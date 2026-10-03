// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Non-temporal (streaming) stores, in a scope that ends with `SFENCE`.
//!
//! ```
//! use fearless_simd::{prelude::*, Avx2, Level, u64x4};
//!
//! #[fearless_simd_macros::simd]
//! fn nt_positional<S: Simd, V: SimdInt<S, Element = u64> + plumb_lines::NtStore<S>>(simd: S, buf: &mut [u64]) {
//!     let (head, vectors, tail) = plumb_lines::as_vectors_mut::<S, V>(simd, buf);
//!     assert!(head.is_empty() && tail.is_empty(), "this example keeps the buffer vector-aligned");
//!     plumb_lines::nontemporal(simd, vectors, |w| {
//!         let mut idx = V::from_fn(simd, |i| i as u64);
//!         let step = V::splat(simd, V::LEN as u64);
//!         w.fill_with(|_| { let v = idx; idx += step; v });
//!     }); // SFENCE here; `vectors` is borrowed until then
//! }
//!
//! // A 64-byte-aligned buffer of 128 u64 (a Vec<Line> provides the alignment).
//! let mut lines = vec![plumb_lines::Line::default(); 16];
//! // SAFETY: Line is [u64; 8].
//! let buf = unsafe { std::slice::from_raw_parts_mut(lines.as_mut_ptr() as *mut u64, 128) };
//! let avx2: Avx2 = Level::new().as_avx2().expect("x86-64-v3 CPU");
//! nt_positional::<Avx2, u64x4<Avx2>>(avx2, buf);
//! assert!(buf.iter().enumerate().all(|(i, &w)| w == i as u64));
//! ```
//!
//! # The rule, and how the scope enforces it
//!
//! stdarch's contract for streaming stores: after one, and **before any other access to that
//! memory (read or write, including another streaming store)**, the storing thread must execute
//! `SFENCE`, and the memory must not be handed to another thread before then. [`nontemporal`]
//! enforces this in safe code:
//!
//! - the destination is mutably borrowed until the scope's `SFENCE` has run (also on unwind);
//! - the [`NtWriter`] is passed **by value** and every way of writing consumes it
//!   ([`NtWriter::fill_with`], [`NtWriter::into_slots`], [`NtWriter::split_at`]); an [`NtSlot`]
//!   can store once. So each slot is written at most once and never read inside the scope;
//! - writers and slots are neither `Send` nor `Sync`, so no other thread can store through them
//!   (this thread's `SFENCE` wouldn't cover another thread's stores);
//! - the closure is higher-ranked over the writer's lifetime, so nothing escapes the scope.
//!
//! ```compile_fail
//! # use fearless_simd::{prelude::*, Level, u64x4};
//! # let t = Level::new().as_avx2().unwrap();
//! # let mut v = [u64x4::splat(t, 0); 4];
//! // Two writes to the same slots: the writer is moved by the first.
//! plumb_lines::nontemporal(t, &mut v, |w| { w.fill_with(|_| u64x4::splat(t, 1)); w.fill_with(|_| u64x4::splat(t, 2)) });
//! ```
//! ```compile_fail
//! # use fearless_simd::{prelude::*, Level, u64x4};
//! # let t = Level::new().as_avx2().unwrap();
//! # let mut v = [u64x4::splat(t, 0); 4];
//! // Reading the destination inside the scope: it is mutably borrowed.
//! plumb_lines::nontemporal(t, &mut v, |w| { let _x = v[0]; w.fill_with(|_| u64x4::splat(t, 1)) });
//! ```
//! ```compile_fail
//! # use fearless_simd::{prelude::*, Level, u64x4};
//! # let t = Level::new().as_avx2().unwrap();
//! # let mut v = [u64x4::splat(t, 0); 4];
//! // A writer escaping the scope.
//! let mut stash = None;
//! plumb_lines::nontemporal(t, &mut v, |w| stash = Some(w));
//! ```
//! ```compile_fail
//! # use fearless_simd::{prelude::*, Level, u64x4};
//! # let t = Level::new().as_avx2().unwrap();
//! # let mut v = [u64x4::splat(t, 0); 4];
//! // Sending a writer to another thread.
//! fn need_send<T: Send>(_: T) {}
//! plumb_lines::nontemporal(t, &mut v, |w| need_send(w));
//! ```
//! ```compile_fail
//! # use fearless_simd::{prelude::*, Level, u64x4};
//! # let t = Level::new().as_avx2().unwrap();
//! # let mut v = [u64x4::splat(t, 0); 4];
//! // Sending a slot to another thread.
//! fn need_send<T: Send>(_: T) {}
//! plumb_lines::nontemporal(t, &mut v, |w| for s in w.into_slots() { need_send(s) });
//! ```
//!
//! # Using it from generic code
//!
//! [`NtStore`] is implemented for the concrete x86 levels (`Sse2`, `Sse4_2`, `Avx2`, `Avx512`), so
//! a level-generic kernel takes the vector type as a parameter with an `NtStore<S>` bound
//! (`V: SimdInt<S, Element = u64> + NtStore<S>`, as above) and is called with a concrete token.
//! It can't be called through fearless's `dispatch!`, whose token is an opaque `impl Simd`.
//!
//! Call the scope from inside a `#[simd]` function or `kernel!`. Outside one, the code still
//! compiles and is correct, but at 512 bits every fearless op and every store becomes an
//! out-of-line call (the 256-bit case only works because the workspace baseline is x86-64-v3).
//! The bench keeps a kernel that does this (`k_ntw_plplain_512`) so the cost stays visible.
//!
//! # Performance notes
//!
//! [`NtWriter::fill_with`] unrolls 4x: the stream intrinsics are `asm!` inside stdarch, so LLVM
//! won't unroll a loop around them (TMR's `simple_write_nt_positional_simd!` does the same by
//! hand). A cache line written only partly before its write-combining buffer drains takes a slow
//! path on current Intel parts. Sequential fills that write whole lines with adjacent narrower
//! stores (4 x 128-bit or 2 x 256-bit) are fine; in TMR's multi-threaded runs 128-bit was the
//! fastest sequential NT width. The slow case is isolated or scattered partial-line stores.

use core::marker::PhantomData;
use fearless_simd::{Bytes, Simd, SimdBase};

mod sealed {
    #[expect(
        unnameable_types,
        reason = "This is a sealed trait, so being unnameable is the entire point"
    )]
    pub trait Sealed {}
}

/// One non-temporal store of a fearless byte vector (`u8x16`, `u8x32`, `u8x64`) at a given
/// level. Implemented per (level, width) with `kernel!`, the way `fearless_simd` implements its own
/// ops, so a generic caller inlines it once the level's target features are in effect. Sealed:
/// use [`NtStore`] (any fearless vector) or [`nontemporal`].
pub trait NtBytes<S: Simd>: Copy + sealed::Sealed {
    /// # Safety
    /// `dst` must be valid for writes of `size_of::<Self>()` bytes and aligned to that size. Until
    /// the calling thread executes `SFENCE` ([`crate::sfence`]), nothing may access that memory
    /// (no reads and no writes, including another streaming store), and it must not be handed to
    /// another thread.
    unsafe fn stream_bytes(self, dst: *mut Self);
}

macro_rules! nt_bytes {
    // One native-width store.
    ($Tok:ident, $V:ident, native $arch:ty, $stream:ident) => {
        impl sealed::Sealed for fearless_simd::$V<fearless_simd::$Tok> {}
        impl NtBytes<fearless_simd::$Tok> for fearless_simd::$V<fearless_simd::$Tok> {
            #[inline(always)]
            unsafe fn stream_bytes(self, dst: *mut Self) {
                fearless_simd::kernel!(
                    #[inline(always)]
                    #[allow(
                        clippy::not_unsafe_ptr_arg_deref,
                        reason = "the contract is on NtBytes::stream_bytes"
                    )]
                    fn k(_t: $Tok, v: fearless_simd::$V<fearless_simd::$Tok>, dst: *mut $arch) {
                        // SAFETY: forwarded from NtBytes::stream_bytes.
                        unsafe { core::arch::x86_64::$stream(dst, v.into()) }
                    }
                );
                k(self.simd, self, dst.cast())
            }
        }
    };
    // A vector wider than the level's registers: store each register-width part.
    ($Tok:ident, $V:ident, parts $n:literal x $arch:ty, $stream:ident) => {
        impl sealed::Sealed for fearless_simd::$V<fearless_simd::$Tok> {}
        impl NtBytes<fearless_simd::$Tok> for fearless_simd::$V<fearless_simd::$Tok> {
            #[inline(always)]
            unsafe fn stream_bytes(self, dst: *mut Self) {
                fearless_simd::kernel!(
                    #[inline(always)]
                    #[allow(
                        clippy::not_unsafe_ptr_arg_deref,
                        reason = "the contract is on NtBytes::stream_bytes"
                    )]
                    fn k(_t: $Tok, v: fearless_simd::$V<fearless_simd::$Tok>, dst: *mut $arch) {
                        // SAFETY: the vector is `$n` contiguous `$arch` registers wide (fearless
                        // stores it as `[$arch; $n]`); `transmute` checks the size.
                        let parts = unsafe {
                            core::mem::transmute::<
                                fearless_simd::$V<fearless_simd::$Tok>,
                                [$arch; $n],
                            >(v)
                        };
                        for (i, part) in parts.into_iter().enumerate() {
                            // SAFETY: forwarded from NtBytes::stream_bytes; part i is at dst + i.
                            unsafe { core::arch::x86_64::$stream(dst.add(i), part) }
                        }
                    }
                );
                k(self.simd, self, dst.cast())
            }
        }
    };
}

use core::arch::x86_64::{__m128i, __m256i, __m512i};
nt_bytes!(Sse2, u8x16, native __m128i, _mm_stream_si128);
nt_bytes!(Sse2, u8x32, parts 2 x __m128i, _mm_stream_si128);
nt_bytes!(Sse2, u8x64, parts 4 x __m128i, _mm_stream_si128);
nt_bytes!(Sse4_2, u8x16, native __m128i, _mm_stream_si128);
nt_bytes!(Sse4_2, u8x32, parts 2 x __m128i, _mm_stream_si128);
nt_bytes!(Sse4_2, u8x64, parts 4 x __m128i, _mm_stream_si128);
nt_bytes!(Avx2, u8x16, native __m128i, _mm_stream_si128);
nt_bytes!(Avx2, u8x32, native __m256i, _mm256_stream_si256);
nt_bytes!(Avx2, u8x64, parts 2 x __m256i, _mm256_stream_si256);
nt_bytes!(Avx512, u8x16, native __m128i, _mm_stream_si128);
nt_bytes!(Avx512, u8x32, native __m256i, _mm256_stream_si256);
nt_bytes!(Avx512, u8x64, native __m512i, _mm512_stream_si512);

/// A non-temporal store for any fearless vector: it's reinterpreted as its byte vector (a free
/// bitcast) and streamed with [`NtBytes`].
pub trait NtStore<S: Simd>: SimdBase<S> {
    /// # Safety
    /// As [`NtBytes::stream_bytes`]: `dst` valid and aligned for `Self`, and no other access to
    /// that memory (read or write) and no publishing before this thread's `SFENCE`. Prefer the
    /// safe [`nontemporal`] scope.
    unsafe fn stream(self, dst: *mut Self);
}

impl<S: Simd, V: SimdBase<S> + Bytes> NtStore<S> for V
where
    V::Bytes: NtBytes<S>,
{
    #[inline(always)]
    unsafe fn stream(self, dst: *mut Self) {
        // SAFETY: a vector and its byte vector have the same size and alignment; the rest of the
        // contract is the caller's.
        unsafe { self.to_bytes().stream_bytes(dst.cast()) }
    }
}

/// Write-only, write-once access to the destination of a [`nontemporal`] scope. Passed by value;
/// every way of writing consumes it, so each slot is written at most once.
pub struct NtWriter<'a, S: Simd, V> {
    ptr: *mut V,
    len: usize,
    _scope: PhantomData<(&'a mut [V], S)>,
}

// Manual `Debug` impls: they show the length only. Reading the destination before the scope's
// SFENCE would break the NT contract, so nothing here may print the slots' contents.
impl<S: Simd, V> core::fmt::Debug for NtWriter<'_, S, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NtWriter")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<S: Simd, V> core::fmt::Debug for NtSlots<'_, S, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // SAFETY: same allocation, cur <= end.
        let left = unsafe { self.end.offset_from(self.cur) };
        f.debug_struct("NtSlots")
            .field("left", &left)
            .finish_non_exhaustive()
    }
}

impl<S: Simd, V> core::fmt::Debug for NtSlot<'_, S, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NtSlot").finish_non_exhaustive()
    }
}

impl<'a, S: Simd, V: NtStore<S>> NtWriter<'a, S, V> {
    /// Number of vectors in the destination.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the destination is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Split into writers for `[0, mid)` and `[mid, len)`. Panics if `mid > len`.
    #[inline(always)]
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        assert!(
            mid <= self.len,
            "NtWriter::split_at: {mid} out of range for {} vectors",
            self.len
        );
        // SAFETY: mid <= len, so both halves are inside the destination and disjoint.
        let right = unsafe { self.ptr.add(mid) };
        (
            Self {
                ptr: self.ptr,
                len: mid,
                _scope: PhantomData,
            },
            Self {
                ptr: right,
                len: self.len - mid,
                _scope: PhantomData,
            },
        )
    }

    /// Stream `f(i)` into every slot, in increasing `i`, 4 stores per iteration. `f` is called
    /// exactly once per slot in order, so it may keep state (e.g. an index vector it advances).
    #[inline(always)]
    pub fn fill_with(self, mut f: impl FnMut(usize) -> V) {
        let (p, n) = (self.ptr, self.len);
        // A precomputed bound (not `i + 4 <= n`) keeps one induction variable in the loop.
        let end4 = n & !3;
        let mut i = 0;
        while i < end4 {
            let (v0, v1, v2, v3) = (f(i), f(i + 1), f(i + 2), f(i + 3));
            // SAFETY: i + 3 < n; `ptr` came from a `&mut [V]`, so it's aligned; each slot is
            // written once; the scope fences.
            unsafe {
                v0.stream(p.add(i));
                v1.stream(p.add(i + 1));
                v2.stream(p.add(i + 2));
                v3.stream(p.add(i + 3));
            }
            i += 4;
        }
        while i < n {
            // SAFETY: i < n; as above.
            unsafe { f(i).stream(p.add(i)) };
            i += 1;
        }
    }

    /// The destination as write-once slots, in order (a pointer walk).
    #[inline(always)]
    pub fn into_slots(self) -> NtSlots<'a, S, V> {
        // SAFETY: `ptr..ptr+len` is this writer's part of the destination.
        NtSlots {
            cur: self.ptr,
            end: unsafe { self.ptr.add(self.len) },
            _w: PhantomData,
        }
    }
}

impl<'a, S: Simd, V: NtStore<S>> IntoIterator for NtWriter<'a, S, V> {
    type Item = NtSlot<'a, S, V>;
    type IntoIter = NtSlots<'a, S, V>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.into_slots()
    }
}

/// Iterator over the write-once slots of an [`NtWriter`].
pub struct NtSlots<'w, S: Simd, V> {
    cur: *mut V,
    end: *mut V,
    _w: PhantomData<(&'w mut [V], S)>,
}

impl<'w, S: Simd, V: NtStore<S>> Iterator for NtSlots<'w, S, V> {
    type Item = NtSlot<'w, S, V>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.end {
            return None;
        }
        let slot = NtSlot {
            ptr: self.cur,
            _w: PhantomData,
        };
        // SAFETY: cur < end, both in the same allocation.
        self.cur = unsafe { self.cur.add(1) };
        Some(slot)
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // SAFETY: same allocation, cur <= end.
        let n = unsafe { self.end.offset_from(self.cur) } as usize;
        (n, Some(n))
    }
}

impl<S: Simd, V: NtStore<S>> ExactSizeIterator for NtSlots<'_, S, V> {}

/// One write-once destination slot: the only thing you can do with it is stream one vector in.
pub struct NtSlot<'w, S: Simd, V> {
    ptr: *mut V,
    _w: PhantomData<(&'w mut V, S)>,
}

impl<S: Simd, V: NtStore<S>> NtSlot<'_, S, V> {
    /// Stream `v` into this slot.
    #[inline(always)]
    pub fn store(self, v: V) {
        // SAFETY: the slot is an aligned element of the scope's destination, written once (the
        // slot is consumed); the scope fences.
        unsafe { v.stream(self.ptr) }
    }
}

/// Run `f` with write-once, non-temporal access to `dst`, then `SFENCE`. The fence also runs if
/// `f` panics. `dst` stays mutably borrowed until the fence has run, so no code can read it with
/// stores still in flight; see the [module docs](self) for the full argument.
#[inline(always)]
pub fn nontemporal<S: Simd, V: NtStore<S>, R>(
    simd: S,
    dst: &mut [V],
    f: impl FnOnce(NtWriter<'_, S, V>) -> R,
) -> R {
    struct Fence;
    impl Drop for Fence {
        #[inline(always)]
        fn drop(&mut self) {
            crate::sfence();
        }
    }
    let _ = simd;
    let _fence = Fence;
    f(NtWriter {
        ptr: dst.as_mut_ptr(),
        len: dst.len(),
        _scope: PhantomData,
    })
}
