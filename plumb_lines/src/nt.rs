//! Non-temporal (streaming) stores, in a scope that ends with `SFENCE`.
//!
//! ```ignore
//! let (head, vectors, tail) = plumb_lines::as_vectors_mut::<_, u64x8<_>>(simd, buf);
//! plumb_lines::nontemporal(simd, vectors, |w| {
//!     let mut idx = u64x8::from_fn(simd, |i| i as u64);
//!     let step = u64x8::splat(simd, 8);
//!     w.fill_with(|_| { let v = idx ^ base; idx += step; v });
//! }); // SFENCE here; `vectors` is borrowed until then
//! ```
//!
//! The writer never hands out a readable reference into the destination: reading memory with
//! NT stores still in flight is exactly what the stdarch contract forbids. [`NtWriter::fill_with`]
//! unrolls 4x, because the stream intrinsics are `asm!` inside stdarch and LLVM won't unroll a
//! loop around them (TMR's `simple_write_nt_positional_simd!` does the same by hand).
//!
//! Only full 64-byte lines take the fast NT path on current Intel parts (one 512-bit store, or
//! adjacent 256-bit pairs); narrower NT stores pay a per-line cost (TMR `doc/nt_stores.md`).

use core::marker::PhantomData;
use fearless_simd::{Bytes, Simd, SimdBase};

mod sealed {
    pub trait Sealed {}
}

/// One non-temporal store of a fearless byte vector (`u8x16`, `u8x32`, `u8x64`) at a given
/// level. Implemented per (level, width) with `kernel!`, the way fearless_simd implements its own
/// ops, so a generic caller inlines it once the level's target features are in effect. Sealed:
/// use [`NtStore`] (any fearless vector) or [`nontemporal`].
pub trait NtBytes<S: Simd>: Copy + sealed::Sealed {
    /// # Safety
    /// `dst` must be valid for writes of `size_of::<Self>()` bytes and aligned to that size, and
    /// the calling thread must `SFENCE` ([`crate::sfence`]) before anything reads that memory or
    /// before publishing it to another thread.
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
                    #[allow(clippy::not_unsafe_ptr_arg_deref, reason = "the contract is on NtBytes::stream_bytes")]
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
                    #[allow(clippy::not_unsafe_ptr_arg_deref, reason = "the contract is on NtBytes::stream_bytes")]
                    fn k(_t: $Tok, v: fearless_simd::$V<fearless_simd::$Tok>, dst: *mut $arch) {
                        // SAFETY: the vector is `$n` contiguous `$arch` registers wide (fearless
                        // stores it as `[$arch; $n]`); `transmute` checks the size.
                        let parts = unsafe { core::mem::transmute::<fearless_simd::$V<fearless_simd::$Tok>, [$arch; $n]>(v) };
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
    /// As [`NtBytes::stream_bytes`]: `dst` valid and aligned for `Self`, and `SFENCE` before the
    /// memory is read or published. Prefer the safe [`nontemporal`] scope.
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

/// Write-only access to the destination of a [`nontemporal`] scope.
pub struct NtWriter<'a, S: Simd, V> {
    ptr: *mut V,
    len: usize,
    _scope: PhantomData<(&'a mut [V], S)>,
}

impl<S: Simd, V: NtStore<S>> NtWriter<'_, S, V> {
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

    /// Stream `v` into slot `i`. Panics if `i >= len`.
    #[inline(always)]
    pub fn store(&mut self, i: usize, v: V) {
        assert!(i < self.len, "NtWriter::store: index {i} out of range for {} vectors", self.len);
        // SAFETY: in bounds; `ptr` came from a `&mut [V]`, so it's aligned; the scope fences.
        unsafe { v.stream(self.ptr.add(i)) }
    }

    /// Stream `f(i)` into every slot, in increasing `i`, 4 stores per iteration. `f` is called
    /// exactly once per slot in order, so it may keep state (e.g. an index vector it advances).
    #[inline(always)]
    pub fn fill_with(&mut self, mut f: impl FnMut(usize) -> V) {
        let (p, n) = (self.ptr, self.len);
        let mut i = 0;
        while i + 4 <= n {
            let (v0, v1, v2, v3) = (f(i), f(i + 1), f(i + 2), f(i + 3));
            // SAFETY: i + 3 < n; see `store`.
            unsafe {
                v0.stream(p.add(i));
                v1.stream(p.add(i + 1));
                v2.stream(p.add(i + 2));
                v3.stream(p.add(i + 3));
            }
            i += 4;
        }
        while i < n {
            // SAFETY: i < n; see `store`.
            unsafe { f(i).stream(p.add(i)) };
            i += 1;
        }
    }

    /// The destination as write-only slots, in order (a pointer walk).
    #[inline(always)]
    pub fn slots(&mut self) -> NtSlots<'_, S, V> {
        // SAFETY: `ptr..ptr+len` is the borrowed destination.
        NtSlots { cur: self.ptr, end: unsafe { self.ptr.add(self.len) }, _w: PhantomData }
    }
}

/// Iterator over the write-only slots of an [`NtWriter`].
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
        let slot = NtSlot { ptr: self.cur, _w: PhantomData };
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

/// One write-only destination slot: the only thing you can do with it is stream a vector in.
pub struct NtSlot<'w, S: Simd, V> {
    ptr: *mut V,
    _w: PhantomData<(&'w mut V, S)>,
}

impl<S: Simd, V: NtStore<S>> NtSlot<'_, S, V> {
    /// Stream `v` into this slot.
    #[inline(always)]
    pub fn store(self, v: V) {
        // SAFETY: the slot is an aligned element of the scope's destination; the scope fences.
        unsafe { v.stream(self.ptr) }
    }
}

/// Run `f` with write-only, non-temporal access to `dst`, then `SFENCE`. The fence also runs if
/// `f` panics. `dst` stays mutably borrowed until the fence has run, so no code can read it with
/// stores still in flight.
#[inline(always)]
pub fn nontemporal<S: Simd, V: NtStore<S>, R>(simd: S, dst: &mut [V], f: impl FnOnce(&mut NtWriter<'_, S, V>) -> R) -> R {
    struct Fence;
    impl Drop for Fence {
        #[inline(always)]
        fn drop(&mut self) {
            crate::sfence();
        }
    }
    let _ = simd;
    let _fence = Fence;
    let mut w = NtWriter { ptr: dst.as_mut_ptr(), len: dst.len(), _scope: PhantomData };
    f(&mut w)
}
