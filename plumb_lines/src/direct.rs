// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! MOVDIR64B: 64-byte direct stores, in a scope that ends with `SFENCE`.
//!
//! MOVDIR64B copies one 64-byte line from memory to a 64-byte-aligned destination as a single
//! write: no read-for-ownership, no cache allocation, and the 64 bytes land atomically. That
//! makes it a distinct DRAM write path for a memory tester. It measured level with 512-bit NT
//! stores multi-threaded, and slightly ahead single-threaded. Like NT stores it's weakly ordered,
//! so the same rule applies: no other access to the written memory (read or write, including
//! another direct store) and no publishing before this thread's `SFENCE`. [`direct`] enforces it
//! the way [`crate::nontemporal`] does: the destination is borrowed until the scope's fence (also
//! on unwind), and the [`DirectWriter`] is passed by value, write-once and `!Send`.
//!
//! The destination type is [`Line`] (`#[repr(align(64))]`), so the alignment MOVDIR64B demands
//! is a property of the type. [`as_lines_mut`] carves lines out of a `u64` buffer. The source
//! can be any readable 64 bytes; to write a pattern, build one `Line` and copy it with
//! [`DirectWriter::fill`], or generate per line with [`DirectWriter::fill_with`] (the line is
//! built in a stack temporary, then copied).
//!
//! ```compile_fail
//! # let md = plumb_lines::Movdir64b::try_new().unwrap();
//! # let mut dst = [plumb_lines::Line::default(); 4];
//! # let src = plumb_lines::Line::default();
//! // Two writes to the same lines: the writer is moved by the first.
//! plumb_lines::direct(md, &mut dst, |w| { w.fill(&src); w.fill(&src) });
//! ```
//! ```compile_fail
//! # let md = plumb_lines::Movdir64b::try_new().unwrap();
//! # let mut dst = [plumb_lines::Line::default(); 4];
//! fn need_send<T: Send>(_: T) {}
//! plumb_lines::direct(md, &mut dst, |w| need_send(w));
//! ```
//!
//! This is `asm!` on every toolchain; `asm!` needs no target feature and inlines anywhere.
//! rustc's `movdir64b` target feature is on nightly (rust-lang/rust#163742), and stdarch's
//! `_movdir64b` is in review (rust-lang/stdarch#2239); once that reaches nightly the `nightly`
//! feature can use it.

use core::marker::PhantomData;
use core::slice;

/// Proof that the CPU has MOVDIR64B (`CPUID.(EAX=7,ECX=0):ECX[28]`).
#[derive(Clone, Copy, Debug)]
pub struct Movdir64b {
    _private: (),
}

impl Movdir64b {
    /// Detect MOVDIR64B.
    pub fn try_new() -> Option<Self> {
        crate::cpu::has_movdir64b().then_some(Self { _private: () })
    }

    /// # Safety
    /// The CPU must support MOVDIR64B.
    pub unsafe fn assume_supported() -> Self {
        Self { _private: () }
    }
}

/// One 64-byte, 64-byte-aligned cache line.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Line(pub [u64; 8]);

/// Split a `u64` slice into `(head, aligned lines, tail)`, like [`crate::as_vectors`].
#[inline(always)]
pub fn as_lines(buf: &[u64]) -> (&[u64], &[Line], &[u64]) {
    let (head, n, tail) = split(buf.as_ptr() as usize, buf.len());
    let p = buf.as_ptr();
    let mid = align_line(p.wrapping_add(head).cast::<Line>());
    // SAFETY: when n > 0 the middle is 64-byte aligned (align_line is then the identity) and
    // inside `buf`, and `Line` is `[u64; 8]` with that alignment; when n == 0 it's an aligned,
    // zero-length slice (aligned without a branch; see `crate::view`). Head and tail are the
    // disjoint sub-ranges of `buf` around it.
    unsafe {
        (
            slice::from_raw_parts(p, head),
            slice::from_raw_parts(mid, n),
            slice::from_raw_parts(p.add(tail), buf.len() - tail),
        )
    }
}

/// Mutable version of [`as_lines`].
#[inline(always)]
pub fn as_lines_mut(buf: &mut [u64]) -> (&mut [u64], &mut [Line], &mut [u64]) {
    let (head, n, tail) = split(buf.as_ptr() as usize, buf.len());
    let len = buf.len();
    let p = buf.as_mut_ptr();
    let mid = align_line(p.wrapping_add(head).cast::<Line>().cast_const()).cast_mut();
    // SAFETY: as `as_lines`; the ranges are disjoint, so the mutable borrows don't alias.
    unsafe {
        (
            slice::from_raw_parts_mut(p, head),
            slice::from_raw_parts_mut(mid, n),
            slice::from_raw_parts_mut(p.add(tail), len - tail),
        )
    }
}

/// Round up to 64-byte alignment, keeping provenance (identity for aligned pointers).
#[inline(always)]
fn align_line(p: *const Line) -> *const Line {
    p.map_addr(|a| (a + 63) & !63)
}

/// (head elements, whole lines, tail start) for a u64 slice at `addr` of `len` elements.
#[inline(always)]
fn split(addr: usize, len: usize) -> (usize, usize, usize) {
    let head = ((addr.wrapping_neg() & 63) / 8).min(len);
    let n = (len - head) / 8;
    (head, n, head + n * 8)
}

/// Write-only, write-once access to the destination of a [`direct`] scope. Passed by value;
/// every way of writing consumes it, so each line is written at most once.
pub struct DirectWriter<'a> {
    ptr: *mut Line,
    len: usize,
    _scope: PhantomData<&'a mut [Line]>,
}

impl<'a> DirectWriter<'a> {
    /// Number of destination lines.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the destination is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Split into writers for lines `[0, mid)` and `[mid, len)`. Panics if `mid > len`.
    #[inline(always)]
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        assert!(
            mid <= self.len,
            "DirectWriter::split_at: {mid} out of range for {} lines",
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

    /// MOVDIR64B `src[i]` into line `i` for every line. Panics if the lengths differ.
    #[inline(always)]
    pub fn copy_from(self, src: &[Line]) {
        assert_eq!(
            src.len(),
            self.len,
            "DirectWriter::copy_from: length mismatch"
        );
        for (i, s) in src.iter().enumerate() {
            // SAFETY: i < len, aligned (from a `&mut [Line]`), written once; the scope fences.
            unsafe { movdir64b(self.ptr.add(i), s) }
        }
    }

    /// MOVDIR64B the same `src` line into every destination line (a pattern fill).
    #[inline(always)]
    pub fn fill(self, src: &Line) {
        for i in 0..self.len {
            // SAFETY: as `copy_from`.
            unsafe { movdir64b(self.ptr.add(i), src) }
        }
    }

    /// Generate each line with `f(i)` (in increasing `i`, once each) into a stack temporary and
    /// MOVDIR64B it to line `i`.
    #[inline(always)]
    pub fn fill_with(self, mut f: impl FnMut(usize) -> Line) {
        for i in 0..self.len {
            let tmp = f(i);
            // SAFETY: as `copy_from`; `tmp` is a live 64-byte line.
            unsafe { movdir64b(self.ptr.add(i), &tmp) }
        }
    }

    /// The destination as write-once slots, in order.
    #[inline(always)]
    pub fn into_slots(self) -> DirectSlots<'a> {
        // SAFETY: `ptr..ptr+len` is this writer's part of the destination.
        DirectSlots {
            cur: self.ptr,
            end: unsafe { self.ptr.add(self.len) },
            _w: PhantomData,
        }
    }
}

// Manual `Debug` impls: the length only. Nothing may read the destination before the SFENCE.
impl core::fmt::Debug for DirectWriter<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DirectWriter")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for DirectSlots<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // SAFETY: same allocation, cur <= end.
        let left = unsafe { self.end.offset_from(self.cur) };
        f.debug_struct("DirectSlots")
            .field("left", &left)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for DirectSlot<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DirectSlot").finish_non_exhaustive()
    }
}

/// Iterator over the write-once slots of a [`DirectWriter`].
pub struct DirectSlots<'w> {
    cur: *mut Line,
    end: *mut Line,
    _w: PhantomData<&'w mut [Line]>,
}

impl<'w> Iterator for DirectSlots<'w> {
    type Item = DirectSlot<'w>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.end {
            return None;
        }
        let slot = DirectSlot {
            ptr: self.cur,
            _w: PhantomData,
        };
        // SAFETY: cur < end, same allocation.
        self.cur = unsafe { self.cur.add(1) };
        Some(slot)
    }
}

/// One write-once destination line.
pub struct DirectSlot<'w> {
    ptr: *mut Line,
    _w: PhantomData<&'w mut Line>,
}

impl DirectSlot<'_> {
    /// MOVDIR64B `src` into this line.
    #[inline(always)]
    pub fn copy(self, src: &Line) {
        // SAFETY: an aligned line of the scope's destination, written once; the scope fences.
        unsafe { movdir64b(self.ptr, src) }
    }
}

/// Run `f` with write-once MOVDIR64B access to `dst`, then `SFENCE` (also if `f` panics). `dst`
/// stays borrowed until the fence has run.
#[inline(always)]
pub fn direct<R>(md: Movdir64b, dst: &mut [Line], f: impl FnOnce(DirectWriter<'_>) -> R) -> R {
    struct Fence;
    impl Drop for Fence {
        #[inline(always)]
        fn drop(&mut self) {
            crate::sfence();
        }
    }
    let _ = md;
    let _fence = Fence;
    f(DirectWriter {
        ptr: dst.as_mut_ptr(),
        len: dst.len(),
        _scope: PhantomData,
    })
}

/// One MOVDIR64B.
///
/// # Safety
/// MOVDIR64B must be supported (callers hold a [`Movdir64b`]); `dst` must be valid for a 64-byte
/// write and 64-byte aligned (else #GP); until this thread's `SFENCE` nothing may access `dst`
/// (read or write) and it must not be published.
#[inline(always)]
unsafe fn movdir64b(dst: *mut Line, src: &Line) {
    // SAFETY: forwarded. AT&T operand order: memory source, then the register holding the
    // destination address (Intel order is the reverse, which is easy to get backwards).
    unsafe {
        core::arch::asm!("movdir64b ({src}), {dst}",
            src = in(reg) src as *const Line, dst = in(reg) dst,
            options(att_syntax, nostack, preserves_flags));
    }
}
