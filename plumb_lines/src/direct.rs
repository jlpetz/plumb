//! MOVDIR64B: 64-byte direct stores, in a scope that ends with `SFENCE`.
//!
//! MOVDIR64B copies one 64-byte line from memory to a 64-byte-aligned destination as a single
//! write: no read-for-ownership, no cache allocation, and the 64 bytes land atomically. That
//! makes it a distinct DRAM write path for a memory tester. It measured level with 512-bit NT
//! stores multi-threaded, and slightly ahead single-threaded (TMR `../shuffle-test/`). Like NT
//! stores it's weakly ordered, so [`direct`] fences when it ends; see the crate docs.
//!
//! The destination type is [`Line`] (`#[repr(align(64))]`), so the alignment MOVDIR64B demands
//! is a property of the type. [`as_lines_mut`] carves lines out of a `u64` buffer. The source
//! can be any readable 64 bytes; to write a pattern, build one `Line` and copy it with
//! [`DirectWriter::fill`], or generate per line with [`DirectWriter::fill_with`] (the line is
//! built in a stack temporary, then copied).
//!
//! stdarch has no MOVDIR64B intrinsic and rustc has no `movdir64b` target feature yet, so this is
//! `asm!` on every toolchain; `asm!` needs neither and inlines anywhere.

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
    // An empty middle must still be aligned (see `crate::view`).
    let mid: &[Line] = if n == 0 {
        &[]
    } else {
        // SAFETY: the middle is 64-byte aligned, inside `buf`, and `Line` is `[u64; 8]` with
        // that alignment.
        unsafe { slice::from_raw_parts(p.add(head) as *const Line, n) }
    };
    // SAFETY: head and tail are disjoint sub-ranges of `buf`.
    unsafe { (slice::from_raw_parts(p, head), mid, slice::from_raw_parts(p.add(tail), buf.len() - tail)) }
}

/// Mutable version of [`as_lines`].
#[inline(always)]
pub fn as_lines_mut(buf: &mut [u64]) -> (&mut [u64], &mut [Line], &mut [u64]) {
    let (head, n, tail) = split(buf.as_ptr() as usize, buf.len());
    let len = buf.len();
    let p = buf.as_mut_ptr();
    let mid: &mut [Line] = if n == 0 {
        &mut []
    } else {
        // SAFETY: as `as_lines`.
        unsafe { slice::from_raw_parts_mut(p.add(head) as *mut Line, n) }
    };
    // SAFETY: the ranges are disjoint sub-ranges of `buf`.
    unsafe { (slice::from_raw_parts_mut(p, head), mid, slice::from_raw_parts_mut(p.add(tail), len - tail)) }
}

/// (head elements, whole lines, tail start) for a u64 slice at `addr` of `len` elements.
#[inline(always)]
fn split(addr: usize, len: usize) -> (usize, usize, usize) {
    let head = ((addr.wrapping_neg() & 63) / 8).min(len);
    let n = (len - head) / 8;
    (head, n, head + n * 8)
}

/// Write-only access to the destination of a [`direct`] scope.
pub struct DirectWriter<'a> {
    ptr: *mut Line,
    len: usize,
    _scope: PhantomData<&'a mut [Line]>,
}

impl DirectWriter<'_> {
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

    /// MOVDIR64B `src` into line `i`. Panics if `i >= len`.
    #[inline(always)]
    pub fn copy(&mut self, i: usize, src: &Line) {
        assert!(i < self.len, "DirectWriter::copy: index {i} out of range for {} lines", self.len);
        // SAFETY: in bounds and aligned (from a `&mut [Line]`); `src` is 64 readable bytes; the
        // scope fences.
        unsafe { movdir64b(self.ptr.add(i), src) }
    }

    /// MOVDIR64B `src[i]` into line `i` for every line. Panics if the lengths differ.
    #[inline(always)]
    pub fn copy_from(&mut self, src: &[Line]) {
        assert_eq!(src.len(), self.len, "DirectWriter::copy_from: length mismatch");
        for (i, s) in src.iter().enumerate() {
            // SAFETY: i < len; see `copy`.
            unsafe { movdir64b(self.ptr.add(i), s) }
        }
    }

    /// MOVDIR64B the same `src` line into every destination line (a pattern fill).
    #[inline(always)]
    pub fn fill(&mut self, src: &Line) {
        for i in 0..self.len {
            // SAFETY: i < len; see `copy`.
            unsafe { movdir64b(self.ptr.add(i), src) }
        }
    }

    /// Generate each line with `f(i)` (in increasing `i`, once each) into a stack temporary and
    /// MOVDIR64B it to line `i`.
    #[inline(always)]
    pub fn fill_with(&mut self, mut f: impl FnMut(usize) -> Line) {
        for i in 0..self.len {
            let tmp = f(i);
            // SAFETY: i < len; `tmp` is a live 64-byte line.
            unsafe { movdir64b(self.ptr.add(i), &tmp) }
        }
    }
}

/// Run `f` with write-only MOVDIR64B access to `dst`, then `SFENCE` (also if `f` panics). `dst`
/// stays borrowed until the fence has run.
#[inline(always)]
pub fn direct<R>(md: Movdir64b, dst: &mut [Line], f: impl FnOnce(&mut DirectWriter<'_>) -> R) -> R {
    struct Fence;
    impl Drop for Fence {
        #[inline(always)]
        fn drop(&mut self) {
            crate::sfence();
        }
    }
    let _ = md;
    let _fence = Fence;
    let mut w = DirectWriter { ptr: dst.as_mut_ptr(), len: dst.len(), _scope: PhantomData };
    f(&mut w)
}

/// One MOVDIR64B.
///
/// # Safety
/// MOVDIR64B must be supported (callers hold a [`Movdir64b`]); `dst` must be valid for a 64-byte
/// write and 64-byte aligned (else #GP); the thread must `SFENCE` before the data is read.
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
