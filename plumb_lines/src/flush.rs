//! CLFLUSHOPT: evict cache lines to DRAM, so the next read of them is a real DRAM round trip.
//!
//! CLFLUSHOPT is about 15x faster than CLFLUSH (which is globally serialized), and it's weakly
//! ordered, so a flushed range needs `MFENCE` before the reads that must miss. [`Clflushopt::flush`]
//! and [`flush_after`] include it.
//!
//! [`flush_after`] is the "write, then flush everything written, before anything reads it"
//! scope. It needs no dirty-line tracking because it owns the range: it flushes the whole
//! borrowed slice when `f` returns. Flushing isn't a memory-safety matter (caches are
//! coherent), so the scope hands out ordinary `&mut` access.
//!
//! CLFLUSHOPT isn't part of any x86-64 psABI level, so it isn't in any fearless_simd token; it's
//! a separate capability ([`Clflushopt`]). On stable it is emitted with `asm!`, which needs no
//! target feature and inlines into any function. With the `nightly` feature the range flush uses
//! the stdarch intrinsic inside a `clflushopt` target-feature function (same instruction; LLVM
//! may unroll the loop). The per-line [`Clflushopt::flush_line`] stays `asm!` on both, so it
//! inlines inside fearless kernels; for the intrinsic there, see `with_clflushopt!`.

/// Proof that the CPU has CLFLUSHOPT, plus its flush line size.
#[derive(Clone, Copy, Debug)]
pub struct Clflushopt {
    line: u16,
}

impl Clflushopt {
    /// Detect CLFLUSHOPT (CPUID.(EAX=7,ECX=0):EBX[23]) and read the flush line size.
    pub fn try_new() -> Option<Self> {
        crate::cpu::has_clflushopt().then(|| Self { line: crate::cpu::flush_line_bytes() as u16 })
    }

    /// # Safety
    /// The CPU must support CLFLUSHOPT, and `line_bytes` must be a power of two no larger than
    /// the CPU's flush line size (smaller is allowed: it only flushes some lines twice).
    pub unsafe fn assume_supported(line_bytes: usize) -> Self {
        assert!(line_bytes.is_power_of_two() && line_bytes <= u16::MAX as usize);
        Self { line: line_bytes as u16 }
    }

    /// The flush granularity in bytes.
    #[inline(always)]
    pub fn line_bytes(self) -> usize {
        self.line as usize
    }

    /// CLFLUSHOPT the cache line containing the start of `x`. No fence: call
    /// [`crate::mfence`] before reads that must miss, or use [`Self::flush`].
    #[inline(always)]
    pub fn flush_line<T: ?Sized>(self, x: &T) {
        // SAFETY: `self` proves CLFLUSHOPT; the address is that of a live reference.
        unsafe { clflushopt_asm(x as *const T as *const u8) }
    }

    /// CLFLUSHOPT every cache line that overlaps `buf`, then `MFENCE`.
    #[inline(always)]
    pub fn flush<T>(self, buf: &[T]) {
        self.flush_no_fence(buf);
        crate::mfence();
    }

    /// CLFLUSHOPT every cache line that overlaps `buf`, without the fence (for batching several
    /// ranges under one [`crate::mfence`]).
    #[inline(always)]
    pub fn flush_no_fence<T>(self, buf: &[T]) {
        // SAFETY: the range is the live slice `buf`.
        unsafe { self.flush_range(buf.as_ptr() as *const u8, core::mem::size_of_val(buf)) }
    }

    /// CLFLUSHOPT every line overlapping `[ptr, ptr + bytes)`.
    ///
    /// # Safety
    /// Every line overlapping the range must be mapped.
    #[inline(always)]
    unsafe fn flush_range(self, ptr: *const u8, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let line = self.line_bytes();
        let start = (ptr as usize) & !(line - 1);
        let end = ptr as usize + bytes;
        let lines = (end - start).div_ceil(line);
        // SAFETY: `self` proves CLFLUSHOPT; every flushed line overlaps the mapped range. The
        // first line starts below `ptr` (in the same line, so the same page) by design.
        unsafe { flush_lines(ptr.wrapping_sub(ptr as usize - start), lines, line) }
    }
}

/// Run `f` with ordinary mutable access to `buf`, then CLFLUSHOPT every line of `buf` and
/// `MFENCE`, so the next read of any of it comes from DRAM. The flush also runs if `f` panics.
#[inline(always)]
pub fn flush_after<T, R>(cf: Clflushopt, buf: &mut [T], f: impl FnOnce(&mut [T]) -> R) -> R {
    struct Flush {
        cf: Clflushopt,
        ptr: *const u8,
        bytes: usize,
    }
    impl Drop for Flush {
        #[inline(always)]
        fn drop(&mut self) {
            // SAFETY: the range is `flush_after`'s borrowed slice, still mapped: the guard is
            // dropped before `flush_after` returns. Only addresses are used; no reference is
            // re-created.
            unsafe { self.cf.flush_range(self.ptr, self.bytes) };
            crate::mfence();
        }
    }
    let (ptr, len) = (buf.as_mut_ptr(), buf.len());
    let _flush = Flush { cf, ptr: ptr as *const u8, bytes: core::mem::size_of_val(buf) };
    // SAFETY: `ptr..ptr+len` is `buf`. The closure's slice is derived from `ptr`, so `ptr` stays
    // its parent and the guard's later use of it is valid.
    f(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
}

/// One CLFLUSHOPT. No target feature needed, so it inlines anywhere.
///
/// # Safety
/// CLFLUSHOPT must be supported; `p` should be a mapped address (an unmapped one faults).
#[inline(always)]
pub(crate) unsafe fn clflushopt_asm(p: *const u8) {
    // SAFETY: forwarded. No `nomem`/`readonly`: the compiler must keep earlier stores to the
    // line before the flush.
    unsafe { core::arch::asm!("clflushopt [{}]", in(reg) p, options(nostack, preserves_flags)) }
}

/// `lines` CLFLUSHOPTs from `start`, `line` bytes apart.
///
/// # Safety
/// CLFLUSHOPT must be supported and every line must be mapped.
#[cfg(not(feature = "nightly"))]
#[inline(always)]
unsafe fn flush_lines(start: *const u8, lines: usize, line: usize) {
    for i in 0..lines {
        // SAFETY: forwarded.
        unsafe { clflushopt_asm(start.wrapping_add(i * line)) }
    }
}

/// Nightly: the stdarch intrinsic in a `clflushopt` function, so it inlines and LLVM can unroll
/// the loop (TMR's `flush_range_to_dram`). One call per range, never per line.
///
/// # Safety
/// As the stable version.
#[cfg(feature = "nightly")]
#[inline]
#[target_feature(enable = "clflushopt")]
unsafe fn flush_lines(start: *const u8, lines: usize, line: usize) {
    for i in 0..lines {
        // SAFETY: forwarded; the target feature is enabled on this fn.
        unsafe { core::arch::x86_64::_mm_clflushopt(start.wrapping_add(i * line)) }
    }
}
