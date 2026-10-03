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
//! the stdarch intrinsic inside a `clflushopt` target-feature function (same instruction). The
//! per-line [`Clflushopt::flush_line`] stays `asm!` on both, so it inlines inside fearless
//! kernels (unroll hot loops around it by hand; see its docs); for the intrinsic there, see
//! `with_clflushopt!`.

/// Proof that the CPU has CLFLUSHOPT, plus its flush line size.
#[derive(Clone, Copy, Debug)]
pub struct Clflushopt {
    line: u16,
}

impl Clflushopt {
    /// Detect CLFLUSHOPT (`CPUID.(EAX=7,ECX=0):EBX[23]`) and read the flush line size (a power of
    /// two in 32..=4096; anything else CPUID reports, e.g. from an odd hypervisor, becomes 64).
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

    /// CLFLUSHOPT the cache line containing the start of `x`; nothing for a zero-sized `x`. No
    /// fence: call [`crate::mfence`] before reads that must miss, or use [`Self::flush`].
    ///
    /// In a hot loop, unroll by hand. This is `asm!`, and LLVM's runtime unroller won't unroll a
    /// loop containing inline asm (it counts as a call), so `for line in lines { ..;
    /// cf.flush_line(line) }` runs one line per iteration. Iterating `lines.as_chunks_mut::<4>()`
    /// and flushing each line of the chunk gives the unrolled loop. Each flush still costs a
    /// `lea`, because an `asm!` address is a register operand and can't take a displacement
    /// (the intrinsic's `[r9 - 448]`). The range flushes ([`Self::flush`], [`flush_after`]) are
    /// unrolled with displacements already.
    #[inline(always)]
    pub fn flush_line<T: ?Sized>(self, x: &T) {
        // A zero-sized referent (`&()`, an empty slice) can have a dangling address such as 0x1
        // or 0x8, and CLFLUSHOPT faults on unmapped memory. For sized types this folds away.
        if core::mem::size_of_val(x) == 0 {
            return;
        }
        // SAFETY: `self` proves CLFLUSHOPT; `x` has at least one byte, so its address is mapped.
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
        let first = ptr.wrapping_sub(ptr as usize - start);
        // SAFETY: `self` proves CLFLUSHOPT; every flushed line overlaps the mapped range. The
        // first line starts below `ptr` (in the same line, so the same page) by design. The
        // common 64-byte line gets a constant stride (TMR's addressing); others the general loop.
        unsafe {
            if line == 64 {
                flush_lines_64(first, lines)
            } else {
                flush_lines(first, lines, line)
            }
        }
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

/// Eight CLFLUSHOPTs 64 bytes apart in one `asm!` block, so the offsets are displacements. One
/// `asm!` per line would need each address in a register (a `lea` per line).
///
/// # Safety
/// CLFLUSHOPT must be supported and all eight lines must be mapped.
#[cfg(not(feature = "nightly"))]
#[inline(always)]
unsafe fn clflushopt8_64(p: *const u8) {
    // SAFETY: forwarded. No `nomem`/`readonly`, as in `clflushopt_asm`.
    unsafe {
        core::arch::asm!(
            "clflushopt [{p}]",
            "clflushopt [{p} + 64]",
            "clflushopt [{p} + 128]",
            "clflushopt [{p} + 192]",
            "clflushopt [{p} + 256]",
            "clflushopt [{p} + 320]",
            "clflushopt [{p} + 384]",
            "clflushopt [{p} + 448]",
            p = in(reg) p,
            options(nostack, preserves_flags),
        )
    }
}

/// The nightly twin of `clflushopt8_64`: constant offsets, which LLVM folds into displacements.
///
/// # Safety
/// As `clflushopt8_64`.
#[cfg(feature = "nightly")]
#[inline]
#[target_feature(enable = "clflushopt")]
unsafe fn clflushopt8_64(p: *const u8) {
    for k in 0..8 {
        // SAFETY: forwarded; the target feature is enabled on this fn.
        unsafe { core::arch::x86_64::_mm_clflushopt(p.wrapping_add(k * 64)) }
    }
}

/// The flush loops: `lines` CLFLUSHOPTs from `start`, `line` (or 64) bytes apart. Stable: `asm!`,
/// inlined into the caller. Nightly: the stdarch intrinsic in a `clflushopt` function, so it
/// inlines there (TMR's `flush_range_to_dram`); one call per range, never per line, and a
/// separate 64-byte function because a constant argument to an out-of-line function isn't
/// specialised. Both are unrolled 8x by hand: LLVM won't runtime-unroll a loop around `asm!`.
/// The general-stride loop is for line sizes other than 64, which no current x86 CPU reports.
///
/// Safety (both functions): CLFLUSHOPT must be supported and every line must be mapped.
macro_rules! flush_loops {
    ($($attr:meta),*; $flush:path) => {
        $(#[$attr])*
        unsafe fn flush_lines(start: *const u8, lines: usize, line: usize) {
            let end8 = lines & !7;
            let mut i = 0;
            while i < end8 {
                for k in 0..8 {
                    // SAFETY: forwarded.
                    unsafe { $flush(start.wrapping_add((i + k) * line)) }
                }
                i += 8;
            }
            while i < lines {
                // SAFETY: forwarded.
                unsafe { $flush(start.wrapping_add(i * line)) }
                i += 1;
            }
        }
        $(#[$attr])*
        unsafe fn flush_lines_64(start: *const u8, lines: usize) {
            let end8 = lines & !7;
            let mut i = 0;
            while i < end8 {
                // SAFETY: forwarded; lines i..i + 8 are all in the range.
                unsafe { clflushopt8_64(start.wrapping_add(i * 64)) }
                i += 8;
            }
            while i < lines {
                // SAFETY: forwarded.
                unsafe { $flush(start.wrapping_add(i * 64)) }
                i += 1;
            }
        }
    };
}

#[cfg(not(feature = "nightly"))]
flush_loops!(inline(always); clflushopt_asm);
#[cfg(feature = "nightly")]
flush_loops!(inline, target_feature(enable = "clflushopt"); core::arch::x86_64::_mm_clflushopt);
