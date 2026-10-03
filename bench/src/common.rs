//! Constants, buffers, CPU detection and pinning shared by both kernel sets.

use std::arch::x86_64::{__cpuid_count, _xgetbv};

/// TMR's `REFRESH_PATTERN`: not byte-uniform, so fill loops are not rewritten to `memset`.
pub const PATTERN: u64 = 0xA55A_A55A_A55A_A55A;
/// Byte-uniform canary (`0xA5` x 8). Used only by the `fill_uniform` kernels, which are
/// *expected* to collapse to `memset` in both styles (the trap applies to fearless_simd too).
pub const UNIFORM: u64 = 0xA5A5_A5A5_A5A5_A5A5;
/// Positional base, as in `simple_test_nt_impl!` (thread 0).
pub const POS_BASE: u64 = 0xDEAD_BEEF_DEAD_BEEF;
/// LCG constants (Knuth MMIX). The values don't matter for codegen; the 64-bit multiply does.
pub const LCG_MUL: u64 = 6_364_136_223_846_793_005;
pub const LCG_ADD: u64 = 1_442_695_040_888_963_407;
pub const LCG_SEED: u64 = 0x1234_5678_9ABC_DEF0;
/// Cache line. TMR reads it from `CacheInfo`; every x86 part TMR supports uses 64.
pub const LINE: usize = 64;
/// Prefetch distance for the prefetch-verify kernels, in bytes (8 lines ahead).
pub const PF_DIST: usize = 8 * LINE;

#[inline(always)]
pub fn lcg_next(s: u64, m: u64, a: u64) -> u64 {
    s.wrapping_mul(m).wrapping_add(a)
}

/// Seeds, multiplier and addend for an `n`-lane LCG where every lane jumps `n` steps
/// (same scheme as TMR's `LcgSimdN::new`).
pub fn lcg_lanes(n: usize) -> ([u64; 8], u64, u64) {
    let mut seeds = [0u64; 8];
    let mut s = LCG_SEED;
    for seed in seeds.iter_mut().take(n) {
        *seed = s;
        s = lcg_next(s, LCG_MUL, LCG_ADD);
    }
    // m^n and the matching addend: x -> m^n x + a(m^(n-1) + ... + 1).
    let (mut mn, mut an) = (1u64, 0u64);
    for _ in 0..n {
        an = an.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        mn = mn.wrapping_mul(LCG_MUL);
    }
    (seeds, mn, an)
}

/// Page-aligned, pre-touched heap buffer of `u64`s. Ordinary 4 KiB pages: no large pages, so
/// no `SeLockMemoryPrivilege` and none of the large-page risks in the workspace CLAUDE.md.
pub struct AlignedBuf {
    ptr: *mut u64,
    len: usize,
    layout: std::alloc::Layout,
}

impl AlignedBuf {
    pub fn new(bytes: usize) -> Self {
        assert!(bytes > 0 && bytes.is_multiple_of(4096));
        let layout = std::alloc::Layout::from_size_align(bytes, 4096).unwrap();
        // SAFETY: non-zero size; zeroed so every page is touched before timing.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut u64;
        assert!(!ptr.is_null(), "allocation of {bytes} bytes failed");
        Self { ptr, len: bytes / 8, layout }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        // SAFETY: owned allocation of `len` u64s, initialised by alloc_zeroed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout) };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Cpu {
    pub avx2: bool,
    pub avx512f: bool,
    pub clflushopt: bool,
    pub movdir64b: bool,
    pub amx_tile: bool,
    pub amx_int8: bool,
    pub amx_bf16: bool,
    /// XCR0 bits 17 (XTILECFG) and 18 (XTILEDATA): the OS has enabled AMX tile state.
    pub os_amx: bool,
}

/// CPUID leaf 7 directly: `is_x86_feature_detected!` doesn't know `movdir64b`, and TMR's
/// startup gate reads CLFLUSHOPT the same way. Read-only; executes no AMX instruction.
pub fn detect() -> Cpu {
    let l7 = __cpuid_count(7, 0);
    let l1 = __cpuid_count(1, 0);
    let osxsave = l1.ecx & (1 << 27) != 0;
    // SAFETY: XGETBV is valid once CPUID reports OSXSAVE.
    let xcr0 = if osxsave { unsafe { _xgetbv(0) } } else { 0 };
    Cpu {
        avx2: is_x86_feature_detected!("avx2"),
        avx512f: is_x86_feature_detected!("avx512f"),
        clflushopt: l7.ebx & (1 << 23) != 0,
        movdir64b: l7.ecx & (1 << 28) != 0,
        amx_bf16: l7.edx & (1 << 22) != 0,
        amx_tile: l7.edx & (1 << 24) != 0,
        amx_int8: l7.edx & (1 << 25) != 0,
        os_amx: xcr0 & (0b11 << 17) == (0b11 << 17),
    }
}

#[cfg(windows)]
pub fn pin_to_cpu(idx: usize) {
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
    }
    // SAFETY: plain Win32 calls on the current thread's pseudo-handle.
    let prev = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << idx) };
    if prev == 0 {
        eprintln!("warn: SetThreadAffinityMask({idx}) failed; results may be noisier");
    }
}

#[cfg(not(windows))]
pub fn pin_to_cpu(_idx: usize) {}

/// MOVDIR64B: one 64-byte direct store from `src` to `dst` (asm from `../shuffle-test/`; stdarch
/// has no intrinsic). Weakly ordered, like an NT store: the caller must `sfence`.
///
/// # Safety
/// `src` readable and `dst` writable for 64 bytes; `dst` 64-byte aligned; CPU has MOVDIR64B.
#[inline(always)]
pub unsafe fn movdir64b(dst: *mut u8, src: *const u8) {
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        core::arch::asm!("movdir64b ({src}), {dst}",
            src = in(reg) src, dst = in(reg) dst,
            options(att_syntax, nostack, preserves_flags));
    }
}

/// CLFLUSHOPT as inline asm. Needs no target feature, so it inlines into any fn, including a
/// fearless_simd kernel whose token doesn't list `clflushopt`.
///
/// # Safety
/// `p` must be a mapped address; the CPU must have CLFLUSHOPT.
#[inline(always)]
pub unsafe fn clflushopt_asm(p: *const u8) {
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        core::arch::asm!("clflushopt [{}]", in(reg) p, options(nostack, preserves_flags));
    }
}
