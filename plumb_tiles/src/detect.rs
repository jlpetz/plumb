// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Runtime detection: CPUID feature bits, the palette-1 and TMUL limits, OS-enabled tile state
//! (XCR0), and on Linux the per-process permission to use tile data.
//!
//! Detection runs once per process; the result is cached in an atomic.

use core::arch::asm;
use core::arch::x86_64::{__cpuid_count, CpuidResult};
use core::sync::atomic::{AtomicU8, Ordering};

use crate::{ROW_BYTES, ROWS};

/// AMX-TILE, its OS state and (Linux) permission, and a palette 1 that fits the fixed shape.
pub(crate) const TILE: u8 = 1 << 0;
pub(crate) const INT8: u8 = 1 << 1;
pub(crate) const BF16: u8 = 1 << 2;
pub(crate) const FP16: u8 = 1 << 3;
/// Set once detection has run, so a cached 0 ("nothing") is distinguishable from "not yet".
const KNOWN: u8 = 1 << 7;

static CACHE: AtomicU8 = AtomicU8::new(0);

/// Bitmask of the `TILE`/`INT8`/`BF16`/`FP16` flags this process may use.
pub(crate) fn features() -> u8 {
    let cached = CACHE.load(Ordering::Acquire);
    if cached & KNOWN != 0 {
        return cached;
    }
    // Racing threads compute the same value; the Linux permission request is idempotent.
    let found = probe() | KNOWN;
    CACHE.store(found, Ordering::Release);
    found
}

fn cpuid(leaf: u32, sub_leaf: u32) -> CpuidResult {
    // SAFETY: CPUID exists on every x86_64 CPU. The intrinsic is a safe fn on current stable
    // (1.98) but still `unsafe` at the 1.89 MSRV, hence the block and the allow.
    #[allow(
        unused_unsafe,
        reason = "__cpuid_count is unsafe at the 1.89 MSRV, safe on newer Rust"
    )]
    unsafe {
        __cpuid_count(leaf, sub_leaf)
    }
}

fn bit(reg: u32, n: u32) -> bool {
    reg >> n & 1 != 0
}

fn probe() -> u8 {
    let max_leaf = cpuid(0, 0).eax;
    // Leaf 7 has the feature bits, 1Dh the palette limits; AMX implies both.
    if max_leaf < 0x1D {
        return 0;
    }
    let l7 = cpuid(7, 0);
    if !bit(l7.edx, 24) || !tile_state_enabled() || !palette1_fits() || !os_permission() {
        return 0;
    }
    let mut found = TILE;
    // The TDP instructions are additionally bounded by the TMUL limits; if they can't take a
    // 16x64 operand the compute tokens stay unavailable even though the memory ops work.
    if tmul_fits(max_leaf) {
        if bit(l7.edx, 25) {
            found |= INT8;
        }
        if bit(l7.edx, 22) {
            found |= BF16;
        }
        // l7.eax is the highest sub-leaf of leaf 7; AMX-FP16 lives in sub-leaf 1.
        if l7.eax >= 1 && bit(cpuid(7, 1).eax, 21) {
            found |= FP16;
        }
    }
    found
}

/// XCR0 bits 17 (XTILECFG) and 18 (XTILEDATA): the OS saves tile state across context switches.
fn tile_state_enabled() -> bool {
    // XGETBV raises #UD unless CR4.OSXSAVE is set, which CPUID.1:ECX[27] mirrors. Checking it
    // here (instead of enabling the `xsave` target feature for `_xgetbv`) keeps this callable
    // from any function.
    if !bit(cpuid(1, 0).ecx, 27) {
        return false;
    }
    let (lo, hi): (u32, u32);
    // SAFETY: OSXSAVE is set, so XGETBV with ECX = 0 (XCR0) is defined.
    unsafe {
        asm!("xgetbv", in("ecx") 0_u32, out("eax") lo, out("edx") hi,
             options(nomem, nostack, preserves_flags));
    }
    let xcr0 = u64::from(hi) << 32 | u64::from(lo);
    xcr0 & (0b11 << 17) == 0b11 << 17
}

/// Leaf 1Dh: palette 1 must have 8 tile names of at least 16 rows x 64 bytes, or the session's
/// fixed configuration would fault in LDTILECFG. Every AMX CPU so far reports exactly that.
fn palette1_fits() -> bool {
    if cpuid(0x1D, 0).eax < 1 {
        return false;
    }
    let p1 = cpuid(0x1D, 1);
    let bytes_per_tile = p1.eax >> 16;
    let bytes_per_row = p1.ebx & 0xFFFF;
    let max_names = p1.ebx >> 16;
    let max_rows = p1.ecx & 0xFFFF;
    bytes_per_tile as usize >= ROWS * ROW_BYTES
        && bytes_per_row as usize >= ROW_BYTES
        && max_rows as usize >= ROWS
        && max_names >= 8
}

/// Leaf 1Eh: TMUL must accept K = 16 rows of B and N = 64 bytes per row.
fn tmul_fits(max_leaf: u32) -> bool {
    if max_leaf < 0x1E {
        return false;
    }
    let tmul = cpuid(0x1E, 0).ebx;
    let maxk = tmul & 0xFF;
    let maxn = tmul >> 8 & 0xFFFF;
    maxk as usize >= ROWS && maxn as usize >= ROW_BYTES
}

/// Linux enables XTILEDATA in XCR0 but traps its first use (XFD) until the process asks for it.
/// The permission is process-wide and permanent, so asking again is harmless.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn os_permission() -> bool {
    const SYS_ARCH_PRCTL: usize = 158;
    const ARCH_REQ_XCOMP_PERM: usize = 0x1023;
    const XFEATURE_XTILEDATA: usize = 18;
    let ret: usize;
    // SAFETY: arch_prctl(ARCH_REQ_XCOMP_PERM) takes two integers and touches no user memory.
    // The syscall instruction clobbers rcx (return address) and r11 (saved RFLAGS).
    unsafe {
        asm!("syscall",
             inlateout("rax") SYS_ARCH_PRCTL => ret,
             in("rdi") ARCH_REQ_XCOMP_PERM,
             in("rsi") XFEATURE_XTILEDATA,
             lateout("rcx") _, lateout("r11") _,
             options(nostack));
    }
    // 0 on success; a negative errno if the kernel predates 5.16 or refuses (e.g. a too-small
    // sigaltstack), in which case tile instructions would raise SIGILL.
    ret == 0
}

/// Windows needs no opt-in: a plain thread runs tile instructions once XCR0 has the tile bits
/// (checked on Server 2025). Other OSes are untested and get the XCR0 check only.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn os_permission() -> bool {
    true
}
