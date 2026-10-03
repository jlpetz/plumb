// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Detection against raw CPUID/XCR0 read independently of the crate.

use std::arch::asm;
use std::arch::x86_64::{__cpuid_count, CpuidResult};

use plumb_tiles::{Amx, AmxBf16, AmxFp16, AmxInt8};

fn cpuid(leaf: u32, sub_leaf: u32) -> CpuidResult {
    #[allow(
        unused_unsafe,
        reason = "__cpuid_count is unsafe at the 1.89 MSRV, safe on newer Rust"
    )]
    // SAFETY: CPUID exists on every x86_64 CPU.
    unsafe {
        __cpuid_count(leaf, sub_leaf)
    }
}

fn os_tile_state() -> bool {
    if cpuid(1, 0).ecx >> 27 & 1 == 0 {
        return false;
    }
    let (lo, hi): (u32, u32);
    // SAFETY: OSXSAVE is set, so XGETBV(0) is defined.
    unsafe {
        asm!("xgetbv", in("ecx") 0_u32, out("eax") lo, out("edx") hi, options(nomem, nostack));
    };
    (u64::from(hi) << 32 | u64::from(lo)) >> 17 & 0b11 == 0b11
}

/// Leaf 1Dh.1, palette 1, holds the fixed 16 x 64 shape in all 8 tiles. AMX-TILE alone isn't
/// enough: an ACE-only CPU sets it too, with palette 1 all zeros.
fn palette1_fits(max_leaf: u32) -> bool {
    if max_leaf < 0x1D || cpuid(0x1D, 0).eax < 1 {
        return false;
    }
    let p1 = cpuid(0x1D, 1);
    p1.eax >> 16 >= 1024 && p1.ebx & 0xFFFF >= 64 && p1.ebx >> 16 >= 8 && p1.ecx & 0xFFFF >= 16
}

/// Leaf 1Eh: TMUL takes K = 16 rows of B and 64 bytes per row.
fn tmul_fits(max_leaf: u32) -> bool {
    let tmul = if max_leaf >= 0x1E {
        cpuid(0x1E, 0).ebx
    } else {
        0
    };
    tmul & 0xFF >= 16 && tmul >> 8 & 0xFFFF >= 64
}

/// Linux: the per-process XTILEDATA permission, requested here independently of the crate (the
/// request is idempotent). x86_64 Linux syscall ABI; compile-checked only.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn os_permission() -> bool {
    let ret: i64;
    // SAFETY: arch_prctl(ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18) takes two
    // integers and touches no user memory; syscall clobbers rcx and r11.
    unsafe {
        asm!("syscall", inlateout("rax") 158i64 => ret, in("rdi") 0x1023u64, in("rsi") 18u64,
             lateout("rcx") _, lateout("r11") _, options(nostack));
    }
    ret == 0
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn os_permission() -> bool {
    true
}

#[test]
fn detection_matches_raw_cpuid_and_xcr0() {
    let max_leaf = cpuid(0, 0).eax;
    let l7 = cpuid(7, 0);
    let tile =
        l7.edx >> 24 & 1 == 1 && os_tile_state() && palette1_fits(max_leaf) && os_permission();
    let tmul = tile && tmul_fits(max_leaf);
    let int8 = tmul && l7.edx >> 25 & 1 == 1;
    let bf16 = tmul && l7.edx >> 22 & 1 == 1;
    let fp16 = tmul && l7.eax >= 1 && cpuid(7, 1).eax >> 21 & 1 == 1;
    eprintln!("raw: tile={tile} int8={int8} bf16={bf16} fp16={fp16}");

    let amx = Amx::try_new();
    assert_eq!(amx.is_some(), tile, "Amx::try_new");
    assert_eq!(AmxInt8::try_new().is_some(), int8, "AmxInt8::try_new");
    assert_eq!(AmxBf16::try_new().is_some(), bf16, "AmxBf16::try_new");
    assert_eq!(AmxFp16::try_new().is_some(), fp16, "AmxFp16::try_new");
    if let Some(amx) = amx {
        assert_eq!(amx.int8().is_some(), int8);
        assert_eq!(amx.bf16().is_some(), bf16);
        assert_eq!(amx.fp16().is_some(), fp16);
    }
}

/// On a CPU that advertises AMX the tile limits must be the ones the fixed shape relies on.
#[test]
fn palette_and_tmul_limits_fit_the_fixed_shape() {
    let Some(_) = crate::amx_or_skip("palette_and_tmul_limits_fit_the_fixed_shape") else {
        return;
    };
    let p1 = cpuid(0x1D, 1);
    let tmul = cpuid(0x1E, 0).ebx;
    eprintln!(
        "palette 1: total {} B, {} B/tile, {} B/row, {} names, {} rows; tmul maxk {}, maxn {}",
        p1.eax & 0xFFFF,
        p1.eax >> 16,
        p1.ebx & 0xFFFF,
        p1.ebx >> 16,
        p1.ecx & 0xFFFF,
        tmul & 0xFF,
        tmul >> 8 & 0xFFFF
    );
    assert!(p1.ebx & 0xFFFF >= 64 && p1.ecx & 0xFFFF >= 16 && p1.ebx >> 16 >= 8);
    assert!(tmul & 0xFF >= 16 && tmul >> 8 & 0xFFFF >= 64);
}

#[test]
fn detection_is_stable_across_threads() {
    let here = (
        Amx::try_new().is_some(),
        AmxInt8::try_new().is_some(),
        AmxFp16::try_new().is_some(),
    );
    let threads: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                (
                    Amx::try_new().is_some(),
                    AmxInt8::try_new().is_some(),
                    AmxFp16::try_new().is_some(),
                )
            })
        })
        .collect();
    for t in threads {
        assert_eq!(t.join().unwrap(), here);
    }
}

#[test]
fn tokens_are_copy_send_sync_and_zero_sized() {
    fn check<T: Copy + Send + Sync + std::fmt::Debug>() {
        assert_eq!(size_of::<T>(), 0);
    }
    check::<Amx>();
    check::<AmxInt8>();
    check::<AmxBf16>();
    check::<AmxFp16>();
    if let Some(i8) = AmxInt8::try_new() {
        let amx: Amx = i8.into();
        let _ = (amx, i8.amx());
    }
}
