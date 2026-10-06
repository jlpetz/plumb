// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stable: every tile instruction is an `asm!` block with the tile number as a `const`
//! operand, so no target feature or nightly intrinsic is involved.
//!
//! None of the blocks is `pure`, so the compiler never reorders or drops them. The memory
//! options then say what each instruction does to memory the compiler can see: loads (and
//! `LDTILECFG`) are `readonly` (pending stores to the source are completed first), stores (and
//! `STTILECFG`) claim nothing, and zero/compute/release are `nomem`. No tile instruction
//! touches the stack or RFLAGS.

use core::arch::asm;

use crate::reg::TileReg;

/// `LDTILECFG`: configures the tiles from the 64-byte descriptor at `cfg`.
#[inline(always)]
pub(crate) unsafe fn ldtilecfg(cfg: *const u8) {
    // SAFETY (caller): the CPU has AMX-TILE with OS-enabled tile state, and `cfg` is 64
    // readable bytes of a descriptor it accepts.
    unsafe {
        asm!("ldtilecfg [{cfg}]", cfg = in(reg) cfg,
             options(nostack, readonly, preserves_flags));
    }
}

/// `STTILECFG`: writes this thread's live tile configuration (all zero when released) to the
/// 64 bytes at `cfg`.
#[inline(always)]
pub(crate) unsafe fn sttilecfg(cfg: *mut u8) {
    // SAFETY (caller): the CPU has AMX-TILE with OS-enabled tile state, and `cfg` is 64
    // writable bytes. The block isn't readonly, so the compiler treats them as written.
    unsafe {
        asm!("sttilecfg [{cfg}]", cfg = in(reg) cfg, options(nostack, preserves_flags));
    }
}

/// `TILERELEASE`: returns the tiles to their initial, unconfigured state.
#[inline(always)]
pub(crate) unsafe fn tilerelease() {
    // SAFETY (caller): the CPU has AMX-TILE. TILERELEASE is defined configured or not and
    // touches no memory.
    unsafe { asm!("tilerelease", options(nostack, nomem, preserves_flags)) };
}

/// `TILELOADD`: loads tile `T`, row `r` from `src + r * stride`.
#[inline(always)]
pub(crate) unsafe fn tileloadd<T: TileReg>(src: *const u8, stride: usize) {
    // SAFETY (caller): src + r*stride .. +ROW_BYTES is readable for r < ROWS; tiles configured.
    unsafe {
        asm!("tileloadd tmm{t}, [{src} + {stride}*1]", t = const T::N,
             src = in(reg) src, stride = in(reg) stride,
             options(nostack, readonly, preserves_flags));
    }
}

/// `TILELOADDT1`: [`tileloadd`] with the T1 (low reuse) hint.
#[inline(always)]
pub(crate) unsafe fn tileloaddt1<T: TileReg>(src: *const u8, stride: usize) {
    // SAFETY (caller): as tileloadd.
    unsafe {
        asm!("tileloaddt1 tmm{t}, [{src} + {stride}*1]", t = const T::N,
             src = in(reg) src, stride = in(reg) stride,
             options(nostack, readonly, preserves_flags));
    }
}

/// `TILESTORED`: stores tile `T`, row `r` to `dst + r * stride`.
#[inline(always)]
pub(crate) unsafe fn tilestored<T: TileReg>(dst: *mut u8, stride: usize) {
    // SAFETY (caller): dst + r*stride .. +ROW_BYTES is writable for r < ROWS; tiles configured.
    unsafe {
        asm!("tilestored [{dst} + {stride}*1], tmm{t}", t = const T::N,
             dst = in(reg) dst, stride = in(reg) stride,
             options(nostack, preserves_flags));
    }
}

/// `TILEZERO`: sets every byte of tile `T` to zero.
#[inline(always)]
pub(crate) unsafe fn tilezero<T: TileReg>() {
    // SAFETY (caller): tiles configured; TILEZERO touches no memory.
    unsafe { asm!("tilezero tmm{t}", t = const T::N, options(nostack, nomem, preserves_flags)) }
}

macro_rules! tdp {
    ($($name:ident),*) => {$(
        #[doc = concat!("`", stringify!($name), " C, A, B`.")]
        #[inline(always)]
        pub(crate) unsafe fn $name<C: TileReg, A: TileReg, B: TileReg>() {
            // SAFETY (caller): the CPU has the instruction, the tiles are configured with
            // shapes the TMUL limits accept, and C, A and B are three different tiles. It
            // touches only tile registers.
            unsafe {
                asm!(concat!(stringify!($name), " tmm{c}, tmm{a}, tmm{b}"),
                     c = const C::N, a = const A::N, b = const B::N,
                     options(nostack, nomem, preserves_flags));
            }
        }
    )*};
}

tdp!(tdpbssd, tdpbsud, tdpbusd, tdpbuud, tdpbf16ps, tdpfp16ps);
