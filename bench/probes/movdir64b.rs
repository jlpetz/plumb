// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Probe, not built by cargo: what an LLVM-visible MOVDIR64B (a future stdarch `_movdir64b`)
//! would change against `asm!`. Build the asm with:
//!
//! ```text
//! rustc +nightly -O --edition 2024 --crate-type=lib --emit=asm -C target-feature=+movdir64b -C llvm-args=-x86-asm-syntax=intel movdir64b.rs
//! ```
//!
//! Result (nightly 1.100, 2026-10-03): the intrinsic loop unrolls 4x and folds the source
//! displacement, but MOVDIR64B's destination is a register operand by encoding, so each line
//! still needs a `lea`: 13 instructions per 4 lines against `asm!`'s 5 per line. The copy is
//! DRAM-bound (RESULTS.md: the same GiB/s in every style), so the gain is ergonomics, not speed.

#![feature(link_llvm_intrinsics)]
#![allow(internal_features)]
#[repr(C, align(64))]
pub struct Line(pub [u64; 8]);
unsafe extern "C" {
    #[link_name = "llvm.x86.movdir64b"]
    fn movdir64b(dst: *mut u8, src: *const u8);
}
/// LLVM-visible MOVDIR64B, as a stdarch `_movdir64b` would be.
#[unsafe(no_mangle)]
pub unsafe fn k_copy_intrinsic(dst: &mut [Line], src: &[Line]) {
    for (d, s) in dst.iter_mut().zip(src) {
        unsafe { movdir64b(d as *mut Line as *mut u8, s as *const Line as *const u8) };
    }
    unsafe { core::arch::x86_64::_mm_sfence() };
}
/// Today's asm! (plumb_lines' `movdir64b`).
#[unsafe(no_mangle)]
pub unsafe fn k_copy_asm(dst: &mut [Line], src: &[Line]) {
    for (d, s) in dst.iter_mut().zip(src) {
        unsafe { core::arch::asm!("movdir64b {d}, zmmword ptr [{s}]", d = in(reg) d as *mut Line, s = in(reg) s as *const Line, options(nostack, preserves_flags)) };
    }
    unsafe { core::arch::x86_64::_mm_sfence() };
}
