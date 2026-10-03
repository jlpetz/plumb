//! Hardware tests for plumb_tiles. They run tile instructions, so each test first asks for an
//! `Amx` token and skips (with a message) when the machine has none.
//!
//! One test binary with modules: they share the helpers below, and the test harness runs them
//! on parallel threads, which also exercises per-thread tile state.

#![cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]

mod compute;
mod detect;
mod guard_page;
mod memory;
mod session;

/// The asm_check kernels, run here so their results are checked on every `cargo test`.
#[path = "../../examples/asm_kernels.rs"]
mod asm_kernels;

#[test]
fn asm_check_kernels_compute_the_right_results() {
    asm_kernels::main();
}

/// The kernels refuse partial buffers instead of skipping the tail.
#[test]
fn asm_check_kernels_reject_partial_buffers() {
    use asm_kernels::kernels::{
        BLOCK_WORDS, TILE_WORDS, k_tile_copy, k_tile_fill, k_tile_stride_fill, k_tile_stride_load,
        k_tile_stride_load_t1,
    };
    let Some(amx) = amx_or_skip("asm_check_kernels_reject_partial_buffers") else {
        return;
    };
    let pattern = [0u64; TILE_WORDS];
    let src = vec![0u64; BLOCK_WORDS + TILE_WORDS];
    let mut dst = vec![0u64; BLOCK_WORDS + TILE_WORDS];
    let whole = "not a whole number of";
    for (name, msg) in [
        (
            "copy, lengths differ",
            panic_message(|| k_tile_copy(amx, &src, &mut dst[..TILE_WORDS])),
        ),
        (
            "copy, partial tile",
            panic_message(|| k_tile_copy(amx, &src[..TILE_WORDS + 1], &mut dst[..TILE_WORDS + 1])),
        ),
        (
            "fill",
            panic_message(|| k_tile_fill(amx, &pattern, &mut dst[..TILE_WORDS - 1])),
        ),
        (
            "stride_load",
            panic_message(|| k_tile_stride_load(amx, &src)),
        ),
        (
            "stride_load_t1",
            panic_message(|| k_tile_stride_load_t1(amx, &src)),
        ),
        (
            "stride_fill",
            panic_message(|| k_tile_stride_fill(amx, &pattern, &mut dst)),
        ),
    ] {
        assert!(
            msg.contains(whole) || msg.contains("lengths differ"),
            "{name}: unexpected panic: {msg}"
        );
    }
    assert!(dst.iter().all(|&w| w == 0), "a refused kernel wrote");
}

use std::panic::{AssertUnwindSafe, catch_unwind};

use plumb_tiles::Amx;

/// The AMX token, or `None` after printing why the test is skipped.
pub fn amx_or_skip(test: &str) -> Option<Amx> {
    let amx = Amx::try_new();
    if amx.is_none() {
        eprintln!("{test}: skipped, AMX is not available on this machine");
    }
    amx
}

/// Deterministic test data: the 64-bit MMIX LCG, high half only (its low bits are weak).
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u32() as u8).collect()
    }
}

/// Runs `f`, which must panic, and returns the panic message.
#[track_caller]
pub fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else {
        String::from("<non-string panic payload>")
    }
}

/// STTILECFG: the live tile configuration of this thread (all zero when released).
pub fn tile_config(_: Amx) -> [u8; 64] {
    let mut cfg = [0xEEu8; 64];
    // SAFETY: the token proves AMX-TILE; STTILECFG writes exactly 64 bytes, configured or not.
    unsafe {
        std::arch::asm!("sttilecfg [{}]", in(reg) cfg.as_mut_ptr(), options(nostack, preserves_flags));
    }
    cfg
}

/// The descriptor every session should load, written out independently of the crate.
pub fn expected_config() -> [u8; 64] {
    let mut b = [0u8; 64];
    b[0] = 1;
    for t in 0..8 {
        b[16 + 2 * t..18 + 2 * t].copy_from_slice(&64u16.to_le_bytes());
        b[48 + t] = 16;
    }
    b
}
