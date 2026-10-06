// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The raw tile instructions, in two implementations with the same signatures: `asm!` on
//! stable (`insn/asm.rs`) and stdarch's intrinsics with the `nightly` feature
//! (`insn/intrin.rs`).
//!
//! Both are as conservative about ordering as the session needs. The tile registers are
//! invisible to the compiler, so "load T0, then store T0" stays in that order only because
//! every tile instruction is treated as a side effect the compiler never reorders or drops.
//!
//! Callers have checked the span (memory operations) and hold a session (everything except
//! the configuration operations).

#[cfg(not(feature = "nightly"))]
mod asm;
#[cfg(not(feature = "nightly"))]
pub(crate) use asm::{
    ldtilecfg, sttilecfg, tdpbf16ps, tdpbssd, tdpbsud, tdpbusd, tdpbuud, tdpfp16ps, tileloadd,
    tileloaddt1, tilerelease, tilestored, tilezero,
};

#[cfg(feature = "nightly")]
mod intrin;
#[cfg(feature = "nightly")]
pub(crate) use intrin::{
    ldtilecfg, sttilecfg, tdpbf16ps, tdpbssd, tdpbsud, tdpbusd, tdpbuud, tdpfp16ps, tileloadd,
    tileloaddt1, tilerelease, tilestored, tilezero,
};
