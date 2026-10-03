// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPUID helpers for the capability tokens. `std_detect` knows `clflushopt` but not
//! `movdir64b`, and this crate is `no_std`, so detection reads CPUID directly.

use core::arch::x86_64::{__cpuid, __cpuid_count, CpuidResult};

#[inline]
fn cpuid(leaf: u32, sub: u32) -> CpuidResult {
    // SAFETY: CPUID is available on every x86_64 CPU.
    #[allow(
        unused_unsafe,
        reason = "`__cpuid_count` is unsafe on the 1.89 MSRV, safe on newer toolchains"
    )]
    unsafe {
        __cpuid_count(leaf, sub)
    }
}

#[inline]
fn max_leaf() -> u32 {
    // SAFETY: as above.
    #[allow(
        unused_unsafe,
        reason = "`__cpuid` is unsafe on the 1.89 MSRV, safe on newer toolchains"
    )]
    unsafe {
        __cpuid(0).eax
    }
}

/// `CPUID.(EAX=7,ECX=0):EBX[23]`.
pub fn has_clflushopt() -> bool {
    max_leaf() >= 7 && cpuid(7, 0).ebx & (1 << 23) != 0
}

/// `CPUID.(EAX=7,ECX=0):ECX[28]`.
pub fn has_movdir64b() -> bool {
    max_leaf() >= 7 && cpuid(7, 0).ecx & (1 << 28) != 0
}

/// The CLFLUSH line size, `CPUID.1:EBX[15:8]` x 8 bytes. That is the granularity CLFLUSH and
/// CLFLUSHOPT operate on (64 on every current x86 CPU). The range flush masks addresses with it,
/// so anything that isn't a power of two in 32..=4096 (0, or an odd hypervisor value) becomes 64.
pub fn flush_line_bytes() -> usize {
    let bytes = ((cpuid(1, 0).ebx >> 8) & 0xff) as usize * 8;
    if bytes.is_power_of_two() && (32..=4096).contains(&bytes) {
        bytes
    } else {
        64
    }
}
