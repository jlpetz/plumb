// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The same kernels written with fearless_simd: **one generic body per kernel**, the width
//! chosen by the vector type `V` (`u64x2/u64x4/u64x8<S>`) and the instruction set by the token
//! `S` (`Avx2`, `Avx512`). No per-width macro, no `#[target_feature]`, slices instead of raw
//! pointers where the instruction allows it.
//!
//! What fearless_simd doesn't have (NT stores, CLFLUSHOPT, MOVDIR64B) is added here the way the
//! library adds its own ops: a trait method that is `#[inline(always)]` and wraps a `kernel!`.
//!
//! The `k_*` functions at the bottom are thin named entry points for the harness and
//! `asm_check.py`. With `#[simd]`, the hot loop lives in a target-feature helper the wrapper
//! calls, not in the wrapper itself; the checker follows that call.

use crate::common::*;
use fearless_simd::prelude::*;
use fearless_simd::{Avx2, Avx512, Level, dispatch, kernel, u64x2, u64x4, u64x8};
use fearless_simd_macros::simd;
use std::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// Generic kernels: one body each, covering every width and level.
// ---------------------------------------------------------------------------------------------

/// Constant fill (write phase of `stuck_bit_write_verify!`).
#[simd]
pub fn fill<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &mut [u64], pat: u64) {
    let p = V::splat(simd, pat);
    for c in buf.chunks_exact_mut(V::LEN) {
        p.store_slice(c);
    }
}

/// 4-accumulator verify (verify phase of `stuck_bit_write_verify!`). Returns 1 on mismatch.
#[simd]
pub fn verify4<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &[u64], pat: u64) -> u64 {
    let w = V::LEN;
    let p = V::splat(simd, pat);
    let z = V::splat(simd, 0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let mut quads = buf.chunks_exact(4 * w);
    for q in &mut quads {
        a0 |= V::from_slice(simd, &q[..w]) ^ p;
        a1 |= V::from_slice(simd, &q[w..2 * w]) ^ p;
        a2 |= V::from_slice(simd, &q[2 * w..3 * w]) ^ p;
        a3 |= V::from_slice(simd, &q[3 * w..]) ^ p;
    }
    for c in quads.remainder().chunks_exact(w) {
        a0 |= V::from_slice(simd, c) ^ p;
    }
    let acc = (a0 | a1) | (a2 | a3);
    acc.simd_eq(z).any_false() as u64
}

/// Positional write, Mode 0/1 (`simple_write_positional_simd!`).
#[simd]
pub fn pos_write<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &mut [u64], base: u64) {
    let base_v = V::splat(simd, base);
    let step = V::splat(simd, V::LEN as u64);
    let mut idx = V::from_fn(simd, |i| i as u64);
    for c in buf.chunks_exact_mut(V::LEN) {
        (idx ^ base_v).store_slice(c);
        idx += step;
    }
}

/// Positional verify, single accumulator (`simple_verify_positional_simd!`, no check mask).
#[simd]
pub fn pos_verify<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &[u64], base: u64) -> u64 {
    let z = V::splat(simd, 0);
    let base_v = V::splat(simd, base);
    let step = V::splat(simd, V::LEN as u64);
    let mut idx = V::from_fn(simd, |i| i as u64);
    let mut acc = z;
    for c in buf.chunks_exact(V::LEN) {
        acc |= V::from_slice(simd, c) ^ (idx ^ base_v);
        idx += step;
    }
    acc.simd_eq(z).any_false() as u64
}

/// LCG write, Mode 2 (`simple_write_lcg_simd!`): exercises the 64-bit lane multiply.
#[simd]
pub fn lcg_write<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &mut [u64]) {
    let (seeds, m, a) = lcg_lanes(V::LEN);
    let mut state = V::from_slice(simd, &seeds[..V::LEN]);
    let mul = V::splat(simd, m);
    let add = V::splat(simd, a);
    for c in buf.chunks_exact_mut(V::LEN) {
        state.store_slice(c);
        state = state * mul + add;
    }
}

/// 4-accumulator verify with a software prefetch per cache line. `_mm_prefetch` is SSE, so it
/// inlines anywhere without `kernel!`, but a `#[simd]` body still needs `unsafe` to call it.
#[simd]
pub fn verify4_pf<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &[u64], pat: u64) -> u64 {
    let w = V::LEN;
    let lines = (4 * size_of::<V>() / LINE).max(1);
    let p = V::splat(simd, pat);
    let z = V::splat(simd, 0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let mut quads = buf.chunks_exact(4 * w);
    for q in &mut quads {
        let ahead = (q.as_ptr() as *const i8).wrapping_add(PF_DIST);
        for k in 0..lines {
            // SAFETY: SSE is in every x86_64 baseline; `#[simd]` bodies still need `unsafe`.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(ahead.wrapping_add(k * LINE)) };
        }
        a0 |= V::from_slice(simd, &q[..w]) ^ p;
        a1 |= V::from_slice(simd, &q[w..2 * w]) ^ p;
        a2 |= V::from_slice(simd, &q[2 * w..3 * w]) ^ p;
        a3 |= V::from_slice(simd, &q[3 * w..]) ^ p;
    }
    for c in quads.remainder().chunks_exact(w) {
        a0 |= V::from_slice(simd, c) ^ p;
    }
    let acc = (a0 | a1) | (a2 | a3);
    acc.simd_eq(z).any_false() as u64
}

/// Native width for the level (`S::u64s`): this is fearless_simd's version of TMR's Auto.
#[simd]
pub fn fill_native<S: Simd>(simd: S, buf: &mut [u64], pat: u64) {
    fill::<S, S::u64s>(simd, buf, pat)
}

#[simd]
pub fn verify4_native<S: Simd>(simd: S, buf: &[u64], pat: u64) -> u64 {
    verify4::<S, S::u64s>(simd, buf, pat)
}

// ---------------------------------------------------------------------------------------------
// Extension 1: non-temporal store, the op fearless_simd lacks.
// ---------------------------------------------------------------------------------------------

/// One vector NT store. Implemented per (level, width) with `kernel!`, exactly how
/// fearless_simd implements `xor_u64x8` and friends, so generic code calling it inlines it.
///
/// `kernel!` only accepts *safe* fns, but every NT store takes a raw pointer, so the unsafety
/// moves into the trait method's contract and an `unsafe` block inside the kernel body.
pub trait NtStore<S: Simd>: SimdBase<S> {
    /// # Safety
    /// `dst` must be writable for `size_of::<Self>()` bytes and aligned to that size, and the
    /// calling thread must `_mm_sfence()` before anything reads or publishes that memory.
    unsafe fn nt_store(self, dst: *mut u64);
}

macro_rules! nt_store_impl {
    ($V:ident, $Tok:ident, $arch:ty, $stream:ident) => {
        impl NtStore<$Tok> for $V<$Tok> {
            #[inline(always)]
            unsafe fn nt_store(self, dst: *mut u64) {
                kernel!(
                    #[inline(always)]
                    #[allow(clippy::not_unsafe_ptr_arg_deref, reason = "contract is on nt_store")]
                    fn k(_t: $Tok, v: $V<$Tok>, dst: *mut u64) {
                        // SAFETY: forwarded from nt_store's contract.
                        unsafe { $stream(dst.cast::<$arch>(), v.into()) }
                    }
                );
                k(self.simd, self, dst)
            }
        }
    };
}

nt_store_impl!(u64x2, Avx2, __m128i, _mm_stream_si128);
nt_store_impl!(u64x4, Avx2, __m256i, _mm256_stream_si256);
nt_store_impl!(u64x2, Avx512, __m128i, _mm_stream_si128);
nt_store_impl!(u64x4, Avx512, __m256i, _mm256_stream_si256);
nt_store_impl!(u64x8, Avx512, __m512i, _mm512_stream_si512);

/// NT positional write (`simple_write_nt_positional_simd!`): same 4x manual unroll, because
/// the stream intrinsics are `asm!` inside stdarch and LLVM won't unroll across them.
#[simd]
pub fn nt_pos_write<S: Simd, V: SimdInt<S, Element = u64> + NtStore<S>>(
    simd: S,
    buf: &mut [u64],
    base: u64,
) {
    let w = V::LEN;
    assert_eq!(
        buf.as_ptr() as usize % size_of::<V>(),
        0,
        "NT stores need vector alignment"
    );
    assert_eq!(buf.len() % w, 0);
    let base_v = V::splat(simd, base);
    let step1 = V::splat(simd, w as u64);
    let step4 = V::splat(simd, 4 * w as u64);
    let mut idx = V::from_fn(simd, |i| i as u64);
    let p = buf.as_mut_ptr();
    let n = buf.len();
    let unrolled_end = n / (4 * w) * (4 * w);
    let mut i = 0;
    while i < unrolled_end {
        let v0 = idx ^ base_v;
        let i1 = idx + step1;
        let v1 = i1 ^ base_v;
        let i2 = i1 + step1;
        let v2 = i2 ^ base_v;
        let i3 = i2 + step1;
        let v3 = i3 ^ base_v;
        // SAFETY: i + 4w <= n, buf is vector-aligned (asserted), sfence below.
        unsafe {
            v0.nt_store(p.add(i));
            v1.nt_store(p.add(i + w));
            v2.nt_store(p.add(i + 2 * w));
            v3.nt_store(p.add(i + 3 * w));
        }
        idx += step4;
        i += 4 * w;
    }
    while i < n {
        // SAFETY: as above, i + w <= n.
        unsafe { (idx ^ base_v).nt_store(p.add(i)) };
        idx += step1;
        i += w;
    }
    // SAFETY: SSE is in every x86_64 baseline; `#[simd]` bodies still need `unsafe` (E0133).
    unsafe { _mm_sfence() };
}

kernel!(
    /// The same NT write as a plain `kernel!` (no trait): the "drop to intrinsics" route.
    pub fn nt_pos_write_k512(t: Avx512, buf: &mut [u64], base: u64) {
        assert_eq!(
            buf.as_ptr() as usize % 64,
            0,
            "NT stores need 64-byte alignment"
        );
        let base_v = u64x8::splat(t, base);
        let step = u64x8::splat(t, 8);
        let mut idx = u64x8::from_fn(t, |i| i as u64);
        let (quads, rest) = buf.as_chunks_mut::<32>();
        for q in quads {
            let p = q.as_mut_ptr().cast::<__m512i>();
            // SAFETY: q is 32 u64 = 4 aligned lines; sfence below.
            unsafe {
                _mm512_stream_si512(p, (idx ^ base_v).into());
                _mm512_stream_si512(p.add(1), ((idx + step) ^ base_v).into());
                _mm512_stream_si512(p.add(2), ((idx + step + step) ^ base_v).into());
                _mm512_stream_si512(p.add(3), ((idx + step + step + step) ^ base_v).into());
            }
            idx += step + step + step + step;
        }
        for c in rest.as_chunks_mut::<8>().0 {
            // SAFETY: c is one aligned line; sfence below.
            unsafe { _mm512_stream_si512(c.as_mut_ptr().cast(), (idx ^ base_v).into()) };
            idx += step;
        }
        _mm_sfence();
    }
);

/// DRAM -> DRAM copy: vector load + NT store, then sfence.
#[simd]
pub fn copy_nt<S: Simd, V: SimdInt<S, Element = u64> + NtStore<S>>(
    simd: S,
    dst: &mut [u64],
    src: &[u64],
) {
    assert_eq!(dst.len(), src.len());
    assert_eq!(
        dst.as_ptr() as usize % size_of::<V>(),
        0,
        "NT stores need vector alignment"
    );
    let d = dst.as_mut_ptr();
    for (i, c) in src.chunks_exact(V::LEN).enumerate() {
        // SAFETY: i * LEN + LEN <= dst.len(), aligned (asserted), sfence below.
        unsafe { V::from_slice(simd, c).nt_store(d.add(i * V::LEN)) };
    }
    // SAFETY: SSE is in every x86_64 baseline; `#[simd]` bodies still need `unsafe` (E0133).
    unsafe { _mm_sfence() };
}

// ---------------------------------------------------------------------------------------------
// Extension 2: CLFLUSHOPT. No fearless_simd token lists `clflushopt` (it isn't in any x86-64
// psABI level), so the *intrinsic* can't inline into a fearless kernel. Two routes compared:
// ---------------------------------------------------------------------------------------------

kernel!(
    /// Route A: the stdarch intrinsic inside `kernel!(Avx2)`. The kernel's features don't
    /// include `clflushopt`, so LLVM may refuse to inline `_mm_clflushopt` (asm_check shows which).
    pub fn flush_intr(t: Avx2, buf: &[u64]) {
        let p = buf.as_ptr() as *const u8;
        for i in 0..(buf.len() * 8).div_ceil(LINE) {
            // SAFETY: inside `buf`; CLFLUSHOPT is gated at startup like TMR's.
            unsafe { _mm_clflushopt(p.add(i * LINE)) };
        }
        _mm_mfence();
    }
);

kernel!(
    /// Route B: inline `asm!` inside `kernel!(Avx2)`; asm needs no target feature.
    pub fn flush_asm(t: Avx2, buf: &[u64]) {
        let p = buf.as_ptr() as *const u8;
        for i in 0..(buf.len() * 8).div_ceil(LINE) {
            // SAFETY: inside `buf`; CLFLUSHOPT is gated at startup like TMR's.
            unsafe { clflushopt_asm(p.add(i * LINE)) };
        }
        _mm_mfence();
    }
);

/// Mixed write + flush per line, generic, route A (intrinsic).
#[simd]
pub fn wflush_intr<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &mut [u64], pat: u64) {
    let p = V::splat(simd, pat);
    for line in buf.as_chunks_mut::<{ LINE / 8 }>().0 {
        for c in line.chunks_exact_mut(V::LEN) {
            p.store_slice(c);
        }
        // SAFETY: line is inside buf.
        unsafe { _mm_clflushopt(line.as_ptr() as *const u8) };
    }
    // SAFETY: SSE2 is in every x86_64 baseline; `#[simd]` bodies still need `unsafe` (E0133).
    unsafe { _mm_mfence() };
}

/// Mixed write + flush per line, generic, route B (asm).
#[simd]
pub fn wflush_asm<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &mut [u64], pat: u64) {
    let p = V::splat(simd, pat);
    for line in buf.as_chunks_mut::<{ LINE / 8 }>().0 {
        for c in line.chunks_exact_mut(V::LEN) {
            p.store_slice(c);
        }
        // SAFETY: line is inside buf.
        unsafe { clflushopt_asm(line.as_ptr() as *const u8) };
    }
    // SAFETY: SSE2 is in every x86_64 baseline; `#[simd]` bodies still need `unsafe` (E0133).
    unsafe { _mm_mfence() };
}

// ---------------------------------------------------------------------------------------------
// Extension 3: MOVDIR64B (asm only; stdarch has no intrinsic).
// ---------------------------------------------------------------------------------------------

kernel!(
    pub fn copy_movdir(t: Avx2, dst: &mut [u64], src: &[u64]) {
        assert_eq!(dst.len(), src.len());
        assert_eq!(
            dst.as_ptr() as usize % 64,
            0,
            "MOVDIR64B needs a 64-byte aligned dst"
        );
        let d = dst.as_mut_ptr();
        for (i, line) in src.as_chunks::<{ LINE / 8 }>().0.iter().enumerate() {
            // SAFETY: both lines in bounds; dst aligned (asserted); sfence below. The caller
            // checks CPUID for MOVDIR64B.
            unsafe { movdir64b(d.add(i * LINE / 8) as *mut u8, line.as_ptr() as *const u8) };
        }
        _mm_sfence();
    }
);

// ---------------------------------------------------------------------------------------------
// Deliberate footgun: a generic helper that is not `#[simd]` and not `#[inline(always)]`.
// It compiles at the x86-64-v3 baseline, so the Avx512 ops inside it can't inline and become
// calls into fearless_simd's kernel fns. Width is kept (those fns have the features); speed
// is not. Compare `tmr::k_verify4_tmrhelper_512`, where the same mistake silently drops to ymm.
// ---------------------------------------------------------------------------------------------

#[inline(never)]
fn or_xor_step<S: Simd, V: SimdInt<S, Element = u64>>(a: V, x: V, p: V) -> V {
    a | (x ^ p)
}

#[simd]
pub fn verify4_helper<S: Simd, V: SimdInt<S, Element = u64>>(
    simd: S,
    buf: &[u64],
    pat: u64,
) -> u64 {
    let w = V::LEN;
    let p = V::splat(simd, pat);
    let z = V::splat(simd, 0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    for q in buf.chunks_exact(4 * w) {
        a0 = or_xor_step(a0, V::from_slice(simd, &q[..w]), p);
        a1 = or_xor_step(a1, V::from_slice(simd, &q[w..2 * w]), p);
        a2 = or_xor_step(a2, V::from_slice(simd, &q[2 * w..3 * w]), p);
        a3 = or_xor_step(a3, V::from_slice(simd, &q[3 * w..]), p);
    }
    let acc = (a0 | a1) | (a2 | a3);
    acc.simd_eq(z).any_false() as u64
}

// ---------------------------------------------------------------------------------------------
// Named entry points (harness + asm_check.py). The 128-bit variants take the Avx2 token, like
// TMR's 128 variant, which also compiles at the v3 baseline (VEX xmm, 16 registers).
// ---------------------------------------------------------------------------------------------

macro_rules! entry {
    ($name:ident, $Tok:ident, $V:ident, $kernel:ident, ($($arg:ident: $ty:ty),*) $(-> $ret:ty)?, ($($call:expr),*)) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name(t: $Tok, $($arg: $ty),*) $(-> $ret)? {
            $kernel::<$Tok, $V<$Tok>>(t, $($call),*)
        }
    };
}

entry!(k_fill_fs_128, Avx2, u64x2, fill, (buf: &mut [u64]), (buf, PATTERN));
entry!(k_fill_fs_256, Avx2, u64x4, fill, (buf: &mut [u64]), (buf, PATTERN));
entry!(k_fill_fs_512, Avx512, u64x8, fill, (buf: &mut [u64]), (buf, PATTERN));
entry!(k_filluni_fs_256, Avx2, u64x4, fill, (buf: &mut [u64]), (buf, UNIFORM));
entry!(k_filluni_fs_512, Avx512, u64x8, fill, (buf: &mut [u64]), (buf, UNIFORM));

entry!(k_verify4_fs_128, Avx2, u64x2, verify4, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_fs_256, Avx2, u64x4, verify4, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_fs_512, Avx512, u64x8, verify4, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_fshelper_512, Avx512, u64x8, verify4_helper, (buf: &[u64]) -> u64, (buf, PATTERN));

entry!(k_posw_fs_128, Avx2, u64x2, pos_write, (buf: &mut [u64]), (buf, POS_BASE));
entry!(k_posw_fs_256, Avx2, u64x4, pos_write, (buf: &mut [u64]), (buf, POS_BASE));
entry!(k_posw_fs_512, Avx512, u64x8, pos_write, (buf: &mut [u64]), (buf, POS_BASE));

entry!(k_posv_fs_128, Avx2, u64x2, pos_verify, (buf: &[u64]) -> u64, (buf, POS_BASE));
entry!(k_posv_fs_256, Avx2, u64x4, pos_verify, (buf: &[u64]) -> u64, (buf, POS_BASE));
entry!(k_posv_fs_512, Avx512, u64x8, pos_verify, (buf: &[u64]) -> u64, (buf, POS_BASE));

entry!(k_lcgw_fs_128, Avx2, u64x2, lcg_write, (buf: &mut [u64]), (buf));
entry!(k_lcgw_fs_256, Avx2, u64x4, lcg_write, (buf: &mut [u64]), (buf));
entry!(k_lcgw_fs_512, Avx512, u64x8, lcg_write, (buf: &mut [u64]), (buf));

entry!(k_pfv_fs_256, Avx2, u64x4, verify4_pf, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_pfv_fs_512, Avx512, u64x8, verify4_pf, (buf: &[u64]) -> u64, (buf, PATTERN));

entry!(k_ntw_fs_128, Avx2, u64x2, nt_pos_write, (buf: &mut [u64]), (buf, POS_BASE));
entry!(k_ntw_fs_256, Avx2, u64x4, nt_pos_write, (buf: &mut [u64]), (buf, POS_BASE));
entry!(k_ntw_fs_512, Avx512, u64x8, nt_pos_write, (buf: &mut [u64]), (buf, POS_BASE));

entry!(k_wflush_fsintr_256, Avx2, u64x4, wflush_intr, (buf: &mut [u64]), (buf, PATTERN));
entry!(k_wflush_fsintr_512, Avx512, u64x8, wflush_intr, (buf: &mut [u64]), (buf, PATTERN));
entry!(k_wflush_fsasm_256, Avx2, u64x4, wflush_asm, (buf: &mut [u64]), (buf, PATTERN));
entry!(k_wflush_fsasm_512, Avx512, u64x8, wflush_asm, (buf: &mut [u64]), (buf, PATTERN));

entry!(k_copynt_fs_512, Avx512, u64x8, copy_nt, (dst: &mut [u64], src: &[u64]), (dst, src));

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_ntw_fsk_512(t: Avx512, buf: &mut [u64]) {
    nt_pos_write_k512(t, buf, POS_BASE)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_flush_fsintr(t: Avx2, buf: &[u64]) {
    flush_intr(t, buf)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_flush_fsasm(t: Avx2, buf: &[u64]) {
    flush_asm(t, buf)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_copymd_fs(t: Avx2, dst: &mut [u64], src: &[u64]) {
    copy_movdir(t, dst, src)
}

/// Auto: runtime dispatch to the best level, native width. On an AVX-512 (Ice Lake+) CPU this
/// is Avx512 + u64x8; on Skylake-X/Cascade Lake fearless_simd picks Avx2 (its AVX-512 policy).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_fill_fs_auto(level: Level, buf: &mut [u64]) {
    dispatch!(level, simd => fill_native(simd, buf, PATTERN))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_verify4_fs_auto(level: Level, buf: &[u64]) -> u64 {
    dispatch!(level, simd => verify4_native(simd, buf, PATTERN))
}

// ---------------------------------------------------------------------------------------------
// Loop-shape experiment for the 512-bit verify gap: same body, different walk. `chunks_exact`
// gives indexed addressing (`[r9 + 8*r10 + 64]`); these two try to get base+disp like TMR's
// pointer walk.
// ---------------------------------------------------------------------------------------------

/// Safe walk: peel 4 vectors off the front with `split_at` each iteration.
#[simd]
pub fn verify4_split<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &[u64], pat: u64) -> u64 {
    let w = V::LEN;
    let p = V::splat(simd, pat);
    let z = V::splat(simd, 0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let mut rest = buf;
    while rest.len() >= 4 * w {
        let (q, r) = rest.split_at(4 * w);
        a0 |= V::from_slice(simd, &q[..w]) ^ p;
        a1 |= V::from_slice(simd, &q[w..2 * w]) ^ p;
        a2 |= V::from_slice(simd, &q[2 * w..3 * w]) ^ p;
        a3 |= V::from_slice(simd, &q[3 * w..]) ^ p;
        rest = r;
    }
    for c in rest.chunks_exact(w) {
        a0 |= V::from_slice(simd, c) ^ p;
    }
    let acc = (a0 | a1) | (a2 | a3);
    acc.simd_eq(z).any_false() as u64
}

/// Raw-pointer walk, the shape of TMR's `stuck_bit_write_verify!`.
#[simd]
pub fn verify4_ptr<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &[u64], pat: u64) -> u64 {
    let w = V::LEN;
    let p = V::splat(simd, pat);
    let z = V::splat(simd, 0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let n = buf.len() / w * w;
    let mut ptr = buf.as_ptr();
    // SAFETY: `end` is within (or one past) `buf`.
    let end = unsafe { ptr.add(n) };
    // SAFETY (loads below): every `ptr.add(k * w)` read is < `end`, and `V::Array` is `[u64; LEN]`.
    let ld = |q: *const u64| V::load_array_ref(simd, unsafe { &*(q as *const V::Array) });
    while (end as usize - ptr as usize) >= 4 * w * 8 {
        a0 |= ld(ptr) ^ p;
        a1 |= ld(unsafe { ptr.add(w) }) ^ p;
        a2 |= ld(unsafe { ptr.add(2 * w) }) ^ p;
        a3 |= ld(unsafe { ptr.add(3 * w) }) ^ p;
        ptr = unsafe { ptr.add(4 * w) };
    }
    while ptr < end {
        a0 |= ld(ptr) ^ p;
        ptr = unsafe { ptr.add(w) };
    }
    let acc = (a0 | a1) | (a2 | a3);
    acc.simd_eq(z).any_false() as u64
}

entry!(k_verify4_fssplit_128, Avx2, u64x2, verify4_split, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_fssplit_256, Avx2, u64x4, verify4_split, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_fssplit_512, Avx512, u64x8, verify4_split, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_fsptr_128, Avx2, u64x2, verify4_ptr, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_fsptr_256, Avx2, u64x4, verify4_ptr, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_fsptr_512, Avx512, u64x8, verify4_ptr, (buf: &[u64]) -> u64, (buf, PATTERN));

/// Prefetching verify with the raw-pointer walk (see `verify4_ptr`).
#[simd]
pub fn verify4_pf_ptr<S: Simd, V: SimdInt<S, Element = u64>>(
    simd: S,
    buf: &[u64],
    pat: u64,
) -> u64 {
    let w = V::LEN;
    let lines = (4 * size_of::<V>() / LINE).max(1);
    let p = V::splat(simd, pat);
    let z = V::splat(simd, 0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let n = buf.len() / w * w;
    let mut ptr = buf.as_ptr();
    // SAFETY: `end` is within (or one past) `buf`.
    let end = unsafe { ptr.add(n) };
    // SAFETY (loads below): every `ptr.add(k * w)` read is < `end`, and `V::Array` is `[u64; LEN]`.
    let ld = |q: *const u64| V::load_array_ref(simd, unsafe { &*(q as *const V::Array) });
    while (end as usize - ptr as usize) >= 4 * w * 8 {
        let ahead = (ptr as *const i8).wrapping_add(PF_DIST);
        for k in 0..lines {
            // SAFETY: SSE is in every x86_64 baseline; prefetch never faults.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(ahead.wrapping_add(k * LINE)) };
        }
        a0 |= ld(ptr) ^ p;
        a1 |= ld(unsafe { ptr.add(w) }) ^ p;
        a2 |= ld(unsafe { ptr.add(2 * w) }) ^ p;
        a3 |= ld(unsafe { ptr.add(3 * w) }) ^ p;
        ptr = unsafe { ptr.add(4 * w) };
    }
    while ptr < end {
        a0 |= ld(ptr) ^ p;
        ptr = unsafe { ptr.add(w) };
    }
    let acc = (a0 | a1) | (a2 | a3);
    acc.simd_eq(z).any_false() as u64
}

entry!(k_pfv_fsptr_256, Avx2, u64x4, verify4_pf_ptr, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_pfv_fsptr_512, Avx512, u64x8, verify4_pf_ptr, (buf: &[u64]) -> u64, (buf, PATTERN));
