//! Reference kernels in TMR-APP's current style: one `macro_rules!` expansion per width, inside
//! a `#[target_feature]` fn, using `std::simd` types and `std::arch` intrinsics. Each loop body
//! is copied from the TMR macro named in its doc comment so the comparison is like-for-like.
//!
//! Every kernel is `#[inline(never)]` + `#[unsafe(no_mangle)]` so `asm_check.py` can find it.
//! Arguments are `(ptr, n_u64)` like TMR's `ChunkCtx`; the harness passes whole buffers.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::common::*;
use std::arch::x86_64::*;
use std::simd::cmp::SimdPartialEq;
use std::simd::{u64x2, u64x4, u64x8};

// Feature strings are TMR-APP's exact ones (tests.rs, `stuck_bit_impl!` / `simple_test_nt_impl!`).
// `#[target_feature(enable = ...)]` takes only a literal, so they are repeated, as in TMR.

/// Fill, verify (4 accumulators), positional write/verify and LCG write at one width.
macro_rules! tmr_width {
    ($V:ty, $lanes:expr, $tf:literal,
     $fill:ident, $filluni:ident, $verify4:ident, $posw:ident, $posv:ident, $lcgw:ident, $pfv:ident) => {
        /// Write phase of `stuck_bit_write_verify!` (constant splat store).
        #[unsafe(no_mangle)]
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $fill(p: *mut u64, n: usize) {
            let base = p as *mut $V;
            let pat = <$V>::splat(PATTERN);
            for i in 0..n / <$V>::LEN {
                *base.add(i) = pat;
            }
        }

        /// Same loop with a byte-uniform constant: expected to become `memset`.
        #[unsafe(no_mangle)]
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $filluni(p: *mut u64, n: usize) {
            let base = p as *mut $V;
            let pat = <$V>::splat(UNIFORM);
            for i in 0..n / <$V>::LEN {
                *base.add(i) = pat;
            }
        }

        /// Verify phase of `stuck_bit_write_verify!`: 4 independent XOR/OR chains.
        #[unsafe(no_mangle)]
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $verify4(p: *const u64, n: usize) -> u64 {
            let base = p as *const $V;
            let end = n / <$V>::LEN;
            let pat = <$V>::splat(PATTERN);
            let mut a0 = <$V>::splat(0);
            let mut a1 = <$V>::splat(0);
            let mut a2 = <$V>::splat(0);
            let mut a3 = <$V>::splat(0);
            let mut i = 0;
            while i + 4 <= end {
                a0 |= *base.add(i) ^ pat;
                a1 |= *base.add(i + 1) ^ pat;
                a2 |= *base.add(i + 2) ^ pat;
                a3 |= *base.add(i + 3) ^ pat;
                i += 4;
            }
            while i < end {
                a0 |= *base.add(i) ^ pat;
                i += 1;
            }
            let acc = (a0 | a1) | (a2 | a3);
            acc.simd_ne(<$V>::splat(0)).any() as u64
        }

        /// `simple_write_positional_simd!` (Mode 0/1: idx ^ base).
        #[unsafe(no_mangle)]
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $posw(p: *mut u64, n: usize) {
            let w = <$V>::LEN;
            let base_vec = <$V>::splat(POS_BASE);
            let step = <$V>::splat(w as u64);
            let mut idx_vec = <$V>::splat(0) + <$V>::from_array($lanes);
            for i in (0..n).step_by(w) {
                *(p.add(i) as *mut $V) = idx_vec ^ base_vec;
                idx_vec += step;
            }
        }

        /// `simple_verify_positional_simd!`, `check_mask = None` arm (single accumulator).
        #[unsafe(no_mangle)]
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $posv(p: *const u64, n: usize) -> u64 {
            let w = <$V>::LEN;
            let zero = <$V>::splat(0);
            let base_vec = <$V>::splat(POS_BASE);
            let step = <$V>::splat(w as u64);
            let mut idx_vec = <$V>::splat(0) + <$V>::from_array($lanes);
            let mut error_acc = zero;
            for i in (0..n).step_by(w) {
                let actual = *(p.add(i) as *const $V);
                error_acc |= actual ^ (idx_vec ^ base_vec);
                idx_vec += step;
            }
            error_acc.simd_ne(zero).any() as u64
        }

        /// `simple_write_lcg_simd!` with `LcgSimdN::next` inlined (Mode 2: state * m + a).
        #[unsafe(no_mangle)]
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $lcgw(p: *mut u64, n: usize) {
            let w = <$V>::LEN;
            let (seeds, m, a) = lcg_lanes(w);
            let mut state = <$V>::from_slice(&seeds[..w]);
            let mul = <$V>::splat(m);
            let add = <$V>::splat(a);
            for i in (0..n).step_by(w) {
                *(p.add(i) as *mut $V) = state;
                state = state * mul + add;
            }
        }

        /// The 4-accumulator verify plus a software prefetch per cache line, `PF_DIST` ahead.
        #[unsafe(no_mangle)]
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $pfv(p: *const u64, n: usize) -> u64 {
            const LINES: usize = (4 * size_of::<$V>() / LINE) as usize;
            let base = p as *const $V;
            let end = n / <$V>::LEN;
            let pat = <$V>::splat(PATTERN);
            let mut a0 = <$V>::splat(0);
            let mut a1 = <$V>::splat(0);
            let mut a2 = <$V>::splat(0);
            let mut a3 = <$V>::splat(0);
            let mut i = 0;
            while i + 4 <= end {
                let ahead = (base.add(i) as *const i8).wrapping_add(PF_DIST);
                for k in 0..LINES.max(1) {
                    _mm_prefetch::<_MM_HINT_T0>(ahead.wrapping_add(k * LINE));
                }
                a0 |= *base.add(i) ^ pat;
                a1 |= *base.add(i + 1) ^ pat;
                a2 |= *base.add(i + 2) ^ pat;
                a3 |= *base.add(i + 3) ^ pat;
                i += 4;
            }
            while i < end {
                a0 |= *base.add(i) ^ pat;
                i += 1;
            }
            let acc = (a0 | a1) | (a2 | a3);
            acc.simd_ne(<$V>::splat(0)).any() as u64
        }
    };
}

tmr_width!(u64x2, [0, 1], "sse4.2,sse4.1,ssse3,sse3,sse2,popcnt",
    k_fill_tmr_128, k_filluni_tmr_128, k_verify4_tmr_128, k_posw_tmr_128, k_posv_tmr_128,
    k_lcgw_tmr_128, k_pfv_tmr_128);
tmr_width!(u64x4, [0, 1, 2, 3], "avx2,avx,fma,bmi1,bmi2",
    k_fill_tmr_256, k_filluni_tmr_256, k_verify4_tmr_256, k_posw_tmr_256, k_posv_tmr_256,
    k_lcgw_tmr_256, k_pfv_tmr_256);
tmr_width!(u64x8, [0, 1, 2, 3, 4, 5, 6, 7], "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,avx,fma,bmi1,bmi2",
    k_fill_tmr_512, k_filluni_tmr_512, k_verify4_tmr_512, k_posw_tmr_512, k_posv_tmr_512,
    k_lcgw_tmr_512, k_pfv_tmr_512);

/// `simple_write_nt_positional_simd!` verbatim: 4x manual unroll, stream intrinsic, sfence.
macro_rules! tmr_nt {
    ($name:ident, $V:ty, $lanes:expr, $arch:ty, $stream:path, $tf:literal) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $name(p: *mut u64, n: usize) {
            let w = <$V>::LEN;
            let base_vec = <$V>::splat(POS_BASE);
            let step1 = <$V>::splat(w as u64);
            let step4 = <$V>::splat((w * 4) as u64);
            let mut idx_vec = <$V>::splat(0) + <$V>::from_array($lanes);
            let unrolled_end = (n / (w * 4)) * (w * 4);
            let mut i = 0;
            while i < unrolled_end {
                let v0: $V = idx_vec ^ base_vec;
                let idx1 = idx_vec + step1;
                let v1: $V = idx1 ^ base_vec;
                let idx2 = idx1 + step1;
                let v2: $V = idx2 ^ base_vec;
                let idx3 = idx2 + step1;
                let v3: $V = idx3 ^ base_vec;
                $stream(p.add(i) as *mut $arch, std::mem::transmute::<$V, $arch>(v0));
                $stream(p.add(i + w) as *mut $arch, std::mem::transmute::<$V, $arch>(v1));
                $stream(p.add(i + w * 2) as *mut $arch, std::mem::transmute::<$V, $arch>(v2));
                $stream(p.add(i + w * 3) as *mut $arch, std::mem::transmute::<$V, $arch>(v3));
                idx_vec += step4;
                i += w * 4;
            }
            while i < n {
                let val: $V = idx_vec ^ base_vec;
                $stream(p.add(i) as *mut $arch, std::mem::transmute::<$V, $arch>(val));
                idx_vec += step1;
                i += w;
            }
            _mm_sfence();
        }
    };
}

tmr_nt!(k_ntw_tmr_128, u64x2, [0, 1], __m128i, _mm_stream_si128, "sse4.2,sse4.1,ssse3,sse3,sse2,popcnt");
tmr_nt!(k_ntw_tmr_256, u64x4, [0, 1, 2, 3], __m256i, _mm256_stream_si256, "avx2,avx,fma,bmi1,bmi2");
tmr_nt!(k_ntw_tmr_512, u64x8, [0, 1, 2, 3, 4, 5, 6, 7], __m512i, _mm512_stream_si512, "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,avx,fma,bmi1,bmi2");

/// `flush_range_to_dram` verbatim (the CLFLUSHOPT intrinsic in a `clflushopt` fn, then MFENCE).
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "clflushopt")]
pub unsafe fn k_flush_tmr(p: *const u64, n: usize) {
    let base = p as *const u8;
    let line_count = (n * 8).div_ceil(LINE);
    for i in 0..line_count {
        _mm_clflushopt(base.add(i * LINE));
    }
    _mm_mfence();
}

/// Mixed loop TMR doesn't have yet: write each line, then CLFLUSHOPT it in the same loop.
/// TMR style gets `clflushopt` into the fn's feature list, so the intrinsic inlines.
macro_rules! tmr_wflush {
    ($name:ident, $V:ty, $tf:literal) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $name(p: *mut u64, n: usize) {
            const PER_LINE: usize = LINE / size_of::<$V>();
            let base = p as *mut $V;
            let pat = <$V>::splat(PATTERN);
            for line in 0..n * 8 / LINE {
                for k in 0..PER_LINE {
                    *base.add(line * PER_LINE + k) = pat;
                }
                _mm_clflushopt(base.add(line * PER_LINE) as *const u8);
            }
            _mm_mfence();
        }
    };
}

tmr_wflush!(k_wflush_tmr_256, u64x4, "avx2,avx,fma,bmi1,bmi2,clflushopt");
tmr_wflush!(k_wflush_tmr_512, u64x8, "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,avx,fma,bmi1,bmi2,clflushopt");

/// DRAM -> DRAM line copy: 512-bit load + NT store, then sfence (shuffle-test's NT-512 path).
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,avx,fma,bmi1,bmi2")]
pub unsafe fn k_copynt_tmr_512(dst: *mut u64, src: *const u64, n: usize) {
    for i in (0..n).step_by(8) {
        let v = *(src.add(i) as *const u64x8);
        _mm512_stream_si512(dst.add(i) as *mut __m512i, std::mem::transmute::<u64x8, __m512i>(v));
    }
    _mm_sfence();
}

/// DRAM -> DRAM line copy with MOVDIR64B (shuffle-test's asm; no stdarch intrinsic exists).
/// MOVDIR64B is weakly ordered like an NT store, so it needs the same sfence.
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe fn k_copymd_tmr(dst: *mut u64, src: *const u64, n: usize) {
    for i in (0..n).step_by(LINE / 8) {
        movdir64b(dst.add(i) as *mut u8, src.add(i) as *const u8);
    }
    _mm_sfence();
}

/// Deliberate footgun: a non-inlined helper called from the 512-bit verify. The helper has no
/// `#[target_feature]`, so std::simd compiles its `u64x8` ops at the x86-64-v3 baseline:
/// 2 x ymm, silently, with correct results. This is the failure TMR's macro rule exists for.
#[inline(never)]
fn or_xor_step_std(a: u64x8, x: u64x8, p: u64x8) -> u64x8 {
    a | (x ^ p)
}

#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,avx,fma,bmi1,bmi2")]
pub unsafe fn k_verify4_tmrhelper_512(p: *const u64, n: usize) -> u64 {
    let base = p as *const u64x8;
    let end = n / 8;
    let pat = u64x8::splat(PATTERN);
    let (mut a0, mut a1, mut a2, mut a3) =
        (u64x8::splat(0), u64x8::splat(0), u64x8::splat(0), u64x8::splat(0));
    let mut i = 0;
    while i + 4 <= end {
        a0 = or_xor_step_std(a0, *base.add(i), pat);
        a1 = or_xor_step_std(a1, *base.add(i + 1), pat);
        a2 = or_xor_step_std(a2, *base.add(i + 2), pat);
        a3 = or_xor_step_std(a3, *base.add(i + 3), pat);
        i += 4;
    }
    let acc = (a0 | a1) | (a2 | a3);
    acc.simd_ne(u64x8::splat(0)).any() as u64
}

// Fences, for the memory-model question in TODO 84. asm only, never timed.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_fence_seqcst() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_fence_release() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_fence_sfence() {
    // SAFETY: SSE is in every x86_64 baseline (a safe fn without `#[target_feature]` still
    // needs `unsafe` to call it).
    unsafe { _mm_sfence() };
}
