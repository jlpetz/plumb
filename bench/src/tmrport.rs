//! Ports of three TMR-APP tests, each written twice:
//!
//! - **TMR style** (`*_tmr_*`): the per-chunk hot loops of `stuck_bit_impl!` /
//!   `stuck_bit_write_verify!`, `refresh_impl!`, and `simple_write_nt_positional_simd!` +
//!   `simple_verify_positional_simd!` (TMR-APP `src/tests.rs`), with raw pointers, `std::simd`
//!   and per-width `#[target_feature]`. TMR's scaffolding (timers, progress, stats) is left out:
//!   the bench times whole passes.
//! - **plumb style** (`*_pl_*`): one generic fearless body per test using plumb_lines:
//!   `flush_after` for write-then-flush, aligned views for fill/verify, `nontemporal` for NT.
//!
//! Both take an `inject` hook, called after each write phase and before its verify, so the
//! equivalence tests can flip bits and check that both styles report the same errors. The bench
//! entries pass a no-op closure, which compiles away.
//!
//! Semantics kept from TMR: chunk sizes (`CHUNK_U64`, at least TMR's minimum vectors per chunk);
//! StuckBit's three phases P1, P2, P1 with `fence(SeqCst)` after each write; flush (when
//! enabled) between write and verify; Refresh's single phase (its 64 ms sleep is omitted: it's a
//! fixed wall-clock cost, not bandwidth); SimpleNT's global positional index across chunks.
//! Errors are counted per bad chunk-phase, as in TMR.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::common::*;
use crate::lines::{fill_view, ntw_scope_at, pos_verify_view_at, verify4_view};
use fearless_simd::{Avx2, Avx512, Simd, SimdInt, u64x4, u64x8};
use fearless_simd_macros::simd;
use plumb_lines::{Clflushopt, NtStore};
use std::arch::x86_64::*;
use std::simd::cmp::SimdPartialEq;
use std::sync::atomic::{Ordering, fence};

/// TMR's StuckBit patterns: exact complements, not byte-uniform (TMR-APP tests.rs).
pub const STUCKBIT_P1: u64 = 0xAA55_AA55_AA55_AA55;
pub const STUCKBIT_P2: u64 = 0x55AA_55AA_55AA_55AA;
/// TMR's `REFRESH_PATTERN` (not byte-uniform).
pub const REFRESH_PATTERN: u64 = 0xA55A_A55A_A55A_A55A;
/// Chunk size in u64: 1 MiB, a typical TMR chunk.
pub const CHUNK_U64: usize = 1 << 17;

// ---------------------------------------------------------------------------------------------
// TMR style
// ---------------------------------------------------------------------------------------------

macro_rules! tmr_ports {
    ($V:ty, $tf:literal, $sb:ident, $refresh:ident, $simplent:ident, $lanes:expr, $arch:ty, $stream:ident) => {
        /// `stuck_bit_impl!` per chunk: write P1/P2/P1, fence, optional flush, 4-acc verify.
        ///
        /// # Safety
        /// `p..p+n` valid and 64-byte aligned; the CPU has the features in `$tf` and CLFLUSHOPT.
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $sb<F: FnMut(&mut [u64])>(p: *mut u64, n: usize, flush: bool, mut inject: F) -> u64 {
            let lanes = size_of::<$V>() / 8;
            let base = p as *mut $V;
            let len = n / lanes;
            let chunk = (CHUNK_U64 / lanes).max(1024);
            let mut errs = 0u64;
            let mut start = 0;
            while start < len {
                let end = (start + chunk).min(len);
                for pat_u in [STUCKBIT_P1, STUCKBIT_P2, STUCKBIT_P1] {
                    let pat = <$V>::splat(pat_u);
                    for i in start..end {
                        *base.add(i) = pat;
                    }
                    fence(Ordering::SeqCst);
                    if flush {
                        crate::tmr::k_flush_tmr(base.add(start) as *const u64, (end - start) * lanes);
                    }
                    inject(std::slice::from_raw_parts_mut(base.add(start) as *mut u64, (end - start) * lanes));
                    let mut a0 = <$V>::splat(0);
                    let mut a1 = <$V>::splat(0);
                    let mut a2 = <$V>::splat(0);
                    let mut a3 = <$V>::splat(0);
                    let mut i = start;
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
                    if acc.simd_ne(<$V>::splat(0)).any() {
                        errs += 1;
                    }
                }
                start = end;
            }
            errs
        }

        /// `refresh_impl!` per chunk (sleep omitted): write, flush if enabled, fence, verify.
        ///
        /// # Safety
        /// As above.
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $refresh<F: FnMut(&mut [u64])>(p: *mut u64, n: usize, flush: bool, mut inject: F) -> u64 {
            let lanes = size_of::<$V>() / 8;
            let base = p as *mut $V;
            let len = n / lanes;
            let chunk = (CHUNK_U64 / lanes).max(256);
            let pattern = <$V>::splat(REFRESH_PATTERN);
            let mut errs = 0u64;
            let mut start = 0;
            while start < len {
                let end = (start + chunk).min(len);
                for i in start..end {
                    *base.add(i) = pattern;
                }
                if flush {
                    crate::tmr::k_flush_tmr(base.add(start) as *const u64, (end - start) * lanes);
                }
                fence(Ordering::SeqCst);
                inject(std::slice::from_raw_parts_mut(base.add(start) as *mut u64, (end - start) * lanes));
                let mut a0 = <$V>::splat(0);
                let mut a1 = <$V>::splat(0);
                let mut a2 = <$V>::splat(0);
                let mut a3 = <$V>::splat(0);
                let mut i = start;
                while i + 4 <= end {
                    a0 |= *base.add(i) ^ pattern;
                    a1 |= *base.add(i + 1) ^ pattern;
                    a2 |= *base.add(i + 2) ^ pattern;
                    a3 |= *base.add(i + 3) ^ pattern;
                    i += 4;
                }
                while i < end {
                    a0 |= *base.add(i) ^ pattern;
                    i += 1;
                }
                let acc = (a0 | a1) | (a2 | a3);
                if acc.simd_ne(<$V>::splat(0)).any() {
                    errs += 1;
                }
                start = end;
            }
            errs
        }

        /// SimpleNT per chunk: `simple_write_nt_positional_simd!` (4x unroll, SFENCE) then
        /// `simple_verify_positional_simd!` (check_mask = None). The index is global.
        ///
        /// # Safety
        /// As above (no CLFLUSHOPT needed).
        #[inline(never)]
        #[target_feature(enable = $tf)]
        pub unsafe fn $simplent<F: FnMut(&mut [u64])>(p: *mut u64, n: usize, mut inject: F) -> u64 {
            let w = $lanes.len();
            let lane_offsets = <$V>::from_array($lanes);
            let base_vec = <$V>::splat(POS_BASE);
            let zero = <$V>::splat(0);
            let chunk = CHUNK_U64;
            let mut errs = 0u64;
            let mut cs = 0;
            while cs < n {
                let ce = (cs + chunk).min(n);
                // Write: 4x unrolled NT stores, then SFENCE.
                let step1 = <$V>::splat(w as u64);
                let step4 = <$V>::splat((w * 4) as u64);
                let mut idx_vec = <$V>::splat(cs as u64) + lane_offsets;
                let unrolled_end = cs + ((ce - cs) / (w * 4)) * (w * 4);
                let mut i = cs;
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
                while i < ce {
                    let val: $V = idx_vec ^ base_vec;
                    $stream(p.add(i) as *mut $arch, std::mem::transmute::<$V, $arch>(val));
                    idx_vec += step1;
                    i += w;
                }
                _mm_sfence();
                inject(std::slice::from_raw_parts_mut(p.add(cs), ce - cs));
                // Verify.
                let mut idx_vec = <$V>::splat(cs as u64) + lane_offsets;
                let mut error_acc = zero;
                for i in (cs..ce).step_by(w) {
                    let actual = *(p.add(i) as *const $V);
                    error_acc |= actual ^ (idx_vec ^ base_vec);
                    idx_vec += step1;
                }
                if error_acc.simd_ne(zero).any() {
                    errs += 1;
                }
                cs = ce;
            }
            errs
        }
    };
}

tmr_ports!(std::simd::u64x4, "avx2,avx,fma,bmi1,bmi2",
    sb_tmr_256, refresh_tmr_256, simplent_tmr_256, [0u64, 1, 2, 3], __m256i, _mm256_stream_si256);
tmr_ports!(std::simd::u64x8, "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,avx,fma,bmi1,bmi2",
    sb_tmr_512, refresh_tmr_512, simplent_tmr_512, [0u64, 1, 2, 3, 4, 5, 6, 7], __m512i, _mm512_stream_si512);

// ---------------------------------------------------------------------------------------------
// plumb style: one generic body per test
// ---------------------------------------------------------------------------------------------

/// StuckBit: per chunk, three phases; `flush_after` makes "write then flush before verify" a
/// scope. Without flush, TMR's `fence(SeqCst)` after the write is kept.
#[simd]
pub fn stuckbit_pl<S: Simd, V: SimdInt<S, Element = u64>, F: FnMut(&mut [u64])>(
    simd: S,
    cf: Clflushopt,
    buf: &mut [u64],
    flush: bool,
    mut inject: F,
) -> u64 {
    let mut errs = 0;
    for chunk in buf.chunks_mut(CHUNK_U64.max(1024 * V::LEN)) {
        for pat in [STUCKBIT_P1, STUCKBIT_P2, STUCKBIT_P1] {
            if flush {
                plumb_lines::flush_after(cf, chunk, |c| fill_view::<S, V>(simd, c, pat));
            } else {
                fill_view::<S, V>(simd, chunk, pat);
                fence(Ordering::SeqCst);
            }
            inject(chunk);
            errs += verify4_view::<S, V>(simd, chunk, pat);
        }
    }
    errs
}

/// Refresh (sleep omitted): write, flush if enabled, verify.
#[simd]
pub fn refresh_pl<S: Simd, V: SimdInt<S, Element = u64>, F: FnMut(&mut [u64])>(
    simd: S,
    cf: Clflushopt,
    buf: &mut [u64],
    flush: bool,
    mut inject: F,
) -> u64 {
    let mut errs = 0;
    for chunk in buf.chunks_mut(CHUNK_U64.max(256 * V::LEN)) {
        if flush {
            plumb_lines::flush_after(cf, chunk, |c| fill_view::<S, V>(simd, c, REFRESH_PATTERN));
        } else {
            fill_view::<S, V>(simd, chunk, REFRESH_PATTERN);
            fence(Ordering::SeqCst);
        }
        inject(chunk);
        errs += verify4_view::<S, V>(simd, chunk, REFRESH_PATTERN);
    }
    errs
}

/// SimpleNT: per chunk, positional NT write through the `nontemporal` scope (its fence), then
/// positional verify. The index is global, as in TMR.
#[simd]
pub fn simplent_pl<S: Simd, V: SimdInt<S, Element = u64> + NtStore<S>, F: FnMut(&mut [u64])>(
    simd: S,
    buf: &mut [u64],
    mut inject: F,
) -> u64 {
    let mut errs = 0;
    for (k, chunk) in buf.chunks_mut(CHUNK_U64).enumerate() {
        let start = (k * CHUNK_U64) as u64;
        ntw_scope_at::<S, V>(simd, chunk, POS_BASE, start);
        inject(chunk);
        errs += pos_verify_view_at::<S, V>(simd, chunk, POS_BASE, start);
    }
    errs
}

// ---------------------------------------------------------------------------------------------
// Named bench entries (no-op injection)
// ---------------------------------------------------------------------------------------------

fn none(_: &mut [u64]) {}

macro_rules! tmr_entry {
    ($name:ident, $f:ident, flush $flush:expr) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name(buf: &mut [u64]) -> u64 {
            // SAFETY: page-aligned whole buffer; the harness gates the CPU features.
            unsafe { $f(buf.as_mut_ptr(), buf.len(), $flush, none) }
        }
    };
    ($name:ident, $f:ident) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name(buf: &mut [u64]) -> u64 {
            // SAFETY: as above.
            unsafe { $f(buf.as_mut_ptr(), buf.len(), none) }
        }
    };
}

tmr_entry!(k_sb_tmr_256, sb_tmr_256, flush true);
tmr_entry!(k_sb_tmr_512, sb_tmr_512, flush true);
tmr_entry!(k_sbnf_tmr_256, sb_tmr_256, flush false);
tmr_entry!(k_sbnf_tmr_512, sb_tmr_512, flush false);
tmr_entry!(k_refresh_tmr_256, refresh_tmr_256, flush true);
tmr_entry!(k_refresh_tmr_512, refresh_tmr_512, flush true);
tmr_entry!(k_simplent_tmr_256, simplent_tmr_256);
tmr_entry!(k_simplent_tmr_512, simplent_tmr_512);

macro_rules! pl_entry {
    ($name:ident, $Tok:ident, $V:ident, $f:ident, flush $flush:expr) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name(t: $Tok, cf: Clflushopt, buf: &mut [u64]) -> u64 {
            $f::<$Tok, $V<$Tok>, _>(t, cf, buf, $flush, none)
        }
    };
    ($name:ident, $Tok:ident, $V:ident, $f:ident) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name(t: $Tok, buf: &mut [u64]) -> u64 {
            $f::<$Tok, $V<$Tok>, _>(t, buf, none)
        }
    };
}

pl_entry!(k_sb_pl_256, Avx2, u64x4, stuckbit_pl, flush true);
pl_entry!(k_sb_pl_512, Avx512, u64x8, stuckbit_pl, flush true);
pl_entry!(k_sbnf_pl_256, Avx2, u64x4, stuckbit_pl, flush false);
pl_entry!(k_sbnf_pl_512, Avx512, u64x8, stuckbit_pl, flush false);
pl_entry!(k_refresh_pl_256, Avx2, u64x4, refresh_pl, flush true);
pl_entry!(k_refresh_pl_512, Avx512, u64x8, refresh_pl, flush true);
pl_entry!(k_simplent_pl_256, Avx2, u64x4, simplent_pl);
pl_entry!(k_simplent_pl_512, Avx512, u64x8, simplent_pl);
