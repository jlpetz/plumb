//! AMX kernels (plumb_tiles) for the bench: tile loads and stores as a DRAM access path, next to
//! the AVX-512 equivalents. Names use the `k_amx_*` prefix (plumb_tiles' own example uses
//! `k_tile_*`, with its own asm expectations).
//!
//! A tile is 16 rows x 64 bytes. At stride 64 one tile load/store moves 1 KiB of consecutive
//! memory; at stride 4096 it touches one line in each of 16 pages, an access shape no single
//! vector instruction produces (TMR TODO 78's question).

#![allow(unsafe_op_in_unsafe_fn)]

use crate::common::*;
use fearless_simd::prelude::*;
use fearless_simd::{Avx512, u64x8};
use fearless_simd_macros::simd;
use plumb_lines::Line;
use plumb_tiles::{Amx, ROW_BYTES, ROWS, T0, T1, T2, T3};

/// u64 per tile at stride 64 (1 KiB).
const TILE_U64: usize = ROWS * ROW_BYTES / 8;
const PAGE: usize = 4096;

/// Sequential read sweep: 4 tile loads per iteration, so 4 KiB is in flight.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_amx_read(amx: Amx, buf: &[u64]) -> u64 {
    amx.with_tiles(|t| {
        let (tiles, rest) = buf.as_chunks::<TILE_U64>();
        assert!(rest.is_empty(), "buffer must be whole tiles");
        let (quads, rest) = tiles.as_chunks::<4>();
        for q in quads {
            t.load_u64::<T0>(&q[0], ROW_BYTES);
            t.load_u64::<T1>(&q[1], ROW_BYTES);
            t.load_u64::<T2>(&q[2], ROW_BYTES);
            t.load_u64::<T3>(&q[3], ROW_BYTES);
        }
        for b in rest {
            t.load_u64::<T0>(b, ROW_BYTES);
        }
    });
    0
}

/// Copy (first half -> second half) through tiles, two in flight.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_amx_copy(amx: Amx, dst: &mut [u64], src: &[u64]) {
    amx.with_tiles(|t| {
        let (s, _) = src.as_chunks::<TILE_U64>();
        let (d, _) = dst.as_chunks_mut::<TILE_U64>();
        let (sp, sr) = s.as_chunks::<2>();
        let (dp, dr) = d.as_chunks_mut::<2>();
        for (s2, d2) in sp.iter().zip(dp.iter_mut()) {
            t.load_u64::<T0>(&s2[0], ROW_BYTES);
            t.load_u64::<T1>(&s2[1], ROW_BYTES);
            t.store_u64::<T0>(&mut d2[0], ROW_BYTES);
            t.store_u64::<T1>(&mut d2[1], ROW_BYTES);
        }
        for (s1, d1) in sr.iter().zip(dr.iter_mut()) {
            t.load_u64::<T0>(s1, ROW_BYTES);
            t.store_u64::<T0>(d1, ROW_BYTES);
        }
    })
}

/// Pattern fill: one tile loaded with the pattern, stored to every 1 KiB block.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_amx_fill(amx: Amx, buf: &mut [u64]) {
    let pattern = [PATTERN; TILE_U64];
    amx.with_tiles(|t| {
        t.load_u64::<T0>(&pattern, ROW_BYTES);
        let (tiles, rest) = buf.as_chunks_mut::<TILE_U64>();
        for b in tiles {
            t.store_u64::<T0>(b, ROW_BYTES);
        }
        rest.fill(PATTERN);
    })
}

/// Strided read: each tile load reads one 64-byte line from each of 16 pages (stride 4096).
/// Covers every byte: for each 64 KiB group, 64 loads at line offsets 0..4096.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_amx_strided_read(amx: Amx, buf: &[u64]) -> u64 {
    const GROUP_U64: usize = ROWS * PAGE / 8;
    amx.with_tiles(|t| {
        let (groups, _) = buf.as_chunks::<GROUP_U64>();
        for g in groups {
            for line in 0..PAGE / ROW_BYTES {
                // The span is (ROWS-1)*PAGE + 64 bytes from line*64: always inside the group.
                t.load_u64::<T0>(&g[line * ROW_BYTES / 8..], PAGE);
            }
        }
    });
    0
}

/// The same strided order with zmm loads (16 rows x 1 load each), 4 accumulators, for
/// comparison with `k_amx_strided_read`.
///
/// # Safety
/// `p..p+n` valid and 64-byte aligned, `n` a multiple of 64 KiB in u64; AVX-512F present.
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,avx,fma,bmi1,bmi2")]
pub unsafe fn k_amx_strided_read_zmm_512(p: *const u64, n: usize) -> u64 {
    use std::simd::u64x8 as V;
    const GROUP_U64: usize = ROWS * PAGE / 8;
    let (mut a0, mut a1, mut a2, mut a3) = (V::splat(0), V::splat(0), V::splat(0), V::splat(0));
    for g in 0..n / GROUP_U64 {
        let base = p.add(g * GROUP_U64);
        for line in 0..PAGE / ROW_BYTES {
            let l = base.add(line * ROW_BYTES / 8);
            for r in (0..ROWS).step_by(4) {
                a0 |= *(l.add(r * PAGE / 8) as *const V);
                a1 |= *(l.add((r + 1) * PAGE / 8) as *const V);
                a2 |= *(l.add((r + 2) * PAGE / 8) as *const V);
                a3 |= *(l.add((r + 3) * PAGE / 8) as *const V);
            }
        }
    }
    let acc = (a0 | a1) | (a2 | a3);
    acc.to_array().iter().fold(0, |x, y| x | y)
}

/// Verify through tiles: tile-load a 1 KiB block from DRAM, store it to an L1 scratch, then
/// XOR/OR the scratch rows against the pattern with zmm (4 accumulators). Returns 1 on mismatch.
#[simd]
pub fn amx_verify<S: Simd>(simd: S, amx: Amx, buf: &[u64], pat: u64) -> u64 {
    let mut scratch = [Line([0; 8]); ROWS];
    let p = u64x8::splat(simd, pat);
    let z = u64x8::splat(simd, 0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    amx.with_tiles(|t| {
        let (tiles, _) = buf.as_chunks::<TILE_U64>();
        for b in tiles {
            t.load_u64::<T0>(b, ROW_BYTES);
            // SAFETY: Line is [u64; 8]; the scratch is 16 lines = one tile at stride 64.
            let words = unsafe { std::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut u64, TILE_U64) };
            t.store_u64::<T0>(words, ROW_BYTES);
            let (quads, _) = scratch.as_chunks::<4>();
            for q in quads {
                a0 |= u64x8::from_slice(simd, &q[0].0) ^ p;
                a1 |= u64x8::from_slice(simd, &q[1].0) ^ p;
                a2 |= u64x8::from_slice(simd, &q[2].0) ^ p;
                a3 |= u64x8::from_slice(simd, &q[3].0) ^ p;
            }
        }
    });
    ((a0 | a1) | (a2 | a3)).simd_eq(z).any_false() as u64
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_amx_verify_512(t5: Avx512, amx: Amx, buf: &[u64]) -> u64 {
    amx_verify(t5, amx, buf, PATTERN)
}
