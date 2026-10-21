// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Correctness of the AMX bench kernels (tiles.rs) on the real CPU. They skip, with a message,
//! when the CPU or OS has no AMX tile state.

use crate::common::*;
use crate::tiles;
use fearless_simd::Level;
use plumb_lines::Line;
use plumb_tiles::Amx;

const GROUP_U64: usize = 16 * 4096 / 8; // 64 KiB: the strided kernels' unit

fn amx() -> Option<Amx> {
    let a = Amx::try_new();
    if a.is_none() {
        eprintln!("skip: no AMX tile state on this CPU/OS");
    }
    a
}

/// `k_amx_verify_512` enables fearless_simd's Avx512 list plus `amx-tile`. plumb_lines' copy of
/// that list (plus `clflushopt`) is checked against fearless_simd's source by plumb_lines'
/// `nightly_entry` test, so matching it here ties this one to the source too.
#[test]
fn avx512_amx_features_are_fearless_avx512_plus_amx_tile() {
    let fearless = plumb_lines::entry::AVX512_FEATURES
        .strip_suffix(",clflushopt")
        .expect("plumb_lines' list ends with clflushopt");
    assert_eq!(tiles::AVX512_AMX_FEATURES, format!("{fearless},amx-tile"));
}

fn buf(n: usize) -> Vec<Line> {
    vec![Line([0; 8]); n.div_ceil(8)]
}
fn words(v: &mut [Line]) -> &mut [u64] {
    // SAFETY: Line is [u64; 8].
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u64, v.len() * 8) }
}

#[test]
fn amx_fill_read_and_verify() {
    let Some(amx) = amx() else { return };
    let n = GROUP_U64 * 3;
    let mut store = buf(n + 8);
    let all = words(&mut store);
    // Fill handles a tail that isn't a whole tile (scalar rest).
    for len in [n, n + 5] {
        let b = &mut all[..len];
        b.fill(0);
        tiles::k_amx_fill(amx, b);
        assert!(b.iter().all(|&w| w == PATTERN), "fill len {len}");
    }
    let b = &mut all[..n];
    assert_eq!(tiles::k_amx_read(amx, b), 0);
    assert_eq!(tiles::k_amx_strided_read(amx, b), 0);
    let Some(t5) = Level::new().as_avx512() else {
        return eprintln!("skip: verify needs Avx512");
    };
    assert_eq!(tiles::k_amx_verify_512(t5, amx, b), 0, "false positive");
    for (i, bit) in [(0, 0), (127, 63), (n / 2, 31), (n - 1, 7), (n - 128, 17)] {
        b[i] ^= 1 << bit;
        assert_eq!(
            tiles::k_amx_verify_512(t5, amx, b),
            1,
            "missed word {i} bit {bit}"
        );
        b[i] ^= 1 << bit;
    }
    // SAFETY: aligned, whole 64 KiB groups. The fn's feature list (TMR's 512 set) is a subset of
    // fearless_simd's Avx512 level, which `t5` proves.
    let or = unsafe { tiles::k_amx_strided_read_zmm_512(b.as_ptr(), b.len()) };
    assert_eq!(or, PATTERN, "zmm strided read must see every word");
}

#[test]
fn amx_copy() {
    let Some(amx) = amx() else { return };
    let half = GROUP_U64 * 2;
    let mut store = buf(half * 2);
    let all = words(&mut store);
    let (src, dst) = all.split_at_mut(half);
    for (i, w) in src.iter_mut().enumerate() {
        *w = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    tiles::k_amx_copy(amx, dst, src);
    assert_eq!(&*dst, &*src);
    // An odd number of tiles exercises the single-tile remainder.
    let n = 3 * 128;
    dst[..n].fill(0);
    tiles::k_amx_copy(amx, &mut dst[..n], &src[..n]);
    assert_eq!(&dst[..n], &src[..n]);
}

/// TMR's mirror one line step at a time: step x swaps line x with line n-1-x.
fn mirror_ref(b: &mut [u64], line_steps: usize) {
    let n = b.len() / 8;
    for x in 0..line_steps {
        for k in 0..8 {
            b.swap(x * 8 + k, (n - 1 - x) * 8 + k);
        }
    }
}

/// The tile mirror pairs lines as TMR does (the -64 stride really reads and writes the high end
/// downwards), at every point of the round trip, and a whole round trip restores the chunk.
#[test]
fn amx_mirror_matches_tmrs_pairing() {
    let Some(amx) = amx() else { return };
    let n_lines = 256; // 16 KiB: 16 steps of 16 lines
    let mut store = buf(n_lines * 8);
    let orig: Vec<u64> = (0..n_lines as u64 * 8)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    for steps in [2, 6, 8, 10, 16] {
        let b = words(&mut store);
        b.copy_from_slice(&orig);
        tiles::amx_mirror_steps(amx, b, steps);
        let mut want = orig.clone();
        mirror_ref(&mut want, steps * 16);
        assert_eq!(&*b, &want[..], "after {steps} steps");
    }
    // Half way, every line is mirrored; the full round trip is the identity.
    let b = words(&mut store);
    b.copy_from_slice(&orig);
    tiles::amx_mirror_steps(amx, b, 8);
    for i in 0..n_lines {
        assert_eq!(
            b[i * 8..i * 8 + 8],
            orig[(n_lines - 1 - i) * 8..][..8],
            "line {i}"
        );
    }
    let b = &mut words(&mut store)[..8 * 64];
    let odd = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tiles::amx_mirror_steps(amx, b, 3)
    }));
    assert!(odd.is_err(), "odd steps must be refused");
}

/// The bench entries leave memory as they found it (one round trip per 4 MiB chunk), on more
/// than one chunk, and so does TMR's zmm swap.
#[test]
fn mirror_entries_round_trip() {
    let Some(amx) = amx() else { return };
    let n = MIRROR_CHUNK_U64 * 2 + 4096;
    let mut store = buf(n);
    let b = words(&mut store);
    for (i, w) in b.iter_mut().enumerate() {
        *w = (i as u64) ^ POS_BASE;
    }
    let positional = |b: &[u64]| b.iter().enumerate().all(|(i, &w)| w == i as u64 ^ POS_BASE);
    tiles::k_amx_mirror(amx, b);
    assert!(positional(b), "amx round trip");
    if Level::new().as_avx512().is_some() {
        // SAFETY: aligned, a multiple of 8 u64; AVX-512F checked.
        unsafe { crate::tmr::k_mirror_tmr_512(b.as_mut_ptr(), b.len()) };
        assert!(positional(b), "zmm round trip");
    }
}

#[test]
fn amx_read_rejects_partial_tiles() {
    let Some(amx) = amx() else { return };
    let mut store = buf(256);
    let b = &words(&mut store)[..200];
    assert!(std::panic::catch_unwind(|| tiles::k_amx_read(amx, b)).is_err());
}
