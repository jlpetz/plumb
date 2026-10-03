// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tile loads and stores: strides, offsets, overlap, every register, and the bounds checks.

use std::ops::Range;

use plumb_tiles::{ROW_BYTES, ROWS, T0, T1, T2, T3, T4, T5, T6, T7, TILE_BYTES, tile_span};

use crate::{Lcg, amx_or_skip, panic_message};

/// What untouched destination bytes hold.
const SENTINEL: u8 = 0xEE;

/// Bytes of row `r` of a tile placed `offset` bytes into a buffer at `stride`.
fn row(offset: usize, stride: usize, r: usize) -> Range<usize> {
    let start = offset + r * stride;
    start..start + ROW_BYTES
}

/// Source data for a tile at (`offset`, `stride`): random bytes, with distinct rows.
fn source(seed: u64, offset: usize, stride: usize) -> Vec<u8> {
    let len = offset + tile_span(stride).unwrap() + 32;
    let src = Lcg::new(seed).bytes(len);
    if stride >= ROW_BYTES {
        for a in 0..ROWS {
            for b in a + 1..ROWS {
                assert_ne!(
                    src[row(offset, stride, a)],
                    src[row(offset, stride, b)],
                    "test data rows {a}, {b}"
                );
            }
        }
    }
    src
}

#[test]
fn zero_then_store_gives_zeros() {
    let Some(amx) = amx_or_skip("zero_then_store_gives_zeros") else {
        return;
    };
    let ones = vec![0xFF_u8; TILE_BYTES];
    let mut out = vec![SENTINEL; TILE_BYTES];
    amx.with_tiles(|t| {
        t.load::<T5>(&ones, ROW_BYTES);
        t.zero::<T5>();
        t.store::<T5>(&mut out, ROW_BYTES);
    });
    assert!(out.iter().all(|&b| b == 0));
}

#[test]
fn round_trip_at_every_stride_and_offset() {
    let Some(amx) = amx_or_skip("round_trip_at_every_stride_and_offset") else {
        return;
    };
    for (i, &stride) in [64_usize, 72, 200, 4096, 8192].iter().enumerate() {
        for offset in [0_usize, 1, 37] {
            let src = source(i as u64 * 100 + offset as u64, offset, stride);
            let mut same = vec![SENTINEL; src.len()];
            let mut packed = vec![SENTINEL; TILE_BYTES];
            amx.with_tiles(|t| {
                t.load::<T3>(&src[offset..], stride);
                t.store::<T3>(&mut same[offset..], stride);
                t.store::<T3>(&mut packed, ROW_BYTES);
            });
            let mut in_row = vec![false; src.len()];
            for r in 0..ROWS {
                let rr = row(offset, stride, r);
                assert_eq!(
                    same[rr.clone()],
                    src[rr.clone()],
                    "stride {stride} offset {offset} row {r}"
                );
                assert_eq!(
                    packed[row(0, ROW_BYTES, r)],
                    src[rr.clone()],
                    "packed, stride {stride} row {r}"
                );
                in_row[rr].fill(true);
            }
            let stray = (0..src.len()).find(|&b| !in_row[b] && same[b] != SENTINEL);
            assert_eq!(
                stray, None,
                "stride {stride} offset {offset}: byte written between rows"
            );
        }
    }
}

#[test]
fn load_t1_reads_the_same_as_load() {
    let Some(amx) = amx_or_skip("load_t1_reads_the_same_as_load") else {
        return;
    };
    for (i, &stride) in [64_usize, 72, 4096].iter().enumerate() {
        let src = source(7 + i as u64, 5, stride);
        let (mut a, mut b) = (vec![0_u8; TILE_BYTES], vec![1_u8; TILE_BYTES]);
        amx.with_tiles(|t| {
            t.load::<T0>(&src[5..], stride);
            t.load_t1::<T1>(&src[5..], stride);
            t.store::<T0>(&mut a, ROW_BYTES);
            t.store::<T1>(&mut b, ROW_BYTES);
        });
        assert_eq!(a, b, "stride {stride}");
    }
}

#[test]
fn stride_zero_load_repeats_the_first_row() {
    let Some(amx) = amx_or_skip("stride_zero_load_repeats_the_first_row") else {
        return;
    };
    let src = Lcg::new(42).bytes(ROW_BYTES + 3);
    let mut out = vec![SENTINEL; TILE_BYTES];
    amx.with_tiles(|t| {
        t.load::<T6>(&src[3..], 0);
        t.store::<T6>(&mut out, ROW_BYTES);
    });
    for r in 0..ROWS {
        assert_eq!(out[row(0, ROW_BYTES, r)], src[3..], "row {r}");
    }
}

/// Strides below 64 (overlapping rows) and 0 (one row 16 times) through every load variant:
/// row `r` is still exactly the 64 bytes at `r * stride`, from an exact-fit buffer.
#[test]
fn small_stride_loads_read_overlapping_rows() {
    let Some(amx) = amx_or_skip("small_stride_loads_read_overlapping_rows") else {
        return;
    };
    for stride in [0_usize, 1, 8, 32, 63] {
        let span = tile_span(stride).unwrap();
        let src = Lcg::new(50 + stride as u64).bytes(span + 3);
        for offset in [0_usize, 3] {
            let s = &src[offset..][..span];
            let (mut a, mut b) = (vec![SENTINEL; TILE_BYTES], vec![SENTINEL; TILE_BYTES]);
            amx.with_tiles(|t| {
                t.load::<T0>(s, stride);
                t.load_t1::<T1>(s, stride);
                t.store::<T0>(&mut a, ROW_BYTES);
                t.store::<T1>(&mut b, ROW_BYTES);
            });
            for r in 0..ROWS {
                let want = &s[row(0, stride, r)];
                assert_eq!(
                    a[row(0, ROW_BYTES, r)],
                    *want,
                    "load, stride {stride} offset {offset} row {r}"
                );
                assert_eq!(
                    b[row(0, ROW_BYTES, r)],
                    *want,
                    "load_t1, stride {stride} offset {offset} row {r}"
                );
            }
        }
        let mut lcg = Lcg::new(60 + stride as u64);
        let words: Vec<u64> = (0..span.div_ceil(8) + 1)
            .map(|_| u64::from(lcg.next_u32()) << 32 | u64::from(lcg.next_u32()))
            .collect();
        for offset in [0_usize, 1] {
            let w = &words[offset..];
            let (mut a, mut b) = (vec![SENTINEL; TILE_BYTES], vec![SENTINEL; TILE_BYTES]);
            amx.with_tiles(|t| {
                t.load_u64::<T0>(w, stride);
                t.load_t1_u64::<T1>(w, stride);
                t.store::<T0>(&mut a, ROW_BYTES);
                t.store::<T1>(&mut b, ROW_BYTES);
            });
            let wb = bytes(w);
            for r in 0..ROWS {
                let want = &wb[row(0, stride, r)];
                assert_eq!(
                    a[row(0, ROW_BYTES, r)],
                    *want,
                    "load_u64, stride {stride} row {r}"
                );
                assert_eq!(
                    b[row(0, ROW_BYTES, r)],
                    *want,
                    "load_t1_u64, stride {stride} row {r}"
                );
            }
        }
    }
}

#[test]
fn overlapping_stores_write_rows_in_order() {
    let Some(amx) = amx_or_skip("overlapping_stores_write_rows_in_order") else {
        return;
    };
    let src = Lcg::new(9).bytes(TILE_BYTES);
    let rows = src.as_chunks::<ROW_BYTES>().0;
    let mut s0 = vec![SENTINEL; ROW_BYTES];
    let mut s32 = vec![SENTINEL; tile_span(32).unwrap()];
    amx.with_tiles(|t| {
        t.load::<T2>(&src, ROW_BYTES);
        t.store::<T2>(&mut s0, 0);
        t.store::<T2>(&mut s32, 32);
    });
    assert_eq!(s0, rows[15], "stride 0: the last row wins");
    // At stride 32 row r covers 32-byte chunks r and r + 1, so chunk j ends up as the first
    // half of row j, except the last chunk, which only row 15 writes.
    for j in 0..ROWS {
        assert_eq!(s32[j * 32..][..32], rows[j][..32], "stride 32 chunk {j}");
    }
    assert_eq!(s32[ROWS * 32..], rows[15][32..]);
}

/// Loads eight different tiles into T0..T7 and stores them back: no two names alias.
#[test]
fn all_eight_tiles_are_independent() {
    let Some(amx) = amx_or_skip("all_eight_tiles_are_independent") else {
        return;
    };
    let src = Lcg::new(77).bytes(8 * TILE_BYTES);
    let mut out = vec![SENTINEL; 8 * TILE_BYTES];
    macro_rules! each {
        ($t:ident, $($reg:ident = $i:literal),*) => {{
            $( $t.load::<$reg>(&src[$i * TILE_BYTES..], ROW_BYTES); )*
            $( $t.store::<$reg>(&mut out[$i * TILE_BYTES..], ROW_BYTES); )*
        }};
    }
    amx.with_tiles(|t| {
        each!(
            t,
            T0 = 0,
            T1 = 1,
            T2 = 2,
            T3 = 3,
            T4 = 4,
            T5 = 5,
            T6 = 6,
            T7 = 7
        );
    });
    for i in 0..8 {
        assert_eq!(
            out[i * TILE_BYTES..][..TILE_BYTES],
            src[i * TILE_BYTES..][..TILE_BYTES],
            "tmm{i}"
        );
    }
}

#[test]
fn u64_variants_round_trip_with_byte_strides() {
    let Some(amx) = amx_or_skip("u64_variants_round_trip_with_byte_strides") else {
        return;
    };
    for stride in [64_usize, 72, 4096] {
        let words = tile_span(stride).unwrap().div_ceil(8) + 1;
        let mut lcg = Lcg::new(stride as u64);
        let src: Vec<u64> = (0..words)
            .map(|_| u64::from(lcg.next_u32()) << 32 | u64::from(lcg.next_u32()))
            .collect();
        let mut dst = vec![u64::from_ne_bytes([SENTINEL; 8]); words];
        let mut dst_t1 = dst.clone();
        amx.with_tiles(|t| {
            t.load_u64::<T0>(&src, stride);
            t.store_u64::<T0>(&mut dst, stride);
            t.load_t1_u64::<T1>(&src, stride);
            t.store_u64::<T1>(&mut dst_t1, stride);
        });
        let (sb, db, tb) = (bytes(&src), bytes(&dst), bytes(&dst_t1));
        for r in 0..ROWS {
            assert_eq!(
                db[row(0, stride, r)],
                sb[row(0, stride, r)],
                "stride {stride} row {r}"
            );
        }
        assert_eq!(db, tb, "load_t1_u64, stride {stride}");
        if stride > ROW_BYTES {
            assert!(
                db[ROW_BYTES..stride].iter().all(|&b| b == SENTINEL),
                "gap after row 0"
            );
        }
    }
}

fn bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_ne_bytes()).collect()
}

#[track_caller]
fn assert_out_of_bounds(f: impl FnOnce()) {
    let msg = panic_message(f);
    assert!(
        msg.contains("tile access out of bounds"),
        "unexpected panic: {msg}"
    );
}

/// Strides whose span overflows `usize` (the first is the smallest such stride), plus one that
/// doesn't overflow but is far too large for any real buffer.
const HUGE_STRIDES: [usize; 5] = [
    (usize::MAX - ROW_BYTES) / (ROWS - 1) + 1,
    usize::MAX / 2,
    isize::MAX as usize,
    usize::MAX,
    1 << 40,
];

#[test]
fn loads_reject_short_buffers_and_overflowing_strides() {
    let Some(amx) = amx_or_skip("loads_reject_short_buffers_and_overflowing_strides") else {
        return;
    };
    for stride in [0_usize, 1, 64, 72, 4096] {
        let span = tile_span(stride).unwrap();
        let buf = vec![0_u8; span];
        // An exact fit is fine.
        amx.with_tiles(|t| {
            t.load::<T0>(&buf, stride);
            t.load_t1::<T1>(&buf, stride);
        });
        let short = &buf[..span - 1];
        assert_out_of_bounds(|| amx.with_tiles(|t| t.load::<T0>(short, stride)));
        assert_out_of_bounds(|| amx.with_tiles(|t| t.load_t1::<T0>(short, stride)));
    }
    assert_out_of_bounds(|| amx.with_tiles(|t| t.load::<T0>(&[], 0)));
    let big = vec![0_u8; 1 << 20];
    let words = vec![0_u64; 1 << 17];
    for stride in HUGE_STRIDES {
        assert_out_of_bounds(|| amx.with_tiles(|t| t.load::<T0>(&big, stride)));
        assert_out_of_bounds(|| amx.with_tiles(|t| t.load_t1::<T0>(&big, stride)));
        assert_out_of_bounds(|| amx.with_tiles(|t| t.load_u64::<T0>(&words, stride)));
        assert_out_of_bounds(|| amx.with_tiles(|t| t.load_t1_u64::<T0>(&words, stride)));
    }
    // u64 buffers are measured in bytes: 128 words hold a packed tile, 127 don't.
    amx.with_tiles(|t| t.load_u64::<T0>(&words[..128], ROW_BYTES));
    assert_out_of_bounds(|| amx.with_tiles(|t| t.load_u64::<T0>(&words[..127], ROW_BYTES)));
    assert_out_of_bounds(|| amx.with_tiles(|t| t.load_t1_u64::<T0>(&words[..127], ROW_BYTES)));
    let need = tile_span(4096).unwrap() / 8; // 61504 bytes = 7688 words exactly
    amx.with_tiles(|t| t.load_u64::<T0>(&words[..need], 4096));
    assert_out_of_bounds(|| amx.with_tiles(|t| t.load_u64::<T0>(&words[..need - 1], 4096)));
}

#[test]
fn stores_reject_short_buffers_and_write_nothing() {
    let Some(amx) = amx_or_skip("stores_reject_short_buffers_and_write_nothing") else {
        return;
    };
    let ones = vec![0xFF_u8; TILE_BYTES];
    for stride in [0_usize, 64, 72, 4096] {
        let span = tile_span(stride).unwrap();
        let mut buf = vec![SENTINEL; span];
        assert_out_of_bounds(|| {
            amx.with_tiles(|t| {
                t.load::<T0>(&ones, ROW_BYTES);
                t.store::<T0>(&mut buf[..span - 1], stride);
            });
        });
        assert!(
            buf.iter().all(|&b| b == SENTINEL),
            "stride {stride}: a rejected store wrote"
        );
        amx.with_tiles(|t| {
            t.load::<T0>(&ones, ROW_BYTES);
            t.store::<T0>(&mut buf, stride);
        });
        assert_eq!(
            buf[span - 1],
            0xFF,
            "stride {stride}: exact fit reaches the last byte"
        );
    }
    let mut big = vec![SENTINEL; 1 << 20];
    let mut words = vec![0_u64; 1 << 17];
    for stride in HUGE_STRIDES {
        assert_out_of_bounds(|| amx.with_tiles(|t| t.store::<T0>(&mut big, stride)));
        assert_out_of_bounds(|| amx.with_tiles(|t| t.store_u64::<T0>(&mut words, stride)));
    }
    assert!(big.iter().all(|&b| b == SENTINEL) && words.iter().all(|&w| w == 0));
    assert_out_of_bounds(|| amx.with_tiles(|t| t.store_u64::<T0>(&mut words[..127], ROW_BYTES)));
    assert_out_of_bounds(|| amx.with_tiles(|t| t.store::<T0>(&mut [], 0)));
}
