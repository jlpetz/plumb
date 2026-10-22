// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Correctness of the transform-verify prototype (xform.rs) on the real CPU. Skips, with a
//! message, when the CPU or OS has no AMX-INT8.
//!
//! The prototype plan's test list (design doc): expected values match an independent
//! computation (enforced inside `PatternSet::new` itself, so every test here that builds a set
//! already exercises it); a clean pass reports no errors; every single-bit error is detected,
//! both in the stored result and in the pattern before the transform; two-bit errors; tier 2
//! reports the right tile (and, where its classifier recognizes the signature, the right byte
//! and bit).

use plumb_tiles::AmxInt8;

use crate::xform::{self, PatternSet, TileError};

fn int8() -> Option<AmxInt8> {
    let t = AmxInt8::try_new();
    if t.is_none() {
        eprintln!("skip: no AMX-INT8 on this CPU/OS");
    }
    t
}

fn buf(words: usize) -> Vec<u64> {
    vec![0u64; words]
}

/// A small set for most tests: 16 patterns, 1 check (`r` = 0), fixed seed.
fn small_set(int8: AmxInt8) -> PatternSet {
    PatternSet::new(int8, 16, 1, 0x1234_5678_9ABC_DEF0)
}

#[test]
fn x_and_checks_are_range_checked() {
    let Some(int8) = int8() else { return };
    // x must be a power of two in 2..=64.
    assert!(std::panic::catch_unwind(|| PatternSet::new(int8, 3, 1, 1)).is_err());
    assert!(std::panic::catch_unwind(|| PatternSet::new(int8, 128, 1, 1)).is_err());
    // checks (1 + r) must leave 2 tiles for the operands.
    assert!(std::panic::catch_unwind(|| PatternSet::new(int8, 16, 7, 1)).is_err());
    assert!(std::panic::catch_unwind(|| PatternSet::new(int8, 16, 0, 1)).is_err());
    // The design doc's suggested sizes should all build (and self-check) fine.
    for x in [16usize, 32, 64] {
        for checks in [1usize, 2, 6] {
            let set = PatternSet::new(int8, x, checks, 0xA5A5_A5A5);
            assert_eq!(set.x(), x);
            assert_eq!(set.checks(), checks);
        }
    }
}

/// `PatternSet::new` itself asserts the AMX-computed expected values match the independent
/// scalar reference (see its doc); building sets at every size and several seeds, with `r` up
/// to 5, exercises that broadly. If the multiply unit disagreed with the scalar path, these
/// would have already panicked before this test's own assertions run.
#[test]
fn expected_values_are_self_consistent_across_sizes_and_seeds() {
    let Some(int8) = int8() else { return };
    for x in [16usize, 32, 64] {
        for checks in [1usize, 2, 6] {
            for seed in [0u64, 1, 0xDEAD_BEEF, u64::MAX] {
                PatternSet::new(int8, x, checks, seed);
            }
        }
    }
}

#[test]
fn a_clean_pass_reports_no_errors() {
    let Some(int8) = int8() else { return };
    let set = small_set(int8);
    for groups in [1usize, 3] {
        let mut b = buf(set.group_words() * groups);
        for pass in 0..set.x() {
            let mut errors = Vec::new();
            let bad = xform::pass(int8, &set, &mut b, pass, &mut |e| errors.push(e));
            assert_eq!(bad, 0, "pass {pass}: {groups} groups, bad count");
            assert!(
                errors.is_empty(),
                "pass {pass}: tier 2 fired with no tier-1 mismatch"
            );
        }
    }
}

/// After a clean pass, each tile's stored bytes equal exactly the pattern the rotation says
/// belongs there, transformed — confirms the rotation and the transform agree with each other
/// (not just that tier 1 happens to pass).
#[test]
fn each_tile_holds_its_rotated_patterns_transform() {
    let Some(int8) = int8() else { return };
    let set = small_set(int8);
    let mut b = buf(set.group_words() * 2);
    for pass in [0usize, 1, set.x() - 1, set.x(), set.x() + 3] {
        xform::pass(int8, &set, &mut b, pass, &mut |e| {
            panic!("clean buffer: {e:?}")
        });
        for (g, group) in b.chunks(set.group_words()).enumerate() {
            for (t, tile) in group.chunks(xform::TILE_U64).enumerate() {
                let pi = set.rotation(g, pass, t);
                assert_eq!(
                    tile,
                    set.expected(pi).0,
                    "pass {pass} group {g} tile {t}: expected pattern {pi}'s transform"
                );
            }
        }
    }
}

/// Fault injection: flipping one bit in the *stored* result (after RW, before R) must be
/// caught by tier 1 and located exactly by tier 2. `fill` runs once; the corrupted buffer is
/// then checked with `verify` alone — calling `pass` (which re-fills from the clean pattern)
/// here would overwrite the injected bit before anything saw it.
#[test]
fn single_bit_in_the_stored_result_is_always_detected() {
    let Some(int8) = int8() else { return };
    let set = small_set(int8);
    let mut b = buf(set.group_words());
    // One representative byte/bit per pattern tile, plus the tile's first and last word, to
    // catch an off-by-one at the tile boundary.
    for tile_idx in 0..set.x() {
        for &(word, bit) in &[(0usize, 0u32), (0, 7), (63, 0), (127, 63), (64, 32)] {
            xform::fill(int8, &set, &mut b, 0);
            let at = tile_idx * xform::TILE_U64 + word;
            b[at] ^= 1u64 << bit;
            let mut errors = Vec::new();
            let bad = xform::verify(int8, &set, &b, 0, &mut |e| errors.push(e));
            assert_eq!(
                bad, 1,
                "tile {tile_idx} word {word} bit {bit}: group not flagged"
            );
            assert!(
                errors.iter().any(|e: &TileError| e.tile == tile_idx),
                "tile {tile_idx} word {word} bit {bit}: tier 2 didn't name the bad tile (found {:?})",
                errors.iter().map(|e| e.tile).collect::<Vec<_>>()
            );
        }
    }
}

/// Fault injection: flipping one bit of a *pattern* before RW reads it, through `fill_inject`'s
/// hook, then `verify`. Every case must be caught, and tier 2's classifier should recover the
/// exact byte and bit (the signature a bit propagated through the transform matrix leaves).
#[test]
fn single_bit_before_the_transform_is_detected_and_located() {
    let Some(int8) = int8() else { return };
    let set = small_set(int8);
    let mut group = buf(set.group_words());
    for tile_idx in 0..set.x() {
        // `fill_inject` always runs on group 0: this is the pattern the rotation puts at this
        // tile position there, for pass 0.
        let pi = set.rotation(0, 0, tile_idx);
        for byte in [0usize, 31, 63] {
            for bit in 0..8u32 {
                let word = byte / 8;
                let shift = byte % 8 * 8 + bit as usize;
                // The sign tier 2's classifier should report: +1 if the bit is currently 0
                // (flipping sets it, increasing the byte), -1 if it's currently 1.
                let original = (set.pattern(pi).0[word] >> shift) & 1;
                let sign = if original == 0 { 1i64 } else { -1 };

                xform::fill_inject(int8, &set, &mut group, 0, &mut |g| {
                    let word_idx = tile_idx * xform::TILE_U64 + word;
                    g[word_idx] ^= 1u64 << shift;
                });
                let mut errors = Vec::new();
                let bad = xform::verify(int8, &set, &group, 0, &mut |e| errors.push(e));
                assert!(
                    bad > 0,
                    "tile {tile_idx} byte {byte} bit {bit}: not flagged"
                );
                let hit = errors.iter().find(|e: &&TileError| e.tile == tile_idx);
                let hit = hit.unwrap_or_else(|| {
                    panic!(
                        "tile {tile_idx} byte {byte} bit {bit}: tier 2 didn't name the bad tile \
                         (found {:?})",
                        errors.iter().map(|e| e.tile).collect::<Vec<_>>()
                    )
                });
                assert_eq!(
                    hit.before_transform,
                    Some((byte, bit, sign)),
                    "tile {tile_idx} byte {byte} bit {bit}: classifier result"
                );
            }
        }
    }
}

/// Two flipped bits in a group: the design's distinct-rows construction should still catch
/// them (tested by injection, not proof — the design doc's own framing).
#[test]
fn two_bit_errors_in_a_group_are_detected() {
    let Some(int8) = int8() else { return };
    let set = small_set(int8);
    let mut b = buf(set.group_words());
    let positions: Vec<(usize, u32)> = (0..set.x())
        .flat_map(|t| {
            [
                (t * xform::TILE_U64, 3u32),
                (t * xform::TILE_U64 + 1, 50u32),
            ]
        })
        .collect();
    let mut undetected = Vec::new();
    for i in 0..positions.len() {
        for j in i + 1..positions.len() {
            xform::fill(int8, &set, &mut b, 0);
            let (w1, bit1) = positions[i];
            let (w2, bit2) = positions[j];
            b[w1] ^= 1u64 << bit1;
            b[w2] ^= 1u64 << bit2;
            let bad = xform::verify(int8, &set, &b, 0, &mut |_| {});
            if bad == 0 {
                undetected.push((i, j));
            }
        }
    }
    assert!(
        undetected.is_empty(),
        "{} two-bit combinations went undetected: {undetected:?}",
        undetected.len()
    );
}

/// `checks` > 1 (`r` > 0) only adds verification work; it must never turn a clean pass into a
/// false positive.
#[test]
fn extra_checks_dont_cause_false_positives() {
    let Some(int8) = int8() else { return };
    for checks in [1usize, 3, 6] {
        let set = PatternSet::new(int8, 16, checks, 0x9999);
        let mut b = buf(set.group_words());
        let bad = xform::pass(int8, &set, &mut b, 0, &mut |e| {
            panic!("checks={checks}: {e:?}")
        });
        assert_eq!(bad, 0, "checks={checks}: clean pass flagged");
    }
}

/// Every extra check (`r` > 0) also independently catches a corrupted group, not just the
/// first ("real") one — confirms the load dial really does add independent verification, not
/// just busywork.
#[test]
fn every_extra_check_independently_catches_corruption() {
    let Some(int8) = int8() else { return };
    for checks in [2usize, 4, 6] {
        let set = PatternSet::new(int8, 16, checks, 0x1357_9bdf);
        let mut b = buf(set.group_words());
        xform::fill(int8, &set, &mut b, 0);
        b[0] ^= 1;
        let bad = xform::verify(int8, &set, &b, 0, &mut |_| {});
        assert_eq!(bad, 1, "checks={checks}: corrupted group not flagged");
    }
}
