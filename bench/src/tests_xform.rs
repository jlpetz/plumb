// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Correctness of the transform-verify prototype (xform.rs) on the real CPU. Skips, with a
//! message, when the CPU or OS has no AMX-INT8.
//!
//! The design's test list (prototype plan, item 2): the expected values computed on the tiles
//! match an independent computation (the scalar reference here, which never touches AMX); the
//! built-in set's stored values are right; a clean pass reports no errors; every single-bit
//! error is detected, in the stored result and in the pattern before the transform, both
//! exhaustively for the built-in set; two-bit errors; tier 2 reports the right words. Also the
//! construction the guarantees rest on, the rotation's two properties, and three or more flips
//! measured by random injection.
//!
//! The injection sweeps put one fault in each group of a batch: each group's checksum is its
//! own, so every group's report is one case.

use std::collections::HashSet;

use plumb_tiles::AmxInt8;

use crate::xform::{
    self, BUILTIN_CHECKSUM, BUILTIN_SEED, BadWord, GroupError, MAX_R, Matrix, PatternSet,
    REREAD_RESCANS, RESCAN_LOGGED, Reread, Rescans, Signature, TILE_U64, Tile,
};

fn int8() -> Option<AmxInt8> {
    let t = AmxInt8::try_new();
    if t.is_none() {
        eprintln!("skip: no AMX-INT8 on this CPU/OS");
    }
    t
}

/// A buffer of `groups` groups.
fn buf<const X: usize>(groups: usize) -> Vec<u64> {
    vec![0; PatternSet::<X>::group_words() * groups]
}

/// Bits in a tile.
const TILE_BITS: usize = TILE_U64 * 64;

/// Bit `bit` of a tile's words (byte `bit / 8`, bit `bit % 8` of it, little-endian).
fn get_bit(words: &[u64], bit: usize) -> bool {
    (words[bit / 64] >> (bit % 64)) & 1 == 1
}

fn flip_bit(words: &mut [u64], bit: usize) {
    words[bit / 64] ^= 1 << (bit % 64);
}

// ----- The scalar reference ------------------------------------------------------------------

/// Byte `i` of a tile's words, little-endian.
fn byte(words: &[u64; TILE_U64], i: usize) -> u8 {
    (words[i / 8] >> (i % 8 * 8)) as u8
}

/// `A x B` as TDPBUUD defines it, `C[m][n] = sum(j < 64) A[m][j] * B[j][n]` (u8 x u8 into u32,
/// wrapping), in plain scalar code that never touches AMX: the independent computation.
/// `A` is 16 rows of 64 bytes, `B` the VNNI-packed matrix, `C` in TILESTORED's layout.
fn scalar_multiply(a: &[u64; TILE_U64], b: &Matrix) -> Tile {
    let c: [u32; 256] = std::array::from_fn(|mn| {
        let (m, n) = (mn / 16, mn % 16);
        (0..64).fold(0u32, |s, j| {
            s.wrapping_add(u32::from(byte(a, m * 64 + j)) * u32::from(b.get(j, n)))
        })
    });
    Tile(std::array::from_fn(|w| {
        u64::from(c[2 * w]) | (u64::from(c[2 * w + 1]) << 32)
    }))
}

/// `T(P_i)` for each pattern and the group checksum at `r` = 0, by the scalar reference.
fn scalar_values<const X: usize>(set: &PatternSet<X>) -> (Vec<Tile>, Tile) {
    let t: Vec<Tile> = (0..X)
        .map(|i| scalar_multiply(&set.pattern(i).0, set.transform_matrix()))
        .collect();
    let mut sum = [0u64; TILE_U64];
    for (i, ti) in t.iter().enumerate() {
        let p = scalar_multiply(&ti.0, set.key(i));
        for (s, q) in sum.iter_mut().zip(p.0) {
            *s = lanes(*s, q, u32::wrapping_add);
        }
    }
    (t, Tile(sum))
}

/// Two words of u32 lanes combined lane by lane with `f`.
fn lanes(a: u64, b: u64, f: impl Fn(u32, u32) -> u32) -> u64 {
    let lo = f(a as u32, b as u32);
    let hi = f((a >> 32) as u32, (b >> 32) as u32);
    u64::from(lo) | (u64::from(hi) << 32)
}

/// `n` times a checksum, lane by lane, wrapping: what `n` repeats of every product add up to.
fn scale(e: &Tile, n: usize) -> Tile {
    let n = n as u64 | ((n as u64) << 32);
    Tile(e.0.map(|w| lanes(w, n, u32::wrapping_mul)))
}

// ----- The set and its expected values -------------------------------------------------------

#[test]
fn r_is_range_checked() {
    let Some(int8) = int8() else { return };
    // 1 + r times a fault's change must stay below 2^32 (X is a const generic, checked at
    // compile time: 16, 32 or 64).
    assert!(std::panic::catch_unwind(|| PatternSet::<16>::new(int8, MAX_R + 1, 1)).is_err());
    for r in [0, 1, MAX_R] {
        assert_eq!(PatternSet::<16>::new(int8, r, 1).r(), r);
    }
}

/// What the guarantees rest on ([`PatternSet::new`]): every byte odd, the 16-byte rows distinct
/// across the set's matrices, the transform's rows distinct.
fn assert_odd_and_distinct<const X: usize>(set: &PatternSet<X>, what: &str) {
    let odd = |m: &Matrix| {
        m.0.iter()
            .all(|w| w & 0x0101_0101_0101_0101 == 0x0101_0101_0101_0101)
    };
    let k = set.transform_matrix();
    assert!(odd(k), "{what}: transform has an even byte");
    let rows: HashSet<[u8; 16]> = (0..64).map(|j| k.row(j)).collect();
    assert_eq!(rows.len(), 64, "{what}: transform rows repeat");
    let mut rows = HashSet::new();
    for i in 0..X {
        let m = set.key(i);
        assert!(odd(m), "{what}: pattern {i}'s matrix has an even byte");
        for j in 0..64 {
            assert!(
                rows.insert(m.row(j)),
                "{what}: row {j} of pattern {i}'s matrix repeats"
            );
        }
    }
}

#[test]
fn matrices_are_odd_with_distinct_rows() {
    let Some(int8) = int8() else { return };
    for seed in [0u64, 1, BUILTIN_SEED, u64::MAX] {
        assert_odd_and_distinct(&PatternSet::<16>::new(int8, 0, seed), "X=16");
        assert_odd_and_distinct(&PatternSet::<32>::new(int8, 0, seed), "X=32");
        assert_odd_and_distinct(&PatternSet::<64>::new(int8, 0, seed), "X=64");
    }
}

fn assert_matches_scalar<const X: usize>(int8: AmxInt8, r: usize, seed: u64) {
    let set = PatternSet::<X>::new(int8, r, seed);
    let (t, e) = scalar_values(&set);
    for (i, ti) in t.iter().enumerate() {
        assert_eq!(set.expected(i), ti, "X={X} seed {seed:#x}: T(P_{i})");
    }
    assert_eq!(set.checksum(), &e, "X={X} seed {seed:#x}: checksum");
    assert_eq!(
        set.expected_sum(),
        &scale(&e, 1 + r),
        "X={X} seed {seed:#x} r={r}: expected sum"
    );
}

#[test]
fn expected_values_match_an_independent_computation() {
    let Some(int8) = int8() else { return };
    for r in [0, 1, 4, 255] {
        for seed in [0u64, 1, 0xDEAD_BEEF, u64::MAX] {
            assert_matches_scalar::<16>(int8, r, seed);
            assert_matches_scalar::<32>(int8, r, seed);
            assert_matches_scalar::<64>(int8, r, seed);
        }
    }
}

#[test]
fn the_builtin_sets_stored_checksum_is_right() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<16>::new(int8, 0, BUILTIN_SEED);
    let (_, e) = scalar_values(&set);
    assert_eq!(
        e.0, BUILTIN_CHECKSUM,
        "xform_builtin.rs is stale; regenerate it with `cargo test -p plumb-bench --release \
         xform_builtin_regenerate -- --ignored`"
    );
    assert!(
        xform::self_test(int8),
        "the startup check failed on this CPU"
    );
    for r in [0, 1, 8] {
        assert!(PatternSet::<16>::builtin(int8, r, 9).is_some(), "r={r}");
    }
}

/// Writes `src/xform_builtin.rs`: the built-in set's group checksum, from the scalar reference
/// (the tiles build the set's patterns and matrices, which the seed decides, but no value).
#[test]
#[ignore = "writes src/xform_builtin.rs; run after changing how a set is generated"]
fn xform_builtin_regenerate() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<16>::new(int8, 0, BUILTIN_SEED);
    let (_, e) = scalar_values(&set);
    let mut s = String::from(
        "// Copyright 2026 the plumb Authors\n\
         // SPDX-License-Identifier: Apache-2.0 OR MIT\n\n\
         // Generated by `cargo test -p plumb-bench --release xform_builtin_regenerate -- --ignored`\n\
         // (tests_xform.rs) from the scalar reference, never the tiles: the built-in set's group\n\
         // checksum at r = 0 (X = 16, BUILTIN_SEED), as TILESTORED lays it out. Don't edit by\n\
         // hand.\n\n\
         /// The built-in set's group checksum at r = 0 ([`PatternSet::builtin`]).\n\
         pub const BUILTIN_CHECKSUM: [u64; TILE_U64] = [\n",
    );
    for line in e.0.chunks(4) {
        let words: Vec<String> = line.iter().map(|w| format!("{w:#018x}")).collect();
        s.push_str(&format!("    {},\n", words.join(", ")));
    }
    s.push_str("];\n");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/xform_builtin.rs");
    std::fs::write(path, s).expect("write xform_builtin.rs");
}

// ----- Clean passes and the rotation ---------------------------------------------------------

fn clean_passes<const X: usize>(int8: AmxInt8, r: usize) {
    let set = PatternSet::<X>::new(int8, r, 0xC1EA);
    let mut b = buf::<X>(3);
    for pass in 0..2 * X + 1 {
        let errors = xform::pass(int8, &set, &mut b, pass, &mut Rescans::default());
        assert!(errors.is_empty(), "X={X} r={r} pass {pass}: {errors:?}");
        // Each tile holds its rotated pattern's transform: the rotation and the transform agree
        // with each other, not just with the checksum.
        for (g, group) in b.chunks(PatternSet::<X>::group_words()).enumerate() {
            for (t, tile) in group.chunks(TILE_U64).enumerate() {
                let pi = (t + set.offset(g, pass)) % X;
                assert_eq!(
                    tile,
                    set.expected(pi).0,
                    "X={X} pass {pass} group {g} tile {t}"
                );
            }
        }
    }
}

#[test]
fn a_clean_pass_reports_no_errors() {
    let Some(int8) = int8() else { return };
    for r in [0, 3] {
        clean_passes::<16>(int8, r);
        clean_passes::<32>(int8, r);
        clean_passes::<64>(int8, r);
    }
}

fn every_pattern_each_cycle<const X: usize>(int8: AmxInt8) {
    let set = PatternSet::<X>::new(int8, 0, 0xC1C1E);
    for g in 0..64 {
        for cycle in 0..3 {
            let seen: HashSet<usize> = (0..X).map(|p| set.offset(g, cycle * X + p)).collect();
            assert_eq!(seen.len(), X, "X={X} group {g} cycle {cycle}");
        }
    }
}

#[test]
fn every_tile_holds_every_pattern_each_cycle() {
    let Some(int8) = int8() else { return };
    every_pattern_each_cycle::<16>(int8);
    every_pattern_each_cycle::<32>(int8);
    every_pattern_each_cycle::<64>(int8);
}

/// Groups an addressing fault would pair (one address bit apart: neighbours and every
/// power-of-two distance) hold the same data for a whole cycle about 1 time in `X^4 / 4`
/// (16,384 at X = 16), never for two cycles running; and the layout follows the run's seed.
#[test]
fn the_rotation_differs_between_groups_and_runs() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<16>::builtin(int8, 0, 0x5EED).expect("self-check");
    let cycle =
        |g: usize, c: usize| -> [usize; 16] { std::array::from_fn(|p| set.offset(g, c * 16 + p)) };
    let (mut pairs, mut blind) = (0, 0);
    for g in 0..4096usize {
        for k in 0..12 {
            let h = g ^ (1 << k);
            if h < g {
                continue;
            }
            pairs += 1;
            let same: Vec<bool> = (0..2).map(|c| cycle(g, c) == cycle(h, c)).collect();
            blind += same.iter().filter(|&&s| s).count();
            assert!(
                !(same[0] && same[1]),
                "groups {g} and {h}: the same data two cycles running"
            );
        }
    }
    // 24,576 pairs x 2 cycles / 16,384: about 3 expected.
    assert!(
        blind <= 12,
        "{blind} of {pairs} pairs x 2 cycles held the same data for a cycle"
    );
    let other = PatternSet::<16>::builtin(int8, 0, 0x5EEE).expect("self-check");
    let differ = (0..64)
        .filter(|&g| cycle(g, 0) != std::array::from_fn(|p| other.offset(g, p)))
        .count();
    assert!(
        differ >= 60,
        "only {differ} of 64 groups' layouts changed with the seed"
    );
}

// ----- Fault injection -----------------------------------------------------------------------

/// Groups per batch in the sweeps.
const BATCH: usize = 512;

/// `cases` fault injections, one per group, `BATCH` groups at a time, on pass `pass`. `before`
/// corrupts group `g`'s pattern between W and RW, `after` its stored result after RW (either may
/// do nothing); `check` gets each case's report, `None` if its group passed.
fn sweep<const X: usize>(
    int8: AmxInt8,
    set: &PatternSet<X>,
    cases: usize,
    pass: usize,
    before: impl Fn(&mut [u64], usize, usize),
    after: impl Fn(&mut [u64], usize, usize),
    mut check: impl FnMut(usize, usize, Option<&GroupError>),
) {
    let gw = PatternSet::<X>::group_words();
    for start in (0..cases).step_by(BATCH) {
        let n = BATCH.min(cases - start);
        let mut b = buf::<X>(n);
        xform::write(int8, set, &mut b, pass);
        for (g, group) in b.chunks_mut(gw).enumerate() {
            before(group, start + g, g);
        }
        xform::transform(int8, set, &mut b);
        for (g, group) in b.chunks_mut(gw).enumerate() {
            after(group, start + g, g);
        }
        let mut rescans = Rescans::default();
        let errors = xform::verify(int8, set, &b, pass, &mut rescans);
        assert_eq!(rescans.alarms, errors.len() as u64);
        let mut by_group: Vec<Option<&GroupError>> = vec![None; n];
        for e in &errors {
            assert!(
                by_group[e.group].replace(e).is_none(),
                "group {} twice",
                e.group
            );
        }
        for (g, e) in by_group.into_iter().enumerate() {
            check(start + g, g, e);
        }
    }
}

/// Every one of the 131,072 single-bit errors a group's stored result can have (each tile, each
/// bit), at r = 3 (every product added four times): caught, and tier 2 names the tile, the word,
/// the byte and bit, rereads it (in each batch's first [`REREAD_RESCANS`] rescans, its budget),
/// and the tiles still see it.
#[test]
fn single_bit_in_the_stored_result_is_caught_and_located() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<16>::builtin(int8, 3, 0x5106).expect("self-check");
    let pass = 3;
    let case = |c: usize| (c / TILE_BITS, c % TILE_BITS);
    sweep(
        int8,
        &set,
        16 * TILE_BITS,
        pass,
        |_, _, _| {},
        |group, c, _| {
            let (t, bit) = case(c);
            flip_bit(&mut group[t * TILE_U64..][..TILE_U64], bit);
        },
        |c, g, e| {
            let (t, bit) = case(c);
            let e = e.unwrap_or_else(|| panic!("tile {t} bit {bit}: not caught"));
            let [te] = &e.tiles[..] else {
                panic!("tile {t} bit {bit}: tiles {:?}", e.tiles)
            };
            let pattern = (t + set.offset(g, pass)) % 16;
            let expected = set.expected(pattern).0[bit / 64];
            let actual = expected ^ (1 << (bit % 64));
            assert_eq!((te.tile, te.pattern), (t, pattern), "bit {bit}");
            assert_eq!((e.bad_words, te.bad_words), (1, 1), "tile {t} bit {bit}");
            assert_eq!(
                te.words,
                [BadWord {
                    word: bit / 64,
                    expected,
                    actual
                }],
                "tile {t} bit {bit}"
            );
            assert_eq!(
                te.signature,
                Signature::Stored {
                    byte: bit / 8,
                    bit: (bit % 8) as u32
                }
            );
            let reread = Reread {
                tile: t,
                word: bit / 64,
                expected,
                still_wrong: 4,
                after_flush: actual,
                via_tiles: actual,
            };
            let rereads: &[Reread] = if g < REREAD_RESCANS as usize {
                &[reread]
            } else {
                &[]
            };
            assert_eq!(e.rereads, rereads, "tile {t} bit {bit} group {g}");
            assert!(e.tile_recheck, "tile {t} bit {bit}: recheck");
        },
    );
}

/// Every single-bit error in the built-in set's patterns before the transform (all 16 patterns
/// x 8,192 bits, the design's exhaustive check): caught at r = 0, and tier 2 works back to
/// the pattern's byte and bit and the way it flipped. Not guaranteed by construction, so this is
/// a property of the built-in set's matrices.
#[test]
fn single_bit_before_the_transform_is_caught_and_located() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<16>::builtin(int8, 0, 0xBEF0).expect("self-check");
    let pass = 5;
    let case = |c: usize| (c / TILE_BITS, c % TILE_BITS);
    sweep(
        int8,
        &set,
        16 * TILE_BITS,
        pass,
        |group, c, _| {
            let (t, bit) = case(c);
            flip_bit(&mut group[t * TILE_U64..][..TILE_U64], bit);
        },
        |_, _, _| {},
        |c, g, e| {
            let (t, bit) = case(c);
            let e = e.unwrap_or_else(|| panic!("tile {t} bit {bit}: not caught"));
            let [te] = &e.tiles[..] else {
                panic!("tile {t} bit {bit}: tiles {:?}", e.tiles)
            };
            let pattern = (t + set.offset(g, pass)) % 16;
            let rose = !get_bit(&set.pattern(pattern).0, bit);
            assert_eq!((te.tile, te.pattern), (t, pattern), "bit {bit}");
            assert_eq!(
                te.signature,
                Signature::BeforeTransform {
                    byte: bit / 8,
                    bit: (bit % 8) as u32,
                    rose
                },
                "tile {t} bit {bit}"
            );
        },
    );
}

/// The two flips at bit `b` of data row `m` that could cancel, for case `c` of the sweep below:
/// a 0 set in tile `t1` and a 1 cleared in tile `t2`, as bit indices in their tiles. Two tiles
/// take the same byte `j` where their bits allow, since two patterns sharing a matrix would
/// collide there; one tile takes two bytes. `None` if the stored data has no such pair.
fn opposite_flips(stored: &[&[u64; TILE_U64]], c: usize) -> Option<(usize, usize, usize, usize)> {
    let pairs: Vec<(usize, usize)> = (0..16).flat_map(|a| (a..16).map(move |b| (a, b))).collect();
    let (t1, t2) = pairs[c / 128];
    let (m, b) = (c / 8 % 16, c % 8);
    let at = |j: usize| (m * 64 + j) * 8 + b;
    let from = |s: usize| (0..64).map(move |j| (s + j) % 64);
    let (rises, falls) = (
        |j| !get_bit(stored[t1], at(j)),
        |j| get_bit(stored[t2], at(j)),
    );
    let start = (c * 7) % 64;
    let same = (t1 != t2)
        .then(|| from(start).find(|&j| rises(j) && falls(j)))
        .flatten();
    let (j1, j2) = match same {
        Some(j) => (j, j),
        None => {
            let j1 = from(start).find(|&j| rises(j))?;
            (
                j1,
                from(start + 32).find(|&j| (t1, j1) != (t2, j) && falls(j))?,
            )
        }
    };
    Some((t1, at(j1), t2, at(j2)))
}

/// Two flipped bits can only cancel at the same bit of the same data row, in opposite
/// directions (see [`PatternSet::new`]); the distinct rows rule that out. Every such pair of
/// tiles (136 with the same tile twice), in every row, at every bit, at r = 1 (every product
/// added twice, so the change is doubled).
#[test]
fn two_flips_that_could_cancel_are_caught() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<16>::builtin(int8, 1, 0x2B17).expect("self-check");
    let pass = 7;
    let stored = |g: usize| -> Vec<&[u64; TILE_U64]> {
        (0..16)
            .map(|t| &set.expected((t + set.offset(g, pass)) % 16).0)
            .collect()
    };
    let cases = 136 * 16 * 8;
    let mut injected = 0;
    sweep(
        int8,
        &set,
        cases,
        pass,
        |_, _, _| {},
        |group, c, g| {
            if let Some((t1, bit1, t2, bit2)) = opposite_flips(&stored(g), c) {
                flip_bit(&mut group[t1 * TILE_U64..][..TILE_U64], bit1);
                flip_bit(&mut group[t2 * TILE_U64..][..TILE_U64], bit2);
            }
        },
        |c, g, e| {
            if let Some(flips) = opposite_flips(&stored(g), c) {
                injected += 1;
                assert!(
                    e.is_some_and(GroupError::reproduced),
                    "case {c} {flips:?}: not caught"
                );
            } else {
                assert!(e.is_none(), "case {c}: nothing injected, yet {e:?}");
            }
        },
    );
    assert!(
        injected > cases * 9 / 10,
        "only {injected} of {cases} cases found a pair to flip"
    );
}

/// A seeded xorshift for the random injections.
struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Three or more flips could cancel in principle, so this measures it: random faults in the
/// stored result, each case either 3 to 16 distinct flipped bits anywhere in the group or 1 to 8
/// words XORed with random masks, all caught at r = 0.
#[test]
fn three_or_more_flips_are_caught_in_random_injection() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<16>::builtin(int8, 0, 0x3F11).expect("self-check");
    let fault = |c: usize, group: &mut [u64]| {
        let mut rng = TestRng(xform_hash(c as u64) | 1);
        if c % 4 == 0 {
            for _ in 0..1 + rng.below(8) {
                let w = rng.below(group.len());
                group[w] ^= rng.next() | 1;
            }
        } else {
            let mut bits = HashSet::new();
            while bits.len() < 3 + rng.below(14) {
                bits.insert(rng.below(group.len() * 64));
            }
            for bit in bits {
                flip_bit(group, bit);
            }
        }
    };
    sweep(
        int8,
        &set,
        32 * 1024,
        11,
        |_, _, _| {},
        |group, c, _| fault(c, group),
        |c, _, e| {
            assert!(
                e.is_some_and(GroupError::reproduced),
                "case {c}: not caught"
            )
        },
    );
}

fn xform_hash(x: u64) -> u64 {
    x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA076_1D64_78BD_642F
}

/// Each product goes into the checksum `1 + r` times, so a fault changes it by `1 + r` times its
/// change at r = 0, which can't wrap to zero up to [`MAX_R`]. Single bits and two flips that could
/// cancel, at load settings up to the largest, with `1 + r` a power of two (the tightest case for
/// wrapping) and at the cap.
#[test]
fn detection_holds_at_any_r() {
    let Some(int8) = int8() else { return };
    for r in [1, 7, 255, 32767, MAX_R] {
        let set = PatternSet::<16>::builtin(int8, r, 0xA11).expect("self-check");
        assert_eq!(set.expected_sum(), &scale(set.checksum(), 1 + r), "r={r}");
        let pass = r;
        let stored = |g: usize| -> Vec<&[u64; TILE_U64]> {
            (0..16)
                .map(|t| &set.expected((t + set.offset(g, pass)) % 16).0)
                .collect()
        };
        let flips = |g: usize, c: usize| opposite_flips(&stored(g), c * 2179 % (136 * 128));
        sweep(
            int8,
            &set,
            16,
            pass,
            |_, _, _| {},
            |group, c, g| {
                if c % 2 == 0 {
                    let (t, bit) = (c * 5 % 16, c * 1031 % TILE_BITS);
                    flip_bit(&mut group[t * TILE_U64..][..TILE_U64], bit);
                } else if let Some((t1, bit1, t2, bit2)) = flips(g, c) {
                    flip_bit(&mut group[t1 * TILE_U64..][..TILE_U64], bit1);
                    flip_bit(&mut group[t2 * TILE_U64..][..TILE_U64], bit2);
                }
            },
            |c, g, e| {
                if c % 2 == 0 || flips(g, c).is_some() {
                    assert!(
                        e.is_some_and(GroupError::reproduced),
                        "r={r} case {c}: not caught"
                    );
                }
            },
        );
    }
}

/// After a bad group the R pass goes on: every bad group reported once, clean ones not.
#[test]
fn verify_reports_each_bad_group_and_reads_on() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<32>::new(int8, 2, 0x600D);
    let mut b = buf::<32>(7);
    xform::write(int8, &set, &mut b, 1);
    xform::transform(int8, &set, &mut b);
    let gw = PatternSet::<32>::group_words();
    for g in [0, 3, 6] {
        b[g * gw + 100] ^= 1 << 40;
    }
    let errors = xform::verify(int8, &set, &b, 1, &mut Rescans::default());
    let groups: Vec<usize> = errors.iter().map(|e| e.group).collect();
    assert_eq!(groups, [0, 3, 6]);
    assert!(errors.iter().all(GroupError::reproduced));
}

/// A check that disagreed while every word reads right is reported as not reproduced, as TMR's
/// rescan reports it, and the recheck through the tiles says whether the tile path still
/// disagrees.
#[test]
fn a_check_the_rescan_cant_reproduce_is_reported_not_reproduced() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<16>::new(int8, 2, 0x7E57);
    let mut b = buf::<16>(2);
    assert!(xform::pass(int8, &set, &mut b, 4, &mut Rescans::default()).is_empty());
    let mut rescans = Rescans::default();
    let e = xform::rescan_group(int8, &set, &b, 1, 4, &mut rescans);
    assert_eq!((e.group, e.bad_words), (1, 0));
    assert!(!e.reproduced());
    assert!(e.tiles.is_empty() && e.rereads.is_empty());
    assert!(!e.tile_recheck, "the tile path agrees on a clean group");
    assert_eq!((rescans.alarms, rescans.not_reproduced), (1, 1));
}

/// Tier 2 is capped as TMR's is: a group's report counts every bad word and keeps the first
/// [`RESCAN_LOGGED`], and only a test's first [`REREAD_RESCANS`] rescans reread, so a run full
/// of errors doesn't slow to the rereads' pace.
#[test]
fn tier_2_is_capped_as_tmrs_is() {
    let Some(int8) = int8() else { return };
    let set = PatternSet::<16>::new(int8, 0, 0xCA9);
    let mut b = buf::<16>(20);
    xform::write(int8, &set, &mut b, 0);
    xform::transform(int8, &set, &mut b);
    for w in &mut b {
        *w = !*w;
    }
    let mut rescans = Rescans::default();
    let errors = xform::verify(int8, &set, &b, 0, &mut rescans);
    assert_eq!(errors.len(), 20);
    for e in &errors {
        assert_eq!(e.bad_words, 16 * TILE_U64 as u64, "group {}", e.group);
        assert_eq!(
            e.tiles.len(),
            16,
            "group {}: every bad tile listed",
            e.group
        );
        assert!(e.tiles.iter().all(|t| t.bad_words == TILE_U64));
        let kept: usize = e.tiles.iter().map(|t| t.words.len()).sum();
        assert_eq!(kept, RESCAN_LOGGED, "group {}", e.group);
        let rereads = if e.group < REREAD_RESCANS as usize {
            4
        } else {
            0
        };
        assert_eq!(e.rereads.len(), rereads, "group {}", e.group);
    }
    assert_eq!((rescans.alarms, rescans.not_reproduced), (20, 0));
    // The test's budget is spent: a second R pass in it doesn't reread.
    let again = xform::verify(int8, &set, &b, 0, &mut rescans);
    assert!(again.iter().all(|e| e.rereads.is_empty()));
    assert_eq!(rescans.alarms, 40);
}
