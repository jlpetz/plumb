// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Transform-verify: a memory stability test under AMX load (plumb TODO 6, design in
//! `fearlessD/design/amx-transform-verify.md`, owner's idea 2026-10-21). Not wired into the
//! bench's `--only` groups yet (that's timing, item 3 of the design's prototype plan, which
//! needs the box); this module is items 1 and 2, the kernels and their correctness tests.
//!
//! Three phases, each one pass over the whole buffer in one tile session, as TM5 streams its
//! phases. A group is `X` consecutive 1 KiB tiles and holds each of the set's `X` patterns once.
//! **W** ([`write`]) stores each tile's pattern, in a rotation that changes every pass
//! ([`rotation`]). **RW** ([`transform`]) loads each tile, multiplies it by the set's transform
//! matrix `K` (`TDPBUUD`) and stores the 16x16 u32 result in place, so memory holds
//! `T(P) = P x K`. **R** ([`verify`]) loads each tile once with its pattern's own matrix and
//! adds their product into the group's running checksum `1 + r` times (`r` is the CPU-load
//! setting, owner's decision 2026-10-22), then compares the checksum with the constant every
//! clean group gives. Every multiply's result is in the checksum, so a matrix unit that
//! miscomputes under load shows as a mismatch too. A mismatch runs tier 2 at once
//! ([`GroupError`]): a rescan of the group against `T(P_i)`, which locates every bad word, TMR's
//! rereads, and the checksum again through the tiles.
//!
//! [`PatternSet::new`] has what's guaranteed: any one or two flipped bits in a group's stored
//! result change the checksum, at any `r`. Flips in the pattern before the transform, and three
//! or more, are measured by `tests_xform.rs`, not guaranteed.
//!
//! The expected values are computed on the tiles, by these kernels. The built-in set's group
//! checksum is also stored in the binary (`xform_builtin.rs`, generated offline by the scalar
//! reference in `tests_xform.rs`), and [`self_test`] compares the two before a run relies on
//! the matrix unit.

#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::{_mm_clflush, _mm_mfence};
use std::collections::HashSet;
use std::ptr::read_volatile;

use plumb_tiles::{AmxInt8, ROW_BYTES, T0, T1, T4, T5, T6, T7, TILE_BYTES, Tiles};

/// u64 words per tile (1 KiB).
pub const TILE_U64: usize = TILE_BYTES / 8;

/// The most extra multiplies per tile (`r`): `1 + r` times any one- or two-bit fault's change
/// to the checksum, which is below 2^16, stays below 2^32, so it can't wrap to zero.
pub const MAX_R: usize = 65534;

/// The built-in set's seed (X = 16): its patterns and matrices, whose group checksum
/// `xform_builtin.rs` stores. "TMR_XFM1".
pub const BUILTIN_SEED: u64 = 0x544D_525F_5846_4D31;

include!("xform_builtin.rs");

/// How many bad words a rescan rereads, and how many times each before the flush: TMR's
/// `REREAD_WORDS` and `REREADS` (TMR-APP `src/tests.rs`).
const REREAD_WORDS: usize = 4;
const REREADS: u32 = 4;

/// One tile's worth of data: a pattern, or memory holding `T(P)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tile(pub [u64; TILE_U64]);

/// A VNNI-packed 64x16 byte matrix (one tile): logical `[j][n]` (column `j` < 64, output
/// column `n` < 16) sits at tile row `j / 4`, byte `4n + j % 4` of that row (see
/// plumb_tiles' [Compute](plumb_tiles#compute) doc for the packing TDPBUUD expects).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Matrix(pub [u64; TILE_U64]);

impl Matrix {
    /// Logical entry `[j][n]`, `j` < 64, `n` < 16.
    pub fn get(&self, j: usize, n: usize) -> u8 {
        let byte = (j / 4) * ROW_BYTES + 4 * n + j % 4;
        (self.0[byte / 8] >> (byte % 8 * 8)) as u8
    }

    /// Logical row `j`, `[j][0..16]`: what byte `j` of a data row is multiplied by, one entry
    /// per output column.
    pub fn row(&self, j: usize) -> [u8; 16] {
        std::array::from_fn(|n| self.get(j, n))
    }
}

/// A tiny xorshift64 stream, seeded, used only to generate deterministic test data (patterns
/// and matrices): not a cryptographic generator, and not part of what's being tested.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// `count` matrices with every byte odd and every logical row ([`Matrix::row`]) distinct,
/// within each matrix and across them: the two properties [`PatternSet::new`]'s guarantees
/// rest on. Rows are 16 odd bytes (a 2^112 space), so a repeat is astronomically unlikely; a
/// matrix with one is drawn again.
fn distinct_odd_matrices(count: usize, rng: &mut Rng) -> Vec<Matrix> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let m = Matrix(std::array::from_fn(|_| {
            rng.next_u64() | 0x0101_0101_0101_0101
        }));
        let rows: HashSet<[u8; 16]> = (0..64).map(|j| m.row(j)).collect();
        if rows.len() == 64 && rows.is_disjoint(&seen) {
            seen.extend(rows);
            out.push(m);
        }
    }
    out
}

/// The set's `X` pattern tiles: the classic memory-test patterns (solid zeros and ones, the two
/// checkerboards, walking ones and walking zeros along each word), then seeded pseudo-random
/// tiles.
fn gen_patterns<const X: usize>(rng: &mut Rng) -> Box<[Tile; X]> {
    let walk = |w: usize| 1u64 << (w % 64);
    let classic = [
        Tile([0; TILE_U64]),
        Tile([!0; TILE_U64]),
        Tile([0xAAAA_AAAA_AAAA_AAAA; TILE_U64]),
        Tile([0x5555_5555_5555_5555; TILE_U64]),
        Tile(std::array::from_fn(walk)),
        Tile(std::array::from_fn(|w| !walk(w))),
    ];
    let random = std::iter::repeat_with(|| Tile(std::array::from_fn(|_| rng.next_u64())));
    let all: Vec<Tile> = classic.into_iter().chain(random).take(X).collect();
    all.try_into().expect("X tiles")
}

/// splitmix64's finalizer: a cheap, well-mixed hash, used to key the rotation.
fn hash64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

/// The rotation offset of group `g` on pass `pass`: tile `t` of the group holds pattern
/// `(t + offset) % X`.
///
/// Over each cycle of `X` passes the offsets are a permutation of `0..X`, so every tile holds
/// every pattern once a cycle. The permutation is an affine step, an xorshift and another
/// affine step, each a bijection mod `X` (odd multipliers), keyed by a hash of the layout seed,
/// the group and the cycle. Its `X^4 / 4` keys give as many distinct permutations (counted at
/// X = 16 and 32), so two groups hold the same data on a given pass 1 time in `X`, and on every
/// pass of a cycle (hiding an addressing fault between them for that cycle) 1 time in
/// `X^4 / 4`, a different pair each cycle.
#[inline(always)]
fn rotation<const X: usize>(layout: u64, g: usize, pass: usize) -> usize {
    let m = X - 1;
    let k = hash64(hash64(layout ^ g as u64).wrapping_add((pass / X) as u64)) as usize;
    let v = (pass & m).wrapping_mul(k | 1).wrapping_add(k >> 8) & m;
    let v = v ^ (v >> (X.trailing_zeros() / 2));
    v.wrapping_mul((k >> 16) | 1).wrapping_add(k >> 24) & m
}

/// A pattern set: `X` pattern tiles (16, 32 or 64), the transform matrix, a checksum matrix per
/// pattern, the CPU-load setting `r`, and every expected value the test compares against.
pub struct PatternSet<const X: usize> {
    r: usize,
    /// Keys the rotation: the run's seed.
    layout: u64,
    patterns: Box<[Tile; X]>,
    transform: Matrix,
    /// The checksum matrix of each pattern.
    keys: Box<[Matrix; X]>,
    /// `T(P_i)` for each pattern.
    expected_t: Box<[Tile; X]>,
    /// The group checksum at `r` = 0, `E = sum_i T(P_i) x K_i`. It is the same for every group
    /// and pass: a group holds every pattern once and multiplies each by its own matrix,
    /// whatever the rotation.
    checksum: Tile,
    /// The group checksum the R pass compares with, at this set's `r`: `(1 + r) E`.
    expected_sum: Tile,
}

impl<const X: usize> PatternSet<X> {
    /// A set from `seed` (patterns, matrices and the rotation's layout), with `r` extra
    /// multiplies per tile in the R pass (the CPU-load setting), its expected values computed
    /// on the tiles.
    ///
    /// Every matrix byte is odd, and the logical matrix rows ([`Matrix::row`]) are all distinct
    /// across the set's `X` matrices. That gives two guarantees:
    ///
    /// - **Any single flipped bit in a group's stored result** changes byte `j` of data row `m`
    ///   by `d = +-2^b`, and so the checksum's row `m` by `d * K[j][n]` in each of its 16
    ///   entries. With `K` odd, `|d * K| < 2^15` is never 0 mod 2^32.
    /// - **Any two.** Changes in different rows can't cancel. In one row, `d1 * K1[j1][n]` and
    ///   `d2 * K2[j2][n]` cancel for all 16 `n` only if they're equal and opposite: since both
    ///   matrices are odd, `|d1| = |d2|` (the same bit, flipped in opposite directions) and the
    ///   rows `K1[j1]` and `K2[j2]` are equal, which the construction rules out.
    ///
    /// Each product goes into the checksum `1 + r` times, so a fault changes it by `1 + r` times
    /// its change at `r` = 0. That is below 2^16 in both cases above, and `1 + r` is at most
    /// 65535 ([`MAX_R`]), so the product stays below 2^32 and nonzero.
    ///
    /// The transform's 64 rows are distinct too, so a bit flipped before the transform leaves a
    /// row delta that names its byte (tier 2's [`Signature::BeforeTransform`]).
    ///
    /// # Panics
    ///
    /// If `r` is over [`MAX_R`].
    pub fn new(int8: AmxInt8, r: usize, seed: u64) -> Self {
        Self::build(int8, r, seed, seed)
    }

    fn build(int8: AmxInt8, r: usize, seed: u64, layout: u64) -> Self {
        const { assert!(X == 16 || X == 32 || X == 64, "X must be 16, 32 or 64") };
        assert!(r <= MAX_R, "r must be at most {MAX_R}");
        let mut rng = Rng::new(seed);
        let patterns = gen_patterns::<X>(&mut rng);
        let transform = distinct_odd_matrices(1, &mut rng)[0];
        let keys: Box<[Matrix; X]> = distinct_odd_matrices(X, &mut rng)
            .try_into()
            .expect("X matrices");

        // T(P_i) through the RW kernel, then the checksums of a group holding them in order
        // (rotation 0) through the R kernel's loop, at r = 0 and at this set's r.
        let mut t: Vec<u64> = patterns.iter().flat_map(|p| p.0).collect();
        // SAFETY: `int8` proves AMX-TILE and AMX-INT8, the features `transform_tiles` and
        // `group_checksum` enable.
        let (checksum, expected_sum) = unsafe {
            transform_tiles(int8, &transform, &mut t);
            (
                group_checksum::<X>(int8, &keys, &t, 0),
                group_checksum::<X>(int8, &keys, &t, r),
            )
        };
        let expected_t = t
            .as_chunks::<TILE_U64>()
            .0
            .iter()
            .map(|w| Tile(*w))
            .collect::<Vec<_>>()
            .try_into()
            .expect("X tiles");
        PatternSet {
            r,
            layout,
            patterns,
            transform,
            keys,
            expected_t,
            checksum,
            expected_sum,
        }
    }

    pub fn r(&self) -> usize {
        self.r
    }

    /// Words per group (`X` whole tiles).
    pub const fn group_words() -> usize {
        X * TILE_U64
    }
}

impl PatternSet<16> {
    /// The built-in set (X = 16, [`BUILTIN_SEED`]) with `r` extra multiplies and the run's
    /// `layout` seed for the rotation. `None` if its group checksum, computed on the tiles,
    /// differs from the one stored in the binary: the matrix unit computes wrongly, before any
    /// memory is tested.
    ///
    /// Comparing the checksum covers every `T(P_i)` as well: it sums them all, each times its
    /// own matrix, so a wrong `T(P_i)` shows there with the guarantees of [`PatternSet::new`].
    pub fn builtin(int8: AmxInt8, r: usize, layout: u64) -> Option<Self> {
        let set = Self::build(int8, r, BUILTIN_SEED, layout);
        (set.checksum.0 == BUILTIN_CHECKSUM).then_some(set)
    }
}

/// The startup check: the built-in set, recomputed on the tiles and compared with its stored
/// checksum. False means the matrix unit is wrong before the test starts.
pub fn self_test(int8: AmxInt8) -> bool {
    PatternSet::<16>::builtin(int8, 0, 0).is_some()
}

/// One differing word of a tile, by its index in the tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BadWord {
    pub word: usize,
    pub expected: u64,
    pub actual: u64,
}

/// What a tile's differences look like.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Signature {
    /// One bit of the stored result: byte `byte` of the tile (0..1024), bit `bit`.
    Stored { byte: usize, bit: u32 },
    /// One bit of the pattern, flipped before RW read it: byte `byte` of the pattern tile, bit
    /// `bit`, and whether it went from 0 to 1. One row of the result is off by `d * K[j][..]`,
    /// which names the byte (the transform's rows are distinct).
    BeforeTransform { byte: usize, bit: u32, rose: bool },
    /// Anything else: more than one bit, or neither shape.
    Other,
}

/// A bad tile the rescan found.
#[derive(Debug)]
pub struct TileError {
    /// Its position in the group.
    pub tile: usize,
    /// The pattern it should hold.
    pub pattern: usize,
    /// Every differing word, by volatile loads.
    pub words: Vec<BadWord>,
    pub signature: Signature,
}

/// TMR's rereads of one bad word: [`REREADS`] volatile loads, one after flushing its line, and
/// one through a tile load after flushing it again.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Reread {
    pub tile: usize,
    pub word: usize,
    pub expected: u64,
    /// How many of the [`REREADS`] rereads were still wrong.
    pub still_wrong: u32,
    pub after_flush: u64,
    pub via_tiles: u64,
}

/// Tier 2's report on a group whose checksum disagreed. As TMR's, none of the rereads locates
/// the fault: a reread can hit a cached copy of a bad read, and a flush writes back a dirty line
/// first. They're reported as seen.
#[derive(Debug)]
pub struct GroupError {
    pub group: usize,
    /// The bad tiles. Empty: the rescan found every word right, the check is not reproduced.
    pub tiles: Vec<TileError>,
    /// The first [`REREAD_WORDS`] bad words, reread.
    pub rereads: Vec<Reread>,
    /// Whether the checksum still disagrees when it's recomputed through the tiles after the
    /// rescan. With nothing reproduced, true means the tile path sees what word loads don't: a
    /// fault in the tile loads or the multiply.
    pub tile_recheck: bool,
}

impl GroupError {
    pub fn reproduced(&self) -> bool {
        !self.tiles.is_empty()
    }
}

/// The shape of `words` (one tile's differences, in word order) against the transform `k`.
fn signature(words: &[BadWord], k: &Matrix) -> Signature {
    if let [w] = words {
        let x = w.expected ^ w.actual;
        if x.count_ones() == 1 {
            let b = x.trailing_zeros() as usize;
            return Signature::Stored {
                byte: w.word * 8 + b / 8,
                bit: (b % 8) as u32,
            };
        }
    }
    // A flip in byte `j` of pattern row `m` moves all 16 entries of result row `m` (8 words) by
    // `d * k[j][n]`, `d = +-2^b`; every entry of `k` is odd, so all 16 move, by `2^b` times an
    // odd byte.
    let row = words[0].word / 8;
    if words.len() != 8 || words.iter().any(|w| w.word / 8 != row) {
        return Signature::Other;
    }
    let delta: [i32; 16] = std::array::from_fn(|n| {
        let (w, shift) = (&words[n / 2], n % 2 * 32);
        ((w.actual >> shift) as u32).wrapping_sub((w.expected >> shift) as u32) as i32
    });
    let bit = delta[0].trailing_zeros();
    if bit > 7 {
        return Signature::Other;
    }
    let d = if delta[0] < 0 { -(1 << bit) } else { 1 << bit };
    let col = delta.map(|x| x / d);
    let odd_bytes = col.iter().all(|&c| (1..=255).contains(&c) && c % 2 == 1);
    if !odd_bytes || delta.iter().zip(&col).any(|(&x, &c)| x != c * d) {
        return Signature::Other;
    }
    match (0..64).find(|&j| (0..16).all(|n| i32::from(k.get(j, n)) == col[n])) {
        Some(j) => Signature::BeforeTransform {
            byte: row * ROW_BYTES + j,
            bit,
            rose: d > 0,
        },
        None => Signature::Other,
    }
}

/// Panics unless `buf` is a whole number of `X`-tile groups.
fn check_groups<const X: usize>(buf: &[u64]) {
    assert!(
        buf.len().is_multiple_of(PatternSet::<X>::group_words()),
        "buf must be a whole number of groups"
    );
}

/// **W**: stores each tile's pattern, rotated for its group and `pass` (TM5's write phase).
pub fn write<const X: usize>(int8: AmxInt8, set: &PatternSet<X>, buf: &mut [u64], pass: usize) {
    check_groups::<X>(buf);
    // SAFETY: `int8` proves AMX-TILE and AMX-INT8, the features `write_tiles` enables.
    unsafe { write_tiles(int8, set, buf, pass) }
}

/// **RW**: transforms every tile in place, `T(P) = P x K`. Any whole number of tiles.
pub fn transform<const X: usize>(int8: AmxInt8, set: &PatternSet<X>, buf: &mut [u64]) {
    assert!(
        buf.len().is_multiple_of(TILE_U64),
        "buf must be whole tiles"
    );
    // SAFETY: `int8` proves AMX-TILE and AMX-INT8, the features `transform_tiles` enables.
    unsafe { transform_tiles(int8, &set.transform, buf) }
}

/// **R** and tier 2: checks every group's checksum against what [`write`] and [`transform`] with
/// the same `pass` leave there, and rescans each group that disagrees as soon as it's found,
/// before the pass reads on. Returns one [`GroupError`] per such group, in buffer order; empty
/// when the buffer is clean.
pub fn verify<const X: usize>(
    int8: AmxInt8,
    set: &PatternSet<X>,
    buf: &[u64],
    pass: usize,
) -> Vec<GroupError> {
    check_groups::<X>(buf);
    let mut errors = Vec::new();
    let mut sums = [[0; TILE_U64]; 2];
    let groups = buf.as_chunks::<TILE_U64>().0.as_chunks::<X>().0;
    let mut from = 0;
    // SAFETY: `int8` proves AMX-TILE and AMX-INT8, the features `verify_tiles` and `rescan`
    // enable.
    while let Some(g) = unsafe { verify_tiles::<X>(int8, set, buf, pass, from, &mut sums) } {
        let off = rotation::<X>(set.layout, g, pass);
        errors.push(unsafe { rescan::<X>(int8, set, &groups[g], g, off, &mut sums) });
        from = g + 1;
    }
    errors
}

/// [`write`], [`transform`] and [`verify`], one after the other.
pub fn pass<const X: usize>(
    int8: AmxInt8,
    set: &PatternSet<X>,
    buf: &mut [u64],
    pass: usize,
) -> Vec<GroupError> {
    write(int8, set, buf, pass);
    transform(int8, set, buf);
    verify(int8, set, buf, pass)
}

// The kernels. Each runs its phase over the buffer in one session, from a function compiled
// with the AMX features (plumb_tiles' calling practice 2), and gives its tile loop its own
// function (TMR's Rule 6.1).

#[inline(never)]
#[target_feature(enable = "amx-tile,amx-int8")]
unsafe fn write_tiles<const X: usize>(
    int8: AmxInt8,
    set: &PatternSet<X>,
    buf: &mut [u64],
    pass: usize,
) {
    int8.with_tiles(|t| {
        let (tiles, _) = buf.as_chunks_mut::<TILE_U64>();
        for (g, group) in tiles.as_chunks_mut::<X>().0.iter_mut().enumerate() {
            let off = rotation::<X>(set.layout, g, pass);
            let (pairs, _) = group.as_chunks_mut::<2>();
            for (i, pair) in pairs.iter_mut().enumerate() {
                t.load_u64::<T0>(&set.patterns[(2 * i + off) & (X - 1)].0, ROW_BYTES);
                t.load_u64::<T1>(&set.patterns[(2 * i + 1 + off) & (X - 1)].0, ROW_BYTES);
                t.store_u64::<T0>(&mut pair[0], ROW_BYTES);
                t.store_u64::<T1>(&mut pair[1], ROW_BYTES);
            }
        }
    })
}

/// The transform matrix stays in T7 for the session; tiles go two at a time (T4, T5 into T0,
/// T1).
#[inline(never)]
#[target_feature(enable = "amx-tile,amx-int8")]
unsafe fn transform_tiles(int8: AmxInt8, k: &Matrix, buf: &mut [u64]) {
    int8.with_tiles(|t| {
        t.load_u64::<T7>(&k.0, ROW_BYTES);
        let (tiles, _) = buf.as_chunks_mut::<TILE_U64>();
        let (pairs, odd) = tiles.as_chunks_mut::<2>();
        for pair in pairs {
            t.load_u64::<T4>(&pair[0], ROW_BYTES);
            t.load_u64::<T5>(&pair[1], ROW_BYTES);
            t.zero::<T0>();
            t.zero::<T1>();
            t.dpbuud::<T0, T4, T7>(int8);
            t.dpbuud::<T1, T5, T7>(int8);
            t.store_u64::<T0>(&mut pair[0], ROW_BYTES);
            t.store_u64::<T1>(&mut pair[1], ROW_BYTES);
        }
        for tile in odd {
            t.load_u64::<T4>(tile, ROW_BYTES);
            t.zero::<T0>();
            t.dpbuud::<T0, T4, T7>(int8);
            t.store_u64::<T0>(tile, ROW_BYTES);
        }
    })
}

/// The R pass's kernel from group `from`: each group's checksum, compared with the expected
/// one. Stops at the first group that disagrees and returns its index. `sums` is scratch.
#[inline(never)]
#[target_feature(enable = "amx-tile,amx-int8")]
unsafe fn verify_tiles<const X: usize>(
    int8: AmxInt8,
    set: &PatternSet<X>,
    buf: &[u64],
    pass: usize,
    from: usize,
    sums: &mut [[u64; TILE_U64]; 2],
) -> Option<usize> {
    int8.with_tiles(|t| {
        let (tiles, _) = buf.as_chunks::<TILE_U64>();
        for (i, group) in tiles.as_chunks::<X>().0[from..].iter().enumerate() {
            let g = from + i;
            let off = rotation::<X>(set.layout, g, pass);
            group_sums::<X>(t, int8, &set.keys, group, off, set.r, sums);
            if !sums_match(sums, &set.expected_sum) {
                return Some(g);
            }
        }
        None
    })
}

/// The R pass's loop: `group`'s checksum, as two chains in `sums` that add up to it. Tiles go in
/// pairs, each tile with its own registers (data T4 and matrix T5 into T0; T6 and T7 into T1),
/// so the two chains of multiplies don't wait on each other; each tile's product goes into its
/// chain `1 + r` times.
#[inline(always)]
fn group_sums<const X: usize>(
    t: &mut Tiles<'_>,
    int8: AmxInt8,
    keys: &[Matrix; X],
    group: &[[u64; TILE_U64]; X],
    off: usize,
    r: usize,
    sums: &mut [[u64; TILE_U64]; 2],
) {
    t.zero::<T0>();
    t.zero::<T1>();
    for (i, pair) in group.as_chunks::<2>().0.iter().enumerate() {
        t.load_u64::<T4>(&pair[0], ROW_BYTES);
        t.load_u64::<T5>(&keys[(2 * i + off) & (X - 1)].0, ROW_BYTES);
        t.load_u64::<T6>(&pair[1], ROW_BYTES);
        t.load_u64::<T7>(&keys[(2 * i + 1 + off) & (X - 1)].0, ROW_BYTES);
        t.dpbuud::<T0, T4, T5>(int8);
        t.dpbuud::<T1, T6, T7>(int8);
        for _ in 0..r {
            t.dpbuud::<T0, T4, T5>(int8);
            t.dpbuud::<T1, T6, T7>(int8);
        }
    }
    t.store_u64::<T0>(&mut sums[0], ROW_BYTES);
    t.store_u64::<T1>(&mut sums[1], ROW_BYTES);
}

/// Two words of u32 lanes added lane by lane, each lane wrapping on its own (as the multiply's
/// accumulator does).
#[inline(always)]
fn add_lanes(a: u64, b: u64) -> u64 {
    let lo = (a as u32).wrapping_add(b as u32);
    let hi = ((a >> 32) as u32).wrapping_add((b >> 32) as u32);
    u64::from(lo) | (u64::from(hi) << 32)
}

/// Whether the two chains add up to `expected`.
#[inline(always)]
fn sums_match(sums: &[[u64; TILE_U64]; 2], expected: &Tile) -> bool {
    let diff = sums[0]
        .iter()
        .zip(&sums[1])
        .zip(&expected.0)
        .fold(0, |acc, ((a, b), e)| acc | (add_lanes(*a, *b) ^ e));
    diff == 0
}

/// Tier 2 for group `g`: the rescan against `T(P_i)` with volatile loads, then in one session
/// the rereads and the checksum again through the tiles. `sums` is scratch.
#[cold]
#[inline(never)]
#[target_feature(enable = "amx-tile,amx-int8")]
unsafe fn rescan<const X: usize>(
    int8: AmxInt8,
    set: &PatternSet<X>,
    group: &[[u64; TILE_U64]; X],
    g: usize,
    off: usize,
    sums: &mut [[u64; TILE_U64]; 2],
) -> GroupError {
    let mut tiles = Vec::new();
    for (i, tile) in group.iter().enumerate() {
        let pattern = (i + off) & (X - 1);
        let words: Vec<BadWord> = set.expected_t[pattern]
            .0
            .iter()
            .zip(tile)
            .enumerate()
            .filter_map(|(word, (&expected, actual))| {
                let actual = read_volatile(actual);
                (actual != expected).then_some(BadWord {
                    word,
                    expected,
                    actual,
                })
            })
            .collect();
        if !words.is_empty() {
            tiles.push(TileError {
                tile: i,
                pattern,
                signature: signature(&words, &set.transform),
                words,
            });
        }
    }
    let (rereads, tile_recheck) = int8.with_tiles(|t| {
        let rereads = reread(t, group, &tiles, &mut sums[0]);
        group_sums::<X>(t, int8, &set.keys, group, off, set.r, sums);
        (rereads, !sums_match(sums, &set.expected_sum))
    });
    GroupError {
        group: g,
        tiles,
        rereads,
        tile_recheck,
    }
}

/// TMR's rereads (`Rescan::reread`) of the first [`REREAD_WORDS`] bad words, plus one through
/// the tiles: every word's volatile rereads before any flush, since a flush evicts the whole
/// line; then each reread after flushing the lines; then each line flushed again and its tile
/// loaded (stored to `scratch` to read the word).
unsafe fn reread(
    t: &mut Tiles<'_>,
    group: &[[u64; TILE_U64]],
    tiles: &[TileError],
    scratch: &mut [u64; TILE_U64],
) -> Vec<Reread> {
    let suspects: Vec<(usize, BadWord)> = tiles
        .iter()
        .flat_map(|e| e.words.iter().map(|&w| (e.tile, w)))
        .take(REREAD_WORDS)
        .collect();
    let at = |tile: usize, w: &BadWord| &raw const group[tile][w.word];
    let still: Vec<u32> = suspects
        .iter()
        .map(|(tile, w)| {
            (0..REREADS)
                .map(|_| u32::from(read_volatile(at(*tile, w)) != w.expected))
                .sum()
        })
        .collect();
    for (tile, w) in &suspects {
        _mm_clflush(at(*tile, w).cast());
    }
    _mm_mfence();
    let after: Vec<u64> = suspects
        .iter()
        .map(|(tile, w)| read_volatile(at(*tile, w)))
        .collect();
    let mut out = Vec::with_capacity(suspects.len());
    for (((tile, w), still_wrong), after_flush) in suspects.iter().zip(still).zip(after) {
        _mm_clflush(at(*tile, w).cast());
        _mm_mfence();
        t.load_u64::<T6>(&group[*tile], ROW_BYTES);
        t.store_u64::<T6>(scratch, ROW_BYTES);
        out.push(Reread {
            tile: *tile,
            word: w.word,
            expected: w.expected,
            still_wrong,
            after_flush,
            via_tiles: scratch[w.word],
        });
    }
    out
}

/// The checksum, at `r`, of a group holding `T(P_i)` at tile `i` (rotation 0).
#[target_feature(enable = "amx-tile,amx-int8")]
unsafe fn group_checksum<const X: usize>(
    int8: AmxInt8,
    keys: &[Matrix; X],
    group: &[u64],
    r: usize,
) -> Tile {
    let mut sums = [[0; TILE_U64]; 2];
    let group = group
        .as_chunks::<TILE_U64>()
        .0
        .try_into()
        .expect("one group");
    int8.with_tiles(|t| group_sums::<X>(t, int8, keys, group, 0, r, &mut sums));
    Tile(std::array::from_fn(|w| add_lanes(sums[0][w], sums[1][w])))
}

/// Test-only: tier 2 on group `g` of `buf`, as if its checksum had disagreed.
#[cfg(test)]
pub fn rescan_group<const X: usize>(
    int8: AmxInt8,
    set: &PatternSet<X>,
    buf: &[u64],
    g: usize,
    pass: usize,
) -> GroupError {
    check_groups::<X>(buf);
    let mut sums = [[0; TILE_U64]; 2];
    let group = &buf.as_chunks::<TILE_U64>().0.as_chunks::<X>().0[g];
    let off = rotation::<X>(set.layout, g, pass);
    // SAFETY: `int8` proves AMX-TILE and AMX-INT8, the features `rescan` enables.
    unsafe { rescan::<X>(int8, set, group, g, off, &mut sums) }
}

/// The set's internals, for `tests_xform.rs`'s independent computation.
#[cfg(test)]
impl<const X: usize> PatternSet<X> {
    pub fn pattern(&self, i: usize) -> &Tile {
        &self.patterns[i]
    }

    pub fn transform_matrix(&self) -> &Matrix {
        &self.transform
    }

    /// Pattern `i`'s checksum matrix.
    pub fn key(&self, i: usize) -> &Matrix {
        &self.keys[i]
    }

    /// `T(pattern i)`.
    pub fn expected(&self, i: usize) -> &Tile {
        &self.expected_t[i]
    }

    /// The group checksum at `r` = 0.
    pub fn checksum(&self) -> &Tile {
        &self.checksum
    }

    /// The group checksum at this set's `r`.
    pub fn expected_sum(&self) -> &Tile {
        &self.expected_sum
    }

    /// The rotation offset of group `g` on pass `pass`: tile `t` holds pattern
    /// `(t + offset) % X`.
    pub fn offset(&self, g: usize, pass: usize) -> usize {
        rotation::<X>(self.layout, g, pass)
    }
}

// The asm gate's entry points (`asm_check.py`, `expect/xform.json`): W and R at both group
// sizes, and RW. R's is the kernel alone: tier 2 is the cold path a mismatch takes, outside it.

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_xform_write_16(int8: AmxInt8, set: &PatternSet<16>, buf: &mut [u64], pass: usize) {
    write(int8, set, buf, pass)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_xform_write_64(int8: AmxInt8, set: &PatternSet<64>, buf: &mut [u64], pass: usize) {
    write(int8, set, buf, pass)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_xform_transform(int8: AmxInt8, set: &PatternSet<16>, buf: &mut [u64]) {
    transform(int8, set, buf)
}

macro_rules! verify_entry {
    ($($name:ident, $x:literal;)*) => {$(
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name(
            int8: AmxInt8,
            set: &PatternSet<$x>,
            buf: &[u64],
            pass: usize,
            sums: &mut [[u64; TILE_U64]; 2],
        ) -> Option<usize> {
            check_groups::<$x>(buf);
            // SAFETY: `int8` proves AMX-TILE and AMX-INT8, the features `verify_tiles` enables.
            unsafe { verify_tiles::<$x>(int8, set, buf, pass, 0, sums) }
        }
    )*};
}

verify_entry!(
    k_xform_verify_16, 16;
    k_xform_verify_64, 64;
);
