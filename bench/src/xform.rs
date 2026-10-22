// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Transform-verify: a memory stability test under AMX load (plumb TODO 6, design in
//! `fearlessD/design/amx-transform-verify.md`, owner's idea 2026-10-21). Not wired into the
//! bench's `--only` groups yet (that's timing, item 3 of the design's prototype plan, which
//! needs the box); this module is items 1 and 2, the kernels and their correctness tests.
//!
//! Per chunk ("group" of `x` tiles): **W** fills each 1 KiB tile with a pattern from a set of
//! `x` (a rotation decides which tile gets which pattern); **RW** loads each tile, multiplies
//! it by a fixed transform matrix `K` (`TDPBUUD`), and stores the 16x16 u32 result back in
//! place, so memory now holds `T(P) = P x K`; **R** loads each tile again and multiplies it
//! into a group checksum with that pattern's own key matrix, `1 + r` times with `1 + r`
//! independent keys (`r` is the CPU-load setting), comparing each checksum with the value
//! expected for a clean group. A mismatch runs tier 2: an exact compare of every tile in the
//! group against its expected value, which locates the bad tile and, in the common case of one
//! flipped bit, the byte and bit.
//!
//! Every matrix byte is odd, by construction, which alone guarantees a single flipped bit in a
//! tile always changes a checksum it's part of (see [`PatternSet::new`]'s doc). Two flipped
//! bits could in principle cancel only if two matrix rows coincided; rows are generated
//! distinct across the whole set to rule that out. Three or more is not ruled out formally;
//! `tests_xform.rs`'s fault injection measures it rather than claims it (the design doc's own
//! framing).
//!
//! **Independence of the expected values.** `PatternSet::new` computes every expected value
//! two ways: on the AMX multiply unit (the operational path) and with a plain scalar
//! implementation ([`scalar_multiply`], never touching AMX). They're asserted equal, which
//! checks the multiply unit computes correctly before any test trusts it. A shipped build could
//! also embed one fixed pattern set's values as literal constants generated offline (skipping
//! even this scalar recompute at startup for that one set); this prototype always recomputes
//! and cross-checks, which is strictly more evidence and works for every seed, not just one.

#![allow(unsafe_op_in_unsafe_fn)]

use plumb_tiles::{AmxInt8, ROW_BYTES, T0, T4, T5, TILE_BYTES};

/// u64 words per tile (1 KiB).
pub const TILE_U64: usize = TILE_BYTES / 8;

/// One tile's worth of data: a pattern, or memory holding `T(P)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tile(pub [u64; TILE_U64]);

/// A VNNI-packed 64x16 byte matrix (one tile): logical `[j][n]` (column `j` < 64, output
/// column `n` < 16) sits at tile row `j / 4`, byte `4n + j % 4` of that row (see
/// plumb_tiles' [Compute](plumb_tiles#compute) doc for the packing TDPBUUD expects).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Matrix(pub [u64; TILE_U64]);

impl Matrix {
    fn bytes(&self) -> [u8; TILE_BYTES] {
        words_to_bytes(&self.0)
    }

    /// Logical entry `[j][n]`, `j` < 64, `n` < 16.
    fn get(&self, j: usize, n: usize) -> u8 {
        self.bytes()[(j / 4) * ROW_BYTES + 4 * n + j % 4]
    }

    /// Physical tile row `r` (64 bytes): `r = j / 4` of the logical layout above.
    fn row(&self, r: usize) -> [u64; 8] {
        self.0[r * 8..r * 8 + 8].try_into().expect("8 words")
    }
}

fn words_to_bytes(words: &[u64; TILE_U64]) -> [u8; TILE_BYTES] {
    let mut b = [0u8; TILE_BYTES];
    for (w, chunk) in words.iter().zip(b.as_chunks_mut::<8>().0) {
        *chunk = w.to_le_bytes();
    }
    b
}

fn bytes_to_words(bytes: &[u8; TILE_BYTES]) -> [u64; TILE_U64] {
    std::array::from_fn(|i| {
        u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().expect("8 bytes"))
    })
}

/// A tiny xorshift64 stream, seeded, used only to generate deterministic test data (patterns
/// and matrices) — not a cryptographic generator, and not part of what's being tested.
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
    fn next_byte(&mut self) -> u8 {
        self.next_u64() as u8
    }
}

/// A matrix with every byte odd (forced with `| 1`): the property that makes a single flipped
/// bit in an operand always change the product (see [`PatternSet::new`]).
fn odd_matrix(rng: &mut Rng) -> Matrix {
    let bytes: [u8; TILE_BYTES] = std::array::from_fn(|_| rng.next_byte() | 1);
    Matrix(bytes_to_words(&bytes))
}

/// `count` matrices, every byte odd, with every physical row (64 bytes) distinct across all of
/// them: the property that rules out two flipped bits cancelling in a checksum (see
/// [`PatternSet::new`]). Collisions are astronomically unlikely (64 odd bytes: a 2^504 space)
/// and are retried if they somehow occur, so this always terminates.
fn distinct_odd_matrices(count: usize, rng: &mut Rng) -> Vec<Matrix> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut m = odd_matrix(rng);
        while (0..16).any(|r| !seen.insert(m.row(r))) {
            for r in 0..16 {
                seen.remove(&m.row(r)); // undo partial inserts from the rejected draw
            }
            m = odd_matrix(rng);
        }
        for r in 0..16 {
            seen.insert(m.row(r));
        }
        out.push(m);
    }
    out
}

/// `x` pattern tiles: four classic patterns (all-zero, all-one, and the two checkerboards),
/// then `x - 4` seeded pseudo-random tiles, up to `x`. (`x` <= 4 just takes the first `x`
/// classics.)
fn gen_patterns(x: usize, rng: &mut Rng) -> Vec<Tile> {
    let classic: [u64; 4] = [
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
    ];
    let mut out = Vec::with_capacity(x);
    for &w in classic.iter().take(x) {
        out.push(Tile([w; TILE_U64]));
    }
    while out.len() < x {
        out.push(Tile(std::array::from_fn(|_| rng.next_u64())));
    }
    out
}

/// `A x B` (INT8 x INT8 -> INT32, wrapping), as TDPBUUD defines it: `A` the 16x64-byte operand
/// (`tile`, row-major), `B` the VNNI-packed matrix. A plain scalar loop, independent of AMX;
/// used only to cross-check the AMX-computed expected values ([`PatternSet::new`]) and in
/// `tests_xform.rs`'s fault injection, never in the hot path.
fn scalar_multiply(tile: &[u64; TILE_U64], b: &Matrix) -> [[u32; 16]; 16] {
    let a = words_to_bytes(tile);
    std::array::from_fn(|m| {
        std::array::from_fn(|n| {
            let mut sum = 0u32;
            for j in 0..64 {
                sum = sum.wrapping_add((a[m * 64 + j] as u32).wrapping_mul(b.get(j, n) as u32));
            }
            sum
        })
    })
}

/// The 16x16 u32 grid packed the way `TILESTORED` lays an accumulator out in memory: row `m`,
/// dwords `n` and `n+1`, little-endian, at word `m*8 + n/2`.
fn grid_to_tile(c: &[[u32; 16]; 16]) -> Tile {
    let mut words = [0u64; TILE_U64];
    for (m, row) in c.iter().enumerate() {
        for half in 0..8 {
            words[m * 8 + half] = (row[half * 2] as u64) | ((row[half * 2 + 1] as u64) << 32);
        }
    }
    Tile(words)
}

/// The inverse of [`grid_to_tile`].
fn tile_to_grid(t: &Tile) -> [[u32; 16]; 16] {
    std::array::from_fn(|m| {
        std::array::from_fn(|n| {
            let w = t.0[m * 8 + n / 2];
            if n % 2 == 0 {
                w as u32
            } else {
                (w >> 32) as u32
            }
        })
    })
}

/// `A x B` via AMX (`TDPBUUD`, accumulator zeroed first). Setup-time use only (building
/// [`PatternSet`]'s expected values): not performance-sensitive, so this captures its by-value
/// arguments without extra ceremony.
fn amx_multiply(int8: AmxInt8, a: &[u64; TILE_U64], b: &Matrix) -> Tile {
    let (a, b) = (*a, b.0);
    let mut out = [0u64; TILE_U64];
    // `&mut out_ref` (a reference, never `Copy`) is what the `move` closure takes: a `move`
    // closure over a `Copy` local like `&mut out` directly would capture ITS OWN COPY of
    // `out`, write into that, and leave the outer `out` untouched (`[u64; N]` arrays are
    // `Copy`, unlike `load.rs`'s pattern, which always moves an already-`&mut`-typed
    // parameter, never a local array).
    let out_ref = &mut out;
    int8.with_tiles(move |t| {
        t.load_u64::<T4>(&a, ROW_BYTES);
        t.load_u64::<T5>(&b, ROW_BYTES);
        t.zero::<T0>();
        t.dpbuud::<T0, T4, T5>(int8);
        t.store_u64::<T0>(out_ref, ROW_BYTES);
    });
    Tile(out)
}

/// `sum over i of a[i] x b[i]` via AMX, one accumulator across the whole list (`TDPBUUD`
/// without re-zeroing between calls, matching the hardware's own INT32-wrapping `+=`). Setup
/// time only, as [`amx_multiply`].
fn amx_accumulate(int8: AmxInt8, pairs: &[(&[u64; TILE_U64], &Matrix)]) -> Tile {
    let mut out = [0u64; TILE_U64];
    int8.with_tiles(|t| {
        t.zero::<T0>();
        for (a, b) in pairs {
            t.load_u64::<T4>(*a, ROW_BYTES);
            t.load_u64::<T5>(&b.0, ROW_BYTES);
            t.dpbuud::<T0, T4, T5>(int8);
        }
        t.store_u64::<T0>(&mut out, ROW_BYTES);
    });
    Tile(out)
}

/// `sum over i of scalar_multiply(a[i], b[i])`, INT32-wrapping. The scalar counterpart of
/// [`amx_accumulate`], for the same cross-check.
fn scalar_accumulate(pairs: &[(&[u64; TILE_U64], &Matrix)]) -> [[u32; 16]; 16] {
    let mut sum = [[0u32; 16]; 16];
    for (a, b) in pairs {
        let c = scalar_multiply(a, b);
        for m in 0..16 {
            for n in 0..16 {
                sum[m][n] = sum[m][n].wrapping_add(c[m][n]);
            }
        }
    }
    sum
}

/// splitmix64's finalizer: a cheap, well-mixed hash of a group index, used to offset the
/// pattern rotation (see [`pattern_index`]).
fn hash64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

/// Which pattern (0..`x`) sits at tile position `t` (0..`x`) within group `group`, on pass
/// `pass`: a cyclic rotation by `pass`, offset by a hash of the group index, so every location
/// sees every pattern over `x` passes, and two locations `x` tiles apart don't always hold the
/// same pattern. `x` must be a power of two (checked by [`PatternSet::new`]), so `% x` is cheap.
fn pattern_index(x: usize, group: usize, pass: usize, t: usize) -> usize {
    let h = hash64(group as u64) as usize;
    (t + pass + h) % x
}

/// A pattern set: `x` pattern tiles, the shared transform matrix, `checks` = `1 + r` checksum
/// matrices per pattern, and every expected value the test compares against.
pub struct PatternSet {
    x: usize,
    checks: usize,
    patterns: Vec<Tile>,
    transform: Matrix,
    /// `checksum_keys[c][i]`: check `c`'s matrix for pattern `i`.
    checksum_keys: Vec<Vec<Matrix>>,
    /// `T(P_i)` for each pattern.
    expected_t: Vec<Tile>,
    /// The group checksum for each check, the same for every group and pass (every group holds
    /// every pattern exactly once).
    expected_checksum: Vec<Tile>,
}

impl PatternSet {
    /// Builds a set of `x` patterns (`x` a power of two, 16/32/64 suggested; TMR's chunk sizes
    /// are multiples of 4 KiB = 4 tiles, so `x` up to 64 always fits) with `checks` = `1 + r`
    /// independent checksum matrices per pattern (`r` the CPU-load setting: `r` is extra work
    /// in the R pass over and above the one real check).
    ///
    /// Every matrix byte is odd (`[odd_matrix`]/[`distinct_odd_matrices`]): flipping one bit of
    /// a byte changes it by `d = +-2^b`, and for any odd `k`, `d * k` is never 0 mod 2^32 (`d`
    /// is a nonzero power of two up to 128, `k` is odd and nonzero, so `|d*k| < 2^16`, far from
    /// wrapping). So a single flipped bit always changes every checksum it contributes to.
    /// Two flips can only cancel in one checksum if they hit the same bit of two matrix entries
    /// whose *rows* coincide; [`distinct_odd_matrices`] makes every row of every checksum
    /// matrix distinct, which rules that out. Three or more is not ruled out by construction;
    /// `tests_xform.rs` measures it with fault injection.
    ///
    /// Panics if the AMX-computed expected values disagree with an independent scalar
    /// computation (see the module doc): that would mean the CPU's multiply unit is already
    /// wrong, before any memory is tested.
    pub fn new(int8: AmxInt8, x: usize, checks: usize, seed: u64) -> PatternSet {
        assert!(
            x.is_power_of_two() && (2..=64).contains(&x),
            "x must be a power of two, 2..=64"
        );
        assert!(
            checks >= 1 && checks + 2 <= 8,
            "checks (1 + r) must leave room for 2 operand tiles"
        );
        let mut rng = Rng::new(seed);
        let patterns = gen_patterns(x, &mut rng);
        let transform = odd_matrix(&mut rng);
        let checksum_keys: Vec<Vec<Matrix>> = (0..checks)
            .map(|_| distinct_odd_matrices(x, &mut rng))
            .collect();

        let expected_t: Vec<Tile> = patterns
            .iter()
            .map(|p| {
                let amx = amx_multiply(int8, &p.0, &transform);
                let scalar = grid_to_tile(&scalar_multiply(&p.0, &transform));
                assert_eq!(
                    amx, scalar,
                    "AMX multiply unit disagrees with the scalar reference (transform)"
                );
                amx
            })
            .collect();

        let expected_checksum: Vec<Tile> = (0..checks)
            .map(|c| {
                let pairs: Vec<(&[u64; TILE_U64], &Matrix)> = expected_t
                    .iter()
                    .zip(&checksum_keys[c])
                    .map(|(t, k)| (&t.0, k))
                    .collect();
                let amx = amx_accumulate(int8, &pairs);
                let scalar = grid_to_tile(&scalar_accumulate(&pairs));
                assert_eq!(
                    amx, scalar,
                    "AMX multiply unit disagrees with the scalar reference (checksum)"
                );
                amx
            })
            .collect();

        PatternSet {
            x,
            checks,
            patterns,
            transform,
            checksum_keys,
            expected_t,
            expected_checksum,
        }
    }

    pub fn x(&self) -> usize {
        self.x
    }

    pub fn checks(&self) -> usize {
        self.checks
    }

    /// Words per group (`x` whole tiles).
    pub fn group_words(&self) -> usize {
        self.x * TILE_U64
    }

    /// `T(pattern i)`: what a clean tile holding pattern `i` looks like after RW. For tests.
    #[cfg(test)]
    pub fn expected(&self, i: usize) -> Tile {
        self.expected_t[i]
    }

    /// Pattern `i`'s raw bytes, before the transform. For tests.
    #[cfg(test)]
    pub fn pattern(&self, i: usize) -> Tile {
        self.patterns[i]
    }

    /// Which pattern (0..`x`) the rotation puts at tile position `t` of group `group` on pass
    /// `pass`. For tests; the real pipeline computes this itself in [`run_group`].
    #[cfg(test)]
    pub fn rotation(&self, group: usize, pass: usize, t: usize) -> usize {
        pattern_index(self.x, group, pass, t)
    }
}

/// What tier 2 found for one bad tile. `group`, `pattern` and `diffs` aren't read by this
/// prototype's own tests (which key on `tile` and `before_transform`); they're here for the
/// bench reporting and TMR integration the design doc's item 3 plans.
#[derive(Debug)]
#[expect(
    dead_code,
    reason = "diagnostic fields for the reporting this prototype doesn't build yet"
)]
pub struct TileError {
    pub group: usize,
    /// Tile position within the group.
    pub tile: usize,
    /// Which pattern that tile should have held.
    pub pattern: usize,
    /// `(row, col, expected, actual)` for every differing dword of the 16x16 result.
    pub diffs: Vec<(usize, usize, u32, u32)>,
    /// Set when every difference lies in one output row and matches one flipped input bit
    /// propagated through the transform matrix: `(byte, bit, sign)`. That signature means the
    /// error was in the pattern *before* the transform (RW read a wrong byte from `A`), not in
    /// the stored result. `None` either means the corruption was in the stored result directly
    /// (the common case: an isolated byte or few bytes differ, not a whole row), or that more
    /// than one bit's worth of corruption happened.
    pub before_transform: Option<(usize, u32, i64)>,
}

/// Compares one tile's actual, stored value against its expected `T(pattern)` and reports every
/// differing dword, with the before/after-transform classification the design doc describes.
fn diagnose_tile(
    group: usize,
    tile: usize,
    pattern: usize,
    actual: &Tile,
    expected: &Tile,
    transform: &Matrix,
) -> TileError {
    let (a, e) = (tile_to_grid(actual), tile_to_grid(expected));
    let mut diffs = Vec::new();
    for (m, (ar, er)) in a.iter().zip(&e).enumerate() {
        for (n, (&av, &ev)) in ar.iter().zip(er).enumerate() {
            if av != ev {
                diffs.push((m, n, ev, av));
            }
        }
    }
    let before_transform = classify_before_transform(&a, &e, transform);
    TileError {
        group,
        tile,
        pattern,
        diffs,
        before_transform,
    }
}

/// If every differing dword lies in one row `m` of the 16x16 result, tries every byte `j` < 64
/// and signed power-of-two `d` (a single flipped bit) to see whether the whole row's delta
/// matches `d * transform[j][*]` exactly: that's the signature a flipped input bit leaves after
/// propagating through the transform matrix. 64 bytes x 8 bits x 2 signs = 1024 candidates,
/// each checked against the row's 16 entries: cheap, and only runs on a tier-2 mismatch.
fn classify_before_transform(
    a: &[[u32; 16]; 16],
    e: &[[u32; 16]; 16],
    transform: &Matrix,
) -> Option<(usize, u32, i64)> {
    let mut rows = (0..16).filter(|&m| (0..16).any(|n| a[m][n] != e[m][n]));
    let m = rows.next()?;
    if rows.next().is_some() {
        return None; // more than one row differs: not a single propagated bit
    }
    for j in 0..64 {
        for bit in 0..8u32 {
            for sign in [1i64, -1i64] {
                let d = sign * (1i64 << bit);
                let matches = (0..16).all(|n| {
                    let delta = a[m][n].wrapping_sub(e[m][n]);
                    let want = (d.wrapping_mul(transform.get(j, n) as i64) as i32) as u32;
                    delta == want
                });
                if matches {
                    return Some((j, bit, sign));
                }
            }
        }
    }
    None
}

/// W and RW for one group (`group.len() == set.group_words()`): fills it with the pattern set,
/// rotated for group `g` and `pass`, then transforms every tile in place. `inject`, if not a
/// no-op, runs on the freshly-filled patterns between W and RW, so a test can corrupt a
/// pattern "before the transform" rather than the stored result after it ([`fill`] always
/// passes a no-op; [`pass_inject`] below is the test-only hook).
///
/// The RW session captures a raw pointer into `group`, not a `&mut [u64]`: plumb_tiles' calling
/// practices (crate docs) want captures by value, which a `Copy` pointer gives for free (a
/// `move` closure over a `Copy` local, including a slice reference re-derived each call, needs
/// care — see `amx_multiply`'s comment for the sharper version of this trap). One session for
/// the whole group, not per tile, keeps the `LDTILECFG`/`TILERELEASE` overhead down (as
/// `tiles.rs`'s `k_amx_mirror` also does).
fn fill_group(
    int8: AmxInt8,
    set: &PatternSet,
    group: &mut [u64],
    g: usize,
    pass: usize,
    inject: &mut dyn FnMut(&mut [u64]),
) {
    let indices: Vec<usize> = (0..set.x)
        .map(|t| pattern_index(set.x, g, pass, t))
        .collect();
    let tile_count = group.len() / TILE_U64;

    // W: fill, rotated.
    let (tiles, _) = group.as_chunks_mut::<TILE_U64>();
    for (tile, &pi) in tiles.iter_mut().zip(&indices) {
        tile.copy_from_slice(&set.patterns[pi].0);
    }
    inject(group);

    // RW: transform each tile in place.
    let transform = set.transform.0;
    let p = group.as_mut_ptr();
    int8.with_tiles(move |t| {
        for i in 0..tile_count {
            // SAFETY: tile i (i < tile_count) lies inside `group`, which nothing else touches
            // for the duration of this session.
            let tile = unsafe { std::slice::from_raw_parts_mut(p.add(i * TILE_U64), TILE_U64) };
            t.load_u64::<T4>(tile, ROW_BYTES);
            t.load_u64::<T5>(&transform, ROW_BYTES);
            t.zero::<T0>();
            t.dpbuud::<T0, T4, T5>(int8);
            t.store_u64::<T0>(tile, ROW_BYTES);
        }
    });
}

/// R and tier 2 for one already-filled group: checks `set.checks()` independent checksums
/// against what a clean group holding `g`/`pass`'s rotation should show (tier 1), and on a
/// mismatch, an exact compare of every tile against its expected value (tier 2), reported
/// through `on_error`. Doesn't touch W or RW, so it sees whatever is actually in `group` —
/// including corruption that happened after a previous [`fill_group`] wrote it. Returns
/// whether the group's checksums disagreed.
fn verify_group(
    int8: AmxInt8,
    set: &PatternSet,
    group: &[u64],
    g: usize,
    pass: usize,
    on_error: &mut dyn FnMut(TileError),
) -> bool {
    let indices: Vec<usize> = (0..set.x)
        .map(|t| pattern_index(set.x, g, pass, t))
        .collect();

    let mut bad = false;
    let p: *const u64 = group.as_ptr();
    for c in 0..set.checks {
        let keys: Vec<Matrix> = indices.iter().map(|&pi| set.checksum_keys[c][pi]).collect();
        let mut sum = [0u64; TILE_U64];
        // See `amx_multiply`'s comment: move a reference to `sum`, not `sum` itself (`Copy`).
        let sum_ref = &mut sum;
        int8.with_tiles(move |t| {
            t.zero::<T0>();
            for (i, key) in keys.iter().enumerate() {
                // SAFETY: tile i (i < tile_count == keys.len()) lies inside `group`, read-only
                // here.
                let tile = unsafe { std::slice::from_raw_parts(p.add(i * TILE_U64), TILE_U64) };
                t.load_u64::<T4>(tile, ROW_BYTES);
                t.load_u64::<T5>(&key.0, ROW_BYTES);
                t.dpbuud::<T0, T4, T5>(int8);
            }
            t.store_u64::<T0>(sum_ref, ROW_BYTES);
        });
        if sum != set.expected_checksum[c].0 {
            bad = true;
        }
    }
    if !bad {
        return false;
    }

    // Tier 2: which tile(s), exactly.
    let (tiles, _) = group.as_chunks::<TILE_U64>();
    for (ti, (tile, &pi)) in tiles.iter().zip(&indices).enumerate() {
        let actual = Tile(*tile);
        if actual != set.expected_t[pi] {
            on_error(diagnose_tile(
                g,
                ti,
                pi,
                &actual,
                &set.expected_t[pi],
                &set.transform,
            ));
        }
    }
    true
}

/// W and RW over the whole of `buf`, a whole number of `set.group_words()`-sized groups: fills
/// every group with the pattern set (rotated per `pass`) and transforms each tile in place.
/// Pairs with [`verify`], as a separate step, so something can happen to the memory in between
/// (the sleep a real deployment would add between TM5-style write and verify phases; a test
/// corrupting the stored result directly).
pub fn fill(int8: AmxInt8, set: &PatternSet, buf: &mut [u64], pass: usize) {
    let group_words = set.group_words();
    assert!(
        buf.len().is_multiple_of(group_words),
        "buf must be a whole number of groups"
    );
    for (g, group) in buf.chunks_mut(group_words).enumerate() {
        fill_group(int8, set, group, g, pass, &mut |_| {});
    }
}

/// R and tier 2 over the whole of `buf`: checks every group's checksums against what [`fill`]
/// with the same `pass` should have left there, running tier 2 on any mismatch. Returns the
/// number of groups with at least one wrong checksum.
pub fn verify(
    int8: AmxInt8,
    set: &PatternSet,
    buf: &[u64],
    pass: usize,
    on_error: &mut dyn FnMut(TileError),
) -> u64 {
    let group_words = set.group_words();
    assert!(
        buf.len().is_multiple_of(group_words),
        "buf must be a whole number of groups"
    );
    let mut bad_groups = 0u64;
    for (g, group) in buf.chunks(group_words).enumerate() {
        if verify_group(int8, set, group, g, pass, on_error) {
            bad_groups += 1;
        }
    }
    bad_groups
}

/// [`fill`] then [`verify`], immediately: convenient where nothing happens in between.
pub fn pass(
    int8: AmxInt8,
    set: &PatternSet,
    buf: &mut [u64],
    pass: usize,
    on_error: &mut dyn FnMut(TileError),
) -> u64 {
    fill(int8, set, buf, pass);
    verify(int8, set, buf, pass, on_error)
}

/// Test-only: as [`fill`], but on exactly one group, with `inject` run on the freshly-filled
/// patterns before RW reads them — a hook for fault injection "before the transform" that
/// [`fill`]'s public signature has no room for. Follow with [`verify`] (on the same one-group
/// buffer, same `pass`) to run tier 1/2 against it.
#[cfg(test)]
pub fn fill_inject(
    int8: AmxInt8,
    set: &PatternSet,
    group: &mut [u64],
    pass: usize,
    inject: &mut dyn FnMut(&mut [u64]),
) {
    assert_eq!(
        group.len(),
        set.group_words(),
        "fill_inject takes exactly one group"
    );
    fill_group(int8, set, group, 0, pass, inject);
}
