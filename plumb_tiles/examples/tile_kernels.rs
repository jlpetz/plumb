// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Memory-tester-shaped tile kernels for `bench/asm_check.py`, which checks (via
//! `bench/expect/tiles.json`) that each kernel's tile instruction sits inside its loop with no
//! call there, and that the session's STTILECFG/LDTILECFG/TILERELEASE wrap it.
//!
//! ```text
//! cd bench && python asm_check.py --package plumb_tiles --example tile_kernels
//! ```
//!
//! `main` runs each kernel once on small buffers and checks the results: memory kernels byte
//! for byte, compute kernels against a scalar reference on varied data (so a kernel that wires
//! the wrong tiles together, or swaps A and B, fails). The `amx` test binary includes this
//! file as a module and calls it, so `cargo test` runs the kernels too.
//!
//! The kernels take whole units (tiles, or 64 KiB blocks for the 4096-stride ones) and panic on
//! anything else rather than skip a tail: a memory tester that quietly leaves bytes out hasn't
//! tested them.

pub(crate) fn main() {
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    kernels::run();
    #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
    println!("tile instructions need x86_64 with 64-bit pointers; nothing to run");
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "test data: noise bytes and small loop counts, truncated on purpose"
)]
pub(crate) mod kernels {
    use plumb_tiles::{
        Amx, AmxBf16, AmxFp16, AmxInt8, ROW_BYTES, ROWS, T0, T1, T2, T3, T4, T5, T6, T7,
        TILE_BYTES, tile_span,
    };

    /// u64s per packed tile.
    pub(crate) const TILE_WORDS: usize = TILE_BYTES / 8;
    pub(crate) const PAGE: usize = 4096;
    /// u64s per 16-page block: one tile at a 4096-byte stride spans it.
    pub(crate) const BLOCK_WORDS: usize = ROWS * PAGE / 8;

    /// `buf` as whole `N`-word units; panics if there is a remainder.
    #[inline(always)]
    #[track_caller]
    fn whole<const N: usize>(buf: &[u64]) -> &[[u64; N]] {
        let (units, rest) = buf.as_chunks::<N>();
        assert!(
            rest.is_empty(),
            "{} words is not a whole number of {N}-word units",
            buf.len()
        );
        units
    }

    #[inline(always)]
    #[track_caller]
    fn whole_mut<const N: usize>(buf: &mut [u64]) -> &mut [[u64; N]] {
        let len = buf.len();
        let (units, rest) = buf.as_chunks_mut::<N>();
        assert!(
            rest.is_empty(),
            "{len} words is not a whole number of {N}-word units"
        );
        units
    }

    /// Copies `src` to `dst` a packed 1 KiB tile at a time (a move test's inner loop). Two
    /// tiles in flight, so the second load doesn't wait for the first store; an odd last tile
    /// goes on its own.
    ///
    /// # Panics
    ///
    /// Unless `src` and `dst` have the same length, a whole number of tiles.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_copy(amx: Amx, src: &[u64], dst: &mut [u64]) {
        assert_eq!(
            src.len(),
            dst.len(),
            "k_tile_copy: src and dst lengths differ"
        );
        let (src2, src1) = whole::<TILE_WORDS>(src).as_chunks::<2>();
        let (dst2, dst1) = whole_mut::<TILE_WORDS>(dst).as_chunks_mut::<2>();
        amx.with_tiles(|t| {
            for ([s0, s1], [d0, d1]) in src2.iter().zip(dst2) {
                t.load_u64::<T0>(s0, ROW_BYTES);
                t.load_u64::<T1>(s1, ROW_BYTES);
                t.store_u64::<T0>(d0, ROW_BYTES);
                t.store_u64::<T1>(d1, ROW_BYTES);
            }
            for (s, d) in src1.iter().zip(dst1) {
                t.load_u64::<T0>(s, ROW_BYTES);
                t.store_u64::<T0>(d, ROW_BYTES);
            }
        });
    }

    /// Fills `dst` with a 1 KiB pattern tile (a fill pass): one load, then a store per tile.
    ///
    /// # Panics
    ///
    /// Unless `dst` is a whole number of tiles.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_fill(amx: Amx, pattern: &[u64; TILE_WORDS], dst: &mut [u64]) {
        let dst = whole_mut::<TILE_WORDS>(dst);
        amx.with_tiles(|t| {
            t.load_u64::<T2>(pattern, ROW_BYTES);
            for d in dst {
                t.store_u64::<T2>(d, ROW_BYTES);
            }
        });
    }

    /// Reads `buf` in 16-page blocks, one 64-byte column of all 16 pages per load: each
    /// TILELOADD touches 16 different pages (16 DRAM rows on most interleavings).
    ///
    /// # Panics
    ///
    /// Unless `buf` is a whole number of 64 KiB blocks.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_stride_load(amx: Amx, buf: &[u64]) {
        let blocks = whole::<BLOCK_WORDS>(buf);
        amx.with_tiles(|t| {
            for block in blocks {
                for col in (0..PAGE / 8).step_by(ROW_BYTES / 8) {
                    t.load_u64::<T0>(&block[col..], PAGE);
                }
            }
        });
    }

    /// [`k_tile_stride_load`] with the T1 (low reuse) hint.
    ///
    /// # Panics
    ///
    /// As [`k_tile_stride_load`].
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_stride_load_t1(amx: Amx, buf: &[u64]) {
        let blocks = whole::<BLOCK_WORDS>(buf);
        amx.with_tiles(|t| {
            for block in blocks {
                for col in (0..PAGE / 8).step_by(ROW_BYTES / 8) {
                    t.load_t1_u64::<T0>(&block[col..], PAGE);
                }
            }
        });
    }

    /// Writes a pattern tile across `buf` at a 4096-byte stride: each TILESTORED writes one
    /// line in each of 16 pages.
    ///
    /// # Panics
    ///
    /// Unless `buf` is a whole number of 64 KiB blocks.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_stride_fill(amx: Amx, pattern: &[u64; TILE_WORDS], buf: &mut [u64]) {
        let blocks = whole_mut::<BLOCK_WORDS>(buf);
        amx.with_tiles(|t| {
            t.load_u64::<T3>(pattern, ROW_BYTES);
            for block in blocks {
                for col in (0..PAGE / 8).step_by(ROW_BYTES / 8) {
                    t.store_u64::<T3>(&mut block[col..], PAGE);
                }
            }
        });
    }

    /// A strided read whose geometry comes from a config, as in TMR: `tiles` loads at byte
    /// `stride`, the `i`-th starting `i * step` words into `buf`. With stride and step known
    /// only at run time, each load keeps its bounds checks (compares in the loop, panic paths
    /// outside it), which is what this kernel shows `asm_check` that the constant-stride ones
    /// can't. Stores the last tile read to `last`. The closure is `move`: with the `nightly`
    /// feature, values it captured by reference would be reloaded after every tile load.
    ///
    /// # Panics
    ///
    /// If a tile doesn't fit in `buf`.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_rt_stride_load(
        amx: Amx,
        buf: &[u64],
        stride: usize,
        step: usize,
        tiles: usize,
        last: &mut [u64; TILE_WORDS],
    ) {
        amx.with_tiles(move |t| {
            for i in 0..tiles {
                t.load_u64::<T0>(&buf[i * step..], stride);
            }
            t.store_u64::<T0>(last, ROW_BYTES);
        });
    }

    /// The store side of [`k_tile_rt_stride_load`], on a byte buffer: `tiles` copies of
    /// `pattern` at byte `stride`, the `i`-th starting `i * step` bytes into `buf`. `move` for
    /// the same reason.
    ///
    /// # Panics
    ///
    /// If a tile doesn't fit in `buf`.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_rt_stride_store(
        amx: Amx,
        pattern: &[u8; TILE_BYTES],
        buf: &mut [u8],
        stride: usize,
        step: usize,
        tiles: usize,
    ) {
        amx.with_tiles(move |t| {
            t.load::<T3>(pattern, ROW_BYTES);
            for i in 0..tiles {
                t.store::<T3>(&mut buf[i * step..], stride);
            }
        });
    }

    /// A TMUL heat load: 2x2-blocked INT8 GEMM steps, four accumulators fed by two A tiles
    /// (T4 = `a[0]`, T5 = `a[1]`) and two B tiles (T6 = `b[0]`, T7 = `b[1]`), `iters` times:
    /// `c[0] = A0 B0`, `c[1] = A0 B1`, `c[2] = A1 B0`, `c[3] = A1 B1`.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_dpbssd(
        i8: AmxInt8,
        a: &[[u8; TILE_BYTES]; 2],
        b: &[[u8; TILE_BYTES]; 2],
        c: &mut [[u8; TILE_BYTES]; 4],
        iters: usize,
    ) {
        i8.with_tiles(|t| {
            t.load::<T4>(&a[0], ROW_BYTES);
            t.load::<T5>(&a[1], ROW_BYTES);
            t.load::<T6>(&b[0], ROW_BYTES);
            t.load::<T7>(&b[1], ROW_BYTES);
            t.zero::<T0>();
            t.zero::<T1>();
            t.zero::<T2>();
            t.zero::<T3>();
            for _ in 0..iters {
                t.dpbssd::<T0, T4, T6>(i8);
                t.dpbssd::<T1, T4, T7>(i8);
                t.dpbssd::<T2, T5, T6>(i8);
                t.dpbssd::<T3, T5, T7>(i8);
            }
            let [c0, c1, c2, c3] = c;
            t.store::<T0>(c0, ROW_BYTES);
            t.store::<T1>(c1, ROW_BYTES);
            t.store::<T2>(c2, ROW_BYTES);
            t.store::<T3>(c3, ROW_BYTES);
        });
    }

    /// One accumulator, `iters` BF16 dot products: `c = iters * A B`.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_dpbf16ps(
        bf16: AmxBf16,
        a: &[u8; TILE_BYTES],
        b: &[u8; TILE_BYTES],
        c: &mut [u8; TILE_BYTES],
        iters: usize,
    ) {
        bf16.with_tiles(|t| {
            t.load::<T1>(a, ROW_BYTES);
            t.load::<T2>(b, ROW_BYTES);
            t.zero::<T0>();
            for _ in 0..iters {
                t.dpbf16ps::<T0, T1, T2>(bf16);
            }
            t.store::<T0>(c, ROW_BYTES);
        });
    }

    /// One accumulator, `iters` FP16 dot products: `c = iters * A B`.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub(crate) fn k_tile_dpfp16ps(
        fp16: AmxFp16,
        a: &[u8; TILE_BYTES],
        b: &[u8; TILE_BYTES],
        c: &mut [u8; TILE_BYTES],
        iters: usize,
    ) {
        fp16.with_tiles(|t| {
            t.load::<T1>(a, ROW_BYTES);
            t.load::<T2>(b, ROW_BYTES);
            t.zero::<T0>();
            for _ in 0..iters {
                t.dpfp16ps::<T0, T1, T2>(fp16);
            }
            t.store::<T0>(c, ROW_BYTES);
        });
    }

    // ---- checks ------------------------------------------------------------------------------

    /// Deterministic varied data (xorshift64).
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        fn tile(&mut self) -> [u8; TILE_BYTES] {
            core::array::from_fn(|_| self.next() as u8)
        }
    }

    /// The TDP layout on element values: tiles of 16 rows x 16 dwords, `lanes` elements per
    /// dword (4 for INT8, 2 for 16-bit types), `C[m][n] = sum(k < 16, i < lanes)
    /// A[m][lanes k + i] * B[k][lanes n + i]`.
    fn tdp_reference(a: &[i32], b: &[i32], lanes: usize) -> Vec<i32> {
        let row = 16 * lanes;
        (0..16 * 16)
            .map(|mn| {
                let (m, n) = (mn / 16, mn % 16);
                (0..16)
                    .flat_map(|k| (0..lanes).map(move |i| (k, i)))
                    .map(|(k, i)| a[m * row + lanes * k + i] * b[k * row + lanes * n + i])
                    .sum()
            })
            .collect()
    }

    fn dwords<T>(tile: &[u8; TILE_BYTES], from: fn([u8; 4]) -> T) -> Vec<T> {
        tile.as_chunks::<4>().0.iter().map(|d| from(*d)).collect()
    }

    /// A tile of small integers in -4..=4 as bf16 or f16 (`one_to_four` holds the encodings of
    /// 1..=4), and their values. Products and sums of these are exact in f32, so the hardware
    /// and the reference can't round differently.
    fn small_int_halves(noise: &mut Noise, one_to_four: [u16; 4]) -> ([u8; TILE_BYTES], Vec<i32>) {
        let values: Vec<i32> = (0..TILE_BYTES / 2)
            .map(|_| (noise.next() % 9).cast_signed() - 4)
            .collect();
        let mut tile = [0_u8; TILE_BYTES];
        for (h, &v) in tile.as_chunks_mut::<2>().0.iter_mut().zip(&values) {
            let magnitude = if v == 0 {
                0
            } else {
                one_to_four[v.unsigned_abs() as usize - 1]
            };
            *h = (magnitude | if v < 0 { 0x8000 } else { 0 }).to_le_bytes();
        }
        (tile, values)
    }

    pub(crate) fn run() {
        let Some(amx) = Amx::try_new() else {
            println!("no AMX on this machine; nothing to run");
            return;
        };
        check_memory_kernels(amx);
        let mut noise = Noise(0x9E37_79B9_7F4A_7C15);
        if let Some(i8) = amx.int8() {
            let iters = 3;
            let (a, b) = ([noise.tile(), noise.tile()], [noise.tile(), noise.tile()]);
            let mut c = [[0_u8; TILE_BYTES]; 4];
            k_tile_dpbssd(i8, &a, &b, &mut c, iters);
            let signed = |t: &[u8; TILE_BYTES]| -> Vec<i32> {
                t.iter().map(|&x| i32::from(x.cast_signed())).collect()
            };
            let (a, b) = (a.each_ref().map(signed), b.each_ref().map(signed));
            for (j, (x, y)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                let want: Vec<i32> = tdp_reference(&a[x], &b[y], 4)
                    .iter()
                    .map(|v| v * iters as i32)
                    .collect();
                assert_eq!(
                    dwords(&c[j], i32::from_le_bytes),
                    want,
                    "k_tile_dpbssd c[{j}] = A{x} B{y}"
                );
            }
        }
        let iters = 2;
        if let Some(bf16) = amx.bf16() {
            let one_to_four = [0x3F80, 0x4000, 0x4040, 0x4080];
            let ((a, av), (b, bv)) = (
                small_int_halves(&mut noise, one_to_four),
                small_int_halves(&mut noise, one_to_four),
            );
            let mut c = [0_u8; TILE_BYTES];
            k_tile_dpbf16ps(bf16, &a, &b, &mut c, iters);
            let want: Vec<f32> = tdp_reference(&av, &bv, 2)
                .iter()
                .map(|&v| (v * iters as i32) as f32)
                .collect();
            assert_eq!(dwords(&c, f32::from_le_bytes), want, "k_tile_dpbf16ps");
        }
        if let Some(fp16) = amx.fp16() {
            let one_to_four = [0x3C00, 0x4000, 0x4200, 0x4400];
            let ((a, av), (b, bv)) = (
                small_int_halves(&mut noise, one_to_four),
                small_int_halves(&mut noise, one_to_four),
            );
            let mut c = [0_u8; TILE_BYTES];
            k_tile_dpfp16ps(fp16, &a, &b, &mut c, iters);
            let want: Vec<f32> = tdp_reference(&av, &bv, 2)
                .iter()
                .map(|&v| (v * iters as i32) as f32)
                .collect();
            assert_eq!(dwords(&c, f32::from_le_bytes), want, "k_tile_dpfp16ps");
        }
        println!("all tile kernels ran and checked out");
    }

    fn check_memory_kernels(amx: Amx) {
        // An odd number of tiles, so k_tile_copy's single-tile tail runs too.
        let src: Vec<u64> = (0..(2 * BLOCK_WORDS + TILE_WORDS) as u64).collect();
        let mut dst = vec![0_u64; src.len()];
        k_tile_copy(amx, &src, &mut dst);
        assert_eq!(src, dst, "k_tile_copy");

        let pattern: [u64; TILE_WORDS] = core::array::from_fn(|i| 0xA5A5_0000_0000_0000 | i as u64);
        k_tile_fill(amx, &pattern, &mut dst);
        assert!(
            whole::<TILE_WORDS>(&dst).iter().all(|d| *d == pattern),
            "k_tile_fill"
        );

        let blocks = &src[..2 * BLOCK_WORDS];
        k_tile_stride_load(amx, blocks);
        k_tile_stride_load_t1(amx, blocks);

        let blocks = &mut dst[..2 * BLOCK_WORDS];
        blocks.fill(0);
        k_tile_stride_fill(amx, &pattern, blocks);
        // Row r of the pattern lands at the same column of page r in every block.
        for (w, &v) in blocks.iter().enumerate() {
            let (page, word) = (w / (PAGE / 8) % ROWS, w % (PAGE / 8));
            assert_eq!(
                v,
                pattern[page * (ROW_BYTES / 8) + word % (ROW_BYTES / 8)],
                "k_tile_stride_fill word {w}"
            );
        }

        // A diagonal: each row one page and one line further on, a new tile every line.
        let (stride, step, tiles) = (PAGE + ROW_BYTES, ROW_BYTES, 16);
        let mut last = [0_u64; TILE_WORDS];
        k_tile_rt_stride_load(amx, &src, stride, step / 8, tiles, &mut last);
        let first_word = (tiles - 1) * step / 8;
        for r in 0..ROWS {
            assert_eq!(
                last[r * ROW_BYTES / 8..][..ROW_BYTES / 8],
                src[first_word + r * stride / 8..][..ROW_BYTES / 8],
                "k_tile_rt_stride_load row {r}"
            );
        }

        let mut noise = Noise(0x2545_F491_4F6C_DD1D);
        let pattern = noise.tile();
        let tiles = 8;
        // Two spare lines after the last row, all checked: the span is a whole number of lines.
        let mut buf =
            vec![0xEE_u8; (tiles - 1) * step + tile_span(stride).unwrap() + 2 * ROW_BYTES];
        k_tile_rt_stride_store(amx, &pattern, &mut buf, stride, step, tiles);
        // Tile i row r covers line i + 65 r; with fewer than 65 tiles no two rows share a line.
        for (line, bytes) in buf.as_chunks::<ROW_BYTES>().0.iter().enumerate() {
            let (i, r) = (line % (stride / ROW_BYTES), line / (stride / ROW_BYTES));
            if i < tiles && r < ROWS {
                assert_eq!(
                    bytes[..],
                    pattern[r * ROW_BYTES..][..ROW_BYTES],
                    "k_tile_rt_stride_store line {line}"
                );
            } else {
                assert!(
                    bytes.iter().all(|&b| b == 0xEE),
                    "k_tile_rt_stride_store wrote line {line}"
                );
            }
        }
    }
}
