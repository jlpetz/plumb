// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU load to run next to a memory test (plumb TODO 6, owner's design 2026-10-21). A load thread
//! runs multiply-add batches in registers on its own core or sibling hyperthread, so the test's
//! code and register allocation are untouched. Every batch checks its result exactly against a
//! reference computed once at startup, so the load also catches compute errors.
//!
//! - `sse2`: 128-bit multiply then add (TMR's 128-bit level has no FMA)
//! - `avx2`: 256-bit FMA
//! - `avx512`: 512-bit FMA
//! - `amx`: INT8 tile dot products (`TDPBUUD`), no memory traffic
//!
//! A batch starts from the same values every time, so its result never changes. The
//! floating-point references are scalar: `f64::mul_add` is the same fused operation as the FMA,
//! and Rust never contracts `x * m + a`, so they match bit for bit. After `k` iterations an AMX
//! accumulator holds `k * (A x B)` mod 2^32, computed in scalar integer code.

use crate::common::Cpu;
use plumb_tiles::{AmxInt8, ROW_BYTES, T0, T1, T2, T3, T4, T5, T6, T7};
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Independent accumulator chains: enough for two FMA ports at four cycles' latency.
pub const CHAINS: usize = 8;
/// Iterations per floating-point batch, two multiply-adds per chain each.
pub const ITERS: usize = 2048;
/// Iterations per AMX batch, four tile multiplies each.
pub const AMX_ITERS: u32 = 256;
const TILE: usize = 1024;

/// The values a batch starts from, kept in memory so the compiler can't fold the batch away.
#[derive(Clone)]
#[repr(C, align(64))]
pub struct Seeds {
    pub start: [[f64; 8]; CHAINS],
    /// Alternating multiply-adds keep the values changing inside a batch, and bounded.
    pub m: [f64; 2],
    pub a: [f64; 2],
}

impl Seeds {
    pub fn new() -> Seeds {
        Seeds {
            start: std::array::from_fn(|c| {
                std::array::from_fn(|l| 1.0 + (c * 8 + l) as f64 / 64.0)
            }),
            m: [1.0 - 1.0 / 2048.0, 1.0 + 1.0 / 4096.0],
            a: [0.75 / 2048.0, -0.25 / 4096.0],
        }
    }

    /// The batch result, in scalar code: fused (FMA) or a multiply then an add (SSE2).
    pub fn reference(&self, fused: bool) -> [[f64; 8]; CHAINS] {
        let step = |x: f64, k: usize| {
            if fused {
                x.mul_add(self.m[k], self.a[k])
            } else {
                x * self.m[k] + self.a[k]
            }
        };
        std::array::from_fn(|c| {
            std::array::from_fn(|l| {
                let mut x = self.start[c][l];
                for _ in 0..ITERS {
                    x = step(step(x, 0), 1);
                }
                x
            })
        })
    }
}

/// 128-bit batch: `CHAINS` chains of `xmm`, a multiply then an add per step.
///
/// # Safety
/// None beyond the signature; `unsafe` only because SSE2 is a target feature.
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "sse2")]
pub unsafe fn k_load_sse2(s: &Seeds, out: &mut [[f64; 8]; CHAINS]) {
    let (m1, m2) = (_mm_set1_pd(s.m[0]), _mm_set1_pd(s.m[1]));
    let (a1, a2) = (_mm_set1_pd(s.a[0]), _mm_set1_pd(s.a[1]));
    // SAFETY: each row has 8 f64, the load reads 2.
    let mut x: [__m128d; CHAINS] =
        std::array::from_fn(|c| unsafe { _mm_loadu_pd(s.start[c].as_ptr()) });
    for _ in 0..ITERS {
        for v in &mut x {
            *v = _mm_add_pd(_mm_mul_pd(*v, m1), a1);
            *v = _mm_add_pd(_mm_mul_pd(*v, m2), a2);
        }
    }
    for (o, v) in out.iter_mut().zip(x) {
        // SAFETY: as the loads.
        unsafe { _mm_storeu_pd(o.as_mut_ptr(), v) };
    }
}

/// 256-bit batch: `CHAINS` chains of `ymm`, one FMA per step.
///
/// # Safety
/// The CPU has AVX2 and FMA.
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn k_load_avx2(s: &Seeds, out: &mut [[f64; 8]; CHAINS]) {
    let (m1, m2) = (_mm256_set1_pd(s.m[0]), _mm256_set1_pd(s.m[1]));
    let (a1, a2) = (_mm256_set1_pd(s.a[0]), _mm256_set1_pd(s.a[1]));
    // SAFETY: each row has 8 f64, the load reads 4.
    let mut x: [__m256d; CHAINS] =
        std::array::from_fn(|c| unsafe { _mm256_loadu_pd(s.start[c].as_ptr()) });
    for _ in 0..ITERS {
        for v in &mut x {
            *v = _mm256_fmadd_pd(*v, m1, a1);
            *v = _mm256_fmadd_pd(*v, m2, a2);
        }
    }
    for (o, v) in out.iter_mut().zip(x) {
        // SAFETY: as the loads.
        unsafe { _mm256_storeu_pd(o.as_mut_ptr(), v) };
    }
}

/// 512-bit batch: `CHAINS` chains of `zmm`, one FMA per step.
///
/// # Safety
/// The CPU has AVX-512F.
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "avx512f")]
pub unsafe fn k_load_avx512(s: &Seeds, out: &mut [[f64; 8]; CHAINS]) {
    let (m1, m2) = (_mm512_set1_pd(s.m[0]), _mm512_set1_pd(s.m[1]));
    let (a1, a2) = (_mm512_set1_pd(s.a[0]), _mm512_set1_pd(s.a[1]));
    // SAFETY: each row has 8 f64, the load reads 8.
    let mut x: [__m512d; CHAINS] =
        std::array::from_fn(|c| unsafe { _mm512_loadu_pd(s.start[c].as_ptr()) });
    for _ in 0..ITERS {
        for v in &mut x {
            *v = _mm512_fmadd_pd(*v, m1, a1);
            *v = _mm512_fmadd_pd(*v, m2, a2);
        }
    }
    for (o, v) in out.iter_mut().zip(x) {
        // SAFETY: as the loads.
        unsafe { _mm512_storeu_pd(o.as_mut_ptr(), v) };
    }
}

/// One 1 KiB tile's bytes.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Tile(pub [u8; TILE]);

/// AMX operands: two `A` and two `B` tiles of seeded bytes.
#[derive(Clone)]
pub struct AmxSeeds {
    pub a: [Tile; 2],
    pub b: [Tile; 2],
}

impl AmxSeeds {
    pub fn new(seed: u64) -> AmxSeeds {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        };
        let mut tile = || Tile(std::array::from_fn(|_| next()));
        AmxSeeds {
            a: [tile(), tile()],
            b: [tile(), tile()],
        }
    }

    /// `AMX_ITERS * (a[i] x b[j])` mod 2^32 in TDPBUUD's layout, for (i, j) = 00, 01, 10, 11:
    /// `C[m][n] = sum(k < 16, e < 4) A[m][4k + e] * B[k][4n + e]`.
    pub fn reference(&self) -> [[u32; 256]; 4] {
        let prod = |a: &Tile, b: &Tile| -> [u32; 256] {
            std::array::from_fn(|mn| {
                let (m, n) = (mn / 16, mn % 16);
                let mut sum = 0u32;
                for k in 0..16 {
                    for e in 0..4 {
                        let x = a.0[m * ROW_BYTES + 4 * k + e] as u32;
                        let y = b.0[k * ROW_BYTES + 4 * n + e] as u32;
                        sum = sum.wrapping_add(x * y);
                    }
                }
                sum.wrapping_mul(AMX_ITERS)
            })
        };
        [
            prod(&self.a[0], &self.b[0]),
            prod(&self.a[0], &self.b[1]),
            prod(&self.a[1], &self.b[0]),
            prod(&self.a[1], &self.b[1]),
        ]
    }
}

/// AMX batch: `C[2i + j] += A[i] x B[j]` (`TDPBUUD`) `AMX_ITERS` times from zero, in tiles 0-3,
/// with the operands in tiles 4-7; then the accumulators are stored to `out` for the check.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_load_amx(int8: AmxInt8, s: &AmxSeeds, out: &mut [Tile; 4]) {
    int8.with_tiles(move |t| {
        t.load::<T4>(&s.a[0].0, ROW_BYTES);
        t.load::<T5>(&s.a[1].0, ROW_BYTES);
        t.load::<T6>(&s.b[0].0, ROW_BYTES);
        t.load::<T7>(&s.b[1].0, ROW_BYTES);
        t.zero::<T0>();
        t.zero::<T1>();
        t.zero::<T2>();
        t.zero::<T3>();
        for _ in 0..AMX_ITERS {
            t.dpbuud::<T0, T4, T6>(int8);
            t.dpbuud::<T1, T4, T7>(int8);
            t.dpbuud::<T2, T5, T6>(int8);
            t.dpbuud::<T3, T5, T7>(int8);
        }
        let [o0, o1, o2, o3] = out;
        t.store::<T0>(&mut o0.0, ROW_BYTES);
        t.store::<T1>(&mut o1.0, ROW_BYTES);
        t.store::<T2>(&mut o2.0, ROW_BYTES);
        t.store::<T3>(&mut o3.0, ROW_BYTES);
    })
}

/// Which load to run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Sse2,
    Avx2,
    Avx512,
    Amx,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "sse2" => Kind::Sse2,
            "avx2" => Kind::Avx2,
            "avx512" => Kind::Avx512,
            "amx" => Kind::Amx,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Sse2 => "sse2",
            Kind::Avx2 => "avx2",
            Kind::Avx512 => "avx512",
            Kind::Amx => "amx",
        }
    }
}

/// On and off phases in microseconds; `off_us == 0` is continuous load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Duty {
    pub on_us: u64,
    pub off_us: u64,
}

impl Duty {
    pub const CONTINUOUS: Duty = Duty {
        on_us: 1000,
        off_us: 0,
    };

    /// `on:off` in microseconds, e.g. `500:500`.
    pub fn parse(s: &str) -> Option<Duty> {
        let (on, off) = s.split_once(':')?;
        let d = Duty {
            on_us: on.parse().ok()?,
            off_us: off.parse().ok()?,
        };
        (d.on_us > 0).then_some(d)
    }
}

/// Counters a load thread updates and the coordinator reads.
#[derive(Default)]
pub struct Stats {
    pub batches: AtomicU64,
    /// Values that didn't match the reference: compute errors.
    pub errors: AtomicU64,
}

enum Work {
    Fp {
        seeds: Box<Seeds>,
        want: Box<[[f64; 8]; CHAINS]>,
        lanes: usize,
    },
    Amx {
        int8: AmxInt8,
        seeds: Box<AmxSeeds>,
        want: Box<[[u32; 256]; 4]>,
    },
}

/// A load, with its reference computed: `Load::new`, then `run` on the load thread.
pub struct Load {
    kind: Kind,
    work: Work,
}

impl Load {
    /// `None` if the CPU (or, for AMX, the OS) doesn't support `kind`.
    pub fn new(kind: Kind, cpu: &Cpu) -> Option<Load> {
        let fp = |lanes, fused| {
            let seeds = Box::new(Seeds::new());
            let want = Box::new(seeds.reference(fused));
            Work::Fp { seeds, want, lanes }
        };
        let work = match kind {
            Kind::Sse2 => fp(2, false),
            Kind::Avx2 if cpu.avx2 && is_x86_feature_detected!("fma") => fp(4, true),
            Kind::Avx512 if cpu.avx512f => fp(8, true),
            Kind::Amx => {
                let int8 = AmxInt8::try_new()?;
                let seeds = Box::new(AmxSeeds::new(0x9E37_79B9_7F4A_7C15));
                let want = Box::new(seeds.reference());
                Work::Amx { int8, seeds, want }
            }
            _ => return None,
        };
        Some(Load { kind, work })
    }

    #[cfg(test)]
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// One batch; returns how many values differ from the reference.
    pub fn batch(&self) -> u64 {
        match &self.work {
            Work::Fp { seeds, want, lanes } => {
                let mut out = [[0.0; 8]; CHAINS];
                // SAFETY: `new` checked the features each kernel needs.
                unsafe {
                    match self.kind {
                        Kind::Sse2 => k_load_sse2(seeds, &mut out),
                        Kind::Avx2 => k_load_avx2(seeds, &mut out),
                        _ => k_load_avx512(seeds, &mut out),
                    }
                }
                let mut bad = 0;
                for (o, w) in out.iter().zip(want.iter()) {
                    bad += o[..*lanes]
                        .iter()
                        .zip(&w[..*lanes])
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count() as u64;
                }
                bad
            }
            Work::Amx { int8, seeds, want } => {
                let mut out = [Tile([0; TILE]); 4];
                k_load_amx(*int8, seeds, &mut out);
                let mut bad = 0;
                for (o, w) in out.iter().zip(want.iter()) {
                    bad +=
                        o.0.as_chunks::<4>()
                            .0
                            .iter()
                            .zip(w)
                            .filter(|(b, w)| u32::from_le_bytes(**b) != **w)
                            .count() as u64;
                }
                bad
            }
        }
    }

    /// Batches until `stop`: `duty.on_us` of batches, then `duty.off_us` of spinning (PAUSE).
    pub fn run(&self, duty: Duty, stop: &AtomicBool, stats: &Stats) {
        let (on, off) = (
            Duration::from_micros(duty.on_us),
            Duration::from_micros(duty.off_us),
        );
        while !stop.load(Ordering::Relaxed) {
            let t0 = Instant::now();
            while t0.elapsed() < on {
                let bad = self.batch();
                stats.batches.fetch_add(1, Ordering::Relaxed);
                if bad > 0 {
                    stats.errors.fetch_add(bad, Ordering::Relaxed);
                }
            }
            let t1 = Instant::now();
            while t1.elapsed() < off && !stop.load(Ordering::Relaxed) {
                std::hint::spin_loop();
            }
        }
    }

    /// For tests: a load whose reference is wrong, so every batch must report errors.
    #[cfg(test)]
    pub fn with_wrong_reference(mut self) -> Load {
        match &mut self.work {
            Work::Fp { want, .. } => want[3][1] = f64::from_bits(want[3][1].to_bits() ^ 1),
            Work::Amx { want, .. } => want[2][17] ^= 1 << 9,
        }
        self
    }
}
