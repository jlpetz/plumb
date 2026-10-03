//! fearless-test: can fearless_simd replace TMR-APP's per-width `macro_rules!` SIMD kernels?
//! (TODO 84.) Each TMR kernel shape is written twice, in `tmr.rs` (TMR's current style) and
//! `fs.rs` (one generic fearless_simd body), and timed side by side on the same buffers.
//!
//! The asm verdicts come from `asm_check.py`, not from these timings (see FINDINGS.md).
//!
//! Two regimes:
//! - **L2**: one pinned thread, 256 KiB warm buffer. Where codegen differences show (Rule 3:
//!   widths converge once DRAM-bound). Page size is irrelevant at this size.
//! - **DRAM**: thread sweep (default 1,2,4,6,8), one large-page region per thread (default
//!   2 GiB on 1 GiB pages), threads pinned physical-cores-first. Each worker times its own pass
//!   between barriers; the batch time is the slowest thread, aggregate = bytes / that time
//!   (`../shuffle-test/`'s fix: no unpinned thread in the timing path, never fake-fast).
//!
//! Method (`TMR-APP/doc/simd_codegen_rules.md`, Rule 5): variants sampled round-robin so drift
//! hits all of them; flushes and re-dirtying are untimed and outside the barriers; median and
//! [min..max] (the full spread goes to the CSV); `black_box` on every buffer and result.
//!
//! Usage: fearless-test [--quick] [--regime l2|dram|both] [--only g1,g2] [--threads 1,2,4]
//!        [--per-thread-mib N] [--pages huge|large|small] [--samples N] [--cpu N] [--csv FILE]

#![feature(portable_simd)]
#![feature(clflushopt_target_feature)]
#![feature(simd_x86_clflushopt)]

mod cap;
mod common;
mod fs;
mod mem;
mod tmr;
#[cfg(test)]
mod tests;

use common::*;
use fearless_simd::{Avx2, Avx512, Level};
use mem::{Pages, Region};
use std::hint::black_box;
use std::io::Write as _;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

type Run<'a> = Box<dyn Fn(&mut [u64]) -> u64 + Send + Sync + 'a>;

struct Variant<'a> {
    name: &'static str,
    run: Run<'a>,
}

#[derive(Clone, Copy, PartialEq)]
enum Prime {
    /// Nothing between samples (warm buffer).
    None,
    /// Untimed CLFLUSHOPT of the whole buffer before each sample, so reads come from DRAM.
    Flush,
    /// Untimed re-fill before each sample, so the flush kernels have dirty lines to write back.
    Dirty,
}

struct Group<'a> {
    id: &'static str,
    title: &'static str,
    /// Run once before the group (e.g. prefill the pattern the verify kernels expect).
    setup: Option<Run<'a>>,
    prime: Prime,
    /// Verify kernels must return 0 on the clean buffer; anything else is a port bug.
    expect_zero: bool,
    /// Bytes counted per rep = buffer bytes x this (copies use half the buffer as source).
    bytes_mult: f64,
    variants: Vec<Variant<'a>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Regime {
    L2,
    Dram,
}

struct Opts {
    regimes: Vec<Regime>,
    only: Option<Vec<String>>,
    threads: Vec<usize>,
    per_thread_mib: usize,
    pages: Pages,
    samples: usize,
    l2_reps: usize,
    l2_samples: usize,
    cpu: usize,
    csv: String,
}

fn usage() -> ! {
    eprintln!(
        "usage: fearless-test [--quick] [--regime l2|dram|both] [--only ids] [--threads 1,2,4,6,8]\n\
         \x20      [--per-thread-mib 2048] [--pages huge|large|small] [--samples 5] [--cpu 2]\n\
         \x20      [--csv results.csv]\n\
         group ids: fill verify4 posw posv lcgw flush ntw wflush pfv copy"
    );
    std::process::exit(2)
}

fn parse_opts() -> Opts {
    let mut o = Opts {
        regimes: vec![Regime::L2, Regime::Dram],
        only: None,
        threads: vec![1, 2, 4, 6, 8],
        per_thread_mib: 2048,
        pages: Pages::Huge,
        samples: 5,
        l2_reps: 2000,
        l2_samples: 21,
        cpu: 2,
        csv: "results.csv".into(),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let val = |i: usize| args.get(i + 1).cloned().unwrap_or_else(|| usage());
        let num = |i: usize| val(i).parse::<usize>().unwrap_or_else(|_| usage());
        match args[i].as_str() {
            "--quick" => {
                o.threads = vec![1, 4, 8];
                o.per_thread_mib = 1024;
                o.samples = 3;
                o.l2_reps = 400;
                o.l2_samples = 9;
                i += 1;
                continue;
            }
            "--regime" => {
                o.regimes = match val(i).as_str() {
                    "l2" => vec![Regime::L2],
                    "dram" => vec![Regime::Dram],
                    "both" => vec![Regime::L2, Regime::Dram],
                    _ => usage(),
                }
            }
            "--only" => o.only = Some(val(i).split(',').map(str::to_string).collect()),
            "--threads" => {
                o.threads = val(i).split(',').map(|t| t.parse().unwrap_or_else(|_| usage())).collect()
            }
            "--per-thread-mib" => o.per_thread_mib = num(i),
            "--pages" => {
                o.pages = match val(i).as_str() {
                    "huge" => Pages::Huge,
                    "large" => Pages::Large,
                    "small" => Pages::Small,
                    _ => usage(),
                }
            }
            "--samples" => o.samples = num(i).max(1),
            "--cpu" => o.cpu = num(i),
            "--csv" => o.csv = val(i),
            _ => usage(),
        }
        i += 2;
    }
    o
}

/// Tokens for the variants this CPU can run. `None` skips the variant (no false comparisons).
struct Toks {
    t2: Avx2,
    t512: Option<Avx512>,
    /// TMR-style 512 needs only AVX-512F/BW/CD/DQ/VL; fearless_simd's Avx512 needs Ice Lake.
    tmr512: bool,
    level: Level,
    cpu: Cpu,
    /// CLFLUSHOPT capability token (prototype, `cap.rs`); gated at startup like TMR's.
    cf: cap::Clflushopt,
}

macro_rules! v {
    ($name:literal, $f:expr) => {
        Variant { name: $name, run: Box::new($f) }
    };
}

/// Pointer-style TMR kernel adapters.
fn w(f: unsafe fn(*mut u64, usize)) -> impl Fn(&mut [u64]) -> u64 + Send + Sync {
    move |b: &mut [u64]| {
        // SAFETY: whole, page-aligned buffer; the caller checked the CPU features.
        unsafe { f(b.as_mut_ptr(), b.len()) };
        0
    }
}
fn r(f: unsafe fn(*const u64, usize) -> u64) -> impl Fn(&mut [u64]) -> u64 + Send + Sync {
    // SAFETY: as above.
    move |b: &mut [u64]| unsafe { f(b.as_ptr(), b.len()) }
}
fn split(b: &mut [u64]) -> (&mut [u64], &[u64]) {
    let half = b.len() / 2;
    let (src, dst) = b.split_at_mut(half);
    (dst, &*src)
}

fn groups<'a>(t: &'a Toks, regime: Regime) -> Vec<Group<'a>> {
    let (t2, t512, tmr512, lvl, cf) = (t.t2, t.t512, t.tmr512, t.level, t.cf);
    let dram = regime == Regime::Dram;
    let mut gs = Vec::new();
    let push_if = |vs: &mut Vec<Variant<'a>>, ok: bool, v: Variant<'a>| {
        if ok {
            vs.push(v)
        }
    };

    let mut vs = vec![
        v!("tmr_128", w(tmr::k_fill_tmr_128)),
        v!("fs_128", move |b| { fs::k_fill_fs_128(t2, b); 0 }),
        v!("tmr_256", w(tmr::k_fill_tmr_256)),
        v!("fs_256", move |b| { fs::k_fill_fs_256(t2, b); 0 }),
    ];
    push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_fill_tmr_512)));
    if let Some(t5) = t512 {
        vs.push(v!("fs_512", move |b| { fs::k_fill_fs_512(t5, b); 0 }));
    }
    vs.push(v!("fs_auto", move |b| { fs::k_fill_fs_auto(lvl, b); 0 }));
    gs.push(Group { id: "fill", title: "constant fill (StuckBit/Refresh write)", setup: None,
        prime: Prime::None, expect_zero: false, bytes_mult: 1.0, variants: vs });

    let mut vs = vec![
        v!("tmr_128", r(tmr::k_verify4_tmr_128)),
        v!("fs_128", move |b| fs::k_verify4_fs_128(t2, b)),
        v!("tmr_256", r(tmr::k_verify4_tmr_256)),
        v!("fs_256", move |b| fs::k_verify4_fs_256(t2, b)),
    ];
    push_if(&mut vs, tmr512, v!("tmr_512", r(tmr::k_verify4_tmr_512)));
    if let Some(t5) = t512 {
        vs.push(v!("fs_512", move |b| fs::k_verify4_fs_512(t5, b)));
    }
    vs.push(v!("fs_auto", move |b| fs::k_verify4_fs_auto(lvl, b)));
    // Loop-shape experiment (fs.rs): the same body with a raw-pointer walk (TMR's shape), and
    // in L2 also a split_at walk.
    vs.push(v!("fsptr_128", move |b| fs::k_verify4_fsptr_128(t2, b)));
    vs.push(v!("fsptr_256", move |b| fs::k_verify4_fsptr_256(t2, b)));
    if let Some(t5) = t512 {
        vs.push(v!("fsptr_512", move |b| fs::k_verify4_fsptr_512(t5, b)));
    }
    if !dram {
        vs.push(v!("fssplit_128", move |b| fs::k_verify4_fssplit_128(t2, b)));
        vs.push(v!("fssplit_256", move |b| fs::k_verify4_fssplit_256(t2, b)));
        if let Some(t5) = t512 {
            vs.push(v!("fssplit_512", move |b| fs::k_verify4_fssplit_512(t5, b)));
        }
        // Footguns: a non-inlined helper in each style (L2 only; slow by design).
        push_if(&mut vs, tmr512, v!("tmrhelper_512", r(tmr::k_verify4_tmrhelper_512)));
        if let Some(t5) = t512 {
            vs.push(v!("fshelper_512", move |b| fs::k_verify4_fshelper_512(t5, b)));
        }
    }
    gs.push(Group { id: "verify4", title: "4-accumulator verify (StuckBit verify)",
        setup: Some(Box::new(w(tmr::k_fill_tmr_256))),
        prime: if dram { Prime::Flush } else { Prime::None },
        expect_zero: true, bytes_mult: 1.0, variants: vs });

    let mut vs = vec![
        v!("tmr_128", w(tmr::k_posw_tmr_128)),
        v!("fs_128", move |b| { fs::k_posw_fs_128(t2, b); 0 }),
        v!("tmr_256", w(tmr::k_posw_tmr_256)),
        v!("fs_256", move |b| { fs::k_posw_fs_256(t2, b); 0 }),
    ];
    push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_posw_tmr_512)));
    if let Some(t5) = t512 {
        vs.push(v!("fs_512", move |b| { fs::k_posw_fs_512(t5, b); 0 }));
    }
    gs.push(Group { id: "posw", title: "positional write idx^base (SimpleTest Mode 0/1)",
        setup: None, prime: Prime::None, expect_zero: false, bytes_mult: 1.0, variants: vs });

    let mut vs = vec![
        v!("tmr_128", r(tmr::k_posv_tmr_128)),
        v!("fs_128", move |b| fs::k_posv_fs_128(t2, b)),
        v!("tmr_256", r(tmr::k_posv_tmr_256)),
        v!("fs_256", move |b| fs::k_posv_fs_256(t2, b)),
    ];
    push_if(&mut vs, tmr512, v!("tmr_512", r(tmr::k_posv_tmr_512)));
    if let Some(t5) = t512 {
        vs.push(v!("fs_512", move |b| fs::k_posv_fs_512(t5, b)));
    }
    gs.push(Group { id: "posv", title: "positional verify, 1 accumulator (SimpleTest verify)",
        setup: Some(Box::new(w(tmr::k_posw_tmr_256))),
        prime: if dram { Prime::Flush } else { Prime::None },
        expect_zero: true, bytes_mult: 1.0, variants: vs });

    let mut vs = vec![
        v!("tmr_128", w(tmr::k_lcgw_tmr_128)),
        v!("fs_128", move |b| { fs::k_lcgw_fs_128(t2, b); 0 }),
        v!("tmr_256", w(tmr::k_lcgw_tmr_256)),
        v!("fs_256", move |b| { fs::k_lcgw_fs_256(t2, b); 0 }),
    ];
    push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_lcgw_tmr_512)));
    if let Some(t5) = t512 {
        vs.push(v!("fs_512", move |b| { fs::k_lcgw_fs_512(t5, b); 0 }));
    }
    gs.push(Group { id: "lcgw", title: "LCG write state*m+a (SimpleTest Mode 2, 64-bit mul)",
        setup: None, prime: Prime::None, expect_zero: false, bytes_mult: 1.0, variants: vs });

    // flush: L2 = 256 KiB of dirty lines; DRAM = a mostly-evicted range (TMR's use).
    let vs = vec![
        v!("tmr", |b: &mut [u64]| {
            // SAFETY: whole buffer; CLFLUSHOPT gated in main.
            unsafe { tmr::k_flush_tmr(b.as_ptr(), b.len()) };
            0
        }),
        v!("fsintr", move |b| { fs::k_flush_fsintr(t2, b); 0 }),
        v!("fsasm", move |b| { fs::k_flush_fsasm(t2, b); 0 }),
        v!("fscap", move |b| { cap::k_flush_fscap(t2, cf, b); 0 }),
    ];
    gs.push(Group { id: "flush", title: "CLFLUSHOPT range + MFENCE (flush_range_to_dram)",
        setup: None, prime: Prime::Dirty, expect_zero: false, bytes_mult: 1.0, variants: vs });

    if dram {
        let mut vs = vec![
            v!("tmr_128", w(tmr::k_ntw_tmr_128)),
            v!("fs_128", move |b| { fs::k_ntw_fs_128(t2, b); 0 }),
            v!("tmr_256", w(tmr::k_ntw_tmr_256)),
            v!("fs_256", move |b| { fs::k_ntw_fs_256(t2, b); 0 }),
        ];
        push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_ntw_tmr_512)));
        if let Some(t5) = t512 {
            vs.push(v!("fs_512", move |b| { fs::k_ntw_fs_512(t5, b); 0 }));
            vs.push(v!("fsk_512", move |b| { fs::k_ntw_fsk_512(t5, b); 0 }));
        }
        gs.push(Group { id: "ntw", title: "NT positional write, 4x unroll + SFENCE (SimpleNT)",
            setup: None, prime: Prime::None, expect_zero: false, bytes_mult: 1.0, variants: vs });

        let mut vs = vec![
            v!("tmr_256", w(tmr::k_wflush_tmr_256)),
            v!("fsintr_256", move |b| { fs::k_wflush_fsintr_256(t2, b); 0 }),
            v!("fsasm_256", move |b| { fs::k_wflush_fsasm_256(t2, b); 0 }),
            v!("fscap_256", move |b| { cap::k_wflush_fscap_256(t2, cf, b, PATTERN); 0 }),
        ];
        push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_wflush_tmr_512)));
        if let Some(t5) = t512 {
            vs.push(v!("fsintr_512", move |b| { fs::k_wflush_fsintr_512(t5, b); 0 }));
            vs.push(v!("fsasm_512", move |b| { fs::k_wflush_fsasm_512(t5, b); 0 }));
            vs.push(v!("fscap_512", move |b| { cap::k_wflush_fscap_512(t5, cf, b, PATTERN); 0 }));
        }
        gs.push(Group { id: "wflush", title: "write line + CLFLUSHOPT it, same loop (mixed)",
            setup: None, prime: Prime::None, expect_zero: false, bytes_mult: 1.0, variants: vs });

        let mut vs = vec![
            v!("tmr_256", r(tmr::k_pfv_tmr_256)),
            v!("fs_256", move |b| fs::k_pfv_fs_256(t2, b)),
        ];
        push_if(&mut vs, tmr512, v!("tmr_512", r(tmr::k_pfv_tmr_512)));
        if let Some(t5) = t512 {
            vs.push(v!("fs_512", move |b| fs::k_pfv_fs_512(t5, b)));
            vs.push(v!("fsptr_512", move |b| fs::k_pfv_fsptr_512(t5, b)));
        }
        vs.insert(2, v!("fsptr_256", move |b| fs::k_pfv_fsptr_256(t2, b)));
        gs.push(Group { id: "pfv", title: "4-accumulator verify + PREFETCHT0 per line",
            setup: Some(Box::new(w(tmr::k_fill_tmr_256))), prime: Prime::Flush,
            expect_zero: true, bytes_mult: 1.0, variants: vs });

        let mut vs = Vec::new();
        if tmr512 {
            vs.push(v!("nt_tmr_512", |b| {
                let (d, s) = split(b);
                // SAFETY: equal halves of an aligned buffer; AVX-512F checked.
                unsafe { tmr::k_copynt_tmr_512(d.as_mut_ptr(), s.as_ptr(), s.len()) };
                0
            }));
        }
        if let Some(t5) = t512 {
            vs.push(v!("nt_fs_512", move |b| { let (d, s) = split(b); fs::k_copynt_fs_512(t5, d, s); 0 }));
        }
        if t.cpu.movdir64b {
            vs.push(v!("md_tmr", |b| {
                let (d, s) = split(b);
                // SAFETY: equal halves of an aligned buffer; MOVDIR64B checked.
                unsafe { tmr::k_copymd_tmr(d.as_mut_ptr(), s.as_ptr(), s.len()) };
                0
            }));
            vs.push(v!("md_fs", move |b| { let (d, s) = split(b); fs::k_copymd_fs(t2, d, s); 0 }));
        }
        gs.push(Group { id: "copy", title: "copy half -> half: NT-512 vs MOVDIR64B (GiB/s copied)",
            setup: None, prime: Prime::Flush, expect_zero: false, bytes_mult: 0.5, variants: vs });
    }
    gs
}

#[derive(Clone, Copy)]
struct Stat {
    med: f64,
    min: f64,
    max: f64,
}

fn gib_s(bytes: f64, secs: f64) -> f64 {
    bytes / secs / (1u64 << 30) as f64
}

/// Median/min/max of throughputs (already in GiB/s).
fn stat(mut xs: Vec<f64>) -> Stat {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Stat { med: xs[xs.len() / 2], min: xs[0], max: xs[xs.len() - 1] }
}

fn prime_buf(p: Prime, buf: &mut [u64]) {
    match p {
        Prime::None => {}
        // SAFETY: whole buffer; CLFLUSHOPT gated in main.
        Prime::Flush => unsafe { tmr::k_flush_tmr(buf.as_ptr(), buf.len()) },
        // SAFETY: whole buffer; AVX2 gated in main.
        Prime::Dirty => unsafe { tmr::k_fill_tmr_256(buf.as_mut_ptr(), buf.len()) },
    }
}

/// Warm-up rep, which doubles as the port-correctness check on real-size data.
fn warm_and_check(g: &Group, buf: &mut [u64], bad: &AtomicBool) {
    if let Some(s) = &g.setup {
        s(buf);
    }
    for v in &g.variants {
        let res = (v.run)(black_box(&mut *buf));
        if g.expect_zero && res != 0 {
            eprintln!("  !! {} reported a mismatch on clean data: port bug, numbers invalid", v.name);
            bad.store(true, Ordering::Relaxed);
        }
    }
}

/// L2 regime: one thread, many reps per sample (one rep when a prime is needed between reps).
fn run_l2(g: &Group, buf: &mut [u64], reps: usize, samples: usize) -> Vec<Stat> {
    let bad = AtomicBool::new(false);
    warm_and_check(g, buf, &bad);
    let reps = if g.prime == Prime::None { reps } else { 1 };
    let mut gibs: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); g.variants.len()];
    let bytes = (buf.len() * 8) as f64 * g.bytes_mult * reps as f64;
    let mut sink = 0u64;
    for _ in 0..samples {
        for (k, v) in g.variants.iter().enumerate() {
            prime_buf(g.prime, buf);
            let t0 = Instant::now();
            for _ in 0..reps {
                sink = sink.wrapping_add((v.run)(black_box(&mut *buf)));
            }
            gibs[k].push(gib_s(bytes, t0.elapsed().as_secs_f64()));
        }
    }
    black_box(sink);
    gibs.into_iter().map(stat).collect()
}

/// DRAM regime at `t` threads. Worker k is pinned to `order[k]` and drives `regions[k]` only.
fn run_mt(g: &Group, regions: &[Region], order: &[usize], t: usize, samples: usize) -> Vec<Stat> {
    let nv = g.variants.len();
    let barrier = Barrier::new(t);
    let bad = AtomicBool::new(false);
    let per_thread: Vec<Vec<Vec<f64>>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..t)
            .map(|k| {
                let (barrier, bad, region, cpu) = (&barrier, &bad, &regions[k], order[k]);
                s.spawn(move || {
                    pin_to_cpu(cpu);
                    // SAFETY: region k is used by this worker only.
                    let buf = unsafe { region.slice() };
                    warm_and_check(g, buf, bad);
                    let mut secs: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); nv];
                    let mut sink = 0u64;
                    for _ in 0..samples {
                        for (vi, v) in g.variants.iter().enumerate() {
                            prime_buf(g.prime, buf);
                            barrier.wait();
                            let t0 = Instant::now();
                            sink = sink.wrapping_add((v.run)(black_box(&mut *buf)));
                            secs[vi].push(t0.elapsed().as_secs_f64());
                            // Nobody primes the next sample while another thread is timed.
                            barrier.wait();
                        }
                    }
                    black_box(sink);
                    secs
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    });
    let bytes = (regions[0].bytes() as f64) * g.bytes_mult * t as f64;
    (0..nv)
        .map(|vi| {
            stat((0..samples)
                .map(|smp| {
                    let wall = per_thread.iter().map(|th| th[vi][smp]).fold(0.0, f64::max);
                    gib_s(bytes, wall)
                })
                .collect())
        })
        .collect()
}

/// The TMR-style row a variant is compared against: same width suffix, or plain "tmr".
fn baseline_of(name: &str, names: &[&str]) -> Option<usize> {
    if name.starts_with("tmr") || name.ends_with("_tmr") || name.contains("_tmr_") {
        return None;
    }
    let width = name.rsplit('_').next().unwrap_or("");
    let want = match width {
        "128" | "256" | "512" => {
            if name.starts_with("nt_") {
                format!("nt_tmr_{width}")
            } else {
                format!("tmr_{width}")
            }
        }
        "auto" => {
            return names.iter().position(|n| *n == "tmr_512")
                .or_else(|| names.iter().position(|n| *n == "tmr_256"));
        }
        _ if name.starts_with("md_") => "md_tmr".to_string(),
        _ => "tmr".to_string(),
    };
    names.iter().position(|n| *n == want)
}

/// Allocate one region per worker, each from a thread pinned to that worker's CPU (so a
/// multi-node box would place it locally). 1 GiB pages first, then 2 MiB; never 4 KiB unless
/// asked for, and never a silent fallback: the page mix is printed.
fn alloc_regions(n: usize, bytes: usize, want: Pages, order: &[usize]) -> Result<Vec<Region>, String> {
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..n)
            .map(|k| {
                let cpu = order[k];
                s.spawn(move || {
                    pin_to_cpu(cpu);
                    let ladder: &[Pages] = match want {
                        Pages::Huge => &[Pages::Huge, Pages::Large],
                        Pages::Large => &[Pages::Large],
                        Pages::Small => &[Pages::Small],
                    };
                    let mut last = 0;
                    for &p in ladder {
                        match mem::alloc(bytes, p) {
                            Ok(r) => return Ok(r),
                            Err(e) => last = e,
                        }
                    }
                    Err(match last {
                        1450 => format!("region {k}: no contiguous physical memory for large pages (1450)"),
                        1314 => format!("region {k}: SeLockMemoryPrivilege not held (1314)"),
                        e => format!("region {k}: VirtualAlloc2 failed ({e})"),
                    })
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("alloc thread panicked")).collect()
    })
}

fn main() {
    let o = parse_opts();
    let cpu = detect();
    if !(cpu.avx2 && cpu.clflushopt) {
        eprintln!("needs AVX2 + CLFLUSHOPT (TMR's startup gate)");
        std::process::exit(1);
    }
    let level = Level::new();
    let toks = Toks {
        t2: level.as_avx2().expect("AVX2 detected but fearless_simd found no Avx2 level"),
        t512: level.as_avx512(),
        tmr512: cpu.avx512f && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512cd") && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl"),
        level,
        cpu,
        cf: cap::Clflushopt::try_new().expect("CLFLUSHOPT checked above"),
    };
    let order = mem::cpu_order();

    println!("fearless-test (TODO 84): TMR-style macro_rules!/std::simd vs fearless_simd 1.0");
    println!("cpu: avx2={} avx512f={} clflushopt={} movdir64b={} | amx tile={} int8={} bf16={} os_xtile={}",
        cpu.avx2, cpu.avx512f, cpu.clflushopt, cpu.movdir64b, cpu.amx_tile, cpu.amx_int8,
        cpu.amx_bf16, cpu.os_amx);
    println!("fearless_simd Level::new() = {:?}  (TMR-style 512: {}, fs Avx512 token: {})",
        level, toks.tmr512, toks.t512.is_some());
    println!("topology: {} logical / {} physical, pin order {:?}", order.len(), mem::physical_cores(), order);
    if toks.tmr512 && toks.t512.is_none() {
        println!("note: AVX-512F present but not the Ice Lake set; fs_512 rows are skipped");
    }

    let mut csv = std::fs::File::create(&o.csv).expect("create csv");
    writeln!(csv, "regime,group,variant,threads,pages,median_gib_s,min_gib_s,max_gib_s,samples").unwrap();

    for &regime in &o.regimes {
        match regime {
            Regime::L2 => {
                pin_to_cpu(o.cpu);
                println!("\n######## L2 regime: 1 thread on CPU {}, 256 KiB warm, median GiB/s [min..max]",
                    o.cpu);
                let mut buf = AlignedBuf::new(256 << 10);
                for g in groups(&toks, regime) {
                    if g.variants.is_empty() || o.only.as_ref().is_some_and(|ids| !ids.iter().any(|x| x == g.id)) {
                        continue;
                    }
                    let primed = g.prime != Prime::None;
                    let (reps, samples) = if primed { (1, o.l2_samples * 10) } else { (o.l2_reps, o.l2_samples) };
                    println!("\n== {} | {} ({} samples x {} reps)", g.id, g.title, samples, reps);
                    let stats = run_l2(&g, buf.as_mut_slice(), reps, samples);
                    let names: Vec<&str> = g.variants.iter().map(|v| v.name).collect();
                    println!("   {:<14} {:>9}   {:>19}   {:>7}", "variant", "median", "[min .. max]", "vs tmr");
                    for (k, s) in stats.iter().enumerate() {
                        let rel = baseline_of(names[k], &names)
                            .map(|b| format!("{:>6.1}%", 100.0 * s.med / stats[b].med))
                            .unwrap_or_default();
                        println!("   {:<14} {:>9.2}   [{:>7.2} .. {:>7.2}]   {}", names[k], s.med, s.min, s.max, rel);
                        writeln!(csv, "l2,{},{},1,4 KiB,{:.3},{:.3},{:.3},{}", g.id, names[k], s.med, s.min, s.max, samples).unwrap();
                    }
                }
            }
            Regime::Dram => {
                let threads: Vec<usize> = o.threads.iter().copied().filter(|&t| t >= 1 && t <= order.len()).collect();
                let tmax = *threads.iter().max().unwrap_or(&1);
                let bytes = o.per_thread_mib << 20;
                if o.pages != Pages::Small
                    && let Err(e) = mem::enable_lock_memory_privilege()
                {
                    eprintln!("large pages unavailable: {e}\n(use --pages small to run on 4 KiB pages)");
                    std::process::exit(1);
                }
                let regions = match alloc_regions(tmax, bytes, o.pages, &order) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("allocation failed: {e}\n(try --per-thread-mib smaller, or --pages large)");
                        std::process::exit(1);
                    }
                };
                let mix: Vec<&str> = regions.iter().map(|r| r.pages.label()).collect();
                let pages = if mix.iter().all(|m| *m == mix[0]) { mix[0].to_string() } else { format!("mixed {mix:?}") };
                println!("\n######## DRAM regime: {} MiB/thread on {} pages, threads {:?}, aggregate GiB/s \
                    (bytes / slowest thread), median of {}", o.per_thread_mib, pages, threads, o.samples);
                for g in groups(&toks, regime) {
                    if g.variants.is_empty() || o.only.as_ref().is_some_and(|ids| !ids.iter().any(|x| x == g.id)) {
                        continue;
                    }
                    println!("\n== {} | {}", g.id, g.title);
                    let names: Vec<&str> = g.variants.iter().map(|v| v.name).collect();
                    let t_start = Instant::now();
                    let per_t: Vec<Vec<Stat>> = threads.iter().map(|&t| run_mt(&g, &regions, &order, t, o.samples)).collect();
                    let mut hdr = format!("   {:<12}", "variant");
                    for t in &threads {
                        hdr += &format!(" {:>13}", format!("{t}T"));
                    }
                    println!("{hdr}");
                    let mut worst = (0.0f64, "", 0usize);
                    let mut spreads = Vec::new();
                    for (k, name) in names.iter().enumerate() {
                        let mut line = format!("   {:<12}", name);
                        for (ti, &t) in threads.iter().enumerate() {
                            let s = per_t[ti][k];
                            let rel = baseline_of(name, &names)
                                .map(|b| format!(" ({:>3.0}%)", 100.0 * s.med / per_t[ti][b].med))
                                .unwrap_or_else(|| "       ".into());
                            line += &format!(" {:>6.2}{}", s.med, rel);
                            let spread = (s.max - s.min) / s.med;
                            spreads.push(spread);
                            if spread > worst.0 {
                                worst = (spread, name, t);
                            }
                            writeln!(csv, "dram,{},{},{},{},{:.3},{:.3},{:.3},{}", g.id, name, t, pages, s.med, s.min, s.max, o.samples).unwrap();
                        }
                        println!("{line}");
                    }
                    spreads.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    println!("   spread (max-min)/median: typical {:.1}%, worst {:.1}% ({} @ {}T); group took {:.0} s",
                        100.0 * spreads[spreads.len() / 2], 100.0 * worst.0, worst.1, worst.2,
                        t_start.elapsed().as_secs_f64());
                }
                drop(regions);
            }
        }
    }
    println!("\nfull per-cell [min..max]: {}", o.csv);
}
