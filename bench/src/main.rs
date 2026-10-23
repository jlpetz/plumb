// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! plumb-bench: can fearless_simd replace TMR-APP's per-width `macro_rules!` SIMD kernels?
//! (TODO 84.) Each TMR kernel shape is written twice, in `tmr.rs` (TMR's current style) and
//! `fs.rs` (one generic fearless_simd body), and timed side by side on the same buffers.
//!
//! The asm verdicts come from `asm_check.py`, not from these timings (see FINDINGS-TODO84.md).
//!
//! Two regimes:
//! - **L2**: one pinned thread, 256 KiB warm buffer. Where codegen differences show (Rule 3:
//!   widths converge once DRAM-bound). Page size is irrelevant at this size.
//! - **DRAM**: thread sweep (default 1,2,4,6,8), one large-page region per thread (default
//!   2 GiB on 1 GiB pages), threads pinned physical-cores-first. Each worker times its own pass
//!   between barriers; the batch time is the slowest thread, aggregate = bytes / that time
//!   (no unpinned thread in the timing path, never fake-fast).
//!
//! Method (`TMR-APP/doc/simd_codegen_rules.md`, Rule 5): variants sampled round-robin so drift
//! hits all of them; flushes and re-dirtying are untimed and outside the barriers; median and
//! [min..max] (the full spread goes to the CSV); `black_box` on every buffer and result.
//!
//! Usage: plumb-bench [--quick] [--regime l2|dram|both] [--only g1,g2] [--threads 1,2,4]
//!        [--per-thread-mib N] [--pages huge|large|small] [--samples N] [--cpu N] [--csv FILE]

#![feature(portable_simd)]
#![feature(clflushopt_target_feature)]
#![feature(simd_x86_clflushopt)]
#![feature(x86_amx_intrinsics)] // `amx-tile` in k_amx_verify_512's target features

mod cap;
mod common;
mod fs;
mod lines;
mod load;
mod mem;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_load;
#[cfg(test)]
mod tests_plumb;
#[cfg(test)]
mod tests_tiles;
#[cfg(test)]
mod tests_xform;
mod tiles;
mod tmr;
mod tmrport;
#[allow(dead_code)] // the `xform` group runs `pass`; the tests read the reports' fields
mod xform;

use common::*;
use fearless_simd::{Avx2, Avx512, Level};
use mem::{Pages, Region};
use std::hint::black_box;
use std::io::Write as _;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

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
    /// CPU load next to the test threads (load.rs), and its on/off duty.
    load: Option<load::Kind>,
    load_duty: load::Duty,
}

fn usage() -> ! {
    eprintln!(
        "usage: plumb-bench [--quick] [--regime l2|dram|both] [--only ids] [--threads 1,2,4,6,8]\n\
         \x20      [--per-thread-mib 2048] [--pages huge|large|small] [--samples 5] [--cpu 2]\n\
         \x20      [--csv results.csv] [--load sse2|avx2|avx512|amx] [--load-duty on_us:off_us]\n\
         group ids: fill verify4 posw posv lcgw flush sbnf amxmirror xform xformw xformrw xformr; DRAM only: ntw\n\
         \x20 wflush pfv copy fillflush sb refresh refreshx refreshnf simplent"
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
        load: None,
        load_duty: load::Duty::CONTINUOUS,
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
                o.threads = val(i)
                    .split(',')
                    .map(|t| t.parse().unwrap_or_else(|_| usage()))
                    .collect()
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
            "--load" => o.load = Some(load::Kind::parse(&val(i)).unwrap_or_else(|| usage())),
            "--load-duty" => o.load_duty = load::Duty::parse(&val(i)).unwrap_or_else(|| usage()),
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
    /// plumb_lines' CLFLUSHOPT and MOVDIR64B tokens.
    pcf: plumb_lines::Clflushopt,
    md: Option<plumb_lines::Movdir64b>,
    /// plumb_tiles' AMX token (TILE + OS-enabled tile state).
    amx: Option<plumb_tiles::Amx>,
}

macro_rules! v {
    ($name:literal, $f:expr) => {
        Variant {
            name: $name,
            run: Box::new($f),
        }
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
/// A transform-verify pass (W, RW and R) with `set`, returning the bad groups (0 when clean).
fn xform_run<const X: usize>(
    int8: plumb_tiles::AmxInt8,
    set: xform::PatternSet<X>,
) -> impl Fn(&mut [u64]) -> u64 + Send + Sync {
    move |b: &mut [u64]| xform::pass(int8, &set, b, 0, &mut xform::Rescans::default()).len() as u64
}
/// Transform-verify's R alone with `set`, on a buffer W and RW prepared.
fn xform_verify<const X: usize>(
    int8: plumb_tiles::AmxInt8,
    set: xform::PatternSet<X>,
) -> impl Fn(&mut [u64]) -> u64 + Send + Sync {
    move |b: &mut [u64]| {
        xform::verify(int8, &set, b, 0, &mut xform::Rescans::default()).len() as u64
    }
}
fn split(b: &mut [u64]) -> (&mut [u64], &[u64]) {
    let half = b.len() / 2;
    let (src, dst) = b.split_at_mut(half);
    (dst, &*src)
}

fn groups<'a>(t: &'a Toks, regime: Regime) -> Vec<Group<'a>> {
    let (t2, t512, tmr512, lvl, cf, pcf, md) = (t.t2, t.t512, t.tmr512, t.level, t.cf, t.pcf, t.md);
    let dram = regime == Regime::Dram;
    let mut gs = Vec::new();
    let push_if = |vs: &mut Vec<Variant<'a>>, ok: bool, v: Variant<'a>| {
        if ok {
            vs.push(v)
        }
    };

    let mut vs = vec![
        v!("tmr_128", w(tmr::k_fill_tmr_128)),
        v!("fs_128", move |b| {
            fs::k_fill_fs_128(t2, b);
            0
        }),
        v!("tmr_256", w(tmr::k_fill_tmr_256)),
        v!("fs_256", move |b| {
            fs::k_fill_fs_256(t2, b);
            0
        }),
    ];
    push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_fill_tmr_512)));
    if let Some(t5) = t512 {
        vs.push(v!("fs_512", move |b| {
            fs::k_fill_fs_512(t5, b);
            0
        }));
    }
    vs.push(v!("fs_auto", move |b| {
        fs::k_fill_fs_auto(lvl, b);
        0
    }));
    vs.push(v!("pv_128", move |b| {
        lines::k_fill_pv_128(t2, b);
        0
    }));
    vs.push(v!("pv_256", move |b| {
        lines::k_fill_pv_256(t2, b);
        0
    }));
    if let Some(t5) = t512 {
        vs.push(v!("pv_512", move |b| {
            lines::k_fill_pv_512(t5, b);
            0
        }));
    }
    gs.push(Group {
        id: "fill",
        title: "constant fill (StuckBit/Refresh write)",
        setup: None,
        prime: Prime::None,
        expect_zero: false,
        bytes_mult: 1.0,
        variants: vs,
    });

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
    // plumb_lines aligned views: the pointer-walk shape with no raw pointers in user code.
    vs.push(v!("pv_128", move |b| lines::k_verify4_pv_128(t2, b)));
    vs.push(v!("pv_256", move |b| lines::k_verify4_pv_256(t2, b)));
    if let Some(t5) = t512 {
        vs.push(v!("pv_512", move |b| lines::k_verify4_pv_512(t5, b)));
    }
    // Loop-shape experiment (fs.rs): the same body with a raw-pointer walk (TMR's shape), and
    // in L2 also a split_at walk.
    vs.push(v!("fsptr_128", move |b| fs::k_verify4_fsptr_128(t2, b)));
    vs.push(v!("fsptr_256", move |b| fs::k_verify4_fsptr_256(t2, b)));
    if let Some(t5) = t512 {
        vs.push(v!("fsptr_512", move |b| fs::k_verify4_fsptr_512(t5, b)));
    }
    if !dram {
        vs.push(v!("pvpat_128", move |b| lines::k_verify4_pvpat_128(t2, b)));
        vs.push(v!("pvpat_256", move |b| lines::k_verify4_pvpat_256(t2, b)));
        vs.push(v!("fssplit_128", move |b| fs::k_verify4_fssplit_128(t2, b)));
        vs.push(v!("fssplit_256", move |b| fs::k_verify4_fssplit_256(t2, b)));
        if let Some(t5) = t512 {
            vs.push(v!("fssplit_512", move |b| fs::k_verify4_fssplit_512(t5, b)));
            // The view verify at 512 in the other two loop shapes (L2 gap experiment):
            // `as_chunks` (LLVM-unrolled) and 8 accumulators.
            vs.push(v!("pvch_512", move |b| lines::k_verify4_pvch_512(t5, b)));
            vs.push(v!("pv8_512", move |b| lines::k_verify8_pv_512(t5, b)));
        }
        // Footguns: a non-inlined helper in each style (L2 only; slow by design).
        push_if(
            &mut vs,
            tmr512,
            v!("tmrhelper_512", r(tmr::k_verify4_tmrhelper_512)),
        );
        if let Some(t5) = t512 {
            vs.push(v!("fshelper_512", move |b| fs::k_verify4_fshelper_512(
                t5, b
            )));
        }
    }
    gs.push(Group {
        id: "verify4",
        title: "4-accumulator verify (StuckBit verify)",
        setup: Some(Box::new(w(tmr::k_fill_tmr_256))),
        prime: if dram { Prime::Flush } else { Prime::None },
        expect_zero: true,
        bytes_mult: 1.0,
        variants: vs,
    });

    let mut vs = vec![
        v!("tmr_128", w(tmr::k_posw_tmr_128)),
        v!("fs_128", move |b| {
            fs::k_posw_fs_128(t2, b);
            0
        }),
        v!("tmr_256", w(tmr::k_posw_tmr_256)),
        v!("fs_256", move |b| {
            fs::k_posw_fs_256(t2, b);
            0
        }),
    ];
    push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_posw_tmr_512)));
    if let Some(t5) = t512 {
        vs.push(v!("fs_512", move |b| {
            fs::k_posw_fs_512(t5, b);
            0
        }));
    }
    gs.push(Group {
        id: "posw",
        title: "positional write idx^base (SimpleTest Mode 0/1)",
        setup: None,
        prime: Prime::None,
        expect_zero: false,
        bytes_mult: 1.0,
        variants: vs,
    });

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
    vs.push(v!("pv_128", move |b| lines::k_posv_pv_128(t2, b)));
    vs.push(v!("pv_256", move |b| lines::k_posv_pv_256(t2, b)));
    if let Some(t5) = t512 {
        vs.push(v!("pv_512", move |b| lines::k_posv_pv_512(t5, b)));
    }
    gs.push(Group {
        id: "posv",
        title: "positional verify, 1 accumulator (SimpleTest verify)",
        setup: Some(Box::new(w(tmr::k_posw_tmr_256))),
        prime: if dram { Prime::Flush } else { Prime::None },
        expect_zero: true,
        bytes_mult: 1.0,
        variants: vs,
    });

    let mut vs = vec![
        v!("tmr_128", w(tmr::k_lcgw_tmr_128)),
        v!("fs_128", move |b| {
            fs::k_lcgw_fs_128(t2, b);
            0
        }),
        v!("tmr_256", w(tmr::k_lcgw_tmr_256)),
        v!("fs_256", move |b| {
            fs::k_lcgw_fs_256(t2, b);
            0
        }),
    ];
    push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_lcgw_tmr_512)));
    if let Some(t5) = t512 {
        vs.push(v!("fs_512", move |b| {
            fs::k_lcgw_fs_512(t5, b);
            0
        }));
    }
    gs.push(Group {
        id: "lcgw",
        title: "LCG write state*m+a (SimpleTest Mode 2, 64-bit mul)",
        setup: None,
        prime: Prime::None,
        expect_zero: false,
        bytes_mult: 1.0,
        variants: vs,
    });

    // flush: L2 = 256 KiB of dirty lines; DRAM = a mostly-evicted range (TMR's use).
    let vs = vec![
        v!("tmr", |b: &mut [u64]| {
            // SAFETY: whole buffer; CLFLUSHOPT gated in main.
            unsafe { tmr::k_flush_tmr(b.as_ptr(), b.len()) };
            0
        }),
        v!("fsintr", move |b| {
            fs::k_flush_fsintr(t2, b);
            0
        }),
        v!("fsasm", move |b| {
            fs::k_flush_fsasm(t2, b);
            0
        }),
        v!("fscap", move |b| {
            cap::k_flush_fscap(t2, cf, b);
            0
        }),
        v!("pl", move |b| {
            lines::k_flush_pl(pcf, b);
            0
        }),
    ];
    gs.push(Group {
        id: "flush",
        title: "CLFLUSHOPT range + MFENCE (flush_range_to_dram)",
        setup: None,
        prime: Prime::Dirty,
        expect_zero: false,
        bytes_mult: 1.0,
        variants: vs,
    });

    if dram {
        let mut vs = vec![
            v!("tmr_128", w(tmr::k_ntw_tmr_128)),
            v!("fs_128", move |b| {
                fs::k_ntw_fs_128(t2, b);
                0
            }),
            v!("tmr_256", w(tmr::k_ntw_tmr_256)),
            v!("fs_256", move |b| {
                fs::k_ntw_fs_256(t2, b);
                0
            }),
        ];
        push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_ntw_tmr_512)));
        if let Some(t5) = t512 {
            vs.push(v!("fs_512", move |b| {
                fs::k_ntw_fs_512(t5, b);
                0
            }));
            vs.push(v!("fsk_512", move |b| {
                fs::k_ntw_fsk_512(t5, b);
                0
            }));
            vs.push(v!("pl_512", move |b| {
                lines::k_ntw_pl_512(t5, b);
                0
            }));
        }
        vs.insert(
            2,
            v!("pl_128", move |b| {
                lines::k_ntw_pl_128(t2, b);
                0
            }),
        );
        vs.insert(
            5,
            v!("pl_256", move |b| {
                lines::k_ntw_pl_256(t2, b);
                0
            }),
        );
        gs.push(Group {
            id: "ntw",
            title: "NT positional write, 4x unroll + SFENCE (SimpleNT)",
            setup: None,
            prime: Prime::None,
            expect_zero: false,
            bytes_mult: 1.0,
            variants: vs,
        });

        let mut vs = vec![
            v!("tmr_256", w(tmr::k_wflush_tmr_256)),
            v!("fsintr_256", move |b| {
                fs::k_wflush_fsintr_256(t2, b);
                0
            }),
            v!("fsasm_256", move |b| {
                fs::k_wflush_fsasm_256(t2, b);
                0
            }),
            v!("fscap_256", move |b| {
                cap::k_wflush_fscap_256(t2, cf, b, PATTERN);
                0
            }),
        ];
        push_if(&mut vs, tmr512, v!("tmr_512", w(tmr::k_wflush_tmr_512)));
        if let Some(t5) = t512 {
            vs.push(v!("fsintr_512", move |b| {
                fs::k_wflush_fsintr_512(t5, b);
                0
            }));
            vs.push(v!("fsasm_512", move |b| {
                fs::k_wflush_fsasm_512(t5, b);
                0
            }));
            vs.push(v!("fscap_512", move |b| {
                cap::k_wflush_fscap_512(t5, cf, b, PATTERN);
                0
            }));
            vs.push(v!("pltok_512", move |b| {
                lines::k_wflush_pltok_512(t5, pcf, b);
                0
            }));
            vs.push(v!("plentry_512", move |b| {
                lines::k_wflush_plentry_512(t5, pcf, b);
                0
            }));
        }
        vs.insert(
            4,
            v!("pltok_256", move |b| {
                lines::k_wflush_pltok_256(t2, pcf, b);
                0
            }),
        );
        vs.insert(
            5,
            v!("plentry_256", move |b| {
                lines::k_wflush_plentry_256(t2, pcf, b);
                0
            }),
        );
        gs.push(Group {
            id: "wflush",
            title: "write line + CLFLUSHOPT it, same loop (mixed)",
            setup: None,
            prime: Prime::None,
            expect_zero: false,
            bytes_mult: 1.0,
            variants: vs,
        });

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
        gs.push(Group {
            id: "pfv",
            title: "4-accumulator verify + PREFETCHT0 per line",
            setup: Some(Box::new(w(tmr::k_fill_tmr_256))),
            prime: Prime::Flush,
            expect_zero: true,
            bytes_mult: 1.0,
            variants: vs,
        });

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
            vs.push(v!("nt_fs_512", move |b| {
                let (d, s) = split(b);
                fs::k_copynt_fs_512(t5, d, s);
                0
            }));
        }
        if t.cpu.movdir64b {
            vs.push(v!("md_tmr", |b| {
                let (d, s) = split(b);
                // SAFETY: equal halves of an aligned buffer; MOVDIR64B checked.
                unsafe { tmr::k_copymd_tmr(d.as_mut_ptr(), s.as_ptr(), s.len()) };
                0
            }));
            vs.push(v!("md_fs", move |b| {
                let (d, s) = split(b);
                fs::k_copymd_fs(t2, d, s);
                0
            }));
        }
        if let Some(m) = md {
            vs.push(v!("md_pl", move |b| {
                let (d, s) = split(b);
                lines::k_copymd_pl(m, d, s);
                0
            }));
        }
        gs.push(Group {
            id: "copy",
            title: "copy half -> half: NT-512 vs MOVDIR64B (GiB/s copied)",
            setup: None,
            prime: Prime::Flush,
            expect_zero: false,
            bytes_mult: 0.5,
            variants: vs,
        });

        // Write a chunk then flush everything written (TMR: fill + flush_range_to_dram).
        let mut vs = vec![
            v!("tmr_256", |b| {
                lines::k_fillflush_tmr_256(b);
                0
            }),
            v!("pl_256", move |b| {
                lines::k_fillflush_pl_256(t2, pcf, b);
                0
            }),
        ];
        push_if(
            &mut vs,
            tmr512,
            v!("tmr_512", |b| {
                lines::k_fillflush_tmr_512(b);
                0
            }),
        );
        if let Some(t5) = t512 {
            vs.push(v!("pl_512", move |b| {
                lines::k_fillflush_pl_512(t5, pcf, b);
                0
            }));
        }
        if let Some(m) = md {
            vs.push(v!("md_pl", move |b| {
                lines::k_fillmd_pl(m, b);
                0
            }));
        }
        gs.push(Group {
            id: "fillflush",
            title: "fill then flush the range (flush_after scope); md_pl = MOVDIR64B fill",
            setup: None,
            prime: Prime::None,
            expect_zero: false,
            bytes_mult: 1.0,
            variants: vs,
        });

        // Ported TMR tests (tmrport.rs). Throughput counts TMR's traffic: StuckBit 3 writes +
        // 3 reads, Refresh 1 + 1, SimpleNT 4 x (1 write + 5 reads).
        gs.push(port_group(
            "sb",
            "TMR StuckBit port, flush before verify (3 phases)",
            t,
            6.0,
        ));
        gs.push(port_group(
            "refresh",
            "TMR Refresh port, flush before verify (sleep omitted)",
            t,
            2.0,
        ));
        // TODO 11: what costs the TMR-style port at 256 bits, and whether Refresh without the
        // flush (TMR's default) shows it too.
        gs.push(port_group(
            "refreshx",
            "TMR Refresh port at 256 bits, one phase swapped (TODO 11)",
            t,
            2.0,
        ));
        gs.push(port_group(
            "refreshnf",
            "TMR Refresh port, no flush (TMR's default; sleep omitted)",
            t,
            2.0,
        ));
        gs.push(port_group(
            "simplent",
            "TMR SimpleNT port: 4 x (NT positional write, 5 verifies) per chunk",
            t,
            24.0,
        ));

        // AMX (plumb_tiles, tiles.rs): tile loads and stores as a DRAM access path, against
        // AVX-512 kernels moving the same bytes.
        if let Some(amx) = t.amx {
            let mut vs = vec![v!("amx", move |b| tiles::k_amx_read(amx, b))];
            push_if(
                &mut vs,
                tmr512,
                v!("zmm_tmr_512", r(tmr::k_verify4_tmr_512)),
            );
            if let Some(t5) = t512 {
                vs.push(v!("amx_verify_512", move |b| tiles::k_amx_verify_512(
                    t5, amx, b
                )));
            }
            gs.push(Group { id: "amxread",
                title: "AMX read: 4 tile loads in flight vs zmm 4-acc verify; amx_verify = tile load + zmm check",
                setup: Some(Box::new(w(tmr::k_fill_tmr_256))), prime: Prime::Flush,
                expect_zero: true, bytes_mult: 1.0, variants: vs });

            let mut vs = vec![v!("amx", move |b| {
                tiles::k_amx_fill(amx, b);
                0
            })];
            push_if(&mut vs, tmr512, v!("zmm_tmr_512", w(tmr::k_fill_tmr_512)));
            gs.push(Group {
                id: "amxfill",
                title: "AMX pattern fill (1 KiB tile stores) vs zmm fill",
                setup: None,
                prime: Prime::None,
                expect_zero: false,
                bytes_mult: 1.0,
                variants: vs,
            });

            let mut vs = vec![v!("amx", move |b| {
                let (d, s) = split(b);
                tiles::k_amx_copy(amx, d, s);
                0
            })];
            if tmr512 {
                vs.push(v!("nt_tmr_512", |b| {
                    let (d, s) = split(b);
                    // SAFETY: equal halves of an aligned buffer; AVX-512F checked.
                    unsafe { tmr::k_copynt_tmr_512(d.as_mut_ptr(), s.as_ptr(), s.len()) };
                    0
                }));
            }
            if t.cpu.movdir64b {
                vs.push(v!("md_tmr", |b| {
                    let (d, s) = split(b);
                    // SAFETY: equal halves of an aligned buffer; MOVDIR64B checked.
                    unsafe { tmr::k_copymd_tmr(d.as_mut_ptr(), s.as_ptr(), s.len()) };
                    0
                }));
            }
            gs.push(Group {
                id: "amxcopy",
                title: "copy half -> half: AMX tiles vs NT-512 vs MOVDIR64B (GiB/s copied)",
                setup: None,
                prime: Prime::Flush,
                expect_zero: false,
                bytes_mult: 0.5,
                variants: vs,
            });

            let mut vs = vec![v!("amx", move |b| tiles::k_amx_strided_read(amx, b))];
            push_if(
                &mut vs,
                tmr512,
                v!("zmm_512", r(tiles::k_amx_strided_read_zmm_512)),
            );
            gs.push(Group { id: "amxstride",
                title: "strided read, stride 4096 (one line in each of 16 pages): 1 tile load vs 16 zmm loads",
                setup: None, prime: Prime::Flush, expect_zero: false, bytes_mult: 1.0, variants: vs });
        }
    }
    // MirrorMove through tiles: TMR's whole mirror (4 MiB chunks), 16 line pairs per pair of
    // tiles with the high end at stride -64, against TMR's 512-bit swap. Both regimes: in DRAM
    // the swap is bandwidth-bound; from L2 the tiles' 16x fewer instructions could show.
    if let Some(amx) = t.amx {
        let mut vs = vec![v!("amx", move |b| {
            tiles::k_amx_mirror(amx, b);
            0
        })];
        push_if(&mut vs, tmr512, v!("zmm_tmr_512", w(tmr::k_mirror_tmr_512)));
        gs.push(Group {
            id: "amxmirror",
            title: "MirrorMove round trip, 4 MiB chunks: AMX tiles (high end at stride -64) vs TMR's zmm swap (2R+2W per byte)",
            setup: None,
            prime: if dram { Prime::Flush } else { Prime::None },
            expect_zero: false,
            bytes_mult: 4.0,
            variants: vs,
        });
    }
    gs.push(port_group(
        "sbnf",
        "TMR StuckBit port, no flush (3 phases)",
        t,
        6.0,
    ));
    // Transform-verify (xform.rs, TODO 6) under the AMX load it makes itself: the built-in set
    // (X = 16) at load settings r = 0, 1, 4 and 8, and X = 64 at r = 0. Each pass writes (W),
    // reads and writes (RW) and reads (R) the buffer: 4 bytes of traffic per byte, against
    // StuckBit's 6 (`sbnf`). Skipped if the matrix unit fails the startup check.
    if let Some(int8) = t.amx.and_then(plumb_tiles::Amx::int8) {
        if xform::self_test(int8) {
            let x16 = |r| xform::PatternSet::<16>::builtin(int8, r, 0).expect("self-tested");
            let x64 = xform::PatternSet::<64>::new(int8, 0, xform::BUILTIN_SEED);
            gs.push(Group {
                id: "xform",
                title: "Transform-verify (W, RW, R) through AMX tiles, X = 16 at r = 0/1/4/8 and X = 64",
                setup: None,
                prime: if dram { Prime::Flush } else { Prime::None },
                expect_zero: true,
                bytes_mult: 4.0,
                variants: vec![
                    v!("x16_r0", xform_run(int8, x16(0))),
                    v!("x16_r1", xform_run(int8, x16(1))),
                    v!("x16_r4", xform_run(int8, x16(4))),
                    v!("x16_r8", xform_run(int8, x16(8))),
                    v!("x64_r0", xform_run(int8, x64)),
                ],
            });
            // Each phase alone, to see which sets the pace: W (1 byte of traffic per byte), RW
            // (2) and R (1, at r = 0 and 8, on a buffer W and RW prepared once).
            let (w_set, rw_set, prep) = (x16(0), x16(0), x16(0));
            gs.push(Group {
                id: "xformw",
                title: "Transform-verify's W alone: pattern tiles stored (X = 16)",
                setup: None,
                prime: if dram { Prime::Flush } else { Prime::None },
                expect_zero: false,
                bytes_mult: 1.0,
                variants: vec![v!("x16", move |b| {
                    xform::write(int8, &w_set, b, 0);
                    0
                })],
            });
            gs.push(Group {
                id: "xformrw",
                title: "Transform-verify's RW alone: each tile multiplied in place (X = 16)",
                setup: None,
                prime: if dram { Prime::Flush } else { Prime::None },
                expect_zero: false,
                bytes_mult: 2.0,
                variants: vec![v!("x16", move |b| {
                    xform::transform(int8, &rw_set, b);
                    0
                })],
            });
            gs.push(Group {
                id: "xformr",
                title: "Transform-verify's R alone, X = 16 at r = 0 and 8",
                setup: Some(Box::new(move |b: &mut [u64]| {
                    xform::write(int8, &prep, b, 0);
                    xform::transform(int8, &prep, b);
                    0
                })),
                prime: if dram { Prime::Flush } else { Prime::None },
                expect_zero: true,
                bytes_mult: 1.0,
                variants: vec![
                    v!("x16_r0", xform_verify(int8, x16(0))),
                    v!("x16_r8", xform_verify(int8, x16(8))),
                ],
            });
        } else {
            eprintln!("xform: the matrix unit failed its startup check; group skipped");
        }
    }
    gs
}

/// The ported-TMR-test groups (TMR style vs plumb style at 256/512). They return error counts,
/// which must be 0 on clean memory.
fn port_group<'a>(
    id: &'static str,
    title: &'static str,
    t: &'a Toks,
    bytes_mult: f64,
) -> Group<'a> {
    let (t2, t512, tmr512, pcf) = (t.t2, t.t512, t.tmr512, t.pcf);
    let mut vs: Vec<Variant<'a>> = Vec::new();
    match id {
        "sb" => {
            vs.push(v!("tmr_256", tmrport::k_sb_tmr_256));
            vs.push(v!("pl_256", move |b| tmrport::k_sb_pl_256(t2, pcf, b)));
            if tmr512 {
                vs.push(v!("tmr_512", tmrport::k_sb_tmr_512));
            }
            if let Some(t5) = t512 {
                vs.push(v!("pl_512", move |b| tmrport::k_sb_pl_512(t5, pcf, b)));
                // The fill's earlier shape (LLVM-unrolled loop): does the lost unroll cost?
                vs.push(v!("plplain_512", move |b| tmrport::k_sb_plplain_512(
                    t5, pcf, b
                )));
            }
        }
        "sbnf" => {
            vs.push(v!("tmr_256", tmrport::k_sbnf_tmr_256));
            vs.push(v!("pl_256", move |b| tmrport::k_sbnf_pl_256(t2, pcf, b)));
            if tmr512 {
                vs.push(v!("tmr_512", tmrport::k_sbnf_tmr_512));
            }
            if let Some(t5) = t512 {
                vs.push(v!("pl_512", move |b| tmrport::k_sbnf_pl_512(t5, pcf, b)));
                // The fill's earlier shape (LLVM-unrolled loop): does the lost unroll cost?
                vs.push(v!("plplain_512", move |b| tmrport::k_sbnf_plplain_512(
                    t5, pcf, b
                )));
            }
        }
        "refresh" => {
            vs.push(v!("tmr_256", tmrport::k_refresh_tmr_256));
            vs.push(v!("pl_256", move |b| tmrport::k_refresh_pl_256(t2, pcf, b)));
            if tmr512 {
                vs.push(v!("tmr_512", tmrport::k_refresh_tmr_512));
            }
            if let Some(t5) = t512 {
                vs.push(v!("pl_512", move |b| tmrport::k_refresh_pl_512(t5, pcf, b)));
            }
        }
        "refreshx" => {
            // Each tmr*_256 swaps one thing in tmr_256; tmrx_256 swaps nothing (the control).
            // bnd: verify bound computed once; x2: 8 vectors per verify iteration; vpl: plumb's
            // verify; fpl: plumb's fill and flush.
            vs.push(v!("tmr_256", tmrport::k_refresh_tmr_256));
            vs.push(v!("tmrx_256", move |b| tmrport::k_refresh_tmrx_256(
                t2, pcf, b
            )));
            vs.push(v!("tmrbnd_256", move |b| tmrport::k_refresh_tmrbnd_256(
                t2, pcf, b
            )));
            vs.push(v!("tmrx2_256", move |b| tmrport::k_refresh_tmrx2_256(
                t2, pcf, b
            )));
            vs.push(v!("tmrvpl_256", move |b| tmrport::k_refresh_tmrvpl_256(
                t2, pcf, b
            )));
            vs.push(v!("tmrfpl_256", move |b| tmrport::k_refresh_tmrfpl_256(
                t2, pcf, b
            )));
            vs.push(v!("pl_256", move |b| tmrport::k_refresh_pl_256(t2, pcf, b)));
        }
        "refreshnf" => {
            vs.push(v!("tmr_256", tmrport::k_refreshnf_tmr_256));
            vs.push(v!("pl_256", move |b| tmrport::k_refreshnf_pl_256(
                t2, pcf, b
            )));
            if tmr512 {
                vs.push(v!("tmr_512", tmrport::k_refreshnf_tmr_512));
            }
            if let Some(t5) = t512 {
                vs.push(v!("pl_512", move |b| tmrport::k_refreshnf_pl_512(
                    t5, pcf, b
                )));
            }
        }
        "simplent" => {
            vs.push(v!("tmr_256", tmrport::k_simplent_tmr_256));
            vs.push(v!("pl_256", move |b| tmrport::k_simplent_pl_256(t2, b)));
            if tmr512 {
                vs.push(v!("tmr_512", tmrport::k_simplent_tmr_512));
            }
            if let Some(t5) = t512 {
                vs.push(v!("pl_512", move |b| tmrport::k_simplent_pl_512(t5, b)));
            }
        }
        _ => unreachable!("unknown port group {id}"),
    }
    Group {
        id,
        title,
        setup: None,
        prime: Prime::None,
        expect_zero: true,
        bytes_mult,
        variants: vs,
    }
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
    Stat {
        med: xs[xs.len() / 2],
        min: xs[0],
        max: xs[xs.len() - 1],
    }
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
            eprintln!(
                "  !! {} reported a mismatch on clean data: port bug, numbers invalid",
                v.name
            );
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
        handles
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });
    let bytes = (regions[0].bytes() as f64) * g.bytes_mult * t as f64;
    (0..nv)
        .map(|vi| {
            stat(
                (0..samples)
                    .map(|smp| {
                        let wall = per_thread.iter().map(|th| th[vi][smp]).fold(0.0, f64::max);
                        gib_s(bytes, wall)
                    })
                    .collect(),
            )
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
            return names
                .iter()
                .position(|n| *n == "tmr_512")
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
/// Test and load CPUs for `t` test threads with a load running: the load on each test core's
/// other hyperthread, or, where cores have one thread, test and load on alternating cores.
/// `None` if `t` test threads don't fit.
fn load_placement(cores: &[Vec<usize>], t: usize) -> Option<(Vec<usize>, Vec<usize>)> {
    if cores.iter().all(|c| c.len() >= 2) {
        let c = cores.get(..t)?;
        Some((
            c.iter().map(|c| c[0]).collect(),
            c.iter().map(|c| c[1]).collect(),
        ))
    } else {
        let c = cores.get(..2 * t)?;
        let pick = |odd| c.iter().skip(odd).step_by(2).map(|c| c[0]).collect();
        Some((pick(0), pick(1)))
    }
}

/// The load CPU for a single test thread on `cpu`: its core's other hyperthread, or the next
/// core.
fn load_partner(cores: &[Vec<usize>], cpu: usize) -> Option<usize> {
    let k = cores.iter().position(|c| c.contains(&cpu))?;
    if let Some(&sib) = cores[k].iter().find(|&&c| c != cpu) {
        return Some(sib);
    }
    cores
        .get((k + 1) % cores.len())
        .filter(|_| cores.len() > 1)
        .map(|c| c[0])
}

/// What the load threads did while a measurement ran.
struct LoadReport {
    /// Batches per second, per load thread.
    rate: f64,
    /// Values that differed from the reference: compute errors.
    errors: u64,
}

/// Runs `f` with a load thread pinned to each of `cpus` (none if `load` is `None`).
fn with_load<R>(
    load: Option<(load::Kind, load::Duty)>,
    cpus: &[usize],
    f: impl FnOnce() -> R,
) -> (R, Option<LoadReport>) {
    let Some((kind, duty)) = load else {
        return (f(), None);
    };
    let stop = AtomicBool::new(false);
    let stats: Vec<load::Stats> = cpus.iter().map(|_| load::Stats::default()).collect();
    let t0 = Instant::now();
    let r = std::thread::scope(|s| {
        for (&cpu, st) in cpus.iter().zip(&stats) {
            let stop = &stop;
            s.spawn(move || {
                pin_to_cpu(cpu);
                let l = load::Load::new(kind, &detect()).expect("load checked at startup");
                l.run(duty, stop, st);
            });
        }
        // Let the load threads reach their loops before the measurement starts.
        std::thread::sleep(Duration::from_millis(50));
        let r = f();
        stop.store(true, Ordering::Relaxed);
        r
    });
    let secs = t0.elapsed().as_secs_f64();
    let sum = |f: fn(&load::Stats) -> u64| stats.iter().map(f).sum::<u64>();
    let batches = sum(|s| s.batches.load(Ordering::Relaxed));
    let errors = sum(|s| s.errors.load(Ordering::Relaxed));
    let rate = batches as f64 / secs / cpus.len().max(1) as f64;
    (r, Some(LoadReport { rate, errors }))
}

/// One report line for a group's load threads.
fn load_line(kind: load::Kind, duty: load::Duty, reports: &[(usize, LoadReport)]) -> String {
    let mut line = format!(
        "   load {} ({}:{} us), batches/s per load thread:",
        kind.name(),
        duty.on_us,
        duty.off_us
    );
    for (t, r) in reports {
        line += &format!(" {t}T {:.0}", r.rate);
    }
    let errors: u64 = reports.iter().map(|(_, r)| r.errors).sum();
    line += &if errors == 0 {
        "; no compute errors".to_string()
    } else {
        format!("; COMPUTE ERRORS: {errors}")
    };
    line
}

fn alloc_regions(
    n: usize,
    bytes: usize,
    want: Pages,
    order: &[usize],
) -> Result<Vec<Region>, String> {
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
                        1450 => format!(
                            "region {k}: no contiguous physical memory for large pages (1450)"
                        ),
                        1314 => format!("region {k}: SeLockMemoryPrivilege not held (1314)"),
                        e => format!("region {k}: VirtualAlloc2 failed ({e})"),
                    })
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("alloc thread panicked"))
            .collect()
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
        t2: level
            .as_avx2()
            .expect("AVX2 detected but fearless_simd found no Avx2 level"),
        t512: level.as_avx512(),
        tmr512: cpu.avx512f
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512cd")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl"),
        level,
        cpu,
        cf: cap::Clflushopt::try_new().expect("CLFLUSHOPT checked above"),
        pcf: plumb_lines::Clflushopt::try_new().expect("CLFLUSHOPT checked above"),
        md: plumb_lines::Movdir64b::try_new(),
        amx: plumb_tiles::Amx::try_new(),
    };
    let order = mem::cpu_order();
    let cores = mem::cores();
    let load = o.load.map(|k| (k, o.load_duty));
    if let Some((kind, _)) = load
        && load::Load::new(kind, &cpu).is_none()
    {
        eprintln!("--load {}: not supported on this CPU/OS", kind.name());
        std::process::exit(2);
    }

    println!("plumb-bench (TODO 84): TMR-style macro_rules!/std::simd vs fearless_simd 1.0");
    println!(
        "cpu: avx2={} avx512f={} clflushopt={} movdir64b={} | amx tile={} int8={} bf16={} os_xtile={}",
        cpu.avx2,
        cpu.avx512f,
        cpu.clflushopt,
        cpu.movdir64b,
        cpu.amx_tile,
        cpu.amx_int8,
        cpu.amx_bf16,
        cpu.os_amx
    );
    println!(
        "fearless_simd Level::new() = {:?}  (TMR-style 512: {}, fs Avx512 token: {})",
        level,
        toks.tmr512,
        toks.t512.is_some()
    );
    println!(
        "topology: {} logical / {} physical, pin order {:?}",
        order.len(),
        mem::physical_cores(),
        order
    );
    if toks.tmr512 && toks.t512.is_none() {
        println!("note: AVX-512F present but not the Ice Lake set; fs_512 rows are skipped");
    }

    let mut csv = std::fs::File::create(&o.csv).expect("create csv");
    writeln!(
        csv,
        "regime,group,variant,threads,pages,median_gib_s,min_gib_s,max_gib_s,samples,load"
    )
    .unwrap();
    let load_tag = load.map_or("none".to_string(), |(k, d)| {
        format!("{}:{}:{}", k.name(), d.on_us, d.off_us)
    });

    for &regime in &o.regimes {
        match regime {
            Regime::L2 => {
                pin_to_cpu(o.cpu);
                println!(
                    "\n######## L2 regime: 1 thread on CPU {}, 256 KiB warm, median GiB/s [min..max]",
                    o.cpu
                );
                let l2_load: Vec<usize> = match load {
                    Some((kind, _)) => {
                        let Some(c) = load_partner(&cores, o.cpu) else {
                            eprintln!("--load: no CPU to put the load on next to CPU {}", o.cpu);
                            std::process::exit(2);
                        };
                        println!("load: {} on CPU {c}", kind.name());
                        vec![c]
                    }
                    None => Vec::new(),
                };
                let mut buf = AlignedBuf::new(256 << 10);
                for g in groups(&toks, regime) {
                    if g.variants.is_empty()
                        || o.only
                            .as_ref()
                            .is_some_and(|ids| !ids.iter().any(|x| x == g.id))
                    {
                        continue;
                    }
                    let primed = g.prime != Prime::None;
                    let (reps, samples) = if primed {
                        (1, o.l2_samples * 10)
                    } else {
                        (o.l2_reps, o.l2_samples)
                    };
                    println!(
                        "\n== {} | {} ({} samples x {} reps)",
                        g.id, g.title, samples, reps
                    );
                    let (stats, report) = with_load(load, &l2_load, || {
                        run_l2(&g, buf.as_mut_slice(), reps, samples)
                    });
                    let names: Vec<&str> = g.variants.iter().map(|v| v.name).collect();
                    println!(
                        "   {:<14} {:>9}   {:>19}   {:>7}",
                        "variant", "median", "[min .. max]", "vs tmr"
                    );
                    for (k, s) in stats.iter().enumerate() {
                        let rel = baseline_of(names[k], &names)
                            .map(|b| format!("{:>6.1}%", 100.0 * s.med / stats[b].med))
                            .unwrap_or_default();
                        println!(
                            "   {:<14} {:>9.2}   [{:>7.2} .. {:>7.2}]   {}",
                            names[k], s.med, s.min, s.max, rel
                        );
                        writeln!(
                            csv,
                            "l2,{},{},1,4 KiB,{:.3},{:.3},{:.3},{},{}",
                            g.id, names[k], s.med, s.min, s.max, samples, load_tag
                        )
                        .unwrap();
                    }
                    if let (Some((kind, duty)), Some(r)) = (load, report) {
                        println!("{}", load_line(kind, duty, &[(1, r)]));
                    }
                }
            }
            Regime::Dram => {
                // With a load, the test threads take the placement's test CPUs and only as many
                // thread counts as leave room for a load thread each.
                let fits = |t: usize| match load {
                    Some(_) => load_placement(&cores, t).is_some(),
                    None => t <= order.len(),
                };
                let threads: Vec<usize> = o
                    .threads
                    .iter()
                    .copied()
                    .filter(|&t| t >= 1 && fits(t))
                    .collect();
                if threads.len() < o.threads.len() {
                    println!(
                        "note: thread counts {:?} skipped (no room for them{})",
                        o.threads
                            .iter()
                            .filter(|t| !threads.contains(t))
                            .collect::<Vec<_>>(),
                        if load.is_some() {
                            " with a load thread each"
                        } else {
                            ""
                        }
                    );
                }
                let tmax = *threads.iter().max().unwrap_or(&1);
                let (test_cpus, load_cpus) = match load {
                    Some(_) => load_placement(&cores, tmax).expect("filtered above"),
                    None => (order.clone(), Vec::new()),
                };
                if let Some((kind, _)) = load {
                    println!(
                        "load: {} on CPUs {:?} next to test CPUs {:?}",
                        kind.name(),
                        load_cpus,
                        test_cpus
                    );
                }
                let bytes = o.per_thread_mib << 20;
                if o.pages != Pages::Small
                    && let Err(e) = mem::enable_lock_memory_privilege()
                {
                    eprintln!(
                        "large pages unavailable: {e}\n(use --pages small to run on 4 KiB pages)"
                    );
                    std::process::exit(1);
                }
                let regions = match alloc_regions(tmax, bytes, o.pages, &test_cpus) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!(
                            "allocation failed: {e}\n(try --per-thread-mib smaller, or --pages large)"
                        );
                        std::process::exit(1);
                    }
                };
                let mix: Vec<&str> = regions.iter().map(|r| r.pages.label()).collect();
                let pages = if mix.iter().all(|m| *m == mix[0]) {
                    mix[0].to_string()
                } else {
                    format!("mixed {mix:?}")
                };
                println!(
                    "\n######## DRAM regime: {} MiB/thread on {} pages, threads {:?}, aggregate GiB/s \
                    (bytes / slowest thread), median of {}",
                    o.per_thread_mib, pages, threads, o.samples
                );
                for g in groups(&toks, regime) {
                    if g.variants.is_empty()
                        || o.only
                            .as_ref()
                            .is_some_and(|ids| !ids.iter().any(|x| x == g.id))
                    {
                        continue;
                    }
                    println!("\n== {} | {}", g.id, g.title);
                    let names: Vec<&str> = g.variants.iter().map(|v| v.name).collect();
                    let t_start = Instant::now();
                    let mut reports = Vec::new();
                    let per_t: Vec<Vec<Stat>> = threads
                        .iter()
                        .map(|&t| {
                            let cpus = &load_cpus[..load_cpus.len().min(t)];
                            let (s, r) = with_load(load, cpus, || {
                                run_mt(&g, &regions, &test_cpus, t, o.samples)
                            });
                            reports.extend(r.map(|r| (t, r)));
                            s
                        })
                        .collect();
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
                            writeln!(
                                csv,
                                "dram,{},{},{},{},{:.3},{:.3},{:.3},{},{}",
                                g.id, name, t, pages, s.med, s.min, s.max, o.samples, load_tag
                            )
                            .unwrap();
                        }
                        println!("{line}");
                    }
                    spreads.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    println!(
                        "   spread (max-min)/median: typical {:.1}%, worst {:.1}% ({} @ {}T); group took {:.0} s",
                        100.0 * spreads[spreads.len() / 2],
                        100.0 * worst.0,
                        worst.1,
                        worst.2,
                        t_start.elapsed().as_secs_f64()
                    );
                    if let Some((kind, duty)) = load {
                        println!("{}", load_line(kind, duty, &reports));
                    }
                }
                drop(regions);
            }
        }
    }
    println!("\nfull per-cell [min..max]: {}", o.csv);
}
