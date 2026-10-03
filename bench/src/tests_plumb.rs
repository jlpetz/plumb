// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Correctness of the plumb_lines kernels (lines.rs) and the ported TMR tests (tmrport.rs).
//! The view-based kernels take any `&[u64]`, so they're tested on misaligned sub-slices too.

use crate::common::*;
use crate::tmrport::{CHUNK_U64, REFRESH_PATTERN, STUCKBIT_P1, STUCKBIT_P2, VERIFY_REPS, WRC};
use crate::{lines, tmrport};
use fearless_simd::{Avx2, Avx512, Level, u64x4, u64x8};
use plumb_lines::{Clflushopt, Line, Movdir64b};

struct T {
    t2: Avx2,
    t5: Option<Avx512>,
    cf: Clflushopt,
    md: Option<Movdir64b>,
    tmr512: bool,
}

fn toks() -> Option<T> {
    let l = Level::new();
    let t = T {
        t2: l.as_avx2()?,
        t5: l.as_avx512(),
        cf: Clflushopt::try_new()?,
        md: Movdir64b::try_new(),
        tmr512: is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512cd")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl"),
    };
    if t.t5.is_none() || !t.tmr512 {
        eprintln!("skip: no AVX-512 here; the 512-bit cases are not run");
    }
    Some(t)
}

fn skip_all() {
    eprintln!("skip: needs AVX2 + CLFLUSHOPT");
}

/// A 64-byte-aligned buffer of `n` u64.
fn buf(n: usize) -> Vec<Line> {
    vec![Line([0; 8]); n.div_ceil(8)]
}
fn words(v: &mut [Line]) -> &mut [u64] {
    // SAFETY: Line is [u64; 8].
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u64, v.len() * 8) }
}

type Writer<'a> = Box<dyn Fn(&mut [u64]) + 'a>;
type Reader<'a> = Box<dyn Fn(&[u64]) -> u64 + 'a>;
type Port<'a> = &'a dyn Fn(&mut [u64], &mut dyn FnMut(&mut [u64])) -> u64;
type Entry<'a> = Box<dyn Fn(&mut [u64]) -> u64 + 'a>;

const OFFSETS: [usize; 5] = [0, 1, 3, 7, 9];
const LENS: [usize; 5] = [0, 5, 64, 4096 + 13, 8192];

#[test]
fn view_fill_and_verify_any_alignment() {
    let Some(t) = toks() else { return skip_all() };
    let mut fills: Vec<(&str, Writer)> = vec![
        ("pv_128", Box::new(move |b| lines::k_fill_pv_128(t.t2, b))),
        ("pv_256", Box::new(move |b| lines::k_fill_pv_256(t.t2, b))),
    ];
    let mut verifies: Vec<(&str, Reader)> = vec![
        (
            "pv_128",
            Box::new(move |b| lines::k_verify4_pv_128(t.t2, b)),
        ),
        (
            "pv_256",
            Box::new(move |b| lines::k_verify4_pv_256(t.t2, b)),
        ),
    ];
    if let Some(t5) = t.t5 {
        fills.push(("pv_512", Box::new(move |b| lines::k_fill_pv_512(t5, b))));
        verifies.push(("pv_512", Box::new(move |b| lines::k_verify4_pv_512(t5, b))));
        verifies.push((
            "pvch_512",
            Box::new(move |b| lines::k_verify4_pvch_512(t5, b)),
        ));
        verifies.push(("pv8_512", Box::new(move |b| lines::k_verify8_pv_512(t5, b))));
    }
    verifies.push((
        "pvpat_128",
        Box::new(move |b| lines::k_verify4_pvpat_128(t.t2, b)),
    ));
    verifies.push((
        "pvpat_256",
        Box::new(move |b| lines::k_verify4_pvpat_256(t.t2, b)),
    ));
    let mut store = buf(8192 + 32);
    let all = words(&mut store);
    for off in OFFSETS {
        for len in LENS {
            let b = &mut all[off..off + len];
            for (name, f) in &fills {
                b.fill(0x0123_4567_89AB_CDEF);
                f(b);
                assert!(
                    b.iter().all(|&w| w == PATTERN),
                    "fill {name} off {off} len {len}"
                );
            }
            for (name, v) in &verifies {
                assert_eq!(v(b), 0, "verify {name} false positive off {off} len {len}");
                for i in [0, len / 2, len.saturating_sub(1), len.saturating_sub(3)] {
                    if i >= len {
                        continue;
                    }
                    b[i] ^= 1 << 41;
                    assert_eq!(v(b), 1, "verify {name} missed word {i} off {off} len {len}");
                    b[i] ^= 1 << 41;
                }
            }
        }
    }
}

#[test]
fn view_positional_nt_and_verify_any_alignment_and_start() {
    let Some(t) = toks() else { return skip_all() };
    let t2 = t.t2;
    let mut store = buf(8192 + 32);
    let all = words(&mut store);
    for start in [0u64, 1, 12345, 1 << 40] {
        let mut writes: Vec<(&str, Writer)> = vec![
            (
                "pl_128",
                Box::new(move |b| {
                    lines::ntw_scope_at::<Avx2, fearless_simd::u64x2<Avx2>>(t2, b, POS_BASE, start)
                }),
            ),
            (
                "pl_256",
                Box::new(move |b| lines::ntw_scope_at::<Avx2, u64x4<Avx2>>(t2, b, POS_BASE, start)),
            ),
        ];
        let mut verifies: Vec<(&str, Reader)> = vec![
            (
                "pv_128",
                Box::new(move |b| {
                    lines::pos_verify_view_at::<Avx2, fearless_simd::u64x2<Avx2>>(
                        t2, b, POS_BASE, start,
                    )
                }),
            ),
            (
                "pv_256",
                Box::new(move |b| {
                    lines::pos_verify_view_at::<Avx2, u64x4<Avx2>>(t2, b, POS_BASE, start)
                }),
            ),
        ];
        if let Some(t5) = t.t5 {
            writes.push((
                "pl_512",
                Box::new(move |b| {
                    lines::ntw_scope_at::<Avx512, u64x8<Avx512>>(t5, b, POS_BASE, start)
                }),
            ));
            verifies.push((
                "pv_512",
                Box::new(move |b| {
                    lines::pos_verify_view_at::<Avx512, u64x8<Avx512>>(t5, b, POS_BASE, start)
                }),
            ));
        }
        for off in OFFSETS {
            for len in LENS {
                let b = &mut all[off..off + len];
                for (name, w) in &writes {
                    b.fill(0);
                    w(b);
                    assert!(
                        b.iter()
                            .enumerate()
                            .all(|(i, &x)| x == (start + i as u64) ^ POS_BASE),
                        "ntw {name} start {start} off {off} len {len}"
                    );
                }
                for (name, v) in &verifies {
                    assert_eq!(
                        v(b),
                        0,
                        "posv {name} false positive start {start} off {off} len {len}"
                    );
                    for i in [0, len / 2, len.saturating_sub(1)] {
                        if i >= len {
                            continue;
                        }
                        b[i] ^= 1 << 63;
                        assert_eq!(
                            v(b),
                            1,
                            "posv {name} missed word {i} start {start} off {off} len {len}"
                        );
                        b[i] ^= 1 << 63;
                    }
                }
            }
        }
    }
    // The named bench entries (start 0, whole aligned buffers).
    let b = &mut all[..4096];
    lines::k_ntw_pl_256(t2, b);
    assert_eq!(lines::k_posv_pv_256(t2, b), 0);
    assert!(b.iter().enumerate().all(|(i, &x)| x == i as u64 ^ POS_BASE));
}

#[test]
fn flush_and_direct_kernels() {
    let Some(t) = toks() else { return skip_all() };
    let mut store = buf(16384 + 16);
    let all = words(&mut store);
    // Whole aligned buffer, then a misaligned one (head/tail partial lines).
    for off in [0usize, 3] {
        let b = &mut all[off..off + 16384];
        let mut ks: Vec<(&str, Writer)> = vec![
            (
                "fillflush_pl_256",
                Box::new(|b| lines::k_fillflush_pl_256(t.t2, t.cf, b)),
            ),
            (
                "wflush_pltok_256",
                Box::new(|b| lines::k_wflush_pltok_256(t.t2, t.cf, b)),
            ),
            (
                "wflush_plentry_256",
                Box::new(|b| lines::k_wflush_plentry_256(t.t2, t.cf, b)),
            ),
        ];
        if let Some(t5) = t.t5 {
            ks.push((
                "fillflush_pl_512",
                Box::new(move |b| lines::k_fillflush_pl_512(t5, t.cf, b)),
            ));
            ks.push((
                "wflush_pltok_512",
                Box::new(move |b| lines::k_wflush_pltok_512(t5, t.cf, b)),
            ));
            ks.push((
                "wflush_plentry_512",
                Box::new(move |b| lines::k_wflush_plentry_512(t5, t.cf, b)),
            ));
        }
        for (name, k) in &ks {
            b.fill(0);
            k(b);
            assert!(b.iter().all(|&w| w == PATTERN), "{name} off {off}");
        }
        lines::k_flush_pl(t.cf, b);
        assert!(b.iter().all(|&w| w == PATTERN));
    }
    let b = &mut all[..16384];
    if t.tmr512 {
        b.fill(0);
        lines::k_fillflush_tmr_512(b);
        assert!(b.iter().all(|&w| w == PATTERN));
    }
    if let Some(t5) = t.t5 {
        b.fill(0);
        lines::k_ntw_plplain_512(t5, b);
        assert!(
            b.iter().enumerate().all(|(i, &x)| x == i as u64 ^ POS_BASE),
            "footgun kernel still correct"
        );
    }
    let Some(md) = t.md else {
        return eprintln!("skip: MOVDIR64B kernels (not on this CPU)");
    };
    b.fill(0);
    lines::k_fillmd_pl(md, b);
    assert!(b.iter().all(|&w| w == PATTERN));
    let half = b.len() / 2;
    for (i, w) in b[..half].iter_mut().enumerate() {
        *w = i as u64 * 7;
    }
    let (src, dst) = b.split_at_mut(half);
    lines::k_copymd_pl(md, dst, src);
    assert_eq!(&*dst, &*src);
}

/// Run a ported test in both styles with the same injected faults. Before each injection the
/// oracle checks that memory holds what TMR's sequence should have written at that point (so a
/// wrong phase pattern fails), then a bit is flipped at a varied position (first word, middle,
/// last word) and bit (0, 31, 63, ...). Both styles must report `injections x per_fault` errors
/// and leave the same memory.
fn check_port(
    name: &str,
    n_u64: usize,
    calls_per_chunk: usize,
    per_fault: u64,
    oracle: &dyn Fn(usize, &[u64]) -> bool,
    tmr: Port,
    pl: Port,
) {
    let mut a = buf(n_u64);
    let mut b = buf(n_u64);
    let chunks = n_u64.div_ceil(CHUNK_U64);
    let total_calls = chunks * calls_per_chunk;
    // Clean run, oracle-checked at every call: no errors, same memory.
    let (mut ka, mut kb) = (0usize, 0usize);
    assert_eq!(
        tmr(words(&mut a), &mut |c| {
            assert!(oracle(ka, c), "{name}: tmr memory wrong at call {ka}");
            ka += 1
        }),
        0,
        "{name}: tmr false positive"
    );
    assert_eq!(
        pl(words(&mut b), &mut |c| {
            assert!(oracle(kb, c), "{name}: plumb memory wrong at call {kb}");
            kb += 1
        }),
        0,
        "{name}: plumb false positive"
    );
    assert_eq!(
        (ka, kb),
        (total_calls, total_calls),
        "{name}: inject call counts"
    );
    assert_eq!(
        words(&mut a),
        words(&mut b),
        "{name}: styles left different memory"
    );
    // Faulted run.
    let targets = [0, 1, total_calls / 2, total_calls - 1];
    let inject = |k: &mut usize, c: &mut [u64]| {
        assert!(
            oracle(*k, c),
            "{name}: memory wrong before injection at call {k}"
        );
        if let Some(t) = targets.iter().position(|&x| x == *k) {
            let i = [0, c.len() / 2, c.len() - 1, (*k * 7919) % c.len()][t];
            let bit = [0, 31, 63, *k % 64][t];
            c[i] ^= 1u64 << bit;
        }
        *k += 1;
    };
    let (mut ka, mut kb) = (0, 0);
    let ea = tmr(words(&mut a), &mut |c| inject(&mut ka, c));
    let eb = pl(words(&mut b), &mut |c| inject(&mut kb, c));
    let want = targets
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u64
        * per_fault;
    assert_eq!(ea, want, "{name}: tmr errors");
    assert_eq!(eb, want, "{name}: plumb errors");
    assert_eq!(
        words(&mut a),
        words(&mut b),
        "{name}: styles left different memory after faults"
    );
}

#[test]
fn ported_tests_agree_and_catch_faults() {
    let Some(t) = toks() else { return skip_all() };
    let cf = t.cf;
    // Three full chunks and a partial one of 40 u64 (5 or 10 vectors: not a multiple of 4, so
    // the remainder loops of both styles run).
    let n = CHUNK_U64 * 3 + 40;
    let sb = |k: usize, c: &[u64]| {
        c.iter()
            .all(|&w| w == [STUCKBIT_P1, STUCKBIT_P2, STUCKBIT_P1][k % 3])
    };
    let refresh = |_: usize, c: &[u64]| c.iter().all(|&w| w == REFRESH_PATTERN);
    let simplent = |k: usize, c: &[u64]| {
        let start = (k / WRC * CHUNK_U64) as u64;
        c.iter()
            .enumerate()
            .all(|(i, &w)| w == (start + i as u64) ^ POS_BASE)
    };
    let vr = VERIFY_REPS as u64;
    macro_rules! both {
        ($w:literal, $Tok:ty, $tok:expr, $V:ty, $sb:ident, $refresh:ident, $simplent:ident) => {{
            let tok = $tok;
            for flush in [true, false] {
                // SAFETY (tmr closures): aligned whole buffers, n a multiple of the lanes; features checked.
                check_port(&format!("sb_{}_flush{flush}", $w), n, 3, 1, &sb,
                    &|b, f| unsafe { tmrport::$sb(b.as_mut_ptr(), b.len(), flush, f) },
                    &|b, f| tmrport::stuckbit_pl::<$Tok, $V, _>(tok, cf, b, flush, f));
                check_port(&format!("refresh_{}_flush{flush}", $w), n, 1, 1, &refresh,
                    &|b, f| unsafe { tmrport::$refresh(b.as_mut_ptr(), b.len(), flush, f) },
                    &|b, f| tmrport::refresh_pl::<$Tok, $V, _>(tok, cf, b, flush, f));
            }
            check_port(&format!("simplent_{}", $w), n, WRC, vr, &simplent,
                &|b, f| unsafe { tmrport::$simplent(b.as_mut_ptr(), b.len(), f) },
                &|b, f| tmrport::simplent_pl::<$Tok, $V, _>(tok, b, f));
        }};
    }
    both!(
        "256",
        Avx2,
        t.t2,
        u64x4<Avx2>,
        sb_tmr_256,
        refresh_tmr_256,
        simplent_tmr_256
    );
    if let (Some(t5), true) = (t.t5, t.tmr512) {
        both!(
            "512",
            Avx512,
            t5,
            u64x8<Avx512>,
            sb_tmr_512,
            refresh_tmr_512,
            simplent_tmr_512
        );
    }
}

/// The named bench entries run what they say: right final memory, no errors on clean memory.
/// (k_sb_* and k_sbnf_* share a body and differ only in the flush flag they pass.)
#[test]
fn port_entries_are_wired_right() {
    let Some(t) = toks() else { return skip_all() };
    let n = CHUNK_U64 * 2;
    let mut store = buf(n);
    let b = words(&mut store);
    let positional = |b: &[u64]| b.iter().enumerate().all(|(i, &w)| w == i as u64 ^ POS_BASE);
    let mut cases: Vec<(&str, Entry, u64)> = vec![
        ("k_sb_tmr_256", Box::new(tmrport::k_sb_tmr_256), STUCKBIT_P1),
        (
            "k_sbnf_tmr_256",
            Box::new(tmrport::k_sbnf_tmr_256),
            STUCKBIT_P1,
        ),
        (
            "k_sb_pl_256",
            Box::new(|b| tmrport::k_sb_pl_256(t.t2, t.cf, b)),
            STUCKBIT_P1,
        ),
        (
            "k_sbnf_pl_256",
            Box::new(|b| tmrport::k_sbnf_pl_256(t.t2, t.cf, b)),
            STUCKBIT_P1,
        ),
        (
            "k_refresh_tmr_256",
            Box::new(tmrport::k_refresh_tmr_256),
            REFRESH_PATTERN,
        ),
        (
            "k_refresh_pl_256",
            Box::new(|b| tmrport::k_refresh_pl_256(t.t2, t.cf, b)),
            REFRESH_PATTERN,
        ),
    ];
    if let (Some(t5), true) = (t.t5, t.tmr512) {
        cases.push(("k_sb_tmr_512", Box::new(tmrport::k_sb_tmr_512), STUCKBIT_P1));
        cases.push((
            "k_sbnf_tmr_512",
            Box::new(tmrport::k_sbnf_tmr_512),
            STUCKBIT_P1,
        ));
        cases.push((
            "k_sb_pl_512",
            Box::new(move |b| tmrport::k_sb_pl_512(t5, t.cf, b)),
            STUCKBIT_P1,
        ));
        cases.push((
            "k_sbnf_pl_512",
            Box::new(move |b| tmrport::k_sbnf_pl_512(t5, t.cf, b)),
            STUCKBIT_P1,
        ));
        cases.push((
            "k_sb_plplain_512",
            Box::new(move |b| tmrport::k_sb_plplain_512(t5, t.cf, b)),
            STUCKBIT_P1,
        ));
        cases.push((
            "k_sbnf_plplain_512",
            Box::new(move |b| tmrport::k_sbnf_plplain_512(t5, t.cf, b)),
            STUCKBIT_P1,
        ));
        cases.push((
            "k_refresh_tmr_512",
            Box::new(tmrport::k_refresh_tmr_512),
            REFRESH_PATTERN,
        ));
        cases.push((
            "k_refresh_pl_512",
            Box::new(move |b| tmrport::k_refresh_pl_512(t5, t.cf, b)),
            REFRESH_PATTERN,
        ));
    }
    for (name, k, want) in &cases {
        b.fill(0);
        assert_eq!(k(b), 0, "{name}: errors on clean memory");
        assert!(b.iter().all(|&w| w == *want), "{name}: final memory");
    }
    let mut nt: Vec<(&str, Entry)> = vec![
        ("k_simplent_tmr_256", Box::new(tmrport::k_simplent_tmr_256)),
        (
            "k_simplent_pl_256",
            Box::new(|b| tmrport::k_simplent_pl_256(t.t2, b)),
        ),
    ];
    if let (Some(t5), true) = (t.t5, t.tmr512) {
        nt.push(("k_simplent_tmr_512", Box::new(tmrport::k_simplent_tmr_512)));
        nt.push((
            "k_simplent_pl_512",
            Box::new(move |b| tmrport::k_simplent_pl_512(t5, b)),
        ));
    }
    for (name, k) in &nt {
        b.fill(0);
        assert_eq!(k(b), 0, "{name}: errors on clean memory");
        assert!(positional(b), "{name}: final memory");
    }
}
