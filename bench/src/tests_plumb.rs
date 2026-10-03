//! Correctness of the plumb_lines kernels (lines.rs) and the ported TMR tests (tmrport.rs).
//! The view-based kernels take any `&[u64]`, so they're tested on misaligned sub-slices too.

use crate::common::*;
use crate::{lines, tmrport};
use fearless_simd::{Avx2, Avx512, Level};
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
    Some(T {
        t2: l.as_avx2()?,
        t5: l.as_avx512(),
        cf: Clflushopt::try_new()?,
        md: Movdir64b::try_new(),
        tmr512: is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512cd")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl"),
    })
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

const OFFSETS: [usize; 5] = [0, 1, 3, 7, 9];
const LENS: [usize; 5] = [0, 5, 64, 4096 + 13, 8192];

#[test]
fn view_fill_and_verify_any_alignment() {
    let Some(t) = toks() else { return };
    let mut fills: Vec<(&str, Writer)> = vec![
        ("pv_128", Box::new(move |b| lines::k_fill_pv_128(t.t2, b))),
        ("pv_256", Box::new(move |b| lines::k_fill_pv_256(t.t2, b))),
    ];
    let mut verifies: Vec<(&str, Reader)> = vec![
        ("pv_128", Box::new(move |b| lines::k_verify4_pv_128(t.t2, b))),
        ("pv_256", Box::new(move |b| lines::k_verify4_pv_256(t.t2, b))),
    ];
    if let Some(t5) = t.t5 {
        fills.push(("pv_512", Box::new(move |b| lines::k_fill_pv_512(t5, b))));
        verifies.push(("pv_512", Box::new(move |b| lines::k_verify4_pv_512(t5, b))));
    }
    let mut store = buf(8192 + 32);
    let all = words(&mut store);
    for off in OFFSETS {
        for len in LENS {
            let b = &mut all[off..off + len];
            for (name, f) in &fills {
                b.fill(0x0123_4567_89AB_CDEF);
                f(b);
                assert!(b.iter().all(|&w| w == PATTERN), "fill {name} off {off} len {len}");
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
fn view_positional_nt_and_verify_any_alignment() {
    let Some(t) = toks() else { return };
    let mut writes: Vec<(&str, Writer)> = vec![
        ("pl_128", Box::new(move |b| lines::k_ntw_pl_128(t.t2, b))),
        ("pl_256", Box::new(move |b| lines::k_ntw_pl_256(t.t2, b))),
    ];
    let mut verifies: Vec<(&str, Reader)> = vec![
        ("pv_128", Box::new(move |b| lines::k_posv_pv_128(t.t2, b))),
        ("pv_256", Box::new(move |b| lines::k_posv_pv_256(t.t2, b))),
    ];
    if let Some(t5) = t.t5 {
        writes.push(("pl_512", Box::new(move |b| lines::k_ntw_pl_512(t5, b))));
        verifies.push(("pv_512", Box::new(move |b| lines::k_posv_pv_512(t5, b))));
    }
    let mut store = buf(8192 + 32);
    let all = words(&mut store);
    for off in OFFSETS {
        for len in LENS {
            let b = &mut all[off..off + len];
            for (name, w) in &writes {
                b.fill(0);
                w(b);
                assert!(b.iter().enumerate().all(|(i, &x)| x == (i as u64) ^ POS_BASE), "ntw {name} off {off} len {len}");
            }
            for (name, v) in &verifies {
                assert_eq!(v(b), 0, "posv {name} false positive off {off} len {len}");
                for i in [0, len / 2, len.saturating_sub(1)] {
                    if i >= len {
                        continue;
                    }
                    b[i] ^= 1;
                    assert_eq!(v(b), 1, "posv {name} missed word {i} off {off} len {len}");
                    b[i] ^= 1;
                }
            }
        }
    }
}

#[test]
fn flush_and_direct_kernels() {
    let Some(t) = toks() else { return };
    let mut store = buf(16384);
    let b = words(&mut store);
    lines::k_fillflush_pl_256(t.t2, t.cf, b);
    assert!(b.iter().all(|&w| w == PATTERN));
    b.fill(0);
    lines::k_wflush_pltok_256(t.t2, t.cf, b);
    assert!(b.iter().all(|&w| w == PATTERN));
    b.fill(0);
    lines::k_wflush_plentry_256(t.t2, t.cf, b);
    assert!(b.iter().all(|&w| w == PATTERN));
    if let Some(t5) = t.t5 {
        for f in [lines::k_fillflush_pl_512 as fn(Avx512, Clflushopt, &mut [u64]), lines::k_wflush_pltok_512, lines::k_wflush_plentry_512] {
            b.fill(0);
            f(t5, t.cf, b);
            assert!(b.iter().all(|&w| w == PATTERN));
        }
    }
    lines::k_flush_pl(t.cf, b);
    assert!(b.iter().all(|&w| w == PATTERN));
    if t.tmr512 {
        b.fill(0);
        lines::k_fillflush_tmr_512(b);
        assert!(b.iter().all(|&w| w == PATTERN));
    }
    let Some(md) = t.md else { return eprintln!("skip MOVDIR64B kernels") };
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

/// Run a ported test in both styles with the same injected faults; both must report the same
/// error count (the number of injections) and leave the same memory.
fn check_port(
    name: &str,
    n_u64: usize,
    calls_per_chunk: usize,
    tmr: Port,
    pl: Port,
) {
    let mut a = buf(n_u64);
    let mut b = buf(n_u64);
    let chunks = n_u64.div_ceil(tmrport::CHUNK_U64);
    let total_calls = chunks * calls_per_chunk;
    // Clean run: no errors.
    assert_eq!(tmr(words(&mut a), &mut |_| {}), 0, "{name}: tmr false positive");
    assert_eq!(pl(words(&mut b), &mut |_| {}), 0, "{name}: plumb false positive");
    assert_eq!(words(&mut a), words(&mut b), "{name}: styles left different memory");
    // Faults at chosen (chunk, phase) calls, at varied positions and bits.
    let targets: Vec<usize> = [0, 1, total_calls / 2, total_calls - 1].into_iter().collect();
    let inject = |hits: &mut usize, c: &mut [u64]| {
        let k = *hits;
        *hits += 1;
        if targets.contains(&k) {
            let i = (k * 7919) % c.len();
            c[i] ^= 1u64 << (k % 64);
        }
    };
    let (mut ha, mut hb) = (0, 0);
    let ea = tmr(words(&mut a), &mut |c| inject(&mut ha, c));
    let eb = pl(words(&mut b), &mut |c| inject(&mut hb, c));
    assert_eq!(ha, total_calls, "{name}: tmr inject calls");
    assert_eq!(hb, total_calls, "{name}: plumb inject calls");
    let want = targets.iter().collect::<std::collections::BTreeSet<_>>().len() as u64;
    assert_eq!(ea, want, "{name}: tmr errors");
    assert_eq!(eb, want, "{name}: plumb errors");
    assert_eq!(words(&mut a), words(&mut b), "{name}: styles left different memory after faults");
}

#[test]
fn ported_tests_agree_and_catch_faults() {
    let Some(t) = toks() else { return };
    let (t2, cf) = (t.t2, t.cf);
    // 3.5 chunks: three full and one partial.
    let n = tmrport::CHUNK_U64 * 7 / 2;
    // SAFETY (all tmr closures): aligned whole buffers; features checked by toks().
    check_port("sb_256", n, 3,
        &|b, f| unsafe { tmrport::sb_tmr_256(b.as_mut_ptr(), b.len(), true, f) },
        &|b, f| tmrport::stuckbit_pl::<Avx2, fearless_simd::u64x4<Avx2>, _>(t2, cf, b, true, f));
    check_port("sbnf_256", n, 3,
        &|b, f| unsafe { tmrport::sb_tmr_256(b.as_mut_ptr(), b.len(), false, f) },
        &|b, f| tmrport::stuckbit_pl::<Avx2, fearless_simd::u64x4<Avx2>, _>(t2, cf, b, false, f));
    check_port("refresh_256", n, 1,
        &|b, f| unsafe { tmrport::refresh_tmr_256(b.as_mut_ptr(), b.len(), true, f) },
        &|b, f| tmrport::refresh_pl::<Avx2, fearless_simd::u64x4<Avx2>, _>(t2, cf, b, true, f));
    check_port("simplent_256", n, 1,
        &|b, f| unsafe { tmrport::simplent_tmr_256(b.as_mut_ptr(), b.len(), f) },
        &|b, f| tmrport::simplent_pl::<Avx2, fearless_simd::u64x4<Avx2>, _>(t2, b, f));
    if let (Some(t5), true) = (t.t5, t.tmr512) {
        check_port("sb_512", n, 3,
            &|b, f| unsafe { tmrport::sb_tmr_512(b.as_mut_ptr(), b.len(), true, f) },
            &|b, f| tmrport::stuckbit_pl::<Avx512, fearless_simd::u64x8<Avx512>, _>(t5, cf, b, true, f));
        check_port("refresh_512", n, 1,
            &|b, f| unsafe { tmrport::refresh_tmr_512(b.as_mut_ptr(), b.len(), true, f) },
            &|b, f| tmrport::refresh_pl::<Avx512, fearless_simd::u64x8<Avx512>, _>(t5, cf, b, true, f));
        check_port("simplent_512", n, 1,
            &|b, f| unsafe { tmrport::simplent_tmr_512(b.as_mut_ptr(), b.len(), f) },
            &|b, f| tmrport::simplent_pl::<Avx512, fearless_simd::u64x8<Avx512>, _>(t5, b, f));
    }
}
