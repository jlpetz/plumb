//! Port-correctness: every fearless_simd kernel must produce exactly what its TMR-style twin
//! produces, and every verify must catch a single flipped bit anywhere, including the tail
//! loops. Fast numbers from a wrong kernel would be worthless, so these run before any timing.
//!
//! Lengths are multiples of 8 u64 (one 512-bit vector, as TMR's chunks are) but not of 32,
//! so the 4-accumulator remainder loops run too.

use crate::common::*;
use crate::{fs, tmr};
use fearless_simd::{Avx2, Avx512, Level};

const LENS: [usize; 4] = [8, 4096 + 8, 4096 + 24, 8192];

struct T {
    t2: Avx2,
    t5: Option<Avx512>,
    tmr512: bool,
    level: Level,
    cpu: Cpu,
}

fn toks() -> Option<T> {
    let cpu = detect();
    if !(cpu.avx2 && cpu.clflushopt) {
        eprintln!("skipping: needs AVX2 + CLFLUSHOPT");
        return None;
    }
    let level = Level::new();
    Some(T {
        t2: level.as_avx2()?,
        t5: level.as_avx512(),
        tmr512: cpu.avx512f
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512cd")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl"),
        level,
        cpu,
    })
}

/// Two independent page-aligned buffers of `n` u64 (prefix of a 64 KiB allocation).
fn bufs() -> (AlignedBuf, AlignedBuf) {
    (AlignedBuf::new(64 << 10), AlignedBuf::new(64 << 10))
}

type W<'a> = Box<dyn Fn(&mut [u64]) + 'a>;
type R<'a> = Box<dyn Fn(&[u64]) -> u64 + 'a>;

fn tw(f: unsafe fn(*mut u64, usize)) -> impl Fn(&mut [u64]) {
    // SAFETY: valid aligned slice; features checked in toks().
    move |b: &mut [u64]| unsafe { f(b.as_mut_ptr(), b.len()) }
}
fn tr(f: unsafe fn(*const u64, usize) -> u64) -> impl Fn(&[u64]) -> u64 {
    // SAFETY: as above.
    move |b: &[u64]| unsafe { f(b.as_ptr(), b.len()) }
}

/// Run every writer on a fresh buffer and require identical output to `reference(i)`.
fn check_writers(name: &str, writers: Vec<(&str, W)>, reference: impl Fn(usize) -> u64) {
    let (mut a, _) = bufs();
    for n in LENS {
        for (wname, w) in &writers {
            let buf = &mut a.as_mut_slice()[..n];
            buf.fill(0x0123_4567_89AB_CDEF);
            w(buf);
            for (i, &x) in buf.iter().enumerate() {
                assert_eq!(x, reference(i), "{name}/{wname}: n={n} word {i}");
            }
        }
    }
}

/// Every verifier returns 0 on `fill`ed data and 1 with any single bit flipped.
fn check_verifiers(name: &str, fill: &dyn Fn(&mut [u64]), verifiers: Vec<(&str, R)>) {
    let (mut a, _) = bufs();
    for n in LENS {
        let buf = &mut a.as_mut_slice()[..n];
        fill(buf);
        for (vname, v) in &verifiers {
            assert_eq!(v(buf), 0, "{name}/{vname}: false positive on clean data, n={n}");
            for &i in &[0, n / 2, n - 1, n.saturating_sub(9), n.saturating_sub(17)] {
                for bit in [0u32, 37, 63] {
                    buf[i] ^= 1 << bit;
                    assert_eq!(v(buf), 1, "{name}/{vname}: missed flip at word {i} bit {bit}, n={n}");
                    buf[i] ^= 1 << bit;
                }
            }
        }
    }
}

#[test]
fn fill_matches() {
    let Some(t) = toks() else { return };
    let (t2, lvl) = (t.t2, t.level);
    let mut ws: Vec<(&str, W)> = vec![
        ("tmr_128", Box::new(tw(tmr::k_fill_tmr_128))),
        ("tmr_256", Box::new(tw(tmr::k_fill_tmr_256))),
        ("fs_128", Box::new(move |b| fs::k_fill_fs_128(t2, b))),
        ("fs_256", Box::new(move |b| fs::k_fill_fs_256(t2, b))),
        ("fs_auto", Box::new(move |b| fs::k_fill_fs_auto(lvl, b))),
        ("wflush_tmr_256", Box::new(tw(tmr::k_wflush_tmr_256))),
        ("wflush_fsintr_256", Box::new(move |b| fs::k_wflush_fsintr_256(t2, b))),
        ("wflush_fsasm_256", Box::new(move |b| fs::k_wflush_fsasm_256(t2, b))),
    ];
    if t.tmr512 {
        ws.push(("tmr_512", Box::new(tw(tmr::k_fill_tmr_512))));
        ws.push(("wflush_tmr_512", Box::new(tw(tmr::k_wflush_tmr_512))));
    }
    if let Some(t5) = t.t5 {
        ws.push(("fs_512", Box::new(move |b| fs::k_fill_fs_512(t5, b))));
        ws.push(("wflush_fsintr_512", Box::new(move |b| fs::k_wflush_fsintr_512(t5, b))));
        ws.push(("wflush_fsasm_512", Box::new(move |b| fs::k_wflush_fsasm_512(t5, b))));
    }
    check_writers("fill", ws, |_| PATTERN);

    let mut ws: Vec<(&str, W)> = vec![
        ("tmr_256", Box::new(tw(tmr::k_filluni_tmr_256))),
        ("fs_256", Box::new(move |b| fs::k_filluni_fs_256(t2, b))),
    ];
    if let Some(t5) = t.t5 {
        ws.push(("fs_512", Box::new(move |b| fs::k_filluni_fs_512(t5, b))));
    }
    check_writers("filluni", ws, |_| UNIFORM);
}

#[test]
fn verify4_matches() {
    let Some(t) = toks() else { return };
    let (t2, lvl) = (t.t2, t.level);
    let mut vs: Vec<(&str, R)> = vec![
        ("tmr_128", Box::new(tr(tmr::k_verify4_tmr_128))),
        ("tmr_256", Box::new(tr(tmr::k_verify4_tmr_256))),
        ("fs_128", Box::new(move |b| fs::k_verify4_fs_128(t2, b))),
        ("fs_256", Box::new(move |b| fs::k_verify4_fs_256(t2, b))),
        ("fs_auto", Box::new(move |b| fs::k_verify4_fs_auto(lvl, b))),
        ("pfv_tmr_128", Box::new(tr(tmr::k_pfv_tmr_128))),
        ("pfv_tmr_256", Box::new(tr(tmr::k_pfv_tmr_256))),
        ("pfv_fs_256", Box::new(move |b| fs::k_pfv_fs_256(t2, b))),
        ("fssplit_128", Box::new(move |b| fs::k_verify4_fssplit_128(t2, b))),
        ("fssplit_256", Box::new(move |b| fs::k_verify4_fssplit_256(t2, b))),
        ("fsptr_128", Box::new(move |b| fs::k_verify4_fsptr_128(t2, b))),
        ("fsptr_256", Box::new(move |b| fs::k_verify4_fsptr_256(t2, b))),
        ("pfv_fsptr_256", Box::new(move |b| fs::k_pfv_fsptr_256(t2, b))),
    ];
    if t.tmr512 {
        vs.push(("tmr_512", Box::new(tr(tmr::k_verify4_tmr_512))));
        vs.push(("pfv_tmr_512", Box::new(tr(tmr::k_pfv_tmr_512))));
    }
    if let Some(t5) = t.t5 {
        vs.push(("fs_512", Box::new(move |b| fs::k_verify4_fs_512(t5, b))));
        vs.push(("pfv_fs_512", Box::new(move |b| fs::k_pfv_fs_512(t5, b))));
        vs.push(("fssplit_512", Box::new(move |b| fs::k_verify4_fssplit_512(t5, b))));
        vs.push(("fsptr_512", Box::new(move |b| fs::k_verify4_fsptr_512(t5, b))));
        vs.push(("pfv_fsptr_512", Box::new(move |b| fs::k_pfv_fsptr_512(t5, b))));
    }
    check_verifiers("verify4", &|b: &mut [u64]| b.fill(PATTERN), vs);
}

/// The footgun variants are slow, not wrong. Their remainder handling differs (helpers skip
/// the tail), so they get whole-quad lengths only.
#[test]
fn helper_footguns_are_correct() {
    let Some(t) = toks() else { return };
    let (Some(t5), true) = (t.t5, t.tmr512) else { return };
    let (mut a, _) = bufs();
    let buf = &mut a.as_mut_slice()[..4096];
    buf.fill(PATTERN);
    // SAFETY: valid aligned slice; AVX-512 checked.
    let tmr = |b: &[u64]| unsafe { tmr::k_verify4_tmrhelper_512(b.as_ptr(), b.len()) };
    assert_eq!(tmr(buf), 0);
    assert_eq!(fs::k_verify4_fshelper_512(t5, buf), 0);
    buf[1234] ^= 1 << 17;
    assert_eq!(tmr(buf), 1);
    assert_eq!(fs::k_verify4_fshelper_512(t5, buf), 1);
}

#[test]
fn positional_matches() {
    let Some(t) = toks() else { return };
    let t2 = t.t2;
    let mut ws: Vec<(&str, W)> = vec![
        ("tmr_128", Box::new(tw(tmr::k_posw_tmr_128))),
        ("tmr_256", Box::new(tw(tmr::k_posw_tmr_256))),
        ("fs_128", Box::new(move |b| fs::k_posw_fs_128(t2, b))),
        ("fs_256", Box::new(move |b| fs::k_posw_fs_256(t2, b))),
        ("ntw_tmr_128", Box::new(tw(tmr::k_ntw_tmr_128))),
        ("ntw_tmr_256", Box::new(tw(tmr::k_ntw_tmr_256))),
        ("ntw_fs_128", Box::new(move |b| fs::k_ntw_fs_128(t2, b))),
        ("ntw_fs_256", Box::new(move |b| fs::k_ntw_fs_256(t2, b))),
    ];
    if t.tmr512 {
        ws.push(("tmr_512", Box::new(tw(tmr::k_posw_tmr_512))));
        ws.push(("ntw_tmr_512", Box::new(tw(tmr::k_ntw_tmr_512))));
    }
    if let Some(t5) = t.t5 {
        ws.push(("fs_512", Box::new(move |b| fs::k_posw_fs_512(t5, b))));
        ws.push(("ntw_fs_512", Box::new(move |b| fs::k_ntw_fs_512(t5, b))));
        ws.push(("ntw_fsk_512", Box::new(move |b| fs::k_ntw_fsk_512(t5, b))));
    }
    check_writers("posw", ws, |i| (i as u64) ^ POS_BASE);

    let mut vs: Vec<(&str, R)> = vec![
        ("tmr_128", Box::new(tr(tmr::k_posv_tmr_128))),
        ("tmr_256", Box::new(tr(tmr::k_posv_tmr_256))),
        ("fs_128", Box::new(move |b| fs::k_posv_fs_128(t2, b))),
        ("fs_256", Box::new(move |b| fs::k_posv_fs_256(t2, b))),
    ];
    if t.tmr512 {
        vs.push(("tmr_512", Box::new(tr(tmr::k_posv_tmr_512))));
    }
    if let Some(t5) = t.t5 {
        vs.push(("fs_512", Box::new(move |b| fs::k_posv_fs_512(t5, b))));
    }
    let fill = |b: &mut [u64]| {
        for (i, x) in b.iter_mut().enumerate() {
            *x = (i as u64) ^ POS_BASE;
        }
    };
    check_verifiers("posv", &fill, vs);
}

/// Every width must reproduce the plain sequential LCG: lane j of vector k is step k*L + j.
#[test]
fn lcg_matches_scalar_stream() {
    let Some(t) = toks() else { return };
    let t2 = t.t2;
    let mut ws: Vec<(&str, W)> = vec![
        ("tmr_128", Box::new(tw(tmr::k_lcgw_tmr_128))),
        ("tmr_256", Box::new(tw(tmr::k_lcgw_tmr_256))),
        ("fs_128", Box::new(move |b| fs::k_lcgw_fs_128(t2, b))),
        ("fs_256", Box::new(move |b| fs::k_lcgw_fs_256(t2, b))),
    ];
    if t.tmr512 {
        ws.push(("tmr_512", Box::new(tw(tmr::k_lcgw_tmr_512))));
    }
    if let Some(t5) = t.t5 {
        ws.push(("fs_512", Box::new(move |b| fs::k_lcgw_fs_512(t5, b))));
    }
    let mut stream = vec![0u64; 8192];
    let mut s = LCG_SEED;
    for x in stream.iter_mut() {
        *x = s;
        s = lcg_next(s, LCG_MUL, LCG_ADD);
    }
    check_writers("lcgw", ws, |i| stream[i]);
}

#[test]
fn flush_preserves_contents() {
    let Some(t) = toks() else { return };
    let t2 = t.t2;
    let (mut a, _) = bufs();
    let buf = a.as_mut_slice();
    for (i, x) in buf.iter_mut().enumerate() {
        *x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    let want: Vec<u64> = buf.to_vec();
    // SAFETY: valid slice; CLFLUSHOPT checked.
    unsafe { tmr::k_flush_tmr(buf.as_ptr(), buf.len()) };
    fs::k_flush_fsintr(t2, buf);
    fs::k_flush_fsasm(t2, buf);
    assert_eq!(buf, &want[..]);
}

#[test]
fn copies_match() {
    let Some(t) = toks() else { return };
    let (mut a, mut b) = bufs();
    let src = a.as_mut_slice();
    for (i, x) in src.iter_mut().enumerate() {
        *x = (i as u64) ^ 0x5A5A_0F0F_3C3C_9696;
    }
    let src = &*src;
    let mut run = |name: &str, f: &dyn Fn(&mut [u64], &[u64])| {
        let dst = b.as_mut_slice();
        dst.fill(0);
        f(dst, src);
        assert_eq!(dst, src, "copy {name}");
    };
    if t.tmr512 {
        // SAFETY: equal-length aligned buffers; AVX-512 checked.
        run("nt_tmr_512", &|d, s| unsafe { tmr::k_copynt_tmr_512(d.as_mut_ptr(), s.as_ptr(), s.len()) });
    }
    if let Some(t5) = t.t5 {
        run("nt_fs_512", &|d, s| fs::k_copynt_fs_512(t5, d, s));
    }
    if t.cpu.movdir64b {
        // SAFETY: as above; MOVDIR64B checked.
        run("md_tmr", &|d, s| unsafe { tmr::k_copymd_tmr(d.as_mut_ptr(), s.as_ptr(), s.len()) });
        let t2 = t.t2;
        run("md_fs", &|d, s| fs::k_copymd_fs(t2, d, s));
    } else {
        eprintln!("MOVDIR64B absent: copy_movdir not tested");
    }
}

/// Read-only: what this machine offers (CPUID + XGETBV; executes no AMX/MOVDIR64B instruction).
#[test]
fn report_cpu() {
    let c = detect();
    eprintln!("cpu: {c:?}");
    eprintln!("fearless_simd Level::new() = {:?}", Level::new());
}

/// Capability-token prototype (`cap.rs`): same output as the other write+flush variants.
#[test]
fn cap_clflushopt_matches() {
    let Some(t) = toks() else { return };
    let cf = crate::cap::Clflushopt::try_new().expect("CLFLUSHOPT checked in toks()");
    let t2 = t.t2;
    let mut ws: Vec<(&str, W)> = vec![
        ("fscap_256", Box::new(move |b| crate::cap::k_wflush_fscap_256(t2, cf, b, PATTERN))),
    ];
    if let Some(t5) = t.t5 {
        ws.push(("fscap_512", Box::new(move |b| crate::cap::k_wflush_fscap_512(t5, cf, b, PATTERN))));
    }
    check_writers("wflush_cap", ws, |_| PATTERN);
    let (mut a, _) = bufs();
    let buf = a.as_mut_slice();
    buf.fill(PATTERN);
    crate::cap::k_flush_fscap(t2, cf, buf);
    assert!(buf.iter().all(|&x| x == PATTERN));
}

/// Guard for `cap.rs`'s copied feature lists: the macro literals equal the documented consts,
/// and every feature they enable is detected whenever the matching fearless token exists.
/// (`clflushopt` itself is checked by CPUID in `Clflushopt::try_new`; std_detect lacks it.)
#[test]
fn level_features_are_detected() {
    use crate::cap::{AVX2_CLFLUSHOPT, AVX512_CLFLUSHOPT};
    let src = include_str!("cap.rs");
    for list in [AVX2_CLFLUSHOPT, AVX512_CLFLUSHOPT] {
        assert!(src.contains(&format!("enable = \"{list}\"")), "macro literal drifted from {list}");
    }
    fn detected(f: &str) -> bool {
        macro_rules! d { ($($n:tt),*) => { match f { $($n => is_x86_feature_detected!($n),)* "clflushopt" => detect().clflushopt, _ => panic!("unknown feature {f}") } } }
        d!("fxsr", "adx", "aes", "avx2", "avx512bitalg", "avx512bw", "avx512cd", "avx512dq", "avx512f",
           "avx512ifma", "avx512vbmi", "avx512vbmi2", "avx512vl", "avx512vnni", "avx512vpopcntdq",
           "bmi1", "bmi2", "cmpxchg16b", "f16c", "fma", "gfni", "lzcnt", "movbe", "pclmulqdq",
           "popcnt", "rdrand", "rdseed", "sha", "vaes", "vpclmulqdq", "xsave", "xsavec", "xsaveopt", "xsaves")
    }
    let level = Level::new();
    if level.as_avx2().is_some() {
        for f in AVX2_CLFLUSHOPT.split(',') {
            assert!(detected(f), "Avx2 entry enables {f}, which this CPU lacks");
        }
    }
    if level.as_avx512().is_some() {
        for f in AVX512_CLFLUSHOPT.split(',') {
            assert!(detected(f), "Avx512 entry enables {f}, which this CPU lacks");
        }
    }
}
