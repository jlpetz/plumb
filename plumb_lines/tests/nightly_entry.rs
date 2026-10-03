//! The `with_clflushopt!` entry macro (nightly feature): output, and its copied feature lists.
#![cfg(feature = "nightly")]
#![feature(clflushopt_target_feature, simd_x86_clflushopt)]

use fearless_simd::prelude::*;
use fearless_simd::{Avx2, Avx512, Level, u64x4, u64x8};
use plumb_lines::Clflushopt;
use plumb_lines::entry::{AVX2_FEATURES, AVX512_FEATURES};

/// Write a pattern into each line, then CLFLUSHOPT that line with the intrinsic, in one loop.
#[inline(always)]
fn write_flush<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, _cf: Clflushopt, buf: &mut [u64], pat: u64) {
    let p = V::splat(simd, pat);
    for line in buf.as_chunks_mut::<8>().0 {
        for c in line.chunks_exact_mut(V::LEN) {
            p.store_slice(c);
        }
        // SAFETY: `line` is inside `buf`; `_cf` proves CLFLUSHOPT and the entry enables it.
        unsafe { core::arch::x86_64::_mm_clflushopt(line.as_ptr() as *const u8) };
    }
    plumb_lines::mfence();
}

plumb_lines::with_clflushopt!(avx2, fn entry256 = write_flush::<Avx2, u64x4<Avx2>>, (buf: &mut [u64], pat: u64));
plumb_lines::with_clflushopt!(avx512, fn entry512 = write_flush::<Avx512, u64x8<Avx512>>, (buf: &mut [u64], pat: u64));

#[test]
fn entry_points_write_and_flush() {
    let cf = Clflushopt::try_new().expect("CLFLUSHOPT");
    let l = Level::new();
    let mut buf = vec![0u64; 4096];
    entry256(l.as_avx2().expect("AVX2"), cf, &mut buf, 0xA55A_A55A_A55A_A55A);
    assert!(buf.iter().all(|&w| w == 0xA55A_A55A_A55A_A55A));
    if let Some(t5) = l.as_avx512() {
        entry512(t5, cf, &mut buf, 0x1234_5678_9ABC_DEF0);
        assert!(buf.iter().all(|&w| w == 0x1234_5678_9ABC_DEF0));
    }
}

/// The macro enables these features without checking them, so they must (1) equal the macro
/// text, and (2) all be present whenever the matching fearless token exists.
#[test]
fn copied_feature_lists_are_sound() {
    let src = include_str!("../src/entry.rs");
    for list in [AVX2_FEATURES, AVX512_FEATURES] {
        assert!(src.contains(&format!("enable = \"{list}\"")), "macro text drifted from {list}");
    }
    fn detected(f: &str) -> bool {
        macro_rules! d { ($($n:tt),*) => { match f { $($n => is_x86_feature_detected!($n),)* "clflushopt" => Clflushopt::try_new().is_some(), _ => panic!("unknown feature {f}") } } }
        d!("fxsr", "adx", "aes", "avx2", "avx512bitalg", "avx512bw", "avx512cd", "avx512dq", "avx512f",
           "avx512ifma", "avx512vbmi", "avx512vbmi2", "avx512vl", "avx512vnni", "avx512vpopcntdq",
           "bmi1", "bmi2", "cmpxchg16b", "f16c", "fma", "gfni", "lzcnt", "movbe", "pclmulqdq",
           "popcnt", "rdrand", "rdseed", "sha", "vaes", "vpclmulqdq", "xsave", "xsavec", "xsaveopt", "xsaves")
    }
    let l = Level::new();
    if l.as_avx2().is_some() {
        for f in AVX2_FEATURES.split(',') {
            assert!(detected(f), "Avx2 entry enables {f}, which this CPU lacks");
        }
    }
    if l.as_avx512().is_some() {
        for f in AVX512_FEATURES.split(',') {
            assert!(detected(f), "Avx512 entry enables {f}, which this CPU lacks");
        }
    }
}

/// fearless_simd's own `Avx2`/`Avx512` target-feature lists, read from the source of the
/// fearless_simd 1.0.0 this crate builds against (located with `cargo metadata`).
fn fearless_lists() -> Option<(String, String)> {
    let out = std::process::Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--offline"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    let json = String::from_utf8(out.stdout).ok()?;
    let key = "\"manifest_path\":\"";
    let manifest = json
        .match_indices(key)
        .map(|(i, _)| {
            let rest = &json[i + key.len()..];
            &rest[..rest.find('"').unwrap_or(0)]
        })
        .find(|p| p.contains("fearless_simd-1.0.0"))?
        .replace("\\\\", "\\");
    let src = std::fs::read_to_string(std::path::Path::new(&manifest).parent()?.join("src").join("kernel_macros.rs")).ok()?;
    let body = &src[src.find("macro_rules! __fearless_simd_kernel_target_fn")?..];
    let list = |tok: &str| -> Option<String> {
        let at = body.find(&format!("({tok}, $item:item)"))?;
        let from = at + body[at..].find("enable = \"")? + "enable = \"".len();
        Some(body[from..from + body[from..].find('"')?].to_string())
    };
    Some((list("Avx2")?, list("Avx512")?))
}

/// The real soundness condition: each copied list is exactly fearless_simd's list (what its token
/// proves) plus `clflushopt` (what our token proves). Fails if a fearless upgrade changes a list.
#[test]
fn copied_lists_equal_fearless_simd_source() {
    let (avx2, avx512) = fearless_lists().expect("couldn't locate fearless_simd 1.0.0's source via cargo metadata");
    assert_eq!(AVX2_FEATURES, format!("{avx2},clflushopt"));
    assert_eq!(AVX512_FEATURES, format!("{avx512},clflushopt"));
}
