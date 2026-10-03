//! Correctness of the AMX bench kernels (tiles.rs) on the real CPU. They skip, with a message,
//! when the CPU or OS has no AMX tile state.

use crate::common::*;
use crate::tiles;
use fearless_simd::Level;
use plumb_lines::Line;
use plumb_tiles::Amx;

const GROUP_U64: usize = 16 * 4096 / 8; // 64 KiB: the strided kernels' unit

fn amx() -> Option<Amx> {
    let a = Amx::try_new();
    if a.is_none() {
        eprintln!("skip: no AMX tile state on this CPU/OS");
    }
    a
}

fn buf(n: usize) -> Vec<Line> {
    vec![Line([0; 8]); n.div_ceil(8)]
}
fn words(v: &mut [Line]) -> &mut [u64] {
    // SAFETY: Line is [u64; 8].
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u64, v.len() * 8) }
}

#[test]
fn amx_fill_read_and_verify() {
    let Some(amx) = amx() else { return };
    let n = GROUP_U64 * 3;
    let mut store = buf(n + 8);
    let all = words(&mut store);
    // Fill handles a tail that isn't a whole tile (scalar rest).
    for len in [n, n + 5] {
        let b = &mut all[..len];
        b.fill(0);
        tiles::k_amx_fill(amx, b);
        assert!(b.iter().all(|&w| w == PATTERN), "fill len {len}");
    }
    let b = &mut all[..n];
    assert_eq!(tiles::k_amx_read(amx, b), 0);
    assert_eq!(tiles::k_amx_strided_read(amx, b), 0);
    let Some(t5) = Level::new().as_avx512() else { return eprintln!("skip: verify needs Avx512") };
    assert_eq!(tiles::k_amx_verify_512(t5, amx, b), 0, "false positive");
    for (i, bit) in [(0, 0), (127, 63), (n / 2, 31), (n - 1, 7), (n - 128, 17)] {
        b[i] ^= 1 << bit;
        assert_eq!(tiles::k_amx_verify_512(t5, amx, b), 1, "missed word {i} bit {bit}");
        b[i] ^= 1 << bit;
    }
    // SAFETY: aligned, whole 64 KiB groups. The fn's feature list (TMR's 512 set) is a subset of
    // fearless_simd's Avx512 level, which `t5` proves.
    let or = unsafe { tiles::k_amx_strided_read_zmm_512(b.as_ptr(), b.len()) };
    assert_eq!(or, PATTERN, "zmm strided read must see every word");
}

#[test]
fn amx_copy() {
    let Some(amx) = amx() else { return };
    let half = GROUP_U64 * 2;
    let mut store = buf(half * 2);
    let all = words(&mut store);
    let (src, dst) = all.split_at_mut(half);
    for (i, w) in src.iter_mut().enumerate() {
        *w = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    tiles::k_amx_copy(amx, dst, src);
    assert_eq!(&*dst, &*src);
    // An odd number of tiles exercises the single-tile remainder.
    let n = 3 * 128;
    dst[..n].fill(0);
    tiles::k_amx_copy(amx, &mut dst[..n], &src[..n]);
    assert_eq!(&dst[..n], &src[..n]);
}

#[test]
fn amx_read_rejects_partial_tiles() {
    let Some(amx) = amx() else { return };
    let mut store = buf(256);
    let b = &words(&mut store)[..200];
    assert!(std::panic::catch_unwind(|| tiles::k_amx_read(amx, b)).is_err());
}
