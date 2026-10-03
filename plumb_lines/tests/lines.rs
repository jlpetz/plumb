// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration tests on the real CPU. They skip (with a message) what the CPU lacks.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "test data: small indices, truncated or wrapped on purpose"
)]

use fearless_simd::prelude::*;
use fearless_simd::{
    Avx2, Avx512, Level, Sse2, Sse4_2, f64x8, i32x16, u8x64, u32x16, u64x2, u64x4, u64x8,
};
use plumb_lines::{
    Clflushopt, Line, Movdir64b, NtStore, as_lines, as_lines_mut, as_vectors, as_vectors_mut,
    direct, flush_after, nontemporal,
};

/// A 64-byte-aligned u64 buffer (Vec<Line> gives the alignment).
fn aligned(n_u64: usize) -> Vec<Line> {
    vec![Line([0; 8]); n_u64.div_ceil(8)]
}
fn words(v: &mut [Line]) -> &mut [u64] {
    // SAFETY: Line is [u64; 8].
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u64, v.len() * 8) }
}

fn level() -> Level {
    Level::new()
}

fn skip(what: &str) {
    eprintln!("skip: {what} not available on this CPU");
}

// ------------------------------------------------------------------------------------- view

fn check_view<S: Simd, V: SimdBase<S, Element = u64>>(simd: S) {
    let mut lines = aligned(1024);
    let all = words(&mut lines);
    for (i, w) in all.iter_mut().enumerate() {
        *w = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    for off in 0..V::LEN + 3 {
        for len in [0, 1, V::LEN - 1, V::LEN, V::LEN + 1, 5 * V::LEN + 3, 700] {
            let buf = &all[off..off + len];
            let (head, mid, tail) = as_vectors::<S, V>(simd, buf);
            assert_eq!(
                head.len() + mid.len() * V::LEN + tail.len(),
                len,
                "off {off} len {len}"
            );
            assert!(
                head.len() < V::LEN && tail.len() < V::LEN,
                "head/tail must be partial vectors"
            );
            assert_eq!(
                mid.as_ptr() as usize % align_of::<V>(),
                0,
                "middle unaligned (off {off} len {len})"
            );
            if !mid.is_empty() {
                // The middle is as long as possible: head is exactly up to the first aligned address.
                let first_aligned = (buf.as_ptr() as usize).next_multiple_of(align_of::<V>());
                assert_eq!(
                    head.len(),
                    (first_aligned - buf.as_ptr() as usize) / 8,
                    "head must end at the first aligned address"
                );
            }
            for (k, v) in mid.iter().enumerate() {
                for j in 0..V::LEN {
                    assert_eq!(
                        v[j],
                        buf[head.len() + k * V::LEN + j],
                        "vector {k} lane {j}"
                    );
                }
            }
            assert_eq!(
                tail,
                &buf[head.len() + mid.len() * V::LEN..],
                "tail must be the rest"
            );
        }
    }
    // Writes through the mutable view land in the buffer.
    let buf = &mut all[3..300];
    let (h, mid, _) = as_vectors_mut::<S, V>(simd, buf);
    let h = h.len();
    for v in mid.iter_mut() {
        *v = V::splat(simd, 7);
    }
    assert!(
        buf[h..h + 8].iter().all(|&w| w == 7),
        "writes through the view must land"
    );
}

#[test]
fn view_splits_every_offset() {
    let l = level();
    let t2 = l.as_avx2().expect("AVX2");
    check_view::<Avx2, u64x2<Avx2>>(t2);
    check_view::<Avx2, u64x4<Avx2>>(t2);
    check_view::<Avx2, u64x8<Avx2>>(t2);
    match l.as_avx512() {
        Some(t5) => check_view::<Avx512, u64x8<Avx512>>(t5),
        None => skip("Avx512"),
    }
}

#[test]
fn view_other_element_types() {
    let Some(t5) = level().as_avx512() else {
        return skip("Avx512");
    };
    let mut bytes = vec![0_u8; 4096 + 100];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = i as u8;
    }
    for off in 0..70 {
        let buf = &bytes[off..];
        let (h, mid, t) = as_vectors::<Avx512, u8x64<Avx512>>(t5, buf);
        assert_eq!(mid.as_ptr() as usize % 64, 0);
        assert_eq!(h.len() + mid.len() * 64 + t.len(), buf.len());
        assert_eq!(mid[0][0], buf[h.len()]);
    }
    // f32: positional data, so a wrong head length for 4-byte elements would show.
    let floats: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    for off in 0..17 {
        let buf = &floats[off..];
        let (h, mid, t) = as_vectors::<Avx512, fearless_simd::f32x16<Avx512>>(t5, buf);
        assert_eq!(mid.as_ptr() as usize % 64, 0);
        assert_eq!(h.len() + mid.len() * 16 + t.len(), buf.len());
        for (k, v) in mid.iter().enumerate() {
            for j in 0..16 {
                assert_eq!(v[j], buf[h.len() + k * 16 + j]);
            }
        }
    }
}

// --------------------------------------------------------------------------------------- nt

const BASE: u64 = 0xDEAD_BEEF_DEAD_BEEF;

/// Positional fill through the scope, using a stateful generator (as TMR's NT test does).
fn nt_positional<S: Simd, V: NtStore<S> + SimdInt<S, Element = u64>>(simd: S, dst: &mut [V]) {
    nontemporal(simd, dst, |w| {
        let mut idx = V::from_fn(simd, |i| i as u64);
        let step = V::splat(simd, V::LEN as u64);
        let base = V::splat(simd, BASE);
        w.fill_with(|_| {
            let v = idx ^ base;
            idx += step;
            v
        });
    });
}

fn check_nt<S: Simd, V: NtStore<S> + SimdInt<S, Element = u64>>(simd: S) {
    for n_vectors in [0, 1, 3, 4, 5, 17, 256] {
        let mut lines = aligned(n_vectors * V::LEN + 8);
        let buf = words(&mut lines);
        let (h, mid, _) = as_vectors_mut::<S, V>(simd, buf);
        assert!(h.is_empty(), "an aligned buffer has no head");
        let mid = &mut mid[..n_vectors];
        nt_positional(simd, mid);
        for (k, v) in mid.iter().enumerate() {
            for j in 0..V::LEN {
                assert_eq!(
                    v[j],
                    (k * V::LEN + j) as u64 ^ BASE,
                    "n {n_vectors} vec {k} lane {j}"
                );
            }
        }
    }
    // split_at + fill_with on one half, write-once slots on the other.
    let mut lines = aligned(8 * V::LEN);
    let (_, mid, _) = as_vectors_mut::<S, V>(simd, words(&mut lines));
    nontemporal(simd, &mut mid[..8], |w| {
        assert_eq!(w.len(), 8, "writer length");
        let (a, b) = w.split_at(4);
        assert_eq!((a.len(), b.len()), (4, 4), "split lengths");
        a.fill_with(|i| V::splat(simd, i as u64));
        for (i, slot) in b.into_slots().enumerate() {
            slot.store(V::splat(simd, 104 + i as u64));
        }
    });
    for (i, v) in mid[..8].iter().enumerate() {
        let want = if i < 4 { i as u64 } else { 100 + i as u64 };
        assert!((0..V::LEN).all(|j| v[j] == want), "slot {i}");
    }
}

#[test]
fn nt_every_level_and_width() {
    let l = level();
    match l.as_sse2() {
        Some(t) => {
            check_nt::<Sse2, u64x2<Sse2>>(t);
            check_nt::<Sse2, u64x4<Sse2>>(t); // two xmm stores
            check_nt::<Sse2, u64x8<Sse2>>(t); // four
        }
        None => skip("Sse2"),
    }
    match l.as_sse4_2() {
        Some(t) => {
            check_nt::<Sse4_2, u64x2<Sse4_2>>(t);
            check_nt::<Sse4_2, u64x4<Sse4_2>>(t);
            check_nt::<Sse4_2, u64x8<Sse4_2>>(t);
        }
        None => skip("Sse4_2"),
    }
    let t2 = l.as_avx2().expect("AVX2");
    check_nt::<Avx2, u64x2<Avx2>>(t2);
    check_nt::<Avx2, u64x4<Avx2>>(t2);
    check_nt::<Avx2, u64x8<Avx2>>(t2); // two ymm stores
    match l.as_avx512() {
        Some(t5) => {
            check_nt::<Avx512, u64x2<Avx512>>(t5);
            check_nt::<Avx512, u64x4<Avx512>>(t5);
            check_nt::<Avx512, u64x8<Avx512>>(t5);
        }
        None => skip("Avx512"),
    }
}

#[test]
fn nt_other_element_types() {
    let Some(t5) = level().as_avx512() else {
        return skip("Avx512");
    };
    let mut lines = aligned(64);
    let buf = words(&mut lines);
    // SAFETY: u64 -> u32 view of the same aligned memory.
    let buf32 =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u32, buf.len() * 2) };
    let (_, mid, _) = as_vectors_mut::<Avx512, u32x16<Avx512>>(t5, buf32);
    nontemporal(t5, mid, |w| {
        w.fill_with(|i| u32x16::splat(t5, i as u32 + 1));
    });
    assert!(
        mid.iter()
            .enumerate()
            .all(|(i, v)| (0..16).all(|j| v[j] == i as u32 + 1))
    );

    let mut f = vec![f64x8::splat(t5, 0.0); 9];
    nontemporal(t5, &mut f, |w| {
        w.fill_with(|i| f64x8::splat(t5, i as f64 * 0.5));
    });
    assert!(
        f.iter()
            .enumerate()
            .all(|(i, v)| (0..8).all(|j| v[j] == i as f64 * 0.5))
    );

    let mut s = vec![i32x16::splat(t5, 0); 5];
    nontemporal(t5, &mut s, |w| {
        w.fill_with(|i| i32x16::splat(t5, -(i as i32) - 1));
    });
    assert!(
        s.iter()
            .enumerate()
            .all(|(i, v)| (0..16).all(|j| v[j] == -(i as i32) - 1))
    );
}

#[test]
fn nt_empty_destination() {
    let t2 = level().as_avx2().expect("AVX2");
    let mut empty: [u64x4<Avx2>; 0] = [];
    nontemporal(t2, &mut empty, |w| {
        assert!(w.is_empty());
        assert_eq!(w.into_slots().count(), 0);
    });
    nontemporal(t2, &mut empty, |w| w.fill_with(|_| unreachable!()));
}

#[test]
#[should_panic(expected = "out of range")]
fn nt_split_out_of_range_panics() {
    let t2 = level().as_avx2().expect("AVX2");
    let mut v = vec![u64x4::splat(t2, 0); 4];
    nontemporal(t2, &mut v, |w| {
        let _ = w.split_at(5);
    });
}

/// What this checks: a panic inside the scope unwinds through it (the fence guard runs; that the
/// guard executes SFENCE is visible in the asm, not observable here), and stores made before the
/// panic are visible afterwards.
#[test]
fn nt_unwind_ends_scope_and_keeps_earlier_stores() {
    let t2 = level().as_avx2().expect("AVX2");
    let mut v = vec![u64x4::splat(t2, 0); 4];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        nontemporal(t2, &mut v, |w| {
            w.into_slots().next().unwrap().store(u64x4::splat(t2, 42));
            panic!("boom");
        })
    }));
    assert!(r.is_err());
    assert_eq!(v[0][0], 42);
}

// ------------------------------------------------------------------------------------ flush

#[test]
fn flush_after_keeps_contents_and_returns() {
    let cf = Clflushopt::try_new().expect("CLFLUSHOPT");
    assert!(cf.line_bytes().is_power_of_two() && (32..=4096).contains(&cf.line_bytes()));
    let mut lines = aligned(4096);
    let buf = words(&mut lines);
    for off in [0_usize, 1, 3, 7] {
        let sub = &mut buf[off..off + 1000];
        let r = flush_after(cf, sub, |b| {
            for (i, w) in b.iter_mut().enumerate() {
                *w = i as u64 ^ 0xA55A_A55A_A55A_A55A;
            }
            b.len()
        });
        assert_eq!(r, 1000);
        assert!(
            sub.iter()
                .enumerate()
                .all(|(i, &w)| w == i as u64 ^ 0xA55A_A55A_A55A_A55A)
        );
    }
    flush_after(cf, &mut buf[..0], |_| ());
    cf.flush(&buf[5..9]);
    cf.flush_line(&buf[17]);
    plumb_lines::mfence();
}

/// Zero-sized referents have dangling addresses (0x1, 0x8, one past the end): no CLFLUSHOPT.
#[test]
fn flush_line_on_zero_sized_is_a_no_op() {
    let cf = Clflushopt::try_new().expect("CLFLUSHOPT");
    cf.flush_line(&());
    cf.flush_line(&[] as &[u64]);
    cf.flush_line(Vec::<u64>::new().as_slice());
    let v = [1_u64; 8];
    cf.flush_line(&v[8..]);
    cf.flush(&v[8..]);
    cf.flush::<u64>(&[]);
}

/// What this checks: a panic inside the scope unwinds through it (the flush guard runs) and the
/// write before the panic is kept.
#[test]
fn flush_after_unwind_ends_scope_and_keeps_writes() {
    let cf = Clflushopt::try_new().expect("CLFLUSHOPT");
    let mut v = vec![0_u64; 512];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        flush_after(cf, &mut v, |b| {
            b[10] = 99;
            panic!("boom")
        })
    }));
    assert!(r.is_err());
    assert_eq!(v[10], 99);
}

// ----------------------------------------------------------------------------------- direct

#[test]
fn lines_split_every_offset() {
    let mut lines = aligned(1024);
    let buf = words(&mut lines);
    for off in 0..10 {
        for len in [0, 7, 8, 9, 100, 500] {
            let b = &buf[off..off + len];
            let (h, mid, t) = as_lines(b);
            assert_eq!(h.len() + mid.len() * 8 + t.len(), len);
            assert_eq!(mid.as_ptr() as usize % 64, 0);
            if !mid.is_empty() {
                assert_eq!(h.len(), (8 - off % 8) % 8);
            }
        }
    }
    // Mutable split at misaligned offsets: writes through the lines land in the buffer.
    for off in 0..10 {
        let b = &mut buf[off..off + 200];
        let (h, mid, _) = as_lines_mut(b);
        let h = h.len();
        for l in mid.iter_mut() {
            *l = Line([off as u64 + 1; 8]);
        }
        assert!(b[h..h + 8].iter().all(|&w| w == off as u64 + 1));
    }
}

#[test]
fn direct_copy_fill_and_generate() {
    let Some(md) = Movdir64b::try_new() else {
        return skip("MOVDIR64B");
    };
    let mut dst = vec![Line::default(); 100];
    let pattern = Line([1, 2, 3, 4, 5, 6, 7, 8]);
    direct(md, &mut dst, |w| {
        assert_eq!(w.len(), 100);
        w.fill(&pattern);
    });
    assert!(dst.iter().all(|l| *l == pattern));

    let src: Vec<Line> = (0..100).map(|i| Line([i; 8])).collect();
    direct(md, &mut dst, |w| w.copy_from(&src));
    assert_eq!(dst, src);

    direct(md, &mut dst, |w| w.fill_with(|i| Line([i as u64 * 3; 8])));
    assert!(
        dst.iter()
            .enumerate()
            .all(|(i, l)| *l == Line([i as u64 * 3; 8]))
    );

    direct(md, &mut dst, |w| {
        let (a, b) = w.split_at(57);
        a.fill(&Line::default());
        for (i, slot) in b.into_slots().enumerate() {
            slot.copy(&Line([1000 + i as u64; 8]));
        }
    });
    assert!(dst[..57].iter().all(|l| *l == Line::default()));
    assert!(
        dst[57..]
            .iter()
            .enumerate()
            .all(|(i, l)| *l == Line([1000 + i as u64; 8]))
    );
}

#[test]
fn direct_length_mismatch_and_split_panic() {
    let Some(md) = Movdir64b::try_new() else {
        return skip("MOVDIR64B");
    };
    let mut dst = vec![Line::default(); 4];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        direct(md, &mut dst, |w| w.copy_from(&[Line::default(); 3]));
    }));
    assert!(r.is_err());
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        direct(md, &mut dst, |w| {
            let _ = w.split_at(5);
        });
    }));
    assert!(r.is_err());
}

/// What this checks: a panic inside `direct` unwinds through it (the fence guard runs) and the
/// line written before the panic is kept.
#[test]
fn direct_unwind_ends_scope_and_keeps_earlier_stores() {
    let Some(md) = Movdir64b::try_new() else {
        return skip("MOVDIR64B");
    };
    let mut dst = vec![Line::default(); 4];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        direct(md, &mut dst, |w| {
            w.into_slots().next().unwrap().copy(&Line([5; 8]));
            panic!("boom")
        })
    }));
    assert!(r.is_err());
    assert_eq!(dst[0], Line([5; 8]));
}
