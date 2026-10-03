//! Named kernels for the asm gate on plumb_lines' own codegen:
//! `python bench/asm_check.py --package plumb_lines --example asm_kernels [--toolchain stable]`.
//! The bench binary is built with the `nightly` feature; this example checks the default
//! (stable, `asm!`) paths. Run as a program, it calls each kernel once as a smoke test.

use fearless_simd::prelude::*;
use fearless_simd::{Avx2, Avx512, Level, u64x4, u64x8};
use fearless_simd_macros::simd;
use plumb_lines::{Clflushopt, Line, Movdir64b, NtStore};
use std::hint::black_box;

const PAT: u64 = 0x0123_4567_89AB_CDEF;

/// Constant fill through the aligned view, 8 vectors per iteration.
#[simd]
fn fill<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &mut [u64]) {
    let (head, mid, tail) = plumb_lines::as_vectors_mut::<S, V>(simd, buf);
    head.iter_mut().take(V::LEN - 1).for_each(|w| *w = PAT);
    let p = V::splat(simd, PAT);
    let (eights, rest) = mid.as_chunks_mut::<8>();
    for e in eights {
        *e = [p; 8];
    }
    for v in rest {
        *v = p;
    }
    tail.iter_mut().take(V::LEN - 1).for_each(|w| *w = PAT);
}

/// Constant NT fill through the scope.
#[simd]
fn nt_fill<S: Simd, V: SimdInt<S, Element = u64> + NtStore<S>>(simd: S, buf: &mut [u64]) {
    let (_, mid, _) = plumb_lines::as_vectors_mut::<S, V>(simd, buf);
    plumb_lines::nontemporal(simd, mid, |w| w.fill_with(|_| V::splat(simd, PAT)));
}

/// Write a line, flush it: the per-line `flush_line` in a hand-unrolled loop (its docs).
#[allow(clippy::chunks_exact_to_as_chunks, reason = "V::LEN is generic; as_chunks needs a concrete constant")]
#[simd]
fn wflush<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, cf: Clflushopt, lines: &mut [Line]) {
    let p = V::splat(simd, PAT);
    let (quads, rest) = lines.as_chunks_mut::<4>();
    for quad in quads {
        for line in quad {
            for c in line.0.chunks_exact_mut(V::LEN) {
                p.store_slice(c);
            }
            cf.flush_line(line);
        }
    }
    for line in rest {
        for c in line.0.chunks_exact_mut(V::LEN) {
            p.store_slice(c);
        }
        cf.flush_line(line);
    }
    plumb_lines::mfence();
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_lines_flush(cf: Clflushopt, buf: &[u64]) {
    cf.flush(buf)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_lines_fillflush_256(t: Avx2, cf: Clflushopt, buf: &mut [u64]) {
    plumb_lines::flush_after(cf, buf, |b| fill::<Avx2, u64x4<Avx2>>(t, b))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_lines_fillflush_512(t: Avx512, cf: Clflushopt, buf: &mut [u64]) {
    plumb_lines::flush_after(cf, buf, |b| fill::<Avx512, u64x8<Avx512>>(t, b))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_lines_ntfill_256(t: Avx2, buf: &mut [u64]) {
    nt_fill::<Avx2, u64x4<Avx2>>(t, buf)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_lines_ntfill_512(t: Avx512, buf: &mut [u64]) {
    nt_fill::<Avx512, u64x8<Avx512>>(t, buf)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_lines_wflush_256(t: Avx2, cf: Clflushopt, lines: &mut [Line]) {
    wflush::<Avx2, u64x4<Avx2>>(t, cf, lines)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_lines_wflush_512(t: Avx512, cf: Clflushopt, lines: &mut [Line]) {
    wflush::<Avx512, u64x8<Avx512>>(t, cf, lines)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_lines_directfill(md: Movdir64b, dst: &mut [Line], line: &Line) {
    plumb_lines::direct(md, dst, |w| w.fill(line))
}

/// The lines as u64 words, from word 3 on (so the views have a misaligned head).
fn words(lines: &mut [Line]) -> &mut [u64] {
    // SAFETY: Line is [u64; 8]; the result borrows `lines`.
    let all = unsafe { std::slice::from_raw_parts_mut(lines.as_mut_ptr() as *mut u64, lines.len() * 8) };
    &mut all[3..]
}

fn main() {
    let level = Level::new();
    let mut lines = vec![Line::default(); 4096 + 3];
    let Some(cf) = Clflushopt::try_new() else { return println!("no CLFLUSHOPT; nothing run") };
    k_lines_flush(cf, black_box(words(&mut lines)));
    if let Some(t) = level.as_avx2() {
        k_lines_fillflush_256(t, cf, black_box(words(&mut lines)));
        k_lines_ntfill_256(t, black_box(words(&mut lines)));
        k_lines_wflush_256(t, cf, black_box(&mut lines));
    }
    if let Some(t) = level.as_avx512() {
        k_lines_fillflush_512(t, cf, black_box(words(&mut lines)));
        k_lines_ntfill_512(t, black_box(words(&mut lines)));
        k_lines_wflush_512(t, cf, black_box(&mut lines));
    }
    if let Some(md) = Movdir64b::try_new() {
        let src = Line([PAT; 8]);
        k_lines_directfill(md, black_box(&mut lines), &src);
    }
    assert!(lines.iter().all(|l| *l == Line([PAT; 8])));
    println!("all kernels ran");
}
