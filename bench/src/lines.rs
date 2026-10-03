// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Kernels built on plumb_lines, for comparison with the TMR-style kernels (`tmr.rs`) and the
//! plain fearless kernels (`fs.rs`). Naming: `k_<kernel>_pv_<width>` uses plumb_lines' aligned
//! views (`pv`), `_pl_` the other plumb_lines scopes. Every kernel is a named `#[unsafe(no_mangle)]`
//! entry for `asm_check.py`.

use crate::common::*;
use fearless_simd::prelude::*;
use fearless_simd::{Avx2, Avx512, u64x2, u64x4, u64x8};
use fearless_simd_macros::simd;
use plumb_lines::{Clflushopt, Line, Movdir64b, NtStore};

// ---------------------------------------------------------------------------------------------
// Generic bodies
// ---------------------------------------------------------------------------------------------

/// Constant fill through the aligned view, 8 vectors per iteration (TMR's unroll), scalar
/// head/tail. Unrolled by hand: a plain `for v in mid { *v = p }` unrolls standalone but ran one
/// store per iteration once inlined into the StuckBit port at 512 bits (asm_check twin rule).
#[simd]
pub fn fill_view<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &mut [u64], pat: u64) {
    let (head, mid, tail) = plumb_lines::as_vectors_mut::<S, V>(simd, buf);
    debug_assert!(head.len() < V::LEN && tail.len() < V::LEN);
    // Head and tail hold fewer than V::LEN elements; saying so (`take`) stops LLVM from
    // auto-vectorising them wider than the variant's width.
    head.iter_mut().take(V::LEN - 1).for_each(|w| *w = pat);
    let p = V::splat(simd, pat);
    let (eights, rest) = mid.as_chunks_mut::<8>();
    for e in eights {
        *e = [p; 8];
    }
    for v in rest {
        *v = p;
    }
    tail.iter_mut().take(V::LEN - 1).for_each(|w| *w = pat);
}

/// Reproduction, kept for timing: `fill_view`'s earlier shape, a plain loop over the middle that
/// relies on LLVM's unroller. Inlined into the StuckBit port at 512 bits it runs one store per
/// iteration (`k_sb_plplain_512`); the bench measures whether that costs anything.
#[simd]
pub fn fill_view_plain<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &mut [u64], pat: u64) {
    let (head, mid, tail) = plumb_lines::as_vectors_mut::<S, V>(simd, buf);
    head.iter_mut().take(V::LEN - 1).for_each(|w| *w = pat);
    let p = V::splat(simd, pat);
    for v in mid.iter_mut() {
        *v = p;
    }
    tail.iter_mut().take(V::LEN - 1).for_each(|w| *w = pat);
}

/// 4-accumulator verify through the aligned view. The vectors are a `&[V]`, so the loop is a
/// pointer walk with base+displacement addressing. The loop shape depends on the width, as
/// measured from L2 (`bench/RESULTS.md`): at 512 bits it walks one quad per iteration, because
/// LLVM unrolls an `as_chunks` loop and reassociates the ORs, which turns each fused
/// `vpternlogq acc, p, [mem]` into `vpxorq` + `vpternlogq` (16% slower). At 128 and 256 bits the
/// unrolled `as_chunks` loop is the faster one (3% and 10% over TMR's).
#[simd]
pub fn verify4_view<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &[u64], pat: u64) -> u64 {
    if size_of::<V>() == 64 {
        verify4_view_pat::<S, V>(simd, buf, pat)
    } else {
        verify4_view_chunks::<S, V>(simd, buf, pat)
    }
}

/// `verify4_view`'s `as_chunks::<4>` shape, at every width (`k_verify4_pvch_512` keeps the
/// 512-bit case measurable).
#[simd]
pub fn verify4_view_chunks<S: Simd, V: SimdInt<S, Element = u64>>(
    simd: S,
    buf: &[u64],
    pat: u64,
) -> u64 {
    let (head, mid, tail) = plumb_lines::as_vectors::<S, V>(simd, buf);
    debug_assert!(head.len() < V::LEN && tail.len() < V::LEN);
    let p = V::splat(simd, pat);
    let z = V::splat(simd, 0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let (quads, rest) = mid.as_chunks::<4>();
    for q in quads {
        a0 |= q[0] ^ p;
        a1 |= q[1] ^ p;
        a2 |= q[2] ^ p;
        a3 |= q[3] ^ p;
    }
    for &v in rest {
        a0 |= v ^ p;
    }
    let bad = ((a0 | a1) | (a2 | a3)).simd_eq(z).any_false()
        || head
            .iter()
            .take(V::LEN - 1)
            .chain(tail.iter().take(V::LEN - 1))
            .any(|&w| w != pat);
    bad as u64
}

/// The 4-accumulator verify over the view as a slice-pattern loop, one quad per iteration: the
/// shape of `verify4_ptr` without raw pointers, which LLVM leaves un-unrolled.
#[simd]
pub fn verify4_view_pat<S: Simd, V: SimdInt<S, Element = u64>>(
    simd: S,
    buf: &[u64],
    pat: u64,
) -> u64 {
    let (head, mid, tail) = plumb_lines::as_vectors::<S, V>(simd, buf);
    let p = V::splat(simd, pat);
    let z = V::splat(simd, 0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let mut rest = mid;
    while let [v0, v1, v2, v3, more @ ..] = rest {
        a0 |= *v0 ^ p;
        a1 |= *v1 ^ p;
        a2 |= *v2 ^ p;
        a3 |= *v3 ^ p;
        rest = more;
    }
    for &v in rest {
        a0 |= v ^ p;
    }
    let bad = ((a0 | a1) | (a2 | a3)).simd_eq(z).any_false()
        || head
            .iter()
            .take(V::LEN - 1)
            .chain(tail.iter().take(V::LEN - 1))
            .any(|&w| w != pat);
    bad as u64
}

/// L2 512-bit gap experiment: 8 accumulators over `as_chunks::<8>`. LLVM still unrolls and
/// reassociates it (83% of TMR's), so more accumulators don't help.
#[simd]
pub fn verify8_view<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, buf: &[u64], pat: u64) -> u64 {
    let (head, mid, tail) = plumb_lines::as_vectors::<S, V>(simd, buf);
    let p = V::splat(simd, pat);
    let z = V::splat(simd, 0);
    let mut a = [z; 8];
    let (eights, rest) = mid.as_chunks::<8>();
    for e in eights {
        for k in 0..8 {
            a[k] |= e[k] ^ p;
        }
    }
    for &v in rest {
        a[0] |= v ^ p;
    }
    let acc = ((a[0] | a[1]) | (a[2] | a[3])) | ((a[4] | a[5]) | (a[6] | a[7]));
    let bad = acc.simd_eq(z).any_false()
        || head
            .iter()
            .take(V::LEN - 1)
            .chain(tail.iter().take(V::LEN - 1))
            .any(|&w| w != pat);
    bad as u64
}

/// Positional verify (Mode 0/1) through the aligned view, single accumulator like TMR's.
/// Element `i` of `buf` must hold `(start + i) ^ base`.
#[simd]
pub fn pos_verify_view_at<S: Simd, V: SimdInt<S, Element = u64>>(
    simd: S,
    buf: &[u64],
    base: u64,
    start: u64,
) -> u64 {
    let (head, mid, tail) = plumb_lines::as_vectors::<S, V>(simd, buf);
    let z = V::splat(simd, 0);
    let base_v = V::splat(simd, base);
    let step = V::splat(simd, V::LEN as u64);
    let h = start + head.len() as u64;
    // TMR's shape, splat(start) + lane offsets. A runtime base inside from_fn made LLVM rebuild
    // the index vector through GPRs every iteration at 512 bits.
    let mut idx = V::splat(simd, h) + V::from_fn(simd, |i| i as u64);
    let mut acc = z;
    for &v in mid {
        acc |= v ^ (idx ^ base_v);
        idx += step;
    }
    let t0 = start + (head.len() + mid.len() * V::LEN) as u64;
    let bad = acc.simd_eq(z).any_false()
        || head
            .iter()
            .take(V::LEN - 1)
            .enumerate()
            .any(|(i, &w)| w != (start + i as u64) ^ base)
        || tail
            .iter()
            .take(V::LEN - 1)
            .enumerate()
            .any(|(i, &w)| w != (t0 + i as u64) ^ base);
    bad as u64
}

/// Positional NT write through the `nontemporal` scope (the fence is the scope's). Element `i`
/// gets `(start + i) ^ base`.
#[simd]
pub fn ntw_scope_at<S: Simd, V: SimdInt<S, Element = u64> + NtStore<S>>(
    simd: S,
    buf: &mut [u64],
    base: u64,
    start: u64,
) {
    let (head, mid, tail) = plumb_lines::as_vectors_mut::<S, V>(simd, buf);
    for (i, w) in head.iter_mut().take(V::LEN - 1).enumerate() {
        *w = (start + i as u64) ^ base;
    }
    let h = start + head.len() as u64;
    let t0 = start + (head.len() + mid.len() * V::LEN) as u64;
    plumb_lines::nontemporal(simd, mid, |w| {
        let base_v = V::splat(simd, base);
        let step = V::splat(simd, V::LEN as u64);
        let mut idx = V::splat(simd, h) + V::from_fn(simd, |i| i as u64);
        w.fill_with(|_| {
            let v = idx ^ base_v;
            idx += step;
            v
        });
    });
    for (i, w) in tail.iter_mut().take(V::LEN - 1).enumerate() {
        *w = (t0 + i as u64) ^ base;
    }
}

/// Mixed write + flush per line with the token's per-line flush (`asm!`; inlines anywhere).
#[simd]
pub fn wflush_token<S: Simd, V: SimdInt<S, Element = u64>>(
    simd: S,
    cf: Clflushopt,
    buf: &mut [u64],
    pat: u64,
) {
    let (head, lines, tail) = plumb_lines::as_lines_mut(buf);
    // Partial head/tail lines (< 8 words): written and flushed too, so nothing stays cached.
    head.iter_mut().take(7).for_each(|w| *w = pat);
    tail.iter_mut().take(7).for_each(|w| *w = pat);
    cf.flush_no_fence(head);
    cf.flush_no_fence(tail);
    let p = V::splat(simd, pat);
    // Four lines per iteration: LLVM won't runtime-unroll a loop around `asm!` (flush_line docs).
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

/// Same mixed loop with the stdarch intrinsic, for the `with_clflushopt!` entry (nightly).
#[inline(always)]
fn wflush_intrinsic<S: Simd, V: SimdInt<S, Element = u64>>(
    simd: S,
    cf: Clflushopt,
    buf: &mut [u64],
    pat: u64,
) {
    let (head, lines, tail) = plumb_lines::as_lines_mut(buf);
    head.iter_mut().take(7).for_each(|w| *w = pat);
    tail.iter_mut().take(7).for_each(|w| *w = pat);
    cf.flush_no_fence(head);
    cf.flush_no_fence(tail);
    let p = V::splat(simd, pat);
    for line in lines.iter_mut() {
        for c in line.0.chunks_exact_mut(V::LEN) {
            p.store_slice(c);
        }
        // SAFETY: `line` is inside `buf`; the entry enables clflushopt and `cf` proves it.
        unsafe { core::arch::x86_64::_mm_clflushopt(line as *const Line as *const u8) };
    }
    plumb_lines::mfence();
}

plumb_lines::with_clflushopt!(avx2, fn wflush_entry256 = wflush_intrinsic::<Avx2, u64x4<Avx2>>, (buf: &mut [u64], pat: u64));
plumb_lines::with_clflushopt!(avx512, fn wflush_entry512 = wflush_intrinsic::<Avx512, u64x8<Avx512>>, (buf: &mut [u64], pat: u64));

/// Footgun, kept visible on purpose: the same NT fill called from a plain generic fn (no
/// `#[simd]`, no `kernel!`). It compiles and is correct, but at 512 bits every fearless op and
/// every NT store is an out-of-line call (asm_check marks it KNOWN).
#[inline(never)]
fn ntw_plain<S: Simd, V: SimdInt<S, Element = u64> + NtStore<S>>(
    simd: S,
    buf: &mut [u64],
    base: u64,
) {
    let (_, mid, _) = plumb_lines::as_vectors_mut::<S, V>(simd, buf);
    plumb_lines::nontemporal(simd, mid, |w| {
        let base_v = V::splat(simd, base);
        let step = V::splat(simd, V::LEN as u64);
        let mut idx = V::from_fn(simd, |i| i as u64);
        w.fill_with(|_| {
            let v = idx ^ base_v;
            idx += step;
            v
        });
    });
}

// ---------------------------------------------------------------------------------------------
// Named entry points
// ---------------------------------------------------------------------------------------------

macro_rules! entry {
    ($name:ident, $Tok:ident, $V:ident, $kernel:ident, ($($arg:ident: $ty:ty),*) $(-> $ret:ty)?, ($($call:expr),*)) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name(t: $Tok, $($arg: $ty),*) $(-> $ret)? {
            $kernel::<$Tok, $V<$Tok>>(t, $($call),*)
        }
    };
}

entry!(k_fill_pv_128, Avx2, u64x2, fill_view, (buf: &mut [u64]), (buf, PATTERN));
entry!(k_fill_pv_256, Avx2, u64x4, fill_view, (buf: &mut [u64]), (buf, PATTERN));
entry!(k_fill_pv_512, Avx512, u64x8, fill_view, (buf: &mut [u64]), (buf, PATTERN));
entry!(k_verify4_pv_128, Avx2, u64x2, verify4_view, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_pv_256, Avx2, u64x4, verify4_view, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_pv_512, Avx512, u64x8, verify4_view, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_pvpat_128, Avx2, u64x2, verify4_view_pat, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_pvpat_256, Avx2, u64x4, verify4_view_pat, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify4_pvch_512, Avx512, u64x8, verify4_view_chunks, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_verify8_pv_512, Avx512, u64x8, verify8_view, (buf: &[u64]) -> u64, (buf, PATTERN));
entry!(k_posv_pv_128, Avx2, u64x2, pos_verify_view_at, (buf: &[u64]) -> u64, (buf, POS_BASE, 0));
entry!(k_posv_pv_256, Avx2, u64x4, pos_verify_view_at, (buf: &[u64]) -> u64, (buf, POS_BASE, 0));
entry!(k_posv_pv_512, Avx512, u64x8, pos_verify_view_at, (buf: &[u64]) -> u64, (buf, POS_BASE, 0));
entry!(k_ntw_pl_128, Avx2, u64x2, ntw_scope_at, (buf: &mut [u64]), (buf, POS_BASE, 0));
entry!(k_ntw_pl_256, Avx2, u64x4, ntw_scope_at, (buf: &mut [u64]), (buf, POS_BASE, 0));
entry!(k_ntw_pl_512, Avx512, u64x8, ntw_scope_at, (buf: &mut [u64]), (buf, POS_BASE, 0));

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_ntw_plplain_512(t: Avx512, buf: &mut [u64]) {
    ntw_plain::<Avx512, u64x8<Avx512>>(t, buf, POS_BASE)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_wflush_pltok_256(t: Avx2, cf: Clflushopt, buf: &mut [u64]) {
    wflush_token::<Avx2, u64x4<Avx2>>(t, cf, buf, PATTERN)
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_wflush_pltok_512(t: Avx512, cf: Clflushopt, buf: &mut [u64]) {
    wflush_token::<Avx512, u64x8<Avx512>>(t, cf, buf, PATTERN)
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_wflush_plentry_256(t: Avx2, cf: Clflushopt, buf: &mut [u64]) {
    wflush_entry256(t, cf, buf, PATTERN)
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_wflush_plentry_512(t: Avx512, cf: Clflushopt, buf: &mut [u64]) {
    wflush_entry512(t, cf, buf, PATTERN)
}

/// `flush_range_to_dram` through the token (nightly: the intrinsic in a clflushopt fn).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_flush_pl(cf: Clflushopt, buf: &[u64]) {
    cf.flush(buf)
}

/// Write the chunk, then flush everything written: TMR's "write, flush_range_to_dram" as a scope.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_fillflush_pl_256(t: Avx2, cf: Clflushopt, buf: &mut [u64]) {
    plumb_lines::flush_after(cf, buf, |b| fill_view::<Avx2, u64x4<Avx2>>(t, b, PATTERN))
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_fillflush_pl_512(t: Avx512, cf: Clflushopt, buf: &mut [u64]) {
    plumb_lines::flush_after(cf, buf, |b| {
        fill_view::<Avx512, u64x8<Avx512>>(t, b, PATTERN)
    })
}

/// TMR style for the same: the fill, then `flush_range_to_dram` (tmr.rs kernels).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_fillflush_tmr_256(buf: &mut [u64]) {
    // SAFETY: whole buffer; AVX2 and CLFLUSHOPT gated by the caller.
    unsafe {
        crate::tmr::k_fill_tmr_256(buf.as_mut_ptr(), buf.len());
        crate::tmr::k_flush_tmr(buf.as_ptr(), buf.len());
    }
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_fillflush_tmr_512(buf: &mut [u64]) {
    // SAFETY: as above, AVX-512.
    unsafe {
        crate::tmr::k_fill_tmr_512(buf.as_mut_ptr(), buf.len());
        crate::tmr::k_flush_tmr(buf.as_ptr(), buf.len());
    }
}

/// MOVDIR64B copy (first half -> second half) through the `direct` scope.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_copymd_pl(md: Movdir64b, dst: &mut [u64], src: &[u64]) {
    let (dh, dl, dt) = plumb_lines::as_lines_mut(dst);
    let (sh, sl, st) = plumb_lines::as_lines(src);
    assert!(
        dh.is_empty() && dt.is_empty() && sh.is_empty() && st.is_empty(),
        "line-aligned halves"
    );
    plumb_lines::direct(md, dl, |w| w.copy_from(sl));
}

/// MOVDIR64B pattern fill: one source line copied to every destination line.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn k_fillmd_pl(md: Movdir64b, buf: &mut [u64]) {
    let pattern = Line([PATTERN; 8]);
    let (h, lines, t) = plumb_lines::as_lines_mut(buf);
    h.fill(PATTERN);
    t.fill(PATTERN);
    plumb_lines::direct(md, lines, |w| w.fill(&pattern));
}
