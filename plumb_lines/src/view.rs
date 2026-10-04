// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Aligned vector views of element slices.
//!
//! [`as_vectors_mut`] splits `&mut [u64]` into `(head, &mut [V], tail)`, where `V` is a fearless
//! vector such as `u64x8<S>`: `head` runs up to the first `V`-aligned address, the middle is the
//! longest run of whole aligned vectors, and `tail` is what's left. Two things follow:
//!
//! - **Alignment is in the type.** A `&mut u64x8<S>` is 64-byte aligned, so NT stores into it
//!   need no runtime check ([`crate::nt`]).
//! - **Loops over `&[V]` are pointer walks** (base+displacement addressing). One loop shape to
//!   know at 512 bits: walk an OR-accumulate verify one group of four vectors per iteration
//!   (`while let [a, b, c, d, rest @ ..] = mid`). LLVM unrolls an `as_chunks::<4>` loop there
//!   and reassociates the ORs, losing the fused `vpternlogq`, which cost 16% from L2
//!   (`bench/RESULTS.md`).
//!
//! The split is computed from the address, not with `align_to`, whose documentation allows it
//! to return everything as the head. Here the middle is always as long as possible.

use core::mem::{align_of, size_of};
use core::slice;
use fearless_simd::{Simd, SimdBase};

/// Compile-time check that `V` is laid out as `[V::Element; V::LEN]` (fearless documents this
/// for every vector type: the same layout as the array, with stronger alignment) and that the
/// token is zero-sized, so reinterpreting element memory as `V` is sound.
struct Layout<S, V>(core::marker::PhantomData<(S, V)>);
impl<S: Simd, V: SimdBase<S>> Layout<S, V> {
    const OK: () = assert!(
        size_of::<V>() == V::LEN * size_of::<V::Element>()
            && size_of::<S>() == 0
            && align_of::<V>().is_multiple_of(align_of::<V::Element>()),
        "fearless vector layout is not [Element; LEN]"
    );
}

/// Head length (in elements) up to the first `V`-aligned address, capped at `len`.
#[inline(always)]
fn head_len<S: Simd, V: SimdBase<S>>(addr: usize, len: usize) -> usize {
    let align = align_of::<V>();
    let elem = size_of::<V::Element>().max(1);
    // Bytes to the next multiple of `align`. Elements are `elem`-aligned and `align` is a
    // multiple of `elem`, so this is a whole number of elements.
    let misaligned = addr.wrapping_neg() & (align - 1);
    (misaligned / elem).min(len)
}

/// Splits `buf` into `(head, vectors, tail)`; see the [module docs](self). Takes the token
/// because every `V` value carries one: a `u64x8<Avx512>` is a proof that AVX-512 is present.
#[inline(always)]
pub fn as_vectors<S: Simd, V: SimdBase<S>>(
    simd: S,
    buf: &[V::Element],
) -> (&[V::Element], &[V], &[V::Element]) {
    let () = Layout::<S, V>::OK;
    let _ = simd;
    let len = buf.len();
    let ptr = buf.as_ptr();
    let head = head_len::<S, V>(ptr as usize, len);
    let n = (len - head) / V::LEN;
    let tail = head + n * V::LEN;
    let mid = align_up::<V>(ptr.wrapping_add(head).cast::<V>());
    // SAFETY: when n > 0 the middle starts `V`-aligned at `ptr + head` (head_len; align_up is
    // then the identity) and holds `n` whole vectors inside `buf`. When n == 0 it is a non-null,
    // aligned, zero-length slice. `V` has the layout of `[Element; LEN]` (Layout::OK), every bit
    // pattern of the element type is a valid element (fearless element types are integers and
    // floats), and the zero-sized token is backed by `simd`. Head and tail are the disjoint
    // sub-ranges of `buf` around the middle.
    unsafe {
        (
            slice::from_raw_parts(ptr, head),
            slice::from_raw_parts(mid, n),
            slice::from_raw_parts(ptr.add(tail), len - tail),
        )
    }
}

/// Round `p` up to `T`'s alignment, keeping its provenance. Identity for aligned pointers. Used so
/// an empty middle is still an aligned slice (`from_raw_parts` requires that even for length 0;
/// a too-short `buf` leaves `ptr + head` unaligned) without a branch, which would stop LLVM from
/// unrolling loops over the middle.
#[inline(always)]
fn align_up<T>(p: *const T) -> *const T {
    let a = align_of::<T>();
    p.map_addr(|addr| (addr + a - 1) & !(a - 1))
}

/// Mutable version of [`as_vectors`].
#[inline(always)]
pub fn as_vectors_mut<S: Simd, V: SimdBase<S>>(
    simd: S,
    buf: &mut [V::Element],
) -> (&mut [V::Element], &mut [V], &mut [V::Element]) {
    let () = Layout::<S, V>::OK;
    let _ = simd;
    let len = buf.len();
    let ptr = buf.as_mut_ptr();
    let head = head_len::<S, V>(ptr as usize, len);
    let n = (len - head) / V::LEN;
    let tail = head + n * V::LEN;
    let mid = align_up::<V>(ptr.wrapping_add(head).cast::<V>().cast_const()).cast_mut();
    // SAFETY: as in `as_vectors`. The three ranges are disjoint (an empty middle covers no
    // bytes), so the mutable borrows don't alias, and any element values written through
    // `&mut V` are valid elements.
    unsafe {
        (
            slice::from_raw_parts_mut(ptr, head),
            slice::from_raw_parts_mut(mid, n),
            slice::from_raw_parts_mut(ptr.add(tail), len - tail),
        )
    }
}
