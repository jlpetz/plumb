//! Aligned vector views of element slices.
//!
//! [`as_vectors_mut`] splits `&mut [u64]` into `(head, &mut [V], tail)`, where `V` is a fearless
//! vector such as `u64x8<S>`: `head` runs up to the first `V`-aligned address, the middle is the
//! longest run of whole aligned vectors, and `tail` is what's left. Two things follow:
//!
//! - **Alignment is in the type.** A `&mut u64x8<S>` is 64-byte aligned, so NT stores into it
//!   need no runtime check ([`crate::nt`]).
//! - **Loops over `&[V]` are pointer walks.** `for v in vectors` (and fixed-size chunks of it)
//!   compile to base+displacement addressing. A `chunks_exact` loop over the `u64` slice
//!   produced indexed addressing (`[r9 + 8*r10 + 64]`) instead, which cost 16% on a 512-bit
//!   L2-resident verify (TODO 84 findings).
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
pub fn as_vectors<S: Simd, V: SimdBase<S>>(simd: S, buf: &[V::Element]) -> (&[V::Element], &[V], &[V::Element]) {
    let () = Layout::<S, V>::OK;
    let _ = simd;
    let len = buf.len();
    let ptr = buf.as_ptr();
    let head = head_len::<S, V>(ptr as usize, len);
    let n = (len - head) / V::LEN;
    let tail = head + n * V::LEN;
    // An empty middle must still be an aligned slice: when `buf` is too short to reach an aligned
    // address, `ptr + head` is not `V`-aligned, and `from_raw_parts` requires alignment even for
    // length 0.
    let mid: &[V] = if n == 0 {
        &[]
    } else {
        // SAFETY: the middle starts `V`-aligned (head_len) and holds `n` whole vectors inside
        // `buf`; `V` has the layout of `[Element; LEN]` (Layout::OK), every bit pattern of the
        // element type is a valid element (fearless element types are integers and floats), and
        // the zero-sized token is backed by `simd`.
        unsafe { slice::from_raw_parts(ptr.add(head) as *const V, n) }
    };
    // SAFETY: head and tail are disjoint sub-ranges of `buf` around the middle.
    unsafe { (slice::from_raw_parts(ptr, head), mid, slice::from_raw_parts(ptr.add(tail), len - tail)) }
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
    // See `as_vectors`: an empty middle must still be aligned.
    let mid: &mut [V] = if n == 0 {
        &mut []
    } else {
        // SAFETY: as in `as_vectors`; any element values written through `&mut V` are valid.
        unsafe { slice::from_raw_parts_mut(ptr.add(head) as *mut V, n) }
    };
    // SAFETY: the three ranges are disjoint sub-ranges of `buf`, so the mutable borrows don't
    // alias.
    unsafe { (slice::from_raw_parts_mut(ptr, head), mid, slice::from_raw_parts_mut(ptr.add(tail), len - tail)) }
}
