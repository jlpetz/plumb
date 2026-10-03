//! The TDP instructions against scalar references written from the SDM's operation sections.
//!
//! Layout (each tile is 16 rows of 16 dwords): `C[m][n] += sum(k < 16) dot(A[m][k], B[k][n])`,
//! where the dword `A[m][k]` holds 4 bytes or 2 halves and `dot` pairs them up element-wise.
//! The references index raw tile bytes the same way, so a wrong operand order, a transposed
//! operand or a wrong signedness shows up as a mismatch.

use plumb_tiles::{Amx, AmxInt8, ROW_BYTES, T0, T1, T2, T4, T6, T7, TILE_BYTES, Tiles};

use crate::{Lcg, amx_or_skip};

/// Dwords per tile row, and rows per tile.
const D: usize = 16;

fn i32s_to_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_i32s(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn u16s_to_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ---- INT8 ----------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Int8Op {
    Ssd,
    Sud,
    Usd,
    Uud,
}

impl Int8Op {
    const ALL: [Self; 4] = [Self::Ssd, Self::Sud, Self::Usd, Self::Uud];

    /// Signedness of the A and B bytes.
    fn signed(self) -> (bool, bool) {
        match self {
            Self::Ssd => (true, true),
            Self::Sud => (true, false),
            Self::Usd => (false, true),
            Self::Uud => (false, false),
        }
    }

    fn run<C: plumb_tiles::TileReg, A: plumb_tiles::TileReg, B: plumb_tiles::TileReg>(
        self,
        t: &mut Tiles<'_>,
        i8: AmxInt8,
    ) {
        match self {
            Self::Ssd => t.dpbssd::<C, A, B>(i8),
            Self::Sud => t.dpbsud::<C, A, B>(i8),
            Self::Usd => t.dpbusd::<C, A, B>(i8),
            Self::Uud => t.dpbuud::<C, A, B>(i8),
        }
    }
}

fn widen(b: u8, signed: bool) -> i32 {
    if signed {
        i32::from(b as i8)
    } else {
        i32::from(b)
    }
}

/// `C[m][n] += sum(k, i < 4) A[m][4k + i] * B[k][4n + i]` on raw tile bytes, 32-bit wrapping.
fn int8_reference(op: Int8Op, c: &[i32], a: &[u8], b: &[u8]) -> Vec<i32> {
    let (sa, sb) = op.signed();
    let mut out = c.to_vec();
    for m in 0..D {
        for n in 0..D {
            let mut acc = c[m * D + n];
            for k in 0..D {
                for i in 0..4 {
                    let p = widen(a[m * ROW_BYTES + 4 * k + i], sa)
                        * widen(b[k * ROW_BYTES + 4 * n + i], sb);
                    acc = acc.wrapping_add(p);
                }
            }
            out[m * D + n] = acc;
        }
    }
    out
}

/// Runs `op` once with C in T0, A in T1, B in T2, and once with C in T7, A in T4, B in T6.
fn int8_hardware(
    amx: Amx,
    i8: AmxInt8,
    op: Int8Op,
    c: &[i32],
    a: &[u8],
    b: &[u8],
) -> [Vec<i32>; 2] {
    let c = i32s_to_bytes(c);
    let (mut out0, mut out1) = (vec![0u8; TILE_BYTES], vec![0u8; TILE_BYTES]);
    amx.with_tiles(|t| {
        t.load::<T0>(&c, ROW_BYTES);
        t.load::<T1>(a, ROW_BYTES);
        t.load::<T2>(b, ROW_BYTES);
        op.run::<T0, T1, T2>(t, i8);
        t.store::<T0>(&mut out0, ROW_BYTES);

        t.load::<T7>(&c, ROW_BYTES);
        t.load::<T4>(a, ROW_BYTES);
        t.load::<T6>(b, ROW_BYTES);
        op.run::<T7, T4, T6>(t, i8);
        t.store::<T7>(&mut out1, ROW_BYTES);
    });
    [bytes_to_i32s(&out0), bytes_to_i32s(&out1)]
}

fn int8_tokens(test: &str) -> Option<(Amx, AmxInt8)> {
    let amx = amx_or_skip(test)?;
    let Some(i8) = amx.int8() else {
        eprintln!("{test}: skipped, no AMX-INT8");
        return None;
    };
    Some((amx, i8))
}

#[test]
fn int8_dot_products_match_reference() {
    let Some((amx, i8)) = int8_tokens("int8_dot_products_match_reference") else {
        return;
    };
    for seed in 1..=4u64 {
        let mut lcg = Lcg::new(seed);
        let a = lcg.bytes(TILE_BYTES);
        let b = lcg.bytes(TILE_BYTES);
        // A non-zero accumulator in +-2^29, so this case can't overflow (see the wrap test).
        let c: Vec<i32> = (0..D * D).map(|_| (lcg.next_u32() as i32) >> 2).collect();
        for op in Int8Op::ALL {
            let want = int8_reference(op, &c, &a, &b);
            assert_ne!(want, c, "{op:?}: the test data must change C");
            for (i, got) in int8_hardware(amx, i8, op, &c, &a, &b).iter().enumerate() {
                assert_eq!(*got, want, "{op:?}, seed {seed}, register set {i}");
            }
        }
    }
}

/// The signedness variants really differ on this data (so the test above can tell them apart).
#[test]
fn int8_variants_disagree_on_mixed_sign_data() {
    let mut lcg = Lcg::new(3);
    let (a, b) = (lcg.bytes(TILE_BYTES), lcg.bytes(TILE_BYTES));
    let c = vec![0; D * D];
    let results: Vec<_> = Int8Op::ALL
        .iter()
        .map(|&op| int8_reference(op, &c, &a, &b))
        .collect();
    for x in 0..4 {
        for y in x + 1..4 {
            assert_ne!(
                results[x],
                results[y],
                "{:?} vs {:?}",
                Int8Op::ALL[x],
                Int8Op::ALL[y]
            );
        }
    }
}

/// The INT32 accumulation wraps (no saturation, unlike `VPDPBSSDS`).
#[test]
fn int8_accumulator_wraps() {
    let Some((amx, i8)) = int8_tokens("int8_accumulator_wraps") else {
        return;
    };
    let a = vec![0x7Fu8; TILE_BYTES];
    let b = vec![0x7Fu8; TILE_BYTES];
    let c = vec![i32::MAX - 5; D * D];
    for op in Int8Op::ALL {
        let want = int8_reference(op, &c, &a, &b);
        assert!(
            want.iter().all(|&x| x < 0),
            "{op:?}: the reference must wrap"
        );
        assert_eq!(int8_hardware(amx, i8, op, &c, &a, &b)[0], want, "{op:?}");
    }
}

/// Accumulating twice adds the product twice.
#[test]
fn int8_accumulates_across_instructions() {
    let Some((amx, i8)) = int8_tokens("int8_accumulates_across_instructions") else {
        return;
    };
    let mut lcg = Lcg::new(11);
    let (a, b) = (lcg.bytes(TILE_BYTES), lcg.bytes(TILE_BYTES));
    let zero = vec![0; D * D];
    let once = int8_reference(Int8Op::Ssd, &zero, &a, &b);
    let twice = int8_reference(Int8Op::Ssd, &once, &a, &b);
    let mut out = vec![0u8; TILE_BYTES];
    amx.with_tiles(|t| {
        t.zero::<T0>();
        t.load::<T1>(&a, ROW_BYTES);
        t.load::<T2>(&b, ROW_BYTES);
        t.dpbssd::<T0, T1, T2>(i8);
        t.dpbssd::<T0, T1, T2>(i8);
        t.store::<T0>(&mut out, ROW_BYTES);
    });
    assert_eq!(bytes_to_i32s(&out), twice);
}

/// In matrix terms: A is 16 x 64 row-major, B is 64 x 16 and must be VNNI-packed, so that
/// logical `B[4k + i][n]` sits at byte `4n + i` of tile row `k`. Then the tile result is the
/// ordinary matrix product.
#[test]
fn int8_is_a_gemm_with_vnni_packed_b() {
    let Some((amx, i8)) = int8_tokens("int8_is_a_gemm_with_vnni_packed_b") else {
        return;
    };
    let mut lcg = Lcg::new(5);
    let a: Vec<i8> = lcg.bytes(16 * 64).into_iter().map(|x| x as i8).collect(); // [m][j]
    let b: Vec<i8> = lcg.bytes(64 * 16).into_iter().map(|x| x as i8).collect(); // [j][n]
    let mut packed = vec![0u8; TILE_BYTES];
    for k in 0..16 {
        for n in 0..16 {
            for i in 0..4 {
                packed[k * ROW_BYTES + 4 * n + i] = b[(4 * k + i) * 16 + n] as u8;
            }
        }
    }
    let mut want = vec![0i32; 16 * 16];
    for m in 0..16 {
        for n in 0..16 {
            want[m * 16 + n] = (0..64)
                .map(|j| i32::from(a[m * 64 + j]) * i32::from(b[j * 16 + n]))
                .sum();
        }
    }
    let a_bytes: Vec<u8> = a.iter().map(|&x| x as u8).collect();
    let mut out = vec![0u8; TILE_BYTES];
    amx.with_tiles(|t| {
        t.zero::<T0>();
        t.load::<T1>(&a_bytes, ROW_BYTES);
        t.load::<T2>(&packed, ROW_BYTES);
        t.dpbssd::<T0, T1, T2>(i8);
        t.store::<T0>(&mut out, ROW_BYTES);
    });
    assert_eq!(bytes_to_i32s(&out), want);
}

// ---- BF16 / FP16 ---------------------------------------------------------------------------

/// f32 to bf16 for values bf16 holds exactly (no rounding to get wrong).
fn bf16_from_f32_exact(x: f32) -> u16 {
    let bits = x.to_bits();
    assert_eq!(bits & 0xFFFF, 0, "{x} is not exact in bf16");
    (bits >> 16) as u16
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// f32 to IEEE binary16 for zeros and normal values binary16 holds exactly.
fn f16_from_f32_exact(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16 & 0x8000) as u16;
    if bits & 0x7FFF_FFFF == 0 {
        return sign;
    }
    let exp = (bits >> 23 & 0xFF) as i32 - 127 + 15;
    let man = bits & 0x7F_FFFF;
    assert!((1..31).contains(&exp), "{x} is not a normal f16");
    assert_eq!(man & 0x1FFF, 0, "{x} is not exact in f16");
    sign | (exp as u16) << 10 | (man >> 13) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from(h >> 10 & 0x1F);
    let man = f32::from(h & 0x3FF);
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 if man == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

#[test]
fn half_conversions_are_right() {
    for (x, bf, h) in [
        (1.0f32, 0x3F80u16, 0x3C00u16),
        (-2.0, 0xC000, 0xC000),
        (0.25, 0x3E80, 0x3400),
        (-3.75, 0xC070, 0xC380),
    ] {
        assert_eq!(bf16_from_f32_exact(x), bf, "bf16({x})");
        assert_eq!(f16_from_f32_exact(x), h, "f16({x})");
        assert_eq!(bf16_to_f32(bf), x);
        assert_eq!(f16_to_f32(h), x);
    }
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    // Every zero and normal f16 survives the round trip.
    for h in 0..=u16::MAX {
        let e = h >> 10 & 0x1F;
        if (e != 0 || h & 0x3FF == 0) && e != 31 {
            assert_eq!(f16_from_f32_exact(f16_to_f32(h)), h, "{h:#06x}");
        }
    }
}

/// `C[m][n] += sum(k, i < 2) A[m][2k + i] * B[k][2n + i]` on raw tile halves, in f32.
/// Returns the f32 result and the same sum in f64, to prove no step rounded.
fn half_reference(c: &[f32], a: &[u16], b: &[u16], to_f32: fn(u16) -> f32) -> (Vec<f32>, Vec<f64>) {
    let (mut out, mut out64) = (c.to_vec(), vec![0f64; c.len()]);
    for m in 0..D {
        for n in 0..D {
            let (mut acc, mut acc64) = (c[m * D + n], f64::from(c[m * D + n]));
            for k in 0..D {
                for i in 0..2 {
                    let (x, y) = (to_f32(a[m * 32 + 2 * k + i]), to_f32(b[k * 32 + 2 * n + i]));
                    acc += x * y;
                    acc64 += f64::from(x) * f64::from(y);
                }
            }
            out[m * D + n] = acc;
            out64[m * D + n] = acc64;
        }
    }
    (out, out64)
}

/// Multiples of 1/4 in [-4, 4] and an accumulator of multiples of 1/2 in [-100, 100]: every
/// product and partial sum is a multiple of 1/16 below 2^10, exact in f32 in any order.
fn half_inputs(seed: u64, from_f32: fn(f32) -> u16) -> (Vec<u16>, Vec<u16>, Vec<f32>) {
    let mut lcg = Lcg::new(seed);
    let mut quarter = || from_f32((lcg.next_u32() % 33) as f32 / 4.0 - 4.0);
    let a: Vec<u16> = (0..512).map(|_| quarter()).collect();
    let b: Vec<u16> = (0..512).map(|_| quarter()).collect();
    let c: Vec<f32> = (0..D * D)
        .map(|_| (lcg.next_u32() % 401) as f32 / 2.0 - 100.0)
        .collect();
    (a, b, c)
}

/// Runs `op` (C in T7, A in T1, B in T4) on four sets of exact inputs against the reference.
fn check_half(
    name: &str,
    amx: Amx,
    from_f32: fn(f32) -> u16,
    to_f32: fn(u16) -> f32,
    op: &dyn Fn(&mut Tiles<'_>),
) {
    for seed in 1..=4u64 {
        let (a, b, c) = half_inputs(seed, from_f32);
        let (want, want64) = half_reference(&c, &a, &b, to_f32);
        for (x, y) in want.iter().zip(&want64) {
            assert_eq!(
                f64::from(*x),
                *y,
                "{name}: the reference rounded; pick smaller test values"
            );
        }
        assert_ne!(want, c, "{name}: the test data must change C");
        let (c_bytes, a_bytes, b_bytes) = (f32s_to_bytes(&c), u16s_to_bytes(&a), u16s_to_bytes(&b));
        let mut out = vec![0u8; TILE_BYTES];
        amx.with_tiles(|t| {
            t.load::<T7>(&c_bytes, ROW_BYTES);
            t.load::<T1>(&a_bytes, ROW_BYTES);
            t.load::<T4>(&b_bytes, ROW_BYTES);
            op(t);
            t.store::<T7>(&mut out, ROW_BYTES);
        });
        assert_eq!(bytes_to_f32s(&out), want, "{name}, seed {seed}");
    }
}

#[test]
fn bf16_dot_product_matches_reference() {
    let Some(amx) = amx_or_skip("bf16_dot_product_matches_reference") else {
        return;
    };
    let Some(bf16) = amx.bf16() else {
        eprintln!("bf16_dot_product_matches_reference: skipped, no AMX-BF16");
        return;
    };
    check_half("tdpbf16ps", amx, bf16_from_f32_exact, bf16_to_f32, &|t| {
        t.dpbf16ps::<T7, T1, T4>(bf16)
    });
}

#[test]
fn fp16_dot_product_matches_reference() {
    let Some(amx) = amx_or_skip("fp16_dot_product_matches_reference") else {
        return;
    };
    let Some(fp16) = amx.fp16() else {
        eprintln!("fp16_dot_product_matches_reference: skipped, no AMX-FP16");
        return;
    };
    check_half("tdpfp16ps", amx, f16_from_f32_exact, f16_to_f32, &|t| {
        t.dpfp16ps::<T7, T1, T4>(fp16)
    });
}

/// `C[0][0]` after one instruction with only `A[0][0].half[0] = a` and `B[0][0].half[0] = b`.
fn single_product(amx: Amx, a: u16, b: u16, op: &dyn Fn(&mut Tiles<'_>)) -> f32 {
    let (mut at, mut bt, mut c) = (
        vec![0u8; TILE_BYTES],
        vec![0u8; TILE_BYTES],
        vec![0u8; TILE_BYTES],
    );
    at[..2].copy_from_slice(&a.to_le_bytes());
    bt[..2].copy_from_slice(&b.to_le_bytes());
    amx.with_tiles(|t| {
        t.zero::<T7>();
        t.load::<T1>(&at, ROW_BYTES);
        t.load::<T4>(&bt, ROW_BYTES);
        op(t);
        t.store::<T7>(&mut c, ROW_BYTES);
    });
    bytes_to_f32s(&c)[0]
}

/// TDPBF16PS ignores MXCSR and flushes: a denormal input counts as zero, and so does a
/// product that would be an f32 denormal.
#[test]
fn bf16_flushes_denormal_inputs_and_results() {
    let Some(amx) = amx_or_skip("bf16_flushes_denormal_inputs_and_results") else {
        return;
    };
    let Some(bf16) = amx.bf16() else {
        eprintln!("bf16_flushes_denormal_inputs_and_results: skipped, no AMX-BF16");
        return;
    };
    let op = |t: &mut Tiles<'_>| t.dpbf16ps::<T7, T1, T4>(bf16);
    let pow2 = |e: i32| bf16_from_f32_exact(2f32.powi(e));
    // 0x0040 is the bf16 denormal 2^-127; times 2^100 it would be a normal 2^-27.
    assert_eq!(single_product(amx, 0x0040, pow2(100), &op), 0.0);
    // 2^-100 x 2^-30 = 2^-130 is an f32 denormal.
    assert_eq!(single_product(amx, pow2(-100), pow2(-30), &op), 0.0);
    // Control: normal in, normal out.
    assert_eq!(
        single_product(amx, pow2(-100), pow2(-20), &op),
        2f32.powi(-120)
    );
}

/// TDPFP16PS keeps FP16 denormal inputs.
#[test]
fn fp16_keeps_denormal_inputs() {
    let Some(amx) = amx_or_skip("fp16_keeps_denormal_inputs") else {
        return;
    };
    let Some(fp16) = amx.fp16() else {
        eprintln!("fp16_keeps_denormal_inputs: skipped, no AMX-FP16");
        return;
    };
    let op = |t: &mut Tiles<'_>| t.dpfp16ps::<T7, T1, T4>(fp16);
    // 0x0001 is the smallest f16 denormal, 2^-24; 0x3C00 is 1.0.
    assert_eq!(single_product(amx, 0x0001, 0x3C00, &op), 2f32.powi(-24));
    assert_eq!(single_product(amx, 0x0001, 0x0001, &op), 2f32.powi(-48));
}
