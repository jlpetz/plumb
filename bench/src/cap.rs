//! Prototype: a **capability token** for CLFLUSHOPT that combines with a fearless_simd level,
//! so the stdarch intrinsic inlines inside fearless kernels (no per-line call, no `asm!`).
//!
//! CLFLUSHOPT isn't on fearless_simd's level ladder (Skylake client and Zen 1 have it with only
//! AVX2), so it is a separate proof token. The kernel body stays generic and
//! `#[inline(always)]`; `with_clflushopt!` emits one entry fn per level whose
//! `#[target_feature]` list is fearless_simd's own list for that level **plus** `clflushopt`.
//! Inlined into that entry, fearless's ops (whose inner fns need a subset of the features) and
//! `_mm_clflushopt` all inline.
//!
//! COUPLING: the level feature strings below are copied from fearless_simd 1.0.0
//! (`__fearless_simd_kernel_target_fn!`). If fearless drops a feature from a level, we would
//! enable one its token no longer proves; `level_features_are_detected` (tests.rs) guards the
//! strings against the CPU, and they must be re-checked on every fearless_simd upgrade. If it
//! adds one, nothing breaks: its ops just stop inlining here, which `asm_check.py` flags.
//! RFC 3525 (struct target features) would make this whole file unnecessary.

use crate::common::LINE;
use fearless_simd::{Avx2, Avx512, Simd, SimdInt, u64x4, u64x8};
use std::arch::x86_64::{_mm_clflushopt, _mm_mfence};

/// Proof that the CPU has CLFLUSHOPT (CPUID.(EAX=7,ECX=0):EBX[23]).
#[derive(Clone, Copy, Debug)]
pub struct Clflushopt {
    _private: (),
}

impl Clflushopt {
    pub fn try_new() -> Option<Self> {
        crate::common::detect().clflushopt.then_some(Self { _private: () })
    }
}

/// fearless_simd 1.0.0's exact feature lists, plus `clflushopt` (checked by tests.rs).
#[cfg(test)]
pub const AVX2_CLFLUSHOPT: &str =
    "fxsr,avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,xsave,clflushopt";
#[cfg(test)]
pub const AVX512_CLFLUSHOPT: &str = "fxsr,adx,aes,avx512bitalg,avx512bw,avx512cd,avx512dq,avx512f,\
    avx512ifma,avx512vbmi,avx512vbmi2,avx512vl,avx512vnni,avx512vpopcntdq,bmi1,bmi2,cmpxchg16b,fma,\
    gfni,lzcnt,movbe,pclmulqdq,popcnt,rdrand,rdseed,sha,vaes,vpclmulqdq,xsave,xsavec,xsaveopt,xsaves,\
    clflushopt";

/// One entry fn per level for a generic body taking `(S, Clflushopt, args...)`; `$body` is the
/// fully instantiated path, e.g. `wflush_body::<Avx512, u64x8<Avx512>>`.
/// `#[target_feature]` needs a literal, so the feature string is repeated here, matching the
/// consts above (tests.rs checks they agree).
macro_rules! with_clflushopt {
    (avx2, $name:ident, $body:path, ($($arg:ident: $ty:ty),*) $(-> $ret:ty)?) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name(t: Avx2, cf: Clflushopt, $($arg: $ty),*) $(-> $ret)? {
            #[inline]
            #[target_feature(enable = "fxsr,avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,xsave,clflushopt")]
            fn inner(t: Avx2, cf: Clflushopt, $($arg: $ty),*) $(-> $ret)? {
                $body(t, cf, $($arg),*)
            }
            // SAFETY: `t` proves fearless_simd's Avx2 feature list, `cf` proves clflushopt, and
            // `inner` enables exactly their union.
            unsafe { inner(t, cf, $($arg),*) }
        }
    };
    (avx512, $name:ident, $body:path, ($($arg:ident: $ty:ty),*) $(-> $ret:ty)?) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name(t: Avx512, cf: Clflushopt, $($arg: $ty),*) $(-> $ret)? {
            #[inline]
            #[target_feature(enable = "fxsr,adx,aes,avx512bitalg,avx512bw,avx512cd,avx512dq,avx512f,avx512ifma,avx512vbmi,avx512vbmi2,avx512vl,avx512vnni,avx512vpopcntdq,bmi1,bmi2,cmpxchg16b,fma,gfni,lzcnt,movbe,pclmulqdq,popcnt,rdrand,rdseed,sha,vaes,vpclmulqdq,xsave,xsavec,xsaveopt,xsaves,clflushopt")]
            fn inner(t: Avx512, cf: Clflushopt, $($arg: $ty),*) $(-> $ret)? {
                $body(t, cf, $($arg),*)
            }
            // SAFETY: as above, for the Avx512 list.
            unsafe { inner(t, cf, $($arg),*) }
        }
    };
}

/// Mixed write + flush per line, using the CLFLUSHOPT *intrinsic*. Generic over width; the
/// `Clflushopt` argument is the proof, unused at runtime.
#[inline(always)]
fn wflush_body<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, _cf: Clflushopt, buf: &mut [u64], pat: u64) {
    let p = V::splat(simd, pat);
    for line in buf.as_chunks_mut::<{ LINE / 8 }>().0 {
        for c in line.chunks_exact_mut(V::LEN) {
            p.store_slice(c);
        }
        // SAFETY: `line` is inside `buf`; CLFLUSHOPT is proven by `_cf`.
        unsafe { _mm_clflushopt(line.as_ptr() as *const u8) };
    }
    // SAFETY: SSE2 is in every x86_64 baseline.
    unsafe { _mm_mfence() };
}

/// Standalone range flush (TMR's `flush_range_to_dram`) through the token.
#[inline(always)]
fn flush_body<S: Simd>(_simd: S, _cf: Clflushopt, buf: &mut [u64]) {
    let p = buf.as_ptr() as *const u8;
    for i in 0..(buf.len() * 8).div_ceil(LINE) {
        // SAFETY: inside `buf`; CLFLUSHOPT is proven by `_cf`.
        unsafe { _mm_clflushopt(p.add(i * LINE)) };
    }
    // SAFETY: SSE2 is in every x86_64 baseline.
    unsafe { _mm_mfence() };
}

with_clflushopt!(avx2, k_wflush_fscap_256, wflush_body::<Avx2, u64x4<Avx2>>, (buf: &mut [u64], pat: u64));
with_clflushopt!(avx512, k_wflush_fscap_512, wflush_body::<Avx512, u64x8<Avx512>>, (buf: &mut [u64], pat: u64));
with_clflushopt!(avx2, k_flush_fscap, flush_body::<Avx2>, (buf: &mut [u64]));
