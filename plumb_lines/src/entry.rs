//! `with_clflushopt!` (nightly): a fearless kernel entry point whose target features are the
//! level's plus `clflushopt`, so `_mm_clflushopt` inlines inside fearless loops.
//!
//! fearless_simd's tokens can't carry CLFLUSHOPT (it isn't in any x86-64 level, and `kernel!`
//! only enables its six audited feature lists). Called from a fearless kernel, the intrinsic is
//! therefore an out-of-line call per line. This macro emits an entry fn whose
//! `#[target_feature]` is fearless_simd 1.0.0's exact list for the level plus `clflushopt`,
//! around a generic `#[inline(always)]` body. Inside it both fearless's ops and the intrinsic
//! inline (TODO 84: the same 8x-unrolled loop as TMR's `flush_range_to_dram`).
//!
//! **Coupling**: the feature lists are copied from fearless_simd 1.0.0 (pinned with `=` in the
//! workspace). Enabling a feature its token doesn't prove would be unsound, so the lists are
//! exported as [`AVX2_FEATURES`]/[`AVX512_FEATURES`] and a test checks them against the CPU and
//! against the macro text. Re-check on every fearless_simd upgrade. The language-level fix is
//! struct target features (rust-lang/rfcs#3525).
//!
//! The token types are named through `$crate` (plumb_lines' own fearless_simd), so a call site
//! that shadows `fearless_simd` can't pass a forged token, and the token is always the one the
//! copied feature lists belong to.
//!
//! The expansion lands in *your* crate, so your crate needs
//! `#![feature(clflushopt_target_feature)]` (and `simd_x86_clflushopt` to call the intrinsic).
//!
//! ```
//! #![feature(clflushopt_target_feature, simd_x86_clflushopt)]
//! use fearless_simd::prelude::*;
//! use fearless_simd::{Avx512, Level, u64x8};
//! use plumb_lines::{Clflushopt, Line};
//!
//! /// Write each line and flush it. Inside the entry, `_mm_clflushopt` inlines.
//! #[inline(always)]
//! fn body<S: Simd, V: SimdInt<S, Element = u64>>(simd: S, _cf: Clflushopt, lines: &mut [Line]) {
//!     let p = V::splat(simd, 7);
//!     for line in lines.iter_mut() {
//!         for c in line.0.chunks_exact_mut(V::LEN) {
//!             p.store_slice(c);
//!         }
//!         // SAFETY: the entry enables clflushopt and `_cf` proves the CPU has it.
//!         unsafe { core::arch::x86_64::_mm_clflushopt(line as *const Line as *const u8) };
//!     }
//!     plumb_lines::mfence();
//! }
//! plumb_lines::with_clflushopt!(avx512, pub fn entry512 = body::<Avx512, u64x8<Avx512>>, (lines: &mut [Line]));
//!
//! let (Some(t), Some(cf)) = (Level::new().as_avx512(), Clflushopt::try_new()) else { return };
//! let mut lines = vec![Line::default(); 64];
//! entry512(t, cf, &mut lines);
//! assert!(lines.iter().all(|l| *l == Line([7; 8])));
//! ```

/// A token from anywhere but plumb_lines' own fearless_simd is rejected:
///
/// ```compile_fail
/// #![feature(clflushopt_target_feature)]
/// mod fearless_simd {
///     #[derive(Clone, Copy)]
///     pub struct Avx512; // a forged "proof"
/// }
/// #[inline(always)]
/// fn body(_t: plumb_lines::__fearless_simd::Avx512, _cf: plumb_lines::Clflushopt) {}
/// plumb_lines::with_clflushopt!(avx512, fn entry = body, ());
/// entry(fearless_simd::Avx512, plumb_lines::Clflushopt::try_new().unwrap());
/// ```
pub const _FORGED_TOKEN_IS_REJECTED: () = ();

/// fearless_simd 1.0.0's `Avx2` target features plus `clflushopt` (must equal the macro text).
pub const AVX2_FEATURES: &str = "fxsr,avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,xsave,clflushopt";
/// fearless_simd 1.0.0's `Avx512` target features plus `clflushopt` (must equal the macro text).
pub const AVX512_FEATURES: &str = "fxsr,adx,aes,avx512bitalg,avx512bw,avx512cd,avx512dq,avx512f,avx512ifma,avx512vbmi,avx512vbmi2,avx512vl,avx512vnni,avx512vpopcntdq,bmi1,bmi2,cmpxchg16b,fma,gfni,lzcnt,movbe,pclmulqdq,popcnt,rdrand,rdseed,sha,vaes,vpclmulqdq,xsave,xsavec,xsaveopt,xsaves,clflushopt";

/// Build a fearless kernel entry point with `clflushopt` added to the level's target features.
/// `$body` is the fully instantiated generic body, called as `$body(token, clflushopt, args..)`.
/// See the [module docs](crate::entry) (nightly feature).
#[macro_export]
macro_rules! with_clflushopt {
    (avx2, $vis:vis fn $name:ident = $body:path, ($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?) => {
        $vis fn $name(t: $crate::__fearless_simd::Avx2, cf: $crate::Clflushopt, $($arg: $ty),*) $(-> $ret)? {
            #[inline]
            #[target_feature(enable = "fxsr,avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,xsave,clflushopt")]
            fn inner(t: $crate::__fearless_simd::Avx2, cf: $crate::Clflushopt, $($arg: $ty),*) $(-> $ret)? {
                $body(t, cf, $($arg),*)
            }
            // SAFETY: `t` proves fearless_simd's Avx2 feature list, `cf` proves clflushopt, and
            // `inner` enables exactly their union.
            unsafe { inner(t, cf, $($arg),*) }
        }
    };
    (avx512, $vis:vis fn $name:ident = $body:path, ($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?) => {
        $vis fn $name(t: $crate::__fearless_simd::Avx512, cf: $crate::Clflushopt, $($arg: $ty),*) $(-> $ret)? {
            #[inline]
            #[target_feature(enable = "fxsr,adx,aes,avx512bitalg,avx512bw,avx512cd,avx512dq,avx512f,avx512ifma,avx512vbmi,avx512vbmi2,avx512vl,avx512vnni,avx512vpopcntdq,bmi1,bmi2,cmpxchg16b,fma,gfni,lzcnt,movbe,pclmulqdq,popcnt,rdrand,rdseed,sha,vaes,vpclmulqdq,xsave,xsavec,xsaveopt,xsaves,clflushopt")]
            fn inner(t: $crate::__fearless_simd::Avx512, cf: $crate::Clflushopt, $($arg: $ty),*) $(-> $ret)? {
                $body(t, cf, $($arg),*)
            }
            // SAFETY: as above, for the Avx512 list.
            unsafe { inner(t, cf, $($arg),*) }
        }
    };
}
