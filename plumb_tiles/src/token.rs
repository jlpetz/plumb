// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Capability tokens: zero-sized proofs that the CPU and OS support an instruction set.
//!
//! A token can only come from runtime detection or an `unsafe` promise, so holding one is the
//! precondition every tile instruction needs. Tokens are `Copy` and `Send + Sync`: the CPU
//! features, XCR0 and the Linux permission are all process-wide.

use crate::detect;

/// Proof of the AMX tile architecture (AMX-TILE): the CPU has it, palette 1 holds the fixed
/// 16 x 64-byte shape, and the OS has enabled tile state for this process.
///
/// This is what [`with_tiles`](Amx::with_tiles) and the memory operations need. The compute
/// instructions need one of [`AmxInt8`], [`AmxBf16`] or [`AmxFp16`] as well.
#[derive(Clone, Copy, Debug)]
pub struct Amx {
    _proof: (),
}

impl Amx {
    /// Detects AMX-TILE and its OS support. On Linux this also asks the kernel for permission
    /// to use tile data (`arch_prctl(ARCH_REQ_XCOMP_PERM)`), which is process-wide.
    ///
    /// Checks `CPUID.(EAX=7,ECX=0):EDX[24]`, XCR0 bits 17-18 (via XGETBV, after
    /// `CPUID.1:ECX[27]` OSXSAVE) and the palette-1 limits in leaf 1Dh. The result is cached
    /// after the first call.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        (detect::features() & detect::TILE != 0).then_some(Self { _proof: () })
    }

    /// Creates the token without checking.
    ///
    /// # Safety
    ///
    /// The CPU must support AMX-TILE with a palette 1 of at least 8 tiles of 16 rows x 64 bytes
    /// (every AMX CPU to date), the OS must have enabled XTILECFG and XTILEDATA in XCR0, and on
    /// Linux the process must hold XTILEDATA permission (`ARCH_REQ_XCOMP_PERM`). Otherwise the
    /// first tile instruction faults (`#UD`, `#GP` or `SIGILL`).
    #[must_use]
    pub const unsafe fn assume_supported() -> Self {
        Self { _proof: () }
    }

    /// AMX-INT8 (`TDPB[SU][SU]D`), if supported.
    #[must_use]
    pub fn int8(self) -> Option<AmxInt8> {
        (detect::features() & detect::INT8 != 0).then_some(AmxInt8 { amx: self })
    }

    /// AMX-BF16 (`TDPBF16PS`), if supported.
    #[must_use]
    pub fn bf16(self) -> Option<AmxBf16> {
        (detect::features() & detect::BF16 != 0).then_some(AmxBf16 { amx: self })
    }

    /// AMX-FP16 (`TDPFP16PS`), if supported.
    #[must_use]
    pub fn fp16(self) -> Option<AmxFp16> {
        (detect::features() & detect::FP16 != 0).then_some(AmxFp16 { amx: self })
    }
}

macro_rules! compute_token {
    ($(#[$doc:meta])* $name:ident, $method:ident, $isa:literal, $cpuid:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            amx: Amx,
        }

        impl $name {
            #[doc = concat!("Detects AMX-TILE (as [`Amx::try_new`]) and ", $isa, ".")]
            #[must_use]
            pub fn try_new() -> Option<Self> {
                Amx::try_new()?.$method()
            }

            /// Creates the token without checking.
            ///
            /// # Safety
            ///
            #[doc = concat!("Everything [`Amx::assume_supported`] requires, plus ", $isa, " (",
                            $cpuid, ") and TMUL limits (leaf 1Eh) of at least K = 16 and N = 64 bytes.")]
            #[must_use]
            pub const unsafe fn assume_supported() -> Self {
                // SAFETY: the caller's promise includes Amx's.
                Self { amx: unsafe { Amx::assume_supported() } }
            }

            /// The tile-architecture token this one implies, for [`Amx::with_tiles`].
            #[must_use]
            pub const fn amx(self) -> Amx {
                self.amx
            }
        }

        impl From<$name> for Amx {
            fn from(t: $name) -> Amx {
                t.amx
            }
        }
    };
}

compute_token!(
    /// Proof of AMX-INT8: the signed/unsigned byte dot products [`Tiles::dpbssd`],
    /// [`Tiles::dpbsud`], [`Tiles::dpbusd`] and [`Tiles::dpbuud`].
    ///
    /// [`Tiles::dpbssd`]: crate::Tiles::dpbssd
    /// [`Tiles::dpbsud`]: crate::Tiles::dpbsud
    /// [`Tiles::dpbusd`]: crate::Tiles::dpbusd
    /// [`Tiles::dpbuud`]: crate::Tiles::dpbuud
    AmxInt8, int8, "AMX-INT8", "`CPUID.(EAX=7,ECX=0):EDX[25]`"
);
compute_token!(
    /// Proof of AMX-BF16: [`Tiles::dpbf16ps`](crate::Tiles::dpbf16ps).
    AmxBf16, bf16, "AMX-BF16", "`CPUID.(EAX=7,ECX=0):EDX[22]`"
);
compute_token!(
    /// Proof of AMX-FP16: [`Tiles::dpfp16ps`](crate::Tiles::dpfp16ps).
    AmxFp16, fp16, "AMX-FP16", "`CPUID.(EAX=7,ECX=1):EAX[21]`"
);
