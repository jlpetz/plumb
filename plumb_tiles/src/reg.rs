// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tile register names as types.
//!
//! The register number is encoded in the instruction (there is no "tile in a register" operand
//! the compiler could allocate), so it has to be known at compile time. A type per register lets
//! the methods take it as a type parameter, `tiles.zero::<T3>()`, and lets `const` operands
//! splice it into the `asm!` template.
//!
//! stdarch's intrinsics (`nightly` feature) take the number as a const generic instead, and
//! `_tile_zero::<{ T::N }>()` from a generic `T` would need `generic_const_exprs`. So each
//! register's sealed impl passes its own literal on: `Sealed::with_reg` runs a `WithReg`
//! operation with the number as a const generic, one hop per tile operand.

mod sealed {
    #[expect(
        unnameable_types,
        reason = "This is a sealed trait, so being unnameable is the entire point"
    )]
    pub trait Sealed {
        /// Runs `op` with this register's number as its const generic.
        ///
        /// # Safety
        ///
        /// Whatever `op` requires.
        #[cfg(feature = "nightly")]
        unsafe fn with_reg<Op: WithReg>(op: Op);
    }

    /// An operation that needs a tile register number as a const generic (an intrinsic's).
    #[cfg(feature = "nightly")]
    #[expect(
        unnameable_types,
        reason = "Only reachable through the sealed trait; not for use outside the crate"
    )]
    pub trait WithReg {
        /// # Safety
        ///
        /// The operation's own requirements, with `N` as the register.
        unsafe fn run<const N: i32>(self);
    }
}

pub(crate) use sealed::Sealed;
#[cfg(feature = "nightly")]
pub(crate) use sealed::WithReg;

/// A tile register, `tmm0`..`tmm7`. Sealed: [`T0`]..[`T7`] are the only implementations, so
/// [`N`](TileReg::N) is always a valid register number.
pub trait TileReg: Sealed {
    /// The register number, 0..=7.
    const N: u8;
}

macro_rules! tile_regs {
    ($($name:ident = $n:literal),* $(,)?) => {$(
        #[doc = concat!("Tile register `tmm", stringify!($n), "` (a type-level name; it has no values).")]
        #[derive(Debug)]
        pub enum $name {}
        impl Sealed for $name {
            #[cfg(feature = "nightly")]
            #[inline(always)]
            unsafe fn with_reg<Op: WithReg>(op: Op) {
                // SAFETY: the caller upholds `op`'s requirements for this register.
                unsafe { op.run::<$n>() }
            }
        }
        impl TileReg for $name {
            const N: u8 = $n;
        }
    )*};
}

tile_regs!(
    T0 = 0,
    T1 = 1,
    T2 = 2,
    T3 = 3,
    T4 = 4,
    T5 = 5,
    T6 = 6,
    T7 = 7
);
