//! Tile register names as types.
//!
//! The register number is encoded in the instruction (there is no "tile in a register" operand
//! the compiler could allocate), so it has to be known at compile time. A type per register lets
//! the methods take it as a type parameter, `tiles.zero::<T3>()`, and lets `const` operands
//! splice it into the `asm!` template.

mod sealed {
    pub trait Sealed {}
}

/// A tile register, `tmm0`..`tmm7`. Sealed: [`T0`]..[`T7`] are the only implementations, so
/// [`N`](TileReg::N) is always a valid register number.
pub trait TileReg: sealed::Sealed {
    /// The register number, 0..=7.
    const N: u8;
}

macro_rules! tile_regs {
    ($($name:ident = $n:literal),* $(,)?) => {$(
        #[doc = concat!("Tile register `tmm", stringify!($n), "` (a type-level name; it has no values).")]
        #[derive(Debug)]
        pub enum $name {}
        impl sealed::Sealed for $name {}
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
