// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Nightly: stdarch's AMX intrinsics (`x86_amx_intrinsics`, rust-lang/rust#126622), in the
//! forms that name the tile register (`_tile_loadd::<N>`). The `__tile1024i` forms let LLVM
//! pick the registers and configure the tiles itself, which can't share a session that
//! configures all eight once.
//!
//! The register number reaches each intrinsic through the register's sealed impl
//! ([`with_reg`](crate::reg::Sealed::with_reg)): one [`WithReg`] hop per tile operand.
//!
//! Each intrinsic has `#[target_feature(enable = "amx-…")]`, so it inlines only into code
//! compiled with that feature. The sessions provide it ([`Amx::with_tiles`](crate::Amx::with_tiles) and
//! the compute tokens' `with_tiles`); anywhere else it is an out-of-line call.
//!
//! LLVM declares every AMX intrinsic without memory attributes, so it treats each as a call
//! that may read and write any memory: tile instructions are never dropped or reordered with
//! each other or with other memory accesses. That is stricter than the `asm!` forms' options.

use core::arch::x86_64 as arch;
use core::marker::PhantomData;

use crate::reg::{TileReg, WithReg};

/// `LDTILECFG`: configures the tiles from the 64-byte descriptor at `cfg`.
#[inline(always)]
pub(crate) unsafe fn ldtilecfg(cfg: *const u8) {
    // SAFETY (caller): the CPU has AMX-TILE with OS-enabled tile state, and `cfg` is 64
    // readable bytes of a descriptor it accepts.
    unsafe { arch::_tile_loadconfig(cfg) }
}

/// `STTILECFG`: writes this thread's live tile configuration (all zero when released) to the
/// 64 bytes at `cfg`.
#[inline(always)]
pub(crate) unsafe fn sttilecfg(cfg: *mut u8) {
    // SAFETY (caller): the CPU has AMX-TILE with OS-enabled tile state, and `cfg` is 64
    // writable bytes.
    unsafe { arch::_tile_storeconfig(cfg) }
}

/// `TILERELEASE`: returns the tiles to their initial, unconfigured state.
#[inline(always)]
pub(crate) unsafe fn tilerelease() {
    // SAFETY (caller): the CPU has AMX-TILE. TILERELEASE is defined configured or not.
    unsafe { arch::_tile_release() }
}

/// `TILELOADD` (source, stride) on the register it runs with.
struct Load(*const u8, usize);
/// `TILELOADDT1` (source, stride) on the register it runs with.
struct LoadT1(*const u8, usize);
/// `TILESTORED` (destination, stride) on the register it runs with.
struct Store(*mut u8, usize);

impl WithReg for Load {
    #[inline(always)]
    unsafe fn run<const N: i32>(self) {
        // SAFETY (caller): as tileloadd.
        unsafe { arch::_tile_loadd::<N>(self.0, self.1) }
    }
}

impl WithReg for LoadT1 {
    #[inline(always)]
    unsafe fn run<const N: i32>(self) {
        // SAFETY (caller): as tileloaddt1.
        unsafe { arch::_tile_stream_loadd::<N>(self.0, self.1) }
    }
}

impl WithReg for Store {
    #[inline(always)]
    unsafe fn run<const N: i32>(self) {
        // SAFETY (caller): as tilestored.
        unsafe { arch::_tile_stored::<N>(self.0, self.1) }
    }
}

/// `TILELOADD`: loads tile `T`, row `r` from `src + r * stride`.
#[inline(always)]
pub(crate) unsafe fn tileloadd<T: TileReg>(src: *const u8, stride: usize) {
    // SAFETY (caller): src + r*stride .. +ROW_BYTES is readable for r < ROWS; tiles configured.
    unsafe { T::with_reg(Load(src, stride)) }
}

/// `TILELOADDT1`: [`tileloadd`] with the T1 (low reuse) hint.
#[inline(always)]
pub(crate) unsafe fn tileloaddt1<T: TileReg>(src: *const u8, stride: usize) {
    // SAFETY (caller): as tileloadd.
    unsafe { T::with_reg(LoadT1(src, stride)) }
}

/// `TILESTORED`: stores tile `T`, row `r` to `dst + r * stride`.
#[inline(always)]
pub(crate) unsafe fn tilestored<T: TileReg>(dst: *mut u8, stride: usize) {
    // SAFETY (caller): dst + r*stride .. +ROW_BYTES is writable for r < ROWS; tiles configured.
    unsafe { T::with_reg(Store(dst, stride)) }
}

/// `TILEZERO` on the register it runs with.
struct Zero;

impl WithReg for Zero {
    #[inline(always)]
    unsafe fn run<const N: i32>(self) {
        // SAFETY (caller): as tilezero.
        unsafe { arch::_tile_zero::<N>() }
    }
}

/// `TILEZERO`: sets every byte of tile `T` to zero.
#[inline(always)]
pub(crate) unsafe fn tilezero<T: TileReg>() {
    // SAFETY (caller): tiles configured.
    unsafe { T::with_reg(Zero) }
}

/// A `TDP*` instruction, once its three tile numbers are known.
trait Tdp {
    /// # Safety
    ///
    /// As the instruction's wrapper below.
    unsafe fn run<const C: i32, const A: i32, const B: i32>();
}

/// First hop of a `TDP*`: runs with `C`'s number and resolves `A`.
struct TdpC<Op, A, B>(PhantomData<(Op, A, B)>);
/// Second hop: `C` known, resolves `B`.
struct TdpA<Op, B, const C: i32>(PhantomData<(Op, B)>);
/// Last hop: `C` and `A` known, runs the instruction with `B`.
struct TdpB<Op, const C: i32, const A: i32>(PhantomData<Op>);

impl<Op: Tdp, A: TileReg, B: TileReg> WithReg for TdpC<Op, A, B> {
    #[inline(always)]
    unsafe fn run<const C: i32>(self) {
        // SAFETY (caller): as the instruction's.
        unsafe { A::with_reg(TdpA::<Op, B, C>(PhantomData)) }
    }
}

impl<Op: Tdp, B: TileReg, const C: i32> WithReg for TdpA<Op, B, C> {
    #[inline(always)]
    unsafe fn run<const A: i32>(self) {
        // SAFETY (caller): as the instruction's.
        unsafe { B::with_reg(TdpB::<Op, C, A>(PhantomData)) }
    }
}

impl<Op: Tdp, const C: i32, const A: i32> WithReg for TdpB<Op, C, A> {
    #[inline(always)]
    unsafe fn run<const B: i32>(self) {
        // SAFETY (caller): as the instruction's.
        unsafe { Op::run::<C, A, B>() }
    }
}

macro_rules! tdp {
    ($($name:ident => $op:ident, $intrinsic:ident;)*) => {$(
        enum $op {}

        impl Tdp for $op {
            #[inline(always)]
            unsafe fn run<const C: i32, const A: i32, const B: i32>() {
                // SAFETY (caller): as the wrapper's.
                unsafe { arch::$intrinsic::<C, A, B>() }
            }
        }

        #[doc = concat!("`", stringify!($name), " C, A, B`.")]
        #[inline(always)]
        pub(crate) unsafe fn $name<C: TileReg, A: TileReg, B: TileReg>() {
            // SAFETY (caller): the CPU has the instruction, the tiles are configured with
            // shapes the TMUL limits accept, and C, A and B are three different tiles.
            unsafe { C::with_reg(TdpC::<$op, A, B>(PhantomData)) }
        }
    )*};
}

tdp!(
    tdpbssd => Dpbssd, _tile_dpbssd;
    tdpbsud => Dpbsud, _tile_dpbsud;
    tdpbusd => Dpbusd, _tile_dpbusd;
    tdpbuud => Dpbuud, _tile_dpbuud;
    tdpbf16ps => Dpbf16ps, _tile_dpbf16ps;
    tdpfp16ps => Dpfp16ps, _tile_dpfp16ps;
);
