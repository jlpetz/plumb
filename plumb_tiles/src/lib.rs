//! Safe wrappers for the x86 2D tile instruction sets: Intel AMX now, shaped so the x86 ACE
//! extension can join later.
//!
//! Built for a memory tester first. One `TILELOADD` or `TILESTORED` moves 16 rows of 64 bytes
//! at any byte stride, an access shape no vector instruction has, so the memory operations are
//! the main feature. The AMX dot-product instructions are here too, mainly as a dense
//! power/heat load.
//!
//! Stable Rust: every AMX instruction is an `asm!` block with the tile number as a `const`
//! operand, so no target feature or nightly intrinsic is involved (an ACE backend can't keep all
//! of that; see [ACE roadmap](#ace-roadmap)). x86_64 with 64-bit pointers only (see
//! [Platforms](#platforms)); on other targets only the shape constants are defined.
//!
//! # Model
//!
//! 1. **Capability tokens** prove what the CPU and OS support. [`Amx`] covers the tile
//!    architecture (sessions, memory ops, zeroing); [`AmxInt8`], [`AmxBf16`] and [`AmxFp16`]
//!    cover the compute instructions. Get them from [`Amx::try_new`] and then
//!    [`int8`](Amx::int8), [`bf16`](Amx::bf16) and [`fp16`](Amx::fp16).
//! 2. **A session**: [`Amx::with_tiles`] configures the tiles (`LDTILECFG`), runs a closure with
//!    a [`Tiles`] handle and releases them (`TILERELEASE`) when the closure returns or unwinds.
//! 3. **Tile registers are types**, [`T0`]..[`T7`], because the register number is part of the
//!    instruction encoding: `t.load::<T0>(buf, 64)`.
//!
//! ```
//! # #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))] {
//! use plumb_tiles::{Amx, T0, T1};
//!
//! let Some(amx) = Amx::try_new() else { return };
//! // 16 pages; a 4096-byte stride puts row r of the tile at the start of page r.
//! let src: Vec<u64> = (0..16 * 512).collect();
//! let mut dst = vec![0u64; 16 * 512];
//! amx.with_tiles(|t| {
//!     t.load_u64::<T0>(&src, 4096);
//!     t.store_u64::<T0>(&mut dst, 4096);
//!     t.zero::<T1>();
//! });
//! for page in 0..16 {
//!     assert_eq!(src[page * 512..][..8], dst[page * 512..][..8]); // the first 64 bytes
//!     assert_eq!(dst[page * 512 + 8], 0); // the rest of the page untouched
//! }
//! # }
//! ```
//!
//! # One tile shape
//!
//! Every session configures all eight tiles as [`ROWS`] = 16 rows of [`ROW_BYTES`] = 64 bytes
//! ([`TILE_BYTES`] = 1 KiB). That's AMX's largest shape and ACE's only one. AMX's smaller
//! shapes are deliberately not offered: with one shape a tile op's footprint is a constant, and
//! every op is bounds-checked against it. The shape constants, the tile names and
//! [`zero`](Tiles::zero) carry over to ACE unchanged; the memory ops and the `TDP*` instructions
//! don't exist there (see [AMX and ACE](#amx-and-ace)).
//!
//! # Memory operations (AMX only)
//!
//! [`Tiles::load`], [`Tiles::load_t1`] and [`Tiles::store`] take a byte slice and the byte
//! `stride` between rows: row `r` is `buf[r * stride..][..64]`. The slice must hold
//! [`tile_span(stride)`](tile_span) = `15 * stride + 64` bytes, or the call panics before the
//! instruction runs. Any stride is allowed and there is no alignment requirement: 64 is a packed
//! 1 KiB block, 4096 touches one line in each of 16 pages, 0 reads one row 16 times. The `_u64`
//! variants take `u64` buffers (stride still in bytes).
//!
//! # Compute
//!
//! The `TDP*` instructions read each tile as 16 rows of 16 dwords. With `C` the accumulator and
//! `A`, `B` the sources:
//!
//! ```text
//! C[m][n] += sum over k < 16 of dot(A[m][k], B[k][n])      m, n < 16
//! ```
//!
//! where `dot` multiplies the 4 bytes (INT8) or 2 halves (BF16, FP16) of two dwords pairwise
//! and adds them up; `C` is i32 (wrapping) or f32. In matrix terms `A` is a row-major 16 x 64
//! INT8 (16 x 32 BF16/FP16) matrix and `B` holds the 64 x 16 (32 x 16) right-hand matrix
//! VNNI-packed: logical element `B[4k + i][n]` sits at byte `4n + i` of tile row `k` (for 16-bit
//! types, `B[2k + i][n]` at element `2n + i`). `C`, `A` and `B` must be three different tiles
//! (the ISA raises `#UD` otherwise), which is checked at compile time; [`Tiles`] shows the
//! error. The check fires when the call is monomorphized, so `cargo build` reports it and
//! `cargo check` may not.
//!
//! # Why the tile operations are safe
//!
//! A tile instruction is only defined while the tiles are configured, and the configuration
//! decides how many bytes a load or store touches. The session pins both down:
//!
//! - A [`Tiles`] exists only inside [`with_tiles`](Amx::with_tiles), after `LDTILECFG` has run
//!   with this crate's one configuration, and the closure only gets `&mut Tiles`, which can't
//!   escape it.
//! - `TILERELEASE` runs from a drop guard, so a panic in the closure releases the tiles too.
//! - Tile state is per thread, and `Tiles` is neither `Send` nor `Sync`. A session never starts
//!   on top of a live configuration, which it would overwrite and then release: a second
//!   `with_tiles` on the same thread panics (a per-thread flag), and so does a session on a
//!   thread whose tiles anything else configured, such as another copy of this crate or another
//!   AMX library. That second check reads the live configuration with `STTILECFG`, so it
//!   doesn't depend on any state of this crate.
//! - Memory ops are bounds-checked against the fixed shape with overflow-checked arithmetic.
//!   That matters beyond off-by-one: the CPU adds the stride as a *signed* index, so a stride
//!   that wrapped would address memory below the slice.
//! - Compute ops need their token, and naming a tile twice (which the ISA rejects with `#UD`)
//!   doesn't compile.
//!
//! The compile-time parts (the handle can't leave the closure, tokens can't be built by hand,
//! a repeated tile is rejected) have `compile_fail` tests.
//!
//! What remains the caller's job: code that runs its own `LDTILECFG`/`TILERELEASE` (another
//! AMX library, raw intrinsics or `asm!`) must not run *inside* `with_tiles`, because the entry
//! check can't see what happens mid-session. If it does, the session's next tile op either
//! touches less than was bounds-checked (on every AMX CPU so far, palette 1 can't be
//! configured larger than the 16 x 64 used here) or, if the tiles were released, raises `#UD`
//! and the process dies.
//!
//! # Platforms
//!
//! - **Windows** 11 / Server 2022+: nothing to do per thread. Once XCR0 shows the tile state,
//!   a plain thread can run tile instructions (checked on Server 2025, Xeon 6975P-C).
//! - **Linux**: tile data is opt-in per process. [`Amx::try_new`] asks for it with
//!   `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`, a raw syscall (no libc). That path
//!   is compile-checked only (`cargo check --target x86_64-unknown-linux-gnu`); it hasn't run
//!   on Linux hardware yet.
//! - **Other OSes**: the CPUID and XCR0 checks only; untested.
//! - **x32** (`x86_64-unknown-linux-gnux32`, 32-bit pointers on x86_64) gets only the shape
//!   constants, like non-x86 targets. A 32-bit pointer or `usize` in an `asm!` register operand
//!   leaves the upper half of the 64-bit register the instruction addresses through undefined,
//!   so a bounds-checked load or store could still land anywhere.
//!
//! # AMX and ACE
//!
//! ACE (the x86 Ecosystem Advisory Group's AI Compute Extensions, v1.15 of 2026-05-15) reuses
//! AMX's tile registers and management instructions but not its data path:
//!
//! | | AMX (palette 1) | ACE v1 (palette 2) |
//! |---|---|---|
//! | Tile registers | 8, each up to 16 rows x 64 B (this crate: always 16 x 64) | 8, fixed at 16 rows x 64 B |
//! | Configuration | `LDTILECFG` with rows/colsb per tile | `LDTILECFG` with only `palette = 2` |
//! | Tile <-> memory | `TILELOADD`, `TILELOADDT1`, `TILESTORED`, any byte stride | none (`#UD` under palette 2) |
//! | Getting data in and out | the memory ops; parts with AMX-AVX512 can also read rows into ZMM (`TILEMOVROW`, `TCVTROW*`) | through ZMM only: `TILEMOVROW` (row read/write), `TILEMOVCOL` (column write), `TCVTROW*` (row read with conversion) |
//! | Role of the tiles | operands and accumulator: a `TDP*` names 3 tiles, so a 2x2-blocked kernel has 4 accumulators | accumulators only: operands are ZMM registers, so all 8 tiles can accumulate |
//! | Compute | dot products `TDP*`, tile x tile into a tile | outer products `TOP*`, zmm x zmm into a tile |
//! | Input types | INT8 (s/u), BF16, FP16; complex FP16, FP8, TF32 on later parts | INT8 (s/u), BF16, MX FP8 (E4M3/E5M2), MXINT8; no FP16 |
//! | Accumulators | INT32, FP32 | INT32, FP32 |
//! | Extra state | none | Block Scale Register (1024 bit, XCR0 bit 20) for the MX formats |
//! | Prerequisites | none beyond AMX-TILE | AVX10.1 + AVX10 aux, AVX-512 state |
//! | Detection | CPUID `7.0:EDX[24]`, palette 1 present in leaf `1Dh.1` (an ACE-only part sets AMX-TILE too, with palette 1 zeroed), `XCR0[18:17]`; compute: the type's CPUID bit and the TMUL limits in leaf `1Eh` | AMX-TILE, CPUID `7.1:ECX[11]`, ACE_VSN >= 1 (`1Dh.2:EAX[7:0]`), `XCR0[20,18:17]`, `XCR0[7:5]` |
//! | Compute feature bits | one per type: AMX-INT8, AMX-BF16, AMX-FP16 | none per type: ACE / ACE_VSN enumerates every v1 outer product and BSR op |
//!
//! ACE's narrow data types come from block-scaled (MX) formats, not from smaller tiles, so its
//! one tile shape is no loss there.
//!
//! ## ACE roadmap
//!
//! Not implemented: there is no hardware, no assembler support (LLVM PRs #208408 and #208706
//! are open) and nothing in rustc. The plan:
//!
//! | This crate (AMX) | ACE backend (later) |
//! |---|---|
//! | [`Amx`], [`AmxInt8`], [`AmxBf16`], [`AmxFp16`] | one `Ace` token (the ACE detection above) for all of ACE v1, since v1 has no per-type feature bits; a later ACE_VSN may add tokens |
//! | [`Amx::with_tiles`]: palette 1, all 16 x 64, hands out [`Tiles`] | `Ace::with_tiles`: palette 2, hands out its own handle (say `AceTiles<'_>`), because `Tiles`' memory and `TDP*` methods are `#UD` under palette 2. Same entry checks (the per-thread flag and the `STTILECFG` palette byte), so an AMX and an ACE session can't overlap on a thread (one palette at a time; GCC warns against mixing them) |
//! | [`T0`]..[`T7`], [`ROWS`], [`ROW_BYTES`], [`TILE_BYTES`] | unchanged |
//! | [`Tiles::zero`] | `AceTiles::zero`, the same `TILEZERO` |
//! | `load`, `load_t1`, `store` | AMX only; ACE moves rows through ZMM (`TILEMOVROW`, `TILEMOVCOL`, `TCVTROW*`) |
//! | (later, with AMX-AVX512) row reads into ZMM | the same `TILEMOVROW` (read) and `TCVTROW*`, valid under both palettes, so designed once: the same methods on both handles, or a shared sealed trait |
//! | `dpb**d`, `dpbf16ps` | `top4b**d`, `top2bf16ps`, with ZMM operands |
//! | `dpfp16ps` | no counterpart: ACE v1 has no FP16 outer product |
//! | (none) | `top4mx*` (MX FP8, MXINT8) and the Block Scale Register it reads (`BSRINIT`, `BSRMOV*`) |
//!
//! ZMM operands are where ACE can't follow the AMX model of single-instruction
//! `#[inline(always)]` `asm!` blocks with no target feature. An `asm!` operand in a ZMM register
//! (`zmm_reg`, or a named `zmm0`) needs `avx512f` enabled at the block, and `#[target_feature]`
//! can't be combined with `#[inline(always)]`. Two ways round it:
//!
//! 1. **Through memory**: the method takes the vector by reference (or by value, which the
//!    compiler then spills), and the `asm!` block loads it into an explicitly clobbered ZMM
//!    register before the ACE op. That needs no target feature and stays `#[inline(always)]`,
//!    at the cost of a store and reload per operand; and since the compiler doesn't know the
//!    block dirtied the upper ZMM state, `vzeroupper` becomes the block's job.
//! 2. **AVX-512 callers**: safe `#[inline(always)]` methods, licensed by the `Ace` token (ACE
//!    requires AVX10.1, so the token implies AVX-512), call `#[target_feature]` `#[inline]`
//!    helpers that hold the ZMM operands. Inside an AVX-512 kernel (fearless_simd's `kernel!`,
//!    a `#[simd]` function at an AVX-512 level, or any `#[target_feature]` function with those
//!    features) they inline to the bare instruction; anywhere else each is an out-of-line call.
//!
//! The plan is 2, with 1 as the fallback for callers without an AVX-512 entry point. Until an
//! assembler knows the ACE mnemonics they will be `.byte` encodings with fixed ZMM registers,
//! which have the same target-feature requirement.
//!
//! Also not here yet: AMX-COMPLEX (`TCMMIMFP16PS`, `TCMMRLFP16PS`; this crate's test machine
//! lacks it), and the later AMX extensions (AMX-FP8, AMX-TF32, AMX-MOVRS, and AMX-AVX512,
//! enumerated in leaf 1Eh.1, whose row reads are shared with ACE as above). Each would be a new
//! token plus methods. stdarch has AMX intrinsics behind the unstable `x86_amx_intrinsics`
//! feature (rust-lang/rust#126622); this crate doesn't use them, so it builds on stable.

/// Rows in every tile.
pub const ROWS: usize = 16;
/// Bytes per tile row.
pub const ROW_BYTES: usize = 64;
/// Bytes per tile: [`ROWS`] x [`ROW_BYTES`] = 1 KiB.
pub const TILE_BYTES: usize = ROWS * ROW_BYTES;

/// The bytes a tile load or store at `stride` covers, from the start of row 0 to the end of
/// row 15: `(ROWS - 1) * stride + ROW_BYTES`. `None` if that overflows `usize` (no slice is
/// that long, so the access would panic).
///
/// ```
/// use plumb_tiles::{tile_span, TILE_BYTES};
/// assert_eq!(tile_span(64), Some(TILE_BYTES)); // packed rows
/// assert_eq!(tile_span(4096), Some(15 * 4096 + 64));
/// assert_eq!(tile_span(0), Some(64));
/// assert_eq!(tile_span(usize::MAX / 2), None);
/// ```
#[must_use]
pub const fn tile_span(stride: usize) -> Option<usize> {
    match stride.checked_mul(ROWS - 1) {
        Some(last_row) => last_row.checked_add(ROW_BYTES),
        None => None,
    }
}

// Everything that runs a tile instruction. `target_pointer_width` keeps out x32, whose 32-bit
// pointers and `usize` would reach `asm!` in registers the instructions read as 64-bit.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod detect;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod reg;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod session;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod token;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use reg::{T0, T1, T2, T3, T4, T5, T6, T7, TileReg};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use session::Tiles;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use token::{Amx, AmxBf16, AmxFp16, AmxInt8};

#[cfg(all(doctest, target_arch = "x86_64", target_pointer_width = "64"))]
mod compile_fail;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_span_matches_the_shape_and_rejects_overflow() {
        assert_eq!(TILE_BYTES, 1024);
        assert_eq!(tile_span(0), Some(ROW_BYTES));
        assert_eq!(tile_span(1), Some(15 + 64));
        assert_eq!(tile_span(ROW_BYTES), Some(TILE_BYTES));
        assert_eq!(tile_span(72), Some(15 * 72 + 64));
        // The largest stride whose span still fits, and the first that doesn't.
        let max = (usize::MAX - ROW_BYTES) / (ROWS - 1);
        assert_eq!(tile_span(max), Some(max * 15 + 64));
        assert_eq!(tile_span(max + 1), None);
        assert_eq!(tile_span(usize::MAX / 2), None);
        assert_eq!(tile_span(usize::MAX), None);
    }
}
