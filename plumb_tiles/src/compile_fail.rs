//! `compile_fail` tests for the compile-time half of the crate's safety argument ("Why the tile
//! operations are safe"). Built only for doctests. Each block differs from working code (the
//! crate examples) in the one thing it tests, and names the error code where rustc has one, so
//! a block can't pass by failing for some other reason.
//!
//! The `Tiles` handle can't leave its session: not stored outside the closure,
//!
//! ```compile_fail,E0521
//! let amx = plumb_tiles::Amx::try_new().unwrap();
//! let mut leak = None;
//! amx.with_tiles(|t| leak = Some(t));
//! ```
//!
//! nor returned from it (a lifetime error, which has no code):
//!
//! ```compile_fail
//! let amx = plumb_tiles::Amx::try_new().unwrap();
//! let t = amx.with_tiles(|t| t);
//! ```
//!
//! Tokens come only from detection or `unsafe`, never from a struct literal:
//!
//! ```compile_fail,E0451
//! let amx = plumb_tiles::Amx { _proof: () };
//! ```
//!
//! ```compile_fail,E0451
//! let amx = plumb_tiles::Amx::try_new().unwrap();
//! let int8 = plumb_tiles::AmxInt8 { amx };
//! ```
//!
//! ```compile_fail,E0451
//! let amx = plumb_tiles::Amx::try_new().unwrap();
//! let bf16 = plumb_tiles::AmxBf16 { amx };
//! ```
//!
//! ```compile_fail,E0451
//! let amx = plumb_tiles::Amx::try_new().unwrap();
//! let fp16 = plumb_tiles::AmxFp16 { amx };
//! ```
//!
//! A `TDP*` op rejects every repeated tile, not just `C == A` (shown on `Tiles`): `C == B`,
//!
//! ```compile_fail,E0080
//! use plumb_tiles::{Amx, T0, T1};
//! let amx = Amx::try_new().unwrap();
//! let int8 = amx.int8().unwrap();
//! amx.with_tiles(|t| t.dpbssd::<T0, T1, T0>(int8));
//! ```
//!
//! `A == B`,
//!
//! ```compile_fail,E0080
//! use plumb_tiles::{Amx, T0, T1};
//! let amx = Amx::try_new().unwrap();
//! let int8 = amx.int8().unwrap();
//! amx.with_tiles(|t| t.dpbssd::<T0, T1, T1>(int8));
//! ```
//!
//! and the same in the other instruction families:
//!
//! ```compile_fail,E0080
//! use plumb_tiles::{Amx, T2, T3};
//! let amx = Amx::try_new().unwrap();
//! let bf16 = amx.bf16().unwrap();
//! amx.with_tiles(|t| t.dpbf16ps::<T3, T2, T2>(bf16));
//! ```
//!
//! ```compile_fail,E0080
//! use plumb_tiles::{Amx, T4, T5};
//! let amx = Amx::try_new().unwrap();
//! let fp16 = amx.fp16().unwrap();
//! amx.with_tiles(|t| t.dpfp16ps::<T5, T5, T4>(fp16));
//! ```
