<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

Nothing has been released to crates.io yet. The first releases will be `plumb_lines` 0.1.0 and
`plumb_tiles` 0.1.0.

## [Unreleased]

This release has an [MSRV][] of 1.89.

### Added

#### `plumb_lines`

- Added `as_vectors` and `as_vectors_mut`: split any slice into an unaligned head, a slice of
  aligned fearless_simd vectors, and a tail, so loops over the middle are pointer walks.
- Added `nontemporal`, a closure scope for non-temporal stores of any fearless_simd vector at any
  x86 level. It ends with `SFENCE`, including on unwind. Its writer is by value, write-once and
  `!Send`, so safe code can't store a slot twice, read it, or publish it before the fence.
  `NtWriter::fill_with` issues each group of four stores from one `asm!` block, so the offsets
  are displacements rather than a `lea` per store.
- Added `Clflushopt` (a CPUID-checked token with the flush line size) and `flush_after`, a scope
  that flushes everything it lent out and then runs `MFENCE`.
- Added `Movdir64b` and `direct`, a closure scope for MOVDIR64B 64-byte direct stores ending with
  `SFENCE`.
- Added the `nightly` feature: range flushes use stdarch's `_mm_clflushopt`, and
  `with_clflushopt!` builds a fearless kernel entry point with `clflushopt` added to the level's
  target features.

#### `plumb_tiles`

- Added `Amx`, `AmxInt8`, `AmxBf16` and `AmxFp16` tokens, checked against CPUID, XCR0, the
  palette 1 shape and the TMUL limits.
- Added `Amx::with_tiles`, a per-thread session with all eight tiles configured as 16 rows x 64
  bytes, and `Tiles` with bounds-checked `load`, `load_t1`, `store` and `zero`, plus the AMX dot
  products. Repeated tiles in a dot product are rejected at compile time.
- Added `with_tiles` on `AmxInt8`, `AmxBf16` and `AmxFp16`: the same session, which with the
  `nightly` feature also enables that compute set.
- Added the `nightly` feature: the tile instructions use stdarch's AMX intrinsics
  (`x86_amx_intrinsics`) instead of `asm!`, with the same API. Each session runs inside a
  function that enables the AMX features its token proves, so the intrinsics inline into the
  session closure.

[Unreleased]: https://github.com/jlpetz/plumb/commits/main
[MSRV]: README.md#minimum-supported-rust-version-msrv
