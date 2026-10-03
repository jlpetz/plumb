# plumb TODO

This project's own tracker. It is kept separate from TMR-APP's TODO.md on purpose: if this
succeeds, the result merges into TMR; if not, nothing here clouds TMR's list. One entry per item:
status, decisions, next step. Background is in [PLAN.md](PLAN.md).

### 1. Owner review of the workspace
**Status**: Ready for review (2026-10-03).
**Next**: the owner reviews `plumb_lines`, `plumb_tiles`, `bench/` and the results; then create
the public repo `jlpetz/plumb` and push.

### 2. Timing runs on an idle box
**Status**: Partly done. TODO 84's full sweep ran clean on 2026-10-03; the new plumb kernels and
ported tests need their own run.
**Next**: with the box idle, `cargo +nightly run --release -p plumb-bench` (about 15 min; 1 GiB
pages, 1-8 threads) and record the results in `bench/RESULTS.md`.

### 3. Share the NT design with Shnatsel
**Status**: Not started. He asked to see NT stores tried in an extension crate first.
**Next**: once the repo is public, post the `plumb_lines::nt` design (scoped writer, write-only
slots, SFENCE on exit, `&mut V` alignment) and the asm/parity results in the #simd topic.

### 4. clflushopt stabilization report (upstream)
**Status**: Not started. `clflushopt` target feature (rust-lang/rust#157098) and `_mm_clflushopt`
(rust-lang/stdarch#2141) landed in nightly 2026-05/06 under rust-lang/rust#157096.
**Next**: draft the stabilization report for #157096 (usage: TMR, plumb_lines); the owner posts.

### 5. MOVDIR64B in rustc, std_detect and stdarch (upstream)
**Status**: Not started. rustc rejects `#[target_feature(enable = "movdir64b")]` (LLVM-only),
std_detect doesn't know it, stdarch has no intrinsic.
**Next**: three patches modelled on the clflushopt ones (`../clflushopt-rustc.patch`,
`../clflushopt-stdarch.patch`). plumb_lines keeps `asm!` until it stabilizes.

### 6. plumb_tiles: AMX
**Status**: In progress (built and reviewed by a workflow 2026-10-03; see the crate).
**Next**: AMX kernels in the bench harness (tile load/store DRAM sweeps vs AVX-512); then the
TMR TODO 86 probe: does a 16-row x 8-byte AMX tile store beat AVX-512 scatter for TM5's
strided u64 writes? Raw asm first; only add shapes to the crate if it wins.

### 7. ACE backend
**Status**: Waiting: no hardware; LLVM PRs open (llvm-project#208408/#208706), nothing in rustc.
**Next**: re-check when hardware or rustc support appears; keep the shared tile layer
ACE-shaped meanwhile.

### 8. TMR integration and pilot
**Status**: After items 1-2.
**Next**: a `src/simd/` module in TMR-APP; pilot StuckBit behind `asm_check.py` and the
equivalence tests, then Refresh, SimpleTest, SimpleNT, MirrorMove. Updating TMR's settled
`macro_rules!` decision is part of the pilot's close-out.

### 9. Deferred cache-line ops
**Status**: Deferred (owner decision 2026-10-03). CLWB, CLZERO, CLDEMOTE, MOVDIRI, PREFETCHW:
none is faster than existing paths or adds coverage TMR needs today.
**Next**: none until a test needs one.
