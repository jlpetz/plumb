# plumb TODO

This project's own tracker. It is kept separate from TMR-APP's TODO.md on purpose: if this
succeeds, the result merges into TMR; if not, nothing here clouds TMR's list. One entry per item:
status, decisions, next step. Background is in [PLAN.md](PLAN.md).

### 1. Owner review of the workspace
**Status**: Ready for review (2026-10-03). Both crates built, adversarially reviewed and fixed,
and restyled to fearless_simd's (Linebender) conventions; tests and clippy clean on stable,
nightly and MSRV 1.89; asm gate 130/130 (bench), 8/8 (plumb_lines example, stable and nightly),
10/10 (plumb_tiles example). Timing done (item 2).
**Next**: the owner reviews `plumb_lines`, `plumb_tiles`, `bench/` and `bench/RESULTS.md`; then
create the public repo `jlpetz/plumb` and push (owner approved pushing after review + timing).

### 2. Timing runs on an idle box
**Status**: Done 2026-10-03 (full sweep 825c9a9, verify groups re-run at 4254f13):
`bench/RESULTS.md`. DRAM parity everywhere (98-105%); the 512-bit view verify's L2 gap found and
fixed (loop shape, not addressing); `asm!` costs and the lost unroll measured as front-end only.
**Next**: none; re-run after changes that touch a hot loop.

### 3. Share the NT design with Shnatsel
**Status**: Not started. They asked to see NT stores tried in an extension crate first.
**Next**: once the repo is public, post in the #simd topic: the `plumb_lines::nt` design (scoped
writer, write-once slots, SFENCE on exit, four stores per `asm!` block) with the asm/parity
results. Also say: (a) a correction to my earlier post, the 16% 512-bit L2 verify gap was LLVM
unrolling and reassociating the OR loop (losing the fused `vpternlogq`), not indexed addressing;
fearless's own `chunks_exact` verify shows it (`fs_512` 83%); (b) `plumb_tiles`' Linux path
(`arch_prctl`) is compile-checked only, never run. Shnatsel (2026-10-04) suggests upstreaming
`as_vectors`/`as_vectors_mut` into fearless_simd as an `as_simd()` safe `align_to` wrapper, built on
its internal bytemuck-like layer (`SimdPod`); offer a PR, with the 512-bit loop-shape note.

### 4. clflushopt stabilization report (upstream)
**Status**: Not started. `clflushopt` target feature (rust-lang/rust#157098) and `_mm_clflushopt`
(rust-lang/stdarch#2141) landed in nightly 2026-05/06 under rust-lang/rust#157096.
**Next**: draft the stabilization report for #157096 (usage: TMR, plumb_lines); the owner posts.

### 5. MOVDIR64B in rustc, std_detect and stdarch (upstream)
**Status**: Probed 2026-10-03, not started. rustc rejects `#[target_feature(enable =
"movdir64b")]`, std_detect doesn't know it, stdarch has no intrinsic; LLVM has
`llvm.x86.movdir64b`. `bench/probes/movdir64b.rs`: the intrinsic unrolls and folds the source
address but the destination is a register by encoding (13 vs 20 instructions per 4 lines); the
copy is DRAM-bound, so the gain is ergonomics (detection, `target_feature`, a documented
intrinsic), not speed. NT stores are a different case: `asm!` there is rustc policy
(rust-lang/rust#114582, #128149), so no compiler PR applies.
**Next**: owner decides if it's worth sending. If yes: three patches on new branches of
`../rust-patch` and `../stdarch-patch`, the same shape as the clflushopt commits.

### 6. plumb_tiles: AMX
**Status**: Built, reviewed (16 confirmed findings, all fixed) and wired into the bench
(2026-10-03); timing is item 2.
**Next**: after the timing run, the TMR TODO 86 probe: does a 16-row x 8-byte AMX tile store beat
AVX-512 scatter for TM5's strided u64 writes? Raw asm first; only add shapes to the crate if it
wins.

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

### 10. NT fill without a `lea` per store
**Status**: Done 2026-10-03 (825c9a9): `NtStore::stream4`, one `asm!` block of four stores with
displacements. Positional NT loop 19 -> 16 instructions (TMR 19), constant fill 11 -> 8. No
bandwidth change (DRAM-bound), as expected.
**Next**: none.

### 11. Refresh at 256 bits: plumb 110-117% of the TMR-style port
**Status**: Open (2026-10-03, three runs). The TMR-style port's 256-bit verify loop has an extra
induction variable (14 instructions per 4 loads; 10 at 512); not shown to be the cause.
**Next**: compare TMR-APP's own Refresh verify loop; if TMR has the same shape, it's a TMR fix
(precomputed bound) and a TMR TODO, not a plumb claim.
