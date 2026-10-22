# plumb TODO

This project's own tracker. It is kept separate from TMR's own tracker (not published) on purpose: if this
succeeds, the result merges into TMR; if not, nothing here clouds TMR's list. One entry per item:
status, decisions, next step. Background is in [PLAN.md](PLAN.md).

### 1. Owner review of the workspace
**Status**: Published 2026-10-05 at https://github.com/jlpetz/plumb. Both crates built,
adversarially reviewed and fixed, and restyled to fearless_simd's (Linebender) conventions; tests
and clippy clean on stable, nightly and MSRV 1.89; asm gate 130/130 (bench), 8/8 (plumb_lines
example, stable and nightly), 10/10 (plumb_tiles example). Timing done (item 2). First CI
runs green (e19c848, 2026-10-05).
**Next**: none.

### 2. Timing runs on an idle box
**Status**: Done 2026-10-03 (full sweep 825c9a9, verify groups re-run at 4254f13):
`bench/RESULTS.md`. DRAM parity everywhere (98-105%); the 512-bit view verify's L2 gap found and
fixed (loop shape, not addressing); `asm!` costs and the lost unroll measured as front-end only.
**Next**: none; re-run after changes that touch a hot loop.

### 3. Share the NT design with Shnatsel
**Status**: Draft ready 2026-10-08, statuses checked that day (`fearlessD/upstream/zulip-followup.md`,
about 4.9k characters, one reply in the existing TMR topic): the `plumb_lines::nt` design and
parity results; hold NT upstreaming until llvm/llvm-project#229240 settles, since it would make
NT a plain safe store with a hint; offer an `as_simd()` PR (Shnatsel's 2026-10-04 suggestion,
built on fearless's `SimdPod` layer) with the 512-bit loop-shape note; the correction (the 16%
L2 verify gap was unroll-and-reassociate, not indexed addressing); the untested Linux path.
**Next**: the owner posts it; then the `as_simd()` PR in the shape Shnatsel picks.

### 4. clflushopt stabilization report (upstream)
**Status**: Drafted 2026-10-08 as the stabilization PR's description
(`fearlessD/upstream/rust-clflushopt-stabilization.md`). It plans one rust-lang/rust PR for the
target feature, its detection and `_mm_clflushopt` (`library/stdarch` is now a subtree), shaped
like the keylocker stabilization (#140766). Nightly users: TMR, plumb_lines, valarauca/cafetensor.
Evidence probe `TMR/clflushopt-proof-test`, timed 2026-10-21 (bench/RESULTS.md): after a
clflushopt every read takes DRAM latency (141 ns against 5.6 from L2 and 48 from L3), and a
dirty-range flush is 33x faster than with clflush (5.7 against 187 ns per line). The draft's
tables are filled in.
**Next**: prepare the branch (file list in the draft) and run tidy and the tests; a Reference
PR; the owner opens the PR. Optional: AMD uProf counters on the m8a.metal host.

### 5. MOVDIR64B in rustc, std_detect and stdarch (upstream)
**Status**: Sent upstream 2026-10-04, with MOVDIRI added: rust-lang/rust#163742 (target
features + `is_x86_feature_detected!`), rust-lang/stdarch#2239 (`_movdir64b`,
`_directstoreu_u32/u64`; waits for #163742 to merge and sync), tracking issue
rust-lang/rust#163741. `bench/probes/movdir64b.rs` found the copy DRAM-bound, so the gain is
ergonomics, not speed. #163742 merged 2026-10-04; #2239 waits for stdarch's rustc sync
(#2240). NT stores: llvm/llvm-project#228875 proposes an opaque `llvm.x86.movnt`; RKSimon
prefers fixing `!nontemporal` itself, and jyknight's llvm/llvm-project#229240 does that (an x86
pass inserts SFENCE, keeping `!nontemporal` a hint). Tested and reviewed 2026-10-06: plumb's
fill loops keep MOVNT, unroll and get one SFENCE at the exit; posted a funclet-EH bug and the
cost of per-iteration fences. #228875 closed 2026-10-08 in favour of #229240 (jyknight and
RalfJung prefer it); offered #229240 five fix commits (branch `nt-fence-fixes` on
jlpetz/llvm-project). A patch giving six x86 intrinsics `nosync` is ready locally, held until
asked. #2239's CI is green since #2240 merged; waits for review.
**Next**: when #229240 lands, take the Rust side to Rust (rustc's `nontemporal_store` and
stdarch's `_mm_stream_*` back to `!nontemporal` on x86); when `_movdir64b` reaches nightly, use
it behind `nightly` in `plumb_lines::direct`.

### 6. plumb_tiles: AMX
**Status**: Built, reviewed (16 confirmed findings, all fixed) and wired into the bench
(2026-10-03); timing is item 2. 2026-10-05: the `nightly` feature uses stdarch's AMX intrinsics
(owner decision: be ready when they stabilize); asm gate 10/10 on stable and nightly, bench
130/130. The bench's AMX groups now build through the intrinsics: LLVM unrolls their loops, and
`amx_verify_512`'s zmm check became the fused `vpternlogq` form (21 vs 29 instructions per
tile). Re-timed 2026-10-05 (RESULTS.md, last section): intrinsics = `asm!` in DRAM and from
L1/L2, except by-reference closure captures in an ordinary caller, ~10% slower from L2.
Documented as two calling practices (capture by value, or call from an `amx-tile` function);
both measured equal to `asm!`. LLVM fix for the intrinsics' memory effects sent as
llvm/llvm-project#229025 (approved by phoebewang 2026-10-05; plumb doesn't depend on it).
The TMR TODO 86 probe (16-row x 8-byte tile store vs AVX-512 scatter) was dropped 2026-10-08:
TMR closed 86 as obsolete once its TODO 93 made the strided walk move whole lines, and for whole
lines tiles already tie zmm in the bench (`amxstride`, `amxfill`). 2026-10-08: bench group
`amxmirror`, TMR's whole MirrorMove through tiles, with the high end read and written at stride
-64 (raw `asm!`, since plumb_tiles' stride is a `usize`). On this CPU it matches TMR's pairing at
every step and takes 13 instructions per 32 line pairs, against about 160 for TMR's zmm swap.
Timed 2026-10-21: 86-89% of TMR's swap in DRAM, 97% from L2, so no AMX MirrorMove for TMR.
Owner's idea the same day, a memory test under tile-multiply load (write a pattern, transform it
in place with `TDPBUUD`, verify against the precomputed result):
`fearlessD/design/amx-transform-verify.md`. 2026-10-21/22: prototyped as `bench/src/xform.rs`
(not wired into a bench group): `PatternSet` (x patterns, 1+r checksum matrices per pattern,
every matrix byte odd and every row distinct), `fill`/`verify` as separate steps, tier-2 exact
locating (and, for a single bit flipped before the transform, the byte/bit/sign). 9 tests pass:
expected-value self-check, a clean pass, the rotation/transform agreement, single-bit detection
both in the stored result and before the transform (with the classifier verified against the
known flip), an exhaustive two-bit sweep (496 combinations, all caught), and the load dial's
extra checks adding no false positives while each independently catching corruption.
**Next**: time it against StuckBit's zmm write/verify (design doc item 3, needs the box); decide
whether to wire it into a bench `--only` group; a VNNI/AVX-512 version for AMD is still open.

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
**Status**: Done 2026-10-21. TMR-APP's own Refresh has the port's verify loop and a less
unrolled fill, so the port was faithful. Timed with one phase swapped at a time (bench
`refreshx`): the gap moves between copies of identical code. `tmr_256` is now at parity with
plumb, while its control `tmrx_256` runs at 86%: the same instructions, the same 64-byte
offsets. So it's code placement or run order, not a source change TMR could make, and not a
plumb advantage. Without the flush the styles are at parity (`refreshnf`). README and RESULTS
corrected.
**Next**: none; no TMR TODO.
