# plumb: plan and decisions

Extension crates for [fearless_simd](https://crates.io/crates/fearless_simd) that add what a
memory tester needs and a portable SIMD library doesn't have. They started from TMR-APP's
TODO 84 (can fearless_simd replace TMR's per-width SIMD macros?) and from Shnatsel's reply on
the Linebender Zulip: non-temporal stores and tiles should be experimented with in my own
extension crates first, and NT may be upstreamed once the design has settled. Tiles must avoid
the `fearless_` prefix.

Status tracking lives in [TODO.md](TODO.md). The TODO 84 evidence is in
[bench/FINDINGS-TODO84.md](bench/FINDINGS-TODO84.md).

## Decisions (settled with the project owner, 2026-10-03)

| Topic | Decision |
|---|---|
| Structure | Two crates, `plumb_lines` and `plumb_tiles`, plus a TMR-internal integration module later (not a third crate). `bench/` holds the asm gate, equivalence tests and the thread-sweep harness. |
| Repo | Its own repo, public from the start (`jlpetz/plumb`). Created and pushed only after the owner has reviewed it. |
| Toolchain | Both crates build on **stable** via `asm!`. `plumb_lines` has a `nightly` feature that uses stdarch intrinsics where they exist (`_mm_clflushopt`) and exports the `with_clflushopt!` entry macro. `plumb_tiles` has one that uses stdarch's AMX intrinsics for every tile instruction (owner decision 2026-10-05: be ready when they stabilize). The bench needs nightly (TMR-style kernels use `std::simd`) and builds both crates' `nightly` paths. |
| Order | Aligned views (`as_vectors`) first, then NT stores (what Shnatsel wants to see), then CLFLUSHOPT/MOVDIR64B, then AMX tiles, then the TMR pilot. |
| Cache-line ops | **CLFLUSHOPT and MOVDIR64B only for now**: both are proven useful, one has landed in nightly and one hasn't. CLWB, CLZERO, CLDEMOTE, MOVDIRI and PREFETCHW are deferred: none is faster than existing paths, and none adds coverage TMR needs today. |
| Line ops vs SIMD levels | Capability tokens (`Clflushopt`, `Movdir64b`), not fearless levels: they're orthogonal to the SIMD ladder (Zen 4 has AVX-512 without MOVDIR64B; Alder Lake the reverse). |
| Tiles | AMX implemented now; the x86 ACE extension kept firmly in mind (ACE v1.15 spec) but not implemented (no hardware, no assembler or rustc support). |
| Tile shape | **Static 16 rows x 64 bytes only**, matching ACE's fixed shape and AMX's maximum. AMX's smaller configurable shapes stay out unless TMR TODO 86's strided test proves it needs them (probe it in raw asm first). |
| Tile data types | Faithful per-ISA support (AMX: INT8 x4 variants, BF16, FP16 on this hardware), plus a small shared, ACE-shaped layer (session, zero, shape). No lowest-common-denominator API. |
| AVX-512 policy | Take fearless_simd's (its `Avx512` token needs the Ice Lake set). It costs nothing for DDR5: Skylake-X/Cascade Lake/Cooper Lake are DDR4. Only matters at TMR integration. |
| NT memory model | NT and MOVDIR64B writes happen inside closure scopes that SFENCE on exit (including unwind); the destination stays borrowed until then and writers are write-only. Rust's fences don't order NT stores (`fence(Release)` emits nothing, `fence(SeqCst)` emits `lock or`). |
| Flush scope | `flush_after` (the owner's idea): ordinary writes, then CLFLUSHOPT of the whole borrowed range and MFENCE on exit. Not a soundness matter, so it gives plain `&mut`. |
| Upstream compiler work | Parallel and off the critical path: a stabilization report for `clflushopt` (rust-lang/rust#157096); MOVDIR64B target feature + std_detect + stdarch intrinsic, using the clflushopt PRs (rust-lang/rust#157098, rust-lang/stdarch#2141) as templates. |

## ACE vs AMX (facts from the ACE v1.15 spec and the LLVM/GCC patches)

- Both have 8 tiles of up to 16 x 64 bytes. ACE's shape is fixed (palette 2; `LDTILECFG` carries
  only `palette = 2`).
- **AMX tiles load and store memory directly** (strided `TILELOADD`/`TILESTORED`). **ACE has no
  tile memory instructions**: tiles are accumulators only, filled and drained through ZMM
  (`TILEMOVROW`, `TILEMOVCOL`, `TCVTROW*`). So for a memory tester AMX is the interesting one;
  ACE's value is compute density (heat/power load).
- AMX computes dot products with all three operands in tiles (C += A.B), so an AMX kernel spends
  tiles on inputs (typically 4 accumulators). ACE computes outer products of two ZMM vectors into
  a tile, so all 8 tiles can be accumulators.
- ACE v1 types: INT8 (s/u), BF16, MX FP8 (E4M3/E5M2) and MXINT8 with a 1024-bit Block Scale
  Register (XCR0 bit 20); accumulators INT32/FP32 only. AMX: INT8, BF16, FP16, complex FP16, FP8,
  TF32 (by sub-extension). The data types overlap only partly and the operations differ.
- ACE detection: AMX-TILE + CPUID 7.1:ECX[11] + ACE_VSN (leaf 1Dh.2) + XCR0[20,18:17] + AVX-512
  state; baseline AVX10.1. GCC: legacy AMX and ACE are not to be mixed (one palette at a time).

## Phases

| Phase | Work | Gate | Status |
|---|---|---|---|
| P0 | Workspace, bench moved in from the probe | builds; asm gate reproduces 78/78 | done |
| P1 | `plumb_lines`: `as_vectors`, `NtStore`, `nontemporal` scope | NT loop no worse than TMR's (now 16 vs 19 instructions); tests | done |
| P2 | `plumb_lines`: `Clflushopt` + `flush_after`, `Movdir64b` + `direct`, `with_clflushopt!` | intrinsic/asm inlined in loops; tests | done |
| P2b | TMR StuckBit/Refresh/SimpleNT ported in both styles | same errors under fault injection, memory checked against TMR's sequence; every plumb kernel within 25% of its TMR twin per memory op (asm gate; the per-line `asm!` flush, `wflush_pltok`, is allowed 50%, documented) | done |
| P2c | Adversarial review of plumb_lines + bench (29 raised, 26 confirmed) | all confirmed findings fixed or documented | done (2026-10-03) |
| P3 | Upstream: clflushopt stabilization report; MOVDIR64B in rustc/std_detect/stdarch | upstream review | in progress: MOVDIR64B/MOVDIRI sent (rust-lang/rust#163742, rust-lang/stdarch#2239), opaque NT stores proposed (llvm/llvm-project#228875); clflushopt report not started |
| P4 | `plumb_tiles` AMX | tests on real AMX; asm gate; review (21 raised, 16 confirmed, all fixed); timing (no bandwidth gain over zmm; `bench/RESULTS.md`) | done |
| P4b | Style pass to fearless_simd's (Linebender) conventions; timing sweep | lint set v8 clean, fmt, headers, CI; `bench/RESULTS.md` | done (2026-10-03) |
| P5 | ACE backend | hardware + assembler/rustc support | waiting |
| P6 | TMR integration module and pilot (StuckBit first) | parity at every width/thread count; TMR results unchanged | after review |
