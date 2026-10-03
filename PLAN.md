# Plan: TMR's extensions to fearless_simd

**Date**: 2026-10-03. Follows TODO 84 (FINDINGS.md) and Shnatsel's reply on Zulip: NT stores and
tiles belong in our own extension crates first ("we can upstream it later once we're confident
in the design"); tiles must avoid the `fearless_` prefix.

## Decisions needed (summary)

1. **Structure**: two published crates, `plumb_lines` and `plumb_tiles`, plus a TMR-internal
   integration module (not a third crate). Neither a single super crate nor one crate per addition.
2. **Toolchain**: both crates build on **stable Rust via `asm!`**, with an optional `nightly`
   feature that switches to stdarch intrinsics and target-feature tokens. Nothing has to wait on
   stabilization.
3. **Tile shapes**: compile-time constant shapes (`Tile<ROWS, COLSB>`), with ACE limited to its
   fixed 16x64. That's "static" in the code sense, and keeps AMX's small-row shapes, which
   TODO 86's strided writes may want.
4. **Order**: plumb_lines NT first, then the line ops, then plumb_tiles (AMX), then the TMR
   pilot. Compiler/stdarch PRs run in parallel. ACE waits on hardware and on rustc.

## 1. Structure

| Option | Verdict |
|---|---|
| One super crate with features | No. It mixes stable code (NT) with nightly-ish code (tiles), the name can't fit both, and it makes NT harder to lift into fearless_simd later. |
| One crate per addition (nt, lineops, tiles) | Close, but NT and the line ops share one core abstraction (a scoped writer whose exit runs a fence or flush), so splitting them duplicates it. |
| **Two crates + TMR module** | **Recommended.** |

- **`plumb_lines`**: memory ordering and cache-line control. It covers NT stores, direct stores
  (MOVDIR64B/MOVDIRI), flush and writeback (CLFLUSHOPT/CLWB/CLDEMOTE/CLZERO) and prefetch, with
  the scoped ordering API, the capability tokens and aligned vector views. A plumb line is the
  tool for "true", and these are cache-line ops. The NT module stays self-contained so it can
  move upstream on its own.
- **`plumb_tiles`**: 2D tile ISAs. AMX now; ACE when hardware and rustc support exist.
- **TMR integration**: a module inside TMR-APP (e.g. `src/simd/`), not a crate. It holds TMR's
  conventions (width variants and Auto, the verify accumulator pattern, test registration, the
  level policy) and nothing reusable outside TMR. Promote it to a crate only if a second user
  appears.
- **Repo**: a new workspace repo (e.g. `jlpetz/plumb`) with `plumb_lines/`, `plumb_tiles/` and
  `bench/`. `bench` is this probe crate grown up: `asm_check.py`, the equivalence tests and the
  large-page thread-sweep harness, all of which carry over directly. TMR depends on the crates by
  path or git during development, and on crates.io later.
- **Version pinning**: the capability-token entry macro copies fearless_simd's per-level feature
  lists, so pin `fearless_simd = "=1.0.x"` and re-run `asm_check.py` plus
  `level_features_are_detected` on every upgrade.

## 2. `plumb_lines`

### 2.1 Aligned vector views (do first; it fixes two problems at once)

```rust
/// Split `buf` into an unaligned head, an aligned slice of vectors, and a tail. Takes the token
/// because a `u64x8<Avx512>` value carries an Avx512 proof.
pub fn as_vectors_mut<S: Simd, V: SimdBase<S, Element = u64>>(simd: S, buf: &mut [u64])
    -> (&mut [u64], &mut [V], &mut [u64]);
```
- **Alignment**: a `&mut V` proves vector alignment, which is Shnatsel's `&mut u32x8<S>` point.
  NT and direct stores can then be safe in that respect.
- **Loop shape**: iterating a `&[V]` with `iter()` compiles to a pointer walk (base+disp). That is
  the fix for the 512-bit verify gap that `chunks_exact` caused (84% vs 101%), so it's also the
  answer to the question we put to Shnatsel about a slice-as-`V`s helper.

### 2.2 Ordering scopes: one abstraction for NT, direct stores and flush-after-write

The memory is mutably borrowed for the whole scope, and the scope's exit runs the finalizer, so
nothing can read the data before it's ordered or flushed. It has to be a closure scope, like
`std::thread::scope`, not a guard object: `mem::forget` on a guard would skip the finalizer.

| Scope | Stores inside | Finalizer on exit | Why the finalizer |
|---|---|---|---|
| `nontemporal(simd, &mut [V], \|w\| ..)` | `_mm*_stream_*` (4x unrolled helper) | `SFENCE` | **Soundness** (stdarch contract) |
| `direct(md: Movdir64b, &mut [Line], \|w\| ..)` | `MOVDIR64B` line copies | `SFENCE` | Soundness (weakly ordered, like NT) |
| `flush_after(simd, cf: Clflushopt, &mut [V], \|buf\| ..)` | ordinary cached stores | `CLFLUSHOPT` every line + `MFENCE` | **Test semantics**: the verify must round-trip DRAM |
| `writeback_after(simd, wb: Clwb, ..)` | ordinary cached stores | `CLWB` every line + `SFENCE` | Write the data to DRAM but keep it cached (a new test variant) |

- **NT / direct**: the writer hands out **write-only slots** (`NtSlot` has `store(v)` and nothing
  else). A plain `&mut V` inside the scope would let code read memory with NT stores still in
  flight, which the stdarch contract forbids.
- **flush_after** is your idea (#3), and yes, it works. The scope owns the range, so it doesn't
  need to track dirty lines: it flushes the borrowed range, exactly what `flush_range_to_dram`
  does today, but now the type system enforces "flush before verify". It isn't a soundness
  matter (caches are coherent), so it can be a fully safe API and code may read inside the
  scope. Caveats: unaligned ends flush neighbouring lines, and hardware prefetch during the
  verify still fetches from DRAM, which is what we want.
- **Mixed form**: a per-line `w.store_line_and_flush(..)` for the write-then-flush-each-line loop
  (measured the same speed as flush-at-end in FINDINGS).
- **Rust fences are no substitute**: `fence(Release)` emits nothing and `fence(SeqCst)` emits
  `lock or`, so the finalizers emit `sfence`/`mfence` explicitly.

### 2.3 Capability tokens

`Clflushopt`, `Clwb`, `Movdir64b`, `Movdiri`, `Cldemote`, `Clzero` (AMD) and `Prefetchw`.
These are ZST proofs constructed by CPUID detection (std_detect knows none of them except
`clflushopt`). They're orthogonal to fearless levels, because they're on different CPUs
(e.g. Zen 4: AVX-512 without MOVDIR64B; Alder Lake: MOVDIR64B without AVX-512).
- **Stable path**: `asm!`. It needs no target feature, inlines anywhere, and measured the same as
  the intrinsic (FINDINGS, `../clflush-test/`).
- **`nightly` feature**: stdarch intrinsics, plus the `with_*!` entry macro from `src/cap.rs` that
  enables fearless's level list plus the extra feature. It's proven to inline `_mm_clflushopt`
  with the same 8x unroll as `flush_range_to_dram`.

### 2.4 Upstream compiler work: is it worth it?

What each instruction lacks today (checked on nightly 1.100):

| Instruction | rustc `#[target_feature]` | std_detect | stdarch intrinsic | TMR value |
|---|---|---|---|---|
| CLFLUSHOPT | yes, unstable (rust#157098, yours) | yes | yes, unstable (stdarch#2141, yours) | high: in use |
| MOVDIR64B | **no** (LLVM-only) | no | no | medium: alternate write path (TODO 78) |
| MOVDIRI | no | no | no | low |
| CLWB | no | no | no | medium: `writeback_after` variant |
| CLZERO (AMD) | no | no | no | medium: AMD's full-line no-RFO write |
| CLDEMOTE | no | no | no | low: cache-tier tests |
| PREFETCHW | yes (`prfchw`) | ? | no (`_m_prefetchw` missing) | low |

**My take:** worth doing, but it's not on the critical path and won't make anything faster.
- **What it buys**: `is_x86_feature_detected!` instead of hand-rolled CPUID;
  `#[target_feature]`/`cfg` support, so tokens are type-checked; no operand-order traps (the
  MOVDIR64B asm has its destination as a register operand, which is easy to get backwards); LLVM
  visibility for scheduling and unrolling; and backends that lack `asm!`.
- **What it doesn't buy**: speed (asm and the intrinsic measured the same), or safety (they
  still take pointers).
- **Cost**: three small patches per instruction, using the clflushopt patches in this workspace
  as templates, then a long wait (clflushopt is still unstable four months after landing).
- **Suggested batches**: (1) the clflushopt stabilization report; (2) MOVDIR64B + MOVDIRI
  ("direct stores", same CPUID leaf); (3) CLWB + CLZERO + CLDEMOTE plus `_m_prefetchw`. The crate
  never waits on these: it ships on `asm!` and flips to intrinsics behind `nightly` as each
  lands.

## 3. `plumb_tiles`

### 3.1 Facts: AMX vs ACE v1

From the ACE v1.15 spec (x86 Ecosystem Advisory Group, 2026-05-15), LLVM #208408/#208706 and the
GCC ACEv1 commits:

| | AMX (palette 1) | ACE v1 (palette 2) |
|---|---|---|
| Tiles | 8 x up to 16 rows x 64 B | **8 x 16 rows x 64 B, fixed** |
| Shape config | `LDTILECFG` with per-tile rows/colsb | `LDTILECFG` with just `palette = 2` |
| Tile <-> memory | **`TILELOADD`/`TILESTORED`/`TILESTREAMLOADD`, strided** | **none**: rows/cols via ZMM only (`TILEMOVROW` r/w, `TILEMOVCOL` write) |
| Tile role | A, B and C operands | accumulator only (C) |
| Compute | dot products `TDP*` (tile x tile) | outer products `TOP*` (zmm x zmm into tile) |
| Types | INT8, BF16, FP16, complex FP16, FP8, TF32 (by sub-extension) | INT8 (s/u), BF16, MX FP8 (E4M3/E5M2) and MXINT8 with block scale; accumulators INT32/FP32 only |
| Extra state | none | Block Scale Register, 1024-bit, x1 (XCR0 bit 20) |
| Baseline | AVX-512 not required | AVX10.1 required |
| Detection | CPUID 7.0:EDX[24] + XCR0[18:17] | AMX-TILE + CPUID 7.1:ECX[11] + ACE_VSN (leaf 1DH.2) + XCR0[20,18:17] + AVX-512 state |
| Mixing | | GCC: "legacy AMX and ACE should not be used together"; one palette is configured at a time |

**Corrections to the brief:**
- **The register count is the same.** Both have 8 tiles of up to 1 KiB.
- **The load/store difference runs the other way.** AMX is the one with memory load/store; ACE's
  tiles never touch memory directly.
- **Data types overlap only partly** (INT8, BF16, plus FP8 on AMX-FP8 parts), and the
  *operations* differ (dot vs outer product, tile vs vector operands). So a shared API is only
  possible at the "accumulate a GEMM block" level, not per instruction.

### 3.2 What this means for TMR

- **AMX is the memory-tester feature.** One `TILELOADD`/`TILESTORED` moves 16 rows at an arbitrary
  stride, an access shape no vector instruction produces. With a tile configured as 16 rows x 8 B,
  one `TILESTORED` writes **16 strided u64s**, which is TODO 86's TM5 strided pattern. That's a
  candidate alongside AVX-512 scatter; measure both.
- **ACE adds no new memory access shape.** Data still moves through ZMM loads and stores, which
  fearless_simd already covers. Its TMR value is compute density, a power/heat load for the
  heat-soak work (TODO 64), not memory coverage.

### 3.3 Design

- **Shapes are compile-time constants**: `Tile<const ROWS: usize, const COLSB: usize>`. The AMX
  backend accepts any valid shape; the ACE backend only `<16, 64>`. "Static" in the code sense,
  as you proposed, without losing AMX's small-row shapes that TODO 86 may need. The default and
  most common shape is 16x64, identical across both.
- **Tile registers as type-level indices** (`T0`..`T7`), since tile numbers are immediates in the
  instructions.
- **A scoped tile context**: `amx.with_tiles(cfg, |t| ..)` runs `LDTILECFG` on entry and
  `TILERELEASE` on exit. Tile instructions are only valid while configured, so the scope is what
  makes them safe to call.
- **Memory ops (AMX only) take bounds-checked slices**: a load or store of `ROWS x COLSB` at
  `stride` needs `(ROWS-1)*stride + COLSB <= len`.
- **Shared layer**: config, zero, release, and row moves to and from ZMM (AMX-AVX512 has row
  reads; ACE adds writes and column writes). The backends then split: `amx` (memory ops, `TDP*`)
  and `ace` (`TOP*`, BSR).
- **Stable via `asm!`**: tile registers are named in the instruction, not allocated by Rust, so
  `asm!` with hard-coded `tmm0..7` works. GCC's legacy AMX intrinsics are inline asm for the
  same reason. A `nightly` feature can switch to stdarch's `x86_amx_intrinsics` (unstable,
  rust#126622).
- **OS enablement**: check XCR0[18:17]. On Linux, request permission with
  `arch_prctl(ARCH_REQ_XCOMP_PERM)`. **Open**: what Windows needs per thread (this Server 2025
  VM reports the tile state enabled in XCR0).
- **ACE**: no hardware yet, LLVM PRs still open, nothing in rustc or stdarch. Keep the shared
  layer ACE-shaped (fixed 16x64, row/col moves), write the ACE backend when hardware appears,
  and consider contributing ACE target features to rustc then, like clflushopt.

## 4. TMR integration (later)

- A `src/simd/` module re-exporting fearless_simd and plumb. It encodes TMR's conventions: width
  variants plus Auto; TMR's level policy (TMR gates 512 on AVX-512F/BW/CD/DQ/VL, fearless on the
  Ice Lake set; decide which wins); pointer-walk verify loops via `as_vectors`; and the error
  accumulation pattern.
- **Pilot**: StuckBit (fill + 4-accumulator verify) behind `asm_check.py` and the equivalence
  tests, then an A/B on TMR's own runs. Next Refresh, SimpleTest (positional/LCG), SimpleNT (the
  `nontemporal` scope), and MirrorMove. Keep the macros until each family is proven.
- When the pilot passes, update the settled `macro_rules!` decision in TMR-APP's CLAUDE.md and
  in skill `tmr-design-rationale`.

## 5. Phases and gates

| Phase | Work | Gate |
|---|---|---|
| P0 | Agree structure, names and repo; create the workspace; move the probe in as `bench/` | (decision) |
| P1 | `plumb_lines`: `as_vectors`, `NtStore`, `nontemporal` scope | asm identical to TMR SimpleNT; equivalence tests; parity at 1-8T |
| P2 | `plumb_lines`: capability tokens, `flush_after`, `direct`, `writeback_after` | intrinsic/asm inlined in loops; flush scope matches `flush_range_to_dram` |
| P3 (parallel) | rustc/std_detect/stdarch PRs (batches in 2.4); clflushopt stabilization report | upstream review |
| P4 | `plumb_tiles` AMX: context, memory ops, TODO 86 probe (16x8 strided store vs scatter vs scalar) | measured; asm checked |
| P5 | ACE notes kept current; backend when hardware and rustc allow | (external) |
| P6 | TMR pilot (StuckBit), then the other families | parity at every width/thread count; TMR results unchanged |

## Open questions

- Repo name and visibility (public from the start, or private until P2)?
- Pointing Shnatsel at the NT design once P1 is done (he asked to see it)?
- TMR's 512-bit gate vs fearless's Ice Lake set, for the integration module.
- Windows AMX per-thread requirements (P4).
