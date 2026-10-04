# fearless_simd vs TMR's SIMD style: findings (TODO 84)

> **Correction (see `bench/RESULTS.md`, commit 4254f13):** the 16% 512-bit L2 verify gap blamed
> below on `chunks_exact`'s indexed addressing was LLVM unrolling and reassociating the
> OR-accumulate loop, which loses the fused `vpternlogq`. A base+displacement version was just
> as slow; the pointer walk helped because of its loop shape. The text below is kept as the
> record of what was measured at the time.

**Date**: 2026-10-03
**Crates**: `fearless_simd` 1.0.0, `fearless_simd_macros` 0.1.0
**Rust**: 1.100.0-nightly (LLVM 23.1.1), baseline `x86-64-v3` (TMR's)
**Hardware**: Intel Xeon 6975P-C (Granite Rapids), AWS 4 cores / 8 threads. CPUID: AVX-512 with
the full Ice Lake set, CLFLUSHOPT, MOVDIR64B, **AMX-TILE/INT8/BF16 with the OS tile state
enabled (XCR0 bits 17-18)**. `Level::new()` = `Avx512`.

## The question

TODO 84: could fearless_simd replace TMR-APP's per-width `macro_rules!` + `std::simd` +
`#[target_feature]` kernels without losing what the codegen rules protect (width, no
`memset`/`memcpy` collapse, 4 accumulators)? And does it fit, or can we extend it, for NT stores
(+ fences), CLFLUSHOPT, prefetch, MOVDIR64B, and later AMX/ACE tiles?

## Method

Every TMR kernel shape is written twice and compared on the same buffers:

- `src/tmr.rs`: TMR's style. Loop bodies are copied from `stuck_bit_write_verify!`,
  `simple_write/verify_positional_simd!`, `simple_write_lcg_simd!`,
  `simple_write_nt_positional_simd!` and `flush_range_to_dram`, with TMR's exact feature strings.
- `src/fs.rs`: fearless_simd. **One generic body per kernel** (`#[simd] fn k<S: Simd, V:
  SimdInt<S, Element = u64>>`). Width comes from the type (`u64x2/x4/x8<S>`), the ISA from the
  token (`Avx2`, `Avx512`). The code uses slices, not raw pointers. Missing ops are added as
  extension traits over `kernel!`.

Three checks:
1. **`asm_check.py`** (the verdict): finds each kernel's loops in the emitted asm, following
   the call from the named entry point into the `#[simd]`/`kernel!` target-feature helper.
   It reports loop width, expected instructions, calls inside loops, memset/memcpy calls and
   vector spills, then PASS/FAIL against per-kernel expectations. Exit code 1 on an unexpected
   result, so it can gate a fearless_simd upgrade.
2. **`cargo test`**: each fearless kernel must produce exactly the TMR-style output, and each
   verify must catch a single flipped bit anywhere (including the tail loops). The LCG test
   also requires every width to reproduce the scalar LCG stream.
3. **`cargo run --release`**: throughput, L2-resident and DRAM. See "Timing" below.

## Results: asm (78/78 kernels as expected)

| Kernel (TMR source) | fearless_simd vs TMR style |
|---|---|
| fill (StuckBit/Refresh write) | Same width at 128/256/512, 8x unrolled store loop, no memset. Uses `vmovups`/`vmovdqu64` (unaligned form) where TMR gets `vmovaps`: same speed on aligned data. |
| verify, 4 accumulators | Same: 4 independent loop-carried accumulators survive (512: `vpxorq` + `vpternlogq`, 2x unrolled). Bounds checks from `from_slice` fully elided, no panic calls in loops. But a `chunks_exact` loop gets indexed addressing, which costs 16% at 512-bit L2 speed (see Timing); a pointer walk (`verify4_ptr`) gets TMR's base+disp form. |
| positional write/verify (Mode 0/1) | Same instructions, LLVM unrolls the fearless loop further (27 vs 27 instrs write; 35 vs 19 verify). |
| LCG write (Mode 2, 64-bit mul) | fearless writes AVX2 `mul_u64x4` as 4 scalar `wrapping_mul`s, **but LLVM's SLP pass re-vectorizes it to the same `vpmuludq` x3 sequence std::simd emits** (4x unrolled). 512 uses `vpmullq` in both. No loss, but it depends on SLP. |
| verify + PREFETCHT0 | Same; `_mm_prefetch` is SSE, so it inlines anywhere. |
| NT write, 4x unroll + SFENCE | **Instruction-for-instruction identical** (19-instruction loop, same registers) via the `NtStore` extension trait. Also identical as a plain `kernel!`. |
| CLFLUSHOPT, stdarch intrinsic | **Call per line.** No fearless token lists `clflushopt` (it's in no x86-64 psABI level), so `_mm_clflushopt` can't inline. In the mixed write+flush loop each line pays `vzeroupper` + `call` + a re-broadcast of the pattern (2-5% at DRAM speed). **Fixed by the capability-token prototype** (`cap.rs`, below): the intrinsic inlines, with the same 8x unroll as TMR. |
| CLFLUSHOPT, inline `asm!` | Inlines cleanly in fearless code (5-instruction mixed loop). |
| MOVDIR64B copy | `asm!` inside `kernel!`: identical to TMR's asm. |
| NT-512 copy | Identical. |
| Byte-uniform fill (canary) | **The memset trap applies**: `fs_256` became `memset`. `fs_512` happened to stay a loop in this build, which is pass-order luck. Keep the non-byte-uniform rule. |

### When inlining fails: fearless_simd is loud, std::simd is silent

The `*helper_512` variants call one non-inlined generic helper from the 512-bit verify, the
mistake TMR's macro rule guards against:

- **std::simd** (`k_verify4_tmrhelper_512`): the helper compiles at the v3 baseline and does
  its `u64x8` work as **2 x `vxorps ymm` + 2 x `vorps ymm`**. The "512" variant silently runs at
  256-bit with correct results. This is exactly the failure that got `test_framework.rs` deleted.
- **fearless_simd** (`k_verify4_fshelper_512`): the helper calls fearless_simd's per-op
  target-feature functions out of line, which are **`vxorps zmm` / `vorps zmm`**. The width is
  kept; the cost is two calls per vector, so it is slow, not wrong.

So fearless_simd doesn't remove the inlining requirement, but it changes the failure from
silently lossy (the class TMR's rules exist for) to a performance cliff that `asm_check.py`
flags as "call in loop" and that a benchmark would expose.

### Forcing a width

- Width is the vector type and is independent of the token, so `u64x2/x4/x8` give
  128/256/512. Lower tokens are available on bigger CPUs: `Level::new().as_avx2()` works on
  this AVX-512 box, which is how the fs 128/256 variants get VEX encoding, as TMR's do.
- `dispatch!` *normalizes the level up to the build baseline*: at `x86-64-v3` an `Sse2`/`Sse4_2`
  level dispatches as `Avx2` (`Level::__dispatch_target`). That's harmless, since the type sets
  the width, but don't expect `dispatch!` to honour a forced SSE level.
- **AVX-512 policy**: the `Avx512` token needs the whole Ice Lake set (IFMA, VBMI2, VNNI,
  BITALG, VPOPCNTDQ, GFNI, VAES, SHA...), not just F/BW/CD/DQ/VL. Skylake-X/Cascade Lake fall
  back to `Avx2`. Those are DDR4 platforms; every DDR5 platform with AVX-512 I know of (Sapphire
  Rapids and later, Zen 4/5) has the full set, so for TMR this probably costs nothing.

### Instructions TMR needs that fearless_simd doesn't have

None of NT stores, SFENCE/MFENCE, CLFLUSHOPT, PREFETCH, MOVDIR64B or AMX appear in fearless_simd.

- **NT stores**: a downstream extension trait works and codegen is identical (`fs.rs`
  `NtStore`). It wraps `kernel!` with `#[inline(always)]`, the same pattern the library uses
  for its own ops. The stdarch stream intrinsics are still `asm!` inside, so the manual 4x
  unroll is still needed (`nt_stores.md` is unchanged).
- **CLFLUSHOPT**: use `asm!` (`common.rs` `clflushopt_asm`). `kernel!` has no way to add a
  feature beyond its fixed level list: it accepts only the six audited tokens, and attributes
  you add land on the outer wrapper, not the `#[target_feature]` fn. (TMR's standalone
  `flush_range_to_dram` is unaffected; it's its own `clflushopt` fn.)
- **MOVDIR64B**: no stdarch intrinsic, so `asm!` as today; it works inside `kernel!`.
- **AMX / ACE**: no tile types, and `kernel!` can't enable `amx-tile`. Tile code would be
  TMR's own `#[target_feature(enable = "amx-tile")]` fns using stdarch's unstable
  `x86_amx_intrinsics`; fearless_simd adds nothing there. **This box has AMX with OS support, so
  TODO 78 could start without new hardware.**

### Memory model and fences

- `std::sync::atomic::fence(SeqCst)` emits `lock or dword ptr [rsp], 0`, **not `mfence`**, and
  `fence(Release)` emits nothing. Linux keeps `mfence` for its mandatory `mb()` and uses the
  locked form only for `smp_mb()` on ordinary memory, so don't treat a Rust fence as ordering NT
  stores, MOVDIR64B or CLFLUSHOPT. Keep explicit `_mm_sfence()` / `_mm_mfence()` (relevant to
  TODO 16).
- fearless_simd has no model for this. A sound *safe* NT API would have to guarantee the
  sfence. One shape worth proposing upstream: a scoped writer, `simd.nontemporal(|nt| { ...
  nt.store(v, &mut chunk) ... })`, that sfences on scope exit. That would let TMR's NT kernels
  drop their `unsafe`.

### Friction (maintainability costs found while porting)

1. **`unsafe` doesn't go away for TMR's interesting instructions.** Every one of them takes a
   raw pointer (NT, CLFLUSHOPT, MOVDIR64B). And because a `#[simd]` body isn't lexically in a
   `#[target_feature]` fn, even `_mm_sfence`, `_mm_mfence` and `_mm_prefetch` need `unsafe`
   there (rustc E0133: "being enabled in the build configuration does not remove the
   requirement"). Inside `kernel!` they are safe.
2. `kernel!` only accepts **safe** fns, so pointer intrinsics need an `unsafe` block inside a safe
   fn, with the contract moved to the caller (`NtStore::nt_store` is `unsafe`).
3. The extension impls need one small `macro_rules!` per op (`nt_store_impl!`), five lines per
   (width, level). That replaces TMR's per-test macros with per-op macros, which is a smaller
   surface.
4. Build cost: a clean release build of fearless_simd plus this crate takes 6.8 s, against
   1.6 s for this crate alone (about 100k lines of generated code in the dependency).
5. Asm is harder to read: hot loops live in symbols like
   `fs::verify4::__FearlessDispatch::call::entry<Avx512, u64x8<Avx512>>`. `asm_check.py`
   follows them for you.

What it buys: one type-checked generic body per kernel instead of `$simd_type` substitution
(real IDE support and error messages), safe slice loads/stores with the bounds checks
provably elided, and the loud-not-silent failure mode above.

## Timing (2026-10-03, idle box)

Full output: `run.log` / `results.csv` (every cell with [min..max]), plus `run-ptr.log` /
`results-ptr.csv` for the loop-shape follow-up. These came from the earlier probe and aren't
published; the tables below are copied from them. Two regimes:
- **L2**: one thread pinned to CPU 2, 256 KiB warm, 21 samples x 2000 reps. This is where
  codegen differences show.
- **DRAM**: 2 GiB per thread on 1 GiB pages (16 GiB at 8 threads), threads pinned
  physical-cores-first (`[0,2,4,6,1,3,5,7]`), 5 samples. Each worker times its own pass, and
  aggregate = bytes / slowest thread.

Typical sample spread was 1-4%.

### Result: parity, once the verify loop is a pointer walk

"fs" is the generic fearless kernel, as a % of the TMR-style kernel at the same width:

| Kernel | L2 128 / 256 / 512 | DRAM, 1T to 8T, all widths |
|---|---|---|
| fill | 100 / 100 / 100 | 98-101 |
| positional write | 100 / 100 / 100 | 99-102 |
| positional verify | 96 / 98 / 100 | 98-105 |
| LCG write (64-bit mul) | 99 / 99 / 100 | 99-102 |
| CLFLUSHOPT range (intrinsic, asm) | 100 | 100-101 |
| NT write, 4x unroll + SFENCE | (n/a) | 99-101 (22 to 91 GiB/s) |
| NT-512 copy / MOVDIR64B copy | (n/a) | 99-103 / 99-101 |
| 4-acc verify, `chunks_exact` loop | **104 / 103 / 84** | 93-95 at 1T (128/256), parity by 4T |
| 4-acc verify, **pointer-walk loop** | 103 / 102 / **101** | **98-101** |
| prefetch verify 512, `chunks_exact` | (n/a) | **83** at 1T, 89 at 2T, 91 at 4T, 98 at 8T |
| prefetch verify 512, pointer walk | (n/a) | 94 at 1T, 97 at 2-4T, 104 at 8T |
| write + CLFLUSHOPT per line, intrinsic (call per line) | (n/a) | 95-98, parity at 8T |
| write + CLFLUSHOPT per line, `asm!` | (n/a) | 95-101 |

**The 512-bit verify gap is the loop shape, not fearless_simd.** A `chunks_exact` loop gets
indexed addressing (`[r9 + 8*r10 + 64]`); TMR's pointer walk gets base+disp (`[r8 - 192]`).
At 512-bit L2 speed (about 172 GiB/s, two 64-byte loads per cycle) the indexed form costs 16%.
The same fearless body with a raw-pointer walk (`fs::verify4_ptr`) runs at 172.7 vs 171.2 GiB/s.
A safe `split_at` walk only half-fixes it (86.6%). At 128/256 the `chunks_exact` form is
actually 3-7% faster. The pointer-walk prefetch verify at 512 has the same instructions as TMR's
(4 PREFETCHT0 + 4 `vpternlogq` per iteration, two instructions shorter). Its remaining 1-thread
difference is within spread and layout effects (Rule 5), and it's 104% at 8 threads.

**The CLFLUSHOPT call per line is a small cost.** In the mixed write+flush loop it costs 2-5%
until DRAM saturates. For TMR's standalone `flush_range_to_dram` it doesn't arise.

**Footguns (L2, 512-bit verify):** std::simd helper 9.3 GiB/s (and 256-bit underneath),
fearless helper 16.7 GiB/s (512-bit, out-of-line calls). Both are about 9-18x slower than the
inlined kernels; only the std::simd one changes the width.

### Scaling (TMR style; fearless tracks it)

| GiB/s | 1T | 2T | 4T | 6T | 8T |
|---|---|---|---|---|---|
| fill 512 | 12.2 | 23.2 | 38.2 | 33.9 | 40.2 |
| 4-acc verify 256 | 19.5 | 32.9 | 59.4 | 49.3 | 63.4 |
| NT write 256 | 22.3 | 44.1 | 88.6 | 69.3 | 91.3 |
| CLFLUSHOPT range | 24.6 | 47.9 | 96.2 | 74.4 | 99.1 |
| copy NT-512 / MOVDIR64B | 11.1 / 13.2 | 19.4 / 20.4 | 36.7 / 38.7 | 30.7 / 30.3 | 38.8 / 39.2 |
| LCG write 128 (compute-bound) | 6.1 | 12.3 | 24.5 | 32.3 | 39.1 |

Memory-bound kernels saturate at 4 threads (one per physical core). 8 threads adds 3-8%.
The 6-thread dip is mostly the accounting: on 4 cores, two cores carry two threads each, and the
batch counts the slowest thread (an earlier TMR probe saw the same). Width barely matters once
saturated (Rule 3). The compute-bound LCG keeps scaling through SMT.

### Capability-token prototype (`src/cap.rs`)

A downstream `Clflushopt` token (CPUID-proven) plus `with_clflushopt!`, which emits one entry fn
per level. Its `#[target_feature]` is fearless_simd's exact level list plus `clflushopt`, around a
generic `#[inline(always)]` body. In `k_wflush_fscap_256/512` and `k_flush_fscap`, the stdarch
`_mm_clflushopt` **inlines inside the fearless loop**: no call, and the same 8x unroll as
`flush_range_to_dram`. Output is identical. The cost is a copied feature list per level;
`level_features_are_detected` checks it against the CPU, and it must be re-checked on any
fearless_simd upgrade. RFC 3525 (struct target features) would remove the need for it.

### Reproduce

```bash
cargo test --release                    # port correctness (also: cargo test, debug)
python asm_check.py                     # codegen gate; --dump <kernel> prints its loops
cargo run --release                     # both regimes, ~10 min, needs SeLockMemoryPrivilege
cargo run --release -- --quick          # 1/4/8 threads, 1 GiB/thread, 3 samples
cargo run --release -- --regime dram --only verify4,pfv --threads 1,2,4,8
cargo run --release -- --pages small    # 4 KiB pages, no privilege needed
```

Large pages use one plain `VirtualAlloc2(MEM_RESERVE|MEM_COMMIT|MEM_LARGE_PAGES)` per thread,
with matching alignment (`src/mem.rs`). The box must be idle while it runs.
