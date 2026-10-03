# plumb

Extension crates for [fearless_simd](https://crates.io/crates/fearless_simd) that add the
instructions a memory tester needs and a portable SIMD library doesn't have. Built for
[TMR](https://github.com/jlpetz/Test-Memory-R), a Rust DDR5 memory stability tester, following
advice from fearless_simd's maintainers to try these in extension crates first.

| Crate | What |
|---|---|
| [`plumb_lines`](plumb_lines) | Aligned vector views (`as_vectors`), **scoped non-temporal stores** that `SFENCE` on exit, CLFLUSHOPT (`flush_after`: write, then flush everything written), MOVDIR64B direct stores. |
| [`plumb_tiles`](plumb_tiles) | Safe x86 tile instructions: Intel AMX now (strided tile load/store from memory, INT8/BF16/FP16 dot products), designed for the x86 ACE extension later. |
| [`bench`](bench) | The evidence: an asm gate (`asm_check.py`), equivalence tests, and a large-page thread-sweep benchmark comparing these crates with TMR's hand-written per-width kernels. |

Status: experimental, `0.1`, not published. Plan and decisions: [PLAN.md](PLAN.md); tracker:
[TODO.md](TODO.md).

## What's been shown so far

- **Codegen parity.** Generic fearless kernels plus plumb_lines produce the same hot loops as
  TMR's per-width `macro_rules!` kernels at 128/256/512 bits: the same width, 4 independent
  accumulators, no `memset` collapse, no calls in loops. `bench/asm_check.py` checks every
  kernel against its TMR-style twin in instructions per memory op (within 25%). Most are equal or
  denser, e.g. fill 1.09 vs 1.50, range flush 1.19 vs 1.50, verify 2.19 vs 2.38. The NT write loop
  has the same 19 instructions as TMR's (registers and order differ). The one looser kernel is
  a per-line `asm!` flush in a user loop, which costs a `lea` per line (documented).
- **Throughput parity** (TODO 84 runs, 1 GiB pages, 1-8 threads): within about ±2% everywhere
  once verify loops are pointer walks. A `chunks_exact` loop over the `u64` slice gets indexed
  addressing and lost 16% on a 512-bit L2-resident verify; `as_vectors` gives the pointer walk in
  safe code.
- **Ported TMR tests.** StuckBit, Refresh and SimpleNT, each written in TMR's style and in plumb
  style, report identical errors under injected faults and leave identical memory. Before every
  injection the test checks memory holds what TMR's sequence should have written (phase pattern,
  global positional index), so a wrong pattern or a skipped verify fails.
- **Instructions fearless_simd can't carry yet**, made to work: NT stores (any fearless vector,
  any level), CLFLUSHOPT inlined inside fearless loops (via a capability-token entry macro on
  nightly, or `asm!` on stable), MOVDIR64B, and AMX tiles.

## Running

```bash
cargo test -p plumb_lines                          # stable
cargo +nightly test -p plumb_lines --features nightly
cargo test -p plumb_tiles                          # needs an AMX CPU for the hardware tests
cargo +nightly test --release -p plumb-bench       # equivalence tests
cd bench && python asm_check.py                    # the codegen gate (bench, nightly feature)
python asm_check.py --package plumb_lines --example asm_kernels --toolchain stable
cargo +nightly run --release -p plumb-bench        # benchmark: idle box, ~15 min, 1 GiB pages
```

The benchmark needs `SeLockMemoryPrivilege` for 1 GiB pages (or `--pages small`). It allocates
one plain large-page region per thread (2 GiB by default).

x86_64 only. Licensed under either of MIT or Apache-2.0, at your option.
