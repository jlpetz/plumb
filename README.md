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

Status: experimental, `0.1`, not yet on crates.io. Plan and decisions: [PLAN.md](PLAN.md); tracker:
[TODO.md](TODO.md).

## What's been shown so far

- **Codegen parity.** Generic fearless kernels plus plumb_lines produce the same hot loops as
  TMR's per-width `macro_rules!` kernels at 128/256/512 bits: the same width, 4 independent
  accumulators, no `memset` collapse, no calls in loops. `bench/asm_check.py` checks every
  kernel against its TMR-style twin in instructions per memory op (within 25%). Most are equal or
  denser, e.g. fill 1.09 vs 1.50, range flush 1.09 vs 1.50, positional NT write 4.00 vs 4.75 (four
  NT stores per `asm!` block). The one looser kernel is a per-line `asm!` flush in a user loop,
  which costs a `lea` per line (documented). Table: [bench/CODEGEN.md](bench/CODEGEN.md).
- **Throughput parity** ([bench/RESULTS.md](bench/RESULTS.md); 1 GiB pages, 1/2/4/6/8 threads):
  every plumb kernel is at 97-105% of its TMR twin in DRAM (Refresh at 256 bits is 110-117%,
  not yet explained: TODO 11), and at parity or faster from L2. One
  loop-shape rule came out of it: at 512 bits, walk an OR-accumulate verify one group of four
  vectors per iteration. LLVM unrolls an `as_chunks` loop there and loses the fused
  `vpternlogq`, which cost 16% from L2 (the TODO 84 runs blamed indexed addressing; it wasn't).
- **`asm!` costs are front-end only.** The `lea`s an `asm!` store or flush costs, and the lost
  automatic unrolling, made no measurable difference to DRAM bandwidth.
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
cd bench && python asm_check.py --toolchain nightly # the codegen gate (bench, nightly feature)
python asm_check.py --package plumb_lines --example asm_kernels --toolchain stable
cargo +nightly run --release -p plumb-bench        # benchmark: idle box, ~25 min, 1 GiB pages
```

The benchmark needs `SeLockMemoryPrivilege` for 1 GiB pages (or `--pages small`). It allocates
one plain large-page region per thread (2 GiB by default).

## Minimum supported Rust Version (MSRV)

This version of plumb has been verified to compile with **Rust 1.89** and later, the same MSRV
as fearless_simd 1.0. The `nightly` feature of plumb_lines needs a
nightly toolchain.

Future versions might increase the Rust version requirement. This will be accompanied by a minor
version bump.

## Community

Discussion happens in the [Linebender Zulip](https://xi.zulipchat.com/), in
[#simd](https://xi.zulipchat.com/#narrow/channel/514230-simd), where fearless_simd is discussed.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](./LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Contributions are welcome by pull request. The [Rust code of conduct] applies.
Please feel free to add your name to the [AUTHORS] file in any substantive pull request.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[AUTHORS]: ./AUTHORS
