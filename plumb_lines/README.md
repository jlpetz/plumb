# plumb_lines

Cache-line and memory-ordering operations for
[fearless_simd](https://crates.io/crates/fearless_simd): the instructions a memory tester (or any
bandwidth-bound kernel) needs that a portable SIMD library doesn't have.

| Module | What |
|---|---|
| `view` | `as_vectors(_mut)`: split `&[u64]` into `(head, &[V], tail)` with `V` an aligned fearless vector (`u64x8<S>` etc.). Alignment is in the type, and loops over `&[V]` compile to pointer walks. |
| `nt` | Non-temporal stores for every fearless vector, in a `nontemporal` scope that `SFENCE`s on exit. |
| `flush` | CLFLUSHOPT: `Clflushopt` token, `flush`, `flush_line`, and `flush_after` (write, then flush everything written before anything reads it). |
| `direct` | MOVDIR64B: `Movdir64b` token, 64-byte-aligned `Line`, and a `direct` scope that `SFENCE`s on exit. |

```rust
use fearless_simd::{prelude::*, Avx2, Level, u64x4};
use plumb_lines::NtStore;

/// Streaming fill. Level-generic code takes the vector type with an `NtStore<S>` bound and is
/// called with a concrete token (see the `nt` module docs).
#[fearless_simd_macros::simd]
fn nt_fill<S: Simd, V: SimdInt<S, Element = u64> + NtStore<S>>(simd: S, buf: &mut [u64], value: u64) {
    let (head, vectors, tail) = plumb_lines::as_vectors_mut::<S, V>(simd, buf);
    head.fill(value);
    tail.fill(value);
    let v = V::splat(simd, value);
    plumb_lines::nontemporal(simd, vectors, |w| w.fill_with(|_| v)); // 4 streaming stores/iteration
} // SFENCE ran when the scope ended; `vectors` was borrowed until then

let mut buf = vec![0u64; 1000];
let avx2: Avx2 = Level::new().as_avx2().expect("an x86-64-v3 CPU");
nt_fill::<Avx2, u64x4<Avx2>>(avx2, &mut buf[3..], 7); // any alignment: head/tail are scalar
assert!(buf[3..].iter().all(|&w| w == 7));
```

## Safety model

NT stores and MOVDIR64B are weakly ordered: until the writing thread executes `SFENCE`, nothing
may access the memory (reads or writes, including another streaming store) and it must not be
handed to another thread. Rust's fences don't provide that on x86. So
both happen inside **closure scopes** that borrow the destination mutably and fence when they
end, including on unwind. The writer they pass is by value, write-once and `!Send`, so each slot is
stored at most once, never read inside the scope, and only from this thread. (A guard object can't do this: `mem::forget` would
skip the fence.) Flushing is different: caches are coherent, so `flush_after` is about making a
verify read come from DRAM, not memory safety, and it gives ordinary `&mut` access.

## Codegen

Every hot-path function is `#[inline(always)]`. Call the scopes from inside a fearless `#[simd]`
function or `kernel!`, so they inline into the function with the target features. The
workspace's `bench/asm_check.py` checks the loops: the expected instruction, the named width, no
calls, no `memset`, and instructions per memory op within 25% of the hand-written per-width
twin (equal or denser for every range and view kernel). Throughput is
at parity with hand-written kernels (see the workspace README).

One loop shape to know about: for an OR-accumulate verify over `as_vectors` at 512 bits, walk
one group of four vectors per iteration (`while let [a, b, c, d, rest @ ..] = mid`). LLVM unrolls
an `as_chunks::<4>` loop there and reassociates the ORs, which turns each fused
`vpternlogq acc, p, [mem]` into two instructions; from L2 that was 16% slower. At 128 and 256 bits
the unrolled loop is fine (faster, if anything).

## Toolchains

Stable Rust uses `asm!` for CLFLUSHOPT and MOVDIR64B (`asm!` needs no target feature and inlines
anywhere). The `nightly` feature uses stdarch's `_mm_clflushopt` (rust-lang/rust#157096) for range
flushes and exports `with_clflushopt!`, which builds a fearless kernel entry point with
`clflushopt` added to the level's target features, so the intrinsic also inlines inside fearless
loops. MOVDIR64B has no stdarch intrinsic yet, so it's always `asm!`.

## Minimum supported Rust Version (MSRV)

This version of plumb_lines has been verified to compile with **Rust 1.89** and later, the same MSRV
as fearless_simd 1.0. The `nightly` feature needs a nightly
toolchain. x86_64 only.

Future versions might increase the Rust version requirement. This will be accompanied by a minor
version bump.

## Community

Discussion happens in the [Linebender Zulip](https://xi.zulipchat.com/), in
[#simd](https://xi.zulipchat.com/#narrow/channel/514230-simd), where fearless_simd is discussed.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Contributions are welcome by pull request. The [Rust code of conduct] applies.
Please feel free to add your name to the [AUTHORS] file in any substantive pull request.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[AUTHORS]: ../AUTHORS
