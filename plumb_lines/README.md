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
use fearless_simd::{prelude::*, u64x8};

#[fearless_simd_macros::simd]
fn nt_fill<S: Simd>(simd: S, buf: &mut [u64], base: u64) {
    let (head, vectors, tail) = plumb_lines::as_vectors_mut::<S, u64x8<S>>(simd, buf);
    head.fill(base);
    tail.fill(base);
    plumb_lines::nontemporal(simd, vectors, |w| {
        let v = u64x8::splat(simd, base);
        w.fill_with(|_| v); // 4 streaming stores per iteration
    }); // SFENCE here; `vectors` is borrowed until then
}
```

## Safety model

NT stores and MOVDIR64B are weakly ordered: the writing thread must `SFENCE` before anything
reads the memory or hands it to another thread, and Rust's fences don't provide that on x86. So
both happen inside **closure scopes** that borrow the destination mutably, give write-only access,
and fence when they end, including on unwind. (A guard object can't do this: `mem::forget` would
skip the fence.) Flushing is different: caches are coherent, so `flush_after` is about making a
verify read come from DRAM, not memory safety, and it gives ordinary `&mut` access.

## Codegen

Every hot-path function is `#[inline(always)]`. Call the scopes from inside a fearless `#[simd]`
function or `kernel!`, so they inline into the function with the target features. The
workspace's `bench/asm_check.py` checks the loops: the expected instruction, the named width, no
calls, no `memset`. Measured against hand-written per-width kernels, NT write loops are
instruction-for-instruction identical, and throughput is at parity (see the workspace README).

## Toolchains

Stable Rust uses `asm!` for CLFLUSHOPT and MOVDIR64B (`asm!` needs no target feature and inlines
anywhere). The `nightly` feature uses stdarch's `_mm_clflushopt` (rust-lang/rust#157096) for range
flushes and exports `with_clflushopt!`, which builds a fearless kernel entry point with
`clflushopt` added to the level's target features, so the intrinsic also inlines inside fearless
loops. MOVDIR64B has no stdarch intrinsic yet, so it's always `asm!`.

x86_64 only. Licensed under MIT or Apache-2.0.
