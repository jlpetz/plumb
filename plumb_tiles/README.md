# plumb_tiles

Safe wrappers for the x86 2D tile instruction sets: Intel AMX now, shaped so the x86 ACE
extension can be added later.

It was written for a memory tester. One `TILELOADD` or `TILESTORED` moves 16 rows of 64 bytes
at any byte stride, an access shape no vector instruction has. The AMX dot products
(`TDPB[SU][SU]D`, `TDPBF16PS`, `TDPFP16PS`) are included too, mainly as a dense power/heat load.

```rust
use plumb_tiles::{Amx, T0};

let Some(amx) = Amx::try_new() else { return };       // CPUID + XCR0 (+ Linux permission)
let src = vec![0u64; 16 * 512];                        // 16 pages
let mut dst = vec![0u64; 16 * 512];
amx.with_tiles(|t| {                                   // LDTILECFG ... TILERELEASE
    t.load_u64::<T0>(&src, 4096);                      // one 64-byte line from each page
    t.store_u64::<T0>(&mut dst, 4096);
});
```

- **Stable Rust**: every AMX instruction is an `asm!` block. No target features, no nightly.
  (An ACE backend will need AVX-512 where it's called; the crate docs explain why.)
- **One tile shape**: all eight tiles are always 16 rows x 64 bytes. That's the largest AMX
  shape and the only ACE shape.
- **Safe**: tokens prove CPU and OS support. A closure-scoped session configures the tiles and
  releases them even on panic. A session refuses to start (panics) while the thread's tiles are
  configured, whether by an outer session (nesting), another copy of the crate or another AMX
  library. Memory ops are bounds-checked with overflow checks. Compute ops reject repeated
  tiles at compile time.
- **Platforms**: Windows 11 / Server 2022+ (tested). Linux via `arch_prctl(ARCH_REQ_XCOMP_PERM)`
  (compile-checked only). x86_64 with 64-bit pointers only (not x32).

The crate docs cover the safety argument, the compute layouts (VNNI-packed B), and how AMX
compares with ACE, including the roadmap for an ACE backend.

Licensed under MIT or Apache-2.0, at your option.
