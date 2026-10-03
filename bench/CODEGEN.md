# Codegen: plumb kernels vs their TMR-style twins

Generated with `python asm_check.py --no-build --twins` (commit 4254f13, nightly 1.100, release
profile, `target-cpu=x86-64-v3`). For each plumb kernel with a `twin` rule in `expect/lines.json`,
and each kind of key memory op its TMR-style twin has, the figure is the instructions per op in
the kernel's densest innermost loop (lower is better). `asm_check.py` fails a kernel that is
more than 25% looser than its twin (`twin_tol`).

How to read it:

- It is a codegen check, not a performance predictor. `k_verify4_pvch_512` (the 512-bit verify's
  old shape) is denser than TMR's loop yet 16% slower from L2, because instruction count doesn't
  see that its `vpxorq` + `vpternlogq` pair replaced one fused `vpternlogq` (RESULTS.md).
- "Densest innermost loop" can be a cold loop. LLVM vectorizes remainder loops too, so for some
  kernels (the verifies, the flush in `wflush`) the figure comes from a loop that runs a few
  times per call. The throughput tables in RESULTS.md are the verdict.
- The two looser rows are the per-line `asm!` flush in a user loop (`wflush_pltok`): an `asm!`
  address is a register operand, so each line costs a `lea` that the intrinsic twin folds into a
  displacement. It runs at 98-101% of TMR's in DRAM; the rule allows it 50% (`twin_tol`).

| plumb kernel | TMR-style twin | op | plumb instrs/op | twin instrs/op | plumb vs twin |
|---|---|---|---:|---:|---|
| `k_copymd_pl` | `k_copymd_tmr` | movdir64b | 5.00 | 5.00 | equal |
| `k_fill_pv_128` | `k_fill_tmr_128` | store | 1.09 | 1.50 | denser (27% fewer) |
| `k_fill_pv_256` | `k_fill_tmr_256` | store | 1.09 | 1.50 | denser (27% fewer) |
| `k_fill_pv_512` | `k_fill_tmr_512` | store | 1.09 | 1.50 | denser (27% fewer) |
| `k_fillflush_pl_256` | `k_fillflush_tmr_256` | flush | 1.09 | 1.50 | denser (27% fewer) |
| `k_fillflush_pl_256` | `k_fillflush_tmr_256` | store | 1.09 | 1.50 | denser (27% fewer) |
| `k_fillflush_pl_512` | `k_fillflush_tmr_512` | flush | 1.09 | 1.50 | denser (27% fewer) |
| `k_fillflush_pl_512` | `k_fillflush_tmr_512` | store | 1.09 | 1.50 | denser (27% fewer) |
| `k_flush_pl` | `k_flush_tmr` | flush | 1.09 | 1.50 | denser (27% fewer) |
| `k_ntw_pl_128` | `k_ntw_tmr_128` | nt | 4.00 | 4.75 | denser (16% fewer) |
| `k_ntw_pl_256` | `k_ntw_tmr_256` | nt | 4.00 | 4.75 | denser (16% fewer) |
| `k_ntw_pl_512` | `k_ntw_tmr_512` | nt | 4.00 | 4.75 | denser (16% fewer) |
| `k_posv_pv_128` | `k_posv_tmr_128` | load | 4.38 | 4.75 | denser (8% fewer) |
| `k_posv_pv_256` | `k_posv_tmr_256` | load | 4.38 | 4.75 | denser (8% fewer) |
| `k_posv_pv_512` | `k_posv_tmr_512` | load | 3.00 | 3.50 | denser (14% fewer) |
| `k_refresh_pl_256` | `k_refresh_tmr_256` | flush | 1.09 | 1.50 | denser (27% fewer) |
| `k_refresh_pl_256` | `k_refresh_tmr_256` | load | 2.19 | 2.38 | denser (8% fewer) |
| `k_refresh_pl_256` | `k_refresh_tmr_256` | store | 1.09 | 1.38 | denser (20% fewer) |
| `k_refresh_pl_512` | `k_refresh_tmr_512` | flush | 1.09 | 1.50 | denser (27% fewer) |
| `k_refresh_pl_512` | `k_refresh_tmr_512` | load | 1.88 | 1.88 | equal |
| `k_refresh_pl_512` | `k_refresh_tmr_512` | store | 1.09 | 1.38 | denser (20% fewer) |
| `k_sb_pl_256` | `k_sb_tmr_256` | flush | 1.09 | 1.50 | denser (27% fewer) |
| `k_sb_pl_256` | `k_sb_tmr_256` | load | 2.19 | 2.38 | denser (8% fewer) |
| `k_sb_pl_256` | `k_sb_tmr_256` | store | 1.09 | 1.38 | denser (20% fewer) |
| `k_sb_pl_512` | `k_sb_tmr_512` | flush | 1.09 | 1.50 | denser (27% fewer) |
| `k_sb_pl_512` | `k_sb_tmr_512` | load | 1.88 | 1.88 | equal |
| `k_sb_pl_512` | `k_sb_tmr_512` | store | 1.38 | 1.38 | equal |
| `k_sbnf_pl_256` | `k_sbnf_tmr_256` | flush | 1.09 | 1.50 | denser (27% fewer) |
| `k_sbnf_pl_256` | `k_sbnf_tmr_256` | load | 2.19 | 2.38 | denser (8% fewer) |
| `k_sbnf_pl_256` | `k_sbnf_tmr_256` | store | 1.09 | 1.38 | denser (20% fewer) |
| `k_sbnf_pl_512` | `k_sbnf_tmr_512` | flush | 1.09 | 1.50 | denser (27% fewer) |
| `k_sbnf_pl_512` | `k_sbnf_tmr_512` | load | 1.88 | 1.88 | equal |
| `k_sbnf_pl_512` | `k_sbnf_tmr_512` | store | 1.38 | 1.38 | equal |
| `k_simplent_pl_256` | `k_simplent_tmr_256` | load | 4.38 | 4.75 | denser (8% fewer) |
| `k_simplent_pl_256` | `k_simplent_tmr_256` | nt | 4.00 | 4.75 | denser (16% fewer) |
| `k_simplent_pl_512` | `k_simplent_tmr_512` | load | 3.00 | 3.50 | denser (14% fewer) |
| `k_simplent_pl_512` | `k_simplent_tmr_512` | nt | 4.00 | 5.25 | denser (24% fewer) |
| `k_verify4_pv_128` | `k_verify4_tmr_128` | load | 2.19 | 2.38 | denser (8% fewer) |
| `k_verify4_pv_256` | `k_verify4_tmr_256` | load | 2.19 | 2.38 | denser (8% fewer) |
| `k_verify4_pv_512` | `k_verify4_tmr_512` | load | 1.88 | 1.88 | equal |
| `k_wflush_plentry_256` | `k_wflush_tmr_256` | flush | 1.09 | 3.50 | denser (69% fewer) |
| `k_wflush_plentry_256` | `k_wflush_tmr_256` | store | 1.69 | 1.75 | denser (4% fewer) |
| `k_wflush_plentry_512` | `k_wflush_tmr_512` | flush | 1.09 | 2.50 | denser (56% fewer) |
| `k_wflush_plentry_512` | `k_wflush_tmr_512` | store | 2.38 | 2.50 | denser (5% fewer) |
| `k_wflush_pltok_256` | `k_wflush_tmr_256` | flush | 1.09 | 3.50 | denser (69% fewer) |
| `k_wflush_pltok_256` | `k_wflush_tmr_256` | store | 2.25 | 1.75 | looser (29% more) |
| `k_wflush_pltok_512` | `k_wflush_tmr_512` | flush | 1.09 | 2.50 | denser (56% fewer) |
| `k_wflush_pltok_512` | `k_wflush_tmr_512` | store | 3.50 | 2.50 | looser (40% more) |
