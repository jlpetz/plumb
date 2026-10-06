# Results: plumb kernels vs TMR-style kernels (2026-10-03)

Xeon 6975P-C (Granite Rapids, 4 cores / 8 threads exposed), 64 GB, Windows Server 2025, Rust
nightly 1.100 (2026-09-25). `cargo +nightly run --release -p plumb-bench`. The box was idle:
CPU under 6% before the run, and no cargo/rustc/tmr process appeared at checks every 5 minutes.

- **DRAM regime**: 2 GiB per thread on 1 GiB pages (plain `VirtualAlloc2` commits), threads
  pinned physical-cores-first, 1/2/4/6/8 threads, aggregate GiB/s counted at the slowest thread,
  median of 5. Percentages are against the TMR-style kernel of the same width.
- **L2 regime**: one thread, 256 KiB warm, 21 samples x 2000 reps, median.

The logs and CSVs named here are local (gitignored); the tables below are copied from them.

Runs: the full sweep at commit 825c9a9 (`run-2026-10-03.log`, `results-2026-10-03.csv`); then
the groups that use the view verify (`verify4`, `sb`, `sbnf`, `refresh`) re-run at 4254f13 after
the 512-bit verify fix (`run-2026-10-03-verify.log`). Those groups below are from the re-run.
`run-2026-10-03-repeat-oldshape.log` is an accidental repeat of the earlier shape on the old
binary; it agrees with the full sweep within about 1%, which is a fair measure of run-to-run
noise.

## Summary

- **DRAM: every plumb kernel is at parity with its TMR twin** (98-105% at every thread count),
  including the ported StuckBit, Refresh and SimpleNT tests; the one exception is Refresh at 256
  bits, where plumb is faster (below).
- **L2: at parity or faster**, after one fix: the 512-bit view verify was 84% of TMR's. The cause
  is LLVM unrolling the `as_chunks` OR-accumulate loop and reassociating it, which replaces the
  fused `vpternlogq acc, p, [mem]` (one uop per load) with `vpxorq` + `vpternlogq`. It was not
  indexed addressing, as the TODO 84 findings said: `pv_512` uses base+displacement and was just
  as slow. One quad per iteration (`pvpat`) keeps the fused form, at 99.8-101% of TMR's. At 128
  and 256 bits the unrolled loop is the faster one (103%, 110%), so `verify4_view` picks by width.
  `pvch_512` and `pv8_512` (8 accumulators: no help) keep the slow shape measurable.
- **`asm!` costs are front-end only.** Dropping three `lea`s per four NT stores (19 -> 16
  instructions) changed nothing at DRAM speed (`ntw` 99-101%). The per-line `asm!` flush
  (`wflush pltok`, one `lea` per line) and the intrinsic (`plentry`) both run at 98-101%.
- **The lost unroll was not a performance issue.** The StuckBit port with the fill's old shape
  (one zmm store per iteration, `plplain_512`) runs at 97-101% of TMR's in DRAM (the unrolled
  `pl_512`: 99-101%) and 101.4% in L2.
- **Refresh at 256 bits: plumb 110-117% of the TMR-style port** (three runs). The TMR-style
  port's 256-bit verify loop carries an extra induction variable (14 instructions per 4 loads, 10
  at 512); not yet shown to be the cause, and the loop is this bench's port of TMR's macro, so
  check TMR-APP's own loop before reading this as "plumb beats TMR".
- **AMX: no bandwidth advantage.** Tile loads read at 95-100% of the zmm verify's rate, tile
  stores fill at 98-99% of zmm stores, the stride-4096 tile load matches 16 zmm loads (98-100%),
  and a tile copy is 62-67% of NT-512 (tile stores are ordinary stores, so they pay the
  read-for-ownership that NT and MOVDIR64B avoid). AMX stays a power/heat
  load and an alternative access path, not a faster one.

## L2 regime

```text
== fill | constant fill (StuckBit/Refresh write) (21 samples x 2000 reps)
   variant           median          [min .. max]    vs tmr
   tmr_128            58.41   [  56.72 ..   58.52]   
   fs_128             58.44   [  55.31 ..   58.55]    100.1%
   tmr_256            58.14   [  55.71 ..   58.23]   
   fs_256             58.06   [  55.89 ..   58.18]     99.9%
   tmr_512            57.33   [  53.92 ..   57.44]   
   fs_512             57.37   [  55.25 ..   57.50]    100.1%
   fs_auto            57.37   [  55.82 ..   57.48]    100.1%
   pv_128             58.35   [  55.55 ..   58.48]     99.9%
   pv_256             57.97   [  57.29 ..   58.06]     99.7%
   pv_512             57.27   [  56.22 ..   57.37]     99.9%
```

```text
== verify4 | 4-accumulator verify (StuckBit verify) (21 samples x 2000 reps)
   variant           median          [min .. max]    vs tmr
   tmr_128            78.44   [  77.04 ..   78.66]   
   fs_128             81.73   [  75.80 ..   81.88]    104.2%
   tmr_256           113.47   [  98.76 ..  113.66]   
   fs_256            116.57   [ 111.14 ..  117.97]    102.7%
   tmr_512           172.08   [ 146.66 ..  172.79]   
   fs_512            143.26   [ 123.29 ..  144.10]     83.3%
   fs_auto           143.05   [ 116.75 ..  144.54]     83.1%
   pv_128             81.14   [  77.98 ..   81.25]    103.4%
   pv_256            124.96   [ 109.62 ..  125.61]    110.1%
   pv_512            174.22   [ 149.69 ..  175.09]    101.2%
   fsptr_128          80.58   [  76.61 ..   80.76]    102.7%
   fsptr_256         115.57   [ 110.82 ..  115.71]    101.9%
   fsptr_512         175.00   [ 173.44 ..  175.42]    101.7%
   pvpat_128          79.81   [  79.63 ..   80.12]    101.8%
   pvpat_256         115.45   [ 114.99 ..  115.68]    101.7%
   fssplit_128        79.34   [  75.23 ..   79.48]    101.2%
   fssplit_256       121.21   [ 105.75 ..  121.78]    106.8%
   fssplit_512       140.31   [ 138.10 ..  142.20]     81.5%
   pvch_512          145.53   [ 129.80 ..  146.41]     84.6%
   pv8_512           142.65   [ 139.74 ..  145.45]     82.9%
   tmrhelper_512       9.37   [   9.31 ..    9.39]   
   fshelper_512       16.78   [  16.46 ..   16.83]      9.8%
```

```text
== posw | positional write idx^base (SimpleTest Mode 0/1) (21 samples x 2000 reps)
   variant           median          [min .. max]    vs tmr
   tmr_128            58.32   [  55.64 ..   58.43]   
   fs_128             58.37   [  55.22 ..   58.46]    100.1%
   tmr_256            57.95   [  57.84 ..   58.03]   
   fs_256             57.98   [  56.33 ..   58.09]    100.0%
   tmr_512            57.60   [  56.73 ..   57.73]   
   fs_512             57.65   [  57.24 ..   57.70]    100.1%
```

```text
== posv | positional verify, 1 accumulator (SimpleTest verify) (21 samples x 2000 reps)
   variant           median          [min .. max]    vs tmr
   tmr_128            38.89   [  37.48 ..   38.95]   
   fs_128             37.30   [  36.22 ..   37.35]     95.9%
   tmr_256            74.80   [  66.96 ..   75.01]   
   fs_256             74.88   [  72.24 ..   75.03]    100.1%
   tmr_512           130.54   [ 129.44 ..  131.38]   
   fs_512            132.43   [ 131.55 ..  134.00]    101.4%
   pv_128             41.66   [  38.96 ..   41.81]    107.1%
   pv_256             80.30   [  68.41 ..   80.87]    107.4%
   pv_512            128.38   [ 116.70 ..  132.87]     98.4%
```

```text
== lcgw | LCG write state*m+a (SimpleTest Mode 2, 64-bit mul) (21 samples x 2000 reps)
   variant           median          [min .. max]    vs tmr
   tmr_128             6.21   [   6.10 ..    6.22]   
   fs_128              6.15   [   6.13 ..    6.15]     99.0%
   tmr_256            12.43   [  12.39 ..   12.45]   
   fs_256             12.30   [  12.21 ..   12.31]     99.0%
   tmr_512            12.21   [  12.07 ..   12.23]   
   fs_512             12.21   [  12.02 ..   12.23]    100.0%
```

```text
== flush | CLFLUSHOPT range + MFENCE (flush_range_to_dram) (210 samples x 1 reps)
   variant           median          [min .. max]    vs tmr
   tmr                11.10   [   6.15 ..   11.46]   
   fsintr             11.15   [   8.30 ..   11.57]    100.5%
   fsasm              11.15   [   7.00 ..   11.63]    100.5%
   fscap              11.15   [   8.25 ..   11.63]    100.5%
   pl                 11.15   [   5.61 ..   11.63]    100.5%
```

```text
== sbnf | TMR StuckBit port, no flush (3 phases) (21 samples x 2000 reps)
   variant           median          [min .. max]    vs tmr
   tmr_256            72.83   [  72.14 ..   73.10]   
   pl_256             74.43   [  73.69 ..   74.64]    102.2%
   tmr_512            80.66   [  79.63 ..   80.74]   
   pl_512             81.08   [  80.36 ..   81.13]    100.5%
   plplain_512        81.81   [  80.75 ..   81.87]    101.4%
```

## DRAM regime

```text
== fill | constant fill (StuckBit/Refresh write)
   variant                 1T            2T            4T            6T            8T
   tmr_128       10.90         20.66         35.87         36.08         41.47       
   fs_128        10.90 (100%)  20.59 (100%)  35.67 ( 99%)  36.32 (101%)  41.67 (100%)
   tmr_256       11.54         21.76         37.35         33.63         39.13       
   fs_256        11.52 (100%)  21.67 (100%)  37.66 (101%)  32.25 ( 96%)  38.40 ( 98%)
   tmr_512       12.44         23.56         40.56         34.68         40.44       
   fs_512        12.48 (100%)  23.51 (100%)  40.61 (100%)  34.36 ( 99%)  40.23 ( 99%)
   fs_auto       12.55 (101%)  23.55 (100%)  40.49 (100%)  34.65 (100%)  40.56 (100%)
   pv_128        10.90 (100%)  20.57 (100%)  36.31 (101%)  36.29 (101%)  41.51 (100%)
   pv_256        11.68 (101%)  21.99 (101%)  37.37 (100%)  32.53 ( 97%)  39.23 (100%)
   pv_512        12.60 (101%)  23.65 (100%)  40.92 (101%)  35.13 (101%)  40.59 (100%)
   spread (max-min)/median: typical 2.3%, worst 15.7% (fs_auto @ 4T); group took 79 s
```

```text
== verify4 | 4-accumulator verify (StuckBit verify)
   variant                 1T            2T            4T            6T            8T
   tmr_128       18.36         32.18         59.22         47.39         59.09       
   fs_128        17.37 ( 95%)  31.40 ( 98%)  58.97 (100%)  48.10 (101%)  59.45 (101%)
   tmr_256       19.58         33.03         59.79         49.34         63.34       
   fs_256        18.35 ( 94%)  32.37 ( 98%)  59.63 (100%)  49.26 (100%)  63.60 (100%)
   tmr_512       15.29         29.97         54.69         44.96         55.87       
   fs_512        15.12 ( 99%)  28.94 ( 97%)  53.11 ( 97%)  44.62 ( 99%)  54.48 ( 98%)
   fs_auto       15.19 ( 99%)  28.94 ( 97%)  52.97 ( 97%)  44.78 (100%)  55.75 (100%)
   pv_128        18.29 (100%)  32.42 (101%)  58.95 (100%)  48.26 (102%)  60.76 (103%)
   pv_256        19.38 ( 99%)  33.07 (100%)  60.55 (101%)  49.67 (101%)  64.68 (102%)
   pv_512        15.31 (100%)  29.92 (100%)  54.54 (100%)  44.96 (100%)  55.18 ( 99%)
   fsptr_128     18.06 ( 98%)  32.18 (100%)  59.61 (101%)  47.99 (101%)  60.04 (102%)
   fsptr_256     19.50 (100%)  33.08 (100%)  59.88 (100%)  49.49 (100%)  63.74 (101%)
   fsptr_512     15.30 (100%)  29.90 (100%)  54.68 (100%)  44.87 (100%)  54.29 ( 97%)
   spread (max-min)/median: typical 1.6%, worst 8.8% (fs_512 @ 8T); group took 109 s
```

```text
== posw | positional write idx^base (SimpleTest Mode 0/1)
   variant                 1T            2T            4T            6T            8T
   tmr_128       10.84         20.42         34.73         35.34         41.14       
   fs_128        10.90 (101%)  20.44 (100%)  35.79 (103%)  35.43 (100%)  40.86 ( 99%)
   tmr_256       11.49         21.61         37.27         31.75         38.58       
   fs_256        11.57 (101%)  21.68 (100%)  36.82 ( 99%)  31.87 (100%)  38.14 ( 99%)
   tmr_512       12.30         23.01         39.58         33.83         40.46       
   fs_512        12.26 (100%)  22.92 (100%)  38.28 ( 97%)  34.18 (101%)  40.41 (100%)
   spread (max-min)/median: typical 3.2%, worst 10.4% (tmr_128 @ 8T); group took 48 s
```

```text
== posv | positional verify, 1 accumulator (SimpleTest verify)
   variant                 1T            2T            4T            6T            8T
   tmr_128       17.64         31.50         58.21         46.80         57.94       
   fs_128        17.69 (100%)  31.82 (101%)  58.91 (101%)  46.86 (100%)  58.25 (101%)
   tmr_256       17.47         31.17         58.31         48.26         61.75       
   fs_256        17.04 ( 98%)  30.83 ( 99%)  58.39 (100%)  48.66 (101%)  60.94 ( 99%)
   tmr_512       13.13         25.44         47.57         41.76         51.87       
   fs_512        13.74 (105%)  26.27 (103%)  48.21 (101%)  43.10 (103%)  53.64 (103%)
   pv_128        18.00 (102%)  32.01 (102%)  59.14 (102%)  46.45 ( 99%)  58.11 (100%)
   pv_256        17.96 (103%)  31.90 (102%)  58.94 (101%)  48.61 (101%)  62.29 (101%)
   pv_512        13.81 (105%)  25.99 (102%)  47.73 (100%)  42.80 (102%)  53.42 (103%)
   spread (max-min)/median: typical 2.3%, worst 8.3% (pv_128 @ 4T); group took 78 s
```

```text
== lcgw | LCG write state*m+a (SimpleTest Mode 2, 64-bit mul)
   variant                 1T            2T            4T            6T            8T
   tmr_128        6.16         12.31         24.51         32.29         40.11       
   fs_128         6.09 ( 99%)  12.17 ( 99%)  24.23 ( 99%)  31.95 ( 99%)  39.04 ( 97%)
   tmr_256       10.89         20.52         35.46         32.86         38.29       
   fs_256        10.92 (100%)  20.49 (100%)  36.35 (103%)  32.79 (100%)  38.71 (101%)
   tmr_512       11.83         22.81         39.31         34.52         39.67       
   fs_512        11.86 (100%)  22.69 ( 99%)  39.00 ( 99%)  34.26 ( 99%)  40.01 (101%)
   spread (max-min)/median: typical 1.8%, worst 11.3% (tmr_128 @ 4T); group took 53 s
```

```text
== flush | CLFLUSHOPT range + MFENCE (flush_range_to_dram)
   variant                 1T            2T            4T            6T            8T
   tmr           24.80         48.21         96.75         74.46         98.11       
   fsintr        24.88 (100%)  48.32 (100%)  96.73 (100%)  74.59 (100%)  98.46 (100%)
   fsasm         24.77 (100%)  48.18 (100%)  96.10 ( 99%)  74.55 (100%)  98.93 (101%)
   fscap         24.85 (100%)  48.14 (100%)  96.46 (100%)  74.55 (100%)  98.71 (101%)
   pl            24.80 (100%)  48.20 (100%)  96.61 (100%)  74.54 (100%)  98.61 (101%)
   spread (max-min)/median: typical 0.6%, worst 3.3% (pl @ 8T); group took 52 s
```

```text
== ntw | NT positional write, 4x unroll + SFENCE (SimpleNT)
   variant                 1T            2T            4T            6T            8T
   tmr_128       22.40         44.18         88.09         68.80         89.87       
   fs_128        22.30 (100%)  44.16 (100%)  88.10 (100%)  68.79 (100%)  90.44 (101%)
   pl_128        22.44 (100%)  44.10 (100%)  87.90 (100%)  68.70 (100%)  90.00 (100%)
   tmr_256       22.43         44.14         88.24         69.37         91.19       
   fs_256        22.45 (100%)  44.26 (100%)  87.86 (100%)  69.40 (100%)  89.87 ( 99%)
   pl_256        22.42 (100%)  44.20 (100%)  88.09 (100%)  69.40 (100%)  90.68 ( 99%)
   tmr_512       22.29         44.22         88.52         69.39         91.07       
   fs_512        22.44 (101%)  44.23 (100%)  88.56 (100%)  69.34 (100%)  90.51 ( 99%)
   fsk_512       22.45 (101%)  44.24 (100%)  88.15 (100%)  69.32 (100%)  90.60 ( 99%)
   pl_512        22.48 (101%)  44.25 (100%)  87.99 ( 99%)  69.32 (100%)  90.51 ( 99%)
   spread (max-min)/median: typical 0.8%, worst 9.5% (fs_128 @ 1T); group took 37 s
```

```text
== wflush | write line + CLFLUSHOPT it, same loop (mixed)
   variant                 1T            2T            4T            6T            8T
   tmr_256        6.72         13.08         24.70         24.08         29.86       
   fsintr_256     6.48 ( 96%)  12.39 ( 95%)  23.77 ( 96%)  23.28 ( 97%)  29.31 ( 98%)
   fsasm_256      6.75 (100%)  13.07 (100%)  24.72 (100%)  23.74 ( 99%)  30.07 (101%)
   fscap_256      6.75 (100%)  13.15 (100%)  24.76 (100%)  23.93 ( 99%)  29.88 (100%)
   pltok_256      6.72 (100%)  13.10 (100%)  24.59 (100%)  23.70 ( 98%)  29.83 (100%)
   plentry_256    6.71 (100%)  13.11 (100%)  24.82 (101%)  23.81 ( 99%)  30.05 (101%)
   tmr_512        6.85         13.34         25.72         23.28         29.59       
   fsintr_512     6.58 ( 96%)  12.73 ( 95%)  23.97 ( 93%)  23.41 (101%)  29.45 (100%)
   fsasm_512      6.85 (100%)  13.31 (100%)  25.26 ( 98%)  22.48 ( 97%)  29.14 ( 98%)
   fscap_512      6.86 (100%)  13.39 (100%)  25.48 ( 99%)  23.27 (100%)  29.55 (100%)
   pltok_512      6.83 (100%)  13.31 (100%)  25.23 ( 98%)  23.26 (100%)  29.25 ( 99%)
   plentry_512    6.86 (100%)  13.43 (101%)  25.58 ( 99%)  23.31 (100%)  29.29 ( 99%)
   spread (max-min)/median: typical 1.6%, worst 15.9% (pltok_512 @ 4T); group took 143 s
```

```text
== pfv | 4-accumulator verify + PREFETCHT0 per line
   variant                 1T            2T            4T            6T            8T
   tmr_256       18.49         32.17         58.46         48.62         64.24       
   fs_256        17.61 ( 95%)  31.54 ( 98%)  58.65 (100%)  47.92 ( 99%)  62.75 ( 98%)
   fsptr_256     18.61 (101%)  32.31 (100%)  59.24 (101%)  47.99 ( 99%)  63.27 ( 98%)
   tmr_512       17.27         31.41         59.24         45.02         58.70       
   fs_512        14.23 ( 82%)  27.60 ( 88%)  52.90 ( 89%)  44.81 (100%)  58.57 (100%)
   fsptr_512     16.69 ( 97%)  30.78 ( 98%)  57.04 ( 96%)  47.43 (105%)  62.35 (106%)
   spread (max-min)/median: typical 2.4%, worst 13.4% (tmr_256 @ 4T); group took 51 s
```

```text
== copy | copy half -> half: NT-512 vs MOVDIR64B (GiB/s copied)
   variant                 1T            2T            4T            6T            8T
   nt_tmr_512    11.79         20.18         38.68         30.73         40.42       
   nt_fs_512     11.76 (100%)  20.17 (100%)  38.18 ( 99%)  30.73 (100%)  40.20 ( 99%)
   md_tmr        13.10         20.37         38.41         30.30         39.27       
   md_fs         13.14 (100%)  20.39 (100%)  38.65 (101%)  30.32 (100%)  39.52 (101%)
   md_pl         13.15 (100%)  20.38 (100%)  38.00 ( 99%)  30.29 (100%)  39.54 (101%)
   spread (max-min)/median: typical 2.2%, worst 11.2% (md_pl @ 8T); group took 35 s
```

```text
== fillflush | fill then flush the range (flush_after scope); md_pl = MOVDIR64B fill
   variant                 1T            2T            4T            6T            8T
   tmr_256        7.75         14.78         26.48         22.84         28.65       
   pl_256         7.84 (101%)  14.89 (101%)  26.77 (101%)  22.64 ( 99%)  28.40 ( 99%)
   tmr_512        8.15         15.48         27.78         23.55         29.55       
   pl_512         8.23 (101%)  15.64 (101%)  28.06 (101%)  23.60 (100%)  29.25 ( 99%)
   md_pl         22.78         44.97         89.08         70.19         91.58       
   spread (max-min)/median: typical 1.7%, worst 6.2% (pl_256 @ 4T); group took 49 s
```

```text
== sb | TMR StuckBit port, flush before verify (3 phases)
   variant                 1T            2T            4T            6T            8T
   tmr_256       11.03         21.40         41.37         39.14         49.20       
   pl_256        11.13 (101%)  21.59 (101%)  41.71 (101%)  39.27 (100%)  49.61 (101%)
   tmr_512       10.63         20.68         39.72         38.35         47.70       
   pl_512        10.62 (100%)  20.66 (100%)  39.70 (100%)  38.35 (100%)  47.79 (100%)
   plplain_512   10.60 (100%)  20.61 (100%)  39.53 (100%)  38.08 ( 99%)  47.27 ( 99%)
   spread (max-min)/median: typical 0.5%, worst 3.1% (plplain_512 @ 8T); group took 219 s
```

```text
== refresh | TMR Refresh port, flush before verify (sleep omitted)
   variant                 1T            2T            4T            6T            8T
   tmr_256        7.72         15.02         28.21         27.22         33.82       
   pl_256         8.96 (116%)  17.37 (116%)  32.72 (116%)  30.79 (113%)  38.41 (114%)
   tmr_512        8.77         17.04         31.71         28.13         37.35       
   pl_512         8.91 (102%)  17.27 (101%)  32.12 (101%)  28.03 (100%)  36.55 ( 98%)
   spread (max-min)/median: typical 2.2%, worst 9.8% (tmr_512 @ 6T); group took 76 s
```

```text
== simplent | TMR SimpleNT port: 4 x (NT positional write, 5 verifies) per chunk
   variant                 1T            2T            4T            6T            8T
   tmr_256       31.24         61.05        119.32        104.53        132.83       
   pl_256        31.90 (102%)  62.26 (102%) 121.52 (102%) 106.33 (102%) 133.96 (101%)
   tmr_512       31.97         62.49        120.27         94.18        119.71       
   pl_512        33.38 (104%)  65.19 (104%) 125.31 (104%)  95.75 (102%) 122.39 (102%)
   spread (max-min)/median: typical 0.8%, worst 9.6% (tmr_512 @ 4T); group took 252 s
```

```text
== sbnf | TMR StuckBit port, no flush (3 phases)
   variant                 1T            2T            4T            6T            8T
   tmr_256       38.80         75.27        136.36        109.08        135.63       
   pl_256        40.02 (103%)  77.01 (102%) 140.06 (103%) 111.18 (102%) 135.50 (100%)
   tmr_512       44.12         84.61        151.15        111.09        135.82       
   pl_512        44.00 (100%)  84.71 (100%) 153.22 (101%) 110.55 (100%) 135.86 (100%)
   plplain_512   43.01 ( 97%)  83.61 ( 99%) 150.30 ( 99%) 111.16 (100%) 137.40 (101%)
   spread (max-min)/median: typical 2.0%, worst 6.8% (pl_512 @ 4T); group took 68 s
```

```text
== amxread | AMX read: 4 tile loads in flight vs zmm 4-acc verify; amx_verify = tile load + zmm check
   variant                 1T            2T            4T            6T            8T
   amx           15.36         29.93         54.97         42.51         52.78       
   zmm_tmr_512   15.29         29.79         54.64         44.69         54.49       
   amx_verify_512  13.19         25.67         49.00         42.92         52.09       
   spread (max-min)/median: typical 2.8%, worst 7.6% (amx_verify_512 @ 8T); group took 28 s
```

```text
== amxfill | AMX pattern fill (1 KiB tile stores) vs zmm fill
   variant                 1T            2T            4T            6T            8T
   amx           12.25         23.09         39.17         33.89         40.01       
   zmm_tmr_512   12.37         23.27         39.85         34.53         40.86       
   spread (max-min)/median: typical 2.0%, worst 3.5% (amx @ 8T); group took 15 s
```

```text
== amxcopy | copy half -> half: AMX tiles vs NT-512 vs MOVDIR64B (GiB/s copied)
   variant                 1T            2T            4T            6T            8T
   amx            7.66         14.63         24.82         20.92         24.91       
   nt_tmr_512    11.85         20.20         39.19         31.05         40.03       
   md_tmr        13.22         20.45         38.91         30.51         39.01       
   spread (max-min)/median: typical 1.4%, worst 4.4% (amx @ 4T); group took 23 s
```

```text
== amxstride | strided read, stride 4096 (one line in each of 16 pages): 1 tile load vs 16 zmm loads
   variant                 1T            2T            4T            6T            8T
   amx           16.60         31.34         57.54         45.18         57.14       
   zmm_512       16.71         31.60         57.70         46.03         57.04       
   spread (max-min)/median: typical 1.9%, worst 4.2% (amx @ 4T); group took 17 s
```

## AMX: stdarch intrinsics vs `asm!` (2026-10-05)

plumb_tiles' `nightly` feature (stdarch's AMX intrinsics) against its stable `asm!` path, on
rustc 1.101 nightly (2026-10-04, LLVM 23.1.3). Both bench binaries were built from the same
source, one with the bench's `plumb_tiles` dependency set back to no features; four DRAM runs
of `--only amxread,amxfill,amxcopy,amxstride`, alternating builds. Only four 1 GiB pages were
free that day (the other threads got 2 MiB pages), so these numbers aren't comparable with the
tables above; the two builds ran under the same conditions.

Codegen (asm gate, instructions per innermost loop): the intrinsic loops are unrolled
(`k_amx_read` 66 instructions with 32 tile loads, `asm!` 10 with 4; `k_amx_strided_read` 130
with 64 loads, `asm!` 9 with 1), and `k_amx_verify_512` is 21 per tile against 29.

DRAM: no measurable difference. The AMX variants of the intrinsic build are at -2.3% to +0.7%
of the `asm!` build at every thread count; the variants whose code is identical in both builds
(`zmm_tmr_512`, `nt_tmr_512`, `md_tmr`) moved by up to -4.6%, and one build's two runs differed
by up to 4.8%.

Cache-resident (probe crate, one thread, one session per sample, ns per 1 KiB tile, median of
31, three alternating runs agreed within 2%):

| kernel | L1 (16 KiB) `asm!` | L1 intrinsics | L2 (512 KiB) `asm!` | L2 intrinsics |
|---|---|---|---|---|
| runtime-stride load, closure captures by reference | 14.8 | 13.7 | 5.46 | 5.99 |
| runtime-stride load, `move` closure | 14.8 | 14.8 | 5.46 | 5.46 |
| 4 loads in flight (`k_amx_read`'s shape) | 14.7 | 14.5 | 5.49 | 5.56 |
| copy, 2 tiles in flight | 6.0 | 5.9 | 10.8 | 10.8 |

The unrolling buys nothing: AMX throughput, not loop overhead, is the limit. The one real
difference is the by-reference capture on the intrinsic path, about 10% from L2: the session
function is called, not inlined, and LLVM declares the AMX intrinsics as touching any memory,
so captured values are reloaded after every tile op (19 instructions per tile vs 10). Either
calling practice removes it (crate docs, Toolchains). A `move` closure gives 10 instructions,
5.54 ns from L2. So does the same by-reference closure in a function compiled with `amx-tile`,
5.56 ns, against 5.54 for `asm!`. By-reference stores showed no time cost: from L2 a tile store
takes about 17 ns, which hides the extra instructions. An LLVM change that declares the
intrinsics' real memory effects (the tile state as a target memory location, as AArch64 does
for SME) takes the plain by-reference loop to 16; it was sent upstream as
llvm/llvm-project#229025, though plumb doesn't depend on it, since both practices already match
`asm!`. Loads from an L1-resident buffer being
~2.7x slower per tile than from L2 holds on both paths; not investigated.
