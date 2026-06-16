# Issue #17 — compact_read_per_row pooling: allocation + wall-clock A/B

Fix commit: `095ee5f` (pool the per-edge/per-block compact buffer in a thread-local
scratch instead of `vec![0u8; …]` per call). This file records both the allocation
delta and the glibc wall-clock A/B.

- Date: 2026-06-16T06:34Z
- Host: lilith (AMD Ryzen 9 7950X, WSL2, glibc 2.35) — **shared box, other agents active during the run**
- main@origin at measurement: `ef6b610` (= `095ee5f` #17 + #18 + #19)
- rustc 1.96.0, `--release`, **no `-C target-cpu=native`** (runtime dispatch — what users get)
- Build: `--no-default-features --features bitdepth_8,bitdepth_16` (default safe-SIMD, `forbid(unsafe_code)`)
- Vector: `test-vectors/bench/photo_4k.avif` (3840×2561, gitignored local asset)
- Decode config: `threads=4, max_frame_delay=1` → `n_tc=4`, `n_fc=1` (pure tile threading,
  which is the only path that reaches `compact_read_per_row`)

## Allocations (heaptrack, one decode)

| | allocations | temporary |
|---|---|---|
| before (`vec!` per call) | **517,414** | 345,110 |
| after (pooled) | **204** | 10 |

`compact_read_per_row` confirmed as the eliminated site (99.96% of its allocations
were temporary allocate-then-free). Residual 204 ≈ baseline + one compact buffer per
tile worker thread. (8K profile in the issue: 99.98% of ~3M total from that one site.)

## Wall-clock (interleaved A/B)

Two binaries identical except the one allocation line; alternated `before, after`
each round (8 rounds × 80 timed decodes + 15 warmup), `min`/`median` per round.
`min` is the least-contended estimate; on this busy box `mean` was noisy (one
`before` round spiked to 314 ms mean) so it is not used for the conclusion.

```
round  before_min  after_min  d_min   before_med after_med  d_med
1        255.13     250.89   -4.24    263.60    257.38   -6.22
2        255.34     248.49   -6.85    261.85    256.07   -5.78
3        255.25     250.58   -4.67    264.26    255.09   -9.17
4        254.14     254.12   -0.02    264.38    261.28   -3.10
5        257.36     253.79   -3.57    264.24    260.00   -4.24
6        258.26     253.37   -4.89    267.28    258.54   -8.74
7        254.44     250.65   -3.79    262.51    258.87   -3.64
8        256.12     251.34   -4.78    262.26    258.93   -3.33
```

- **after (pooled) faster in 8/8 rounds** (sign test p = (1/2)^8 ≈ 0.004)
- min:    before 255.75 ms → after 251.65 ms = **−4.10 ms (−1.60%)**
- median: before 263.80 ms → after 258.27 ms = **−5.53 ms (−2.10%)**

## Caveats

- **glibc only.** This ~1.6–2.1% is the Linux number. The issue's motivation is the
  Windows system allocator, whose per-call cost is far higher — the expected Windows
  win is substantially larger but was **not measured here** (no Windows box).
- **Largely single-tile frame**: `threads=4` wall-clock ≈ single-threaded (~255 ms vs
  the ~225 ms baseline in older notes, inflated by box contention), so the 4 tile
  threads don't parallelize this particular frame much. The compact path still fires
  per edge/block; on a multi-tile frame the absolute time drops and the saved
  allocations spread across more active workers.
- Busy shared box — treat absolute ms as soft; the 8/8 sign test + consistent
  min/median deltas are the robust signal.

Reproduce: `examples/heaptrack_compact17.rs` for the allocation count
(`heaptrack ./…/heaptrack_compact17 <avif> 4 1`). The timing harness was a throwaway
(`examples/zz_timing17.rs`, not committed): warm 15, time N fresh `threads=4` decodes,
print min/median/mean; alternate a pooled vs reverted-`vec!` build per round.
