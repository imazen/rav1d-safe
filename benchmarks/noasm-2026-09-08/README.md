# Checked rav1d-safe versus upstream without assembly

On this 12-input still-image corpus, checked rav1d-safe takes less time than
upstream rav1d built without assembly. **Upstream has no memory-safe build
mode:** disabling its assembly leaves unsafe Rust. rav1d-safe keeps its default
borrow/bounds checks, `forbid(unsafe_code)`, and safe SIMD enabled. This is not
an ablation of SIMD and does not establish parity with upstream's faster,
assembly-enabled default.

| Workers | Geometric mean checked / upstream time | Less decode time |
| ---: | ---: | ---: |
| 1 | 0.680× | 32.0% |
| 4 | 0.821× | 17.9% |
| 8 | 0.813× | 18.7% |

Ratios are medians of five paired repetitions per input/worker cell, then
geometrically averaged across inputs. They describe these fixtures, not all AV1.

## Builds and protocol

Measured 2026-09-08 UTC on AMD Ryzen 9 7900X (24 logical CPUs). Both builds use
release, fat LTO, one codegen unit, unwind panics, line-table debug information,
and `RUSTFLAGS="-C llvm-args=-align-all-functions=4"`. Rust 1.98.1 (LLVM 22.1.8). Exact compiler and host
information, feature graphs, lockfiles, binary hashes, commands and raw output
are in the external evidence bundle.

- Upstream: `memorysafety/rav1d` commit
  `d3d1cd67059f47803919be8276650e5870c9fd02`, package 1.1.0;
  `default-features=false`, `bitdepth_8,bitdepth_16` only. Source archive identity
  was checked against all 1,364 Git blobs in the preceding upstream comparison.
- rav1d-safe: `be5b244a35fbbaf4f9402ee69bd925134c0c02a0`, staged 0.6.0,
  checked features, no `unchecked`, `c-ffi`, or `asm`. This includes the pinned
  archmage main dependency. The published 0.5.7 documentation examples are
  separate consumers and are **not** the benchmarked binary.
- Photo and map, each at 1920×1080, 3840×2160, and 7680×4320, encoded as
  8-bit 4:2:0 single-frame IVF. Each has minimum legal tiles and eight tiles
  (`t8`; exact row/column layouts in corpus metadata). 8K minimum is 2×2 tiles,
  not a single tile. This is the two-source seed corpus, not the expanded holdout.
- One decoder, one frame in flight, 1/4/8 workers, no priming. Strict decoding,
  all loop filters and film grain enabled. The fork's extra strict checks remain.
- Five paired repetitions with fixed A/B order, target 150 ms per batch, calibrated using
  upstream without assembly. 360 measured runs plus 36 calibration pilots;
  no samples discarded. Every measured run matched visible-pixel MD5 references
  from upstream and independent dav1d 1.5.3, before and after timing. Timed frame
  counts were checked. The timed operation reuses a warmed decoder and includes flush; creation,
  warmup and teardown are excluded. Fixed arm order can introduce order bias;
  encoded input is already loaded, and AVIF parsing/RGB conversion are excluded.
- Builds and measurement ran serially under `run-heavy --mem 16G --jobs 8`.
  The measurement took 292 seconds. No change to decoder runtime code was made
  for this experiment.

## Per-input results

Median milliseconds per decoded frame; ratio uses paired samples and therefore
can differ slightly from dividing the two displayed medians.

| Input | Workers | Upstream no ASM ms | Checked ms | Checked / upstream |
| --- | ---: | ---: | ---: | ---: |
| photo-2k-min | 1 | 45.041 | 28.501 | 0.631× |
| photo-2k-min | 4 | 42.762 | 29.896 | 0.698× |
| photo-2k-min | 8 | 43.279 | 30.948 | 0.705× |
| photo-2k-t8 | 1 | 45.435 | 28.400 | 0.625× |
| photo-2k-t8 | 4 | 12.756 | 9.728 | 0.778× |
| photo-2k-t8 | 8 | 9.416 | 7.190 | 0.767× |
| photo-4k-min | 1 | 162.177 | 99.427 | 0.613× |
| photo-4k-min | 4 | 147.026 | 100.036 | 0.685× |
| photo-4k-min | 8 | 147.325 | 100.773 | 0.681× |
| photo-4k-t8 | 1 | 160.609 | 98.483 | 0.614× |
| photo-4k-t8 | 4 | 43.443 | 33.466 | 0.776× |
| photo-4k-t8 | 8 | 29.329 | 23.038 | 0.779× |
| photo-8k-min | 1 | 451.903 | 231.372 | 0.512× |
| photo-8k-min | 4 | 138.710 | 95.260 | 0.687× |
| photo-8k-min | 8 | 138.660 | 89.721 | 0.644× |
| photo-8k-t8 | 1 | 450.191 | 230.949 | 0.513× |
| photo-8k-t8 | 4 | 120.031 | 79.816 | 0.662× |
| photo-8k-t8 | 8 | 82.053 | 55.467 | 0.697× |
| map-2k-min | 1 | 43.601 | 35.356 | 0.814× |
| map-2k-min | 4 | 39.858 | 36.690 | 0.919× |
| map-2k-min | 8 | 40.253 | 36.767 | 0.913× |
| map-2k-t8 | 1 | 43.719 | 35.217 | 0.805× |
| map-2k-t8 | 4 | 12.525 | 11.842 | 0.945× |
| map-2k-t8 | 8 | 11.980 | 11.350 | 0.946× |
| map-4k-min | 1 | 129.269 | 104.905 | 0.816× |
| map-4k-min | 4 | 110.664 | 105.604 | 0.950× |
| map-4k-min | 8 | 113.559 | 104.974 | 0.924× |
| map-4k-t8 | 1 | 129.767 | 105.860 | 0.815× |
| map-4k-t8 | 4 | 36.628 | 36.713 | 1.002× |
| map-4k-t8 | 8 | 32.550 | 32.222 | 0.985× |
| map-8k-min | 1 | 401.723 | 303.705 | 0.757× |
| map-8k-min | 4 | 143.611 | 127.048 | 0.895× |
| map-8k-min | 8 | 138.382 | 127.328 | 0.921× |
| map-8k-t8 | 1 | 404.906 | 304.973 | 0.755× |
| map-8k-t8 | 4 | 111.131 | 107.582 | 0.971× |
| map-8k-t8 | 8 | 97.013 | 87.939 | 0.897× |

## Reproduction and evidence

[The manifest](EVIDENCE.json) pins the public R2 archive and its per-file hash
index. It includes all twelve encoded IVF inputs, full samples and validation
output, build manifests/lockfiles, and the documentation example run logs.
No credentials are needed:

```sh
python3 tools/fetch-benchmark-artifacts.py \
  --manifest benchmarks/noasm-2026-09-08/EVIDENCE.json
python3 benchmarks/noasm-2026-09-08/reproduce.py \
  --upstream /path/to/pinned/upstream --work-dir /path/to/new/scratch
```

Use a clean upstream checkout at the revision above and the same compiler.
The reproduction script builds the current rav1d-safe checkout; use the recorded
revision for the measured runtime, retaining this directory and the shared
benchmark scripts. Run under your machine's resource limiter. The script refuses
to reuse an existing work directory and runs builds and measurements serially.
The exact original local build orchestration is retained in the evidence too.

See [the Rust AV1/AVIF guide](../../docs/RUST_CODEC_WORKFLOW.md) for the tested
published APIs, and [the assembly-enabled upstream comparison](../upstream-2026-09-07/README.md)
for that separate baseline. No performance improvement is claimed from this
documentation change.
