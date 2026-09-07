# 2K / 4K / 8K still decoding, 2026-09-07

At the historical baseline below, the default checked decoder takes about **1.9× upstream for serial stills**
and **2.1–2.5× for the eight-tile, eight-worker stills in this development set**.
Disabling slice checks does little for serial decoding. The next substantial
opportunities are coefficient/entropy decoding, scalar inverse-transform
fallbacks, and the registration/copy work around tiled loop filtering.
Spin backoff is a low-priority lead for these particular inputs.

This change establishes the still benchmark and profiling protocol and fixes
misleading threading documentation. The measured decoder implementation is
the previously shipped `b50ac0d6e0a239167f5ac13231fc82ffdea259ca`; no decoder
algorithm or borrow-exclusion rule was changed in this investigation.
The [proposed parity goal](../../docs/PERFORMANCE_PARITY_GOAL.md) defines the
larger campaign and its completion criteria. The user subsequently activated
it with 10% per-group and 25% per-cell limits; these remain historical baseline
results, not the current implementation's performance.

## Inputs and comparison

The size names mean **1920×1080**, **3840×2160**, and **7680×4320**, respectively.
Two native large imazen-26 sources were center cropped and downsampled,
never upscaled. The earlier description of them as **training** images was
incorrect: the pinned canonical registry places both in its **test** split.
They have already been used for optimization and therefore cannot count as
untouched holdouts for this campaign. Their hashes match that registry; see
the [source audit](../still-expanded-2026-09-07/README.md).

- Photo: image 1407, rocky coastline, 8160×6120, Lilith, PD-own.
- Map: image 5017, Great Smoky Mountains trail map, 9146×5272, NPS,
  PD-USGov in the corpus manifest.

[sources.json](sources.json) pins the public downloads, provenance, and actual
SHA-256 values. The first candidate, a zune 8K smooth-render fixture, was
rejected before encoding or timing. No Kodak or gradient fixture was used.

The 12 bitstreams are libaom **aomenc 3.13.1**, all-intra, CPU-used 6, CQ 24,
8-bit 4:2:0. For each content/size pair there is minimum legal tiling and
eight tiles (4 columns × 2 rows). Minimum tiling is one tile at 2K/4K and
2×2 at 8K: the AV1 tile-width/area limits prevent a single 7680×4320 tile.
The diagnostic decoder's parsed geometry was checked in all nine census
cases, including the 8K minimum layout. Encoder logs and the complete corpus
manifest are preserved in `results/`.

The comparator is **memorysafety/rav1d**, commit
`d3d1cd67059f47803919be8276650e5870c9fd02`, with assembly enabled. It is not
the unrelated historical crates.io `rav1d` 0.1.0. The four primary arms are
default checked rav1d-safe, rav1d-safe `unchecked`, rav1d-safe `asm`, and that
upstream. `unchecked` does **not** disable the overlap tracker; the diagnostic
untracked configurations remain rejected by the build.

All arms use standalone in-memory consumers, Rust 1.98.1 / LLVM 22.1.8,
fat LTO, one codegen unit, panic unwind, normal CPU dispatch, and
`RUSTFLAGS='-C llvm-args=-align-all-functions=4'`. No `target-cpu=native` or
root dev-dependency feature unification. This machine is a Ryzen 9 7900X
(12 cores / 24 logical CPUs), Linux x86-64. Complete host and binary metadata
are in the compressed provenance files. The baseline collector initially
captured an empty dav1d version because `dav1d -v` writes stderr; the observed
version was **1.5.3**. Later runs capture both streams correctly.

## Results

Elapsed milliseconds per frame, medians of five rotated rounds, **eight tiles
and eight workers in one persistent decoder**:

| Content | Size | Checked | Upstream ASM | Ratio |
|---|---|---:|---:|---:|
| Photo | 2K | 8.661 | 3.456 | 2.51× |
| Photo | 4K | 26.663 | 12.970 | 2.06× |
| Photo | 8K | 65.968 | 31.119 | 2.12× |
| Map | 2K | 12.344 | 5.233 | 2.36× |
| Map | 4K | 33.743 | 14.472 | 2.33× |
| Map | 8K | 97.314 | 39.216 | 2.48× |

At 4K, the diagnostic arms help separate costs:

| Photo configuration | Checked | Unchecked | Own ASM | Upstream ASM |
|---|---:|---:|---:|---:|
| One tile, one worker | 117.732 | 117.691 | 71.413 | 61.440 |
| Eight tiles, eight workers | 26.663 | 25.189 | 16.502 | 12.970 |

The first row refutes bounds-check removal as the main serial-still solution.
Own ASM also remains slower than upstream, so replacing DSP kernels alone
does not remove all fork overhead.

The primary matrix covers 1, 4, 8, and 24 workers on every input; supplementary
photo eight-tile runs at all three sizes add 2 and 16 workers. More workers
are not uniformly faster. For the 4K eight-tile photo, checked times are
116.991 / 72.924 / 39.385 / 26.663 / 27.241 / 27.896 ms at
1 / 2 / 4 / 8 / 16 / 24 workers. The 8K photo benefits somewhat from 16
workers (61.370 ms versus 65.968 at eight). Minimum-tile 2K/4K stills show
little benefit from more workers.

Four simultaneous independent decoders were also measured, each with 1, 4,
or 8 workers. For the 4K eight-tile photo, aggregate checked throughput is
33.3 / 76.9 / 87.2 frames/s; upstream is 62.2 / 186.3 / 217.5 frames/s.
The aggregate ms/frame in `summary.tsv` is **not individual-request latency**:
four decoders contribute frames to that denominator. Per-instance elapsed
times are retained in every raw record.

Opening and dropping a 24-worker decoder before the serial 4K decoder caused
no material slowdown in these paired checks: photo 118.151 → 118.595 ms,
map 111.863 → 112.017 ms. This is evidence for these two process-history
cases, not a proof covering every interleaving or decoder lifecycle.

## Profiles and borrowing

`perf record` sampled `cycles:u` at 499 Hz for about two seconds per case,
enabled by FIFO only during the harness timer. No lost samples were reported.
Percentages below are named **self CPU samples**, not fractions of elapsed
latency. Cold `lock_slow` is explicitly non-inlined in the tracker.

| Checked case | Coefficients + MSAC | Transforms | Tracker | Loopfilter | Explicit spin/slow lock |
|---|---:|---:|---:|---:|---:|
| Photo 2K, one tile, t1 | 45.39% | 16.32% | 6.94% | 10.06% | 0.00% |
| Photo 4K, one tile, t1 | 46.59% | 17.28% | 6.05% | 8.07% | 0.00% |
| Photo 4K, eight tiles, t8 | 36.06% | 13.70% | 12.90% | 13.66% | 0.00% |
| Photo 8K, eight tiles, t8 | 37.08% | 12.35% | 13.32% | 16.35% | 0.00% |
| Map 4K, eight tiles, t8 | 26.26% | 5.94% | 19.58% | 15.37% | 0.12% |

`decode_coefs` includes coefficient/context work and inlined entropy code;
the profile does not assign its whole cost to a single MSAC function.
`inv_txfm_add`, scalar ADST-8/16, and scalar DCT-8 appear in the photo profile.
Use a transform-shape census before deciding which SIMD variant to implement.
Upstream reports contain 11.6–13.1% anonymous NASM/local labels, so their tiny
*named* transform buckets cannot be read as complete transform attribution.
Buckets are name-based lower bounds, and spin can overlap the tracker bucket.

A separate `probe-usage,probe-tasktime` build recorded borrow extents, sites,
shard occupancy, tile geometry, and task-stage/worker activity. Its timing is
instrumentation-distorted and excluded from the speed results. It found **zero
seven-slot overflow attempts** in all nine census runs. Normal profiles show
little explicit spinning, while ordinary acquisition/drop work remains.

Mean borrow counts per lifetime-decoded frame:

| Census case | Total registrations | Compact loopfilter reads | Context setters |
|---|---:|---:|---:|
| Photo 4K, one tile, t1 | 1,176,630 | — | 574,591 |
| Photo 4K, eight tiles, t8 | 2,699,362 | 1,345,925 | 562,216 |
| Photo 8K, eight tiles, t8 | 6,333,389 | 3,439,376 | 1,087,560 |
| Map 4K, eight tiles, t8 | 4,026,844 | 1,869,837 | 970,692 |
| Map 8K, eight tiles, t8 | 12,067,186 | 6,064,336 | 2,635,329 |

The census denominator is five frames: one serial reference plus four frames
on the requested worker configuration (warmup, two timed, final validation).
Thus the t8 values mix one serial frame with four parallel frames; they are
not a claim that each steady-state t8 frame has exactly that count. Context
setters collapse multiple callers at `src/ctx.rs:110`. The main read site is
`src/safe_simd/loopfilter.rs:5196`, which copies exact rows into compact scratch;
the paired write-back diffs against a pristine copy and registers only changed
spans. Both copies and exact guards are currently load-bearing for correctness.

In this all-intra seed, task probes showed no CDEF/restoration/super-resolution
stage activity. Tile entropy and reconstruction run together under the
`tile_recon` stage, so `tile_entropy = 0` does not mean entropy decoding is free.
Stage hooks also do not cover the serial path. These facts limit this corpus;
the parity goal explicitly requires additional filter/bit-depth/encoder strata.

## Next experiments, in order

1. **Coefficient/entropy decode:** inspect the inlined adapt4 selection and CDF
   update in `src/msac.rs`, then test safe SIMD and control-flow improvements
   with differential symbol/state tests and independent full-frame hashes.
   This is a larger serial-photo target than slice-check removal.
2. **Transforms:** count coded shapes and SIMD fallback decisions for these
   exact stills, then target the costly ADST/mixed transforms actually used.
3. **Loopfilter registrations and copies:** measure exact rectangle eligibility
   and guard lifetimes at the compact-read site. Test coalescing exact row
   records or a kernel interface with explicit read/write footprints. Keep
   read-only taps separate from modified spans. Never substitute a hull
   reference or whole-row copy across unregistered gaps.
4. **Scheduling and batches:** minimize unnecessary work on minimum-tile stills
   and benchmark worker budgets for several simultaneous decoders. Keep
   mixed-size, primed, and repeated-lifecycle adversarial cases in the gate.

The [ownership ledger](../../docs/OWNERSHIP_MODELS.md) records why earlier
tile-keyed locks, full-plane copies, hull references, and filter-band copies
failed. This census justifies re-examining exact loopfilter registrations; it
does not revive those rejected designs or prove that a wider lifetime is safe.

## Integrity, scope, and reproduction

**1,200 measured runs**, plus 72 upstream calibration runs, completed with
ordered visible-frame hashes checked before and after timing. Every input also
matched independent **dav1d 1.5.3** output. Every timed frame was counted.
No failed or shortened decode entered an aggregate. Calibration selects at
least 150 ms of upstream work per round (minimum two passes); these five-round
development results are not the longer confidence-gated parity acceptance run.

The timer covers repeated decode/drain/reset on persistent contexts, with
input already in memory. It excludes pool creation, file I/O, packet parsing,
hashing, and process startup. Lifecycle latency, p95 latency, peak-memory
acceptance, HDR/10/12-bit, other chroma layouts, film grain, and AVIF container
or color conversion remain future campaign work. There are only two sources;
there is no untouched holdout evaluation yet. No parity or release-proof claim
is made from this dataset.

Public signatures, crate features, and runtime behavior are unchanged. The
API documentation now correctly says that checked builds retain tile workers
while capping frames in flight at one; the README no longer promises a fixed
two-thread photo speedup. Release doctests passed: 9 passed, 13 ignored,
0 failed. The log is recorded in `results/`.

Run heavy commands one at a time through
`/home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 --`, with
`TMPDIR=/home/lilith/tmp`. Corpus creation uses system Python/Pillow:

```sh
/usr/bin/python3 benchmarks/stills-2026-09-07/prepare.py --work-dir "$STILL_WORK"
python3 benchmarks/stills-2026-09-07/compare.py --work-dir "$STILL_WORK" \
  --name baseline --upstream "$UPSTREAM_BIN" \
  --arms "checked=$CHECKED_BIN" "unchecked=$UNCHECKED_BIN" \
         "asm=$ASM_BIN" "upstream=$UPSTREAM_BIN"
python3 benchmarks/stills-2026-09-07/probe.py --work-dir "$STILL_WORK" --build
python3 benchmarks/stills-2026-09-07/analyze.py "$STILL_WORK"
python3 benchmarks/stills-2026-09-07/collect.py "$STILL_WORK"
```

Exact supplemental matrix arguments, profile commands, binary hashes, input
hashes, medians, per-round samples, stdout/stderr, and frame counts are in
`results/`. [`build.py`](../tracker-sharding-2026-09-07/build.py) builds the
standalone checked/unchecked/ASM consumers from a pinned checkout. The
[upstream consumer](../upstream-2026-09-07/driver/src/main.rs) and its lockfile
are also tracked. Set its path dependency to the pinned upstream checkout.

Large immutable assets and raw perf recordings are described in
[artifacts.pointer.md](artifacts.pointer.md). Small text evidence is compressed
and split by input so each tracked file is under 30 KB.
