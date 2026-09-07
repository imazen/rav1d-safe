# Still transform specialization — 2026-09-07

Retain the 8×8, 8-bit fallback specialization: it reduces decode time by
2.2–3.4% on the confirmed 2K/4K photo cells, with a smaller serial 8K gain.
The gain persists with a second matched function alignment. The arithmetic
and borrow footprint are unchanged; generated transform code grows about
53 KiB. Threaded 8K gains remain unresolved, and upstream parity is unmet.

Draft [PR #528](https://github.com/imazen/rav1d-safe/pull/528), following
`docs/PERFORMANCE_PARITY_GOAL.md`. The entropy experiments retain the earlier
leaf CDF kernel; later candidates regressed or failed to show a convincing
gain. This campaign measures the next existing profile lead: inverse
transforms account for about 17% of serial 4K photo self CPU samples.

## Census protocol

The `itx-census` standalone driver derives from the matched current driver.
It adds `__ablate` and reports the existing transform-shape counters after
all clients finish. All SIMD families remain enabled. Each cell starts a
fresh process, so there are no counter resets or process-global mode changes.
The internal RESULT times are diagnostic only and are never speed evidence.

`census.py` validates every input checksum and visible image against independent
dav1d 1.5.3. All 12 frozen development streams pass at one and eight workers:
24 cells, four visible decodes each, 96 validated decodes. Counts are exact
multiples of four, and each stream has identical counts at both worker levels.
This is the same two-source development seed, not the full acceptance corpus.

The counter's `SCALAR` label means that the outer SIMD dispatcher declined
the transform. A fallback can still use SIMD internally; conversely, the
`simd` label records acceptance and does not establish instruction counts.
Coefficient area
is the sum of transform width × height × calls; it is neither elapsed time
nor a count of unique output pixels.

| Eight-tile input | Outer-fallback coefficient area | 8×8 fallback area | 16×16 fallback area |
|---|---:|---:|---:|
| Photo 2K | 39.10% | 29.60% | 7.82% |
| Photo 4K | 34.87% | 21.75% | 10.75% |
| Photo 8K | 15.80% | 4.68% | 10.03% |
| Map 2K | 25.20% | 22.53% | 0.65% |
| Map 4K | 24.36% | 19.46% | 1.56% |
| Map 8K | 30.59% | 18.36% | 3.47% |

Minimum-tile results show the same dominant shapes; all exact counts are
archived. The first implementation lead is the 8×8 generic fallback, which
currently passes dimensions and two function pointers to a shared function
marked `inline(never)`. Measure whether specializing that path removes
indirect transform calls and dynamic indexing before writing new SIMD.

## Reproduction

```sh
ROOT=/home/lilith/tmp/rav1d-still-transforms-2026-09-07
env TMPDIR=/home/lilith/tmp /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 benchmarks/tracker-sharding-2026-09-07/build.py \
  --repo /home/lilith/work/zen/rav1d-safe --work-dir "$ROOT" \
  --label census --driver itx-census --modes checked
env TMPDIR=/home/lilith/tmp /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 benchmarks/still-transforms-2026-09-07/census.py \
  --work-dir "$ROOT" --binary "$ROOT/bin/census-checked"
```

The census uses the default checked implementation at `dba3aef2`, with the
same decoder source as validated `8b13ed69`. `results/index.json` lists index
pages with uncompressed SHA-256 checksums. Compressed raw files contain
commands, stdout/stderr, visible hashes, dependency and binary hashes, and
the complete per-shape counts. Compiler, machine and stream provenance
inherit `benchmarks/stills-2026-09-07/README.md`. Large assets remain at their
recorded scratch paths, with no verified off-host backup.

## 8×8 fallback specialization screen

`inline8-specialized.patch.gz`, against `dba3aef2`, shares the existing
transform arithmetic between the ordinary out-of-line wrapper and an
inlined 8×8, 8-bit path. The arithmetic body is unchanged byte for byte.
Constant dimensions and transform types let the compiler remove the
indirect row/column calls for this path. Other shapes retain the shared
wrapper. The symbol inventory grows by about 53 KiB across the affected
specializations; this is a code-size tradeoff, not a new allocation.

All 76 checked unit/fixture tests pass. Five rotated paired rounds validate
all 160 measured runs against upstream rav1d and independent dav1d. Times
below are paired changes from the retained CDF-kernel baseline:

| Input | Workers | Formatting control | 8×8 specialization |
|---|---:|---:|---:|
| Photo 2K, minimum tiles | 1 | -0.65% | -3.05% |
| Photo 2K, minimum tiles | 8 | +0.41% | -2.94% |
| Photo 4K, minimum tiles | 1 | +0.21% | -2.77% |
| Photo 4K, minimum tiles | 8 | -0.62% | -2.72% |
| Photo 4K, eight tiles | 1 | +0.02% | -1.73% |
| Photo 4K, eight tiles | 8 | +1.06% | -4.53% |
| Map 4K, eight tiles | 1 | +0.50% | -0.79% |
| Map 4K, eight tiles | 8 | -0.61% | -0.09% |

The formatting control comes from an edit-script failure: an assertion
failed before writing the intended change, but the shell continued to
rustfmt and build. That first binary, labelled `inline8`, contains only
import ordering and macro formatting changes. A follow-up byte comparison
detected those differences; normalized source and transform code sizes match
the baseline. It is explicitly a control, never presented as an optimization.
The intended candidate was then built under the new immutable label
`inline8-specialized`. `inline8-edit-recovery.json.gz` records the failure.

This screen supports longer confirmation, not a parity claim. The 2K/4K
photo gains need nine-round 2K/4K/8K confirmation, code-placement checks,
and broader correctness/regression gates before promotion.

## Longer confirmation

Nine rotated paired rounds with at least 500 ms calibrated upstream work.
All 324 measured runs match both references. Upper bounds are exact one-sided
95% empirical-bootstrap bounds on paired median ratios against the retained
CDF baseline. No samples are discarded; paired ratios and ratios of medians
can differ. The run reported peak host load 3.71 and minimum available memory
27,342 MiB.

| Input | Workers | Baseline ms | Candidate ms | Upstream ms | Paired change | Upper ratio |
|---|---:|---:|---:|---:|---:|---:|
| photo-2k-min | 1 | 32.064 | 31.010 | 17.319 | -3.35% | 0.9705 |
| photo-2k-min | 8 | 33.998 | 33.299 | 19.879 | -2.26% | 0.9816 |
| photo-4k-min | 1 | 111.371 | 108.976 | 59.861 | -2.19% | 0.9799 |
| photo-4k-min | 8 | 113.253 | 109.918 | 65.576 | -2.61% | 0.9759 |
| photo-4k-t8 | 1 | 111.038 | 108.354 | 59.739 | -2.42% | 0.9767 |
| photo-4k-t8 | 8 | 26.765 | 25.881 | 13.246 | -2.45% | 0.9797 |
| photo-8k-t8 | 1 | 262.730 | 260.383 | 146.392 | -0.92% | 0.9917 |
| photo-8k-t8 | 8 | 61.440 | 60.989 | 31.454 | -1.22% | 1.0023 |
| map-4k-t8 | 1 | 107.740 | 106.886 | 57.354 | -0.86% | 0.9951 |
| map-4k-t8 | 8 | 32.924 | 32.677 | 14.221 | -0.91% | 1.0011 |
| map-8k-t8 | 1 | 319.039 | 315.679 | 164.367 | -1.18% | 0.9896 |
| map-8k-t8 | 8 | 90.672 | 90.728 | 38.743 | -0.88% | 1.0045 |

The 2K/4K photo gains survive the confidence rule, as does the smaller serial
8K photo gain. Eight-worker 8K photo and threaded map improvements remain
unresolved because their upper bounds exceed 1.0. Candidate/upstream paired
median ratios still range from about 1.67 to 2.34, outside the parity goal.
These are development-seed results; holdout, lifecycle, memory and video
gates remain outstanding. The implementation changes no public source
declaration, Cargo dependency, MSAC arithmetic, borrowing policy or constructor.
The declaration audit is not a cargo-semver-checks run.

## Alignment and correctness checks

A second matched build changes LLVM `align-all-functions` from 4 to 5 for
both baseline and candidate. All 1,096 named Rust decoder functions in each
binary are verified to start on 32-byte boundaries. `alignment.py` records
source hashes and restores the tested candidate in `finally`. The upstream
binary only calibrates work and checks images in this auxiliary run; no
alternate-alignment upstream speed ratio is reported.

Nine paired rounds, 500 ms calibrated work, 108 validated measured runs:

| Input | Workers | Paired change with alignment 5 | Upper ratio |
|---|---:|---:|---:|
| photo-2k-min | 1 | -3.54% | 0.9687 |
| photo-2k-min | 8 | -2.29% | 0.9777 |
| photo-4k-t8 | 1 | -2.45% | 0.9758 |
| photo-4k-t8 | 8 | -3.40% | 0.9850 |
| photo-8k-t8 | 1 | -0.96% | 0.9932 |
| photo-8k-t8 | 8 | -0.99% | 1.0066 |

The 2K/4K gains and smaller serial 8K gain survive the alignment change.
Threaded 8K improvement remains unresolved. This supports retaining the
8×8 specialization as an incremental improvement in the draft; it does not
establish the full performance goal or an improvement for every workload.

Checked conformance passes 766 vectors at one worker and again at eight
workers, with two existing infrastructure exclusions each. The selected
CPU-permutation smoke/CDF tests pass. All five committed-vector/concurrent
fixtures pass in debug with overflow checks. The 76-test checked release
unit/fixture gate, formatting, and strict release and debug Clippy also pass.
Cross-platform CI and
the remaining performance acceptance gates still apply.
