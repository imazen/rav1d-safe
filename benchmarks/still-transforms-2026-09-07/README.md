# Still transform specialization — 2026-09-07

Retain the 8×8 fallback specialization and connect fourteen existing mixed
16×16 SIMD kernels to checked dispatch. The first change cuts 2.2–3.4% from
the confirmed 2K/4K photo cells. Against that improved baseline, the dispatch
connection cuts about 6.1–6.2% from 8K photo decoding at one and eight workers.
Both gains survive matched alternate-alignment checks. Arithmetic and
reference construction are unchanged. That candidate took 1.64–2.28×
upstream time. The [mixed 8×8 follow-up](MIXED8.md) adds another measured
improvement and records its placement-sensitive threaded 8K limitation;
latest selected ratios are 1.52–2.22×. Parity and release acceptance remain
unmet.

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

## Rejected 16×16 scalar specialization

`inline16.patch.gz`, against `0c0a9eac`, extends the retained same-body
specialization to 16×16, 8-bit fallbacks. Affected 16×16/shared-fallback
symbols grow from 12,084 to 65,986 bytes (about 53 KiB). This additional
specialization is reverted; the retained 8×8 implementation is unchanged.

Five rotated paired rounds, 150 ms calibrated upstream work, validate all
150 measured runs. Baseline is the retained 8×8 specialization. No cell's
one-sided 95% upper ratio is below 1.0:

| Input | Workers | Paired change | Upper ratio |
|---|---:|---:|---:|
| photo-2k-min | 1 | -0.24% | 1.0008 |
| photo-2k-min | 8 | -0.69% | 1.0083 |
| photo-4k-t8 | 1 | -0.02% | 1.0002 |
| photo-4k-t8 | 8 | -2.00% | 1.0523 |
| photo-8k-t8 | 1 | -0.40% | 1.0018 |
| photo-8k-t8 | 8 | -0.63% | 1.0309 |
| map-4k-t8 | 1 | -0.20% | 1.0001 |
| map-4k-t8 | 8 | +1.59% | 1.0286 |
| map-8k-t8 | 1 | -0.43% | 1.0052 |
| map-8k-t8 | 8 | +2.35% | 1.0374 |

The short screen does not justify the code-size cost or promotion; no
longer confirmation was run. The next lead is that checked dispatch skips
existing mixed 16×16 SIMD kernels. Connecting them requires differential
validation against the real scalar fallback before measuring speed.

## Mixed 16×16 SIMD dispatch candidate

The checked x86 dispatcher omitted all fourteen mixed 16×16, 8-bit
transforms even though their safe SIMD kernels already exist. The candidate
connects those kernels through the existing token and block-view path.
Kernel names list row then column; `TxfmType` lists column then row. No
transform arithmetic or borrowing helper changes. The executable's ELF text
grows 29,520 bytes; the selected dispatcher/16×16 symbols grow 16,219 bytes.

The new test calls production dispatch and the real scalar fallback. It
checks 5,824 combinations of all fourteen types, scan-reachable coefficient
prefixes, extreme/random coefficients, two strides and two offsets. Whole
output buffers and coefficient tails are compared, including sentinel pixels
outside the block. A token sweep additionally exercises 504 SIMD and 336
declined cells on this host; declined calls must leave both buffers intact.
The test uses aligned picture storage and serializes token-state testing.
All 78 release library/committed-fixture tests pass. The first broader test
invocation named a nonexistent `decode_md5` target and exited before building;
the corrected `decode_md5_committed` invocation is archived separately.

Five rotated paired rounds validate all 150 measured image runs against
both references. Baseline is the retained 8×8 specialization:

| Input | Workers | Paired change | Upper ratio |
|---|---:|---:|---:|
| photo-2k-min | 1 | -1.87% | 0.9873 |
| photo-2k-min | 8 | -2.25% | 1.0015 |
| photo-4k-t8 | 1 | -3.47% | 0.9671 |
| photo-4k-t8 | 8 | -5.16% | 0.9937 |
| photo-8k-t8 | 1 | -5.99% | 0.9444 |
| photo-8k-t8 | 8 | -7.21% | 0.9632 |
| map-4k-t8 | 1 | +0.44% | 1.0106 |
| map-4k-t8 | 8 | -1.21% | 1.0204 |
| map-8k-t8 | 1 | -0.40% | 1.0038 |
| map-8k-t8 | 8 | +0.28% | 1.0274 |

This screen supports longer confirmation for photos. Map changes are
unresolved; the candidate has not established a general workload benefit.
`wire16-dispatch.patch.gz` records the production dispatch change against
`0c0a9eac`. The standalone binary includes the same retained 8×8/CDF changes;
`wire16-source-audit.json.gz` records its inputs and
`wire16-final-source-audit.json.gz` records the final test-harness source.
Runtime sources are identical between those two audits.

## Mixed 16×16 longer confirmation

Nine rotated paired rounds, 500 ms calibrated upstream work; all 324
measured runs match both references. No samples discarded. Peak host load
3.51, minimum available memory 27,265 MiB. Baseline includes the retained
8×8 specialization and CDF kernel.

| Input | Workers | Baseline ms | Candidate ms | Upstream ms | Paired change | Upper ratio |
|---|---:|---:|---:|---:|---:|---:|
| photo-2k-min | 1 | 30.979 | 30.467 | 17.369 | -1.83% | 0.9848 |
| photo-2k-min | 8 | 33.150 | 32.652 | 19.837 | -1.66% | 0.9977 |
| photo-4k-min | 1 | 108.936 | 105.087 | 59.878 | -3.40% | 0.9677 |
| photo-4k-min | 8 | 110.478 | 106.104 | 64.828 | -3.68% | 0.9689 |
| photo-4k-t8 | 1 | 108.319 | 104.687 | 59.843 | -3.35% | 0.9673 |
| photo-4k-t8 | 8 | 26.142 | 24.857 | 13.109 | -3.02% | 0.9957 |
| photo-8k-t8 | 1 | 260.301 | 244.642 | 146.309 | -6.10% | 0.9445 |
| photo-8k-t8 | 8 | 59.918 | 56.542 | 31.140 | -6.19% | 0.9470 |
| map-4k-t8 | 1 | 106.823 | 107.099 | 57.361 | +0.23% | 1.0065 |
| map-4k-t8 | 8 | 32.955 | 32.223 | 14.287 | -2.12% | 0.9818 |
| map-8k-t8 | 1 | 315.902 | 314.725 | 163.901 | -0.32% | 0.9998 |
| map-8k-t8 | 8 | 90.890 | 88.415 | 38.648 | -3.10% | 0.9838 |

All photo cells pass the confidence rule for improvement: about 1.7–1.8%
at 2K, 3.0–3.7% at 4K, and 6.1–6.2% at 8K. Threaded maps improve in this
confirmation, while serial 4K map remains unresolved (median +0.23%).
The serial 8K map upper bound is only just below 1.0, so its tiny change is
not a useful headline. Candidate/upstream paired medians still range from
1.64 to 2.28×, outside the parity goal.

The dispatch change reuses the existing `with_block_mut` callback, so its
picture/owned-band reference construction and token gate are unchanged.
Each selected kernel consumes and clears the 256 transform coefficients and
writes sixteen pixels in each of sixteen rows. Scan-prefix and sentinel
checks exercise those footprints; the full decoder conformance checks the
result against independently recorded images. These checks are evidence for
this dispatch change, not a proof of the whole decoder's soundness.

`mutation.py` deliberately maps `ADST_DCT` to the transposed kernel. The
new scalar differential test fails on pixel comparison, and the script
restores the exact candidate source in `finally`. Full checked conformance
then passes 766 vectors at one worker and again at eight workers, with two
existing infrastructure exclusions each. The selected CPU-permutation
smoke/CDF tests also pass. The test logs retain the expected mutation failure.

## Mixed 16×16 alignment confirmation

`alignment.py --variant wire16 --baseline-revision 0c0a9eac` builds both
arms with LLVM function alignment 5 instead of 4. Every named Rust decoder
text symbol (1,096 baseline, 1,116 candidate) is verified 32-byte aligned.
The script restores candidate dispatch in `finally`; test-only additions
are present but excluded from both standalone release consumers.
Upstream is used only for calibration and image validation in this check.

Nine paired rounds, 500 ms calibrated work, 108 validated measured runs:

| Input | Workers | Paired change | Upper ratio |
|---|---:|---:|---:|
| photo-2k-min | 1 | -1.90% | 0.9831 |
| photo-2k-min | 8 | -2.38% | 0.9817 |
| photo-4k-t8 | 1 | -3.76% | 0.9647 |
| photo-4k-t8 | 8 | -2.22% | 0.9894 |
| photo-8k-t8 | 1 | -6.43% | 0.9374 |
| photo-8k-t8 | 8 | -7.88% | 0.9380 |

All selected photo gains survive the alignment change. This supports
retaining the dispatch connection as an incremental improvement in the draft.
The primary confirmation remains the headline result; alternate-alignment
measurements are not substituted for it. Holdout, lifecycle, memory, video,
and remaining concurrency performance gates still apply.

## Candidate profiles and final local checks

Nine timer-only profiles cover baseline/candidate/upstream for serial 4K,
serial 8K, and eight-worker 8K photos. Each samples about three seconds of
measured decode work with `cycles:u` at 499 Hz; output validation succeeds
for every run. `profile_summary.py` groups reported self-cycle percentages:

| Checked build and workload | Entropy | Transforms | Loopfilter | Tracker |
|---|---:|---:|---:|---:|
| photo-4k-t8-t1-baseline | 44.82% | 15.95% | 9.44% | 6.19% |
| photo-4k-t8-t1-wire16 | 46.15% | 11.78% | 10.61% | 7.97% |
| photo-8k-t8-t1-baseline | 46.44% | 16.57% | 12.22% | 5.11% |
| photo-8k-t8-t1-wire16 | 43.00% | 12.43% | 12.77% | 8.48% |
| photo-8k-t8-t8-baseline | 33.54% | 12.28% | 18.58% | 13.77% |
| photo-8k-t8-t8-wire16 | 35.55% | 8.84% | 17.23% | 14.09% |

Transform self share drops in each paired profile. This is not an absolute
stage-time estimate: inlining, sampling variation, and the changing total
cost affect percentages. Upstream reports include 10–14% anonymous NASM
labels; their ownership was unresolved in this initial report, so incomplete
named stage sums could not establish a stage-level slowdown ratio.

The initial reading prioritized entropy by its checked self share. The
[follow-up mixed 8×8 investigation](MIXED8.md) resolves the sampled anonymous
labels to upstream entropy functions and normalizes cycles by timed frames.
That correction shows much closer sampled entropy cost and redirects the
next investigation toward transforms and loopfilter, alongside tracker
overhead. The follow-up also fixes and connects the mixed 8×8 SIMD kernels;
its direct arithmetic tests, paired timings, and scope are recorded there.
These profiles do not provide new evidence of substantial spinning.

Final local checks pass all seven selected debug tests (the new differential
and token sweeps plus five committed-vector/concurrency fixtures), strict
release/debug Clippy, root formatting and formatting of the included files.
The measured production sources match their recorded hashes after the
mutation and alignment builds. No public source declaration, dependency,
borrow policy or constructor changes. The declaration audit is narrower
than a cargo-semver-checks run. Cross-platform CI on this new dispatch
commit and the remaining performance acceptance gates still apply.
