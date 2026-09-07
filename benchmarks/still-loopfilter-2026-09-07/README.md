# Still loopfilter mask investigation — 2026-09-07

Follow-up to checked mixed 8×8 SIMD, commit `984d41dde603cc274f9751ca9e5d098a83ee099c`,
in draft [PR #528](https://github.com/imazen/rav1d-safe/pull/528). The latest
[transform profiles](../still-transforms-2026-09-07/MIXED8.md) estimate
57.6 / 175.1 / 312.6 self Mcycles/frame in checked loopfilter for serial 4K,
serial 8K, and eight-worker 8K photo, versus 6.8 / 28.3 / 42.8 upstream.
These single-profile estimates have no confidence bounds and guide the next
experiment; they are not stage-latency or acceptance measurements.

The x86 SIMD kernels compute narrow and wide filter alternatives and then
select original or filtered pixels by lane masks. There is no early return
when every filter-mask lane is false. Before adding branches, measure how
often no output lane uses the calculated values. This investigation changes
no borrowing/copy footprint and does not revive the rejected wide filter
bands documented in [OWNERSHIP_MODELS.md](../../docs/OWNERSHIP_MODELS.md).

## Census protocol

`instrument.py` inserts diagnostic calls after mask construction in all
thirteen 8-bit x86 SIMD leaf kernels. Monotonic atomic counters record calls,
lanes passing the filter mask, final narrow/mid/wide selections, and calls
with no selected lane for each alternative. Counters live under `__ablate`,
are never reset, and are read after clients finish. Every family stays enabled.
The script builds an isolated census consumer and restores both production
files in `finally`; hashes verify restoration. The exact diagnostic patch
and source/site audit are archived.

`census.py` covers twelve frozen development bitstreams at 1, 2, 4, 8, 16,
and 24 requested workers: 72 fresh processes, 288 completed visible decodes.
Each process runs one serial reference and three requested-worker decodes.
The matching serial per-frame counts are subtracted before dividing by three.
All resulting counts are integral, and every kernel's per-requested-decode
counts are identical across all six worker levels for each stream.

There are 216 before/after/reference hash-validated decodes and 72 timed
decodes checked by exact visible-frame count. Every case matches independent
dav1d 1.5.3. The diagnostic RESULT times are excluded from all speed results.
Counter traffic changes scheduling and cost; this is an arithmetic-use census,
not a waiting/latency profile. These streams still come from only two sources;
expanded development and holdout requirements remain outstanding.

| Input | Kernel calls/frame | Calls with no eligible lane |
|---|---:|---:|
| photo-2k-min | 124,758 | 29.83% |
| photo-2k-t8 | 123,230 | 29.75% |
| photo-4k-min | 449,546 | 25.87% |
| photo-4k-t8 | 444,753 | 25.92% |
| photo-8k-min | 1,251,201 | 15.77% |
| photo-8k-t8 | 1,246,929 | 15.73% |
| map-2k-min | 193,318 | 20.36% |
| map-2k-t8 | 193,130 | 20.39% |
| map-4k-min | 632,654 | 17.69% |
| map-4k-t8 | 632,810 | 17.69% |
| map-8k-min | 1,937,974 | 16.72% |
| map-8k-t8 | 1,937,895 | 16.75% |

Counts above include all instrumented kernels. On the eight-tile 4K photo,
the 8-tap eight-lane vertical kernel rejects every lane in 84.7% of calls,
and its horizontal counterpart in 64.4%. At 8K photo these are 55.2% and
32.3%. The 16-tap horizontal kernel has no eligible lane in 23.6% of 4K photo
calls and 32.7% at 8K, versus only 2.2% for 8K map. Chroma 6-tap calls reject
all lanes only about 1–3% of the time. A blanket branch in every kernel may
therefore cost more than a targeted change; the census does not establish
the break-even point.

The mid/wide/narrow counts classify **final output selection**. For width 16,
`wide = fm & flat_inner & flat_outer`, `mid = fm & flat_inner & !flat_outer`,
and `narrow = fm & !flat_inner`. Their sum equals passing lanes. A lane that
fails `fm` contributes to none of them. An unused mid result need not mean
that every expression shared with wide arithmetic is dead.

## Experiments

`candidate.py` builds two uninstrumented candidates from the same frozen
source and restores the baseline in `finally`. `early8` returns immediately
when the filter mask is zero in the four 8-tap kernels. `earlywide` adds the
same check to the four 16-tap kernels. The 4- and 6-tap kernels stay untouched.
All existing mask and output arithmetic remains unchanged. The new control
flow skips writes that would reproduce original pixels; token dispatch and
reference/guard geometry stay the same. For an all-zero mask the original implementation
selects the original pixel at every store, so returning has the same visible
result without calculating unused alternatives. Direct scalar differential
and adversarial partial-lane tests are required before retaining a candidate;
benchmark output checks alone do not replace them.

These are experimental binaries, not retained runtime changes. Source patches,
binary/manifest/lock hashes, and build commands are archived. Baseline timing
uses the immutable `wire8-row-checked` consumer from the transform campaign.
Both new consumers use the same checked features, compiler, fat LTO,
codegen-units=1, and function-alignment=4, without target-cpu=native.

## Reproduction and evidence

Use `instrument.py --repo ... --work-dir ... --baseline-revision 984d41dd
--shared-target ...`, then `census.py --work-dir ... --binary ...`, sequentially
through the workspace's `scripts/run-heavy --mem 16G --jobs 8`. The default
census matrix includes all six worker levels and all twelve streams.
`candidate.py` takes the same build arguments and creates both uninstrumented
variants. `benchmarks/stills-2026-09-07/compare.py` performs rotated paired
measurements with upstream calibration and independent visible-frame checks.

Large assets, binaries, and source snapshots remain under
`/home/lilith/tmp/rav1d-still-loopfilter-2026-09-07`, with inputs shared from
the frozen still corpus. There is no verified off-host backup. Small lossless
records are compressed and indexed under `results`; each index page and
uncompressed evidence payload has a SHA-256 checksum. Baseline machine,
compiler, and input provenance inherit the still and transform campaign records.


Before retaining either candidate, the direct differential gate should use
`src/loopfilter.rs::loop_filter` through a test-only slice adapter. Cover zero,
fully active, and mixed eligible lanes; narrow/mid/wide results; extrema and
random pixels; varied strides and offsets; and every affected SIMD width.
Compare the entire destination buffer, including surrounding sentinels.
Mutating the early-return condition to reject a partially eligible group
must fail this gate. Then require default checked conformance, debug overflow
checks, live/disabled CPU-token coverage, and longer paired/placement controls.
The arithmetic shortcut itself does not justify weakening any concurrency or
reservation test.


## Initial three-arm screen

Five rotated paired rounds, 150 ms upstream calibration, twelve selected
cells and three timing arms. All 180 measured runs pass visible-frame and
before/after hash validation against upstream rav1d and independent dav1d.
No samples discarded. Normal checked features and alignment are matched.
Changes below are paired median elapsed-time changes against the retained
mixed 8×8 baseline; the last column is the one-sided 95% upper bound for
`earlywide` versus baseline.

| Input | Workers | 8-tap only | 8/16-tap | 8/16-tap upper ratio |
|---|---:|---:|---:|---:|
| photo-2k-min | 1 | -1.97% | -2.10% | 0.9856 |
| photo-2k-min | 8 | +0.17% | +0.09% | 1.0157 |
| photo-4k-min | 1 | -1.61% | -1.54% | 0.9896 |
| photo-4k-min | 8 | -1.26% | -1.34% | 1.0083 |
| photo-4k-t8 | 1 | -1.55% | -1.03% | 0.9919 |
| photo-4k-t8 | 8 | +1.58% | +0.45% | 1.0368 |
| photo-8k-t8 | 1 | -0.67% | -1.23% | 0.9894 |
| photo-8k-t8 | 8 | +0.27% | -2.22% | 1.0760 |
| map-4k-t8 | 1 | -0.81% | -0.72% | 0.9949 |
| map-4k-t8 | 8 | -1.06% | -1.21% | 1.0008 |
| map-8k-t8 | 1 | -0.71% | -0.80% | 1.0068 |
| map-8k-t8 | 8 | -0.40% | +1.60% | 1.0328 |

`earlywide` shows about 1.0–2.1% improvement in the selected serial photo
cells, but threaded results remain unresolved. Eight-worker 8K map has a
+1.60% median with upper 1.0328; this is a reason to investigate, not a
universal speedup claim. No runtime change from this experiment is retained.
Longer measurements and code-placement controls are still needed before
deciding whether to keep the early-return variant. The later direct scalar
and adversarial gates are recorded in the follow-up below.
The diagnostic census and timing binaries are distinct, and both scripts
restore the production baseline after their builds.


## Packed six-tap follow-up

The [packed six-tap record](PACKED6.md) adds a direct production-scalar oracle,
5,200 original-kernel cells, and a mutation that proves the partial-mask gate
can reject an incorrect early return. The early-return implementation remains
experimental. Six-tap kernels reject few groups, so a separate candidate
combines adjacent equal-level UV groups in eight signed 16-bit lanes. Its
6,000 leaf cells, 64 production-grouping cases, arithmetic and memory argument,
corrected conformance worker-setting record, and independent timing results
are documented there. This follow-up changes no tracker or picture footprint.
