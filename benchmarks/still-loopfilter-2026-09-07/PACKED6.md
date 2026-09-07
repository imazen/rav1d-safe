# Packed six-tap SIMD — implementation and evidence

Starting production source is `984d41dd`, unchanged by the evidence-only
`e14d9bbd` head of draft [PR #528](https://github.com/imazen/rav1d-safe/pull/528).
The [mask census](README.md) measured 308,636 vertical and 328,628 horizontal
six-tap calls per 8K photo frame: about 51% of its 1,246,929 SIMD filter calls.
Only 1–3% of these calls have no eligible lane, so the early-return experiment
does little for this population. Existing kernels process four positions
using 32-bit arithmetic. This experiment processes eight using 16-bit lanes.

## Arithmetic and memory argument

The private compute helper receives six vectors of zero-extended bytes and
three u8 thresholds. It applies the same formulas as the existing six-tap
kernel, with signed 16-bit operations instead of 32-bit operations.

| Quantity | Bound before any signed 16-bit operation could overflow |
|---|---|
| Input pixel / threshold | 0–255 |
| Difference / absolute difference | −255–255 / 0–255 |
| Filter-mask expression | 0–637 |
| Six-tap weighted sum including rounding | 0–2044 |
| Narrow expression before clipping to −128–127 | −893–892 |
| Narrow primary correction after shifting | −16–15 |
| Result before final pixel clipping | −16–271 |

All values fit signed i16. Unsigned packing performs the final clamp to
0–255. Mask values are zero or all-one bits; they select lanes rather than
participating in weighted sums. The scalar oracle is the production
`src/loopfilter.rs::loop_filter`, reached through a test-only slice adapter.
The test does not reproduce the filter formulas or compare one SIMD width
with another SIMD width. `packed_bounds.py` additionally tracks the actual
source's four linear weighted sums and all 64 input lanes through its unpack
network. The sums normalize to eight with rounding four, and the modeled
permutation is exactly a transpose. This script models named intrinsics; it
does not verify compiler lowering, masks, or the full decoder.

Horizontal loading reads exactly six bytes per row at taps −3 through +2,
zero-fills two local padding lanes, then transposes eight rows of u16.
Vertical loading reads eight contiguous bytes on each of six tap rows.
Outputs write only taps −2 through +1: four bytes per horizontal row, or
eight bytes on each of four vertical tap rows. There is no rounded read into
the +3/+4 tail and no write to the unchanged outer taps. Ordinary checked
slice operations remain in these entry points even with `unchecked` enabled.

The UV driver combines two groups only when they are adjacent, both select
six taps, and their effective filter levels match, including the existing
lookback rule. It also requires absolute stride at least six horizontally
or eight vertically, so the eight lane footprints are independent. A zero
next mask bit, a gap, a different width or level, or the final mask bit keeps
the existing four-position path. The caller advances exactly two groups
when it combines them. The groups have no cross-lane arithmetic dependence,
so evaluating them together preserves the result of the original order.

This changes arithmetic grouping inside an existing slice. It does not
change the outer dispatcher, `lf_run_reach`, compact window, guard extent,
copy/write-back policy, picture publication, or process-global mode. No new
borrowed reference spans a gap. These local facts do not constitute a proof
of every other decoder or disjoint-mut operation.

## Direct tests and mutation checks

The new leaf gate first passed 5,200 cells on the unchanged thirteen kernels.
It varies positive/negative strides, offsets, threshold extremes, pixel
extremes, random data, inactive/fully active/partially active masks, and the
flat versus narrow alternatives. Whole-buffer comparison includes surrounding
sentinels. Explicit partial-mask cases must change their active pixels.

The early-return candidate also passes those 5,200 cells. A mutation that
returns when *any* lane is ineligible fails the partial-mask case; restoring
the correct predicate passes again. Exact source hashes and logs are in
`earlywide-mask-validation.json` and the corresponding logs.

Adding the two packed kernels expands the leaf gate to 6,000 live cells on
this AVX2/AVX-512 host. A separate 64-case gate calls the production horizontal
and vertical UV drivers and checks actual packed-kernel invocation counts.
It covers both U/V level bytes, positive/negative strides, equal/different
levels, zero-level lookback, zero effective levels, differing widths, gaps,
and mask bits 30/31. The full buffer must match scalar filtering group by
group. Its counters are test-only, monotonic atomics; they never reset and
cannot change dispatch. Deliberately accepting a nonzero but unequal next
level fails the grouping gate. The correct source is restored afterward.

The first grouping-test build failed because the fixture used `Align16` as
a constructor although it is a type alias. The fixture now uses its existing
`ArrayDefault` implementation. Before rerunning, its leading buffer reserve
was also corrected for 128 negative-stride positions. Those were test setup
changes, not arithmetic corrections. The failed log and
`packed6-fixture-correction.json` are retained.

The final candidate passes all 83 checked release library/fixture tests.
Two initial conformance invocations both passed 766 vectors, but the intended
eight-worker invocation supplied the wrong environment variable. Both are
**one-worker controls**. `packed6-conformance-env-audit.json` records this
before results are interpreted. The recognized variable is
`RAV1D_MD5_THREADS`; the separate explicit eight-worker gate now also passes
766 vectors with the same two infrastructure exclusions and is recorded
in `packed6-conformance-explicit-t8.json` and its log. Do not treat the earlier filename ending in `t8` as
proof of its worker setting.

## Measurements and selection

The isolated `packed6-checked` consumer uses the same checked features,
compiler, lockfile, fat LTO, codegen-units=1, and function-alignment=4 as the
previous consumers. Its source and binary hashes are recorded. The timing
baseline is the immutable mixed 8×8 consumer; an independent early-return
arm keeps the two optimization ideas separate. All primary timings exclude
instrumentation, startup, I/O, and hashing, and validate visible output
against upstream rav1d and independent dav1d around the measured work.

The five-round, 150 ms screen passes all 180 measured runs. It shows
roughly 1–3% serial photo improvement, while most threaded cells remain
unresolved. The separate nine-round confirmation uses 500 ms upstream
calibration, twelve cells, and three measured arms: baseline, packed6, and
upstream. All **324 measured runs** pass the visible-frame and before/after
hash checks. No samples are discarded.

| Input | Workers | Paired change vs baseline | One-sided 95% upper ratio | Paired ratio vs upstream |
|---|---:|---:|---:|---:|
| photo-2k-min | 1 | -0.67% | 0.9970 | 1.6320 |
| photo-2k-min | 8 | +0.21% | 1.0100 | 1.5235 |
| photo-4k-min | 1 | -1.15% | 0.9912 | 1.6325 |
| photo-4k-min | 8 | -0.40% | 0.9995 | 1.5175 |
| photo-4k-t8 | 1 | -1.14% | 0.9899 | 1.6317 |
| photo-4k-t8 | 8 | -1.23% | 0.9992 | 1.8002 |
| photo-8k-t8 | 1 | -2.28% | 0.9779 | 1.5906 |
| photo-8k-t8 | 8 | -2.32% | 1.0106 | 1.7065 |
| map-4k-t8 | 1 | -0.66% | 0.9957 | 1.8185 |
| map-4k-t8 | 8 | -1.59% | 0.9875 | 2.1908 |
| map-8k-t8 | 1 | -1.01% | 0.9930 | 1.8523 |
| map-8k-t8 | 8 | -1.30% | 0.9891 | 2.1556 |

Normal alignment confirms a 0.7–2.3% serial photo gain and 0.7–1.0% serial
map gain. Four of the six threaded cells also have improvement bounds below
1.0, but two cells remain unresolved: the eight-worker 2K
photo has a +0.21% median, and the eight-worker 8K photo has a −2.32% median
with an upper bound above 1.0. The latter is not evidence of a confirmed
threaded gain. These are selected development cells, not full-goal strata.
ELF `.text` increases by **3,648 bytes**, from 2,649,910 to 2,653,558.

A second pair of consumers with function-alignment=5 passes all **216
measured runs** across the same twelve cells (nine paired rounds, 500 ms
upstream calibration). `packed6_controls.py` restores the two existing source
files from the pinned baseline for its baseline build and restores the final
candidate in `finally`. The new module/test files remain on disk but are not
included or compiled by the baseline. Source hashes for each variant and
restoration are recorded. Both timing arms share alignment=5; the separately
used upstream calibration/hash oracle is not an alignment=5 timing arm.
The serial gains survive this placement change. The threaded gains are less
consistent: only 4K minimum-tile photo and 8K map have improvement bounds below
1.0 in both placements. No general threaded improvement is claimed.

| Input | Workers | Alternate-placement paired change | One-sided 95% upper ratio |
|---|---:|---:|---:|
| photo-2k-min | 1 | -1.25% | 0.9909 |
| photo-2k-min | 8 | -0.11% | 1.0072 |
| photo-4k-min | 1 | -1.71% | 0.9839 |
| photo-4k-min | 8 | -0.42% | 0.9982 |
| photo-4k-t8 | 1 | -1.32% | 0.9895 |
| photo-4k-t8 | 8 | +0.01% | 1.0062 |
| photo-8k-t8 | 1 | -2.48% | 0.9764 |
| photo-8k-t8 | 8 | -2.43% | 1.0059 |
| map-4k-t8 | 1 | -0.99% | 0.9932 |
| map-4k-t8 | 8 | -0.67% | 1.0019 |
| map-8k-t8 | 1 | -1.21% | 0.9904 |
| map-8k-t8 | 8 | -2.57% | 0.9819 |

The candidate passes seven selected debug tests (both new loopfilter gates and
five committed/concurrent fixtures), strict release/debug Clippy, root
formatting, and separate formatting of the included files. The runtime source
hashes still match the conformance-tested and benchmarked candidate. Final
CPU-permutation smoke and CDF-update gates pass ten permutations each. Both
new leaf/grouping gates also pass with `unchecked` enabled; these new entry
points retain ordinary checked slices in that build. All six timer-only
profiles complete with matching visible output. The six-tap implementation
is selected for retention based on the serial improvement in both placements and these correctness
checks; full goal acceptance remains a separate gate.

Fresh profiles assign 12.32% → 9.01% of serial 4K self cycles and 13.11% →
9.37% at serial 8K to loopfilter symbols. Eight-worker 8K changes from 17.10%
to 16.37%. The old four-position leaf's samples fall, while the new UV driver
absorbs its packed arithmetic. These are single-profile diagnostic proportions,
affected by inlining and denominators; they have no confidence bounds and are
not stage latency or acceptance measurements. Full symbol reports and profile
control/provenance records are archived.

## Reproduction and remaining scope

Use the pinned standalone builder in
`benchmarks/tracker-sharding-2026-09-07/build.py` for the normal checked consumer
and `packed6_controls.py` for alternate-placement controls. Invoke both through
`run-heavy --mem 16G --jobs 8`, with no other heavy job. The exact build,
conformance, benchmark, analysis, and validation commands are archived in the
indexed `results` files. The full candidate patch, both mutation patches, and
all measured samples are retained. To rerun just the algebra/permutation model:

```sh
python3 benchmarks/still-loopfilter-2026-09-07/packed_bounds.py \
  --repo /home/lilith/work/zen/rav1d-safe \
  --output /home/lilith/tmp/packed6-bounds-new.json
```

All production changes are private x86 8-bit filter code. The scalar adapter
and invocation counters are test-only. Public signatures, dependencies, the
tracker, process-global policy, and const constructors are unchanged. This
source-level statement is narrower than a rustdoc API or semver-checker run.
No safety or performance conclusion for the entire crate follows from this
local arithmetic change. Broad development/holdout sources, remaining
concurrency/lifecycle/memory/video cells, and full parity gates remain
outstanding. The current candidate is still 1.52–2.19× upstream in the selected
normal-alignment cells.

The next measured lead is the horizontal 16-tap kernel: 302,304 calls per 8K
photo frame still process four positions each. A scratch eight-position
proposal is **uncompiled and not included in production**. Its positive linear
sums also normalize correctly, with every weighted intermediate at most
4,088, but that does not establish its machine correctness or speed. This
proposal is archived separately so its status cannot be mistaken for a
validated implementation.
