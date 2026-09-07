# Eight-position horizontal wide filter — 2026-09-07

Follow-up to retained six-tap SIMD, commit
`f092774b48ec7eedd180617d1ae1e5700ac7dc99`, in draft
[PR #528](https://github.com/imazen/rav1d-safe/pull/528). The original census
measured 302,304 four-position horizontal width-16 calls per eight-tile 8K
photo. The existing vertical driver already combines such groups; horizontal
width-16 filtering still processes four positions at a time. This candidate
processes eight, using signed 16-bit arithmetic for 8-bit pixels.

## Arithmetic and footprint

The private compute body is derived from the existing eight-position vertical
width-16 kernel, replacing 32-bit operations with their signed 16-bit
counterparts. It computes the same wide, mid, and narrow alternatives and
selects them with the same masks. The caller supplies zero-extended byte taps
and three u8 thresholds.

All positive wide-filter sums normalize to 16 with rounding 8, so their
maximum is `16*255+8 = 4088`. The mid-filter sums normalize to eight with
rounding four, bounded by 2044. Pixel differences stay within −255–255; the
filter-mask expression is at most 637. Narrow pre-clipping arithmetic is
within −893–892, corrections within −16–15, and resulting pixels within
−16–271. These bounds fit signed i16. Signed-word to unsigned-byte saturation
performs the final scalar pixel clamp. Ineligible lanes select original
pixels; no early-return experiment is included here.

`packed_bounds.py` models the actual source's twelve wide and six mid linear
sums and all 64 lanes in its unpack network. The normalized sums and exact
8×8 transpose pass this model. This verifies those modeled operations, not
compiler lowering, all masks, or the complete decoder. Actual SIMD output is
independently compared with the decoder's production scalar `loop_filter`.

The horizontal loader reads eight bytes then exactly six bytes per row,
covering taps −7 through +6. Two local padding bytes are zero-filled before
the second transpose. Output stores cover exactly twelve bytes per row,
taps −6 through +5, using an eight-byte and a four-byte store. Neither outer
tap is written. Ordinary checked slices remain in this entry point even
with the `unchecked` feature.

The production horizontal Y driver combines only adjacent width-16 groups
whose effective levels match, including the existing lookback rule. Absolute
stride must be at least 14, so rows are independent. The second mask bit must
exist, and advancing consumes exactly two four-position groups. Different
widths/levels, mask gaps, a zero effective level, or the final single mask bit
retain the original path. The existing width-8 case keeps its kernel.

All arithmetic occurs inside the caller's existing slice. The compact-window
calculation, mask-derived tap reach, per-row reservations, copy/write-back
policy, picture publication, and process-global mode are unchanged. Combining
these independent rows adds no picture access or borrowed reference across a
gap. These facts are a local argument, not a proof of the whole borrow tracker
or decoder protocol.

## Correctness gates

The first leaf gate passes 6,400 cells before production dispatch is changed.
The final gate adds explicit narrow-filter clipping below zero and above 255
to every leaf, for **7,680 live SIMD/scalar cells** on this host. It covers
positive/negative strides, offsets, pixel and threshold extremes, random data,
partially eligible masks, and wide/mid/narrow alternatives. Whole-buffer
comparison includes surrounding sentinels.

A separate **32-case production grouping gate** exercises both level-byte
selectors, positive/negative strides, same/different levels, lookback, zero
effective levels, differing widths, gaps, and mask bits 30/31. A monotonic,
test-only invocation counter proves the new kernel executes exactly when
expected. The complete output must match scalar filtering group by group.

Another gate gives the kernel **32 exact input spans**, including packed
14-byte rows, larger strides, negative strides, offsets, and narrow clipping.
Each valid span matches scalar and changes its active pixels. Removing its
final input byte must panic while loading, before any writes; the full parent
buffer remains unchanged. These are 32 additional short-view rejection cases.

Both deliberate mutations fail their intended gate:

- Combining a nonzero but unequal second level fails the grouping invocation
  count in case 1.
- Rounding the six-byte tail read up to eight bytes fails a valid exact-span
  case. The extra bytes do not participate in arithmetic, which is why output
  comparisons with oversized buffers would miss this footprint defect.

Both mutations are restored in `finally`. The restored candidate passes all
**85 checked release library/fixture tests**, with no skipped test in that
selection. Independent conformance passes **766 vectors at one worker and
again at eight**, with two existing infrastructure exclusions per run. Exact
commands explicitly set `RAV1D_MD5_THREADS`, the recognized worker variable.
The initial leaf-only build emitted an unused test-counter warning before
the production-grouping gate was added; final release tests emit no such
warning. No production arithmetic correction was needed after the first
leaf gate.

## Measurements and retention decision

The isolated checked consumer shares the baseline's compiler, lockfile,
features, fat LTO, one codegen unit, and function-alignment=4. Its source and
binary hashes are recorded. ELF `.text` grows by **5,312 bytes** relative to
the retained six-tap consumer. Instrumentation is absent from timing builds.

The five-round screen used 150 ms upstream calibration. Confirmation used
nine rotated paired rounds and 500 ms upstream calibration per cell. The
second placement uses function-alignment=5 for both baseline and candidate;
upstream supplies calibration and output references there, without becoming
a mismatched timing arm. All twelve selected cells come from the two already
exposed 8-bit sources. No sample was discarded.

All **720 measured runs** (180 screen, 324 confirmation, 216 placement) pass
exact timed frame counts and before/after visible hashes against pinned
upstream rav1d and independent dav1d. Timed frames are counted, not individually
hashed. The table gives paired median elapsed-time change against the six-tap
baseline and the one-sided empirical-bootstrap 95% upper ratio. Negative
changes are faster; an upper bound above one leaves improvement unresolved.

| Input | Workers | Normal change | Normal upper | Alignment 5 change | Alignment 5 upper |
|---|---:|---:|---:|---:|---:|
| photo-2k-min | 1 | -0.82% | 0.9933 | -0.57% | 0.9949 |
| photo-2k-min | 8 | -0.40% | 1.0107 | -0.12% | 1.0037 |
| photo-4k-min | 1 | -0.55% | 0.9952 | -0.79% | 0.9935 |
| photo-4k-min | 8 | +0.67% | 1.0097 | +0.27% | 1.0054 |
| photo-4k-t8 | 1 | -0.63% | 0.9974 | -0.87% | 0.9935 |
| photo-4k-t8 | 8 | -0.78% | 0.9965 | -1.05% | 1.0053 |
| photo-8k-t8 | 1 | -2.49% | 0.9759 | -2.65% | 0.9760 |
| photo-8k-t8 | 8 | -3.01% | 0.9736 | -3.69% | 0.9918 |
| map-4k-t8 | 1 | -0.63% | 0.9941 | -0.35% | 0.9980 |
| map-4k-t8 | 8 | +0.26% | 1.0061 | -0.60% | 0.9983 |
| map-8k-t8 | 1 | -1.25% | 0.9876 | -1.07% | 0.9914 |
| map-8k-t8 | 8 | -1.20% | 0.9986 | -0.27% | 1.0106 |

All six serial cells have upper bounds below one in both placements. The
8K photo improves **2.49% / 2.65% serially** and **3.01% / 3.69% at eight
workers** in normal / alternate alignment. Most other threaded gains remain
unresolved; minimum-tile 4K at eight workers has small positive medians in
both placements. This is not a universal threaded speedup. The candidate is
retained for the repeatable serial and 8K photo improvement. Percentages from
separate experiments are not added together.

The candidate still takes **1.5179–2.2040× upstream** in these selected
normal-alignment cells. Neither these two sources nor this limited concurrency
matrix satisfies the full campaign's acceptance protocol.

Final validation also passes eight selected debug tests, strict release/debug
Clippy, root formatting, and formatting of the included Rust files. CPU smoke
and CDF-update checks each pass ten permutations. The three new leaf/grouping/
span tests pass with `unchecked` enabled, retaining ordinary checked input
slices. Source hashes match the conformance-tested and timed implementation.
All 26 GitHub checks passed on preceding six-tap head `f092774b`; the new
wide-filter implementation requires its own CI run.

Six fresh timer-only profiles complete with matching output. Loopfilter's
share of self cycles changes from 10.30% to 8.54% at serial 4K, 11.07% to 9.07%
at serial 8K, and 15.79% to 13.42% at eight-worker 8K. Inlining, sampling, and
denominators affect these single-profile proportions; they have no confidence
bounds and are not stage latency or acceptance measurements.

## Reproduction and API scope

Exact command arrays, feature sets, binary/source SHA-256 values, every paired
sample, scalar/mutation logs, profile control records, and the source/API audit
are compressed and indexed in `results`. The complete runtime patch and both
mutation patches are in `experiments`. Use the pinned standalone builder in
`benchmarks/tracker-sharding-2026-09-07/build.py`, and
`packed6_controls.py --experiment packed16` for the matched alternate
placement. Heavy commands run sequentially through the workspace's
`run-heavy --mem 16G --jobs 8`. To rerun the arithmetic/transpose model:

```sh
python3 benchmarks/still-loopfilter-2026-09-07/packed_bounds.py \
  --repo /home/lilith/work/zen/rav1d-safe \
  --wide-kernel src/safe_simd/loopfilter_packed16.rs \
  --output /home/lilith/tmp/packed16-bounds-new.json
```

The new module and functions are private or `pub(super)`; the invocation
counter and scalar gates are test-only. No exported signature, dependency,
borrow tracker, picture reservation, process-global policy, or const
constructor changes. The audit verifies unchanged manifests and relevant
ownership source files against `f092774b`. This is a source declaration review,
not a rustdoc API/semver-checker run or a proof of the entire crate.

Large local binaries and profiles remain under
`/home/lilith/tmp/rav1d-still-loopfilter-2026-09-07`, without a verified off-host
backup. Small evidence is archived losslessly in the PR. The full campaign
still requires expanded development sources, untouched holdouts, remaining
thread/instance/lifecycle/memory/video cells, and the goal's per-group/per-cell
confidence bounds. The [source-admission audit](../still-expanded-2026-09-07/README.md)
corrects the seed sources' canonical split labels and records rejected native
8K candidates. No parity or release-readiness claim is made. See
[PERFORMANCE_PARITY_GOAL.md](../../docs/PERFORMANCE_PARITY_GOAL.md).
