# Still parity checkpoint

The tested runtime is `f423404e42acb0a8883a9d716786bae44566428a` in draft
[PR #528](https://github.com/imazen/rav1d-safe/pull/528). **All 26 GitHub checks
pass on that commit**, including native ARM/Windows checks and ARM decode
permutations. Source-preparation work after it changes no decoder code.

## Current result

The latest horizontal wide-filter optimization improves the selected 8K photo
**2.49–2.65% serially** and **3.01–3.69% at eight workers** against its immediate
six-tap baseline, across two matched code alignments. All six serial cells
have one-sided upper ratios below one in both placements; most other threaded
changes remain unresolved. See the [full record](../benchmarks/still-loopfilter-2026-09-07/PACKED16.md).

**Parity is not achieved.** Current checked decode time is 1.52–2.20× pinned
memorysafety/rav1d upstream in these two-source, 8-bit development cells. The
goal allows 1.10× per workload group and 1.25× per mandatory cell, including
confidence bounds. All twelve cells below still fail the per-cell target.
These are persistent in-memory decode measurements, not request-lifecycle
latency, and not results on the new sources or holdouts.

| Input | Workers | Checked median ms | Upstream median ms | Paired ratio | One-sided 95% upper |
|---|---:|---:|---:|---:|---:|
| photo-2k-min | 1 | 28.02 | 17.38 | 1.613 | 1.619 |
| photo-2k-min | 8 | 30.50 | 20.01 | 1.524 | 1.536 |
| photo-4k-min | 1 | 97.09 | 59.94 | 1.621 | 1.625 |
| photo-4k-min | 8 | 100.02 | 66.11 | 1.518 | 1.543 |
| photo-4k-t8 | 1 | 96.59 | 59.69 | 1.619 | 1.628 |
| photo-4k-t8 | 8 | 23.18 | 13.06 | 1.792 | 1.803 |
| photo-8k-t8 | 1 | 226.33 | 146.30 | 1.550 | 1.554 |
| photo-8k-t8 | 8 | 52.90 | 31.49 | 1.670 | 1.684 |
| map-4k-t8 | 1 | 103.50 | 57.30 | 1.809 | 1.809 |
| map-4k-t8 | 8 | 31.29 | 14.13 | 2.204 | 2.224 |
| map-8k-t8 | 1 | 299.83 | 164.18 | 1.826 | 1.828 |
| map-8k-t8 | 8 | 83.79 | 38.61 | 2.159 | 2.183 |

The paired median ratio can differ from the ratio of the two marginal medians.
Nine rotated rounds used 500 ms upstream calibration, with no discarded
samples. Visible output matches pinned upstream and independent dav1d before
and after timing; every timed frame is counted. Percentages from successive
optimizations are not added together.

## What is ready for review

- Six retained performance changes, their causal baselines, rejected candidates,
  code-placement caveats, and raw evidence are linked from the PR description.
- Latest local gates pass 85 checked release tests, eight selected debug tests,
  766-vector conformance at one and eight workers, strict Clippy, formatting,
  CPU permutations, scalar differential tests, and deliberate wrong-level /
  widened-load mutation checks. Exact slice footprints and arithmetic bounds
  are documented; this is not a universal soundness proof.
- The separate C-FFI allocator lifetime repair and its Miri/callback/retained
  picture tests are documented in [FFI_ALLOCATOR_LIFETIME.md](FFI_ALLOCATOR_LIFETIME.md).
- Source membership, hashes, family separation, and splits are now pinned for
  **12 development and 12 holdout sources**. Each set has four photographs,
  three textures, two maps, and three text documents. Full JPEG validation and
  reviewed native vector crops precede AV1 work. See the
  [source checkpoint](../benchmarks/still-expanded-2026-09-07/SOURCE_FREEZE.md).
  Encoder/quality/edge assignments and generated bitstreams are not frozen yet.
- Performance changes introduce no exported API, dependency, borrowing-policy,
  or const-constructor change. The latest audit is a source declaration review,
  narrower than a rustdoc/semver-checker run. The 0.3.x release branch is separate.

## Work remaining, in order

1. Freeze the expanded workload assignments and bitstreams using libaom plus
   an independently implemented encoder. The clean local zenrav1e source is
   available at `605946821afa839ca80f2b9bb226917238e9dba3`, package 0.2.0;
   its CLI has not been built for this campaign. Keep its name/version distinct
   from upstream rav1e. Prepare 10-bit/chroma and explicit edge coverage.
2. Establish and profile the expanded **development** baseline. Keep holdout
   performance sealed. The current seed profiles identify remaining transforms,
   narrow filtering, and tracker/copy metadata costs; actual spinning was a
   small fraction of sampled time. A spinlock rewrite has no demonstrated
   payoff. The loop-filter census identifies many narrow horizontal map calls
   as one concrete follow-up; any algebra/fusion change still needs paired
   timing and the same footprint/correctness gates.
3. Test the required 1/2/4/8/16/24-thread and independent-instance matrix,
   mixed sizes, process priming, repeated create/decode/flush/drop, cold-start
   and persistent modes, p95 request latency, peak RSS, and video regressions.
4. Once development candidates meet the thresholds, run the untouched holdout
   acceptance matrix and confidence-bound aggregation. Missing or failing
   cells cannot be averaged away. Cross-platform release checks remain scoped
   separately from parity on this Ryzen host.

There is no defensible completion ETA for parity at this gap. Further small
kernel wins alone are not evidence that the remaining 52–120% slowdown will
close. This checkpoint is ready for review and resumption; it is not a crate
release or a completed performance goal. The authoritative criteria remain in
[PERFORMANCE_PARITY_GOAL.md](PERFORMANCE_PARITY_GOAL.md).
