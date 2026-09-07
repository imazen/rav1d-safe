# Checked still entropy experiments — 2026-09-07

Draft PR: https://github.com/imazen/rav1d-safe/pull/528. This is an ongoing
investigation, not a parity or release claim. The active acceptance limits
are 10% per group and 25% per mandatory cell, with the full corpus and
confidence protocol in `docs/PERFORMANCE_PARITY_GOAL.md`.

The initial serial photo profile puts about 46% of self CPU samples in
coefficient/entropy decoding. The checked/unchecked comparison shows very
little serial difference, so simply removing borrow checks would not solve
this bottleneck. All candidates here retain checking and `forbid(unsafe_code)`.

## Experiment sequence

1. `candidate`: safe SSE2 adapt4 selection and CDF update through archmage's
   existing X64V1Token. Preserve short-slice fallback, probabilities at the
   32768 boundary, disabled CDF updates, and every unused CDF entry.
2. `specialized`: specialize the same private kernel for counts 1, 2, 3.
   The public entry point and its signature are unchanged.
3. `split`: move the rare probability-boundary fallback outside the SIMD
   body so it cannot enlarge the hot kernel's register-save requirements.
   This also failed to produce a meaningful serial win. The SSE2 kernels
   remain archived experiments, not part of the current source.
4. `cdf-select`: portable scalar CDF update chooses its direction before
   shifting, reducing two variable shifts per probability to one. This
   regressed serial time by 4–7% and is reverted. Fewer instructions did not
   make the critical dependency chain faster.
5. `cdf-simd`: vectorize only the three-entry probability update, keeping
   symbol selection and coder normalization/refill in the existing scalar
   caller. A leaf kernel avoids the earlier refill-related register saves.
   Retain the original scalar arithmetic for other counts and fallback.
   The six-cell screen improves all medians by 2–4%; longer 2K/4K/8K
   confirmation and the remaining correctness gates are in progress.

Median time change against the pinned checked baseline; negative is faster:

| 3840×2160 input | Workers | Generic SSE2 | Specialized SSE2 | Split fallback | Scalar one-shift | Leaf SIMD CDF |
|---|---:|---:|---:|---:|---:|---:|
| Photo, minimum tiling | 1 | -0.58% | -0.81% | +0.07% | +7.00% | -3.68% |
| Photo, minimum tiling | 8 | +0.36% | -1.54% | +1.02% | +6.91% | -2.78% |
| Photo, eight tiles | 1 | +0.05% | -3.21% | -0.28% | +6.91% | -3.23% |
| Photo, eight tiles | 8 | -4.53% | -2.71% | -1.16% | +5.43% | -3.98% |
| Map, eight tiles | 1 | -0.68% | +0.57% | -0.03% | +4.21% | -2.07% |
| Map, eight tiles | 8 | -2.50% | -2.49% | -2.27% | +1.00% | -2.41% |

These are five-round screening results (150 ms adaptive upstream work),
not the nine-round / 500 ms acceptance gate. The first four variants have no convincing
serial win. The fifth has a consistent small gain that needs longer runs and
code-layout checks before promotion. All five completed screens validated
all ordered visible hashes
against upstream and independent dav1d; each contains 90 measured runs.

A new timer-only profile assigns 34.83% of serial self CPU samples to the
first SIMD kernel and 11.70% to the remaining `decode_coefs` body. Work moved
between symbols without a significant total reduction. Generated code has
an out-of-line kernel, six saved general registers, and a large inlined rare
scalar fallback. Specializing and separating that fallback reduced the
count-3 kernel from 834 to 399 bytes but did not improve serial time.

## Longer confirmation

Nine rotated paired rounds, at least 500 ms calibrated upstream work per
round. All 216 measured runs validated against upstream and dav1d. Ratios
below are paired medians; the confidence bound is the exact one-sided 95%
empirical-bootstrap upper bound against the checked baseline. No samples
were discarded. See `analyze.py` and the raw confirmation evidence.

| Input | Workers | Baseline ms | Candidate ms | Upstream ms | Paired change | Upper ratio |
|---|---:|---:|---:|---:|---:|---:|
| photo-2k-min | 1 | 32.981 | 31.987 | 17.346 | -3.14% | 0.9711 |
| photo-2k-min | 8 | 35.296 | 34.223 | 20.009 | -3.36% | 0.9702 |
| photo-4k-t8 | 1 | 114.998 | 111.017 | 59.566 | -3.46% | 0.9671 |
| photo-4k-t8 | 8 | 27.715 | 26.434 | 13.273 | -3.61% | 0.9718 |
| photo-8k-t8 | 1 | 271.842 | 262.053 | 146.258 | -3.62% | 0.9649 |
| photo-8k-t8 | 8 | 63.008 | 62.079 | 31.331 | -2.55% | 0.9916 |
| map-8k-t8 | 1 | 346.602 | 349.203 | 167.987 | -1.78% | 0.9968 |
| map-8k-t8 | 8 | 94.538 | 92.872 | 39.291 | -1.18% | 1.0137 |

The photo cells improve by 2.6–3.6% in paired medians, with all upper bounds
below 1.0. The eight-worker 8K map result is unresolved: its upper bound is
1.0137. The serial 8K map has a large timing outlier; paired median and
ratio-of-medians differ, and the original sample is retained. The run-heavy
log reported peak host load 11.07 and minimum available memory 23,287 MiB;
this motivates a controlled-load repeat for that stratum, not outlier removal.

These cases remain about 1.7–2.4× upstream in ratios of median times. They
do not meet the 1.10 group / 1.25 cell goal. They are still only development
seed images and do not establish video, lifecycle, memory, holdout, or ARM
performance. The next measured lead is the remaining per-symbol SIMD call
boundary: compare magetypes token-backed operations that can inline inside
the scalar caller, with a dependency-only control build if dependencies change.

## Reproduction and evidence

Use the standalone build and still harnesses from the existing baseline:

```sh
ROOT=/home/lilith/tmp/rav1d-still-parity-2026-09-07
env TMPDIR=/home/lilith/tmp /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 benchmarks/tracker-sharding-2026-09-07/build.py \
  --repo /home/lilith/work/zen/rav1d-safe --work-dir "$ROOT" --label LABEL --modes checked
```

Exact comparison commands, immutable binary SHA-256s, inputs, frame counts,
raw stdout, profiles, and test/build logs are in `results/*.gz`. Decompress
JSONL chunks in numbered order to recover each original run log. `index.json`
contains uncompressed checksums. The recorded `args` in each provenance JSON
reproduce the corresponding `benchmarks/stills-2026-09-07/compare.py` command.

`experiments/*.patch.gz` reconstruct each source experiment against baseline
`d3231aea` (including its tests). They preserve rejected and intermediate
implementations for review; only the current source is compiled normally.

Compiler: Rust 1.98.1 / LLVM 22.1.8, release fat LTO, one codegen unit,
`-C llvm-args=-align-all-functions=4`, default CPU target. No root development
feature unification in timed consumers. All heavy work runs sequentially
under run-heavy on the Ryzen 9 7900X host. Corpus, upstream source pin,
source-image acquisition/license data, and full machine provenance inherit
`benchmarks/stills-2026-09-07/README.md` and its evidence.

Large immutable binaries and perf.data remain under the scratch ROOT above;
encoded streams and native source images remain at the paths in corpus.json.
They have not been mirrored off-host: `/mnt/v` and Tower are unavailable.
No scratch-only asset is represented as a verified backup.

## Correctness scope

Two new differential tests compare symbol, full CDF, range, difference,
normalization count, and input byte position against the scalar reference.
They cover 84,480 consecutive-symbol cases through refill/tail boundaries
and more than 50,000 interval-boundary cases, including zero and 32768
probabilities, short slices, count-rate transitions, and disabled updates.
Both tests passed for the baseline, generic, and specialized implementations.
The initial baseline run failed an invented coverage-count assertion; adding
more rounding-boundary input states made it meaningful. No pixel or coder
state mismatch caused that failure, and no expectation was weakened.

Full conformance, token-fallback, debug-overflow, thread lifecycle, video,
expanded still corpus, and untouched holdout gates remain outstanding for
any retained implementation. The changes add no public API, unsafe block,
dependency, borrow policy, or constructor change.

## Leaf CDF kernel argument

The new private helper handles exactly three probabilities plus their count.
It receives a checked `&mut [u16; 4]`; its sole eight-byte store cannot touch
later CDF entries. `zerocopy::IntoBytes` views these initialized u16 values
as bytes, and native-endian scalar conversion packs them into an SSE2 value.
There are no raw pointers, new references outside the registered footprint,
retained guards, allocations, or process-global mutations in the kernel.

For each probability `p`, SSE2 wrapping subtraction computes the same
`(32768 - p) mod 65536` as the scalar helper. Both right shifts are unsigned.
The lane mask chooses an increase for `i < val` and a decrease otherwise;
wrapping addition/subtraction matches the original u16 arithmetic even for
zero, 32768, and the high half of u16. The fourth lane is replaced with
`count + (count < 32)` before storing. This increment cannot overflow.

Clamping `val` to 3 preserves `i < val` for all three probability lanes,
including unusually large callers' values, and makes the signed lane compare
exact. Only shifts below 16 enter the SIMD kernel; other rates retain the
original scalar behavior. Other CDF sizes and a disabled baseline token also
retain the original scalar implementation. Selection, normalization/refill,
and the disabled-update branch execute in their original order.

`archmage::X64V1Token` proves SSE/SSE2 support. The kernel is compiled only
for x86-64 without assembly MSAC; other architectures retain their existing
implementation. The checked release consumer has no testable-dispatch
feature union, so summoning this architectural baseline token folds away.
Test builds deliberately disable it and must demonstrate both paths.

The exhaustive arithmetic test covers 7,864,320 combinations over every
u16 value, all AV1 rates, both one- and three-probability CDFs, every symbol
direction, and the count-rate boundaries. It computes expected values with
u32 division/modulo, independently of the SIMD shift/mask implementation.
Additional dispatch tests use mixed extreme probabilities, every shift from
0 through 15, and oversized symbol/count values.

The source API diff adds only a private helper and tests. Public signatures,
Cargo manifests/lockfile, the disjoint-mut crate, and const constructors are
unchanged. Miri/Loom publication-protocol gates are not triggered by this
private arithmetic change; conformance and dispatch gates still apply.

## Checkpoint validation

- All 70 checked release library tests pass, including explicit live/disabled
  baseline-token coverage and existing same-process decoder lifetime tests.
- Checked reference conformance: 766 vectors pass at one worker and again
  at eight workers, with the harness's two existing infrastructure exclusions
  reported in each log (not introduced by this change).
- The selected committed-vector/lifecycle tests and smoke/CDF-update CPU
  permutations pass in release; all nine selected debug entropy and committed
  vector tests also pass with overflow checks enabled.
- `mutation.py` deliberately puts the count in the wrong SIMD lane. The
  exhaustive test rejects it. The runner restores the original source in
  `finally`, and the complete 70-test library run passes afterward.
- The source-declaration API audit confirms unchanged public MSAC declarations
  and no manifest, lockfile, or disjoint-mut edits. It is not presented as a
  cargo-semver-checks run.

CI on the preceding experimental checkpoint exposed two additional issues:
`__simd_test`'s existing whole-plane save/restore harness collides with the
new concurrent committed-vector test; assembly unit tests also crash in
`picture_policy_is_local_and_survives_decoder_lifetimes`. The assembly test
fails in code where the entropy candidates are compiled out. Both issues
are being investigated; the draft is not ready to merge or release.
