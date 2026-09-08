# Checked still entropy experiments — 2026-09-07

Raw `.gz` evidence is in [verified R2 storage](../EXTERNAL_ARTIFACTS.md).
Run `python3 tools/fetch-benchmark-artifacts.py` from the repository root to
restore the original paths referenced below. Reports and manifests remain here.

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
performance. The follow-ups below test the remaining per-symbol SIMD call
boundary, including a dependency-only control. Neither improves on the
retained leaf CDF kernel.

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
JSONL chunks in numbered order to recover each original run log. `results/index.json`
lists index pages containing uncompressed checksums. The recorded `args` in each provenance JSON
reproduce the corresponding `benchmarks/stills-2026-09-07/compare.py` command.

The initial `experiments/*.patch.gz` reconstruct each source experiment against
baseline `d3231aea` (including its tests); follow-ups name their base below.
They preserve rejected and intermediate
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

The checkpoint below records conformance, token-fallback, debug-overflow,
and selected lifecycle results. Video, the expanded still corpus, untouched
holdouts, and the full performance acceptance matrix remain outstanding.
The entropy change adds no public API, unsafe block, dependency, borrow
policy, or constructor change.

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

CI exposed two independent issues, repaired in separate commits:

- The default C-FFI allocator cookie pointed into a dropped decoder. Its
  internal allocator clones now retain the pool Arc and pass owned handles
  directly to the allocation helper. Default callbacks also work when their
  addresses are not recognized, a second failure exposed by Miri. All 27
  assembly and 61 C-FFI library/integration checks pass, including the former
  SIGSEGV and a 16-generation retained-picture copy chain. The minimal test
  passes Stacked and Tree Borrows with strict provenance, exercising both
  owned defaults and C callback fallback. Both models are added to CI. The default checked allocator already owned
  its pool. The public C ABI is unchanged. See
  [the lifetime argument](../../docs/FFI_ALLOCATOR_LIFETIME.md).
- The whole-plane `__simd_test` save/restore protocol requires serial
  decodes. The same-process concurrency test now has its own binary and
  explicit regular-feature release/debug CI invocations. Its worker counts,
  repeated decodes, fixtures, and reference hashes are preserved. Locally,
  all 76 checked release library/integration tests, all five debug fixture
  tests, all five assembly fixture tests, and all three serial
  `__simd_test` tests pass. No failing hash
  assertion or concurrent workload was removed. See the protocol in
  [the ownership ledger](../../docs/OWNERSHIP_MODELS.md#7e-the-whole-plane-guard-audit-479-and-why-only-one-of-the-three-sites-was-a-bug).

The default checked doctest gate passes (nine tests, 13 pre-existing ignored
examples). All 26 GitHub checks on `8b13ed69` pass, including both Miri models,
x86/ARM conformance, native ARM and Windows builds, and threading gates.
The remaining performance gates are still required; the draft is not ready
to merge or release.

## Follow-up: move dispatch to a coefficient block

`block-v1.patch.gz` (against `8b13ed69`) gives the coefficient decoder one
baseline SSE2 boundary and forces its scalar body to inline there. The goal
was to inline the retained CDF leaf into the coefficient loop. Generated code
still has 15 CDF calls across both bit depths, down from 17, and the overall
coefficient routines remain approximately 38 KiB each. All 76 checked unit
and fixture tests pass, and the six-cell screen validates 90 timed runs.

| 4K input | Workers | Paired change from retained CDF kernel |
|---|---:|---:|
| Photo, minimum tiles | 1 | -0.30% |
| Photo, minimum tiles | 8 | -0.48% |
| Photo, eight tiles | 1 | +0.41% |
| Photo, eight tiles | 8 | -1.09% |
| Map, eight tiles | 1 | -0.26% |
| Map, eight tiles | 8 | +3.31% |

This is rejected and reverted: no convincing serial win, with a tiled-map
regression in the screen. The original coefficient source is restored byte
for byte. Results, code-generation counts, build hashes, and exact patch are
retained. An independent four-lane array formulation also compiled to scalar
shifts, so it was rejected at code inspection without a decode speed claim;
its source and assembly are archived under `experiments/cdf-auto-codegen.*`.

## Follow-up: inline through magetypes

Use archmage source `7a67c74c569148e5c3470bc95538649dec60d8b4` (0.9.29),
with a dependency-only control that keeps the retained CDF kernel. Freeze
that standalone consumer lockfile and reuse it for every new-dependency arm.
The build helper's optional `--archmage-repo`, `--lockfile`, and explicit
`--refresh-lock` support this comparison; normal builds remain locked.

The new CDF formulation uses safe `u16x8` operations with X64V3Token, falling
back to the retained SSE2 kernel. On the unmodified dependency this leaves
492 CDF/magetypes calls across the coefficient routines and their outlined
helpers. A narrow generator change reuses the existing checked SSE2 baseline
wrapper for 128-bit, 16-bit splat, comparison, and uniform shifts. It removes
the magetypes calls in these routines, but grows each coefficient routine
from about 38 KiB to about 46.6 KiB. This is a code-size observation, not proof
that size alone explains the timings.

Five rotated paired rounds, 150 ms calibrated upstream work, six cells and
150 measured runs per screen. All ordered visible hashes match upstream and
independent dav1d. Percentages are paired changes from the retained leaf
kernel, using the same rounds for each comparison:

| 4K input | Workers | Dependency only | Existing magetypes | SSE2 backend patch |
|---|---:|---:|---:|---:|
| Photo, minimum tiles | 1 | -0.06% | +42.32% | +0.88% |
| Photo, minimum tiles | 8 | +0.27% | +40.58% | +1.25% |
| Photo, eight tiles | 1 | +0.49% | +43.04% | +1.03% |
| Photo, eight tiles | 8 | +2.71% | +37.12% | +4.54% |
| Map, eight tiles | 1 | +0.34% | +28.14% | +0.80% |
| Map, eight tiles | 8 | -0.29% | +20.42% | +1.21% |

A second experiment caches the checked X64V3Token once in MsacContext,
removing per-CDF token summoning. The other decoder and dependency changes
remain the same. A fresh paired screen includes both controls:

| 4K input | Workers | Dependency only | Patched, uncached | Patched, cached |
|---|---:|---:|---:|---:|
| Photo, minimum tiles | 1 | +0.38% | +1.16% | +1.91% |
| Photo, minimum tiles | 8 | -0.16% | +1.66% | +2.13% |
| Photo, eight tiles | 1 | +0.47% | +0.77% | +1.78% |
| Photo, eight tiles | 8 | +3.98% | +4.28% | +8.85% |
| Map, eight tiles | 1 | +0.68% | +0.88% | +1.99% |
| Map, eight tiles | 8 | -2.64% | -1.81% | +0.37% |

Reject all three decoder variants. The backend patch fixes the large
out-of-line penalty but does not produce an overall improvement here;
caching the token also fails to help. The tiled results have visible
control variation and are screening evidence, not acceptance estimates.
All 76 checked decoder unit/fixture tests pass for the patched uncached and
cached variants. The patched dependency passes 584 selected integer/generic
tests, including uniform shift edge counts. This does not substitute for its
full release gate or establish explicit liveness of all three dispatch paths.
No dependency change is retained in the implementation or proposed for release.

`mage-sse2-backend.patch.gz` applies to archmage `7a67c74c` and includes both
the generator and regenerated output. `mage-cached-decoder.patch.gz` applies
to rav1d-safe `e4357c0d`; `mage-sse2-msac.rs.gz` preserves the uncached source.
`mage-consumer.lock.gz` freezes dependencies. Build records and provenance
contain immutable binary and lockfile hashes; `mage-*-screen*` contains all
raw samples. Both repositories' source and the decoder Cargo files were
restored byte for byte after archiving. The clean dependency's generator,
registry, token, and soundness health checks also passed before editing.

## Follow-up: interval selection without an indexed stack lookup

`interval-cmov.patch.gz`, against `19e859ed`, replaces adapt4's two interval
lookups with balanced `core::hint::select_unpredictable` expressions. This
API has been stable since Rust 1.88, within the manifest's 1.89 minimum
([standard-library documentation](https://doc.rust-lang.org/std/hint/fn.select_unpredictable.html)).
The four possible symbols choose exactly the same `(upper, lower)` pairs:
`(rng, v0)`, `(v0, v1)`, `(v1, v2)`, `(v2, 0)`. CDF updates, refill, and
normalization are unchanged.

Generated coefficient routines go from 112 to 12 indexed stack-address
operands each, with substantially more conditional moves. All 76 checked
decoder unit/fixture tests pass. The five-round six-cell screen validates
all 90 measured runs against both references, but serial time regresses:

| 4K input | Workers | Paired change from retained CDF kernel |
|---|---:|---:|
| Photo, minimum tiles | 1 | +0.99% |
| Photo, minimum tiles | 8 | +1.45% |
| Photo, eight tiles | 1 | +1.28% |
| Photo, eight tiles | 8 | -0.25% |
| Map, eight tiles | 1 | +0.87% |
| Map, eight tiles | 8 | -2.59% |

Reject and restore the original source. The tiled-map screen improvement
does not justify serial regressions or establish an acceptance result.
`interval-cmov-*` records the code-generation counts, binaries, tests, and
every timing sample. Removing indexed memory operations alone did not help.

## Entropy usage probe

`entropy-probe.patch.gz`, against `19e859ed`, adds ordinary counters owned by
each exclusively accessed MsacContext. Its diagnostic field prints on drop;
there are no atomics, shared counters, concurrent resets, or process-global
policy changes. `probe.py` checks every output against independent dav1d and
keeps the diagnostic build's timings out of performance analysis. This patch
is archived and reverted, not part of the shipped decoder.

All 12 development streams were probed at one and eight workers. Each
invocation decodes four visible frames, giving 96 checked visible decodes.
Every normalization is accounted for by a symbol or boolean operation, and
every recorded CDF lookup has one symbol. Counts match exactly between one
and eight workers for every stream. The bitstreams are the same frozen
two-source development seed used by the timing screens.

| Input, eight tiles | Entropy operations/frame | Four-symbol share | Equal-probability boolean share | Four-symbol CDF already saturated |
|---|---:|---:|---:|---:|
| Photo 2K | 2,242,531 | 55.83% | 31.6% | 96.10% |
| Photo 4K | 7,683,137 | 57.24% | 31.1% | 98.67% |
| Photo 8K | 18,121,706 | 61.42% | 29.8% | 99.48% |
| Map 2K | 2,101,676 | 49.93% | 27.9% | 96.12% |
| Map 4K | 5,756,076 | 49.13% | 27.3% | 98.15% |
| Map 8K | 15,614,584 | 48.96% | 27.9% | 99.14% |

Across all layouts, symbol zero accounts for 35–45% of four-symbol calls;
it is not an overwhelmingly dominant early exit. Refills occur once per
approximately 38–40 entropy operations. Saturated CDFs account for 96.10–99.80%
of four-symbol calls: their update rate is seven and their count stays 32.
This makes a fixed-rate CDF specialization a stronger measured lead than
further optimizing the uncommon refill. `entropy-probe-analysis.json.gz`
contains exact fractions; the summary and per-cell files retain every local
counter, command, visible hash, and the probe binary hash.

## Follow-up: saturated CDF and constant lane masks

Two independent candidates apply to `19e859ed`. `cdf-steady.patch.gz` handles
`count == 32 && rate == 7` with immediate vector shifts and leaves the already
saturated count lane intact; callers load `count` from that lane. Other
states retain the original kernel. `cdf-mask-table.patch.gz` instead keeps
the variable-rate arithmetic and obtains the lane mask from a four-element
constant table indexed by the already-clamped symbol. Neither changes the
caller or expands its borrowed footprint.

Both pass all 76 checked decoder unit/fixture tests, including exhaustive
CDF arithmetic and live/disabled baseline-token permutations. The mask-table
leaf is 77 bytes versus 98; the steady-state function, including its fallback,
is 197 bytes. These are code-generation observations, not speed claims.

| 4K input | Workers | Saturated CDF | Mask table |
|---|---:|---:|---:|
| Photo, minimum tiles | 1 | -0.49% | -1.14% |
| Photo, minimum tiles | 8 | +0.04% | -1.70% |
| Photo, eight tiles | 1 | +0.76% | -0.53% |
| Photo, eight tiles | 8 | +1.80% | +2.74% |
| Map, eight tiles | 1 | -0.04% | -0.21% |
| Map, eight tiles | 8 | +0.26% | +0.44% |

Five rotated paired rounds validate all 120 measured runs against both
references. The fixed-rate variant provides no convincing gain. The mask
table's small serial changes and mixed tiled results do not justify promotion
from this screen. Both are archived and restored, with raw samples and code
in `cdf-steady-*`, `cdf-mask-table-*`, and `cdf-steady-mask-screen*`.

## Follow-up: CPU tier around coefficient decoding

`block-v3.patch.gz`, against `19e859ed`, revisits the earlier block boundary
with X64V3Token. This tier proves BMI2 and LZCNT as well as AVX2, allowing the
compiler to select independent variable shifts and leading-zero counts in
scalar code. Runtime detection and the original fallback remain in place;
build-wide target flags and dependencies are unchanged.

Code inspection shows that the first formulation leaves three large inner
coefficient-class helpers outlined at the baseline tier. A separate
`block-v3-inline.patch.gz` forces these helpers to inline. Its two hot
bit-depth routines now contain 63 LZCNT sites each and 138–139 SHLX / 144 SHRX
sites, with no BSR. They also have 25 CDF calls and 60 VZEROUPPER sites each;
the baseline has 8–9 CDF calls and no VZEROUPPER sites. These are static
instruction counts, not dynamically weighted costs. The complete symbol
inventory includes the outlined helpers and scalar fallback, so it does not
mistake a smaller outer function for eliminated work.

Both formulations pass all 76 checked decoder unit/fixture tests. Five
rotated paired rounds validate all 120 measured runs against both references:

| 4K input | Workers | Partial V3 boundary | Full inner-loop V3 boundary |
|---|---:|---:|---:|
| Photo, minimum tiles | 1 | -0.79% | +3.41% |
| Photo, minimum tiles | 8 | -0.37% | +3.95% |
| Photo, eight tiles | 1 | +0.22% | +4.33% |
| Photo, eight tiles | 8 | +0.53% | +3.61% |
| Map, eight tiles | 1 | +0.37% | +4.03% |
| Map, eight tiles | 8 | -1.88% | +0.80% |

Neither is retained. The partial boundary does not provide a convincing
serial benefit; the full boundary regresses serial time. The source is
restored exactly. A possible follow-up would need evidence that CDF
operations inline within the same CPU tier, eliminating the extra boundaries,
before repeating this design. `block-v3-*` records both builds, tests, all
timings, and the complete code-generation inventory.

## Follow-up: vector operations inside the V3 caller

`block-v3-mage.patch.gz`, against `e86da579`, combines the fully inlined V3
coefficient body with the earlier uncached magetypes CDF formulation. Use
unmodified archmage `7a67c74c` and the same frozen consumer lockfile as before;
`block-v3-mage-control` changes only the dependencies of `block-v3-inline`.
The root development lockfile is separately archived for the test run.

The V3 routines have no magetypes calls, so the vector operations do inline
inside this CPU tier. However, token summoning and CDF fallback code remain:
each V3 routine has 29 baseline CDF calls and 80 VZEROUPPER sites. Each also
grows to about 44 KiB. Static call counts include untaken fallbacks and must
not be mistaken for runtime operation counts.

All 76 checked tests pass. The CDF token-permutation test explicitly records
the V3, V1, and scalar choices and requires all three when the host supports
V3. The six-cell, five-round screen validates 150 measured runs:

| 4K input | Workers | V3 with old dependencies | Dependency control | Inlined magetypes |
|---|---:|---:|---:|---:|
| Photo, minimum tiles | 1 | +4.21% | +3.39% | +6.33% |
| Photo, minimum tiles | 8 | +4.03% | +2.82% | +7.16% |
| Photo, eight tiles | 1 | +4.71% | +3.63% | +6.24% |
| Photo, eight tiles | 8 | +5.94% | -0.94% | +6.36% |
| Map, eight tiles | 1 | +4.34% | +3.93% | +5.94% |
| Map, eight tiles | 8 | +1.39% | +0.03% | +4.92% |

All percentages are paired changes against the retained CDF kernel. The
dependency-control variation is visible rather than folded into a claimed
kernel speedup. Reject and restore both decoder and Cargo files. A future
version would need to pass an existing token through the hot path instead
of summoning again; inlining alone did not produce a gain. Transform-shape
profiling is the next independent lead after these unsuccessful CDF screens.
