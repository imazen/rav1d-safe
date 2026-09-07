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
   shifting, reducing two variable shifts per probability to one. Screening
   in progress; a new exhaustive test checks every u16 input bit pattern.

Median time change against the pinned checked baseline; negative is faster:

| 3840×2160 input | Workers | Generic SSE2 | Specialized SSE2 | Split fallback |
|---|---:|---:|---:|---:|
| Photo, minimum tiling | 1 | −0.58% | −0.81% | +0.07% |
| Photo, minimum tiling | 8 | +0.36% | −1.54% | +1.02% |
| Photo, eight tiles | 1 | +0.05% | −3.21% | −0.28% |
| Photo, eight tiles | 8 | −4.53% | −2.71% | −1.16% |
| Map, eight tiles | 1 | −0.68% | +0.57% | −0.03% |
| Map, eight tiles | 8 | −2.50% | −2.49% | −2.27% |

These are five-round screening results (150 ms adaptive upstream work),
not the nine-round / 500 ms acceptance gate. There is no convincing serial
win. The few-percent tiled improvements need confirmation and code-layout
checks before promotion. All three screens validated all ordered visible hashes
against upstream and independent dav1d; each contains 90 measured runs.

A new timer-only profile assigns 34.83% of serial self CPU samples to the
first SIMD kernel and 11.70% to the remaining `decode_coefs` body. Work moved
between symbols without a significant total reduction. Generated code has
an out-of-line kernel, six saved general registers, and a large inlined rare
scalar fallback. Specializing and separating that fallback reduced the
count-3 kernel from 834 to 399 bytes but did not improve serial time.

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
