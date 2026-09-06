# Release and soundness review setup

Prepared 2026-09-05 on `r7900x`. This records review inputs and environment
checks, not a soundness verdict. Historical audit claims still need independent
verification against published artifacts and current source.

## Sources and pinned inputs

- Guidance: `~/work/claudehints/CLAUDE.md`, its benchmarking and Rust topics,
  this repository's `CLAUDE.md`, `docs/AGENT_BRIEF.md`, and
  `crates/rav1d-disjoint-mut/AUDIT.md`.
- Zen workspace is `~/work/zen`, remote
  <https://github.com/imazen/zen-workspace>. Its fleet configuration is
  `scripts/.mise.toml`; do not apply ARM target flags to this x86 host.
- Review baseline: `26acb2ba45ab42aa61fa7f940fc713fcddbcfe1b` (`main@origin`).
  The previous local checkout was `f9458f43`; the clean checkout was advanced
  through jj. A colocated jj repository and read-only upstream remote are ready.
- Upstream: <https://github.com/memorysafety/rav1d>, fetched as `upstream`,
  pinned at `d3d1cd67059f47803919be8276650e5870c9fd02`.
  **There is no common Git ancestor with this fork.** The 8,320 commits returned
  by `origin/main..upstream/main` are not 8,320 missing ports. Compare patches,
  source, and historical fork provenance before classifying a fix as missing.
- Corpus: <https://code.videolan.org/videolan/dav1d-test-data>, commit
  `61afa1ceb6029be1a8ea3f6b9c9a9672700f13bb`, downloaded into
  `test-vectors/dav1d-test-data`. It contains 774 IVF files; this is a file count,
  not a count of passing conformance tests.

## Published artifacts

Registry metadata was fetched from `https://crates.io/api/v1/crates/<crate>`;
each downloaded `.crate` archive was verified against its registry SHA-256.
Full checksums, dates, yank state and embedded VCS metadata are in
[`review-releases-2026-09-05.json`](review-releases-2026-09-05.json).
Archives, extracted sources and raw registry responses are cached at
`~/tmp/rav1d-review-2026-09-05/`.

| Crate | Last three published versions | Provenance detail |
| --- | --- | --- |
| rav1d-safe | 0.5.7, 0.5.6, 0.5.5 | 0.5.6 is yanked; 0.5.5 records `dirty: true` |
| rav1d-disjoint-mut | 0.3.1, 0.3.0, 0.2.1 | All three were unyanked when fetched |

Current source declares rav1d-safe 0.6.0 and DisjointMut 0.3.1. A matching
manifest version does not make current DisjointMut source identical to the
published 0.3.1. At the pinned baseline there are 440 commits since the decoder
0.5.7 VCS commit, and 73 commits touching the DisjointMut directory since its
0.3.1 VCS commit. For 0.5.5, use the archive as the source of truth because
the recorded Git commit omits dirty changes.

## Local execution

Host: AMD Ryzen 9 7900X (Zen 4), 24 logical CPUs, about 29 GiB RAM.
Stable rustc is 1.98.1 (`48a229cea`, LLVM 22.1.8). Installed nightly Miri is
0.1.0 (`da86f4d072`, 2026-07-24). Record/recheck these versions on later runs;
`nightly` is a mutable toolchain name. `perf stat` cycles/instructions works.
nextest, cargo-fuzz, NASM, Clang, Meson, Ninja, Valgrind and just are installed.

Run from the repository root:

```sh
scripts/review.sh disjoint
scripts/review.sh no-std
scripts/review.sh miri-stacked
scripts/review.sh miri-tree
scripts/review.sh clippy
scripts/review.sh decoder-smoke
scripts/review.sh decoder-debug
scripts/review.sh bench-list
scripts/review.sh bench-smoke
```

The runner uses the workspace `run-heavy` helper with a 16 GiB cgroup cap,
eight build jobs, nice/ionice, and scratch under `~/tmp`. Review invocations
serialize via `flock`. Logs include commands, commit, working-copy status,
compiler and lockfile checksum. It clears ambient target flags for baseline
builds. `REVIEW_LOG_DIR` and `REVIEW_HEAVY_RUNNER` override local paths.
The helper is external to this repository; verify its availability on other
hosts. The lock only coordinates invocations of this runner.

Miri runs explicitly enable `aligned,pic-buf,zerocopy`, in addition to default
`std`, so storage-specific tests actually execute. Do not use `--all-features`:
several experimental probes deliberately bypass tracking. Full Miri runs retain
`--no-fail-fast` so one target's failure does not hide other targets.

`clippy` selects the production library. The existing `just clippy` uses dev
`--all-targets`, conflicting with release-only test compile guards (documented
in `CLAUDE.md`). Decoder smoke uses nextest process isolation and release
mode. `decoder-debug` runs the same suites with overflow checks in the dev
profile. Full corpus conformance is a separate review gate, not implied by a
smoke pass.

`bench-list` lists benchmark groups and `bench-smoke` exercises cases in
Divan test mode; neither reports a performance result. For actual comparisons,
use the in-process decode benchmarks, identical input hashes, explicit thread
counts, separate preserved binaries for each revision/feature arm, and
interleaved repeated measurements on an idle host. Benchmark checked,
unchecked and ASM separately. Verify decoded output and completed frame count
before accepting timings: some existing loaders/decoders silently omit missing
inputs or errors. The optional photo AVIF fixtures are not downloaded here.
Record flags and any function-alignment experiment in both arms; no native ISA
flags in a production baseline. The macOS E-core/nice guidance in AGENT_BRIEF
does not describe this Linux Zen 4 machine. Do not extrapolate to ARM or Zen 5.

## Review order

1. Diff consecutive published tarballs, then each latest tarball against main.
   Map unsafe implementations, feature changes, dependencies and public APIs.
2. Recheck historical guard-move/protector UB against published versions;
   inspect `e0187a30`, `825df674` and `tests/guard_move_release.rs`.
   The extracted 0.3.1 archive still has `slice: &'a mut V` / `slice: &'a V`
   guard fields. A published-version reproducer remains to be run.
3. Audit tracker publication/release ordering, narrow/wide exclusion, rectangle
   footprints and element-vs-byte units (`02660934`), panic paths and auto-traits.
   Verify that test assertions demonstrate actual contention and coverage.
4. Audit decoder callers, FFI/unchecked modes and SIMD dependency contracts.
   `forbid(unsafe_code)` in the decoder alone is not a transitive soundness proof.
5. Triage upstream patches by behavior. `9d78516f` fixes T.35 underflow;
   current `src/obu.rs` already uses `checked_sub` in that path, so establish
   equivalence and release coverage before porting. Also inspect upstream
   `ed159a3e` (remove non-Send/Sync Rc storage) and `921942d6` plus `6f95e808`
   (fallible chroma location API). These are candidates, not confirmed missing
   fixes; this fork uses different storage and public APIs.

## Setup verification

- Formatting check: passed for both workspace packages.
- DisjointMut native tests with `aligned,pic-buf,zerocopy`: 102 tests plus one
  doctest passed; two pre-existing doctests ignored.
- DisjointMut without default features: 81 tests plus one doctest passed;
  storage-feature tests are intentionally absent in this configuration and
  were exercised in the storage-enabled run above. Two doctests ignored.
- Miri smoke: all seven tests across `aligned_miri`, `guard_move_release` and
  `pic_buf_overflow` passed under each aliasing model. This is not the full Miri
  suite or a multi-seed concurrency sweep.
- Production library Clippy with `-D warnings`: passed.
- Release decoder smoke: all 22 tests across three binaries passed, zero skipped.
- Dev-profile decoder smoke: the same 22 tests passed, zero skipped, on x86_64.
  Test names mentioning ARM do not imply ARM kernels executed on this host.
- Decode benchmark built and all 15 selected 8-bit, 10-bit and film-grain cases
  executed in test mode. No comparative timing claim was made.

Full logs are in `~/tmp/rav1d-review-2026-09-05/logs/`. The resolved lockfile
is copied alongside them (SHA-256
`e56c59914325c0750308e3e9831a70fd5d8235e6a750fd9a56cb64bf969fe9c0`).
In particular, the fresh resolution selected archmage 0.9.26; these setup runs
do not reproduce the historical releases' dependency resolutions.
