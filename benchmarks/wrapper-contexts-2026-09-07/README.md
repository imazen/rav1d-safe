# Wrapper feature contracts

Raw `.gz` evidence is in [verified R2 storage](../EXTERNAL_ARTIFACTS.md).
Run `python3 tools/fetch-benchmark-artifacts.py` from the repository root to
restore the original paths referenced below. Reports and manifests remain here.

The two token-audit commits `576a0cb6` and `63015ed9` were dropped from PR
#528. Deliberate token-name/constant counterfeiting is outside the agreed
threat model. This change fixes the source feature contracts instead.

All **237** former `forge_token_dangerously()` call sites now acquire
`X64V3Token::from_context()` or `NeonToken::from_context()` as the first
statement of a function annotated `#[archmage::rite(v3)]` or
`#[archmage::rite(neon)]`. This includes macro templates and 31 private MC
helpers, not only extern-C wrappers. X86 names end in `_v3`; ARM names retain
`_neon`. Private call sites and macro invocations use the new names.

Each acquisition has `#[deny(unsafe_op_in_unsafe_fn)]`. Without that local
lint, an unsafe FFI function could implicitly authorize calling a stronger
`from_context()` even after its feature declaration was accidentally weakened.
Acquisitions are outside explicit unsafe blocks. Pointer conversions retain
separate unsafe scopes; this change does not establish their memory validity.
The token is zero-sized and acquisition adds no runtime feature detection.

## Dispatch and validation scope

`lib.rs` includes `safe_simd` only without `asm`; the affected functions require
`asm`. These wrappers and helpers are therefore dormant in every supported
build. In particular, the `not(asm)` Rust-wrapper assignments inside the
ASM-only `Rav1dMCDSPContext::init_x86` are unreachable. Active ASM dispatch uses
NASM. We do **not** strengthen its AVX2 gate to V3: that would unnecessarily
reduce CPU coverage. Checked Rust dispatch retains its existing token checks.

The new `tools/check-wrapper-contexts.py` explicitly handles this coverage gap.
It checks all 237 source acquisition statements, their annotations and suffixes
(including macro invocation names), then compiles their projected feature
contracts in an isolated crate. It checks 216 x86 contracts and 21 ARM contracts;
weakening every x86 annotation to V2 must produce an E0133 at every acquisition.
The projections copy the source annotation/statement contracts; they do not
compile the dormant raw-pointer bodies or prove their memory safety. The CI
`Wrapper feature contracts` job runs both architectures. Any future activation
of these wrappers needs whole-body compilation, pointer/extent tests, and
feature-admitted dispatch tests before use.

## Compatibility and release

The renamed functions are beneath `pub(crate) mod safe_simd`, not part of the
public Rust API. C exports and active dispatch tables are unchanged. No ownership,
const-constructor, runtime borrowing, or CPU-detection policy changes are made.

Archmage and archmage-macros are pinned to main revision
`7a67c74c569148e5c3470bc95538649dec60d8b4` (version 0.9.29). Neither the previous
lockfile's 0.9.26 nor published 0.9.28 provides `from_context()`. The lockfile
changes only those two package sources/versions and the macro crate's existing
syn selection. The minimum Rust version remains 1.89. Registry publication
requires publishing a compatible archmage release first, then using that
registry dependency. This PR does not publish either crate.

## Performance protocol

Compare immutable baseline consumers at `6bdd91e0` with the dependency migration
using `benchmarks/tracker-sharding-2026-09-07/build.py`. Build checked, unchecked,
and ASM separately, without dev-dependency feature unification. Keep the same
LLVM function alignment, compiler, input hashes, and harness. Run
`benchmarks/stills-2026-09-07/compare.py` on the existing 12 seed stills
(photo/map, 2K/4K/8K, minimum/8-column tiles), at one and eight workers, with
five rotated rounds and 150 ms upstream calibration. Visible outputs must match
pinned upstream rav1d and independent dav1d before and after timing; every timed
frame must be counted. This is a dependency regression screen, not the expanded
parity campaign or untouched-holdout acceptance test.

## Results

All **720 screen runs + 54 confirmation runs** matched both output references.
Candidate/baseline time ratios (smaller is faster), using paired medians per
cell and a geometric mean across the 24 cells:

| Mode | Mean time change | Per-cell median range |
|---|---:|---:|
| Checked | -0.43% | -1.62% to +0.74% |
| Unchecked | +0.23% | -4.59% to +4.27% |
| ASM | +0.01% | -3.05% to +2.18% |

**There is a reproduced unchecked regression in selected eight-worker cells.**
Nine-round confirmation with 500 ms upstream calibration, using final binaries:

| Input, eight workers | Screen change | Confirmation change |
|---|---:|---:|
| photo-2k-t8 | +3.73% | **+4.08%** |
| map-8k-min | +3.49% | **+2.46%** |
| map-8k-t8 | +4.27% | **+4.86%** |

All nine confirmation ratios exceed 1 for both map cells. The photographic
cell has one ratio below 1. No samples were discarded. This is an open
regression associated with the dependency update/build; code generation or
placement versus a dependency implementation change has not been isolated.
The wrapper changes themselves remain excluded from supported builds. Do not
call this update performance-neutral or claim parity from the overall mean.

For the new build, across the same 24 screen cells, unchecked takes **0.9854x**
checked time on average (cell medians 0.9123–1.0098x); ASM takes **0.7066x**
checked time (0.5779–0.8227x). These are current seed-still measurements, not
the older historical whole-video comparisons or expanded acceptance corpus.

Final consumers were rebuilt after the remaining dormant-helper annotations
and lint scoping. Their `.text` and `.rodata` sections match the measured
screen consumers byte-for-byte in all three modes. ASM `.data.rel.ro` also
matches; checked/unchecked differ only in 651 source-Location line fields
verified through the associated relative relocation and source-path string.
The confirmation uses the final unchecked binary directly.

## Checks

- 216 source-derived V3 contracts compile; all 216 weakened V2 contracts reject
  acquisition with E0133. All 21 ARM contracts compile. The final check seeds
  transitive versions from the decoder lockfile (syn 3.0.4).
- Release library suites: **80 checked, 69 unchecked, 22 ASM** tests pass.
- Debug committed-vector/crash/fuzz suites: **22** tests pass.
- Full MD5 conformance: **14 test groups** pass at one worker and again at
  eight workers, including the comprehensive corpus test.
- Checked and C-FFI library clippy with `-D warnings` pass; aarch64 default
  cross-compilation passes. ARM runtime execution remains delegated to CI.
- A Rust token-stream scan of `src` and `crates` finds zero forging identifiers.
  Formatting and whitespace checks pass.

The first clippy attempt failed because removing module-wide `allow(deprecated)`
exposed five existing zerocopy slice adapters. Their allowance is now local to
those five let-statements; the operations are unchanged. Both original failures
and passing reruns are retained. No module-wide token deprecation suppression
remains in the changed modules.

Evidence is in `artifacts/` with original/payload SHA256s and byte counts in
`ARTIFACTS.json`. Multi-part payloads are reconstructed by decompressing and
concatenating their numbered parts. `ratios.json` preserves every screen pair;
`unchecked-confirm-summary.json` preserves the longer confirmation samples.
