# rav1d-safe 0.6.0 release candidate

Prepared 2026-09-08 UTC on PR #528. **Not published or release-ready yet.**
This record separates preparation checks from publication gates. The current
candidate retains checked safe SIMD, runtime borrow/bounds checks, and the
crate-wide `forbid(unsafe_code)` default. The diagnostic environment policy now requires explicit private features;
ordinary owned-reconstruction policy is fixed to its prior environment-unset behavior.

## Versions and migration

- **rav1d-safe 0.6.0**, upgrading from published 0.5.7. This needs a minor bump:
  managed `Result<T>` now defaults to `whereat::At<Error>`, `Error` gains
  `Cancelled`, and the old `simd_test` feature was renamed `__simd_test`.
  Callers matching managed errors use `err.error()` or `err.decompose().0`.
  Exhaustive matches must handle cancellation. Strict parsing is the default;
  `Strictness::Lenient` is available for dav1d-compatible acceptance behavior.
- **rav1d-disjoint-mut 0.3.2**, upgrading from published 0.3.1. It preserves
  `const fn new()` and `const is_checked()`; `new_eager()` and rectangle APIs
  are additive. No 0.4.0 is required for this soundness correction. Previously
  published 0.3.1 admits guard-move UB in a safe-client Miri reproducer.
  See the [abstraction protocol](RELEASE_SOUNDNESS_PROTOCOL.md) and
  [API/semver evidence](../audit/disjoint-032-current/README.md).
- **archmage and archmage-macros 0.9.29** now resolve from crates.io.
  Both normal and test dependencies use the registry release, including
  `Token::from_context()` and the unified `archmage::intrinsics` memory API.
  The direct `safe_unaligned_simd` dependency remains for its size-bound trait,
  which archmage does not re-export.

The [changelog](../CHANGELOG.md) records decoder correctness fixes, including
film-grain reservations and worker panic cleanup, ARM debug-overflow fixes,
exact write-back footprints, process-global threading-policy isolation, and
C allocator ownership. The [Rust workflow guide](RUST_CODEC_WORKFLOW.md)
contains working AV1 and full AVIF examples pinned to published versions.
Those 0.5.7 examples are not validation of the 0.6.0 package.

## Checks completed during preparation

- Existing runtime/documentation head `e233adcb` has 27 successful PR checks,
  including native platforms, conformance, wrapper feature contracts and
  C allocator Miri checks. These do not substitute for the separate
  disjoint-mut workflow or checks on a subsequently changed head.
- `cargo +1.89.0 check -p rav1d-safe --lib --locked` passes on x86_64 Linux.
  The manifest minimum remains 1.89; a new CI job makes it an ongoing gate.
- `cargo package -p rav1d-disjoint-mut --features aligned,pic-buf,zerocopy`
  builds and verifies 0.3.2 from source head `e233adcb`. Exact archive identity
  and command results are retained in [the preparation evidence](../release/0.6.0/preparation.json).
- `cargo publish -p rav1d-safe --dry-run` reaches registry resolution and
  previously failed because `archmage ^0.9.29` was absent. This historical
  blocker is resolved by the registry migration; it uploaded nothing.
- A rehearsal using only Cargo-selected source files reproduced missing NASM
  inputs. The package whitelist now includes `src/**/*.asm` and `src/**/*.S`,
  including the shared x86 include. The new source-package gate builds default,
  unchecked, C-FFI, ASM and partial ASM independently. It uses local/Git
  dependencies explicitly and is not a successful registry package dry run.
- The last disjoint-mut workflow's only failed job passed its seven Loom
  models, then failed because `rg` was not installed. The safety compiler
  diagnostic was present. The script now uses portable `grep -Fq`, without
  changing which errors qualify as successful rejection.

The fresh API check against 0.5.7 ran 196 checks: 194 passed and two reported
breaking changes (`Error::Cancelled` and removal of `simd_test`). It was run
with `--release-type minor` to expose breaks; that option means a logically
nonbreaking release, not a literal pre-1.0 minor-number increment. These known
breaks require **0.6.0**, not 0.5.8. This result is not a clean patch-compatibility
claim; the managed error-wrapper migration also remains documented above.

## Performance scope

The still optimizations are documented in
[the checkpoint](STILL_PARITY_CHECKPOINT.md). Safe SIMD checked mode takes
32.0%, 17.9%, and 18.7% less decode time than upstream **without assembly** at
1, 4, and 8 workers on the two-source 2K/4K/8K seed matrix. Upstream's Rust
implementation remains unsafe; this is not a comparison of two memory-safe
implementations or a claim of assembly-enabled upstream parity. The
[full report](../benchmarks/noasm-2026-09-08/README.md) records fixed A/B order,
warmed decoders, input limits and all 360 output-validated runs.

The archmage migration has confirmed unchecked regressions at eight workers:
photo-2k-t8 +4.08%, map-8k-min +2.46%, map-8k-t8 +4.86%. Checked and ASM aggregate
results do not erase these cells. The user explicitly accepted these slowdowns for release. They remain
disclosed and are no longer a release blocker. The broader performance-parity goal remains
unfinished; this candidate makes no parity or universal soundness-proof claim.

The fresh disjoint-mut patch-level API gate passes all 223 executed checks
against registry 0.3.1 (30 skipped). Native production-adapter tests, no-std,
doctests, strict Clippy, i686 and no-std WASM checks also pass.

Raw build/test logs, registry snapshots, the source-file inventory, and the
verified disjoint-mut `.crate` are preserved in a public, hash-pinned bundle:

```sh
python3 tools/fetch-benchmark-artifacts.py --manifest release/0.6.0/EVIDENCE.json
```

The [artifact manifest](../release/0.6.0/EVIDENCE.json) pins all 33 files.
No credentials are needed, and conflicting local files are refused.

## Remaining publication sequence

1. Diagnostic feature cleanup is complete: all experimental switches use `__`,
   three obsolete no-ops are removed, and runtime environment overrides require
   a private feature. See [the inventory and enforcement](DIAGNOSTIC_FEATURES.md).
   Published disjoint-mut `instrument` retains its name for patch compatibility.
2. Obtain successful full disjoint-mut platform/Miri/Loom CI on the candidate
   head, and successful decoder CI including the new package and MSRV jobs.
   The earlier interrupted local broad Miri run is not a passing result.
3. Publish rav1d-disjoint-mut 0.3.2 after its own gates. Archmage and
   archmage-macros 0.9.29 are already published.
4. Both archmage declarations now use crates.io and the local lockfile is
   refreshed. Full conformance and performance measurements on the registry
   candidate remain release validation; old measurements identify the Git pin.
5. Run `cargo publish --dry-run -p rav1d-safe` with registry dependencies and
   verify the resulting `.crate` in supported modes and Rust 1.89. Retain its
   SHA256, normalized manifest, included-file list and exact source revision.
6. Complete PR review; the measured unchecked slowdown is accepted. Squash merge keeps old raw
   evidence blobs out of main history. Build/verify the final merged revision,
   date the release notes, then publish/tag that same verified source.

A successful local source rehearsal cannot satisfy steps 3–5. No tag, GitHub
release, crate publication, dependency publication, or yank was performed.

## ARM package follow-up

The new ARM package job exposed a missing `src/arm/asm-offsets.h` and then
previously uncompiled ARM-ASM Rust paths. The header is restored from the pinned
upstream source and included in the package; 21 compile-time assertions parse
its numeric constants and check the actual target Rust layouts. Changing the
seed offset deliberately fails its intended assertion; restoring it passes.
The ARM-only aligned scratch buffers now state their 16-byte alignment and use
the current pointer-access API. Compact loop-filter helpers use their existing
scalar fallbacks when the safe-SIMD module is excluded by `asm`.

All five ARM package-source modes cross-compile with Clang. Native ARM ASM unit
and committed-vector execution now has its own CI matrix entry; cross-compilation
does not substitute for those runtime checks. [Commands and scope](../release/0.6.0/arm-package.json).

The [raw log bundle](../release/0.6.0/ARM_PACKAGE_EVIDENCE.json) is public and
hash-verified; restore it with `python3 tools/fetch-benchmark-artifacts.py
--manifest release/0.6.0/ARM_PACKAGE_EVIDENCE.json`.

## Registry archmage follow-up

Both dependency declarations and Cargo.lock now resolve crates.io 0.9.29.
Shared memory macros and selected existing kernels use `archmage::intrinsics`
on x86, ARM and WASM. Untouched ARM modules keep the equivalent direct
`safe_unaligned_simd` imports to stay within the PR size budget. The wrapper
checker supports registry manifests. Checked/unchecked/ASM checks, Rust 1.89,
ARM/WASM cross-checks, 237 wrapper contracts (including 216 weaker-context
rejections), the feature policy, and 22 selected runtime regressions pass.
See [commands and outcomes](../release/0.6.0/archmage-registry.json).

The new publish dry run gets past archmage and fails on the still-unpublished
`rav1d-disjoint-mut ^0.3.2`. No upload occurred. Memory wrappers re-export the
same implementations, but the published proc macros differ from the prior Git
pin; performance has not been remeasured for this candidate.

## ARM ASM unit-test gate follow-up

The ARM ASM library and packaged-source builds passed, but its first unit-test
build exposed `neon_parity` importing the excluded `safe_simd` module. Gate that
module with `not(feature = "asm")`, matching the implementation it tests. The
safe-SIMD ARM job continues to run those parity tests; the ASM job still runs
its library tests and committed-vector regressions.

On the fix, the exact CI nextest command (`--profile ci --no-default-features
--features asm,bitdepth_8,bitdepth_16 --release --lib`) passes 23 tests on x86.
`cargo check --tests --lib --target aarch64-unknown-linux-gnu` with the same
features and release profile passes with a staged ARM C sysroot. This is ARM
compile validation; native ARM runtime validation remains the CI job.

## Workspace dependency refresh

All direct external dependency requirements in both workspace crates now name
current stable crates.io releases. Cargo.lock refreshes compatible transitive
versions. This includes cc 1.4.5, zenbench 0.1.9, tango-bench 0.8.0,
zenavif-parse 0.6.2 and Criterion 0.8.2. Local disjoint-mut remains the reviewed
0.3.2 candidate. Historical benchmark and published-workflow snapshots retain
the dependencies they actually measured.

All-target release compilation, 22 selected runtime regressions, ARM ASM test
cross-compilation and feature-policy validation pass. The library still builds
on Rust 1.89 and disjoint-mut const API tests pass on Rust 1.85. Decoder dev
tooling now needs Rust 1.93; optional Criterion tooling needs Rust 1.86 and is
gated by `__bench`. See [dependency versions and checks](../release/0.6.0/dependency-update.json).

The preceding ARM ASM gate fix also passed native ARM CI job `asm-arm64` in
run 34195096629; that run predates this dependency refresh.

## Verified workspace packages

At source `1bcd6ce6`, `cargo package --workspace --features
rav1d-disjoint-mut/aligned,rav1d-disjoint-mut/pic-buf,rav1d-disjoint-mut/zerocopy`
creates and verifies both actual archives. Cargo stages disjoint-mut in a
temporary local registry; this resolves the local rehearsal dependency gap
without publishing it. A standalone registry publish still requires 0.3.2 to
be published first. Nothing was uploaded.

The disjoint-mut archive contains 22 files: 162,633 compressed bytes and
571,303 unpacked file bytes. The decoder contains 268 files: 2,190,640 compressed
bytes and 14,985,506 unpacked file bytes. Assembly accounts for 8,329,428 unpacked
bytes and safe SIMD Rust for 3,734,957. These are source archives, not linked
binary sizes or dependency downloads. [Sizes, hashes and largest files](../release/0.6.0/package-sizes.json)
identify these tested artifacts; subsequent documentation edits change archive
identity and require final packaging again at the release revision.
