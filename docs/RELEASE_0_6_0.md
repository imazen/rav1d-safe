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
- **archmage and archmage-macros 0.9.29** are required by the current pinned
  main revision `7a67c74c569148e5c3470bc95538649dec60d8b4`. Their latest registry
  releases at preparation time are 0.9.28. `Token::from_context()` requires
  the new version. Do not replace it with token forging to unblock publication.

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
  **fails** because `archmage ^0.9.29` is absent. It uploaded nothing.
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
3. Publish the reviewed archmage-macros/archmage dependency release in dependency
   order, and rav1d-disjoint-mut 0.3.2 after its own gates.
4. Replace both Git archmage declarations with registry requirements, refresh
   the lockfile, and rerun tests/conformance and the affected performance cells
   against the actual registry artifacts. Verify source equivalence to the pin.
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
