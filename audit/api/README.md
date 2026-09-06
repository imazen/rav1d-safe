# Published API comparison

**0.3.2 maintenance update:** the 0.4 requirement below applies to the full
development implementation. A separate backport retains the published 0.3
tracker and const constructor while fixing guard references. Its patch-level
semver checks pass for default, no-std, and all published features. See the
[0.3.2 release evidence](../disjoint-032/README.md) for exact API differences,
candidate revision, and safety gates; the historical comparison below is kept
as evidence for the development line.

Generated 2026-09-06 UTC with cargo-public-api 0.52.0 and the installed nightly
toolchain. Inputs are the checksum-verified crates.io tarballs listed in
[release metadata](../../docs/review-releases-2026-09-05.json), including the
yanked decoder 0.5.6. Decoder 0.5.5 reports a dirty VCS tree, so comparing only
its nominal commit would not establish the published API.

The comparison omits blanket implementations but **retains auto traits and
`const` qualifiers**. Disjoint features: `std,aligned,pic-buf,zerocopy`.
Decoder features: defaults (`bitdepth_8,bitdepth_16`), on x86_64 Linux.
Full snapshots and build logs are in the review workspace's `api` directory;
[manifest.json](manifest.json) records commands, results, and snapshot hashes.
The compact exact diffs are retained here.

Cargo feature additions/removals are recorded separately in
[features.json](features.json); a rustdoc signature snapshot cannot show them.

| Comparison | Result |
| --- | --- |
| [Disjoint 0.2.1 → 0.3.0](rav1d-disjoint-mut-0.2.1-to-0.3.0.diff) | Removed `Aligned` re-exports in the selected feature surface |
| [Disjoint 0.3.0 → 0.3.1](rav1d-disjoint-mut-0.3.0-to-0.3.1.diff) | Sealed `DisjointMutIndex`, `SliceBounds`, and `TranslateRange`; external implementations cease to compile |
| [Disjoint 0.3.1 → current](rav1d-disjoint-mut-0.3.1-to-current.diff) | `new` loses `const`; adds borrowed mutable-slice storage, exact rectangle APIs, placement hints, and probe methods; guard thread bounds become explicit implementations |
| [Decoder 0.5.5 → 0.5.6](rav1d-safe-0.5.5-to-0.5.6.diff) | Adds `with_pixel_guard_immut` |
| Decoder 0.5.6 → 0.5.7 | Identical selected API snapshots; this does not imply identical implementation or soundness |
| [Decoder 0.5.7 → current](rav1d-safe-0.5.7-to-current.diff) | Adds strictness, cancellation, block helpers, and exposed experiment helpers |

The disjoint 0.3.1-to-main semver check failed specifically for
`inherent_method_const_removed` ([log](semver-disjoint.log)). The local release
version and decoder dependency are therefore prepared as **0.4.0**. Runtime
initialization replaces const/static construction, for example:

```rust
use std::sync::LazyLock;
use rav1d_disjoint_mut::DisjointMut;
static BUFFER: LazyLock<DisjointMut<[u8; 64]>> =
    LazyLock::new(|| DisjointMut::new([0; 64]));
```

Decoder `Settings` gained a public `strictness` field. It was already
`#[non_exhaustive]` in 0.5.7, so this field addition is compatible; outside the
crate, initialize with `Settings::default()` and assign fields. `Error::Cancelled`
adds a case to exhaustive error matches. The Cargo feature `simd_test` was
removed; the current experimental spelling is `__simd_test`.
`Decoder::set_stop` and the stop
traits are additive. Current defaults request strict decoding; that behavioral
change is absent from a signature diff. Current decoder version 0.6.0 is
consistent with a breaking pre-1.0 release.

The decoder check deliberately overrode the release type to `minor` to inventory
breaks that the real 0.6.0 version bump permits. It reported the new exhaustive
error variant and removed Cargo feature; the text diff also records the
compatible settings field. The disjoint check
with its corrected 0.4.0 version passed. These are compatibility checks, not
soundness proofs.

The auto-trait text changes for existing disjoint guards do not themselves
establish a changed `Send`/`Sync` contract: old automatic bounds and new explicit
bounds must be compared semantically. The checker found the `const` removal,
not a `Send`/`Sync` regression, on the selected feature surface. Negative
compile examples and Miri separately challenge guard use.

Several methods described as probes are actually publicly reachable in the
default API. Their names/comments do not remove compatibility obligations.
The five enforcement-disabling disjoint features now fail compilation;
experiments requiring them must use their historical source revision.

To regenerate against already-extracted, verified releases:

```sh
TMPDIR="$HOME/tmp" "$HOME/work/zen/scripts/run-heavy" --mem 16G --jobs 8 -- \
  python3 scripts/review-api.py \
  --releases "$HOME/tmp/rav1d-review-2026-09-05/releases" \
  --output "$HOME/tmp/rav1d-review-2026-09-05/api"
```

This inventory covers the stated features and target, not C ABI compatibility,
every target-specific API, all experimental features, or behavioral equivalence.
