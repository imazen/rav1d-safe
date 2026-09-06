# Const-compatible 0.3.2 on the current tracker

This candidate preserves `const fn DisjointMut::new` while retaining the current
sharded tracker, exact rectangle APIs, pointer-based guards, storage fixes, and
feature-unification hardening. The earlier assumption that a separate older
tracker was needed for 0.3.2 was unnecessarily restrictive. Removing `const`
would require 0.4.0; deferring that removal avoids the break.

The [historical backport](../disjoint-032/README.md) and its package are retained
as evidence and a fallback. Its test results do not validate this candidate.

Candidate source commit: **`09109d91180ed1eb2ae16cff2793013533679b25`**. Local `main`
and `release/disjoint-mut-0.3` carry this source and its release evidence.

## Construction and safety

`new` creates a small const-initialized wrapper and initializes one boxed tracker
on first use. `spin::Once` publishes exactly one tracker to all concurrent first
callers. `is_checked` stays const and true before initialization. Cleanup and
guard drop can only retrieve the published tracker, never create another one.

The additive `new_eager` constructor allocates immediately. The decoder,
`Default`, and allocating slice constructors use it, avoiding the Once load on
each borrow. Both variants use the same admission/retirement algorithm. Neither
variant changes its allocation or shard mapping while usable guards exist.

The [release protocol](../../docs/RELEASE_SOUNDNESS_PROTOCOL.md#const-compatible-tracker-initialization-032)
records the initialization argument and its boundaries. In particular, Loom
checks the production record protocol after initialization; it does not
instrument spin::Once. The wrapper contains no new unsafe code. The `spin`
0.12.3 dependency is built with only `once`, supports no_std, and has an MSRV
below this crate's unchanged Rust 1.85 requirement. Native tests and Miri
exercise its publication, concurrent first use, const/static use, both
constructors, resize/stride transitions, moved buffers, poison, leaked guards,
and live capabilities across global threading-hint calls.

## API contract and completed checks

The [exact API diff](api.diff) retains const qualifiers and auto traits. Existing
function signatures and constructor constness are preserved. Rectangle guard
types/methods, borrowed mutable-slice storage, placement/probe functions, and
`new_eager` are additive. Four old automatic guard Send/Sync implementations
become explicit with equivalent bounds. Trait sealing already happened in
0.3.1 and is unchanged here. Unsafe adapter documentation spells out necessary
storage/provenance obligations; it adds no safe-client scheduling requirement.

All three [patch-level semver checks](semver.json) against the verified crates.io
0.3.1 archive pass: 223 checks passed and 30 skipped in each configuration
(default, no-std, and published storage/instrument features). The
[feature/dependency comparison](feature-contract.json) confirms that every
published feature definition and the Rust 1.85 requirement are retained. The
new normal dependency is spin; parking_lot is optional for an experimental
waiting-policy feature.

| Completed gate | Result |
| --- | --- |
| Native production adapters, all targets | 115 tests and 5 benchmark smoke cases pass |
| No-std with production adapters | 113 tests pass |
| Doctests | 2 runnable and 4 compile-fail examples pass; 2 pre-existing illustrative snippets ignored |
| Clippy | Production adapters with/without std pass with warnings denied |
| Published feature combinations | All 24 distinct combinations compile |
| Cross compilation | i686 with adapters and wasm32 without std pass |
| Loom | All 6 record-protocol models pass, 2-preemption bound, no permutation/time cap |
| Miri, each memory model | 54 focused API/guard/storage tests, 2 initialization tests, and 5 no-std constructor tests pass |
| Decoder regressions | 23 tests pass, plus task-context size (9,088 B, below 48 KiB) and same-process threading-policy regression |
| Package and MSRV | Package verification, Rust 1.85 const/static clients, and packaged storage builds with/without std pass |
| Published downstream | rav1d-safe 0.5.7 matches reference MD5 for 4 vectors at 1 and 4 threads using the packaged crate |
| Safe-feature boundary | All 5 enforcement-disabling features fail for the intended reason |

The broad Stacked Borrows [run](miri-stacked.log) was interrupted after
1,514 seconds in the unchanged 160,000-iteration concurrent-access stress test.
It is **not a passing full-suite result**. Earlier test targets completed
without a UB report; [miri.json](miri.json) records the interruption. The full
Miri CI matrix remains a publication gate. Focused checks of the changed
construction paths and guard/storage/rectangle behavior pass under both models,
including no-std const clients; see [the focused record](miri-focused.json).

## Reproducing the candidate checks

Run groups serially under the workspace heavy-job runner:

```sh
TMPDIR="$HOME/tmp" "$HOME/work/zen/scripts/run-heavy" --mem 16G --jobs 8 -- \
  python3 scripts/review-const-032.py native \
  --output "$HOME/tmp/rav1d-review-const-0.3.2"
```

Other groups are `miri` (full), `miri-focused`, `loom`, `semver`, `api`, `decoder`, `package`, and
`downstream`. The API and semver groups require `--baseline` pointing to the
checksum-verified published 0.3.1 source. Package only a clean, committed
candidate; downstream uses that package with published rav1d-safe 0.5.7.
Commands, flags, outcomes, elapsed times and log hashes are recorded separately.

No performance measurements were added for this change. The earlier bounded
performance investigation and [ownership experiment ledger](../../docs/OWNERSHIP_MODELS.md)
remain applicable as historical findings. Preserving the eager path is not a
new performance result.

## Package provenance and remaining publication gates

Archive: `target/package/rav1d-disjoint-mut-0.3.2.crate` in the primary repository.
SHA-256: `153fd574588f5754ffd340b34f73a13407e409a7203598629e0745fc8e25b46c`.
[Package provenance](package-provenance.json) checks all 18 packaged source,
manifest-original, license and documentation files against the candidate commit.
VCS metadata records that commit and no dirty flag. The normalized manifest
and lockfile are generated by Cargo. [Downstream resolution](downstream-resolution.json)
confirms the published decoder used this packaged 0.3.2 source.

API snapshots exceeding 30 KiB remain in
`~/tmp/rav1d-review-const-0.3.2/api-0.3.{1,2}.log`; hashes and commands are in
[api.json](api.json). All source hashes, tool versions, results and smaller logs
are retained here. The feature matrix is reproduced with the historical
`audit/disjoint-032/review.py matrix` runner, passing the primary repository as
`--candidate`, the verified 0.3.1 archive as `--baseline`, and the current output
directory. Its 24 production-feature combinations apply to both candidates.

Nothing has been pushed, tagged, published or yanked. This candidate still needs
the full Miri/platform CI and the repository's release approval/README review
and publication sequence. The partial broad Miri run does not satisfy that gate.
The existing published-crate negative controls and prior record-protocol mutation
evidence remain linked from the release protocol; they were not rerun as part of
this constructor correction. A 0.3.2 dependency update does not backport the
decoder's separate process-global threading-policy changes.
