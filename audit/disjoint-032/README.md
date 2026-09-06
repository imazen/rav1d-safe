# Historical 0.3.2 backport candidate

The preferred release route now preserves const construction on the current
sharded implementation; see [current 0.3.2 evidence](../disjoint-032-current/README.md).
The candidate, logs and package below are retained as historical evidence for
the earlier conservative backport. They do not validate the newer candidate.

Historical candidate: **`fe45fd6c4af2accba0a61646350a2d96c4a736d2`**, local branch
`audit/disjoint-mut-032-backport` (originally `release/disjoint-mut-0.3`).
It starts at the published 0.3.1 revision
`dd60e0a61d88121fa764094f71d1d1de537f1f9f`; all baseline Rust implementation
files were byte-identical to the verified crates.io archive before editing.
The full [release protocol](RELEASE-0.3.2.md) is copied from the candidate.

**0.4.0 is not required for the soundness fix.** It was required for publishing
the development implementation unchanged, because `new` lost `const`.
The maintenance release preserves the original const-initialized tracker and
backports the guard fix. It does not require a new lazy-initialization scheme.

## Exact API contract

The all-feature [public API diff](api.diff), retaining auto traits and const
qualifiers, changes only four `Send`/`Sync` implementation spellings. Previously
automatic implementations become explicit with equivalent bounds. No public
functions, methods, constants, traits or feature names are added or removed.
Lifetime/variance and the existing const constructor are retained.

Each of the [default](semver-default.log), [no-std](semver-no-std.log), and
[all-feature](semver-all.log) patch-level checks against published 0.3.1 reports
**223 checks passed, 30 skipped, no semver update required**. The [feature and
dependency comparison](feature-contract.json) confirms identical published
features/defaults, normal dependencies, and Rust 1.85 requirement. All 24
distinct feature combinations compile. The model-only Loom dependency is
selected by a dedicated non-feature cfg.

The unsafe storage contract now explicitly states stable identity/provenance,
initialized elements, one authority over aliases, and correct thread traits.
Those are necessary obligations for an unsafe adapter, not new safe-client
scheduling requirements. Index/range trait sealing happened in **0.3.1** and
remains unchanged. Historical changelog claims that every excluded external
implementation was necessarily unsound were corrected: sealing can also break
correct implementations.

Full API snapshots (33 KiB each) and their stderr are retained at
`~/tmp/rav1d-review-0.3.2/api-0.3.{1,2}.log`, with commands and hashes in
[api.json](api.json). The compact exact diff is committed here.

## Safety and compatibility evidence

| Gate | Result |
| --- | --- |
| All published features, all test targets | 54 tests pass; 5 benchmark smoke cases pass |
| No-std | 35 tests pass |
| No-std with all storage adapters | 53 tests pass |
| Doctests | 1 runnable example and 4 compile-fail examples pass; 2 pre-existing illustrative snippets ignored |
| Miri, Stacked Borrows | Complete 54-test all-feature suite and 5 no-std API tests pass |
| Miri, Tree Borrows | Complete 54-test all-feature suite and 5 no-std API tests pass |
| Loom | 3 models pass: inline, real-capacity overflow, shared-reader retirement |
| Published negative controls | Both guard kinds report UB under both Miri models |
| Deliberate missing mutable-overlap scan | Loom reports `currently writing to cell` |
| Cargo package | Builds and verifies with all features |
| Packaged source, Rust 1.85 | All-feature and no-std checks pass |
| Cross-target compilation | i686 with all features; wasm32 without std pass |
| Published decoder 0.5.7 + packaged patch | 4 committed vectors match reference MD5 at 1 and 4 threads |

The new adversarial tests exercise const/static use across threads, moved
guards, an independent interval oracle, all four cast paths, exhausted slots,
overflow reuse, forgotten guards, zero-sized elements and invalid ranges.
The scheduler adjustment to the guard-move stress test retains all attempt
counts and liveness thresholds. The same adjusted tests still fail against
0.3.1 for the original aliasing defect in all four negative-control runs.

The no-std adapter matrix found an existing compile failure: `aligned` used
`std::collections::TryReserveError` in three return paths. Switching to its
`alloc` definition preserves the type. That correction is also applied to the
development line, where the no-std/storage clippy check passes.

Only overlap diagnostics move outside the instance lock; successful
admission/retirement retain the original single-lock record protocol. Loom
checks that production algorithm with instrumented metadata and payload,
abstracting native spin waiting as an Acquire/Release mutex. Its default
preemption bound is two with no permutation/time cap. This is conditional
soundness evidence, not a formal proof of the entire Rust abstraction or AV1
decoder. In particular, the dependency patch does not backport the decoder's
separate process-global threading-policy changes.

Gate commands, outcomes, flags and log hashes are in the JSON files alongside
the logs. The [mutation record](mutation.json) verifies that the candidate
checkout remained unchanged; the defect was planted in a separate copy.
[downstream-resolution.json](downstream-resolution.json) confirms the reverse
dependency used registry rav1d-safe 0.5.7 and the packaged disjoint-mut 0.3.2.
The downstream fixture was subsequently rustfmt-formatted without semantic
changes. `review.py` and `mutation.py` reproduce the independent checks; the
maintenance branch's `scripts/review-disjoint-032.py` runs its release gates.

## Package and release state

Archive: `~/tmp/rav1d-review-0.3.2/package-checkout/target/package/rav1d-disjoint-mut-0.3.2.crate`.
SHA-256: `bcda843df1c804237d6bbb508ba48aff4b8584398acb7f3ccc19f4d736446f74`.
[Package provenance](package-provenance.json) verifies 14 packaged source,
test, manifest and documentation files byte-for-byte against the candidate
commit. Its VCS metadata records that exact commit and no dirty flag.
Packaging used an isolated local checkout so it could not accidentally record
the enclosing zen-workspace repository as its origin.

Nothing has been pushed, tagged, published or yanked. Before publishing, the
maintenance revision needs the declared platform CI and release sequence in
the protocol. After publication, `^0.3` users can resolve 0.3.2; existing
lockfiles require an update, and exact `=0.3.1` pins require changing the pin.
The newer tracker and APIs remain on the development line.
