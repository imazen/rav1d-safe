# Diagnostic feature policy and optimization history

All experimental Cargo features in the decoder and disjoint-mut use the `__`
prefix. They are opt-in diagnostics, not a stable public tuning API. No public
feature or default-feature dependency chain enables them. A double underscore
is a naming convention, not protection against Cargo feature unification:
check-disabling experiments continue to fail compilation deliberately.

The established public features keep their names: decoder bit depths, `asm`,
`partial_asm`, `unchecked`, `c-ffi`, `dav1d-compat`, ARM ASM extensions, and
storage/std adapters in disjoint-mut. Its published `instrument` feature also
keeps its name to preserve 0.3.x compatibility. This policy prefixes experiments,
not every supported feature in the crates.

## What this optimization journey added or used

| Facility | Provenance and use | Current name / decision |
| --- | --- | --- |
| Borrow-usage census | Added in `87d7edbf`: site/read-write/extent/shard/occupancy/construction counters in thread-local maps. Used across single-tile stills, tiled stills and multi-frame video. | `__probe_usage` in both crates; retained. |
| Wide-path and task timing | Pre-existing probes reused in a separate diagnostic binary for promotions, contention, stages and active-worker counts. | `__probe_wide`, `__probe_tasktime`; retained separately from usage maps. |
| Site/extent guards | Existing `probe-sites` reused for validation and extent budgets. Existing `probe-count` was deliberately avoided for production contention measurements because it changes the tracker selection. | `__probe_sites`, `__probe_bounds`, `__probe_count`; independent features remain. |
| Transform census / SIMD comparison | Existing `__ablate` supplied transform-shape counts with every SIMD family enabled. `__simd_test` and CPU/token permutation tests validated dispatch and output. | Existing prefixed names retained. |
| Entropy and loop-filter census | New temporary source instrumentation in isolated benchmark builds: per-decoder entropy counters and loop-filter mask/width counters. | Archived patches and instrumentation scripts, not new release Cargo features. |
| Checked / unchecked / ASM comparisons | Used existing supported modes; separate immutable consumers prevented dev-feature unification. | Supported names unchanged; these are not experimental switches. |
| Owned-reconstruction override | Older same-binary owned/copy experiment left an unconditional `RAV1D_OWNED_RECON` read. Found during this release cleanup. | New `__probe_owned_recon` is required to consult the variable. Ordinary builds always use the prior environment-unset policy. |

The [usage report](../audit/concurrency-profile/README.md),
[ownership experiment ledger](OWNERSHIP_MODELS.md),
[entropy record](../benchmarks/still-entropy-2026-09-07/README.md),
[transform record](../benchmarks/still-transforms-2026-09-07/README.md), and
[loop-filter census](../benchmarks/still-loopfilter-2026-09-07/README.md)
retain original names and exact historical commands. Those reports describe
recorded revisions; their names are not rewritten as though the old runs used
new flags. Executable current source, CI and consumer templates use the new
names. Historical scripts requiring removed no-ops run at their recorded commit.

## Consolidation

45 formerly unprefixed decoder experiments are renamed to `__snake_case`, with
no old unprefixed aliases. Corresponding disjoint-mut names already follow that
scheme. [The complete rename map](../release/0.6.0/feature-renames.json) records
individual names. Families are:

- `__probe_*`: observations, contention-wait comparisons and controlled overrides.
- `__tracker_legacy`, `__shards_*`, `__shard_ident`, `__blockshift_*`, `__bps_*`,
  `__rpb_*`, `__msb_5`, `__held_row_guards`: alternative tracker/guard policies.
- `__pad*`: binary-placement controls, distinct from actual optimizations.
- `__simd_test*`, `__test_induce_worker_panic`, `__lrpoison`, `__lrvarcov`,
  `__ablate`, `__probe_x86tier`: correctness/census diagnostics.

Removed no-ops: decoder `__lf_rect` and `__rows_rect`, disjoint-mut `__rect_mut`.
The measured winning rectangle paths remain unconditional. `__lf_rect1` still
selects a real one-shard-only experiment; it no longer depends on a no-op.

Do not combine these into a feature that enables everything. Counting, changing
shard placement, forcing panics, and selecting different algorithms answer
different questions. An umbrella would confound measurements and can combine
incompatible experiments. In particular, `__probe_untracked`, `__probe_noscan`,
`__probe_lockonly`, `__probe_tinynop`, and `__probe_addnop` remain compile errors.

## Environment reads

These are all direct environment-variable reads in production library source:

| Variable | Required private feature | Ordinary build |
| --- | --- | --- |
| `RAV1D_OWNED_RECON` | `__probe_owned_recon` | Never read; owned reconstruction remains eligible, subject to its normal frame checks. |
| `RAV1D_LF_HULL`, `RAV1D_LF_PERROW`, `RAV1D_LF_DOUBLE` | `__probe_lf_hull` | Never read; all overrides false. |
| `RAV1D_RECT_HULL` | `__probe_rect_hull` | Never read; override false. |
| `RAV1D_CDEF_DOUBLE` | `__probe_cdef_double` | Never read; override false. |
| `RAV1D_PIN_SHIFT` | disjoint-mut `__probe_shiftpin`, forwarded by decoder `__probe_shiftpin` | Never read; normal per-instance placement. |

Test-harness variables (`RAV1D_ROW_GUARD_CHILD`, `RAV1D_TEST_IVF`,
`RAV1D_TEST_EXPECTED_HASH`, Loom controls) live only under `cfg(test)` and are
absent from distributed library builds. Cargo/build-script environment,
benchmark-process configuration and dependency/standard-library internals are
outside this runtime-library policy. This is not a claim that the entire process
or Rust standard library never reads its environment.

`tools/check-feature-policy.py` checks feature naming and public/default feature
closures, then parses every Rust file in both libraries with syn. Direct env
reads/import aliases/macros must be excluded when `cfg(test)` and every `__`
feature are false. Unknown platform/public-feature conditions are treated
conservatively. A runtime `cfg!` branch does not satisfy the gate. The scanner
is a source-policy guard, not whole-program analysis of arbitrary macro
expansions or dependencies. Its adversarial tests challenge missing, inverted,
and bypassable `any` gates and imported aliases.

CI also runs the ordinary owned-reconstruction test with
`RAV1D_OWNED_RECON=0`: the variable must have no effect. Private diagnostic
builds retain their deliberate overrides. This change makes no performance
claim; the previously measured unchecked slowdown is accepted for release and
remains disclosed in the release notes.

## Validation of this cleanup

The ordinary and diagnostic owned-reconstruction tests pass with the override
set to zero. The default decoder regression fixtures, combined probe Clippy,
seven Loom models, all five rejected-feature controls, disjoint-mut's 223
patch-compatibility checks, a current diagnostic consumer, and all five
package-source build modes pass. Full new-head platform CI remains separate.

[Command records](../release/0.6.0/feature-validation.json) retain initial setup
failures as well as successful corrections. The
[31-file evidence bundle](../release/0.6.0/FEATURE_EVIDENCE.json) contains logs,
source inventory and the candidate patch against the recorded parent:

```sh
python3 tools/fetch-benchmark-artifacts.py --manifest release/0.6.0/FEATURE_EVIDENCE.json
```
