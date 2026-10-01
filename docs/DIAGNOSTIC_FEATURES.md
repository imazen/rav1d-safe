# Diagnostic feature policy and optimization history

All experimental Cargo features in the decoder and disjoint-mut use the `__`
prefix. They are opt-in diagnostics, not a stable public tuning API. No public
feature or default-feature dependency chain enables them. A double underscore
is a naming convention, not protection against Cargo feature unification:
check-disabling experiments continue to fail compilation deliberately.

The established public features keep their names: decoder bit depths, `asm`,
`partial_asm`, `untracked`, `c-ffi`, `dav1d-compat`, ARM ASM extensions, and
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

## Current feature inventory

Everything below exists at HEAD. Anything else that older records, scripts or
benchmark drivers name has been removed; see [Removed 2026-10](#removed-2026-10).

**rav1d-safe (33).** Public: `default`, `bitdepth_8`, `bitdepth_16`,
`untracked`, `c-ffi`, `dav1d-compat`, `partial_asm`, `asm`, `asm_arm64_dotprod`,
`asm_arm64_i8mm`, `asm_arm64_sve2`. Internal: `__probe_count`, `__probe_wide`,
`__probe_usage`, `__probe_sites`, `__probe_bounds`, `__probe_untracked`,
`__probe_tasktime`, `__probe_x86tier`, `__probe_owned_recon`, `__bisect`,
`__ablate`, `__simd_test`, `__simd_test_log`, `__lrvarcov`, `__lrpoison`,
`__pad_text`, `__pad_small`, `__pad_far`, `__pad2`, `__pad3`, `__pad4`,
`__test_induce_worker_panic`.

**rav1d-disjoint-mut (12).** Public: `default`, `std`, `aligned`, `pic-buf`,
`instrument`, `untracked`. Internal: `__bench`, `__probe_count`,
`__probe_wide`, `__probe_usage`, `__probe_sites`, `__probe_bounds`. The Loom
model is selected by `--cfg disjoint_mut_loom` (not a feature) and runs the
production tracker with 4 shards instead of 128.

The decoder's `__probe_untracked` is not an alias of its `untracked`: it turns
off only the disjoint-mut tracker and keeps the decoder's tracked-mode compact
copy path and frame-thread clamp (the "tracker-off" ceiling the benchmark drivers
build). `untracked` also switches to zero-copy in-place guards.

## Consolidation

45 formerly unprefixed decoder experiments are renamed to `__snake_case`, with
no old unprefixed aliases. Corresponding disjoint-mut names already follow that
scheme. [The complete rename map](../release/0.6.0/feature-renames.json) records
individual names. Families are:

- `__probe_*`: observations and controlled overrides.
- `__pad*`: binary-placement controls, distinct from actual optimizations.
- `__simd_test*`, `__test_induce_worker_panic`, `__lrpoison`, `__lrvarcov`,
  `__ablate`, `__probe_x86tier`: correctness/census diagnostics.

Removed no-ops (0.6.0): decoder `__lf_rect` and `__rows_rect`, disjoint-mut
`__rect_mut`. The measured winning rectangle paths remain unconditional. The
alternative tracker/guard policies were removed in 2026-10 (below).

Do not combine these into a feature that enables everything. Counting, changing
shard placement, forcing panics, and selecting different algorithms answer
different questions. An umbrella would confound measurements and can combine
incompatible experiments; `__probe_count` and `__probe_usage`, for example, are
a deliberate compile error together because `__probe_count` selects a different
tracker.

## Environment reads

These are all direct environment-variable reads in production library source:

| Variable | Required private feature | Ordinary build |
| --- | --- | --- |
| `RAV1D_OWNED_RECON` | `__probe_owned_recon` | Never read; owned reconstruction remains eligible, subject to its normal frame checks. |

`RAV1D_LF_HULL`, `RAV1D_LF_PERROW`, `RAV1D_LF_DOUBLE`, `RAV1D_RECT_HULL`,
`RAV1D_CDEF_DOUBLE` and `RAV1D_PIN_SHIFT` were read only by experiment features
removed in 2026-10; no build reads them now.

`__bisect` also gates `MC_SCALAR`, `IPRED_SCALAR`, `MCT_PREP_LOG` and
`RAV1D_LOG` (see its comment in `Cargo.toml`).

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

## Removed 2026-10

Finished A/B experiments whose winner already ships were purged: every
`cfg(feature = ...)` was resolved to its feature-off (shipped) arm, the other
arms were deleted, and the features left both manifests. A fingerprint of every
tracker constant, the block-shift / shard-mask rules over a 600-line input grid,
`BorrowId` encodings and live registration placement was byte-identical before
and after, in debug and release. To reproduce any historical arm, check out
**`087242f1`** (the last commit that has all of them) and build with the
feature named below.

| Group | Removed features (crate) | Shipped arm kept | Env var it armed |
| --- | --- | --- | --- |
| Single-lock A/B tracker | `__tracker_legacy` (both) | sharded tracker; the single-lock tracker survives only as `__probe_count`'s instrumented tracker (`tracker_count.rs`) | |
| Disabled unsound probes | `__probe_noscan`, `__probe_lockonly`, `__probe_tinynop`, `__probe_addnop` (both; already `compile_error!`) | full tracking | |
| Shard-sizing simulator | `__probe_shardsim` (both) | none (counter-only) | |
| Shard count ladder | `__shards_{1,4,8,16,32,64,128,256}` (both) | 128 shards (4 under `cfg(disjoint_mut_loom)`) | |
| Shard mapping | `__shard_ident` (both) | Fibonacci hash | |
| Fixed block-shift ladder | `__blockshift_{8,10,13,14,15,16}`, `__blockshift_adaptive` (both) | shift 12 serial / single-tile, adaptive for concurrent multi-tile | |
| Block-count granularity | `__bps_{quarter,half,1,4,8,blocks}` (both) | `BPS = (2, 1)` + derived rows rule on | |
| Rows-per-block ladder | `__rpb_{2,8,16}` (both) | `ROWS_PER_BLOCK_MIN = 4` | |
| Per-plane shift pin | `__probe_shiftpin` (both) | rule-derived shift | `RAV1D_PIN_SHIFT` |
| Shards per borrow | `__msb_5` (both) | `MAX_SHARDS_PER_BORROW = 4` | |
| One-shard rectangles | `__rect_1shard` (disjoint-mut), `__lf_rect1` (decoder) | multi-shard rectangle records | |
| Shard-lock waiting policy | `__probe_lock_{backoff,yield,relax,park}` (both), optional `parking_lot` dep of disjoint-mut | pure spin | |
| Loop-filter hull/per-row/double | `__probe_lf_hull` (decoder) | per-row (threaded) / hull (serial) reads | `RAV1D_LF_HULL`, `RAV1D_LF_PERROW`, `RAV1D_LF_DOUBLE` |
| Recon/MC hull | `__probe_rect_hull` (decoder) | per-row guards under tile threading | `RAV1D_RECT_HULL` |
| CDEF double registration | `__probe_cdef_double` (decoder) | single registration | `RAV1D_CDEF_DOUBLE` |
| Held row guards | `__held_row_guards` (decoder) | two-pass compact `BlockMut` | |
| Aliases | disjoint-mut `__probe_untracked` (use `untracked`), decoder `__probe_tasktime_untracked` (use `__probe_tasktime,__probe_untracked`) | | |
| No-op feature | disjoint-mut `zerocopy` (cast API is now unconditional) | | |

Benchmark records under `benchmarks/` and `audit/` keep their original feature
names and pinned drivers; they describe the revisions they were measured at.
The `scripts/perf/` drivers that only built removed arms (`bpsrows_*`,
`shardgran_*`, `shardsize_{build,gates,miri}`, `shard_sweep.sh`, `c256_*`,
`rect_gates.sh`) were deleted with them; mixed scripts had the removed arms
taken out of their lists.

## Validation of the 0.6.0 cleanup

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

## Benchmark tooling

Disjoint-mut's `__bench` enables Criterion 0.8.2 for the `lock_overhead`
benchmark. Run `cargo bench -p rav1d-disjoint-mut --features __bench --bench
lock_overhead` on Rust 1.86 or newer. It is optional so the crate and its const
API tests retain Rust 1.85 support. It adds no library environment overrides.
The decoder's current test/example/benchmark dependencies require Rust 1.93
because of zenavif-parse 0.6.2; the library itself retains Rust 1.89 support.
