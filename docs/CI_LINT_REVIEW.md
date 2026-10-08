# CI lint review (2026-10-08)

The push workflow is active. Use an explicit repository when inspecting runs:
`gh run list --repo imazen/rav1d-safe`. An inherited fork default can show a
different workflow history.

Rust 1.99 CI reported six library errors: the deprecated compatibility field
in `Settings::default`, four constant-size chunk loops, and a manual ceiling
division. The default initializer retains the compatibility field with a
scoped deprecation allowance. Four-element chunks use `as_chunks::<4>().0`,
which preserves the old treatment of trailing elements. Positive kernel
heights use `div_ceil(8)`.

The prepared tree passes release all-target clippy on Rust 1.98.1. It also
passes ARM, WASM and C-FFI compilation and both isolated and eight-thread
library runs: 115 tests in each run, no failures or ignored tests.
`run-heavy`: rc=0, 42s, peak-RSS 1.50GiB, min-avail 25169MiB,
peak-load 4.82. The follow-up on Rust 1.99 also passes these gates, untracked all-target
clippy, library clippy with C-FFI and `__probe_sites`, and a Rust 1.89
library compile check. Its scope completed in 87s with peak-RSS 1.61GiB,
min-avail 24800MiB and peak-load 4.69. WASM's unused import is removed;
its unused compact-window helper has the same allowance as other non-x86
builds. Panic helpers used only by tracked tests are cfg-gated accordingly.
Remote CI remains the platform gate.

The Linux i686 container recipe passes 60 release tests (38 library and 22
committed/crash/fuzz tests) and repeats the 22 regression tests with dev
overflow checks. A repeat after architecture-gating unused test helpers
passes without warnings. [Commands, raw logs and container limits](../benchmarks/i686_cross_2026-10-08.meta.json)
record both runs; wrapper RSS excludes the daemon-owned container.
The CI leg complements the existing native i686 nextest leg. The library's
Intel macOS runner is `macos-26-intel`; Windows ARM remains covered.
Checkout v7 and Codecov v7 match the current official major releases checked
on 2026-10-08. The revised remote matrix passed all 32 jobs at `83c928d3`, as linked below.

The first revised workflow was rejected before jobs started: `runner.temp`
is unavailable in job-level environment expressions. Container options now
live in the test step's environment. Actionlint 1.7.12 validates the workflow;
`just lint-ci-workflow` retains that check for future expression edits.

`just test-integration` now uses release mode and calls the explicit corpus
recipe. `just test-integration-selection <directory>` selects that path and
runs all nine bodies, including the existing eight ignored tests without
changing their attributes. Missing files/directories, empty eligible sweeps,
and a stream that produces no HDR frame fail the gate. The existing corpus
passes 9/9; an absent selected corpus fails 9/9. All-target clippy passes.
[Raw positive/negative results](../benchmarks/integration_selection_2026-10-08.meta.json)
record a 15s scope, peak-RSS 0.97GiB, min-avail 25346MiB and peak-load 0.73.
The conformance workflow calls the explicit selection on both architectures.

The updated main revision `83c928d3` passed every one of 32 jobs in the
[2026-10-08 CI run](https://github.com/imazen/rav1d-safe/actions/runs/37721109550).
This includes Windows ARM, macOS Intel, native and container i686 tests,
both conformance architectures, token permutations, and both allocator
Miri models. Native ARM all-target lint is a separate local recipe gate.

Both native ARM release all-target recipes now pass on Rust 1.99.0. Seventy-five
unused scalar ITX reference items have non-ASM test allowances scoped to
individual items; new unused helpers and assembly-wrapper configurations keep
dead-code linting. One unused test variable is removed. The MC reservation test
module is compiled only where one of its existing tests is compiled; this
removes empty-module imports on ARM untracked builds without dropping tests.
C-FFI and ASM compile checks also pass; ASM still emits existing warnings.
[Commands, full logs and source hashes](../benchmarks/arm_lint_feature_2026-10-08.meta.json)
record the 67s native scope: peak-RSS 0.70GiB, min-avail 26612MiB, peak-load 7.99.
The x86 lint recipes, ARM/WASM/C-FFI compile checks and formatting check also
pass after these edits; their 24s scope peaked at 0.97GiB RSS.

For the complete sidecar matrix, `just conformance-all-modes <tracked-binary>
<untracked-binary>` runs each native runtime tier at 1/2/4/8 workers, delay 0,
with an expected 803-vector selection and stop-on-first-failure. The caller
selects both binaries; this command does not infer successful coverage from
nextest or silently build a different mode. The existing five runner boundary
tests and recipe expansion pass. Execution results belong with each candidate.

The later lead-owned main revision `a34a4233` also passed all 32 jobs in
[the complete CI run](https://github.com/imazen/rav1d-safe/actions/runs/37728965312),
verified on 2026-10-08. The separate Disjoint workflow at `a076772f`
passed all 14 jobs, including Stacked and Tree Borrows Miri. Native decoder
experiments remain separate from these published-revision checks.

The first native all-tier/all-worker sidecar invocation exposed a TSV-shape
regression before any decode: OSS-fuzz sanitizer arrays emitted six fields
although the writer required seven. Those rows are excluded by the default
caller, but still must serialize successfully. Their empty `extra_args`
field is now explicit. A new end-to-end default-extraction test fails before
the repair and passes afterward alongside all five existing runner tests;
it also checks operating-point/frame-type/limit flags survive extraction.
[Failure, restored gate and native build proof](../benchmarks/conformance_extractor_2026-10-08.meta.json)
record the distinction. The separate full sidecar matrix remains pending.

The dedicated `Conformance runner protocol` CI job now runs all six boundary
tests on pushes, including the default extraction path. Actionlint 1.7.12
accepts the updated workflow. This adds one job to the previously green
32-job matrix. All 33 jobs pass at `ae6c26d8` in the
[complete CI run](https://github.com/imazen/rav1d-safe/actions/runs/37732635299).
[The captured result](../benchmarks/ci_main_2026-10-08.json) identifies every job
and the full revision. `just ci-status <run-id>` queries this repository
explicitly. `just build-bench-modes <tracked-target> <untracked-target>` expands
the paired generic release commands; use fresh target directories to preserve
reference binaries. Its command expansion is checked; each future candidate's
actual builds and timing need their own evidence.

ARM CPU masks do not disable every baseline NEON dispatcher. The x86 mask
also leaves plain msac incants and autoversioned coefficient decoding free
to select higher tokens. CPU-mask labels therefore do not prove fallback
tier coverage. See [the existing ARM applicability review](X64_APPLICABILITY.md#a6-cpulevelscalar-does-not-disable-safe-simd-measurement-infrastructure-gap).

The `decode_md5` example now holds archmage's token-testing lock and caps
x86 tokens for scalar/v2/v3/v4. On ARM scalar selection it disables the NEON
token and its descendants. Guards remain alive until decoder workers join;
pre-decode assertions check the requested cap. Production managed CPU-mask
behavior remains unchanged.

`just test-conformance-token-tiers` passes natively on x86 and ARM. Removing
token disabling makes each unchanged worker oracle fail; exact restoration
passes again. CI calls this example gate on both conformance architectures.
Release all-target clippy passes in both modes, and ARM cross-check passes.
[Guard, mutation, lint and build evidence](../benchmarks/conformance_token_guards_2026-10-08.meta.json)
records the x86 13/12/12-second scopes (1.29/1.29/1.30GiB peak RSS), 14-second
lint scope (0.96GiB) and 127-second ARM binary build (0.91GiB).
[Native ARM guard evidence](../benchmarks/arm_token_guard_2026-10-08.meta.json)
records the independent four-worker check.

`just conformance-scalar-modes <tracked-binary> <untracked-binary>` selects
803 sidecars at 1/2/4/8 workers in both modes. Those enforced ARM scalar legs
are running. Earlier [CPU-mask selections](../benchmarks/arm_cpu_mask_sidecars_2026-10-08.meta.json)
pass24/24 legs but establish no fully scalar ARM sidecar claim. New x86
sidecar executions remain candidate-specific work.

A failed sidecar invocation now prints the decoder exit status and the unchanged
120-second deadline. Fake-decoder controls for exit 17 and timeout status 124
fail before this diagnostic change; all seven runner boundary tests pass
afterward. [Before/after logs](../benchmarks/conformance_exit_diagnostics_2026-10-08.meta.json)
make a timeout distinguishable from a reported pixel mismatch.

Published revision `770fc6d6` passed all 33 jobs in the
[complete CI run](https://github.com/imazen/rav1d-safe/actions/runs/37738399273),
verified on 2026-10-08. The ARM signed-source repair subsequently landed in
`507890b7` after its separate full decoder, CPU-mask sidecar and enforced
scalar gates passed; see [the ARM source-row review](ARM_MC_SOURCE_STRIDES.md).

The `decode_md5` frame limit now stops packet submission and frame draining,
rather than only limiting hashes. A committed valid frame followed by a
malformed second IVF packet proves the old behavior fails and limits zero/one
stop correctly. Both ARM modes pass that gate; the original scalar two-worker
300-frame sidecar matches its MD5 under the unchanged 120-second deadline.
`just test-conformance-token-tiers` now also builds the example and runs
`just test-decode-md5-limit`, so both existing CI architecture steps exercise
the real command. The x86 worker/CLI gates, both-mode all-target lint and ARM
compile pass in a 42-second scope, peak-RSS 1.29 GiB, min-avail 24645 MiB,
peak-load 0.93. [Before/after full logs and binary hashes](../benchmarks/decode_md5_limit_2026-10-08.meta.json)
record the failure and repair; the full enforced ARM scalar rerun is separate.

The repeated CPU-mask descriptions in `Settings::cpu_level`, the core flag
comment, the profiling example label and the overlap-test header also identify
pixel-dispatch selection. Token dispatch and compiler vectorization remain
independent of that mask. No settings or decoder behavior changes accompany
these documentation corrections.

Frame-threading descriptions now match `src/lib.rs::get_num_threads` and
`Settings::effective_frame_delay`: explicit delays greater than one enable
tracked frame contexts, and managed auto delay selects two when workers are
requested. Existing historical comparison values and release entries remain
as recorded; the old unconditional-clamp statement is labeled by its run.
The [full native ARM gate](../benchmarks/arm_decoder_full_2026-10-08.meta.json)
includes generated threaded-vector and backpressure tests in both modes.

Published revision `189acd88`, including that ARM repair and the guarded
benchmark example, passed all 33 jobs in the
[complete CI run](https://github.com/imazen/rav1d-safe/actions/runs/37750574432).
[Job identities, outcomes and timestamps](../benchmarks/ci_guarded_builds_2026-10-08.json)
record the 2026-10-08 observation, including both conformance architectures,
both allocator Miri models, Windows ARM, macOS Intel and native/container i686.
Private MC, CDEF and glue experiments remain outside this CI result.

The completed archmage comparison and its reporting/output-check changes
also have [all 33 CI jobs green](../benchmarks/archmage_comparison_ci_2026-10-08.json)
on ebceb0ca, run 37756911363. Both architecture permutation jobs complete,
as do Windows ARM, macOS Intel and i686 native/cross tests. This covers the
published benchmark tools and existing decoder source; the private MC,
CDEF, retained-row and initialization candidates remain separate gates.

`just test-threading-races` now accepts an explicit feature list, retaining
its existing default. `just test-threading-races
bitdepth_8,bitdepth_16,untracked` exercises the same concurrent-MD5,
three overlap and two worker-panic checks in the fast mode. The candidate's
[both-mode selected gate](../benchmarks/mc_feature_gates_2026-10-08/meta.json)
passes all six checks per mode with these commands; source optimizations
are still unpublished. No test expectation or selection is relaxed.

Reviewed documentation revision `01cf4826` passes all 33 jobs in
[CI run 37766267699](https://github.com/imazen/rav1d-safe/actions/runs/37766267699).
[Job outcomes, timestamps and API-capture hash](../benchmarks/ci_lead_docs_2026-10-08.json)
record both completed corpus-permutation architectures, Windows ARM, macOS
Intel, native/container i686 and both allocator Miri models. The earlier
MC build-record run completed 31 jobs before two unfinished corpus jobs were
cancelled by the subsequent push; it is not an all-green record. This complete
run covers the published decoder and tools. Private MC, CDEF, retained-row
and glue candidates still require their separate validation and timing.

Published initial MC passes all 33 CI jobs on revision `96dcf44a`.
The [complete job record](../benchmarks/ci_mc_landing_2026-10-08.json)
includes Windows ARM, macOS Intel and i686 and links the successful run.
This coverage does not include the unpublished signed warp destination
repair or the remaining performance experiments.

The subsequent current-head run on `bba380f5` also passes all 33 jobs.
Its [complete platform record](../benchmarks/ci_mc_evidence_2026-10-08.json)
verifies the published MC source plus the newly committed warp investigation
evidence. The warp repair itself remains unpublished and is excluded from
this green run.
