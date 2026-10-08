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
