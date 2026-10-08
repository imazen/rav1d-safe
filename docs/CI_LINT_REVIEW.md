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
on 2026-10-08. The revised remote matrix remains pending until its run finishes.

The first revised workflow was rejected before jobs started: `runner.temp`
is unavailable in job-level environment expressions. Container options now
live in the test step's environment. Actionlint 1.7.12 validates the workflow;
`just lint-ci-workflow` retains that check for future expression edits.
