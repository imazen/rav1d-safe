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
