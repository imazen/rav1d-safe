# Archmage 0.9.30 update review

The native ARM full suites, doctests and explicit integration selection are
green with the signed-source repair. Its separate CPU-mask
sidecar matrix passes 24/24 legs. The explicit token-disabled scalar
supplement passes all eight legs, each 803/803 vectors at 1/2/4/8 workers
in both modes ([evidence](../benchmarks/arm_scalar_sidecars_2026-10-08.meta.json)). The updated remote CI matrix
passes all 33 jobs on `ae6c26d8`: [CI run](https://github.com/imazen/rav1d-safe/actions/runs/37732635299).
The manifest selects full revision
`e2dbab66ef5aa08f8e23ed05248e7d1217f58475` for both normal and dev
archmage dependencies. The previous pin was
`b8d6c0777e55e4ea8d26cde0879faae858cba662` (0.9.29).

This is a 63-commit update, including macro emission and dispatch changes,
not only a token-detection fix. Preserve same-pin benchmark comparisons
when assessing subsequent kernel changes.

The generated cold token detection publishes its result with
compare-and-exchange instead of an unconditional store and rechecks the
disabled flag. This prevents an in-flight detection from persistently
resurrecting a disabled token. The repaired race oracle checks summons
after disable returns in the toggler; the concurrent summoner supplies
pressure and confirms an enabled window. Observations across an in-progress
disable are not an oracle for a persistent stale-cache update.

`#[arcane]` preserves an input `unsafe fn` on its generated sibling and
nested body; autoversion variants preserve it too. Safe inputs remain safe.
`#[track_caller]` follows both halves and autoversion variants. Body lint
expectations remain on the body; forwarding wrappers receive corresponding
allowances. Windows raw snapshot comparison normalizes CRLF to LF.

Verified upstream on 2026-10-08: the full-SHA CI and publication workflows
passed. These upstream results do not substitute for decoder gates.

- [Detection fix](https://github.com/imazen/archmage/commit/5abd4f662146)
- [Race oracle repair](https://github.com/imazen/archmage/commit/d784c272)
- [Unsafe sibling and body attributes](https://github.com/imazen/archmage/commit/707d7c17)
- [Windows snapshots](https://github.com/imazen/archmage/commit/ed3e8508)
- [Complete update](https://github.com/imazen/archmage/compare/b8d6c0777e55e4ea8d26cde0879faae858cba662...e2dbab66ef5aa08f8e23ed05248e7d1217f58475)
- [CI](https://github.com/imazen/archmage/actions/runs/37676492882)
- [Publication](https://github.com/imazen/archmage/actions/runs/37676524132)

The full x86 release suite passed 232/232 selected tracked tests (18 existing
ignored) with Rust 1.98.1 and 215/215 untracked tests (18 existing ignored)
with Rust 1.99.0. Both doctest runs passed ten tests, with thirteen existing
ignored. The compiler changed after the tracked executable was built;
previous benchmark binaries remain preserved. [Raw logs and provenance](../benchmarks/archmage_downstream_2026-10-08.meta.json)
record the phases. The combined scope completed in 1868s with peak-RSS
1.51GiB, min-avail 17590MiB and peak-load 15.72.

The compatible resolver update also advances shared syn from 3.0.5 to 3.0.6.
Rust 1.99 release all-target clippy passes in both modes; C-FFI and
`__probe_sites` library clippy pass too. ARM, WASM and C-FFI compile checks
and both library runners (115/115 each) pass on current stable. Rust 1.89
compiles the library. [Follow-up logs and commands](../benchmarks/archmage_stable_msrv_2026-10-08.meta.json)
record an 87s scope, peak-RSS 1.61GiB, min-avail 24800MiB and peak-load 4.69.
Native Neoverse-N1 execution with Rust 1.99.0 passes 226/226 tracked tests
and 208/208 untracked tests, with 26 existing skips per mode. Ten active
doctests pass per mode (three existing ignored), and both explicit legacy
integration selections pass all nine bodies without skips.
[Complete logs and source catalog](../benchmarks/arm_decoder_full_2026-10-08.meta.json)
record the 2856-second scope: peak-RSS 1.05 GiB, min-avail 23306 MiB,
peak-load 17.37. The catalog identifies all 262 runtime/test/build source files for that
recorded run. Later conformance token/limit tool changes have their own
[validation and hashes](CI_LINT_REVIEW.md). The signed-source repair also passes the separate CPU-mask and enforced
scalar sidecar matrices described above. No decoder performance benefit is
claimed for this dependency update.
