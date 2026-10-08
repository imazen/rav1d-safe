# Pre-optimization byte MC source-row repair

Missing: production x86 source repair, full decoder and sidecar gates, cross
checks and matched performance binaries. The independent negative control
fails and the unchanged oracle passes after repair. No speed claim is made.

The original V-only eight-tap put helpers accept a source suffix and convert
the signed byte pitch to an unsigned row offset. A suffix cannot contain the
earlier addresses needed when rows walk backwards. The repair passes the
complete bounded source and a separate tap-zero base to both AVX2 and AVX-512
helpers; vector loads and scalar tails calculate signed row addresses within
that slice. Coefficients, rounding, SIMD arithmetic and the dispatch gate stay
unchanged. The separate MC optimization uses the same source-address repair.

The oracle uses the actual reference-window guard, nine eight-tap filter
combinations, all 256 phase pairs and five shapes through 128x128. It compares
whole put/prep outputs with scalar for direct AVX2, available AVX-512 and
production dispatch, while holding an unrelated picture-row write. This
fixture is default-only and establishes no negative-stride C-FFI coverage.
Run `just test-mc-source-strides` before paired performance builds.

The pre-optimization code fails at `mc.rs:3100:25`, indexing
18446744073709551424 into a two-byte suffix. The signed-address repair
passes the unchanged oracle in 5.361 seconds. Both wrapped scopes took
19 seconds; the failing run peaked at 1.51 GiB RSS and the passing run
at 1.52 GiB. [Raw before/after logs and source hashes](../benchmarks/mc_baseline_source_2026-10-08.meta.json)
keep this correctness result separate from the pending MC optimization.

The final matched before source can be reconstructed from the
[recorded repair patch and independent oracle](../benchmarks/mc_final_baseline_repair_2026-10-08/meta.json).
The patch starts at MC SHA `c3a73a1f` and produces `a4538140`; the complete
hashes and fixture placement are in that record. This preserves the exact
source used for the fresh Rust 1.99 / archmage 0.9.30 benchmark baseline,
without adopting the optimization candidate or claiming a speed benefit.
