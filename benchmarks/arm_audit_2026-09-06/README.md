# ARM decode tier audit, 2026-09-06

Coverage is five IVF fixtures / 52 decoded frames, not the complete AV1 corpus or every kernel. No decoder algorithm changed. These measurements do not cover x86 or multi-thread scaling.

Apple M4 Pro, macOS Darwin 25.5, 24 GiB RAM; rustc 1.98 / LLVM 22. Decoder source: `26acb2ba45ab42aa61fa7f940fc713fcddbcfe1b`, with the accompanying benchmark rewrite. Default features, one decoder thread, film grain enabled, no target-cpu=native. The command was `CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR=/Users/lilith/tmp nice -n 19 cargo bench --locked -p rav1d-safe --bench tier_isolation -- --format=llm`. No memory usage claim is made.

The benchmark now uses zenbench's interleaved comparison. Token switching and exact comparison of every YUV plane happen outside timing. Decode and flush failures fail the run. The previous ARM early return prevented measurement; the current decoder successfully exercises its scalar fallback with NeonToken disabled.

| Fixture | Frames | NEON mean (ms) | Forced scalar mean (ms) |
| --- | ---: | ---: | ---: |
| 8-bit/data/00000795.ivf | 12 | 38.66 | 124.68 |
| 10-bit/data/00000775.ivf | 10 | 10.45 | 25.88 |
| 12-bit/data/00000790.ivf | 10 | 10.27 | 25.97 |
| 8-bit/film_grain/av1-1-b8-23-film_grain-50.ivf | 10 | 17.72 | 33.95 |
| 10-bit/film_grain/av1-1-b10-23-film_grain-50.ivf | 10 | 24.02 | 42.77 |

All five comparisons favor NEON with paired confidence intervals excluding zero; raw intervals and variance are in [rav1d-tiers.log](rav1d-tiers.log). All 52 frames have exact tier parity, including 12-bit and film grain. This benchmark selects dispatch with testable archmage tokens, not Settings::cpu_level. Scalar compiler auto-vectorization is allowed.

`cargo clippy --locked -p rav1d-safe --bench tier_isolation -- -D warnings` passed. Existing decoder source was unchanged; the five fixture parity checks are the validation for this benchmark change.

Fixtures are external at `test-vectors/dav1d-test-data`, or the explicit `RAV1D_BENCH_VECTORS` root. Missing fixtures fail loudly. Hashes are in [fixtures.tsv](fixtures.tsv).
