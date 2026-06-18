# AVX-512 wd=16 v-filter x16 — 4K AVIF A/B (2026-05-26)

Commit: 96e78d1 (96e78d1 fix on 64096fe feat)
Host: AMD Ryzen 9 7950X (Zen 4, full AVX512ICL), single-thread, pinned cores 14,15
Build: release, checked (default features, forbid(unsafe_code)), no -Ctarget-cpu=native
Input: test-vectors/bench/photo_4k.avif (3840x2561, AVIF intra)
Method: profile_avif 30 iters/run, level arg (v3 = AVX2 x8 fallback, v4 = AVX-512 x16)

| run | v3 AVX2 x8 (ms) | v4 AVX-512 x16 (ms) |
|-----|-----------------|---------------------|
| 1   | 192.0           | 192.6               |
| 2   | 198.2           | 205.2               |
| 3   | 189.4           | 186.0               |
| 4   | 191.4           | 191.9               |

Finding: no measurable wall-clock delta. The x16 dispatch requires four
consecutive wd=16 v-edges sharing the same deblock level; this rarely fires
on real/conformance content (0 fires across the dav1d conformance oracle
run). Loopfilter is ~8-9% of the 4K profile and only the wd=16 v-filter
subset is AVX-512-accelerated, so the lane-width gain is below noise (and is
offset by AVX-512 downclocking on Zen 4 for a sparsely-firing kernel).

Correctness: 14/14 decode_md5_verify (unchecked) pass with the v4 path live;
__simd_test per-call SIMD-vs-scalar oracle shows zero divergence on the
previously-failing vectors after the unsigned-pack clamp fix (96e78d1).
