# Paired horizontal warp experiment

Outcome: rejected for production after complete controlled timing. Full
decoder/sidecar gates were not run for this candidate; no production change
was adopted.
The experiment starts from the landed signed-destination fix `80069eb0`.

The private 8-bit horizontal helper computes two adjacent dot products
in separate 128-bit halves of an AVX2 vector. Pixels widen from unsigned
bytes to i16 and coefficients sign-extend from i8 to i16 before signed
multiply-add. Two lane-local horizontal sums produce independent i32
results. Warp coefficients never enter saturating byte-pair accumulation.
The 15-row horizontal pass advances each phase in its original order and
stores the same rounded i16 intermediates. Vertical arithmetic, 10/12-bit
horizontal arithmetic, source footprints, destination bases and public
signatures are unchanged.

The independent dot-product oracle covers all 193-by-193 filter pairs with
coefficient-sign extrema and deterministic random pixels, 256 uniform-byte
values per filter with complementary lanes, and sixteen impulses per filter.
An absolute bound of `8 * 255 * 128` prevents i32 partial-sum overflow.
The unchanged whole-warp oracle adds 7,768 8-bit put/prep cases, signed
source rows, all filter indices, destination padding and held unrelated
source rows. Separate negative-destination and high-depth whole-plane
fixtures retain the landed addressing gate.

`just test-mc-warp-paired` selects the two dot tests, three original scalar
tests and bounded reference-window test. Runtime results must demonstrate
actual native SIMD execution; unavailable-token returns do not establish
coverage. Both decoder modes, feature boundaries, whole-clip MD5 and the
8/10-bit one/four-worker timing matrix remain required before landing.
Zen 5 has not been measured.

## Focused, feature and build gate

The [exact-source gate and reproducible candidate patch](../benchmarks/warp_paired_horizontal_2026-10-08/gate-build.meta.json)
pass six focused tests in each decoder mode. Both release all-target lint
modes and ARM/WASM/C-FFI compilation pass. Native AVX2 was present; the
negative-destination fixtures require its token, while the paired dot and
whole-warp comparisons passed without a token-disable permutation active.
Matched generic release fat-LTO examples use Rust 1.99.0, archmage 0.9.30,
the same lockfile and guarded timer as the signed-row baseline. Both
example builds unify `testable_dispatch`; ordinary consumer builds without
it are outside this measurement scope.

The wrapped scope returned rc=0 after 134 seconds, peak RSS 1.67 GiB,
minimum available 23,859 MiB and peak load 4.10. Full logs and exact source
and executable fingerprints are retained. Static disassembly of the byte
horizontal pass contains unsigned/signed widening and `vpmaddwd`; it
contains no `vpmaddubsw`. These are static code-generation observations,
not throughput or instruction-count measurements. The full decoder and
sidecar gates were not run; real-clip results are below.

## Whole-clip identity and error controls

All [32 before/after whole-clip cases](../benchmarks/warp_paired_horizontal_2026-10-08/clips-controls.meta.json)
match dav1d 1.5.3 with grain enabled, across four 8/10-bit clips, both
modes and one/four workers at frame delay one. Both new timing binaries
reject a malformed second packet with no timing RESULT; valid controls
and MD5 frame limits zero/one pass. The wrapped scope returned rc=0
after 121 seconds, peak RSS 0.20 GiB, minimum available 25,540 MiB and
peak load 1.96. Full raw output and executable provenance are retained.
The sixteen-case controlled timing matrix is complete. The MD5 checks establish
identity for those clips; they do not replace the full decoder/sidecar gates.

## Controlled timing and decision

The [complete raw matrix and provenance](../benchmarks/warp_paired_horizontal_2026-10-08/timing.meta.json)
contains all sixteen A/B cases and sixteen A/A controls. Each case has four
alternating process pairs and three timed passes per invocation, with unchanged
frame counts and matching source/executable/stream fingerprints. Percentages
below are recomputed after/before changes; positive values mean slower.

| Mode | Clip | Workers | A/B median | A/B minimum | A/A median | A/A minimum |
|---|---|---:|---:|---:|---:|---:|
| safe | aom_real1080_96.ivf | 1 | +0.0329% | +0.1021% | +0.2814% | +0.3828% |
| safe | aom_real1080_96.ivf | 4 | -0.5435% | -0.0551% | +0.4682% | +0.9680% |
| safe | aom_real4k_48.ivf | 1 | +0.8896% | +0.6956% | -0.3482% | -0.2074% |
| safe | aom_real4k_48.ivf | 4 | -0.0412% | -0.4451% | -1.6137% | +0.8454% |
| safe | svt_real1080_96.ivf | 1 | +0.3629% | +0.1898% | -0.5232% | +0.1140% |
| safe | svt_real1080_96.ivf | 4 | -1.0798% | +1.4361% | -2.6264% | -1.6716% |
| safe | aom10_real1080.ivf | 1 | -0.4825% | +0.4637% | -0.1336% | +0.2439% |
| safe | aom10_real1080.ivf | 4 | +0.1294% | +0.0924% | +0.3936% | -0.4832% |
| untracked | aom_real1080_96.ivf | 1 | +0.1083% | +0.4782% | -0.0706% | -0.0553% |
| untracked | aom_real1080_96.ivf | 4 | +1.3401% | +0.7342% | -1.0671% | -0.7878% |
| untracked | aom_real4k_48.ivf | 1 | -1.2325% | -0.8073% | -0.1440% | +0.9835% |
| untracked | aom_real4k_48.ivf | 4 | +0.2416% | -1.1332% | +0.1304% | -0.1600% |
| untracked | svt_real1080_96.ivf | 1 | -0.3846% | -0.0760% | -0.1695% | +0.0824% |
| untracked | svt_real1080_96.ivf | 4 | +2.4153% | +1.8269% | +0.2630% | -0.9964% |
| untracked | aom10_real1080.ivf | 1 | +0.0761% | +0.2989% | -0.7397% | -0.2810% |
| untracked | aom10_real1080.ivf | 4 | +0.2379% | -0.0188% | -0.4038% | +0.5235% |

The common candidate is rejected. Untracked 4K at one worker improves in all
four process-pair medians, while tracked 4K at one worker slows in all four.
Several untracked four-worker cases also slow, with variable A/A controls.
The measurements do not establish a shared tracked/untracked improvement or
a universal effect on the horizontal kernel. No source optimization is landed.
The candidate patch and full focused/feature/clip evidence remain reproducible.

The wrapped timing scope returned rc=0 after 2,974 seconds, peak RSS 0.21 GiB,
minimum available 25,457 MiB and peak load 1.80. Both modes use the same generic
release fat-LTO compiler, lockfile and guarded timer. Both example builds unify
`testable_dispatch`; consumer builds without it and Zen 5 remain unmeasured.
