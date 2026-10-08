# Fused small-block MC experiment

Missing: full decoder/sidecar gates, standard MC signed-destination validation
and a performance decision on the completed timing matrix.
The completed timing has case-specific gains and mixed effects; production
acceptance remains pending.

This isolated child shares the published signed warp destination repair
`80069eb0` with its benchmark baseline.
Only 8-bit 4x4 blocks with both horizontal and vertical four-tap filters
select the prepared branch. Both AVX2 and AVX-512 entry points use the
existing AVX2 token context for this narrow block. Other block sizes,
integer phases, filter widths, bit depths and dispatch gates are unchanged.

Seven active horizontal rows feed a four-vector ring. Each vertical result
uses the ring's last four rows. The implementation reads exactly seven
active bytes per source row, including a tight final row, and handles both
source-stride signs. Horizontal rounding remains `(sum + 2) >> 2`; vertical
rounding remains `(sum + 512) >> 10` for put and `(sum + 32) >> 6` for prep.
The prepared put branch clips to bytes; prep preserves signed i16 values.
No pooled mid-buffer allocation or take/return is needed in this branch.

The new oracle compares independent nested scalar arithmetic against raw
fused output, for every pair of current four-tap table rows, eight pixel
patterns, both stride signs, and a source slice containing only 49 active
bytes. Existing whole-buffer MC parity also covers 4x4 dispatch, direct
AVX2 and available AVX-512 tiers, every phase, destination padding, and the
bounded negative-source-stride fixture. Focused execution is recorded below; those tests do not establish full
decoder coverage.

Run `just test-mc-fused-small` and the full MC gates before building matched
benchmark examples. Real-clip timing must include 8-bit and 10-bit clips,
one/four workers, tracked/untracked builds, and A/A controls. Zen 5 has
not been measured.

Independent source review against the shared warp-fixed baseline finds only
the new helper, four narrow fast paths and the test module registration.
The ring schedule supplies horizontal rows y through y+3 to vertical
coefficients 2 through 5. Existing table-wide adjacent-pair and subset-sum
bounds cover the saturating horizontal multiply-add and i16 addition; the
new oracle separately bounds the rounded intermediate and final prep range.
Put packs i32 through signed i16 and then unsigned bytes, preserving byte
clipping even when the signed intermediate lies beyond the byte range.
These source checks are not executed parity or throughput measurements.

## Focused and feature gate

The [exact-source gate and reproducible patch](../benchmarks/mc_fused_small_2026-10-08/gate-build.meta.json)
pass all ten selected tests in each mode: the tight fused scalar oracle,
whole-buffer MC parity, active four-tap rows and the signed warp/reference
fixtures. The native AVX2 token is required by the fused fixture; it passed
without a token-disable permutation active. Both release all-target lint
modes and ARM/WASM/C-FFI checks pass. Matched generic release fat-LTO
examples were built with Rust 1.99.0, archmage 0.9.30, the same dependency
lock and guarded timer. Both examples unify `testable_dispatch`.

The wrapped scope returned rc=0 after 171 seconds, peak RSS 1.62 GiB,
minimum available 23,861 MiB and peak load 3.65. Complete raw output and
source/executable fingerprints are retained. The whole-clip identity and error
control results are below. This focused result does not establish full decoder
coverage or a performance benefit; Zen 5 and ordinary consumers remain
unmeasured.

## Whole-clip identity and error controls

All [32 before/after cases](../benchmarks/mc_fused_small_2026-10-08/clips-controls.meta.json)
match dav1d 1.5.3 with grain enabled: four 8/10-bit clips, tracked/untracked
builds and one/four workers at delay one. Both candidate timing binaries
reject a malformed second packet with rc101 and no RESULT; valid controls
and MD5 frame limits zero/one pass. The original no-limit MD5 example
still reports its decode error separately and is not claimed repaired.
The wrapped scope returned rc=0 after 120 seconds, peak RSS 0.20 GiB,
minimum available 25,499 MiB and peak load 1.73. The controlled timing
matrix is complete; full decoder/sidecar gates and a production landing
remain missing.

## Static code generation

The [preserved disassemblies](../benchmarks/mc_fused_small_2026-10-08/codegen.json)
resolve two private helper instantiations, shifts six and ten. Each emitted
symbol has 1,186 machine-code bytes, fourteen static byte multiply-adds
and eight static word multiply-adds. `just dump-simd-codegen` reproduces
each exact address range; executable and raw-output hashes are recorded.
These are static occurrences, not executed instruction counts, call
frequency or measured savings from avoiding the pooled intermediate.
The completed controlled timing results are below.

## Controlled timing

The [complete raw phases and provenance](../benchmarks/mc_fused_small_2026-10-08/artifacts.meta.json)
contain sixteen A/B cases and sixteen A/A controls, four alternating process
pairs per case and three timed passes per invocation. Positive percentages
mean slower. All compared executable and stream fingerprints were reverified.

| Mode | Clip | Workers | A/B median | A/B minimum | A/A median | A/A minimum |
|---|---|---:|---:|---:|---:|---:|
| safe | aom_real1080_96.ivf | 1 | -1.4168% | -1.5381% | -0.3419% | -0.0625% |
| safe | aom_real1080_96.ivf | 4 | -1.7606% | -1.8635% | -0.6313% | -1.1502% |
| safe | aom_real4k_48.ivf | 1 | -0.8776% | -0.7943% | -0.2991% | -0.1720% |
| safe | aom_real4k_48.ivf | 4 | +0.1494% | +1.1682% | +0.4397% | +0.5288% |
| safe | svt_real1080_96.ivf | 1 | -0.4128% | +0.1504% | -0.0928% | +0.5420% |
| safe | svt_real1080_96.ivf | 4 | -0.3988% | +0.7887% | -2.3866% | -0.2369% |
| safe | aom10_real1080.ivf | 1 | -0.3934% | +0.0012% | +0.0694% | -0.5758% |
| safe | aom10_real1080.ivf | 4 | +0.5182% | -0.5653% | +1.0593% | -0.7111% |
| untracked | aom_real1080_96.ivf | 1 | -0.8508% | -0.9301% | +0.8059% | +0.3854% |
| untracked | aom_real1080_96.ivf | 4 | -2.0400% | -2.4694% | +0.4208% | +1.5918% |
| untracked | aom_real4k_48.ivf | 1 | +0.2125% | +0.3456% | +0.7247% | +0.3376% |
| untracked | aom_real4k_48.ivf | 4 | -0.2521% | -1.0992% | -0.1765% | +1.9737% |
| untracked | svt_real1080_96.ivf | 1 | -0.0053% | +0.0844% | +0.3954% | +0.0629% |
| untracked | svt_real1080_96.ivf | 4 | +1.6804% | +2.8678% | -0.0680% | +2.5756% |
| untracked | aom10_real1080.ivf | 1 | -0.6606% | -0.3494% | +0.5038% | +0.4204% |
| untracked | aom10_real1080.ivf | 4 | -0.5631% | -1.1525% | -0.4085% | -0.8055% |

AOM 1080p improves at one and four workers in both modes. Tracked one-worker
AOM 4K also improves in all four process-pair medians. Untracked one-worker
4K is mixed. Untracked four-worker SVT has a slower median in three of four
process pairs, with a variable A/A minimum; a shared throughput benefit is
not established for that case. The candidate remains unlanded pending the
standard MC signed-destination validation, performance decision and full
decoder/sidecar gates. The scope returned rc=0 after 2,964 seconds, peak RSS
0.21 GiB, minimum available 25,426 MiB and peak load 1.69. These binaries
unify testable_dispatch; ordinary consumers and Zen 5 remain unmeasured.
