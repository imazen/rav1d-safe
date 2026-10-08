# Fused small-block MC experiment

Missing: completed whole-clip identity/error checks, full decoder/sidecar
gates and matched tracked/untracked A/A plus A/B timing.
No throughput improvement is claimed.

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
matrix is now running; full decoder/sidecar gates and a production landing
remain missing.

## Static code generation

The [preserved disassemblies](../benchmarks/mc_fused_small_2026-10-08/codegen.json)
resolve two private helper instantiations, shifts six and ten. Each emitted
symbol has 1,186 machine-code bytes, fourteen static byte multiply-adds
and eight static word multiply-adds. `just dump-simd-codegen` reproduces
each exact address range; executable and raw-output hashes are recorded.
These are static occurrences, not executed instruction counts, call
frequency or measured savings from avoiding the pooled intermediate.
The whole-clip controlled timing matrix remains pending.
