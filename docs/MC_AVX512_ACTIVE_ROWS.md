# AVX-512 vertical active-row experiment

Missing: matched builds/codegen, full decoder and sidecar gates,
and matched A/A plus A/B timing. No speedup
is claimed. The experiment shares the isolated signed warp destination
repair; fused small blocks and paired warp arithmetic are absent.

The existing eight-bit vertical put and prep paths choose tap count at
runtime. Their four-tap vector loops still load two inactive rows. The
prepared change specializes the same arithmetic at four, six and eight
active taps and omits inactive loads and multiplication. Source layout,
i32 accumulation, rounding, output packing and scalar tails stay as
before. Narrow widths use the existing AVX2 constant-tap implementation.
The horizontal and high-depth kernels are unchanged.

The put oracle provides exactly the active row extent and compares whole
output buffers with an independent scalar sum. It covers every filter,
i16 minimum/maximum, alternating and sign-selected extrema, deterministic
random values, odd widths and blocks through 128 columns. Eight products
of a signed byte coefficient and signed i16 input have absolute sum at
most 33,554,432, so i32 arithmetic plus the actual rounding cannot overflow.

Prep uses the table-wide horizontal interval -1785 through 5865, already
established by the pair-window bounds gate. It includes endpoint and
coefficient-sign-selected rows, compares scalar rounded values after an
explicit checked i16 conversion, and checks destination padding. Arbitrary
i16 inputs are outside that prep contract because final signed packing
saturates while the scalar tail casts; the put clipping comparison does
cover the full signed intermediate range.

`just test-mc-v512-rows` prints executed comparisons by tap count when the
AVX-512 token is available, so a token-unavailable return cannot be reported
as measured SIMD coverage. The preserved pre-specialization child contains
the original put extent oracle; the before helper failure and focused/feature checks are now
recorded below. After focused gates, use the same 8/10-bit
clips, both modes, one/four workers and controls as the other MC comparisons.
Zen 5 is not measured.

## Executed focused and feature gates

The [before-failure and after-pass records](../benchmarks/mc_avx512_active_rows_2026-10-08/artifacts.meta.json)
reproduce a read of row six from a six-row intermediate in the original
private AVX-512 helper. This is a helper-extent defect; a production decoder
crash has not been established. The specialized source passes eleven selected
tests each in tracked and untracked modes, and six under C-FFI. Put executes
2,970/1,080/810 scalar comparisons for four/six/eight taps in each mode; prep
executes 1,980/720/540. Both release all-target lint modes and ARM/WASM/C-FFI
compile checks pass. Before returns rc100 after nineteen seconds, peak RSS
1.59 GiB, minimum available 23,859 MiB and peak load 1.35. After returns rc0
after 117 seconds, peak RSS 1.62 GiB, minimum available 23,970 MiB and peak
load 3.52. The source shares the signed warp repair; the separate standard MC
signed-origin repair must be included in its next foundation and gates.
