# Paired horizontal warp experiment

Missing: full decoder/sidecar gates, matched A/A plus A/B and production landing. No speedup is claimed.
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
not throughput or instruction-count measurements. Full gate results and
real-clip performance remain pending.

## Whole-clip identity and error controls

All [32 before/after whole-clip cases](../benchmarks/warp_paired_horizontal_2026-10-08/clips-controls.meta.json)
match dav1d 1.5.3 with grain enabled, across four 8/10-bit clips, both
modes and one/four workers at frame delay one. Both new timing binaries
reject a malformed second packet with no timing RESULT; valid controls
and MD5 frame limits zero/one pass. The wrapped scope returned rc=0
after 121 seconds, peak RSS 0.20 GiB, minimum available 25,540 MiB and
peak load 1.96. Full raw output and executable provenance are retained.
The sixteen-case controlled timing matrix is running; no throughput
conclusion or production landing follows from these MD5 checks.
