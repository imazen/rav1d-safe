# MC pair-window review

Performance and full decoder validation remain pending for the candidate
`25ef8af1124cecccab38c54efc521a69cae6b3d5`. The focused tests described here
passed on a Ryzen 9 7900X (Zen 4) on 2026-10-08. Zen 5 was not measured.

`mc_x86_8bpc_parity.rs` compares put/prep against scalar for the existing
inter-block matrix, ten filter combinations, and all 256 phase pairs. The
endpoint gate adds two-pixel chroma blocks, odd widths, the 4/8/16/32/64/128
vector boundaries, and 128x128 blocks. It compares dispatch and AVX2 directly,
and AVX-512 when its token is available. Whole output buffers are compared,
including destination padding. Tests hold `token_test_lock` throughout.

The pair-window implementation uses saturating `maddubs` followed by wrapping
i16 additions. The arithmetic gate checks every adjacent coefficient pair
in all 90 current table rows, including odd-aligned pairs used by six-tap
kernels. Pair extrema on byte inputs are -3060 through 20400; extrema of any
subset of a row are bounded by -7140 through 23460. Adding the largest rounding
term, 34, remains inside i16. These linear bounds cover the entire byte input
range; pixel parity also checks deterministic random and endpoint patterns.

The original handed-over before binary has 1,323,456 bytes of `.tbss` and inline
loop-restoration scratch arrays. It predates the boxed TLS change despite its
reported revision label. Exact pre-MC tracked/untracked rebuilds and the MC
candidate have 11,368 bytes. Earlier reported ratios cannot isolate MC.
[Build and executable evidence](../benchmarks/mc_baseline_identity_2026-10-08.json)
records the corrected baseline. [The complete tracked A/A control](../benchmarks/mc_exact_control_safe_2026-10-08.meta.json)
records per-case observations before any optimization conclusion.

The current comparison uses Rust 1.99.0 and archmage `e2dbab6`, with
target-CPU flags unset and release fat LTO. The preserved baseline at
`2b96f04d` and first MC snapshot `f910b2f2` use identical lockfiles and
compiler/dependency versions.
[Baseline build provenance](../benchmarks/mc_current_baseline_build_2026-10-08.meta.json)
records executable hashes and TLS sections. Its 66-second wrapped build peaked
at 1.29 GiB RSS. Historical Rust 1.98.1 / archmage 0.9.29 measurements remain
separate evidence and do not establish the current candidate's performance.

The follow-up uses `#[rite]` for six pair-window helpers that run inside an
existing SIMD context, reserving `#[arcane]` for entry points. The first
current-pin candidate already has no separate helper symbols under fat LTO;
changing the annotation alone has no measured throughput benefit.

Review found that the fixture called default-only `copy_pixels_to` under
C-FFI and allocated destinations without guaranteed 64-byte alignment.
It now uses aligned destination storage in both modes. C-FFI reads the caller
storage after dropping the temporary picture wrapper; default mode keeps its
explicit copy-back. Assertions and compared pixels remain unchanged.

Review found unsigned source-row arithmetic in the batched horizontal
helpers and V-only put helpers. A new oracle uses the actual bounded
reference guard with negative source pitch, nine eight-tap filter
combinations, every phase pair and five sizes through 128x128. It compares
whole put/prep buffers against scalar while an unrelated row is borrowed
mutably. The oracle fails before repair at the V-only tail with an unsigned index
18446744073709551232 into a two-byte suffix. Keeping the full bounded
source and a separate base in V-only and batched H helpers passes the
unchanged oracle in 6.810 seconds. The 21-second wrapped run peaks at 1.59 GiB
RSS. Full decoder, sidecar, cross-compile and timing gates remain pending.

The existing three parity tests now use 64-byte-aligned destination storage
and read outputs according to the default copy or C-FFI zero-copy model.
Run `just test-mc-pair-windows bitdepth_8,bitdepth_16,c-ffi` for the latter.
The new negative-stride oracle currently uses default-only PicBuf fixture
construction; it adds coverage without removing any existing C-FFI tests.

The signed-source follow-up passes all four default focused tests in
33.255 seconds, and all three C-FFI focused tests in 2.718 seconds after
repairing alignment and read-back. The 49-second wrapped scope peaks
at 1.43 GiB RSS. Negative-stride coverage is still the default-only fixture;
these results do not establish negative-stride C-FFI fixture coverage.
Full decoder/sidecar, cross-compile and matched timing gates remain pending.

The pre-MC source also passes a suffix and unsigned row addresses to the
V-only byte filter. An isolated source-address repair and the same reversed-row
oracle are prepared on that baseline so its performance comparison can share
the correctness repair. That independent baseline has not yet compiled or run.
No speed claim follows from either prepared source change.


Static review found that the AVX2 four-tap vertical put/prep helpers still
loaded two inactive outer rows and built their zero-coefficient pair.
A separate child now omits those loads under the existing constant tap count.
Its extent oracle provides only rows 0..6, where active taps occupy 2..6;
reading either absent inactive row must fail. It compares whole put/prep
outputs at four widths through 128 against scalar integer arithmetic for
every current four-tap table row. The unchanged extent test fails before the change at row 6 beyond a six-row
slice. Afterward all five default focused tests pass in 33.261 seconds and
all four C-FFI tests pass in 2.678 seconds. The combined 63-second scope
peaks at 1.54 GiB RSS. [Before/after evidence and source hashes](../benchmarks/mc_four_tap_rows_2026-10-08.meta.json)
record this AVX2 extent result. Dynamic AVX-512 inactive-row loads are
unchanged. Full decoder/sidecar/feature gates and throughput remain missing. The previously validated
signed-source snapshot remains preserved for comparison.
