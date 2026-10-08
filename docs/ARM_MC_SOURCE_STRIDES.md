# Signed source addressing in ARM motion compensation

Missing: the explicit token-disabled scalar supplement at 1/2/4/8 workers in both modes, and
publication.
The complete native nextest/doctest/integration selections are green with
the local source repair. No performance change is claimed.

The expanded put/prep oracle exposed a negative source pitch converted to
`usize` before computing vertical-filter tail addresses. The 16-bit dispatch
also divided that unsigned representation by two, losing the original sign.
A negative-stride source therefore indexed beyond its registered slice.

Keep the byte pitch signed and divide it by pixel size before converting
coordinates. Put/prep receive the complete bounded source slice and a separate
base index, so a backwards row walk stays inside that slice. Bilinear and
8-tap scalar tails use the same signed row addressing as vector loads.
The source hull reservation and concurrent-write conflict checks stay intact.
This repair addresses source pitch; it does not establish new guarantees for
negative destination strides in legacy assembly FFI wrappers.

The native Neoverse-N1 oracle compares whole put/prep outputs with scalar
at 8/10/12 bits, both pitch signs, ten filters, all 256 phase pairs, endpoint
and random source pixels, eight dimensions including odd tails and 128x128.
An unrelated last-row reconstruction write remains borrowed during decoding.
[Focused validation](../benchmarks/arm_mc_signed_stride_2026-10-08.meta.json)
passes the unchanged expanded oracle; its wrapped run peaks at 1.17 GiB RSS.

The first complete tracked run passed 224/226 selected tests. Restoring the
missing photo fixture and applying the user-approved bounded worker-name
observation resolved both failures. The worker assertion passes 100 repetitions
and still fails when deliberately spawning only three workers.
[Startup observation evidence](THREAD_CLEANUP_OBSERVATION.md) records that
separate test repair.

The complete rerun passed 226/226 tracked and 208/208 untracked nextest tests,
ten active doctests per mode and all nine explicitly selected integration
bodies per mode. Its 2,856-second wrapped scope peaked at 1.05 GiB RSS.
[Full native validation](../benchmarks/arm_decoder_full_2026-10-08.meta.json)
records the source inventory and existing ignored selections. The separate CPU-mask matrix passed all 24 legs, each 803/803 at 1/2/4/8
workers in both modes. Its 2,763-second scope peaked at 0.24 GiB RSS.
[CPU-mask sidecar results](../benchmarks/arm_cpu_mask_sidecars_2026-10-08.meta.json)
record the mask limitation below; fully scalar decoding still needs the supplement.

Review confirmed the ARM mask limitation documented in
[X64_APPLICABILITY A6](X64_APPLICABILITY.md#a6-cpulevelscalar-does-not-disable-safe-simd-measurement-infrastructure-gap). Several dispatchers summon
`Arm64` without testing the CPU mask, so a sidecar invocation labelled
`scalar` can still run NEON. The token-permutation suite exercises the real
fallbacks, but the CPU-mask matrix alone cannot claim fully scalar sidecar
coverage. The prepared `decode_md5` scalar guard disables the NEON token and
its descendants under the existing process-wide testing lock through decoder
worker shutdown. The worker guard passes, fails when token disabling is deliberately omitted,
and passes after exact source restoration. The unchanged assertions check
four concurrent workers and token restoration.
[Guard and mutation evidence](../benchmarks/arm_token_guard_2026-10-08.meta.json)
records the42/27/28-second scopes, each peaking at0.98GiB RSS. Fresh decoder
builds and eight scalar sidecar legs remain pending. The managed API's ARM
mask limitation remains open.
