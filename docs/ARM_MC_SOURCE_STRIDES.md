# Signed source addressing in ARM motion compensation

Missing: green complete native decoder suites and both-mode doctests and
legacy integration selection on the new dependency pin. The source repair is
local and has not been published. No performance change is claimed.

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

The first complete tracked run passes 224/226 selected tests, including that
oracle, Argon coverage, generated/threaded streams and token permutations.
The two failures are missing `photo_4k.avif` and premature worker-name counting;
restoring the fixture and the user-approved bounded name observation pass all
seven focused tests. The name assertion also passes 100 repetitions and still
fails when deliberately spawning only three workers.
[Startup observation evidence](THREAD_CLEANUP_OBSERVATION.md) records this
separate test repair. These results are not a green full-suite claim.
