# Scoped picture policy and bounded motion-compensation reads

Work based on synchronized main `b31692de` (2026-09-07 UTC), including the
remote film-grain threading fixes and local soundness/profiling history.
The prior profiling screen remains in `audit/concurrency-profile`; its
measurements are not silently replaced by these implementation results.

## Changes and safety argument

`DisjointMut::configure_parallelism(&mut self, threads, tiles)` is additive.
It changes placement hints only, including when a caller underestimates
concurrency. Both sides of an overlap still use the same instance mapping.
An exclusive borrow prevents reconfiguration while any guard remains usable.
The implementation keeps poison and outstanding records, preserves the local
hints across resize/stride declarations, and retains the existing monotone
globals as the fallback for callers that do not opt in. `const new` is unchanged.
The shard registration, conflict checks, and acquire/release protocol are
unchanged. This is not a new soundness proof of the complete crate.

Decoder picture planes receive their policy before the new allocation is
shared. Threaded post-filters need row guards even for a single tile, so
worker count selects row guards; tile count only influences tracker placement.
Retained pictures and new allocations copied from them retain their policy.
Legacy/raw scratch components retain the conservative global fallback.

Six x86 source reservations (put/prep at both depths and warp put/prep) use
a bounded contiguous source slice instead of a whole-component guard.
Eight-tap filtering needs [-3, +4] padding on each filtered axis; bilinear
needs [0, +1]; an unfiltered axis needs no padding. Warp reads 15 rows of
15 pixels regardless of its filter coefficients. The SIMD load loops consume
no pixels beyond these tap windows. The largest regular output is 128x128,
so the helper asserts a maximum 135x135 window. Signed-stride hull arithmetic
uses checked offsets and the DisjointMut slice validates allocation bounds.

The **entire Rust slice**, including gaps between rows, remains registered.
No rectangle record is used to justify a wider contiguous reference. Concurrent
writes anywhere within the slice are still rejected; adjacent storage remains
available. Reference windows use an explicit, private narrowing of the existing
whole-component read exemption from the reconstruction extent ceiling. The
ordinary reconstruction ceiling and its tests are unchanged. These are the
same source sites previously exempt as whole-picture reads; the prior bounds
map found no concurrent foreign writes there. The usage probe still observes
each actual reservation through the normal DisjointMut registration path.

## Measurement protocol

Baseline and candidate use the same Cargo.lock, default checked features,
release fat LTO, and `RUSTFLAGS=-C llvm-args=-align-all-functions=4`.
The hash-validating `profile_concurrency` harness rotates arm order, hashes
visible output before/after timing, and excludes startup and validation.
All heavy jobs are serialized under the workstation's mandatory run-heavy
16 GiB / 8 build job / nice 19 / idle-I/O limits. No concurrent build or profiler
runs during timing. Detailed raw records and source/binary identities follow.

Initial policy-only screen: five rotations, four inputs, 1/8 workers and a
1-worker process primed by opening/dropping a 24-worker decoder. Primed
candidate/baseline medians: multi-frame 0.583, first tiled frame 0.827,
32-tile still 0.702, one-tile 10-bit still 0.964. Fresh single-worker ratios
were 0.998–1.009. Multi-frame t8 was 1.083 in this screen; the combined change
requires its own screen and this result is not called an across-the-board win.

## Verification

Focused adversarial gates exercise exclusive reconfiguration, concurrent borrowers
under underestimated hints, global promotions, retained frame lifetimes,
copied pictures, and gap reservation. Tight-vs-full SIMD tests cover all ten
filters, all phase pairs, widths through 128, and 8/10/12-bit pixels. Corpus
hash checks provide an independent pixel oracle; tight-vs-full alone only
proves parity with the existing kernels.

## Measured result and remaining cost

The main grid contains seven repetitions of 36 cells per arm: four workloads,
1/2/4/8/16/24 workers, four simultaneous decoders at 1 and 8 workers each, and
a single-worker process primed by a 24-worker decoder. Every measured run
passed frame-count and before/after pixel-hash checks. Baseline and combined
results are preserved per input under `results/`, alongside the earlier
policy-only screen. `summary.json.gz` contains paired ratios, signs, and ranges.

| Workload | Workers × instances | Baseline ms/frame | Combined ms/frame | Ratio of medians |
| --- | ---: | ---: | ---: | ---: |
| Multi-frame, 4 tiles | 1 × 1 | 2.060 | 2.074 | 1.007 |
| Multi-frame, 4 tiles | 2 × 1 | 2.999 | 2.035 | 0.679 |
| Multi-frame, 4 tiles | 4 × 1 | 2.680 | 1.896 | 0.707 |
| Multi-frame, 4 tiles | 8 × 1 | 2.807 | 2.007 | 0.715 |
| Multi-frame, 4 tiles | 16 × 1 | 3.112 | 2.376 | 0.763 |
| Multi-frame, 4 tiles | 24 × 1 | 3.383 | 2.426 | 0.717 |
| Multi-frame, 4 tiles | 8 × 4 | 0.842 | 0.584 | 0.693 |
| Multi-frame, after priming | 1 × 1 | 3.889 | 2.148 | 0.552 |
| First tiled frame | 8 × 1 | 2.149 | 2.054 | 0.956 |
| 32-tile still | 16 × 1 | 3.287 | 3.481 | 1.059 |
| 32-tile still | 24 × 1 | 3.394 | 3.632 | 1.070 |
| One-tile 10-bit still | 1 × 1 | 1.781 | 1.777 | 0.998 |

A nine-rotation follow-up compared baseline, its **byte-identical copy**, the
policy-only binary, the combined binary, and an outlined MC helper. The
combined multi-frame paired ratios at 8/16/24 workers were 0.694/0.716/0.706
(9/9 faster in each cell). The 32-tile still paired ratios were
1.010/1.070/1.068 (4/9, 0/9, 0/9 faster). Its 16/24-worker regression is larger
than the identity-control drift and remains unresolved. Outlining did not
resolve it and was reverted. No blanket improvement is claimed for stills;
fresh single-worker medians remain within about 1% in the main grid. These
results are from this Linux x86 host, compiler, and corpus, not ARM measurements.

Timed-region perf sampling on the same immutable benchmark binaries confirms
the intended reduction in wide-path work. At eight workers, wide admission
plus retirement falls from 26.51% to 4.90% of sampled cycles; with four decoder
instances, 27.42% to 4.92%. Total tracker share falls from 66.60% to 53.63%, and
65.70% to 52.22%, respectively. Spinning remains 16–18% of the reduced profile;
ordinary read/write registration is now more prominent. This does not establish
that a different spin wait policy will help. The lock algorithm and memory
orderings were deliberately kept intact. The profiles are self-IP samples;
worker DWARF caller chains remain unreliable as in the prior screen.

## Completed gates and API

- 66 decoder unit tests passed, including retained/copy policy checks, mixed
  concurrently active serial/threaded decoders, exact source hull exclusion,
  and warp windows at positive/negative strides and all three bit depths.
- 47 release regression tests passed, including the existing comprehensive
  dav1d MD5 corpus gate, film-grain concurrency, fuzz regressions, and strictness.
  The 23 committed-vector debug tests also passed with overflow checks enabled.
- 118 disjoint-mut native tests and seven doctests passed; two old illustrative
  doctests remain ignored. The new API's no-std adversarial test passed.
- All seven production-record Loom models passed at the existing two-preemption
  bound without a time/permutation cap. The new public API's concurrent test
  passed under Stacked Borrows and Tree Borrows. These are focused new Miri
  gates, not a claim that the previously interrupted full suite now passed.
- Clippy passed with warnings denied for default libraries, C-FFI, and the
  combined usage/site instrumentation. The legacy tracker compatibility build
  passed. Instrumentation-only type aliases resolve two existing Clippy warnings.
- The extent gate passed after it began inspecting decoded plane views; no
  ceiling, minimum count, or assertion was weakened.
- Patch-level semver checking against verified crates.io 0.3.1 passed all 223
  applicable checks (30 skipped). The exact delta from the previous const-
  compatible candidate is one additive `configure_parallelism(&mut self,
  usize, usize)` method: `results/api.diff`. Constructor constness is preserved.

No crate was published or tagged. The previous 0.3.2 release evidence/bookmark
is preserved; these checks validate this new source change, and do not rewrite
that package's historical provenance. This remains an argument backed by
bounded model checking and adversarial tests, not a proof of every decoder path.

The new exact-window test also runs under token permutations: on this host it
covers both AVX2 and AVX-512 dispatch, 53,760 cases per mode. The targeted
mutation run forces all pictures onto global policy and removes the final
source pixel from each MC window. Both policy tests and all three window tests
fail (five expected failures); restoring the code makes all five pass.
Commands and results are in `results/mutations.json` and the compressed logs.

Remaining leads: explicit policy for private scratch/non-picture buffers may
remove the residual priming cost; narrower source windows leave substantial
ordinary registration and shard-coherence cost. Spinning still accounts for
16–18% of sampled cycles, although its estimated absolute sampled cycles fall
about 30% with one eight-worker decoder and 35% with four. These are sampled
cycle estimates, not a separate measurement of removable wall time. The
high-worker still regression needs explanation before claiming a still-image
throughput improvement. The outlined-helper experiment did not provide one.
