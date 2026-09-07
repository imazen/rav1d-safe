# Shared scratch sharding, 2026-09-07

Source baseline: `73a13d1759ae` (decoder code unchanged from the previous
release comparison). This experiment keeps runtime checking, guard extents,
record publication, retirement, and lock memory ordering intact.

## Measured result

Five rotated repetitions on the 720×300, 24-frame, four-tile video, with frame
delay 1; medians in milliseconds per visible frame:

| Mode | Workers | Baseline | Change | Faster |
|---|---:|---:|---:|---:|
| Checked | 1 | 2.078 | 2.084 | −0.3% |
| Checked | 8 | 1.886 | 1.598 | 15.2% |
| Checked | 24 | 2.277 | 1.768 | 22.3% |
| Unchecked | 8 | 1.402 | 1.181 | 15.7% |
| Unchecked | 24 | 1.679 | 1.249 | 25.6% |
| Assembly | 8 | 0.463 | 0.328 | 29.2% |
| Assembly | 24 | 0.639 | 0.393 | 38.5% |

With four simultaneous eight-worker decoders, throughput improves 19.1%
checked, 20.0% unchecked, and 35.6% assembly. Additional 2/4/16-worker video
cells all improve. Priming a 24-worker decoder before serial checked decoding
now measures 2.118 ms/frame versus the baseline's 2.137; the exploratory
process-history regression is removed.

With automatic frame threading, assembly improves 27.2% at eight workers
(0.291 → 0.212) and 30.0% at 24 (0.345 → 0.242). Unchecked improves about 6%
there. The auto driver preserves and retries an unaccepted packet on decoder
backpressure; the full ordered 24-frame hash sequence is checked.

Upstream remains faster: its assembly build measures 0.243 ms/frame at eight
workers/frame delay 1, versus this change's 0.328 (1.35×). With automatic frame
threading the corresponding figures are 0.149 and 0.212 (1.42×). Upstream is
`memorysafety/rav1d` at `d3d1cd67059f47803919be8276650e5870c9fd02`, using the
same consumer settings and output checks as the preceding comparison.

The three still inputs show no consistent material win. Two initial cells
looked about 2% slower (checked single-tile/t8 and assembly tiled-first/t24).
Nine longer rotated repetitions put them at +0.25% and +0.64%, respectively;
both sets of samples are retained. These results establish a video contention
fix on this corpus, not a general AV1 speedup or upstream parity.

## Cause and implementation

The previous 65,536-element threshold forced the tiled video's motion-vector
storage onto a single tracker shard. At eight workers, `r` contains 26,881
twelve-byte elements and `rp_proj` contains 6,144 elements. Each still got one
lock. Concurrent row reservations also overflowed its seven inline slots.
The large-still census that originally justified this threshold did not cover
this video workload.

The threshold is now 1,024 elements: enough for sixteen of the smallest
64-element address blocks. Tiny context arrays keep one shard. Serial policy
also keeps one shard regardless of buffer length. This uses the existing
address mapping and retains the same exclusion domain for every shared byte.
It does not assign different tracking domains to different callers or tiles.

Lowering the threshold alone exposed a process-history regression: a serial
checked decoder opened after a 24-worker decoder became 17.1% slower on the
video in the exploratory matrix. Frame initialization now configures its
scratch buffers while holding exclusive access. Resizing retains that local
policy. Newly allocated reference-MV and segmentation arrays are configured
before publishing their Arc; reused segmentation maps retain their existing
policy. Temporary `wrap_buf` scratch components also get serial tracker
placement, preserving their existing row-guard selection and full overlap
checking even if shared. Without that last step the video still regressed by
about 8% in the primed serial cell. This does not reset process-global hints
underneath other decoders.

## Diagnostic evidence

Separate `asm,probe-usage,probe-wide` builds, eight workers, one instance, the
24-frame video decoded for 96 lifetime output frames (24 timed). Counts include
warmups and validation where the probe specifies that; do not use instrumented
times as performance evidence.

| Timed tracker counter | Baseline | Threshold only |
|---|---:|---:|
| Full-slot promotions | 19,017 | 0 |
| All wide promotions | 19,017 | 390 |
| Contended fast registrations | 14,522 | 108 |
| Slow lock acquisitions | 26,599 | 28 |

The usage census confirms that the MV sites move from one shard to 128 shards;
the context arrays stay on one. These scheduling-sensitive counts establish
the mechanism, not a universal speedup or a peak-memory claim.

Timed-region self-IP cycle profiles of uninstrumented consumers corroborate
the result. Assembly's `TinyLock::lock_slow` falls from 31.53% to 0.10%; named
tracker symbols fall from 58.10% to 21.26%. The final checked build spends
3.15% in that spin path and 38.29% in named tracker symbols. Its next large
samples are memmove (15.31%) and warped MC (13.54%). Unchecked spends 22.75%
in warped MC, 19.23% in memmove, and 7.34% in named tracker symbols; its spin
path is below the report's 0.1% display threshold. Percentages are shares of
CPU cycles, not removable wall time, and sums omit symbols below that threshold.

The next useful investigations are reducing scratch copy volume/temporary
wrappers, improving warped MC, and reducing ordinary registration work.
This evidence gives little reason to prioritize a different waiting policy
for the assembly build now. Copy attribution needs caller-level investigation;
the self-IP samples alone do not identify every memmove source.

## Protocol and reproduction

The standalone consumer uses the same harness and locked dependencies as
`../release-perf-2026-09-07`. All arms use Rust 1.98.1, fat LTO, one codegen
unit, default target, panic unwind, and
`RUSTFLAGS='-C llvm-args=-align-all-functions=4'`. The four inputs and ordered
visible-frame oracle are inherited from `../upstream-2026-09-07`.

Host: Ryzen 9 7900X, 24 logical CPUs. Heavy jobs run one at a time, at nice 19
and idle I/O priority, with an enforced 16 GiB cap and eight build jobs. The
final matrix's monitor reports peak RSS 0.13 GiB, minimum available RAM
27,463 MiB, and peak load 3.72. The final checks/profiles/supplementary timings
report peak RSS 1.10 GiB, minimum available 24,547 MiB, peak load 2.95. These
are wrapper monitoring figures, not a decoder memory comparison.

`compare.py` rotates five repetitions, times in-memory decode calls in warmed
persistent decoders, checks every reference frame hash and validates output
before and after each timed run. Failed commands, incorrect frame counts, and
hash discrepancies abort the matrix. Multi-instance values are throughput
normalized per frame, not individual-decoder latency. `profile.py` enables
hardware sampling only while the internal timer is active.

Run every heavy phase serially through
`/home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8`, with
`TMPDIR=/home/lilith/tmp`. Example:

```text
python3 benchmarks/tracker-sharding-2026-09-07/build.py \
  --repo /absolute/source --work-dir /absolute/evidence --label candidate
python3 benchmarks/tracker-sharding-2026-09-07/compare.py \
  --work-dir /absolute/evidence --name comparison \
  --arms baseline=/absolute/baseline solution=/absolute/evidence/bin/candidate-checked
```

`build.py --driver auto --modes unchecked asm` builds the frame-threading
consumer; the default driver fixes frame delay to 1. Frozen consumer source,
manifests, and lockfiles live in `drivers/`. `compare.py` extracts the first
video packet before timing, so reproducing the still input needs no old scratch
directory. The build helper was exercised for both final auto binaries.

Full-sized binaries, perf.data, driver build caches, and original logs are in
`/home/lilith/tmp/rav1d-perf-solution-2026-09-07`; the baseline binaries remain
in `/home/lilith/tmp/rav1d-release-perf-2026-09-07/bin`. Committed results retain
the complete samples, source and binary identities, commands, and validation.

Raw labels document the experiment sequence: `matrix-*` is threshold only;
`final-*` adds frame/reference-array policy but predates scratch-wrapper policy;
`solution-*` is the shipped implementation. `pilot-*`, `scaling-*`, `auto-*`,
and `recheck-*` retain their stated experiments. The archive includes 2,331
timing runs across these stages, 675,072 timed frames, and 9,663 complete
ordered hash passes, all successful. The current results are the `solution-*`
rows and their scaling/auto/recheck supplements; exploratory rows are not
pooled into them. [Summary table](results/summary.tsv),
[source/binary provenance](results/provenance.json),
[validation commands](results/validation.json).

## Safety argument

Only shard placement changes in disjoint-mut. Two ranges containing the same
element still share an address block and therefore a shard; wide registrations
still coordinate with every active shard. The release/acquire protocol, the
in-lock wide-state recheck, exact range/rectangle footprints, poisoning, and
guard retirement are unchanged. Placement remains fixed while any usable guard
exists: local configuration and resizing require exclusive access.

The new public-API tests exercise medium-sized typed storage, simultaneous
guards crossing block boundaries, repeated slot reuse, wide-reader/narrow-writer
exclusion, global-hint promotion, resizing, and serial/parallel transitions.
The placement regression test rejects restoration of the old threshold. The
existing [release protocol](../../docs/RELEASE_SOUNDNESS_PROTOCOL.md) remains
the governing argument; finite testing and bounded Loom exploration do not
constitute an unrestricted proof.

Completed gates: 121 native disjoint-mut tests, seven executed doctests, 113
release decoder tests (including the dav1d MD5 corpus at eight workers), 23
debug regressions, both new tests under Stacked and Tree Borrows Miri, both
under no_std, seven Loom models bounded to two preemptions, Clippy, and 223
patch-compatibility checks against disjoint-mut 0.3.1. The final scratch-wrapper
change was followed by a fresh decoder/debug/Clippy run. Restoring the old
threshold fails the new placement test; restoring the tested source passes.

No borrowing API, `const new()`, feature contract, or package version changes.
This work does not publish a crate or advance the separate 0.3 release bookmark.
