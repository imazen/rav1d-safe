# Checked rav1d-safe versus upstream rav1d, 2026-09-07

Current checked rav1d-safe takes **1.8–8.6 times as long** as upstream across
this four-input concurrency grid with one frame in flight. On the 24-frame
tiled video, upstream's frame threading increases the eight-worker gap to
**12.3 times**. These are measurements on the Ryzen 9 7900X, using the inputs
from the preceding concurrency investigation. Ratios describe these fixtures,
thread settings, and builds; they are not a general codec score.

The compared implementations are:

- Checked rav1d-safe with the picture policy and bounded MC reads in main
  `d1767a1bb62e8d6408d19f7f98c357fb86486a2b`. The immutable `combined` binary
  from [the preceding comparison](../../audit/concurrency-fixes/README.md)
  is reused and its SHA-256 verified. Its exact baseline/patch identity is
  retained there; later tests, documentation, and probe-only changes did not
  change the default decoder algorithm.
- Actual [memorysafety/rav1d](https://github.com/memorysafety/rav1d/tree/d3d1cd67059f47803919be8276650e5870c9fd02),
  `d3d1cd67059f47803919be8276650e5870c9fd02`, dated 2026-08-14. Its package
  version is 1.1.0. All 1,364 source blobs in the archive were verified against
  the Git tree after the build. Default features enable its hand-written
  assembly; the profile confirms calls into SSE2, AVX2, and AVX-512 assembly.

The upstream arm calls the Rust `Decoder` API in that source. No decoder source
was modified. Default checked features remain enabled on the safe arm.

## Matched decoding settings

All times below are median milliseconds per displayed frame; lower is better.
Both arms have one frame in flight (`max_frame_delay=1`), strict decoding,
film grain enabled, all inloop filters, all layers, operating point zero,
visible output only, all frame types, a 120-million-pixel limit, and native
runtime CPU dispatch. The safe fork has additional strictness checks, so
matching strict settings does not imply identical validation implementations.

| Input | Workers | Checked ms/frame | Upstream ms/frame | Checked / upstream |
| --- | ---: | ---: | ---: | ---: |
| 720×300, 24 frames, four tiles | 1 | 2.075 | 0.359 | 5.77× |
| 720×300, 24 frames, four tiles | 4 | 1.877 | 0.229 | 8.21× |
| 720×300, 24 frames, four tiles | 8 | 1.931 | 0.249 | 7.77× |
| 720×300, 24 frames, four tiles | 24 | 2.424 | 0.281 | 8.62× |
| First frame of that video, four tiles | 1 | 4.082 | 1.369 | 2.98× |
| First frame of that video, four tiles | 8 | 2.014 | 0.704 | 2.86× |
| 1024×1024 still, 32 tiles | 1 | 12.984 | 3.019 | 4.30× |
| 1024×1024 still, 32 tiles | 8 | 3.847 | 0.702 | 5.48× |
| 256×256 10-bit still, one tile | 1 | 1.763 | 0.871 | 2.02× |
| 256×256 10-bit still, one tile | 8 | 1.644 | 0.927 | 1.77× |

The complete [table](summary.tsv) includes 1/2/4/8/16/24 workers per decoder,
and four simultaneous decoders with one or eight workers each. For the video
with four eight-worker decoders, checked/upstream is 0.584/0.069 ms per output
frame, or 8.52×. Those multi-instance numbers measure aggregate throughput,
not an individual frame's latency. Thirty-two workers oversubscribe this
24-logical-CPU host equally in both arms.

## Upstream frame threading

The checked fork currently forces its frame-context count to one in
[`get_num_threads`](../../src/lib.rs), while preserving tile parallelism.
An additional upstream binary uses `max_frame_delay=0` and follows the upstream
Rust API's recommendation to retrieve one picture per submitted packet,
draining all pending pictures only at backpressure or end of stream. This
allows multiple frames in flight. It uses the same decoder source, compiler,
release profile, input copies, strictness, and hash validation as the matched arm.

| Video workers | Upstream one frame, ms/frame | Upstream automatic delay, ms/frame | Checked / automatic upstream |
| ---: | ---: | ---: | ---: |
| 1 | 0.359 | 0.362 | 5.74× |
| 2 | 0.265 | 0.213 | 9.51× |
| 4 | 0.229 | 0.155 | 12.10× |
| 8 | 0.249 | 0.156 | 12.34× |
| 16 | 0.279 | 0.179 | 12.54× |
| 24 | 0.281 | 0.189 | 12.82× |

At eight workers, upstream's frame pipeline improves its throughput by another
1.59×. Four eight-worker upstream decoders achieve 0.053 ms/frame aggregate,
versus checked's 0.584, a 10.93× gap. The 24-frame input includes pipeline fill
and drain on every pass. Enabling a setting alone cannot give the checked fork
this behavior: its frame-threading restriction needs a sound implementation.

## Profiles and next leads

Separate timed-region `cycles:u` profiles use the exact measured binaries.
Hashes, input loading, worker creation, and validation are outside perf's
enabled region. All three profiles report zero lost samples. These are
self-instruction-pointer cycle shares, not removable wall-time percentages.

- Checked video, eight workers: explicitly named tracker functions account
  for **54.38%** of sampled cycles, including **19.05%** in `TinyLock::lock_slow`.
  Ordinary read/write registration accounts for 23.00%; wide admission and
  retirement account for 3.97%. This agrees with the preceding fix's result:
  ordinary registration and contention remain substantial after narrowing reads.
- Checked video, one worker: **23.33%** is in the 8-bit warp-affine kernel,
  **18.84%** in `memmove`, and 14.12% in explicitly named tracker functions.
  The warp vertical pass still expresses a per-output dot-product loop in
  `warp_v_pass_8bpc_put`; inspecting its generated instructions and batching
  outputs is a concrete next investigation.
- Upstream video, eight workers: the profile is distributed over block and
  coefficient decoding, entropy assembly, motion compensation, and filters.
  `memmove` accounts for 1.64%. Different thread counts and total work prevent
  interpreting the difference between these percentages as a speedup estimate.

The next measured experiments should address ordinary borrow registration and
shard contention, warp-affine computation, and the sources of copying. The
`memmove` samples alone do not identify their callers or establish that every
copy is avoidable. Safe frame threading is a separate throughput opportunity.
No production optimization or soundness rule was changed in this comparison.

## Measurement integrity and evidence

Rust 1.98.1 / LLVM 22.1.8, default compilation target, fat LTO, one codegen unit,
and `RUSTFLAGS='-C llvm-args=-align-all-functions=4'` apply to both Rust arms.
Both use `panic=unwind`, matching the checked consumer profile. Upstream's own
workspace release profile specifies `panic=abort`; this comparison deliberately
uses a common consumer profile. Each project retains its own dependency lock.
Every dependency used by upstream retains the version and checksum in its lock;
only the benchmark package and untimed MD5 helper were added.

The controller starts separate processes, but reads **their internal timing**:
direct decoder calls on in-memory packets with persistent warmed decoders,
untimed file parsing, and untimed ordered visible-plane MD5 validation before
and after timing. Both adapters copy each submitted input packet. Errors,
incomplete IVF packets, missing output, output-count changes, and pixel changes
fail the run. The auto-delay adapter preserves and retries pending input.
Twelve additional checks pass malformed inputs to all three binaries; each
fails before producing a timing result (`invalid-input-checks.json`).

Seven rotations change arm order and workload order. There are **504 successful
runs, 294,784 timed output frames, and 2,268 validation passes**. Every arm and
concurrency setting reproduces the same ordered frame hashes for its input.
Both raw timings and paired-ratio ranges are retained, without outlier removal.
For example, the eight-worker video's paired matched-setting ratios range
7.36–8.28×; against upstream automatic delay they range 11.56–13.07×.

Heavy jobs ran sequentially through `run-heavy --mem 16G --jobs 8`, with nice 19,
idle I/O priority, and a hard cgroup memory limit. A shared file lock serialized
this timing controller. The matrix completed in 265 seconds; the wrapper
reported peak RSS 0.13 GiB, minimum available memory 27,492 MiB, peak load 5.13.
The profiles completed in 24 seconds, peak RSS 0.06 GiB, minimum available
27,638 MiB. No build or profiler overlapped the timing grid. These wrapper
figures are guard telemetry, not a decoder heap-memory comparison.

Evidence: `provenance.json`, `inputs.json`, `verification.json`, `summary.tsv`,
`summary.json.gz`, per-input/rotation raw JSONL archives, `frame-reference.json`,
`preflight.json`, profile reports and commands, and the complete controller logs.
The checked Cargo.lock and upstream driver lock are retained for reproduction.
Large binaries and raw perf recordings remain in
`/home/lilith/tmp/rav1d-upstream-perf-2026-09-07`.

For a new run, create an empty scratch directory with `bin/` and `upstream/`.
Extract the pinned upstream Git tree into `upstream/`, and copy this report's
`driver/` alongside it. Build that driver outside the repository workspace:

```sh
env TMPDIR=/home/lilith/tmp RUSTFLAGS='-C llvm-args=-align-all-functions=4' \
  /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  cargo +stable build --manifest-path SCRATCH/driver/Cargo.toml --release --locked --bins
```

Copy `rav1d-upstream-profile` to `bin/upstream-fd1` and
`rav1d-upstream-frame-threading` to `bin/upstream-auto`. For the exact measured
checked source, extract `b31692dee0727ffb63e159a4271fc2c015cb7a50`, apply
`audit/concurrency-fixes/results/combined.patch.gz`, restore the compressed
checked Cargo.lock, and build `--release --locked --example profile_concurrency`
with the same compiler flags and wrapper. Copy that example to `bin/checked`.

```sh
env TMPDIR=/home/lilith/tmp /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 benchmarks/upstream-2026-09-07/compare.py --repo REPO --work-dir SCRATCH --reps 7
env TMPDIR=/home/lilith/tmp /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 benchmarks/upstream-2026-09-07/profile.py --repo REPO --work-dir SCRATCH
```

`REPO` needs the dav1d test-data checkout and tracked crash vectors named in
`inputs.json`. The controller extracts the tiled single-frame input directly
from the video's first IVF packet. Inputs, binaries, source, and settings are
identified by hashes or pinned revisions; a changed build is a new experiment.
