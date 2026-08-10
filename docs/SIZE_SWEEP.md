# Decode cost vs image size, and where 10bpc loses

Measure-only round. No optimisation, no library source change — the diff is
four measurement scripts and one example.

## Why

Every gap number the rav1d-safe campaign produced across ~15 rounds is 4K, on
two vectors (`v4k_8tile`, `v4k_8tile_10b`). The product case is AVIF **still
images**, which are usually far smaller. The project's own sweep discipline
says to fit `total = alpha + beta*pixels` and report the **intercept**, not just
the slope; nobody had done it for decode. Two questions:

* **Q1** — does our ratio to dav1d get worse as images get smaller? Is our
  per-frame fixed cost (`alpha`) larger than dav1d's?
* **Q2** — 10bpc t=1 has trailed 8bpc all campaign. Where, in ms/frame?

## Method

* Host: Apple M4 Pro (`Mac16,11`, 8P+4E, 24 GB), macOS 26.5.2 build 25F84.
  `rav1d-safe` at `main` **b0a00c3** (the prompt named `b0f70ee`; `main` had
  moved two commits on, both docs/test-only — `b0f70ee` is an ancestor).
* dav1d **1.5.4** (homebrew), measured in the **same interleaved sweep**, arm
  order rotating per round, at `--framedelay 1` — the tile-threading-only model
  our checked build implements (it hard-pins `n_fc = 1`).
* One instrument on both sides: wall clock of the whole process at **two frame
  counts**, `total = a + b*frames` fitted, so process startup (exec, mmap,
  container parse, decoder construction) drops out and `b` is ms/frame. Frame
  counts are **per cell** — a 64x36 frame decodes in ~45 us, so the campaign's
  usual 2-vs-20 fit would have been pure timer noise. Counts run from
  5,000/50,000 at 64x36 to 2/16 at 4K.
* No `nice` on any timed run (Darwin maps niced processes to E-cores, ~40x wall
  distortion). No `-C target-cpu=native`. Default features. Strict idle gate:
  a cell is discarded and re-run if any foreign process exceeded 25% CPU during
  it. Whole thing serialised behind `measlock`.
* Ratios are **paired within a round** — both arms saw the same box state — and
  reduced by median with the min/max band printed, per `docs/AGENT_BRIEF.md`.

## Vectors

Built for this round, kept **out of git** (1.1 GB of IVF): sources in
`~/tmp/szsweep/src`, AVIF in `~/tmp/szsweep/vec`, IVF in `~/tmp/szsweep/ivf`,
recipe in `~/tmp/szsweep/enc.sh` + `mkivf.sh`.

One encoder config at every point of the ladder — `avifenc -s 6 -q 70
--tilerowslog2 0 --tilecolslog2 0` (aom 3.14.1, **single tile**, no film
grain) — over six 16:9 sizes downscaled from the campaign's own `src4k.png`
with `sips`, at YUV420 (the product case for AVIF stills) and YUV444
(continuity: the campaign's `v256`/`v1024`/`v4k_*` vectors are all 4:4:4), at
8 and 10 bpc. 6 x 2 x 2 = 24 vectors.

**Correctness of the new vectors, before any timing:** every one of the 24
decodes bit-identically to dav1d 1.5.4 — `decode_md5` vs `dav1d --muxer md5`,
**24/24 MATCH**, recorded in `benchmarks/size_sweep_vector_md5_2026-08-10.tsv`.

`avif_to_ivf` wraps each AVIF's payload into an N-frame IVF so dav1d is fed
exactly the bitstream `bench_ab_decode` feeds us: same temporal delimiter, same
sequence header, same key frame, re-parsed once per frame by both sides.

### Known asymmetry, stated before the results

dav1d reads its frames from an IVF file (a 12-byte header `fread` plus the
frame payload, from page cache) while our arm re-decodes one in-memory buffer.
That demux is a per-frame cost dav1d pays and we do not, and it lands in
dav1d's `alpha`. It is bounded by a few hundred nanoseconds against a 34.6 us
tiny-cell frame — under 1% — and it biases in the direction that makes OUR
small-image ratio look BETTER, so it cannot manufacture a small-image
regression, only mask one.

## Q2 scope note — already known, NOT re-reported here as new

* The zerocopy cast frame (#459) — `#[cold] #[inline(never)]` on the
  `CastError` construction site plus `#[inline(always)]` on
  `slice_as`/`mut_slice_as` — took 10bpc t=1 to 0.9380 and is on `main`.
* The 16bpc CDEF port and the 16bpc itx port each moved 10bpc materially and
  are on `main`.
* 16bpc itx above 16x16 (32- and 64-point, and `WHT_WHT`) is still scalar.
  Confirmed in source, not assumed: `src/safe_simd/itx_arm_hbd.rs` sets
  `MAXDIM = 16` and `hbd_supported(w, h) = w <= 16 && h <= 16`, and
  `src/safe_simd/itx_arm.rs:8455` is the only 16bpc arm in the dispatch.

What this round adds on top of those is the SHAPE of the depth penalty across
image size, and a ranked ms/frame attribution of what is left.
