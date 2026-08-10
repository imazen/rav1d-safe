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

## The tool set is constant across the ladder — checked in the bitstream

Before reading anything into "the ratio changes with size", the obvious
confound had to go: libaom picks superblock size, CDEF and loop restoration per
resolution, so two cells of one ladder can be running different kernels.
`scripts/perf/seq_header_flags.py` reads the flags out of the AV1 sequence
header (spec 5.5.1, reduced-still-picture form).

| | 64x36 | 256x144 | 512x288 | 1024x576 | 2048x1152 | 3840x2160 |
|---|---|---|---|---|---|---|
| `use_128x128_superblock` | 0 | 0 | 0 | 0 | 0 | 0 |
| `enable_filter_intra` | 1 | 1 | 1 | 1 | 1 | 1 |
| `enable_intra_edge_filter` | 1 | 1 | 1 | 1 | 1 | 1 |
| `enable_superres` | 0 | 0 | 0 | 0 | 0 | 0 |
| `enable_cdef` | 1 | 1 | 1 | 1 | 1 | 1 |
| `enable_restoration` | **0** | **0** | **0** | **0** | **0** | **0** |

Identical at every size and both depths. **So the size effect below is not the
encoder turning a tool on — it is ours.**

Two things fall out of the same table:

* **Loop restoration is off at every point of this ladder**, so this round
  inherits the campaign's structural blindness rather than curing it. Recorded
  as a limitation, not buried.
* The campaign's own seven vectors read the same way — `v256`, `v1024`,
  `v1024_10b`, `v4k_1tile{,_10b}`, `v4k_8tile{,_10b}` all have
  `enable_restoration = 0`. That independently confirms the brief's claim for
  the two 4K gap vectors and extends it to the other five.

**Teeth on the parser** (it would be worthless if it could only ever print 0):
`dav1d-test-data/.../00000791.ivf` reads `use_128x128_superblock = 1`,
`enable_restoration = 1`. It also refuses to guess past a non-reduced-still
sequence header rather than mis-parsing one.

Caveat stated in the tool's own header: `enable_cdef` / `enable_restoration`
are SEQUENCE-level permissions. Absence is conclusive; presence is not — which
frame actually applies CDEF, or picks anything but `RESTORE_NONE` per plane,
lives in the uncompressed frame header. Execution is proved from profile leaf
samples, not from this table.

## Not measured — said before the results, not after

* **No loop restoration anywhere.** `enable_restoration = 0` in all 24 ladder
  vectors and all 7 campaign vectors (table above). LR is active in 696 of 768
  corpus vectors, so the filter-chain share here is understated and every other
  share is correspondingly overstated. This round does not cure the campaign's
  structural blindness; it documents its extent.
* **No inter prediction.** Every vector is a still image, so `mc_arm.rs`
  registers no borrows and runs no kernels. An AVIF still is exactly this case,
  so the omission is right for the product question and wrong for video.
* **One content class.** The whole ladder is downscales of a single photo
  (`src4k.png`). Content class moves the transform-size mix and the
  coefficient density, and it was not swept. Screen content and line art are
  not represented.
* **One quality point** (`-q 70`). The project's sweep discipline asks for
  q5-q100 with low-q density equal to high-q; that axis is not in this round.
* **One encoder** (libaom 3.14.1 via avifenc 1.4.2). zenrav1e/SVT streams pick
  different tools and were not measured.
* **aarch64 only.** No x86_64 build or run. Do not extrapolate any of these
  ratios there.
* **t=1 for the whole ladder**; only 1024x576 and 3840x2160 were also taken at
  t=8, and only at 4:2:0.
* **No 12bpc.**

## Independent corroboration of the shape, from before this round existed

The size effect is not a property of today's harness or today's vectors. The
pre-campaign yardstick sweep (`~/tmp/recon-yard/yard_report.txt`, 2026-08-06,
n=9, a *different build* on a *different day* with *different vectors*) shows
the same ordering at t=1:

| vector | geom | ours ms | dav1d ms | ratio |
|---|---|---|---|---|
| `v256` | 256x144 8bpc | 1.197 | 0.770 | **1.56** |
| `v1024` | 1024x576 8bpc | 52.951 | 15.431 | **3.43** |
| `v4k_1tile` | 3840x2160 8bpc | 597.506 | 246.514 | **2.42** |

Best at 256x144, worst at 1024x576, 4K in between — the same U with the same
hump, at ratios 2-3x larger before the campaign's work. So the campaign
improved every cell (4K 2.42 -> 1.30, 1024 3.43 -> 1.54) without changing the
shape.

That table is also an instrument cross-check: its **dav1d** column reproduces
this round's to under 1% four days apart (256x144 0.770 vs 0.765; 1024x576
15.431 vs 15.392; 4K 1-tile 246.514 vs 246.179), which is worth more confidence
than any single sweep's internal spread.

---

# Q1 — the answer is not the one the question expected

n=4 complete rounds, **0 of 220 rows under foreign load**, dav1d 1.5.4 in the
same interleaved sweep, t=1. A fifth round was started and dropped: a second
agent began its own timed campaign on the box without taking `measlock`, and
partial rounds are excluded automatically so every cell in the table has the
same n. Full tables: `benchmarks/size_sweep_report_2026-08-10.txt`.

## Our ratio to dav1d does NOT get worse as images get smaller

YUV420, t=1, ratio ours/dav1d, with each size's ratio band checked against the
previous size's:

| size | Mpx | 8bpc | band | 10bpc | band |
|---|---|---|---|---|---|
| 64x36 | 0.0023 | 1.312 | [1.301..1.322] | **1.648** | [1.647..1.655] |
| 256x144 | 0.0369 | **1.144** | [1.140..1.153] | 1.288 | [1.284..1.298] |
| 512x288 | 0.1475 | **1.165** | [1.147..1.168] | **1.266** | [1.262..1.269] |
| 1024x576 | 0.5898 | 1.479 | [1.454..1.487] | 1.625 | [1.605..1.668] |
| 2048x1152 | 2.3593 | **1.562** | [1.542..1.605] | **1.739** | [1.707..1.751] |
| 3840x2160 | 8.2944 | 1.311 | [1.309..1.314] | 1.459 | [1.451..1.469] |

Every step is disjoint from the previous size except 256->512 at 8bpc, and the
YUV444 ladder has the same shape (1.319 / 1.131 / 1.150 / 1.542 / 1.459 / 1.298
at 8bpc). **The curve is a U with a hump in the middle, not a slope.**

Three things follow, and the middle one is the finding:

1. **256x144 and 512x288 are our BEST cells in the entire campaign** — 1.13 to
   1.17, comfortably under the ~1.30x bar that has been met at exactly one 4K
   cell in fifteen rounds. Small AVIFs are not worse off. That worry is retired
   for 8bpc.
2. **The worst cells are 0.59 and 2.36 MP — 1.48 and 1.56 at 8bpc, 1.63 and
   1.74 at 10bpc — and that is the size range a web image server actually
   serves.** 4K, where every gap number in the campaign was taken, is our
   *second-best* cell at 1.31. The campaign has not been optimising the wrong
   END; it has been optimising the wrong SIZE, by 15 to 27 ratio points.
3. **The exception is the 10bpc tiny cell**, 1.648 (4:2:0) and 1.728 (4:4:4) —
   the two worst ratios in the whole 24-cell matrix. A 10-bit thumbnail IS
   worse off than the 4K numbers suggest.

## alpha and beta, reported separately

`ms/frame` versus pixels is **not** a straight line for either decoder, so the
standard OLS is misspecified and its intercept is an artifact — it hands back
**alpha = -587 us/frame for dav1d**, which is not a physical quantity. The
ms/MP column is the tell (YUV420 8bpc, ours/dav1d):

```
2 kpx 19.7/15.0   37 kpx 16.5/14.4   147 kpx 19.3/16.6
590 kpx 27.4/18.5  2.36 Mpx 26.3/16.9  8.29 Mpx 24.0/18.3
```

Both curve; ours curves more. The intercept therefore has to come from an
affine fit through the two SMALLEST cells, which assumes nothing about the rest
of the ladder:

| | alpha (us/frame) | beta (ms/MP) | alpha as % of the 64x36 frame |
|---|---|---|---|
| ours, 420 8bpc | **+7.9** | 16.3 | 17.4% |
| dav1d, 420 8bpc | **+1.5** | 14.4 | 4.3% |
| ours, 420 10bpc | **+17.1** | 18.6 | 28.5% |
| dav1d, 420 10bpc | **+2.4** | 14.7 | 6.6% |
| ours, 444 8bpc | +3.2 | 23.4 | 5.5% |
| dav1d, 444 8bpc | -4.9 | 20.9 | (indistinguishable from 0) |
| ours, 444 10bpc | +10.8 | 28.3 | 14.2% |
| dav1d, 444 10bpc | -5.3 | 21.4 | (indistinguishable from 0) |

**Our per-frame fixed cost is 6 to 16 us larger than dav1d's.** dav1d's own is
indistinguishable from zero at this resolution of measurement (two of four
ladders fit it slightly negative). So the ratio of the alphas is large, but the
DIFFERENCE is 6-16 us — which is 17-29% of a 64x36 frame, ~1-2% of a 256x144
frame, and under 0.3% of anything from 512x288 up.

**Verdict on the question as posed:** our alpha is bigger than dav1d's, and it
matters only below about 0.04 MP. It is not what makes small AVIFs slow,
because small AVIFs are not slow — the 0.6-2.4 MP band is. The intercept worry
is retired above thumbnail size and quantified below it.

## Two harness hazards worth a line in the brief

* **Never edit a shell script while it is running.** Bash reads a script
  incrementally and keeps a file offset; an in-place edit that changes byte
  lengths makes it resume parsing at the wrong place. I added a load-recording
  line to `depth_profile.sh` mid-run and reverted it within the minute to
  restore the original byte layout. The recorder now lives in a separate
  process (`~/tmp/szsweep/loadwatch.sh`) that watches from outside.
* **A "disjoint bands" tick has to compare the arms the CLAIM compares.** The
  first version of the size table printed ours-vs-dav1d disjointness, which is
  trivially true for two different decoders and could never have failed. The
  claim the table makes is about how the RATIO moves with size, so it now
  compares each size's ratio band with the previous size's — and that check
  does fire (256->512 at 8bpc reads OVERLAP).
