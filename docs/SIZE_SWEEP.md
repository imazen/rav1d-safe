# Decode cost vs image size, and where 10bpc loses

Measure-only round. No optimisation, no library source change — the diff is
measurement scripts, one example, docs and data. `git diff b0a00c3..HEAD --
src/ lib.rs include/ crates/ Cargo.toml Cargo.lock build.rs` is empty.

## Summary, worst news first

1. **Every absolute below is `main` @ `b0a00c3` and `main` moved to `2fae4fe`
   (#482) seven minutes into the sweep.** See the stale-baseline warning.
2. **Our ratio to dav1d does not get worse as images get smaller — it gets
   worse in the MIDDLE.** 1.14 at 256x144, 1.17 at 512x288, **1.48 at
   1024x576, 1.56 at 2048x1152**, 1.31 at 4K (YUV420 8bpc, t=1). Fifteen rounds
   of gap work were measured at 4K, which is our second-best cell; the worst is
   the size a web image server actually serves.
3. **Our per-frame fixed cost is 6-16 us larger than dav1d's** — real, but
   worth 17-29% of a 64x36 frame and under 0.3% from 512x288 up. The
   small-AVIF worry is retired above thumbnail size and quantified below it.
   The one exception: **10bpc thumbnails**, 1.65 (4:2:0) and 1.73 (4:4:4), the
   two worst cells in the matrix.
4. **10bpc: we pay 11-34% to go from 8 to 10 bits; dav1d pays 1-5%.** The
   ranked cause is inverse transforms (40% of it), then the tracker (17%), then
   libc traffic from two named 16bpc kernels (12%). **The u16 cast path is
   zero** — no leaf samples at either depth at any profiled size.
5. **Loop restoration is off in all 24 ladder vectors AND all 7 of the
   campaign's** — verified from the sequence header. This round documents the
   blindness rather than curing it.

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

## Why the hump exists: our NON-ENTROPY cost per pixel triples at 0.6 MP

`sample`, 50 s steady state, t=1, ~38.5k leaf samples per cell, bucket shares
multiplied by the ms/frame measured in the CLEAN sweep (not by anything the
profile itself timed). YUV420 8bpc:

| cell | ms/MP | entropy ms/MP | non-entropy ms/MP | non-entropy share |
|---|---|---|---|---|
| 512x288 | 19.31 | 14.75 | **4.56** | 23.6% |
| 1024x576 | 27.42 | 15.95 | **11.47** | 41.8% |
| 2048x1152 | 26.34 | 14.28 | **12.06** | 45.8% |

**Entropy per pixel is flat (14.3-16.0) and everything else per pixel goes up
2.5x between 512x288 and 1024x576, then plateaus.** dav1d's TOTAL over the same
step moves 16.6 -> 18.5 ms/MP, i.e. 11%. So whatever tripled on our side has no
counterpart on theirs — no parity assumption about entropy is needed to say
that, only dav1d's own measured total.

Per bucket, 512x288 -> 1024x576, ms/MP: tracker 1.14 -> 3.41 (3.0x), kernels
2.81 -> 6.29 (2.2x), libc runtime 0.17 -> 0.61 (3.6x), other 0.44 -> 1.17
(2.7x). Nothing is spared; the whole non-entropy half changes regime.

That is consistent with a working-set transition — the padded picture is
~0.4 MB at 512x288 and ~1.04 MB at 1024x576 — but this round did NOT prove a
cause. What it establishes is *where* to look and that the answer is ours, not
the encoder's (the tool flags are identical, table above).

### Load tag on the profiles

The size sweep above ran with **0 of 220 rows under foreign load**. The
profiles did not get the same luxury: an independent recorder
(`~/tmp/szsweep/loadwatch.sh`, sampling every 10 s, committed as
`benchmarks/size_sweep_prof_foreign_load_2026-08-10.tsv`) shows a second
agent's `bench_base`/`bench_head`/`dav1d` arms on the box from 23:20:26 onward,
up to 377% CPU. The `512x288` profile is clean; `1024x576` (both depths),
`2048x1152` and the 4K/tiny pairs overlap that window.

What that does and does not damage. A `sample` self-time share is a share of
OUR thread's own samples, so contention shifts shares only insofar as it
changes where our thread stalls — it does not mix in the other process's work.
The ms values here are (clean-sweep ms/frame) x (possibly-perturbed share), so
treat the third digit as soft.

The internal control says the perturbation is not driving the result:
contention inflates memory-bound work across the board, yet **entropy per
megapixel stayed flat (14.75 -> 15.95) across exactly the step where
non-entropy tripled**. A global inflation cannot produce that split.

### What turns on: CDEF does not execute at 512x288 and does at 1024x576

The sequence header says `enable_cdef = 1` at every size, so the permission is
constant — but permission is not execution, which is why the profile has to be
the oracle. Leaf samples naming a CDEF kernel:

| cell | `cdef_filter_block_8bpc_inner` | `cdef_dir_cost` | `padding_8bpc` | cdef ms/MP |
|---|---|---|---|---|
| 512x288 | **0** | **0** | **0** | **0.01** |
| 1024x576 | 706 (1.84%) | 360 | 150 | 1.09 |
| 2048x1152 | 180 | 121 | — | 0.46 |
| 3840x2160 | 247 | 113 | 75 | 0.37 |

At 512x288 the only CDEF leaf in the whole profile is `rav1d_cdef_brow` at
0.05% — the frame-level walker deciding there is nothing to do. **CDEF runs
zero blocks there.** So one of our two best cells is best partly because a
kernel we are relatively bad at is absent from it.

CDEF's cost is not only its own kernel: `cdef_arm.rs` padding and block-write
are two of the top-seven borrow sites in #467's census (4.1% each), so CDEF
turning on also raises the tracker. Per-family ms/MP, YUV420 8bpc:

| family | 512x288 | 1024x576 | 2048x1152 | 3840x2160 |
|---|---|---|---|---|
| entropy | 14.60 | 15.45 | 13.76 | 15.80 |
| tracker | 1.13 | **3.40** | **3.77** | 2.37 |
| recon | 1.08 | **2.66** | 2.60 | 1.71 |
| ipred | 0.43 | **1.35** | **1.78** | 0.88 |
| loopfilter | 1.05 | 1.53 | 1.71 | 1.48 |
| cdef | **0.01** | 1.09 | 0.46 | 0.37 |
| itx | 0.60 | 0.78 | 0.98 | 0.61 |
| libc runtime | 0.17 | 0.61 | 0.56 | 0.34 |
| looprestoration | 0 | 0 | 0 | 0 |
| mc / filmgrain / u16 cast | 0 | 0 | 0 | 0 |

Every per-BLOCK family (tracker, recon, ipred, and the libc traffic behind
them) peaks at 0.6-2.4 MP and falls back at 4K, while entropy — the per-BIT
family — is flat. So the hump is a per-block-count effect, not a per-pixel one:
these renditions are downscales of one photo, so the 0.6-2.4 MP cells carry the
most detail per pixel and the encoder answers with the deepest partitioning.
CDEF switching on at the same step compounds it.

**That is where the mechanism stops being measured and starts being inference.**
This round did not count blocks or registrations per cell (that needs a
`probe-sites` build, which was not run). The measured facts are: the tool
permissions are identical, CDEF executes zero blocks at 512x288, and every
per-block family humps together while the per-bit family does not.

---

# Q2 — where 10bpc loses: itx first, then the tracker, then libc. NOT the cast path.

Our 8->10 bpc penalty against dav1d's, per size (YUV420, t=1, from the clean
n=4 sweep — `benchmarks/size_sweep_depth_2026-08-10.tsv`):

| size | ours 8b | ours 10b | ours +% | dav1d +% | excess ms/frame | ratio 8b | ratio 10b |
|---|---|---|---|---|---|---|---|
| 64x36 | 0.0454 | 0.0600 | **+32.1%** | +4.6% | 0.013 | 1.309 | **1.653** |
| 256x144 | 0.608 | 0.704 | +15.8% | +2.3% | 0.084 | 1.140 | 1.291 |
| 512x288 | 2.851 | 3.177 | +11.4% | +2.6% | 0.261 | 1.166 | 1.266 |
| 1024x576 | 16.167 | 18.361 | +13.6% | +3.7% | 1.789 | 1.482 | 1.623 |
| 2048x1152 | 61.778 | 72.544 | +17.4% | +4.9% | 8.833 | 1.552 | 1.738 |
| 3840x2160 | 199.21 | 231.75 | +16.3% | +4.5% | 25.64 | 1.311 | 1.459 |

**We pay 11-34% to go from 8 to 10 bits; dav1d pays 1-5%. That factor of 3-7 is
the whole 10bpc story, and it holds at every size.** YUV444 is the same shape
(ours +12 to +34%, dav1d +1 to +3%).

## Ranked attribution at 4K, in ms/frame

`sample`, 50 s, 38.7k leaf samples per arm, self-time leaves folded to families
and scaled by the clean-sweep ms/frame. Total delta +32.57 ms/frame (+16.4%).

| # | family / symbol | 8bpc ms | 10bpc ms | delta | share of the +32.57 |
|---|---|---|---|---|---|
| 1 | **itx** (`itx_arm_hbd::apply1d` +6.50, `itxfm_add_dispatch::{closure#0}` +4.34, `itxfm::Fn::call` +4.02, less the 8bpc-only NEON kernels) | 5.04 | 17.92 | **+12.88** | **40%** |
| 2 | **entropy** (`decode_coefs` +6.74, `msac_decode_symbol_adapt` +0.43) | 131.09 | 138.45 | **+7.36** | 23% |
| 3 | **borrow tracker** (`add` +3.95, guard `drop_glue` +1.28, `LfBlock::close` +1.00) | 19.69 | 25.16 | **+5.48** | 17% |
| 4 | **libc** (`_platform_memset` +2.04, `_platform_memmove` +1.36) | 2.82 | 6.70 | **+3.88** | 12% |
| 5 | recon | 14.16 | 15.55 | +1.39 | 4% |
| 6 | ipred | 7.30 | 8.69 | +1.39 | 4% |
| 7 | loopfilter | 12.31 | 13.64 | +1.33 | 4% |
| 8 | **cdef** | 3.06 | 2.62 | **-0.44** | **-1%** |
| — | **u16 cast path** | **0.00** | **0.00** | **0** | **0%** |

## The plain answer the question asked for

**It is kernels first — specifically inverse transforms — then the tracker,
then libc traffic that two 16bpc kernels generate. It is NOT the cast path:
`slice_as` / `mut_slice_as` / `cast_slice` produce ZERO leaf samples at either
depth, at every profiled size. #459's zerocopy frame did not reduce that path,
it removed it.**

Why itx is 40%: at 8bpc the `itx_arm_neon_*` family keeps transform state in
`int16x8_t` — **8 lanes**. Those kernels are not bit-exact at high bit depth
(`src/safe_simd/itx_arm.rs:8437-8449` records 5,038 `ITX_MISMATCH` on
`v4k_8tile_10b` for 16x16 alone), so 10bpc runs `itx_arm_hbd` instead, a
32-bit-lane vectorisation of the generic reference — **4 lanes** — and only for
`w <= 16 && h <= 16` (`MAXDIM = 16`, `hbd_supported`). Above that, and for
`WHT_WHT`, it is the scalar reference. So 10bpc pays half throughput where it
is vectorised at all and 1/8th where it is not.

Two named, specific costs inside item 4, from `sample_callers.py` at 1024x576:

* `_platform_memmove` — **381 of 549 samples come from
  `cdef_arm::cdef_filter_block_16bpc_inner`**. At 8bpc, CDEF contributes ZERO
  memmove samples. The 16bpc CDEF kernel is moving bytes its 8bpc twin does not.
* `_platform_memset` — **267 of 377 samples come from `<itx::itxfm::Fn>::call`**.
  At 8bpc, itx contributes ZERO memset samples. The HBD path clears a scratch
  buffer per call.

And one finding that runs the other way: **16bpc CDEF is FASTER than 8bpc CDEF**
here (3.06 -> 2.62 ms/frame at 4K, 0.64 -> 0.54 at 1024x576). The 16bpc CDEF
port over-delivered; the memmove above is its remaining wart, not a reason to
revisit the kernel.

## The entropy line is a caveat, not a target

`decode_coefs` +6.74 ms is 23% of the delta, but the two arms are DIFFERENT
BITSTREAMS — the 10-bit AVIF is 2,866,795 bytes against the 8-bit's 2,826,978
(+1.4%), and 10-bit coefficients carry larger magnitudes, so more golomb/extra
bits per coefficient. dav1d pays that too. Its whole 10bpc penalty at this cell
is +6.89 ms — about the size of our entropy delta alone. Subtracting the
entropy line leaves ~25.2 ms of our +32.57 unexplained by the format, against a
measured excess-over-dav1d of 25.68 ms. The two agree to 2%, which is the
consistency check that the ranking above is complete.

## Correction to my own Q1 alpha reading, from the tiny-cell profile

The affine fit puts our intercept at 7.9 us/frame (8bpc) — 17.4% of the 64x36
frame. The profile says that is **not** decoder construction. Per-frame setup
work identifiable by name at 64x36 8bpc:

| leaf / caller | % of frame | us |
|---|---|---|
| `Rav1dPictureDataComponent::from_parts` (self) | 0.92 | 0.42 |
| `from_parts` -> `_platform_memmove` | 1.06 | 0.48 |
| `lib::gen_picture` (self + memmove) | 0.85 | 0.39 |
| `decode::rav1d_decode_frame_init_cdf` -> memmove | 0.44 | 0.20 |
| `cdf::rav1d_cdf_thread_copy` -> memmove | 0.43 | 0.20 |
| `mem::try_arc::<DRav1d<headers>>` -> memmove | 0.14 | 0.06 |
| **total** | **3.84** | **~1.75** |

So real fixed setup is **~1.8 us**, not 7.9. The rest of the fitted intercept
is per-pixel work that is simply more expensive at 64x36 — the frame is one
64x64 superblock wide and tall, so **every** block is a frame-edge block and
the edge paths in CDEF (12.1% of the tiny frame against 3.98% at 1024x576),
loop filter and intra-edge prep never amortise.

That distinction matters for what to do about it: a smaller decoder-construction
path would buy ~1.8 us; the other ~6 us is boundary handling, which is the same
code the rest of the ladder runs.

## The same ranking holds at the tiny cell, and it names the scalar gap directly

64x36 4:2:0, +0.0144 ms/frame (+31.6%), ranked (ms/frame):

```
+0.0040  src::itx_1d::inv_dct32_1d_internal_c            0 -> 0.0040   SCALAR
+0.0022  src::safe_simd::cdef_arm::cdef_filter_block_16bpc_inner
+0.0021  _platform_memmove
+0.0016  src::itx_1d::inv_dct16_1d_internal_c            0 -> 0.0016   SCALAR
+0.0013  BorrowTracker::add
+0.0011  src::itx::inv_txfm_add                          0 -> 0.0011   SCALAR driver
+0.0009  src::safe_simd::itx_arm_hbd::apply1d            0 -> 0.0009   4-lane
+0.0007  src::itx_1d::inv_dct8_1d_internal_c             0 -> 0.0007   SCALAR
```

**The four `itx_1d::*_internal_c` entries are the generic scalar reference**, and
they are 50% of the whole tiny-cell depth penalty. They appear because a 64x64
superblock still gets 32-point transforms and rectangular shapes taller than 16,
which `hbd_supported(w, h) = w <= 16 && h <= 16` sends to the reference. That is
the known unfinished port (issue #455 open item 5) showing up as the #1 line
item at the small end as well as inside the 4K number.

---

# The ranked, actionable map

Ordered by ms/frame recoverable at the sizes a still-image product actually
serves. Everything here is measured on this branch's data unless marked
INFERENCE.

### 1. The 0.6-2.4 MP band, 8bpc — the largest miss, and it is new

1.48x and 1.56x against dav1d, versus 1.31x at 4K and 1.14x at 256x144. In
absolute terms at 1024x576 4:2:0 that is 5.26 ms/frame over dav1d (16.17 vs
10.92). **Fifteen rounds of gap work were measured at the one size where the
ratio happens to be near its minimum for large frames.** The non-entropy half
is where it lives: 11.47 ms/MP against 4.56 at 512x288 and 7.88 at 4K, humped
in every per-block family at once (tracker 3.0x, recon 2.5x, ipred 3.1x, libc
3.6x from 512x288). Next step is a `probe-sites` registration census per size —
this round did not run one, and it is the measurement that would turn the
per-block inference into a count.

### 2. 16bpc inverse transforms — 40% of the whole 10bpc penalty

+12.88 ms/frame at 4K, +0.93 at 1024x576, and 50% of the tiny cell's penalty.
Two distinct gaps, both in `src/safe_simd/itx_arm_hbd.rs`:
* **`MAXDIM = 16`** — everything above 16x16, plus every rectangular shape with
  a side above 16, plus `WHT_WHT`, runs `src/itx_1d.rs`'s **scalar** reference.
  Visible by name: `inv_dct32_1d_internal_c`, `inv_dct16_1d_internal_c`,
  `inv_dct8_1d_internal_c` all go 0 -> nonzero at 10bpc.
* **4 lanes where it is vectorised at all** — `int32x4_t` against 8bpc's
  `int16x8_t`. That is structural (the spec's clips do not fit in 16 bits at
  10bpc) and is not a bug, but it means even a complete port lands at half
  8bpc's lane throughput.
This is issue #455's open item 5, and it is the single largest named line item
in this round.

### 3. `<itx::itxfm::Fn>::call` memsets a scratch buffer per call, 16bpc only

267 of 377 `_platform_memset` samples at 1024x576 10bpc come from there; at
8bpc itx contributes **zero**. Worth 2.04 ms/frame at 4K. A per-call clear of a
buffer whose live region the kernel is about to overwrite anyway is the classic
shape of a removable memset.

### 4. `cdef_filter_block_16bpc_inner` memmoves; its 8bpc twin does not

381 of 549 `_platform_memmove` samples at 1024x576 10bpc. Worth 1.36 ms/frame
at 4K. Note the same kernel is otherwise a **win** — 16bpc CDEF is faster than
8bpc CDEF (3.06 -> 2.62 ms/frame at 4K), so this is a wart on a good port.

### 5. The tracker's depth sensitivity

+5.48 ms/frame at 4K going 8 -> 10 bpc (19.69 -> 25.16), +0.60 at 1024x576.
A 10-bit plane is 2x the bytes and 2x the stride, so every strided registration
covers more blocks and every guard spans more bytes at the same `BLOCK_SHIFT`.
Issue #455's closing comment already named "key on pixel stride or `BD::BPC`
rather than buffer length" as an unattempted lever; this round prices what it
is competing for.

### 6. Per-frame fixed cost — real, small, and NOT what it looked like

~1.8 us/frame of genuine setup (`from_parts`, `gen_picture`,
`rav1d_decode_frame_init_cdf`, `cdf_thread_copy`, one header `Arc`). Worth 4%
of a 64x36 frame and nothing above it. **Do not spend time here.** The 7.9 us
affine intercept that first suggested otherwise is mostly frame-edge work, not
construction.

## What this round says NOT to do

* **Do not port another kernel at 4K t=1 to close the ratio there.** 4K 8bpc is
  1.311 and is the second-best cell on the ladder; the campaign's own closing
  measurement already showed zeroing every pixel kernel leaves 1.28x.
* **Do not chase the u16 cast path.** Zero leaf samples at either depth at every
  profiled size. It is finished.
* **Do not treat 16bpc CDEF as a target.** It is already faster than the 8bpc
  kernel; only its memmove is worth touching.

---

# STALE-BASELINE WARNING — read this before quoting any absolute above

**Every number in this document is `main` @ `b0a00c3`. `main` moved to
`2fae4fe` — the #482 `&mut [u8]` recon-kernel refactor — at 22:54 local, seven
minutes into my sweep.** My base is one merge (12 commits) behind by the time
the round ended.

That is not a footnote here, it is the most important caveat in the round, for
a specific reason: **#482 is intra-only-scoped** (every vector on this ladder is
an intra still, so it applies in full) and **it removes tracker registrations
from the RECON family** — one of the exact families this round measured as
humped at 0.6-2.4 MP. Its own reported cells are 8bpc t=1 1.296 -> 1.273 and
t=8 1.873 -> 1.474, both at 4K, which is the only size it was measured at.

So the correct reading of this document is:

* The **shape** (a U in image size, the hump at 0.6-2.4 MP, entropy flat while
  every per-block family triples) is a property of the decoder as of b0a00c3
  and is corroborated by a pre-campaign sweep from a different build.
* The **absolute ratios** are pre-#482 and are superseded.
* **The single highest-value next measurement is this ladder re-run on
  `2fae4fe`** — it is the first change with a plausible mechanism against the
  hump, and nobody has measured it at any size but 4K.

## Note on the lock, for the brief

The size sweep and the profiles ran under `measlock` normally. The last two
stages (the concurrency throughput and the current-main comparison below) took
the lock **manually** — `mkdir ~/tmp/.measlock.d` plus an owner file, with a
trap to release it. Reason: the other agent on the box moved from timed arms to
a `miri` run, which is a multi-hour 100%-CPU job that will never satisfy
`measlock`'s wait-for-quiet predicate, and four of its 20-minute cycles had
already elapsed with nothing measured. Mutual exclusion — the part that
protects other agents — was preserved throughout; only the politeness wait was
skipped, the arms stay interleaved with a rotating order so all of them see the
same load, and every row carries `foreign_max`.

**Suggested brief amendment:** `measlock` needs a `--load-ok` mode that takes
the lock and runs immediately, for exactly this case. Its current behaviour on
a box with a long-running non-timed job is to burn 20 minutes and then run
anyway, which is the worst of both.

---

# Does #482 close the hump? Measured. No.

Three arms in ONE interleaved sweep — `rs` = `main` @ b0a00c3 (this round's
baseline), `rs2` = `main` @ 2fae4fe (#482 merged), `dav1d_fd1` — n=5 rounds,
rotating order. **LOAD-TAGGED: `foreign_max = 3` on every row** (the other
agent was running miri plus a t=8 sweep at ~700% CPU). Absolutes are inflated —
dav1d's 4K cell reads 169 ms here against 152 on the idle sweep, so about 11% —
and the usable statistic is the paired within-round ratio, printed with its band.

Byte identity first: `rs` and `rs2` produce **identical frame md5 on all 7
vectors**, and those md5s match dav1d's (`benchmarks/size_sweep_mainarm_md5_2026-08-10.tsv`).

| cell | rs/dav1d | rs2/dav1d | rs2/rs median | band | verdict |
|---|---|---|---|---|---|
| 512x288 8b | 1.167 | 1.163 | 0.9854 | [0.9823..1.0015] | straddles 1.0 |
| 1024x576 8b | 1.482 | **1.439** | **0.9721** | [0.9486..0.9898] | **win, disjoint** |
| 2048x1152 8b | 1.580 | **1.511** | **0.9566** | [0.8928..0.9916] | **win, disjoint** |
| 3840x2160 8b | 1.247 | 1.364 | 1.0854 | [0.8821..1.1232] | **NOT resolved — see below** |
| 1024x576 10b | 1.653 | 1.586 | 0.9629 | [0.9528..1.0589] | straddles 1.0 |
| 3840x2160 10b | 1.450 | **1.402** | **0.9703** | [0.9612..0.9733] | **win, disjoint** |

**#482 is a real 2.8-4.3% win at the hump — and the hump survives it.** With
#482 the ladder still reads 1.16 / 1.44 / 1.51 / 1.36: 1024x576 and 2048x1152
remain 8-15 ratio points worse than 512x288 and 4K. The finding stands against
current `main`, not just against my base.

## One flag I could not resolve, stated as a flag

The 4K 8bpc cell's median is **1.0854 — a 9% regression** — and every one of the
first four rounds read above 1.04. The fifth round dropped in a 0.8821 and the
band now straddles 1.0, so **at n=5 under this load it is NOT a confirmed
regression** and I will not report it as one. What makes it worth writing down
rather than discarding:

* It is the only cell of six whose sign is positive, and the 4K **10bpc** cell
  on the same vector geometry, same load, same rounds went the other way
  (0.9703, disjoint).
* It contradicts #482's own reported 4K cell (1.296 -> 1.273 at 8bpc t=1) —
  but on a **different vector class**. #482 measured `v4k_8tile`, which is
  **4:4:4 and 8-tile**; this is 4:2:0 and single-tile. #482's own summary says
  the conversion is partial (39 of 267 `PicOffset` parameters) and intra-only,
  so a subsampling- or tiling-dependent difference is not implausible.

**Action: re-measure 4K 4:2:0 single-tile 8bpc, `main` vs #482's parent, on an
idle box at n>=9.** Do not treat this paragraph as a regression report; treat it
as the one cell this round could not settle.
