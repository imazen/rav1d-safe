# Did #482 regress 4K 8bpc? No. What moved was the measurement.

Measure-only round. No library source changed — `git diff 0f6bf10..HEAD --
src/ lib.rs include/ crates/ Cargo.toml Cargo.lock build.rs` is empty.

## Answer, worst news first

1. **The one thing I could not get is an idle box.** All 140 groups of the main
   sweep and all 22 of the confirmation are load-tagged; **0 idle**. Other
   agents ran timed arms from at least four separate worktrees, a long `miri`,
   test suites and a 10-way parallel `rustc` build on this machine during the
   round. Every absolute below is inflated; the paired ratios are the claim.
2. **`head/parent` at the disputed cell straddles 1.0 even at n=20**
   (band [0.9201..1.1018]). The verdict below rests on the median and its
   bootstrap CI, not on a disjoint band. Only two cells in the whole round are
   band-disjoint.
3. **#482 did NOT regress 4K 8bpc.** At the disputed cell — 3840x2160 4:2:0
   single-tile, 8bpc, t=1 — `head/parent` is **0.9913**, n=20, 16/20 rounds
   faster, p=0.012, 95% bootstrap CI **[0.9837..0.9968]**. The claimed 1.0854
   sits 0.089 above an interval 0.013 wide.
4. **The measurement that was wrong is the size sweep's 1.0854**, and it is not
   the binaries, the vector, the subsampling or the tiling. Re-running **its
   own two staged binaries** under a counterbalanced design gives
   `szrs2/szrs = 0.9913` (10/11, p=0.012) — the same number my freshly built
   arms give.
5. **What #482 actually bought at 4K t=1 is small, and now I can say why.**
   The band is worth **−3.8%**; the `ReconDst` seam it shipped with cost
   **+3.0%**; net **−0.9%**. On current `main` the seam is down to **+0.6%**
   and the net is **−2.8%**. The commit that did that — `2a7ff51`, 37
   `#[inline]` → `#[inline(always)]` — **is not in `2fae4fe`**; it landed
   afterwards, inside the #483 range.
6. Unchanged and still true: **t=8 is where #482 pays.** `v4k_8tile` t=8 is
   **0.7826**, 20/20, p=0.0000, and it is one of the two **band-disjoint**
   results in this round.

## The question

Three prior measurements of #482 (tile-owned intra recon, merged `2fae4fe`) at
4K, t=1, 8bpc:

| round | vector | n | head/base | band | as reported |
|---|---|---|---|---|---|
| #482 §4 | `v4k_8tile` 4:4:4 8-tile | 9, idle | 0.9823 | [0.9699..1.0218] | win, 8/9, **the one cell whose arms overlapped** |
| #482 §12 | `v4k_8tile` 4:4:4 8-tile | 9, loaded | 0.9790 | [0.9702..0.9920] | win, 9/9, disjoint |
| size sweep | `L3840x2160_420_8b` 4:2:0 1-tile | 5, loaded | **1.0854** | [0.8821..1.1232] | **"the one cell this round could not settle"** |

The third contradicts the first two by 11 points, on a different vector class,
so "subsampling- or tiling-dependent" was live. The size-sweep round did not
claim a regression — it flagged the cell and asked for exactly this. Several
ranked plans are built on #482's 4K number, so the arithmetic was blocked.

## Arms, and how they are proved to be the commits they claim

| arm | commit | what it is |
|---|---|---|
| `parent` | `b0a00c3` | #482's first parent on `main` |
| `head` | `2fae4fe` | #482 as merged |
| `main` | `0f6bf10` | current `main` |
| `headoff` | `2fae4fe` | **same binary** as `head`, `RAV1D_OWNED_RECON=0` |
| `mainoff` | `0f6bf10` | **same binary** as `main`, `RAV1D_OWNED_RECON=0` |
| `szrs` / `szrs2` | — | the size sweep's **own staged binaries**, reused unmodified |

The `*off` arms are what make this a root cause rather than a verdict. #482 is
two things at once: a new owned-band recon path, and a `ReconDst` seam that
every shared-path write branches through. Disarming the band leaves the seam,
so within one round `headoff/parent` prices the seam alone, `head/headoff`
prices the band alone, and `head/parent` is what shipped. Armed and disarmed
are one binary, so no inter-arm delta can be a codegen artefact.

### The correction that changes what "current main" means

`2a7ff51` — the commit taking the 37 seam accessors from `#[inline]` to
`#[inline(always)]`, i.e. the fix that cut the seam tax from 1.15–3.01% to
0.3–1.3% — **is not in `2fae4fe`**:

```
git merge-base --is-ancestor 2a7ff51 2fae4fe  -> 1  (NO)
git merge-base --is-ancestor 2a7ff51 0f6bf10  -> 0  (YES)
```

So `head` carries the seam **out of line** and `main` carries it inlined. That
is visible without running anything — `nm` counts 28 `owned_recon` symbols in
`head`, 16 in `main`, 0 in `parent` — and the three binaries reproduce, to the
byte, the sizes recorded in `2a7ff51`'s own commit message:

| arm | built here | recorded in `2a7ff51` |
|---|---|---|
| `parent` | 2,868,624 | 2,868,624 (base) |
| `head` | 2,888,816 | 2,888,816 (pre-inline head) |
| `main` | 2,887,376 | 2,887,376 (post-inline head) |

An independent check that the arms are the intended commits, before a timing
was taken.

## Correctness gate, with teeth

Set-diff **by name**, hash in the key, 6 vectors × {t=1, t=8}:
`benchmarks/recheck_482_md5_2026-08-10.tsv` — **12 of 12 keys MATCH** across
all three binaries (4:2:0 and 4:4:4, 8bpc and 10bpc, 1-tile and 8-tile). The
size sweep's binaries pass the same gate:
`benchmarks/recheck_482_md5_sizesweep_bins_2026-08-10.tsv`, 4 of 4 keys across
five arms.

A gate that cannot fail proves nothing, so it was made to fail: planting
`ac_dq + 1` at `src/recon.rs:1180` (the AC dequant, live on every vector) and
rebuilding `parent` turned both probed keys **DIFFER** and the script exited
nonzero. Restoring and rebuilding returned sha256 `63bd95ff…`, byte-identical
to the original — which also shows the builds are reproducible.

## The band arms on the disputed vector — counted, not assumed

A regression at a cell where the new path silently declines would be a
different bug, so this was checked before timing. `--features probe-sites`,
registrations per frame:

| vector | `RAV1D_OWNED_RECON=1` | `=0` | removed |
|---|---|---|---|
| `L3840x2160_420_8b` (disputed) | 4,166,874 | 5,419,614 | **1,252,740 (23.1%)** |
| `v4k_8tile` (#482's own) | 6,005,602 | 7,924,706 | 1,919,104 (24.2%) |

## Method

* Apple M4 Pro (`Mac16,11`, 8P+4E, 24 GB), macOS 26.5.2 build 25F84,
  rustc 1.97.1, default features, no `asm`, no `-C target-cpu=native`, no
  `nice` on any timed run, serialised behind `measlock`.
* Vectors reused unchanged from the size-sweep ladder (`~/tmp/szsweep/vec`)
  plus the campaign's `v4k_8tile`. The ladder's `L3840x2160_444_8b` is
  byte-identical to the campaign's `v4k_1tile` (md5 `690b8601…`).
* **Two independent instruments per cell**, both always reported: the
  two-point external wall fit at 2 and 16 frames — the instrument both
  disputed measurements used — and the harness's in-process timer, which
  starts after the file read, container parse, decoder construction and a
  warmup decode. They agree throughout; where the text quotes one number it is
  the in-process one.
* Arms interleaved back-to-back, rotating order, paired within (round, cell),
  median with min/max band. A pair is called only if its **ratio** band
  excludes 1.0 — the two arms the claim compares, so the tick can fail. It
  mostly does fail here, which is the point of item 2 above.

## The verdict at the disputed cell

`L3840x2160_420_8b` t=1, n=20, in-process instrument
(`benchmarks/recheck_482_report_2026-08-10.txt`):

| pair | median | band | wins | p | reading |
|---|---|---|---|---|---|
| **`head`/`parent`** | **0.9913** | [0.9201..1.0739] | 16/20 | 0.012 | **#482 as merged: −0.9%** |
| `headoff`/`parent` | 1.0303 | [0.9024..1.1156] | 3/20 | 0.003 | the seam alone: **+3.0%** |
| `head`/`headoff` | 0.9621 | [0.8937..1.1044] | 17/20 | 0.003 | the band alone: **−3.8%** |
| `main`/`parent` | 0.9724 | [0.8531..1.0275] | 17/20 | 0.003 | current main: **−2.8%** |
| `mainoff`/`parent` | 1.0060 | [0.8899..1.1179] | 8/20 | 0.503 | main's seam: **+0.6%** |
| `main`/`head` | 0.9887 | [0.8559..1.0381] | 15/20 | 0.041 | what #483's range added |

`0.9621 × 1.0303 = 0.9913` — band times seam reproduces the shipped ratio to
four decimals, and `main`'s halves do the same (`main`/`mainoff` = 0.9666,
`× 1.0060 = 0.9724`). Medians of ratios are not obliged to chain like that
(`median(a/b) · median(b/c) ≠ median(a/c)` in general), so this is a modest
check that the per-round distributions are well behaved — not independent
evidence for the decomposition.

The band is worth about the same at both commits (−3.8% at `head`, −3.3% at
`main`); what changed between them is the seam, from +3.0% to +0.6%.

**The seam tax reproduces #482's own honest negative on a vector class it was
never measured on.** `headoff/parent` reads +3.03 / +2.47 / +0.67 / +2.61 % at
420-8b / 420-10b / 444-8b / `v4k_8tile` t=1 against #482 §4's 1.15–3.01%
band; `mainoff/parent` reads +0.60 / +0.55 / +0.42 / +0.80 % against §12b's
0.3–1.3%.

### 1.0854 is not in the sampling distribution

Percentile bootstrap of the median, 20,000 resamples, seed 20260810
(`benchmarks/recheck_482_power_2026-08-10.txt`):

```
median 0.9913   95% CI [0.9837..0.9968]
   1.0854 (the size sweep's median): OUTSIDE
   1.0000 (parity):                  OUTSIDE
```

And exhaustively, over **all 15,504 ways** to draw 5 of my 20 rounds and take
the median exactly as the size sweep did:

```
range [0.9539..1.0100]   p5 0.9668   p50 0.9921   p95 0.9984
P(5-round median >= 1.0000) =  3.20%
P(5-round median >= 1.0400) =  0.00%   (0 of 15,504)
P(5-round median >= 1.0854) =  0.00%   (0 of 15,504)
```

So a 5-round window is enough to lose the sign 3% of the time — but **not**
enough, at my box's load, to reach 1.04. Their number needed their box as well
as their n. Which is the next section.

## Every other 4K cell, same round

`head`/`parent`, in-process, n=20:

| cell | median | wins | p | note |
|---|---|---|---|---|
| 4K 4:2:0 8b t=1 | 0.9913 | 16/20 | 0.012 | the disputed cell |
| 4K 4:2:0 10b t=1 | 0.9763 | 14/20 | 0.115 | |
| 4K 4:4:4 8b t=1 | 0.9943 | 14/20 | 0.115 | = the campaign's `v4k_1tile` |
| `v4k_8tile` 8b t=1 | 0.9979 | 12/20 | 0.503 | #482 reported 0.9823 / 0.9790 |
| 4K 4:2:0 8b t=8 | 0.8569 | 16/20 | 0.012 | single-tile: no tile parallelism |
| **`v4k_8tile` 8b t=8** | **0.7826** | **20/20** | **0.0000** | **band-disjoint**; #482 reported 0.7845 / 0.7662 |

**Nothing here is subsampling- or tiling-dependent.** All four t=1 cells sit
between 0.976 and 0.998; 4:2:0 single-tile is not the outlier the size sweep's
number implied. The t=8 headline reproduces #482's own to within 0.002.

One thing fell out that nobody asked for. On the **single-tile** 4K vector,
`parent` at t=8 is **slower than at t=1** — 218.7 against 203.8 ms/frame
(mirror sweep, min-reduced, n=11). One tile means no tile parallelism, so eight
threads buy nothing and the frame pays the coordination anyway. `head` does
not: 185.0 at t=8 against 201.3 at t=1. So #482's band turns a 7% threading
*penalty* into a 8% gain on a frame with no parallelism in it, which is
independent support for the win being tracker traffic rather than pixel work.

## Why the size sweep got 1.0854 — from its own rows, and then from a probe

Its raw rows are still on disk, so the first move is not to argue with the
number but to reproduce it. `scripts/perf/recheck_482_reanalyse_sizesweep.py`
re-fits them and returns the published median exactly, and its "4 of 5 rounds
above 1.04" (`benchmarks/recheck_482_sizesweep_reanalysis_2026-08-10.txt`):

```
per round: 1.0942  1.0427  1.0854  1.1232  0.8821
median 1.0854   band [0.8821..1.1232]   spread 24.1 points
rs   min  208.07   median  212.43   max  232.64   spread  11.8%
rs2  min  205.21   median  230.57   max  237.21   spread  15.6%
rs2/rs by MINIMUM : 0.9863
rs2/rs by MEDIAN  : 1.0854
```

**Each arm's own spread across those five rounds is 11.8% and 15.6%.** A
5-sample paired median cannot resolve a 1% effect against that. Reduced by
per-arm minimum — the least-disturbed observation, which is the right
estimator when the disturbance is one-sided-positive — the *same rows* give
**0.9863**.

### What the dispersion actually is: contention, not the cell

The cell itself is quiet. Sixteen back-to-back reps of one binary, with
foreign-process count recorded per rep
(`benchmarks/recheck_482_position_probe_2026-08-10.txt`):

```
parent, foreign=1 throughout : 202 200 201 202 201 202 202 203 201 201 202 201 202 201 199 203
head,   foreign=1 throughout : 192 198 198 199 198 199 198 198 199 199 200 200 199 199 200 201
```

**Flat to ±1% over a 56-second burst.** An earlier run of the same probe
drifted 205 → 230 across reps 9-15 and looked exactly like a thermal ramp;
adding the per-rep foreign-load column killed that reading — the drift was a
neighbour's job arriving. A later run under load opened at **340 ms against
the 202 ms baseline, a 68% inflation**, and decayed back to 202 as the
neighbour finished. I had written the thermal explanation down before the load
column existed; it was wrong.

That is the mechanism, and it sharpens a rule in the brief rather than
confirming it. *"A busy box invalidates absolutes but not paired ratios"* holds
only when the disturbance is **common-mode** — shared by both arms of the pair.
A 100%-CPU neighbour that runs for the whole sweep is common-mode and cancels.
A **burst** that lands inside one arm's 3.5-second window and not the other's
does not cancel; it injects a ratio error the size of the burst, which is tens
of percent. Rotation does not fix it and neither does pairing. Only n does.

**Honest limit on this explanation.** Bursty contention is what my probe
measured and what their per-arm spreads are consistent with; I did not
instrument their box during their run and cannot prove burst placement caused
their specific five draws. What is proved is narrower and enough: their
binaries, their vector and their fit produce 0.9913 when measured again.

### Their execution-order pattern — checked, and it does NOT explain it

Their rows show the later of the two Rust arms slower in 4 of 5 rounds, median
later/earlier 1.0942 — the same magnitude as the disputed 1.0854, and the
3-arm rotation puts `rs2` later in 3 of 5 rounds. That is a tidy story and I
chased it for a while. It does not survive: at n=20 my own execution-position
table spreads only **1.12%** at this cell, the counterbalanced re-run below
removes order by construction and moves the answer by 0.0004, and 4-of-5 is
p=0.375. **Order is not the cause. It is recorded here so the next agent does
not re-derive it and believe it.**

## The confirmation: their binaries, counterbalanced

A mirrored order — `A,B,C,D,D,C,B,A` — gives every arm a position and its
mirror, so any drift that is monotone across the group cancels in every pair by
construction instead of by averaging. Each arm's two passes are reduced by
minimum. n=11, `benchmarks/recheck_482_mirror_2026-08-10.tsv`:

| cell | pair | median | band | wins | p |
|---|---|---|---|---|---|
| 4K 4:2:0 8b t=1 | **`szrs2`/`szrs`** (their binaries) | **0.9913** | [0.9391..1.0366] | 10/11 | 0.012 |
| | `head`/`parent` (mine) | 0.9909 | [0.9428..1.1018] | 8/11 | 0.227 |
| | `head`/`szrs2` (same commit, two builds) | 1.0027 | [0.9494..1.1134] | 5/11 | 1.000 |
| | `parent`/`szrs` (same commit, two builds) | 0.9944 | [0.9659..1.0481] | 7/11 | 0.549 |
| 4K 4:2:0 8b t=8 | `szrs2`/`szrs` | 0.8582 | [0.8417..1.3579] | 10/11 | 0.012 |
| | `head`/`parent` | 0.8505 | [0.8268..1.1204] | 10/11 | 0.012 |

**The size sweep's own two binaries, on the size sweep's own vector, give
0.9913** — indistinguishable from my independently built pair, and 11 points
from 1.0854. Two separate builds of each commit agree to 0.3–0.6%. The code,
the binaries and the vector are all exonerated; what is left is n and the box.

## What this un-blocks, and what it does not

* **#482's 4K number stands, with a correction.** Plans built on "#482 bought
  ~2% at 4K t=1 and ~21% at t=8" are safe on the t=8 half and should use
  **−0.9% for `2fae4fe` at t=1, −2.8% for current `main`** — the two are
  different commits and the difference is the seam, not the band.
* **Do not spend a round hunting a 4:2:0 recon regression. There is none.**
* **The seam is still worth attention, and less than it was.** +0.6% on `main`
  at t=1, on every frame that declines the band — which today is every inter
  frame. That is #482's own "first half of a conversion" argument, re-priced.
* **Not measured here:** anything below 4K, t=2/t=4, 12bpc, x86_64, wasm,
  `--features asm`/`c-ffi`/`unchecked`, inter frames, and any vector with loop
  restoration live. The ladder's `enable_restoration = 0` blindness is
  unchanged.
* **Never idle.** Every row is load-tagged. A genuinely idle repeat would
  tighten the bands; it would have to move the median 9 points to change the
  verdict, and the bootstrap CI is 1.3 points wide.

## Two things for the brief

1. **Qualify the paired-ratio rule.** "A busy box invalidates absolutes but not
   paired ratios" is true for **steady** foreign load and false for **bursty**
   load. A burst inside one arm's window is not shared by the pair and does not
   cancel — it lands whole in the ratio. When `foreign_max > 0`, either show the
   load is steady or raise n until the median's bootstrap CI is narrower than
   the effect being claimed.
2. **Record foreign load per RUN, not per group.** A per-group `foreign_max`
   cannot distinguish "one neighbour all round" from "a neighbour during arm
   B". The per-rep column is what falsified my own thermal-drift reading inside
   one minute.
