# The hump is not a size law — it is content, and screen content is our worst case

Follow-on to [`SIZE_SWEEP.md`](SIZE_SWEEP.md) (unmerged, branch
`measure/size-sweep-intercept`) and issue #455.

Base SHA **`5606efe`** (`main`; the task named `0f6bf10`, which `main` had moved
four docs-only commits past — `141e07f`, `02dda0f`, `f87b12c`, `5606efe`).
Diff against the recorded SHA, never against the `main` ref.

Worst news first.

---

## 0. What is NOT done

* **No fix for the thing this round measured.** The finding is a redirect, not a
  patch: the ratio-to-dav1d is a content property, our worst content class is
  screen content at 2.1–3.2x, and nobody has profiled that class yet. This
  round profiles it and names the leaves; it does not port anything.
* **The one code change here is 16bpc CDEF padding**, worth low single digits at
  10bpc, and it is the *smaller* half of `SIZE_SWEEP.md`'s Q2 "check while you
  are there" item. Its sibling (the itx scratch memset) was already fixed by
  PR #484, which also **refuted** the headline 16bpc-itx candidate by census —
  see §5. This round adds nothing to 16bpc itx.
* **Loop restoration is off in every vector used here** as well
  (`enable_restoration = 0` in all 24 new vectors, verified with
  `scripts/perf/seq_header_flags.py` from the size-sweep branch). The
  campaign's structural blindness is inherited whole, again.
* **No inter prediction, no 12bpc, no x86_64 run, no wasm, no `--features
  asm`/`c-ffi`/`unchecked`, no t=2/t=4.** t=1 only for the content grid.
* **One encoder** (libaom 3.14.1 via avifenc 1.4.2). zenrav1e and SVT streams
  pick different tools and are not represented.
* **Screen content is CROPS, photos are one crop and one downscale.** Cropping
  keeps a screenshot at native scale, which is the point; but it means the
  screen ladder covers 64x36 to 2048x1152 only (the sources are 2560x1664 and
  2940x1912), and there is no 4K screen cell.
* **The box was never idle.** Two other agents held 100%-CPU jobs (`miri`,
  `decode_permutations`) throughout. Every row is load-tagged; only paired
  within-round ratios are claimed. See §6.

---

## 1. The census `SIZE_SWEEP.md` asked for: the inference was RIGHT, and it does
   not matter

`SIZE_SWEEP.md` §"Why the hump exists" ended: *"That is where the mechanism
stops being measured and starts being inference. This round did NOT count blocks
or registrations per cell (that needs a `probe-sites` build, which was not
run)."* It was the first thing this task asked for, so it was the first thing
run — `examples/probe_tracker --features probe-sites`, t=1, counts only (that
build's wall clock is perturbed and is used nowhere).

Borrow registrations per frame and **per megapixel**, 4:2:0 8bpc, the size
sweep's own ladder:

| cell | MP | regs/frame | regs/MP | vs 512x288 | its ratio to dav1d |
|---|---|---|---|---|---|
| 64x36 | 0.0023 | 1,037 | 450,087 | 1.99 | 1.300 |
| 256x144 | 0.0369 | 6,527 | 177,056 | 0.78 | 1.145 |
| 512x288 | 0.1475 | 33,315 | 225,932 | 1.00 | 1.147 |
| **1024x576** | 0.5898 | 477,714 | **809,926** | **3.58** | 1.428 |
| **2048x1152** | 2.3593 | 1,837,330 | **778,762** | **3.45** | 1.515 |
| 3840x2160 | 8.2944 | 4,166,874 | 502,372 | 2.22 | 1.274 |

**The inference is confirmed exactly.** Registrations per megapixel is humped in
the same place and by the same factor as the tracker's ms/MP: the size sweep
measured tracker 1.13 → 3.40 → 3.77 → 2.37 ms/MP across 512 → 1024 → 2048 → 4K,
i.e. 1.00 / 3.01 / 3.34 / 2.10 normalised; the census gives 1.00 / 3.58 / 3.45 /
2.22. Nothing else is needed to explain the tracker line.

Per site (`regs/MP`, 4:2:0 8bpc), the whole per-block family moves together —
so it is block COUNT, not one hot site:

| site | 256 | 512 | 1024 | 2048 | 4K | x512→1024 |
|---|---|---|---|---|---|---|
| `src/ctx.rs:99:27` | 74,870 | 101,508 | **355,177** | 357,977 | 220,661 | 3.50 |
| `src/loopfilter.rs:769:33` | 12,126 | 17,022 | 35,338 | 38,608 | 31,090 | 2.08 |
| `src/ipred_prepare.rs:{34,37,47}` | 17,091 | 21,729 | 78,247 | 78,372 | 49,137 | 3.60 |
| `src/recon.rs:{2352,2353,2734,2735}` | 17,904 | 29,362 | 80,316 | 97,286 | 56,868 | 2.74 |
| `src/decode.rs` (8 sites) | 19,936 | 25,347 | 108,000 | 104,496 | 65,520 | 4.26 |
| `src/env.rs` (4 sites) | 13,428 | 16,682 | 39,652 | 60,062 | 38,074 | 2.38 |
| **CDEF (6 sites)** | **0** | **0** | **102,836** | **30,015** | **33,281** | **∞** |

Two things fall out:

* `src/ctx.rs:99:27` — `CaseSetter::set_disjoint`'s `index_mut` — is **43.9% of
  every registration in the frame** at the hump cell, at a mean extent of
  **3.3 bytes**. PR #488 attributed it across its 12 callers and named the
  removable half (`t.l`, a per-worker field reachable through `&mut`); it is
  still not reduced. On THIS ladder it is twice as dominant as on the
  campaign's `v4k_8tile` (22.2%), so the prize is bigger here than #488 priced
  it: half of 43.9% at ~4.2 ns/registration is **0.44 ms of a 16.17 ms frame,
  2.7%**.
* **CDEF's six sites appear from 1024x576 up and are exactly 0 at 512x288**,
  independently confirming `SIZE_SWEEP.md`'s profile-based "CDEF executes zero
  blocks at 512x288" from a completely different instrument. 12.7% of the hump
  cell's registrations are CDEF's.

**And then it stops mattering**, because of §2.

---

## 2. Registrations per pixel does not predict the gap. Neither does pixel count.

The size ladder is one content class (`sips` downscales of one photo) at one
quality (q70). Block density is not a function of pixel count — it is a function
of how finely the encoder partitions, which content and quality move by an order
of magnitude at constant size. `scripts/perf/content_sweep.sh` varies those two
axes; `scripts/perf/content_report.py` fits the result.

Four content classes x four qualities, **all at 1024x576**, 4:2:0 8bpc, t=1,
ratio ours/dav1d 1.5.4 `--framedelay 1`, paired per round, median of n rounds
with the min/max band:

| content class, all at 1024x576 | q20 | q40 | q70 | q90 |
|---|---|---|---|---|
| photo, downscaled (the ladder&#39;s own content) | **1.8237** [1.7932..1.8719] | **1.6156** [1.6002..1.6514] | **1.4320** [1.2616..1.4624] | **1.4443** [1.4168..1.5142] |
| photo, native crop | **2.6427** [2.5717..2.7262] | **2.0231** [1.9977..2.0525] | **1.4336** [1.3948..1.4548] | **1.3968** [1.3772..1.4517] |
| screen: wiki page (text) | **2.4747** [2.4451..2.5655] | **2.4640** [2.4168..2.5374] | **2.2537** [2.1667..2.3258] | **2.0891** [2.0630..2.1619] |
| screen: macOS UI (dark) | **3.1637** [2.9490..3.4387] | **3.1150** [3.0676..3.2049] | **2.8967** [2.8211..2.9873] | **2.5733** [2.4472..2.6653] |

The same cells in ms/frame, so nobody has to take a ratio on faith:

| ours ms/f vs dav1d ms/f | q20 | q40 | q70 | q90 |
|---|---|---|---|---|
| photo, downscaled (the ladder&#39;s own content) | 6.5519 / 3.5852 | 9.0051 / 5.5859 | 16.0513 / 11.1966 | 25.6250 / 17.8333 |
| photo, native crop | 5.9549 / 2.2674 | 5.9697 / 2.9461 | 9.2121 / 6.4192 | 17.2130 / 12.1574 |
| screen: wiki page (text) | 1.1549 / 0.4618 | 1.2704 / 0.5141 | 1.4513 / 0.6436 | 1.6303 / 0.7778 |
| screen: macOS UI (dark) | 6.1145 / 1.9495 | 6.8506 / 2.1762 | 7.6872 / 2.5967 | 8.4676 / 3.2593 |

**At one pixel count the ratio spans 1.397 to 3.164 — a 2.26x spread that
contains the size ladder's entire 1.12–1.49 range twice over.** n=9 complete
rounds (run 1). An independent **n=4 re-run** after the harness incident in §6
reproduces every cell: 1.827/1.639/1.447/1.480 (`pdown`), 2.663/2.071/1.440/1.403
(`pnat`), 2.607/2.632/2.359/2.123 (`swiki`), 3.241/3.282/2.879/2.546 (`simac`) —
each within 0.12 of run 1, with the same ordering and the same regression verdict
(`log10(pixels)` R^2 = 0.000, `regs/pixel` 0.015, `log10(dav1d ms/MP)` 0.632).
Both runs are committed.

Regressed over all 30 cells (4 content classes, 4 qualities, 5 sizes):

```
ratio ~ regs/pixel        : slope -0.1866  intercept +2.1474  R^2 0.009  n=30
ratio ~ log10(pixels)     : slope +0.0171  intercept +1.9561  R^2 0.001  n=30
ratio ~ log10(ours ms/MP) : slope -1.0147  intercept +3.0828  R^2 0.354  n=30
ratio ~ log10(dav1d ms/MP): slope -1.0226  intercept +2.7929  R^2 0.578  n=30
```

* **`log10(pixels)`: R² ≈ 0.00.** Pixel count explains nothing.
* **`regs/pixel`: R² ≈ 0.01.** The mechanism §1 confirmed explains nothing
  either — it explains the tracker's *share*, not the *ratio*.
* The best single predictor is **decode work per pixel**, and the sign is
  negative: the less work a frame carries per pixel, the worse we do against
  dav1d.

That relation replicates inside every content class: the ratio falls as quality
(and so coefficient density) rises, in 4 of 4 classes (`pdown`'s only break is
q70 → q90, 1.427 → 1.435, inside its own band).

**So `SIZE_SWEEP.md`'s "U with a hump at 0.6–2.4 MP" is not a size law.** It is
the ladder's accidental correlation between rendition size and coefficient
density: downscaling one photo to 512x288 throws away the detail that makes
1024x576 and 2048x1152 expensive. The finding was real and correctly measured;
the *generalisation* to "0.6–2.4 MP is our bad zone" does not survive a second
content class. Anyone choosing an optimisation target by rendition size is
choosing by a proxy with R² = 0.00.

---

## 3. What replaces it: screen content, an axis the campaign has never measured

Ratio to dav1d at q70, screen content (native-scale CROPS, so no downscale
softens the text) against the photo ladder at the same size:

| size (q70) | screen wiki | band | screen macOS UI | band | photo ladder |
|---|---|---|---|---|---|
| 64x36 | 3.2667 | [3.0917..3.3846] | 1.8194 | [1.7902..1.9071] | 1.3003 [1.2140..1.3154] |
| 256x144 | 2.3516 | [2.3200..2.4799] | 1.6663 | [1.6368..1.7500] | 1.1216 [1.1089..1.1749] |
| 512x288 | 2.4582 | [2.4394..2.4970] | 2.0766 | [2.0116..2.1486] | 1.1487 [1.1400..1.1772] |
| 1024x576 | 2.2537 | [2.1667..2.3258] | 2.8967 | [2.8211..2.9873] | 1.4292 [1.4149..1.4846] |
| 2048x1152 | 2.6639 | [2.6367..2.7007] | 2.5689 | [2.4753..2.6731] | 1.4940 [1.4625..1.5332] |
| 3840x2160 | &mdash; | &mdash; | &mdash; | &mdash; | 1.2738 [1.2026..1.2915] |


**Screen content is 2.08x to 3.26x dav1d. The photo ladder is 1.12x to 1.49x.**
Our worst photo cell in the whole matrix (`pnat` q20, 2.64) is beaten by the
*best* macOS-UI cell at any quality (2.57), and the wiki-text ladder never drops
below 2.25 above thumbnail size.

Two honest qualifications before this is quoted:

* **The 64x36 cells are mostly fixed cost, not content.** `swiki_64x36` is
  7.7 us/frame against `SIZE_SWEEP.md`'s fitted alpha of ~7.9 us, so its 3.27
  is the intercept story that round already told, not a screen-content one.
  Read the 512x288-and-up rows.
* **Two screenshots are not a content class.** They are two crops of two
  sources. What they establish is that the axis MOVES the answer by more than
  the entire size ladder does, not a per-class constant.

`docs/AGENT_BRIEF.md` §2 already requires ">= 3 content classes (photo,
screen/synthetic, line-art)" for source-informing sweeps. Every vector the
rav1d-safe campaign has ever measured — `v256`, `v1024`, `v4k_1tile`,
`v4k_8tile`, both 10bpc twins, and all 24 of the size ladder — is a photo.
`rg -i 'screen content'` over `docs/` returns two hits, both the rule itself.

---

## 4. Where the screen-content gap goes

`sample`, 45–50 s steady state, t=1, ~38k self-time leaves per cell, bucketed
by `scripts/perf/bucket_selftime.py`. All three cells at the SAME size and
quality (1024x576, q70, 4:2:0 8bpc), so the only variable is content. Shares are
shares of OUR thread's own samples, so foreign load shifts them only where it
changes where our thread stalls; treat the third digit as soft.

| bucket | photo (`pdown`) | screen text (`swiki`) | screen UI (`simac`) |
|---|---|---|---|
| entropy | **58.80%** | 11.37% | 13.22% |
| **borrow tracker** | 11.49% | **26.88%** | **30.58%** |
| kernels | 23.43% | 25.32% | 11.88% |
| libc / runtime | 2.43% | 3.95% | 9.83% |
| other | 3.85% | 32.48% | 34.49% |
| — of which `decode::read_pal_indices` | — | **20.93%** | **17.19%** |
| — of which `refmvs::splat_mv` + `add_spatial_candidate` | — | 2.85% | 3.93% |
| — of which `recon::rav1d_read_pal_plane` | — | 2.68% | 2.20% |

Read in milliseconds against the same cells' measured wall clock:

| cell | ours | dav1d | excess | tracker ms | of excess | `read_pal_indices` ms | of excess | both |
|---|---|---|---|---|---|---|---|---|
| `pdown` (photo) | 16.051 | 11.197 | 4.855 | 1.844 | **38.0%** | 0 | 0% | 38.0% |
| `swiki` (screen text) | 1.451 | 0.644 | 0.808 | 0.390 | **48.3%** | 0.304 | 37.6% | **85.9%** |
| `simac` (screen UI) | 7.687 | 2.597 | 5.091 | 2.351 | **46.2%** | 1.321 | 26.0% | **72.1%** |

Three findings, most robust first:

1. **The borrow tracker is 38–48% of our excess over dav1d in EVERY content
   class**, and dav1d has no counterpart to it at all. Its *share of the frame*
   triples on screen content (11.5% → 26.9 / 30.6%) not because it does more
   work per block but because there is so much less other work to hide it
   behind: `simac`'s registration count (497,582/frame) is within 5% of the
   photo's (477,714) while its frame is less than half as long. **On screen
   content our tracker alone (2.35 ms) is 91% of dav1d's entire frame time.**
   That is the strongest argument yet for #469/#474/#481 — and it was invisible
   in fifteen rounds of photo-only measurement, where the tracker sits behind a
   59%-entropy wall.
2. **`decode::read_pal_indices` is 17–21% of a screen-content frame and 0% of a
   photo's.** Stated carefully, because the bucketer files it under "other": it
   is the palette colour-map wavefront (`src/decode.rs:738`), an `order_palette`
   context derivation plus an MSAC symbol decode **per pixel**, so most of it is
   genuinely entropy work that the `entropy` regex does not match by name.
   Counting it as entropy, entropy is ~32% (`swiki`) / ~30% (`simac`) against
   the photo's 59% — still a different regime, and the specific loop is a named,
   never-profiled target.
3. **A still image is running the motion-vector machinery.** `refmvs::splat_mv`
   and `add_spatial_candidate` take 2.9–3.9% combined, and `recon::mc` another
   3.66% on `simac`, on a key-frame-only AVIF. That is **intra block copy**
   spending its display-vector budget through the inter path. Photos show none
   of it.

What this does not say: two screenshots are not a content class, this is one
size and one quality, and nothing here is a fix.

---

## 5. The one code change: 16bpc CDEF padding stops calling `memmove`

`SIZE_SWEEP.md` Q2 named two libc items and asked which kind of defect each was
— "a scratch buffer being zeroed or copied per call is a different fix from a
missing kernel":

* `_platform_memset`, **267 of 377 samples from `<itx::itxfm::Fn>::call`** — the
  itx scratch, **fixed by PR #484** (`032d917`, "size the itx scratch to the
  shape"). Not touched here.
* `_platform_memmove`, **381 of 549 samples from
  `cdef_arm::cdef_filter_block_16bpc_inner`**, and **zero from 8bpc CDEF** —
  unclaimed by any open PR. That is this change.

The asymmetry is one line of spelling. `padding_8bpc` copies each row through
`widen_row` → `widen_n::<N>`, a **compile-time** trip count, so the
load/widen/store is inline. `padding_16bpc` needs no widening, so the obvious
spelling was `copy_from_slice` — which takes a **runtime** length and lowers to
a `_platform_memmove` call for 4 to 12 `u16`s. Up to `h + 4` of those calls per
CDEF block (one per block row, two top rows, two bottom rows).

`copy_row_u16` is `widen_row`'s 16bpc twin over the same length set
{4, 6, 8, 10, 12} — `w` is 4 or 8, and a padding row is `w`, `w + 2`, or
`x_end - x_start` wide — falling back to `copy_from_slice` for anything else.
**Byte-identical by construction**: same bytes, same order, same destination,
and the three call sites keep exactly the borrow guards they had.

### Measured: NULL except at the tiny cell, and the control says why

`base` = `be289f4^` (`src/safe_simd/cdef_arm.rs` at the base SHA), `head` =
`be289f4`. Two binaries built from the same tree, differing only inside
`padding_16bpc`; sha256 `b5965101…` vs `3261e6e6…`, and a rebuild of `head`
reproduced its sha exactly. n=9 complete rounds, arms interleaved back-to-back
with the order rotating per round, `measlock --load-ok`, load-tagged.
`scripts/perf/pair_report.py`:

```
cell                      n  base ms/f  head ms/f  head/base   ratio band       win    p     DJ
L64x36_420_10b            9     0.0602     0.0590   0.9782  [0.9637..0.9894]   9/9  0.004   no
L512x288_420_10b          9     3.1978     3.1844   0.9954  [0.9896..1.0430]   7/9  0.180   no
L1024x576_420_10b         9    17.8833    17.9333   0.9946  [0.9798..1.0146]   5/9  1.000   no
L2048x1152_420_10b        9    68.8222    69.3778   1.0003  [0.9857..1.0231]   4/9  1.000   no
L3840x2160_420_10b        9   227.9286   226.5714   0.9978  [0.9625..1.0374]   6/9  0.508   no
L1024x576_444_10b         9    25.8111    25.7167   0.9957  [0.9752..1.0024]   7/9  0.070   no
L3840x2160_444_10b        9   356.2857   354.5000   1.0002  [0.9876..1.0134]   4/9  1.000   no
L256x144_420_10b          9     0.7069     0.7116   1.0078  [0.9768..1.0139]   1/9  0.039   no
--- 8bpc CONTROLS: the change cannot reach this code ---
L1024x576_420_8b          9    16.0833    16.2556   1.0094  [0.9784..1.0206]   3/9  0.508   no
L3840x2160_420_8b         9   200.1429   201.7143   1.0021  [0.9838..1.0157]   4/9  1.000   no
```

**The only defensible claim is the 64x36 cell: 0.9782, 9 of 9 rounds faster,
sign-test p=0.004.** Everything else is inside the instrument:

* `L256x144_420_10b` reads **+0.78%, 1 of 9, p=0.039** — nominally a
  *significant regression*, and exactly the magnitude of
* the **8bpc control at 1024x576: +0.94%, 3 of 9** — on a code path the change
  cannot execute. Two binaries that differ only inside
  `if BD::BPC == BPC::BPC16` still have different code LAYOUT, and that alone
  produces ~1% here. So ~1% is this A/B's floor, and six of the eight 10bpc
  cells move less than that.
* No cell is band-disjoint, at any n.

**This is what the price says should happen.** The count, computed before the
build from the `probe-sites` census: 120 CDEF blocks/frame at 64x36 10bpc and
7,616 at 1024x576, times ~8 removed `memmove` calls each (one per block row,
two top, two bottom). At 64x36 that is ~960 calls against a 60.2 us frame; at
1024x576 ~61,000 against 17.9 ms; at 4K ~290,000 against 227 ms — i.e. the same
absolute saving is 2% of a thumbnail, 0.5% of a 0.6 MP frame and 0.2% of a 4K
frame. The measured 0.978 / 0.995 / 0.998 track that ordering. **The mechanism
is real and its size is, at every cell but the smallest, below what this box
resolves.** #488's lesson generalises: convert the count into milliseconds
first — and then check whether the milliseconds are above your floor.

### The mechanism, verified independently of the wall clock

`sample`, 45 s, `L1024x576_420_10b`, t=1, ~34.6k leaves per arm,
`scripts/perf/sample_callers.py _platform_memmove`:

| | base | head |
|---|---|---|
| `_platform_memmove`, inclusive | **636 (1.84%)** | **201 (0.58%)** |
| from `cdef_arm::cdef_filter_block_16bpc_inner` | **421 (1.22%)** | **0 — absent from the caller list** |
| from `ipred_prepare::rav1d_prepare_intra_edges` | 83 | 75 |
| from `owned_recon::stitch_sbrow` | 77 | 65 |

**The change does exactly what it was built to do: CDEF's memmove traffic goes
to zero and total memmove traffic falls 66%.** It reproduces
`SIZE_SWEEP.md`'s attribution on a different vector — CDEF was 66% of this
cell's memmove samples, it measured 69% (381/549) of theirs at the same size.

And that is why the wall clock barely moves: the bytes are still copied, just
inline, so what is saved is the call/return/dispatch, not the 1.22%. **A leaf
share is an upper bound on what removing the leaf can buy, not an estimate of
it.**

**Kept, not on the number.** It is byte-identical, it strictly removes libc
calls, it costs no new machinery, and it makes `padding_16bpc` match the
`padding_8bpc` pattern whose own justification is already in the source. A
reviewer who would rather not carry an unmeasurable change should revert
`be289f4` alone; nothing else on the branch depends on it.

---

## 6. Load tag, and why paired ratios are still claimed

`measlock --load-ok` throughout, `ALLOW_LOAD=1`, foreign load recorded **per arm**
(`f_arm`) as well as per group (`f_grp`) — a per-group maximum cannot tell "one
neighbour all round" from "a neighbour during arm B".

* **0 of 540 content-sweep rows and 0 of 180 A/B rows are load-free.** Two other
  agents held 100%-CPU jobs for the whole session: a `nightly miri` and a
  `decode_permutations` release test, both at `nice 20`. `f_arm` is 1 on most
  rows, 2–5 on the rest.
* Both arms of every pair ran back-to-back inside one round, with the arm ORDER
  rotating per round, so steady foreign load is common-mode and cancels.
* **Absolutes are inflated and are not claimed.** The calibration that says by
  how little: this round's six size-ladder cells reproduce `SIZE_SWEEP.md`'s own
  n=4 idle-box ratios to within 3.4% at every point (1.12 vs 1.14, 1.15 vs 1.17,
  1.43 vs 1.48, 1.49 vs 1.56, 1.27 vs 1.31, 1.30 vs 1.31), all in the same
  direction — a different day, a different build, a busy box instead of an idle
  one, and the SHAPE is identical.
* **A harness defect cost this round its first raw TSV**, and it is worth more
  than the data: `kill`ing a `measlock` released the lock and left the process
  running, so a holder I believed dead sat out the 20-minute politeness wait
  with no lock and then started a second copy of the content sweep on top of the
  A/B, truncating the completed run's rows with `: > "$OUT"`. The derived n=9
  report survived (`benchmarks/content_gap_report_run1_2026-08-10.txt`) and the
  sweep was re-run from scratch and **stopped at 4 complete rounds** once it had
  replicated run 1, to give the box back to the profiles in §4 — so the n=9
  claim rests on run 1's derived report and the n=4 re-run is the replication,
  not a second n=9. Both are committed
  (`content_gap_report_run1_2026-08-10.txt`, `content_gap_run2_2026-08-10.tsv.zst`
  + `content_gap_report_run2_2026-08-10.txt`). `~/bin/measlock` is fixed (signal handlers now terminate and the
  payload is interruptible) and `docs/AGENT_BRIEF.md` §2 records it.

---

## 7. Gates

All on the final tree unless stated. `crates/` is **byte-identical to base**
(`git diff --quiet 5606efe..HEAD -- crates/`), and the only library file this
branch touches is `src/safe_simd/cdef_arm.rs`.

* **Vector correctness before any timing:** all 24 new AVIFs decode
  bit-identically to dav1d 1.5.4 — `decode_md5` vs `dav1d --muxer md5` over
  1-frame IVFs, **24/24 MATCH**, `benchmarks/content_vector_md5_2026-08-10.tsv`.
* **Corpus by NAME with the md5 as the value** (`scripts/perf/md5_setdiff.py`
  against the committed `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst`):
  * `--threads 1`: **766 PASS / 768 keys / 0 differing — SETDIFF: CLEAN**
  * `--threads 8`, with **`8-bit/film_grain` and `10-bit/film_grain` dropped from
    BOTH sides** per #479 (13 vectors): **753 PASS / 755 keys / 0 differing —
    SETDIFF: CLEAN**
* **Teeth, planted and confirmed failing, then restored byte-exact.** `copy_n`
  made to write `a[0]` into its last lane: `10-bit/data` **71 MISMATCH of 71**,
  and `8-bit/data` **358 PASS, 0 mismatch** — which also proves the new path is
  LIVE on 71 corpus vectors and is 16bpc-only, the change's own scope claim.
  Restored: sha256 `1c424deb…` before and after, `git diff --exit-code` rc=0.
* **`#![forbid(unsafe_code)]` proved ACTIVELY** from `src/ablate.rs`, an
  always-compiled file with no module-local attribute: the planted
  `#[allow(unsafe_code)] unsafe { 42 }` fails with *"allow(unsafe_code)
  incompatible with previous forbid … overruled by previous forbid"* anchored on
  **`lib.rs:13:12`**, plus *"usage of an `unsafe` block"*. Restored byte-exact
  (sha256 `353fac8c…`, `git diff --exit-code` rc=0).
* **Standing hazards under `--features __probe_wide`**, run even though `crates/`
  is untouched (`tracker_shard.rs` sha `d4e03d4a…`, unchanged): control 6/6 pass;
  **deleting the in-lock `state` re-read** → `wide_exclusion` FAILS;
  **`active()` cut to one shard** → `wide_exclusion` FAILS (and `shard_liveness`
  passes, so only the one test has teeth for it). Both restored byte-exact
  (sha256 + `git diff --exit-code`).
* **x86 clippy reproduced LOCALLY**, the strict form CI does not run
  (`--target x86_64-apple-darwin --all-targets --keep-going -- -D warnings`):
  11 failing sites on head, 11 on base, **set-diff empty in both directions**;
  all 11 are `_dev` examples/benches/tests plus one `src/safe_simd/mod.rs`
  cfg-dead site, none in the changed file.

Still to run at hand-off, listed so nobody assumes they passed: Miri both
models (in flight), `mt_stress` 1/2/4/8/16, `multi_decoder_pressure.sh`,
`cargo test --lib` in the DEBUG profile, and CI on the branch.

## 8. The ranked next step, from this round's numbers

1. **Profile and fix screen content.** It is 2.1–3.2x, it is 20–40% of real web
   AVIF traffic, and it has never been measured. §4 names the leaves.
2. **`ctx.rs:99`'s `t.l` half** — 22% of every registration on this ladder at the
   hump cell, removable with borrowck as the proof and no `unsafe`, priced at
   2.7% of a 1024x576 frame. Attributed by PR #488, not reduced by it. The
   blocker is shape: `CaseSet::many` takes a homogeneous `[T; N]`, so splitting
   the two directions means splitting 22 call sites or expanding each body twice.
3. **Stop selecting targets by rendition size.** Ratio-to-dav1d against
   `log10(pixels)` is R^2 = 0.001 over 30 cells. A gap number needs its content
   class and quality named next to it or it does not identify a cell.

