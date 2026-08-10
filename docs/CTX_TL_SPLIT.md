# The `t.l` split: the left neighbour context needs no borrow tracker

Branch `perf/ctx-tl-split`, based on `main` @ `5e9975f`. Box: Apple M4 Pro
(12 cores, 8P+4E), macOS 26.5.2. Campaign index: issue #455.

**Read §0 first.** What is NOT done is more useful to the next agent than what
is.

---

## 0. What is NOT done, and what is only measured on n=1 source per class

* **The READ side is only PARTLY done.** A second commit (`f7c4e6f`, §6b)
  converted the 17 direct `t.l.<field>.index(..)` reads in
  `decode.rs`/`recon.rs`. It did **not** convert the 14
  `(a: &BlockContext, l: &BlockContext)` helpers in `src/env.rs` — that needs a
  signature change and `sm_flag`/`sm_uv_flag` are called with both roles. What
  remains is ~**8,200 registrations/frame** on screen UI q20 t=1 (`env.rs:105:72`
  4,288 + `env.rs:89:18` 3,926), ~4.3% of that arm's remaining population.
* **The read-side increment is NOT established at t=8 for screen UI, nor at
  q20 for photo** (§6b): 0.9854 at 6/11 and 1.0013 at 4/11 for UI t=8, 1.0000 at
  5/11 for photo q20. Those rows carry `foreign_max=3` — my own Miri run was on
  the box — so the nulls there may be load rather than signal, and they are
  reported as not-established rather than as nulls.
* **`decode::read_pal_indices` was profiled but NOT changed.** §5b has the
  census, and it moves the target: only ~45% of that symbol is addressable
  (`order_palette`, inlined), the other ~39% is the msac symbol decode every
  other part of the decoder pays for too.
* **One source image per content class.** `text` and `ui` are Quick Look
  renders, not screenshots (this box has no screen-recording permission), and
  each class has exactly one source. Every per-class number below is a claim
  about THAT image, and should be read that way. The corpus recipe is
  `scripts/perf/mk_content_classes.sh`.
* **Every timed row is load-tagged.** Another agent held a ~100% CPU `miri` on
  this box for the whole sweep. `measlock --load-ok` was used deliberately
  rather than letting the quiet gate wait forever (the campaign's own rule for
  exactly this case). **Paired ratios only; the absolute ms/frame below are not
  comparable to an idle campaign's.** `foreign_max` was 2 on the main sweep and
  1 on the 4K and gap sweeps.
* **The t=8 corpus leg covers 755 of 768 vectors, not 768.** Both film-grain
  groups abort the inventory process at `--threads 8` on `main` as well as on
  this branch (issue #479), and the abort kills the run *before* the 10-bit and
  12-bit groups — so the naive t=8 run reaches only 571 vectors. The
  set-diff below therefore drops `film_grain` from BOTH arms to recover the
  other 184. §5 states both runs.
* **Not measured:** x86_64 / wasm32 / linux at runtime (compile-only),
  `unchecked`, `asm` at runtime, t=2/4/16 wall clock, 10-bit and 12-bit content
  classes, any 4:4:4 screen vector, and any inter (video) content — the corpus
  here is stills, so the 11 converted sites that only fire on inter frames are
  covered by the 766-vector md5 gate but not by any wall-clock cell.

---

## 1. The change, in one paragraph

`t.l` is a field of `Rav1dTaskContext` — the worker's own struct, already `&mut`
in every decode/recon signature, nameable by no other worker. Yet every write to
it went through the borrow tracker, because `CaseSet::many` takes a homogeneous
`[T; N]` and so forced both neighbour directions — the worker-local left `t.l`
and the genuinely shared above `f.a[t.a]` — into the same reference type. This is
§4d of `docs/OWNERSHIP_MODELS.md` ("the consumer must be a single worker for the
region's whole lifetime") applied to a context array instead of a picture band:
where that holds, exclusion is a **borrowck fact with no runtime record**.

`CaseSetter::set_exclusive` is `set_disjoint`'s twin over `&mut DisjointMut` —
same bytes, same `small_memset`, same offset/len arithmetic, no registration.
`case_set_al!` splats one field list into both directions, LEFT through
`set_exclusive` and ABOVE through `set_disjoint`.

**This is not a coarsening.** `docs/BOUNDS_MAP.md` says `ctx.rs:99:27` has ZERO
widening headroom — 36 acquisitions with a concurrent foreign WRITE at gap 0. No
extent anywhere in this change moves by a byte; the reservations that remain are
byte-for-byte the ones `main` took. The count falls because half of them stop
existing, which is the one direction the bounds map leaves open.

### Why a macro and not two hand-written bodies

The two directions need different *reference types*, which no closure parameter
can abstract over without either a per-field accessor trait or a runtime branch
per field. Two hand-written copies would work and would drift: the biggest list
is twelve fields, and a field updated on one side only is a silent bitstream bug
that no type checks. The value is always a `(left, above)` pair, even where both
are the same, so the asymmetric sites (`t_dim.lh`/`lw`, `b_dim[3]`/`[2]`,
`dav1d_al_part_ctx[1]`/`[0]`) cannot be misread.

### Scope

22 `CaseSet::many` sites (11 in `src/decode.rs`, 11 in `src/recon.rs`) plus the
5 direct `t.l.*.index_mut` sites — the two `rav1d_create_lf_mask_*` calls' left
`tx_lpf_y`/`tx_lpf_uv` guards and `recon.rs`'s fused luma-coef guard.
`crates/rav1d-disjoint-mut` is **byte-identical to `5e9975f`**
(`git diff 5e9975f -- crates/` is empty).

---

## 2. The instrument that made this measurable: per-LINE attribution

Before this round the census could say `ctx.rs:99:27` is 43.9% of all
registrations and could not say *which* of the ~40 call sites. `set_disjoint`
now carries a `probe-sites`-only `track_caller`, so `Location::caller()`
propagates into the closure body's own line. Absent from the default build and
from every published feature.

The first thing it said contradicted the natural guess. On screen UI q20 t=1 the
single biggest cluster is not a coefficient site but **`decode.rs:1997..2005` —
nine lines of ONE `CaseSet::many`, 7,958 registrations each, 71,622 together,
26.6% of the whole decoder's population**. Reading a self-time profile would
never have found that; it is nine source lines inside one inlined closure.

Full per-site tables: `benchmarks/ctx_tl_split_census_site_2026-08-10.tsv.zst`.

---

## 3. Registrations removed — census, both arms, `lost=0`

`--features probe-sites`, 3 iterations, per frame.
`benchmarks/ctx_tl_split_census_tot_2026-08-10.tsv`.

| vector | t | base | head | delta | |
|---|---|---|---|---|---|
| `Cui_1024x576_q20` | 1 | 268,763 | 212,602 | −56,161 | **−20.9%** |
| `Cui_1024x576_q20` | 8 | 416,765 | 360,604 | −56,161 | −13.5% |
| `Cui_1024x576_q70` | 1 | 328,796 | 260,867 | −67,929 | **−20.7%** |
| `Ctext_1024x576_q20` | 1 | 175,096 | 134,158 | −40,938 | **−23.4%** |
| `Ctext_1024x576_q20` | 8 | 267,772 | 226,834 | −40,938 | −15.3% |
| `Ctext_1024x576_q70` | 1 | 299,936 | 224,201 | −75,735 | **−25.3%** |
| `Cphoto_1024x576_q20` | 1 | 288,028 | 264,583 | −23,445 | −8.1% |
| `Cphoto_1024x576_q20` | 8 | 710,988 | 687,543 | −23,445 | −3.3% |
| `Cphoto_1024x576_q70` | 1 | 478,459 | 359,377 | −119,082 | −24.9% |
| `photo_4k` | 1 | 2,055,056 | 1,570,898 | −484,158 | −23.6% |
| `photo_4k` | 8 | 4,241,467 | 3,757,309 | −484,158 | −11.4% |

**The removed population is thread-count invariant** — the same −56,161 at t=1
and t=8, the same −40,938, the same −484,158. That is measured, not assumed, and
it is the signature of a per-block population rather than a threading-policy one
(contrast `LfBlock::fill`, where 5.4 M of 6.4 M registrations exist *only* at
t>1).

---

## 4. Wall clock — 12 of 12 cells faster, all bands disjoint

`bench_ab_decode`, in-process `Instant` over 100 decodes, arms interleaved with
a rotating order inside every round, **n=11**.
`benchmarks/ctx_tl_split_ab_2026-08-10.tsv`; report
`scripts/perf/content_ab_report.py`.

| vector | t | base med [min..max] | head med [min..max] | head/base | faster | p |
|---|---|---|---|---|---|---|
| `Ctext…q70` | 8 | 3.491 [3.383..3.619] | 3.082 [3.029..3.105] | **0.8819** | 11/11 | 0.001 |
| `Ctext…q20` | 8 | 1.835 [1.808..1.904] | 1.631 [1.611..1.651] | **0.8888** | 11/11 | 0.001 |
| `Ctext…q70` | 1 | 5.236 [5.147..5.406] | 4.791 [4.775..4.937] | **0.9128** | 11/11 | 0.001 |
| `Ctext…q20` | 1 | 2.997 [2.943..3.045] | 2.768 [2.745..2.826] | **0.9265** | 11/11 | 0.001 |
| `Cui…q20` | 8 | 2.963 [2.889..3.003] | 2.749 [2.713..2.793] | 0.9300 | 11/11 | 0.001 |
| `Cui…q20` | 1 | 3.731 [3.677..3.858] | 3.475 [3.421..3.656] | 0.9336 | 11/11 | 0.001 |
| `Cui…q70` | 8 | 4.477 [4.408..4.522] | 4.238 [4.162..4.315] | 0.9442 | 11/11 | 0.001 |
| `Cui…q70` | 1 | 5.785 [5.723..5.853] | 5.481 [5.431..5.602] | 0.9489 | 11/11 | 0.001 |
| `Cphoto…q70` | 8 | 7.572 [7.515..7.635] | 7.227 [7.160..7.276] | 0.9553 | 11/11 | 0.001 |
| `Cphoto…q70` | 1 | 15.520 [15.490..15.630] | 14.922 [14.850..14.970] | 0.9590 | 11/11 | 0.001 |
| `Cphoto…q20` | 8 | 2.596 [2.592..2.610] | 2.505 [2.493..2.517] | 0.9644 | 11/11 | 0.001 |
| `Cphoto…q20` | 1 | 6.410 [6.360..6.445] | 6.259 [6.239..6.288] | 0.9775 | 11/11 | 0.001 |

**Every one of the 12 cells has fully disjoint base/head [min..max] bands** —
`base_min > head_max` on all 12. That is the check the campaign's own
`88.0 -> 85.6` mistake failed, and it is why these are reported as results and
not as trends.

### The campaign's own gap vector shows almost nothing

`benchmarks/ctx_tl_split_ab4k_2026-08-10.tsv`, n=9, iters=4:

| vector | t | base med [min..max] | head med [min..max] | head/base | faster | p |
|---|---|---|---|---|---|---|
| `photo_4k` | 1 | 368.492 [367.902..370.940] | 365.758 [364.990..366.923] | 0.9923 | 9/9 | 0.004 |
| `photo_4k` | 8 | 58.874 [57.396..59.508] | 59.007 [56.987..59.456] | **0.9987** | 6/9 | 0.508 |

**At 4K photo t=8 — the cell every number in this campaign was taken on — this
change measures NULL** (bands overlap, 6/9, p=0.508), while removing 484,158
registrations per frame. At the same time it is worth −11.8% on screen text at
t=8. If this round had been measured the way the campaign has been measured so
far, it would have been reported as a refutation.

---

## 5. The mechanism, and why it generalises: tracker DENSITY

A registration costs the same everywhere. What varies by two orders of magnitude
is how many of them a millisecond of frame contains.

Dividing the measured Δms by the measured Δregistrations, at t=1:

| cell | Δregs | Δms | ns per registration |
|---|---|---|---|
| `Cui…q20` | 56,161 | 0.256 | 4.56 |
| `Cui…q70` | 67,929 | 0.304 | 4.48 |
| `Ctext…q20` | 40,938 | 0.229 | 5.59 |
| `Ctext…q70` | 75,735 | 0.445 | 5.88 |
| `Cphoto…q20` | 23,445 | 0.151 | 6.44 |
| `Cphoto…q70` | 119,082 | 0.598 | 5.02 |
| `photo_4k` | 484,158 | 2.734 | 5.65 |

**4.5–6.4 ns/registration across a 13x range of content and a 42x range of frame
time** — and inside the 2.8–9.4 ns/registration band #481 measured for recon's
population, and near §11f's 4.04 ns for the filter chain's at t=8. Three
independent populations now price a registration the same. This is the first
number in the campaign that PREDICTS rather than reports:

| cell | base regs/ms | count cut | predicted | measured |
|---|---|---|---|---|
| `Ctext…q20` t=1 | 58,424 | 23.4% | 7.6% | **7.35%** |
| `Cui…q20` t=1 | 72,035 | 20.9% | 6.9% | **6.64%** |
| `photo_4k` t=1 | 5,577 | 23.6% | 0.74% | **0.77%** |

Screen text carries **10x the tracker density of a 4K photo** (58,424 vs 5,577
registrations per millisecond). That is the whole content-class effect, and it is
why the brief's regression of ours/dav1d ratio on dav1d's own ms/MP had slope
−1.02: a fixed per-block tax over a variable per-pixel denominator.

**The corollary is a rule for the next round:** *before* building a count
reduction, divide the population you intend to remove by the frame time of the
cell you intend to prove it on. Below ~10,000 registrations/ms, a 20% count cut
cannot produce 1% of wall no matter how sound it is.

---

## 5a. The profile confirms the TRACKER bucket is what shrank

Wall clock alone cannot distinguish "6% faster because the tracker shrank" from
"6% faster because something else did". `/usr/bin/sample`, 25 s window at 1 ms,
t=1, self-time bucketed by `scripts/perf/bucket_selftime.py`
(`benchmarks/ctx_tl_split_selftime_*_2026-08-10.tsv.zst`):

| class | tracker base | tracker head | `set_disjoint` as a named leaf |
|---|---|---|---|
| screen text | **17.91%** | **13.41%** | 1.16% → 0.61% |
| screen UI | **24.24%** | **19.50%** | 1.52% → 0.69% |
| photo 1024 | 11.48% | 10.12% | below 0.5% both |

Read these as SHARES inside a fixed wall window, so they understate the absolute
drop (the head arm decoded more frames in the same 25 s). The direction is the
point: the bucket that shrank is the bucket the change aimed at, and no other
bucket shrank (`kernels` and `entropy` shares rise, which is what a smaller
denominator does).

## 5b. `read_pal_indices`: profiled, and the target is HALF of it

The brief's secondary item, measured before touching it — and the census moves
the target, which is the same lesson `examples/itx_shape_census.rs` taught one
level down.

On screen UI q20 t=1, `decode::read_pal_indices` is **18.63% of self time** on
`base` (20.55% on `head`, a bigger share of a smaller total). It is the single
largest self-time leaf on that class — larger than any one tracker component.

**`order_palette` does not appear in the profile at all: 0 occurrences.** It is
fully inlined into `read_pal_indices`, so the symbol's self time is a blend. The
sample records a source line per node, so the blend can be split
(`read_pal_indices` calls only `pal_idx_finish_rust`, 2 samples, so its
inclusive-by-line histogram is a self-time proxy):

| line | share of the symbol | what it is |
|---|---|---|
| `decode.rs:766` | **44.7%** | the inlined `order_palette` wave-front |
| `decode.rs:768` | **39.3%** | `rav1d_msac_decode_symbol_adapt8`, inlined |
| `decode.rs:767` | 3.9% | the `(last..=first).rev().enumerate()` loop head |
| `decode.rs:788` | 3.7% | `pal_idx_finish` dispatch |
| `decode.rs:762` | 2.4% | the wave-front `for i` head |

So of the 18.63%, roughly **8.3% of the whole frame is `order_palette`** and
**7.3% is entropy decode** that happens to be attributed to this symbol because
it inlined. The entropy half is not a palette opportunity — it is the same msac
work the rest of the decoder pays. **Anyone porting or rewriting "the 18.6%
symbol" would be aiming at 45% of it.**

Second finding, and it is a corpus caveat: **`read_pal_indices` is 0 samples on
this corpus's `text` vector.** The brief reported 20.93% on ITS screen-text
image; on the text image here (a Quick Look render of source code) the encoder
chose no palette blocks at all, while the `ui` image is palette-heavy. Palette
share is a property of the individual image, not of "screen content" — so a
palette claim needs the specific vector named, and a palette *optimisation* needs
a corpus with several palette-heavy images before its win can be called
representative.

## 6. Where the gap to dav1d now sits, and what is left

`scripts/perf/verify_gap.sh`, two-point fit `total = a + b*frames` at 20 and 200
frames, median of 5 rounds, load-tagged.
`benchmarks/ctx_tl_split_gap_2026-08-10.tsv`.

| vector | t | dav1d `--framedelay 1` | base/dav1d | head/dav1d |
|---|---|---|---|---|
| `Ctext…q20` | 8 | 0.517 | 3.538 | **3.161** |
| `Ctext…q20` | 1 | 0.772 | 3.878 | **3.590** |
| `Cui…q20` | 8 | 1.122 | 2.614 | **2.436** |
| `Cui…q20` | 1 | 1.700 | 2.180 | **2.033** |
| `Cui…q70` | 8 | 2.039 | 2.193 | **2.071** |
| `Ctext…q70` | 8 | 1.806 | 2.028 | **1.708** |
| `Cphoto…q20` | 8 | 1.489 | 1.746 | **1.675** |
| `Cphoto…q20` | 1 | 3.539 | 1.807 | **1.768** |
| `Cphoto…q70` | 8 | 5.506 | 1.377 | **1.313** |

This independently reproduces the brief's redirect finding on a corpus built
here: at ONE pixel count the ours/dav1d ratio spans **1.38x (photo) to 3.88x
(screen text)**, which is wider than the whole size ladder's spread. The
head/base ratios from this whole-process two-point fit agree with §4's in-process
`Instant` to under 1% on every shared cell (0.9258 vs 0.9265; 0.8936 vs 0.8888;
0.9325 vs 0.9336; 0.9783 vs 0.9775) — two instruments, one answer.

### The next local population, sized

`t.l`'s READ sites, from the base census on `Cui_1024x576_q20` t=1. Each is one
half of a paired `(a, l)` read on one source line, so the per-site count below is
the LOCAL half only:

| site | regs/frame | what reads it |
|---|---|---|
| `src/recon.rs:2735` (now `:2739`) | 7,770 | `l_ccoef` read into `decode_coefs` |
| `src/decode.rs:1446:61` (now `:1437`) | 4,388 | `*t.l.skip.index(by4)` |
| `src/env.rs:105:72` | 4,288 | `get_partition_ctx`'s `l.partition` |
| `src/recon.rs:2353` | 4,265 | `t.l.lcoef` read into `decode_coefs` |
| `src/decode.rs:1682:53` | 3,979 | `t.l.mode` → `dav1d_intra_mode_context` |
| `src/env.rs:89:18` | 3,926 | `get_tx_ctx`'s `l.tx_intra` |
| **total** | **28,616** | **13.5% of `head`'s 212,602** |

At the 4.5–6.4 ns/registration measured above that is **another 0.13–0.18 ms on
a 3.475 ms cell, i.e. ~4–5%**, and it lands on the same content class. The
blocker is shape again, one level up: the 14 `(a: &BlockContext,
l: &BlockContext)` helpers in `src/env.rs` take `l` behind a shared reference, so
they cannot use `get_mut`. Both spellings work — give `l` a `&mut BlockContext`
parameter (smallest diff, reads through an exclusive reference), or split
`BlockContext` into a tracked and an untracked type (cleaner end state, bigger
diff, and `sm_flag`/`sm_uv_flag` in `ipred_prepare.rs` are called with BOTH so
they would need generalising). One gotcha found while scoping it:
`recon.rs:2011-2012` reads `t.l.filter[1]` and `t.l.filter[0]` in one
expression, which under `&mut` is two mutable borrows of one array at runtime
indices and must be sequenced into locals first.

## 6b. The read side, part one — built and priced (commit `f7c4e6f`)

Reads through `&mut` need no record either: `DisjointMut::get_mut` hands back a
plain `&mut T` and borrowck owns the exclusion. So the 17 direct
`t.l.<field>.index(..)` sites plus the three `let l_ccoef = &t.l.ccoef[pl]`
bindings that feed `decode_coefs` became `get_mut()` slices. `src/env.rs`'s 14
`(a, l)` helpers were left alone — see §0.

Two sites needed **sequencing rather than substitution**, and this is the one
non-mechanical thing in the change: `filter[0]` and `filter[1]` are elements of
ONE `[DisjointMut<_>; 2]`, so two `&mut` borrows at runtime indices cannot
coexist. `recon.rs:3274` and `obmc`'s at `recon.rs:2014` now read each into a
local first, and `obmc`'s pair is hoisted out of `mc`'s argument list (where the
borrow would have had to outlive the call). Both carry a comment saying why.

Registrations/frame at t=1, on top of the write-side arm:

| vector | write-side arm | + reads | vs `base` composed |
|---|---|---|---|
| `Cui_1024x576_q20` | 212,602 | 189,450 | 268,763 → **−29.5%** |
| `Ctext_1024x576_q20` | 134,158 | 119,983 | 175,096 → **−31.5%** |
| `Cphoto_1024x576_q70` | 359,377 | 318,145 | 478,459 → **−33.5%** |

Wall, n=11, three arms interleaved in one sweep
(`benchmarks/ctx_tl_split_ab_reads_2026-08-10.tsv`). **`foreign_max = 3` on this
sweep — my own Miri run was on the box — so it is noisier than §4's, and the t=8
cells especially.** The isolated increment (reads / write-side arm):

| vector | t | ratio | faster | verdict |
|---|---|---|---|---|
| `Ctext…q20` | 8 | 0.9552 | 11/11 | real |
| `Ctext…q70` | 1 | 0.9645 | 11/11 | real |
| `Ctext…q20` | 1 | 0.9678 | 11/11 | real |
| `Ctext…q70` | 8 | 0.9770 | 11/11 | real |
| `Cui…q20` | 1 | 0.9547 | 11/11 | real |
| `Cui…q70` | 1 | 0.9830 | 10/11 | real |
| `Cphoto…q70` | 1 | 0.9899 | 11/11 | real, small |
| `Cphoto…q70` | 8 | 0.9925 | 11/11 | real, small |
| `Cui…q20` | 8 | 0.9854 | 6/11 | **not established** |
| `Cui…q70` | 8 | 1.0013 | 4/11 | **not established** |
| `Cphoto…q20` | 1 | 1.0000 | 5/11 | **not established** |
| `Cphoto…q20` | 8 | 1.0022 | 5/11 | **not established** |

Composed against `base`, same sweep: screen text **0.846–0.900**, screen UI
0.882–0.952, photo 0.942–0.983 — 11/11 on all 12 cells.

The read side is the cheaper half per registration removed, which is what §11e
of `MUT_RECON_KERNELS.md` predicted for an immutable population ("the cheap
kind"): 23,152 registrations on `Cui…q20` t=1 bought 0.137 ms, i.e. **5.9 ns
each** — the same order as the write side, so the "cheap kind" claim is NOT
confirmed here at t=1. Where it does show is the four not-established cells, all
at t=8 or on the photo class.

---

## 7. Correctness gates

### Corpus, BY NAME, with the actual md5 as the value, at t=1 AND t=8

`examples/md5_inventory`, joined on `(group, name)` with `(status, actual_md5)`
as the value. Both arms against `base`:

| leg | rows | statuses | write-side arm | + read side |
|---|---|---|---|---|
| t=1 | 768 | 766 PASS / 2 SKIP | **0 diffs** | **0 diffs** |
| t=8, `--skip-group film_grain` | 755 | 753 PASS / 2 SKIP | **0 diffs** | **0 diffs** |
| t=8, no skip (for the record) | 571 | 569 PASS / 2 SKIP | **0 diffs** | — |

The two SKIPs are `8-bit/features/{annexb,section5}` on both arms. The 571-row
t=8 run is what the inventory produces unfiltered: the film-grain groups abort
the process (#479) before the 10-bit and 12-bit groups are reached, on `main`
identically. Records:
`benchmarks/ctx_tl_split_inv_head_{t1,t8}_2026-08-10.tsv.zst`.

Re-run after the last code-touching commit (`f17015c`): `cargo test --release
--test decode_md5_verify` 14/14, full `cargo test --release` **21 targets, 0
failures** (including `decode_permutations` 19/19, 705 s).

### The gate has teeth — planted, failed, restored

A one-token directional swap in the biggest `case_set_al!` site
(`tx_intra = (t_dim.lw, t_dim.lh)` — the two values exchanged, which is exactly
the class of bug the macro exists to prevent):

| | `decode_md5_verify` |
|---|---|
| baseline | 14 passed |
| planted swap | **12 failed, 2 passed** |
| restored | 14 passed |

Restored byte-exact: `src/decode.rs` sha256
`0f9cc569…689333f`, `git diff --exit-code` clean.

And again on the read-side arm, an off-by-one on a converted read
(`t.l.skip.get_mut()[by4 + 1]`):

| | `decode_md5_verify` |
|---|---|
| planted off-by-one | **13 failed, 1 passed** |
| restored | 14 passed |

**Gotcha worth recording, because it cost real work:** `git checkout --
src/decode.rs` to undo a plant on a tree whose read-side edits were still
UNCOMMITTED reverted those edits too, silently — the sha matched the committed
file and looked like a clean restore. Commit the change first, then plant. (The
edits were scripted, so re-applying was mechanical; had they been hand-made they
would have been gone.)

### `#![forbid(unsafe_code)]`, proved actively and non-vacuously

Four plants on the plain default build, each `unsafe { core::ptr::null::<u8>(); }`:

| file | anchor the error names |
|---|---|
| `src/ctx.rs:147` | `src/ctx.rs:1` (module-level `forbid`) |
| `src/decode.rs:3817` | `src/decode.rs:1` (module-level `forbid`) |
| `src/internal.rs:2` | **`lib.rs:13`** (the crate-level `forbid`) |

`src/internal.rs` is the one that proves the CRATE attribute: it is always
compiled and has no module-level `forbid` of its own, so nothing else could have
caught the plant. All restored, `git diff --exit-code` clean, `src/ctx.rs` sha256
`6c6ce3d7…aae26d5c`.

### And the exclusion property itself, the same way

`forbid(unsafe_code)` says no `unsafe` was written. It does not say
`set_exclusive` cannot be pointed at something shared. That is a separate claim,
and it is also a compile-time fact, so it was planted: `case_set_al!`'s ABOVE
arm switched from `set_disjoint` to `set_exclusive`.

```
error[E0596]: cannot borrow `a.lcoef` as mutable, as it is behind a `&` reference
    --> src/ctx.rs:252:35
     |
 252 |               $( case.set_exclusive(&mut a.$f $([$fi])?, $av); )*
     |                                     ^^^^^^^^^ `a` is a `&` reference, so it cannot be borrowed as mutable
     |
    ::: src/recon.rs:1513:9
```

One error per converted site — the shared above-context is reached through
`&Rav1dFrameData`, so borrowck refuses `set_exclusive` there and would refuse it
at any future site where the buffer is genuinely shared. **The design cannot be
misapplied.** Restored byte-exact (sha256 above). There is deliberately no
run-time test asserting this; a test that passes because a string matched would
prove nothing.

### Standing hazards, replanted

`--features __probe_wide`, `tests/wide_exclusion.rs`:

| plant | result |
|---|---|
| baseline | ok (0.06 s) |
| `4af62ae`'s in-lock `state` re-read deleted from `add_contended` | **FAILED** |
| `active()` cut to one shard | **FAILED** |
| after both restores | ok (0.06 s) |

Restored byte-exact: `crates/rav1d-disjoint-mut/src/tracker_shard.rs` sha256
`d4e03d4a70183660cde4ef18cde777d5ef29530501c5a0e029a524e9c423176d` — the same
digest §7 of `MUT_RECON_KERNELS.md` recorded — and `git diff --exit-code` clean.
`touch`ed after each restore, per that section's mtime gotcha.

### Threading

* `tests/mt_stress.rs` (threads 1/2/4/8/16 × 5 trials on `photo_4k`): pass.
* `scripts/perf/multi_decoder_pressure.sh`, 12 concurrent decoder processes ×
  3 iters, thread counts 1/2/4/8/16, over `photo_4k` + 4 content-class vectors:
  **PASS — all 12 md5s match the serial t=1 reference**, no wedge.

### Debug as well as release

`cargo test --lib` in DEBUG: 75 passed, 0 failed, 8 ignored. (Three integration
targets carry a `compile_error!` demanding `--release`; that is the repo's
existing structure, not this branch.)

### Compile matrix

| configuration | |
|---|---|
| default, aarch64-apple-darwin | OK + every runtime gate above |
| default, x86_64-apple-darwin, `clippy --all-targets` | OK (one pre-existing `unused import` in `tests/thread_cleanup_test.rs`, present on `main`) |
| default, wasm32-unknown-unknown | OK (compile only) |
| default, aarch64-unknown-linux-gnu | OK (compile only) |
| `--features c-ffi`, aarch64-unknown-linux-gnu | OK (compile only) |
| `--features asm`, x86_64-unknown-linux-gnu | OK (compile only) |

### Miri

`crates/rav1d-disjoint-mut` is byte-identical to `5e9975f`, so neither memory
model can regress from a change that is not there — but both were run anyway,
each target in isolation via `--no-fail-fast`. Results in §8; **a partial run is
not reported as green.**

---

## 8. Miri, both models

`cargo +nightly miri test -p rav1d-disjoint-mut --no-fail-fast`, 9 targets each.

| model | result |
|---|---|
| Stacked Borrows (`MIRIFLAGS=""`) | **9 targets, 61 passed, 0 failed**, no UB (`soundness` 992.69 s) |
| Tree Borrows (`-Zmiri-tree-borrows`) | **9 targets, 61 passed, 0 failed**, no UB (`soundness` 993.11 s) |

Both runs are COMPLETE — 9 of 9 targets each, which is what the previous round's
§7b also reported (61/61, `soundness` 992.7 s / 993.1 s). Neither run is
partial, and neither is inherited: both were re-run on this branch.
Raw: `benchmarks/ctx_tl_split_miri_2026-08-10.txt`.

`crates/rav1d-disjoint-mut` being byte-identical to `5e9975f` is why a
regression here was not expected; the runs exist because "expected" is not
evidence, and because #469 and #478 were both caught by Miri and by nothing
else.

**What Miri does NOT cover here.** Miri runs `rav1d-disjoint-mut`'s own tests,
not the decoder. This change's soundness argument is not about the tracker's
internals at all — it is that `&mut DisjointMut<_>` is exclusive, which is
borrowck's claim and is checked at every one of the 22 sites by the compiler
(see the `set_exclusive`-on-the-shared-direction plant in §7). There is no new
guard SHAPE here to run Miri against, which is the case §6 of
`OWNERSHIP_MODELS.md` says to reach for Miri on.

---

## 9. Reproducing this

```sh
cargo build --release --example avif_to_ivf --example decode_md5
scripts/perf/mk_content_classes.sh            # corpus (needs avifenc/avifdec/qlmanage/sips)
scripts/perf/content_md5.sh                   # PROVE bit-identity to dav1d first
scripts/perf/ctx_census.sh  ~/tmp/ctxtl/census    # registrations, both arms
measlock ctxtl-ab -- scripts/perf/content_ab.sh ~/tmp/ctxtl/ab1.tsv 11 100
scripts/perf/content_ab_report.py ~/tmp/ctxtl/ab1.tsv
```

`test-vectors/bench/photo_4k.avif` is gitignored; copy it in or `mt_stress` and
the 4K cells hard-fail.
