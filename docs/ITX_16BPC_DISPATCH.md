# The 16bpc itx dispatch: what the size sweep's ranking actually named

Branch `perf/itx-hbd-wide`, base `main` @ `2fae4fe` (the sweep's brief said
`b0f70ee`; `main` had moved twice — `b0a00c3` then `2fae4fe`/#482 — and this
work is based on the newest).

Follow-on to [`SIZE_SWEEP.md`](SIZE_SWEEP.md) and issue #455. That round
measured; this one builds. Worst news first.

---

## 0. What is NOT done, and one thing the sweep named that I refuted

**Refuted before it was built: "16bpc itx above 16x16".** That is open item 5 of
issue #455 and one of the two candidates this task named. On the vectors this
campaign measures it is worth **nothing**, and I did not port it.

A shape census — an `__ablate`-gated counter at the `itxfm::Fn::call` seam,
recording `(tx_size, handled)` per call, run once and reverted — over the size
ladder and the campaign's own gap vectors:

| vector | 16bpc calls | of those, >16x16 (scalar) | scalar share of coeff area |
|---|---|---|---|
| `L3840x2160_420_10b` | 272,929 | **20** | **0.15%** |
| `L2048x1152_420_10b` | 131,067 | **0** | **0.00%** |
| `L1024x576_420_10b` | 26,448 | **9** | **1.05%** |
| `L512x288_420_10b` | 2,145 | **14** | **6.50%** |
| `v4k_8tile_10b` | 369,678 | **0** | **0.00%** |

The self-time profile agrees and is the reason I looked: at 4K 10bpc every
`src::itx_1d::*` leaf **combined** is 16 samples of 42,119 — **0.04%**. The
scalar reference is not what 16bpc itx spends its time on. `<itx::itxfm::Fn>::call`
carries 3.36% of *self* time, but that is the hbd dispatch and driver **inlined
into it**, not the fallback.

Caveat, stated plainly: this is one content class (`sips` downscales of one
photo) at one quality point (q70) plus two campaign vectors, inheriting
`SIZE_SWEEP.md`'s §3 limitation. 32x32/64x64 are common on flat and synthetic
content. **The claim here is narrow: the ladder and the gap vectors do not
support the candidate, so it is not where this session spent its time.** A
content sweep could revive it.

**Also not done:**

- **Nothing here touches 8bpc.** All three changes are inside
  `if BD::BPC == BPC::BPC16`. The 0.6–2.4 MP hump, which is `SIZE_SWEEP.md`'s
  headline and is measured at 8bpc, is **untouched**. The 8bpc arms in the
  tables below are a layout control, not a result.
- **The tracker is now the largest addressable family and I did not touch it.**
  After these changes, at 4K 10bpc, `BorrowTracker::add` is 6.66% of self time
  with `drop_glue` at 1.72%; at 1024x576 8bpc the tracker subtree is 10.13%
  inclusive, from `decode_b` 2.69%, `ctx::CaseSetter::set_disjoint` 2.09%,
  `rav1d_recon_b_intra` 1.65%, `decode_sb` 1.13%. Those are block-context
  arrays, not pixels.
- **`decode_coefs` is 54–59% of self time at both depths and is not addressed.**
  It is the entropy decoder and it is the whole ball game; see §6.
- No inter prediction, no 12bpc timing, no x86_64 run (compile only), no
  `--features unchecked`, no loop restoration (still `enable_restoration = 0`
  in every ladder vector — `SIZE_SWEEP.md` §2's blindness is inherited whole).

---

## 1. The ranking that chose the target

`sample` self-time, `L3840x2160_420_10b`, t=1, 55 s, 42,119 leaves, 2026-08-10,
load-tagged (another agent held a `miri` job at ~100% for the whole session —
see §5). Absolutes are soft; the ordering and the paired ratios are not.

| symbol | share | what it is |
|---|---|---|
| `recon::decode_coefs` | 54.84% | entropy — untouched, see §6 |
| `recon::rav1d_recon_b_intra` | 4.16% | |
| **`<itx::itxfm::Fn>::call`** | **3.36%** | hbd dispatch + driver, inlined |
| `BorrowTracker::add::<true>` | 3.13% | |
| **`itx_arm_hbd::apply1d`** | **3.11%** | the 1-D kernels |
| `BorrowTracker::add::<false>` | 3.00% | |
| **`itxfm_add_dispatch::{closure#0}`** | **1.92%** | the per-row pixel add |
| **`ReconDst::slice_mut`** | **1.27%** | the per-row guard |
| **`_platform_memset`** | **1.01%** | 0.76% of it charged to `Fn::call` |

Summed, the 16bpc itx family is **10.90% of the frame**. Three separate
overheads inside it, none of them transform math:

1. **One DisjointMut guard per row** in the pixel-add loop — `h` registrations
   per transform block. 1.92 + 1.27 + 0.91 (tracker charged to that closure's
   subtree) = **4.10%**.
2. **A 1 KiB scratch zeroed per call**, whatever the shape. 181,768 of the
   frame's 272,929 16bpc transform calls are 4x4, which needs 64 B.
3. **The driver's state arrays forced to the stack.** `[V; MAXDIM]` — 16 NEON
   vectors — indexed by a runtime bound and handed to `apply1d` as a slice of
   unknown length, so LLVM zeroed all 16 lanes per group of four and reloaded
   each element across the 1-D call. Again: the dominant shape uses 4 of the 16.

---

## 2. The three changes

| | commit | what |
|---|---|---|
| **A** | `50e878b` | the pixel-add loop uses `ReconDst::for_rows_mut` — one guard over the block's strided hull when tile threading is off, one narrow per-row guard when it is on |
| **B** | `032d917` | the scratch is sized to `w*h` (five arms, 16/32/64/128/256) instead of to `16*16` |
| **C** | `8a24835` | `inv_txfm_hbd_neon` becomes a shape dispatcher over `txfm_core::<W, H>` (nine instantiations); `apply1d` takes `&mut [V; N]` (three) |

**On A and the campaign's own `block_mut` negative.** Issue #455 records
"`block_mut` holding mutable row guards — null; halving the guard count bought
nothing." That is not this. There, guard COUNT went 2h → h and the cost per
access was unchanged. Here the count goes h → 1 *at t=1*, and the comment being
replaced justified the per-row choice with a CfL measurement — but CfL runs one
kernel call over the block, so its registration count was 1 either way. The
measurement did not transfer to a loop that runs `h` times.

Under tile threading `for_rows_mut` still takes one narrow guard per row, so
the soundness argument for concurrent tiles is unchanged, and the `v4k_8tile:8`
cells in §3 are the control that says so.

---

## 3. Measured

<!--RESULTS-->

---

## 4. Correctness

<!--GATES-->

---

## 5. Method, and what weakens these numbers

- Every timed run went through `measlock`, which now has a **`--load-ok`** mode
  (added this session, installed by atomic rename so a running `measlock` cannot
  be corrupted mid-flight — see `SIZE_SWEEP.md`'s trap 1). It keeps the mutual
  exclusion and skips the politeness wait. The previous behaviour on a box with
  a long-running non-timed job was the worst of both: wait 20 minutes, then run
  anyway.
- **The box was never idle.** Another agent held a multi-hour `miri` job at
  ~100% of one core for this whole session, and my own gate runs (`nice -n 19`,
  per the brief's E-core rule) add to it. Every row carries `foreign_max`.
  **Absolute ms/frame here is inflated and is not comparable to an idle
  campaign's; the arm-vs-arm and ours-vs-dav1d ratios are paired inside a round
  with a rotating arm order, and those are the statistic to read.**
- Two-point wall fit (`total = a + b*frames`) at per-cell frame counts, the same
  instrument as `scripts/perf/verify_gap.sh` and `SIZE_SWEEP.md`.
- No `nice` on any timed run. No `-C target-cpu=native`. dav1d 1.5.4
  `--framedelay 1` interleaved in the same sweep.
- **8bpc is a layout control, not a result.** All three changes are inside
  `BD::BPC == BPC16`, so any 8bpc movement is instruction-cache and code-layout
  noise. An early n=3 probe saw the 8bpc arm swing 198.5 → 195.1 → 199.3 ms
  across base/A/AB with disjoint bands — **±2% of pure layout noise on a depth
  the diff cannot reach**. Treat sub-2% 8bpc deltas anywhere in this repo with
  that in mind.

---

## 6. What the next round should take, by measured cost

1. **`decode_coefs`, 54.84% → 59.25% of self time.** It is the entropy decoder
   and it is more than half the frame at both depths and every size, and
   `SIZE_SWEEP.md` measured its per-pixel cost as *flat* across 3.5 decades. It
   is not the hump and it is not the depth penalty; it is simply the largest
   thing, by a factor of nine over the next symbol. Nothing in fifteen rounds
   has attacked it.
2. **The borrow tracker's context-array traffic.** 10.13% inclusive at
   1024x576 8bpc — the hump cell — and its callers are `decode_b`,
   `ctx::CaseSetter::set_disjoint`, `rav1d_recon_b_intra`, `decode_sb`. Note
   #455's standing warning: reducing registration COUNT has measured null twice;
   what paid was shard GRANULARITY. `set_disjoint` writes small fixed-size
   context arrays, which is a different shape from both.
3. **The remaining 5.07% of 16bpc itx** — `Fn::call` 1.34%, the driver 1.28%,
   `apply1d::<16>` 1.16%, `apply1d::<8>` 0.79%. Halved already; the rest is real
   transform work at 4 lanes against 8bpc's 8, which is also what dav1d does.
4. **`cdef_filter_block_16bpc_inner`'s memmoves** — 165 of the frame's 466
   `_platform_memmove` samples, and **zero** from the 8bpc twin. 16bpc CDEF is
   otherwise *faster* than 8bpc CDEF, so this is a padding/copy artefact, not
   the filter.
