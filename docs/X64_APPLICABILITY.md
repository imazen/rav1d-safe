# x86_64 applicability of the 2026-08-07/08 aarch64 work

Everything below was **measured on aarch64** (Apple M4 Pro, 8P+4E, macOS 26.5.2). This file
exists so an x86_64 session can tell, without re-deriving anything, which findings are
**architecture-independent** (should port or already apply) and which are **aarch64-only**
(x86 was already correct, or the kernel does not exist there).

Do not treat any row as proven on x64. Each one names what to run.

Records: `benchmarks/verify_compose_2026-08-07.meta`, `verify_compose2_2026-08-08.meta`,
`aarch64_md5_attribution_2026-08-07/`, `p1_scaling_2026-08-07.meta`,
`p2_kernel_profile_2026-08-07.meta`, `tracker_blockshift_2026-08-08.meta`.

---

## A. Architecture-independent — these should apply to x64 directly

These live in threading, borrow-tracking, or guard-extent logic. None of them is SIMD.

### A1. Adaptive tracker block shift (the largest multi-thread win)

`BLOCK_SHIFT` in `crates/rav1d-disjoint-mut/src/tracker_shard.rs` was a **constant**. Making it a
function of the buffer, adapted once `set_parallelism(n>1)` is declared, cut shard cache-line
traffic per strided access ~4x.

- Measured aarch64: `add_slow` 72,585 -> 23,009 calls per 6-frame run (8bpc), 135,648 -> 23,966
  (10bpc), **zero** wide promotions. v4k_8tile t=8 117.9 -> 76.3 ms/frame.
- Ladder (t=8, 8bpc): shift 12 -> 119.0 ms, 13 -> 89.3, **14 -> 72.8**, 15 -> 73.0, 16 -> 75.5.
- **x64 expectation: applies, but RE-FIT THE LADDER.** The optimum is a cache-geometry property
  (line size, L2 size, prefetcher behaviour), not an ISA property. M4's 128-byte lines differ from
  x86's 64-byte lines — the aarch64 optimum of 14 may well be 15 on x64. Re-run the ladder before
  adopting the constant.
- **Known cost, also expected on x64:** +3.08% on single-tile 4K at t=8, because `block_shift_for`
  reads buffer length and cannot see tiling, so the coarse shift applies where there is no tile
  concurrency to pay for it. Fix in shape: plumb tile count / observed concurrency into the choice.

### A2. The frame-global deblock barrier in `check_tile` (the scaling plateau)

`src/thread_task.rs` gated reconstruction of frame-sbrow N **in any tile** on
`fc.frame_thread_progress.deblock` (a frame-wide monotone counter advanced by the single serial
deblock chain) reaching N-1. **dav1d has no equivalent.** Independent tile rows were serialised by
it: worker occupancy 2.86 of 8, never >= 6; the barrier was 86% of all dispatch deferrals
(771/frame, 5.7 rejected dispatches per accepted one).

- **x64 expectation: fully applicable** — this is pure task-scheduling logic, no ISA content.
- **The two costs multiply and neither pays alone.** Barrier alone 1.19x (and it re-introduces
  anti-scaling); tracker alone 2.17x; both 4.74x. The barrier was *hiding* the tracker by capping
  concurrency at ~3.2 workers. **An x64 session measuring the barrier fix in isolation will see
  ~19% and wrongly conclude it is not the problem.** Measure both together.
- **It is only sound with A3 in place.** See below — do not remove the barrier first.

### A3. Exact-window padding guards in CDEF (the precondition for A2)

Commit `fdd6a35`. The CDEF top/bottom padding loops read `x_start..x_end` but **guarded from
`offset`**, locking two columns the code never reads. When `HAVE_LEFT` is absent (`x_start == 2`)
at the left frame edge, `offset - 2` is the **tail of the previous row**, which a concurrent tile
worker legitimately writes (`backup2lines` saves whole rows). Result: false `overlapping
DisjointMut` panics under concurrent load, which the barrier had been masking.

Fix: every guard starts at `offset + x_start` and is `x_end - x_start` long.

- **x64 expectation: ALREADY PARTLY DONE, VERIFY IT.** The commit message states it fixed *"both
  bottom-edge loops in the portable/x86 file"* alongside the aarch64 ones, and that `src/cdef.rs`'s
  scalar reference has carried the discipline since the i686 report. **Confirm on x64 before
  removing any barrier.** If a guard there still starts at `offset`, A2 is unsound on x64.
- Note the direction: this **narrows** a guard to what is actually read. It removes false
  positives; it does not weaken real detection.

### A4. The sharded-tracker TOCTOU (`4af62ae`)

`add` read `state` **before** acquiring the shard lock. A wide registrant publishes into
`self.wide` and only *then* bumps `state`, so the pre-lock read could miss it — 115/18/22 missed
overlaps per ~1.4x10^9 acquisitions. Fix: re-read `state` **inside** the lock.

- **x64 expectation: fully applicable, it is a lock-ordering bug with no ISA content.**
- **Regression gate exists and is fast:** `wide_exclusion.rs::a_wide_borrow_excludes_every_narrow_shard`
  fails in **0.03 s**, deterministically (5/5 and 3/3 runs). A doc header in this repo briefly
  claimed no gate existed — that was wrong and is corrected. The 25 `soundness.rs` tests **do**
  miss it, so do not use those to conclude you are covered.
- Any new tracker fast path on x64 must not reintroduce this shape. Prove it by mutation.

### A5. Batching guard acquisitions (PR #450)

The #445 tile-threading fix shipped a 4.3-8.2% decode regression. Root cause was **not** memcpys
but `2h` extra tracker lock round-trips. Fixed by batching 4 rows per lock.
Also `ba74cc4`: one guard per tap row in the loop filter, not one per tap.

- **x64 expectation: applies.** Lock round-trip count is architecture-independent. The specific
  batch factor (4) may want re-fitting.

### A6. `CpuLevel::Scalar` does not disable safe SIMD (measurement infrastructure gap)

Nothing under `src/safe_simd/` reads `rav1d_cpu_flags_mask`; every dispatcher gates on
`archmage::<Token>::summon()`. So the documented way to get a scalar baseline **silently did
nothing**. An ablation switch had to be built (`src/ablate.rs`, `__ablate` feature, 26 guard sites).

- **x64 expectation: THE SAME HOLE ALMOST CERTAINLY EXISTS.** Check before trusting any x64
  "scalar vs SIMD" comparison — this invalidated a prior aarch64 conclusion
  (`neon_tier_isolation_2026-07-28.meta`'s "vq_suite is NOT SIMD-related at all three CPU levels";
  the premise was false, all 18 are SIMD).
- `src/ablate.rs` is written generically enough to extend to x86 tokens.

---

## B. aarch64-only — x86 was already correct

These were **bugs in the aarch64 kernels where the x86 kernel and the scalar reference agreed**.
An x64 session should **verify the x86 side matches the scalar reference** and then move on.

| Defect | aarch64 had | scalar + x86 have | Vectors |
|---|---|---|---|
| CDEF 16bpc pri_tap parity (#446) | `4 - (pri_strength & 1)` | `4 - (pri_strength >> bitdepth_min_8 & 1)` | 92 |
| `avg_8bpc` rounding | `vqrdmulhq_n_s16(sum, 2048)` = `(sum+8)>>4` | `(sum+16)>>5` | part of 80 |
| `w_avg`/`mask` 8bpc | rounded twice; tails ÷512/÷2048 | ÷256/÷1024, one shift | part of 80 |
| `w_mask` 8bpc (4 defects) | `rnd += 8192*64`; `sign` on pixel blend; 4:2:2 store dropped `sign`; 4:2:0 wrote `(m+n+1)>>1` on even rows | `PREP_BIAS`=0; raw `m+n` folded with odd row at `>>2` | part of 80 |
| itx 16x16 `eob_half` | 36 for every type | **8** for H_DCT/V_DCT (`def_fn_16x16`) | **243** |
| itx 16x32/32x16 identity | collapsed to `round2(c,2)` | exact rect2 + identity16 roundings | ~64 |
| 5x 16bpc MC kernels | hardcoded `intermediate_bits = 4` | `14 - bitdepth` (2 at 12bpc) | part of 91 |
| 16bpc `PREP_BIAS` | two conventions live at once | one (the scalar one) | part of 91 |

`bitdepth_min_8` is 0 at 8bpc, which is why several of these were invisible to every 8-bit test.
**The generic bug shape to grep for on x64: a bit-depth-conditional term dropped or mis-shifted.**
The aarch64 audit found no *further* siblings in cdef/loopfilter/looprestoration/filmgrain (all
match their scalar twins); mc and itx use `intermediate_bits` rather than that identifier, and
that is where the shape recurred five times.

Corpus effect on aarch64: **302/766 -> 766/766 PASS, 0 regressions** (set-diffed by name).
**An x64 session should run the same corpus first** — if x64 is already 766/766, section B is
informational only.

---

## C. SIMD ports — structure transfers, code does not

Real NEON tiers were written for kernels that were previously the scalar reference wearing a NEON
name. The **finding** that matters on x64 is the audit method, not the intrinsics.

- **Count real intrinsic uses per module.** Three `*_arm.rs` modules had **zero**. At HEAD,
  `looprestoration_arm.rs` (1,531 lines) **still does** — a hand-written duplicate of the
  reference, dispatched unconditionally at both depths. Run the same count on the x86 modules.
- Loop filter: all 4 widths x 2 directions x 8/10/12bpc ported; 1.074x whole-decoder at t=1 8bpc.
- CDEF: 16bpc filter + direction search; 1.019x (8bpc t=1), 1.156x (10bpc t=1).
- 16bpc itx: 10bpc t=1 508.0 -> 438.1 ms.
- **Transferable trap:** the CDEF direction search is a sum-of-squares argmax over 8 directions;
  its **tie-break must match the reference exactly** or flat blocks pick a different (still
  plausible) direction. Test on flat/synthetic content, not photos.
- **Transferable trap:** in the loop filter, getting the flat/hev **mask** wrong produces output
  correct on most pixels and wrong only at edges — passes an eyeball, fails bit-identity.

---

## D. Measured negatives — do not repeat these on x64 without a contradicting profile

| Idea | aarch64 result |
|---|---|
| `TinyLock` backoff/yield under contention | **null, measured twice** (despite `lock_slow` at 6-23% of profile) |
| `block_mut` holding mutable row guards across the kernel (halves registrations 2h->h) | **null** — 131.3 vs 128.5 ms. Correct, bit-identical, does exactly what was predicted, buys nothing |
| `CompInterType` guard drop glue as a target | **does not exist** — the 15.19% symbol is ICF-folded drop glue shared by every guard whose payload needs no drop |
| Strided tracker records | rejected by reasoning (under address-sharding a strided borrow still registers per block) — **not measured** |
| 64 shards instead of 128 | 47% of the t=8 win for 34% of the t=1 cost — strictly worse on both sides |

The `block_mut` one is the instructive negative: **halving the guard count did nothing, while
changing the shard granularity of each access was the whole win.** The cost was traffic per
access, not the number of accesses. An x64 session should reach for A1 before any count-reduction
scheme.

---

## E. Measurement traps that cost real time here

1. **Loop restoration is switched OFF in both 4K gap vectors.** Every perf number in these records
   contains **no LR at all** — yet `md5_inventory --activity` shows LR active in **696 of 768**
   corpus vectors, and 11.5% of decode self-time on `10-bit/issues/318_tx_4x4`. A benchmark grid
   can be structurally blind to a whole kernel. Check activity counters before concluding a kernel
   is cheap.
2. **`nice` on macOS maps to background QoS and lands work on E-cores** — ~40x wall-clock
   distortion. Builds may be niced; timed runs never. (x64 Linux `nice` does not do this, but the
   habit of never nicing a timed run is still right.)
3. **`--features unchecked` silently switches to FRAME threading** — checked builds hard-pin
   `n_fc=1` at `src/lib.rs:126`. A "no-tracking ceiling" arm must force `n_fc=1` too or it is
   measuring a different threading model.
4. **A busy box invalidates absolutes but not paired ratios.** One campaign had 354/360 rows taken
   under load; its ratios held to three digits while its absolute ms were inflated ~2.5%.
5. **Two-point wall fits** (2 and 20 frames) remove process startup and made our numbers agree with
   an in-process instrument to <=0.5%.
