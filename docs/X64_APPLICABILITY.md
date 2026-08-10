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

---

> Sections F and G were written by different sessions. **F** (PR #493) is the first x86_64
> execution pass. **G** is the root cause and fix for the t=8 defect F found. If only one of the
> two is present in your checkout, the other is still in review.

## G. The x86-only guard-policy duplication, and the t=8 race it caused (#494)

**Verdict on A2/A3: A3's CDEF check was necessary but not sufficient, and A2's soundness argument
was true of the reference and FALSE of the x86 loop-filter window.** Fixed; the argument is now an
executable assertion rather than a comment.

### What was wrong

`src/loopfilter.rs::loopfilter_sb_direct` dispatches to a **different** SIMD entry point per
architecture, and the two sit at different levels:

| arch | entry | who sizes the guard |
|---|---|---|
| aarch64 | `loopfilter_arm::loopfilter_sb_dispatch` returns **false** | `LfBlock::open`, per fused group, from that group's own `wd` |
| x86_64 | `safe_simd::loopfilter::loopfilter_sb_dispatch` handles the whole superblock edge | the dispatcher itself, from a **constant** |

The x86 dispatcher sized a V run (horizontal edges) as `tap_before = tap_after = 7` for luma — the
widest tap span the *plane* allows. The filter's real reach is per group: the mask level is
`min(log2(tx_h) above, log2(tx_h) below)` capped at 2 (`src/lf_mask.rs`,
`masks[1][by4 + y][cmp::min(ttx, btx)]`), a transform never crosses a superblock boundary, and
`lf_reach` maps level 2/1/0 to 7/4/2 rows against 16/8/4 rows of headroom. **At every level-0 edge
in the last 4-row band of a superblock row the window therefore read 3 rows PAST the bottom of its
own superblock row** — rows owned by whoever is working on the next one.

### The writer, named

`--features probe-sites` did not reproduce (F records that as a measured negative: its per-registration
hash + three atomic RMWs perturb the window away). What does reproduce, and keeps per-record
`Loc`s, is a plain **`-C debug-assertions=on` release build**: `ShardRecs::locs` and the
`track_caller` propagation through `picture.rs`'s helpers are both `debug_assertions`-gated, so this
costs one non-atomic store per registration and no hash. It reproduced at a HIGHER rate than release — its two aborts
(`00000037`, `00000051`) were the 20th and 27th vectors attempted of `8-bit/data` at t=8, against
release's 4 in a full 358 — and named both sides:

```
 current:    & _[98304..98432] at src/safe_simd/loopfilter.rs:5134:44   <- the V-run compact read
existing: &mut _[98304..98688] at src/owned_recon.rs:937:42            <- stitch_sbrow, next sbrow

 current: &mut _[98439..98441] at src/safe_simd/loopfilter.rs:5203:25   <- an LF diff write-back
existing:    & _[98432..98448] at src/safe_simd/loopfilter.rs:5134:44   <- another LF compact read
```

So there are **two** concurrent writers, which is why #482 was correctly ruled out and yet the
failure rate went UP with `RAV1D_OWNED_RECON=0`: one pairing is `stitch_sbrow` copying the next
superblock row out of the owned band, and the other is that row's own `DeblockCols` task — which
does not involve owned recon at all, and which the barrier also used to order.

### Attribution to the barrier: yes, and it is still the wrong remedy

`054e2ed` removed a `check_tile` predicate that made reconstruction of superblock row S wait for
`deblock_progress >= S`, i.e. for `DeblockRows(S-1)`. Every candidate pairing above is downstream of
that: recon of row N+1, and `DeblockCols(N+1)` (which needs recon of N+1), both used to be ordered
after `DeblockRows(N)`. So removing it is what exposed this — **and putting it back is still the
wrong fix**, because it costs 2.19x at t=8 to paper over a 3-row over-read. The window is the defect.

### The fix, and why it cannot move a single output byte

`lf_run_reach(is_y, mask)` (new, beside `lf_reach`) returns the reach of the widest width the
run's mask can select; both BPC arms of the x86 dispatcher use it for the V window.
`lf_group_wd` is extracted from `loop_filter_sb128_rust` so the new function and its test share the
driver's ladder rather than copying it. Two things deliberately did NOT change:

* the **scalar-fallback predicate** still tests the plane worst case, so the SIMD-vs-scalar decision
  is bit-for-bit what it was, and the narrowed window is always a subset of an extent that test
  already proved in bounds;
* the **H direction** keeps the constant. Its perpendicular extent is COLUMNS of rows already inside
  this superblock row, so it has no cross-row ordering to violate, and its `tap_after` (9 luma /
  5 chroma at 8bpc) is the 4-byte chunked transpose load's rounding, not a tap bound — narrowing it
  would mean modelling the kernels' loads for no correctness gain.

### The defect is now DETERMINISTIC and single-threaded

`loopfilter_sb_direct` carries a `debug_assert!` — arch-independent, placed where the superblock
geometry is in scope — that a V run's window stays inside its superblock row. With the old constant
planted, a `-C debug-assertions=on` release build aborts on the **second vector of `8-bit/data` at
`--threads 1`**:

```
V-run window leaves the superblock row: row 380 (+7) in a 128-row superblock row,
is_y=true, mask=[f0000000, 00000000, 00000000]
```

380 % 128 = 124, i.e. the last 4-row band, with only level 0 present. Unmutated: 358/358 PASS, 0
firings. A race that needed a 10-minute t=8 corpus pass to show up once is now a 1.5-second t=1
abort — that is the reusable part of this fix.

## H. The two x86 misfits F flagged

### H1. `Shard` is 128 bytes on a 64-byte-line machine — DOCUMENTED, NOT CHANGED

`crates/rav1d-disjoint-mut/src/tracker_shard.rs` has `#[repr(align(128))] struct Shard` with
`const _: () = assert!(size_of::<Shard>() == 128 || cfg!(debug_assertions))`, and the comment says
128 is the M-series line size. On x86 (`clflush size: 64`) that means:

* **no false sharing between shards** — a 128-byte aligned 128-byte object still occupies whole
  lines, so the stated purpose of the alignment holds on x86 too;
* **but the steady-state fast path touches TWO lines.** Field offsets are `lock` 0, `live` 1..8,
  `allocated`/`mutable` 8..10, `starts[0..7]` 16..72, `ends[0..7]` 72..128. The measured steady
  state is occupancy 0-1, so the hot path reads `lock`, `live[0]`, `allocated`, `mutable`,
  `starts[0]` (line 0) and `ends[0]` at offset **72** (line 1).

Two refits would put slot 0 entirely in one x86 line, both pure layout with no semantic change:
store the records as `[(usize, usize); SLOTS]` pairs (slot 0's pair lands at 16..32), or take
`SLOTS` from 7 to 3 (`1 + 3 + 8 + 48 = 60`, one 64-byte shard). **Neither is landed and no speedup
is claimed: TCG has no cache model, so this box cannot measure either one.** The `SLOTS = 3` variant
additionally raises the shard-full rate, which pushes borrows onto the wide path — that is
measurable without timing, via `--features __probe_wide`, and should be checked before it is tried.
What is landed is the corrected comment.

### H2. `CpuLevel::X86V2` is inert in a default build — DOCUMENTED, AND FILED

`CpuLevel::X86V2` sets `SSE2 | SSSE3 | SSE41` in `rav1d_cpu_flags_mask`, and **nothing under
`src/safe_simd/` reads those three flags**: every x86 dispatcher gates on `summon_avx2()` /
`summon_avx512()` / `summon_avx512x()`. So in the default (safe-SIMD, checked) build `X86V2`
behaves exactly like `Scalar`, and a pre-Haswell x86 gets no vector kernels at all. It is not
entirely inert in every build: `--features asm` links dav1d's SSSE3/SSE4.1 asm, and `unchecked`
uses SSE2 intrinsics in msac. The doc comment now says which builds the level affects; the missing
safe-SIMD SSE tier is filed as a coverage gap rather than fixed here.
