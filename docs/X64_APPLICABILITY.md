# x86_64 applicability of the 2026-08-07/08 aarch64 work

Everything in sections A-E was **measured on aarch64** (Apple M4 Pro, 8P+4E, macOS 26.5.2). This
file exists so an x86_64 session can tell, without re-deriving anything, which findings are
**architecture-independent** (should port or already apply) and which are **aarch64-only**
(x86 was already correct, or the kernel does not exist there).

**Section F (2026-08-10) is the first x86_64 EXECUTION pass.** It carries per-row verdicts —
including two REFUTED predictions and one x86-only defect this file did not anticipate. Read F
before acting on any A/B row: where they disagree, F ran the code.

Do not treat any A-E row as proven on x64. Each one names what to run.

Records: `benchmarks/verify_compose_2026-08-07.meta`, `verify_compose2_2026-08-08.meta`,
`aarch64_md5_attribution_2026-08-07/`, `p1_scaling_2026-08-07.meta`,
`p2_kernel_profile_2026-08-07.meta`, `tracker_blockshift_2026-08-08.meta`.

**x86_64 records this file originally failed to cite** — read them before any x86 perf work:
`benchmarks/x64_i265_gap_2026-08-08.meta`, `x64_i265_postmerge_2026-08-08.meta` and their
`x64_i265_CORRECTION_2026-08-08.md`, plus issue #458. Real Intel hardware ran this decoder two
days before section A was written, and it measured x86 **anti-scaling** plus a **+59% x86
single-thread regression** from the campaign's own tracker chain. Section F9 summarises.

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

## F. First x86_64 EXECUTION pass — 2026-08-10

Base: `main` @ `5e9975f` (the brief named `312dcc3`; `main` moved under the session — all
worktrees share one `.git`, so every number here is against `5e9975f`, verified with
`git log --oneline -1`, not against the `main` ref).

Records: `benchmarks/x64_verify_2026-08-10.meta` + `x64_verify_2026-08-10_*.tsv.zst`.
Recipe: `scripts/x64/build_and_run_x86.sh`. Instrument added: `--features __probe_x86tier`
(`src/cpu.rs::tier_census`, `examples/x86_tier_census.rs`).

> **Landed 2026-08-12, after §G and §H, and reconciled against them rather than rewritten.**
> This section was written on 2026-08-10 against `main` @ `5e9975f`; §G (PR #497) and §H were
> written and merged before it. The **one** part of F that §G supersedes is F2's open-questions
> list and its A2/A3 consequence — both carry an inline SUPERSEDED note below. Everything else
> (F0-F1, F3-F9) is unaffected: F5's zero-MISMATCH corpus verdict, F1's tier census, F3's
> film-grain both-ways measurement, F4's A6 refutation and F9's citation of the i265 record are
> all still the current position, and F7's list of what x86 still does not verify is still
> accurate on `main` — in particular the corpus leg in `ci.yml` still decodes at `threads = 1`
> and is still `continue-on-error: true`, so the CI gap F7 names is **not** closed.
> The two misfits F4/F7 flag as unmeasurable under TCG are followed up in §H1 / §H2 (the latter
> filed as #496).

### F0. What "executed on x86_64" means here — and what it cannot mean

* Host is aarch64 macOS and **Rosetta 2 is absent** (`arch -x86_64 /usr/bin/true` =>
  "Bad CPU type in executable"; no `/Library/Apple/usr/libexec/oah`). So
  `--target x86_64-apple-darwin` builds and **cannot run**. Every earlier x86 claim in this repo
  was compile-only for that reason.
* x86_64 code DID execute, in a **colima QEMU-TCG Linux VM** (`colima --profile x86 --arch
  x86_64`, 4 vCPU, 6 GiB; guest reports `QEMU TCG CPU version 2.5+`, `AuthenticAMD`,
  `clflush size: 64`).
* **Supported by this setup:** bit-exactness (per-vector md5 vs the dav1d reference), pass/fail
  BY NAME, borrow-tracker and threading behaviour, deterministic counters, symbol/static facts.
* **NOT supported: any wall-clock, ms/frame, ratio-to-dav1d, or cache-geometry number.** TCG is a
  JIT interpreter with no cache model, no store buffer, no branch predictor and no PMU. The
  campaign's gap-to-dav1d shape (1.273 / 1.332 / 1.321 / 1.474 at t=1/2/4/8, 8bpc, aarch64) is
  therefore **still unmeasured on x86_64**, and so is A1's shift ladder. See F7.
* **NOT covered: AVX-512.** TCG exposes SSE..SSE4.2, AVX, AVX2, FMA, BMI1/2, and **no AVX-512**,
  so `summon_avx512` / `summon_avx512x` refuse at every `CpuLevel`. **1,754 `_mm512_*` call sites
  went unexecuted** (mc 671, looprestoration 391, itx 390, ipred 206, cdef 51, loopfilter 45).
  Everything below is a statement about the **AVX2 tier only**.

### F1. Prove the SIMD ran before believing the corpus (`__probe_x86tier`)

Counting grants/refusals at the three x86 gates, `8-bit/data/00000001`, 3 frames, t=1:

| CpuLevel | avx2 grant | avx2 refuse | avx512 grant/refuse | avx512x grant/refuse | md5 |
|---|---|---|---|---|---|
| Scalar | 0 | 337,378 | 0 / 0 | 0 / 0 | 98b8c18a… |
| X86V2 | 0 | 337,378 | 0 / 0 | 0 / 0 | 98b8c18a… |
| X86V3 | **384,221** | 0 | 0 / 173,257 | 0 / 123,675 | 98b8c18a… |
| X86V4 | 384,221 | 0 | 0 / 173,257 | 0 / 123,675 | 98b8c18a… |
| Native | 384,221 | 0 | 0 / 173,257 | 0 / 123,675 | 98b8c18a… |

Three findings, none of them guessable from source alone:

1. The corpus run really does execute the AVX2 kernels — 384 k dispatches on a 3-frame vector.
2. **A6 is REFUTED on x86_64** (see F4).
3. **`CpuLevel::X86V2` is indistinguishable from `Scalar`.** Nothing in `safe_simd` dispatches
   below the AVX2 (`Desktop64`) token, so every `_mm_*` SSE kernel in the tree is reachable only
   through it. A pre-Haswell x86 CPU gets **no vector kernels at all** — worth knowing before
   anyone reads "x86-64-v2" in `CpuLevel` as a supported SIMD tier.

### F2. Corpus, x86_64 — and one x86-ONLY defect

`examples/md5_inventory`, all 19 meson groups, film grain applied on the two `film_grain/` groups
exactly as `decode_md5_verify` does. Same commit, same corpus, same example on both arches.

| arch | threads | PASS | MISMATCH | ERROR | SKIP |
|---|---|---|---|---|---|
| x86_64 | 1 | **766** | **0** | **0** | 2 |
| x86_64 | 8 | 762 | 0 | **4** | 2 |
| aarch64 | 1 | 766 | 0 | 0 | 2 |
| aarch64 | 8 / 16 / 32 | 753 (film_grain skipped, #479) | 0 | 0 | 2 |

The 2 SKIPs are `8-bit/features/{annexb,section5}` (not `.ivf`) on both arches. The x86 t=1 row is
two containers — `--group 8-bit/data` (358 PASS) and `--skip-group 8-bit/data` (408 PASS + the 2
SKIPs) — because a first single-container attempt stalled (F7 item 5); the union is all 19 groups,
768 rows, the same set as the aarch64 run.

**`MISMATCH` is zero on x86_64 at both thread counts. No bit-exactness defect exists on the x86
AVX2 tier** — which is the section-B verdict (F5).

**The 4 ERRORs are an x86-only threading defect this file did not predict.** Set-diffed BY NAME
against the aarch64 t=8 run, the difference is exactly:

```
8-bit/data/00000325                                              ERROR (x86) vs PASS (aarch64)
8-bit/data/00000625                                              ERROR (x86) vs PASS (aarch64)
8-bit/issues/issue_48                                            ERROR (x86) vs PASS (aarch64)
8-bit/vq_suite/Syntax_AV1_mainb8ss420_432x240_019_vq_aom_ctest_4.2  ERROR (x86) vs PASS (aarch64)
```

Each is a worker-thread `overlapping DisjointMut` panic, surfaced as
`decode error: generic error` (the post-`49df1fc0` behaviour: a dead worker fails the decode in ms
instead of wedging). The four extents:

```
current:    & _[82432..82560] (128 B)   existing: &mut _[82488..82496]  (8 B)
current: &mut _[73844..73848]  (4 B)    existing:    & _[73728..73848] (120 B)
current:    & _[16640..16752] (112 B)   existing: &mut _[16672..16688] (16 B)
current:    & _[49536..49660] (124 B)   existing: &mut _[49536..49568] (32 B)
```

Every one is an immutable READ of a picture plane holding a smaller concurrent WRITE inside it.

**The site is named.** A second x86 t=8 pass with `RUST_BACKTRACE=1` caught it with a full chain:

```
rav1d_worker_task                       src/thread_task.rs:1370
rav1d_filter_sbrow_deblock_rows         src/recon.rs:3753
rav1d_loopfilter_sbrow_rows             src/lf_apply.rs:711
filter_plane_rows_y                     src/lf_apply.rs:439
<loopfilter Fn>::call                   src/loopfilter.rs:234
loopfilter_sb_direct                    src/loopfilter.rs:103
loopfilter_sb_dispatch                  src/safe_simd/loopfilter.rs:5134   <-- x86-only file
compact_read_per_row                    include/dav1d/picture.rs:1176
slice                                   include/dav1d/picture.rs:729
```

It is the **deblock task's compact read**, issued from the x86 AVX2 loop-filter dispatcher's
`use_compact` arm (`tile_threading_active()`), which takes one guard per row of `cw` pixels —
`cw = max_iter * 4` for a V run (up to 128) and `tap_before + tap_after` for an H run (8-16).
Both widths appear in the panics, so both arms of that `if !is_v` collide. `src/loopfilter.rs:101`
routes x86 here and aarch64 to `loopfilter_arm::loopfilter_sb_dispatch` instead, so the colliding
read is issued by **arch-specific code** even though the guard helper is shared.

**At least one collision is NOT over-reservation.** From the second pass:

```
current:    & _[73728..73736] (8 B)     existing: &mut _[73728..73736] (8 B)   <-- IDENTICAL extent
current:    & _[73728..73744] (16 B)    existing: &mut _[73728..73732] (4 B)
```

A read and a write of *exactly the same eight bytes*, concurrently live. No narrowing can remove
that: it is a genuine read/write race on picture pixels, which in a `--features unchecked` build
is a silent half-written-pixel read rather than a panic. That makes this a **correctness** finding,
not just a false-positive-panic finding.

Established: the failures are real tracker panics (not emulation noise — the extents and the
backtrace are self-consistent and the same site recurs); reproducible in the aggregate across two
independent passes (**4 in 768 vectors** with defaults; **3 in the first 207** with
`RAV1D_OWNED_RECON=0`); seven distinct vectors have failed so far and the failing SET is not
stable, which is what a timing window looks like rather than a content bug; and they do NOT occur
on aarch64 in three full passes at t=8, t=16 and t=32 (`0` overlap panics each). They are
load-dependent: all four of the first pass's vectors PASS at t=8 in isolation (12 targeted
retries), so the window needs the sustained many-decoder run.

**`#482` (the tile-owned recon path) is ruled OUT as the cause** — `RAV1D_OWNED_RECON=0` did not
suppress it and in fact failed at a higher rate.

What is NOT established, and must be before anyone "fixes" it:

> **SUPERSEDED 2026-08-11 by §G (PR #497, merged as `b700489`) — all three bullets below are
> ANSWERED and the defect is FIXED.** Kept as written because the failed attempts in them are
> reusable method, not because they are the current position. §G names both writers
> (`owned_recon.rs:937:42` `stitch_sbrow` and a second LF compact read at
> `safe_simd/loopfilter.rs:5134:44`), attributes the exposure to the `054e2ed` barrier removal
> **and rejects restoring the barrier as the remedy** (2.19x at t=8 to paper over a 3-row
> over-read), and answers the third bullet in the negative: **aarch64 was not merely lucky** —
> its dispatcher returns `false` and lets `LfBlock::open` size each rectangle from the fused
> group's own `wd`, so only the x86 dispatcher ever used the plane-worst-case constant. The
> reproduction recipe the first bullet could not find is a plain `-C debug-assertions=on`
> release build (not `probe-sites`, whose per-record hash perturbs the window away — that
> negative, recorded below, is what sent §G to the right instrument).

* **Who the writer is — attempted and NOT caught.** The `existing:` record prints no location in a
  release build (no `track_caller` past the wrapper). A `--features probe-sites` build keeps
  per-record `Loc`s and would name BOTH sides; one full `--group 8-bit/data --threads 8` pass with
  that build came back **358/358 PASS, 0 overlaps**. So either the per-record `Loc` store perturbs
  the window, or ~2 hits per 358 vectors simply needs several passes. Prime suspect from the shape
  is still a tile worker's recon write into rows the deblock task is reading. (That pass is also a
  third independent x86 t=8 run with **0 MISMATCH**, so bit-exactness holds under threading even
  where the abort fires.)
* **Whether A2 (the `check_tile` deblock-barrier removal) is the enabling change.** The removed
  barrier gated *reconstruction* of sbrow N on deblock progress reaching N-1 — exactly the
  ordering whose absence would let these two overlap — and A3 already records that the barrier had
  been masking this class. There is no ablation switch left (the `probe-nodeblockgate` family went
  with the barrier in `054e2ed`), so attribution needs the barrier put back temporarily.
* **Whether aarch64 is merely lucky.** `compact_read_per_row` is shared code; only the dispatcher
  that calls it is arch-specific. `--features __probe_bounds` (`docs/BOUNDS_MAP.md`) prints each
  site's distance to the nearest concurrently-live foreign WRITE and is the designed instrument
  for deciding whether aarch64 has the same zero-gap neighbour and just never lands in the window.

**Consequence for A2/A3: the x86 verification A3 demanded is NOT satisfied.** A3 checked CDEF
padding and found it narrow (F4), and that much is true — but some other x86 site is still coarse
enough to collide, and the barrier that used to hide it is gone. Do not treat "x86 CDEF guards are
narrow" as "x86 is safe to run barrier-less".

> **SUPERSEDED 2026-08-11 by §G.** The coarse site was found — the x86 loop-filter V window — and
> narrowed to the mask-derived reach, so the four aborts are fixed rather than open. The
> *generalised* warning in the paragraph above still stands and §G restates it: A3's per-subsystem
> CDEF check was necessary but not sufficient, and "this arch's CDEF guards are narrow" was never
> the same statement as "this arch is safe barrier-less". What §G adds is an executable form of
> the argument — a `debug_assert!` in `loopfilter_sb_direct` that turns the race into a
> deterministic t=1 abort on the second vector of `8-bit/data`.

### F3. #479 (film-grain multi-thread) is aarch64-ONLY — measured both ways

This belongs in section B, and it is the first *concurrency* defect there rather than a
bit-exactness one.

| arch | film_grain groups, t=1 | t=4 | t=8 |
|---|---|---|---|
| x86_64 | 13/13 PASS | **13/13 PASS** | **13/13 PASS** |
| aarch64 | 13/13 PASS | exit 101, 0 rows | exit 101, 0 rows |

aarch64 dies with `overlapping DisjointMut` at `include/dav1d/picture.rs:742` before completing a
single vector. The source difference is exact and one-sided:

* `src/safe_simd/filmgrain.rs` (**x86**) already guards a BAND:
  `total_pixels = (bh - 1) * pixel_stride + pw` from
  `row_num * FG_BLOCK_SIZE * stride`, for `dst`, `src` and the `fguv` luma reference alike.
* `src/safe_simd/filmgrain_arm.rs` (**aarch64**) takes `full_guard_mut::<BD>()` /
  `full_guard::<BD>()` — the WHOLE picture component — per band, so N workers each reserve the
  entire plane.

So the x86 kernel has always had the shape that PR #491 introduces on aarch64, which is
independent corroboration that the narrowing is the right fix. The
`--skip-group film_grain` habit that ran through ~20 rounds of this campaign was an
**aarch64-only tax**; x86 never needed it.

The second #479 instance (the odd-width 2-pixel padding write into the input luma plane,
`src/fg_apply.rs:171-178`) is also not reachable on x86: that write is `slice_mut::<BD>(2)` inside
the current band's rows, and the x86 `fguv` luma guard covers only its own band, so the only
overlap is with the same worker, sequentially, in the same call.

### F4. Section A, per row, on x86_64

| row | verdict | evidence |
|---|---|---|
| **A1** adaptive block shift | **applies as CODE, LADDER NOT REFITTABLE HERE, and the doc named the wrong x86 knob** | `block_shift_rule` is arch-independent and its rule is `log2(len) - 8` from `BLOCKS_PER_SHARD * N_SHARDS`, not a hardcoded 14, so there is no "aarch64 constant" to port. The x86 mis-fit is elsewhere and it is explicit in source: `#[repr(align(128))] struct Shard` with `const _: () = assert!(size_of::<Shard>() == 128)` and the comment *"128 bytes is the M-series line size (`hw.cachelinesize`)… the alignment is load-bearing"*. The guest reports `clflush size: 64`, so **on x86 every shard spans TWO cache lines** and a scan that walks `lock` + `live[]` + the record arrays touches both. The x86 refit knobs are therefore `SLOTS` (7 records/shard) and `BLOCKS_PER_SHARD`, NOT the shift rung. Unmeasurable under TCG — needs a real x86 box. |
| **A2** `check_tile` deblock barrier | **code confirmed removed and arch-independent — but NOT proven sound on x86** *(as of 2026-08-10; **UPDATED 2026-08-11 by §G**: the one unsound x86 site this pass found is fixed, and the barrier is explicitly NOT the remedy)* | `src/thread_task.rs:589` carries the deliberate "there is no frame-global deblock barrier here" note. See F2: four t=8 aborts of exactly the class A3 says the barrier used to mask — root-caused and fixed in §G. |
| **A3** exact-window CDEF padding guards | **CONFIRMED narrow on both x86-relevant files** | `src/safe_simd/cdef.rs` (portable, compiled on x86): top loop takes `left_ext = 0` when `!HAVE_LEFT` so the guard starts at `offset`, not `offset - 2`; both bottom loops (8bpc `:1392,1400`, 16bpc `:1851,1858`) take `bottom_row.offset + x_start` for `x_end - x_start`. Scalar `src/cdef.rs:487,514,516` likewise. |
| **A4** sharded-tracker TOCTOU | **CONFIRMED, and the gate is proven to have TEETH on x86** | `crates/rav1d-disjoint-mut/tests/wide_exclusion.rs` passes on x86_64 in 0.61 s. Mutation planted (in-lock `state` re-read at `tracker_shard.rs:1476` replaced by `if false`), rebuilt, **FAILED 3/3 runs**; mutation reverted, green again. |
| **A5** guard batching | **CONFIRMED present and arch-independent** | `LF_BATCH_MAX = 4` at `src/loopfilter.rs:345`, outside any `target_arch` gate; `LfBlock::close` writes back only `changed_span` per row. Re-fitting the factor is a timing question => not answerable here. |
| **A6** `CpuLevel::Scalar` does not disable safe SIMD | **REFUTED on x86_64** | The three x86 gates (`cpu.rs::summon_avx2/summon_avx512/summon_avx512x`) each test `simd_enabled(...)`, i.e. `rav1d_cpu_flags_mask`, BEFORE summoning — 78 + 78 + 9 call sites, and nothing in `safe_simd/*.rs` summons an x86 token directly except four `#[cfg(all(feature = "asm", target_arch = "x86_64"))]` FFI wrappers in `safe_simd/loopfilter.rs:4188-4323`, which the file's own AUDITED banner records as having **no callers** (under `asm` the table resolves to the NASM symbol, not to these). Measured in F1: `Scalar` gives 0 grants / 337,378 refusals where `Native` gives 384,221 grants. **An x64 scalar-vs-SIMD A/B is valid**, unlike the aarch64 one; the `__ablate` feature is not needed on x86 (it stays the right tool for per-FAMILY ablation). |

### F5. Section B verdict: CONFIRMED for the AVX2 tier

x86_64 is **0 MISMATCH across the whole corpus at t=1 and t=8**, including every group the eight
aarch64 defects were found in (`8-bit/data` 358, `8-bit/size` 100, `8-bit/quantizer` 64,
`10-bit/data` 71, `10-bit/quantizer` 64, `12-bit/data` 46, `8-bit/vq_suite` 18). All eight rows in
section B were aarch64-only, as claimed, and section B is now informational.

Two boundaries on that verdict: it covers the **AVX2** tier only (F0 — AVX-512 unexecuted), and
`12-bit` coverage is the corpus's own 47 vectors, all 160x90 or thereabouts.

### F6. Section C on x86: the audit method ports, and it finds one hole

Real `_mm*` call sites per x86-dispatched module (`_mm_` / `_mm256_` / `_mm512_`):

| module | lines | sites | SSE / AVX2 / AVX-512 |
|---|---|---|---|
| `mc.rs` | 13,196 | 1,628 | 79 / 878 / 671 |
| `loopfilter.rs` | 5,684 | 1,054 | 727 / 282 / 45 |
| `looprestoration.rs` | 5,933 | 886 | 14 / 481 / 391 |
| `ipred.rs` | 7,130 | 478 | 36 / 236 / 206 |
| `cdef.rs` | 2,698 | 251 | 190 / 10 / 51 |
| `filmgrain.rs` | 2,767 | 131 | 10 / 121 / 0 |
| `pal.rs` | 200 | 17 | 8 / 9 / 0 |
| `refmvs.rs` | 59 | 2 | 2 / 0 / 0 |
| `itx/part*.rs` (10 files) | 24,277 | 2,672 | 785 / 2,272 / 390 |

So x86 has no `looprestoration_arm.rs`-shaped hole — but it does have **`filmgrain.rs` with zero
AVX-512 sites** and, more interestingly, the whole x86 SSE surface is unreachable without an AVX2
token (F1, finding 3).

### F7. Still NOT verified on x86, and the infrastructure that would close it

Named first, because none of it is closeable on this hardware:

1. **The gap to dav1d on x86_64 — not measured BY THIS PASS, but it was already measured on real
   x86 hardware in this repo. See F9: the shape is materially different and the campaign's own
   tracker chain regressed x86 single-thread by +59%.** Nothing new can come out of TCG.
2. **A1's shift ladder and the `SLOTS`/128-byte-shard question — NOT MEASURED**, same reason.
3. **#482's seam cost on x86 (0.3-1.3% on aarch64) — NOT MEASURED.** What IS checked: no
   out-of-line `ReconSrc::slice` symbol exists in either the x86 or the aarch64 release binary at
   this commit (`nm | grep owned_recon` returns the same 8 symbols on both), so the #483 symptom —
   the accessor going out-of-line — does not appear on x86 either. Whether the owned path is
   actually TAKEN on x86 is unproven; prove it with `--features probe-sites` and a
   picture-plane registration-count diff between `RAV1D_OWNED_RECON=0` and the default.
4. **The AVX-512 tier: entirely unexecuted** (1,754 sites). No `__simd_test` differential, no
   corpus pass.
5. **One unexplained x86 stall.** A t=1 corpus pass stopped making progress on
   `8-bit/data/00001114` (34 ms on aarch64) with **0.00% CPU, exactly one thread, wchan
   `__do_sys_pause`** — and the static binary contains no call site to `pause` at all, so the
   wait channel is not explicable from our code. The same vector passes in isolation and the
   re-run passed the same group. Most likely a colima/virtiofs/TCG artifact; recorded rather than
   attributed, because it is the one observation here I cannot pin down.

**What would close 1-4.** Be precise about what CI already has, because "add an x86 job" is the
wrong ask: `.github/workflows/ci.yml` ALREADY runs `--test decode_md5_verify` on
`ubuntu-latest` (x86_64) as well as `ubuntu-24.04-arm`, and also has `SIMD Permutation Tests
(x86_64)`, `Conformance (decode permutations, x86_64)`, `Tile Threading` and `Threading race
gates` on x86 runners. Two properties are why F2 slipped through anyway:

* **`decode_md5_verify` decodes at `threads = 1`** (`Settings::default().threads == 1`,
  `src/managed.rs:196`) — and x86 is 766/766 clean at t=1, so a single-threaded corpus leg
  structurally cannot see this.
* **that step is `continue-on-error: true`** (ci.yml:137,142), so even a failure would not gate.

So the cheap fix is one extra step on the existing x86 leg — the corpus at `--threads 8`, e.g.
`examples/md5_inventory --threads 8` with a non-zero exit on any ERROR row — plus dropping
`continue-on-error` once it is green. That would have caught F2's four aborts. The perf rows (1-3)
need a dedicated bare-metal x86 host — the i265 rig of F9, or equivalent — run with the same
discipline as the aarch64 side (`measlock`, interleaved arms, median >= 5, dav1d in the same
sweep). Until then, "verified on x64" can honestly mean **single-thread correctness on the AVX2
tier** and nothing more.

### F8. Also settled on the way past

* **`c-ffi` on x86 Linux WORKS.** CI's exact leg
  (`cargo clippy --no-default-features --features "c-ffi,bitdepth_8,bitdepth_16" -- -D warnings`)
  exits 0 for `--target x86_64-unknown-linux-musl`, and the same command on the macOS host fails
  with exactly the documented `error[E0080] assertion failed: Rav1dError::EAGAIN as u8 ==
  libc::EAGAIN as u8` — so that blocker is host-OS (Darwin `EAGAIN` 35 vs Linux 11), not
  architecture. Adding `--all-targets` (which CI does not use) surfaces 5 latent lints on x86:
  `token_test_lock` unused in the lib-test cfg, and 4 in `examples/{itx_shape_census,md5_ablate}.rs`.
* **Rebuilding into a path a running measurement reads is the same mistake as editing a running
  shell script.** A `cargo build` mid-session replaced `examples/md5_inventory` under three live
  container runs. They survived (the old inode stays mapped), but the arm identity was destroyed;
  distinct `_plain` / `_sites` copies are the fix.

### F9. The premise "x86 has only ever been compile-checked" is FALSE — and the x86 shape IS different

**Every number in this subsection is CITED from a committed record, not measured by this pass.**
Source: `benchmarks/x64_i265_gap_2026-08-08.meta` + `x64_i265_postmerge_2026-08-08.meta` +
`x64_i265_CORRECTION_2026-08-08.md` + issue #458. Host: **i265 — Intel Core Ultra 7 265K**
(Arrow Lake, 20C/20T, single-channel DDR5-6000, Ubuntu 26.04), idle, dedicated, dav1d
1.5.3-46-g1718ff9a built on-box, `scripts/perf/ab_sweep.sh` 3 rounds x 3 reps interleaved. So real
x86_64 hardware DID run this decoder, two days before sections A-E were written — and **sections
A-E cite none of those files.** Read them before planning any x86 perf work.

What that record says, restricted to the arms its own CORRECTION leaves standing
(`main` = `a6a7e232` and `pre445`; the `audit445` and `attrib` arms are VOID — a silent
`git checkout` refusal rebuilt pre-merge main for both, verified byte-identical):

* **Single thread, x86 is NARROWER than aarch64 vs dav1d:** 1.51x at 8bpc and 1.99x at 10bpc
  (v4k_8tile), against the same-week aarch64 reference's 1.62x / 2.01x.
* **But x86-64 tile threading ANTI-SCALES, on every genuine arm.** v4k_8tile 8bpc ms/frame:
  t=1 220.1, t=2 428.7, t=4 590.6, t=8 622.2 — **t1->t8 = 0.35x**, i.e. t=8 is 2.8x SLOWER than
  t=1. dav1d on the same box scales 4.46x and saturates at t=8 (32.7 ms/f). **The gap therefore
  goes 1.5x at t=1 to 19.0-20.8x at t=8**, where the aarch64 figure for the same measurement is
  3.44x. That is the materially-different x86 shape, and it is a far bigger deal than any row in
  section A.
* **The campaign's tracker work costs x86 +59% at t=1** (issue #458, and the CORRECTION's
  re-bisect): 220.6 -> 350.8 ms/frame across the compose-2 merge, 10bpc 275 -> 412 (+50%),
  bisected to **the sharded-tracker chain itself**, with aarch64 t=1 flat (0.994) over the same
  merge. Post-merge the x86 ST gap is therefore ~2.4x dav1d. **The base this pass verified
  (`5e9975f`) is post-merge, so that regression is in the code above.**
* Also standing from that record, and consistent with section F here: frame md5 identical across
  arms x vectors x t{1,2,4,8} on x64, and #446's CDEF `pri_tap` divergence does not exist on x64.

Consequences for section A, which must not be read as-is any more:

* **A1's "x64 expectation: applies, but RE-FIT THE LADDER" is too optimistic in the wrong
  direction.** On x86 the sharded-tracker chain that A1 belongs to is a measured net LOSS at t=1
  (+59%), and the MT win it buys on aarch64 has no x86 counterpart to buy because x86 anti-scales
  before the tracker is reached.
* **A2's "fully applicable" is unproven on x86.** The record's `attrib` arm (which composed the
  barrier removal) is VOID, so nobody has a valid x86 measurement of the barrier fix at all — and
  F2 above shows the barrier's removal is what the four t=8 aborts sit behind.
* **A5's batch factor** likewise: the only x86 datum ("audit445's batching is noise on x64") is
  VOID.

The honest x86 status of this campaign is therefore: **correctness verified on the AVX2 tier
(F1-F6); performance UNVERIFIED and pointing the wrong way on the one real x86 record that
exists.** A dedicated x86 box (the i265 rig, or a bare-metal CI runner) is the only way forward,
and the first thing to run on it is not a ladder — it is why x86 anti-scales at t=2.


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
