# Composing #472 (strided-rectangle record) with #474 (owned per-tile recon)

Round record. Branch `perf/compose-rect-tileown`, base `main` @ `ee07b00`.
Apple M4 Pro (12 cores), macOS 26.5.2, aarch64, default features, no
`-C target-cpu=native`, no `nice` on any timed run, `foreign_max = 0` on every
committed row.

Data: `benchmarks/compose_rect_tileown_2026-08-09.meta` + six files beside it.

---

## Verdict first

**The composition is a measured regression, at both depths, at every thread
count where the feature is active.** Adding #474's private tile buffers on top
of #472's rectangle record costs **6.7–9.7 %** of wall clock, bands disjoint,
`n = 9`. The best arm in this round is **#472 alone**.

Two independent comparisons agree — one across builds (`both` vs `rect`), one
inside a single binary via the runtime switch (`both` vs `bothoff`), so the
delta cannot be a codegen artifact:

| | 8bpc t=2 | t=4 | t=8 | 10bpc t=2 | t=4 | t=8 |
|---|---|---|---|---|---|---|
| `both` / `bothoff` (one binary) | **1.067** | **1.091** | **1.091** | **1.069** | **1.076** | **1.097** |
| `both` / `rect` (two builds) | **1.055** | **1.080** | **1.097** | **1.062** | **1.082** | 1.092 ᵒ |

All disjoint except the one marked ᵒ. `t = 1` is OVERLAP everywhere — the
feature declines below two workers, as designed.

## The thesis this round was sent to test

#474's report proposed:

> #469 removes the split from reconstruction *and* the filter chain (−65 %);
> this removes it from reconstruction only (−41 %). … With private recon
> buffers the filter chain is the *only* remaining split population, which is
> precisely what #469's rectangle record shrinks. **So they are complementary,
> not competing.**

**They are not complementary. They are nested.** Registrations per frame,
`--features probe-sites`, `lost = 0` on every row:

| arm | 8bpc t=1 | 8bpc t=8 | 10bpc t=1 | 10bpc t=8 |
|---|---|---|---|---|
| `base` (main) | 7,924,706 | 22,700,725 | 10,853,960 | 21,119,397 |
| `rect` (#472) | 7,924,706 | 8,250,328 | 10,853,960 | 10,852,772 |
| `tko` (#474) | 7,924,706 | 13,372,343 | 10,853,960 | 16,665,593 |
| **`both`** | 7,924,706 | **7,975,358** | 10,853,960 | **10,904,612** |

There are 14,776,019 removable registrations above the `t = 1` floor at 8bpc.
`rect` removes 14,450,397; `tko` removes 9,328,382. If the populations were
disjoint the sum would be 23.8 M — more than exist. Composed removes
14,725,367: **274,970 more than `rect` alone, 1.9 % of what `rect` had already
removed.** At 10bpc the composition is *worse* than `rect` alone by 51,840.

`tko`'s population is a subset of `rect`'s, and the ledger's 1.00 ns per
removed registration prices the remainder at 0.27 ms/frame — nothing.

## What the composition does uniquely, and why it still loses

`block_mut`'s compact copy survives #472. The rectangle makes the *record*
exact, which lets `compact_read_per_row` reserve the block once instead of `h`
times, but the caller still takes a **flat slice plus a stride**, so the copy
in and the write-back out both remain. Only exclusive ownership of the rows
makes a flat hull reference sound — that is what a private tile buffer buys,
and it is real. Per-site diff, `v4k_8tile` 8bpc t=8, `rect` → `both`:

| site | `rect` | `both` |
|---|---|---|
| `include/dav1d/picture.rs:2159` — compact read / write-back | 326,810 | **0** |
| `src/recon.rs:2725`, `:2726` — compact branch | 188,130 ×2 | 0 |
| `src/recon.rs:2730`, `:2731` — direct branch | 0 | 188,130 ×2 |
| `src/tile_recon.rs:358` — per-sbrow stitch | 0 | 25,920 |

**326,810 compact copy pairs per frame disappear — and the frame still gets
slower.** The explanation is in the same table: the copies are not removed,
they are *relocated*. A compact read+write-back moves each block's pixels
twice; the per-sbrow stitch moves every reconstructed pixel of the frame once
more, out of the private buffer and into the picture. The traffic is roughly
conserved, and the private buffers add `tile_columns ×` the plane working set
on top. That is what the 6–10 % is.

## Gap to dav1d, n = 9

`ms/frame`, two-point wall fit (2 and 20 frames), median of 9 rounds, dav1d
1.5.4 `--framedelay 1` interleaved in the same sweep.

| 8bpc | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| base | 316.7 | 197.9 | 106.8 | 68.1 |
| **rect** | 325.3 | **177.5** | **92.9** | **56.9** |
| tko | 320.8 | 185.6 | 96.4 | 59.2 |
| both | 321.2 | 187.3 | 100.3 | 62.4 |
| dav1d | 246.4 | 125.2 | 65.6 | 36.1 |

ratio to dav1d — base 1.285 / 1.581 / 1.627 / 1.885; **rect 1.320 / 1.418 /
1.417 / 1.575**; tko 1.302 / 1.482 / 1.470 / 1.638; both 1.303 / 1.497 / 1.529
/ 1.728.

| 10bpc (ceiling raised so `both` is live) | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| base | 358.8 | 206.0 | 110.7 | 71.4 |
| **rect** | 362.1 | 191.8 | **100.1** | **63.6** |
| tko | 360.8 | 203.1 | 105.9 | 65.4 |
| both | 363.2 | 203.6 | 108.3 | 69.4 |
| dav1d | 251.4 | 127.6 | 66.8 | 37.3 |

ratio to dav1d — base 1.427 / 1.615 / 1.657 / 1.912; **rect 1.440 / 1.503 /
1.497 / 1.702**; both 1.445 / 1.596 / 1.620 / 1.859.

Scaling `t1 → t8`, 8bpc: base 4.65× · both 5.15× · tko 5.42× · **rect 5.72×** ·
dav1d 6.82×.

**Is ~1.3× met at every thread count? No — and no arm here improves on that.**
The only cell at or under the bar is 8bpc `t = 1`, where **`base` is 1.285 and
every arm in this round makes it slightly worse** (rect 1.320, tko 1.302, both
1.303, all OVERLAP against base except rect's 10bpc t=1). Best cell above t=1
is rect at 1.417 (8bpc t=4). Against the untracked ceiling
(1.160 / 1.264 / 1.262 / 1.345, #473) rect still leaves 0.16 / 0.15 / 0.15 /
0.23.

## Memory

`/usr/bin/time -l`, 20 decodes per process, n = 3, `v4k_8tile` t=8. `t = 1` is
untouched at every arm.

| | 8bpc (base 111.4 MB) | 10bpc (base 164.9 MB) |
|---|---|---|
| feature off | +0.0 | +0.0 |
| **ceiling declines** (`MAX_MB=64`) | **+0.0** | **+0.0** |
| reuse, no flush release (= #474) | +100.8 | +200.3 |
| **reuse + release on flush (shipped)** | **+201.5** | **+402.4** |
| release at every frame exit | +205.7 | +402.4 |

Four findings, and the first two contradict the round's brief:

1. **Releasing the buffers at frame exit — the fix this round was briefed to
   make — doubles peak RSS.** macOS does not hand a freed 25 MB region back
   before the next frame has faulted a fresh set in, so two generations are
   resident at once. Kept as `RAV1D_TILE_OWNED_RELEASE=1`.
2. **Releasing on flush costs exactly the same doubling**, whenever any decode
   follows the flush (the bench warms up, flushes, then decodes 20 times).
   Measured directly rather than inferred, via
   `RAV1D_TILE_OWNED_KEEP_ON_FLUSH=1`: +100.8 keeping vs +201.5 releasing.
   **On this platform there is no "free it and take it again" that is cheap.**
   The trade is real and it is a trade: retention when idle, or 2× peak when
   decoding resumes.
3. **What ships**: reuse across frames, `release_all()` from `rav1d_flush` (so
   an idle decoder holds *nothing* — the actual complaint against #474), and a
   ceiling, `RAV1D_TILE_OWNED_MAX_MB`, default 256, checked before allocating.
   The ceiling is charged on the allocation `n_tiles × plane_set_bytes`, which
   for a 2-row tiling is ~2× the resident-per-generation — so it also bounds
   the 2-generation peak to about its own value. Over the ceiling the frame
   takes the ordinary shared-picture decline path.
4. **Peak cannot be brought near baseline in this design.** Resident cost is
   `tile_columns × plane_set_bytes`: a tile writes a narrow column band of
   *every row it owns*, and a 16 KiB page spans ~4 rows of a 3840-byte stride,
   so every page in the tile's row range goes resident even though the tile
   writes a quarter of it. 4 tile columns × 25.07 MB = the measured 100.8 MB.
   Allocating only the tile's own rows does **not** help (`alloc_zeroed` never
   faults the phantom prefix), and no affine offset shift can change a stride.
   The only fix is a column-**compact** tile buffer with its own stride — which
   is #473's "coordinates, not ownership" blocker: 22 frame-coordinate recon
   sites plus ~10 `f.cur.stride[..]` reads. Not attempted.

### One arm was measuring nothing, and a counter caught it

The first sweep's `both` arm at **10bpc silently declined**: 8 tiles × ~50 MB
of 10-bit planes is 398 MB, over the 256 MB ceiling this branch adds, so the
arm ran the shared-picture path and reported a clean null. Proof —
registrations/frame at t=8 with the default ceiling: **10,852,772, byte-identical
to `rect`'s count**; with `RAV1D_TILE_OWNED_MAX_MB=1024`: 10,904,612. All 10bpc
cells were re-measured with the ceiling raised, and only then did 10bpc show the
same 6.9–9.7 % regression as 8bpc. Without the registration counter this round
would have shipped "the composition is a null at 10bpc."

## Gates

* Corpus, `md5_inventory`, set-diff **by name with the actual md5 as the
  value**: `t=1` **766/766**. Four threaded legs (`base` and `both` × `t=1` and
  `t=8`): **755 rows each, 0 differing** (753 PASS + 2 SKIP; the 13 film-grain
  vectors excluded, see below).
* Frame md5 identical over 40 cells (2 depths × t ∈ {1,2,4,8,16} × 4 arms):
  exactly one hash per vector.
* `cargo test --release --features tile-owned-recon`: **164 passed, 0 failed**,
  including `decode_permutations` 19/19 and `mt_stress` 1/2/4/8/16.
* `scripts/perf/multi_decoder_pressure.sh` 12 × 3 × {1,2,4,8,16}: PASS.
* Miri `rect_hull_aliasing`, **all 9 tests run in isolation**: 9/9 clean.
* `rect_oracle` at `RECT_ORACLE_ITERS=200000`: 2/2 (1.79 s vs 0.02 s at 2 k —
  the env var is honoured, so the 200 k really ran).
* Both standing hazards re-planted under `--features __probe_wide`: 4af62ae's
  in-lock `state` re-read deleted → `wide_exclusion` FAILS; `active()` cut to
  one shard → FAILS. Each restored, proved byte-exact by `shasum -c` and an
  empty `git diff`.
* `forbid(unsafe_code)` proved actively: a planted `unsafe {}` in
  `tile_recon::release` fails the build with `usage of an 'unsafe' block`.
* New gate `tests/tile_recon_lifetime.rs`, teeth by two mutations (delete
  `release_all` → flush assertion fails; make release unconditional → reuse
  assertion fails), each restored sha256-exact.

## Dead-gate audit

The brief named `md5_inventory`'s `--threads 1` default as this campaign's
third dead gate. The same shape is in the gate the campaign has been quoting:

1. **`tests/decode_md5_verify.rs` — the documented 766/766 corpus gate — runs
   entirely at `threads = 1`.** It builds `Settings::default()`
   (`src/managed.rs:196`, `threads: 1`) and never touches it. Every "766/766"
   in this campaign, including those offered as cover for tile-threading
   changes, is a single-threaded number. Same defect as `md5_inventory`, in the
   test CI actually runs. **Not fixed here** — fixing it means giving the test
   a caller-visible threads knob and a CI leg, and it should land on its own.
2. **`tests/decode_permutations.rs` also runs at `threads = 1`**
   (`Settings::default()`, line 187). Defensible for its own purpose, but it
   means the SIMD-permutation × tile-threading cross product is untested — and
   the loop-filter and CFL dispatchers it covers both branch on
   `needs_row_split()`. The `macos-15-intel` `compact_read_per_row` panic lives
   in exactly that cross product.
3. **`tests/tile_threading_parity.rs` does have teeth** (decodes at 1 and at 4
   and compares) — over **three committed vectors**. That is the whole threaded
   parity corpus, against 766 single-threaded ones.
4. **A fourth, and this one was already known to be silent**: the wasm32 cross
   job runs `cargo check`, which stops before the MIR pass that raises
   `arithmetic_overflow`. `crates/rav1d-disjoint-mut/src/tracker_shard.rs`
   carried a test-side `1usize << 32` that is a hard error where `usize` is 32
   bits, and the i686 legs run `cargo nextest --lib` over both workspace
   members. cfg-gated to 64-bit here, and A/B'd directly on a 32-bit target
   rather than argued: `cargo build --target wasm32-wasip1 -p
   rav1d-disjoint-mut --features std --all-targets` is clean at HEAD and, with
   the `cfg` removed, fails with `this arithmetic operation will overflow` /
   `#[deny(arithmetic_overflow)]` in the **lib test** target. `cargo check`
   with the same arguments is green in BOTH states — the third time this
   campaign that `check` has been mistaken for a build gate.
5. **`disjoint-mut CI` has been red on `main` since at least `6c17d8c`** — four
   legs: `Test (…, --no-default-features)` ×3 and `Test (ubuntu-latest,
   --all-features)`. The first three are fixed here (`extern crate std` was
   gated on `feature = "std"` and not on `test`, so the unit tests, which need
   threads and `catch_unwind`, could not compile in a `no_std` configuration:
   28 errors → 0, `cargo test -p rav1d-disjoint-mut --no-default-features`
   now runs 35 + 9 + 2 + 5 + 25 + … tests green). The fourth is partly fixed:
   `wide_exclusion`'s probe block spelled only `__probe_wide` where the lib's
   re-export of `wide_probe` also requires the sharded tracker, so
   `--all-features` (which turns on the mutually exclusive tracker selectors at
   once) failed to COMPILE the binary and took the job's other test binaries
   with it. Two lib unit tests (`test_overlapping_mut`,
   `test_new_always_tracked`) still fail there because they assert the checked
   tracker while `--all-features` selects an untracked one. **That leg is
   asking for a self-contradictory configuration** and the fix is a decision
   for the crate — either drop the leg or make the tracker-selector features
   mutually exclusive with a `compile_error!`. Not taken here.

### Running the corpus threaded immediately found a live bug on `main`

`av1-1-b8-23-film_grain-50` at `--threads 8`:

```
overlapping DisjointMut:
 current: &mut _[0..36864] on ThreadId(3) at include/dav1d/picture.rs:736:19
existing: &mut _[0..36864]
```

Both ranges are the **same whole plane**. `rav1d_apply_grain_row` hands each
worker one row band, but `src/safe_simd/filmgrain_arm.rs:1550` takes
`dst_row.full_guard_mut::<BD>()` — a guard over the entire component — so two
grain-row workers hold overlapping `&mut` over the whole plane. The dead worker
then wedges the main thread on `ttd.delayed_fg.try_write().unwrap()`
(`src/thread_task.rs:534`).

**Reproduced on unmodified `main` @ ee07b00** with none of this branch compiled
in. Not caused by the compose and not fixed here. It caps the threaded corpus at
753 of 766, which is why `md5_inventory` grew `--skip-group`: the exclusion
belongs in the invocation, where a set-diff can see the run shrink, not inside
the runner.

## What this branch is for

Not for turning `tile-owned-recon` on — the measurement says don't. It carries:

* the **compose itself**, so the question is answered with a number instead of
  an argument, and so the two mechanisms are known to be sound together
  (Miri, oracle, corpus, pressure, md5 all clean composed);
* the **lifetime and ceiling work**, which makes #474 shippable-if-ever-wanted
  and replaces its unbounded retention with a bounded one;
* the **gate fixes** — `--skip-group`, the 32-bit shift, `ARM_BIN_`/`ARM_ENV_`
  — and the audit above.

`tile-owned-recon` stays a non-default cargo feature, off at compile time.
