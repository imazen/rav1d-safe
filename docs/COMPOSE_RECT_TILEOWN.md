# Composing #472 (strided-rectangle record) with #474 (owned per-tile recon)

Round record. Branch `perf/compose-rect-tileown`, base `main` @ `ee07b00`.
Host: Apple M4 Pro (12 cores), macOS 26.5.2, aarch64, default features,
no `-C target-cpu=native`, no `nice` on any timed run.

Data: `benchmarks/compose_rect_tileown_2026-08-09.meta` and the four TSVs
beside it.

---

## The thesis this round was sent to test, and the answer

#474's report proposed:

> #469 removes the split from reconstruction *and* the filter chain (−65%);
> this removes it from reconstruction only (−41%). … With private recon buffers
> the filter chain is the *only* remaining split population, which is precisely
> what #469's rectangle record shrinks. **So they are complementary, not
> competing.**

**Measured: they are not complementary. They are nested.** The rectangle record
already removes the reconstruction split, so private recon buffers have almost
nothing left to remove.

Registrations per frame, `probe-sites`, `lost=0` on every row:

| arm | 8bpc t=1 | 8bpc t=8 | 10bpc t=1 | 10bpc t=8 |
|---|---|---|---|---|
| `base` (main) | 7,924,706 | 22,700,725 | 10,853,960 | 21,119,397 |
| `rect` (#472) | 7,924,706 | 8,250,328 | 10,853,960 | 10,852,772 |
| `tko` (#474) | 7,924,706 | 13,372,343 | 10,853,960 | 16,665,593 |
| **`both`** | 7,924,706 | **7,975,358** | 10,853,960 | **10,904,612** |

Above the `t=1` floor there are 14,776,019 removable registrations at 8bpc.
`rect` removes 14,450,397 of them, `tko` removes 9,328,382. If the populations
were disjoint the sum would be 23.8 M — more than exist. Composed removes
14,725,367, i.e. **274,970 more than `rect` alone, 1.9 % of what `rect`
already removed**. At 10bpc the composition is *worse* than `rect` alone by
51,840 registrations.

So on the registration axis the answer is a flat no, and the arithmetic says
why: `tko`'s population is a subset of `rect`'s.

## What the composition uniquely buys, which is not registrations

`block_mut`'s compact path survives #472. The rectangle makes the *record*
exact, which lets `compact_read_per_row` reserve the block once instead of `h`
times — but the caller still takes a **flat slice plus a stride**, so the copy
into the compact buffer and the write-back out of it both remain. Only
exclusive ownership of the rows makes the flat hull reference sound, and that
is what a private tile buffer is.

Per-site diff, `v4k_8tile` 8bpc t=8, `rect` → `both`:

| site | rect | both |
|---|---|---|
| `include/dav1d/picture.rs:2159` (compact read/write-back) | 326,810 | **0** |
| `src/recon.rs:2725/2726` (compact branch) | 188,130 ×2 | 0 |
| `src/recon.rs:2730/2731` (direct branch) | 0 | 188,130 ×2 |
| `src/tile_recon.rs:358` (per-sbrow stitch) | 0 | 25,920 |

**326,810 compact copy pairs per frame disappear.** That, not the 275 k
registrations, is the whole case for composing — and at the ledger's measured
1.00 ns per removed registration the registration delta is worth 0.27 ms/frame,
i.e. nothing.

## Memory

`/usr/bin/time -l`, 20 decodes per process, n=3, median. `v4k_8tile`.

| | 8bpc t=8 | 10bpc t=8 |
|---|---|---|
| base | 111.4 MB | 164.9 MB |
| `rect` | 111.3 | 164.8 |
| composed, **reuse** (shipped) | 212.2 (**+100.8**) | 365.2 (**+200.3**) |
| composed, release every frame | 317.1 (**+205.7**) | 567.1 (**+402.3**) |
| composed, `RAV1D_TILE_OWNED=0` | 111.4 (+0.0) | 164.8 (−0.1) |

`t=1` is untouched at every arm (feature declines below two workers).

Three findings:

1. **Releasing the buffers at every frame exit — the obvious fix for #474's
   retention — doubles peak RSS.** macOS does not hand the freed large regions
   back before the next frame has faulted a fresh set in, so two generations
   are resident at once. Kept behind `RAV1D_TILE_OWNED_RELEASE=1`.
2. **What ships instead**: reuse across frames, plus `release_all()` from
   `rav1d_flush` (which `Decoder::flush()`, a seek and end-of-stream all
   reach), plus a ceiling (`RAV1D_TILE_OWNED_MAX_MB`, default 256) checked
   before allocating. An idle decoder now holds nothing; an actively decoding
   one holds one generation; and the footprint can no longer surprise, because
   over the ceiling the frame takes the ordinary shared-picture decline path.
3. **Peak is not near baseline and cannot be brought near it in this design.**
   Resident cost is `tile_columns × plane_set_bytes`: a tile writes a narrow
   column band of *every row it owns*, a 16 KiB page spans ~4 rows of a 3840-byte
   stride, so every page in the tile's row range becomes resident even though
   the tile writes a quarter of it. 4 tile columns × 25.07 MB of planes = the
   measured 100.8 MB. Allocating only the tile's own rows does **not** help
   (`alloc_zeroed` never faults the phantom prefix in the first place), and an
   affine offset shift cannot change the stride. The only fix is a
   column-**compact** tile buffer with its own stride, which is #473's
   "coordinates, not ownership" blocker: 22 frame-coordinate recon sites plus
   ~10 `f.cur.stride[..]` reads. Not attempted here.

## Gates

* Corpus, `md5_inventory`, set-diff **by name with the actual md5 as the
  value**: `t=1` **766/766**. Threaded legs at `t=1` and `t=8`, `base` and
  `both`, **755 rows each, 0 differing** in all four (753 PASS + 2 SKIP; the
  13 film-grain vectors excluded — see below).
* Frame md5 identical across 40 cells (2 depths × t∈{1,2,4,8,16} × 4 arms):
  exactly one hash per vector.
* Full `cargo test --release --features tile-owned-recon`: **164 passed, 0
  failed**, including `decode_permutations` 19/19 and `mt_stress` 1/2/4/8/16.
* `multi_decoder_pressure.sh` 12 × 3 × {1,2,4,8,16}: PASS.
* Miri `rect_hull_aliasing`, all **9 tests run in isolation**: 9/9 clean.
* `rect_oracle` at `RECT_ORACLE_ITERS=200000`: 2/2 (1.79 s, vs 0.02 s at 2 k —
  the env var is honoured).
* Both standing hazards re-planted under `--features __probe_wide`:
  4af62ae's in-lock `state` re-read deleted → `wide_exclusion` FAILS; `active()`
  cut to one shard → FAILS. Each restored and proved byte-exact by sha256 and
  an empty `git diff`.
* `forbid(unsafe_code)` proved actively: a planted `unsafe {}` in
  `tile_recon::release` fails the build with `usage of an 'unsafe' block`.
* New mechanism gate `tests/tile_recon_lifetime.rs`, teeth by 2 mutations, each
  restored sha256-exact.

## Dead-gate audit

The brief named `md5_inventory`'s `--threads 1` default as the third dead gate
of this campaign. Grepping the siblings found the same shape in the gate the
campaign has actually been quoting:

1. **`tests/decode_md5_verify.rs` — the documented 766/766 corpus gate — runs
   entirely at `threads = 1`.** It builds a `Settings::default()`
   (`src/managed.rs:196`, `threads: 1`) and never touches it. Every "766/766"
   in this campaign, including the ones offered as cover for tile-threading
   changes, is a single-threaded number. It is the same defect as
   `md5_inventory`, in the test that CI runs.
2. **`tests/decode_permutations.rs` also runs at `threads = 1`**
   (`Settings::default()` at line 187). Defensible for its own purpose — it
   permutes SIMD tokens — but it means the permutation × tile-threading cross
   product is untested, and the loop-filter and CFL dispatchers it covers both
   branch on `needs_row_split()`. The `macos-15-intel` `compact_read_per_row`
   panic lives in exactly that cross product.
3. **`tests/tile_threading_parity.rs` has teeth** (it decodes at 1 and at 4 and
   compares) — over **three committed vectors**. That is the entire threaded
   parity corpus in the suite, against 766 single-threaded ones.

### And running the corpus threaded immediately found a live bug on `main`

`av1-1-b8-23-film_grain-50` at `--threads 8` panics:

```
overlapping DisjointMut:
 current: &mut _[0..36864] on ThreadId(3) at include/dav1d/picture.rs:736:19
existing: &mut _[0..36864]
```

Both ranges are the **same whole plane**. `rav1d_apply_grain_row` hands each
worker a row band, but `safe_simd/filmgrain_arm.rs:1550` takes
`dst_row.full_guard_mut::<BD>()` — a guard over the entire component — so two
grain-row workers hold overlapping `&mut` over the whole plane. The dead worker
then wedges the main thread's `ttd.delayed_fg.try_write().unwrap()`
(`src/thread_task.rs:534`).

**Reproduced on unmodified `main` @ ee07b00**, with none of this branch
compiled in. Not caused by the compose; not fixed here. It caps the threaded
corpus at 753 of 766 until it is fixed, which is why `md5_inventory` grew
`--skip-group` — the exclusion belongs in the invocation, where a set-diff can
see the run shrank, not inside the runner.
