# Owned per-tile reconstruction buffers — issue #455, Variant 1

Round of 2026-08-09. Base `main` @ `ee07b00`, branch `perf/tile-owned-recon`.
Behind `--features tile-owned-recon` (**default off**) plus a runtime
`RAV1D_TILE_OWNED` switch, so both arms of an A/B are the same binary.

Numbers: `benchmarks/tile_owned_recon_2026-08-09.meta` + the TSVs beside it.
The feasibility work this builds on is `perf/tile-keyed`'s
`docs/TILE_KEYED_BORROWS.md` (PR #473) — §2 "Variant 1" priced the stitch and
found the locality bonus null; this round builds the thing.

---

## 1. What is being fixed

The borrow tracker's whole `t > 1` cost exists because two tile workers
legitimately write the **same picture rows at different columns**. A strided
`w x h` block borrow therefore cannot be one contiguous hull — the hull spans
inter-row gaps owned by the next tile column — so `block_mut` splits into `h`
per-row borrows plus a compact copy and a per-row write-back
(`include/dav1d/picture.rs`, the `tile_threading_active()` branch family).

That split is #455's measured 7,924,706 -> 22,700,725 registrations/frame
explosion at `v4k_8tile` 8bpc, and on top of it the per-registration cost rises
79% from t=2 to t=8 through shard contention.

AV1 guarantees tile regions do not overlap during reconstruction. Give each tile
its own buffer and the collision cannot happen — no key, no shard policy, no
`unsafe`. The exclusion is **static**, so there is nothing to prove at run time.

## 2. Why this variant and not the other two

PR #473 killed both alternatives with a specific mechanism, and neither defect
applies here:

* **Shard-by-tile-key is unsound while the filter chain is unkeyed.** A keyed
  borrow lands in its tile's shard, an unkeyed one in an address-hashed shard;
  different partitions of the same array never meet, so a real
  recon-vs-filter overlap is missed. Private buffers have no such split
  universe: a tile buffer is a **different `DisjointMut` instance**, and the
  filter chain never touches it — it runs on the picture, after the stitch,
  with exactly today's tracking.
* **The hull arm hands out a `&mut` over the hull**, so two tile columns' hulls
  on the same rows are two live overlapping `&mut` — UB whatever the tracker
  recorded. Here the hull a tile takes covers only its own buffer, which no
  other thread can reach, so the reference is as exclusive as the record.

## 3. The geometry trick — why this is a three-line seam

#473 named the blocker as **coordinates, not ownership**: 22 recon sites compute
`4 * (t.b.y * pixel_stride + t.b.x)` in FRAME coordinates plus ~10 direct reads
of `f.cur.stride[..]`, and a tight-stride tile buffer needs a different origin
*and* stride at every one of them.

This round side-steps all of it: **each tile buffer has the picture plane's byte
length AND stride.** A frame coordinate therefore indexes the same pixel in the
tile buffer as it would in the picture, and not one offset computation changes.

The seam is then exactly the three `&f.cur.data.as_ref().unwrap().data`
acquisitions reconstruction makes, replaced by
`Rav1dFrameData::recon_planes(tile_idx)`:

| site | function |
|---|---|
| `src/recon.rs:2132` | `rav1d_recon_b_intra` |
| `src/recon.rs:2822` | `rav1d_recon_b_inter` |
| `src/recon.rs:3860` | `rav1d_backup_ipred_edge` |

`src/recon.rs:1758` (`mc`) is deliberately **not** redirected: it takes
`cur_data` only for the `ref_eq` identity test at `:1775` that detects intra
block copy, and that test must keep comparing against the real picture.

#473 costed the translation-free alternative at "4x memory for the tile columns
of a tile row, ~50 MB at 4K". That estimate assumed a full-width buffer covering
only the tile's rows. Full-plane geometry is *larger* virtually and *smaller*
resident, because a tile only ever writes its own rows and the rest of the
allocation is never faulted in — see the measured RSS in the record. The buffers
are also cached across frames on a `(n_tiles, byte_len, stride)` key, so the
first-touch cost is paid once per sequence, not once per frame.

## 4. Ordering, and why the filter chain still sees whole rows

`stitch_sbrow` runs at the end of `rav1d_decode_tile_sbrow`, which returns
**before** `src/thread_task.rs` stores `ts.progress[..]`. The filter task for
superblock row N is gated on every tile having published N, so it can never
observe a partially-stitched row.

Each stitched row borrow covers exactly the tile's own columns — `w` pixels
reserved, `w` written, no inter-row gap — so two tile columns stitching the same
rows are disjoint by construction. At 4K with 4x2 tiles the stitch costs 128 row
borrows per tile per superblock row (64 luma + 2x32 chroma), i.e. 34,816 per
frame, against the 14.8 M/frame the split reconstruction path costs at t=8.

## 5. What declines to the shared picture

`setup` sets `f.tile_recon = None` — restoring today's behaviour with no other
branch — for:

* `allow_intrabc` frames (intra block copy reads the current picture as an MC
  reference; it would be stale in a tile buffer until the stitch);
* `n_fc > 1` (frame threading splits a tile into entropy and reconstruction
  passes; out of scope for this round — checked builds hard-pin `n_fc = 1`);
* fewer than 2 worker threads, or a single-tile frame — nothing to win;
* any plane with a non-positive stride, or a geometry whose bottom-right tile
  pixel is not provably inside the plane. The stitch then cannot clamp silently.

## 6. Soundness

No `unsafe` was written. `#![forbid(unsafe_code)]` was proved ACTIVELY: an
`unsafe` block planted in `tile_recon::setup` gives
`error: usage of an unsafe block` against the attribute at `lib.rs:13`;
restored, `git diff` clean.

The per-instance predicate `Rav1dPictureDataComponent::needs_row_split()`
replaces the process-global `tile_threading_active()` at the six branch sites
that choose between one hull borrow and `h` per-row ones. It answers
`tile_threading_active() && !private`, so:

* the shared picture behaves exactly as today (the filter chain, `mc`,
  everything post-stitch);
* a private component keeps the single-guard path even with the latch on.

The latch itself stays monotone and is unchanged.

## 7. Gates and their teeth

Every gate below was proved to be able to fail before it was believed.

* **Unit mechanism tests** (`src/tile_recon.rs::tests`, 4;
  `include/dav1d/picture.rs::row_guard_policy_tests`, 2 new). The load-bearing
  one asserts the stitch rectangles **exactly partition the block grid** — no
  pixel written twice, none dropped — over the real v4k_8tile tiling.
* **Teeth by mutation, x3**, each restored byte-exact (sha256 + `git diff`):
  `needs_row_split` ignoring `private` (2 tests fail), the stitch dropping the
  last row of every sbrow (2 fail), the final short sbrow rounded up instead of
  clamped (3 fail).
* **The decode-md5 gate's own teeth, measured at both ends.** Planting the
  "drop the last row" mutation into the real decoder changes the frame md5 at
  t=8 (`a00c11f4…` -> `0bcfec0b…`) and **does not change it at t=1** — because
  the feature is disabled below 2 workers by construction. So an identical md5
  at t>1 is a real result, but **a t=1 corpus run gates nothing here**, which is
  why the corpus was re-run threaded.
* **Both standing hazards re-planted** under `--features __probe_wide`: the
  4af62ae in-lock `state` re-read deleted, and `active()` cut to one shard.
  `wide_exclusion` FAILS 3/3 on each; restored byte-exact.

### A pre-existing panic the threaded corpus found

`examples/md5_inventory` gained `--threads`. It had defaulted to 1, which is
exactly the blindness #455's verification round warned about ("the 766/766
corpus gate is single-threaded, so it cannot see any tile-threading hazard").

Running it at `--threads 2` or `--threads 8` **panics on `main`'s default
build**, at the first film-grain vector (`8-bit/film_grain/av1-1-b8-23-film_grain-50`):

```
thread 'main' panicked at src/thread_task.rs:534:56:
called `Option::unwrap()` on a `None` value
```

That is `ttd.delayed_fg.try_write().unwrap()` in the delayed film-grain path.
It reproduces with **no tile-owned code compiled in**, so it is pre-existing and
independent of this change — but it means the threaded corpus can only be run
with the two film-grain groups excluded until it is fixed, and the threaded
numbers below cover 755 of 768 vectors for that reason.

## 8. What this does NOT do

* **The filter chain is untouched**, by design. It is 5.4 M of the 13.4 M
  registrations/frame that remain at t=8 and it genuinely straddles tile
  boundaries (`src/lf_apply.rs:563`, `:608`), so it needs a different mechanism.
* **The tracker is still on the reconstruction path**, just at its t=1 shape and
  on an uncontended private instance. Removing it entirely needs either
  `unsafe` (the brief forbids it, and rightly — the whole argument for this
  design is that no run-time proof is required) or a `&mut [u8]` refactor of
  every kernel signature that currently takes
  `WithOffset<&Rav1dPictureDataComponent>`. **Not attempted.**
* **Frame threading, intrabc frames, and c-ffi picture allocators** all fall
  back. `new_private_like` is `not(feature = "c-ffi")` only.
* **x86_64, Miri, `--features unchecked`, t=16 in the timed grid, and any
  vector with loop restoration live** — not run. #455 item 4's structural
  blindness is unchanged: both 4K gap vectors are intra-only with LR off.
