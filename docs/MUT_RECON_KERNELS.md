# The `&mut [u8]` recon-kernel refactor, written

Round of 2026-08-09. Base `main` @ `cebb97f`. Branch `perf/mut-recon-kernels`.

PR #481 priced this change and did not write it. This writes it, on a band that
is column-compact instead of full-plane, so the memory blocker #481 named is
paid down in the same pass.

---

## 0. What is NOT done — before anything that works

* **Inter frames are not converted, at all.** The scope is frames whose
  reconstruction is entirely intra (`KEY` / `INTRA_ONLY`). `mc`, OBMC,
  inter-intra blending and warp still write the shared picture through the
  tracker, and a frame containing them declines wholesale. Every writer of a
  recon plane has to move together — intra prediction, palette, CfL and the
  inverse transform all write the same pixels — so a subset conversion leaves
  the tracker on the path and measures null. Inter is the next coherent slice,
  and it is bigger than this one (`src/mc.rs` alone has 54 `PicOffset`
  parameters against `src/ipred.rs`'s 20).
* **The filter chain is not converted.** `loopfilter`, `cdef`, `looprestoration`
  and `filmgrain` genuinely cross tile boundaries and run after the stitch on
  the unified picture, exactly as before and with exactly today's tracking.
  They are 47% of the post-#474 registration population (#481 §3) and no
  picture-buffer design reaches them.
* **`src/ctx.rs:99` is untouched and is larger than everything here.**
  2,534,988 registrations/frame at `v4k_8tile` t=8 — `CaseSetter::set_disjoint`
  on 32-byte instances, mean extent 2.1 bytes. It is not a picture buffer.
* **Also declining:** `allow_intrabc`, frame threading (`n_fc > 1`), the
  `c-ffi`/`asm` configuration, `__simd_test`, negative strides, and a frame
  with no picture data. Each leaves `f.owned_recon = false`, i.e. today's path.
* **Not measured:** x86_64 (compile-checked only), `--features unchecked`,
  t=16, any vector below 4K, and **any vector with loop restoration live** —
  #455 item 4's structural blindness is unchanged.

## 1. The design, and why it needs no run-time proof

#474 gave each TILE a private full-plane `Rav1dPictureDataComponent` hanging off
`Rav1dFrameData`. Tile tasks reach that through `fc.data.try_read()` — a SHARED
reference — so the only mutation it can express is interior, which is the
tracker. That is why #474 kept the tracker and #481 could only remove it by
setting `tracker: None`, which is unsound.

The band here is a plain `Vec<Chunk>` field of **`Rav1dTaskContext`**, the
worker's own struct, which is already `&mut` in every reconstruction signature
(`rav1d_decode_tile_sbrow`, `rav1d_recon_b_intra`, `rav1d_backup_ipred_edge`).
So `ReconPlanes::dst()` hands out a genuine `&mut [u8]`, exclusion is a
**borrowck fact**, and no record is taken at run time. `#![forbid(unsafe_code)]`
is untouched and was proved actively (§6).

The kernels take `&mut ReconDst<'_>`, a two-arm enum: `Pic` for the shared
picture (unchanged, tracked) and `Own` for the band. Rust reborrows `&mut T`
implicitly in argument position, so passing it down a dispatch chain costs no
syntax, and the two backings share ONE code path — the shared-picture arm is
byte-identical to today at every one of the 24 md5 cells in §5.

### Geometry: column-compact, one superblock row, its own stride

#474 bought a three-line seam by giving each tile buffer the picture's byte
length AND stride, so every frame-coordinate offset indexed the same pixel. The
cost was `tile_columns × plane` residency: **+96.3 MB at 8bpc, +191.0 MB at
10bpc**, measured, and #481 called that unshippable.

A kernel signature taking `(&mut [u8], base, stride)` already carries the origin
and stride a compact buffer needs, so the translation is free once the signature
is changing anyway. `recon.rs` therefore asks for a destination by **plane pixel
`(row, col)`** instead of a flat frame offset — #473's 22-site coordinate cost,
paid once, in the same change rather than twice.

The band is `max_tile_width × (sb_step * 4)` per plane, per WORKER, with its own
64-byte-aligned stride. At 4K 4:4:4 with 8 workers that is
`8 × 960 × 128 × 3 ≈ 2.9 MB` against #474's 100 MB.

**Why one superblock row is enough, and how it is enforced.** Reconstruction
never reads a picture row above the current superblock row: `src/recon.rs`
gates the top edge on `t.b.y & f.sb_step - 1 == 0` and, at a superblock-row
boundary, sources it from `f.ipred_edge` rather than the plane
(`rav1d_prepare_intra_edges`'s `prefilter_toplevel_sb_edge`). Left, top-right
and bottom-left reads are clamped to the TILE
(`ts.tiling.col_start/col_end/row_end`) — AV1 tile independence. This is
load-bearing, so `ReconBand::at` subtracts the band origin and asserts the row
is in range: an access above the band is a **panic**, never a silent wrong
pixel. Across 766 corpus vectors × {t=1, t=8} the assert never fired.

## 2. Where the registrations went — census, by name, `lost=0`

`--features probe-sites`, `v4k_8tile` 8bpc, 3 iterations, registrations per
frame. Both arms are the SAME BINARY; `RAV1D_OWNED_RECON=0` disarms the band, so
an inter-arm delta cannot be a codegen artefact.

| arm | t=1 | t=8 | distinct sites |
|---|---|---|---|
| band off (= `main`) | 7,924,706 | 22,700,725 | 54 |
| **band on** | **6,005,602** | **11,401,399** | 49 / 48 |

The off column reproduces #455's and #481's committed census for `base` **bit
for bit** (7,924,706 / 22,700,725), which is the instrument's control.

**The on column is #481's unsound ceiling arm, exactly: 6,005,602 and
11,401,399.** #481 obtained those by composing #474 with `tracker: None` on the
private planes. This reaches the same two numbers with borrowck, from `main`,
without #474 — because the same change that removes the tracker also removes the
per-row split that took `base` from 7.9 M to 22.7 M at t=8.

Removed: **1,919,104/frame at t=1 and 11,299,326/frame at t=8 (49.8%)**.

## 3. Memory — the blocker, closed

Peak RSS, `/usr/bin/time -l`, 20 frames, one decoder.

| arm | 8bpc t=1 | 8bpc t=8 | 10bpc t=1 | 10bpc t=8 |
|---|---|---|---|---|
| `main` | 99.4 MB | 106.3 MB | 149.1 MB | 157.3 MB |
| **band on** | **99.8** | **108.0** | **149.7** | **160.6** |
| #474 / #481 (measured there) | 195.7 | 202.5 | 340.1 | 348.3 |

**+1.7 MB at 8bpc t=8 and +3.3 MB at 10bpc t=8**, against #474's +96.3 and
+191.0 — 57× and 58× smaller, and the t=1 column is +0.4 / +0.6 MB.

## 4. Wall clock

See §4 of the record below; measured with `scripts/perf/verify_gap.sh`,
two-point fit at 2 and 20 frames, rotating arm order, dav1d 1.5.4
`--framedelay 1` in the same interleaved sweep, strict idle gate.

## 5. Correctness gates

* **Corpus, set-diffed BY NAME with the md5 as the value**, against
  `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst`:
  * `--threads 1`: **766 PASS / 0 mismatch / 0 error**, 768 keys,
    0 only-in-baseline / 0 only-in-head / **0 differing**.
  * `--threads 8`: **753 PASS**, 755 keys after dropping the two film-grain
    groups (#479: `rav1d_apply_grain_row` bands rows per worker while
    `filmgrain_arm.rs` takes a whole-plane mutable guard, which aborts 13 of 768
    vectors at `threads > 1` on unmodified `main`), **0 differing**.
    `md5_inventory --skip-group` was added for this and prints
    `skipped_groups=` on the TOTAL line so an empty filter cannot pass as a full
    run. Groups skipped: `8-bit/film_grain`, `10-bit/film_grain`.
  * Band-on vs band-off at both thread counts: 0 differing, both directions.
* **Frame md5 across `main` / band-on / band-off**, t=1/2/4/8, 8bpc and 10bpc:
  **24 of 24 identical** (`a00c11f454328023c58af14d55544cff` /
  `4f218411bc6ee4cc9c630fe827337fa2`).
* `mt_stress` 1/2/4/8/16 × 5: pass.
  `multi_decoder_pressure` 12 procs × 3 iters × {1,2,4,8,16}: **PASS**, every
  md5 matching the serial reference.
* **Liveness is asserted, not assumed.** An `assert!(false)` planted in
  `arm_sbrow` after `ReconBand::arm` fires on the first superblock row of
  `v4k_8tile` at t=1 — the band is on the path, before any timing is quoted.
  (Restored; the registration census in §2 is the standing proof.)

## 6. `#![forbid(unsafe_code)]`, proved actively and NON-VACUOUSLY

#481 recorded that its first proof was vacuous: an `unsafe` planted in
`src/tile_recon.rs` built cleanly because that whole module is
`#[cfg(feature = "tile-owned-recon")]`.

`src/owned_recon.rs` carries **no `cfg` on its `mod` declaration** — it is
compiled on the default feature set, which is what makes a plant there
meaningful. Two plants, on the plain default build:

| file | result |
|---|---|
| `src/owned_recon.rs:75` (the new module) | `error: usage of an 'unsafe' block`, `lib.rs:13` |
| `include/dav1d/picture.rs:384` | `error: usage of an 'unsafe' block`, `lib.rs:13` |

Both restored **byte-exact**, verified by sha256 AND `git diff --exit-code`:
`owned_recon.rs` `2372afdf…81732`, `picture.rs` `aa92199b…62875d`.

There is no compile-time feature to re-prove under: arming is a runtime switch
(`RAV1D_OWNED_RECON`) precisely so both A/B arms are one binary. The band's code
is in the default build either way.

## 7. Standing hazards, replanted

`--features __probe_wide` on `rav1d-disjoint-mut` (untouched by this branch —
`git diff cebb97f -- crates/` is empty).

| plant | `wide_exclusion` |
|---|---|
| baseline | ok (0.06 s) |
| the `4af62ae` in-lock `state` re-read deleted from **`add_contended` alone** (the CONTENDED path; the `try_lock` half is not touched) | **FAILED** |
| `active()` cut to one shard | **FAILED** |
| after both restores | ok |

Both restored byte-exact (`sha256 d4e03d4a…423176d`, `git diff --exit-code`
clean).

**Gotcha worth recording:** restoring a file with `shutil.move` preserves its
mtime, and cargo then reuses the *mutated* build artefact — the first re-run
after restore reported FAILED on a byte-identical tree. `touch` the file after
any restore that does not bump mtime, and re-run before believing a green (or a
red).

## 8. Mechanical scope of the conversion

Function signatures moved from `WithOffset<&Rav1dPictureDataComponent>` to
`&mut ReconDst<'_>` (writes) or `&ReconSrc<'_>` (reads):

| file | signatures |
|---|---|
| `src/ipred.rs` | 20 |
| `src/itx.rs` | 4 |
| `src/safe_simd/ipred.rs` (x86-64) | 3 |
| `src/safe_simd/ipred_arm.rs` | 2 |
| `src/safe_simd/itx/part10_dispatch.rs` (x86-64) | 2 |
| `src/safe_simd/itx_arm.rs` | 1 |
| `src/safe_simd/itx_wasm.rs` | 1 |
| `src/ipred_prepare.rs` | 1 |
| `include/common/dump.rs` | 2 |
| `src/recon.rs` (the seam + 15 destination constructions) | 3 |

Two kernel bodies needed more than a type change:

* **`ipred_filter_rust`** held a live READ guard on `dst` (`top`) across the
  write loop, refreshed once per row pair. An owned band cannot hand out `&` and
  `&mut` to itself simultaneously — that is the whole point — so the row is
  copied into a 64-pixel local first. Same values, same output; the copy is of a
  row this kernel wrote one iteration earlier.
* **`ipred_z3_rust`** took `dst.pixel_stride()` inside a `dst + …` expression;
  the stride is hoisted.

The `asm` dispatch arms recover the tracked `PicOffset` via `ReconDst::as_pic()`
and `expect`. That is not a latent panic: `asm` implies `c-ffi`, and
`frame_setup` refuses to arm under `c-ffi`.
