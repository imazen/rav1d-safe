# A true zero-tracker recon path — priced before it is built

Round of 2026-08-09. Base `main` @ `cebb97f`, composed with PR #474
(`perf/tile-owned-recon`, owned per-tile reconstruction buffers).

The brief for this round named the remaining step after #474 precisely, quoting
#474's own author:

> True zero-tracker needs either `unsafe` (forbidden) or a `&mut [u8]` refactor
> of every kernel signature taking `WithOffset<&Rav1dPictureDataComponent>`.
> **NOT ATTEMPTED.**

and scoped it at ~27 kernel call sites plus ~40 `with_pixel_guard_mut`
closures, with the untracked ceiling (1.160 / 1.264 / 1.262 / 1.345 at
t=1/2/4/8, 8bpc) as the prize.

**This round did not write that refactor. It priced it, and the price says
not to.** What follows is the measurement, its liveness proof, and the reason
the ceiling is not on the other side of it.

---

## 1. The instrument: a ceiling arm, not a partial conversion

A partial conversion is uninterpretable. The brief says so itself — "a
conversion that misses sites leaves the tracker on the path and the win will not
appear, which reads as *the design fails* when it means *the conversion is
incomplete*." So the arm built here is the **completed** refactor's effect,
obtained without the refactor:

`--features probe-recon-untracked` constructs the private per-tile planes
(`Rav1dPictureDataComponent::new_private_like`) with **no borrow tracker**,
leaving the shared picture and everything that runs on it — the whole filter
chain — tracked exactly as today.

That is, by construction, the union of what a complete `&mut [u8]` conversion of
every recon kernel could remove: every registration a recon kernel makes on a
plane it exclusively owns, and nothing else.

* UNSOUND by construction (`__`-gated in the sub-crate, `probe-`-gated here,
  absent from `default` and from every published feature).
* `#![forbid(unsafe_code)]` is **not** relaxed in `rav1d-safe`. `forbid` cannot
  be locally allowed, so the sub-crate — which already contains the `unsafe`
  the whole tracker exists to encapsulate — exposes a safe
  `DisjointMut::probe_untracked` behind `__probe_untracked_ctor`. The consumer
  crate calls a safe function.
* It is a slight UNDER-estimate of the refactor in one direction (the refactor
  would also delete the guard structs and the `with_pixel_guard_*` closure
  indirection, which this keeps) and an OVER-estimate in another (it assumes
  every one of the 21 sites below actually converts).

## 2. Liveness, by COUNT and by name — not by timing

`--features probe-sites`, `v4k_8tile` 8bpc, registrations per frame, `lost=0`
on every run:

| arm | t=1 | t=8 | distinct sites |
|---|---|---|---|
| `base` (main) | 7,924,706 | 22,700,725 | 68 |
| `tko` (#474) | 7,976,546 | 13,372,343 | 70 / 69 |
| **`ceil`** (#474 + untracked recon) | **6,005,602** | **11,401,399** | 49 / 48 |

`base` and `tko` reproduce #474's committed census **bit for bit**
(7,924,706 / 22,700,725 / 13,372,343), which is the instrument's control.

The ceiling arm removes **1,970,944 registrations per frame, the same number at
t=1 and at t=8** — the recon population is thread-count-invariant once #474 has
taken the per-row split off it. Set-diffed BY NAME, every one of the 21 sites
that vanish is a reconstruction site and **not one is a filter-chain or
`BlockContext` site**:

| per frame | mean bytes | site |
|---|---|---|
| 629,080 | 10.2 | `src/safe_simd/ipred_arm.rs:1995` (cfl_ac) |
| 283,778 | 16.9 | `src/ipred_prepare.rs:235` (left column) |
| 280,464 | 28,154 | `include/dav1d/picture.rs:173` (`with_pixel_guard_immut`) |
| 233,102 | 26,888 | `src/safe_simd/itx_arm.rs:8725` |
| 133,494 | 31,628 | `src/safe_simd/ipred_arm.rs:1515` |
| 99,264 | 1.0 | `src/ipred.rs:1471` |
| 78,326 | 32,278 | `src/ipred.rs:532` |
| 58,439 | 11,524 | `src/safe_simd/itx_arm.rs:8537` |
| 43,404 | 22,705 | `src/ipred.rs:1238` |
| 39,760 | 11,524 | `src/ipred.rs:538` |
| 35,269 | 57,616 | `src/safe_simd/itx_arm.rs:8894` |
| 25,920 | 960 | `src/tile_recon.rs:287` (the stitch's READ side) |
| 23,710 | 25,128 | `src/ipred.rs:947` |
| 2,271 / 1,822 / 1,180 / 1,149 | 17.7k–25.3k | `src/ipred.rs:887 / 1367 / 806 / 846` |
| 272 / 136 | 960 | `src/recon.rs:3898 / 3880` (ipred-edge backup) |
| 92 / 12 | 1.0 | `src/ipred_prepare.rs:336 / 376` |

**Output identity**: frame md5 is identical across all three arms at
t=1/2/4/8 on both depths — 24 of 24 cells,
`a00c11f454328023c58af14d55544cff` (8bpc) and
`4f218411bc6ee4cc9c630fe827337fa2` (10bpc).

## 3. What is left after #474, and why the recon share is small

The same census answers the question the brief's framing turns on. With #474
active at t=8, the 13,372,343 registrations that remain split as:

| class | per frame | share |
|---|---|---|
| filter chain (`loopfilter.rs`, `cdef*`) | ~6.35 M | **47%** |
| non-picture instances (`ctx.rs`, `env.rs`, `decode.rs`, `ipred_prepare` 1-byte reads, `recon.rs` contexts) | ~4.9 M | **37%** |
| **recon on the private planes** — everything a `&mut [u8]` refactor can reach | **1.97 M** | **15%** |

`src/ctx.rs:99` alone (2,534,988/frame, `CaseSetter::set_disjoint`, mean 2.1
bytes on 32-byte instances) is **larger than the entire recon population** the
refactor targets, and no picture-buffer design can touch it.


## 4. The price — and it is NOT small

n = 9 complete rounds, 288 rows, **`foreign_max = 0` on every one**, idle Apple
M4 Pro, `scripts/perf/verify_gap.sh`, two-point wall fit at 2 and 20 frames,
rotating arm order, dav1d 1.5.4 `--framedelay 1` in the SAME interleaved sweep,
one cell-round discarded and re-run. `RAV1D_TILE_OWNED_T1=1` throughout, so the
private buffers are live at t=1 as well (#474 declines below two workers; the
ceiling has to be priced at the cell the bar is closest at).

ms/frame, median:

| | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| `base` 8bpc | 319.5 | 197.4 | 107.1 | 68.1 |
| `tko` 8bpc | 320.2 | 185.3 | 96.9 | 58.9 |
| **`ceil`** 8bpc | **314.6** | **166.8** | **86.2** | **53.3** |
| dav1d 8bpc | 246.5 | 125.1 | 65.6 | 36.2 |

Ratio to dav1d `--framedelay 1`:

| | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| `base` 8bpc | 1.296 | 1.578 | 1.632 | 1.883 |
| `tko` 8bpc (#474) | 1.299 | 1.481 | 1.478 | 1.628 |
| **`ceil`** 8bpc | **1.276** | **1.333** | **1.313** | **1.475** |
| `base` 10bpc | 1.432 | 1.617 | 1.662 | 1.926 |
| `tko` 10bpc | 1.444 | 1.587 | 1.586 | 1.749 |
| **`ceil`** 10bpc | **1.393** | **1.452** | **1.467** | **1.610** |

Paired within-round ratios (the arms in one round saw the same machine state;
exact two-sided sign test):

| `ceil` / `tko` | median | band | p | bands |
|---|---|---|---|---|
| 8bpc t=1 | 0.9806 | [0.9580..0.9910] | 0.004 | disjoint |
| 8bpc t=2 | **0.8996** | [0.8896..0.9061] | 0.004 | disjoint |
| 8bpc t=4 | **0.8955** | [0.8826..0.9045] | 0.004 | disjoint |
| 8bpc t=8 | **0.9071** | [0.8923..0.9173] | 0.004 | disjoint |
| 10bpc t=1/2/4/8 | 0.9634 / 0.9127 / 0.9158 / 0.9206 | | 0.004 | disjoint ×3, OVERLAP at t=8 |

`ceil` / `base` at 8bpc: 0.9833 / **0.8437** / **0.8087** / **0.7708**.

**So the answer to "is the `&mut [u8]` refactor worth building" is YES, and my own
first estimate said no.** 1,970,944 registrations removed bought 18.5 ms at t=2,
10.7 at t=4 and 5.6 at t=8 — **9.4 / 5.4 / 2.8 ns per removed registration**,
against the 1.00 ns/registration the #469 verification round measured for the
rectangle's removals at t=8. These are not the same registrations: the recon
population's mean extent is 10 B–58 KB with a heavy tail of 26–32 KB strided
hulls, and each also carries guard-drop glue that vanishes with the record.
**"Count is not cost" cuts both ways, and the campaign's 1 ns/registration
constant does not transfer between site classes.**

Ranking against the sound alternatives on the same box, 8bpc:

| | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| #472 (sound rectangle, measured this session) | 1.335 | 1.420 | 1.412 | 1.569 |
| #474 (`tko`) | 1.299 | 1.481 | 1.478 | 1.628 |
| **zero-tracker recon (`ceil`)** | **1.276** | **1.333** | **1.313** | **1.475** |
| untracked ceiling (whole tracker gone, #467) | 1.160 | 1.264 | 1.262 | 1.345 |

The zero-tracker recon path **beats #472 at every one of the eight cells**, and
at 8bpc t=1 it is the only arm this campaign has produced that is under the
~1.30x bar (1.276). t=2 and t=4 land within 0.07 and 0.05 of the whole-tracker
ceiling. **t=8 still misses at 1.475**, and no amount of recon work closes it —
the whole-tracker ceiling there is 1.345.

## 5. The blocker is memory, and it is bigger than reported

Peak RSS, `/usr/bin/time -l`, 20 frames, one decoder:

| arm | 8bpc t=1 | 8bpc t=8 | 10bpc t=1 | 10bpc t=8 |
|---|---|---|---|---|
| base | 99.5 MB | 106.2 MB | 149.2 MB | 157.3 MB |
| tko | 195.7 | 202.5 | 340.1 | 348.3 |
| ceil | 195.1 | 202.0 | 339.6 | 347.8 |

**+96.3 MB at 8bpc and +191.0 MB at 10bpc**, and with `RAV1D_TILE_OWNED_T1=1`
the t=1 column pays it too. The arithmetic is structural, not an allocation bug:
`v4k_8tile` is 4:4:4 with three 8,355,840-byte planes (25.07 MB per frame),
tiled 4×2, so each of the 8 private buffers has half its rows touched —
8 × 12.5 MB ≈ 100 MB. It is not the tracker: `ceil` and `tko` differ by 0.5 MB.

**A version that is this fast and takes +191 MB is not shippable, and the fix is
not a separate follow-up — it is the same change.** #474 bought its
translation-free seam ("one expression, not 22") by giving each tile buffer the
picture's byte length AND stride; that is exactly what makes the whole plane
addressable and the residency `tile_columns × plane`. A column-compact buffer
needs its own origin and stride at every offset computation — which is precisely
what a kernel signature that takes `(&mut [BD::Pixel], stride)` instead of
`WithOffset<&Rav1dPictureDataComponent>` already carries. **Do the `&mut`
conversion and the compact geometry in one pass; doing them in sequence pays
#473's 22-site coordinate cost twice.**

## 6. What this round did NOT do — say it before the wins

* **The refactor itself is NOT written.** Zero of the ~27 kernel call sites and
  zero of the ~40 `with_pixel_guard_mut` closures were converted. Every number
  in §4 comes from an UNSOUND probe that deletes the tracker on the private
  planes; a sound implementation has to earn them back.
* **The `&mut` is not obtainable where #474 puts the buffers.** Tile tasks reach
  `f.tile_recon` through `fc.data.try_read()` → `&Rav1dFrameData`
  (`src/thread_task.rs:1237`), i.e. a SHARED reference, so `recon_planes` can
  only ever hand back `&Rav1dPictureDataComponent` — interior mutability, which
  is the tracker. The conversion therefore has a prerequisite nobody has costed:
  the buffers must move to something the task owns exclusively (`t: &mut
  Rav1dTaskContext` is the obvious candidate, and it is already `&mut` in every
  recon signature). This is the same `&mut`-availability blocker #468 hit one
  level up, and it should be checked in a minute rather than discovered in an
  afternoon.
* **Composition with the rectangle was NOT re-tested** — #475 measured it a
  negative (nested populations: 1.9% more registrations removed for 6.7–9.7%
  more wall) and nothing here changes that argument.
* **`allow_intrabc` frames, `n_fc > 1`, c-ffi allocators and negative strides**
  all decline to the shared picture, as in #474.
* Not measured: **x86_64**; Miri on the probe arm; `--features unchecked`; t=16;
  any vector below 4K; **any vector with loop restoration live** — #455 item 4's
  structural blindness is unchanged, `looprestoration*.rs` registers zero borrows
  in this grid while LR is active in 696 of 768 corpus vectors, so the filter
  class in §3 is understated relative to a corpus vector and the recon share is
  correspondingly overstated.

## 7. Gates

* Corpus `md5_inventory`, set-diffed BY NAME with the md5 as the value against
  `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst`:
  * `tko` and `ceil` at **`--threads 1`: 766 PASS / 0 mismatch / 0 error**,
    768 keys, 0 only-in-baseline / 0 only-in-head / **0 differing**;
  * `tko` and `ceil` at **`--threads 8`: 753 PASS**, 754 keys after dropping the
    two film-grain groups from BOTH sides, **0 differing**. The groups are
    skipped for issue #479 (`rav1d_apply_grain_row` bands the rows per worker
    while `filmgrain_arm.rs:1550` takes a whole-plane mutable guard), which
    aborts 13 of 768 vectors at `threads > 1` on unmodified `main`. `--skip-group`
    was added to `md5_inventory` for this and prints `skipped_groups=` on the
    TOTAL line so an empty filter cannot pass as a full run.
  * This is the first time the private-buffer path has been corpus-gated at
    **t=1** at all — #474 declines below two workers, so its t=1 corpus run
    exercised none of it.
* Frame md5 identical across `base`/`tko`/`ceil` at t=1/2/4/8 on both depths:
  24 of 24.
* `mt_stress` 1/2/4/8/16 × 5 pass. `multi_decoder_pressure` 12 × 3 ×
  {1,2,4,8,16} **PASS on both `tko` and `ceil`**.
* Both standing hazards re-planted and each restored byte-exact (sha256 **and**
  `git diff --exit-code`), under `--features __probe_wide`:
  the 4af62ae in-lock `state` re-read deleted **from `add_contended` alone**
  (the contended path — the `try_lock` fast-path half is not gated) →
  `wide_exclusion` FAILS; `active()` cut to one shard → FAILS. Green again after
  each restore.
* `#![forbid(unsafe_code)]` proved ACTIVELY, and the first attempt was VACUOUS:
  an `unsafe` block planted in `src/tile_recon.rs` **built cleanly** on the
  default feature set, because that whole module is `#[cfg(feature =
  "tile-owned-recon")]` and is not compiled at all. With the feature on it is
  rejected at `lib.rs:13`; a second plant in `include/dav1d/picture.rs` is
  rejected on the default build. **A forbid proof on a cfg'd-out file proves
  nothing** — pick a file the build under test actually compiles.
