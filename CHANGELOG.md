# Changelog

All notable changes to the `rav1d-safe` crate are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/). `rav1d-safe` is a fork of [rav1d](https://github.com/memorysafety/rav1d), which is itself a Rust port of [dav1d](https://code.videolan.org/videolan/dav1d); this fork adds archmage-based SIMD dispatch and removes the C FFI path. Entries below cover only changes made in this fork — upstream rav1d and dav1d release notes remain the canonical record for the shared decoder core. This file was backfilled from git history on 2026-04-15; the `[0.5.4]` date reflects the commit date of tag `v0.5.4` rather than the crates.io publish date.

## [Unreleased]

### Added
- **`examples/crash_sweep.rs` and the farm-artifact triage it produced**
  (`benchmarks/fuzz_triage_2026-08-27.meta`). The 2026-08-14 record could not
  reach `s3://zenfuzz`; this run could. Every artifact the farm holds for the
  three "RECURRED after #399/#403/#407" issues (#430, #436, #439 — 1,960
  inputs, all uploaded 2026-06-15, the day BEFORE the fix they cite) ran
  through the four `fuzz_regression` entry points on unmodified `main`: 7,840
  runs, 0 panics. They are the stale-build re-files that record hypothesised.
  The same sweep is what found #444 live (2 of 2 artifacts panic) and is the
  documented first step before committing any farm seed.

- **A fuzz-crash regression harness, and CI wiring that can actually fail**
  (`tests/fuzz_regression.rs`, `.github/workflows/{ci,fuzz}.yml`,
  `benchmarks/fuzz_regression_2026-08-14.meta`). The repo had ~22 open
  fuzz-crash issues, 13 titled `RECURRED after <link>`, and no
  `tests/fuzz_regression.rs`; the repros lived only in issue bodies and block
  storage. The suite walks `fuzz/regression/` and `tests/crash_vectors/` and
  runs every seed through all three fuzz targets' entry points plus the
  production-default decoder, on **stable** — 30 seeds, 4 entry points, 124
  runs, 0 panics, 0.08 s. Panics are collected with `catch_unwind`, so one run
  names every failing (seed, entry point) pair. Four anti-vacuity guards
  (missing/empty root, `MIN_SEEDS`, a `GUARDED` table mapping 10 farm repros to
  the 17 issues they gate, and a floor on how many seeds still decode a frame)
  plus a `--features __ablate` liveness test asserting the corpus reaches itx
  (4,146,624 units), CDEF (2,981,696) and loop restoration (2,154,542). Teeth
  proven by planting a `panic!` in all seven aarch64 MC dispatchers: the suite
  went red naming 24 pairs, and that run doubles as the MC liveness map — each
  `arm_mc16_*` seed reaches exactly the dispatcher it was filed against. Two
  vacuity findings: `crash-cdef-tile-overlap.avif` is an AVIF container that
  did 0 units of work when fed verbatim (the harness now unwraps AVIF seeds via
  `zenavif-parse`), and the 4-byte `crash-edc01b37...` guards nothing. On the
  CI side, `fuzz.yml`'s `regression` job ran `cargo test ... 2>/dev/null || echo
  "No regression test found"` inside an `if [ -d ... ]` — it could not fail,
  and reported green while no such test existed; its corpus seeding was a flat
  `cp fuzz/regression/*` that silently skipped every per-target subdirectory.
  Both fixed, and the suite added to the 9-leg `build-test` matrix so it runs
  on pull requests at all.

- **The `+1%` at t=1 that kept the rectangle record default-off is CODE
  PLACEMENT, measured — and two new static instruments that say so**
  (`scripts/perf/text_layout_diff.py`, `scripts/perf/text_symbol_diff.sh`,
  `loopfilter::text_pad`, `src/text_pad.rs`, `--features
  __pad_text/__pad_small/__pad2/__pad3/__pad4/__pad_far`,
  `docs/RECT_SHIP.md`). **Nothing about the shipped decoder changes**: the
  default build's `__text` is verified byte-equivalent to the base commit's
  (1,839,536 → 1,839,536, 0 symbols resized, 0 added), and the corpus is
  766/766 by NAME at t=1 and t=8 in both the default and `__lf_rect` arms.
  4,828 bytes of provably-dead `#[used]` text — a build in which no symbol
  changes size, every hot loop-filter symbol keeps a byte-identical instruction
  stream, and a planted `panic!` in the pad leaves the md5 unchanged so it
  provably never executes — costs **+1.10% wall at t=1 on `v4k8tile`, 0 of 11
  rounds**. Nine binaries differing from `main`'s by +1,132 B to +19,420 B
  (dead code; a pure refactor that *shrinks* `LfBlock::open` by 17%; near and
  far modules) all land in **+1.1% to +1.6%** with 9/9–0/11 signs and are
  mutually within ±0.4%, while a byte-identical copy reads 1.0006 (4/11).
  Against a same-source control **the rectangle costs nothing at t=1**: 0.9967
  (7/9). The tax scales with working set — +1.4% at 4K, +0.7% at 1024×576,
  **0** at 256×2048. Consequence for the campaign: "the same source built in a
  second worktree" is NOT a layout control, and a t=1 claim must be judged
  against a same-source arm. Re-measured at the shipped configuration, the
  rectangle's t=8 win replicates on the 1024-wide family (−1.5% to −2.4% wall,
  11/11 and 12/12 signs; `text_q20` −2.6% CPU 13/13) and is narrower than #505
  reported (`c3840x256` is null here).
- **`RAV1D_CDEF_DOUBLE`, the marginal-price arm `docs/RECT_RECORDS.md` §7b asked
  for** (`--features __probe_cdef_double`, `picture::{dup_rows, dup_rows_mut}`).
  Doubles the five CDEF registration sites in ONE binary, changing the count and
  nothing else. A CDEF registration costs **3.27 ns on `c256x2048` t=8**
  (+159,424 regs/frame = 28.0% of the population, +1.34% wall) and **5.27 ns on
  `c1024x576` t=8** (+121,856, **+4.09% wall, 0 of 12 rounds**), against
  `LfBlock::fill`'s 2.42–2.71 ns. Null control: `text_q20` files **zero** CDEF
  registrations and the arm reads 1.0000 (5/12). §7b expected "under ~1%,
  nothing to win"; on the 1024-wide family it is four times that, which makes a
  CDEF rectangle the best-looking remaining target.
- **Exact strided-rectangle borrow records in the tracker, and a
  `LfBlock::fill` arm that uses them — MEASURED, NOT SHIPPED as a default**
  (`crates/rav1d-disjoint-mut`, `src/loopfilter.rs`, `--features __lf_rect`,
  `docs/RECT_RECORDS.md`). One registration describing `h` exact row segments
  with NO inter-row gap reserved: the third shape after the per-row split and
  the refuted hull. Storage is free — the record keeps the rectangle's hull in
  the two words a plain interval already used and `(rows, seg)` is recovered
  from the hull and the instance's declared row stride, an exact bijection, so
  `Shard` stays 128 bytes. Detection is exact in both directions (shard
  selection uses the hull's blocks, a sound superset; overlap detection walks
  rows and never reports a gap byte), and `add_rect` DECLINES rather than
  approximating whenever the geometry is not representable. Registrations drop
  569,690 -> 409,349 per frame (−28.1%) on `c256x2048` t=8.
  **Measured −1.0% to −1.8% wall (up to −3.3% CPU) on 5 of 6 multi-tile t=8
  cells, replicated across two sessions with 10/10-11/11 sign counts; NULL on
  `c256x2048` t=8, the cell it was built for; and +0.9% to +1.3% wall at t=1 on
  `v4k8tile` (0 of 11 rounds either session) where the path never executes.**
  Default-off for that last reason. The round's other deliverable is a
  correction to the campaign's cost model: a `fill` per-row registration costs
  **2.42-2.71 ns at the margin** (measured by doubling the population in one
  binary, 0 of 25 rounds on the other side) against a 19.71 ns/registration
  cell AVERAGE, so the largest registration site in the decoder is 31.7% of the
  population and 3.9-4.4% of the tracker's CPU. The cost tracks distinct shard
  LINES visited, not records filed.

### Fixed
- **Managed `Decoder::flush()` discarded the frames it was documented to
  return** (#423; `src/managed.rs`, `tests/flush_drains.rs`). It called
  `rav1d_flush` — dav1d's reset: drop pending input, drop the ready output
  picture, drop every frame-threading `out_delayed` slot — and only then looped
  `rav1d_get_picture`, which by that point had nothing to give back. Two silent
  losses: under frame threading (`threads >= 2`, `max_frame_delay != 1`; needs
  `unchecked`/`asm`, the default build clamps `n_fc` to 1) `decode()` returns
  `Ok(None)` with the frame in flight and the natural `decode(); flush()` pump
  lost it depending on scheduling (the `asm` CI flavour hashed 0 frames in
  `lr_sgr_vectors_threaded_match_reference_md5`); and at ANY thread count a
  chunk holding several temporal units lost every unit after the first, because
  the unparsed remainder is pending input and `rav1d_flush` drops it.
  `flush()` now drains first (`rav1d_get_picture` until `EAGAIN` — the drain
  protocol: parses queued temporal units, waits for in-flight frames, and fails
  rather than wedges if a worker panicked) and resets after, so it is still the
  "start a fresh stream" call. Gated by `tests/flush_drains.rs`: the vector fed
  twice in one `decode()` must yield two frames (teeth in the default build:
  reverting the order fails it at `threads = 1`, `1 != 2`), and one temporal
  unit at `threads = 2/4/8, max_frame_delay = auto` must yield exactly one frame
  (under `--features unchecked` the frame is deferred and the reverted order
  returns `0`). Both mutations verified; the test is wired into the CI matrix's
  committed-vector step so the `asm` legs run its frame-threaded half.

- **x86_64 16bpc AVX2 horizontal 8-tap MC loaded 4 pixels past the last tap it
  uses, and panicked when the source row ended inside them** (#516;
  `src/safe_simd/mc.rs` `h_filter_8tap_16bpc_avx2_inner`,
  `h_filter_8tap_16bpc_put_avx2_inner`, `h_filter_8tap_16bpc_prep_direct_avx2_inner`;
  `loadu_64!` in `src/safe_simd/pixel_access.rs`). The three kernels build each
  256-bit source register from two 128-bit loads and then
  `_mm256_unpacklo_epi16`, which only consumes the low 4 `u16` of each lane —
  but the loads were 8 wide, so tap `k` read `src[col+k..col+k+8]` and the
  loop's furthest read was `col + 18` for a filter whose last tap is at
  `col + 14`. The callers pass an open-ended `&src[row - 3..]` slice of the
  whole plane, so this only bites when the row is one of the last in the
  buffer: the farm's artifact hit `range end index 18 out of range for slice
  of length 17` on the H-only put path (w = 8, w + 9 pixels to the end of the
  plane). All 36 loads are now 64-bit (`loadu_64!`, `_mm_loadu_si64` behind
  it), reading exactly `col + 14` at most; the consumed lanes are unchanged,
  so output is bit-identical. The AVX-512 variants already loaded exactly
  `w + 7` and were never affected. Gated by
  `fuzz/regression/parse_seq_header/crash-mc16-h8tap-avx2-src-overread` (the
  farm artifact, 61 bytes). **Verified on this aarch64 box only as far as it
  can be**: the x86_64 target compiles and passes `fuzz_regression` under
  Rosetta, which does not expose AVX2, so the kernel itself ran only on the
  farm — the before-evidence is the farm's own run of this artifact against
  `d26c404` (whose line 5337 is byte-identical to `main` before this fix); the
  after-evidence is CI's `ubuntu-latest` `fuzz_regression` leg.

- **aarch64 16bpc bilinear MC sliced two source rows the kernel never reads,
  and panicked on the range for a block in the plane's bottom-right corner**
  (#444; `src/safe_simd/mc_arm.rs`, `mc_put_dispatch_inner` and
  `mct_prep_dispatch` BPC16 `Bilinear` arms). Both dispatchers took
  `&src_bytes[start..start + ((h + 1) * stride + w + 1) * 2]` unconditionally,
  but `put_bilin_16bpc_inner` / `prep_bilin_16bpc_inner` read `h` rows plus one
  only when `my != 0`, `w` columns plus one only when `mx != 0`, and on the last
  row only those columns — never the rest of its stride. A 16bpc plane buffer
  ends exactly where its last row's `w` pixels end, so the copy case
  (`mx = my = 0`) overshot the guard by two full strides and the range panicked
  (`range end index 33088 out of range for slice of length 32768` and `8226 /
  8192` on the farm's two artifacts). Reproduced from both `s3://zenfuzz`
  artifacts for the signature on unmodified `main` (8 of 8 seed/entry-point
  pairs panic at `mc_arm.rs:6034:61`), fixed by a shared `bilin_16bpc_src_extent`
  that sizes the slice to the rows and columns actually read, and gated: the
  two artifacts are committed as
  `fuzz/regression/parse_seq_header/crash-mc16-bilin-src-overshoot-{a,b}` with
  `GUARDED` rows; reverting the two call sites makes `fuzz_regression` fail on
  all 8 pairs. The 8bpc arms were never affected (they slice open-ended).
  `examples/crash_sweep.rs` is the tool that ran the farm's artifacts — every
  file in a synced crash directory through the four `fuzz_regression` entry
  points, panics grouped by site.

- **`--features c-ffi` without `asm` failed to build any test target**
  (`src/safe_simd/ipred_arm.rs`, `src/safe_simd/itx_arm_parity.rs`, edfddee).
  The `cfl_parity` and `itx_arm_parity` modules round-trip their scratch buffer
  through `Rav1dPictureDataComponent::copy_pixels_to`, which is declared
  `#[cfg(not(feature = "c-ffi"))]`, but both gated themselves on
  `not(feature = "asm")`. `asm = ["c-ffi"]` and not the reverse, so plain
  `--features c-ffi` compiled them against a method that does not exist (three
  `E0599`s). CI's c-ffi leg is `cargo clippy --features c-ffi` with no
  `--all-targets`, which never builds the test modules, so the breakage stayed
  latent. Both gates are now `not(feature = "c-ffi")`, which excludes every
  configuration the old gate did plus `--features c-ffi` alone.

- **The reconstruction band reserved the tile's exact WIDTH, so a block
  overhanging the last column wrote into the next band row — silently wrong
  pixels, and a panic on the block's last row** (`src/owned_recon.rs`,
  `fuzz/regression/parse_seq_header/crash-owned-recon-band-row-short`). Found
  by fuzzing `main` for 600 s on aarch64 with the new harness's corpus as the
  seed: `owned_recon.rs:367:37: range end index 128 out of range for slice of
  length 64`, minimised to 21 bytes, reproducing through the plain stable
  harness on all four entry points at `threads = 1`. Measured at the overrun
  site: a 32-pixel-wide band and a 64-pixel `splat_dc` block at 16bpc.
  `band_geometry` sized the column extent as exactly `(col_end - col_start) *
  4` while the ROW extent has always been a full `sb_step * 4` with live rows
  clamped separately — but AV1 codes whole blocks and crops on output, so a
  32x32 frame can legally be one 64x64 block. The picture absorbs the overhang
  in its padding; a column-compact band has none. 8bpc was mostly shielded by
  the 64-byte `CHUNK` rounding, which is 64 pixels of slack at one byte per
  pixel and only 32 at two. `arm()` now rounds the ALLOCATED width up to a
  whole superblock while `live` keeps the exact `cols`, so `stitch` copies what
  it always did and output cannot change: `decode_md5_verify` is 766 passed / 0
  failed / 2 skipped at **both** t=1 and t=8, with the widening instrumented
  and confirmed live on that corpus (`cols=352 -> alloc_cols=384` luma,
  `176 -> 192` chroma).

- **x86_64 at `--threads 8`: the loop filter read 3 picture rows past its own
  superblock row and raced concurrent reconstruction** (#494,
  `src/safe_simd/loopfilter.rs`, `src/loopfilter.rs`). The x86-only
  `loopfilter_sb_dispatch` sizes the guard for a whole superblock edge itself
  (aarch64 returns `false` and lets `LfBlock::open` size per fused group), and
  it used a CONSTANT perpendicular reach for V runs — 7 rows either side of a
  horizontal edge for luma, the widest the plane allows. The filter's real
  reach is `lf_reach` of the level the MASK selected, and the level is
  `min(log2(tx_h) above, log2(tx_h) below)` capped at 2, with a transform never
  crossing a superblock boundary: level 2 (reach 7) has >= 16 rows of headroom,
  level 1 (reach 4) >= 8, level 0 (reach 2) >= 4. At every level-0 edge in the
  last 4-row band the window therefore read 3 rows into the NEXT superblock
  row, which since `054e2ed` dropped the frame-global deblock barrier is being
  written concurrently. Reproduced as 4 aborts per 358-vector `8-bit/data` pass
  at t=8 (the failing set is a timing window, not content — three passes named
  seven different vectors), with both sides identified by a
  `-C debug-assertions=on` release build (which reproduces it at a HIGHER rate
  than release — its two aborts were the 20th and 27th vectors attempted): the
  V-run compact read
  (`loopfilter.rs:5134`) against `owned_recon.rs:937`'s `stitch_sbrow` copy-out
  of the next superblock row, and against that row's own DeblockCols
  write-back. In an `unchecked` build there is no panic — the read returns
  half-written pixels. Fix: `lf_run_reach` derives the V window from the run's
  mask; `lf_group_wd` is extracted from `loop_filter_sb128_rust` so the new
  function and its test share the driver's ladder. The scalar-fallback
  predicate still tests the plane worst case, so the SIMD-vs-scalar decision
  and every output byte are unchanged (aarch64 base-vs-head set-diffed by name
  with the MD5 as the value: identical, 768 rows, at t=1 and t=8; x86_64 head
  t=1 == head t=8 == aarch64, also 768 rows). The over-read fired on **0.632%
  of V runs** before the fix and 0 after (counted, aarch64 `8-bit/data` t=1),
  so it was rare in coincidence rather than rare in the filter. The H
  direction keeps the constant deliberately — its extent is columns of rows
  already inside this superblock row, and its `tap_after` is the chunked
  transpose load's rounding rather than a tap bound. Gates:
  `run_reach_equals_the_widest_group_it_can_meet` (every mask combination
  against the driver's own ladder, with a liveness assert that all four reaches
  occur) plus a `debug_assert!` in `loopfilter_sb_direct` that a V run's window
  stays inside its superblock row — which makes the whole class deterministic
  and single-threaded: with the old constant planted, a
  `-C debug-assertions=on` release build aborts on the second vector of
  `8-bit/data` at `--threads 1`.
- **Film grain could not be decoded above one thread at all** (#479,
  `src/safe_simd/filmgrain_arm.rs`). `rav1d_apply_grain_row` is claimed by N
  workers that each `fetch_add` a *different* `FG_BLOCK_SIZE` row band off
  `TaskThreadData::delayed_fg_progress[0]`, while all four film-grain guards
  reserved the WHOLE picture component per band (`full_guard_mut` /
  `full_guard`). The workers collided on their first band each
  (`overlapping DisjointMut: &mut _[0..147456]` x6 at 8 threads) and the dead
  worker then wedged the main thread on `thread_task.rs`'s `unwrap()` of a
  `None`. **13 of 768 corpus vectors could not be decoded at any thread count
  above 1**, and every corpus run of the 2026-08 campaign passed
  `--skip-group film_grain` to work around it. Each guard is now narrowed to
  the band the call actually touches — `(bh-1)*stride + pw` for dst/src, and
  for `fguv`'s luma read rows `y << sy` for `y in 0..bh` (last row
  `(bh-1) << sy`, *not* `bh << sy`) by columns up to `((pw-1) << sx) + sx`.
  Bands are provably disjoint: consecutive bands start `FG_BLOCK_SIZE * stride`
  apart, a band spans at most `(FG_BLOCK_SIZE-1)*stride + pw`, and
  `pw <= stride`. A narrowing, which is the direction
  `docs/OWNERSHIP_MODELS.md` §3/§6/§7d says is always available — the three
  refuted schemes all *widened* a reservation. Corpus 766/766 at `--threads 8`
  with **no** `--skip-group`, set-diffed by name with the MD5 as the value
  against `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst`; also clean
  at t=1. Gate: `tests/filmgrain_threads.rs` (every film-grain vector x
  {1,2,4,8} threads, 3 reps above t=1, plus two liveness tests), proven to have
  teeth by planting the old guard and watching it abort.
  Also on this path: negative strides now fall through to the `PicOffset`-based
  scalar kernel, because `fgy_inner_*`'s `row_off(y) = (y*stride) as usize`
  wraps for `stride < 0` and indexes out of bounds — a panic before this change
  too, at any thread count. No corpus vector has a negative stride; only a
  caller-supplied `Rav1dPicAllocator` under `c-ffi` can produce one.
- **`wide_exclusion` had gone vacuous, and with it the only gate for the
  wide-path TOCTOU** (`crates/rav1d-disjoint-mut/tests/wide_exclusion.rs`).
  Since `SHARDS_SERIAL = 1` (#458) an instance built by a process that has
  declared no parallelism gets `mask = 0`, i.e. ONE shard — so this test's
  whole-buffer borrow took `add`'s fast path and never entered the wide path
  at all. Measured on 2aa00c5: 100 whole-buffer borrows of an 8 MiB instance
  produced 0 promotions on every `wide_probe` counter, while the test still
  reported `ok`; its two liveness assertions count GRANTED borrows, which a
  single-shard instance grants fine. With the 4af62ae TOCTOU hazard planted
  (delete `add`'s in-lock `state` re-read) `wide_exclusion`, `soundness` (25),
  `shard_liveness` (5) and `narrow_release` ALL passed — no gate at all. The
  test now declares parallelism first and, under `__probe_wide`, asserts that
  promotions actually happened, so it cannot go quiet again. Post-repair the
  same hazard fails it with 1393 witnesses (probe-wide) / 988 (plain).
  Test-only: the library binary is byte-identical (sha256-verified).
  Record: `benchmarks/verify_compose4_2026-08-08.meta` §6.

### Changed
- **A picture plane's borrow-tracker block shift is now DERIVED FROM ITS ROW
  STRIDE, not from a target block count** (#455, `docs/BPS_ROWS_DEFAULT.md`,
  `benchmarks/bps_rows_default_2026-08-11.*`). The block-count rule targets
  `N_SHARDS * BPS` blocks, so a block spans `~aligned_h / 256` picture ROWS while
  the hot strided accesses are a fixed number of rows — a defect in the rule's
  shape that no constant fixes. The new rule coarsens from the block-count
  answer until a block spans `ROWS_PER_BLOCK_MIN = 4` picture rows, never finer
  than before and never past a `MIN_BLOCKS = 32` floor; buffers with no declared
  stride (everything that is not a picture plane) and every serial or
  single-tile decode keep the old rule exactly. Measured on the SHIPPED build,
  n=7 paired per-round ratios, idle box, twice on independently built binaries:
  **wall 0.7656 / 0.7530 / 0.7473 / 0.8889 at 1024x192, 1024x384, 3840x256 and
  1024x576 (8 tiles, t=8), all seven rounds below 1.0 on each**, and 0.9000 on
  the 10-bit 1024x576 twin. Against dav1d 1.5.4 `--framedelay 1` on the same
  clock: **2.093 -> 1.601, 2.268 -> 1.708, 2.195 -> 1.649, 1.825 -> 1.616,
  1.834 -> 1.645.** CPU 0.85-0.93 on the same cells. Nothing regresses: the two
  cells that read ~1.000 (`c256x2048`, `v4k_8tile`) are cells where the rule
  provably computes the SAME shift, so they are identity controls and their
  spread is the grid's noise floor (±1.0% and ±3.1%). t=1 is unchanged by
  construction and measured so (medians 0.9972-1.0004). Registration counts are
  identical between the arms on every cell — the knob changes the COST of a
  registration, taking it from 5.8-9.3 ns to 3.2-3.9 ns, which is the
  uncontended rate. `w_full` is 0 everywhere, so the coarser block does not
  trade wide-by-shard-count for wide-by-slot-exhaustion. Corpus 766/766 at t=1
  AND t=8 on the DEFAULT build with no `--skip-group`, set-diffed by name with
  the actual md5 as the value; 10/10 sweep vectors bit-identical to dav1d at
  both thread counts before any timing. #501's one unexplained cell (512x576)
  **does not reproduce**: a 3x3 per-plane shift factorial (new
  `--features probe-shiftpin`) puts the derived rule between `bps1` and
  `bps-half` with the planes separable to ±2%; the real residual is that one
  rows target serves both planes while a 4:2:0 chroma tap window is half the
  luma one, worth ~0.7% there. Still open: no x86_64 leg, no held-out size for
  `ROWS_PER_BLOCK_MIN` (the new `__rpb_{2,8,16}` ladder is for re-fitting it and
  was not swept), no 4:4:4 / 12-bit / inter content. (318a4bc, f97a168, c2b5d0f)
- **Feature rename in the shard ladder**: `bps-rows` is gone (it is the default,
  and a flag for the default can never fail) and `bps-blocks` is new — the
  block-count rule that shipped before this, and the base arm any A/B against
  the default must be differenced against. Selecting ANY `__bps_*` rung now also
  turns the derived rule off, so the ladder stays a clean re-fit instrument.
  Added `rpb-2` / `rpb-8` / `rpb-16` (the ladder for the constant that actually
  ships) and `probe-shiftpin` (`RAV1D_PIN_SHIFT="<stride>:<shift>,…"`, the only
  instrument that separates a luma shift from a chroma one). (318a4bc, c2b5d0f)
- **The zerocopy slice cast no longer puts a 112-byte cold-error frame on the
  10-bit hot path** (`crates/rav1d-disjoint-mut/src/lib.rs`). `.unwrap()` on
  `mut_from_bytes`/`ref_from_bytes` required a several-word `CastError` to be
  materialisable in the caller, which is what stopped LLVM inlining
  `slice_as`/`mut_slice_as::<_, u16>`; a `#[cold] #[inline(never)]`
  `cast_slice_failed` plus `#[inline(always)]` removes all six out-of-line cast
  symbols from the binary. The predicate is unchanged — zerocopy still decides.
  Measured idle-box, paired per-round ratios, n=9, md5-identical: **10bpc 4K
  t=1 0.9380 [0.9261, 0.9517], t=8 0.9351**; 8bpc 1.0043 (null, and correctly
  so — at one byte per pixel the cast folds away). Gap to dav1d 1.5.4
  `--framedelay 1` at 10bpc: 1.77 -> 1.65 (t=1), 2.26 -> 2.09 (t=8); 8bpc
  unmoved at 1.48/2.05. Self-time bucketing puts the whole win in the tracker
  bucket (105.5 -> 82.3 ms/frame, −22.0%) with `add` flat in absolute terms.
  Composed from `perf/blockshift-bpc` part 2; that branch's part 1 and all of
  `perf/shard-mapping` measured null on current main and were NOT composed
  (both target a wide path that #458 already removed — see the .meta §1).
  The two halves of this change are super-additive and neither ships alone:
  `inline(always)` only 0.9862, `cast_slice_failed` only 0.9820, both 0.9374
  (n=9, 10bpc t=1). The handover's claim that the inline ALONE regresses
  2.3-2.7% does not reproduce on main — see the .meta §8, which also corrects
  that number having been restated here as if this campaign measured it.
  Record: `benchmarks/verify_compose4_2026-08-08.meta`.

### Added
- **The borrow tracker's block-shift rule swept across PICTURE SIZE, and a
  derived alternative to it** (#455, `docs/SHARD_SIZE_SWEEP.md`,
  `benchmarks/shard_size_sweep_2026-08-10.*`). The adaptive shift targets a
  fixed BLOCK COUNT, so a block spans `~aligned_h / 256` picture ROWS while the
  hot strided accesses are a fixed number of rows — meaning the rule is
  miscalibrated by picture geometry, not by its constant. Measured over 17
  multi-tile cells (a fixed-width height ladder, a fixed-height width ladder,
  the 16:9 diagonal and two discriminating cells), all bit-identical to dav1d
  1.5.4 at t=1 and t=8 before timing: coarsening by two shifts is worth
  0.761x-0.930x wall with disjoint bands on 9 of the 10 cells below 2.13 rows
  per block, and is a null on all 7 cells at or above 3.76. That crossover is
  why the prior round measured 14.6% at 1024x576 and nothing at 4K. `len` alone
  provably cannot express the rule (the wanted coarsening is not monotone in
  it), so `rav1d-disjoint-mut` gained `DisjointMut::declare_row_stride` — called
  unconditionally by the picture allocator, a no-op for the shipped rule — and a
  default-off `__bps_rows` arm that picks the shift per instance to give a block
  at least 4 picture rows. Over the 17 cells it is the only arm with no
  regression anywhere (worst 0.998 vs 1.004/1.013/1.031 for the fixed rungs) and
  has the best CPU geomean; on wall geomean it ties `bps-half`. Not enabled: one
  cell (512x576) is an unexplained 7% miss, and there is no 10-bit, 4:4:4 or
  x86_64 leg. Corpus 766/766 at t=1 and t=8 on both the default build and the
  arm. (dad8f15, eb4169c) **Superseded 2026-08-11: `__bps_rows` is now the
  DEFAULT (see the Changed entry above), the 512x576 miss did not reproduce, and
  a 10-bit leg landed; the 4:4:4 and x86_64 legs are still missing.**
- `examples/decode_md5 --threads N` (was hardcoded to 1) and an `__ablate`
  per-stage execution census in `examples/probe_tracker`, so an ad-hoc vector
  can be identity-checked at t=8 and "did this stage run at all?" is a counter
  rather than a header flag. The census corrects a committed claim: "CDEF
  executes zero blocks at 512x288" is a property of the DOWNSCALED size-ladder
  vector, not of the size — a 1:1 crop at that size runs 63,488 units/frame.
  (dad8f15)
- `scripts/perf/av1_tile_info.py`, which parses tile_cols/tile_rows out of an
  AVIF's AV1 bitstream. libaom clamps a `--tilecolslog2` request against the
  frame's superblock count and the encoder log records only the request, so an
  unnoticed clamp turns a multi-tile cell into a silently void one. (dad8f15)
- **10/12-bit inverse transforms now have an aarch64 NEON tier**
  (`src/safe_simd/itx_arm_hbd.rs`). `itxfm_add_dispatch` had been 8bpc-only on
  purpose: the `itx_arm_neon_*` kernels hold transform state in `int16x8_t`,
  which is exactly the spec's 8bpc row/column clip and nothing wider, so their
  `*_16bpc_*` entry points were never reachable from the safe build. The new
  module vectorises the *generic* reference (`src/itx.rs` + `src/itx_1d.rs`)
  in `int32x4_t` lanes instead — four independent 1-D transforms per vector,
  each lane running the identical i32 op sequence the scalar reference runs.
  Wired for every shape with `max(w, h) <= 16` and all 16 non-WHT types
  (4x4, 8x8, 16x16, 4x8, 8x4, 4x16, 16x4, 8x16, 16x8); 32/64-point transforms
  and WHT still run the reference. Measured idle-box at t=1 on a 4K 10-bit
  still: **508.3 -> 455.8 ms/frame, 2.02x -> 1.81x of dav1d 1.5.4
  `--framedelay 1`** (t=2 2.16 -> 2.04, t=4 2.43 -> 2.33); itx goes from 21.93%
  to 12.79% of decode inclusive with no `itx_1d` sample left at all. 8bpc is
  unchanged, as it must be. Record: `benchmarks/itx_hbd_neon_2026-08-07.meta`.
- **NEON chroma-from-luma prediction on aarch64** (`cfl_pred_dispatch` in
  `src/safe_simd/ipred_arm.rs`). `cfl_pred_direct` had an x86_64 dispatch and
  nothing beside it; measured at t=1 the scalar loop was 2.46% of decode self
  time at 8bpc (3.37% inclusive) and 1.79% at 10bpc.
- **Per-family kernel activity counters** (`src/ablate.rs`
  `note`/`activity_snapshot`/`activity_reset`, `__ablate`-gated exactly like
  `is_off`) plus `md5_inventory --activity`, which emits them per corpus
  vector. This is what a `sample` profile cannot tell you: whether a 0.0 ms
  kernel is fast or simply never called. Record:
  `benchmarks/family_activity_2026-08-07.tsv.zst`.
- `examples/profile_ivf` gains `RAV1D_ABLATE` / `RAV1D_REPS` / `RAV1D_LABEL`,
  so any kernel family can be A/B'd against its scalar reference from a single
  binary; `scripts/perf/lr_ab.sh` is the rotating-order driver.

### Removed
- **`src/safe_simd/looprestoration_arm.rs`'s scalar duplicate (1,436 lines).**
  The file claimed to be "Safe ARM NEON implementations for Loop Restoration",
  contained zero aarch64 intrinsics, and `lr_filter_dispatch` returned `true`
  unconditionally — so every loop restoration call on aarch64 ran a second
  hand-written copy of `src/looprestoration.rs` rather than the reference.
  Interleaved A/B (rotating order, median of 9): the copy was **6.01% slower
  whole-decode at 8bpc** (204.42 vs 192.83 ms/frame on `8-bit/data/00001147`)
  and a wash at 10bpc. Now the dispatcher returns `false` and the caller runs
  the reference; 766/766 conformance with byte-identical per-vector MD5s, which
  is also the proof the duplicate bought nothing. aarch64 loop restoration is
  still scalar — a real NEON tier is named as remaining work in
  `ROADMAP_SIMD_PORTING.md` R3 and `benchmarks/lr_arm_vs_reference_2026-08-07.meta`.
### Changed
- **`rav1d-disjoint-mut`: releasing a borrow no longer takes the shard lock**
  (`1f09769`). The occupancy bitmap moved out of the lock-protected record
  block into an `AtomicU8` on the shard, so `remove` is one `fetch_and` against
  the `fetch_or` a registration publishes with. Registration is unchanged and
  still re-reads `state` inside its lock (the wide-list TOCTOU fix in
  `4af62ae`). New `threaded_churn_leaks_no_slots` test, mutation-proven against
  the lost-update shape it guards. Isolated effect on decode: -0.9%.
- **`rav1d-disjoint-mut`: the borrow-tracker block shift is re-openable, and can
  size itself from the buffer** (`c003d2f`). Profiling by CALLER rather than by
  symbol showed the tracker's cost is the shard cache line, not the
  registration — a strided access pays one line per ROW, and the single largest
  consumer is `rav1d_prepare_intra_edges`' 1-pixel-wide left-column read (9.19%
  of a t=8 4K frame) at one registration per row per BYTE. New
  `blockshift-13/14/15/16` rungs and a `blockshift-adaptive` rule
  (`log2(len) - 8`, so a 4K 8-bit plane gets 14 and its 10-bit twin 15, both at
  ~4.3 picture rows per block, while a 64 KiB buffer keeps 8). Sound for any
  value — the no-missed-overlap argument constrains only that both registrants
  agree, which the module header now states, and the premise is mutation-proven
  (making consecutive registrations disagree by one bit fails
  `cross_shard_overlaps_are_all_caught`).
- **The adaptive shift is now the DEFAULT for a threaded decode** (`fd5239f`),
  with serial decode left byte-for-byte on the old constant — the same split,
  for the same reason, as `SHARDS_SERIAL` vs `SHARDS_CONCURRENT`. Idle box,
  median of 5, zero foreign, ms/frame: v4k_8tile 131.3 -> 119.8 at t=4 and
  117.9 -> 76.3 at t=8; 10bpc 162.0 -> 155.3 and 141.6 -> 97.1. Against
  dav1d 1.5.4 `--framedelay 1` that is **t=8 3.15x -> 2.04x (8bpc) and
  3.74x -> 2.56x (10bpc)**, t=4 2.00x -> 1.83x and 2.43x -> 2.32x. Cost,
  measured and reproducible across all five rounds: single-tile 4K at t=8 is
  **2.6% slower** (364.6 -> 373.9) — one tile means the concurrency is
  post-filter tasks sharing planes, so a coarser block buys no strided-read
  locality and only adds collisions; the tracker cannot detect that from the
  buffer length. 1024x1024 is neutral. Output bit-identical on all 769 corpus
  vectors; wide-path promotions zero.
  Record: `benchmarks/tracker_blockshift_2026-08-08.meta`, raw
  `benchmarks/tracker_blockshift_confirm_2026-08-08.tsv`.
- **`held-row-guards` (default off), a measured negative kept on purpose**
  (`94f1bdb`). `WithOffset::block_mut`'s compact path can hold its per-row
  MUTABLE guards across the kernel instead of taking immutable ones to read and
  mutable ones to write back, halving registrations from `2h` to `h` with
  byte-identical extents and 766/766 correctness. It measures null, and
  combined with a larger block shift it is a collapse (402-986 ms/frame against
  a 120 ms base) because holding 64 guards overflows a shard's 7 slots onto the
  all-shards wide path.
- `examples/probe_tracker` gains `--features probe-wide`: wide-path promotion
  counters that, unlike `probe-count`, keep the sharded tracker rather than
  switching to the legacy one.

### Fixed
- **10/12-bit CDEF on aarch64 decoded to the wrong pixels** (issue #446).
  `src/safe_simd/cdef_arm.rs`'s 16bpc filter computed `pri_tap` as
  `4 - (pri_strength & 1)` where the spec, `src/cdef.rs`'s scalar reference and
  both x86 16bpc kernels use `4 - (pri_strength >> bitdepth_min_8 & 1)`.
  `bitdepth_min_8` is 0 at 8 bpc, which is why only high-bit-depth ARM output was
  affected and why no 8-bit hash ever moved. Carried in with `perf/cdef-neon`
  (whose 16bpc vector kernel cannot be bit-exact without it) and independently
  confirmed: frame MD5 now matches dav1d 1.5.4 on 7/7 local vectors at t=1/2/4/8,
  and `cargo test --release --test decode_md5_verify` over dav1d-test-data goes
  from 556 mismatching vectors to 464 — 92 newly matching, **0 regressions**
  (10-bit/quantizer 3 -> 64 passing, 10-bit/data 0 -> 26, 10-bit/film_grain
  1 -> 6, every other suite unchanged to the vector).

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->
- `rav1d-disjoint-mut`: **`DisjointMut::new` is no longer `const`**. The borrow
  tracker now sizes itself from the container's length at construction, which
  needs a call to `AsMutPtr::len`. `dangerously_unchecked` stays `const`. No
  in-repo caller constructed a `DisjointMut` in const context.

### Changed
- **VERIFIED COMPOSE (2026-08-07, `verify/compose`).** `perf/lf-neon-port`,
  `perf/cdef-neon` and `perf/p3-t8-inversion` merged onto `perf/p2-kernels` and
  independently re-measured; nothing was left out. Record:
  `benchmarks/verify_compose_2026-08-07.meta`, raw
  `benchmarks/verify_gap_2026-08-07.tsv`, harness `scripts/perf/verify_gap.sh`.
  The composed tree is **not** bit-identical to its baseline at 10 bpc, and the
  divergence is a fix (see Fixed). Idle-box, median of 7, one instrument on both
  sides: v4k_8tile ms/frame 412.1/260.6/175.9/187.8 -> 400.3/223.8/131.1/116.5 at
  t=1/2/4/8, i.e. 1.67/2.08/2.68/5.02x -> 1.62/1.79/2.00/3.12x of dav1d 1.5.4
  `--framedelay 1`; 10 bpc 2.39/2.80/3.37/5.65x -> 2.01/2.16/2.37/3.76x. The
  t=4 -> t=8 inversion is gone (t=8/t=4 1.068 -> 0.889 at 8 bpc, 0.951 -> 0.899
  at 10 bpc) and t=8 is the best thread count again. Still 3.1-3.8x off dav1d at
  t=8; its t=1 -> t=8 scaling is 6.59x against our 3.44x.

- **The aarch64 deblocking loop filter has a real NEON tier**
  (`src/safe_simd/loopfilter_arm.rs`). That file previously imported
  `core::arch::aarch64::*` and contained no intrinsic call, and its
  `loopfilter_sb_dispatch` returned `false` unconditionally — so on aarch64 the
  "NEON" loop filter was the scalar reference. Ported bit-exactly: all four tap
  widths (`wd` 4 / 6 / 8 / 16 = the spec's filter4 / filter6 / filter8 /
  filter14), both edge directions (vertical edges transpose 8x8 tiles in
  registers; horizontal edges are tap-major and need none), at 8, 10 and 12
  bits, over fused runs of 1..4 groups. One `u16`-lane kernel serves every bit
  depth. Two neighbours of the filter that the profile showed cost MORE than
  the arithmetic went with it: `LfBlock::close`'s write-back diff scan is now
  one `vceqq` plus a nibble movemask, and `LfBlock::open`'s per-row copy is
  monomorphized on the six widths it can take instead of a `memmove` call per
  row. No `DisjointMut` guard changed extent or count, and no `unsafe` was
  added. Bit-identical decode output: the whole-corpus MISMATCH set (4,945
  lines, 2,845 distinct triples across 5 CPU tiers x 989 vectors) is unchanged,
  and frame md5 matches on all 7 local vectors at every thread count.
  `benchmarks/lf_neon_2026-08-07.meta` (3b44f6d, d751493, a5606dc).
- **The borrow tracker's shard set is sized from the declared decode
  parallelism, and 8-bit decode no longer gets slower past 4 threads**
  (`crates/rav1d-disjoint-mut/src/tracker_shard.rs`, `set_parallelism`). On a
  3840x2160 8-tile 8bpc stream the decoder had been running 175.8 ms/frame at
  t=4 and 187.0 at t=8 — the best configuration was 4 threads. Worker occupancy
  was not the problem (7.55 of 8 busy); the same 136 tile-recon and 3x34 filter
  tasks simply cost 62% more CPU at t=8 than at t=4, and a build with the
  tracker compiled out did not slow down at all. The default shard count per big
  instance moves 32 -> 128 for a threaded decode and stays at 32 for a serial
  one, because the two want opposite things: more shards cut the cross-core
  contention that caused the inversion, while a serial decode is dominated by
  the wide-borrow path, whose cost is proportional to the shard count and which
  is common exactly when tile threading is off. That path now holds only the
  shards an instance can actually reach (`0..=mask`) rather than the whole
  array, which is what makes the larger array affordable — for a sub-64-KiB
  instance it is one lock instead of 128. Measured, M4 Pro, median of 9,
  ms/frame: 8bpc t=1 413.3 -> 430.5, t=4 175.8 -> 139.7, t=8 187.0 -> 125.3,
  t=16 208.8 -> 135.6; 10bpc t=8 213.7 -> 161.7. Best shippable configuration
  175.8 -> 125.3 (1.40x), and the gap to dav1d 1.5.4 at 8 threads goes 5.23x ->
  3.44x (8bpc) and 5.65x -> 4.18x (10bpc). The 4.2% single-thread cost is the
  larger shard array itself and is not recovered. Bit-identical output at every
  arm and thread count; `tests/wide_exclusion.rs` gates the wide path's
  exclusion with a race that fails on a deliberately shortened prefix.
  `benchmarks/p3_inversion_2026-08-07.meta` (14873a6).
- **The `DisjointMut` borrow tracker is address-block sharded, and tile
  threading now scales** (`crates/rav1d-disjoint-mut/src/tracker_shard.rs`). Each
  instance's tracker is split into 32 independently locked, cache-line-isolated
  shards chosen by a hash of the borrow's address block, instead of one spin
  lock plus one 64-slot table that every tile worker funnelled through. A borrow
  registers its exact interval in every shard its blocks map to and checks
  exactly those, so overlaps are still caught (two overlapping borrows share a
  byte, hence a block, hence a shard) and disjoint borrows are still never
  refused (records are whole intervals, never clipped). Decode had been getting
  *slower* with more threads; it now gets faster at every step. Measured on
  3840x2160 8-tile 8bpc, M4 Pro, median of 9, ms/frame (speedup vs that arm's
  own t=1): t=1 602 (1.00x) -> 601 (1.00x), t=2 513 (1.17x) -> 423 (1.42x),
  t=4 688 (0.88x) -> 365 (1.65x), t=8 1679 (0.36x) -> 332 (1.81x) — 5.1x faster
  at t=8, and 1.55x faster than the best thread count the old tracker could
  ship. Single-tile content gains too (1024x576 8bpc: no scaling at all before,
  1.36x now). Costs, measured and not yet diagnosed: 10-bit content is 6-11%
  slower single-threaded, and 256x144 is 4.5-6.8% slower at every thread count.
  Bit-identical output at every thread count and every arm.
  `benchmarks/shard_tracker_2026-08-07.meta` (91169df, e1a3e85, 1401cb3,
  f01ada8).
- `DisjointMut::tracker` is boxed, so the wrapper is pointer-sized: this drops
  `Rav1dTaskContext` well under its 48 KiB stack-weight gate.

## [0.6.0] - 2026-07-04

Staged release: the batched 0.x-breaking changes below ship together with the
issue-#14 aarch64 loop-restoration closure. `cargo semver-checks
--baseline-version 0.5.7 --default-features` confirms the break (new
`Error::Cancelled` variant on a non-`#[non_exhaustive]` enum; `simd_test`
feature renamed to `__simd_test`), hence 0.6.0 rather than a 0.5.x patch.

### Breaking
- **`managed::Result` is now `Result<T, whereat::At<Error>>`** (was
  `Result<T, Error>`), so the safe `Decoder` API (`new`/`with_settings`/
  `decode`/`get_frame`/`flush`) attaches a source location to errors for
  server-side logs. The internal `Rav1dError`/`Rav1dResult` hot path is a
  separate, unchanged type — the `At<>` wrapper applies only at the per-frame
  managed boundary, never in an inner loop, and decode output is byte-identical.
  Callers matching the error unwrap first: `err.error()` (`&Error`),
  `err.decompose().0` (owned).
- **New `Error::Cancelled` variant** on the (non-`#[non_exhaustive]`)
  `managed::Error` enum, returned when a decode is aborted through the new
  cooperative cancellation token (see Added below). Exhaustive matches on
  `Error` need a new arm.
- **Testing-only feature `simd_test` renamed to `__simd_test`** (with the new
  `__simd_test_log` inventory variant). The old feature name no longer exists;
  it was never meant for downstream use.

### Added
- **Cooperative in-flight decode cancellation** (issue #412). `Decoder::set_stop(Some(Arc<dyn Stop>))` installs an [`enough`](https://github.com/imazen/enough) `Stop` token that the decode loop polls at superblock-row granularity; when it fires, the in-flight frame is aborted and the `decode`/`get_frame`/`flush` call returns the new `Error::Cancelled` instead of running a crafted-but-spec-legal stream to completion. Both decode paths honor it: the single-threaded loop checks per sbrow (`src/decode.rs`), and tile-threaded workers check per task and abort via the same per-frame error path the internal flush uses (`src/thread_task.rs`). `None` (default) means never check — zero overhead (`enough::Stop::may_stop` short-circuits). Re-exports `Stop`/`StopReason`/`Unstoppable` from `managed`; adds internal `Rav1dError::ECANCELED`. Lets an untrusted-AV1 server bound a slow decode without abandoning the worker thread. Tested in `tests/cancellation.rs` (single-threaded + tile-threaded). Pure safe Rust; default `forbid(unsafe_code)` build unaffected.
- Versioned public-API surface snapshots at `docs/public-api/<crate>.txt` (rav1d-safe + rav1d-disjoint-mut), regenerated by `tests/public_api_doc.rs` on every `cargo test`; `ZEN_API_DOC=check` gates staleness in the CI clippy job, `=off` skips. Justfile recipes `api-doc` / `api-doc-check`.

### Fixed
- **Tile-threaded loop-filter compact-COW guards raced CDEF — worker panic
  (checked builds) or latent stale-byte clobber (`unchecked` builds)**
  (zenavif#30 root cause; the in-process futex hang that froze 4/220 zenavif
  two-pass conformance cells for 76-90 minutes). Two composing defects in
  `loopfilter_sb_dispatch`'s tile-threading path: (1)
  `compact_write_back_per_row` rewrote — and mutably guarded — every pixel of
  the filter's READ window, including the 7 tap rows/cols the filter only
  reads; dav1d's CDEF task legitimately reads (bottom-edge padding) and writes
  (its own blocks) in that zone concurrently, since dav1d's CDEF lag ahead of
  deblock is exactly 2 pad rows + max-modified rows. Fixed by diffing against
  a pristine pre-filter copy and writing back only modified pixel spans
  (`compact_write_back_per_row_diff`) — the write-set now equals dav1d's by
  construction. (2) The compact window used the luma tap reach (7) for chroma
  too; chroma deblock reads at most 3 rows/cols past the edge (wd6), and rows
  4..=7 above a chroma edge belong to the previous sbrow's CDEF writes
  (4-chroma-row lag). Fixed with plane-accurate `tap_before` (luma 7 /
  chroma 3). In default checked builds the race surfaced as an
  `overlapping DisjointMut` worker panic; in `unchecked` builds the wide
  write-back could instead silently overwrite concurrent CDEF output with
  stale copied bytes. Verified: 6000/6000 parallel decode-stress iterations
  clean on the trigger streams (pre-fix: ~10/12 loops panicked within their
  first ~20), full `decode_md5_verify` conformance + `decode_md5_committed`
  bit-exact, `tile_threading_parity` clean. Regression test
  `tests/tile_threading_overlap.rs::multi_threaded_cdef_lpf_race` (+ committed
  trigger vector `tile_threading_cdef_lpf_race.obu`) fails in 0.1 s on the
  pre-fix code. Repro harness: `examples/decode_stress.rs`.
- **A worker-thread panic wedged every decode wait forever** (zenavif#30
  aftermath-half). A worker that dies by panic can never complete its claimed
  task, so `task_counter` never reaches 0 and `rav1d_decode_frame`'s
  completion condvar wait — plus the frame-threaded submit/drain waits,
  `rav1d_flush`'s per-worker flushed wait, and the delayed-film-grain wait —
  blocked forever with 0 CPU (the observed `futex_` hang; the panic message
  sat unread in stderr). Now `TaskThreadData::panicked` is set by an unwind
  guard in `rav1d_worker_task` (which also marks the dead worker `flushed`
  and wakes every waiter class), and the waits fail the frame with `EGeneric`
  after parking live workers flush-style — decode+drop completes in
  milliseconds with an error instead. A panicked decoder stays failed
  (poisoned) but drops cleanly; fresh decoders are unaffected. Tested in
  `tests/worker_panic_recovery.rs` via the new private
  `__test_induce_worker_panic` feature hook.
- `docs/public-api/rav1d-disjoint-mut.txt` snapshot caught up with the merged
  `DisjointMut`-from-`&mut [V]` support (#416) — regenerated alongside this
  change.
- **`read_segment_id` clamp underflowed when no segment carried an active
  feature** (fuzz #415, found by `differential_dav1d` on arm64). When
  segmentation is enabled with `update_map`/`update_data` set but *every* segment
  feature bit is 0, `last_active_segid` stays at its `-1` sentinel. dav1d widens
  that `int8_t(-1)` to `unsigned` (`0xFFFFFFFF`) before the out-of-range check, so
  a decoded segment id is kept; this fork's port computed
  `(last_active_segid + 1) as u8 == 0` and then tested `seg_id >= 0`, which is
  *always* true for a `u8` — forcing the segment id to 0. The wrong id is written
  into `cur_segmap`, so a later block's neighbour-derived segment context (`l/a/al`)
  diverges, the symbol-decode trajectory drifts, and the MSAC overread that should
  trip the per-sbrow `cnt <= -15` guard never fires: rav1d-safe returned a frame
  for a malformed stream that dav1d rejects with `EINVAL`. The fix mirrors dav1d's
  unsigned semantics exactly — clamp only when `last_active_segid >= 0` — at both
  the pre-skip and post-skip `read_segment_id` sites in `src/decode.rs`. For
  `last_active_segid >= 0` (every well-formed segmented stream) the test is
  byte-for-byte the same as before. Same bug present in upstream rav1d. Verified on
  a native arm64 box: the repro now matches dav1d 1.4.1; `decode_md5_verify` output
  is identical fork-vs-baseline across all 558 conformance-vector frame hashes
  (no valid-stream pixel change); `decode_md5_committed` stays bit-exact; and the
  5,880-input `decode_obu` corpus shows zero `differential_dav1d` divergence.
- **aarch64 NEON 16bpc motion-compensation blend dst-slice overshoot** (issues
  #417–#421, commit 5a208de). The non-asm (default, safe-Rust SIMD) BPC16 path of
  seven aarch64 MC blend dispatchers — `avg`, `w_avg`, `mask`, `blend`,
  `blend_dir` (vertical + horizontal) and `w_mask` — sized the destination `u16`
  slice as `(h*stride + w)*2`, but `narrow_guard_mut` sizes the dst guard to only
  `(h-1)*stride + w` pixels (the last row needs `w` pixels, not a full stride).
  The slice index then overshot the guard by `stride − w`, panicking with `range
  end index N out of range for slice of length M` deep inside frame
  reconstruction for high-bit-depth (10/12-bit) inter prediction. One
  value-independent arithmetic bug copy-pasted across all seven sites — it
  affects the default build on aarch64, not just malformed input. The identical
  bug in the `mc_put`/resize BPC16 branch was already fixed (`arm_mc16_overshoot`
  regression test); this propagates the same `h.saturating_sub(1)` fix to the
  remaining dispatchers. Verified on a native arm64 box: all five fuzz repros and
  the 70-file arm64 farm crash corpus are repro-clean, `decode_md5_committed`
  stays bit-exact (incl. the HDR rec2020 16bpc vector), and `differential_dav1d`
  shows zero pixel divergence vs dav1d 1.4.1 on the repros plus 35 valid 10/12-bit
  frames. New regression tests `arm_mc16_{avg,mask,blend,blend_dir,w_mask}_overshoot`
  in `tests/safe_simd_crashes.rs` guard each path.
- **aarch64 NEON 16x64 / 64x16 DC-only inverse transform off-by-1** (issue #400):
  the `eob == 0` fast path for these two sizes routed through the shared
  `dc_only_rect64_{8,16}bpc` helper, which (a) applied the rect2 `√2` (`*2896`)
  input scaling **unconditionally** and (b) was called with intermediate
  `shift = 1` instead of `2`. 16x64 / 64x16 are 4:1, not rect2 (1:2), so dav1d's
  NEON `idct_dc` macro applies neither — its extra `sqrdmulh` is gated on
  `w == 2*h || h == 2*w` and it uses `idct_dc 16,64,2` / `idct_dc 64,16,2`. The
  result was a DC biased by 1 (e.g. a flat 16-wide column decoded as 126 where
  dav1d / the generic scalar give 127), so the same stream decoded to a different
  YUV MD5 on aarch64 and the `differential_dav1d` fuzz target diverged from dav1d
  on arm64 (`DIVERGENCE Y row 0 col 16: rav1d=126 dav1d=127`). Fixed by gating the
  extra scale on the true rect2 condition (32x64 / 64x32 still get it) and passing
  `shift = 2` for 16x64 / 64x16. Verified bit-exact: the `__simd_test`
  per-transform NEON-vs-scalar gate now reports zero 16x64 mismatches across the
  428-input arm64 fuzz-farm crash corpus, and the new committed vector
  `arm_itx_16x64_dc_rect2.obu` guards it in `tests/decode_md5_committed.rs` on x86
  **and** aarch64. The earlier #400 work fixed the non-DC 16x64 path but missed
  this DC-only fast path.
- **aarch64 NEON inverse transforms are now bit-exact with the spec** (issue #400):
  they were silently non-bit-exact — the NEON-vs-scalar unit tests passed only
  under a `MAX_DIFF = 15/40` tolerance, but AV1 decode requires exact output, so
  aarch64 produced wrong pixels (the same stream decoded to a different YUV MD5
  on aarch64 vs x86 — e.g. kodim03, alpha_noispe — and the `differential_dav1d`
  fuzz target diverged from dav1d on arm64). Root-caused and fixed per transform
  on a native arm64 box using the `__simd_test` dual-compute (NEON vs the generic
  spec itx) across the full dav1d conformance corpus until **zero** divergence:
  the large DCTs (64x64/64x32/32x64/16x64/64x16) had missing intermediate
  clipping, wrong `sh=32` coefficient layout, bogus rect2 scaling and wrong
  shifts — routed through the spec's clipping 1-D DCT with the NEON pixel add;
  the frequent 8x32/32x8/32x16 NEON 1-D kernels had wrong intermediate shifts
  (`<1>` vs `<2>`), a spurious/absent rect2 `√2` input scale, eob row-group skips
  that dropped non-zero rows, a 32x8 that ran the wrong (rect2) row kernel, and
  an off-by-1 DC path — all fixed bit-exact while keeping the NEON kernels. The
  earlier release shipped a temporary generic-scalar fallback (a6bc4d57); this
  removes it and re-enables the (now correct) NEON itx by default. Verified
  end-to-end as identical YUV MD5 to x86 on native arm64; `tests/decode_md5_committed.rs`
  guards it in CI on x86 **and** native arm64. The dispatch mapping was always
  correct (itx.rs uses a deliberately flipped row/col convention) — the bug was
  in the SIMD arithmetic.
- **aarch64 16bpc loop-restoration no longer panics on threaded decode**
  (issue #14, fixed upstream of this entry by da53bfa3): `selfguided_filter_16bpc`
  / `boxsum3_16bpc` in `looprestoration_arm` wrote one element past the
  `68×390 = 26520` box-sum buffer (`index out of bounds: len is 26520 but the
  index is 26520`) on Apple-Silicon/Graviton decodes of 16-bit content. Now
  verified fixed on real aarch64 under qemu (the `arm_boxsum3_oob_16bpc` /
  16bpc-HDR crash vectors decode cleanly, and `decode_md5_committed` confirms the
  16bpc output matches x86), and exercised in CI on the native arm64 runner.
- **Issue #14, full closure — verified on native aarch64 with conformant
  streams; last SGR bit-exactness gap fixed** (2026-07-04). The da53bfa3 fix
  family above was re-verified end-to-end on a Neoverse-N1 against real
  encodes (not just fuzz repros), which (a) reproduced every pre-fix failure
  mode of registry 0.5.7 and (b) surfaced one REMAINING 16bpc correctness bug,
  fixed here:
  * Registry 0.5.7 negative control (native arm64, conformant zenrav1e
    still-picture encodes): 8bpc SGR streams panic at
    `looprestoration_arm.rs:465:30` (`attempt to multiply with overflow`,
    overflow-checked builds — the exact failure that turned zenrav1e's ARM CI
    red on Windows/macOS/Linux arm64 via its `intrabc_fires_and_roundtrips…`
    test) and at `:399:13` (`len is 26520 but the index is 26520`, release
    bounds check); 10-bit SGR-3x3 streams panic at `:999:30` / `:935:13` (the
    exact issue-#14 report). Streams that avoid the panicking paths (wiener5,
    SGR-5x5-only) decoded **without error but with wrong pixels** on 0.5.7 —
    silent corruption, caught only by MD5 comparison.
  * **New fix: `selfguided_filter_16bpc` scaled its box sums with truncating
    shifts; the scalar (and dav1d) round** (`a + (1 << 2*bmin8 >> 1) >>
    2*bmin8`, same for `b`). The rounding addend is 0 at 8bpc — the 8bpc twin
    was unaffected, which is exactly why the port slip survived — but at
    10/12-bit the truncation skewed `z → x →` both SGR coefficients: the
    `__simd_test` census showed 421 LR mismatches over a 20-vector SGR/wiener
    suite (sgr_3x3 units up to ~78% wrong pixels, sgr_5x5 a few per unit),
    now **0**. The coefficient loop also now mirrors the scalar's exact
    integer widths (i32/u32 rather than i64/u64) so its overflow semantics
    can never diverge again.
  * Bit-identity matrix after the fix (native Neoverse-N1 NEON vs x86_64 vs
    `aomdec --rawvideo`, 23 conformant vectors: 8bpc/10bpc, 4:2:0/4:4:4,
    SGR-3x3/SGR-5x5/mix/wiener5, 256×256 + 640×256 max-width LR units):
    byte-identical everywhere LR is isolatable — all 8bpc vectors and all
    CDEF-off 10-bit vectors; `threads=1 == threads=8`, deterministic across
    repeat runs, debug == release. CDEF-on 16bpc frames still differ on
    aarch64 through the open CDEF divergence tracked in issue #414 (LR's own
    census is 0 there too).
  * Regression coverage: 6 committed conformant vectors +
    `lr_sgr_vectors_match_reference_md5` /
    `lr_sgr_vectors_threaded_match_reference_md5` (8-worker decode — the
    issue's `rav1d-worker-N` shape) in `tests/decode_md5_committed.rs`, MD5s
    pinned to the aomdec-verified x86 reference, hard-gated in CI on every
    platform including native arm64.
- **aarch64 wiener5 loop restoration produced wrong pixels for every 5-tap
  block** (issue #414 inventory, fixed by 710537f8 — entry backfilled): the
  NEON-path wiener5 special-cased `(center_tap=2, tap_count=5, tap_start=1)`
  and read `tmp[x+0..5]`, mis-centering the window by one column/row;
  `wiener_rust` always applies the 7-tap window centered at +3 with zero outer
  coefficients. Max pixel diff up to 119, ~13k mismatches across the dav1d
  corpus, silent (no panic). Also corrected the 16bpc inner's rounding
  (`round_bits_h = 3 + 2·(12bpc)`, no unconditional `+128` center tap,
  `round_bits_v = 11 − 2·(12bpc)`). Wiener5/wiener7 verified bit-exact on
  native arm64; the committed `lr_wiener5_8bpc_intrabc_s2` vector pins it in
  CI.
- **`compact_read_per_row` no longer allocates a `Vec` per filtered edge /
  predicted block under tile threading** (issue #17): decoding one 4K AVIF with
  `threads = 4` dropped from **517,414 heap allocations to 204** (heaptrack;
  the per-call `vec![0u8; …]` was 99.96% of allocations from that site, and on
  8K it was 99.98% of ~3M total). The compact per-row buffer that the loop
  filter, ipred CFL, and `with_pixel_guard_*` paths materialize when `n_tc > 1`
  is now drawn from a thread-local scratch pool (`recycle_compact_scratch`
  returns it after write-back) instead of freshly allocated each call — the same
  take/put pattern as the MC mid-buffer pool. Cheap on glibc, the old churn was
  pathological on the Windows allocator. **Tile threading stays fully enabled**:
  the pool is per-thread, so each tile worker owns its own buffer with no
  cross-tile aliasing. Output is byte-identical to the single-threaded path (new
  `tests/tile_threading_parity.rs` asserts ST==MT MD5 across 8bpc 4:2:0, 16bpc
  HDR, and the tile-overlap stream). Pure safe Rust — the default
  `forbid(unsafe_code)` build is unaffected.
- **v4x intra-prediction parity tests no longer fail under the parallel test
  suite on AVX-512 hosts** (issue #16): `z1/z2/z3_v4x_matches_avx2` summon AVX2 /
  AVX-512 tokens (via `summon_avx2`/`summon_avx512x`) at a `is_none()` gate and
  then *again* in `run_z*`. Those summons consult archmage's process-wide
  token-disable state, which `test_avg_token_permutations` /
  `test_wht4_token_permutations` mutate (through `for_each_token_permutation`)
  while iterating. When a permutation landed in that TOCTOU window the token
  read as disabled and the later `.expect("avx2"/"v4x")` panicked, failing the
  test (~20 % of parallel runs on a 7950X; invisible on CI, which lacks
  AVX-512). The `panicked at ipred.rs:2408` index-underflow messages in the
  reporter's log were a red herring — those are out-of-reach synthetic configs
  the test's `catch_unwind` probe intentionally skips; the noisy hook print is
  unrelated to the failure. Fix: each v4x test now holds
  `archmage::testing::lock_token_testing()` for its duration — the same mutex
  `for_each_token_permutation` acquires — so token state is stable end-to-end.
  Test-only change; decoder parallelism, performance, and accuracy unchanged.
- **`Rav1dTaskContext` no longer costs ~275 KB of stack per construction site**
  (2b311a5d, issue #15): the 250 KB `scratch` buffer is now a
  `Box<TaskContextScratch>` allocated zeroed directly on the heap via
  zerocopy's `new_box_zeroed()` — no stack temporary exists at any point.
  Previously `rav1d_open` reserved the full-size slot in its frame even on
  branches that never construct it (entry-probe faults on small caller
  stacks — embedders running decodes inside rayon pools needed 32 MB worker
  stacks), and every worker thread parked ~250 KB of its stack for its
  lifetime. `size_of::<Rav1dTaskContext>()`: 281,344 → 31,104 B, pinned by a
  regression test. `TaskContextScratch` is now `#[repr(C, align(64))]` with
  const asserts — its view-cast alignment was previously satisfied only
  incidentally by field offset. Decode A/B benched at parity
  (`benchmarks/taskctx_scratch_box_ab_2026-06-11.tsv`: 8K −0.5 %, 4K +1.5 %
  within overlapping spreads); full md5 conformance suite green.

### Changed
- **README accuracy pass for untrusted-input servers** (found via an insulated external-developer usability test): documented that `frame_size_limit` is a **total-pixel** (`width * height`) cap with a **120 MP default** (not unlimited, not the "4K" the old example implied) and that `0` disables it; added an honest **Cancellation** note that no in-flight decode-abort token exists (the frame-size cap is the only pre-decode guard); added a canonical crate-root `use rav1d_safe::{…}` import and reconciled the `Planes` variant names (`Planes::Depth8`/`Depth16` enum variants vs `Planes8`/`Planes16` structs); and documented the **planar-YUV output** format (`y()`/`u()`/`v()` accessors, `PixelLayout` subsampling, 8 vs 10/12-bit planes) (#411).
- **Test runner switched to `cargo-nextest`** (justfile + CI). Nextest runs each
  test in its own process, which removes a whole class of cross-test global-state
  races structurally — the archmage token-permutation vs. v4x parity race
  (issue #16), the process-global CPU-flags mask switched per tier by
  `decode_permutations` / `decode_cpu_levels`, and the worker-thread-count
  baselines in `thread_cleanup_test` — so the `--test-threads=1` workarounds in
  the permutation/conformance/threading CI jobs are gone. `.config/nextest.toml`
  adds a `default`/`ci` profile and a `heavy-threading` test-group that
  serializes the multi-threaded decode tests (reproduce_overlap, mt_stress,
  thread_cleanup) so separate processes don't oversubscribe small runners.
  Doctests (which nextest doesn't run) get a dedicated `cargo test --doc` step.
  Cross-target recipes that nextest can't host (`test-wasm`) and the QEMU
  `cross test` aarch64 path stay on `cargo test`. No production-code change.

## [0.5.7] - 2026-05-26

Memory-safety release. Supersedes 0.5.6 (yanked).

### Fixed
- **Depend on `rav1d-disjoint-mut 0.3.1`** (was `0.3.0`). The published 0.3.0 (2026-02-14) predated two safety fixes that existed only locally: the `PicBuf::from_vec_aligned` arithmetic-overflow OOB (reachable on 32-bit with crafted picture dimensions) and the index-trait sealing soundness fix. 0.5.6 was built against the stale registry 0.3.0 and shipped without them; 0.5.7 pulls the fixed 0.3.1. (dd60e0a)
- **Harden the picture allocator size arithmetic** (`alloc_picture_data`): `round_up` and the per-plane `sz + RAV1D_PICTURE_ALIGNMENT` were unchecked adds that could wrap on 32-bit for plane sizes just below `usize::MAX`, yielding an under-sized allocation. Both now use `checked_add` and surface `ENOMEM` instead of wrapping.

## [0.5.6] - 2026-05-26

Performance release: closes the safe-checked-vs-ASM 4K AVIF gap from **1.66× to ~1.55×**
(1.98× at the 2026-02 baseline → ~43% of the original gap closed) with zero new `unsafe` —
`#![forbid(unsafe_code)]` holds for the default build throughout. No public API changes
(`cargo semver-checks`: no semver update required).

This release also adds AVX-512 (x86-64-v4 / ICL) and modern-ARM dispatch tiers (all
bit-exact vs the AVX2/scalar reference; the benefit lands on native-512 hardware and is
flat on AVX2-double-pump CPUs like Zen4) plus two ISA-independent dav1d algorithmic ports
(decode_coefs index-offset, itx eob-pruning) that add a further ~3% on the AVX2 path
(per-change interleaved A/B measurements in `benchmarks/decode_coefs_index_offset_2026-05-26.md`).

### Performance — inverse transforms
- i16-packed `pmaddwd` DCT row passes for 8bpc dct8/dct16/dct32 (`dct{8,16,32}_row_pass_i16_simd`), replacing the i32 `mullo_epi32` path. `_mm256_madd_epi16` (5c Zen3) does a multiply-add per pair where i32 needed `mullo`+`add` (10c). Bit-exact vs scalar reference across seeded fuzz; wired into the live 8x8/16x16/32x32 transforms (b660acc, 04ff4cf, 94fa376)
- i16-packed `pmaddwd` DCT column passes (`dct{8,16,32}_col_pass_i16`) — completes the full i16 row+col pipeline for 8bpc DCT_DCT, extended to all 8bpc DCT-32 column sites (16x32/64x32/8x32) with runtime 8bpc/16bpc dispatch (aeba194, 1f2fd70, 802e22c)
- SIMD row 1D transforms for 8bpc dct8/16/32 across all sizes (8x8/8x16/8x32; 16x8/16x16/16x32/16x64; 32x8/32x16/32x32/32x64) + 16x16 dct/adst/flipadst-row mixed variants via `impl_16x16_transform_simd_row_{dct,adst}_col!` macros (464bcc3, edd008a, 6becd5b, 0caef66, a6a8457)
- DC-only fast path for DCT_DCT when `eob == 0` — broadcasts the scaled DC via width-tiered AVX2/SSE2/scalar `dc_only_add_{8,16}bpc` instead of running the full 2-pass transform (33f7402)
- Right-size scalar `inv_txfm_add` tmp buffer: 16 KB stack `[0; 64*64]` → const-generic `[[0i32; W]; H]`, dropping per-call memset (was ~3% of profile) (96e93d7)
- 8x8 add-to-dst: contiguous `loadu_256!` replaces 8× `_mm_set_epi32` scatter (27 `vmovd`/block) (ac490f9)

### Performance — loopfilter
- Token upgrade `X64V2Token` (SSE4.2) → `Desktop64`/`X64V3Token` (AVX2+FMA+BMI2) — unlocks YMM-width loopfilter. `cargo asm` confirms YMM in the dispatcher (was 0) (aa23eb8)
- YMM x8 widen for wd=4/8/16 v-filter and wd=8 h-filter: when two adjacent edges share a filter level, one 8-lane kernel processes both. wd=6 h-filter newly SIMD (was scalar). 4K AVIF loopfilter share ~9% → measurably reduced; ratio 1.60 → ~1.51 (c320fa1, 70576f3, cb09031, a2c2b24, 40b6b3e)
- Hoist target_feature region to outer dispatch + `#[arcane]`→`#[rite]` inner conversions so per-edge SIMD inlines into the per-superblock region (eee9005, 2d0d05c, f018bb5, ffbdca4)

### Performance — other
- SIMD `cfl_ac_dispatch` (4:2:0/4:2:2/4:4:4 8bpc) — `cfl_ac_rust` profile share 1.49% → 1.05% (6512b9b, d48656c)
- `ctx_refill` bulk 8-byte BSWAP load via `u64::from_be_bytes`, matching dav1d's 5-instruction refill (4e145dc)
- `with_pixel_guard_immut` + `decode_coefs` immutable a/l context slices + per-block guard fusion — reduces BorrowTracker traffic in `recon.rs`/`ipred_prepare.rs` hot paths (475e61d, 9f3ad5c, c33050e)

### Performance — dav1d algorithmic ports (ISA-independent, help all CPUs)
- `decode_coefs` index-offset optimization (dav1d `5ef6b241` + fix `63bf075a`): cache `slw`/`slh`/`tx2dszctx`, derive the eob context via shifts, and for `TX_CLASS_2D` index the `levels` scratch by `rc`/`rc_i` directly — dropping a per-coefficient `imul` in the single hottest loop of the profile (decode_coefs ~45% of 4K AVIF). **~2.6%** faster 4K AVIF (checked). Bit-exact: 14/14 MD5 + 6/6 cross-level conformance. (Our port matches dav1d's *fixed* form; the original `5ef6b241` had an `if (TX_CLASS_2D)` typo that disabled the hot-loop win in dav1d until June 2025.) (e24b479)
- itx EOB pruning (dav1d `ca83ee6d`): in `dct{32,16}_row_pass_i16_simd`, OR-reduce each 8-row batch (fused with the existing column load) and skip the butterfly+transpose for all-zero batches via `_mm_testz_si128` — self-evidently bit-exact (`out` is pre-zeroed). 32x32: **−0.8% mean / −1.6% median** 4K AVIF; 16x16: neutral on photos, helps sparse content. New sparse-batch unit tests. (ab61fc3, 8bf5e84)

### Added — AVX-512 (x86-64-v4 / ICL) and modern-ARM SIMD tiers
All bit-exact vs the AVX2/scalar reference (14/14 MD5). On AVX2-double-pump hardware (Zen4) these bench flat; the benefit lands on native-512 execution units (Intel Ice Lake server / Sapphire Rapids, Zen5).
- itx (`Server64`/X64V4): AVX-512 16-row DCT row passes + dct4/8/16/32 and identity (IDTX) column passes, wired into the wide transform sites (8997bc4, 75d2e68, 954a252, 4b81aa8, 0d74989)
- cdef (`Server64`/X64V4): AVX-512 8bpc directional filter (629e454)
- loopfilter (`Server64`/X64V4): AVX-512 wd=16 v-filter, 16 lanes (2e24a30, 572cb77)
- ipred (`X64V4xToken`/AVX-512ICL): z1/z2/z3 directional predictors via `vpermi2b` register-resident gather — z3 was previously scalar (5667552, 430e7a8, 4b76f48)
- ARM (`Arm64V2`/`Arm64V3`): scaffolded DotProd/I8MM 8-tap MC dispatch + a `summon_arm64v2/v3` runtime gate, cfg-gated OFF by default (the `vdotq_s32`/`vusdotq_s32` intrinsics are nightly-only). The stable default build is byte-for-byte unchanged NEON; bit-exactness of the DotProd source-bias correction is proven by an exhaustive host test (75f044c)

### Added
- `itx_mul2x_pack!` macro: bit-exact `pmaddwd`+`paddd`+`psrad` equivalent of dav1d's `ITX_MUL2X_PACK`, verified across 13,312 input pairings (8cadc48, a9fbce9)
- `transpose_8x8_i32!` macro: single source of truth for the 24-instruction in-register 8x8 i32 transpose (8cadc48)

### Changed
- Split the ~23K-line `src/safe_simd/itx.rs` into a thin `include!` shell + 10 part files (`itx/part01..part10`) for navigability. Single-module textual split — all 43 `macro_rules!` and 462 items stay mutually visible; zero visibility/macro/logic changes (813358b)
- `debug_assert!` i16-range precondition in `i32_to_i16_pair` — catches future callers feeding unclipped data that `_mm_packs_epi32` would silently saturate (d01bf9c)

### Fixed
- Resolve all clippy warnings (`-D warnings` clean on the default `--all-targets` build) (c916f44)

## [0.5.5] - 2026-04-17

### Changed
- Replace blanket `#![allow(clippy::all)]` with a targeted lint policy across 27 files: 22 specific lint allows (each documented with warning count and rationale) cover pervasive C-port patterns such as `precedence`, `too_many_arguments`, `unnecessary_cast`, `identity_op`, and `needless_range_loop`, while ~100 warnings for the remaining enabled lints were fixed in place (db99f94, #7)
- Add crate-level allows for seven additional clippy lints that fire on CI's clippy 1.87+ (`duplicated_attributes`, `manual_is_multiple_of`, `let_and_return`, `unnecessary_map_on_constructor`, `clone_on_copy`, `option_map_unit_fn`, `unnecessary_lazy_evaluations`) — all pervasive C-port patterns not worth fixing individually (8c6621c, #7)

### Fixed
- Restore `MsacAsmContext` visibility for asm builds: the lint-audit refactor had accidentally gated the type behind `#[cfg(not(asm_msac))]`, breaking the `asm`-feature CI job; the erroneous cfg gate is removed and the manual `Default` impl (needed because the conditionally-compiled `symbol_adapt16` fn-pointer field doesn't derive `Default`) is reinstated (96dde32, #7)

### Tests
- `CpuLevel` doctest in `src/managed.rs` builds `Settings` via `Settings::default()` plus field mutation instead of a bare struct expression, avoiding E0639 on `#[non_exhaustive]` structs across the crate boundary that doctests compile against (008a811)

### Internal
- Ignore the `.workongoing` coordination marker file (008a811)

## [0.5.4] - 2026-04-10

Patch release focused on concurrency safety, parser hardening, and fuzz coverage.

### Fixed
- CDEF tile threading race
- MV parsing overflow guard
- `wrapping_sub` in `read_golomb`

### Tests
- AV1 fuzz dictionary expansion
