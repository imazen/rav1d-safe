# Changelog

All notable changes to the `rav1d-safe` crate are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/). `rav1d-safe` is a fork of [rav1d](https://github.com/memorysafety/rav1d), which is itself a Rust port of [dav1d](https://code.videolan.org/videolan/dav1d); this fork adds archmage-based SIMD dispatch and removes the C FFI path. Entries below cover only changes made in this fork — upstream rav1d and dav1d release notes remain the canonical record for the shared decoder core. This file was backfilled from git history on 2026-04-15; the `[0.5.4]` date reflects the commit date of tag `v0.5.4` rather than the crates.io publish date.

## [Unreleased]

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
