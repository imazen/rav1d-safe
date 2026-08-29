#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global latch: true once any decoder in this process has used tile threading
/// (n_tc > 1). When true, compact_read/compact_write_back and
/// `with_pixel_guard_*` use per-row guards to avoid stride-padding overlap.
/// When false, they use a single wide guard, which is only sound with no
/// concurrent tile workers.
static TILE_THREADING: AtomicBool = AtomicBool::new(false);

/// Latch tile threading on. Called from decoder initialization.
///
/// MONOTONE ON PURPOSE — never stores `false`. This is process-global state
/// but decoders are per-instance, so a store of `false` from a
/// single-threaded `rav1d_open` used to clobber the flag for every
/// CONCURRENTLY LIVE multi-threaded decoder: their tile workers then took the
/// wide `narrow_guard` path, whose extent is `(h-1) * stride + w` — a 1x16
/// intra left-edge column claims 16,321 pixels to read 16 — and any
/// concurrent row write inside that span is a real conflict. In a checked
/// build that surfaced as a spurious `overlapping DisjointMut` panic in
/// `rav1d_prepare_intra_edges`; in an `unchecked` build it is an undetected
/// data race on picture memory.
///
/// Measured on this branch before the latch: 6 concurrent
/// `tile_threading_overlap --ignored` processes (whose
/// `single_threaded_no_panic` case opens an `n_tc == 1` decoder alongside
/// `multi_threaded_cdef_lpf_race`'s threaded ones) panicked in 8-9 of 24
/// runs. After: 0 of 24. See `benchmarks/p2_kernels_2026-08-07.meta`.
///
/// The cost of latching instead of tracking per decoder is that a process
/// which has ever opened a threaded decoder keeps the per-row path for its
/// single-threaded ones too. That is the correct direction to be wrong in,
/// and a purely single-threaded process never sets the latch at all.
pub fn set_tile_threading(active: bool) {
    if active {
        TILE_THREADING.store(true, Ordering::Relaxed);
    }
}

/// Check if tile threading is active.
pub fn tile_threading_active() -> bool {
    TILE_THREADING.load(Ordering::Relaxed)
}

/// THROWAWAY (`__probe_rect_hull` + `RAV1D_RECT_HULL=1`): make the recon/MC
/// block helpers take ONE strided-HULL guard under tile threading instead of
/// `h` per-row ones.
///
/// This is the falsification arm for the strided-rectangle counterfactual in
/// `bounds_probe::eval_rect`. That instrument measures, over
/// `8-bit/data` at t=8, that the hull at `block_mut`'s write-back intersects a
/// concurrently-live foreign reservation ~15.4 K times — ~8.0 K of them a
/// foreign WRITE — while the exact rectangle intersects **zero** times. A
/// reservation overlap involving a mutable record is a `DisjointMut` panic, so
/// this arm MUST fail to decode. If it passes, the counterfactual's `hull_ovl`
/// column is not measuring what it claims.
///
/// UNSOUND BY CONSTRUCTION — measurement only, never a shipping path.
#[cfg(feature = "__probe_rect_hull")]
pub fn rect_hull_arm() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("RAV1D_RECT_HULL").as_deref(), Ok("1")))
}

/// The default build: a constant, so the arm folds away entirely.
#[cfg(not(feature = "__probe_rect_hull"))]
#[inline(always)]
pub fn rect_hull_arm() -> bool {
    false
}

/// MARGINAL-PRICE ARM (`__probe_cdef_double` + `RAV1D_CDEF_DOUBLE=1`): take
/// every CDEF per-row registration TWICE, in ONE binary, changing the borrow
/// COUNT and nothing else.
///
/// # Why this and not a rectangle
///
/// `docs/RECT_RECORDS.md` §7b names five CDEF sites whose per-row registrations
/// a strided-rectangle record could collapse 7-8x (`rows` 7.27-8.00 on **1.000**
/// shards), a combined 118,624 registrations/frame — 20.8% of `c256x2048` t=8's
/// population — and says the arm to build FIRST is this one, because it prices
/// that whole population in one binary without implementing anything. If the
/// population is worth under ~1% there is nothing for a rectangle to win, and
/// #505 already measured that the largest site in the decoder (31.7% of the
/// population) is only 3.9-4.4% of the tracker's CPU.
///
/// # Sound by construction, and it is the ONLY sound direction
///
/// The extra reservation covers **exactly** the bytes the real one is about to
/// cover, and it is dropped before the real one is taken, so it cannot invent an
/// overlap that the real one would not already have found. Immutable sites get
/// an extra immutable guard; the one mutable site
/// (`safe_simd::cdef_arm::cdef_filter_block_*_neon`) gets an extra MUTABLE one,
/// because `find::<IS_MUT>` is a different scan and the mutable site must be
/// priced with the record shape it actually files.
///
/// ADDING a duplicate, rather than removing the original, is what makes this a
/// marginal price rather than a confounded one — removing a registration also
/// removes its guard's copy and its drop. `docs/AGENT_BRIEF.md` §6's closing
/// note is that this is the only arm that changes nothing else.
#[cfg(feature = "__probe_cdef_double")]
pub fn cdef_double_reads() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("RAV1D_CDEF_DOUBLE").as_deref(), Ok("1")))
}

/// The default build: a constant, so every `dup_rows*` call folds away.
#[cfg(not(feature = "__probe_cdef_double"))]
#[inline(always)]
pub fn cdef_double_reads() -> bool {
    false
}

/// Per-FILE ceilings on ONE tracked **partial** picture-plane reservation taken
/// while tile threading is active. See [`note_pic_extent`].
///
/// **Keyed by file, not by `file:line`, on purpose.** The bounds map's own
/// reconciliation had to compare multisets rather than `file:line` keys because
/// inserting two no-op declarations shifted every line number below them. A
/// line-keyed table would go stale on the next unrelated edit and then either
/// fail spuriously or, worse, stop matching and silently pass.
///
/// Only files whose reservation is bounded by a **constant** (a tap count, a
/// block dimension, a scratch width) get a tight entry. Files whose reservation
/// scales with the frame — `owned_recon.rs` (2,688 B observed),
/// `looprestoration.rs` (~2,050 B), `picture.rs`'s `copy_pixels_to` (4,096 B) —
/// would need a frame-size-relative bound, which this table cannot express, so
/// they fall through to [`TILE_THREADED_PIC_EXTENT_MAX_BYTES`].
///
/// Provenance for every number: `benchmarks/bounds_verdicts_2026-08-10.meta`,
/// measured by this module's own counters over the committed crash vectors plus
/// `dav1d-test-data` `8-bit/data` + `10-bit/data` at t=1 and t=8. Each entry is
/// the observed maximum; there is no slack, so the gate fails on the first byte
/// of any widening.
pub const PIC_EXTENT_CEILINGS: &[(&str, usize)] = &[
    // `LfBlock::fill`'s per-row read guard and `write_back`'s per-row write.
    // Both are bounded by the scratch width `LF_BW = 16` pixels, so 32 B at
    // 16bpc. Measured max 32 B over 90,170,282 reservations. This is the
    // decoder's largest guard site AND the one with the least measured headroom
    // (60 B to a concurrent `loopfilter.rs:887:14` write-back), so it is the
    // entry that matters most.
    ("src/loopfilter.rs", 32),
    // CDEF's per-row picture reads: <= 16 px, 32 B at 16bpc. Measured max 24 B
    // over 192,238,382 reservations.
    ("src/safe_simd/cdef_arm.rs", 32),
    // Loop-restoration stripe row reads: 4 px. Measured max 8 B over 1,384,827.
    ("src/lr_apply.rs", 32),
];

/// Fallback ceiling for picture-plane files with no [`PIC_EXTENT_CEILINGS`]
/// entry. Set from the same measurement; catches gross widenings (a strided
/// hull at 4K is 50-60 KB) without pretending to bound the frame-scaling sites
/// tightly.
pub const TILE_THREADED_PIC_EXTENT_MAX_BYTES: usize = 64;

/// Observation counters for [`note_pic_extent`], so every ceiling above can be
/// re-derived rather than trusted, and so the gate can prove it is not vacuous.
///
/// Interning is on the `&'static str` POINTER of `Location::file()` — one
/// static per source file, so pointer identity is file identity and the compare
/// is free. `site_probe` uses the same trick on the `Location` itself.
#[cfg(feature = "probe-sites")]
pub mod extent_budget {
    use core::panic::Location;
    use std::string::String;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
    use std::vec::Vec;

    /// Every partial picture-plane reservation that reached the check.
    pub static CHECKS: AtomicU64 = AtomicU64::new(0);
    /// ... of those, the ones taken while tile threading was active.
    pub static CHECKS_TT: AtomicU64 = AtomicU64::new(0);
    /// Largest partial reservation seen at all, and while tile threading was on.
    pub static MAX_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static MAX_BYTES_TT: AtomicUsize = AtomicUsize::new(0);
    /// Whole-component reservations, which the invariant deliberately exempts.
    pub static WHOLE: AtomicU64 = AtomicU64::new(0);

    /// `(file, max_bytes_under_tile_threading, site_of_that_max, n)`.
    #[allow(clippy::type_complexity)]
    static PER_FILE: Mutex<Vec<(&'static str, usize, &'static Location<'static>, u64, usize)>> =
        Mutex::new(Vec::new());
    /// Largest number of picture ROWS one reservation spanned. A value above 1
    /// means some reservation covered an inter-row gap, which is #469/#475's
    /// defect and is what the one-row bound exists to forbid.
    pub static MAX_ROWS_TT: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn record(bytes: usize, tt: bool, rows: usize, loc: &'static Location<'static>) {
        CHECKS.fetch_add(1, Relaxed);
        MAX_BYTES.fetch_max(bytes, Relaxed);
        if !tt {
            return;
        }
        CHECKS_TT.fetch_add(1, Relaxed);
        MAX_BYTES_TT.fetch_max(bytes, Relaxed);
        MAX_ROWS_TT.fetch_max(rows, Relaxed);
        let Ok(mut v) = PER_FILE.lock() else { return };
        let file = loc.file();
        for e in v.iter_mut() {
            if core::ptr::eq(e.0, file) || e.0 == file {
                e.3 += 1;
                e.4 = e.4.max(rows);
                if bytes > e.1 {
                    e.1 = bytes;
                    e.2 = loc;
                }
                return;
            }
        }
        v.push((file, bytes, loc, 1, rows));
    }

    pub(super) fn record_whole() {
        WHOLE.fetch_add(1, Relaxed);
    }

    /// Per-file maxima under tile threading, sorted widest first:
    /// `(file, max_bytes, "file:line:col" of that max, registrations, max_rows)`.
    pub fn per_file() -> Vec<(String, usize, String, u64, usize)> {
        let Ok(v) = PER_FILE.lock() else {
            return Vec::new();
        };
        let mut out: Vec<_> = v
            .iter()
            .map(|(f, b, l, n, r)| {
                (
                    (*f).to_string(),
                    *b,
                    format!("{}:{}:{}", l.file(), l.line(), l.column()),
                    *n,
                    *r,
                )
            })
            .collect();
        out.sort_by_key(|e| core::cmp::Reverse(e.1));
        out
    }

    /// `(checks, checks_under_tile_threading, max_bytes, max_bytes_tt,
    /// whole_component_reservations)`.
    pub fn report() -> (u64, u64, usize, usize, u64) {
        (
            CHECKS.load(Relaxed),
            CHECKS_TT.load(Relaxed),
            MAX_BYTES.load(Relaxed),
            MAX_BYTES_TT.load(Relaxed),
            WHOLE.load(Relaxed),
        )
    }
}

/// The tight [`PIC_EXTENT_CEILINGS`] entry for `file`, if it has one.
///
/// `None` means the file is held to one picture row instead, which
/// [`pic_extent_ceiling`] cannot express without knowing the plane. The gate
/// uses this to apply exactly the bound the decoder applies.
#[inline]
pub fn pic_extent_ceiling_const(file: &str) -> Option<usize> {
    let mut i = 0;
    while i < PIC_EXTENT_CEILINGS.len() {
        let (f, c) = PIC_EXTENT_CEILINGS[i];
        if file.ends_with(f) {
            return Some(c);
        }
        i += 1;
    }
    None
}

/// The ceiling that applies to a reservation taken at `file`, on a plane whose
/// row is `row_bytes` long.
///
/// A file with a [`PIC_EXTENT_CEILINGS`] entry is held to that constant. Every
/// other file is held to **one picture row**, which is the coarsest extent that
/// cannot contain an inter-row gap — the thing #469's rectangle and #475's hull
/// reserved and that AV1's column-partitioned tiles make unsafe.
#[inline]
pub fn pic_extent_ceiling(file: &str, row_bytes: usize) -> usize {
    let mut i = 0;
    while i < PIC_EXTENT_CEILINGS.len() {
        let (f, c) = PIC_EXTENT_CEILINGS[i];
        // `Location::file()` is the path as written to rustc, so it already
        // starts with `src/` / `include/` in this crate. `ends_with` also makes
        // the entry survive a build that prefixes an absolute path.
        if file.ends_with(f) {
            return c;
        }
        i += 1;
    }
    row_bytes.max(TILE_THREADED_PIC_EXTENT_MAX_BYTES)
}

/// The bounds map's standing extent invariant, checked at the ONE funnel every
/// tracked picture-plane reservation passes through.
///
/// > While tile threading is active, a **partial** reservation against a
/// > picture plane may not exceed [`TILE_THREADED_PIC_EXTENT_MAX_BYTES`].
///
/// # Why this invariant and not "reserved <= footprint"
///
/// `docs/BOUNDS_MAP.md` measured both halves. At t=8 the ratio half is already
/// 1.000 at every hot site — there is no over-reservation left to assert
/// against — while the *absolute* extent is what decides whether a reservation
/// collides with another worker's write. The three refuted attempts
/// (#469 strided rectangle, #475 hull, #485 read band) all widened the absolute
/// extent; two of them kept `reserved == footprint` while doing it, so a ratio
/// assertion would have passed all three. The measured headroom at the decoder's
/// largest site (`loopfilter.rs:710:14`) is **60 bytes** to a concurrent
/// `loopfilter.rs:887:14` write-back, so a ceiling in that neighbourhood is the
/// property that actually gates the failure.
///
/// # What is exempt, and why that is not a hole
///
/// * **Whole-component reservations** (`full_guard`, `full_guard_mut`,
///   `copy_pixels_to`, `copy_from`). A reservation of the entire plane is
///   unambiguous, greppable and deliberate; the map measured those sites
///   (`mc.rs:121:61`, `mc.rs:1342:44`, `mc_arm.rs:5971:41`) as having **no
///   concurrent foreign write at any distance**, because they read reference
///   frames that are immutable for the whole decode. They are a tracker-COST
///   question, not a conflict question.
/// * **Single-element access** (`index`, `index_mut`). One element is the
///   smallest reservation expressible; there is nothing to widen.
/// * **Everything at t=1.** The hull paths deliberately over-reserve by
///   153x-1680x there and it is correct: with no second worker there is nobody
///   for the inter-row gaps to collide with, and the count reduction is worth
///   2.6% (`docs/MUT_RECON_KERNELS.md` §11d).
///
/// # Cost
///
/// Compiled only under `debug_assertions` or `--features probe-sites`. The
/// default release build has no counter, no atomic load and no branch here.
#[cfg(any(debug_assertions, feature = "probe-sites"))]
#[inline]
#[track_caller]
pub(crate) fn note_pic_extent(bytes: usize, whole_component: bool, row_bytes: usize) {
    {
        if whole_component {
            #[cfg(feature = "probe-sites")]
            extent_budget::record_whole();
            return;
        }
        let tt = tile_threading_active();
        let loc = core::panic::Location::caller();
        let rows = if row_bytes == 0 {
            1
        } else {
            bytes.div_ceil(row_bytes)
        };
        #[cfg(feature = "probe-sites")]
        extent_budget::record(bytes, tt, rows, loc);
        let ceiling = pic_extent_ceiling(loc.file(), row_bytes);
        if tt && bytes > ceiling {
            panic!(
                "{}:{}:{} took a {bytes} B picture-plane reservation while tile \
                 threading is active; the measured ceiling for that file is \
                 {ceiling} B.\n\
                 This is the bounds map's standing extent invariant \
                 (docs/BOUNDS_MAP.md, PIC_EXTENT_CEILINGS). A wider reservation \
                 collides with a concurrent foreign WRITE at the rate the map's \
                 widening-budget column predicts — the decoder's largest guard \
                 site has 60 bytes of measured headroom, and the three refuted \
                 attempts (#469 rectangle, #475 hull, #485 band) all failed \
                 exactly here.\n\
                 If the widening is deliberate: price it against the budget \
                 table FIRST (`--features __probe_bounds`), then raise the \
                 ceiling WITH the measurement in the same commit. \
                 (one picture row here is {row_bytes} B; this reservation spans \
                 {rows} rows)",
                loc.file(),
                loc.line(),
                loc.column()
            );
        }
    }
}

thread_local! {
    /// Reusable scratch buffers backing [`WithOffset::compact_read_per_row`]
    /// (and the pristine copy kept by the loopfilter's diff write-back).
    /// Pulled on read, returned via [`recycle_compact_scratch`] after the
    /// compact buffer has been written back / consumed.
    ///
    /// Under tile threading the loop filter / ipred / `with_pixel_guard_*`
    /// paths materialize a per-edge / per-block compact buffer; allocating one
    /// `Vec` per call produced ~3M alloc+free pairs decoding a single 8K frame
    /// (issue #17) — cheap on glibc, pathological on the Windows allocator.
    /// Reusing buffers per thread eliminates that churn. It is per-thread,
    /// so each tile thread owns its own buffers with no cross-tile aliasing —
    /// tile threading stays enabled. Two slots because the loopfilter's
    /// tile-threading path holds a working copy and a pristine copy at once.
    static COMPACT_SCRATCH: Cell<[Option<Vec<u8>>; 2]> = const { Cell::new([None, None]) };
}

/// Take a scratch buffer from the thread-local pool (empty `Vec` if the pool
/// is dry). Pair with [`recycle_compact_scratch`].
#[inline]
pub(crate) fn take_compact_scratch() -> Vec<u8> {
    COMPACT_SCRATCH.with(|c| {
        let mut slots = c.take();
        let buf = slots[0].take().or_else(|| slots[1].take());
        c.set(slots);
        buf.unwrap_or_default()
    })
}

/// Return a compact scratch buffer to the thread-local pool so the next
/// [`WithOffset::compact_read_per_row`] on this thread can reuse the allocation.
///
/// Pairs with the `Vec` that `compact_read_per_row` returns. Calling this is an
/// optimization, not a requirement: simply dropping the `Vec` is always correct
/// (the next read just allocates a fresh buffer). If a panic unwinds past a
/// pending recycle the buffer is freed normally — no unsoundness, the pool is
/// merely empty for the next call (same contract as the MC mid-buffer pool).
///
/// Crate-internal: this is decoder plumbing, not part of the public API.
#[inline]
pub(crate) fn recycle_compact_scratch(buf: Vec<u8>) {
    // Keep the larger buffers so re-entrant reads (a read nested inside a
    // `with_pixel_guard_*` closure) don't shrink the pool.
    COMPACT_SCRATCH.with(|c| {
        let mut slots = c.take();
        // Fill an empty slot, else replace the smallest slot if `buf` is larger.
        let smallest = if slots[0].is_none()
            || slots[1].is_some()
                && slots[0].as_ref().map(Vec::capacity) < slots[1].as_ref().map(Vec::capacity)
        {
            0
        } else {
            1
        };
        match &slots[smallest] {
            Some(existing) if existing.capacity() >= buf.capacity() => {}
            _ => slots[smallest] = Some(buf),
        }
        c.set(slots);
    });
}

use crate::include::common::bitdepth::BitDepth;

/// Execute a closure with mutable byte access to a w×h pixel block.
///
/// The closure receives `(bytes, offset, stride)` — a mutable byte slice, the
/// BYTE offset to the first pixel, and the byte stride between rows.
///
/// This is a closure-shaped front end for [`WithOffset::block_mut`], which owns
/// the single-threaded-vs-tile-threaded policy. The two exist because call sites
/// come in two shapes — a closure suits the x86-64 and generic dispatchers, an
/// RAII guard suits the aarch64/wasm ones that need the slice to outlive a
/// borrow — but there must be exactly ONE implementation of the policy behind
/// them. When there were two, the aarch64 half silently kept reserving inter-row
/// gaps long after the x86-64 half stopped, which is the bug this whole change
/// exists to fix.
#[inline]
pub fn with_pixel_guard_mut<BD: BitDepth, R>(
    pic: &crate::src::with_offset::WithOffset<&Rav1dPictureDataComponent>,
    w: usize,
    h: usize,
    f: impl FnOnce(&mut [u8], usize, isize) -> R,
) -> R {
    let mut block = pic.block_mut::<BD>(w, h);
    // `BlockMut::base()` is a PIXEL index — 0 for a compact block, and row 0's
    // index inside the hull, `(h-1)*|stride|`, for a direct block on a
    // negative-stride picture (whose hull starts at the last row). This
    // closure's contract is a BYTE offset, so the scaling happens here rather
    // than inside `BlockMut`, whose own users index the byte slice directly.
    let offset = block.base() * core::mem::size_of::<BD::Pixel>();
    let stride = block.byte_stride();
    // `block` is dropped at the end of this expression, i.e. after `f` has run:
    // a compact block therefore writes back before the caller observes `R`,
    // exactly as the previous open-coded version did.
    f(block.as_mut_bytes(), offset, stride)
}

/// Execute a closure with read-only byte access to a w×h pixel block.
///
/// In single-threaded mode: zero-copy via `narrow_guard`.
/// In multi-threaded mode: copies into a compact buffer with per-row guards
/// (one guard per row, each covering exactly `w` pixels) so the helper is
/// safe to use alongside concurrent tile-thread writes on adjacent blocks.
///
/// The closure receives `(bytes, offset, stride)` — an immutable byte slice,
/// the byte offset to the first pixel, and the byte stride between rows.
///
/// Counterpart to [`with_pixel_guard_mut`]; see [`feedback_mt_safe_batching`]
/// for the rationale on why wide guards (`strided_slice_mut` / `narrow_guard`)
/// are unsafe under tile threading even in the immutable case (they cover
/// `(h-1)*stride + w` bytes and overlap with adjacent tiles' mutable ranges).
#[inline]
pub fn with_pixel_guard_immut<BD: BitDepth, R>(
    pic: &crate::src::with_offset::WithOffset<&Rav1dPictureDataComponent>,
    w: usize,
    h: usize,
    f: impl FnOnce(&[u8], usize, isize) -> R,
) -> R {
    use crate::src::strided::Strided as _;
    let pixel_size = core::mem::size_of::<BD::Pixel>();
    if tile_threading_active() && !rect_hull_arm() {
        let (buf, byte_stride) = pic.compact_read_per_row::<BD>(w, h);
        let result = f(&buf, 0, byte_stride as isize);
        recycle_compact_scratch(buf);
        result
    } else {
        let (guard, base) = pic.narrow_guard::<BD>(w, h);
        let bytes = guard.as_bytes();
        let offset = base * pixel_size;
        let stride = pic.stride();
        f(bytes, offset, stride)
    }
}
#[cfg(feature = "c-ffi")]
use crate::include::common::validate::validate_input;
#[cfg(feature = "c-ffi")]
use crate::include::dav1d::common::Dav1dDataProps;
use crate::include::dav1d::common::Rav1dDataProps;
use crate::include::dav1d::headers::DRav1d;
use crate::include::dav1d::headers::Dav1dFrameHeader;
use crate::include::dav1d::headers::Dav1dITUTT35;
use crate::include::dav1d::headers::Dav1dPixelLayout;
use crate::include::dav1d::headers::Dav1dSequenceHeader;
use crate::include::dav1d::headers::Rav1dContentLightLevel;
use crate::include::dav1d::headers::Rav1dFrameHeader;
use crate::include::dav1d::headers::Rav1dITUTT35;
use crate::include::dav1d::headers::Rav1dMasteringDisplay;
use crate::include::dav1d::headers::Rav1dPixelLayout;
use crate::include::dav1d::headers::Rav1dSequenceHeader;
#[cfg(feature = "c-ffi")]
use crate::src::assume::assume;
#[cfg(feature = "c-ffi")]
use crate::src::c_arc::RawArc;
use crate::src::disjoint_mut::DisjointImmutGuard;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::disjoint_mut::DisjointMutGuard;
#[cfg(feature = "c-ffi")]
use crate::src::disjoint_mut::ExternalAsMutPtr;
use crate::src::disjoint_mut::SliceBounds;
#[cfg(feature = "c-ffi")]
use crate::src::error::Dav1dResult;
use crate::src::error::Rav1dError;
#[cfg(feature = "c-ffi")]
use crate::src::error::Rav1dError::EINVAL;
use crate::src::error::Rav1dResult;
#[cfg(not(feature = "c-ffi"))]
use crate::src::mem::MemPool;
#[cfg(asm_fn_ptrs)]
use crate::src::pixels::Pixels;
#[cfg(feature = "c-ffi")]
use crate::src::send_sync_non_null::SendSyncNonNull;
use crate::src::strided::Strided;
use crate::src::with_offset::WithOffset;
#[allow(non_camel_case_types)]
type ptrdiff_t = isize;
#[cfg(feature = "c-ffi")]
#[allow(non_camel_case_types)]
type uintptr_t = usize;
#[cfg(not(feature = "c-ffi"))]
use rav1d_disjoint_mut::PicBuf;
use std::array;
use std::ffi::c_int;
#[cfg(feature = "c-ffi")]
use std::ffi::c_void;
use std::iter;
use std::mem;
#[cfg(feature = "c-ffi")]
use std::ptr::NonNull;
use std::sync::Arc;
#[cfg(feature = "c-ffi")]
use to_method::To as _;
use zerocopy::FromBytes;
use zerocopy::Immutable;
use zerocopy::IntoBytes;
use zerocopy::KnownLayout;

pub(crate) const RAV1D_PICTURE_ALIGNMENT: usize = 64;
pub const DAV1D_PICTURE_ALIGNMENT: usize = RAV1D_PICTURE_ALIGNMENT;

/// A raw pointer to picture data that is `Send + Sync`.
///
/// Uses `SendSyncNonNull` from `rav1d-disjoint-mut` so that `Send + Sync` are
/// automatically derived without any `unsafe impl` in this crate.
///
/// Thread safety rationale:
///
/// 1. **Owned buffers**: Points into a `Vec<u8>` stored in
///    the same `Rav1dPictureData` struct, behind `Arc`. The Vec cannot be
///    grown or reallocated through `&Rav1dPictureData`, so the pointer stays
///    valid. Concurrent access is tracked by `DisjointMut`.
///
/// 2. **Borrowed scratch buffers** (`wrap_buf`): Points into a `&mut [BD::Pixel]`
///    that outlives the component. These are single-threaded temporaries in
///    `recon.rs` — they are never shared across threads.
#[cfg(feature = "c-ffi")]
#[derive(Clone, Copy)]
#[repr(transparent)]
struct PicDataPtr(SendSyncNonNull<u8>);

#[cfg(feature = "c-ffi")]
#[allow(unsafe_code)]
impl PicDataPtr {
    /// Create a dangling pointer with [`RAV1D_PICTURE_ALIGNMENT`] alignment.
    fn dangling_aligned() -> Self {
        // SAFETY: NonNull::dangling() is Send+Sync-safe (no real data behind it).
        Self(unsafe {
            SendSyncNonNull::new_unchecked(NonNull::<AlignedPixelChunk>::dangling().cast())
        })
    }

    /// Create from a raw pointer. Returns `None` if null.
    fn new(ptr: *mut u8) -> Option<Self> {
        // SAFETY: The pointer comes from a Vec<u8> or &mut [u8] which are Send+Sync.
        NonNull::new(ptr).map(|nn| Self(unsafe { SendSyncNonNull::new_unchecked(nn) }))
    }

    /// Create from a `NonNull<u8>`.
    fn from_non_null(ptr: NonNull<u8>) -> Self {
        // SAFETY: The pointer comes from a C allocator callback, caller ensures validity.
        Self(unsafe { SendSyncNonNull::new_unchecked(ptr) })
    }

    /// Get the raw pointer.
    fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr().as_ptr()
    }

    /// Check if the pointer is aligned to [`AlignedPixelChunk`].
    fn is_chunk_aligned(self) -> bool {
        self.0.as_ptr().cast::<AlignedPixelChunk>().is_aligned()
    }
}

#[derive(Default)]
#[repr(C)]
pub struct Dav1dPictureParameters {
    pub w: c_int,
    pub h: c_int,
    pub layout: Dav1dPixelLayout,
    pub bpc: c_int,
}

// TODO(kkysen) Eventually the [`impl Default`] might not be needed.
#[derive(Clone, Default)]
#[repr(C)]
pub(crate) struct Rav1dPictureParameters {
    pub w: c_int,
    pub h: c_int,
    pub layout: Rav1dPixelLayout,
    pub bpc: u8,
}

impl From<Dav1dPictureParameters> for Rav1dPictureParameters {
    fn from(value: Dav1dPictureParameters) -> Self {
        let Dav1dPictureParameters { w, h, layout, bpc } = value;
        Self {
            w,
            h,
            layout: layout.try_into().unwrap(),
            bpc: bpc.try_into().unwrap(),
        }
    }
}

impl From<Rav1dPictureParameters> for Dav1dPictureParameters {
    fn from(value: Rav1dPictureParameters) -> Self {
        let Rav1dPictureParameters { w, h, layout, bpc } = value;
        Self {
            w,
            h,
            layout: layout.into(),
            bpc: bpc.into(),
        }
    }
}

#[cfg(feature = "c-ffi")]
#[derive(Default)]
#[repr(C)]
pub struct Dav1dPicture {
    pub seq_hdr: Option<NonNull<Dav1dSequenceHeader>>,
    pub frame_hdr: Option<NonNull<Dav1dFrameHeader>>,
    pub data: [Option<NonNull<c_void>>; 3],
    pub stride: [ptrdiff_t; 2],
    pub p: Dav1dPictureParameters,
    pub m: Dav1dDataProps,
    pub content_light: Option<NonNull<Rav1dContentLightLevel>>,
    pub mastering_display: Option<NonNull<Rav1dMasteringDisplay>>,
    pub itut_t35: Option<NonNull<Dav1dITUTT35>>,
    pub n_itut_t35: usize,
    pub reserved: [uintptr_t; 4],
    pub frame_hdr_ref: Option<RawArc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>, // opaque, so we can change this
    pub seq_hdr_ref: Option<RawArc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>>, // opaque, so we can change this
    pub content_light_ref: Option<RawArc<Rav1dContentLightLevel>>, // opaque, so we can change this
    pub mastering_display_ref: Option<RawArc<Rav1dMasteringDisplay>>, // opaque, so we can change this
    pub itut_t35_ref: Option<RawArc<DRav1d<Box<[Rav1dITUTT35]>, Box<[Dav1dITUTT35]>>>>, // opaque, so we can change this
    pub reserved_ref: [uintptr_t; 4],
    pub r#ref: Option<RawArc<Rav1dPictureData>>, // opaque, so we can change this
    pub allocator_data: Option<SendSyncNonNull<c_void>>,
}

#[derive(Clone, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C, align(64))]
pub struct AlignedPixelChunk([u8; RAV1D_PICTURE_ALIGNMENT]);

const _: () = assert!(mem::align_of::<AlignedPixelChunk>() == RAV1D_PICTURE_ALIGNMENT);
const _: () = assert!(mem::size_of::<AlignedPixelChunk>() == RAV1D_PICTURE_ALIGNMENT);

/// The guaranteed length multiple of [`Rav1dPictureDataComponentInner`]s.
/// This is checked and [`assume`]d.
const RAV1D_PICTURE_GUARANTEED_MULTIPLE: usize = 64;

/// Actual [`Rav1dPictureData`]'s components should be multiples of this,
/// as this is guaranteed by [`Rav1dPicAllocator::alloc_picture_callback`],
/// though wrapped buffers may only be [`RAV1D_PICTURE_GUARANTEED_MULTIPLE`].
const RAV1D_PICTURE_MULTIPLE: usize = 64 * 64;

/// The inner buffer type for picture data components.
///
/// For c-ffi: a struct with raw pointer, length, and stride (supports C allocator callbacks).
/// Without c-ffi: aliases [`StridedBuf`] from the disjoint-mut crate (all unsafe confined there).
#[cfg(feature = "c-ffi")]
pub struct Rav1dPictureDataComponentInner {
    /// A ptr to the start of this slice of `BitDepth::Pixel`s*,
    /// even if [`Self::stride`] is negative.
    ///
    /// It is aligned to [`RAV1D_PICTURE_ALIGNMENT`].
    ptr: PicDataPtr,

    /// The length of [`Self::ptr`] in [`u8`] bytes.
    ///
    /// It is a multiple of [`RAV1D_PICTURE_GUARANTEED_MULTIPLE`].
    len: usize,

    /// The stride of [`Self::ptr`] in [`u8`] bytes.
    stride: isize,
}

/// Without c-ffi, the inner buffer is a [`PicBuf`] from the disjoint-mut crate.
/// All unsafe for `AsMutPtr` is confined to that crate. Stride is stored separately
/// in [`Rav1dPictureDataComponent`].
#[cfg(not(feature = "c-ffi"))]
pub type Rav1dPictureDataComponentInner = PicBuf;

#[cfg(feature = "c-ffi")]
impl Rav1dPictureDataComponentInner {
    /// `len` and `stride` are in terms of [`u8`] bytes.
    ///
    /// # Safety
    ///
    /// `ptr`, `len`, and `stride` must follow the requirements of [`Dav1dPicAllocator::alloc_picture_callback`].
    unsafe fn new(ptr: Option<NonNull<u8>>, len: usize, stride: isize) -> Self {
        let ptr = match ptr {
            None => {
                return Self {
                    ptr: PicDataPtr::dangling_aligned(),
                    len: 0,
                    stride,
                };
            }
            Some(ptr) => ptr,
        };

        assert!(len != 0); // If `len` was 0, `ptr` should've been `None`.
        assert!(ptr.cast::<AlignedPixelChunk>().is_aligned());

        let ptr = if stride < 0 {
            let ptr = ptr.as_ptr();
            // SAFETY: According to `Dav1dPicAllocator::alloc_picture_callback`,
            // if the `stride` is negative, this is how we get the start of the data.
            // `.offset(-stride)` puts us at one element past the end of the slice,
            // and `.sub(len)` puts us back at the start of the slice.
            let ptr = unsafe { ptr.offset(-stride).sub(len) };
            PicDataPtr::new(ptr).unwrap()
        } else {
            PicDataPtr::from_non_null(ptr)
        };
        // Guaranteed by `Dav1dPicAllocator::alloc_picture_callback`.
        assert!(len % RAV1D_PICTURE_MULTIPLE == 0);
        Self { ptr, len, stride }
    }

    /// # Safety
    ///
    /// As opposed to [`Self::new`], this is safe because `buf` is a `&mut` and thus unique,
    /// so it is sound to further subdivide it into disjoint `&mut`s.
    ///
    /// # Panics
    ///
    /// `buf` must satisfy the same two invariants [`Self::new`] enforces on the
    /// C allocator's buffers, because this constructs the same type and the
    /// same [`ExternalAsMutPtr`] impl [`assume`]s both of them:
    ///
    /// 1. it must START on a [`RAV1D_PICTURE_ALIGNMENT`]-byte boundary, and
    /// 2. its BYTE length must be a multiple of
    ///    [`RAV1D_PICTURE_GUARANTEED_MULTIPLE`].
    ///
    /// Neither can be relaxed into a copy here: this path is zero-copy on
    /// purpose, and its callers (`recon.rs`'s OBMC `lap` and interintra `tmp`)
    /// read their results back out of `buf` afterwards — which is why the
    /// safe-mode twin's `copy_pixels_to` call sites are
    /// `cfg(not(feature = "c-ffi"))`.
    ///
    /// A plain `Vec<BD::Pixel>` satisfies (1) only by luck. Every production
    /// caller satisfies it as a TYPE property: the scratch buffers in
    /// `src/internal.rs` are `#[repr(C, align(64))]`, pinned by the
    /// `scratch_alignment` assertions there.
    pub fn wrap_buf<BD: BitDepth>(buf: &mut [BD::Pixel], stride: usize) -> Self {
        let buf = IntoBytes::as_mut_bytes(buf);
        let ptr = PicDataPtr::new(buf.as_mut_ptr()).unwrap();
        assert!(ptr.is_chunk_aligned());
        let len = buf.len();
        assert!(len % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0);
        let stride = (stride * mem::size_of::<BD::Pixel>()) as isize;
        Self { ptr, len, stride }
    }
}

// SAFETY: We only store the raw pointer (via PicDataPtr), so we never materialize a `&mut`.
#[cfg(feature = "c-ffi")]
#[allow(unsafe_code)]
unsafe impl ExternalAsMutPtr for Rav1dPictureDataComponentInner {
    type Target = u8;

    #[inline] // Inline so callers can see the assume.
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        // SAFETY: Safe to dereference by unsafe preconditions.
        // Since we don't store any `&mut`s, just a raw ptr, we can have a `&Self`.
        let this = unsafe { &*ptr };

        // Assume this so that the compiler knows `ptr` is aligned.
        // Normally we'd store this as a slice so the compiler would know,
        // but since it's a ptr due to `DisjointMut`, we explicitly assume it here.
        // SAFETY: We already checked this in `Self::new`.
        unsafe { assume(this.ptr.is_chunk_aligned()) };

        this.ptr.as_ptr()
    }

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
        // SAFETY: Only creates &Self (SharedReadOnly). Data is behind a raw pointer,
        // not inline, so SharedReadOnly doesn't cover element data.
        let this = unsafe { &*ptr };
        // SAFETY: Alignment guaranteed by PicBuf allocation (via AlignedVec).
        unsafe { assume(this.ptr.is_chunk_aligned()) };
        // SAFETY: Length is always a multiple of RAV1D_PICTURE_GUARANTEED_MULTIPLE,
        // enforced by PicBuf::new padding.
        unsafe { assume(this.len % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0) };
        core::ptr::slice_from_raw_parts_mut(this.ptr.as_ptr(), this.len)
    }

    #[inline] // Inline so callers can see the assume.
    fn len(&self) -> usize {
        // SAFETY: We already checked this in `Self::new`.
        unsafe { assume(self.len % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0) };
        self.len
    }
}

/// A picture data component: a disjoint-tracked buffer with stride.
///
/// For c-ffi: stride is stored inside the inner type.
/// Without c-ffi: stride is stored alongside the [`DisjointMut<PicBuf>`].
#[cfg(feature = "c-ffi")]
pub struct Rav1dPictureDataComponent {
    data: DisjointMut<Rav1dPictureDataComponentInner>,
}

#[cfg(not(feature = "c-ffi"))]
pub struct Rav1dPictureDataComponent {
    data: DisjointMut<Rav1dPictureDataComponentInner>,
    stride: isize,
}

impl Rav1dPictureDataComponent {
    /// Access the inner [`DisjointMut`].
    #[inline(always)]
    pub(crate) fn dm(&self) -> &DisjointMut<Rav1dPictureDataComponentInner> {
        &self.data
    }

    /// Construct from parts. For c-ffi, stride is inside inner.
    /// For non-c-ffi, stride is stored separately.
    #[cfg(feature = "c-ffi")]
    fn from_parts(inner: Rav1dPictureDataComponentInner, _stride: isize) -> Self {
        let mut this = Self {
            data: crate::src::disjoint_mut::dm_new(inner),
        };
        // The tracker's block shift is fixed at construction, so the stride has
        // to reach it here — while `data` is still local and no borrow can
        // exist. A no-op for the shipped `len`-only rule; see
        // `BorrowTracker::set_row_stride`.
        this.data.declare_row_stride(_stride.unsigned_abs());
        this.data.probe_declare_stride(_stride);
        this
    }

    #[cfg(not(feature = "c-ffi"))]
    fn from_parts(inner: Rav1dPictureDataComponentInner, stride: isize) -> Self {
        let mut this = Self {
            data: crate::src::disjoint_mut::dm_new(inner),
            stride,
        };
        // See the c-ffi twin: the block shift is chosen once, at construction.
        this.data.declare_row_stride(stride.unsigned_abs());
        // THROWAWAY (`__probe_bounds`): a no-op without the feature. Lets the
        // report price "widen this guard to the full picture rows it spans".
        this.data.probe_declare_stride(stride);
        this
    }

    /// Extract the owned `Vec<u8>` from this component's inner buffer, if any.
    ///
    /// Returns `Some(vec)` for owned allocations (from `alloc_picture_data`),
    /// `None` for borrowed scratch buffers (from `wrap_buf`).
    /// Used by [`Rav1dPictureData::drop`] to return buffers to the memory pool.
    #[cfg(not(feature = "c-ffi"))]
    fn take_buf(&mut self) -> Option<Vec<u8>> {
        self.data.get_mut().take_buf()
    }

    /// Create from a pixel buffer for use as a scratch source or destination.
    ///
    /// In c-ffi mode: wraps a raw pointer into the caller's buffer (zero-copy).
    /// In safe mode: copies the data into an owned `Vec<u8>` (no raw pointers,
    /// auto `Send + Sync`). For dst-scratch usage, call [`copy_pixels_to`] after
    /// writing to retrieve the results.
    ///
    /// # Panics
    ///
    /// `buf`'s BYTE length must be a multiple of
    /// `RAV1D_PICTURE_GUARANTEED_MULTIPLE` (64) in every configuration, and
    /// under `c-ffi` it must additionally START on a
    /// `RAV1D_PICTURE_ALIGNMENT`-byte (64) boundary — the zero-copy path keeps
    /// the caller's pointer, and that
    /// alignment is a type invariant of `Rav1dPictureDataComponentInner`. See
    /// that type's `wrap_buf` (c-ffi only) for the full rationale.
    ///
    /// The default build copies, so it does NOT check the alignment rule. A
    /// caller that only ever runs without `c-ffi` can therefore violate it
    /// undetected; test harnesses should allocate through
    /// `crate::src::safe_simd::aligned_plane`.
    ///
    /// [`copy_pixels_to`]: Self::copy_pixels_to
    pub fn wrap_buf<BD: BitDepth>(buf: &mut [BD::Pixel], stride: usize) -> Self {
        let stride_bytes = (stride * mem::size_of::<BD::Pixel>()) as isize;
        cfg_if::cfg_if! {
            if #[cfg(feature = "c-ffi")] {
                Self::from_parts(Rav1dPictureDataComponentInner::wrap_buf::<BD>(buf, stride), stride_bytes)
            } else {
                let buf_bytes = IntoBytes::as_bytes(buf);
                assert!(buf_bytes.len() % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0);
                let inner = PicBuf::from_slice_copy(buf_bytes);
                Self::from_parts(inner, stride_bytes)
            }
        }
    }

    /// Copy pixels from this component back into a scratch buffer.
    ///
    /// Used after MC/ipred writes into a copy-backed scratch component
    /// to retrieve the results for subsequent operations (e.g., blend).
    #[cfg(not(feature = "c-ffi"))]
    pub fn copy_pixels_to<BD: BitDepth>(&self, dst: &mut [BD::Pixel]) {
        let n = self.pixel_len::<BD>();
        let guard = self.slice::<BD, _>(..);
        dst[..n].copy_from_slice(&guard[..n]);
    }
}

#[cfg(asm_fn_ptrs)]
impl Pixels for Rav1dPictureDataComponent {
    fn byte_len(&self) -> usize {
        self.dm().len()
    }

    fn as_byte_mut_ptr(&self) -> *mut u8 {
        self.dm().as_mut_ptr()
    }
}

#[cfg(feature = "c-ffi")]
#[allow(unsafe_code)]
impl Strided for Rav1dPictureDataComponent {
    fn stride(&self) -> isize {
        // SAFETY: We're only accessing the `stride` field, not `ptr`.
        unsafe { (*self.dm().inner()).stride }
    }
}

#[cfg(not(feature = "c-ffi"))]
impl Strided for Rav1dPictureDataComponent {
    fn stride(&self) -> isize {
        self.stride
    }
}

impl Rav1dPictureDataComponent {
    /// Length in number of bytes.
    pub fn byte_len(&self) -> usize {
        self.dm().len()
    }

    /// Determine if two components reference the same underlying data.
    pub fn ref_eq(&self, other: &Self) -> bool {
        self.dm().as_mut_ptr() == other.dm().as_mut_ptr()
    }

    /// Length in number of `BitDepth::Pixel`s.
    pub fn pixel_len<BD: BitDepth>(&self) -> usize {
        self.dm().len() / mem::size_of::<BD::Pixel>()
    }

    pub fn pixel_offset<BD: BitDepth>(&self) -> usize {
        let stride = self.stride();
        if stride >= 0 {
            return 0;
        }
        BD::pxstride(self.byte_len() - (-stride) as usize)
    }

    pub fn with_offset<BD: BitDepth>(&self) -> Rav1dPictureDataComponentOffset<'_> {
        Rav1dPictureDataComponentOffset {
            data: self,
            offset: self.pixel_offset::<BD>(),
        }
    }

    /// Strided ptr to [`u8`] bytes.
    ///
    /// Only used by asm (mc emu_edge) and c-ffi (Dav1dPicture conversion).
    #[cfg(any(feature = "asm", feature = "c-ffi"))]
    #[allow(unsafe_code)]
    fn as_strided_byte_mut_ptr(&self) -> *mut u8 {
        let ptr = self.dm().as_mut_ptr();
        let stride = self.stride();
        if stride < 0 {
            // SAFETY: This puts `ptr` one element past the end of the slice of pixels.
            let ptr = unsafe { ptr.add(self.byte_len()) };
            // SAFETY: `stride` is negative and `-stride < len`, so this should stay in bounds.
            let ptr = unsafe { ptr.offset(stride) };
            ptr
        } else {
            ptr
        }
    }

    /// Strided ptr to `BitDepth::Pixel`s.
    #[cfg(any(feature = "asm", feature = "c-ffi"))]
    #[allow(unsafe_code)]
    pub fn as_strided_mut_ptr<BD: BitDepth>(&self) -> *mut BD::Pixel {
        // SAFETY: Transmutation is safe because we verify this with `zerocopy` in `Self::slice`.
        self.as_strided_byte_mut_ptr().cast()
    }

    /// Strided ptr to `BitDepth::Pixel`s.
    #[cfg(feature = "asm")]
    #[allow(unsafe_code)]
    pub fn as_strided_ptr<BD: BitDepth>(&self) -> *const BD::Pixel {
        self.as_strided_mut_ptr::<BD>().cast_const()
    }

    #[cfg(feature = "c-ffi")]
    fn as_dav1d(&self) -> Option<NonNull<c_void>> {
        if self.byte_len() == 0 {
            None
        } else {
            NonNull::new(self.as_strided_byte_mut_ptr().cast())
        }
    }

    pub fn copy_from(&self, src: &Self) {
        let dst = &mut *self.dm().index_mut(..);
        let src = &*src.dm().index(..);
        dst.clone_from_slice(src);
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn index<'a, BD: BitDepth>(
        &'a self,
        index: usize,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.dm().element_as(index)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn index_mut<'a, BD: BitDepth>(
        &'a self,
        index: usize,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.dm().mut_element_as(index)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn slice<'a, BD, I>(
        &'a self,
        index: I,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>
    where
        BD: BitDepth,
        I: SliceBounds,
    {
        #[cfg(any(debug_assertions, feature = "probe-sites"))]
        {
            let total = self.pixel_len::<BD>();
            let r = index.clone().to_range(total);
            note_pic_extent(
                r.len() * mem::size_of::<BD::Pixel>(),
                r.start == 0 && r.end == total,
                {
                    use crate::src::strided::Strided as _;
                    self.pixel_stride::<BD>().unsigned_abs() * mem::size_of::<BD::Pixel>()
                },
            );
        }
        self.dm().slice_as(index)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn slice_mut<'a, BD, I>(
        &'a self,
        index: I,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>
    where
        BD: BitDepth,
        I: SliceBounds,
    {
        #[cfg(any(debug_assertions, feature = "probe-sites"))]
        {
            let total = self.pixel_len::<BD>();
            let r = index.clone().to_range(total);
            note_pic_extent(
                r.len() * mem::size_of::<BD::Pixel>(),
                r.start == 0 && r.end == total,
                {
                    use crate::src::strided::Strided as _;
                    self.pixel_stride::<BD>().unsigned_abs() * mem::size_of::<BD::Pixel>()
                },
            );
        }
        self.dm().mut_slice_as(index)
    }
}

pub type Rav1dPictureDataComponentOffset<'a> = WithOffset<&'a Rav1dPictureDataComponent>;

impl<'a> Rav1dPictureDataComponentOffset<'a> {
    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn index<BD: BitDepth>(
        &self,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.data.index::<BD>(self.offset)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn index_mut<BD: BitDepth>(
        &self,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.data.index_mut::<BD>(self.offset)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn slice<BD: BitDepth>(
        &self,
        len: usize,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]> {
        self.data.slice::<BD, _>((self.offset.., ..len))
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn slice_mut<BD: BitDepth>(
        &self,
        len: usize,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]> {
        self.data.slice_mut::<BD, _>((self.offset.., ..len))
    }

    /// Create a tracked mutable guard covering a strided w×h pixel region.
    ///
    /// Handles both positive and negative strides. The returned guard covers
    /// all pixels that would be accessed by iterating h rows with the given
    /// pixel stride, each row being w pixels wide.
    ///
    /// Returns `(guard, base_offset_within_guard)` where `base_offset_within_guard`
    /// is the index within the guard's slice that corresponds to `self.offset`.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn strided_slice_mut<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
    ) -> (
        DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        self.narrow_guard_mut::<BD>(w, h)
    }

    /// A mutable w×h pixel block that is safe to write while OTHER threads
    /// write other blocks on the same rows.
    ///
    /// # Why this exists
    ///
    /// [`Self::strided_slice_mut`] reserves ONE contiguous range covering the
    /// whole strided span, `(h - 1) * stride + w` pixels — everything from the
    /// block's first pixel to its last, INCLUDING the inter-row gaps, which
    /// belong to other blocks. That is correct single-threaded and wrong under
    /// tile threading: AV1 tiles partition a frame by columns, so two tile
    /// workers routinely write the SAME rows at different columns. The second
    /// one lands in the first one's reserved gap and `DisjointMut` panics —
    /// a false positive, since the two writes are genuinely disjoint.
    ///
    /// Concretely, on a 3840-wide frame a 16×16 block reserves 57,616 bytes in
    /// order to write 256, and a neighbouring 16-byte intra-prediction write
    /// anywhere in those rows trips it. (zenavif#30; the same defect class the
    /// loopfilter and `ipred` compact paths already fixed for themselves.)
    ///
    /// # What it does
    ///
    /// * Tile threading OFF — hands back exactly what `strided_slice_mut`
    ///   does: one contiguous guard, the picture's own stride, no copying.
    ///   Byte-identical behavior and identical cost to before.
    /// * Tile threading ON — copies the block into a compact `w × h` scratch
    ///   buffer through PER-ROW guards (each reserving only the `w` pixels
    ///   actually touched), lets the caller work on that, and writes it back
    ///   per row on drop. No gap is ever reserved, so disjoint neighbours
    ///   cannot collide.
    ///
    /// Callers must take the stride from [`BlockMut::byte_stride`] rather than
    /// from the picture, because the compact buffer has its own stride.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn block_mut<BD: BitDepth>(&self, w: usize, h: usize) -> BlockMut<'a, BD> {
        if tile_threading_active() && !rect_hull_arm() {
            #[cfg(feature = "held-row-guards")]
            if w != 0 && h != 0 && h <= MAX_HELD_ROWS {
                return self.block_mut_held::<BD>(w, h);
            }
            let (buf, byte_stride) = self.compact_read_per_row::<BD>(w, h);
            BlockMut {
                storage: BlockMutStorage::Compact { buf },
                dst: *self,
                w,
                h,
                base: 0,
                byte_stride: byte_stride as isize,
            }
        } else {
            use crate::src::strided::Strided as _;
            let byte_stride = self.stride();
            let (guard, base) = self.narrow_guard_mut::<BD>(w, h);
            BlockMut {
                storage: BlockMutStorage::Direct { guard },
                dst: *self,
                w,
                h,
                base,
                byte_stride,
            }
        }
    }

    /// The tile-threading [`Self::block_mut`] path that HOLDS its row guards.
    ///
    /// # Why it is cheaper
    ///
    /// The plain compact path reserves each row twice: `h` IMMUTABLE per-row
    /// guards to copy the block in, and then, on drop, `h` MUTABLE per-row
    /// guards over the very same bytes to copy it back. That is `2h`
    /// registrations for one block. Taking the MUTABLE guards up front and
    /// keeping them alive across the kernel does the same job with `h` — the
    /// read borrows through a guard that already excludes every other borrow,
    /// so the separate read guard was never adding exclusion.
    ///
    /// Measured on a t=8 4K frame before this existed (macOS `sample`, leaf
    /// samples): the read half cost 14,081 samples and the write-back half
    /// 10,105, together 9.3% of the whole decode.
    ///
    /// # Why it is sound
    ///
    /// * **Extents are unchanged.** Each guard covers exactly
    ///   `[row_off, row_off + w)` — byte-for-byte what the read guard and the
    ///   write-back guard each covered. Nothing widens.
    /// * **Exclusion only tightens.** A mutable registration conflicts with
    ///   every live record; the immutable one it replaces conflicted only with
    ///   mutable records. No overlap that the two-pass shape detected can be
    ///   missed here.
    /// * The bytes held are the block this call is about to overwrite in full,
    ///   so a concurrent borrow of them is a real conflict, not an artefact of
    ///   holding longer: dav1d's task schedule never lets another worker read a
    ///   block whose inverse transform has not finished.
    ///
    /// # Why the row cap
    ///
    /// The guards live in an inline array, so `h` is bounded at compile time,
    /// and peak live registrations per thread rise from 1 to `h`. A shard holds
    /// 7 records before it promotes a borrow to the wide list — which locks
    /// every shard of the instance — so the cap is also a guard against turning
    /// a tall block into a wide-path storm. `MAX_HELD_ROWS` covers every
    /// transform height AV1 has; anything taller falls back to the two-pass
    /// path, which is always correct.
    #[cfg(feature = "held-row-guards")]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn block_mut_held<BD: BitDepth>(&self, w: usize, h: usize) -> BlockMut<'a, BD> {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let byte_stride = w * pixel_size;
        let needed = h * byte_stride;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        let mut buf = take_compact_scratch();
        buf.resize(needed, 0);
        let mut rows: RowGuards<'a, BD> = [const { None }; MAX_HELD_ROWS];
        for row in 0..h {
            let row_off = if pxstride >= 0 {
                self.offset + row * abs_stride
            } else {
                self.offset - row * abs_stride
            };
            let guard = self.data.slice_mut::<BD, _>((row_off.., ..w));
            buf[row * byte_stride..][..byte_stride]
                .copy_from_slice(&guard.as_bytes()[..byte_stride]);
            rows[row] = Some(guard);
        }
        BlockMut {
            storage: BlockMutStorage::Held { buf, rows },
            dst: *self,
            w,
            h,
            base: 0,
            byte_stride: byte_stride as isize,
        }
    }

    /// Create a tracked immutable guard covering a strided w×h pixel region.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn strided_slice<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
    ) -> (
        DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        self.narrow_guard::<BD>(w, h)
    }

    /// The contiguous hull of a `w x h` pixel block whose row 0 starts at
    /// `self.offset`, as `(start, total, base, pxstride)`: the block lies
    /// within `[start, start + total)` of the component, and row `r`, column
    /// `x` is at `base + x + r * pxstride` inside that hull.
    ///
    /// Rows run *forward* `w` pixels from their own start whatever the sign of
    /// the stride (see `for_rows`'s per-row branch and `with_pixel_guard_mut`).
    /// On a negative stride the hull therefore starts `(h-1)*|stride|` below
    /// `self.offset` — at the LAST row — and ends at `self.offset + w`, the end
    /// of row 0, with `base = (h-1)*|stride|`. This is the one place that
    /// geometry lives: #520 was four copies of it that started the hull `w-1`
    /// pixels lower (`offset + 1 - total`, base `total - 1`), so every row-0
    /// pixel but the first fell outside the guard while `w-1` pixels below the
    /// block were reserved for nothing.
    #[inline(always)]
    fn block_hull<BD: BitDepth>(&self, w: usize, h: usize) -> (usize, usize, usize, isize) {
        use crate::src::strided::Strided as _;
        let pxstride = self.data.pixel_stride::<BD>();
        if w == 0 || h == 0 {
            return (self.offset, 0, 0, pxstride);
        }
        let span = (h - 1) * pxstride.unsigned_abs();
        let total = span + w;
        if pxstride >= 0 {
            (self.offset, total, 0, pxstride)
        } else {
            let start = self
                .offset
                .checked_sub(span)
                .expect("block extends below the start of a negative-stride picture");
            (start, total, span, pxstride)
        }
    }

    /// Create a tracked immutable guard covering exactly a w×h pixel block.
    ///
    /// Returns `(guard, base)`: `base` is 0 for a positive stride and
    /// `(h-1)*|stride|` for a negative one — see [`Self::block_hull`].
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn narrow_guard<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
    ) -> (
        DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        let (start, total, base, pxstride) = self.block_hull::<BD>(w, h);
        let ps = mem::size_of::<BD::Pixel>();
        let guard = self.data.slice::<BD, _>((start.., ..total));
        guard.probe_declare_rows(start * ps, w * ps, h, pxstride * ps as isize);
        (guard, base)
    }

    /// Visit `h` consecutive picture rows of `w` pixels each, IMMUTABLY, taking
    /// ONE borrow registration instead of `h` when no tile worker can be alive.
    ///
    /// # The policy, and why it is the existing one
    ///
    /// A `w×h` strided block is either `h` registrations of exactly `w` pixels,
    /// or one registration of `(h-1)*stride + w` — the hull, which additionally
    /// reserves the inter-row gaps belonging to other columns of the same rows.
    /// [`Rav1dPictureDataComponentOffset::block_mut`] already documents the
    /// trade in full: the hull is "correct single-threaded and wrong under tile
    /// threading", because AV1 tiles partition a frame by COLUMNS, so two tile
    /// workers routinely write the same rows at different columns and the gap
    /// reservation turns a genuinely disjoint pair into a false positive.
    ///
    /// So the choice is made by [`tile_threading_active`] — process-global,
    /// monotone, and never storing `false` — exactly as it is in `block_mut`,
    /// [`with_pixel_guard_immut`] and [`Self::compact_read`]. This helper adds
    /// callers to that policy; it does not introduce one.
    ///
    /// Neither branch can MISS an overlap: the hull is a superset of the `h`
    /// row ranges, and a superset registration conflicts with strictly more.
    /// The only thing at stake is false positives, and the latch is what rules
    /// those out.
    ///
    /// # Why it exists
    ///
    /// Per-row guards over small blocks ARE the decoder's borrow-count
    /// distribution. Measured with `--features probe-sites` on `v4k_8tile` 8bpc
    /// at t=1: collapsing the per-row loops in the loopfilter, `ipred` and
    /// `cdef` took registrations per frame from 15,646,727 to 7,924,706
    /// (-49.4%), at a mean per-row extent of ~10 bytes.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn for_rows<BD: BitDepth, F: FnMut(usize, &[BD::Pixel])>(
        &self,
        w: usize,
        h: usize,
        mut f: F,
    ) {
        use crate::src::strided::Strided as _;
        if w == 0 || h == 0 {
            return;
        }
        let pxstride = self.data.pixel_stride::<BD>();
        if tile_threading_active() {
            let ps = mem::size_of::<BD::Pixel>();
            self.data.dm().probe_eval_rect(
                core::panic::Location::caller(),
                false,
                self.offset * ps,
                w * ps,
                h,
                pxstride * ps as isize,
            );
            // ONE exact strided-rectangle
            // record instead of `h` per-row ones, the `LfBlock::fill_rect`
            // mechanism applied at this seam. `None` is a REFUSAL — no declared
            // stride, a stride mismatch, `w > stride`, `h > MAX_RECT_ROWS`, a
            // hull spanning more than `MAX_SHARDS_PER_BORROW` blocks, a full
            // shard, or a live wide record — and then the per-row loop below
            // runs exactly as it did before rectangles existed. Nothing is ever
            // rounded up to make a rectangle fit.
            //
            // Sound here for the reason it is sound in `LfBlock::fill_threaded`
            // and the hull is not: the record covers only the `h` row segments,
            // so a concurrent writer in an inter-row GAP (another tile column of
            // the same picture rows, the routine case) is neither reserved
            // against nor reported, and `DisjointImmutRectGuard` never
            // materialises a reference wider than one row.
            if let Some(rect) =
                self.data
                    .dm()
                    .index_rect_as::<BD::Pixel>(self.offset, w, h, pxstride)
            {
                for row in 0..h {
                    f(row, rect.row(row));
                }
                return;
            }
            for row in 0..h {
                let off = self.offset.wrapping_add_signed(row as isize * pxstride);
                let guard = self.data.slice::<BD, _>((off.., ..w));
                f(row, &guard);
            }
            return;
        }
        let (lo, total, base, _) = self.block_hull::<BD>(w, h);
        let guard = self.data.slice::<BD, _>((lo.., ..total));
        {
            let ps = mem::size_of::<BD::Pixel>();
            guard.probe_declare_rows(lo * ps, w * ps, h, pxstride * ps as isize);
        }
        for row in 0..h {
            let idx = base.wrapping_add_signed(row as isize * pxstride);
            f(row, &guard[idx..][..w]);
        }
    }

    /// Take `h` EXTRA per-row IMMUTABLE registrations over exactly the bytes a
    /// following [`Self::for_rows`] will cover, then drop them.
    ///
    /// Inert unless `__probe_cdef_double` is compiled in AND
    /// `RAV1D_CDEF_DOUBLE=1`; see [`cdef_double_reads`] for what it prices and
    /// why doubling is the sound direction. Only the tile-threading branch is
    /// doubled, because that is the branch whose per-registration cost is in
    /// question — without tile threading `for_rows` takes ONE hull guard.
    #[inline(always)]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn dup_rows<BD: BitDepth>(&self, w: usize, h: usize) {
        if !cdef_double_reads() || w == 0 || h == 0 || !tile_threading_active() {
            return;
        }
        use crate::src::strided::Strided as _;
        let pxstride = self.data.pixel_stride::<BD>();
        for row in 0..h {
            let off = self.offset.wrapping_add_signed(row as isize * pxstride);
            let guard = self.data.slice::<BD, _>((off.., ..w));
            core::hint::black_box(&guard[0]);
        }
    }

    /// [`Self::dup_rows`] for a following [`Self::for_rows_mut`]: the extra
    /// reservations are MUTABLE, because that is the record shape the real site
    /// files and `find::<true>` is a different scan from `find::<false>`.
    #[inline(always)]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn dup_rows_mut<BD: BitDepth>(&self, w: usize, h: usize) {
        if !cdef_double_reads() || w == 0 || h == 0 || !tile_threading_active() {
            return;
        }
        use crate::src::strided::Strided as _;
        let pxstride = self.data.pixel_stride::<BD>();
        for row in 0..h {
            let off = self.offset.wrapping_add_signed(row as isize * pxstride);
            let mut guard = self.data.slice_mut::<BD, _>((off.., ..w));
            core::hint::black_box(&mut guard[0]);
        }
    }

    /// [`Self::for_rows`], mutably. Same policy, same soundness argument.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn for_rows_mut<BD: BitDepth, F: FnMut(usize, &mut [BD::Pixel])>(
        &self,
        w: usize,
        h: usize,
        mut f: F,
    ) {
        use crate::src::strided::Strided as _;
        if w == 0 || h == 0 {
            return;
        }
        let pxstride = self.data.pixel_stride::<BD>();
        if tile_threading_active() {
            let ps = mem::size_of::<BD::Pixel>();
            self.data.dm().probe_eval_rect(
                core::panic::Location::caller(),
                true,
                self.offset * ps,
                w * ps,
                h,
                pxstride * ps as isize,
            );
            // The write side: ONE exact
            // MUTABLE rectangle record instead of `h` per-row ones. Same
            // refusal list, same soundness argument as the read side above,
            // plus: `DisjointMutRectGuard::row_mut` takes `&mut self`, so at
            // most one row reference is live at a time and no `&mut [_]` wider
            // than one row is ever created.
            if let Some(mut rect) =
                self.data
                    .dm()
                    .index_rect_mut_as::<BD::Pixel>(self.offset, w, h, pxstride)
            {
                for row in 0..h {
                    f(row, rect.row_mut(row));
                }
                return;
            }
            for row in 0..h {
                let off = self.offset.wrapping_add_signed(row as isize * pxstride);
                let mut guard = self.data.slice_mut::<BD, _>((off.., ..w));
                f(row, &mut guard);
            }
            return;
        }
        let (lo, total, base, _) = self.block_hull::<BD>(w, h);
        let mut guard = self.data.slice_mut::<BD, _>((lo.., ..total));
        {
            let ps = mem::size_of::<BD::Pixel>();
            guard.probe_declare_rows(lo * ps, w * ps, h, pxstride * ps as isize);
        }
        for row in 0..h {
            let idx = base.wrapping_add_signed(row as isize * pxstride);
            f(row, &mut guard[idx..][..w]);
        }
    }

    /// Read a w×h pixel block into a compact Vec using per-row DisjointMut guards.
    ///
    /// When tile threading is active ([`set_tile_threading`]), each row guard covers
    /// exactly `w` pixels, avoiding stride-padding overlap between concurrent tiles.
    /// When single-threaded, uses one guard for the whole block (fast path).
    ///
    /// Returns `(buffer, byte_stride)` where `byte_stride` is `w * pixel_size` when
    /// threading (compact layout) or the original stride when single-threaded.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_read<BD: BitDepth>(&self, w: usize, h: usize) -> (Vec<u8>, usize) {
        use crate::src::strided::Strided as _;
        // The fast path hands back the hull in MEMORY order with an unsigned
        // stride, which a caller can only address when row 0 comes first — a
        // positive stride. A negative stride takes the compact, row-0-first
        // path whatever the threading mode (#520).
        if tile_threading_active() || self.data.pixel_stride::<BD>() < 0 {
            self.compact_read_per_row::<BD>(w, h)
        } else {
            self.compact_read_fast::<BD>(w, h)
        }
    }

    /// Fast path: single guard for the whole block, returns the hull in memory
    /// order with the original (unsigned) stride — row 0 first only when the
    /// stride is positive, which is the only case `compact_read` sends here.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn compact_read_fast<BD: BitDepth>(&self, w: usize, h: usize) -> (Vec<u8>, usize) {
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let (start, total, _base, pxstride) = self.block_hull::<BD>(w, h);
        let abs_stride = pxstride.unsigned_abs();
        let guard = self.data.slice::<BD, _>((start.., ..total));
        guard.probe_declare_rows(
            start * pixel_size,
            w * pixel_size,
            h,
            pxstride * pixel_size as isize,
        );
        let byte_stride = abs_stride * pixel_size;
        (guard.as_bytes().to_vec(), byte_stride)
    }

    /// Per-row path: each row guard covers exactly `w` pixels.
    /// Always returns compact stride = w * pixel_size.
    /// Used by the loopfilter (needs compact layout for 2D decomposition)
    /// and by tile threading (needs per-row guards to avoid stride overlap).
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_read_per_row<BD: BitDepth>(&self, w: usize, h: usize) -> (Vec<u8>, usize) {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let byte_stride = w * pixel_size;
        let needed = h * byte_stride;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        // Reuse a thread-local scratch buffer instead of allocating per call
        // (issue #17). `resize` keeps the existing capacity across reuse and
        // only zero-fills when growing past the previous high-water mark; the
        // per-row copy below fully overwrites `buf[..needed]` regardless, so the
        // returned `Vec` is byte-identical to the former `vec![0u8; needed]`.
        let mut buf = take_compact_scratch();
        buf.resize(needed, 0);
        self.data.dm().probe_eval_rect(
            core::panic::Location::caller(),
            false,
            self.offset * pixel_size,
            w * pixel_size,
            h,
            pxstride * pixel_size as isize,
        );
        for row in 0..h {
            let row_off = if pxstride >= 0 {
                self.offset + row * abs_stride
            } else {
                self.offset - row * abs_stride
            };
            let guard = self.data.slice::<BD, _>((row_off.., ..w));
            buf[row * byte_stride..][..byte_stride]
                .copy_from_slice(&guard.as_bytes()[..byte_stride]);
        }
        (buf, byte_stride)
    }

    /// Write a compact buffer back to a w×h pixel block.
    ///
    /// Matches the layout produced by [`compact_read`].
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_write_back<BD: BitDepth>(&self, w: usize, h: usize, buf: &[u8]) {
        use crate::src::strided::Strided as _;
        // Mirrors `compact_read`: a negative stride's buffer is compact and
        // row-0-first, so it must be written back per row (#520).
        if tile_threading_active() || self.data.pixel_stride::<BD>() < 0 {
            self.compact_write_back_per_row::<BD>(w, h, buf);
        } else {
            self.compact_write_back_fast::<BD>(w, h, buf);
        }
    }

    /// Fast path write-back: single guard, original stride layout.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn compact_write_back_fast<BD: BitDepth>(&self, w: usize, h: usize, buf: &[u8]) {
        use zerocopy::IntoBytes;
        let (start, total, _base, pxstride) = self.block_hull::<BD>(w, h);
        let mut guard = self.data.slice_mut::<BD, _>((start.., ..total));
        {
            let ps = mem::size_of::<BD::Pixel>();
            guard.probe_declare_rows(start * ps, w * ps, h, pxstride * ps as isize);
        }
        let dst = guard.as_mut_bytes();
        let len = buf.len().min(dst.len());
        dst[..len].copy_from_slice(&buf[..len]);
    }

    /// Per-row write-back: compact stride = w * pixel_size.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_write_back_per_row<BD: BitDepth>(&self, w: usize, h: usize, buf: &[u8]) {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let byte_stride = w * pixel_size;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        self.data.dm().probe_eval_rect(
            core::panic::Location::caller(),
            true,
            self.offset * pixel_size,
            w * pixel_size,
            h,
            pxstride * pixel_size as isize,
        );
        for row in 0..h {
            let row_off = if pxstride >= 0 {
                self.offset + row * abs_stride
            } else {
                self.offset - row * abs_stride
            };
            let mut guard = self.data.slice_mut::<BD, _>((row_off.., ..w));
            guard.as_mut_bytes()[..byte_stride]
                .copy_from_slice(&buf[row * byte_stride..][..byte_stride]);
        }
    }

    /// Per-row write-back that stores ONLY the pixels the caller actually
    /// modified, by diffing `work` against the `pristine` copy taken before
    /// filtering. Rows (and row spans) that are byte-identical are neither
    /// written nor mutably guarded.
    ///
    /// This is the tile-threading-correct write-back for filters whose
    /// *read* region is wider than their *write* region (the loop filter
    /// reads 7 tap rows/cols beyond the ≤6 it can modify): a plain
    /// [`Self::compact_write_back_per_row`] would rewrite — and take `&mut`
    /// guards on — tap-input pixels it never changed, which (a) collides
    /// with concurrent readers that dav1d's task schedule legitimately
    /// allows (CDEF bottom-edge padding reads the 2 rows the deblock task
    /// of the next sbrow only ever *reads*; dav1d's 8-row CDEF lag is
    /// exactly 2 pad rows + 6 modified rows), and (b) risks clobbering a
    /// concurrent writer's output with stale copied bytes. Diffing makes the
    /// write-set equal dav1d's by construction. Found via zenavif#30: the
    /// `overlapping DisjointMut` worker panic (`cdef.rs` padding read vs
    /// this write-back's 7th tap row) wedged the decode wait forever.
    ///
    /// `work` and `pristine` must both use the compact layout produced by
    /// [`Self::compact_read_per_row`] (stride = `w * pixel_size`).
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_write_back_per_row_diff<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
        work: &[u8],
        pristine: &[u8],
    ) {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let byte_stride = w * pixel_size;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        for row in 0..h {
            let work_row = &work[row * byte_stride..][..byte_stride];
            let pristine_row = &pristine[row * byte_stride..][..byte_stride];
            // Find the modified byte span of this row (usually empty or
            // small — the loop filter touches ≤6 pixels around each edge).
            let Some(first) = iter::zip(work_row, pristine_row).position(|(a, b)| a != b) else {
                continue; // row untouched: no write, no mutable guard
            };
            let last = iter::zip(work_row, pristine_row)
                .rposition(|(a, b)| a != b)
                .expect("a differing byte exists, so rposition finds one");
            // Widen the differing byte span [first..=last] to whole pixels.
            let first_px = first / pixel_size;
            let end_px = last / pixel_size + 1;
            let row_off = if pxstride >= 0 {
                self.offset + row * abs_stride
            } else {
                self.offset - row * abs_stride
            };
            let mut guard = self
                .data
                .slice_mut::<BD, _>((row_off + first_px.., ..end_px - first_px));
            guard.as_mut_bytes()[..(end_px - first_px) * pixel_size]
                .copy_from_slice(&work_row[first_px * pixel_size..end_px * pixel_size]);
        }
    }

    /// Create a tracked mutable guard covering the entire picture component.
    ///
    /// Returns `(guard, offset_within_guard)` where the offset corresponds to
    /// this PicOffset's logical position within the full guard.
    /// Use this when the access pattern is complex (e.g., loopfilter accessing
    /// negative offsets from the base pointer).
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn full_guard_mut<BD: BitDepth>(
        &self,
    ) -> (
        DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        let total_pixels = self.data.pixel_len::<BD>();
        let guard = self.data.slice_mut::<BD, _>((0.., ..total_pixels));
        (guard, self.offset)
    }

    /// Create a tracked mutable guard covering exactly a w×h pixel block
    /// whose row 0 starts at this offset.
    ///
    /// Returns `(guard, base)`: the guard is the block's hull,
    /// `(h-1)*|stride| + w` pixels, and `base` is row 0's index inside it — 0
    /// for a positive stride, `(h-1)*|stride|` for a negative one, where the
    /// hull starts at the last row. See [`Self::block_hull`].
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn narrow_guard_mut<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
    ) -> (
        DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        let (start, total, base, pxstride) = self.block_hull::<BD>(w, h);
        let ps = mem::size_of::<BD::Pixel>();
        let guard = self.data.slice_mut::<BD, _>((start.., ..total));
        guard.probe_declare_rows(start * ps, w * ps, h, pxstride * ps as isize);
        (guard, base)
    }

    /// Create a tracked immutable guard covering the entire picture component.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn full_guard<BD: BitDepth>(
        &self,
    ) -> (
        DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        let total_pixels = self.data.pixel_len::<BD>();
        let guard = self.data.slice::<BD, _>((0.., ..total_pixels));
        (guard, self.offset)
    }
}

#[cfg(feature = "c-ffi")]
pub struct Rav1dPictureData {
    pub data: [Rav1dPictureDataComponent; 3],
    pub(crate) allocator_data: Option<SendSyncNonNull<c_void>>,
    pub(crate) allocator: Rav1dPicAllocator,
}

#[cfg(not(feature = "c-ffi"))]
pub struct Rav1dPictureData {
    pub data: [Rav1dPictureDataComponent; 3],
    pub(crate) allocator: Rav1dPicAllocator,
}

#[cfg(feature = "c-ffi")]
impl Drop for Rav1dPictureData {
    fn drop(&mut self) {
        let Self {
            data,
            allocator_data,
            allocator,
        } = self;
        allocator.dealloc_picture_data(data, *allocator_data);
    }
}

#[cfg(not(feature = "c-ffi"))]
impl Drop for Rav1dPictureData {
    fn drop(&mut self) {
        for component in &mut self.data {
            if let Some(buf) = component.take_buf() {
                if !buf.is_empty() {
                    self.allocator.pool.push(buf);
                }
            }
        }
    }
}

// TODO(kkysen) Eventually the [`impl Default`] might not be needed.
// It's needed currently for a [`mem::take`] that simulates a move,
// but once everything is Rusty, we may not need to clear the `dst` anymore.
// This also applies to the `#[derive(Default)]`
// on [`Rav1dPictureParameters`] and [`Rav1dPixelLayout`].
#[derive(Clone, Default)]
#[repr(C)]
pub(crate) struct Rav1dPicture {
    pub seq_hdr: Option<Arc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>>,
    pub frame_hdr: Option<Arc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>,
    pub data: Option<Arc<Rav1dPictureData>>,
    pub stride: [ptrdiff_t; 2],
    pub p: Rav1dPictureParameters,
    pub m: Rav1dDataProps,
    pub content_light: Option<Arc<Rav1dContentLightLevel>>,
    pub mastering_display: Option<Arc<Rav1dMasteringDisplay>>,
    pub itut_t35: Arc<DRav1d<Box<[Rav1dITUTT35]>, Box<[Dav1dITUTT35]>>>,
}

#[cfg(feature = "c-ffi")]
impl From<Dav1dPicture> for Rav1dPicture {
    fn from(value: Dav1dPicture) -> Self {
        let Dav1dPicture {
            seq_hdr: _,
            frame_hdr: _,
            data: _,
            stride,
            p,
            m,
            content_light: _,
            mastering_display: _,
            itut_t35: _,
            n_itut_t35: _,
            reserved: _,
            frame_hdr_ref,
            seq_hdr_ref,
            content_light_ref,
            mastering_display_ref,
            itut_t35_ref,
            reserved_ref: _,
            r#ref: data_ref,
            allocator_data: _,
        } = value;
        Self {
            // We don't `.update_rav1d()` [`Rav1dSequenceHeader`] because it's meant to be read-only.
            seq_hdr: seq_hdr_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            // We don't `.update_rav1d()` [`Rav1dFrameHeader`] because it's meant to be read-only.
            frame_hdr: frame_hdr_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            data: data_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            stride,
            p: p.into(),
            m: m.into(),
            content_light: content_light_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            mastering_display: mastering_display_ref.map(|raw| {
                // Safety: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            // We don't `.update_rav1d` [`Rav1dITUTT35`] because never read it.
            itut_t35: itut_t35_ref
                .map(|raw| {
                    // SAFETY: `raw` came from [`RawArc::from_arc`].
                    unsafe { raw.into_arc() }
                })
                .unwrap_or_default(),
        }
    }
}

#[cfg(feature = "c-ffi")]
impl From<Rav1dPicture> for Dav1dPicture {
    fn from(value: Rav1dPicture) -> Self {
        let Rav1dPicture {
            seq_hdr,
            frame_hdr,
            data,
            stride,
            p,
            m,
            content_light,
            mastering_display,
            itut_t35,
        } = value;
        Self {
            // [`DRav1d::from_rav1d`] is called right after [`parse_seq_hdr`].
            seq_hdr: seq_hdr.as_ref().map(|arc| (&arc.as_ref().dav1d).into()),
            // [`DRav1d::from_rav1d`] is called in [`parse_frame_hdr`].
            frame_hdr: frame_hdr.as_ref().map(|arc| (&arc.as_ref().dav1d).into()),
            data: data
                .as_ref()
                .map(|arc| arc.data.each_ref().map(|data| data.as_dav1d()))
                .unwrap_or_default(),
            stride,
            p: p.into(),
            m: m.into(),
            content_light: content_light.as_ref().map(|arc| arc.as_ref().into()),
            mastering_display: mastering_display.as_ref().map(|arc| arc.as_ref().into()),
            // [`DRav1d::from_rav1d`] is called in [`rav1d_parse_obus`].
            itut_t35: Some(NonNull::new(itut_t35.dav1d.as_ptr().cast_mut()).unwrap()),
            n_itut_t35: itut_t35.len(),
            reserved: Default::default(),
            frame_hdr_ref: frame_hdr.map(RawArc::from_arc),
            seq_hdr_ref: seq_hdr.map(RawArc::from_arc),
            content_light_ref: content_light.map(RawArc::from_arc),
            mastering_display_ref: mastering_display.map(RawArc::from_arc),
            itut_t35_ref: Some(itut_t35).map(RawArc::from_arc),
            reserved_ref: Default::default(),
            // Order flipped so that the borrow comes before the move.
            allocator_data: data.as_ref().and_then(|arc| arc.allocator_data),
            r#ref: data.map(RawArc::from_arc),
        }
    }
}

impl Rav1dPicture {
    pub fn lf_offsets<BD: BitDepth>(&self, y: c_int) -> [Rav1dPictureDataComponentOffset<'_>; 3] {
        // Init loopfilter offsets. Point the chroma offsets in 4:0:0 to the luma
        // plane here to avoid having additional in-loop branches in various places.
        // We never use those values, so it doesn't really matter what they point
        // at, as long as the offsets are valid.
        let layout = self.p.layout;
        let has_chroma = layout != Rav1dPixelLayout::I400;
        let data = &self.data.as_ref().unwrap().data;
        array::from_fn(|i| {
            let data = &data[has_chroma as usize * i];
            let ss_ver = layout == Rav1dPixelLayout::I420 && i != 0;
            data.with_offset::<BD>() + (y as isize * data.pixel_stride::<BD>() >> ss_ver as u8)
        })
    }
}

#[cfg(feature = "c-ffi")]
#[derive(Clone)]
#[repr(C)]
pub struct Dav1dPicAllocator {
    /// Custom data to pass to the allocator callbacks.
    ///
    /// # Safety
    ///
    /// All accesses to [`Self::cookie`] must be thread-safe
    /// (i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`]).
    ///
    /// If used from Rust, [`Self::cookie`] is a [`SendSyncNonNull`],
    /// whose constructors ensure this [`Send`]` + `[`Sync`] safety.
    pub cookie: Option<SendSyncNonNull<c_void>>,

    /// Allocate the picture buffer based on the [`Dav1dPictureParameters`].
    ///
    /// [`data`]`[0]`, [`data`]`[1]` and [`data`]`[2]`
    /// must be [`DAV1D_PICTURE_ALIGNMENT`]-byte aligned
    /// and with a pixel width/height multiple of 128 pixels.
    /// Any allocated memory area should also be padded by [`DAV1D_PICTURE_ALIGNMENT`] bytes.
    /// [`data`]`[1]` and [`data`]`[2]` must share the same [`stride`]`[1]`.
    ///
    /// # Safety
    ///
    /// See [`Self::cookie`]'s safety requirements.
    ///
    /// ### Additional `rav1d` requirement:
    ///
    /// The allocated data must be initialized.
    /// If newly (e.x. not reused) allocated data is zero initialized using OS APIs,
    /// it is possible for this to not be slower than an uninitialized allocation.
    /// For example, see `dav1d_default_picture_alloc` and `MemPool::pop_init`.
    ///
    /// If the allocated data is not initialized,
    /// it is possible there will be reads of uninitialized data.
    /// `rav1d` should not read this data before writing to it first,
    /// but it does not guarantee that it does so.
    /// Instead, initializing the allocated data guarantees all uses of it will be sound.
    ///
    /// # Args
    ///
    /// * `pic`: The picture to allocate the buffer for.
    ///     The callback needs to fill the picture
    ///     [`data`]`[0]`, [`data`]`[1]`, [`data`]`[2]`,
    ///     [`stride`]`[0]`, and [`stride`]`[1]`.
    ///     The allocator can fill the pic [`allocator_data`] pointer
    ///     with a custom pointer that will be passed to
    ///     [`release_picture_callback`].
    ///
    ///     The only fields of `pic` that will be already set are:
    ///     * [`Dav1dPicture::p`]
    ///     * [`Dav1dPicture::seq_hdr`]
    ///     * [`Dav1dPicture::frame_hdr`]
    ///     
    ///     This is not a change from the original `DAV1D_API`,
    ///     just a clarification of it.
    ///
    /// * `cookie`: Custom pointer passed to all calls.
    ///
    /// *Note*: No fields other than [`data`], [`stride`] and [`allocator_data`]
    /// must be filled by this callback.
    ///
    /// # Return
    ///
    /// 0 on success. A negative `DAV1D_ERR` value on error.
    /// <!--- TODO(kkysen) Translate `DAV1D_ERR` -->
    ///
    /// [`data`]: Dav1dPicture::data
    /// [`stride`]: Dav1dPicture::data
    /// [`allocator_data`]: Dav1dPicture::allocator_data
    /// [`release_picture_callback`]: Self::release_picture_callback
    pub alloc_picture_callback: Option<
        unsafe extern "C" fn(
            pic: *mut Dav1dPicture,
            cookie: Option<SendSyncNonNull<c_void>>,
        ) -> Dav1dResult,
    >,

    /// Release the picture buffer.
    ///
    /// # Safety
    ///
    /// If frame threading is used, accesses to `cookie` must be thread-safe.
    ///
    /// If frame threading is used, this function may be called by the main thread
    /// (the thread which calls [`dav1d_get_picture`]),
    /// or any of the frame threads and thus must be thread-safe.
    /// If frame threading is not used, this function will only be called on the main thread.
    ///
    /// # Args
    ///
    /// * `pic`: The picture that was filled by [`alloc_picture_callback`].
    ///     
    ///     The only fields of `pic` that will be set are
    ///     the ones allocated by [`Self::alloc_picture_callback`]:
    ///     * [`Dav1dPicture::data`]
    ///     * [`Dav1dPicture::allocator_data`]
    ///     
    ///     NOTE: This is a slight change from the original `DAV1D_API`, which was underspecified.
    ///     However, all known uses of this API follow this already:
    ///     * `libdav1d`: [`dav1d_default_picture_release`](https://code.videolan.org/videolan/dav1d/-/blob/16ed8e8b99f2fcfffe016e929d3626e15267ad3e/src/picture.c#L85-87)
    ///     * `dav1d`: [`picture_release`](https://code.videolan.org/videolan/dav1d/-/blob/16ed8e8b99f2fcfffe016e929d3626e15267ad3e/tools/dav1d.c#L180-182)
    ///     * `dav1dplay`: [`placebo_release_pic`](https://code.videolan.org/videolan/dav1d/-/blob/16ed8e8b99f2fcfffe016e929d3626e15267ad3e/examples/dp_renderer_placebo.c#L375-383)
    ///     * `libplacebo`: [`pl_release_dav1dpicture`](https://github.com/haasn/libplacebo/blob/34e019bfedaa5a64f268d8f9263db352c0a8f67f/src/include/libplacebo/utils/dav1d_internal.h#L594-L607)
    ///     * `ffmpeg`: [`libdav1d_picture_release`](https://github.com/FFmpeg/FFmpeg/blob/00b288da73f45acb78b74bcc40f73c7ba1fff7cb/libavcodec/libdav1d.c#L124-L129)
    ///
    ///     Making this API safe without this slight tightening of the API
    ///     [is very difficult](https://github.com/memorysafety/rav1d/pull/685#discussion_r1458171639).
    ///
    /// * `cookie`: Custom pointer passed to all calls.
    ///
    /// [`dav1d_get_picture`]: crate::src::lib::dav1d_get_picture
    /// [`alloc_picture_callback`]: Self::alloc_picture_callback
    pub release_picture_callback: Option<
        unsafe extern "C" fn(pic: *mut Dav1dPicture, cookie: Option<SendSyncNonNull<c_void>>) -> (),
    >,
}

#[cfg(feature = "c-ffi")]
#[derive(Clone)]
#[repr(C)]
pub(crate) struct Rav1dPicAllocator {
    /// See [`Dav1dPicAllocator::cookie`].
    ///
    /// # Safety
    ///
    /// If [`Self::is_default`]`()`, then this cookie is a reference to
    /// [`Rav1dContext::picture_pool`], a `&Arc<MemPool<u8>`.
    /// Thus, its lifetime is that of `&c.picture_pool`,
    /// so the lifetime of the `&`[`Rav1dContext`].
    /// This is used from `dav1d_default_picture_alloc`
    /// ([`Self::default`]`().alloc_picture_callback`),
    /// which is called from [`Self::alloc_picture_data`],
    /// which is called further up on the call stack with a `&`[`Rav1dContext`].
    /// Thus, the lifetime will always be valid where used.
    ///
    /// Note that this is an `&Arc<MemPool<u8>` turned into a raw pointer,
    /// not an [`Arc::into_raw`] of that [`Arc`].
    /// This is because storing the [`Arc`] would require C to
    /// free data owned by a [`Dav1dPicAllocator`] potentially,
    /// which it may not do, as there are no current APIs for doing so.
    ///
    /// [`Rav1dContext::picture_pool`]: crate::src::internal::Rav1dContext::picture_pool
    /// [`Rav1dContext`]: crate::src::internal::Rav1dContext
    pub cookie: Option<SendSyncNonNull<c_void>>,

    /// See [`Dav1dPicAllocator::alloc_picture_callback`].
    ///
    /// # Safety
    ///
    /// `pic` is passed as a `&mut`.
    ///
    /// If frame threading is used, accesses to [`Self::cookie`] must be thread-safe,
    /// i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`].
    pub alloc_picture_callback: unsafe extern "C" fn(
        pic: *mut Dav1dPicture,
        cookie: Option<SendSyncNonNull<c_void>>,
    ) -> Dav1dResult,

    /// See [`Dav1dPicAllocator::release_picture_callback`].
    ///
    /// # Safety
    ///
    /// `pic` is passed as a `&mut`.
    ///
    /// If frame threading is used, accesses to [`Self::cookie`] must be thread-safe,
    /// i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`].
    pub release_picture_callback:
        unsafe extern "C" fn(pic: *mut Dav1dPicture, cookie: Option<SendSyncNonNull<c_void>>) -> (),
}

/// Safe picture allocator using per-plane `Vec<u8>` buffers from a shared pool.
#[cfg(not(feature = "c-ffi"))]
#[derive(Clone, Default)]
pub(crate) struct Rav1dPicAllocator {
    pub(crate) pool: Arc<MemPool<u8>>,
}

#[cfg(feature = "c-ffi")]
impl TryFrom<Dav1dPicAllocator> for Rav1dPicAllocator {
    type Error = Rav1dError;

    fn try_from(value: Dav1dPicAllocator) -> Result<Self, Self::Error> {
        let Dav1dPicAllocator {
            cookie,
            alloc_picture_callback,
            release_picture_callback,
        } = value;
        Ok(Self {
            cookie,
            alloc_picture_callback: validate_input!(alloc_picture_callback.ok_or(EINVAL))?,
            release_picture_callback: validate_input!(release_picture_callback.ok_or(EINVAL))?,
        })
    }
}

#[cfg(feature = "c-ffi")]
impl From<Rav1dPicAllocator> for Dav1dPicAllocator {
    fn from(value: Rav1dPicAllocator) -> Self {
        let Rav1dPicAllocator {
            cookie,
            alloc_picture_callback,
            release_picture_callback,
        } = value;
        Self {
            cookie,
            alloc_picture_callback: Some(alloc_picture_callback),
            release_picture_callback: Some(release_picture_callback),
        }
    }
}

#[cfg(feature = "c-ffi")]
impl Rav1dPicAllocator {
    pub fn alloc_picture_data(
        &self,
        w: c_int,
        h: c_int,
        seq_hdr: Arc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>,
        frame_hdr: Option<Arc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>,
    ) -> Rav1dResult<Rav1dPicture> {
        let pic = Rav1dPicture {
            p: Rav1dPictureParameters {
                w,
                h,
                layout: seq_hdr.layout,
                bpc: 8 + 2 * seq_hdr.hbd,
            },
            seq_hdr: Some(seq_hdr),
            frame_hdr,
            ..Default::default()
        };
        let mut pic_c = pic.to::<Dav1dPicture>();
        // SAFETY: `pic_c` is a valid `Dav1dPicture` with `data`, `stride`, `allocator_data` unset.
        let result = unsafe { (self.alloc_picture_callback)(&mut pic_c, self.cookie) };
        result.try_to::<Rav1dResult>().unwrap()?;
        // `data`, `stride`, and `allocator_data` are the only fields set by the allocator.
        // Of those, only `data` and `allocator_data` are read through `r#ref`,
        // so we need to read those directly first and allocate the `Arc`.
        let data = pic_c.data;
        let allocator_data = pic_c.allocator_data;
        let mut pic = pic_c.to::<Rav1dPicture>();
        let len = pic.p.pic_len(pic.stride)?;
        // TODO fallible allocation
        pic.data = Some(Arc::new(Rav1dPictureData {
            data: array::from_fn(|i| {
                let ptr = data[i].map(|ptr| ptr.cast::<u8>());
                let len = len[(i != 0) as usize];
                let stride = pic.stride[(i != 0) as usize];
                // SAFETY: These args come from `Self::alloc_picture_callback`.
                let component = unsafe { Rav1dPictureDataComponentInner::new(ptr, len, stride) };
                Rav1dPictureDataComponent::from_parts(component, stride)
            }),
            allocator_data,
            allocator: self.clone(),
        }));
        Ok(pic)
    }

    pub fn dealloc_picture_data(
        &self,
        data: &mut [Rav1dPictureDataComponent; 3],
        allocator_data: Option<SendSyncNonNull<c_void>>,
    ) {
        let data = data.each_mut().map(|data| data.as_dav1d());
        let mut pic_c = Dav1dPicture {
            data,
            allocator_data,
            ..Default::default()
        };
        // SAFETY: `pic_c` contains the same `data` and `allocator_data`
        // that `Self::alloc_picture_data` set, which now get deallocated here.
        unsafe {
            (self.release_picture_callback)(&mut pic_c, self.cookie);
        }
    }
}

#[cfg(not(feature = "c-ffi"))]
impl Rav1dPicAllocator {
    pub fn alloc_picture_data(
        &self,
        w: c_int,
        h: c_int,
        seq_hdr: Arc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>,
        frame_hdr: Option<Arc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>,
    ) -> Rav1dResult<Rav1dPicture> {
        let p = Rav1dPictureParameters {
            w,
            h,
            layout: seq_hdr.layout,
            bpc: 8 + 2 * seq_hdr.hbd,
        };

        let hbd = (p.bpc > 8) as c_int;
        let aligned_w = p.w + 127 & !127;
        let has_chroma = p.layout != Rav1dPixelLayout::I400;
        let ss_hor = (p.layout != Rav1dPixelLayout::I444) as c_int;
        let mut y_stride = (aligned_w << hbd) as isize;
        let mut uv_stride = if has_chroma { y_stride >> ss_hor } else { 0 };
        if y_stride & 1023 == 0 {
            y_stride += RAV1D_PICTURE_ALIGNMENT as isize;
        }
        if uv_stride & 1023 == 0 && has_chroma {
            uv_stride += RAV1D_PICTURE_ALIGNMENT as isize;
        }
        let stride = [y_stride, uv_stride];
        let [y_sz, uv_sz] = p.pic_len(stride)?;

        // Round up to RAV1D_PICTURE_MULTIPLE for allocated data. Use checked
        // arithmetic: an overflow here (only reachable on 32-bit with crafted
        // dimensions whose plane size lands just below usize::MAX) must surface
        // as ENOMEM, not wrap into an under-sized allocation.
        let round_up = |sz: usize| -> Rav1dResult<usize> {
            if sz == 0 {
                Ok(0)
            } else {
                Ok(sz
                    .checked_add(RAV1D_PICTURE_MULTIPLE - 1)
                    .ok_or(Rav1dError::ENOMEM)?
                    & !(RAV1D_PICTURE_MULTIPLE - 1))
            }
        };
        let y_sz = round_up(y_sz)?;
        let uv_sz = round_up(uv_sz)?;

        // Allocate per-plane buffers with alignment padding (checked add — see above).
        let alloc_plane = |sz: usize| -> Result<Vec<u8>, Rav1dError> {
            if sz == 0 {
                return Ok(Vec::new());
            }
            let alloc_len = sz
                .checked_add(RAV1D_PICTURE_ALIGNMENT)
                .ok_or(Rav1dError::ENOMEM)?;
            self.pool
                .pop_init(alloc_len, 0)
                .map_err(|_| Rav1dError::ENOMEM)
        };

        let y_buf = alloc_plane(y_sz)?;
        let u_buf = alloc_plane(uv_sz)?;
        let v_buf = alloc_plane(uv_sz)?;

        let data = [
            Rav1dPictureDataComponent::from_parts(
                PicBuf::from_vec_aligned(y_buf, RAV1D_PICTURE_ALIGNMENT, y_sz),
                y_stride,
            ),
            Rav1dPictureDataComponent::from_parts(
                PicBuf::from_vec_aligned(u_buf, RAV1D_PICTURE_ALIGNMENT, uv_sz),
                uv_stride,
            ),
            Rav1dPictureDataComponent::from_parts(
                PicBuf::from_vec_aligned(v_buf, RAV1D_PICTURE_ALIGNMENT, uv_sz),
                uv_stride,
            ),
        ];

        let pic = Rav1dPicture {
            p,
            seq_hdr: Some(seq_hdr),
            frame_hdr,
            stride,
            data: Some(Arc::new(Rav1dPictureData {
                data,
                allocator: self.clone(),
            })),
            ..Default::default()
        };

        Ok(pic)
    }
}
pub type PicOffset<'a> = Rav1dPictureDataComponentOffset<'a>;

/// Tallest block [`WithOffset::block_mut_held`] will hold row guards for.
///
/// 64 is the tallest AV1 transform (`TX_64X64` / `TX_16X64` / `TX_32X64`), so
/// in practice nothing falls back. The array costs
/// `MAX_HELD_ROWS * size_of::<Option<guard>>()` of stack in the frame that owns
/// the [`BlockMut`]; a guard is a fat slice reference, a parent reference and a
/// `BorrowId`, and `Option` is niche-packed into the slice pointer.
#[cfg(feature = "held-row-guards")]
const MAX_HELD_ROWS: usize = 64;

#[cfg(feature = "held-row-guards")]
type RowGuards<'a, BD> = [Option<
    DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [<BD as BitDepth>::Pixel]>,
>; MAX_HELD_ROWS];

/// Backing storage for [`BlockMut`] — see [`WithOffset::block_mut`].
enum BlockMutStorage<'a, BD: BitDepth> {
    /// Tile threading off: the picture's own memory, one contiguous guard.
    Direct {
        guard: DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
    },
    /// Tile threading on: a detached compact copy, written back on drop
    /// through fresh per-row guards.
    Compact { buf: Vec<u8> },
    /// Tile threading on, and the per-row MUTABLE guards are held for the
    /// block's whole life — see [`WithOffset::block_mut_held`]. Half the
    /// registrations of `Compact`, same extents.
    #[cfg(feature = "held-row-guards")]
    Held {
        buf: Vec<u8>,
        rows: RowGuards<'a, BD>,
    },
}

/// A writable w×h pixel block. Obtain one with [`WithOffset::block_mut`], which
/// documents why the compact variant exists.
///
/// On drop, a compact block is written back to the picture through per-row
/// guards. A direct block has nothing to do — the caller wrote the picture in
/// place.
pub struct BlockMut<'a, BD: BitDepth> {
    storage: BlockMutStorage<'a, BD>,
    dst: WithOffset<&'a Rav1dPictureDataComponent>,
    w: usize,
    h: usize,
    base: usize,
    byte_stride: isize,
}

impl<'a, BD: BitDepth> BlockMut<'a, BD> {
    /// Byte stride of the buffer returned by [`Self::as_mut_bytes`].
    ///
    /// NOT necessarily the picture's stride: a compact block is tightly packed
    /// at `w * size_of::<BD::Pixel>()`. Always positive in the compact case;
    /// may be negative in the direct case, exactly as the picture's is.
    #[inline]
    pub fn byte_stride(&self) -> isize {
        self.byte_stride
    }

    /// Index within [`Self::as_mut_bytes`] of the block's first pixel.
    ///
    /// Zero for a compact block; for a direct block with negative stride it is
    /// the last row's offset, matching `strided_slice_mut`'s second return value.
    #[inline]
    pub fn base(&self) -> usize {
        self.base
    }

    /// The writable bytes.
    #[inline]
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        match &mut self.storage {
            BlockMutStorage::Direct { guard } => guard.as_mut_bytes(),
            BlockMutStorage::Compact { buf } => buf.as_mut_slice(),
            #[cfg(feature = "held-row-guards")]
            BlockMutStorage::Held { buf, .. } => buf.as_mut_slice(),
        }
    }
}

impl<BD: BitDepth> Drop for BlockMut<'_, BD> {
    fn drop(&mut self) {
        match &mut self.storage {
            BlockMutStorage::Direct { .. } => {}
            BlockMutStorage::Compact { buf } => {
                // Unconditional per-row write-back rather than the loopfilter's
                // diff variant: an inverse transform adds residual across the
                // whole block, so a diff would compare every byte only to
                // discover that every byte changed. Per-row is what keeps the
                // reservations narrow.
                self.dst
                    .compact_write_back_per_row::<BD>(self.w, self.h, buf);
                recycle_compact_scratch(core::mem::take(buf));
            }
            #[cfg(feature = "held-row-guards")]
            BlockMutStorage::Held { buf, rows } => {
                // The rows are already reserved — write straight through the
                // guards taken in `block_mut_held`, then let them drop. Same
                // bytes as `compact_write_back_per_row` would have written,
                // with no second round of registrations.
                let byte_stride = self.byte_stride as usize;
                for (row, slot) in rows.iter_mut().enumerate().take(self.h) {
                    if let Some(guard) = slot {
                        guard.as_mut_bytes()[..byte_stride]
                            .copy_from_slice(&buf[row * byte_stride..][..byte_stride]);
                    }
                }
                // Drop the guards BEFORE recycling, so the scratch is only
                // handed back once the picture is consistent.
                *rows = [const { None }; MAX_HELD_ROWS];
                recycle_compact_scratch(core::mem::take(buf));
            }
        }
    }
}

#[cfg(test)]
mod tile_threading_latch_tests {
    /// `set_tile_threading` must never be able to turn the flag back off.
    ///
    /// It is one process-global bool shared by every decoder, so a
    /// single-threaded `rav1d_open` storing `false` used to push concurrently
    /// live multi-threaded decoders back onto the wide-guard path — whose
    /// borrow extent is `(h-1) * stride + w`, e.g. 16,321 pixels for a 1x16
    /// intra left-edge column — while their tile workers were running. That
    /// showed up as spurious `overlapping DisjointMut` panics under load (8-9
    /// of 24 concurrent runs) and would be an undetected data race in an
    /// `unchecked` build.
    #[test]
    fn set_tile_threading_is_monotone() {
        use super::{set_tile_threading, tile_threading_active};
        // Whatever another test in this binary has already done, `true` sticks.
        set_tile_threading(true);
        assert!(tile_threading_active());
        set_tile_threading(false);
        assert!(
            tile_threading_active(),
            "a single-threaded open must not clear tile threading for \
             concurrently live multi-threaded decoders"
        );
    }
}

/// MECHANISM gate for [`Rav1dPictureDataComponentOffset::for_rows`] and
/// [`Rav1dPictureDataComponentOffset::for_rows_mut`].
///
/// Those two pick between ONE hull registration and `h` per-row ones on
/// [`tile_threading_active`]. Which branch runs is invisible in the decoded
/// pixels, so nothing else in the tree can catch a regression in either
/// direction:
///
/// * per-row taken with threading OFF — the whole point of the helper is lost
///   and 8.4 M registrations per frame come back, with no test going red.
/// * hull taken with threading ON — the hull reserves the inter-row gaps, which
///   belong to other tile COLUMNS. A neighbouring tile's legitimate write then
///   trips a spurious `overlapping DisjointMut` panic (measured 8-9 of 24
///   concurrent runs before [`set_tile_threading`] was made monotone), or, in
///   an `unchecked` build, races undetected.
///
/// So each test asserts which EXTENT got reserved, by holding a byte that only
/// the hull covers and checking whether the tracker rejects the call.
// These assert what the borrow TRACKER reserved, so they cannot run in an
// `unchecked` build (which `asm` implies) — tracking is compiled out there and
// the tests' own anti-vacuity assertion correctly fires. Same idiom as
// src/disjoint_mut.rs:29 and src/safe_simd/pixel_access.rs:451.
#[cfg(all(test, not(feature = "unchecked")))]
mod row_guard_policy_tests {
    use super::{
        Rav1dPictureDataComponent, Rav1dPictureDataComponentInner, set_tile_threading,
        tile_threading_active,
    };
    use crate::include::common::bitdepth::BitDepth8;
    use crate::src::with_offset::WithOffset;
    use std::panic::{self, AssertUnwindSafe};

    const STRIDE: usize = 64;
    const ROWS: usize = 4;
    const W: usize = 8;
    /// First pixel of row 0 to last pixel of row `ROWS-1`, gaps included.
    const HULL: usize = (ROWS - 1) * STRIDE + W;

    fn plane() -> Rav1dPictureDataComponent {
        // `wrap_buf` asserts the byte length is a multiple of 64.
        let mut buf = vec![0u8; STRIDE * ROWS];
        Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut buf, STRIDE)
    }

    /// Does `for_rows_mut` over the `W x ROWS` block at offset 0 conflict with a
    /// live mutable borrow of the single pixel `probe`?
    fn conflicts_with(probe: usize) -> bool {
        let pic = plane();
        let held = pic.slice_mut::<BitDepth8, _>((probe.., ..1));
        let at = WithOffset {
            data: &pic,
            offset: 0,
        };
        let prev = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            at.for_rows_mut::<BitDepth8, _>(W, ROWS, |_, row| {
                row[0] = 1;
            });
        }));
        panic::set_hook(prev);
        drop(held);
        r.is_err()
    }

    fn assert_hull_branch() {
        assert!(
            !tile_threading_active(),
            "precondition: this must run in a process where no decoder has \
             latched tile threading, or it is testing the other branch"
        );
        // A byte in the INTER-ROW GAP — inside the hull, outside every row's
        // own `[row*STRIDE, row*STRIDE + W)`. Only the hull registration covers
        // it, so a conflict here is proof the hull branch ran.
        assert!(
            conflicts_with(W),
            "with tile threading off, `for_rows_mut` must take ONE hull guard; \
             a gap byte inside [0, {HULL}) did not conflict, so the per-row \
             branch ran instead"
        );
        // Control: past the hull nothing may conflict, or the assertion above
        // would pass for the wrong reason.
        assert!(
            !conflicts_with(HULL),
            "a byte outside the hull must never conflict"
        );
    }

    /// The threading-OFF half, run in a CHILD PROCESS.
    ///
    /// `TILE_THREADING` is a monotone process-global and
    /// `set_tile_threading_is_monotone` (above, same binary) latches it on, so
    /// this branch is simply not observable in the parent — and libtest gives
    /// no ordering guarantee that would fix that. Re-exec'ing ourselves with
    /// `--exact` gives a process running this test and nothing else.
    ///
    /// The child's marker line is checked, not just its exit status: libtest
    /// exits 0 when a filter matches NOTHING, so a renamed test would otherwise
    /// turn this gate green while running no assertions at all.
    #[test]
    fn for_rows_mut_reserves_the_strided_hull_when_tile_threading_is_off() {
        const MARKER: &str = "ROW_GUARD_HULL_CHILD_RAN";
        const NAME: &str = "include::dav1d::picture::row_guard_policy_tests::\
                            for_rows_mut_reserves_the_strided_hull_when_tile_threading_is_off";
        if std::env::var_os("RAV1D_ROW_GUARD_CHILD").is_some() {
            assert_hull_branch();
            println!("{MARKER}");
            return;
        }
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["--exact", NAME, "--nocapture"])
            .env("RAV1D_ROW_GUARD_CHILD", "1")
            .output()
            .expect("re-exec the test binary");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains(MARKER),
            "the child process did not reach the assertions (renamed test? \
             libtest exits 0 on an empty filter). status={:?}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            stdout,
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(out.status.success(), "child failed:\n{stdout}");
    }

    // ---- Negative strides (#520) --------------------------------------------
    //
    // The safe allocator never produces a negative stride, so these build one
    // by hand through the private constructor: a plane whose picture rows run
    // BOTTOM-UP in memory, exactly what a c-ffi `Dav1dPicAllocator` hands in
    // when it returns a negative stride. Picture row `r` occupies memory row
    // `BU_TOP_ROW - r`; row 0 starts at pixel `ROW0`. The first 16 bytes of
    // every memory row are stamped `row * 16 + column` so a read-back names the
    // pixel it came from.
    //
    // The hull guards under test are only taken with tile threading OFF (with
    // it on, the bounds map's extent ceiling rejects any hull-sized
    // reservation), and `TILE_THREADING` is a monotone process-global that
    // other tests in this binary latch, so each of these runs its body in a
    // child process — see `in_child_process`.

    /// Memory rows in the bottom-up plane: one slack row below the block's last
    /// row and one above row 0, so probes just outside the block stay inside
    /// the buffer.
    const BU_MEM_ROWS: usize = ROWS + 2;
    /// Memory row holding picture row 0.
    const BU_TOP_ROW: usize = BU_MEM_ROWS - 2;
    /// Pixel offset of picture row 0.
    const ROW0: usize = BU_TOP_ROW * STRIDE;
    /// `(ROWS-1) * STRIDE`: how far below `ROW0` the block's last row starts,
    /// and row 0's index inside the block's hull.
    const SPAN: usize = (ROWS - 1) * STRIDE;

    fn stamp(row: usize, x: usize) -> u8 {
        (row * 16 + x) as u8
    }

    fn bottom_up_plane() -> Rav1dPictureDataComponent {
        let mut buf = vec![0u8; STRIDE * BU_MEM_ROWS];
        for mem_row in 0..BU_MEM_ROWS {
            for x in 0..16 {
                // The slack row above row 0 has no picture row; mark it with a
                // value no block row can produce.
                buf[mem_row * STRIDE + x] = if mem_row <= BU_TOP_ROW {
                    stamp(BU_TOP_ROW - mem_row, x)
                } else {
                    0xEE
                };
            }
        }
        let inner = Rav1dPictureDataComponentInner::from_slice_copy(&buf);
        Rav1dPictureDataComponent::from_parts(inner, -(STRIDE as isize))
    }

    /// Run `body` in a CHILD PROCESS where no decoder has latched tile
    /// threading, and check that it reached its end. `name` is the calling
    /// test's full path (for `--exact`); `marker` is a line only the body
    /// prints, checked instead of the bare exit status because libtest exits
    /// 0 when a filter matches nothing. Same idiom as
    /// `for_rows_mut_reserves_the_strided_hull_when_tile_threading_is_off`.
    fn in_child_process(name: &str, marker: &str, body: impl FnOnce()) {
        if std::env::var_os("RAV1D_ROW_GUARD_CHILD").is_some() {
            assert!(
                !tile_threading_active(),
                "precondition: a fresh process, or this tests the other branch"
            );
            body();
            println!("{marker}");
            return;
        }
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["--exact", name, "--nocapture"])
            .env("RAV1D_ROW_GUARD_CHILD", "1")
            .output()
            .expect("re-exec the test binary");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains(marker),
            "the child process did not reach the assertions (renamed test? \
             libtest exits 0 on an empty filter). status={:?}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            stdout,
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(out.status.success(), "child failed:\n{stdout}");
    }

    const TESTS: &str = "include::dav1d::picture::row_guard_policy_tests::";

    /// `narrow_guard{,_mut}` on a bottom-up plane: the guard is the block's
    /// hull, `(h-1)*|stride| + w` pixels from the LAST row's first pixel to
    /// the end of row 0, and `base + x + r * stride` (stride negative) lands
    /// on picture row `r`, column `x`. #520 returned a hull shifted `w-1`
    /// pixels down with base `total - 1`, so `base + x` for any `x >= 1` was
    /// past the guard.
    #[test]
    fn narrow_guards_cover_the_block_on_a_negative_stride() {
        in_child_process(
            &format!("{TESTS}narrow_guards_cover_the_block_on_a_negative_stride"),
            "NEG_STRIDE_NARROW_GUARDS_CHILD_RAN",
            || {
                let pic = bottom_up_plane();
                let at = WithOffset {
                    data: &pic,
                    offset: ROW0,
                };
                {
                    let (guard, base) = at.narrow_guard::<BitDepth8>(W, ROWS);
                    assert_eq!(guard.len(), HULL, "hull is (h-1)*|stride| + w pixels");
                    assert_eq!(
                        base, SPAN,
                        "row 0 sits (h-1)*|stride| into a bottom-up hull"
                    );
                    for r in 0..ROWS {
                        for x in 0..W {
                            assert_eq!(
                                guard[base + x - r * STRIDE],
                                stamp(r, x),
                                "row {r} col {x}"
                            );
                        }
                    }
                }
                let (guard, base) = at.narrow_guard_mut::<BitDepth8>(W, ROWS);
                assert_eq!(guard.len(), HULL);
                assert_eq!(base, SPAN);
                for r in 0..ROWS {
                    for x in 0..W {
                        assert_eq!(guard[base + x - r * STRIDE], stamp(r, x), "row {r} col {x}");
                    }
                }
            },
        );
    }

    /// `for_rows` / `for_rows_mut` hand out picture rows top-down with the
    /// right bytes on a bottom-up plane, and writes land where they should.
    /// Both of their branches were already right; this pins them through the
    /// shared hull helper. Runs in the parent (per-row branch when threading is
    /// latched) AND in a child (hull branch).
    #[test]
    fn for_rows_visits_picture_rows_top_down_on_a_negative_stride() {
        fn check() {
            let pic = bottom_up_plane();
            let at = WithOffset {
                data: &pic,
                offset: ROW0,
            };
            let mut seen = 0;
            at.for_rows::<BitDepth8, _>(W, ROWS, |r, row| {
                assert_eq!(row.len(), W);
                for (x, &px) in row.iter().enumerate() {
                    assert_eq!(px, stamp(r, x), "row {r} col {x}");
                }
                seen += 1;
            });
            assert_eq!(seen, ROWS);
            at.for_rows_mut::<BitDepth8, _>(W, ROWS, |r, row| {
                for (x, px) in row.iter_mut().enumerate() {
                    assert_eq!(*px, stamp(r, x), "row {r} col {x}");
                    *px ^= 0x80;
                }
            });
            at.for_rows::<BitDepth8, _>(W, ROWS, |r, row| {
                for (x, &px) in row.iter().enumerate() {
                    assert_eq!(px, stamp(r, x) ^ 0x80, "row {r} col {x} after write");
                }
            });
        }
        if std::env::var_os("RAV1D_ROW_GUARD_CHILD").is_none() {
            check();
        }
        in_child_process(
            &format!("{TESTS}for_rows_visits_picture_rows_top_down_on_a_negative_stride"),
            "NEG_STRIDE_FOR_ROWS_CHILD_RAN",
            check,
        );
    }

    /// The fast compact read returns the hull in MEMORY order: on a bottom-up
    /// plane the block's last picture row comes first and row 0 last, with
    /// the original (unsigned) stride. #520 started the copy `w-1` pixels too
    /// low, so row 0 was cut to its first pixel. `compact_read` itself never
    /// sends a negative stride here (next test); this pins the geometry.
    #[test]
    fn compact_read_fast_hull_starts_at_the_last_row_on_a_negative_stride() {
        in_child_process(
            &format!("{TESTS}compact_read_fast_hull_starts_at_the_last_row_on_a_negative_stride"),
            "NEG_STRIDE_COMPACT_READ_FAST_CHILD_RAN",
            || {
                let pic = bottom_up_plane();
                let at = WithOffset {
                    data: &pic,
                    offset: ROW0,
                };
                let (buf, byte_stride) = at.compact_read_fast::<BitDepth8>(W, ROWS);
                assert_eq!(byte_stride, STRIDE);
                assert_eq!(buf.len(), HULL);
                for r in 0..ROWS {
                    for x in 0..W {
                        assert_eq!(buf[SPAN + x - r * STRIDE], stamp(r, x), "row {r} col {x}");
                    }
                }
            },
        );
    }

    /// `compact_read` (the dispatcher) on a negative stride must return the
    /// compact, row-0-first layout whatever the threading mode: the fast
    /// path's memory-order hull carries no base, so a caller cannot find row
    /// 0 in it when the stride is negative. Threading is off in the child, so
    /// this is the branch that used to pick the fast path.
    #[test]
    fn compact_read_returns_picture_rows_top_down_on_a_negative_stride() {
        in_child_process(
            &format!("{TESTS}compact_read_returns_picture_rows_top_down_on_a_negative_stride"),
            "NEG_STRIDE_COMPACT_READ_CHILD_RAN",
            || {
                let pic = bottom_up_plane();
                let at = WithOffset {
                    data: &pic,
                    offset: ROW0,
                };
                let (buf, byte_stride) = at.compact_read::<BitDepth8>(W, ROWS);
                assert_eq!(byte_stride, W, "compact layout, not the memory-order hull");
                for r in 0..ROWS {
                    for x in 0..W {
                        assert_eq!(buf[r * byte_stride + x], stamp(r, x), "row {r} col {x}");
                    }
                }
            },
        );
    }

    /// Does a live mutable borrow of the single pixel `probe` make
    /// `narrow_guard_mut` over the `W x ROWS` block at `ROW0` of a bottom-up
    /// plane conflict?
    fn bottom_up_block_conflicts_with(probe: usize) -> bool {
        let pic = bottom_up_plane();
        let held = pic.slice_mut::<BitDepth8, _>((probe.., ..1));
        let at = WithOffset {
            data: &pic,
            offset: ROW0,
        };
        let prev = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            let (guard, base) = at.narrow_guard_mut::<BitDepth8>(W, ROWS);
            std::hint::black_box((guard.len(), base));
        }));
        panic::set_hook(prev);
        drop(held);
        r.is_err()
    }

    /// What the TRACKER reserves for a block on a negative stride: exactly the
    /// hull, from the last row's first pixel to the end of row 0. #520's
    /// hull ended at `offset + 1` (every row-0 pixel but the first
    /// unreserved) and began `w-1` pixels below the last row (reserved for
    /// nothing).
    #[test]
    fn narrow_guard_mut_reserves_exactly_the_block_hull_on_a_negative_stride() {
        in_child_process(
            &format!(
                "{TESTS}narrow_guard_mut_reserves_exactly_the_block_hull_on_a_negative_stride"
            ),
            "NEG_STRIDE_RESERVATION_CHILD_RAN",
            || {
                assert!(
                    bottom_up_block_conflicts_with(ROW0 + W - 1),
                    "row 0's last column is inside the block and must be reserved"
                );
                assert!(
                    bottom_up_block_conflicts_with(ROW0 - SPAN),
                    "the last row's first column is inside the block and must be reserved"
                );
                assert!(
                    !bottom_up_block_conflicts_with(ROW0 + W),
                    "the pixel past the end of row 0 must not be reserved"
                );
                assert!(
                    !bottom_up_block_conflicts_with(ROW0 - SPAN - 1),
                    "the pixel below the last row's start must not be reserved"
                );
            },
        );
    }

    /// The threading-ON half. Latches the flag itself, so it is independent of
    /// what else in this binary has run.
    #[test]
    fn for_rows_mut_never_reserves_an_inter_row_gap_when_tile_threading_is_on() {
        set_tile_threading(true);
        assert!(tile_threading_active());
        assert!(
            !conflicts_with(W),
            "with tile threading ON, `for_rows_mut` must take PER-ROW guards; a \
             byte in the inter-row gap conflicted, which is exactly the false \
             positive `block_mut`'s tile-threading branch and the monotone \
             `set_tile_threading` latch exist to prevent"
        );
        // Anti-vacuity: without this, the assertion above would also pass with
        // borrow tracking compiled out entirely (`--features unchecked`).
        assert!(
            conflicts_with(STRIDE),
            "row 1 column 0 IS written by this block, so a live mutable borrow \
             of it must conflict; if it does not, tracking is off in this build \
             and this test gates nothing"
        );
    }
}
