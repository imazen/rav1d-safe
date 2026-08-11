//! Address-block **sharded** borrow tracker.
//!
//! # Why
//!
//! The predecessor ([`crate::tracker_legacy`]) is one spin lock plus one 64-slot
//! record table per [`DisjointMut`](crate::DisjointMut) instance. A 4K frame
//! registers ~50 M borrows, and on multi-tile content every tile worker funnels
//! its share of them through the *same* lock and the *same* four parallel
//! arrays. Measured on an M4 Pro (`benchmarks/tracker_decomp_2026-08-07.meta`):
//! decode gets *slower* with more threads (591 ms at t=1, 1264 ms at t=8), and
//! an arm that takes the lock and does **nothing** inside it still costs 2.7x
//! at t=8. The cost is not the overlap scan — it is that one cache line is
//! written by eight cores tens of millions of times per frame.
//!
//! # The design
//!
//! Split each instance's tracker into `N_SHARDS` independently locked shards,
//! selected by the **address** of the borrow rather than by thread or instance
//! (measured: all 12 hot instances are touched by all 8 workers, so instance
//! sharding buys nothing, and thread sharding would make each registrant read
//! every other thread's line).
//!
//! * The buffer is cut into `1 << shift`-byte blocks — a per-instance shift
//!   under `__blockshift_adaptive`, the [`BLOCK_SHIFT`] constant otherwise.
//! * `shard(block)` is a multiplicative hash of the block index.
//! * A borrow `R` registers its **exact** interval `[R.start, R.end)` — not a
//!   clipped piece — in every shard that `R`'s blocks map to, and checks for
//!   overlaps in exactly those shards.
//!
//! ## Soundness
//!
//! *No missed overlap.* If `R1` and `R2` share a byte `x`, `x` lies in some
//! block `b`, so `shard(b)` is in both `shards(R1)` and `shards(R2)`. Whichever
//! registers second holds `shard(b)`'s lock, scans it, and finds the other's
//! record. Registration happens *before* the reference is created (unchanged
//! from the legacy tracker), so there is no TOCTOU gap.
//!
//! That argument needs only that both registrants agree on where the block
//! boundaries are and on which shard a block maps to. It does NOT constrain
//! either choice, which is why the block shift can be a per-instance value and
//! `shard_of` can be any function of the block index: both are fixed for a
//! tracker's whole life (they move only in [`BorrowTracker::reprovision`],
//! which takes `&mut self` and so runs with no borrow outstanding). Those are
//! locality-versus-collision knobs, never correctness ones.
//!
//! *Release does not need the lock.* Retiring a record is one bit-clear in
//! [`Shard::occupied`], which is atomic and so cannot lose, or be lost by, the
//! `fetch_or` a concurrent registration publishes with. See
//! [`BorrowTracker::remove`] for why this does not widen the add/remove race.
//!
//! *No false positive.* Every stored record is the borrow's full interval, so
//! two records overlap exactly when the two borrows do. (Storing a *clipped*
//! per-shard record instead would be unsound in the other direction — a shard
//! whose blocks are non-contiguous would report a hull that covers bytes the
//! borrow never touched.)
//!
//! *No deadlock.* [`TinyLock`] is not reentrant, so multi-shard operations sort
//! and dedupe their shard indices and acquire strictly ascending. The wide path
//! acquires *all* shards, also ascending, from a state where it holds none.
//!
//! This is **not** the March-2026 strided tracker that was merged and reverted
//! the next day: that one declared the gaps between rows unaccessed, and safe
//! code could write them. Nothing here ever declares a byte unaccessed.

use super::*;
use core::panic::Location;
#[cfg(not(feature = "__probe_lock_park"))]
use core::sync::atomic::AtomicBool;
use core::sync::atomic::{AtomicU8, AtomicU32, AtomicUsize, Ordering};

// =============================================================================
// Tunables (compile-time A/B knobs — see benchmarks/shard_tracker_*.meta)
// =============================================================================

/// Shards per instance. Power of two.
///
/// Trades collision probability against the tracker's cache footprint: every
/// shard is one 128-byte line (the M-series line size), so an instance costs
/// `N_SHARDS * 128` bytes and the 12 hot picture planes cost 12x that. The
/// borrow stream walks the whole table (a 128x128 superblock touches ~120
/// distinct blocks), so the table is not effectively smaller than its size.
///
/// Measured, v4k_8tile 8bpc, M4 Pro, ms/frame (`benchmarks/
/// shard_tracker_2026-08-07.meta`, screening pass):
///
/// ```text
///   shards     t=1     t=8
///   legacy    583.9  1120.8
///        1    624.0  1318.3
///       16    567.6   386.4
///       32    578.0   333.7   <- default
///       64    603.7   304.2
///      128    675.9   286.8
///      256    743.4   (stack overflow in a worker)
/// ```
///
/// In the final interleaved sweep (median of 9, tracker boxed) 32 shards costs
/// nothing single-threaded on 8-bit content — 600.7 ms against the legacy
/// tracker's 602.3 — while running 5.1x faster at t=8. 64 is 1-5% better again
/// at t=4/t=8 for ~3% at t=1 and ~7 MB of RSS, and is available as
/// `shards-64`. 256 is not offered: at 32 KiB per instance it overflowed a
/// worker stack while constructing a `DisjointMut` back when the array was
/// inline, and even boxed it was already past the point of diminishing returns.
///
/// **Re-measured 2026-08-07, and the default moved 32 -> 128**
/// (`benchmarks/p3_inversion_2026-08-07.meta`). The table above was taken
/// before the P2 kernel work; with the kernels cheaper, the tracker is a much
/// larger share of the multi-thread wall and the shard count matters far more
/// than it did. Same host, same vector, median of 9, interleaved:
///
/// ```text
///   shards     t=1     t=4     t=8    t=16
///       32    412.5   175.3   190.5   209.9   <- the old default: INVERTS at t>4
///       64    447.1   163.7   162.0   172.3
///      128    522.5   139.1   123.5   135.6
///   tracker
///   removed   336.5    95.5    57.7    65.2
/// ```
///
/// The t=1 column is not a cache effect — a 128-shard array with only 32 of
/// them ACTIVE measured 528.0 ms, indistinguishable from using all 128 — it is
/// the wide path holding every shard it is given. `SHARDS_SERIAL` and
/// [`BorrowTracker::active`] are what make the two columns independent.
//
// The cascade is priority-ordered rather than a set of independent `cfg`s, so
// enabling two knobs at once (`--all-features`) still compiles instead of
// defining the constant twice.
#[cfg(feature = "__shards_1")]
pub(super) const N_SHARDS: usize = 1;
#[cfg(all(feature = "__shards_4", not(feature = "__shards_1")))]
pub(super) const N_SHARDS: usize = 4;
#[cfg(all(
    feature = "__shards_8",
    not(any(feature = "__shards_1", feature = "__shards_4"))
))]
pub(super) const N_SHARDS: usize = 8;
#[cfg(all(
    feature = "__shards_16",
    not(any(feature = "__shards_1", feature = "__shards_4", feature = "__shards_8"))
))]
pub(super) const N_SHARDS: usize = 16;
#[cfg(all(
    feature = "__shards_32",
    not(any(
        feature = "__shards_1",
        feature = "__shards_4",
        feature = "__shards_8",
        feature = "__shards_16"
    ))
))]
pub(super) const N_SHARDS: usize = 32;
#[cfg(all(
    feature = "__shards_64",
    not(any(
        feature = "__shards_1",
        feature = "__shards_4",
        feature = "__shards_8",
        feature = "__shards_16",
        feature = "__shards_32"
    ))
))]
pub(super) const N_SHARDS: usize = 64;
#[cfg(all(
    feature = "__shards_128",
    not(any(
        feature = "__shards_1",
        feature = "__shards_4",
        feature = "__shards_8",
        feature = "__shards_16",
        feature = "__shards_32",
        feature = "__shards_64"
    ))
))]
pub(super) const N_SHARDS: usize = 128;
/// 256 shards: MEASURED WORSE AT EVERY THREAD COUNT, kept only as an A/B rung.
///
/// The 128 default was chosen while the `check_tile` deblock barrier still
/// capped achieved occupancy at 2.86 of 8, so "more shards for less collision"
/// was worth re-testing once occupancy reached 7.14. It is not: v4k_8tile 8bpc,
/// M4 Pro, interleaved median of 5 on an idle box, against the default
/// (`benchmarks/scaling_shards_2026-08-08.tsv`), holding the block shift at 14
/// so only the table size moves — t=1 1.0730, t=4 1.0594, t=8 1.0295. The
/// bigger table costs more than the collisions it removes, which is the same
/// verdict the original 32/64/128/256 ladder reached for a different reason.
#[cfg(all(
    feature = "__shards_256",
    not(any(
        feature = "__shards_1",
        feature = "__shards_4",
        feature = "__shards_8",
        feature = "__shards_16",
        feature = "__shards_32",
        feature = "__shards_64",
        feature = "__shards_128"
    ))
))]
pub(super) const N_SHARDS: usize = 256;
#[cfg(not(any(
    feature = "__shards_1",
    feature = "__shards_4",
    feature = "__shards_8",
    feature = "__shards_16",
    feature = "__shards_32",
    feature = "__shards_64",
    feature = "__shards_128",
    feature = "__shards_256"
)))]
pub(super) const N_SHARDS: usize = 128;

/// `log2` of the block size in elements.
///
/// Two forces pull opposite ways. Small blocks separate concurrent tile columns
/// better — at 4K with 4 tile columns two workers on the same picture row are
/// only 960 bytes apart, so a 4 KiB block puts them in the same shard. Large
/// blocks keep more borrows inside ONE block, and a borrow that spans several
/// takes the multi-shard path: two or more ordered lock acquisitions plus a
/// sort, instead of one.
///
/// Measured, the second force wins, and not marginally. Borrow lengths on the
/// hot planes (`benchmarks/shard_sizing_2026-08-07.txt`, v4k_8tile 8bpc): 77.3%
/// are a single byte and 99.94% are <= 31 bytes. Fraction spanning exactly one
/// block: 99.875% at shift 8, 99.985% at shift 12 — 8x fewer multi-shard
/// registrations. The same probe shows the peer-collision rate is nearly flat
/// from shift 6 to shift 12 (0.031-0.040 colliding peers per add against the
/// unsharded 2.07), i.e. the tile-column argument barely shows up, because
/// workers are rarely on the same row at the same instant.
///
/// A/B at 32 shards, ms/frame, median of 9, interleaved
/// (`benchmarks/shard_tracker_2026-08-07.meta`, shift screening):
///
/// ```text
///   vector              t    shift 8   shift 12
///   v4k_8tile 8bpc      1      605.0      602.4
///   v4k_8tile 8bpc      8      349.7      333.9   -4.5%
///   v4k_8tile 10bpc     8      433.8      421.2   -2.9%
///   v4k_1tile 10bpc     8      640.5      636.9
/// ```
///
/// So: chosen by measurement, not by the tile-geometry argument, which the
/// measurement does not support. Shift 8 and 10 remain available as
/// `blockshift-8` / `blockshift-10` if a different tiling ever inverts this.
///
/// **Re-opened 2026-08-08, and the ladder does NOT stop at 12**
/// (`benchmarks/tracker_blockshift_2026-08-08.meta`). The 8/10/12 screening
/// above measured the right thing for the wrong quantity: it counted
/// multi-shard REGISTRATIONS, and those really are ~flat past shift 12. What it
/// could not see is that the tracker's cost is the shard CACHE LINE, not the
/// registration — and a strided access pays one line per ROW.
///
/// The biggest single tracker consumer is `rav1d_prepare_intra_edges`'s
/// left-COLUMN read, 9.19% of a t=8 4K frame's samples, which registers one
/// 1-PIXEL interval per row to read one byte per row. At a 3840-byte row and a
/// 4 KiB block no two of those rows share a block, so a 16-row column costs 16
/// distinct shard lines. Doubling the block halves that.
///
/// Measured, `examples/probe_tracker`, v4k_8tile, t=8, ms/frame:
///
/// ```text
///   shift  bytes/block  rows/block   8bpc   10bpc   add_slow(8bpc)
///      12         4096        1.07  119.0   140.7            72585
///      13         8192        2.13   89.3   130.1            36197
///      14        16384        4.27   72.8   100.4            18061
///      15        32768        8.53   73.0    92.1             9038
///      16        65536       17.07   75.5    91.0             4454
/// ```
///
/// Wide-path promotions are ZERO at every one of those, on both bit depths, so
/// nothing here is trading against the all-shards path. `rows/block` is at the
/// 8-bit luma stride; the 10-bit column peaks two shifts later because its
/// stride is twice as wide, which is the observation
/// [`block_shift_for`] turns into a rule.
///
/// The fixed values stay available as `blockshift-13/14/15/16`, but a CONSTANT
/// is the wrong shape: the shift that makes a 4K plane's rows share a block
/// turns a 64 KiB buffer into one block and one lock. Prefer
/// `blockshift-adaptive`.
#[cfg(feature = "__blockshift_8")]
const BLOCK_SHIFT: u32 = 8;
#[cfg(all(feature = "__blockshift_10", not(feature = "__blockshift_8")))]
const BLOCK_SHIFT: u32 = 10;
#[cfg(all(
    feature = "__blockshift_13",
    not(any(feature = "__blockshift_8", feature = "__blockshift_10"))
))]
const BLOCK_SHIFT: u32 = 13;
#[cfg(all(
    feature = "__blockshift_14",
    not(any(
        feature = "__blockshift_8",
        feature = "__blockshift_10",
        feature = "__blockshift_13"
    ))
))]
const BLOCK_SHIFT: u32 = 14;
#[cfg(all(
    feature = "__blockshift_16",
    not(any(
        feature = "__blockshift_8",
        feature = "__blockshift_10",
        feature = "__blockshift_13",
        feature = "__blockshift_14"
    ))
))]
const BLOCK_SHIFT: u32 = 16;
#[cfg(all(
    feature = "__blockshift_15",
    not(any(
        feature = "__blockshift_8",
        feature = "__blockshift_10",
        feature = "__blockshift_13",
        feature = "__blockshift_14",
        feature = "__blockshift_16"
    ))
))]
const BLOCK_SHIFT: u32 = 15;
#[cfg(not(any(
    feature = "__blockshift_8",
    feature = "__blockshift_10",
    feature = "__blockshift_13",
    feature = "__blockshift_14",
    feature = "__blockshift_15",
    feature = "__blockshift_16"
)))]
const BLOCK_SHIFT: u32 = 12;

/// Records per shard. Sized so a shard is exactly one 128-byte cache line.
///
/// Measured max concurrent borrows on one *instance* is 8 at t=8 and those are
/// spread across shards, so per-shard occupancy is ~0.005 on average. A shard
/// that fills up anyway promotes the borrow to the wide list; correctness does
/// not depend on this number.
const SLOTS: usize = 7;

/// Bits 0..SLOTS of the occupancy masks.
const SLOTS_MASK: u8 = ((1u16 << SLOTS) - 1) as u8;

/// Where the RECTANGLE bitmap starts inside [`ShardRecs::flags`]. The mutability
/// bitmap occupies bits `0..SLOTS`.
const RECT_SHIFT: u32 = 8;
const _: () = assert!(SLOTS as u32 <= RECT_SHIFT);
const _: () = assert!(RECT_SHIFT + SLOTS as u32 <= u16::BITS);

/// Rows a single strided-rectangle record may describe.
///
/// This is a REPRESENTABILITY limit, not an approximation: a caller whose
/// rectangle is taller declines the rectangle record and takes its own per-row
/// path (see [`BorrowTracker::add_rect`]), so nothing is ever rounded up. The
/// bound exists because the exact rectangle-vs-rectangle test walks rows, so an
/// unbounded row count would put an unbounded loop inside an overlap scan.
///
/// 64 covers the decoder's real geometry with room to spare: the loop filter's
/// compact read is at most 16 rows (`2 * lf_reach(16)` down a column, or
/// `4 * groups` along one), and the CDEF/MC strided helpers are at most 8-16.
const MAX_RECT_ROWS: usize = 64;

/// `(rows, seg)` of the rectangle whose hull is `[h0, h1)` on stride `s`.
///
/// **Exact, and the inverse of the encoding.** A rectangle stores
/// `h1 - h0 = (rows - 1) * s + seg` with `1 <= seg <= s`, so
/// `h1 - h0 - 1 = (rows - 1) * s + (seg - 1)` with `0 <= seg - 1 < s`: the
/// division recovers `rows - 1` uniquely, and `seg` follows. That bijection is
/// the whole reason a rectangle record needs no storage beyond the two words a
/// plain interval already uses.
///
/// `seg <= s` is enforced at registration ([`BorrowTracker::add_rect`] declines
/// otherwise), and `s > 0` because a record is only marked as a rectangle when
/// the instance has a declared stride.
#[inline(always)]
fn rect_decode(h0: usize, h1: usize, s: usize) -> (usize, usize) {
    debug_assert!(s > 0 && h1 > h0);
    let span = h1 - h0;
    let rows = (span - 1) / s + 1;
    (rows, span - (rows - 1) * s)
}

/// The first row segment of the rectangle hulled by `[h0, h1)` on stride `s`
/// that intersects `[a, b)`, or `None` when the rectangle and the interval are
/// genuinely disjoint.
///
/// This is the test that makes a rectangle record EXACT rather than a hull: the
/// inter-row gaps are not part of the record, so a probe that only touches them
/// gets `None`. It reserves nothing and rounds nothing — the returned interval
/// is one real row segment.
#[cold]
#[inline(never)]
fn rect_hit_range(h0: usize, h1: usize, s: usize, a: usize, b: usize) -> Option<(usize, usize)> {
    let (rows, seg) = rect_decode(h0, h1, s);
    // Clip the probe to the hull first: outside it there is nothing to find, and
    // inside it the first candidate row is a division away.
    let lo = if a > h0 { a } else { h0 };
    let hi = if b < h1 { b } else { h1 };
    if lo >= hi {
        return None;
    }
    let mut r = (lo - h0) / s;
    while r < rows {
        let rs = h0 + r * s;
        if rs >= hi {
            break;
        }
        // `rs + seg` cannot overflow: it is at most `h1`, which the caller
        // already holds as a valid buffer offset.
        if lo < rs + seg {
            return Some((rs, rs + seg));
        }
        r += 1;
    }
    None
}

/// The registration site handed to `alloc`.
///
/// Only debug builds keep it: in release the wrapper methods do not propagate
/// `#[track_caller]` from the borrow site, so the stored value was the
/// wrapper's own line — no diagnostic worth one store per borrow on the
/// hottest path in the decoder. See [`ShardRecs::locs`].
type Loc = &'static Location<'static>;

#[inline(always)]
#[track_caller]
fn here() -> Loc {
    Location::caller()
}

/// A borrow touching more distinct shards than this goes to the wide list
/// instead. Measured 0.000% of hot borrows at the shipped BLOCK_SHIFT
/// (0.009% at shift 8).
///
/// **The cap is not free to raise, and the binding constraint is [`BorrowId`],
/// not the tracker.** The id has to stay one register-sized word, so
/// `KIND_BITS + N_BITS + PAIR_BITS * MAX_SHARDS_PER_BORROW <= 64` — pinned by a
/// const assert below. At the 12-bit pair the id shipped with, 4 pairs fit and 5
/// do not; narrowing the pair to `log2(N_SHARDS) + 3` (10 bits at the default
/// 128 shards) buys the fifth. `__msb_5` is that arm.
///
/// Why an arm at all: the strided-2D record's refuting quantity is
/// `pct_row_wide` — the fraction of would-be 2-D registrations that exceed this
/// cap — measured 0.54%-70.59% per site
/// (`benchmarks/strided_2d_2026-08-10.meta` §4). If a cap raise collapses that,
/// the exact 2-D record becomes viable and the per-row split (2.86x the
/// registrations) can go. Measure `__probe_wide` and the `eval_rect` cap columns
/// BEFORE timing anything: under the shipped per-row scheme a wide promotion is
/// rare, so a cap raise can only pay through the counterfactual.
#[cfg(feature = "__msb_5")]
const MAX_SHARDS_PER_BORROW: usize = 5;
#[cfg(not(feature = "__msb_5"))]
const MAX_SHARDS_PER_BORROW: usize = 4;

/// Blocks scanned before giving up and going wide. Bounds the fast path's work
/// for a pathologically long borrow (e.g. an unbounded `index(..)`).
const MAX_BLOCKS_SCAN: usize = 64;

/// THROWAWAY wide-path reason counters (`__probe_wide`).
///
/// The wide path holds EVERY active shard of an instance, so a workload that
/// reaches it at any rate collapses. Which of its three doors a promotion came
/// through decides what to do about it, and they move in OPPOSITE directions
/// as `BLOCK_SHIFT` grows: `WIDE_SHARDS` and `WIDE_BLOCKS` get rarer, while
/// `WIDE_FULL` (slot exhaustion, because a bigger block funnels more
/// simultaneous borrows onto one shard) gets commoner. Without these the shift
/// ladder's cliffs can only be guessed at.
///
/// Unlike `__probe_count`, this does NOT switch the crate to the legacy
/// tracker — it has to observe the sharded one.
#[cfg(feature = "__probe_wide")]
pub mod wide_probe {
    use core::sync::atomic::AtomicU64;
    use core::sync::atomic::Ordering::Relaxed;

    /// `add_multi` saw more than `MAX_SHARDS_PER_BORROW` distinct shards.
    pub static WIDE_SHARDS: AtomicU64 = AtomicU64::new(0);
    /// The borrow spanned more than `MAX_BLOCKS_SCAN` blocks.
    pub static WIDE_BLOCKS: AtomicU64 = AtomicU64::new(0);
    /// A shard had no free slot, so the record was promoted.
    pub static WIDE_FULL: AtomicU64 = AtomicU64::new(0);
    /// `add_slow` entries (poisoned / a live wide record / multi-block).
    pub static N_SLOW: AtomicU64 = AtomicU64::new(0);
    /// `add_multi` entries.
    pub static N_MULTI: AtomicU64 = AtomicU64::new(0);
    /// [`super::BorrowTracker::add_contended`] entries: the single-block fast
    /// path's `try_lock` LOST — somebody else held the shard. This is the
    /// primary contention rate, and it is strictly larger than [`N_LOCKSLOW`]
    /// because the retry inside `lock()` often succeeds without ever spinning.
    pub static N_CONTENDED: AtomicU64 = AtomicU64::new(0);
    /// [`super::TinyLock::lock_slow`] entries: a thread found a shard lock held
    /// and had to WAIT for it. This is the direct CONTENTION count, and it is
    /// the quantity the wide/multi counters are only a proxy for — a cell can
    /// have zero multi-shard registrations and still be contention-bound
    /// (`c256x2048`). Free: `lock_slow` is `#[cold] #[inline(never)]` and is
    /// about to take a blocking lock anyway.
    pub static N_LOCKSLOW: AtomicU64 = AtomicU64::new(0);
    /// Total spin-loop iterations across all `lock_slow` entries, accumulated
    /// ONCE per entry rather than per iteration, so the counter costs one
    /// relaxed RMW per wait and not one per spin. `N_SPINS / N_LOCKSLOW` is the
    /// mean depth of a wait, which separates "many short waits" (a granularity
    /// problem) from "few long ones" (a scheduling problem).
    pub static N_SPINS: AtomicU64 = AtomicU64::new(0);
    /// Total registrations — NOT counted. One shared `fetch_add` per `add`, at
    /// 136 M adds per 4K frame from eight threads, serialises the decoder hard
    /// enough that slot pressure disappears and `WIDE_FULL` reads zero for the
    /// wrong reason. The counters that remain fire 10^4-10^5 times per frame,
    /// which is free. Kept as a field so the report's shape does not change.
    pub static N_ADD: AtomicU64 = AtomicU64::new(0);
    /// Strided-RECTANGLE registrations that were ACCEPTED
    /// ([`super::BorrowTracker::add_rect`]). This is the liveness proof for the
    /// rectangle path: a timed arm whose `n_rect` is 0 measured nothing.
    pub static N_RECT: AtomicU64 = AtomicU64::new(0);
    /// Rectangle registrations DECLINED — unrepresentable geometry, a >
    /// `MAX_SHARDS_PER_BORROW`-block hull, a full shard, or a live wide record.
    /// Every one of these fell back to the caller's per-row path, so
    /// `n_rect_declined` is the count of `fill`s the mechanism did not reach.
    pub static N_RECT_DECLINED: AtomicU64 = AtomicU64::new(0);
    /// Accepted rectangles whose hull spanned MORE THAN ONE shard. Those pay a
    /// multi-shard `add` (n locks) AND a `remove_multi` (n locks again) against a
    /// per-row registration's ONE `try_lock` and ONE lock-free store, so this
    /// column is what decides whether the record COUNT or the LOCK TRAFFIC is
    /// what a rectangle actually changes.
    pub static N_RECT_MULTI: AtomicU64 = AtomicU64::new(0);

    pub fn report() -> std::string::String {
        use core::fmt::Write as _;
        let mut out = std::string::String::new();
        let w = WIDE_SHARDS.load(Relaxed) + WIDE_BLOCKS.load(Relaxed) + WIDE_FULL.load(Relaxed);
        // Absolute counts only. There is deliberately no denominator: see
        // `N_ADD`. `const_shift` is the compile-time constant and is NOT what
        // an `__blockshift_adaptive` build uses — that one is per instance.
        let _ = writeln!(
            out,
            "WIDEHDR\tconst_shift\tslow\tmulti\tw_shards\tw_blocks\tw_full\twide_total\tcontended\tlockslow\tspins\tn_rect\tn_rect_declined\tn_rect_multi"
        );
        let _ = writeln!(
            out,
            "WIDE\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            super::BLOCK_SHIFT,
            N_SLOW.load(Relaxed),
            N_MULTI.load(Relaxed),
            WIDE_SHARDS.load(Relaxed),
            WIDE_BLOCKS.load(Relaxed),
            WIDE_FULL.load(Relaxed),
            w,
            N_CONTENDED.load(Relaxed),
            N_LOCKSLOW.load(Relaxed),
            N_SPINS.load(Relaxed),
            N_RECT.load(Relaxed),
            N_RECT_DECLINED.load(Relaxed),
            N_RECT_MULTI.load(Relaxed),
        );
        out
    }

    pub fn reset() {
        for a in [
            &WIDE_SHARDS,
            &WIDE_BLOCKS,
            &WIDE_FULL,
            &N_SLOW,
            &N_MULTI,
            &N_ADD,
            &N_CONTENDED,
            &N_LOCKSLOW,
            &N_SPINS,
        ] {
            a.store(0, Relaxed);
        }
    }
}

// =============================================================================
// Lock
// =============================================================================

/// Lightweight spin lock, one per shard.
///
/// `swap` rather than `compare_exchange`: an unconditional store with no branch
/// on the old value, which is what the uncontended path wants. **Not
/// reentrant** — every multi-shard operation in this module depends on that
/// being remembered, hence the ascending acquisition order.
///
/// # The waiting policy is an A/B axis, and it is NOT settled
///
/// `docs/AGENT_BRIEF.md` §6 records "TinyLock backoff: null, measured twice",
/// and both of those measurements were taken where contention is ~0.02% of
/// registrations. On `c256x2048` at t=8 the same lock spends **1.136 CPU
/// ms/frame** in [`Self::lock_slow`], which is a different regime, so the arms
/// below re-open the question THERE rather than overwrite the earlier null.
/// See `docs/C256_CONTENTION.md`.
///
/// | feature | waiting policy |
/// |---|---|
/// | (default) | pure relaxed-load spin, never yields |
/// | `__probe_lock_backoff` | spin 64, then `yield_now`, repeat |
/// | `__probe_lock_yield` | `yield_now` on every iteration |
/// | `__probe_lock_relax` | exponential pause BETWEEN loads, never yields |
/// | `__probe_lock_park` | `parking_lot::RawMutex` — spins, then genuinely parks |
///
/// `__probe_lock_relax` is the only one of the four that changes how often a
/// waiter TOUCHES the line rather than what it does between touches, and it
/// exists because a spin iteration here was measured at **~627 ns** against
/// 7.6 ns for `spin_loop()` on an idle core — the cost is the relaxed load
/// pulling a line the holder is hammering, so a waiter that reads less often
/// may let the holder finish sooner.
#[cfg(not(feature = "__probe_lock_park"))]
struct TinyLock(AtomicBool);

#[cfg(not(feature = "__probe_lock_park"))]
impl TinyLock {
    const fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    #[inline(always)]
    fn lock(&self) {
        if self.0.swap(true, Ordering::Acquire) {
            self.lock_slow();
        }
    }

    /// Acquire without the blocking retry. `false` means somebody else holds
    /// it and NOTHING was acquired.
    ///
    /// Exists so the single-block fast path can contain no `bl` at all: the
    /// contended case tail-calls a cold function that re-takes the lock the
    /// blocking way. A `bl` anywhere in that path forces LLVM to spill
    /// callee-saved registers around it, which is a 0x80-byte stack frame plus
    /// three `stp`/`ldp` pairs on EVERY registration — see
    /// [`BorrowTracker::add_contended`].
    #[inline(always)]
    fn try_lock(&self) -> bool {
        !self.0.swap(true, Ordering::Acquire)
    }

    #[cold]
    #[inline(never)]
    fn lock_slow(&self) {
        #[cfg(feature = "__probe_lock_backoff")]
        let mut spins = 0u32;
        #[cfg(feature = "__probe_lock_relax")]
        let mut pause = 1u32;
        // Total spin iterations for THIS wait, published once at the end so the
        // counter costs one relaxed RMW per wait rather than one per spin.
        #[cfg(feature = "__probe_wide")]
        let mut total = 0u64;
        loop {
            // Spin on a load, not a swap: a read-only spin keeps the line in
            // Shared instead of ping-ponging it Exclusive between waiters.
            while self.0.load(Ordering::Relaxed) {
                #[cfg(feature = "__probe_wide")]
                {
                    total += 1;
                }
                #[cfg(not(any(feature = "__probe_lock_yield", feature = "__probe_lock_relax")))]
                core::hint::spin_loop();
                #[cfg(feature = "__probe_lock_yield")]
                std::thread::yield_now();
                #[cfg(feature = "__probe_lock_relax")]
                {
                    // Pause `pause` times BETWEEN loads, doubling up to a cap,
                    // so a waiter stops pulling the line away from the holder.
                    for _ in 0..pause {
                        core::hint::spin_loop();
                    }
                    pause = (pause * 2).min(64);
                }
                #[cfg(feature = "__probe_lock_backoff")]
                {
                    spins += 1;
                    if spins >= 64 {
                        spins = 0;
                        std::thread::yield_now();
                    }
                }
            }
            if !self.0.swap(true, Ordering::Acquire) {
                #[cfg(feature = "__probe_wide")]
                {
                    wide_probe::N_LOCKSLOW.fetch_add(1, Ordering::Relaxed);
                    wide_probe::N_SPINS.fetch_add(total, Ordering::Relaxed);
                }
                return;
            }
        }
    }

    #[inline(always)]
    fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}

/// THROWAWAY arm (`__probe_lock_park`): the same one-byte shard lock, but a
/// waiter PARKS instead of spinning.
///
/// `parking_lot::RawMutex` is an `AtomicU8`, so this is size- and
/// layout-neutral against [`TinyLock`]'s `AtomicBool` — the arm changes the
/// waiting policy and nothing else, which is what makes it a clean A/B against
/// a pure spin. It already does a bounded adaptive spin before parking, so it
/// is the "spin then really sleep" end of the ladder that `__probe_lock_yield`
/// (deschedule immediately) and `__probe_lock_backoff` (spin 64, then yield)
/// bracket.
///
/// Measurement only; absent from `default` and from every published feature.
#[cfg(feature = "__probe_lock_park")]
struct TinyLock(parking_lot::RawMutex);

#[cfg(feature = "__probe_lock_park")]
impl TinyLock {
    const fn new() -> Self {
        Self(<parking_lot::RawMutex as parking_lot::lock_api::RawMutex>::INIT)
    }

    #[inline(always)]
    fn lock(&self) {
        use parking_lot::lock_api::RawMutex as _;
        #[cfg(feature = "__probe_wide")]
        if self.0.is_locked() {
            wide_probe::N_LOCKSLOW.fetch_add(1, Ordering::Relaxed);
        }
        self.0.lock();
    }

    #[inline(always)]
    fn try_lock(&self) -> bool {
        use parking_lot::lock_api::RawMutex as _;
        self.0.try_lock()
    }

    #[inline(always)]
    fn unlock(&self) {
        use parking_lot::lock_api::RawMutex as _;
        // SAFETY: every caller reached here through `lock`/`try_lock` returning
        // success on this same lock and has not unlocked it since — the same
        // obligation the spin implementation's `store(false)` carries, made
        // explicit by `RawMutex`'s signature.
        unsafe { self.0.unlock() }
    }
}

/// RAII guard for a single shard. Multi-shard sections unlock by hand (in the
/// reverse of acquisition), because the set is dynamic.
struct ShardGuard<'a>(&'a TinyLock);

impl Drop for ShardGuard<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.0.unlock();
    }
}

// =============================================================================
// Records
// =============================================================================

/// An existing-borrow record as reported by the overlap finders:
/// `(start, end, mutable, registration_site)`.
type OverlapHit = (usize, usize, bool, Option<&'static Location<'static>>);

/// The records of one shard. Only ever touched while that shard's [`TinyLock`]
/// is held.
///
/// Liveness is NOT here — it lives in [`Shard::live`], one atomic byte per
/// slot, because releasing a borrow has to be doable without the lock. See
/// [`BorrowTracker::remove`] and [`Shard::live`].
struct ShardRecs {
    /// SUPERSET of the live slots: bit `i` clear ⇒ slot `i` is provably dead.
    ///
    /// The scan needs a *bitmap* to stay one branch wide in the common case,
    /// but liveness itself has to be per-slot and atomic so release can be a
    /// plain store. This reconciles the two: it is written only by lock
    /// holders (so it is a plain field, not an atomic), and it is only ever
    /// grown by a publish and shrunk by [`Shard::live_mask`] refreshing it
    /// against the real flags. A stale-LARGE value costs one extra flag load;
    /// a stale-small one would be unsound, and cannot happen because a slot
    /// becomes live only through a publish that sets its bit here first.
    allocated: u8,
    /// Two per-slot bitmaps in one word: **low** byte bit `i` set iff slot `i`'s
    /// record is a mutable borrow, **high** byte bit `i` set iff slot `i`'s
    /// record is a strided RECTANGLE rather than a plain interval (see
    /// [`RECT_SHIFT`] and [`BorrowTracker::add_rect`]).
    ///
    /// Only meaningful for slots that [`Shard::live_mask`] reports live, and
    /// only ever read or written by a lock holder.
    ///
    /// ONE `u16` and not two `u8`s. [`Self::alloc`]'s empty-shard arm is the
    /// measured steady state (mean occupancy 0.02, `occ_max == 1`) and it
    /// publishes the whole word with a SINGLE store, exactly as it did when
    /// `mutable` stood alone; [`Self::find`] loads it once. Two adjacent bytes
    /// would have put a second store on the dependency chain between the lock
    /// acquire and the record write, which is the only kind of work this path
    /// has been measured to be sensitive to (a single extra load there measured
    /// +0.8%, `benchmarks/tracker_borrowcost_2026-08-08.tsv`).
    flags: u16,
    /// The rectangle case reads the row stride out of the tracker, so a record
    /// stays two words: `starts[i]`/`ends[i]` hold the rectangle's **hull**,
    /// which is what shard selection needs anyway, and `(rows, seg)` is
    /// recovered from the hull and the stride by [`rect_decode`]. That is an
    /// exact bijection, so nothing about the footprint is approximated.
    starts: [usize; SLOTS],
    ends: [usize; SLOTS],
    /// Registration site, for the overlap panic message.
    ///
    /// Debug only. In release the wrapper methods do not propagate
    /// `#[track_caller]` to the caller, so this used to record the *wrapper's*
    /// own line — no diagnostic value for one store per borrow, on the hottest
    /// path in the decoder. Debug and test builds keep the real site.
    #[cfg(debug_assertions)]
    locs: [Option<&'static Location<'static>>; SLOTS],
}

impl ShardRecs {
    const fn new() -> Self {
        Self {
            allocated: 0,
            flags: 0,
            starts: [0; SLOTS],
            ends: [0; SLOTS],
            #[cfg(debug_assertions)]
            locs: [None; SLOTS],
        }
    }

    /// `start`/`end` are passed in rather than read from the record: for a
    /// rectangle record the interesting extent is the ROW SEGMENT that actually
    /// collided, not the hull, and the panic message must name the real one.
    #[inline(always)]
    fn hit(&self, i: usize, start: usize, end: usize) -> OverlapHit {
        // `min` rather than a bounds check: it is a `umin`, and it lets LLVM
        // drop the panic path from the hottest loop in the decoder.
        let i = i.min(SLOTS - 1);
        (
            start,
            end,
            self.flags & (1 << i) != 0,
            #[cfg(debug_assertions)]
            self.locs[i],
            #[cfg(not(debug_assertions))]
            None,
        )
    }

    /// First live record overlapping `[start, end)`.
    ///
    /// `IS_MUT` is the *registrant's* mutability: a mutable borrow conflicts
    /// with every live record, an immutable one only with mutable records. A
    /// const parameter, so each call site compiles to one loop with the mask
    /// folded in — the way the legacy tracker's two separate `find_overlap_*`
    /// functions did.
    ///
    /// `occupied` is passed in because liveness now lives outside the lock, in
    /// [`Shard::live`]; the caller derives it once with [`Shard::live_mask`]
    /// and uses the same snapshot for the scan and for [`Self::alloc`].
    /// `row_stride` is only read when a RECTANGLE record's hull overlapped, i.e.
    /// essentially never: the caller passes the tracker's field and LLVM sinks
    /// the load into that branch.
    #[inline(always)]
    fn find<const IS_MUT: bool>(
        &self,
        occupied: u8,
        start: usize,
        end: usize,
        row_stride: usize,
    ) -> Option<OverlapHit> {
        // TRIED AND REVERTED: `if occupied == 0 { return None }` to skip the
        // `mutable` load on an empty shard. +2.3% at 8bpc t=1 (335.1 -> 342.7,
        // median of 9, idle box). The load is free — same cache line, issued in
        // parallel — and LLVM already folds the test into ONE `ands`+`b.eq`;
        // the early-out just adds a second branch with the same outcome.
        // `benchmarks/tracker_borrowcost_2026-08-08.tsv`, arm `findeo`.
        let flags = self.flags;
        let mut mask = if IS_MUT {
            occupied
        } else {
            occupied & (flags as u8)
        };
        let rect = (flags >> RECT_SHIFT) as u8;
        while mask != 0 {
            let i = (mask.trailing_zeros() as usize).min(SLOTS - 1);
            if self.starts[i] < end && start < self.ends[i] {
                if rect & (1 << i) == 0 {
                    return Some(self.hit(i, self.starts[i], self.ends[i]));
                }
                // COLD, and only reachable once a rectangle record is live in
                // this shard: the test above was against that record's HULL, so
                // it is a prefilter and not the answer. The exact test walks the
                // rows the probe range can reach and reports the row segment
                // that actually collided.
                if let Some((rs, re)) =
                    rect_hit_range(self.starts[i], self.ends[i], row_stride, start, end)
                {
                    return Some(self.hit(i, rs, re));
                }
            }
            mask &= mask - 1;
        }
        None
    }

    /// [`Self::find`] with a strided RECTANGLE as the probe instead of an
    /// interval: `[h0, h1)` is the registrant's hull and `row_stride` the
    /// instance's declared stride, so [`rect_decode`] recovers `(rows, seg)`.
    ///
    /// Cold by construction — one call per rectangle registration, against a
    /// shard whose measured occupancy is 0.02.
    #[inline]
    fn find_from_rect<const IS_MUT: bool>(
        &self,
        occupied: u8,
        h0: usize,
        h1: usize,
        row_stride: usize,
    ) -> Option<OverlapHit> {
        let flags = self.flags;
        let mut mask = if IS_MUT {
            occupied
        } else {
            occupied & (flags as u8)
        };
        let rect = (flags >> RECT_SHIFT) as u8;
        let (rows, seg) = rect_decode(h0, h1, row_stride);
        while mask != 0 {
            let i = (mask.trailing_zeros() as usize).min(SLOTS - 1);
            let (es, ee) = (self.starts[i], self.ends[i]);
            // Hull-vs-hull prefilter, exactly as in `find`.
            if es < h1 && h0 < ee {
                if rect & (1 << i) == 0 {
                    // Stored record is a plain interval: ask whether OUR rows
                    // reach it.
                    if let Some((rs, re)) = rect_hit_range(h0, h1, row_stride, es, ee) {
                        // Report the stored record's extent, clipped to nothing
                        // — it is the counterparty and its extent is exact.
                        let _ = (rs, re);
                        return Some(self.hit(i, es, ee));
                    }
                } else {
                    // Both sides are rectangles on the same stride. Compare row
                    // by row: `rows` and the counterparty's row count are both
                    // capped at `MAX_RECT_ROWS`, so this is bounded, and it is
                    // the honest test — no common-grid assumption is made.
                    for r in 0..rows {
                        let rs = h0 + r * row_stride;
                        if let Some((cs, ce)) = rect_hit_range(es, ee, row_stride, rs, rs + seg) {
                            return Some(self.hit(i, cs, ce));
                        }
                    }
                }
            }
            mask &= mask - 1;
        }
        None
    }

    /// Claim a slot for `[start, end)`, given the `occupied` snapshot the
    /// caller already loaded. `None` when the shard is full.
    ///
    /// Returns the slot index; PUBLISHING it (setting the occupancy bit) is the
    /// caller's job and must happen *after* this returns, so that the record
    /// fields are complete before any other thread can observe the bit.
    #[inline(always)]
    fn alloc<const IS_MUT: bool, const IS_RECT: bool>(
        &mut self,
        occupied: u8,
        start: usize,
        end: usize,
        loc: Loc,
    ) -> Option<u8> {
        // Empty shard: slot 0, with no `rbit`/`clz` and constant store offsets.
        // Measured mean occupancy is 0.02 and measured max is 1, so this is the
        // case essentially always, and `trailing_ones` sits on the dependency
        // chain between the lock acquire and the record write — the only kind
        // of work this path is sensitive to.
        if occupied == 0 {
            self.starts[0] = start;
            self.ends[0] = end;
            // Whole-WORD STORE, not `|= 1` / `&= !1`: with no live slot, no
            // other slot's mutability or rectangle bit is meaningful (`find`
            // masks with the live set), so there is nothing to preserve — and
            // that turns a load-or-store on the dependency chain into a single
            // store. One `u16` store, the same instruction count the `u8`
            // `mutable` field cost before the rectangle bitmap joined it.
            self.flags = IS_MUT as u16 | ((IS_RECT as u16) << RECT_SHIFT);
            #[cfg(debug_assertions)]
            {
                self.locs[0] = Some(loc);
            }
            return Some(0);
        }
        // `!SLOTS_MASK` pre-fills the unusable high bits, so `trailing_ones`
        // reaches SLOTS exactly when the shard is full.
        let free = ((occupied | !SLOTS_MASK).trailing_ones() as usize).min(SLOTS);
        if free == SLOTS {
            return None;
        }
        self.starts[free] = start;
        self.ends[free] = end;
        if IS_MUT {
            self.flags |= 1 << free;
        } else {
            self.flags &= !(1u16 << free);
        }
        if IS_RECT {
            self.flags |= 1 << (RECT_SHIFT + free as u32);
        } else {
            self.flags &= !(1u16 << (RECT_SHIFT + free as u32));
        }
        #[cfg(debug_assertions)]
        {
            self.locs[free] = Some(loc);
        }
        #[cfg(not(debug_assertions))]
        let _ = loc;
        Some(free as u8)
    }
}

/// One shard: its lock, its per-slot live flags, and its records, alone on a
/// cache line.
///
/// 128 bytes is the M-series line size (`hw.cachelinesize`). Two shards sharing
/// a line would halve the effective shard count, so the alignment is
/// load-bearing, not decorative.
///
/// **On x86 the line is 64 bytes (`clflush size: 64`), so read this as two
/// separate claims.** No false sharing BETWEEN shards: a 128-byte-aligned
/// 128-byte object occupies whole lines on either machine, which is what the
/// alignment is for. But the steady-state fast path spans BOTH x86 lines —
/// offsets are `lock` 0, `live` 1..8, `allocated`/`mutable` 8..10,
/// `starts[0..7]` 16..72, `ends[0..7]` 72..128, and measured occupancy is 0-1,
/// so the hot path reads `starts[0]` in line 0 and `ends[0]` at offset **72**
/// in line 1. Two pure-layout refits would fix that — records as
/// `[(usize, usize); SLOTS]` pairs (slot 0 at 16..32), or `SLOTS` 7 -> 3
/// (`1 + 3 + 8 + 48 = 60`, a one-line shard). NEITHER IS DONE AND NO SPEEDUP IS
/// CLAIMED: the only x86 host available to this campaign is QEMU-TCG, which has
/// no cache model. `SLOTS = 3` also raises the shard-full rate and pushes
/// borrows onto the wide path, which IS measurable without a timer
/// (`--features __probe_wide`) and should be checked first.
/// See `docs/X64_APPLICABILITY.md` H1.
#[repr(align(128))]
struct Shard {
    lock: TinyLock,
    /// Slot `i` holds a live record iff `live[i]` is nonzero.
    ///
    /// ONE ATOMIC BYTE PER SLOT, not one shared bitmap word, and that is the
    /// whole point: a per-slot flag has **at most one writer at a time**, so
    /// both publish and release are plain `store`s instead of the `fetch_or` /
    /// `fetch_and` a shared bitmap forces. Only two parties ever write
    /// `live[i]`:
    ///
    /// * the allocator, which holds the lock AND has just observed `live[i]`
    ///   zero, so no other allocator can pick `i` and the previous owner's
    ///   `store(0)` is already globally visible (single-location coherence);
    /// * that borrow's own owner, exactly once, on release.
    ///
    /// They cannot overlap, so there is no update to lose and no RMW to pay.
    /// Measured on an M4 Pro (`examples/probe_borrow_cost`, median of 9): the
    /// shipped `swap` + `fetch_or` + `fetch_and` triple costs 3.68 ns per
    /// acquire/release pair against 1.57 ns for `swap` + plain stores — 2.10 ns
    /// of the tracker's 6.67 ns per-pair cost.
    ///
    /// Ordering: publish is `Release` so a lock holder that loads the flag
    /// `Acquire` sees the record fields behind it; release is `Release` so the
    /// borrower's writes THROUGH the reference are ordered before the slot can
    /// be handed to anyone else. That is the same pairing the bitmap's
    /// `fetch_or(Release)` / `fetch_and(Release)` / `load(Acquire)` gave.
    ///
    /// Scanning wants a bitmap, though — hence [`ShardRecs::allocated`], a
    /// lock-protected superset that keeps the common case to one flag load.
    live: [AtomicU8; SLOTS],
    recs: UnsafeCell<ShardRecs>,
}

impl Shard {
    const fn new() -> Self {
        Self {
            lock: TinyLock::new(),
            live: [const { AtomicU8::new(0) }; SLOTS],
            recs: UnsafeCell::new(ShardRecs::new()),
        }
    }

    /// The live-slot bitmap, refreshed from the per-slot flags.
    ///
    /// `allocated` is [`ShardRecs::allocated`], a superset; this narrows it to
    /// the slots that are actually live. The caller must hold this shard's lock
    /// (so no slot can *become* live underneath it) and should store the result
    /// back into `allocated`, which is what keeps the superset from saturating.
    #[inline(always)]
    fn live_mask(&self, allocated: u8) -> u8 {
        // `allocated <= 1` is the measured steady state and it is straight-line:
        // one load, no `rbit`/`clz`, no loop. `probe-count` reports occ_max == 1
        // on every hot plane at t=1 and mean occupancy 0.02, and the allocator
        // always takes the lowest free slot — so after the first borrow on a
        // shard, `allocated` is 1 and stays 1. This matters because the whole
        // of `live_mask` sits on the dependency chain between the lock acquire
        // and the record write, which is the only kind of work this path has
        // been measured to be sensitive to (see the .meta).
        //
        // `live[i]` stores 1, and slot 0's bit IS 1, so the byte is already the
        // mask for this case.
        //
        // `allocated == 0` needs no separate branch: nothing was ever published
        // in this shard, so `live[0]` reads 0 and the answer is the same.
        if allocated <= 1 {
            // Acquire pairs with the retiring owner's `Release` store, so a
            // slot observed dead carries that borrow's writes with it.
            return self.live[0].load(Ordering::Acquire);
        }
        let mut m = allocated & SLOTS_MASK;
        let mut live = 0u8;
        while m != 0 {
            let i = (m.trailing_zeros() as usize).min(SLOTS - 1);
            if self.live[i].load(Ordering::Acquire) != 0 {
                live |= 1 << i;
            }
            m &= m - 1;
        }
        live
    }

    /// Publish slot `slot` as live. Must be called by the lock holder, after
    /// [`ShardRecs::alloc`] has filled the record in and after `allocated` has
    /// gained the slot's bit.
    #[inline(always)]
    fn publish(&self, slot: u8) {
        self.live[(slot as usize).min(SLOTS - 1)].store(1, Ordering::Release);
    }

    /// Retire slot `slot`. Lock-free — see [`Self::live`].
    #[inline(always)]
    fn retire(&self, slot: u8) {
        let i = (slot as usize).min(SLOTS - 1);
        debug_assert!(
            self.live[i].load(Ordering::Relaxed) != 0,
            "freeing an unoccupied shard slot"
        );
        self.live[i].store(0, Ordering::Release);
    }
}

const _: () = assert!(
    core::mem::size_of::<Shard>() == 128 || cfg!(debug_assertions),
    "Shard must be exactly one cache line in release builds"
);

// =============================================================================
// BorrowId
// =============================================================================

/// Handle returned by a registration, used to release it.
///
/// A borrow can hold a slot in up to [`MAX_SHARDS_PER_BORROW`] shards, so this
/// is no longer a bare slot index — but it stays a single register-sized word,
/// because it is created and destroyed ~50 million times per 4K frame and it
/// travels inside every guard.
///
/// ```text
///   bits  0..2                 kind
///   bits  2..2+N_BITS          narrow: number of (shard, slot) pairs, minus one
///   bits  PAIR_SHIFT..         narrow: MAX_SHARDS_PER_BORROW (slot:3, shard) pairs
///   bits  PAIR_SHIFT..+16      wide:   index into the wide list
/// ```
///
/// `PAIR_BITS` is `3 + log2(N_SHARDS)` — exactly what the shard index needs, not
/// a round number — because the whole word must hold
/// `MAX_SHARDS_PER_BORROW` pairs plus the kind and the count in 64 bits. At the
/// default 128 shards that is 10 bits per pair, so five pairs cost 55 bits.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct BorrowId(u64);

const KIND_EMPTY: u64 = 0;
const KIND_NARROW: u64 = 1;
const KIND_WIDE: u64 = 2;
const KIND_UNCHECKED: u64 = 3;

const KIND_BITS: u32 = 2;
const KIND_MASK: u64 = (1 << KIND_BITS) - 1;
/// Bits holding `pairs - 1`, i.e. `ceil(log2(MAX_SHARDS_PER_BORROW))`.
///
/// Written with `leading_zeros` rather than `ilog2` on purpose: `ilog2` panics on
/// zero, so a cap of 1 would fail const evaluation with a message about a
/// logarithm instead of about the cap. This form yields 0 there, which is
/// correct — with one possible pair the count field is empty.
const N_BITS: u32 = usize::BITS - (MAX_SHARDS_PER_BORROW - 1).leading_zeros();
const N_MASK: u64 = (1 << N_BITS) - 1;
const PAIR_SHIFT: u32 = KIND_BITS + N_BITS;
/// 3 bits of slot + exactly as many shard bits as `N_SHARDS` needs.
const SHARD_ID_BITS: u32 = N_SHARDS.trailing_zeros();
const PAIR_BITS: u32 = 3 + SHARD_ID_BITS;
const SLOT_MASK: u64 = 0b111;
const SHARD_MASK_BITS: u64 = (1u64 << SHARD_ID_BITS) - 1;

const _: () = assert!(SLOTS <= (SLOT_MASK as usize) + 1);
const _: () = assert!(N_SHARDS <= (SHARD_MASK_BITS as usize) + 1);
const _: () = assert!(N_SHARDS.is_power_of_two());
const _: () = assert!(SHARDS_SERIAL.is_power_of_two() && SHARDS_SERIAL <= N_SHARDS);
/// The whole reason [`MAX_SHARDS_PER_BORROW`] cannot just be raised: the id is
/// ONE word and every pair has to fit in it beside the kind and the count.
const _: () = assert!(
    (KIND_BITS + N_BITS + PAIR_BITS * MAX_SHARDS_PER_BORROW as u32) as usize <= u64::BITS as usize
);
const _: () = assert!(MAX_SHARDS_PER_BORROW >= 1);

impl BorrowId {
    pub const UNCHECKED: Self = Self(KIND_UNCHECKED);

    const EMPTY: Self = Self(KIND_EMPTY);

    #[inline(always)]
    const fn narrow1(shard: usize, slot: u8) -> Self {
        Self(KIND_NARROW | ((((shard as u64) << 3) | (slot as u64 & SLOT_MASK)) << PAIR_SHIFT))
    }

    #[inline(always)]
    const fn wide(idx: u16) -> Self {
        Self(KIND_WIDE | ((idx as u64) << PAIR_SHIFT))
    }

    #[inline(always)]
    fn kind(self) -> u64 {
        self.0 & KIND_MASK
    }

    /// Number of `(shard, slot)` pairs. Only meaningful for `KIND_NARROW`.
    #[inline(always)]
    fn pairs(self) -> usize {
        (((self.0 >> KIND_BITS) & N_MASK) as usize) + 1
    }

    /// `(shard, slot)` of pair `i`. Both are masked, so the caller can index
    /// the shard array and shift by the slot without a bounds or overflow
    /// check.
    #[inline(always)]
    fn pair(self, i: usize, mask: usize) -> (usize, u8) {
        let f = (self.0 >> (PAIR_SHIFT + PAIR_BITS * i as u32)) & ((1 << PAIR_BITS) - 1);
        // Two masks: the runtime one restores what `shard_of` produced, the
        // constant one is what lets LLVM index `[Shard; N_SHARDS]` without a
        // bounds check.
        (
            ((f >> 3) as usize) & mask & (N_SHARDS - 1),
            (f & SLOT_MASK) as u8,
        )
    }

    #[inline(always)]
    fn wide_idx(self) -> usize {
        ((self.0 >> PAIR_SHIFT) & 0xFFFF) as usize
    }

    fn from_pairs(shards: &[u16], slots: &[u8]) -> Self {
        debug_assert!(!shards.is_empty() && shards.len() <= MAX_SHARDS_PER_BORROW);
        let mut v = KIND_NARROW | (((shards.len() - 1) as u64) << KIND_BITS);
        for i in 0..shards.len() {
            let f = ((shards[i] as u64) << 3) | (slots[i] as u64 & SLOT_MASK);
            v |= f << (PAIR_SHIFT + PAIR_BITS * i as u32);
        }
        Self(v)
    }
}

impl Default for BorrowId {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Debug for BorrowId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.kind() {
            KIND_EMPTY => write!(f, "BorrowId(empty)"),
            KIND_UNCHECKED => write!(f, "BorrowId(unchecked)"),
            KIND_WIDE => write!(f, "BorrowId(wide #{})", self.wide_idx()),
            _ => {
                write!(f, "BorrowId(")?;
                for i in 0..self.pairs() {
                    let (sh, sl) = self.pair(i, usize::MAX);
                    write!(f, "{}s{sh}/{sl}", if i == 0 { "" } else { " " })?;
                }
                write!(f, ")")
            }
        }
    }
}

// =============================================================================
// Tracker
// =============================================================================

/// A wide record: the borrow's exact interval, its mutability, and its site.
/// `start >= end` marks a tombstone.
type WideRec = (usize, usize, bool, Option<&'static Location<'static>>);

/// All active borrows for one `DisjointMut` instance.
///
/// Poisoning matches `std::sync::Mutex`: a thread that panics while holding a
/// mutable guard leaves the region possibly half-written, so every later borrow
/// fails.
pub(super) struct BorrowTracker {
    shards: [Shard; N_SHARDS],
    /// `log2` of this instance's block size — see [`block_shift_for`]. Read
    /// once per registration off the same line as `mask`.
    shift: u32,
    /// Which of the shards this instance actually uses: `N_SHARDS - 1` for a
    /// buffer big enough to be worth spreading, `0` for a small one, whose
    /// borrows then all land on shard 0 and stay on one cache line.
    ///
    /// The whole tracker lives behind a `Box` (see [`DisjointMut`]'s field) —
    /// inline it is 4 KiB, and `Rav1dTaskContext` embeds ~20 `DisjointMut`s,
    /// which pushed it from under 48 KiB to 97 KiB and tripped its
    /// stack-weight gate. Boxing the tracker rather than the shard array keeps
    /// the array a fixed-size field, so a masked index needs no fat-pointer
    /// load and no bounds check.
    mask: usize,
    /// This instance's picture row stride in BYTES, or 0 when none was declared.
    ///
    /// Written only through [`Self::set_row_stride`] / [`Self::reprovision`],
    /// both `&mut self`, so it is fixed for as long as any record can exist —
    /// which is what lets a rectangle record's `(rows, seg)` be *derived* from
    /// its hull instead of stored (see [`rect_decode`]). Both registrants of a
    /// shared byte read the same value, exactly as they do for `shift` and
    /// `mask`, and it lives on the same line as those.
    row_stride: usize,
    /// THROWAWAY (`__probe_tinynop`): this instance is shorter than
    /// [`SHARD_MIN_LEN`]. Set once in [`Self::new`]/[`Self::reprovision`] off
    /// the same line as `mask`, and read only to SKIP tracking entirely.
    /// UNSOUND — measurement only. See [`Self::add`]'s probe arm.
    #[cfg(feature = "__probe_tinynop")]
    tiny: bool,
    /// Live wide records. Read while holding **any** shard lock; written only
    /// while holding **every** shard lock.
    wide: UnsafeCell<Vec<WideRec>>,
    /// Poison flag (bit 31) and live wide-record count (bits 0..31), in one
    /// word so the hot path tests both with a single load and one branch:
    /// `state == 0` means "not poisoned, no wide records", which is the case
    /// essentially always.
    state: AtomicU32,
}

const POISON_BIT: u32 = 1 << 31;

/// THROWAWAY (`__probe_bounds`): the shard geometry a counterfactual extent
/// must be priced against — this instance's REAL block shift and shard mask,
/// plus the two promotion limits.
///
/// Mirroring these as constants in the probe is wrong and was caught being
/// wrong: the shipped shift is `block_shift_for(len)`, which is 12 for a serial
/// or single-tile decode and 14-15 for a multi-tile 4K plane, so a hull that
/// spans 15 blocks in one configuration spans 3 in another.
#[cfg(feature = "__probe_bounds")]
impl BorrowTracker {
    pub(super) fn probe_geometry(&self) -> crate::bounds_probe::ShardGeom {
        crate::bounds_probe::ShardGeom {
            shift: self.shift,
            mask: self.mask,
            max_shards: MAX_SHARDS_PER_BORROW,
            max_blocks: MAX_BLOCKS_SCAN,
        }
    }

    /// [`shard_of`], for the probe. Same function the tracker registers with.
    pub(super) fn probe_shard_of(block: usize, mask: usize) -> usize {
        shard_of(block, mask)
    }
}

// SAFETY: `wide` and every `Shard::recs` are only accessed under the relevant
// `TinyLock`(s), per the module-level rules.
unsafe impl Send for BorrowTracker {}
unsafe impl Sync for BorrowTracker {}

/// Blocks per shard the adaptive shift aims for.
///
/// The whole point of a bigger block is that a STRIDED access — a `w x h`
/// compact read, or worse `rav1d_prepare_intra_edges`' 1-pixel-wide left
/// column — stops paying one distinct shard line per row. That argues for as
/// few blocks as possible. The opposite pull is that a shard is the unit of
/// mutual exclusion, so collapsing a whole plane onto a handful of shards puts
/// every concurrent tile worker on the same lock.
///
/// Aiming for a fixed ratio of blocks to shards balances the two AT EVERY
/// BUFFER SIZE, which a single global constant cannot: the shift that makes a
/// 4K plane's rows share a block turns a 64 KiB buffer into ONE block and one
/// lock. See [`block_shift_for`].
///
/// 2 is chosen so that the quantity the mechanism actually cares about —
/// PICTURE ROWS PER BLOCK — comes out the same at every bit depth. A 10-bit
/// plane is twice the bytes AND twice the stride of the 8-bit one, so a rule
/// keyed on `len` alone tracks the stride for free: both land on ~4.3 rows per
/// block. On the fixed ladder that is shift 14 for the 8-bit 4K plane (its
/// joint-best) and 15 for the 10-bit one (within 1.2% of its best). A ratio of
/// 1 would give ~8.5 rows and shift 15/16 — a hair better on 10-bit, at half
/// the shard utilisation on every small buffer, which is the wrong trade for a
/// rule that has to hold at all sizes.
///
/// **Re-opened 2026-08-10 on a different OBJECTIVE, and that is the point of
/// the `__bps_*` ladder below.** The value 2 was fitted against WHOLE-FRAME
/// WALL on `v4k_8tile`. The tiled t=8 attribution
/// (`docs/TILED_SCALING.md` §4, §7 item 1) then found that 42.2% / 32.4% of the
/// t=8 wall gap is IDLE CORES, most of it in the post-tile filter TAIL — and a
/// shift that minimises mean wall can be wrong exactly where the cores are
/// idle, because the tail's contention is between ADJACENT SUPERBLOCK ROWS
/// filtering at once and the block size decides whether those land on the same
/// shard. So the ladder is re-swept scored on tail concurrency as well as on
/// wall. `benchmarks/shard_granularity_2026-08-10.*`.
///
/// The ratio is a rational so the ladder can go BELOW one block per shard
/// (coarser blocks than the default), which a `usize` count cannot express.
/// `__bps_half` = 1/2 is one shift COARSER than the default, `__bps_4` = 4/1 is
/// one shift FINER. Rungs are `__`-gated A/B arms.
///
/// **This is no longer the shipped rule for a picture plane (2026-08-11).** The
/// size sweep measured a block COUNT to be the wrong shape and the default is
/// now [`block_shift_rule_rows`], which coarsens from here until a block spans
/// [`ROWS_PER_BLOCK_MIN`] picture rows. `BPS` still decides
/// 1. the **base** the derived rule starts from and never goes finer than;
/// 2. the shift for every buffer with **no declared stride** (everything that
///    is not a picture plane); and
/// 3. the shift for the whole build when a rung is compiled in — selecting any
///    `__bps_*` rung, or `__bps_blocks`, turns the derived rule OFF so the
///    ladder stays a clean re-fit instrument. See [`ROWS_RULE_ACTIVE`].
#[cfg(feature = "__bps_quarter")]
const BPS: (usize, usize) = (1, 4);
#[cfg(all(feature = "__bps_half", not(feature = "__bps_quarter")))]
const BPS: (usize, usize) = (1, 2);
#[cfg(all(
    feature = "__bps_1",
    not(any(feature = "__bps_quarter", feature = "__bps_half"))
))]
const BPS: (usize, usize) = (1, 1);
#[cfg(all(
    feature = "__bps_4",
    not(any(feature = "__bps_quarter", feature = "__bps_half", feature = "__bps_1"))
))]
const BPS: (usize, usize) = (4, 1);
#[cfg(all(
    feature = "__bps_8",
    not(any(
        feature = "__bps_quarter",
        feature = "__bps_half",
        feature = "__bps_1",
        feature = "__bps_4"
    ))
))]
const BPS: (usize, usize) = (8, 1);
#[cfg(not(any(
    feature = "__bps_quarter",
    feature = "__bps_half",
    feature = "__bps_1",
    feature = "__bps_4",
    feature = "__bps_8"
)))]
const BPS: (usize, usize) = (2, 1);

/// Blocks the adaptive rule aims to split an instance into, i.e.
/// `N_SHARDS * BPS`. Named so the rule and its test assert against ONE
/// expression rather than two copies of the arithmetic.
const TARGET_BLOCKS: usize = {
    let t = (N_SHARDS * BPS.0) / BPS.1;
    if t == 0 { 1 } else { t }
};

// =============================================================================
// The rows-per-block rule — a DERIVED shift, not a rung. THE DEFAULT since
// 2026-08-11 (PR #503); `__bps_blocks` is the arm that reverts to the old
// block-count rule.
// =============================================================================

/// Whether the derived rows-per-block rule decides a strided buffer's shift.
///
/// **On unless a rung is compiled in.** The `__bps_*` ladder and `__bps_blocks`
/// are A/B instruments for re-fitting the block-COUNT rule, so a rung means
/// "give me exactly that constant" — mixing a rung with the derived rule would
/// measure neither. `__bps_blocks` is the ladder's centre rung, i.e. the rule
/// that shipped before this one, and is the base arm any re-measurement of the
/// default must be differenced against.
///
/// SOUND EITHER WAY: the block shift is a locality knob, never a correctness
/// one (see the module's soundness note and [`block_shift_for`]).
const ROWS_RULE_ACTIVE: bool = !cfg!(any(
    feature = "__bps_blocks",
    feature = "__bps_quarter",
    feature = "__bps_half",
    feature = "__bps_1",
    feature = "__bps_4",
    feature = "__bps_8"
));

/// Picture rows a block should span, once the buffer's stride is known.
///
/// **Why a rows target rather than a block count.** [`TARGET_BLOCKS`] fixes how
/// many blocks a buffer is cut into, so a block spans `len / TARGET_BLOCKS`
/// BYTES and therefore `len / (TARGET_BLOCKS * stride)` ROWS — which for a
/// picture plane is about `aligned_h / TARGET_BLOCKS`, i.e. a function of the
/// picture's HEIGHT. The hot strided accesses are a fixed number of ROWS
/// (CDEF tap windows and superblock-row compacts; measured `rows_mean` 7.16-9.02
/// at every hot site, on every picture size), so on a short picture one access
/// spreads over many blocks and on a tall one over few. That is a defect in the
/// rule's SHAPE, and no single constant fixes it — which the size sweep then
/// confirmed the hard way: `len` alone cannot even ORDER the sizes by how much
/// coarsening they want (1024x1024's plane is 1.11 MB and wants one shift,
/// 2048x576's is 1.35 MB and wants two, 1024x2048's is 2.23 MB and wants none).
///
/// **4, and it is fitted, not derived from the tap window.** The obvious value
/// is the tap window itself (8-9 rows, so one access fits in one block), and it
/// is measurably too coarse: replaying the rule over the sweep's own arms, a
/// target of 8 picks a worse arm than a target of 4 on five of fifteen cells and
/// picks the WORST available arm (1.023x) on 512x288. 4 is enough because the
/// door that matters is `MAX_SHARDS_PER_BORROW = 4`, not one block: at 4 rows
/// per block an 8-row window spans three blocks and never promotes, while every
/// further doubling only trades shard lines for shard COLLISIONS.
///
/// Measured across the 15-cell size sweep
/// (`benchmarks/shard_size_sweep_2026-08-10.*`, aarch64, 8 tiles, t=8): against
/// the block-count rule the wall win is 10-25% where that rule lands below 1
/// row per block, 8-14% between 1.8 and 2.2, and inside the bands at 3.8 and
/// above — so the crossover sits between 2.1 and 3.8 and this target is on the
/// coarse side of it by one step.
///
/// **Fitted on that grid with no held-out size, and that is a stated weakness of
/// the shipped value** (`docs/SHARD_SIZE_SWEEP.md` §1). `__rpb_2` / `__rpb_8` /
/// `__rpb_16` are the ladder for re-fitting THIS constant — the `__bps_*` rungs
/// re-fit the block-COUNT rule, which is the thing this replaced, so they are
/// the wrong instrument for the question. `__bps_blocks` is the base arm.
#[cfg(feature = "__rpb_2")]
const ROWS_PER_BLOCK_MIN: usize = 2;
#[cfg(all(feature = "__rpb_8", not(feature = "__rpb_2")))]
const ROWS_PER_BLOCK_MIN: usize = 8;
#[cfg(all(
    feature = "__rpb_16",
    not(any(feature = "__rpb_2", feature = "__rpb_8"))
))]
const ROWS_PER_BLOCK_MIN: usize = 16;
#[cfg(not(any(feature = "__rpb_2", feature = "__rpb_8", feature = "__rpb_16")))]
const ROWS_PER_BLOCK_MIN: usize = 4;

/// Floor on how many blocks a buffer keeps, whatever the rows target asks for.
///
/// Coarsening is NOT free: it trades "one borrow touching several shard lines"
/// for "several borrows landing on one shard", and the second cost grows as the
/// block count falls towards the worker count. 32 is the coarsest block count
/// the sweep measured to still be a win — `bps-quarter` cuts the 1024x192 and
/// 1024x384 planes into 34 and 51 blocks and reads 0.78x and 0.74x wall — so
/// this stops a shorter picture than the sweep contains from going past the last
/// point with evidence. It is a MEASURED bound, not a safety one; every value
/// here is sound.
///
/// At [`ROWS_PER_BLOCK_MIN`] = 4 it does not actually bind on any cell of the
/// sweep — 1024x192 ties it exactly — so it is a guard against pictures shorter
/// than the grid contains, not a fitted parameter. It DOES bind at a target of
/// 8, which is part of why 8 was rejected.
const MIN_BLOCKS: usize = 32;

/// Block shift for an instance of `len` bytes: the power of two that lands
/// `len` on about [`TARGET_BLOCKS`] = `N_SHARDS * BPS` blocks.
///
/// At 128 shards and a ratio of 2 that is `log2(len) - 8`, clamped. An 8.3 MB
/// 4K 8-bit luma plane gets 14 and its 16.6 MB 10-bit twin gets 15 — the same
/// ~4.3 picture rows per block either way — while a 1 MB plane gets 12 and a
/// 64 KiB one gets 8, instead of all of them being forced onto whatever suits
/// 4K.
///
/// SOUND FOR ANY VALUE: the "no missed overlap" argument needs only that both
/// registrants of a shared byte agree on the block boundaries, and the shift is
/// read once in [`BorrowTracker::new`] and immutable for that tracker's life
/// (it moves only in [`BorrowTracker::reprovision`], which takes `&mut self`
/// and therefore runs with no borrow outstanding).
#[inline]
fn block_shift_for(len: usize) -> u32 {
    block_shift_rule(len, active_shards(), tile_concurrency())
}

/// The shift decision as a pure function of `len` and the two declared
/// concurrency facts, so the policy can be tested without touching the
/// process-global monotone latches that feed it (they can only move one way,
/// which makes an ordering-dependent test of the gate impossible).
#[inline]
fn block_shift_rule(len: usize, shards: usize, tiles: usize) -> u32 {
    // A fixed rung, if one was selected, wins everywhere.
    if FIXED_SHIFT_SELECTED {
        return BLOCK_SHIFT;
    }
    if !ADAPTIVE_WHEN_SERIAL {
        // Serial decode keeps the old constant. The adaptive shift's whole
        // benefit is cross-core shard-line traffic, which a single thread does
        // not have, and the single-thread column of the ladder is
        // flat-to-slightly-adverse. Same split, for the same reason, as
        // SHARDS_SERIAL vs SHARDS_CONCURRENT — and read at the same moment, so
        // an instance built before `set_parallelism` simply keeps the serial
        // value, exactly like `mask`.
        //
        // Threads are necessary but NOT sufficient: a single-tile frame on
        // eight threads is concurrent, and the coarse shift measured 3.08%
        // SLOWER there while measuring 39% faster on the eight-tile frame at
        // the same thread count. See `set_tile_concurrency`.
        if shards < SHARDS_CONCURRENT || tiles < 2 {
            return BLOCK_SHIFT;
        }
    }
    let target = TARGET_BLOCKS as u64;
    let want = (len as u64 / target.max(1)).max(1);
    // `ilog2` rounds down, so the block count lands at or above the target.
    (u64::BITS - 1 - want.leading_zeros()).clamp(6, 24)
}

/// The rows-per-block refinement of [`block_shift_rule`], for a buffer whose
/// row stride is known. **This is the shipped rule for picture planes.**
///
/// Never FINER than the block-count rule — only coarser, and only far enough to
/// put [`ROWS_PER_BLOCK_MIN`] picture rows in a block, and only while the buffer
/// still holds [`MIN_BLOCKS`] blocks. A buffer with no declared stride keeps the
/// block-count rule exactly, so nothing outside the picture planes moves; so
/// does every buffer when a ladder rung is compiled in ([`ROWS_RULE_ACTIVE`]).
///
/// SOUND FOR ANY VALUE, for the same reason the block-count rule is: the "no
/// missed overlap" argument needs only that both registrants of a shared byte
/// agree on the block boundaries, and this runs from `&mut self` (see
/// [`BorrowTracker::set_row_stride`]) so no borrow can be outstanding when it
/// moves.
#[inline]
fn block_shift_rule_rows(len: usize, shards: usize, tiles: usize, stride: usize) -> u32 {
    let base = block_shift_rule(len, shards, tiles);
    if stride == 0 || FIXED_SHIFT_SELECTED || !ROWS_RULE_ACTIVE {
        return base;
    }
    // The SAME gates the block-count rule uses, restated rather than inferred
    // from `base` (a size whose adaptive shift happens to equal `BLOCK_SHIFT` is
    // not a disarmed one). Arming this rule can therefore never arm a case the
    // plain rule leaves disarmed.
    if !ADAPTIVE_WHEN_SERIAL && (shards < SHARDS_CONCURRENT || tiles < 2) {
        return base;
    }
    // Smallest shift with `2^shift >= ROWS_PER_BLOCK_MIN * stride`, i.e.
    // ceil(log2(want)). Written from `leading_zeros` rather than
    // `next_power_of_two`, which panics in debug on a `want` above 2^63 — the
    // `clamp` below would bound the result either way, but a panic in the
    // tracker's construction path is not something to leave reachable at all.
    let want = (ROWS_PER_BLOCK_MIN as u64)
        .saturating_mul(stride as u64)
        .max(1);
    let rows_shift = u64::BITS - (want - 1).leading_zeros();
    // Coarsest shift that still leaves MIN_BLOCKS blocks.
    let cap_want = (len as u64 / MIN_BLOCKS as u64).max(1);
    let cap_shift = u64::BITS - 1 - cap_want.leading_zeros();
    base.max(rows_shift.min(cap_shift)).clamp(6, 24)
}

/// THROWAWAY (`__probe_shiftpin`): pin a declared-stride buffer's block shift
/// from the environment, so a factorial over PER-PLANE shifts can be measured.
///
/// `RAV1D_PIN_SHIFT="1088:13,512:11"` — a comma-separated `stride:shift` list,
/// matched on the row stride the picture allocator declares. A stride not named
/// keeps the ordinary rule, and no variable at all leaves everything alone.
///
/// It exists because the rows rule and the `__bps_*` ladder move LUMA and CHROMA
/// together in a fixed pattern, so the grid they span cannot separate the two —
/// and the size sweep's one unexplained cell (512x576, where the derived rule
/// reads 0.995 between two rungs that read 0.927/0.930) is exactly a question
/// about whether the two planes' shifts interact.
///
/// SOUND FOR ANY VALUE, for the same reason every other shift choice is: both
/// registrants of a shared byte read the same live `shift`, which moves only
/// from `&mut self`. This is a measurement instrument, not a shipping knob, and
/// it is `__`-gated and absent from every published feature.
#[cfg(feature = "__probe_shiftpin")]
fn pinned_shift(stride: usize) -> Option<u32> {
    use std::sync::OnceLock;
    static PINS: OnceLock<Vec<(usize, u32)>> = OnceLock::new();
    let pins = PINS.get_or_init(|| {
        let Ok(spec) = std::env::var("RAV1D_PIN_SHIFT") else {
            return Vec::new();
        };
        spec.split(',')
            .filter_map(|kv| {
                let (k, v) = kv.split_once(':')?;
                Some((k.trim().parse().ok()?, v.trim().parse::<u32>().ok()?))
            })
            .collect()
    });
    pins.iter()
        .find(|(s, _)| *s == stride)
        .map(|(_, sh)| (*sh).clamp(6, 24))
}

/// True when one of the fixed `blockshift-*` rungs was selected, in which case
/// [`block_shift_for`] hands back [`BLOCK_SHIFT`] and nothing adapts.
const FIXED_SHIFT_SELECTED: bool = cfg!(any(
    feature = "__blockshift_8",
    feature = "__blockshift_10",
    feature = "__blockshift_13",
    feature = "__blockshift_14",
    feature = "__blockshift_15",
    feature = "__blockshift_16"
));

/// `__blockshift_adaptive` forces the adaptive shift even for a serial decode.
/// Only useful for A/B-ing the single-thread column; the default is off.
const ADAPTIVE_WHEN_SERIAL: bool = cfg!(feature = "__blockshift_adaptive");

/// Instances below this many elements get a single shard.
///
/// Sharding only pays when concurrent workers touch *different* addresses of
/// the same instance; a buffer smaller than this cannot spread far enough to
/// matter, and giving it one shard keeps its tracker to a single cache line.
/// Measured: 12 instances (the picture planes, 8.3 MB each) carry 89.8% of all
/// borrows and 100% of the contention, while 1,027 smaller ones see zero
/// contended acquisitions.
const SHARD_MIN_LEN: usize = 64 * 1024;

/// Fibonacci hashing: the multiplicative constant is `2^64 / phi`. Taking the
/// *high* bits mixes the low block bits (the x position within a picture row)
/// into the shard index, which is what separates concurrent tile columns.
///
/// `__shard_ident` swaps in the identity instead — see the A/B note there. Any
/// pure function of the block index is equally SOUND: the "no missed overlap"
/// argument only needs `shard(b)` to agree for both registrants of a shared
/// block, which any function does. The choice is purely locality vs collision.
#[cfg(not(feature = "__shard_ident"))]
#[inline(always)]
fn shard_of(block: usize, mask: usize) -> usize {
    // The second `&` is with a constant, which is what lets LLVM prove the
    // result indexes `[Shard; N_SHARDS]` in bounds. `mask` alone is a runtime
    // value it cannot bound.
    ((((block as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 40) as usize & mask) & (N_SHARDS - 1)
}

/// Identity shard mapping: consecutive blocks land on consecutive shards.
///
/// The hypothesis this arm tests: a `w x h` compact read registers `h` row
/// intervals whose block indices are consecutive-ish (at 4K a 3840-byte row
/// against a 4096-byte block advances the index by ~1), so under the identity
/// they occupy `h` ADJACENT 128-byte shard lines — one prefetchable 2 KiB run —
/// instead of `h` lines scattered over the instance's whole 16 KiB table.
/// The cost is that two tile columns on the same picture row, which differ by
/// at most one block index, can no longer be separated by the hash.
#[cfg(feature = "__shard_ident")]
#[inline(always)]
fn shard_of(block: usize, mask: usize) -> usize {
    (block & mask) & (N_SHARDS - 1)
}

/// `active_shards() - 1` when the buffer is worth spreading, else `0`.
///
/// Read once per [`BorrowTracker::new`] and then immutable for that tracker's
/// life, so two instances built either side of a [`set_parallelism`] call
/// simply get different masks — they share no records, and every index an
/// instance produces is masked with its OWN value on both registration and
/// release.
#[inline]
fn mask_for(len: usize) -> usize {
    if len >= SHARD_MIN_LEN {
        active_shards() - 1
    } else {
        0
    }
}

/// Shards a *concurrent* instance gets. The compile-time array size, i.e. the
/// most this build can ever hand out.
const SHARDS_CONCURRENT: usize = N_SHARDS;

/// Shards a big instance gets when the process has declared no parallelism.
///
/// Sharding buys nothing without concurrent registrants, and it is not free:
/// the wide path holds every ACTIVE shard (see [`BorrowTracker::active`]), and
/// the single-threaded decode path is exactly where wide borrows are common —
/// with tile threading off, `WithOffset::block_mut` reserves the whole strided
/// span `(h - 1) * stride + w`, which is 15 blocks for a 16x16 block on a 4K
/// row and therefore over `MAX_SHARDS_PER_BORROW`. Measured on v4k_8tile 8bpc
/// at t=1 (`benchmarks/p3_inversion_2026-08-07.meta`): raising the shard count
/// from 32 to 128 with the wide path holding all of them costs 413.7 -> 531.3
/// ms/frame, and narrowing the wide path to the active prefix takes 100 of
/// those 118 ms back — the residual 17 ms is the bigger array itself.
///
/// **ONE, not 32 (issue #458).** With 32 serial shards the mask still spreads a
/// strided block guard's ~15 blocks over up to 15 distinct shards, which is
/// over `MAX_SHARDS_PER_BORROW` — so at t=1 EVERY such guard still promoted to
/// the wide path. Each wide add/remove is ~`SHARDS_SERIAL` lock-prefixed RMWs;
/// on x86-64 those are full fences (~20-40 cycles each even uncontended) where
/// Apple LSE atomics are near-free, which is why the M4 ladder that chose 32
/// never saw the cost: on an Ultra 7 265K, v4k_8tile 8bpc t=1 decode measured
/// 352 ms/frame against the legacy tracker's 220 (`add_wide`+`remove_wide` =
/// 31% of self time). At ONE serial shard the mask is 0, every block of a span
/// maps to shard 0, and a strided guard registers as one ordinary narrow
/// interval — legacy-tracker behavior: 242 ms/frame, wide reserved for slot
/// exhaustion. Sharding exists to separate CONCURRENT registrants; with no
/// declared parallelism there is nothing to separate.
const SHARDS_SERIAL: usize = 1;

/// Declared decode parallelism, as a shard count. Monotone.
static ACTIVE_SHARDS: AtomicUsize = AtomicUsize::new(SHARDS_SERIAL);

/// Declare that up to `n` threads will register borrows concurrently.
///
/// One process-global, like the tile-threading flag it is set beside, and
/// **monotone** for the same reason that one had to become monotone: opening a
/// single-threaded decoder must not reconfigure a concurrently live
/// multi-threaded one. Unlike that flag, a stale value here is only ever a
/// performance question — the mask is read once per [`BorrowTracker::new`] and
/// is immutable for the tracker's life, and every shard index it produces is
/// masked with the SAME value on registration and release.
///
/// Measured, v4k_8tile 8bpc, ms/frame (`benchmarks/p3_inversion_2026-08-07.meta`):
/// 32 shards is 176.1 at t=4 and 189.3 at t=8 — SLOWER with twice the threads —
/// while 128 shards is 139.3 and 124.5. The crossover is entirely at t=1, which
/// is why this is a function of the declared parallelism rather than a constant.
pub fn set_parallelism(n: usize) {
    let want = if n > 1 {
        SHARDS_CONCURRENT
    } else {
        SHARDS_SERIAL
    };
    ACTIVE_SHARDS.fetch_max(want, Ordering::Relaxed);
}

#[inline(always)]
fn active_shards() -> usize {
    ACTIVE_SHARDS.load(Ordering::Relaxed).clamp(1, N_SHARDS)
}

/// Tiles the busiest frame seen so far could decode at once. Monotone.
static OBSERVED_TILES: AtomicUsize = AtomicUsize::new(1);

/// Declare how many tiles a frame about to be decoded splits into.
///
/// This is the second half of the adaptive block shift's gate, and it exists
/// because [`set_parallelism`] alone gets it wrong in one measured direction.
///
/// The shift's benefit is that a STRIDED access stops paying one distinct
/// shard line per row, which is worth paying shard collisions for only when
/// there are concurrent tile workers to collide. With `tiling.cols * rows ==
/// 1` the reconstruction of a frame is serial no matter how many threads are
/// open — the remaining concurrency is post-filter sbrow tasks over the same
/// planes — so the coarse block buys no locality and only widens each lock's
/// footprint.
///
/// Measured on this box (Apple M4 Pro 8P+4E, `scripts/perf/verify_gap.sh`,
/// median of 5, no `nice`, default features): v4k_1tile at t=8 cost **+3.08%**
/// (362.7 -> 373.8 ms/frame) from the tracker branch with per-round ranges that
/// do not overlap, while v4k_8tile at t=8 GAINED 39% (117.3 -> 71.3) from the
/// same change. Same thread count, opposite sign, and the only difference is
/// the tile split — which the tracker could not see, because it sizes itself
/// from buffer length.
///
/// Monotone `fetch_max`, for the same reason [`set_parallelism`] is: a later
/// single-tile frame must not reconfigure a decoder that is concurrently
/// running multi-tile ones. And, exactly as there, a stale value is only ever
/// a performance question — the shift is read once in [`BorrowTracker::new`],
/// is immutable for that tracker's life, and SOUNDNESS needs only that both
/// registrants of a shared byte agree on the block boundary, which they do
/// because they share the instance.
pub fn set_tile_concurrency(n: usize) {
    OBSERVED_TILES.fetch_max(n.max(1), Ordering::Relaxed);
}

#[inline(always)]
fn tile_concurrency() -> usize {
    OBSERVED_TILES.load(Ordering::Relaxed)
}

impl BorrowTracker {
    pub fn new(len: usize) -> Self {
        Self {
            shards: [const { Shard::new() }; N_SHARDS],
            shift: block_shift_for(len),
            mask: mask_for(len),
            row_stride: 0,
            #[cfg(feature = "__probe_tinynop")]
            tiny: len < SHARD_MIN_LEN,
            wide: UnsafeCell::new(Vec::new()),
            state: AtomicU32::new(0),
        }
    }

    /// Re-size the shard array after the container's length changed.
    ///
    /// `&mut self` is the whole safety argument: the caller holds `&mut
    /// DisjointMut`, so no borrow can be outstanding and no record can be lost.
    pub fn reprovision(&mut self, len: usize) {
        // Only the mask (and the block shift) moves; the shards are already
        // there, and `&mut self` guarantees every one of them is empty.
        self.shift = block_shift_for(len);
        self.mask = mask_for(len);
        // A resize drops the stride hint, exactly as it drops the derived shift
        // (see `DisjointMut::declare_row_stride`). Rectangle records simply stop
        // being offered until a stride is declared again; every caller has a
        // per-row path to fall back to, so this is a performance question and
        // never a correctness one.
        self.row_stride = 0;
        #[cfg(feature = "__probe_tinynop")]
        {
            self.tiny = len < SHARD_MIN_LEN;
        }
    }

    /// Tell the tracker this buffer's picture row stride in bytes, so the block
    /// shift can be chosen in ROWS rather than in blocks-per-buffer.
    ///
    /// `&mut self` carries the same whole safety argument as
    /// [`Self::reprovision`]: the caller holds `&mut DisjointMut`, so no borrow
    /// can be outstanding and no record can be lost when the shift moves.
    ///
    /// **This decides the shipped block shift for every picture plane** since
    /// 2026-08-11; before that it was a no-op feeding an A/B arm. It reverts to
    /// re-deriving [`block_shift_for`]'s answer — i.e. changes nothing — when a
    /// ladder rung is compiled in ([`ROWS_RULE_ACTIVE`]) or the caller has no
    /// stride to declare.
    #[inline]
    pub fn set_row_stride(&mut self, len: usize, stride: usize) {
        // Stored regardless of which shift rule wins below: it is what makes a
        // rectangle record's geometry derivable, and that is independent of the
        // block size.
        self.row_stride = stride;
        #[cfg(feature = "__probe_shiftpin")]
        if let Some(pinned) = pinned_shift(stride) {
            self.shift = pinned;
            return;
        }
        self.shift = block_shift_rule_rows(len, active_shards(), tile_concurrency(), stride);
    }

    /// The prefix of [`Self::shards`] this instance can actually reach.
    ///
    /// [`shard_of`] ends in `& self.mask`, and `mask` is always `2^k - 1`, so
    /// every index this instance ever produces — in [`Self::add`],
    /// [`Self::add_slow`], [`Self::add_multi`], [`Self::remove`] and
    /// [`Self::remove_multi`] — lies in `0..=mask`. No record of this instance
    /// can exist above that, and no narrow registrant of this instance can
    /// take a lock above it either.
    ///
    /// So the wide path's "hold **every** shard" only has to mean "hold every
    /// shard this instance uses": the exclusion it needs is against this
    /// instance's own narrow registrants, and they are all inside the prefix.
    /// Shards above `mask` are dead weight for this instance, and locking them
    /// costs `N_SHARDS - mask - 1` atomic RMWs per wide borrow — which is the
    /// whole of the wide path for a small instance (`mask == 0`: 1 lock
    /// instead of 32).
    ///
    /// `mask` only moves in [`Self::reprovision`], which takes `&mut self` and
    /// therefore runs with no borrow outstanding, so a record can never be
    /// stranded above a shrunken mask.
    #[inline(always)]
    fn active(&self) -> &[Shard] {
        // `min` keeps the slice in bounds for LLVM without a panic path; the
        // mask is always `<= N_SHARDS - 1` by construction.
        &self.shards[..(self.mask & (N_SHARDS - 1)) + 1]
    }

    /// This instance's block shift: a field when adaptive, the constant
    /// otherwise, so the hot path compiles to the same shape either way.
    #[inline(always)]
    fn block_shift(&self) -> u32 {
        self.shift
    }

    /// Mark this tracker as poisoned. All future borrow attempts will panic.
    pub fn poison(&self) {
        self.state.fetch_or(POISON_BIT, Ordering::Release);
    }

    #[cold]
    #[inline(never)]
    fn poisoned_panic() -> ! {
        panic!("DisjointMut poisoned: a thread panicked while holding a mutable borrow");
    }

    /// Report an overlap violation with diagnostic info.
    ///
    /// Takes the existing record's four fields SEPARATELY rather than as an
    /// `OverlapHit` tuple. A 25-byte tuple is returned and passed indirectly,
    /// so the caller has to build it on its own stack — which is a stack frame
    /// on the registration path even though this call never happens. Seven
    /// scalar arguments all fit in registers, and the call becomes a tail
    /// branch.
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn overlap_panic(
        new_start: usize,
        new_end: usize,
        new_mutable: bool,
        existing_start: usize,
        existing_end: usize,
        existing_mutable: bool,
        existing_loc: Option<&'static Location<'static>>,
    ) -> ! {
        let new_mut_str = if new_mutable { "&mut" } else { "   &" };
        let existing_mut_str = if existing_mutable { "&mut" } else { "   &" };
        let caller = Location::caller();
        struct MaybeLoc(Option<&'static Location<'static>>);
        impl core::fmt::Display for MaybeLoc {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self.0 {
                    Some(loc) => write!(f, " at {loc}"),
                    None => Ok(()),
                }
            }
        }
        let existing_loc = MaybeLoc(existing_loc);
        #[cfg(feature = "std")]
        {
            let thread = std::thread::current().id();
            panic!(
                "\toverlapping DisjointMut:\n current: {new_mut_str} _[{new_start}..{new_end}] \
                 on {thread:?} at {caller}\nexisting: {existing_mut_str} _[{existing_start}..{existing_end}]{existing_loc}",
            );
        }
        #[cfg(not(feature = "std"))]
        panic!(
            "\toverlapping DisjointMut:\n current: {new_mut_str} _[{new_start}..{new_end}] \
             at {caller}\nexisting: {existing_mut_str} _[{existing_start}..{existing_end}]{existing_loc}",
        );
    }

    /// Register a mutable borrow. Checks against ALL existing borrows.
    #[inline]
    #[track_caller]
    pub fn add_mut(&self, bounds: &Bounds) -> BorrowId {
        self.add::<true>(bounds)
    }

    /// Register an immutable borrow. Only checks against mutable borrows.
    #[inline]
    #[track_caller]
    pub fn add_immut(&self, bounds: &Bounds) -> BorrowId {
        self.add::<false>(bounds)
    }

    /// THROWAWAY (`__probe_addnop`): keep the CALL, delete the WORK.
    ///
    /// The question this answers: `probe-untracked` (no tracker at all) is
    /// 77 ms/frame faster than the tracker at 8bpc t=1, but removing 26
    /// instructions and two of the three locked RMWs from `add` moved that cell
    /// 0.3%. Those two facts are only compatible if most of the 77 ms is not
    /// the tracker's instructions but the fact that a call happens at all —
    /// 15.6 M opaque calls per frame that clobber every caller-saved register
    /// and fence the caller's optimizer. This arm keeps the call site, the
    /// argument setup, the `Option`/`Box` indirection and the clobber, and
    /// throws away everything inside. If it lands near `base`, no amount of
    /// shaving inside `add` can reach the ceiling; if it lands near
    /// `untracked`, the internals really are the cost.
    #[cfg(feature = "__probe_addnop")]
    #[inline(never)]
    fn add<const IS_MUT: bool>(&self, bounds: &Bounds) -> BorrowId {
        core::hint::black_box(bounds.range.start);
        BorrowId::UNCHECKED
    }

    #[cfg(not(feature = "__probe_addnop"))]
    #[inline]
    #[track_caller]
    fn add<const IS_MUT: bool>(&self, bounds: &Bounds) -> BorrowId {
        let start = bounds.range.start;
        let end = bounds.range.end;
        #[cfg(feature = "__probe_sites")]
        crate::site_probe::record(Location::caller(), IS_MUT, end.saturating_sub(start));
        // THROWAWAY (`__probe_tinynop`): price the sub-`SHARD_MIN_LEN` instance
        // class — `BlockContext`'s twenty 32-byte arrays and their kin, ~1,027
        // instances, which `mask_for` keeps single-shard at EVERY thread count.
        // Nothing else can separate their cost from the picture planes'.
        // UNSOUND: it stops tracking them.
        #[cfg(feature = "__probe_tinynop")]
        if self.tiny {
            return BorrowId::UNCHECKED;
        }
        if start >= end {
            return BorrowId::EMPTY;
        }
        // ONE-SHARD INSTANCES SKIP THE BLOCK ARITHMETIC ENTIRELY.
        //
        // `shard_of(b, 0)` is 0 for every block, so on a `mask == 0` instance
        // the shift load, both shifts, the multiplicative hash and its two
        // masks cannot change the answer — they are pure latency at the HEAD of
        // this function's dependency chain, feeding the address the lock
        // acquire needs. Skipping them shortens the chain by a dependent L1
        // load plus a multiply before anything else can start.
        //
        // That matters more than it looks: `probe-addnop` (keep the call,
        // delete the body) measured 290.7 ms/frame against 365.0 for the real
        // tracker and 287.2 with no tracker at all, at 8bpc t=1 — so the call
        // barrier is 4% of the tracker's cost and the other 96% is this
        // function's own latency. Two earlier attempts to cut its INSTRUCTION
        // count (two of three locked RMWs; the 26-instruction stack frame)
        // moved that cell 0.9% combined, which is what a throughput fix does to
        // a latency-bound path.
        //
        // Every instance is `mask == 0` when no parallelism has been declared
        // (`SHARDS_SERIAL == 1`, issue #458), and every sub-`SHARD_MIN_LEN`
        // instance is, always. Soundness is untouched: this is the SAME shard
        // the general path picks, and the multi-block case was already allowed
        // through the fast path here for exactly this reason (#458).
        // TRIED AND REVERTED: hoisting the `mask != 0` test so the one-shard
        // case is the fallthrough, and passing shard 0 as a LITERAL through an
        // `#[inline(always)]` body helper so its address folds to `self + 0x78`
        // instead of a `mov`/shift/add the lock acquire waits on. It does fold
        // — and it costs +0.9% (8bpc) / +0.8% (10bpc) at t=1, disjoint bands,
        // idle box, n=9: duplicating the ~70-instruction body took the function
        // from 117 to 208 instructions and put a stack frame back. The
        // dependency-chain win is real and smaller than the I-cache and frame
        // it buys. `benchmarks/tracker_borrowcost_2026-08-08.tsv`, arm `fold`.
        let si = if self.mask == 0 {
            // The pre-lock `state` check is KEPT, and it is not redundant with
            // the authoritative in-lock re-read below the way it looks. Removing
            // it measured +0.8% at 8bpc t=1 (337.4 -> 340.0, median of 9, idle):
            // this load warms the header line and resolves its branch while the
            // lock acquire is still in flight, so deleting it does not shorten
            // the chain, it serialises the in-lock load behind the acquire.
            // `benchmarks/tracker_borrowcost_2026-08-08.tsv`, arm `chain3`.
            if self.state.load(Ordering::Acquire) != 0 {
                return self.add_slow_wide_live::<IS_MUT>(start, end);
            }
            0
        } else {
            let shift = self.block_shift();
            let b0 = start >> shift;
            let b1 = (end - 1) >> shift;
            // One load and one branch covers poisoning, live wide records, and
            // multi-block borrows. All three are cold.
            if b0 != b1 || self.state.load(Ordering::Acquire) != 0 {
                return self.add_slow::<IS_MUT>(start, end, b0, b1);
            }
            shard_of(b0, self.mask)
        };

        // Fast path: the borrow lives in one shard — either one block, or any
        // span on a mask-0 instance. 99.875% of hot-plane borrows at
        // BLOCK_SHIFT = 8.
        let shard = &self.shards[si];
        // `try_lock`, not `lock`: see `TinyLock::try_lock`. A blocking acquire
        // here puts a `bl` in the middle of the hot path and costs the whole
        // function a stack frame.
        if !shard.lock.try_lock() {
            return self.add_contended::<IS_MUT>(start, end, si);
        }
        let g = ShardGuard(&shard.lock);
        // RE-READ `state` INSIDE THE LOCK. The load above happens BEFORE this
        // lock is taken, and a wide registrant publishes into `self.wide` —
        // not into any shard — then bumps `state` and unlocks. A narrow
        // registrant that observed `state == 0` in that window would scan a
        // shard which legitimately holds no record of the wide borrow and
        // register anyway: two overlapping mutable guards, both live, missed.
        //
        // That is a lock-ordering TOCTOU, distinct from the
        // registration-before-reference gap this module's header reasons
        // about — that argument does not account for the wide list at all.
        // Found by an independent 8-thread harness: 115/18/22 violations over
        // three runs of ~1.4e9 acquisitions, with one thread taking wide
        // borrows over a shared pivot byte. `add_multi` is NOT affected (it
        // reads `state` inside its locks), so the hole was unique to this
        // single-block fast path.
        //
        // Cost is one relaxed-ordering re-load on the hot path; the fall
        // through to `add_slow` (which does consult `wide`) is cold.
        if self.state.load(Ordering::Acquire) != 0 {
            drop(g);
            return self.add_slow_wide_live::<IS_MUT>(start, end);
        }
        // One snapshot of the occupancy bitmap drives both the scan and the
        // slot search. A release landing between the two can only clear bits,
        // which at worst wastes a slot search — never loses a record.
        // SAFETY: this shard's lock is held.
        let recs = unsafe { &mut *shard.recs.get() };
        let occ = shard.live_mask(recs.allocated);
        // `allocated` is written ONCE, on the success path below, instead of
        // being narrowed here and re-widened there. Leaving it stale-LARGE on
        // the two exits from here is sound by construction: it is a SUPERSET,
        // never a subset, and the only cost of a stale bit is one extra flag
        // load in a later scan. The success path narrows it on every borrow,
        // so it cannot saturate.
        if let Some(existing) = recs.find::<IS_MUT>(occ, start, end, self.row_stride) {
            drop(g);
            Self::overlap_panic(
                start, end, IS_MUT, existing.0, existing.1, existing.2, existing.3,
            );
        }
        match recs.alloc::<IS_MUT, false>(occ, start, end, here()) {
            Some(slot) => {
                recs.allocated = occ | (1u8 << (slot as usize).min(SLOTS - 1));
                shard.publish(slot);
                BorrowId::narrow1(si, slot)
            }
            None => {
                // Shard full — release and retry on the wide path, which is
                // atomic against everything.
                #[cfg(feature = "__probe_wide")]
                wide_probe::WIDE_FULL.fetch_add(1, Ordering::Relaxed);
                drop(g);
                self.add_wide::<IS_MUT>(start, end)
            }
        }
    }

    /// The single-block registration, but somebody else held the shard lock,
    /// so this one blocks for it.
    ///
    /// Byte-for-byte the same sequence as [`Self::add`]'s fast path — INCLUDING
    /// the in-lock `state` re-read that closes the wide-path TOCTOU (4af62ae);
    /// deleting it here would reopen exactly the hole `tests/wide_exclusion.rs`
    /// gates, on the path a contended multi-threaded decode actually takes. It
    /// is a separate `#[cold] #[inline(never)]` function purely so that the
    /// blocking acquire's call does not live inside the hot function.
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn add_contended<const IS_MUT: bool>(&self, start: usize, end: usize, si: usize) -> BorrowId {
        #[cfg(feature = "__probe_wide")]
        wide_probe::N_CONTENDED.fetch_add(1, Ordering::Relaxed);
        let shard = &self.shards[si & (N_SHARDS - 1)];
        shard.lock.lock();
        let g = ShardGuard(&shard.lock);
        if self.state.load(Ordering::Acquire) != 0 {
            drop(g);
            return self.add_slow_wide_live::<IS_MUT>(start, end);
        }
        // SAFETY: this shard's lock is held.
        let recs = unsafe { &mut *shard.recs.get() };
        let occ = shard.live_mask(recs.allocated);
        recs.allocated = occ;
        if let Some(existing) = recs.find::<IS_MUT>(occ, start, end, self.row_stride) {
            drop(g);
            Self::overlap_panic(
                start, end, IS_MUT, existing.0, existing.1, existing.2, existing.3,
            );
        }
        match recs.alloc::<IS_MUT, false>(occ, start, end, here()) {
            Some(slot) => {
                recs.allocated = occ | (1u8 << (slot as usize).min(SLOTS - 1));
                shard.publish(slot);
                BorrowId::narrow1(si, slot)
            }
            None => {
                #[cfg(feature = "__probe_wide")]
                wide_probe::WIDE_FULL.fetch_add(1, Ordering::Relaxed);
                drop(g);
                self.add_wide::<IS_MUT>(start, end)
            }
        }
    }

    /// [`Self::add_slow`] for a caller that never computed the block indices.
    ///
    /// The `mask == 0` fast path skips the shift and both block divisions
    /// because they cannot change which shard it picks; this recomputes them
    /// for the cold paths that DO need them (`add_slow` consults the wide list
    /// per block). Cold, so the arithmetic is free here.
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn add_slow_wide_live<const IS_MUT: bool>(&self, start: usize, end: usize) -> BorrowId {
        let shift = self.block_shift();
        self.add_slow::<IS_MUT>(start, end, start >> shift, (end - 1) >> shift)
    }

    /// Everything the fast path bailed out of: poisoned, a live wide record, or
    /// a borrow spanning more than one block.
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn add_slow<const IS_MUT: bool>(
        &self,
        start: usize,
        end: usize,
        b0: usize,
        b1: usize,
    ) -> BorrowId {
        #[cfg(feature = "__probe_wide")]
        wide_probe::N_SLOW.fetch_add(1, Ordering::Relaxed);
        if self.state.load(Ordering::Acquire) & POISON_BIT != 0 {
            Self::poisoned_panic();
        }
        if b0 == b1 {
            // Single block, but there is at least one live wide record, so the
            // wide list has to be consulted too.
            let si = shard_of(b0, self.mask);
            let shard = &self.shards[si];
            shard.lock.lock();
            let g = ShardGuard(&shard.lock);
            // SAFETY: this shard's lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            let occ = shard.live_mask(recs.allocated);
            recs.allocated = occ;
            let mut hit = recs.find::<IS_MUT>(occ, start, end, self.row_stride);
            if hit.is_none() {
                // SAFETY: a shard lock is held, and wide records are only
                // written while every shard lock is held.
                hit = Self::find_wide::<IS_MUT>(unsafe { &*self.wide.get() }, start, end);
            }
            if let Some(existing) = hit {
                drop(g);
                Self::overlap_panic(
                    start, end, IS_MUT, existing.0, existing.1, existing.2, existing.3,
                );
            }
            return match recs.alloc::<IS_MUT, false>(occ, start, end, here()) {
                Some(slot) => {
                    recs.allocated = occ | (1u8 << (slot as usize).min(SLOTS - 1));
                    shard.publish(slot);
                    BorrowId::narrow1(si, slot)
                }
                None => {
                    #[cfg(feature = "__probe_wide")]
                    wide_probe::WIDE_FULL.fetch_add(1, Ordering::Relaxed);
                    drop(g);
                    self.add_wide::<IS_MUT>(start, end)
                }
            };
        }
        self.add_multi::<IS_MUT>(start, end, b0, b1)
    }

    /// Borrow spanning several blocks. Registers the same exact interval in
    /// each distinct shard, acquired in ascending order.
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn add_multi<const IS_MUT: bool>(
        &self,
        start: usize,
        end: usize,
        b0: usize,
        b1: usize,
    ) -> BorrowId {
        #[cfg(feature = "__probe_wide")]
        wide_probe::N_MULTI.fetch_add(1, Ordering::Relaxed);
        let nblocks = b1 - b0 + 1;
        if nblocks > MAX_BLOCKS_SCAN {
            #[cfg(feature = "__probe_wide")]
            wide_probe::WIDE_BLOCKS.fetch_add(1, Ordering::Relaxed);
            return self.add_wide::<IS_MUT>(start, end);
        }
        let mut set = [0u16; MAX_SHARDS_PER_BORROW];
        let mut n = 0usize;
        for b in b0..=b1 {
            let s = shard_of(b, self.mask) as u16;
            if set[..n].contains(&s) {
                continue;
            }
            if n == MAX_SHARDS_PER_BORROW {
                #[cfg(feature = "__probe_wide")]
                wide_probe::WIDE_SHARDS.fetch_add(1, Ordering::Relaxed);
                return self.add_wide::<IS_MUT>(start, end);
            }
            set[n] = s;
            n += 1;
        }
        set[..n].sort_unstable();

        for &s in &set[..n] {
            self.shards[(s as usize) & (N_SHARDS - 1)].lock.lock();
        }
        // Check every held shard, plus the wide list.
        let mut hit = None;
        for &s in &set[..n] {
            let shard = &self.shards[(s as usize) & (N_SHARDS - 1)];
            // SAFETY: shard `s`'s lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            let occ = shard.live_mask(recs.allocated);
            recs.allocated = occ;
            if let Some(h) = recs.find::<IS_MUT>(occ, start, end, self.row_stride) {
                hit = Some(h);
                break;
            }
        }
        if hit.is_none() && self.state.load(Ordering::Relaxed) & !POISON_BIT != 0 {
            // SAFETY: shard locks are held.
            hit = Self::find_wide::<IS_MUT>(unsafe { &*self.wide.get() }, start, end);
        }
        if let Some(existing) = hit {
            Self::unlock_all(&self.shards, &set[..n]);
            Self::overlap_panic(
                start, end, IS_MUT, existing.0, existing.1, existing.2, existing.3,
            );
        }
        // Claim a slot in each. If any shard is full, roll the whole thing back
        // and go wide — a partial registration would be unsound.
        let mut slots = [0u8; MAX_SHARDS_PER_BORROW];
        let mut done = 0usize;
        while done < n {
            let shard = &self.shards[(set[done] as usize) & (N_SHARDS - 1)];
            // SAFETY: the shard's lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            // `allocated` was refreshed to the live mask by the scan above and
            // nothing can have published since (this thread holds every one of
            // these locks), so it IS the occupancy snapshot here.
            let occ = recs.allocated;
            match recs.alloc::<IS_MUT, false>(occ, start, end, here()) {
                Some(slot) => {
                    slots[done] = slot;
                    done += 1;
                }
                None => break,
            }
        }
        if done < n {
            #[cfg(feature = "__probe_wide")]
            wide_probe::WIDE_FULL.fetch_add(1, Ordering::Relaxed);
            // Nothing was published yet, so the rollback has nothing to undo:
            // `alloc` only fills fields, and the occupancy bits are set below.
            Self::unlock_all(&self.shards, &set[..n]);
            return self.add_wide::<IS_MUT>(start, end);
        }
        // Publish only once every shard has a slot, so a partially-registered
        // borrow is never observable.
        for i in 0..n {
            let shard = &self.shards[(set[i] as usize) & (N_SHARDS - 1)];
            // SAFETY: the shard's lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            recs.allocated |= 1u8 << (slots[i] as usize).min(SLOTS - 1);
            shard.publish(slots[i]);
        }
        Self::unlock_all(&self.shards, &set[..n]);
        BorrowId::from_pairs(&set[..n], &slots[..n])
    }

    /// Register a strided RECTANGLE — `rows` segments of `seg` bytes, the first
    /// at `lo`, successive ones the instance's declared row stride apart — as
    /// ONE record instead of `rows` of them.
    ///
    /// `None` means *not representable here*, and the caller must take its own
    /// per-row path. Nothing is ever approximated to make a rectangle fit: the
    /// declining cases are
    ///
    /// * no declared row stride, or a stride that does not match the caller's;
    /// * `seg > stride` (rows would overlap) or `rows > MAX_RECT_ROWS`;
    /// * the hull spans more than [`MAX_SHARDS_PER_BORROW`] BLOCKS, which is
    ///   what keeps a rectangle off the wide path — the wide list stores plain
    ///   intervals, so a promoted rectangle would degrade to its hull and could
    ///   then refuse a legitimate borrow;
    /// * a shard along the way is full (same reason: the fallback is the wide
    ///   path);
    /// * poisoning or a live wide record — the per-row path handles both, and
    ///   poisoning must still panic, which it does there.
    ///
    /// # Why this is exact
    ///
    /// The record stores the hull, and [`rect_decode`] recovers `(rows, seg)`
    /// from it — a bijection, not a widening. Shard SELECTION uses the hull's
    /// blocks, which is a superset of the blocks the rows occupy and therefore
    /// sound (a shared byte's block is in both registrants' sets, which is all
    /// the module header's argument needs). Overlap DETECTION uses
    /// [`rect_hit_range`], which knows nothing of the inter-row gaps, so the gap
    /// bytes are neither reserved nor reported: the false positive that gates
    /// `LfBlock::fill_hull` cannot arise here.
    #[inline]
    #[track_caller]
    pub fn add_rect_immut(
        &self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: usize,
    ) -> Option<BorrowId> {
        self.add_rect::<false>(lo, seg, rows, stride)
    }

    /// [`Self::add_rect_immut`] for a mutable rectangle.
    ///
    /// **No shipping caller yet** — `#[cfg(test)]` says so rather than an
    /// `allow(dead_code)` pretending otherwise. It exists because the exact
    /// rectangle-vs-rectangle test can only be reached with a MUTABLE rectangle
    /// (two immutable records never conflict), and that test is the one gate
    /// that distinguishes an exact record from a hull.
    #[cfg(test)]
    #[inline]
    #[track_caller]
    pub fn add_rect_mut(
        &self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: usize,
    ) -> Option<BorrowId> {
        self.add_rect::<true>(lo, seg, rows, stride)
    }

    #[track_caller]
    fn add_rect<const IS_MUT: bool>(
        &self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: usize,
    ) -> Option<BorrowId> {
        let s = self.row_stride;
        if s == 0 || s != stride || seg == 0 || seg > s || rows == 0 || rows > MAX_RECT_ROWS {
            #[cfg(feature = "__probe_wide")]
            wide_probe::N_RECT_DECLINED.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let span = (rows - 1) * s + seg;
        let end = lo.checked_add(span)?;
        // A live wide record or a poisoned tracker. The per-row fallback
        // consults the wide list and panics on poison, so declining here is not
        // a hole — it is the same answer, taken by the path that already knows
        // how to give it.
        if self.state.load(Ordering::Acquire) != 0 {
            #[cfg(feature = "__probe_wide")]
            wide_probe::N_RECT_DECLINED.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        // Shard set: the hull's blocks. On a one-shard instance every block maps
        // to shard 0, so the block span cannot promote and is not consulted.
        let mut set = [0u16; MAX_SHARDS_PER_BORROW];
        let n;
        if self.mask == 0 {
            set[0] = 0;
            n = 1;
        } else {
            let shift = self.block_shift();
            let b0 = lo >> shift;
            let b1 = (end - 1) >> shift;
            // `>= MAX` and not `>`: `b1 - b0 + 1` blocks, and distinct shards
            // are never more than blocks.
            if b1 - b0 >= MAX_SHARDS_PER_BORROW {
                #[cfg(feature = "__probe_wide")]
                wide_probe::N_RECT_DECLINED.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            let mut k = 0usize;
            for b in b0..=b1 {
                let sh = shard_of(b, self.mask) as u16;
                if set[..k].contains(&sh) {
                    continue;
                }
                // Belt as well as braces. The block-span test above already
                // bounds `k`, but this is the invariant the array's size rests
                // on, and `add_multi` writes it the same way: exceeding the cap
                // must DECLINE, never index past `set`.
                if k == MAX_SHARDS_PER_BORROW {
                    #[cfg(feature = "__probe_wide")]
                    wide_probe::N_RECT_DECLINED.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                set[k] = sh;
                k += 1;
            }
            n = k;
            set[..n].sort_unstable();
        }
        // THROWAWAY (`__rect_1shard`): accept ONLY rectangles that land in one
        // shard. A one-shard rectangle is strictly cheaper than the per-row
        // registrations it replaces — one `try_lock` on `add`, one lock-free
        // store on `remove`, exactly what a single per-row guard costs — whereas
        // a 2-shard one pays two locks on `add` and two more in `remove_multi`
        // against the per-row scheme's lock-FREE release. This arm separates the
        // record-count effect from the lock-traffic effect instead of measuring
        // their sum.
        #[cfg(feature = "__rect_1shard")]
        if n != 1 {
            #[cfg(feature = "__probe_wide")]
            wide_probe::N_RECT_DECLINED.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        #[cfg(feature = "__probe_sites")]
        crate::site_probe::record(Location::caller(), IS_MUT, span);

        for &sh in &set[..n] {
            self.shards[(sh as usize) & (N_SHARDS - 1)].lock.lock();
        }
        // RE-READ `state` inside the locks, for the reason spelled out in
        // `Self::add`: a wide registrant publishes into `self.wide` and bumps
        // `state` while holding EVERY shard, so a check made before this
        // thread's first acquire could be stale. Holding one of its shards is
        // enough to make this read authoritative.
        if self.state.load(Ordering::Acquire) != 0 {
            Self::unlock_all(&self.shards, &set[..n]);
            #[cfg(feature = "__probe_wide")]
            wide_probe::N_RECT_DECLINED.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut hit = None;
        for &sh in &set[..n] {
            let shard = &self.shards[(sh as usize) & (N_SHARDS - 1)];
            // SAFETY: shard `sh`'s lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            let occ = shard.live_mask(recs.allocated);
            recs.allocated = occ;
            if let Some(h) = recs.find_from_rect::<IS_MUT>(occ, lo, end, s) {
                hit = Some(h);
                break;
            }
        }
        if let Some(existing) = hit {
            Self::unlock_all(&self.shards, &set[..n]);
            Self::overlap_panic(
                lo, end, IS_MUT, existing.0, existing.1, existing.2, existing.3,
            );
        }
        let mut slots = [0u8; MAX_SHARDS_PER_BORROW];
        let mut done = 0usize;
        while done < n {
            let shard = &self.shards[(set[done] as usize) & (N_SHARDS - 1)];
            // SAFETY: the shard's lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            // Refreshed to the live mask by the scan above, and nothing can have
            // published since — this thread holds every one of these locks.
            let occ = recs.allocated;
            match recs.alloc::<IS_MUT, true>(occ, lo, end, here()) {
                Some(slot) => {
                    slots[done] = slot;
                    done += 1;
                }
                None => break,
            }
        }
        if done < n {
            // Nothing was published yet (`alloc` only fills fields), so there is
            // nothing to undo. Decline rather than promote: see the doc comment.
            #[cfg(feature = "__probe_wide")]
            wide_probe::N_RECT_DECLINED.fetch_add(1, Ordering::Relaxed);
            Self::unlock_all(&self.shards, &set[..n]);
            return None;
        }
        for i in 0..n {
            let shard = &self.shards[(set[i] as usize) & (N_SHARDS - 1)];
            // SAFETY: the shard's lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            recs.allocated |= 1u8 << (slots[i] as usize).min(SLOTS - 1);
            shard.publish(slots[i]);
        }
        Self::unlock_all(&self.shards, &set[..n]);
        #[cfg(feature = "__probe_wide")]
        {
            wide_probe::N_RECT.fetch_add(1, Ordering::Relaxed);
            if n > 1 {
                wide_probe::N_RECT_MULTI.fetch_add(1, Ordering::Relaxed);
            }
        }
        Some(if n == 1 {
            BorrowId::narrow1(set[0] as usize, slots[0])
        } else {
            BorrowId::from_pairs(&set[..n], &slots[..n])
        })
    }

    /// Last-resort registration: hold **every** shard, so the record is atomic
    /// against all narrow registrants, and publish it to the wide list that
    /// they all consult.
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn add_wide<const IS_MUT: bool>(&self, start: usize, end: usize) -> BorrowId {
        let active = self.active();
        for shard in active {
            shard.lock.lock();
        }
        let mut hit = None;
        for shard in active {
            // SAFETY: every shard lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            let occ = shard.live_mask(recs.allocated);
            recs.allocated = occ;
            if let Some(h) = recs.find::<IS_MUT>(occ, start, end, self.row_stride) {
                hit = Some(h);
                break;
            }
        }
        // SAFETY: every shard lock is held. Scoped so the `&mut` is dead
        // before the locks drop — otherwise another thread's `&` read of the
        // same list would alias a live `&mut`.
        if hit.is_none() {
            hit = Self::find_wide::<IS_MUT>(unsafe { &*self.wide.get() }, start, end);
        }
        if let Some(existing) = hit {
            Self::unlock_every(active);
            Self::overlap_panic(
                start, end, IS_MUT, existing.0, existing.1, existing.2, existing.3,
            );
        }
        let idx = {
            // SAFETY: every shard lock is held.
            let wide = unsafe { &mut *self.wide.get() };
            let rec = (start, end, IS_MUT, wide_loc());
            match wide.iter().position(|r| r.0 >= r.1) {
                Some(i) => {
                    wide[i] = rec;
                    i
                }
                None => {
                    wide.push(rec);
                    wide.len() - 1
                }
            }
        };
        assert!(
            idx <= u16::MAX as usize,
            "DisjointMut: too many concurrent wide borrows"
        );
        self.state.fetch_add(1, Ordering::Relaxed);
        Self::unlock_every(active);
        BorrowId::wide(idx as u16)
    }

    #[inline(always)]
    fn find_wide<const IS_MUT: bool>(
        wide: &[WideRec],
        start: usize,
        end: usize,
    ) -> Option<OverlapHit> {
        for &(s, e, m, l) in wide {
            if s < e && (IS_MUT || m) && s < end && start < e {
                return Some((s, e, m, l));
            }
        }
        None
    }

    #[inline(always)]
    fn unlock_all(shards: &[Shard], set: &[u16]) {
        for &s in set.iter().rev() {
            shards[(s as usize) & (N_SHARDS - 1)].lock.unlock();
        }
    }

    #[inline(always)]
    fn unlock_every(shards: &[Shard]) {
        for shard in shards.iter().rev() {
            shard.lock.unlock();
        }
    }

    /// Release a borrow.
    ///
    /// **Lock-free, and not even an atomic RMW.** Retiring a record is one
    /// plain `Release` store to this slot's own [`Shard::live`] byte. No
    /// `fetch_and` is needed because no other thread can be writing that byte:
    /// the only other writer is the allocator that reuses the slot, and it
    /// cannot pick the slot until this store is visible. The record's own
    /// fields are left alone — they are only ever read for live slots, and the
    /// next registration to claim the slot rewrites them under the lock before
    /// publishing.
    ///
    /// What this does NOT change is the size of the add/remove race window.
    /// Whether a registration that overlaps a borrow being dropped sees the
    /// record was already decided by which of the two reached the shard first;
    /// taking the lock here never made that ordering more meaningful, it only
    /// made both sides queue for the same cache line. (It is also why the
    /// deliberate-overlap tests still fire: they hold the first borrow LIVE
    /// across the second registration, which no ordering can hide.)
    #[inline]
    pub fn remove(&self, id: BorrowId) {
        if id.kind() != KIND_NARROW {
            if id.kind() == KIND_WIDE {
                self.remove_wide(id);
            }
            return;
        }
        if id.pairs() != 1 {
            return self.remove_multi(id);
        }
        let (si, slot) = id.pair(0, self.mask);
        self.shards[si].retire(slot);
    }

    /// Releasing a multi-shard borrow must be atomic: dropping the record from
    /// shard A before shard B would leave a window in which a genuinely
    /// disjoint neighbour still sees the dead record and panics.
    ///
    /// The bit-clears are lock-free like [`Self::remove`]'s, but the locks are
    /// still taken, because *that* is what makes the whole set retire as one
    /// step against a registrant that holds several of these shards at once
    /// ([`Self::add_multi`], [`Self::add_wide`]). This path is cold — measured
    /// 0.00% of a 4K frame — so the two extra acquisitions buy the property for
    /// nothing.
    #[cold]
    #[inline(never)]
    fn remove_multi(&self, id: BorrowId) {
        let n = id.pairs();
        // `add_multi` stored the pairs ascending, so this order is safe.
        for i in 0..n {
            self.shards[id.pair(i, self.mask).0].lock.lock();
        }
        for i in 0..n {
            let (si, slot) = id.pair(i, self.mask);
            self.shards[si].retire(slot);
        }
        for i in (0..n).rev() {
            self.shards[id.pair(i, self.mask).0].lock.unlock();
        }
    }

    #[cold]
    #[inline(never)]
    fn remove_wide(&self, id: BorrowId) {
        let active = self.active();
        for shard in active {
            shard.lock.lock();
        }
        {
            // SAFETY: every shard lock is held; scoped so the `&mut` is dead
            // before the locks drop.
            let wide = unsafe { &mut *self.wide.get() };
            let i = id.wide_idx();
            if i < wide.len() {
                wide[i] = (1, 0, false, None); // tombstone
            }
        }
        self.state.fetch_sub(1, Ordering::Relaxed);
        Self::unlock_every(active);
    }
}

/// The wide list keeps the site unconditionally — it is cold, and a wide record
/// is exactly the kind an overlap panic most needs to name.
#[inline(always)]
#[track_caller]
fn wide_loc() -> Option<&'static Location<'static>> {
    Some(Location::caller())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(r: core::ops::Range<usize>) -> Bounds {
        Bounds { range: r }
    }

    /// Two borrows in the same block must collide (the fast path).
    #[test]
    #[should_panic(expected = "overlapping DisjointMut")]
    fn same_block_overlap_is_caught() {
        let t = BorrowTracker::new(1 << 20);
        let _a = t.add_mut(&b(0..16));
        let _c = t.add_mut(&b(8..24));
    }

    /// The interesting case: the two borrows overlap in a block whose shard is
    /// not the shard of either borrow's *first* block. Registering only the
    /// first block's shard would miss this.
    #[test]
    #[should_panic(expected = "overlapping DisjointMut")]
    fn overlap_in_a_later_block_is_caught() {
        let t = BorrowTracker::new(1 << 20);
        let bs = 1usize << t.block_shift();
        // a covers blocks 0..=2, c covers blocks 2..=4; they share block 2.
        let _a = t.add_mut(&b(0..2 * bs + 1));
        let _c = t.add_mut(&b(2 * bs..4 * bs + 1));
    }

    /// Adjacent-but-disjoint borrows either side of a block boundary must NOT
    /// collide, even though hashing may put them in the same shard.
    #[test]
    fn block_boundary_neighbours_do_not_collide() {
        let t = BorrowTracker::new(1 << 20);
        let bs = 1usize << t.block_shift();
        for k in 0..256usize {
            let a = t.add_mut(&b(k * bs..(k + 1) * bs));
            let c = t.add_mut(&b((k + 1) * bs..(k + 2) * bs));
            t.remove(a);
            t.remove(c);
        }
    }

    /// Sharding must not let two borrows that hash to *different* shards slip
    /// past each other when they overlap. Sweep enough offsets to hit many
    /// shard pairs.
    #[test]
    fn cross_shard_overlaps_are_all_caught() {
        let bs = 1usize << BorrowTracker::new(1 << 20).block_shift();
        for k in 0..64usize {
            let t = BorrowTracker::new(1 << 20);
            let _a = t.add_mut(&b(k * bs..k * bs + 4));
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _c = t.add_mut(&b(k * bs + 2..k * bs + 6));
            }))
            .is_err();
            assert!(caught, "missed overlap at block {k}");
            // `_a` is deliberately left registered; `t` is dropped at the end
            // of the iteration.
            let _ = _a;
        }
    }

    /// Immutable borrows may share; a mutable one may not join them.
    #[test]
    fn immutable_sharing_then_mutable_conflict() {
        let t = BorrowTracker::new(1 << 20);
        let a = t.add_immut(&b(0..64));
        let c = t.add_immut(&b(32..96));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _d = t.add_mut(&b(40..48));
        }))
        .is_err();
        assert!(
            caught,
            "mutable borrow overlapping two immutables not caught"
        );
        t.remove(a);
        t.remove(c);
        // With them gone the mutable borrow is fine.
        let d = t.add_mut(&b(40..48));
        t.remove(d);
    }

    /// A borrow wide enough to exceed `MAX_SHARDS_PER_BORROW` takes the wide
    /// path; narrow borrows must still see it.
    #[test]
    fn wide_borrow_is_visible_to_narrow_registrants() {
        let t = BorrowTracker::new(1 << 20);
        let bs = 1usize << t.block_shift();
        let wide = t.add_mut(&b(0..MAX_BLOCKS_SCAN * bs * 4));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _n = t.add_mut(&b(5 * bs..5 * bs + 4));
        }))
        .is_err();
        assert!(caught, "narrow borrow missed a live wide record");
        t.remove(wide);
        // Released: the same narrow borrow now succeeds.
        let n = t.add_mut(&b(5 * bs..5 * bs + 4));
        t.remove(n);
    }

    /// ...and the reverse: a narrow record must be found by a wide registrant.
    #[test]
    fn narrow_borrow_is_visible_to_wide_registrant() {
        let t = BorrowTracker::new(1 << 20);
        let bs = 1usize << t.block_shift();
        let n = t.add_mut(&b(5 * bs..5 * bs + 4));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _w = t.add_mut(&b(0..MAX_BLOCKS_SCAN * bs * 4));
        }))
        .is_err();
        assert!(caught, "wide registrant missed a live narrow record");
        t.remove(n);
    }

    /// The wide path holds only `0..=mask` (see [`BorrowTracker::active`]).
    /// For a SMALL instance that prefix is a single shard, which is the case
    /// where the narrowing is most aggressive — so this is the case that must
    /// still detect. Overflowing the one shard is what forces the promotion.
    #[test]
    fn wide_and_narrow_still_see_each_other_on_a_one_shard_instance() {
        let t = BorrowTracker::new(SHARD_MIN_LEN - 1);
        assert_eq!(t.mask, 0, "a sub-SHARD_MIN_LEN instance must get one shard");
        // Fill the single shard, so the next registration is promoted to wide.
        let mut ids = Vec::new();
        for i in 0..SLOTS {
            ids.push(t.add_mut(&b(i * 2..i * 2 + 1)));
        }
        let wide = t.add_mut(&b(1000..1004));
        // A narrow registrant must still see the promoted record...
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _n = t.add_mut(&b(1002..1006));
        }))
        .is_err();
        assert!(
            caught,
            "narrow borrow missed a wide record on a 1-shard instance"
        );
        t.remove(wide);
        // ...and stop seeing it once released.
        let n = t.add_mut(&b(1002..1006));
        t.remove(n);
        for id in ids {
            t.remove(id);
        }
    }

    /// The same, on an instance sized for the FULL shard set. Guards against a
    /// future `shard_of` that stops masking with `self.mask`, which would put
    /// records outside the prefix the wide path holds.
    #[test]
    fn wide_and_narrow_still_see_each_other_at_full_width() {
        set_parallelism(64);
        let t = BorrowTracker::new(1 << 24);
        assert_eq!(
            t.mask,
            N_SHARDS - 1,
            "declared parallelism must widen the mask"
        );
        let bs = 1usize << t.block_shift();
        // A borrow spanning far more blocks than MAX_SHARDS_PER_BORROW, so it
        // is promoted; the narrow probe sits in a block deep inside it, whose
        // shard is nowhere near the wide registrant's first.
        let wide = t.add_mut(&b(0..MAX_BLOCKS_SCAN * bs * 4));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _n = t.add_mut(&b(200 * bs..200 * bs + 4));
        }))
        .is_err();
        assert!(
            caught,
            "narrow borrow missed a live wide record at full width"
        );
        t.remove(wide);
        let n = t.add_mut(&b(200 * bs..200 * bs + 4));
        t.remove(n);
    }

    /// Issue #458: with no parallelism declared, a BIG instance must get ONE
    /// shard (mask 0), and a strided multi-block guard must register as one
    /// ordinary narrow interval — never promote to the wide path. With
    /// `SHARDS_SERIAL = 32` this failed twice over: the instance got mask 31,
    /// and a ~15-block guard hashed to more than `MAX_SHARDS_PER_BORROW`
    /// distinct shards, so EVERY such guard went wide — ~32 lock-prefixed RMWs
    /// per add/remove on the single-threaded decode path (measured +59%
    /// whole-frame at t=1 on x86-64, where a locked RMW is a full fence).
    ///
    /// Process-state note: `set_parallelism` is a monotone process-global, so
    /// this test is meaningful only while THIS process has not declared
    /// parallelism — under nextest's process-per-test isolation that always
    /// holds. The first assertion fails loudly (never silently skips) if that
    /// assumption breaks.
    #[test]
    fn serial_big_instance_keeps_strided_guards_narrow() {
        // The serial-instance half of the guarantee is compile-time: reverting
        // `SHARDS_SERIAL` to a wider set fails the build, not a race-prone
        // runtime assertion (`set_parallelism` is a monotone process-global,
        // so a big-instance mask check here would flake under plain
        // `cargo test`'s shared process).
        const _: () = assert!(
            SHARDS_SERIAL == 1,
            "issue #458: serial instances must get ONE shard, or strided block \
             guards promote to the wide path on the single-threaded decode path"
        );
        // Behavioral half on an instance that is mask-0 by CONSTRUCTION
        // (below SHARD_MIN_LEN), immune to process state: a multi-block span
        // must stay narrow, keep the wide list empty, and still detect
        // overlap.
        let t = BorrowTracker::new(32 * 1024);
        assert_eq!(t.mask, 0, "sub-SHARD_MIN_LEN instances are single-shard");
        let bs = 1usize << t.block_shift();
        // The shape of a strided block guard: ~15 blocks.
        let id = t.add_mut(&b(bs..7 * bs + 3));
        assert_eq!(
            t.state.load(Ordering::Relaxed),
            0,
            "a multi-block guard on a mask-0 instance must not go wide"
        );
        assert_eq!(id.kind(), KIND_NARROW, "must be a narrow record");
        assert_eq!(id.pairs(), 1, "must occupy exactly one (shard, slot) pair");
        // Overlap detection through that narrow record still fires...
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _o = t.add_mut(&b(4 * bs..4 * bs + 4));
        }))
        .is_err();
        assert!(
            caught,
            "overlap inside the strided span must still be caught"
        );
        t.remove(id);
        // ...and clears cleanly.
        let again = t.add_mut(&b(4 * bs..4 * bs + 4));
        t.remove(again);
    }

    /// A single-threaded open must never shrink the shard set out from under a
    /// concurrently live multi-threaded decoder — the same hazard that forced
    /// the tile-threading flag to become monotone.
    #[test]
    fn set_parallelism_is_monotone() {
        set_parallelism(64);
        let raised = active_shards();
        assert!(raised >= SHARDS_SERIAL);
        set_parallelism(1);
        assert_eq!(
            active_shards(),
            raised,
            "a 1-thread open must not lower the shard count"
        );
    }

    /// Filling a shard past `SLOTS` must promote to the wide list, not drop the
    /// record.
    #[test]
    fn shard_overflow_promotes_and_still_detects() {
        let t = BorrowTracker::new(1 << 20);
        // All in block 0, hence all in one shard: SLOTS + 3 live at once.
        let mut ids = Vec::new();
        for i in 0..(SLOTS + 3) {
            ids.push(t.add_mut(&b(i * 2..i * 2 + 1)));
        }
        // Every one of them, including the promoted ones, is still enforced.
        for i in 0..(SLOTS + 3) {
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _x = t.add_mut(&b(i * 2..i * 2 + 1));
            }))
            .is_err();
            assert!(caught, "record {i} was lost on shard overflow");
        }
        for id in ids {
            t.remove(id);
        }
        // All released: the whole span is borrowable again.
        let x = t.add_mut(&b(0..(SLOTS + 3) * 2));
        t.remove(x);
    }

    #[test]
    fn poison_blocks_everything() {
        let t = BorrowTracker::new(1 << 20);
        t.poison();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _x = t.add_immut(&b(0..1));
        }))
        .is_err();
        assert!(caught);
    }

    /// The adaptive shift must keep the block count near
    /// [`TARGET_BLOCKS`] across the whole range of buffer sizes the
    /// decoder allocates — that is the entire point of it over a constant.
    ///
    /// The sizes below are real: a 4K 8-bit luma plane, its chroma planes, the
    /// same at 10-bit, a 1024x1024 plane, and a buffer just over
    /// `SHARD_MIN_LEN`.
    // Not applicable when a fixed rung is compiled in — `block_shift_for` then
    // hands back the constant by design, which the last assertion below pins
    // from the other side. Gated on the CONFIG, not skipped at runtime.
    #[cfg(not(any(
        feature = "__blockshift_8",
        feature = "__blockshift_10",
        feature = "__blockshift_13",
        feature = "__blockshift_14",
        feature = "__blockshift_15",
        feature = "__blockshift_16"
    )))]
    #[test]
    fn adaptive_shift_keeps_the_block_count_near_target() {
        // This test is about the RULE, not about the gate, so it drives
        // `block_shift_rule` with both concurrency facts declared rather than
        // poking the monotone globals (which the gate test below must be able
        // to see un-poked).
        let sh = |len: usize| block_shift_rule(len, SHARDS_CONCURRENT, 8);
        let target = TARGET_BLOCKS;
        for len in [
            64 * 1024,       // just at SHARD_MIN_LEN
            1024 * 1024,     // 1024x1024 8-bit plane
            1920 * 1080,     // 4K chroma, 4:2:0
            3840 * 2160,     // 4K luma, 8-bit
            2 * 3840 * 2160, // 4K luma, 10/12-bit
            8 * 3840 * 2160, // a generous over-allocation
        ] {
            let shift = sh(len);
            let nblocks = len >> shift;
            assert!(
                nblocks >= target && nblocks < target * 2,
                "len {len}: shift {shift} gives {nblocks} blocks, target {target}"
            );
        }
        // The invariant the RATIO exists for, and it has to hold at every rung
        // of the `__bps_*` ladder: the 8-bit and 10-bit 4K luma planes land on
        // the same PICTURE ROWS PER BLOCK, because a 10-bit plane is twice the
        // bytes AND twice the stride, so a rule keyed on `len` tracks the stride
        // for free.
        assert_eq!(
            (1usize << sh(3840 * 2160)) / 3840,
            (1usize << sh(2 * 3840 * 2160)) / 7680,
            "rows/block must match across bit depth at BPS {BPS:?}"
        );
        // A small buffer must NOT be handed the 4K plane's shift, which would
        // collapse it onto one or two shards. Relative, so it is a real
        // assertion at every rung rather than a constant that only fits one.
        assert!(sh(64 * 1024) < sh(2 * 3840 * 2160));
        // The DEFAULT rung's two measured shifts, pinned by value: the fixed
        // ladder (`benchmarks/tracker_blockshift_2026-08-08.meta`) measured 14
        // joint-best for the 8-bit 4K plane and 15 within 1.2% of best for its
        // 10-bit twin, and this is what stops a ladder rung silently becoming
        // the shipped default.
        if BPS == (2, 1) {
            assert_eq!(sh(3840 * 2160), 14);
            assert_eq!(sh(2 * 3840 * 2160), 15);
            assert_eq!((1usize << sh(3840 * 2160)) / 3840, 4);
        }
    }

    /// The other side of the gate: with a fixed rung compiled in, nothing
    /// adapts and every buffer gets the constant.
    #[test]
    fn a_fixed_rung_overrides_the_adaptive_rule() {
        let sh = |len: usize| block_shift_rule(len, SHARDS_CONCURRENT, 8);
        if FIXED_SHIFT_SELECTED {
            for len in [64 * 1024, 1024 * 1024, 3840 * 2160, 2 * 3840 * 2160] {
                assert_eq!(sh(len), BLOCK_SHIFT, "len {len}");
            }
        } else {
            // No rung selected: the rule must actually be doing something, i.e.
            // two very different buffers must not get the same shift.
            assert_ne!(sh(64 * 1024), sh(2 * 3840 * 2160));
        }
    }

    /// A single-tile frame must NOT get the coarse shift, however many threads
    /// are open.
    ///
    /// This is the measured regression the tile gate exists for: v4k_1tile at
    /// t=8 cost +3.08% from the adaptive shift while v4k_8tile at t=8 gained
    /// 39% from it, same thread count. Thread parallelism alone cannot tell
    /// those two apart, so `block_shift_for` reads both latches.
    #[cfg(not(any(
        feature = "__blockshift_8",
        feature = "__blockshift_10",
        feature = "__blockshift_13",
        feature = "__blockshift_14",
        feature = "__blockshift_15",
        feature = "__blockshift_16",
        feature = "__blockshift_adaptive"
    )))]
    #[test]
    fn one_tile_does_not_get_the_coarse_shift() {
        const LEN: usize = 2 * 3840 * 2160;
        // Threads but one tile: the constant.
        assert_eq!(block_shift_rule(LEN, SHARDS_CONCURRENT, 1), BLOCK_SHIFT);
        // Tiles but one thread: still the constant (the pre-existing gate).
        assert_eq!(block_shift_rule(LEN, SHARDS_SERIAL, 8), BLOCK_SHIFT);
        // Both: adapt. And this must be a real change, or the test is vacuous.
        let adapted = block_shift_rule(LEN, SHARDS_CONCURRENT, 8);
        assert_ne!(adapted, BLOCK_SHIFT);
        if BPS == (2, 1) {
            assert_eq!(adapted, 15);
        }
        // Two tiles is already "multi-tile"; the gate is a threshold, not a
        // proportion.
        assert_eq!(block_shift_rule(LEN, SHARDS_CONCURRENT, 2), adapted);
    }

    /// The rows-per-block rule, checked against the exact plane geometry the
    /// picture allocator produces — not against round numbers.
    ///
    /// `stride = (w + 127 & !127) << hbd`, `+ 64` when that is a multiple of
    /// 1024; `len = stride * (h + 127 & !127)` (`Rav1dPicAllocator`). Those are
    /// the buffers the size sweep measured, so the rule is pinned on the same
    /// inputs the numbers came from.
    ///
    /// This test is about the RULE, so it drives it with both concurrency facts
    /// declared, like its block-count sibling above.
    // Gated on the CONFIG, never skipped at runtime: a compiled-in rung turns
    // the derived rule off by design, and `a_rung_disables_the_derived_rule`
    // below is the assertion for that side. Both configs assert something.
    #[cfg(not(any(
        feature = "__bps_blocks",
        feature = "__bps_quarter",
        feature = "__bps_half",
        feature = "__bps_1",
        feature = "__bps_4",
        feature = "__bps_8",
        feature = "__blockshift_8",
        feature = "__blockshift_10",
        feature = "__blockshift_13",
        feature = "__blockshift_14",
        feature = "__blockshift_15",
        feature = "__blockshift_16"
    )))]
    #[test]
    fn rows_rule_targets_picture_rows_not_block_count() {
        fn plane(w: usize, h: usize, hbd: u32) -> (usize, usize) {
            let mut stride = ((w + 127) & !127) << hbd;
            if stride & 1023 == 0 {
                stride += 64;
            }
            (stride * ((h + 127) & !127), stride)
        }
        let rows_of = |w, h, hbd| {
            let (len, stride) = plane(w, h, hbd);
            let sh = block_shift_rule_rows(len, SHARDS_CONCURRENT, 8, stride);
            ((1usize << sh) / stride, len >> sh)
        };
        // Every cell of the sweep grid must land at or above the rows target,
        // unless the block floor bit first. THIS is the property the rule is
        // for; the block-count rule fails it by construction on short pictures.
        for (w, h) in [
            (512, 288),
            (512, 576),
            (1024, 192),
            (1024, 576),
            (1024, 1024),
            (1024, 2160),
            (2048, 576),
            (2048, 1152),
            (3840, 576),
            (3840, 2160),
        ] {
            let (rows, blocks) = rows_of(w, h, 0);
            // Either the rows target is met, or the ONLY reason it is not is
            // that one more shift would fall under the block floor. `blocks <=
            // MIN_BLOCKS` was the earlier form and is too strict at a coarser
            // target: `ilog2` rounds the cap down, so a floor-bound cell can sit
            // anywhere in [MIN_BLOCKS, 2*MIN_BLOCKS). 1024x192 at a target of 8
            // lands on 34 blocks and would fail it while being exactly right.
            assert!(
                rows >= ROWS_PER_BLOCK_MIN || (blocks >> 1) < MIN_BLOCKS,
                "{w}x{h}: {rows} rows/block, {blocks} blocks — below the target \
                 with the block floor not binding"
            );
            assert!(blocks >= 1, "{w}x{h}: {blocks} blocks");
        }
        // The floor is not decorative: 1024x192 is the cell where it binds, and
        // it must leave at least MIN_BLOCKS/2 blocks rather than collapse the
        // plane onto a handful.
        let (_, blocks) = rows_of(1024, 192, 0);
        assert!(
            blocks >= MIN_BLOCKS / 2,
            "1024x192 kept only {blocks} blocks"
        );
        // Never FINER than the block-count rule, at any cell or bit depth. The
        // narrow-and-tall cells are the ones that make this a real assertion:
        // 256x2048's stride is 256, so its rows target lands at shift 10 while
        // the block-count rule already chose 11. Without the `max` the rule
        // would go a step FINER there, which is the opposite of its purpose —
        // and a list of only wide cells would never notice.
        let mut narrower = 0;
        for (w, h) in [
            (512, 288),
            (1024, 576),
            (3840, 2160),
            (256, 2048),
            (256, 4096),
            (128, 2048),
            (1024, 2048),
            // Narrow enough that the unclamped rows target stays finer than the
            // block rule even at `__rpb_16`: that needs `aligned_h > 256 * R`,
            // and without a cell like this the anti-vacuity check below FAILS at
            // the coarse rungs — which is the check doing its job, not a bug.
            (128, 8192),
        ] {
            for hbd in [0, 1] {
                let (len, stride) = plane(w, h, hbd);
                let base = block_shift_rule(len, SHARDS_CONCURRENT, 8);
                assert!(
                    block_shift_rule_rows(len, SHARDS_CONCURRENT, 8, stride) >= base,
                    "{w}x{h} hbd{hbd}: rows rule went finer than the block rule"
                );
                // Would the unclamped target have been finer? If it never is,
                // the assertion above cannot fail and is decoration.
                let want = (ROWS_PER_BLOCK_MIN as u64)
                    .saturating_mul(stride as u64)
                    .max(1);
                let rows_shift = u64::BITS - (want - 1).leading_zeros();
                let cap_shift =
                    u64::BITS - 1 - ((len as u64 / MIN_BLOCKS as u64).max(1)).leading_zeros();
                if rows_shift.min(cap_shift) < base {
                    narrower += 1;
                }
            }
        }
        assert!(
            narrower > 0,
            "no cell in the list has an unclamped target finer than the block \
             rule, so the never-finer assertion above is vacuous"
        );
        // An undeclared stride keeps the shipped rule EXACTLY — this is what
        // stops every non-picture buffer moving under the arm.
        let (len, _) = plane(1024, 576, 0);
        assert_eq!(
            block_shift_rule_rows(len, SHARDS_CONCURRENT, 8, 0),
            block_shift_rule(len, SHARDS_CONCURRENT, 8)
        );
        // And the arm must actually DO something on the cell it was fitted for,
        // or every assertion above is vacuous.
        let (len, stride) = plane(1024, 576, 0);
        assert!(
            block_shift_rule_rows(len, SHARDS_CONCURRENT, 8, stride)
                > block_shift_rule(len, SHARDS_CONCURRENT, 8),
            "the rows rule is inert on 1024x576, where the sweep measured 0.86x"
        );
        // The gates are shared: one tile, or one thread, and nothing moves.
        assert_eq!(
            block_shift_rule_rows(len, SHARDS_CONCURRENT, 1, stride),
            block_shift_rule(len, SHARDS_CONCURRENT, 1)
        );
        assert_eq!(
            block_shift_rule_rows(len, SHARDS_SERIAL, 8, stride),
            block_shift_rule(len, SHARDS_SERIAL, 8)
        );
    }

    /// The SEAM, not the rule: declaring a stride must actually install the
    /// derived shift on the tracker in the DEFAULT build, and must install the
    /// block-count one when a ladder rung is compiled in.
    ///
    /// The rule's own test above drives [`block_shift_rule_rows`] directly, so
    /// it stays green if [`BorrowTracker::set_row_stride`] is re-gated back into
    /// a no-op — which is exactly what this change reverses, so it needs its own
    /// assertion. Both configs assert; neither skips.
    ///
    /// Process-state note: the two latches are monotone process-globals and
    /// other tests in this module already raise them, so this test raises them
    /// itself and then reads them, rather than assuming an initial value.
    #[test]
    fn declaring_a_stride_installs_the_derived_shift() {
        // 1024x576 8-bit luma, the plane the sweep measured: stride
        // `(1024+127)&!127 = 1024`, a multiple of 1024 so `+64` -> 1088;
        // `len = 1088 * ((576+127)&!127) = 1088 * 640`.
        const STRIDE: usize = 1088;
        const LEN: usize = STRIDE * 640;
        set_parallelism(64);
        set_tile_concurrency(8);
        let (shards, tiles) = (active_shards(), tile_concurrency());
        let mut t = BorrowTracker::new(LEN);
        let from_len = t.shift;
        t.set_row_stride(LEN, STRIDE);
        assert_eq!(
            t.shift,
            block_shift_rule_rows(LEN, shards, tiles, STRIDE),
            "set_row_stride did not install the derived shift"
        );
        assert_eq!(from_len, block_shift_rule(LEN, shards, tiles));
        if ROWS_RULE_ACTIVE {
            assert!(
                t.shift > from_len,
                "the default build must coarsen 1024x576 past the block-count \
                 rule ({from_len}), got {}",
                t.shift
            );
        } else {
            assert_eq!(
                t.shift, from_len,
                "a compiled-in rung must leave the block-count shift alone"
            );
        }
    }

    /// Distinct shards a strided access maps to, i.e. the number of cache lines
    /// the tracker touches for it and the quantity `MAX_SHARDS_PER_BORROW` is
    /// compared against. Test-only mirror of what `add_multi` walks.
    #[cfg(test)]
    fn strided_shards(lo: usize, w: usize, rows: usize, stride: usize, shift: u32) -> usize {
        let mut set = std::vec::Vec::new();
        for i in 0..rows {
            let a0 = (lo + i * stride) >> shift;
            let a1 = (lo + i * stride + w - 1) >> shift;
            for b in a0..=a1 {
                let s = shard_of(b, N_SHARDS - 1);
                if !set.contains(&s) {
                    set.push(s);
                }
            }
        }
        set.len()
    }

    /// **The granularity ladder IS the whole shard-set lever, and "keep a short
    /// RUN of consecutive blocks on one shard" is not a second one.**
    ///
    /// A run mapping groups `2^k` consecutive blocks by hashing `block >> k`.
    /// But `block == addr >> shift`, so `block >> k == addr >> (shift + k)` — the
    /// grouped index IS the block index at a shift `k` coarser. Anything that
    /// depends only on an access's SHARD SET (`pct_row_wide`, the shard lines a
    /// strided read touches, the `MAX_SHARDS_PER_BORROW` promotion door) is
    /// therefore already covered by the `__bps_*` rungs, and a separate run
    /// mapping cannot reach a point the ladder does not.
    ///
    /// What this test pins is the CONSEQUENCE for the worst strided shape in the
    /// decoder — `rav1d_prepare_intra_edges`' one-byte-wide left column, 16 rows
    /// at a 4K luma stride: the shard count falls as the block grows and crosses
    /// `MAX_SHARDS_PER_BORROW` between shift 12 and 14.
    ///
    /// **HOW MUCH THIS GUARDS, honestly: it is a derivation-pin, not a guard on
    /// the hash.** Two planted mutations of `shard_of` — a different
    /// multiplicative constant taking the LOW bits, and the same constant taking
    /// bits 32.. instead of 40.. — both left the vector at `[15, 8, 4, 2, 1, 1]`
    /// and the test green. That is not a weak test so much as the point: the
    /// count is dominated by the number of distinct BLOCKS, which is arithmetic,
    /// and among 16 blocks in 128 shards any decent hash collides about once
    /// (birthday: `C(16,2)/128 = 0.94`). So the ladder — not the mapping — is
    /// where the shard set is decided, which is exactly the claim above.
    /// `__shard_ident` is the arm that changes the mapping's LOCALITY rather than
    /// its cardinality, and it is excluded here for that reason.
    #[cfg(not(feature = "__shard_ident"))]
    #[test]
    fn coarser_blocks_collapse_a_strided_access_onto_fewer_shards() {
        // 16 rows, 1 byte each, 3840-byte stride: the 4K left-column read.
        let n: std::vec::Vec<usize> = (12u32..=17)
            .map(|s| strided_shards(0, 1, 16, 3840, s))
            .collect();
        // Non-increasing, and it must actually MOVE — a flat sequence would make
        // the ladder pointless and the test vacuous.
        for w in n.windows(2) {
            assert!(
                w[1] <= w[0],
                "shard count must not grow with the block: {n:?}"
            );
        }
        assert!(n[0] > n[n.len() - 1], "ladder is inert: {n:?}");
        // Pinned values, so a change to `shard_of` shows up here rather than as a
        // silent shift in every granularity conclusion.
        // 15, not 16, at shift 12: two of the sixteen blocks collide under the
        // Fibonacci hash. The doc comment above says "16 distinct shard lines"
        // for this access; the measured number is 15.
        assert_eq!(n, [15, 8, 4, 2, 1, 1], "shard counts by shift 12..=17");
        // ...and the cross-over past the promotion cap is what the ladder buys:
        // at the shipped shift-12 constant this access is over the cap, at 14 it
        // is not.
        assert!(n[0] > MAX_SHARDS_PER_BORROW);
        assert!(n[2] <= MAX_SHARDS_PER_BORROW);
    }

    /// A later single-tile frame must not undo a multi-tile declaration, for
    /// the same reason `set_parallelism` is monotone.
    #[test]
    fn set_tile_concurrency_is_monotone() {
        set_tile_concurrency(8);
        let raised = tile_concurrency();
        assert!(raised >= 8);
        set_tile_concurrency(1);
        assert_eq!(tile_concurrency(), raised);
    }

    /// Every record registered must be retired — no leaks — under maximal
    /// add/remove interleaving on ONE shard.
    ///
    /// This is the test that guards the lock-free [`BorrowTracker::remove`].
    /// Registration publishes with a plain `Release` store while holding the
    /// shard lock; release stores 0 to the same byte while holding nothing.
    /// They race by construction, and the design's claim is that they cannot
    /// LOSE each other because a slot byte has at most one writer at a time
    /// (see [`Shard::live`]). When the two were a `fetch_or` / `fetch_and` pair
    /// on ONE shared byte, writing `publish` the obvious way —
    /// `occupied.store(occ | bit)` from the snapshot the lock holder already
    /// had — would have let a release landing between the load and the store be
    /// silently undone, leaving the slot occupied forever until the shard
    /// overflowed to the wide list and started reporting overlaps against
    /// borrows that had ended. The per-slot flags make that shape unwritable;
    /// this test is what proves the new one does not reintroduce it.
    ///
    /// Every range lives in block 0, so all eight threads hammer the same
    /// shard's flag bytes. Slot exhaustion is real here (8 threads, `SLOTS`
    /// slots), so some registrations legitimately go wide; those retire
    /// through `remove_wide` and must leave the shard clean too.
    ///
    /// This is the gate for the per-slot-flag design's one new failure mode:
    /// publish and release are plain stores, so if the "at most one writer per
    /// slot byte" argument were wrong, a release would be overwritten by a
    /// publish and the slot would stay set forever. That shows up here, and
    /// only here, as a nonzero flag after every thread has joined.
    #[test]
    fn threaded_churn_leaks_no_slots() {
        use std::sync::Arc;
        let t = Arc::new(BorrowTracker::new(1 << 20));
        let mut hs = Vec::new();
        for th in 0..8usize {
            let t = Arc::clone(&t);
            hs.push(std::thread::spawn(move || {
                // Miri interprets every one of these atomics, and TREE
                // BORROWS is where it bites: at the native count the whole lib
                // leg finishes in 1488.78 s under Stacked Borrows but had not
                // finished this test plus `threaded_disjoint_is_clean` after
                // 7_200 s under Tree Borrows (2026-08-09, M4 Pro). Native is
                // unchanged.
                for _ in 0..if cfg!(miri) { 1_500 } else { 50_000usize } {
                    // Disjoint per thread, all inside block 0 => one shard.
                    let id = t.add_mut(&b(th * 4..th * 4 + 4));
                    t.remove(id);
                }
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        for (i, shard) in t.shards.iter().enumerate() {
            for (s, flag) in shard.live.iter().enumerate() {
                assert_eq!(
                    flag.load(Ordering::Relaxed),
                    0,
                    "shard {i} slot {s} leaked: a release was lost"
                );
            }
            // The lock-protected superset must also narrow back to empty, or
            // every later scan on this shard pays for a phantom slot.
            assert_eq!(
                shard.live_mask(unsafe { (*shard.recs.get()).allocated }),
                0,
                "shard {i}: live_mask disagrees with the flags"
            );
        }
        assert_eq!(t.state.load(Ordering::Relaxed), 0, "a wide record leaked");
        // Proof the tracker is still functional rather than merely empty.
        let x = t.add_mut(&b(0..32));
        t.remove(x);
    }

    /// Concurrent registration of disjoint ranges must never report an overlap,
    /// and overlapping ones must always be caught.
    #[test]
    fn threaded_disjoint_is_clean() {
        use std::sync::Arc;
        let t = Arc::new(BorrowTracker::new(1 << 20));
        let mut hs = Vec::new();
        for th in 0..8usize {
            let t = Arc::clone(&t);
            hs.push(std::thread::spawn(move || {
                // See `threaded_churn_leaks_no_slots` for why this is cut
                // under Miri. Native is unchanged.
                for i in 0..if cfg!(miri) { 1_000 } else { 20_000usize } {
                    // Interleave the threads across the address space so they
                    // land in the same shards constantly.
                    let base = (i * 8 + th) * 4;
                    let id = t.add_mut(&b(base..base + 4));
                    t.remove(id);
                }
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
    }

    // =========================================================================
    // Strided-rectangle records
    //
    // Every test here is against a BRUTE-FORCE BYTE-SET oracle, not against a
    // transcription of the predicate under test, and the two liveness tests
    // (a rectangle must NOT collide with a foreign borrow that only touches an
    // inter-row GAP) each carry a non-vacuity assertion: the same pair
    // registered as the HULL is checked to collide, which proves the gap byte
    // really is inside the hull and the pass is not free.
    // =========================================================================

    /// Sorted byte set of a rectangle. The oracle's ground truth.
    fn rect_set(lo: usize, seg: usize, rows: usize, s: usize) -> alloc::vec::Vec<usize> {
        let mut v = alloc::vec::Vec::new();
        for r in 0..rows {
            for k in 0..seg {
                v.push(lo + r * s + k);
            }
        }
        v.sort_unstable();
        v
    }

    fn oracle_rect_vs_range(
        lo: usize,
        seg: usize,
        rows: usize,
        s: usize,
        a: usize,
        b: usize,
    ) -> bool {
        rect_set(lo, seg, rows, s)
            .iter()
            .any(|&x| x >= a && x < b)
    }

    fn hull_of(lo: usize, seg: usize, rows: usize, s: usize) -> (usize, usize) {
        (lo, lo + (rows - 1) * s + seg)
    }

    /// `rect_decode` is the exact inverse of the hull encoding, over a grid that
    /// includes `seg == s` (the widest legal row) and `seg == 1`.
    #[test]
    fn rect_decode_is_the_exact_inverse_of_the_hull() {
        for s in 1..=40usize {
            for rows in 1..=17usize {
                for seg in 1..=s {
                    let (h0, h1) = hull_of(1000, seg, rows, s);
                    assert_eq!(
                        rect_decode(h0, h1, s),
                        (rows, seg),
                        "s={s} rows={rows} seg={seg}"
                    );
                }
            }
        }
    }

    /// `rect_hit_range` against the byte-set oracle, exhaustively over a grid
    /// that includes gap-only probes, probes wider than the stride, probes
    /// clipped by either end of the hull, and probes entirely outside it.
    #[test]
    fn rect_hit_range_matches_a_brute_force_byte_set_oracle() {
        const LO: usize = 64;
        let mut hits = 0usize;
        let mut misses = 0usize;
        for s in [3usize, 4, 7, 16] {
            for rows in 1..=6usize {
                for seg in 1..=s {
                    let (h0, h1) = hull_of(LO, seg, rows, s);
                    for a in (LO - 3)..(h1 + 3) {
                        for len in 0..(2 * s + 3) {
                            let b = a + len;
                            let got = rect_hit_range(h0, h1, s, a, b);
                            let want = oracle_rect_vs_range(LO, seg, rows, s, a, b);
                            assert_eq!(
                                got.is_some(),
                                want,
                                "s={s} rows={rows} seg={seg} probe=[{a},{b})"
                            );
                            if let Some((rs, re)) = got {
                                // The reported extent must be a REAL row
                                // segment, and it must be the one that collided.
                                assert_eq!(re - rs, seg);
                                assert_eq!((rs - h0) % s, 0);
                                assert!(rs < b && a < re);
                                hits += 1;
                            } else {
                                misses += 1;
                            }
                        }
                    }
                }
            }
        }
        // Liveness of the test itself: both outcomes must be exercised in bulk.
        assert!(hits > 1000, "hits={hits}");
        assert!(misses > 1000, "misses={misses}");
    }

    /// A tracker with a declared stride and the FULL shard set.
    ///
    /// `set_parallelism` is load-bearing, not decoration: with no parallelism
    /// declared `mask_for` returns 0, every block maps to shard 0, and
    /// `add_rect`'s multi-shard path — the one a real strided rectangle takes —
    /// is never entered. A `mask == 0` grid would have made every test below
    /// pass while exercising a single lock, which is the shape of vacuous gate
    /// this repo has shipped six times. The assertion fails loudly if the
    /// process state ever stops cooperating.
    fn rect_tracker(len: usize, stride: usize) -> BorrowTracker {
        set_parallelism(N_SHARDS);
        let mut t = BorrowTracker::new(len);
        t.set_row_stride(len, stride);
        assert_eq!(
            t.mask,
            N_SHARDS - 1,
            "the rectangle tests need the full shard set"
        );
        t
    }

    /// The multi-shard rectangle path is REACHED by the grid below, and a
    /// single-shard one is too. Without this, `add_rect`'s sort/lock/scan loop
    /// could be dead code in every test and nothing would say so.
    #[test]
    fn rect_registrations_reach_both_the_one_shard_and_the_multi_shard_path() {
        let s = 256usize;
        let t = rect_tracker(1 << 20, s);
        let bs = 1usize << t.block_shift();
        // Wholly inside one block.
        let one = t.add_rect_immut(0, 16, 4, s).expect("representable");
        assert_eq!(one.pairs(), 1, "a rectangle inside one block is one shard");
        t.remove(one);
        // Straddling a block boundary: two blocks, and `shard_of` is a
        // multiplicative hash, so two distinct shards.
        let lo = bs - s;
        let many = t.add_rect_immut(lo, 16, 4, s).expect("representable");
        assert!(
            many.pairs() > 1,
            "a rectangle straddling a block boundary must register in >1 shard"
        );
        t.remove(many);
    }

    /// The registration a rectangle replaces, as a control: `rows` plain per-row
    /// records over the same bytes.
    fn add_rows_immut(t: &BorrowTracker, lo: usize, seg: usize, rows: usize, s: usize) -> alloc::vec::Vec<BorrowId> {
        (0..rows)
            .map(|r| t.add_immut(&b(lo + r * s..lo + r * s + seg)))
            .collect()
    }

    /// A mutable borrow ON a rectangle's row segment must be caught — the
    /// rectangle record is a real reservation, not a hint.
    #[test]
    #[should_panic(expected = "overlapping DisjointMut")]
    fn rect_vs_range_on_a_row_is_caught() {
        let t = rect_tracker(1 << 20, 256);
        let _r = t.add_rect_immut(4096, 16, 8, 256).expect("representable");
        // Row 5, one byte in.
        let _x = t.add_mut(&b(4096 + 5 * 256 + 1..4096 + 5 * 256 + 2));
    }

    /// The whole point of an exact record: a mutable borrow in an inter-row GAP
    /// must NOT be caught, because the rectangle never reserved it.
    ///
    /// Non-vacuity: the same borrow against the HULL registered as a plain range
    /// IS caught, which proves the gap byte lies inside the hull.
    #[test]
    fn rect_vs_range_in_a_gap_is_permitted_and_the_hull_would_have_refused_it() {
        let (lo, seg, rows, s) = (4096usize, 16usize, 8usize, 256usize);
        let gap = lo + 3 * s + seg + 4; // inside row 3's gap
        {
            let t = rect_tracker(1 << 20, s);
            let r = t.add_rect_immut(lo, seg, rows, s).expect("representable");
            let x = t.add_mut(&b(gap..gap + 4)); // must not panic
            t.remove(x);
            t.remove(r);
        }
        // NON-VACUITY: the hull over the same bytes refuses it.
        let t = rect_tracker(1 << 20, s);
        let (h0, h1) = hull_of(lo, seg, rows, s);
        let _hull = t.add_immut(&b(h0..h1));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _x = t.add_mut(&b(gap..gap + 4));
        }))
        .is_err();
        assert!(
            caught,
            "the gap byte is not inside the hull — the permitting test above is vacuous"
        );
    }

    /// Two rectangles on the SAME ROWS at OVERLAPPING COLUMNS must be caught.
    /// This is the case a per-row scheme catches and a hull scheme also catches;
    /// the exact record must not lose it.
    #[test]
    #[should_panic(expected = "overlapping DisjointMut")]
    fn rect_vs_rect_same_rows_overlapping_columns_is_caught() {
        let t = rect_tracker(1 << 20, 256);
        let _a = t.add_rect_mut(4096, 16, 8, 256).expect("representable");
        // Same rows, columns [8, 24) — overlaps [0, 16) in every row.
        let _c = t.add_rect_immut(4096 + 8, 16, 8, 256);
    }

    /// Two rectangles on the same rows at DISJOINT columns must be permitted —
    /// this is the pair `fill_hull` turns into a false positive, and it is the
    /// routine case under tile threading (two tile columns, same picture rows).
    ///
    /// Non-vacuity: registering either side as its hull refuses the other.
    #[test]
    fn rect_vs_rect_same_rows_disjoint_columns_is_permitted() {
        let (lo, seg, rows, s) = (4096usize, 16usize, 8usize, 256usize);
        {
            let t = rect_tracker(1 << 20, s);
            let a = t.add_rect_mut(lo, seg, rows, s).expect("representable");
            let c = t
                .add_rect_mut(lo + 64, seg, rows, s)
                .expect("representable");
            t.remove(c);
            t.remove(a);
        }
        let t = rect_tracker(1 << 20, s);
        let (h0, h1) = hull_of(lo, seg, rows, s);
        let _hull = t.add_mut(&b(h0..h1));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _c = t.add_rect_mut(lo + 64, seg, rows, s);
        }))
        .is_err();
        assert!(caught, "the two column ranges do not share a hull — vacuous");
    }

    /// Interleaved rows: rectangle A on even rows, B on odd rows, same columns.
    /// Their hulls overlap heavily and their byte sets are disjoint.
    #[test]
    fn rect_vs_rect_interleaved_rows_is_permitted() {
        let s = 256usize;
        let t = rect_tracker(1 << 20, 2 * s);
        let a = t.add_rect_mut(4096, 16, 4, 2 * s).expect("representable");
        let c = t
            .add_rect_mut(4096 + s, 16, 4, 2 * s)
            .expect("representable");
        t.remove(c);
        t.remove(a);
    }

    /// The rectangle-vs-rectangle predicate against the byte-set oracle, over a
    /// grid of offsets and row counts, run through the REAL tracker so that the
    /// shard selection is exercised too.
    #[test]
    fn rect_vs_rect_agrees_with_the_byte_set_oracle_through_the_tracker() {
        const LEN: usize = 1 << 20;
        let s = 64usize;
        let mut checked = 0usize;
        let mut collisions = 0usize;
        for seg_a in [1usize, 7, 32, 64] {
            for seg_c in [1usize, 7, 32, 64] {
                for rows_a in [1usize, 3, 9] {
                    for rows_c in [1usize, 3, 9] {
                        // Signed offsets. With `lo_c >= lo_a` the registrant's
                        // row 0 is always the nearest to A, so a grid of
                        // non-negative offsets can never require a row k > 0 to
                        // be compared — and a mutation that compares only row 0
                        // passes it. (Measured: it did. See the teeth table in
                        // docs/RECT_RECORDS.md.)
                        for doff in 0..(5 * s) {
                            let lo_a = 8192usize;
                            let lo_c = (lo_a + doff) - 2 * s;
                            let want = {
                                let sa = rect_set(lo_a, seg_a, rows_a, s);
                                let sc = rect_set(lo_c, seg_c, rows_c, s);
                                sa.iter().any(|x| sc.binary_search(x).is_ok())
                            };
                            let t = rect_tracker(LEN, s);
                            let a = t
                                .add_rect_mut(lo_a, seg_a, rows_a, s)
                                .expect("representable");
                            let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                t.add_rect_mut(lo_c, seg_c, rows_c, s)
                            }));
                            match got {
                                Ok(id) => {
                                    assert!(
                                        !want,
                                        "missed overlap: A(lo={lo_a},seg={seg_a},rows={rows_a}) \
                                         C(lo={lo_c},seg={seg_c},rows={rows_c}) s={s}"
                                    );
                                    if let Some(id) = id {
                                        t.remove(id);
                                    }
                                }
                                Err(_) => {
                                    assert!(
                                        want,
                                        "false positive: A(lo={lo_a},seg={seg_a},rows={rows_a}) \
                                         C(lo={lo_c},seg={seg_c},rows={rows_c}) s={s}"
                                    );
                                    collisions += 1;
                                    // The tracker is left with A live and
                                    // possibly poisoned; it is dropped here.
                                    return_early(&t, a);
                                    checked += 1;
                                    continue;
                                }
                            }
                            t.remove(a);
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 500, "checked={checked}");
        assert!(collisions > 100, "collisions={collisions}");
    }

    /// Helper so the panic arm above does not leave a borrow registered in a
    /// tracker that is about to be reused (it is not — but being explicit keeps
    /// the loop readable).
    fn return_early(t: &BorrowTracker, id: BorrowId) {
        t.remove(id);
    }

    /// A rectangle whose hull straddles a block boundary registers in every
    /// shard the hull maps to, so an overlap in the LATER block is still caught.
    #[test]
    #[should_panic(expected = "overlapping DisjointMut")]
    fn rect_overlap_in_a_later_block_is_caught() {
        const LEN: usize = 1 << 20;
        let s = 256usize;
        let t = rect_tracker(LEN, s);
        let bs = 1usize << t.block_shift();
        assert!(bs >= s, "block is at least one row here (bs={bs})");
        // Start one row before a block boundary so the hull spans two blocks.
        let lo = bs - s;
        let rows = (bs / s) + 1;
        assert!(rows <= MAX_RECT_ROWS);
        let _r = t.add_rect_immut(lo, 16, rows, s).expect("representable");
        // Last row, inside the second block.
        let last = lo + (rows - 1) * s;
        let _x = t.add_mut(&b(last..last + 4));
    }

    /// Every declining case returns `None` rather than registering something
    /// approximate. A caller that gets `None` takes its own per-row path.
    #[test]
    fn unrepresentable_rectangles_are_declined_not_approximated() {
        const LEN: usize = 1 << 20;
        // No declared stride at all.
        let t = BorrowTracker::new(LEN);
        assert!(t.add_rect_immut(0, 16, 4, 256).is_none());

        let s = 256usize;
        let t = rect_tracker(LEN, s);
        // A stride the tracker does not know about.
        assert!(t.add_rect_immut(0, 16, 4, 128).is_none());
        // Rows would overlap.
        assert!(t.add_rect_immut(0, s + 1, 4, s).is_none());
        // Degenerate.
        assert!(t.add_rect_immut(0, 0, 4, s).is_none());
        assert!(t.add_rect_immut(0, 16, 0, s).is_none());
        // Too tall to compare row-by-row.
        assert!(t.add_rect_immut(0, 16, MAX_RECT_ROWS + 1, s).is_none());
        // A hull spanning more blocks than a borrow may hold shards for. The
        // stride is chosen so the case is REACHABLE within `MAX_RECT_ROWS`
        // rather than silently skipped: at stride 256 and a 4 KiB block it would
        // take 65 rows, one past the cap.
        let wide_s = 1024usize;
        let tw = rect_tracker(LEN, wide_s);
        let bs = 1usize << tw.block_shift();
        let rows = (MAX_SHARDS_PER_BORROW * bs) / wide_s + 2;
        assert!(
            rows <= MAX_RECT_ROWS,
            "the >{MAX_SHARDS_PER_BORROW}-block case must be reachable: \
             rows={rows} bs={bs} stride={wide_s}"
        );
        assert!(
            tw.add_rect_immut(0, 16, rows, wide_s).is_none(),
            "a {rows}-row hull spans more than {MAX_SHARDS_PER_BORROW} blocks of {bs}"
        );
        // ...and one row fewer than the cap needs still registers, so the
        // assertion above is not passing for an unrelated reason.
        let ok_rows = (MAX_SHARDS_PER_BORROW - 1) * bs / wide_s;
        let id = tw
            .add_rect_immut(0, 16, ok_rows, wide_s)
            .expect("just inside the cap");
        tw.remove(id);
        // And the representable one still works, so the assertions above are not
        // all failing for one shared reason.
        let id = t.add_rect_immut(0, 16, 4, s).expect("representable");
        t.remove(id);
    }

    /// A rectangle record is retired like any other: registering and releasing
    /// the same rectangle far more times than there are slots must not leak.
    #[test]
    fn rect_records_are_retired() {
        let s = 256usize;
        let t = rect_tracker(1 << 20, s);
        for _ in 0..(SLOTS * 8) {
            let id = t.add_rect_immut(4096, 16, 8, s).expect("representable");
            t.remove(id);
        }
        // If any slot leaked, a mutable borrow over the same rows would now trip.
        let x = t.add_mut(&b(4096..4096 + 16));
        t.remove(x);
    }

    /// The per-row control and the rectangle must agree on every verdict. This
    /// is the substitution test: replace `rows` records with one and no answer
    /// may change.
    #[test]
    fn a_rectangle_and_its_per_row_control_give_the_same_verdicts() {
        const LEN: usize = 1 << 20;
        let s = 128usize;
        let (lo, seg, rows) = (4096usize, 12usize, 9usize);
        let mut agreed = 0usize;
        let mut refused = 0usize;
        for a in (lo - 4)..(lo + rows * s + 4) {
            for len in [1usize, 5, 13, 100] {
                let probe = a..a + len;
                let via_rect = {
                    let t = rect_tracker(LEN, s);
                    let r = t.add_rect_immut(lo, seg, rows, s).expect("representable");
                    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        t.add_mut(&b(probe.clone()))
                    }));
                    let _ = r;
                    out.is_err()
                };
                let via_rows = {
                    let t = rect_tracker(LEN, s);
                    let ids = add_rows_immut(&t, lo, seg, rows, s);
                    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        t.add_mut(&b(probe.clone()))
                    }));
                    let _ = ids;
                    out.is_err()
                };
                assert_eq!(via_rect, via_rows, "probe={probe:?}");
                agreed += 1;
                if via_rect {
                    refused += 1;
                }
            }
        }
        assert!(agreed > 100, "agreed={agreed}");
        assert!(refused > 20, "refused={refused}");
    }
}
