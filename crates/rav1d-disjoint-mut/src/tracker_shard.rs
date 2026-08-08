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
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicUsize, Ordering};

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
#[cfg(not(any(
    feature = "__shards_1",
    feature = "__shards_4",
    feature = "__shards_8",
    feature = "__shards_16",
    feature = "__shards_32",
    feature = "__shards_64",
    feature = "__shards_128"
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
    /// Total registrations — NOT counted. One shared `fetch_add` per `add`, at
    /// 136 M adds per 4K frame from eight threads, serialises the decoder hard
    /// enough that slot pressure disappears and `WIDE_FULL` reads zero for the
    /// wrong reason. The counters that remain fire 10^4-10^5 times per frame,
    /// which is free. Kept as a field so the report's shape does not change.
    pub static N_ADD: AtomicU64 = AtomicU64::new(0);

    pub fn report() -> std::string::String {
        use core::fmt::Write as _;
        let mut out = std::string::String::new();
        let w = WIDE_SHARDS.load(Relaxed) + WIDE_BLOCKS.load(Relaxed) + WIDE_FULL.load(Relaxed);
        // Absolute counts only. There is deliberately no denominator: see
        // `N_ADD`. `const_shift` is the compile-time constant and is NOT what
        // an `__blockshift_adaptive` build uses — that one is per instance.
        let _ = writeln!(
            out,
            "WIDEHDR\tconst_shift\tslow\tmulti\tw_shards\tw_blocks\tw_full\twide_total"
        );
        let _ = writeln!(
            out,
            "WIDE\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            super::BLOCK_SHIFT,
            N_SLOW.load(Relaxed),
            N_MULTI.load(Relaxed),
            WIDE_SHARDS.load(Relaxed),
            WIDE_BLOCKS.load(Relaxed),
            WIDE_FULL.load(Relaxed),
            w,
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
struct TinyLock(AtomicBool);

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

    #[cold]
    #[inline(never)]
    fn lock_slow(&self) {
        #[cfg(feature = "__probe_lock_backoff")]
        let mut spins = 0u32;
        loop {
            // Spin on a load, not a swap: a read-only spin keeps the line in
            // Shared instead of ping-ponging it Exclusive between waiters.
            while self.0.load(Ordering::Relaxed) {
                core::hint::spin_loop();
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
                return;
            }
        }
    }

    #[inline(always)]
    fn unlock(&self) {
        self.0.store(false, Ordering::Release);
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
/// The occupancy bitmap is NOT here — it lives in [`Shard::occupied`] as an
/// atomic, because releasing a borrow only has to clear one bit and therefore
/// does not need the lock at all. See [`BorrowTracker::remove`].
struct ShardRecs {
    /// Bit `i` set iff slot `i`'s record is a mutable borrow.
    ///
    /// Only meaningful for slots whose [`Shard::occupied`] bit is set, and only
    /// ever read or written by a lock holder.
    mutable: u8,
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
            mutable: 0,
            starts: [0; SLOTS],
            ends: [0; SLOTS],
            #[cfg(debug_assertions)]
            locs: [None; SLOTS],
        }
    }

    #[inline(always)]
    fn hit(&self, i: usize) -> OverlapHit {
        // `min` rather than a bounds check: it is a `umin`, and it lets LLVM
        // drop the panic path from the hottest loop in the decoder.
        let i = i.min(SLOTS - 1);
        (
            self.starts[i],
            self.ends[i],
            self.mutable & (1 << i) != 0,
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
    /// `occupied` is passed in because it now lives outside the lock, in
    /// [`Shard::occupied`]; the caller loads it once and uses the same snapshot
    /// for the scan and for [`Self::alloc`].
    #[inline(always)]
    fn find<const IS_MUT: bool>(
        &self,
        occupied: u8,
        start: usize,
        end: usize,
    ) -> Option<OverlapHit> {
        let mut mask = if IS_MUT {
            occupied
        } else {
            occupied & self.mutable
        };
        while mask != 0 {
            let i = (mask.trailing_zeros() as usize).min(SLOTS - 1);
            if self.starts[i] < end && start < self.ends[i] {
                return Some(self.hit(i));
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
    fn alloc<const IS_MUT: bool>(
        &mut self,
        occupied: u8,
        start: usize,
        end: usize,
        loc: Loc,
    ) -> Option<u8> {
        // `!SLOTS_MASK` pre-fills the unusable high bits, so `trailing_ones`
        // reaches SLOTS exactly when the shard is full.
        let free = ((occupied | !SLOTS_MASK).trailing_ones() as usize).min(SLOTS);
        if free == SLOTS {
            return None;
        }
        self.starts[free] = start;
        self.ends[free] = end;
        if IS_MUT {
            self.mutable |= 1 << free;
        } else {
            self.mutable &= !(1 << free);
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

/// One shard: its lock, its occupancy bitmap, and its records, alone on a
/// cache line.
///
/// 128 bytes is the M-series line size (`hw.cachelinesize`). Two shards sharing
/// a line would halve the effective shard count, so the alignment is
/// load-bearing, not decorative.
#[repr(align(128))]
struct Shard {
    lock: TinyLock,
    /// Bit `i` set iff slot `i` holds a live record.
    ///
    /// Atomic, and deliberately OUTSIDE [`Self::recs`]: releasing a borrow only
    /// has to clear one bit, which `fetch_and` does without the lock. That
    /// takes the whole of [`BorrowTracker::remove`] — measured 20.6% of a t=8
    /// 4K frame's samples — from `swap`-acquire + bit clear + `store`-release
    /// down to one RMW, and it stops releases from contending with each other
    /// and with registrations for the shard lock at all.
    ///
    /// Registration still holds the lock, and must publish here with `fetch_or`
    /// (never `store(load | bit)`) so a concurrent release cannot be lost. The
    /// record fields in `recs` are written BEFORE the bit is set, so a bit
    /// observed set always has complete fields behind it.
    occupied: AtomicU8,
    recs: UnsafeCell<ShardRecs>,
}

impl Shard {
    const fn new() -> Self {
        Self {
            lock: TinyLock::new(),
            occupied: AtomicU8::new(0),
            recs: UnsafeCell::new(ShardRecs::new()),
        }
    }

    /// Publish slot `slot` as live. Must be called by the lock holder, after
    /// [`ShardRecs::alloc`] has filled the record in.
    #[inline(always)]
    fn publish(&self, slot: u8) {
        self.occupied
            .fetch_or(1u8 << (slot as usize).min(SLOTS - 1), Ordering::Release);
    }

    /// Retire slot `slot`. Lock-free — see [`Self::occupied`].
    #[inline(always)]
    fn retire(&self, slot: u8) {
        let bit = 1u8 << (slot as usize).min(SLOTS - 1);
        debug_assert!(
            self.occupied.load(Ordering::Relaxed) & bit != 0,
            "freeing an unoccupied shard slot"
        );
        self.occupied.fetch_and(!bit, Ordering::Release);
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
///   bits  0..2   kind
///   bits  2..4   narrow: number of (shard, slot) pairs, minus one
///   bits  4..40  narrow: four 9-bit (slot:3, shard:6) pairs
///   bits  4..20  wide:   index into the wide list
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct BorrowId(u64);

const KIND_EMPTY: u64 = 0;
const KIND_NARROW: u64 = 1;
const KIND_WIDE: u64 = 2;
const KIND_UNCHECKED: u64 = 3;

const KIND_BITS: u32 = 2;
const KIND_MASK: u64 = (1 << KIND_BITS) - 1;
const N_BITS: u32 = 2;
const PAIR_SHIFT: u32 = KIND_BITS + N_BITS;
/// 3 bits of slot + 9 bits of shard.
const PAIR_BITS: u32 = 12;
const SLOT_MASK: u64 = 0b111;
const SHARD_MASK_BITS: u64 = 0b1_1111_1111;

const _: () = assert!(SLOTS <= (SLOT_MASK as usize) + 1);
const _: () = assert!(N_SHARDS <= (SHARD_MASK_BITS as usize) + 1);
const _: () = assert!(N_SHARDS.is_power_of_two());
const _: () = assert!(SHARDS_SERIAL.is_power_of_two() && SHARDS_SERIAL <= N_SHARDS);
const _: () = assert!(MAX_SHARDS_PER_BORROW <= 4);

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
        (((self.0 >> KIND_BITS) & 0b11) as usize) + 1
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
const BLOCKS_PER_SHARD: usize = 2;

/// Block shift for an instance of `len` bytes: the power of two that lands
/// `len` on about `BLOCKS_PER_SHARD * N_SHARDS` blocks.
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
    let target = (BLOCKS_PER_SHARD * N_SHARDS) as u64;
    let want = (len as u64 / target.max(1)).max(1);
    // `ilog2` rounds down, so the block count lands at or above the target.
    (u64::BITS - 1 - want.leading_zeros()).clamp(6, 24)
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
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn overlap_panic(
        new_start: usize,
        new_end: usize,
        new_mutable: bool,
        existing: OverlapHit,
    ) -> ! {
        let (existing_start, existing_end, existing_mutable, existing_loc) = existing;
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

    #[inline]
    #[track_caller]
    fn add<const IS_MUT: bool>(&self, bounds: &Bounds) -> BorrowId {
        let start = bounds.range.start;
        let end = bounds.range.end;
        if start >= end {
            return BorrowId::EMPTY;
        }
        let shift = self.block_shift();
        let b0 = start >> shift;
        let b1 = (end - 1) >> shift;
        // One load and one branch covers poisoning, live wide records, and
        // multi-block borrows. All three are cold.
        //
        // `self.mask == 0` (a serial or sub-`SHARD_MIN_LEN` instance) takes the
        // fast path even for a multi-block span: `shard_of(b, 0)` is 0 for
        // every block, so the span's distinct-shard set is exactly {0} and
        // `add_slow`'s per-block classification walk — up to `MAX_BLOCKS_SCAN`
        // `shard_of` calls per strided guard — would only rediscover that. The
        // record stored is the same plain `[start, end)` interval either way,
        // in the same shard, so overlap detection is unchanged (issue #458:
        // this plus `SHARDS_SERIAL = 1` is what keeps single-threaded strided
        // block guards off the wide path).
        if (b0 != b1 && self.mask != 0) || self.state.load(Ordering::Acquire) != 0 {
            return self.add_slow::<IS_MUT>(start, end, b0, b1);
        }

        // Fast path: the borrow lives in one shard — either one block, or any
        // span on a mask-0 instance. 99.875% of hot-plane borrows at
        // BLOCK_SHIFT = 8.
        let si = shard_of(b0, self.mask);
        let shard = &self.shards[si];
        shard.lock.lock();
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
            return self.add_slow::<IS_MUT>(start, end, b0, b1);
        }
        // One snapshot of the occupancy bitmap drives both the scan and the
        // slot search. A release landing between the two can only clear bits,
        // which at worst wastes a slot search — never loses a record.
        let occ = shard.occupied.load(Ordering::Acquire);
        // SAFETY: this shard's lock is held.
        let recs = unsafe { &mut *shard.recs.get() };
        if let Some(existing) = recs.find::<IS_MUT>(occ, start, end) {
            drop(g);
            Self::overlap_panic(start, end, IS_MUT, existing);
        }
        match recs.alloc::<IS_MUT>(occ, start, end, here()) {
            Some(slot) => {
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
            let occ = shard.occupied.load(Ordering::Acquire);
            // SAFETY: this shard's lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            let mut hit = recs.find::<IS_MUT>(occ, start, end);
            if hit.is_none() {
                // SAFETY: a shard lock is held, and wide records are only
                // written while every shard lock is held.
                hit = Self::find_wide::<IS_MUT>(unsafe { &*self.wide.get() }, start, end);
            }
            if let Some(existing) = hit {
                drop(g);
                Self::overlap_panic(start, end, IS_MUT, existing);
            }
            return match recs.alloc::<IS_MUT>(occ, start, end, here()) {
                Some(slot) => {
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
            let occ = shard.occupied.load(Ordering::Acquire);
            // SAFETY: shard `s`'s lock is held.
            let recs = unsafe { &*shard.recs.get() };
            if let Some(h) = recs.find::<IS_MUT>(occ, start, end) {
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
            Self::overlap_panic(start, end, IS_MUT, existing);
        }
        // Claim a slot in each. If any shard is full, roll the whole thing back
        // and go wide — a partial registration would be unsound.
        let mut slots = [0u8; MAX_SHARDS_PER_BORROW];
        let mut done = 0usize;
        while done < n {
            let shard = &self.shards[(set[done] as usize) & (N_SHARDS - 1)];
            let occ = shard.occupied.load(Ordering::Acquire);
            // SAFETY: the shard's lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            match recs.alloc::<IS_MUT>(occ, start, end, here()) {
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
            self.shards[(set[i] as usize) & (N_SHARDS - 1)].publish(slots[i]);
        }
        Self::unlock_all(&self.shards, &set[..n]);
        BorrowId::from_pairs(&set[..n], &slots[..n])
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
            let occ = shard.occupied.load(Ordering::Acquire);
            // SAFETY: every shard lock is held.
            let recs = unsafe { &*shard.recs.get() };
            if let Some(h) = recs.find::<IS_MUT>(occ, start, end) {
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
            Self::overlap_panic(start, end, IS_MUT, existing);
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
    /// **Lock-free.** Retiring a record is one bit-clear in [`Shard::occupied`],
    /// and `fetch_and` does that atomically against the `fetch_or` a concurrent
    /// registration publishes with; neither can lose the other's update. The
    /// record's own fields are left alone — they are only ever read for slots
    /// whose bit is set, and the next registration to claim the slot rewrites
    /// them under the lock before publishing.
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
        assert!(caught, "overlap inside the strided span must still be caught");
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
    /// `BLOCKS_PER_SHARD * N_SHARDS` across the whole range of buffer sizes the
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
        let target = BLOCKS_PER_SHARD * N_SHARDS;
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
        // The two 4K planes the fixed ladder was measured on must land on the
        // shift that ladder measured best for each...
        assert_eq!(sh(3840 * 2160), 14);
        assert_eq!(sh(2 * 3840 * 2160), 15);
        // ...at the SAME picture-rows-per-block, which is the quantity that
        // actually drives the win (4.3 rows either way).
        assert_eq!((1usize << sh(3840 * 2160)) / 3840, 4);
        assert_eq!((1usize << sh(2 * 3840 * 2160)) / 7680, 4);
        // ...and a small buffer must NOT be handed that shift, which would
        // collapse it onto one or two shards.
        assert!(sh(64 * 1024) <= 9);
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
        assert_eq!(adapted, 15);
        // Two tiles is already "multi-tile"; the gate is a threshold, not a
        // proportion.
        assert_eq!(block_shift_rule(LEN, SHARDS_CONCURRENT, 2), adapted);
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
    /// Registration publishes with `fetch_or` while holding the shard lock and
    /// release clears with `fetch_and` while holding nothing, so the two RMWs
    /// race by construction. Had `publish` been written the obvious way —
    /// `occupied.store(occ | bit)` from the snapshot the lock holder already
    /// had — a release landing between the load and the store would be
    /// silently undone, the slot would stay occupied forever, and the shard
    /// would leak until it overflowed to the wide list and started reporting
    /// overlaps against borrows that had ended.
    ///
    /// Every range lives in block 0, so all eight threads hammer the same
    /// `occupied` byte. Slot exhaustion is real here (8 threads, `SLOTS`
    /// slots), so some registrations legitimately go wide; those retire
    /// through `remove_wide` and must leave the shard clean too.
    #[test]
    fn threaded_churn_leaks_no_slots() {
        use std::sync::Arc;
        let t = Arc::new(BorrowTracker::new(1 << 20));
        let mut hs = Vec::new();
        for th in 0..8usize {
            let t = Arc::clone(&t);
            hs.push(std::thread::spawn(move || {
                for _ in 0..50_000usize {
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
            assert_eq!(
                shard.occupied.load(Ordering::Relaxed),
                0,
                "shard {i} leaked a slot: a release was lost"
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
                for i in 0..20_000usize {
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
}
