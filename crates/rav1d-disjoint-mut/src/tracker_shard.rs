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
//! * The buffer is cut into fixed `1 << BLOCK_SHIFT`-byte blocks.
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
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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
/// 32 is the largest count that costs nothing single-threaded. 64 buys another
/// 9% at t=8 for 4% at t=1 and is available as `shards-64`. 256 is not offered:
/// at 32 KiB the tracker makes `DisjointMut` too large to build on a worker
/// stack, which is the hard ceiling on this inline-array design.
#[cfg(not(any(
    feature = "__shards_1",
    feature = "__shards_4",
    feature = "__shards_8",
    feature = "__shards_16",
    feature = "__shards_64",
    feature = "__shards_128"
)))]
pub(super) const N_SHARDS: usize = 32;
#[cfg(feature = "__shards_1")]
pub(super) const N_SHARDS: usize = 1;
#[cfg(feature = "__shards_4")]
pub(super) const N_SHARDS: usize = 4;
#[cfg(feature = "__shards_8")]
pub(super) const N_SHARDS: usize = 8;
#[cfg(feature = "__shards_16")]
pub(super) const N_SHARDS: usize = 16;
#[cfg(feature = "__shards_64")]
pub(super) const N_SHARDS: usize = 64;
#[cfg(feature = "__shards_128")]
pub(super) const N_SHARDS: usize = 128;

/// `log2` of the block size in elements.
///
/// Small enough that two tile columns working the same picture row land in
/// different blocks (at 4K/4 columns they are 960 bytes apart), large enough
/// that essentially every borrow fits in one block. Measured borrow-length
/// distribution on the hot planes (v4k_8tile 8bpc, `benchmarks/
/// shard_sizing_2026-08-07.txt`): 77.3% are a single byte, 99.94% are <= 31
/// bytes, and at this shift 99.875% span exactly one block.
#[cfg(not(any(feature = "__blockshift_10", feature = "__blockshift_12")))]
const BLOCK_SHIFT: u32 = 8;
#[cfg(feature = "__blockshift_10")]
const BLOCK_SHIFT: u32 = 10;
#[cfg(feature = "__blockshift_12")]
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

/// The registration site carried into `alloc`. Debug builds propagate
/// `#[track_caller]` all the way from the borrow site and store it; release
/// builds do not (the wrapper's own line is not a useful diagnostic), so this
/// degrades to a zero-sized value and the store disappears.
#[cfg(debug_assertions)]
type Loc = &'static Location<'static>;
#[cfg(not(debug_assertions))]
type Loc = ();

/// Cheap in release, the real site in debug. See [`Loc`].
#[cfg(debug_assertions)]
#[inline(always)]
#[track_caller]
fn here() -> Loc {
    Location::caller()
}
#[cfg(not(debug_assertions))]
#[inline(always)]
fn here() -> Loc {}

/// A borrow touching more distinct shards than this goes to the wide list
/// instead. Measured 0.009% of hot borrows at `BLOCK_SHIFT == 8`.
const MAX_SHARDS_PER_BORROW: usize = 4;

/// Blocks scanned before giving up and going wide. Bounds the fast path's work
/// for a pathologically long borrow (e.g. an unbounded `index(..)`).
const MAX_BLOCKS_SCAN: usize = 64;

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
        loop {
            // Spin on a load, not a swap: a read-only spin keeps the line in
            // Shared instead of ping-ponging it Exclusive between waiters.
            while self.0.load(Ordering::Relaxed) {
                core::hint::spin_loop();
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
struct ShardRecs {
    /// Bit `i` set iff slot `i` holds a live record.
    occupied: u8,
    /// Bit `i` set iff slot `i`'s record is a mutable borrow.
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
            occupied: 0,
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
    #[inline(always)]
    fn find<const IS_MUT: bool>(&self, start: usize, end: usize) -> Option<OverlapHit> {
        let mut mask = if IS_MUT {
            self.occupied
        } else {
            self.occupied & self.mutable
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

    /// Claim a slot for `[start, end)`. `None` when the shard is full.
    #[inline(always)]
    fn alloc<const IS_MUT: bool>(&mut self, start: usize, end: usize, loc: Loc) -> Option<u8> {
        // `!SLOTS_MASK` pre-fills the unusable high bits, so `trailing_ones`
        // reaches SLOTS exactly when the shard is full.
        let free = ((self.occupied | !SLOTS_MASK).trailing_ones() as usize).min(SLOTS);
        if free == SLOTS {
            return None;
        }
        self.starts[free] = start;
        self.ends[free] = end;
        self.occupied |= 1 << free;
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

    #[inline(always)]
    fn free(&mut self, slot: u8) {
        let bit = 1u8 << (slot as usize).min(SLOTS - 1);
        debug_assert!(self.occupied & bit != 0, "freeing an unoccupied shard slot");
        self.occupied &= !bit;
    }
}

/// One shard: its lock and its records, alone on a cache line.
///
/// 128 bytes is the M-series line size (`hw.cachelinesize`). Two shards sharing
/// a line would halve the effective shard count, so the alignment is
/// load-bearing, not decorative.
#[repr(align(128))]
struct Shard {
    lock: TinyLock,
    recs: UnsafeCell<ShardRecs>,
}

impl Shard {
    const fn new() -> Self {
        Self {
            lock: TinyLock::new(),
            recs: UnsafeCell::new(ShardRecs::new()),
        }
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
const _: () = assert!(MAX_SHARDS_PER_BORROW <= 4);

impl BorrowId {
    pub const UNCHECKED: Self = Self(KIND_UNCHECKED);

    const EMPTY: Self = Self(KIND_EMPTY);

    #[inline(always)]
    const fn narrow1(shard: usize, slot: u8) -> Self {
        Self(
            KIND_NARROW
                | ((((shard as u64) << 3) | (slot as u64 & SLOT_MASK)) << PAIR_SHIFT),
        )
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
        (((f >> 3) as usize) & mask, (f & SLOT_MASK) as u8)
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
#[inline(always)]
fn shard_of(block: usize, mask: usize) -> usize {
    // The second `&` is with a constant, which is what lets LLVM prove the
    // result indexes `[Shard; N_SHARDS]` in bounds. `mask` alone is a runtime
    // value it cannot bound.
    ((((block as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 40) as usize & mask)
        & (N_SHARDS - 1)
}

/// `N_SHARDS - 1` when the buffer is worth spreading, else `0`.
#[inline]
fn mask_for(len: usize) -> usize {
    if len >= SHARD_MIN_LEN { N_SHARDS - 1 } else { 0 }
}

impl BorrowTracker {
    pub fn new(len: usize) -> Self {
        Self {
            shards: [const { Shard::new() }; N_SHARDS],
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
        // Only the mask moves; the shards are already there, and `&mut self`
        // guarantees every one of them is empty.
        self.mask = mask_for(len);
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
        let b0 = start >> BLOCK_SHIFT;
        let b1 = (end - 1) >> BLOCK_SHIFT;
        // One load and one branch covers poisoning, live wide records, and
        // multi-block borrows. All three are cold.
        if b0 != b1 || self.state.load(Ordering::Acquire) != 0 {
            return self.add_slow::<IS_MUT>(start, end, b0, b1);
        }

        // Fast path: the borrow lives in one block, so one shard. 99.875% of
        // hot-plane borrows at BLOCK_SHIFT = 8.
        let si = shard_of(b0, self.mask);
        let shard = &self.shards[si];
        shard.lock.lock();
        let g = ShardGuard(&shard.lock);
        // SAFETY: this shard's lock is held.
        let recs = unsafe { &mut *shard.recs.get() };
        if let Some(existing) = recs.find::<IS_MUT>(start, end) {
            drop(g);
            Self::overlap_panic(start, end, IS_MUT, existing);
        }
        match recs.alloc::<IS_MUT>(start, end, here()) {
            Some(slot) => BorrowId::narrow1(si, slot),
            None => {
                // Shard full — release and retry on the wide path, which is
                // atomic against everything.
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
            let mut hit = recs.find::<IS_MUT>(start, end);
            if hit.is_none() {
                // SAFETY: a shard lock is held, and wide records are only
                // written while every shard lock is held.
                hit = Self::find_wide::<IS_MUT>(unsafe { &*self.wide.get() }, start, end);
            }
            if let Some(existing) = hit {
                drop(g);
                Self::overlap_panic(start, end, IS_MUT, existing);
            }
            return match recs.alloc::<IS_MUT>(start, end, here()) {
                Some(slot) => BorrowId::narrow1(si, slot),
                None => {
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
        let nblocks = b1 - b0 + 1;
        if nblocks > MAX_BLOCKS_SCAN {
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
            // SAFETY: shard `s`'s lock is held.
            let recs = unsafe { &*self.shards[(s as usize) & (N_SHARDS - 1)].recs.get() };
            if let Some(h) = recs.find::<IS_MUT>(start, end) {
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
            // SAFETY: the shard's lock is held.
            let recs = unsafe { &mut *self.shards[(set[done] as usize) & (N_SHARDS - 1)].recs.get() };
            match recs.alloc::<IS_MUT>(start, end, here()) {
                Some(slot) => {
                    slots[done] = slot;
                    done += 1;
                }
                None => break,
            }
        }
        if done < n {
            for i in 0..done {
                // SAFETY: the shard's lock is held.
                let recs = unsafe { &mut *self.shards[(set[i] as usize) & (N_SHARDS - 1)].recs.get() };
                recs.free(slots[i]);
            }
            Self::unlock_all(&self.shards, &set[..n]);
            return self.add_wide::<IS_MUT>(start, end);
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
        for shard in &self.shards {
            shard.lock.lock();
        }
        let mut hit = None;
        for shard in &self.shards {
            // SAFETY: every shard lock is held.
            let recs = unsafe { &*shard.recs.get() };
            if let Some(h) = recs.find::<IS_MUT>(start, end) {
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
            Self::unlock_every(&self.shards);
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
        Self::unlock_every(&self.shards);
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
    fn unlock_all(shards: &[Shard; N_SHARDS], set: &[u16]) {
        for &s in set.iter().rev() {
            shards[(s as usize) & (N_SHARDS - 1)].lock.unlock();
        }
    }

    #[inline(always)]
    fn unlock_every(shards: &[Shard; N_SHARDS]) {
        for shard in shards.iter().rev() {
            shard.lock.unlock();
        }
    }

    /// Release a borrow.
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
        let shard = &self.shards[si];
        shard.lock.lock();
        let _g = ShardGuard(&shard.lock);
        // SAFETY: this shard's lock is held.
        unsafe { &mut *shard.recs.get() }.free(slot);
    }

    /// Releasing a multi-shard borrow must be atomic: dropping the record from
    /// shard A before shard B would leave a window in which a genuinely
    /// disjoint neighbour still sees the dead record and panics.
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
            // SAFETY: the shard's lock is held.
            unsafe { &mut *self.shards[si].recs.get() }.free(slot);
        }
        for i in (0..n).rev() {
            self.shards[id.pair(i, self.mask).0].lock.unlock();
        }
    }

    #[cold]
    #[inline(never)]
    fn remove_wide(&self, id: BorrowId) {
        for shard in &self.shards {
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
        Self::unlock_every(&self.shards);
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
        let bs = 1usize << BLOCK_SHIFT;
        // a covers blocks 0..=2, c covers blocks 2..=4; they share block 2.
        let _a = t.add_mut(&b(0..2 * bs + 1));
        let _c = t.add_mut(&b(2 * bs..4 * bs + 1));
    }

    /// Adjacent-but-disjoint borrows either side of a block boundary must NOT
    /// collide, even though hashing may put them in the same shard.
    #[test]
    fn block_boundary_neighbours_do_not_collide() {
        let t = BorrowTracker::new(1 << 20);
        let bs = 1usize << BLOCK_SHIFT;
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
        let bs = 1usize << BLOCK_SHIFT;
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
        assert!(caught, "mutable borrow overlapping two immutables not caught");
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
        let bs = 1usize << BLOCK_SHIFT;
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
        let bs = 1usize << BLOCK_SHIFT;
        let n = t.add_mut(&b(5 * bs..5 * bs + 4));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _w = t.add_mut(&b(0..MAX_BLOCKS_SCAN * bs * 4));
        }))
        .is_err();
        assert!(caught, "wide registrant missed a live narrow record");
        t.remove(n);
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
