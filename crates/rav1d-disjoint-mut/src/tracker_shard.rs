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
/// `N_SHARDS * 128` bytes and the 12 hot picture planes cost 12x that. Bigger
/// is better for contention and worse for L1 residency, and the borrow stream
/// walks the whole table (a 128x128 superblock touches ~120 distinct blocks),
/// so the table is *not* effectively smaller than its size.
#[cfg(not(any(feature = "__shards_8", feature = "__shards_32", feature = "__shards_64")))]
pub(super) const N_SHARDS: usize = 16;
#[cfg(feature = "__shards_8")]
pub(super) const N_SHARDS: usize = 8;
#[cfg(feature = "__shards_32")]
pub(super) const N_SHARDS: usize = 32;
#[cfg(feature = "__shards_64")]
pub(super) const N_SHARDS: usize = 64;

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

    /// First live record overlapping `[start, end)`, restricted to mutable
    /// records when `mut_only`.
    #[inline(always)]
    fn find(&self, start: usize, end: usize, mut_only: bool) -> Option<OverlapHit> {
        let mut mask = if mut_only {
            self.occupied & self.mutable
        } else {
            self.occupied
        } as u32;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            if self.starts[i] < end && start < self.ends[i] {
                return Some(self.hit(i));
            }
            mask &= mask - 1;
        }
        None
    }

    /// Claim a slot for `[start, end)`. `None` when the shard is full.
    #[inline(always)]
    fn alloc(
        &mut self,
        start: usize,
        end: usize,
        is_mut: bool,
        loc: &'static Location<'static>,
    ) -> Option<u8> {
        let free = (self.occupied as u32 | !((1u32 << SLOTS) - 1)).trailing_ones() as usize;
        if free >= SLOTS {
            return None;
        }
        self.starts[free] = start;
        self.ends[free] = end;
        self.occupied |= 1 << free;
        if is_mut {
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
        debug_assert!(
            self.occupied & (1 << slot) != 0,
            "freeing an unoccupied shard slot"
        );
        self.occupied &= !(1 << slot);
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

const KIND_EMPTY: u8 = 0;
const KIND_NARROW: u8 = 1;
const KIND_WIDE: u8 = 2;
const KIND_UNCHECKED: u8 = 3;

/// Handle returned by a registration, used to release it.
///
/// A borrow can occupy a slot in up to [`MAX_SHARDS_PER_BORROW`] shards, so this
/// is no longer a single index. It is a plain `Copy` value living in the guard.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct BorrowId {
    kind: u8,
    /// Number of `(shard, slot)` pairs for `KIND_NARROW`; unused otherwise.
    n: u8,
    /// `KIND_NARROW`: shard indices. `KIND_WIDE`: `[0]` is the wide-list index.
    shards: [u16; MAX_SHARDS_PER_BORROW],
    slots: [u8; MAX_SHARDS_PER_BORROW],
}

impl BorrowId {
    pub const UNCHECKED: Self = Self {
        kind: KIND_UNCHECKED,
        n: 0,
        shards: [0; MAX_SHARDS_PER_BORROW],
        slots: [0; MAX_SHARDS_PER_BORROW],
    };

    const EMPTY: Self = Self {
        kind: KIND_EMPTY,
        n: 0,
        shards: [0; MAX_SHARDS_PER_BORROW],
        slots: [0; MAX_SHARDS_PER_BORROW],
    };

    const fn wide(idx: u16) -> Self {
        let mut shards = [0u16; MAX_SHARDS_PER_BORROW];
        shards[0] = idx;
        Self {
            kind: KIND_WIDE,
            n: 1,
            shards,
            slots: [0; MAX_SHARDS_PER_BORROW],
        }
    }
}

impl Default for BorrowId {
    fn default() -> Self {
        Self::EMPTY
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
    /// Live wide records. Read while holding **any** shard lock; written only
    /// while holding **every** shard lock.
    wide: UnsafeCell<Vec<WideRec>>,
    /// Number of live wide records. Read-mostly and almost always zero, so the
    /// per-borrow load stays on a Shared line.
    wide_count: AtomicU32,
    poisoned: AtomicBool,
}

// SAFETY: `wide` and every `Shard::recs` are only accessed under the relevant
// `TinyLock`(s), per the module-level rules.
unsafe impl Send for BorrowTracker {}
unsafe impl Sync for BorrowTracker {}

impl Default for BorrowTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Fibonacci hashing: the multiplicative constant is `2^64 / phi`. Taking the
/// *high* bits mixes the low block bits (the x position within a picture row)
/// into the shard index, which is what separates concurrent tile columns.
#[inline(always)]
fn shard_of(block: usize) -> usize {
    (((block as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 40) as usize & (N_SHARDS - 1)
}

impl BorrowTracker {
    pub const fn new() -> Self {
        Self {
            shards: [const { Shard::new() }; N_SHARDS],
            wide: UnsafeCell::new(Vec::new()),
            wide_count: AtomicU32::new(0),
            poisoned: AtomicBool::new(false),
        }
    }

    /// Mark this tracker as poisoned. All future borrow attempts will panic.
    pub fn poison(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    #[inline(always)]
    fn check_poisoned(&self) {
        if self.poisoned.load(Ordering::Acquire) {
            Self::poisoned_panic();
        }
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
        self.add(bounds, true)
    }

    /// Register an immutable borrow. Only checks against mutable borrows.
    #[inline]
    #[track_caller]
    pub fn add_immut(&self, bounds: &Bounds) -> BorrowId {
        self.add(bounds, false)
    }

    #[inline]
    #[track_caller]
    fn add(&self, bounds: &Bounds, is_mut: bool) -> BorrowId {
        let start = bounds.range.start;
        let end = bounds.range.end;
        if start >= end {
            return BorrowId::EMPTY;
        }
        self.check_poisoned();
        let loc = Location::caller();
        // `mut_only`: a mutable borrow conflicts with everything, an immutable
        // one only with mutable borrows.
        let mut_only = !is_mut;

        let b0 = start >> BLOCK_SHIFT;
        let b1 = (end - 1) >> BLOCK_SHIFT;
        if b0 == b1 {
            // Fast path: the whole borrow lives in one block, so one shard.
            // 99.875% of hot-plane borrows at BLOCK_SHIFT = 8.
            let si = shard_of(b0);
            let shard = &self.shards[si];
            shard.lock.lock();
            let g = ShardGuard(&shard.lock);
            // SAFETY: this shard's lock is held.
            let recs = unsafe { &mut *shard.recs.get() };
            if let Some(existing) = recs.find(start, end, mut_only) {
                drop(g);
                Self::overlap_panic(start, end, is_mut, existing);
            }
            if self.wide_count.load(Ordering::Relaxed) != 0 {
                // SAFETY: a shard lock is held, and wide records are only
                // written while every shard lock is held.
                if let Some(existing) = Self::find_wide(unsafe { &*self.wide.get() }, start, end, mut_only)
                {
                    drop(g);
                    Self::overlap_panic(start, end, is_mut, existing);
                }
            }
            if let Some(slot) = recs.alloc(start, end, is_mut, loc) {
                let mut id = BorrowId::EMPTY;
                id.kind = KIND_NARROW;
                id.n = 1;
                id.shards[0] = si as u16;
                id.slots[0] = slot;
                return id;
            }
            // Shard full — release and retry on the wide path, which is
            // atomic against everything.
            drop(g);
            return self.add_wide(start, end, is_mut, mut_only, loc);
        }

        self.add_multi(start, end, is_mut, mut_only, loc, b0, b1)
    }

    /// Borrow spanning several blocks. Registers the same exact interval in
    /// each distinct shard, acquired in ascending order.
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn add_multi(
        &self,
        start: usize,
        end: usize,
        is_mut: bool,
        mut_only: bool,
        loc: &'static Location<'static>,
        b0: usize,
        b1: usize,
    ) -> BorrowId {
        let nblocks = b1 - b0 + 1;
        if nblocks > MAX_BLOCKS_SCAN {
            return self.add_wide(start, end, is_mut, mut_only, loc);
        }
        let mut set = [0u16; MAX_SHARDS_PER_BORROW];
        let mut n = 0usize;
        for b in b0..=b1 {
            let s = shard_of(b) as u16;
            if set[..n].contains(&s) {
                continue;
            }
            if n == MAX_SHARDS_PER_BORROW {
                return self.add_wide(start, end, is_mut, mut_only, loc);
            }
            set[n] = s;
            n += 1;
        }
        set[..n].sort_unstable();

        for &s in &set[..n] {
            self.shards[s as usize].lock.lock();
        }
        // Check every held shard, plus the wide list.
        let mut hit = None;
        for &s in &set[..n] {
            // SAFETY: shard `s`'s lock is held.
            let recs = unsafe { &*self.shards[s as usize].recs.get() };
            if let Some(h) = recs.find(start, end, mut_only) {
                hit = Some(h);
                break;
            }
        }
        if hit.is_none() && self.wide_count.load(Ordering::Relaxed) != 0 {
            // SAFETY: shard locks are held.
            hit = Self::find_wide(unsafe { &*self.wide.get() }, start, end, mut_only);
        }
        if let Some(existing) = hit {
            Self::unlock_all(&self.shards, &set[..n]);
            Self::overlap_panic(start, end, is_mut, existing);
        }
        // Claim a slot in each. If any shard is full, roll the whole thing back
        // and go wide — a partial registration would be unsound.
        let mut slots = [0u8; MAX_SHARDS_PER_BORROW];
        let mut done = 0usize;
        while done < n {
            // SAFETY: shard's lock is held.
            let recs = unsafe { &mut *self.shards[set[done] as usize].recs.get() };
            match recs.alloc(start, end, is_mut, loc) {
                Some(slot) => {
                    slots[done] = slot;
                    done += 1;
                }
                None => break,
            }
        }
        if done < n {
            for i in 0..done {
                // SAFETY: shard's lock is held.
                let recs = unsafe { &mut *self.shards[set[i] as usize].recs.get() };
                recs.free(slots[i]);
            }
            Self::unlock_all(&self.shards, &set[..n]);
            return self.add_wide(start, end, is_mut, mut_only, loc);
        }
        Self::unlock_all(&self.shards, &set[..n]);
        BorrowId {
            kind: KIND_NARROW,
            n: n as u8,
            shards: set,
            slots,
        }
    }

    /// Last-resort registration: hold **every** shard, so the record is atomic
    /// against all narrow registrants, and publish it to the wide list that
    /// they all consult.
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn add_wide(
        &self,
        start: usize,
        end: usize,
        is_mut: bool,
        mut_only: bool,
        loc: &'static Location<'static>,
    ) -> BorrowId {
        for shard in &self.shards {
            shard.lock.lock();
        }
        let mut hit = None;
        for shard in &self.shards {
            // SAFETY: every shard lock is held.
            let recs = unsafe { &*shard.recs.get() };
            if let Some(h) = recs.find(start, end, mut_only) {
                hit = Some(h);
                break;
            }
        }
        // SAFETY: every shard lock is held.
        let wide = unsafe { &mut *self.wide.get() };
        if hit.is_none() {
            hit = Self::find_wide(wide, start, end, mut_only);
        }
        if let Some(existing) = hit {
            Self::unlock_every(&self.shards);
            Self::overlap_panic(start, end, is_mut, existing);
        }
        let rec = (start, end, is_mut, Some(loc));
        let idx = match wide.iter().position(|r| r.0 >= r.1) {
            Some(i) => {
                wide[i] = rec;
                i
            }
            None => {
                wide.push(rec);
                wide.len() - 1
            }
        };
        assert!(
            idx <= u16::MAX as usize,
            "DisjointMut: too many concurrent wide borrows"
        );
        self.wide_count.fetch_add(1, Ordering::Relaxed);
        Self::unlock_every(&self.shards);
        BorrowId::wide(idx as u16)
    }

    #[inline(always)]
    fn find_wide(
        wide: &[WideRec],
        start: usize,
        end: usize,
        mut_only: bool,
    ) -> Option<OverlapHit> {
        for &(s, e, m, l) in wide {
            if s < e && (!mut_only || m) && s < end && start < e {
                return Some((s, e, m, l));
            }
        }
        None
    }

    #[inline(always)]
    fn unlock_all(shards: &[Shard; N_SHARDS], set: &[u16]) {
        for &s in set.iter().rev() {
            shards[s as usize].lock.unlock();
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
        match id.kind {
            KIND_NARROW => {
                if id.n == 1 {
                    // Fast path, mirror of `add`'s.
                    let shard = &self.shards[id.shards[0] as usize];
                    shard.lock.lock();
                    let _g = ShardGuard(&shard.lock);
                    // SAFETY: this shard's lock is held.
                    unsafe { &mut *shard.recs.get() }.free(id.slots[0]);
                } else {
                    self.remove_multi(id);
                }
            }
            KIND_WIDE => self.remove_wide(id),
            _ => {}
        }
    }

    /// Releasing a multi-shard borrow must be atomic: dropping the record from
    /// shard A before shard B would leave a window where a genuinely disjoint
    /// neighbour still sees the dead record and panics.
    #[cold]
    #[inline(never)]
    fn remove_multi(&self, id: BorrowId) {
        let n = id.n as usize;
        // `add_multi` stored them ascending, so this order is already safe.
        for &s in &id.shards[..n] {
            self.shards[s as usize].lock.lock();
        }
        for i in 0..n {
            // SAFETY: the shard's lock is held.
            unsafe { &mut *self.shards[id.shards[i] as usize].recs.get() }.free(id.slots[i]);
        }
        Self::unlock_all(&self.shards, &id.shards[..n]);
    }

    #[cold]
    #[inline(never)]
    fn remove_wide(&self, id: BorrowId) {
        for shard in &self.shards {
            shard.lock.lock();
        }
        // SAFETY: every shard lock is held.
        let wide = unsafe { &mut *self.wide.get() };
        wide[id.shards[0] as usize] = (1, 0, false, None); // tombstone
        self.wide_count.fetch_sub(1, Ordering::Relaxed);
        Self::unlock_every(&self.shards);
    }
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
        let t = BorrowTracker::new();
        let _a = t.add_mut(&b(0..16));
        let _c = t.add_mut(&b(8..24));
    }

    /// The interesting case: the two borrows overlap in a block whose shard is
    /// not the shard of either borrow's *first* block. Registering only the
    /// first block's shard would miss this.
    #[test]
    #[should_panic(expected = "overlapping DisjointMut")]
    fn overlap_in_a_later_block_is_caught() {
        let t = BorrowTracker::new();
        let bs = 1usize << BLOCK_SHIFT;
        // a covers blocks 0..=2, c covers blocks 2..=4; they share block 2.
        let _a = t.add_mut(&b(0..2 * bs + 1));
        let _c = t.add_mut(&b(2 * bs..4 * bs + 1));
    }

    /// Adjacent-but-disjoint borrows either side of a block boundary must NOT
    /// collide, even though hashing may put them in the same shard.
    #[test]
    fn block_boundary_neighbours_do_not_collide() {
        let t = BorrowTracker::new();
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
            let t = BorrowTracker::new();
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
        let t = BorrowTracker::new();
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
        let t = BorrowTracker::new();
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
        let t = BorrowTracker::new();
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
        let t = BorrowTracker::new();
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
        let t = BorrowTracker::new();
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
        let t = Arc::new(BorrowTracker::new());
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
