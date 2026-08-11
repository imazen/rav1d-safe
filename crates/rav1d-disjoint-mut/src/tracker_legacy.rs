//! Legacy single-lock borrow tracker (pre-sharding).
//!
//! Kept so the throwaway `__probe_*` decomposition arms from the 2026-08-07
//! recon (`benchmarks/tracker_decomp_2026-08-07.meta`) stay reproducible: those
//! arms measure THIS tracker, which is the baseline the sharded one is scored
//! against. Selected only when a `__probe_*` feature is on; the default build
//! uses [`crate::tracker_shard`].

use super::*;
use core::panic::Location;
use core::sync::atomic::{AtomicBool, Ordering};

/// Lightweight spinlock for borrow tracking.
///
/// Uses a simple `AtomicBool::swap` for the lock — cheaper than
/// `compare_exchange` on the uncontended fast path because `swap`
/// is an unconditional store (no branch on old value). For the
/// single-threaded case (rav1d `threads=1`), this never spins.
///
struct TinyLock(AtomicBool);

impl TinyLock {
    const fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    #[inline(always)]
    fn lock(&self) -> TinyGuard<'_> {
        // Fast path: swap is cheaper than compare_exchange for uncontended locks.
        // On x86_64, `xchg` is simpler than `lock cmpxchg`.
        if self.0.swap(true, Ordering::Acquire) {
            // Contended — spin. This should essentially never happen in rav1d.
            self.lock_slow();
        }
        TinyGuard(&self.0)
    }

    /// THROWAWAY probe variant: also reports whether the acquisition was
    /// contended and, if so, how many nanoseconds it spun. The uncontended
    /// fast path takes no clock reading at all, so it is undistorted.
    #[cfg(feature = "__probe_count")]
    #[inline(always)]
    fn lock_probe(&self) -> (TinyGuard<'_>, u64, bool) {
        if self.0.swap(true, Ordering::Acquire) {
            let t0 = std::time::Instant::now();
            self.lock_slow();
            let ns = t0.elapsed().as_nanos() as u64;
            return (TinyGuard(&self.0), ns, true);
        }
        (TinyGuard(&self.0), 0, false)
    }

    #[cold]
    #[inline(never)]
    fn lock_slow(&self) {
        loop {
            // Spin-wait: read without writing to avoid cache line bouncing
            while self.0.load(Ordering::Relaxed) {
                core::hint::spin_loop();
            }
            if !self.0.swap(true, Ordering::Acquire) {
                return;
            }
        }
    }
}

struct TinyGuard<'a>(&'a AtomicBool);

impl<'a> Drop for TinyGuard<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Inline capacity: 64 slots with a u64 bitmask for O(1) free-slot finding.
const INLINE_SLOTS: usize = 64;

/// A unique identifier for a borrow registration.
///
/// Encoding:
/// - `0..63`: inline slot index
/// - `64..254`: overflow Vec index (value - INLINE_SLOTS)
/// - `EMPTY_SLOT` (254): empty-range borrow, no real slot
/// - `UNCHECKED` (255): unchecked guard, no tracking
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct BorrowId(u8);

impl BorrowId {
    /// Sentinel value for unchecked guards (no tracking).
    pub const UNCHECKED: Self = Self(u8::MAX);
}

/// Borrow record storage with 64 inline slots and Vec overflow.
///
/// The fast path uses a u64 bitmask for O(1) allocation/deallocation.
/// If all 64 inline slots are occupied, borrows spill into a Vec that
/// is allocated on demand. The Vec is never touched if inline capacity
/// suffices.
struct BorrowSlots {
    // Parallel arrays for cache efficiency during overlap scans.
    starts: [usize; INLINE_SLOTS],
    ends: [usize; INLINE_SLOTS],
    mutable: [bool; INLINE_SLOTS],
    /// Registration site of each inline borrow (diagnostics for overlap
    /// panics). With `debug_assertions` the callers propagate
    /// `#[track_caller]`, so this names the true borrow site; in release
    /// it names the `DisjointMut` wrapper method. One pointer store per
    /// borrow — kept unconditionally so release overlap panics still say
    /// *which* wrapper registered the existing side.
    locs: [Option<&'static Location<'static>>; INLINE_SLOTS],
    /// Bitmask of occupied inline slots. Bit `i` set iff slot `i` is active.
    occupied: u64,
    /// Overflow storage, allocated only when >64 concurrent borrows.
    overflow: Vec<(usize, usize, bool, Option<&'static Location<'static>>)>,
}

/// An existing-borrow record returned by the overlap finders:
/// `(start, end, mutable, registration_site)`.
type OverlapHit = (usize, usize, bool, Option<&'static Location<'static>>);

impl BorrowSlots {
    /// Sentinel slot index for empty-range borrows (no real slot allocated).
    const EMPTY_SLOT: u8 = u8::MAX - 1; // 254

    const fn new() -> Self {
        Self {
            starts: [0; INLINE_SLOTS],
            ends: [0; INLINE_SLOTS],
            mutable: [false; INLINE_SLOTS],
            locs: [None; INLINE_SLOTS],
            occupied: 0,
            overflow: Vec::new(),
        }
    }

    /// Allocate a slot and return its BorrowId encoding.
    #[inline(always)]
    fn alloc(
        &mut self,
        start: usize,
        end: usize,
        is_mutable: bool,
        loc: &'static Location<'static>,
    ) -> u8 {
        if start >= end {
            return Self::EMPTY_SLOT;
        }
        let free = self.occupied.trailing_ones() as usize;
        if free < INLINE_SLOTS {
            // Fast path: inline slot available.
            self.starts[free] = start;
            self.ends[free] = end;
            self.mutable[free] = is_mutable;
            self.locs[free] = Some(loc);
            self.occupied |= 1u64 << free;
            free as u8
        } else {
            // Slow path: spill to Vec.
            self.alloc_overflow(start, end, is_mutable, loc)
        }
    }

    /// Overflow allocation — cold path, never inlined.
    #[cold]
    #[inline(never)]
    fn alloc_overflow(
        &mut self,
        start: usize,
        end: usize,
        is_mutable: bool,
        loc: &'static Location<'static>,
    ) -> u8 {
        // Find a free slot in the overflow Vec (tombstoned entries have start >= end).
        for (i, entry) in self.overflow.iter_mut().enumerate() {
            if entry.0 >= entry.1 {
                // Reuse tombstoned slot.
                *entry = (start, end, is_mutable, Some(loc));
                return (INLINE_SLOTS + i) as u8;
            }
        }
        let idx = self.overflow.len();
        // BorrowId is u8, with 254/255 reserved. Max overflow index:
        // INLINE_SLOTS + idx must be < 254, so idx < 254 - 64 = 190.
        assert!(
            INLINE_SLOTS + idx < Self::EMPTY_SLOT as usize,
            "DisjointMut: too many concurrent borrows (max {})",
            Self::EMPTY_SLOT as usize
        );
        self.overflow.push((start, end, is_mutable, Some(loc)));
        (INLINE_SLOTS + idx) as u8
    }

    /// Free a slot by BorrowId encoding.
    #[inline(always)]
    fn free(&mut self, slot: u8) {
        if slot == Self::EMPTY_SLOT {
            return;
        }
        let idx = slot as usize;
        if idx < INLINE_SLOTS {
            debug_assert!(
                (self.occupied & (1u64 << idx)) != 0,
                "BorrowId slot {slot} not occupied"
            );
            self.occupied &= !(1u64 << idx);
        } else {
            // Overflow slot — tombstone it (set start >= end).
            let ov_idx = idx - INLINE_SLOTS;
            debug_assert!(ov_idx < self.overflow.len(), "overflow index out of range");
            self.overflow[ov_idx] = (1, 0, false, None); // tombstone
        }
    }

    /// Check if the range [start, end) overlaps any active borrow.
    #[inline(always)]
    fn find_overlap_any(&self, start: usize, end: usize) -> Option<OverlapHit> {
        if start >= end {
            return None;
        }
        // Scan inline slots.
        let mut mask = self.occupied;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            if self.starts[i] < end && start < self.ends[i] {
                return Some((self.starts[i], self.ends[i], self.mutable[i], self.locs[i]));
            }
            mask &= mask - 1;
        }
        // Scan overflow (cold — only reached if overflow is non-empty).
        if !self.overflow.is_empty() {
            return self.find_overlap_any_overflow(start, end);
        }
        None
    }

    #[cold]
    #[inline(never)]
    fn find_overlap_any_overflow(&self, start: usize, end: usize) -> Option<OverlapHit> {
        for &(s, e, m, l) in &self.overflow {
            if s < e && s < end && start < e {
                return Some((s, e, m, l));
            }
        }
        None
    }

    /// Check if the range [start, end) overlaps any active MUTABLE borrow.
    #[inline(always)]
    fn find_overlap_mut(&self, start: usize, end: usize) -> Option<OverlapHit> {
        if start >= end {
            return None;
        }
        let mut mask = self.occupied;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            if self.mutable[i] && self.starts[i] < end && start < self.ends[i] {
                return Some((self.starts[i], self.ends[i], true, self.locs[i]));
            }
            mask &= mask - 1;
        }
        if !self.overflow.is_empty() {
            return self.find_overlap_mut_overflow(start, end);
        }
        None
    }

    #[cold]
    #[inline(never)]
    fn find_overlap_mut_overflow(&self, start: usize, end: usize) -> Option<OverlapHit> {
        for &(s, e, m, l) in &self.overflow {
            if m && s < e && s < end && start < e {
                return Some((s, e, true, l));
            }
        }
        None
    }
}

/// All active borrows for a single `DisjointMut` instance.
///
/// Uses 64 inline slots with a u64 bitmask for O(1) allocation and
/// deallocation. If more than 64 concurrent borrows are needed, spills
/// to a heap-allocated Vec (up to 254 total). Overlap checking scans
/// only occupied slots.
///
/// Like `std::sync::Mutex`, the tracker poisons the data structure when a
/// thread panics while holding a mutable borrow guard. This prevents
/// subsequent access to potentially corrupted data.
pub(super) struct BorrowTracker {
    lock: TinyLock,
    slots: UnsafeCell<BorrowSlots>,
    poisoned: AtomicBool,
    /// THROWAWAY probe: lazily-assigned per-instance stats slot.
    #[cfg(feature = "__probe_count")]
    probe_slot: core::sync::atomic::AtomicU32,
}

// SAFETY: BorrowSlots is only accessed while TinyLock is held.
unsafe impl Send for BorrowTracker {}
unsafe impl Sync for BorrowTracker {}

impl BorrowTracker {
    pub fn new(_len: usize) -> Self {
        Self {
            lock: TinyLock::new(),
            slots: UnsafeCell::new(BorrowSlots::new()),
            poisoned: AtomicBool::new(false),
            #[cfg(feature = "__probe_count")]
            probe_slot: core::sync::atomic::AtomicU32::new(u32::MAX),
        }
    }

    /// No-op: the legacy tracker's table does not depend on the length.
    pub fn reprovision(&mut self, _len: usize) {}

    /// No-op: with one lock per instance and no block shift, there is no
    /// granularity for a row stride to choose. Present so the picture allocator
    /// can call it unconditionally against either tracker.
    pub fn set_row_stride(&mut self, _len: usize, _stride: usize) {}

    /// Always DECLINES, so every caller takes its per-row path.
    ///
    /// A strided-rectangle record buys nothing here and could only lose: this
    /// tracker is one lock and one 64-slot table per instance, so a registration
    /// costs the same whatever its shape, and the sharded tracker's motivation
    /// (one shard line touched per row) does not exist. Declining keeps the
    /// legacy A/B arm measuring the tracker it names and nothing else.
    #[inline(always)]
    pub fn add_rect_immut(
        &self,
        _lo: usize,
        _seg: usize,
        _rows: usize,
        _stride: usize,
    ) -> Option<BorrowId> {
        None
    }

    /// Mark this tracker as poisoned. All future borrow attempts will panic.
    pub fn poison(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    /// Panic if the tracker has been poisoned.
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
    #[cfg(not(any(
        feature = "__probe_count",
        feature = "__probe_noscan",
        feature = "__probe_lockonly"
    )))]
    pub fn add_mut(&self, bounds: &Bounds) -> BorrowId {
        let start = bounds.range.start;
        let end = bounds.range.end;
        if start >= end {
            return BorrowId(BorrowSlots::EMPTY_SLOT);
        }
        self.check_poisoned();
        let _guard = self.lock.lock();
        // SAFETY: TinyLock is held, so we have exclusive access to slots.
        let slots = unsafe { &mut *self.slots.get() };
        if let Some(existing) = slots.find_overlap_any(start, end) {
            Self::overlap_panic(start, end, true, existing);
        }
        BorrowId(slots.alloc(start, end, true, Location::caller()))
    }

    /// Register an immutable borrow. Only checks against mutable borrows.
    #[inline]
    #[track_caller]
    #[cfg(not(any(
        feature = "__probe_count",
        feature = "__probe_noscan",
        feature = "__probe_lockonly"
    )))]
    pub fn add_immut(&self, bounds: &Bounds) -> BorrowId {
        let start = bounds.range.start;
        let end = bounds.range.end;
        if start >= end {
            return BorrowId(BorrowSlots::EMPTY_SLOT);
        }
        self.check_poisoned();
        let _guard = self.lock.lock();
        // SAFETY: TinyLock is held, so we have exclusive access to slots.
        let slots = unsafe { &mut *self.slots.get() };
        if let Some(existing) = slots.find_overlap_mut(start, end) {
            Self::overlap_panic(start, end, false, existing);
        }
        BorrowId(slots.alloc(start, end, false, Location::caller()))
    }

    #[cfg(any(
        feature = "__probe_count",
        feature = "__probe_noscan",
        feature = "__probe_lockonly"
    ))]
    pub fn add_mut(&self, bounds: &Bounds) -> BorrowId {
        self.add_probed(bounds, true)
    }

    /// Shared body of `add_mut` / `add_immut`, with the throwaway probe
    /// hooks and the noscan / lockonly probe modes folded in so the two
    /// entry points cannot drift apart.
    #[inline]
    #[track_caller]
    #[cfg(any(
        feature = "__probe_count",
        feature = "__probe_noscan",
        feature = "__probe_lockonly"
    ))]
    fn add_probed(&self, bounds: &Bounds, is_mut: bool) -> BorrowId {
        let start = bounds.range.start;
        let end = bounds.range.end;
        if start >= end {
            return BorrowId(BorrowSlots::EMPTY_SLOT);
        }
        self.check_poisoned();

        // Probe mode: take and release the lock, do nothing else. Isolates
        // raw lock traffic from the scan and from slot bookkeeping.
        #[cfg(feature = "__probe_lockonly")]
        {
            let _guard = self.lock.lock();
            return BorrowId(BorrowSlots::EMPTY_SLOT);
        }

        #[cfg(not(feature = "__probe_lockonly"))]
        {
            #[cfg(feature = "__probe_count")]
            let (occupancy, id, wait_ns, contended) = {
                let (_guard, wait_ns, contended) = self.lock.lock_probe();
                // SAFETY: TinyLock is held, so we have exclusive access to slots.
                let slots = unsafe { &mut *self.slots.get() };
                let occupancy = slots.occupied.count_ones();
                #[cfg(not(feature = "__probe_noscan"))]
                {
                    let hit = if is_mut {
                        slots.find_overlap_any(start, end)
                    } else {
                        slots.find_overlap_mut(start, end)
                    };
                    if let Some(existing) = hit {
                        Self::overlap_panic(start, end, is_mut, existing);
                    }
                }
                let id = slots.alloc(start, end, is_mut, Location::caller());
                // Guard drops HERE: every counter update below happens
                // outside the critical section, so the probe cannot inflate
                // the very lock hold time it is measuring.
                (occupancy, id, wait_ns, contended)
            };

            #[cfg(feature = "__probe_count")]
            {
                let slot = crate::probe::assign_slot(&self.probe_slot);
                let spilled = id != BorrowSlots::EMPTY_SLOT && (id as usize) >= INLINE_SLOTS;
                crate::probe::record_add(
                    slot,
                    is_mut,
                    end,
                    occupancy,
                    spilled,
                    wait_ns,
                    contended,
                    Location::caller(),
                );
                #[cfg(feature = "__probe_shardsim")]
                crate::probe::record_shard(
                    slot,
                    start,
                    end,
                    crate::probe::SLOTS[slot]
                        .max_end
                        .load(core::sync::atomic::Ordering::Relaxed),
                );
                return BorrowId(id);
            }

            #[cfg(not(feature = "__probe_count"))]
            {
                let _guard = self.lock.lock();
                // SAFETY: TinyLock is held, so we have exclusive access to slots.
                let slots = unsafe { &mut *self.slots.get() };
                #[cfg(not(feature = "__probe_noscan"))]
                {
                    let hit = if is_mut {
                        slots.find_overlap_any(start, end)
                    } else {
                        slots.find_overlap_mut(start, end)
                    };
                    if let Some(existing) = hit {
                        Self::overlap_panic(start, end, is_mut, existing);
                    }
                }
                BorrowId(slots.alloc(start, end, is_mut, Location::caller()))
            }
        }
    }

    /// Register an immutable borrow. Only checks against mutable borrows.
    #[inline]
    #[track_caller]
    #[cfg(any(
        feature = "__probe_count",
        feature = "__probe_noscan",
        feature = "__probe_lockonly"
    ))]
    pub fn add_immut(&self, bounds: &Bounds) -> BorrowId {
        self.add_probed(bounds, false)
    }

    /// Remove a borrow by slot index. O(1).
    #[inline]
    #[cfg(not(any(feature = "__probe_count", feature = "__probe_lockonly")))]
    pub fn remove(&self, id: BorrowId) {
        if id.0 == BorrowSlots::EMPTY_SLOT || id == BorrowId::UNCHECKED {
            return;
        }
        let _guard = self.lock.lock();
        // SAFETY: TinyLock is held, so we have exclusive access to slots.
        let slots = unsafe { &mut *self.slots.get() };
        slots.free(id.0);
    }

    /// THROWAWAY probe variants of `remove`.
    #[cfg(feature = "__probe_lockonly")]
    #[inline]
    pub fn remove(&self, id: BorrowId) {
        // `add` handed back EMPTY_SLOT for every borrow, so keep the
        // release-side lock traffic symmetric by hand.
        if id == BorrowId::UNCHECKED {
            return;
        }
        let _guard = self.lock.lock();
    }

    #[cfg(all(feature = "__probe_count", not(feature = "__probe_lockonly")))]
    #[inline]
    pub fn remove(&self, id: BorrowId) {
        if id.0 == BorrowSlots::EMPTY_SLOT || id == BorrowId::UNCHECKED {
            return;
        }
        let (wait_ns, contended) = {
            let (_guard, wait_ns, contended) = self.lock.lock_probe();
            // SAFETY: TinyLock is held, so we have exclusive access to slots.
            let slots = unsafe { &mut *self.slots.get() };
            slots.free(id.0);
            (wait_ns, contended)
        };
        let slot = crate::probe::assign_slot(&self.probe_slot);
        crate::probe::record_remove(slot, wait_ns, contended);
    }
}

/// The legacy tracker is a single lock per instance and has no shards, so the
/// parallelism hint has nothing to size. Present only so the `__probe_*` /
/// `__tracker_legacy` arms still compile against the same crate surface.
pub fn set_parallelism(_n: usize) {}

/// Likewise: with no shards and no block shift there is nothing for the tile
/// split to select.
pub fn set_tile_concurrency(_n: usize) {}
