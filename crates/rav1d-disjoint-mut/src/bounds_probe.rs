//! THROWAWAY guard-extent / footprint / concurrency map (feature `__probe_bounds`).
//!
//! Never merge into a shipping configuration. It exists to answer the one
//! question that has now refuted three "widen the reservation to cut the
//! registration count" attempts (#469 strided rectangle, #475 hull arm,
//! #485 loop-filter read band), each discovered only by building the change
//! and running it:
//!
//! > **Does the proposed extent intersect anything another worker is
//! > concurrently writing?**
//!
//! What it records, per guard acquisition:
//!
//! * **site** — `Location::caller()`, exactly as [`crate::site_probe`] does, so
//!   the two instruments' per-site counts are directly comparable (the reason
//!   `__probe_bounds` turns `__probe_sites` on: they run in ONE binary and
//!   reconcile registration for registration).
//! * **reserved extent** — the `[start, end)` the tracker registered.
//! * **true footprint** — the byte ranges actually touched *through that
//!   guard*, and read-vs-write, recorded at the POINT OF USE (`Deref`,
//!   `DerefMut`, and an explicit row-geometry declaration made by the strided
//!   helpers that already compute `(w, h, stride)`).
//! * **liveness** — a global monotone epoch at acquire and at release, plus the
//!   worker id, which is what makes "were these two guards live at the same
//!   instant" decidable.
//!
//! # What the footprint is, and is NOT
//!
//! Three cases, and the report labels each site with which one it is:
//!
//! * `rows` — the site declared `(base, w, rows, stride)`. **Exact**: the
//!   footprint is those `rows` spans of `w` bytes, and everything the
//!   reservation covers beyond them is measured waste.
//! * `whole` — the guard was `Deref`/`DerefMut`-ed and made no declaration, so
//!   the footprint is taken to be the whole reservation. This is an **UPPER
//!   BOUND** on what was touched: at these sites over-reservation reads as
//!   zero, which is right for the many guards that are exactly their access
//!   (`slice_mut(len)` followed by `copy_from_slice`) and an under-report for
//!   any that are not.
//! * `none` — the guard was never dereferenced. The footprint is empty; the
//!   guard bought exclusion only.
//!
//! Sub-reservation WRITE sets are not measured below the `Deref` granularity.
//! For the conflict question that is the safe direction: a mutable reservation
//! is a superset of the bytes it writes, so testing a proposed extent against
//! foreign *reservations* can only over-predict a conflict, never miss one.
//!
//! # This is an empirical map, not a proof
//!
//! Everything here is the union over the executions that were actually run. A
//! site that never appears is **unknown**, not safe.
//!
//! # It perturbs timing
//!
//! Publishing a live record, a `SeqCst` fence and a cross-worker scan happen on
//! every registration. No wall-clock number may be quoted from a build with
//! this feature on.

use core::panic::Location;
use core::sync::atomic::AtomicI64;
use core::sync::atomic::AtomicU32;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering::AcqRel;
use core::sync::atomic::Ordering::Acquire;
use core::sync::atomic::Ordering::Relaxed;
use core::sync::atomic::Ordering::Release;
use core::sync::atomic::fence;
use std::string::String;
use std::vec::Vec;

/// Worker slots. Slots are RECYCLED when a thread exits, because a corpus run
/// creates one decoder (and its whole worker pool) per vector — thousands of
/// thread lifetimes, at most a handful alive at once. Without recycling the
/// 700th vector's workers get no slot and every registration is dropped.
pub const MAX_TH: usize = 64;
/// Live guards per worker. `block_mut_held` can hold 16; nesting adds a few.
const MAX_LIVE: usize = 64;
/// Distinct source sites. The 4K vectors show 48-49; the corpus adds a few.
const NSITES: usize = 512;
/// Distinct `DisjointMut` instances that get a dense id.
///
/// This has to be GENEROUS. At 1024 the table saturated on a 4K t=8 decode,
/// `inst_id` handed back its `u32::MAX` "unknown" sentinel for everything past
/// the fill point, and the scan's `f_inst != inst` filter then matched every
/// saturated instance against every other — reporting 70 concurrent
/// reservation overlaps that the tracker would have panicked on. The decoder
/// creates thousands of small instances (per-`BlockContext` `CaseSet` arrays,
/// per-worker recon bands, scratch); 64 K costs 2 MB of BSS.
const NINST: usize = 1 << 16;
/// Per-worker site-pair table (open addressed, power of two).
const NPAIR: usize = 4096;
/// Gap histogram buckets, in bytes: 0 (touching/overlapping), <=4, <=16, <=64,
/// <=256, <=1K, <=4K, <=16K, <=64K, <=1M, >1M, and "no concurrent foreign
/// record at all". Powers of four so a proposed widening of k bytes can be
/// priced by summing every bucket below k.
pub const NGAP: usize = 12;

const F_READ: u32 = 1;
const F_WRITE: u32 = 2;
const F_ROWS: u32 = 4;

#[inline]
fn gap_bucket(g: u64) -> usize {
    match g {
        0 => 0,
        1..=4 => 1,
        5..=16 => 2,
        17..=64 => 3,
        65..=256 => 4,
        257..=1024 => 5,
        1025..=4096 => 6,
        4097..=16384 => 7,
        16385..=65536 => 8,
        65537..=1048576 => 9,
        _ => 10,
    }
}

/// Column labels for the gap histogram, in the same order as [`gap_bucket`].
pub const GAP_LABELS: [&str; NGAP] = [
    "g0", "g4", "g16", "g64", "g256", "g1k", "g4k", "g16k", "g64k", "g1m", "gbig", "gnone",
];

// =============================================================================
// Live set: what every worker currently holds, published for cross-worker scan
// =============================================================================

#[repr(align(128))]
struct LiveSlot {
    /// Seqlock AND liveness in one word: **odd = dead or mid-write, even =
    /// live and stable**. Folding liveness in here (rather than relying on the
    /// owner's `mask`) is what stops the scan reporting overlaps the tracker
    /// would have panicked on: the mask lives in a different object, so a
    /// scanner could read a stale set bit and then read the slot's still-intact
    /// fields, and nothing in the read invalidated it.
    ver: AtomicU64,
    inst: AtomicU32,
    site: AtomicU32,
    ismut: AtomicU32,
    flags: AtomicU32,
    start: AtomicU64,
    end: AtomicU64,
    /// Declared footprint: `rows` spans of `w` bytes at `stride`, first at `lo`.
    fp_lo: AtomicU64,
    fp_w: AtomicU64,
    fp_rows: AtomicU64,
    fp_stride: AtomicI64,
    acq: AtomicU64,
    /// Release epoch, or 0 while the record is live. A scanner that reaches a
    /// slot whose mask bit it read a moment ago must consult this: the owner
    /// may have retired it in between, and a retired record that closed BEFORE
    /// the scanner's own acquire epoch was never concurrent with it. Without
    /// this the scan reports overlaps that the tracker would have panicked on,
    /// which is how the instrument gets caught lying.
    rel: AtomicU64,
}

impl LiveSlot {
    const fn new() -> Self {
        Self {
            ver: AtomicU64::new(1),
            inst: AtomicU32::new(0),
            site: AtomicU32::new(0),
            ismut: AtomicU32::new(0),
            flags: AtomicU32::new(0),
            start: AtomicU64::new(0),
            end: AtomicU64::new(0),
            fp_lo: AtomicU64::new(0),
            fp_w: AtomicU64::new(0),
            fp_rows: AtomicU64::new(0),
            fp_stride: AtomicI64::new(0),
            acq: AtomicU64::new(0),
            rel: AtomicU64::new(0),
        }
    }
}

#[repr(align(128))]
struct ThreadLive {
    mask: AtomicU64,
    slots: [LiveSlot; MAX_LIVE],
}

impl ThreadLive {
    const fn new() -> Self {
        Self {
            mask: AtomicU64::new(0),
            slots: [const { LiveSlot::new() }; MAX_LIVE],
        }
    }
}

static LIVE: [ThreadLive; MAX_TH] = [const { ThreadLive::new() }; MAX_TH];

/// Monotone acquire/release epoch. Both events bump it, so two guards were live
/// at the same instant iff `acq_a < rel_b && acq_b < rel_a`.
static EPOCH: AtomicU64 = AtomicU64::new(1);

/// Registrations dropped because a worker ran out of live slots (must stay 0).
pub static LOST_SLOT: AtomicU64 = AtomicU64::new(0);
/// Registrations dropped because more than [`MAX_TH`] workers appeared.
pub static LOST_TH: AtomicU64 = AtomicU64::new(0);
/// Registrations dropped because the site table filled (must stay 0).
pub static LOST_SITE: AtomicU64 = AtomicU64::new(0);
/// Foreign live records skipped because their seqlock read raced.
pub static LOST_SCAN: AtomicU64 = AtomicU64::new(0);
/// Instances that did not fit the interning table. MUST stay 0: the sentinel
/// they fall back to is not an identity, and comparing two of them is what
/// produced 70 phantom overlaps before `NINST` was raised.
pub static LOST_INST: AtomicU64 = AtomicU64::new(0);
/// Foreign records the mask still listed but that had already retired before
/// this acquisition's epoch. Not a loss — a correctly rejected stale hit.
pub static SKIP_DEAD: AtomicU64 = AtomicU64::new(0);

/// Bit `i` set == worker slot `i` is claimed by a live thread. The scan walks
/// only the set bits, so an idle fleet costs one load.
static SLOTS_INUSE: AtomicU64 = AtomicU64::new(0);
/// Threads that found every slot claimed. MUST stay 0.
pub static LOST_TID: AtomicU64 = AtomicU64::new(0);

struct TidSlot(usize);

impl Drop for TidSlot {
    fn drop(&mut self) {
        if self.0 < MAX_TH {
            // Guards are all released before a thread exits, so the slot's
            // live mask is already empty.
            SLOTS_INUSE.fetch_and(!(1u64 << self.0), Release);
        }
    }
}

fn claim_tid() -> TidSlot {
    loop {
        let cur = SLOTS_INUSE.load(Relaxed);
        let slot = (!cur).trailing_zeros() as usize;
        if slot >= MAX_TH {
            LOST_TID.fetch_add(1, Relaxed);
            return TidSlot(usize::MAX);
        }
        if SLOTS_INUSE
            .compare_exchange_weak(cur, cur | (1u64 << slot), AcqRel, Relaxed)
            .is_ok()
        {
            LIVE[slot].mask.store(0, Release);
            return TidSlot(slot);
        }
    }
}

std::thread_local! {
    static TID: TidSlot = claim_tid();
}

#[inline]
fn tid() -> usize {
    TID.with(|t| t.0)
}

// =============================================================================
// Site interning (same key as `site_probe`: the `&'static Location` POINTER)
// =============================================================================

static SITE_KEY: [AtomicUsize; NSITES] = [const { AtomicUsize::new(0) }; NSITES];
static SITE_NAMES: std::sync::Mutex<Vec<(usize, &'static Location<'static>)>> =
    std::sync::Mutex::new(Vec::new());

#[inline]
fn site_id(loc: &'static Location<'static>) -> Option<u32> {
    let key = loc as *const Location<'static> as usize;
    let mut h = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize & (NSITES - 1);
    for _ in 0..128 {
        let cur = SITE_KEY[h].load(Relaxed);
        if cur == key {
            return Some(h as u32);
        }
        if cur == 0
            && SITE_KEY[h]
                .compare_exchange(0, key, Relaxed, Relaxed)
                .is_ok()
        {
            if let Ok(mut n) = SITE_NAMES.lock() {
                n.push((key, loc));
            }
            return Some(h as u32);
        }
        h = (h + 1) & (NSITES - 1);
    }
    LOST_SITE.fetch_add(1, Relaxed);
    None
}

// =============================================================================
// Instance interning + per-instance row stride (for the ROWBAND counterfactual)
// =============================================================================

static INST_KEY: [AtomicUsize; NINST] = [const { AtomicUsize::new(0) }; NINST];
static INST_STRIDE: [AtomicI64; NINST] = [const { AtomicI64::new(0) }; NINST];
static INST_LEN: [AtomicU64; NINST] = [const { AtomicU64::new(0) }; NINST];
static INST_N: [AtomicU64; NINST] = [const { AtomicU64::new(0) }; NINST];
/// Instances that got a `declare_stride` after they had already been indexed.
pub static INST_LATE_STRIDE: AtomicU64 = AtomicU64::new(0);

#[inline]
fn inst_id(base: usize) -> u32 {
    let mut h = (base.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize & (NINST - 1);
    for _ in 0..128 {
        let cur = INST_KEY[h].load(Relaxed);
        if cur == base {
            return h as u32;
        }
        if cur == 0
            && INST_KEY[h]
                .compare_exchange(0, base, Relaxed, Relaxed)
                .is_ok()
        {
            return h as u32;
        }
        h = (h + 1) & (NINST - 1);
    }
    LOST_INST.fetch_add(1, Relaxed);
    u32::MAX
}

/// Declare a buffer's picture row stride in BYTES, so the report can evaluate
/// the "widen this guard to the full picture rows it spans" counterfactual —
/// which is exactly the shape #485's loop-filter band took.
pub fn declare_stride(base: usize, len: usize, stride_bytes: isize) {
    let id = inst_id(base);
    if id != u32::MAX {
        if INST_N[id as usize].load(Relaxed) != 0 {
            INST_LATE_STRIDE.fetch_add(1, Relaxed);
        }
        INST_STRIDE[id as usize].store(stride_bytes as i64, Relaxed);
        INST_LEN[id as usize].store(len as u64, Relaxed);
    }
}

// =============================================================================
// Per-(worker, site) aggregates. Sharded by worker so nothing contends.
// =============================================================================

/// Counter indices inside a per-(worker, site) row.
mod c {
    pub const N: usize = 0;
    pub const N_MUT: usize = 1;
    pub const RES_BYTES: usize = 2;
    pub const FP_BYTES: usize = 3;
    pub const N_ROWS_DECL: usize = 4;
    pub const N_WHOLE: usize = 5;
    pub const N_NEVER: usize = 6;
    pub const ROWS_SUM: usize = 7;
    pub const W_SUM: usize = 8;
    pub const N_READ: usize = 9;
    pub const N_WRITE: usize = 10;
    /// Acquisitions that saw >=1 foreign live record on the same instance.
    pub const N_CONC: usize = 11;
    /// ... where at least one such foreign record was MUTABLE.
    pub const N_CONC_MUT: usize = 12;
    /// Reservations that actually intersected a concurrent foreign reservation.
    pub const N_RES_OVL: usize = 13;
    /// Declared footprints that intersected a concurrent foreign footprint.
    pub const N_FP_OVL: usize = 14;
    /// Counterfactual (#485's band): widening this guard to the full picture
    /// ROWS it spans would have intersected a concurrent foreign RESERVATION.
    /// The conservative test — a reservation is a superset of what it touches,
    /// so this can over-predict a conflict but never miss one.
    pub const N_ROW_OVL: usize = 15;
    /// ... and the foreign side was MUTABLE, i.e. another worker writing.
    pub const N_ROW_OVL_MUT: usize = 16;
    /// The same widening against the foreign side's declared FOOTPRINT — the
    /// tight test. `N_ROW_OVL - N_ROW_FP_OVL` is the conflict volume that
    /// exists only because the foreign reservation is looser than its touches.
    pub const N_ROW_FP_OVL: usize = 17;
    pub const N_ROW_FP_OVL_MUT: usize = 18;
    /// Interior waste (hull of declared footprint minus the footprint).
    pub const GAP_BYTES: usize = 19;
    /// Leading / trailing waste, summed, for declared-footprint acquisitions.
    pub const LEAD_WASTE: usize = 20;
    pub const TAIL_WASTE: usize = 21;
    /// Live-interval length in epochs, summed.
    pub const LIVE_EPOCHS: usize = 22;
    /// Gap histogram to nearest foreign reservation (any / mutable).
    pub const GAP_HIST: usize = 23; // .. 23+NGAP
    pub const GAP_HIST_MUT: usize = 23 + super::NGAP; // .. +NGAP
    pub const NCTR: usize = 23 + 2 * super::NGAP;
}

static SITE_AGG: [[[AtomicU64; c::NCTR]; NSITES]; MAX_TH] =
    [const { [const { [const { AtomicU64::new(0) }; c::NCTR] }; NSITES] }; MAX_TH];

/// Minimum observed gap per (worker, site) — tracked separately since it is a
/// min, not a sum. `u64::MAX` == never had a concurrent foreign record.
static SITE_MINGAP: [[AtomicU64; NSITES]; MAX_TH] =
    [const { [const { AtomicU64::new(u64::MAX) }; NSITES] }; MAX_TH];
static SITE_MINGAP_MUT: [[AtomicU64; NSITES]; MAX_TH] =
    [const { [const { AtomicU64::new(u64::MAX) }; NSITES] }; MAX_TH];

#[inline]
fn agg(t: usize, s: u32, i: usize, by: u64) {
    SITE_AGG[t][s as usize][i].fetch_add(by, Relaxed);
}

#[inline]
fn agg_min(cell: &AtomicU64, v: u64) {
    let mut cur = cell.load(Relaxed);
    while v < cur {
        match cell.compare_exchange_weak(cur, v, Relaxed, Relaxed) {
            Ok(_) => return,
            Err(c) => cur = c,
        }
    }
}

// =============================================================================
// Per-(worker, site-pair) conflict table
// =============================================================================

mod p {
    pub const N: usize = 0;
    pub const N_RES_OVL: usize = 1;
    pub const N_FP_OVL: usize = 2;
    pub const N_ROW_OVL: usize = 3;
    /// The foreign side was a MUTABLE reservation.
    pub const N_FOREIGN_MUT: usize = 4;
    pub const NCTR: usize = 5;
}

static PAIR_KEY: [[AtomicU64; NPAIR]; MAX_TH] =
    [const { [const { AtomicU64::new(u64::MAX) }; NPAIR] }; MAX_TH];
static PAIR_AGG: [[[AtomicU64; p::NCTR]; NPAIR]; MAX_TH] =
    [const { [const { [const { AtomicU64::new(0) }; p::NCTR] }; NPAIR] }; MAX_TH];
static PAIR_MINGAP: [[AtomicU64; NPAIR]; MAX_TH] =
    [const { [const { AtomicU64::new(u64::MAX) }; NPAIR] }; MAX_TH];
pub static LOST_PAIR: AtomicU64 = AtomicU64::new(0);

#[inline]
fn pair_slot(t: usize, key: u64) -> Option<usize> {
    let mut h = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 44) as usize & (NPAIR - 1);
    for _ in 0..64 {
        let cur = PAIR_KEY[t][h].load(Relaxed);
        if cur == key {
            return Some(h);
        }
        if cur == u64::MAX {
            PAIR_KEY[t][h].store(key, Relaxed);
            return Some(h);
        }
        h = (h + 1) & (NPAIR - 1);
    }
    LOST_PAIR.fetch_add(1, Relaxed);
    None
}

// =============================================================================
// Raw reservoir sample of acquisitions that had a concurrent foreign record
// =============================================================================

/// Every reservation-overlap event, captured in full. A concurrent overlap
/// involving a MUTABLE record is impossible — the tracker would have panicked —
/// so any row here with `ismut` on either side is a **false positive of this
/// instrument**, and the count is its measured error floor.
const NOVL: usize = 1024;
static OVL: [[AtomicU64; 12]; NOVL] = [const { [const { AtomicU64::new(0) }; 12] }; NOVL];
static OVL_N: AtomicUsize = AtomicUsize::new(0);
/// Overlaps where EITHER side was a mutable reservation, counted over ALL of
/// them rather than the sampled prefix. **This is the instrument's self-check
/// and it must read 0**: the tracker panics on such a pair, so any nonzero
/// value is the probe reporting a concurrency that did not happen, and every
/// number downstream is then suspect. Immutable-vs-immutable overlaps are
/// legal and expected (two workers reading the same reference-frame window).
pub static OVL_MUT: AtomicU64 = AtomicU64::new(0);

const NSAMP: usize = 1 << 14;
static SAMP: [[AtomicU64; 10]; NSAMP] = [const { [const { AtomicU64::new(0) }; 10] }; NSAMP];
static SAMP_N: AtomicUsize = AtomicUsize::new(0);

// =============================================================================
// The acquisition ticket carried by the guard
// =============================================================================

/// Handle from an acquisition to its live record. `Copy` and 8 bytes so the
/// guard's size and auto-traits are unaffected beyond the field itself.
#[derive(Clone, Copy)]
pub struct Ticket {
    tid: u32,
    slot: u32,
}

impl Ticket {
    pub const NONE: Self = Self {
        tid: 0,
        slot: u32::MAX,
    };

    #[inline]
    pub fn is_none(&self) -> bool {
        self.slot == u32::MAX
    }

    #[inline]
    fn cell(&self) -> Option<&'static LiveSlot> {
        if self.slot == u32::MAX {
            None
        } else {
            Some(&LIVE[self.tid as usize].slots[self.slot as usize])
        }
    }

    /// The guard's reference was materialised for reading.
    #[inline]
    pub fn mark_read(&self) {
        if let Some(cell) = self.cell() {
            let f = cell.flags.load(Relaxed);
            if f & F_READ == 0 {
                cell.flags.store(f | F_READ, Relaxed);
            }
        }
    }

    /// The guard's reference was materialised for writing.
    #[inline]
    pub fn mark_write(&self) {
        if let Some(cell) = self.cell() {
            let f = cell.flags.load(Relaxed);
            if f & F_WRITE == 0 {
                cell.flags.store(f | F_WRITE, Relaxed);
            }
        }
    }

    /// Declare the EXACT footprint: `rows` spans of `w` ELEMENTS each, the
    /// first starting at absolute element offset `lo`, successive ones `stride`
    /// elements apart (stride may be negative; `lo` is then the LOWEST row).
    ///
    /// Callers are the strided helpers that already compute `(w, h, stride)`,
    /// so this costs them nothing but the call.
    #[inline]
    pub fn declare_rows(&self, lo: usize, w: usize, rows: usize, stride: isize) {
        let Some(cell) = self.cell() else { return };
        let v = cell.ver.load(Relaxed);
        cell.ver.store(v | 1, Relaxed);
        fence(Release);
        cell.fp_lo.store(lo as u64, Relaxed);
        cell.fp_w.store(w as u64, Relaxed);
        cell.fp_rows.store(rows as u64, Relaxed);
        cell.fp_stride.store(stride as i64, Relaxed);
        let f = cell.flags.load(Relaxed);
        cell.flags.store(f | F_ROWS, Relaxed);
        fence(Release);
        // `v` was even (live); (v|1)+1 == v+2, still even, so the record stays
        // live and every reader mid-flight is invalidated exactly once.
        cell.ver.store((v | 1) + 1, Release);
    }
}

// =============================================================================
// Footprint geometry helpers
// =============================================================================

#[derive(Clone, Copy)]
struct Fp {
    lo: u64,
    hi: u64,
    w: u64,
    rows: u64,
    stride: i64,
    declared: bool,
}

impl Fp {
    #[inline]
    fn bytes(&self) -> u64 {
        if self.declared {
            self.w * self.rows
        } else {
            self.hi - self.lo
        }
    }

    #[inline]
    fn hull(&self) -> (u64, u64) {
        (self.lo, self.hi)
    }

    /// Sparse row-set intersection. Only correct to call when the hulls
    /// already intersect; the caller checks that first.
    fn intersects(&self, o: &Fp) -> bool {
        if !self.declared || !o.declared {
            return true; // hulls intersect and at least one side is unrefined
        }
        let sa = self.stride.unsigned_abs().max(1);
        let sb = o.stride.unsigned_abs().max(1);
        if sa != sb {
            return true; // different geometries: fall back to the hull answer
        }
        // Rows sit at lo + i*s. Only the two candidate j per i can overlap.
        let s = sa as i64;
        for i in 0..self.rows.min(64) {
            let a0 = self.lo as i64 + i as i64 * s;
            let a1 = a0 + self.w as i64;
            let d = a0 - o.lo as i64;
            let j0 = d.div_euclid(s);
            for j in (j0 - 1)..=(j0 + 1) {
                if j < 0 || j as u64 >= o.rows {
                    continue;
                }
                let b0 = o.lo as i64 + j * s;
                let b1 = b0 + o.w as i64;
                if a0 < b1 && b0 < a1 {
                    return true;
                }
            }
        }
        false
    }
}

#[inline]
fn read_fp(cell: &LiveSlot, start: u64, end: u64) -> Fp {
    let flags = cell.flags.load(Relaxed);
    if flags & F_ROWS != 0 {
        let lo = cell.fp_lo.load(Relaxed);
        let w = cell.fp_w.load(Relaxed);
        let rows = cell.fp_rows.load(Relaxed);
        let stride = cell.fp_stride.load(Relaxed);
        let span = if rows == 0 {
            0
        } else {
            (rows - 1) * stride.unsigned_abs() + w
        };
        Fp {
            lo,
            hi: lo + span,
            w,
            rows,
            stride,
            declared: true,
        }
    } else if flags & (F_READ | F_WRITE) != 0 {
        Fp {
            lo: start,
            hi: end,
            w: end - start,
            rows: 1,
            stride: 0,
            declared: false,
        }
    } else {
        Fp {
            lo: start,
            hi: start,
            w: 0,
            rows: 0,
            stride: 0,
            declared: false,
        }
    }
}

#[inline]
fn rowband(start: u64, end: u64, stride: i64, len: u64) -> (u64, u64) {
    let s = stride.unsigned_abs();
    if s == 0 {
        return (start, end);
    }
    let lo = start / s * s;
    let hi = (end.div_ceil(s) * s).min(if len == 0 { u64::MAX } else { len });
    (lo, hi.max(end))
}

#[inline]
fn overlaps(a0: u64, a1: u64, b0: u64, b1: u64) -> bool {
    a0 < b1 && b0 < a1
}

#[inline]
fn gap_between(a0: u64, a1: u64, b0: u64, b1: u64) -> u64 {
    if overlaps(a0, a1, b0, b1) {
        0
    } else if b0 >= a1 {
        b0 - a1
    } else {
        a0 - b1
    }
}

// =============================================================================
// Acquire / release
// =============================================================================

/// Register an acquisition and scan every other worker's live set.
///
/// `base` is the instance's base address (identity), `start`/`end` are ELEMENT
/// offsets in that instance, exactly as the tracker registered them.
#[inline]
pub fn acquire(
    loc: &'static Location<'static>,
    base: usize,
    is_mut: bool,
    start: usize,
    end: usize,
) -> Ticket {
    let t = tid();
    if t >= MAX_TH {
        LOST_TH.fetch_add(1, Relaxed);
        return Ticket::NONE;
    }
    let Some(site) = site_id(loc) else {
        return Ticket::NONE;
    };
    let inst = inst_id(base);
    if inst != u32::MAX {
        INST_N[inst as usize].fetch_add(1, Relaxed);
    }
    let me = &LIVE[t];
    let mask = me.mask.load(Relaxed);
    let slot = (!mask).trailing_zeros() as usize;
    if slot >= MAX_LIVE {
        LOST_SLOT.fetch_add(1, Relaxed);
        return Ticket::NONE;
    }
    let cell = &me.slots[slot];
    let epoch = EPOCH.fetch_add(1, Relaxed);

    let v = cell.ver.load(Relaxed) | 1;
    cell.ver.store(v, Relaxed);
    fence(Release);
    cell.inst.store(inst, Relaxed);
    cell.site.store(site, Relaxed);
    cell.ismut.store(is_mut as u32, Relaxed);
    cell.flags.store(0, Relaxed);
    cell.start.store(start as u64, Relaxed);
    cell.end.store(end as u64, Relaxed);
    cell.fp_lo.store(start as u64, Relaxed);
    cell.fp_w.store(0, Relaxed);
    cell.fp_rows.store(0, Relaxed);
    cell.fp_stride.store(0, Relaxed);
    cell.acq.store(epoch, Relaxed);
    cell.rel.store(0, Relaxed);
    fence(Release);
    cell.ver.store(v + 1, Release);
    me.mask.store(mask | (1u64 << slot), Release);

    // Publish-then-scan on BOTH sides, so every overlapping pair is seen by at
    // least one of them. Without the full fence two workers can each publish
    // and each miss the other.
    fence(core::sync::atomic::Ordering::SeqCst);

    scan(t, site, inst, is_mut, start as u64, end as u64, epoch);

    Ticket {
        tid: t as u32,
        slot: slot as u32,
    }
}

fn scan(t: usize, site: u32, inst: u32, is_mut: bool, start: u64, end: u64, epoch: u64) {
    if inst == u32::MAX {
        // No identity for this buffer, so no record can be compared against it.
        agg(t, site, c::GAP_HIST + NGAP - 1, 1);
        agg(t, site, c::GAP_HIST_MUT + NGAP - 1, 1);
        return;
    }
    let stride = if inst == u32::MAX {
        0
    } else {
        INST_STRIDE[inst as usize].load(Relaxed)
    };
    let ilen = if inst == u32::MAX {
        0
    } else {
        INST_LEN[inst as usize].load(Relaxed)
    };
    let (hb_lo, hb_hi) = rowband(start, end, stride, ilen);

    let mut n_conc = 0u64;
    let mut n_conc_mut = 0u64;
    let mut n_res_ovl = 0u64;
    let mut n_fp_ovl = 0u64;
    let mut row_ovl = false;
    let mut row_ovl_mut = false;
    let mut row_fp_ovl = false;
    let mut row_fp_ovl_mut = false;
    let mut min_gap = u64::MAX;
    let mut min_gap_mut = u64::MAX;

    let mut inuse = SLOTS_INUSE.load(Relaxed) & !(1u64 << t);
    while inuse != 0 {
        let ft = inuse.trailing_zeros() as usize;
        inuse &= inuse - 1;
        let fmask = LIVE[ft].mask.load(Acquire);
        if fmask == 0 {
            continue;
        }
        let mut bits = fmask;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let cell = &LIVE[ft].slots[b];
            let v0 = cell.ver.load(Acquire);
            if v0 & 1 != 0 {
                // Dead or mid-write: the mask bit we read was stale.
                SKIP_DEAD.fetch_add(1, Relaxed);
                continue;
            }
            let f_inst = cell.inst.load(Relaxed);
            if f_inst != inst {
                continue;
            }
            let f_site = cell.site.load(Relaxed);
            let f_mut = cell.ismut.load(Relaxed) != 0;
            let f_start = cell.start.load(Relaxed);
            let f_end = cell.end.load(Relaxed);
            let f_fp = read_fp(cell, f_start, f_end);
            let f_rel = cell.rel.load(Acquire);
            fence(Acquire);
            if cell.ver.load(Acquire) != v0 {
                LOST_SCAN.fetch_add(1, Relaxed);
                continue;
            }
            // Liveness. The mask bit said "live", but the owner may have
            // retired the record since — and a record that closed BEFORE this
            // acquisition's epoch was never concurrent with it. Skipping those
            // is what stops the instrument reporting overlaps the tracker
            // would have panicked on.
            if f_rel != 0 && f_rel < epoch {
                SKIP_DEAD.fetch_add(1, Relaxed);
                continue;
            }
            if f_rel == 0 && LIVE[ft].mask.load(Acquire) & (1u64 << b) == 0 {
                // The bit we read may itself have been stale; re-read `rel`
                // now that the clear is visible.
                let r2 = cell.rel.load(Acquire);
                if r2 != 0 && r2 < epoch {
                    SKIP_DEAD.fetch_add(1, Relaxed);
                    continue;
                }
            }
            n_conc += 1;
            if f_mut {
                n_conc_mut += 1;
            }
            let res_ovl = overlaps(start, end, f_start, f_end);
            if res_ovl {
                n_res_ovl += 1;
                if is_mut || f_mut {
                    OVL_MUT.fetch_add(1, Relaxed);
                }
                let n = OVL_N.fetch_add(1, Relaxed);
                if n < NOVL {
                    let r = &OVL[n];
                    r[0].store(epoch, Relaxed);
                    r[1].store(t as u64, Relaxed);
                    r[2].store(start, Relaxed);
                    r[3].store(end, Relaxed);
                    r[4].store(is_mut as u64, Relaxed);
                    r[5].store(site as u64, Relaxed);
                    r[6].store(ft as u64, Relaxed);
                    r[7].store(cell.acq.load(Relaxed), Relaxed);
                    r[8].store(cell.rel.load(Relaxed), Relaxed);
                    r[9].store(f_start, Relaxed);
                    r[10].store((f_end << 1) | (f_mut as u64), Relaxed);
                    r[11].store(f_site as u64, Relaxed);
                }
            }
            let g = gap_between(start, end, f_start, f_end);
            if g < min_gap {
                min_gap = g;
            }
            if f_mut && g < min_gap_mut {
                min_gap_mut = g;
            }
            // Real footprint intersection. My own rows are not declared yet at
            // acquire time, so my side is the reservation (a superset) and the
            // foreign side is its declared footprint where it has one.
            let fp_ovl = f_fp.bytes() > 0
                && overlaps(start, end, f_fp.lo, f_fp.hi)
                && f_fp.intersects(&Fp {
                    lo: start,
                    hi: end,
                    w: end - start,
                    rows: 1,
                    stride: 0,
                    declared: false,
                });
            if fp_ovl {
                n_fp_ovl += 1;
            }
            // Counterfactual (#485's band): widen to the full picture rows we
            // span, and test against the foreign RESERVATION (conservative)
            // and against the foreign declared FOOTPRINT (tight).
            let rb = overlaps(hb_lo, hb_hi, f_start, f_end);
            if rb {
                row_ovl = true;
                if f_mut {
                    row_ovl_mut = true;
                }
                if f_fp.bytes() > 0 && overlaps(hb_lo, hb_hi, f_fp.lo, f_fp.hi) {
                    row_fp_ovl = true;
                    if f_mut {
                        row_fp_ovl_mut = true;
                    }
                }
            }

            // Pair table.
            let key = ((site as u64) << 20) | (f_site as u64);
            if let Some(ps) = pair_slot(t, key) {
                PAIR_AGG[t][ps][p::N].fetch_add(1, Relaxed);
                if res_ovl {
                    PAIR_AGG[t][ps][p::N_RES_OVL].fetch_add(1, Relaxed);
                }
                if fp_ovl {
                    PAIR_AGG[t][ps][p::N_FP_OVL].fetch_add(1, Relaxed);
                }
                if rb {
                    PAIR_AGG[t][ps][p::N_ROW_OVL].fetch_add(1, Relaxed);
                }
                if f_mut {
                    PAIR_AGG[t][ps][p::N_FOREIGN_MUT].fetch_add(1, Relaxed);
                }
                agg_min(&PAIR_MINGAP[t][ps], g);
            }

            if n_conc == 1 {
                // Reservoir-ish: keep the first NSAMP concurrent acquisitions.
                let n = SAMP_N.fetch_add(1, Relaxed);
                if n < NSAMP {
                    let r = &SAMP[n];
                    r[0].store(epoch, Relaxed);
                    r[1].store(t as u64, Relaxed);
                    r[2].store(site as u64, Relaxed);
                    r[3].store(start, Relaxed);
                    r[4].store(end, Relaxed);
                    r[5].store(is_mut as u64, Relaxed);
                    r[6].store(ft as u64, Relaxed);
                    r[7].store(f_site as u64, Relaxed);
                    r[8].store(f_start, Relaxed);
                    r[9].store((f_end << 1) | (f_mut as u64), Relaxed);
                }
            }
        }
    }

    if n_conc > 0 {
        agg(t, site, c::N_CONC, 1);
        if n_conc_mut > 0 {
            agg(t, site, c::N_CONC_MUT, 1);
        }
        if n_res_ovl > 0 {
            agg(t, site, c::N_RES_OVL, 1);
        }
        if n_fp_ovl > 0 {
            agg(t, site, c::N_FP_OVL, 1);
        }
        if row_ovl {
            agg(t, site, c::N_ROW_OVL, 1);
        }
        if row_ovl_mut {
            agg(t, site, c::N_ROW_OVL_MUT, 1);
        }
        if row_fp_ovl {
            agg(t, site, c::N_ROW_FP_OVL, 1);
        }
        if row_fp_ovl_mut {
            agg(t, site, c::N_ROW_FP_OVL_MUT, 1);
        }
        agg(t, site, c::GAP_HIST + gap_bucket(min_gap), 1);
        agg_min(&SITE_MINGAP[t][site as usize], min_gap);
        if min_gap_mut != u64::MAX {
            agg(t, site, c::GAP_HIST_MUT + gap_bucket(min_gap_mut), 1);
            agg_min(&SITE_MINGAP_MUT[t][site as usize], min_gap_mut);
        } else {
            agg(t, site, c::GAP_HIST_MUT + NGAP - 1, 1);
        }
    } else {
        agg(t, site, c::GAP_HIST + NGAP - 1, 1);
        agg(t, site, c::GAP_HIST_MUT + NGAP - 1, 1);
    }
}

/// Retire an acquisition and fold it into the per-site aggregates.
#[inline]
pub fn release(tk: Ticket) {
    let Some(cell) = tk.cell() else { return };
    let t = tk.tid as usize;
    let slot = tk.slot as usize;
    let me = &LIVE[t];
    let rel = EPOCH.fetch_add(1, Relaxed);
    // Publish the release epoch, THEN clear the mask, and do NOT bump `ver`.
    //
    // Bumping `ver` here was tried and measured: it invalidated 20,869,655
    // seqlock reads on one `v4k_8tile` t=8 run, i.e. two thirds of every
    // foreign record the scan looked at, because a slot is retired far more
    // often than it is written. The epoch test below does the same job without
    // touching the generation counter.
    cell.rel.store(rel, Relaxed);
    cell.ver.store(cell.ver.load(Relaxed) | 1, Release);
    me.mask
        .store(me.mask.load(Relaxed) & !(1u64 << slot), Release);

    let site = cell.site.load(Relaxed);
    let is_mut = cell.ismut.load(Relaxed) != 0;
    let start = cell.start.load(Relaxed);
    let end = cell.end.load(Relaxed);
    let flags = cell.flags.load(Relaxed);
    let acq = cell.acq.load(Relaxed);
    let res = end - start;

    agg(t, site, c::N, 1);
    if is_mut {
        agg(t, site, c::N_MUT, 1);
    }
    agg(t, site, c::RES_BYTES, res);
    agg(t, site, c::LIVE_EPOCHS, rel.saturating_sub(acq));
    if flags & F_READ != 0 {
        agg(t, site, c::N_READ, 1);
    }
    if flags & F_WRITE != 0 {
        agg(t, site, c::N_WRITE, 1);
    }
    if flags & F_ROWS != 0 {
        let fp = read_fp(cell, start, end);
        agg(t, site, c::N_ROWS_DECL, 1);
        agg(t, site, c::FP_BYTES, fp.bytes());
        agg(t, site, c::ROWS_SUM, fp.rows);
        agg(t, site, c::W_SUM, fp.w);
        let (hl, hh) = fp.hull();
        agg(t, site, c::GAP_BYTES, (hh - hl).saturating_sub(fp.bytes()));
        agg(t, site, c::LEAD_WASTE, hl.saturating_sub(start));
        agg(t, site, c::TAIL_WASTE, end.saturating_sub(hh));
    } else if flags & (F_READ | F_WRITE) != 0 {
        agg(t, site, c::N_WHOLE, 1);
        agg(t, site, c::FP_BYTES, res);
    } else {
        agg(t, site, c::N_NEVER, 1);
    }
}

// =============================================================================
// STRIDED-RECT counterfactual: would a 2-D record be sound, and would it pay?
// =============================================================================
//
// The question this answers is the one the March-2026 strided tracker
// (`884b4b5`, reverted by `424cbbb` the next day on an ARGUMENT) was never
// measured against, and which #472's per-row reference view re-opened:
//
// > If ONE registration covered an `h x w` rectangle exactly — a 2-D record,
// > with per-row references so both sides are exact (§6 of
// > `docs/OWNERSHIP_MODELS.md`) — would it ever REJECT a foreign record that
// > the shipped `h` per-row registrations permit?
//
// A 2-D record only differs from `h` 1-D records in the inter-row gaps. So the
// decisive event is: a foreign record that intersects the rectangle's HULL but
// NOT the rectangle itself. If the overlap test is exact, that pair is
// permitted and the scheme is sound; if the test falls back to the hull (as
// `884b4b5` did whenever the two strides differed), that pair PANICS and the
// scheme is a decode failure. Either way the count of such pairs — split by
// whether the counterparty was a WRITE — is what decides it, and it is a
// measurement, not an argument.
//
// This evaluator REGISTERS NOTHING and holds no guard. It is called by the
// strided helper immediately before that helper takes its shipped per-row
// guards, so the live set it scans is the set the counterfactual registration
// would itself have seen. Behaviour is unchanged.

/// Counters, indexed the same way as [`c`]/[`p`].
pub mod r {
    /// Counterfactual evaluations.
    pub const N: usize = 0;
    /// Evaluations that had at least one co-live foreign record.
    pub const N_CONC: usize = 1;
    /// Co-live foreign records seen (not evaluations).
    pub const N_PEERS: usize = 2;
    /// Peers intersecting the HULL — what a 1-D test over the hull rejects.
    pub const N_HULL: usize = 3;
    /// Peers intersecting the exact RECTANGLE — what a correct 2-D test rejects.
    pub const N_RECT: usize = 4;
    /// **THE ANSWER, read direction.** Peers in the hull but NOT in the
    /// rectangle: the inter-row-gap traffic a 2-D record must permit.
    pub const N_GAP: usize = 5;
    /// **THE ANSWER.** As [`N_GAP`], and the peer was a MUTABLE reservation —
    /// i.e. a concurrently-live foreign WRITE inside the strided read's gap.
    pub const N_GAP_MUT: usize = 6;
    /// Gap peers whose exactness needed the general (different-stride) test.
    pub const N_GAP_XSTRIDE: usize = 7;
    /// Rows in the evaluated rectangle, summed (for a mean).
    pub const ROWS_SUM: usize = 8;
    /// Max rows seen — bounds the row-iteration cap's validity.
    pub const ROWS_MAX: usize = 9;
    /// Distinct 4096-element blocks the HULL spans, summed.
    pub const HULL_BLOCKS_SUM: usize = 10;
    /// Distinct blocks the ROW SET spans, summed (a shard-aware 2-D record's
    /// registration footprint — the cost a 2-D scheme cannot avoid).
    pub const ROW_BLOCKS_SUM: usize = 11;
    /// Evaluations whose hull spans more than `MAX_SHARDS_PER_BORROW` blocks,
    /// hence would promote to the tracker's WIDE path (all active shards).
    pub const N_HULL_WIDE: usize = 12;
    /// Evaluations whose ROW SET spans more than `MAX_SHARDS_PER_BORROW`
    /// blocks — a shard-aware 2-D record goes wide too.
    pub const N_ROW_WIDE: usize = 13;
    /// Distinct SHARDS the row set maps to, summed. This is the cost a 2-D
    /// record pays per registration; the shipped per-row scheme pays 1 each.
    pub const ROW_SHARDS_SUM: usize = 15;
    /// Bitmask of the block shifts observed at this site.
    pub const SHIFT_SEEN: usize = 16;
    /// Evaluations where every row sits in ONE block, i.e. the shipped per-row
    /// scheme took `rows` single-shard registrations.
    pub const N_PERROW_NARROW: usize = 14;
    pub const NCTR: usize = 17;
}

/// This instance's REAL shard geometry, handed in by the tracker.
///
/// It is NOT a mirrored constant. The shipped block shift is
/// `block_shift_for(len)` — 12 for a serial or single-tile decode, 14 for a
/// multi-tile 4K 8-bit luma plane, 15 at 10-bit — so the number of blocks a
/// hull spans, and hence whether it promotes to the all-shards wide path,
/// depends on the configuration. An earlier draft of this probe mirrored 12 and
/// would have reported "100% wide" for a cell that is not wide at all.
#[derive(Clone, Copy)]
pub struct ShardGeom {
    pub shift: u32,
    pub mask: usize,
    pub max_shards: usize,
    pub max_blocks: usize,
}

static RECT_AGG: [[[AtomicU64; r::NCTR]; NSITES]; MAX_TH] =
    [const { [const { [const { AtomicU64::new(0) }; r::NCTR] }; NSITES] }; MAX_TH];

/// Gap events captured in full, so the report can name the SITE PAIR rather
/// than a count (a count cannot be set-diffed).
const NGAPS: usize = 512;
static GAPS: [[AtomicU64; 10]; NGAPS] = [const { [const { AtomicU64::new(0) }; 10] }; NGAPS];
static GAPS_N: AtomicUsize = AtomicUsize::new(0);

/// Rectangle rows iterated before the exact test gives up and answers
/// conservatively. MUST stay 0 or the `N_GAP` counts are upper bounds.
pub static RECT_ROWS_CAPPED: AtomicU64 = AtomicU64::new(0);
const RECT_ROW_CAP: u64 = 512;

/// An `h x w` rectangle at `lo`, rows `|stride|` apart.
#[derive(Clone, Copy)]
struct Rect {
    lo: u64,
    w: u64,
    rows: u64,
    stride: u64,
}

impl Rect {
    #[inline]
    fn hull(&self) -> (u64, u64) {
        (self.lo, self.lo + (self.rows - 1) * self.stride + self.w)
    }

    /// Exact: does any row of this rectangle intersect `[b0, b1)`?
    ///
    /// Closed form, no iteration: the rows that can reach `[b0, b1)` are those
    /// whose index lies in a contiguous range, and only the endpoints need
    /// testing.
    #[inline]
    fn hits_interval(&self, b0: u64, b1: u64) -> bool {
        if b0 >= b1 || self.rows == 0 || self.w == 0 {
            return false;
        }
        let s = self.stride.max(1);
        // Row i covers [lo + i*s, lo + i*s + w). It intersects [b0,b1) iff
        // lo + i*s < b1  &&  b0 < lo + i*s + w.
        let i_hi = if b1 > self.lo {
            ((b1 - 1 - self.lo) / s).min(self.rows - 1)
        } else {
            return false;
        };
        let i_lo = if b0 + 1 > self.lo + self.w {
            (b0 + 1 - self.lo - self.w).div_ceil(s)
        } else {
            0
        };
        i_lo <= i_hi
    }

    /// A rectangle whose rows are at least as wide as its stride has NO gaps:
    /// consecutive rows touch or overlap, so its byte set is exactly its hull.
    /// (Degenerate for a picture — `w <= stride` always holds there — but the
    /// oracle grid covers it and an inexact predicate here is worthless.)
    #[inline]
    fn contiguous(&self) -> bool {
        self.rows <= 1 || self.w >= self.stride.max(1)
    }

    /// A row that starts late enough in its stride period spills into the next
    /// one, so the rectangle occupies two column ranges rather than one and the
    /// closed-form column test does not apply. `w < stride` is NOT enough to
    /// rule this out — the oracle caught exactly that mistake.
    #[inline]
    fn wraps(&self) -> bool {
        let s = self.stride.max(1);
        self.rows > 1 && (self.lo % s) + self.w > s
    }

    /// Exact: does this rectangle intersect `o`?
    ///
    /// Returns `(hit, needed_general_path)`. Three cases, all exact:
    ///
    /// * either side gap-free -> rectangle-vs-interval, closed form;
    /// * equal strides, both gapped -> the column/row product test;
    /// * different strides -> iterate this rectangle's rows as intervals,
    ///   bounded by [`RECT_ROW_CAP`].
    #[inline]
    fn hits_rect(&self, o: &Rect) -> (bool, bool) {
        if o.contiguous() {
            let (b0, b1) = o.hull();
            return (self.hits_interval(b0, b1), false);
        }
        if self.contiguous() {
            let (a0, a1) = self.hull();
            return (o.hits_interval(a0, a1), false);
        }
        // Both gapped and neither wrapping: one column range per side.
        if self.stride == o.stride && !self.wraps() && !o.wraps() {
            let s = self.stride;
            let a_col = self.lo % s;
            let b_col = o.lo % s;
            if !(a_col < b_col + o.w && b_col < a_col + self.w) {
                return (false, false);
            }
            let a_r0 = self.lo / s;
            let b_r0 = o.lo / s;
            return (
                a_r0 <= b_r0 + o.rows - 1 && b_r0 <= a_r0 + self.rows - 1,
                false,
            );
        }
        let xstride = self.stride != o.stride;
        if self.rows > RECT_ROW_CAP {
            RECT_ROWS_CAPPED.fetch_add(1, Relaxed);
            return (true, xstride);
        }
        for i in 0..self.rows {
            let a0 = self.lo + i * self.stride;
            if o.hits_interval(a0, a0 + self.w) {
                return (true, xstride);
            }
        }
        (false, xstride)
    }

    /// Distinct blocks the ROW SET touches, at the instance's real shift.
    /// Rows ascend, so a run-length dedup is exact.
    #[inline]
    fn row_blocks(&self, g: &ShardGeom) -> u64 {
        let s = self.stride.max(1);
        let mut n = 0u64;
        let mut last = u64::MAX;
        for i in 0..self.rows.min(RECT_ROW_CAP) {
            let a0 = (self.lo + i * s) >> g.shift;
            let a1 = (self.lo + i * s + self.w - 1) >> g.shift;
            for b in a0..=a1 {
                if b != last {
                    n += 1;
                    last = b;
                }
            }
        }
        n
    }

    /// Distinct SHARDS the row set maps to, by the tracker's own `shard_of`,
    /// capped at `max_shards + 1` (past that the tracker goes wide anyway).
    #[inline]
    fn row_shards(&self, g: &ShardGeom) -> u64 {
        let s = self.stride.max(1);
        let mut set = [usize::MAX; 8];
        let mut n = 0usize;
        for i in 0..self.rows.min(RECT_ROW_CAP) {
            let a0 = (self.lo + i * s) >> g.shift;
            let a1 = (self.lo + i * s + self.w - 1) >> g.shift;
            for b in a0..=a1 {
                let sh = crate::checked::BorrowTracker::probe_shard_of(b as usize, g.mask);
                if set[..n].contains(&sh) {
                    continue;
                }
                if n == set.len() {
                    return set.len() as u64 + 1;
                }
                set[n] = sh;
                n += 1;
            }
        }
        n as u64
    }
}

/// Evaluate the strided-rectangle counterfactual for the acquisition the caller
/// is ABOUT to make as `rows` per-row guards. Registers nothing.
///
/// `lo`/`w`/`stride` are in ELEMENT offsets of the instance identified by
/// `base`, exactly as the tracker would register them. `stride` may be
/// negative; the rectangle is normalised to its lowest row.
pub fn eval_rect(
    loc: &'static Location<'static>,
    base: usize,
    is_mut: bool,
    lo: usize,
    w: usize,
    rows: usize,
    stride: isize,
    geom: ShardGeom,
) {
    if rows == 0 || w == 0 {
        return;
    }
    let t = tid();
    if t >= MAX_TH {
        return;
    }
    let Some(site) = site_id(loc) else { return };
    let inst = inst_id(base);
    if inst == u32::MAX {
        return;
    }
    let astride = stride.unsigned_abs() as u64;
    let base_lo = if stride >= 0 {
        lo as u64
    } else {
        (lo as u64).saturating_sub((rows as u64 - 1) * astride)
    };
    let me = Rect {
        lo: base_lo,
        w: w as u64,
        rows: rows as u64,
        stride: astride,
    };
    let (h0, h1) = me.hull();

    let epoch = EPOCH.fetch_add(1, Relaxed);
    fence(core::sync::atomic::Ordering::SeqCst);

    let mut n_peers = 0u64;
    let mut n_hull = 0u64;
    let mut n_rect = 0u64;
    let mut n_gap = 0u64;
    let mut n_gap_mut = 0u64;
    let mut n_gap_x = 0u64;

    let mut inuse = SLOTS_INUSE.load(Relaxed) & !(1u64 << t);
    while inuse != 0 {
        let ft = inuse.trailing_zeros() as usize;
        inuse &= inuse - 1;
        let fmask = LIVE[ft].mask.load(Acquire);
        if fmask == 0 {
            continue;
        }
        let mut bits = fmask;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let cell = &LIVE[ft].slots[b];
            let v0 = cell.ver.load(Acquire);
            if v0 & 1 != 0 {
                continue;
            }
            if cell.inst.load(Relaxed) != inst {
                continue;
            }
            let f_site = cell.site.load(Relaxed);
            let f_mut = cell.ismut.load(Relaxed) != 0;
            let f_start = cell.start.load(Relaxed);
            let f_end = cell.end.load(Relaxed);
            let f_fp = read_fp(cell, f_start, f_end);
            let f_rel = cell.rel.load(Acquire);
            fence(Acquire);
            if cell.ver.load(Acquire) != v0 {
                LOST_SCAN.fetch_add(1, Relaxed);
                continue;
            }
            if f_rel != 0 && f_rel < epoch {
                continue;
            }
            n_peers += 1;

            // A 1-D test over the hull — what a hull-extent registration (#475,
            // #485, `lf_hull_reads`) rejects.
            if !overlaps(h0, h1, f_start, f_end) {
                continue;
            }
            n_hull += 1;

            // The exact 2-D answer. The foreign side is its DECLARED rectangle
            // where it has one and its reservation interval otherwise — the
            // reservation is a superset of the bytes it touches, so using it
            // can over-predict a conflict and never miss one.
            let (hit, xstride) = if f_fp.declared && f_fp.rows > 0 && f_fp.w > 0 {
                let fs = f_fp.stride.unsigned_abs();
                let f_lo = if f_fp.stride >= 0 {
                    f_fp.lo
                } else {
                    f_fp.lo.saturating_sub((f_fp.rows - 1) * fs)
                };
                me.hits_rect(&Rect {
                    lo: f_lo,
                    w: f_fp.w,
                    rows: f_fp.rows,
                    stride: fs,
                })
            } else {
                (me.hits_interval(f_start, f_end), false)
            };
            if hit {
                n_rect += 1;
                continue;
            }
            // Hull yes, rectangle no: inter-row-gap traffic.
            n_gap += 1;
            if xstride {
                n_gap_x += 1;
            }
            if f_mut {
                n_gap_mut += 1;
            }
            let n = GAPS_N.fetch_add(1, Relaxed);
            if n < NGAPS {
                let g = &GAPS[n];
                g[0].store(site as u64, Relaxed);
                g[1].store(f_site as u64, Relaxed);
                g[2].store(base_lo, Relaxed);
                g[3].store(me.w, Relaxed);
                g[4].store(me.rows, Relaxed);
                g[5].store(me.stride, Relaxed);
                g[6].store(f_start, Relaxed);
                g[7].store(f_end, Relaxed);
                g[8].store(u64::from(f_mut) | (u64::from(is_mut) << 1), Relaxed);
                g[9].store(u64::from(xstride), Relaxed);
            }
        }
    }

    let hull_blocks = ((h1 - 1) >> geom.shift) - (h0 >> geom.shift) + 1;
    let row_blocks = me.row_blocks(&geom);
    let row_shards = me.row_shards(&geom);
    let per_row_narrow = (0..me.rows.min(RECT_ROW_CAP)).all(|i| {
        let a0 = me.lo + i * me.stride.max(1);
        (a0 >> geom.shift) == ((a0 + me.w - 1) >> geom.shift)
    });

    let a = &RECT_AGG[t][site as usize];
    a[r::N].fetch_add(1, Relaxed);
    a[r::N_PEERS].fetch_add(n_peers, Relaxed);
    if n_peers > 0 {
        a[r::N_CONC].fetch_add(1, Relaxed);
    }
    a[r::N_HULL].fetch_add(n_hull, Relaxed);
    a[r::N_RECT].fetch_add(n_rect, Relaxed);
    a[r::N_GAP].fetch_add(n_gap, Relaxed);
    a[r::N_GAP_MUT].fetch_add(n_gap_mut, Relaxed);
    a[r::N_GAP_XSTRIDE].fetch_add(n_gap_x, Relaxed);
    a[r::ROWS_SUM].fetch_add(me.rows, Relaxed);
    agg_min_neg(&a[r::ROWS_MAX], me.rows);
    a[r::HULL_BLOCKS_SUM].fetch_add(hull_blocks, Relaxed);
    a[r::ROW_BLOCKS_SUM].fetch_add(row_blocks, Relaxed);
    a[r::ROW_SHARDS_SUM].fetch_add(row_shards, Relaxed);
    a[r::SHIFT_SEEN].fetch_or(1u64 << (geom.shift.min(63)), Relaxed);
    // The tracker promotes to the WIDE path (every ACTIVE shard) when a borrow
    // touches more than `max_shards` distinct shards or more than `max_blocks`
    // blocks. Priced with the instance's own geometry, not a mirrored constant.
    if hull_blocks as usize > geom.max_blocks || row_shards as usize > geom.max_shards {
        a[r::N_HULL_WIDE].fetch_add(1, Relaxed);
    }
    if row_shards as usize > geom.max_shards {
        a[r::N_ROW_WIDE].fetch_add(1, Relaxed);
    }
    if per_row_narrow {
        a[r::N_PERROW_NARROW].fetch_add(1, Relaxed);
    }
}

/// Running maximum.
#[inline]
fn agg_min_neg(cell: &AtomicU64, v: u64) {
    let mut cur = cell.load(Relaxed);
    while v > cur {
        match cell.compare_exchange_weak(cur, v, Relaxed, Relaxed) {
            Ok(_) => return,
            Err(c) => cur = c,
        }
    }
}

/// The strided-rectangle counterfactual, per site plus the named gap pairs.
pub fn report_rect() -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "#rectsite\tn\tn_conc\tpeers\thull_ovl\trect_ovl\tgap\tgap_mut\tgap_xstride\trows_mean\trows_max\thull_blocks_mean\trow_blocks_mean\trow_shards_mean\tpct_hull_wide\tpct_row_wide\tpct_perrow_narrow\tshifts\twhere"
    );
    let mut any = false;
    for s in 0..NSITES {
        if SITE_KEY[s].load(Relaxed) == 0 {
            continue;
        }
        let mut c = [0u64; r::NCTR];
        for t in 0..MAX_TH {
            for i in 0..r::NCTR {
                if i == r::ROWS_MAX {
                    c[i] = c[i].max(RECT_AGG[t][s][i].load(Relaxed));
                } else if i == r::SHIFT_SEEN {
                    c[i] |= RECT_AGG[t][s][i].load(Relaxed);
                } else {
                    c[i] += RECT_AGG[t][s][i].load(Relaxed);
                }
            }
        }
        if c[r::N] == 0 {
            continue;
        }
        any = true;
        let n = c[r::N] as f64;
        let _ = writeln!(
            out,
            "RECT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}",
            c[r::N],
            c[r::N_CONC],
            c[r::N_PEERS],
            c[r::N_HULL],
            c[r::N_RECT],
            c[r::N_GAP],
            c[r::N_GAP_MUT],
            c[r::N_GAP_XSTRIDE],
            c[r::ROWS_SUM] as f64 / n,
            c[r::ROWS_MAX],
            c[r::HULL_BLOCKS_SUM] as f64 / n,
            c[r::ROW_BLOCKS_SUM] as f64 / n,
            c[r::ROW_SHARDS_SUM] as f64 / n,
            100.0 * c[r::N_HULL_WIDE] as f64 / n,
            100.0 * c[r::N_ROW_WIDE] as f64 / n,
            100.0 * c[r::N_PERROW_NARROW] as f64 / n,
            {
                let m = c[r::SHIFT_SEEN];
                let mut v = String::new();
                for b in 0..64 {
                    if m >> b & 1 == 1 {
                        if !v.is_empty() {
                            v.push(',');
                        }
                        let _ = write!(v, "{b}");
                    }
                }
                v
            },
            site_name(s as u32),
        );
    }
    if !any {
        let _ = writeln!(
            out,
            "RECT\t(no strided helper declared a rectangle — the counterfactual never armed)"
        );
    }
    let _ = writeln!(
        out,
        "RECTMETA\trows_capped={}\tgaps_sampled={}",
        RECT_ROWS_CAPPED.load(Relaxed),
        GAPS_N.load(Relaxed).min(NGAPS),
    );
    let _ = writeln!(
        out,
        "#gapsite\tmine\tpeer\tlo\tw\trows\tstride\tf_start\tf_end\tpeer_mut\tmine_mut\txstride"
    );
    for i in 0..GAPS_N.load(Relaxed).min(NGAPS) {
        let g = &GAPS[i];
        let fl = g[8].load(Relaxed);
        let _ = writeln!(
            out,
            "GAP\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            site_name(g[0].load(Relaxed) as u32),
            site_name(g[1].load(Relaxed) as u32),
            g[2].load(Relaxed),
            g[3].load(Relaxed),
            g[4].load(Relaxed),
            g[5].load(Relaxed),
            g[6].load(Relaxed),
            g[7].load(Relaxed),
            fl & 1,
            (fl >> 1) & 1,
            g[9].load(Relaxed),
        );
    }
    out
}

#[cfg(test)]
mod rect_oracle {
    //! Differential test of [`Rect`]'s closed-form intersection against a
    //! brute-force byte-set oracle.
    //!
    //! The whole strided-2-D question turns on this predicate being EXACT: a
    //! false negative permits two aliasing references (the March-2026 defect),
    //! a false positive is a decode failure. So it is checked against the
    //! definition — the literal set of bytes each rectangle covers — over an
    //! exhaustive small grid, not against a transcription of itself.
    use super::*;
    use std::collections::BTreeSet;
    use std::vec::Vec;

    fn bytes(r: &Rect) -> BTreeSet<u64> {
        let mut s = BTreeSet::new();
        for i in 0..r.rows {
            let a0 = r.lo + i * r.stride.max(1);
            for b in a0..a0 + r.w {
                s.insert(b);
            }
        }
        s
    }

    fn grid() -> Vec<Rect> {
        let mut v = Vec::new();
        for stride in [1u64, 2, 3, 5, 7] {
            for w in 1u64..=7 {
                for rows in 1u64..=4 {
                    for lo in 0u64..=11 {
                        v.push(Rect {
                            lo,
                            w,
                            rows,
                            stride,
                        });
                    }
                }
            }
        }
        v
    }

    #[test]
    fn hits_interval_matches_the_byte_set() {
        let mut checked = 0u64;
        let mut trues = 0u64;
        for r in grid() {
            let set = bytes(&r);
            for b0 in 0u64..=24 {
                for len in 1u64..=9 {
                    let b1 = b0 + len;
                    let want = (b0..b1).any(|x| set.contains(&x));
                    assert_eq!(
                        r.hits_interval(b0, b1),
                        want,
                        "rect lo={} w={} rows={} stride={} vs [{b0},{b1})",
                        r.lo,
                        r.w,
                        r.rows,
                        r.stride
                    );
                    checked += 1;
                    trues += u64::from(want);
                }
            }
        }
        // Liveness: the grid must actually exercise both answers.
        assert!(checked > 300_000, "grid too small: {checked}");
        assert!(
            trues > checked / 20,
            "almost never intersects: {trues}/{checked}"
        );
        assert!(
            trues < checked - checked / 20,
            "almost always intersects: {trues}/{checked}"
        );
    }

    #[test]
    fn hits_rect_matches_the_byte_set() {
        let g = grid();
        let mut checked = 0u64;
        let mut trues = 0u64;
        let mut xstride_seen = 0u64;
        for a in &g {
            let sa = bytes(a);
            for b in &g {
                let sb = bytes(b);
                let want = sa.intersection(&sb).next().is_some();
                let (got, xstride) = a.hits_rect(b);
                // The general path is exact; the same-stride closed form must be
                // exact too. Neither is allowed to under-report.
                assert_eq!(
                    got, want,
                    "A(lo={},w={},rows={},s={}) vs B(lo={},w={},rows={},s={}) xstride={xstride}",
                    a.lo, a.w, a.rows, a.stride, b.lo, b.w, b.rows, b.stride
                );
                checked += 1;
                trues += u64::from(want);
                xstride_seen += u64::from(xstride);
            }
        }
        assert!(checked > 100_000, "grid too small: {checked}");
        assert!(
            trues > checked / 20 && trues < checked - checked / 20,
            "degenerate: {trues}/{checked}"
        );
        // Liveness: BOTH branches must have run, or half the predicate is untested.
        assert!(xstride_seen > 0, "the different-stride path never ran");
        assert!(xstride_seen < checked, "the same-stride path never ran");
    }

    /// The March-2026 (`884b4b5`) test, transcribed verbatim from
    /// `git show 884b4b5`, and the two inputs on which it is WRONG.
    ///
    /// This is archaeology, not a shipping predicate. It exists so the campaign
    /// record contains the defect as a executed fact rather than a claim.
    #[allow(clippy::too_many_arguments)]
    fn ranges_overlap_884b4b5(
        a_start: usize,
        a_end: usize,
        a_stride: usize,
        a_width: usize,
        b_start: usize,
        b_end: usize,
        b_stride: usize,
        b_width: usize,
    ) -> bool {
        if a_start >= b_end || b_start >= a_end {
            return false;
        }
        let (
            strided_start,
            strided_end,
            stride,
            s_width,
            other_start,
            other_end,
            o_stride,
            o_width,
        ) = if a_stride > 0 && a_width > 0 {
            (
                a_start, a_end, a_stride, a_width, b_start, b_end, b_stride, b_width,
            )
        } else if b_stride > 0 && b_width > 0 {
            (
                b_start, b_end, b_stride, b_width, a_start, a_end, a_stride, a_width,
            )
        } else {
            return true;
        };
        if o_stride == stride && o_width > 0 {
            let s_col = strided_start % stride;
            let o_col = other_start % stride;
            if s_col >= o_col + o_width || o_col >= s_col + s_width {
                return false;
            }
            let s_row = strided_start / stride;
            let o_row = other_start / stride;
            let s_h = (strided_end - strided_start).div_ceil(stride);
            let o_h = (other_end - other_start).div_ceil(stride);
            if s_row >= o_row + o_h || o_row >= s_row + s_h {
                return false;
            }
        } else if o_stride == 0 {
            let other_len = other_end - other_start;
            let o_col = other_start % stride;
            let s_col = strided_start % stride;
            let s_row = strided_start / stride;
            let o_row = other_start / stride;
            let s_h = (strided_end - strided_start).div_ceil(stride);
            let o_h = (other_end - other_start).div_ceil(stride);
            if s_row >= o_row + o_h || o_row >= s_row + s_h {
                return false;
            }
            if other_len < stride && o_col + other_len <= stride {
                if o_col >= s_col + s_width || s_col >= o_col + other_len {
                    return false;
                }
            }
        }
        true
    }

    /// `884b4b5` had a FALSE NEGATIVE — it declares two genuinely overlapping
    /// borrows disjoint — whenever a row wraps past the stride
    /// (`start % stride + width > stride`), which its safe public API
    /// (`index_mut_strided(index, stride, width)`) never checked for.
    #[test]
    fn the_2026_03_predicate_has_a_false_negative_on_a_wrapping_row() {
        // Strided: stride 100, one row of 20 bytes starting at column 90, so it
        // covers [90..110) — i.e. columns 90..100 of row 0 AND 0..10 of row 1.
        // Flat: one byte at 105, squarely inside that.
        let ours = Rect {
            lo: 90,
            w: 20,
            rows: 1,
            stride: 100,
        };
        assert!(
            ours.hits_interval(105, 106),
            "the byte-set definition says these overlap"
        );
        assert!(
            !ranges_overlap_884b4b5(90, 110, 100, 20, 105, 106, 0, 0),
            "884b4b5 is expected to MISS this — that is the point of the test"
        );
    }

    /// The other direction: `884b4b5` also answers conservatively (a false
    /// POSITIVE, i.e. a decode failure if it ever fires) for two strided
    /// borrows with DIFFERENT strides, which its own commit message admits.
    #[test]
    fn the_2026_03_predicate_falls_back_to_1d_on_mixed_strides() {
        // A: stride 10, rows at 0 and 10, 2 bytes each -> {0,1,10,11}.
        // B: stride 7, rows at 3 and 10, 2 bytes each -> {3,4,10,11}. Overlap at
        // {10,11}, so `true` is correct here...
        let a = Rect {
            lo: 0,
            w: 2,
            rows: 2,
            stride: 10,
        };
        let b = Rect {
            lo: 3,
            w: 2,
            rows: 2,
            stride: 7,
        };
        assert!(a.hits_rect(&b).0);
        // ...but shift B down one and the sets are disjoint ({2,3,9,10} vs
        // {0,1,10,11} still shares 10; use 4 instead) — pick a truly disjoint
        // pair and show 884b4b5 still says "overlap".
        let b2 = Rect {
            lo: 4,
            w: 2,
            rows: 2,
            stride: 7,
        };
        // {4,5,11,12} vs {0,1,10,11} shares 11 -> also overlapping. Use stride 7
        // rows at 2,9: {2,3,9,10} vs {0,1,10,11} shares 10. Take w=1.
        let b3 = Rect {
            lo: 2,
            w: 1,
            rows: 2,
            stride: 7,
        }; // {2, 9}
        assert!(
            !a.hits_rect(&b3).0,
            "{{0,1,10,11}} and {{2,9}} are disjoint"
        );
        assert!(
            ranges_overlap_884b4b5(0, 12, 10, 2, 2, 10, 7, 1),
            "884b4b5 falls back to the 1-D hull answer and rejects a disjoint pair"
        );
        let _ = b2;
    }
}

// =============================================================================
// Reset / report
// =============================================================================

/// Zero the counters, keep the interning, so a warmup decode does not pollute.
pub fn reset() {
    for t in 0..MAX_TH {
        for s in 0..NSITES {
            for i in 0..c::NCTR {
                SITE_AGG[t][s][i].store(0, Relaxed);
            }
            SITE_MINGAP[t][s].store(u64::MAX, Relaxed);
            SITE_MINGAP_MUT[t][s].store(u64::MAX, Relaxed);
        }
        for k in 0..NPAIR {
            PAIR_KEY[t][k].store(u64::MAX, Relaxed);
            PAIR_MINGAP[t][k].store(u64::MAX, Relaxed);
            for i in 0..p::NCTR {
                PAIR_AGG[t][k][i].store(0, Relaxed);
            }
        }
    }
    for t in 0..MAX_TH {
        for s in 0..NSITES {
            for i in 0..r::NCTR {
                RECT_AGG[t][s][i].store(0, Relaxed);
            }
        }
    }
    GAPS_N.store(0, Relaxed);
    RECT_ROWS_CAPPED.store(0, Relaxed);
    SAMP_N.store(0, Relaxed);
    OVL_N.store(0, Relaxed);
    OVL_MUT.store(0, Relaxed);
    for i in 0..NINST {
        INST_N[i].store(0, Relaxed);
    }
    INST_LATE_STRIDE.store(0, Relaxed);
    LOST_SLOT.store(0, Relaxed);
    LOST_TH.store(0, Relaxed);
    LOST_SITE.store(0, Relaxed);
    LOST_SCAN.store(0, Relaxed);
    LOST_INST.store(0, Relaxed);
    SKIP_DEAD.store(0, Relaxed);
    LOST_PAIR.store(0, Relaxed);
}

/// Total registrations recorded, and how many of them came from a site whose
/// `file:line:col` contains `needle`. The liveness proof for a code path: an
/// instrument that never saw loop restoration has measured nothing about it.
pub fn regs_matching(needle: &str) -> (u64, u64) {
    let mut total = 0u64;
    let mut hit = 0u64;
    for s in 0..NSITES {
        if SITE_KEY[s].load(Relaxed) == 0 {
            continue;
        }
        let mut n = 0u64;
        for t in 0..MAX_TH {
            n += SITE_AGG[t][s][c::N].load(Relaxed);
        }
        if n == 0 {
            continue;
        }
        total += n;
        if site_name(s as u32).contains(needle) {
            hit += n;
        }
    }
    (total, hit)
}

fn site_name(id: u32) -> String {
    let key = SITE_KEY[id as usize].load(Relaxed);
    SITE_NAMES
        .lock()
        .ok()
        .and_then(|n| {
            n.iter()
                .find(|(k, _)| *k == key)
                .map(|(_, l)| std::format!("{}:{}:{}", l.file(), l.line(), l.column()))
        })
        .unwrap_or_else(|| std::format!("?{key:#x}"))
}

/// The full report. `frames` divides the per-frame columns.
pub fn report(frames: u64) -> String {
    use std::fmt::Write as _;
    let f = frames.max(1) as f64;
    let mut out = String::new();

    // ---- per-site ----------------------------------------------------------
    let mut rows: Vec<(u64, u32, [u64; c::NCTR], u64, u64)> = Vec::new();
    let mut total = 0u64;
    for s in 0..NSITES {
        if SITE_KEY[s].load(Relaxed) == 0 {
            continue;
        }
        let mut ctr = [0u64; c::NCTR];
        let mut mg = u64::MAX;
        let mut mgm = u64::MAX;
        for t in 0..MAX_TH {
            for i in 0..c::NCTR {
                ctr[i] += SITE_AGG[t][s][i].load(Relaxed);
            }
            mg = mg.min(SITE_MINGAP[t][s].load(Relaxed));
            mgm = mgm.min(SITE_MINGAP_MUT[t][s].load(Relaxed));
        }
        if ctr[c::N] == 0 {
            continue;
        }
        total += ctr[c::N];
        rows.push((ctr[c::N], s as u32, ctr, mg, mgm));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));

    let _ = writeln!(
        out,
        "BOUNDS\ttotal_per_frame={:.0}\tdistinct={}\tlost_slot={}\tlost_th={}\tlost_site={}\tlost_scan={}\tskip_dead={}\tlost_inst={}\tlost_pair={}\tlost_tid={}",
        total as f64 / f,
        rows.len(),
        LOST_SLOT.load(Relaxed),
        LOST_TH.load(Relaxed),
        LOST_SITE.load(Relaxed),
        LOST_SCAN.load(Relaxed),
        SKIP_DEAD.load(Relaxed),
        LOST_INST.load(Relaxed),
        LOST_PAIR.load(Relaxed),
        LOST_TID.load(Relaxed),
    );

    let _ = writeln!(
        out,
        "#bsite\tper_frame\tmut\tres_mean\tfp_mean\tover_ratio\tfp_kind\tn_rows_decl\tn_whole\tn_never\trows_mean\tw_mean\tgap_mean\tlead_mean\ttail_mean\tn_read\tn_write\tlive_epochs_mean\twhere"
    );
    for (n, s, ctr, _mg, _mgm) in rows.iter() {
        let nn = *n as f64;
        let res = ctr[c::RES_BYTES] as f64 / nn;
        let fp = ctr[c::FP_BYTES] as f64 / nn;
        let kind = if ctr[c::N_ROWS_DECL] * 2 > *n {
            "rows"
        } else if ctr[c::N_NEVER] * 2 > *n {
            "none"
        } else {
            "whole"
        };
        let nd = ctr[c::N_ROWS_DECL].max(1) as f64;
        let _ = writeln!(
            out,
            "BSITE\t{:.0}\t{:.0}\t{:.2}\t{:.2}\t{:.3}\t{}\t{:.0}\t{:.0}\t{:.0}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.0}\t{:.0}\t{:.1}\t{}",
            nn / f,
            ctr[c::N_MUT] as f64 / f,
            res,
            fp,
            if fp > 0.0 { res / fp } else { f64::INFINITY },
            kind,
            ctr[c::N_ROWS_DECL] as f64 / f,
            ctr[c::N_WHOLE] as f64 / f,
            ctr[c::N_NEVER] as f64 / f,
            ctr[c::ROWS_SUM] as f64 / nd,
            ctr[c::W_SUM] as f64 / nd,
            ctr[c::GAP_BYTES] as f64 / nd,
            ctr[c::LEAD_WASTE] as f64 / nd,
            ctr[c::TAIL_WASTE] as f64 / nd,
            ctr[c::N_READ] as f64 / f,
            ctr[c::N_WRITE] as f64 / f,
            ctr[c::LIVE_EPOCHS] as f64 / nn,
            site_name(*s)
        );
    }

    // ---- per-site concurrency / counterfactual ------------------------------
    let _ = writeln!(
        out,
        "#bconc(RAW totals over {frames} frames, not per-frame)\tn_raw\tn_conc\tpct_conc\tn_conc_mut\tn_res_ovl\tn_fp_ovl\tn_row_ovl\tn_row_ovl_mut\tn_row_fp_ovl\tn_row_fp_ovl_mut\tmin_gap\tmin_gap_mut\t{gaps}\twhere",
        frames = frames,
        gaps = GAP_LABELS.join("\t")
    );
    for (n, s, ctr, mg, mgm) in rows.iter() {
        let nn = *n as f64;
        let mut hist = String::new();
        for i in 0..NGAP {
            let _ = write!(hist, "{}\t", ctr[c::GAP_HIST + i]);
        }
        let _ = writeln!(
            out,
            "BCONC\t{}\t{}\t{:.4}%\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}{}",
            n,
            ctr[c::N_CONC],
            100.0 * ctr[c::N_CONC] as f64 / nn,
            ctr[c::N_CONC_MUT],
            ctr[c::N_RES_OVL],
            ctr[c::N_FP_OVL],
            ctr[c::N_ROW_OVL],
            ctr[c::N_ROW_OVL_MUT],
            ctr[c::N_ROW_FP_OVL],
            ctr[c::N_ROW_FP_OVL_MUT],
            if *mg == u64::MAX {
                String::from("-")
            } else {
                std::format!("{mg}")
            },
            if *mgm == u64::MAX {
                String::from("-")
            } else {
                std::format!("{mgm}")
            },
            hist,
            site_name(*s)
        );
    }

    // ---- the gap histogram against MUTABLE foreign records only ------------
    // "How far is the nearest byte another worker is CONCURRENTLY WRITING?"
    // Widening this site's reservation by k bytes collides in exactly the sum
    // of the buckets strictly below k.
    let _ = writeln!(
        out,
        "#bgapmut(RAW)\tn_raw\t{}\twhere",
        GAP_LABELS
            .iter()
            .map(|l| std::format!("m{}", &l[1..]))
            .collect::<Vec<_>>()
            .join("\t")
    );
    for (n, s, ctr, _, _) in rows.iter() {
        let mut hist = String::new();
        for i in 0..NGAP {
            let _ = write!(hist, "{}\t", ctr[c::GAP_HIST_MUT + i]);
        }
        let _ = writeln!(out, "BGAPMUT\t{}\t{}{}", n, hist, site_name(*s));
    }

    // ---- per-pair ----------------------------------------------------------
    let mut pairs: Vec<(u64, u64, [u64; p::NCTR], u64)> = Vec::new();
    for t in 0..MAX_TH {
        for k in 0..NPAIR {
            let key = PAIR_KEY[t][k].load(Relaxed);
            if key == u64::MAX {
                continue;
            }
            let mut ctr = [0u64; p::NCTR];
            for i in 0..p::NCTR {
                ctr[i] = PAIR_AGG[t][k][i].load(Relaxed);
            }
            if ctr[p::N] == 0 {
                continue;
            }
            let mg = PAIR_MINGAP[t][k].load(Relaxed);
            match pairs.iter_mut().find(|(kk, _, _, _)| *kk == key) {
                Some(e) => {
                    e.1 += ctr[p::N];
                    for i in 0..p::NCTR {
                        e.2[i] += ctr[i];
                    }
                    e.3 = e.3.min(mg);
                }
                None => pairs.push((key, ctr[p::N], ctr, mg)),
            }
        }
    }
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    let _ = writeln!(
        out,
        "#bpair(RAW)\tn\tn_res_ovl\tn_fp_ovl\tn_row_ovl\tn_foreign_mut\tmin_gap\tacquiring_site\tconcurrent_site"
    );
    for (key, _, ctr, mg) in pairs.iter() {
        let a = (key >> 20) as u32;
        let b = (key & 0xFFFFF) as u32;
        let _ = writeln!(
            out,
            "BPAIR\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            ctr[p::N],
            ctr[p::N_RES_OVL],
            ctr[p::N_FP_OVL],
            ctr[p::N_ROW_OVL],
            ctr[p::N_FOREIGN_MUT],
            if *mg == u64::MAX {
                String::from("-")
            } else {
                std::format!("{mg}")
            },
            site_name(a),
            site_name(b)
        );
    }

    // ---- every reservation overlap, in full --------------------------------
    let novl = OVL_N.load(Relaxed);
    let _ = writeln!(
        out,
        "#bovl(a concurrent overlap involving a MUTABLE record is IMPOSSIBLE — the tracker panics on it — so those rows are this instrument's own false positives)\tepoch\ttid\tstart\tend\tismut\tftid\tfacq\tfrel\tfstart\tfend\tfismut\tsite\tfsite"
    );
    let _ = writeln!(
        out,
        "BOVLSUM\tn={novl}\tmutable_overlaps={}\tsampled={}",
        OVL_MUT.load(Relaxed),
        novl.min(NOVL)
    );
    for i in 0..novl.min(NOVL) {
        let r = &OVL[i];
        let fe = r[10].load(Relaxed);
        let _ = writeln!(
            out,
            "BOVL\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r[0].load(Relaxed),
            r[1].load(Relaxed),
            r[2].load(Relaxed),
            r[3].load(Relaxed),
            r[4].load(Relaxed),
            r[6].load(Relaxed),
            r[7].load(Relaxed),
            r[8].load(Relaxed),
            r[9].load(Relaxed),
            fe >> 1,
            fe & 1,
            site_name(r[5].load(Relaxed) as u32),
            site_name(r[11].load(Relaxed) as u32),
        );
    }

    // ---- per-instance ------------------------------------------------------
    let _ = writeln!(out, "#binst\tn_per_frame\tstride_bytes\tlen_bytes\tbase");
    let mut insts: Vec<(u64, i64, u64, usize)> = Vec::new();
    for i in 0..NINST {
        let k = INST_KEY[i].load(Relaxed);
        if k == 0 {
            continue;
        }
        let n = INST_N[i].load(Relaxed);
        if n == 0 {
            continue;
        }
        insts.push((
            n,
            INST_STRIDE[i].load(Relaxed),
            INST_LEN[i].load(Relaxed),
            k,
        ));
    }
    insts.sort_by(|a, b| b.0.cmp(&a.0));
    for (n, st, l, k) in insts.iter().take(40) {
        let _ = writeln!(out, "BINST\t{:.0}\t{}\t{}\t{:#x}", *n as f64 / f, st, l, k);
    }
    let _ = writeln!(
        out,
        "BINSTSUM\tdistinct={}\tlate_stride={}",
        insts.len(),
        INST_LATE_STRIDE.load(Relaxed)
    );

    // ---- raw sample --------------------------------------------------------
    let n = SAMP_N.load(Relaxed).min(NSAMP);
    let _ = writeln!(
        out,
        "#bsamp\tepoch\ttid\tstart\tend\tismut\tftid\tfstart\tfend\tfismut\tsite\tfsite"
    );
    for i in 0..n {
        let r = &SAMP[i];
        let fe = r[9].load(Relaxed);
        let _ = writeln!(
            out,
            "BSAMP\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r[0].load(Relaxed),
            r[1].load(Relaxed),
            r[3].load(Relaxed),
            r[4].load(Relaxed),
            r[5].load(Relaxed),
            r[6].load(Relaxed),
            r[8].load(Relaxed),
            fe >> 1,
            fe & 1,
            site_name(r[2].load(Relaxed) as u32),
            site_name(r[7].load(Relaxed) as u32),
        );
    }

    out
}
