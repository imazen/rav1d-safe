//! THROWAWAY measurement probe for BorrowTracker contention.
//!
//! Feature `__probe_count` only. Never merge this; it exists to answer
//! "where does the tracker's time go" with counts rather than guesses.
//!
//! Per-tracker-instance slots (lazily assigned) record how many borrows each
//! `DisjointMut` sees, how contended its lock is, how full its 64 inline slots
//! get, and how long contended acquisitions actually wait.

use core::panic::Location;
use core::sync::atomic::AtomicU32;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering::Relaxed;
use std::string::String;
use std::vec::Vec;

pub const MAX_SLOTS: usize = 16384;
pub const MAX_THREADS: usize = 64;

pub struct Slot {
    pub n_mut: AtomicU64,
    pub n_immut: AtomicU64,
    pub n_remove: AtomicU64,
    /// Contended acquisitions (lock_slow entered), split by call kind.
    pub cont_add: AtomicU64,
    pub cont_remove: AtomicU64,
    /// Nanoseconds spent spinning in lock_slow (contended path only).
    pub wait_ns_add: AtomicU64,
    pub wait_ns_remove: AtomicU64,
    /// Sum / max of `occupied.count_ones()` observed at add time (before alloc).
    pub occ_sum: AtomicU64,
    pub occ_max: AtomicU64,
    /// Borrows that spilled past the 64 inline slots.
    pub overflow: AtomicU64,
    /// Largest `end` offset ever registered — a size fingerprint for the buffer.
    pub max_end: AtomicUsize,
    /// First registration site seen (`&'static Location` as usize).
    pub loc: AtomicUsize,
    /// Bitmask of thread indices that borrowed this instance.
    pub thread_mask: AtomicU64,
}

impl Slot {
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        Self {
            n_mut: AtomicU64::new(0),
            n_immut: AtomicU64::new(0),
            n_remove: AtomicU64::new(0),
            cont_add: AtomicU64::new(0),
            cont_remove: AtomicU64::new(0),
            wait_ns_add: AtomicU64::new(0),
            wait_ns_remove: AtomicU64::new(0),
            occ_sum: AtomicU64::new(0),
            occ_max: AtomicU64::new(0),
            overflow: AtomicU64::new(0),
            max_end: AtomicUsize::new(0),
            loc: AtomicUsize::new(0),
            thread_mask: AtomicU64::new(0),
        }
    }

    fn total(&self) -> u64 {
        self.n_mut.load(Relaxed) + self.n_immut.load(Relaxed)
    }
}

pub struct ThreadSlot {
    pub n_mut: AtomicU64,
    pub n_immut: AtomicU64,
    pub n_remove: AtomicU64,
    pub contended: AtomicU64,
    pub wait_ns: AtomicU64,
}

impl ThreadSlot {
    pub const fn new() -> Self {
        Self {
            n_mut: AtomicU64::new(0),
            n_immut: AtomicU64::new(0),
            n_remove: AtomicU64::new(0),
            contended: AtomicU64::new(0),
            wait_ns: AtomicU64::new(0),
        }
    }
}

pub static SLOTS: [Slot; MAX_SLOTS] = [const { Slot::new() }; MAX_SLOTS];
pub static THREADS: [ThreadSlot; MAX_THREADS] = [const { ThreadSlot::new() }; MAX_THREADS];
pub static NEXT_SLOT: AtomicU32 = AtomicU32::new(0);
pub static NEXT_THREAD: AtomicU32 = AtomicU32::new(0);
/// Borrows dropped on the floor because we ran out of probe slots.
pub static SLOT_EXHAUSTED: AtomicU64 = AtomicU64::new(0);

std::thread_local! {
    static TID: core::cell::Cell<u32> = const { core::cell::Cell::new(u32::MAX) };
}

#[inline]
pub fn thread_index() -> usize {
    TID.with(|c| {
        let mut t = c.get();
        if t == u32::MAX {
            t = NEXT_THREAD.fetch_add(1, Relaxed);
            if t as usize >= MAX_THREADS {
                t = (MAX_THREADS - 1) as u32;
            }
            c.set(t);
        }
        t as usize
    })
}

/// Assign a probe slot to a tracker on first use.
#[inline]
pub fn assign_slot(cell: &AtomicU32) -> usize {
    let cur = cell.load(Relaxed);
    if cur != u32::MAX {
        return cur as usize;
    }
    let new = NEXT_SLOT.fetch_add(1, Relaxed);
    if new as usize >= MAX_SLOTS {
        SLOT_EXHAUSTED.fetch_add(1, Relaxed);
        return MAX_SLOTS - 1;
    }
    // Racy only if two threads first-touch the same tracker simultaneously;
    // the loser's slot is simply wasted, which does not corrupt any count.
    match cell.compare_exchange(u32::MAX, new, Relaxed, Relaxed) {
        Ok(_) => new as usize,
        Err(actual) => actual as usize,
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn record_add(
    slot: usize,
    is_mut: bool,
    end: usize,
    occupancy: u32,
    spilled: bool,
    wait_ns: u64,
    contended: bool,
    loc: &'static Location<'static>,
) {
    let s = &SLOTS[slot];
    if is_mut {
        s.n_mut.fetch_add(1, Relaxed);
    } else {
        s.n_immut.fetch_add(1, Relaxed);
    }
    s.occ_sum.fetch_add(occupancy as u64, Relaxed);
    if s.occ_max.load(Relaxed) < occupancy as u64 {
        s.occ_max.store(occupancy as u64, Relaxed);
    }
    if spilled {
        s.overflow.fetch_add(1, Relaxed);
    }
    if s.max_end.load(Relaxed) < end {
        s.max_end.store(end, Relaxed);
    }
    if s.loc.load(Relaxed) == 0 {
        s.loc
            .store(loc as *const Location<'static> as usize, Relaxed);
    }
    let tid = thread_index();
    s.thread_mask.fetch_or(1u64 << tid, Relaxed);
    if contended {
        s.cont_add.fetch_add(1, Relaxed);
        s.wait_ns_add.fetch_add(wait_ns, Relaxed);
    }
    let t = &THREADS[tid];
    if is_mut {
        t.n_mut.fetch_add(1, Relaxed);
    } else {
        t.n_immut.fetch_add(1, Relaxed);
    }
    if contended {
        t.contended.fetch_add(1, Relaxed);
        t.wait_ns.fetch_add(wait_ns, Relaxed);
    }
}

#[inline]
pub fn record_remove(slot: usize, wait_ns: u64, contended: bool) {
    let s = &SLOTS[slot];
    s.n_remove.fetch_add(1, Relaxed);
    if contended {
        s.cont_remove.fetch_add(1, Relaxed);
        s.wait_ns_remove.fetch_add(wait_ns, Relaxed);
    }
    let t = &THREADS[thread_index()];
    t.n_remove.fetch_add(1, Relaxed);
    if contended {
        t.contended.fetch_add(1, Relaxed);
        t.wait_ns.fetch_add(wait_ns, Relaxed);
    }
}

/// Tab-separated report. `iters` scales counts to per-frame.
pub fn report(iters: u64) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let iters = iters.max(1);

    let used = (NEXT_SLOT.load(Relaxed) as usize).min(MAX_SLOTS);
    let mut idx: std::vec::Vec<usize> = (0..used).collect();
    idx.sort_by_key(|&i| core::cmp::Reverse(SLOTS[i].total()));

    let mut tot_add = 0u64;
    let mut tot_rem = 0u64;
    let mut tot_cont = 0u64;
    let mut tot_wait = 0u64;
    let mut tot_spill = 0u64;
    for i in 0..used {
        let s = &SLOTS[i];
        tot_add += s.total();
        tot_rem += s.n_remove.load(Relaxed);
        tot_cont += s.cont_add.load(Relaxed) + s.cont_remove.load(Relaxed);
        tot_wait += s.wait_ns_add.load(Relaxed) + s.wait_ns_remove.load(Relaxed);
        tot_spill += s.overflow.load(Relaxed);
    }

    let _ = writeln!(out, "PROBE\ttracker_instances_used\t{used}");
    let _ = writeln!(
        out,
        "PROBE\tslot_exhausted_events\t{}",
        SLOT_EXHAUSTED.load(Relaxed)
    );
    let _ = writeln!(out, "PROBE\ttotal_adds\t{tot_add}");
    let _ = writeln!(out, "PROBE\ttotal_removes\t{tot_rem}");
    let _ = writeln!(out, "PROBE\tadds_per_frame\t{}", tot_add / iters);
    let _ = writeln!(
        out,
        "PROBE\tlock_ops_per_frame\t{}",
        (tot_add + tot_rem) / iters
    );
    let _ = writeln!(out, "PROBE\tcontended_acquisitions\t{tot_cont}");
    let _ = writeln!(
        out,
        "PROBE\tcontended_pct\t{:.4}",
        100.0 * tot_cont as f64 / (tot_add + tot_rem).max(1) as f64
    );
    let _ = writeln!(out, "PROBE\ttotal_spin_wait_ns\t{tot_wait}");
    let _ = writeln!(out, "PROBE\tspin_wait_ns_per_frame\t{}", tot_wait / iters);
    let _ = writeln!(
        out,
        "PROBE\tmean_wait_ns_per_contended\t{:.1}",
        tot_wait as f64 / tot_cont.max(1) as f64
    );
    let _ = writeln!(out, "PROBE\toverflow_spills\t{tot_spill}");

    let _ = writeln!(
        out,
        "SLOT\trank\tadds\tremoves\tadds_pct\tcont_add\tcont_rem\tcont_pct\twait_ns\tocc_mean\tocc_max\tspill\tmax_end\tthreads\tloc"
    );
    for (rank, &i) in idx.iter().enumerate().take(20) {
        let s = &SLOTS[i];
        let adds = s.total();
        if adds == 0 {
            continue;
        }
        let rem = s.n_remove.load(Relaxed);
        let ca = s.cont_add.load(Relaxed);
        let cr = s.cont_remove.load(Relaxed);
        let wait = s.wait_ns_add.load(Relaxed) + s.wait_ns_remove.load(Relaxed);
        let loc_ptr = s.loc.load(Relaxed);
        let loc = if loc_ptr == 0 {
            std::borrow::Cow::Borrowed("?")
        } else {
            // SAFETY-equivalent: the pointer came from a `&'static Location`.
            let l: &'static Location<'static> = unsafe { &*(loc_ptr as *const Location<'static>) };
            std::borrow::Cow::Owned(std::format!("{}:{}", l.file(), l.line()))
        };
        let _ = writeln!(
            out,
            "SLOT\t{rank}\t{adds}\t{rem}\t{:.2}\t{ca}\t{cr}\t{:.3}\t{wait}\t{:.2}\t{}\t{}\t{}\t{}\t{loc}",
            100.0 * adds as f64 / tot_add.max(1) as f64,
            100.0 * (ca + cr) as f64 / (adds + rem).max(1) as f64,
            s.occ_sum.load(Relaxed) as f64 / adds.max(1) as f64,
            s.occ_max.load(Relaxed),
            s.overflow.load(Relaxed),
            s.max_end.load(Relaxed),
            s.thread_mask.load(Relaxed).count_ones(),
        );
    }

    let nthreads = (NEXT_THREAD.load(Relaxed) as usize).min(MAX_THREADS);
    let _ = writeln!(out, "THREAD\ttid\tadds\tremoves\tcontended\twait_ns");
    for t in 0..nthreads {
        let ts = &THREADS[t];
        let a = ts.n_mut.load(Relaxed) + ts.n_immut.load(Relaxed);
        if a == 0 && ts.n_remove.load(Relaxed) == 0 {
            continue;
        }
        let _ = writeln!(
            out,
            "THREAD\t{t}\t{a}\t{}\t{}\t{}",
            ts.n_remove.load(Relaxed),
            ts.contended.load(Relaxed),
            ts.wait_ns.load(Relaxed)
        );
    }
    out
}

/// Zero every counter (call after warmup so the report covers timed work only).
pub fn reset() {
    for s in SLOTS.iter() {
        s.n_mut.store(0, Relaxed);
        s.n_immut.store(0, Relaxed);
        s.n_remove.store(0, Relaxed);
        s.cont_add.store(0, Relaxed);
        s.cont_remove.store(0, Relaxed);
        s.wait_ns_add.store(0, Relaxed);
        s.wait_ns_remove.store(0, Relaxed);
        s.occ_sum.store(0, Relaxed);
        s.occ_max.store(0, Relaxed);
        s.overflow.store(0, Relaxed);
    }
    for t in THREADS.iter() {
        t.n_mut.store(0, Relaxed);
        t.n_immut.store(0, Relaxed);
        t.n_remove.store(0, Relaxed);
        t.contended.store(0, Relaxed);
        t.wait_ns.store(0, Relaxed);
    }
    SLOT_EXHAUSTED.store(0, Relaxed);
}

// =============================================================================
// THROWAWAY shard-sizing probe (`__probe_shardsim`)
// =============================================================================
//
// Answers three questions for a *hypothetical* address-sharded tracker, using
// counts only (no timing, so it is immune to a busy box):
//
//  1. How long are borrows?  -> log2 length histogram.
//  2. How many shards would a borrow span at a given block shift?  -> k.
//     k is the multiplier on lock round trips for the sound multi-shard design.
//  3. Do concurrently-running tile workers actually land on DIFFERENT shards?
//     -> for each add, how many of the other live threads' most recent borrow
//     was on the same (instance, shard) pair. N=1 reproduces today's design and
//     is the calibration point; the ideal for N shards is (threads-1)/N.
//
// Restricted to instances whose observed max_end is >= 1 MiB, i.e. the 12
// picture planes that carry 89.8% of borrows and 100% of contention.

/// (block_shift, n_shards, mixed)
pub const SHARD_CFGS: [(u32, u32, bool); 8] = [
    (0, 1, false),  // 0: status quo — one lock per instance
    (6, 64, true),  // 1: 64 B blocks
    (7, 64, true),  // 2: 128 B
    (8, 64, true),  // 3: 256 B
    (10, 64, true), // 4: 1 KiB
    (12, 64, true), // 5: 4 KiB
    (8, 64, false), // 6: 256 B, linear (no mixing) — alignment-sensitivity check
    (8, 256, true), // 7: 256 B, 256 shards
];
pub const N_CFG: usize = SHARD_CFGS.len();

/// Log2 length histogram, 0..=20 plus overflow.
pub static LEN_HIST: [AtomicU64; 24] = [const { AtomicU64::new(0) }; 24];
/// k (shards spanned) histogram per config: 1, 2, 3, 4, 5..8, 9..16, 17..64, >64
pub static K_HIST: [[AtomicU64; 8]; N_CFG] = [const { [const { AtomicU64::new(0) }; 8] }; N_CFG];
/// Sum of k, per config (exact lock-op multiplier).
pub static K_SUM: [AtomicU64; N_CFG] = [const { AtomicU64::new(0) }; N_CFG];
/// Adds observed by the shard sim (hot instances only).
pub static SHARD_ADDS: AtomicU64 = AtomicU64::new(0);
/// Sum over adds of "how many other threads' most recent borrow was on the
/// same (instance, shard)".  Divided by SHARD_ADDS this is the expected number
/// of colliding peers per registration.
pub static COLLIDE_SUM: [AtomicU64; N_CFG] = [const { AtomicU64::new(0) }; N_CFG];
/// Adds where at least one other thread collided.
pub static COLLIDE_ANY: [AtomicU64; N_CFG] = [const { AtomicU64::new(0) }; N_CFG];
/// Per (config, thread): packed (slot << 24) | shard of that thread's most
/// recent hot-instance borrow.  u64::MAX = never borrowed.
pub static LAST: [[AtomicU64; MAX_THREADS]; N_CFG] =
    [const { [const { AtomicU64::new(u64::MAX) }; MAX_THREADS] }; N_CFG];
/// Per-shard add counts for cfg index 3 (256 B / 64 shards) — uniformity check.
pub static SHARD_SPREAD: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

#[inline]
fn shard_of(block: u64, n: u32, mixed: bool) -> u32 {
    if n == 1 {
        return 0;
    }
    if mixed {
        (((block.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 40) as u32) & (n - 1)
    } else {
        (block as u32) & (n - 1)
    }
}

/// Called from `add_probed` while NOT holding the tracker lock.
/// `slot` is the probe slot for the instance, `start`/`end` the byte range.
pub fn record_shard(slot: usize, start: usize, end: usize, max_end: usize) {
    // Hot instances only: the picture planes.
    if max_end < (1 << 20) {
        return;
    }
    let len = end - start;
    let lz = (usize::BITS - 1 - len.leading_zeros()) as usize;
    LEN_HIST[lz.min(23)].fetch_add(1, Relaxed);
    SHARD_ADDS.fetch_add(1, Relaxed);
    let tid = thread_index();
    for (c, &(shift, n, mixed)) in SHARD_CFGS.iter().enumerate() {
        let b0 = (start as u64) >> shift;
        let b1 = ((end - 1) as u64) >> shift;
        let k = (b1 - b0 + 1).min(1 << 20);
        K_SUM[c].fetch_add(k, Relaxed);
        let bucket = match k {
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            5..=8 => 4,
            9..=16 => 5,
            17..=64 => 6,
            _ => 7,
        };
        K_HIST[c][bucket].fetch_add(1, Relaxed);

        let s = shard_of(b0, n, mixed);
        if c == 3 {
            SHARD_SPREAD[(s & 63) as usize].fetch_add(1, Relaxed);
        }
        let packed = ((slot as u64) << 24) | s as u64;
        let mut hits = 0u64;
        let active = (NEXT_THREAD.load(Relaxed) as usize).min(MAX_THREADS);
        for t in 0..active {
            if t == tid {
                continue;
            }
            if LAST[c][t].load(Relaxed) == packed {
                hits += 1;
            }
        }
        if hits > 0 {
            COLLIDE_SUM[c].fetch_add(hits, Relaxed);
            COLLIDE_ANY[c].fetch_add(1, Relaxed);
        }
        LAST[c][tid].store(packed, Relaxed);
    }
}

pub fn shard_report() -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let adds = SHARD_ADDS.load(Relaxed).max(1);
    let _ = writeln!(out, "SHARD\thot_adds\t{}", SHARD_ADDS.load(Relaxed));
    for (i, h) in LEN_HIST.iter().enumerate() {
        let v = h.load(Relaxed);
        if v > 0 {
            let _ = writeln!(
                out,
                "LEN\t{}\t{}\t{:.4}",
                1usize << i,
                v,
                v as f64 * 100.0 / adds as f64
            );
        }
    }
    let _ = writeln!(
        out,
        "CFGHDR\tcfg\tshift\tn\tmixed\tk_mean\tk1_pct\tk2_pct\tk3_4_pct\tk_gt4_pct\tcollide_per_add\tcollide_any_pct"
    );
    for (c, &(shift, n, mixed)) in SHARD_CFGS.iter().enumerate() {
        let ks: Vec<u64> = K_HIST[c].iter().map(|a| a.load(Relaxed)).collect();
        let k_mean = K_SUM[c].load(Relaxed) as f64 / adds as f64;
        let p = |v: u64| v as f64 * 100.0 / adds as f64;
        let _ = writeln!(
            out,
            "CFG\t{c}\t{shift}\t{n}\t{mixed}\t{k_mean:.4}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.5}\t{:.4}",
            p(ks[0]),
            p(ks[1]),
            p(ks[2] + ks[3]),
            p(ks[4] + ks[5] + ks[6] + ks[7]),
            COLLIDE_SUM[c].load(Relaxed) as f64 / adds as f64,
            p(COLLIDE_ANY[c].load(Relaxed)),
        );
    }
    for (i, s) in SHARD_SPREAD.iter().enumerate() {
        let _ = writeln!(out, "SPREAD\t{i}\t{}", s.load(Relaxed));
    }
    out
}

pub fn shard_reset() {
    for a in LEN_HIST.iter() {
        a.store(0, Relaxed);
    }
    for c in 0..N_CFG {
        K_SUM[c].store(0, Relaxed);
        COLLIDE_SUM[c].store(0, Relaxed);
        COLLIDE_ANY[c].store(0, Relaxed);
        for b in 0..8 {
            K_HIST[c][b].store(0, Relaxed);
        }
        for t in 0..MAX_THREADS {
            LAST[c][t].store(u64::MAX, Relaxed);
        }
    }
    for a in SHARD_SPREAD.iter() {
        a.store(0, Relaxed);
    }
    SHARD_ADDS.store(0, Relaxed);
}
