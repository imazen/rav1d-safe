//! THROWAWAY: price ONE borrow acquire+release pair, and price the individual
//! atomic operations the tracker's fast path is built from, on this host.
//!
//! The decode-level A/B says the tracker costs 83.5 ms/frame at t=1 8bpc
//! (373.5 tracked vs 290.0 `probe-untracked`). This runner answers the
//! question that decides WHICH lever is worth building: of one acquire+release
//! pair's cost, how much is the three locked RMWs (swap / fetch_or /
//! fetch_and) and how much is everything else?
//!
//! Every arm is measured in the SAME process, interleaved, rotating order,
//! against the same `DisjointMut`, so a thermal drift cannot masquerade as a
//! delta. Absolute ns are microbenchmark ns — an L1-resident shard line and a
//! perfectly-predicted loop — and are a LOWER bound on the decoder's cost per
//! borrow. The RATIOS between arms are the deliverable.
//!
//! Usage: probe_borrow_cost [iters] [rounds]

use rav1d_disjoint_mut::DisjointMut;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Instant;

/// One 8.3 MB plane, the shape the hot instances actually have.
const LEN: usize = 3840 * 2176;

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(2_000_000);
    let rounds: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(9);

    let buf: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    // SAFETY: measurement-only arm; every borrow below is a 1-byte read/write
    // taken and dropped before the next, so no two are ever live at once.
    let unch: DisjointMut<Vec<u8>> = unsafe { DisjointMut::dangerously_unchecked(vec![0u8; LEN]) };
    // Warm the tracker's shard line and the buffer page we touch.
    for _ in 0..1000 {
        let g = buf.index_mut(17..18);
        black_box(&g);
        let g = unch.index_mut(17..18);
        black_box(&g);
    }

    // Standalone atomics, on their own cache line, uncontended and L1-hot:
    // the individual instructions the tracker's fast path is made of.
    #[repr(align(128))]
    struct Line {
        lock: AtomicBool,
        occ: AtomicU8,
    }
    let line = Line {
        lock: AtomicBool::new(false),
        occ: AtomicU8::new(0),
    };
    // The V1 candidate's shape: one live-flag BYTE per slot, each holding
    // `1 << i` or 0, so the scan mask is an OR-reduction of 7 relaxed loads and
    // publish/retire become plain stores (no RMW, no lost update -- the only
    // writers of slot i's byte are its owner and the allocator that observed it
    // clear under the lock).
    #[repr(align(128))]
    struct Slots {
        lock: AtomicBool,
        live: [AtomicU8; 7],
    }
    let slots = Slots {
        lock: AtomicBool::new(false),
        live: [const { AtomicU8::new(0) }; 7],
    };

    let names = [
        "pair_index_mut", // full acquire+release, mutable
        "pair_index",     // full acquire+release, immutable
        "pair_unchecked", // same guard, NO tracker: everything-but-the-tracker
        "swap_only",      // TinyLock acquire + release store
        "fetchor_only",   // publish
        "fetchand_only",  // retire
        "three_rmw",      // swap + fetch_or + fetch_and, the shipped shape
        "two_rmw",        // swap + fetch_xor            (lock+occ fused)
        "one_rmw",        // swap only, stores for the rest
        "zero_rmw",       // acquire/release loads+stores only
        "three_rmw_rlx",  // shipped shape with publish RELAXED (lock orders it)
        "slotflags",      // V1: per-slot flag bytes, publish/retire plain stores
        "slotflags_hi",   // V2: ditto + a lock-held watermark, so the scan is 1 load
        "tls_seqcst",     // V3: no lock at all -- publish, dmb ish, scan peers
        "fence_only",     // the dmb ish V3 rests on
        "empty",          // loop overhead
    ];
    let mut acc: Vec<Vec<f64>> = vec![Vec::new(); names.len()];

    for r in 0..rounds {
        for k in 0..names.len() {
            let a = (k + r) % names.len();
            let t0 = Instant::now();
            match a {
                0 => {
                    for i in 0..iters {
                        let g = buf.index_mut(black_box(i & 0xFFFF)..black_box((i & 0xFFFF) + 1));
                        black_box(&g);
                    }
                }
                1 => {
                    for i in 0..iters {
                        let g = buf.index(black_box(i & 0xFFFF)..black_box((i & 0xFFFF) + 1));
                        black_box(&g);
                    }
                }
                2 => {
                    for i in 0..iters {
                        let g = unch.index_mut(black_box(i & 0xFFFF)..black_box((i & 0xFFFF) + 1));
                        black_box(&g);
                    }
                }
                3 => {
                    for _ in 0..iters {
                        black_box(line.lock.swap(true, Ordering::Acquire));
                        line.lock.store(false, Ordering::Release);
                    }
                }
                4 => {
                    for _ in 0..iters {
                        black_box(line.occ.fetch_or(1, Ordering::Release));
                    }
                }
                5 => {
                    for _ in 0..iters {
                        black_box(line.occ.fetch_and(!1u8, Ordering::Release));
                    }
                }
                6 => {
                    for _ in 0..iters {
                        black_box(line.lock.swap(true, Ordering::Acquire));
                        black_box(line.occ.load(Ordering::Acquire));
                        black_box(line.occ.fetch_or(1, Ordering::Release));
                        line.lock.store(false, Ordering::Release);
                        black_box(line.occ.fetch_and(!1u8, Ordering::Release));
                    }
                }
                7 => {
                    for _ in 0..iters {
                        // lock bit fused into the occupancy word: fetch_or to
                        // acquire (and snapshot), fetch_xor to publish+release.
                        black_box(line.occ.fetch_or(0x80, Ordering::Acquire));
                        black_box(line.occ.fetch_xor(0x81, Ordering::Release));
                        black_box(line.occ.fetch_and(!1u8, Ordering::Release));
                    }
                }
                8 => {
                    for _ in 0..iters {
                        black_box(line.lock.swap(true, Ordering::Acquire));
                        black_box(line.occ.load(Ordering::Acquire));
                        // per-slot flags: publish and retire are plain stores
                        line.occ.store(1, Ordering::Release);
                        line.lock.store(false, Ordering::Release);
                        line.occ.store(0, Ordering::Release);
                    }
                }
                9 => {
                    for _ in 0..iters {
                        black_box(line.lock.load(Ordering::Acquire));
                        line.lock.store(true, Ordering::Release);
                        black_box(line.occ.load(Ordering::Acquire));
                        line.occ.store(1, Ordering::Release);
                        line.lock.store(false, Ordering::Release);
                        line.occ.store(0, Ordering::Release);
                    }
                }
                10 => {
                    for _ in 0..iters {
                        black_box(line.lock.swap(true, Ordering::Acquire));
                        black_box(line.occ.load(Ordering::Acquire));
                        black_box(line.occ.fetch_or(1, Ordering::Relaxed));
                        line.lock.store(false, Ordering::Release);
                        black_box(line.occ.fetch_and(!1u8, Ordering::Release));
                    }
                }
                11 => {
                    for _ in 0..iters {
                        black_box(slots.lock.swap(true, Ordering::Acquire));
                        let mut occ = 0u8;
                        for s in &slots.live {
                            occ |= s.load(Ordering::Acquire);
                        }
                        black_box(occ);
                        slots.live[0].store(1, Ordering::Relaxed);
                        slots.lock.store(false, Ordering::Release);
                        slots.live[0].store(0, Ordering::Release);
                    }
                }
                12 => {
                    // Lock-held watermark: nothing above `hi` can be live, so
                    // the scan reads one flag in the measured steady state
                    // (occ_max == 1 on every hot plane at t=1).
                    for _ in 0..iters {
                        black_box(slots.lock.swap(true, Ordering::Acquire));
                        black_box(slots.live[0].load(Ordering::Acquire));
                        slots.live[0].store(1, Ordering::Relaxed);
                        slots.lock.store(false, Ordering::Release);
                        slots.live[0].store(0, Ordering::Release);
                    }
                }
                13 => {
                    // Per-thread registry: no lock, no RMW. Publish own record,
                    // full fence, scan peers (one peer slot here).
                    for _ in 0..iters {
                        slots.live[0].store(1, Ordering::Release);
                        core::sync::atomic::fence(Ordering::SeqCst);
                        black_box(slots.live[1].load(Ordering::Relaxed));
                        slots.live[0].store(0, Ordering::Release);
                    }
                }
                14 => {
                    for _ in 0..iters {
                        core::sync::atomic::fence(Ordering::SeqCst);
                        black_box(0u8);
                    }
                }
                _ => {
                    for i in 0..iters {
                        black_box(i);
                    }
                }
            }
            let ns = t0.elapsed().as_secs_f64() * 1e9 / iters as f64;
            acc[a].push(ns);
        }
    }

    println!("ARM\tn\tns_per_iter_median\tmin\tmax");
    let mut empty = acc[names.len() - 1].clone();
    let e = med(&mut empty);
    for (i, name) in names.iter().enumerate() {
        let mut v = acc[i].clone();
        let m = med(&mut v);
        println!(
            "{name}\t{}\t{:.3}\t{:.3}\t{:.3}",
            v.len(),
            m,
            v.first().copied().unwrap_or(0.0),
            v.last().copied().unwrap_or(0.0)
        );
    }
    println!("# loop overhead subtracted below (empty = {e:.3} ns)");
    println!("ARM\tns_net");
    for (i, name) in names.iter().enumerate() {
        let mut v = acc[i].clone();
        println!("{name}\t{:.3}", med(&mut v) - e);
    }
}
