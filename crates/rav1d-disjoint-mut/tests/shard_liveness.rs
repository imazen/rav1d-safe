//! Liveness gate for the sharded borrow tracker.
//!
//! A tracker that never panics is fast and useless. These tests deliberately
//! create overlaps and require them to be caught — and, just as importantly,
//! require disjoint borrows *not* to be reported, because an address-sharded
//! tracker can fail in that direction too if a record is stored clipped rather
//! than whole.
//!
//! The exhaustive test compares the tracker's verdict against the plain
//! interval predicate `a.start < b.end && b.start < a.end` over tens of
//! thousands of pairs chosen to straddle every plausible block boundary
//! (256 B, 1 KiB, 4 KiB) and to span from one block to dozens — i.e. across
//! the fast path, the multi-shard path, and the wide path. Any disagreement in
//! either direction fails.

use rav1d_disjoint_mut::DisjointMut;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const LEN: usize = 24 * 1024;

/// Offsets chosen to sit on, just below, and just above every block size the
/// tracker might be compiled with, plus a few arbitrary ones.
fn starts() -> Vec<usize> {
    let mut v = vec![0usize, 1, 3, 7, 63, 64, 65];
    for base in [256usize, 512, 1024, 2048, 4096, 8192, 16384] {
        for d in [-1isize, 0, 1] {
            let s = base as isize + d;
            if s >= 0 && (s as usize) < LEN {
                v.push(s as usize);
            }
        }
    }
    v.extend([333usize, 1777, 5000, 9999, 20000]);
    v.sort_unstable();
    v.dedup();
    v
}

/// Lengths spanning one block, a couple of blocks, more shards than
/// `MAX_SHARDS_PER_BORROW`, and more blocks than `MAX_BLOCKS_SCAN` (the wide
/// path).
const LENS: [usize; 9] = [1, 2, 8, 255, 256, 257, 1024, 4096, 20000];

fn clipped(start: usize, len: usize) -> Option<(usize, usize)> {
    let end = start.checked_add(len)?;
    if end > LEN || start >= end {
        None
    } else {
        Some((start, end))
    }
}

/// Does the tracker report a conflict when `b` is registered while `a` is live?
///
/// `a_mut` / `b_mut` pick which of the four mutability combinations to test.
fn conflicts(
    dm: &DisjointMut<Vec<u8>>,
    a: (usize, usize),
    b: (usize, usize),
    a_mut: bool,
    b_mut: bool,
) -> bool {
    // Hold `a`, then try `b`. Both guards must be dropped before returning so
    // the next pair starts clean; the `catch_unwind` is only around `b`.
    let hit = if a_mut {
        let _ga = dm.index_mut(a.0..a.1);
        panic::catch_unwind(AssertUnwindSafe(|| {
            if b_mut {
                drop(dm.index_mut(b.0..b.1));
            } else {
                drop(dm.index(b.0..b.1));
            }
        }))
        .is_err()
    } else {
        let _ga = dm.index(a.0..a.1);
        panic::catch_unwind(AssertUnwindSafe(|| {
            if b_mut {
                drop(dm.index_mut(b.0..b.1));
            } else {
                drop(dm.index(b.0..b.1));
            }
        }))
        .is_err()
    };
    hit
}

/// The gate: for every pair, the tracker's verdict must equal the interval
/// predicate (restricted to the mut/immut rule — two immutable borrows never
/// conflict).
#[test]
fn exhaustive_pairs_match_the_interval_predicate() {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {})); // the expected panics are the data here

    let mut checked = 0usize;
    let mut wrong: Vec<String> = Vec::new();
    let ss = starts();
    // Each pair gets a fresh tracker: a poisoned one (from a caught panic
    // while a mutable guard was live) would mask later results.
    for &a_start in &ss {
        for &a_len in &LENS {
            let Some(a) = clipped(a_start, a_len) else {
                continue;
            };
            for &b_start in &ss {
                for &b_len in &LENS {
                    let Some(b) = clipped(b_start, b_len) else {
                        continue;
                    };
                    for (a_mut, b_mut) in [(true, true), (true, false), (false, true), (false, false)]
                    {
                        let dm = DisjointMut::new(vec![0u8; LEN]);
                        let overlaps = a.0 < b.1 && b.0 < a.1;
                        let expect = overlaps && (a_mut || b_mut);
                        let got = conflicts(&dm, a, b, a_mut, b_mut);
                        checked += 1;
                        if got != expect && wrong.len() < 20 {
                            wrong.push(format!(
                                "a={:?} {} b={:?} {}: expected conflict={expect}, got {got}",
                                a,
                                if a_mut { "mut" } else { "imm" },
                                b,
                                if b_mut { "mut" } else { "imm" },
                            ));
                        }
                    }
                }
            }
        }
    }
    panic::set_hook(prev);
    assert!(
        wrong.is_empty(),
        "{} of {checked} pairs disagreed with the interval predicate:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    // Guard against the test silently degenerating to nothing.
    assert!(checked > 50_000, "only {checked} pairs exercised");
    eprintln!("checked {checked} borrow pairs");
}

/// The liveness half, under real concurrency: every thread hammers the SAME
/// small window, so overlaps are constant. Some registrations must be refused.
#[test]
fn concurrent_overlaps_are_caught() {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let dm = Arc::new(DisjointMut::new(vec![0u8; LEN]));
    let caught = Arc::new(AtomicUsize::new(0));
    let mut hs = Vec::new();
    for t in 0..8usize {
        let dm = Arc::clone(&dm);
        let caught = Arc::clone(&caught);
        hs.push(std::thread::spawn(move || {
            for i in 0..4_000usize {
                // A window every thread wants a piece of.
                let start = (i * 17 + t) % 512;
                let r = panic::catch_unwind(AssertUnwindSafe(|| {
                    let mut g = dm.index_mut(start..start + 64);
                    g[0] = t as u8;
                    // Hold it long enough to overlap someone.
                    std::hint::black_box(&mut g);
                }));
                if r.is_err() {
                    caught.fetch_add(1, Ordering::Relaxed);
                    // A panic while a mutable guard is live poisons the
                    // instance; from here on every borrow fails, which is
                    // correct behaviour, so stop this thread.
                    return;
                }
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    panic::set_hook(prev);
    assert!(
        caught.load(Ordering::Relaxed) > 0,
        "8 threads fought over one 64-byte window and the tracker never objected \
         — the overlap check is not live"
    );
}

/// The precision half, under real concurrency: strictly disjoint per-thread
/// regions, interleaved across the address space so they land in the same
/// shards constantly. Not one panic is allowed.
#[test]
fn concurrent_disjoint_is_never_refused() {
    let dm = Arc::new(DisjointMut::new(vec![0u8; LEN]));
    let mut hs = Vec::new();
    for t in 0..8usize {
        let dm = Arc::clone(&dm);
        hs.push(std::thread::spawn(move || {
            for i in 0..20_000usize {
                // Stride by 8 threads so neighbouring threads' regions are
                // adjacent but never overlapping.
                let start = ((i * 8 + t) * 13) % (LEN - 16);
                let start = start - (start % 8) + (t % 8);
                if start + 1 >= LEN {
                    continue;
                }
                let mut g = dm.index_mut(start..start + 1);
                g[0] = g[0].wrapping_add(1);
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
}

/// A borrow long enough to take the wide path must still be enforced against a
/// one-byte borrow anywhere inside it, and must release cleanly.
#[test]
fn wide_and_narrow_see_each_other() {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let dm = DisjointMut::new(vec![0u8; LEN]);
    {
        let _wide = dm.index_mut(0..LEN);
        for probe in [0usize, 1, 255, 4096, LEN - 1] {
            let hit = panic::catch_unwind(AssertUnwindSafe(|| drop(dm.index(probe..probe + 1))))
                .is_err();
            assert!(hit, "byte {probe} inside a whole-buffer mutable borrow was allowed");
        }
    }
    panic::set_hook(prev);
    // Released: the same probes now succeed.
    for probe in [0usize, 1, 255, 4096, LEN - 1] {
        drop(dm.index(probe..probe + 1));
    }
}

/// `index(..)` and `index(n..)` encode their end as `usize::MAX`; the clamp
/// must not lose the borrow.
#[test]
fn open_ended_ranges_are_still_tracked() {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let dm = DisjointMut::new(vec![0u8; LEN]);
    {
        let _g = dm.index_mut(..);
        let hit =
            panic::catch_unwind(AssertUnwindSafe(|| drop(dm.index(LEN - 1..LEN)))).is_err();
        assert!(hit, "index(..) did not reserve the last byte");
    }
    {
        let _g = dm.index_mut(1024..);
        let hit = panic::catch_unwind(AssertUnwindSafe(|| drop(dm.index(2048..2049)))).is_err();
        assert!(hit, "index(1024..) did not reserve byte 2048");
        // ...and a byte below the start is still free.
        drop(dm.index(0..1));
    }
    panic::set_hook(prev);
}
