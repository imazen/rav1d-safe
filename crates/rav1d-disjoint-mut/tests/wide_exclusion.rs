//! Race gate for the wide path's mutual exclusion against narrow registrants.
//!
//! # What this exists to catch
//!
//! The wide path (a borrow spanning more shards than `MAX_SHARDS_PER_BORROW`,
//! or one that overflows a shard) publishes its record into a side list rather
//! than into any shard, and narrow registrants only consult that list when the
//! tracker's `state` word says a wide record is live. The mutual exclusion that
//! makes that safe is the wide registrant HOLDING every shard lock a narrow
//! registrant of the same instance could take, across the publish.
//!
//! Both halves of that sentence are load-bearing and neither is reachable from
//! a single-threaded test:
//!
//! * A narrow registrant that reads `state == 0`, then takes its shard lock,
//!   then registers, must not be able to interleave with a wide publish. (This
//!   is the TOCTOU that `add`'s re-read of `state` INSIDE the lock closes.)
//! * "Every shard lock a narrow registrant could take" is a *prefix* of the
//!   shard array — the instance's `0..=mask` — not the whole array. If that
//!   prefix is ever computed too short, the exclusion has a hole exactly the
//!   width of the shards it left out.
//!
//! A missed overlap is silent: both borrows succeed and nothing panics. So this
//! test does not look for a panic, it looks for two provably-overlapping
//! mutable borrows being live at the same instant.
//!
//! # How
//!
//! One thread repeatedly takes a WIDE mutable borrow of the whole buffer and,
//! while holding it, raises a flag. Several other threads repeatedly take
//! NARROW mutable borrows at addresses scattered across the buffer — every one
//! of which is inside the wide borrow — and, while holding, check the flag. A
//! correct tracker makes one side of every such race panic; a hole lets both
//! through, and the narrow side sees the flag raised.
//!
//! Registration failures are *expected* and are counted, not propagated: the
//! whole point is that one side loses. What must be zero is `witnesses`.
//!
//! # Liveness of this gate
//!
//! Deliberately shortening the wide path's prefix to one shard makes this fail;
//! see `benchmarks/p3_inversion_2026-08-07.meta`. The single-threaded
//! `wide_and_narrow_still_see_each_other_*` unit tests do NOT fail under that
//! break — they exercise the `state`/side-list half only — which is why this
//! file exists separately.

use rav1d_disjoint_mut::DisjointMut;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Big enough that `mask_for` gives the instance its full shard set (the
/// tracker's `SHARD_MIN_LEN` is 64 KiB) and that a whole-buffer borrow is well
/// past `MAX_BLOCKS_SCAN` blocks, i.e. genuinely wide.
const LEN: usize = 8 * 1024 * 1024;

const NARROW_THREADS: usize = 6;
const ROUNDS: usize = 60_000;

#[test]
fn a_wide_borrow_excludes_every_narrow_shard() {
    // The tracker reports overlaps by panicking, and here that is the DESIRED
    // outcome on one side of each race, so the default hook's backtrace spam is
    // suppressed for the duration.
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let dm = Arc::new(DisjointMut::new(vec![0u8; LEN]));
    let wide_held = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    // A narrow borrow was granted while the wide holder provably still held
    // its overlapping mutable borrow. Any nonzero value is a missed overlap.
    let witnesses = Arc::new(AtomicUsize::new(0));
    let narrow_ok = Arc::new(AtomicUsize::new(0));
    let wide_ok = Arc::new(AtomicUsize::new(0));

    let mut hs = Vec::new();
    for t in 0..NARROW_THREADS {
        let dm = Arc::clone(&dm);
        let wide_held = Arc::clone(&wide_held);
        let stop = Arc::clone(&stop);
        let witnesses = Arc::clone(&witnesses);
        let narrow_ok = Arc::clone(&narrow_ok);
        hs.push(std::thread::spawn(move || {
            // Addresses far enough apart to hash to different shards at every
            // supported block size, so the race is spread over the whole
            // shard prefix rather than repeatedly hitting one line.
            let mut off = t * 977 * 4096 + 61;
            while !stop.load(Ordering::Relaxed) {
                off = (off + 4099) % (LEN - 64);
                let hit = panic::catch_unwind(AssertUnwindSafe(|| {
                    let g = dm.index_mut(off..off + 8);
                    if wide_held.load(Ordering::Acquire) {
                        witnesses.fetch_add(1, Ordering::Relaxed);
                    }
                    drop(g);
                }))
                .is_err();
                if !hit {
                    narrow_ok.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for _ in 0..ROUNDS {
        let hit = panic::catch_unwind(AssertUnwindSafe(|| {
            let g = dm.index_mut(0..LEN);
            wide_held.store(true, Ordering::Release);
            std::hint::spin_loop();
            wide_held.store(false, Ordering::Release);
            drop(g);
        }))
        .is_err();
        if !hit {
            wide_ok.fetch_add(1, Ordering::Relaxed);
        }
    }
    stop.store(true, Ordering::Relaxed);
    for h in hs {
        h.join().unwrap();
    }
    panic::set_hook(prev);

    // Both sides have to have actually run, or a zero witness count proves
    // nothing. (`narrow_ok`/`wide_ok` count only the GRANTED borrows, so a
    // tracker that refused everything would fail here rather than pass.)
    assert!(
        wide_ok.load(Ordering::Relaxed) > ROUNDS / 100,
        "the wide registrant was refused almost every round ({}); the race \
         window was never exercised",
        wide_ok.load(Ordering::Relaxed)
    );
    assert!(
        narrow_ok.load(Ordering::Relaxed) > 1000,
        "the narrow registrants were refused almost every round ({}); the race \
         window was never exercised",
        narrow_ok.load(Ordering::Relaxed)
    );
    assert_eq!(
        witnesses.load(Ordering::Relaxed),
        0,
        "a narrow mutable borrow was granted inside a wide mutable borrow that \
         was provably still held — the wide path's shard exclusion has a hole"
    );
}
