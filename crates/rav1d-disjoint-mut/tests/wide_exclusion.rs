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
/// Native rounds. Under Miri this is 300, for the reason spelled out on
/// `shard_liveness::starts()`: every round here takes a WHOLE-BUFFER borrow of
/// an 8 MiB instance, which promotes to the wide path and locks all 128 shards,
/// while six narrow threads contend — and Miri emulates every one of those
/// atomics. Measured 2026-08-09 on an M4 Pro: **300 rounds takes 181.7 s**, and
/// both liveness floors are met at that count; 60_000 rounds was still running
/// after 10 minutes when it was killed, on both memory models. The native run
/// is unchanged at 60_000 (0.05 s), and the `wide-path-gate` CI job — the one
/// that proves this file is non-vacuous, via `--features __probe_wide` — is
/// native, so the anti-vacuity evidence does not depend on the Miri count.
const ROUNDS: usize = if cfg!(miri) { 300 } else { 60_000 };

/// Narrow-side liveness floor, scaled with `ROUNDS`. The narrow threads spin
/// until the wide loop finishes, so their count tracks the round count.
const MIN_NARROW_OK: usize = if cfg!(miri) { 100 } else { 1000 };

#[test]
fn a_wide_borrow_excludes_every_narrow_shard() {
    // DECLARE PARALLELISM FIRST, or this whole file is vacuous.
    //
    // `mask_for` hands an instance `active_shards() - 1`, and `active_shards()`
    // is `SHARDS_SERIAL` until some caller declares otherwise. Since issue #458
    // set `SHARDS_SERIAL = 1` that default mask is ZERO, and a mask-0 instance
    // has exactly one shard — so `index_mut(0..LEN)` below takes `add`'s
    // one-lock fast path and the wide path is never entered. Measured on
    // 2aa00c5 before this line existed: 100 whole-buffer borrows of an 8 MiB
    // instance produced 0 promotions on every `wide_probe` counter, while the
    // test still reported `ok` — the two liveness assertions at the bottom
    // count GRANTED borrows, which a single-shard instance grants just fine.
    //
    // A real decode reaches the wide path through the concurrent shard set, so
    // that is the configuration this gate has to run in. `set_parallelism` is a
    // monotone process-global and this file holds exactly one test, so there is
    // no ordering hazard here.
    rav1d_disjoint_mut::set_parallelism(NARROW_THREADS + 1);
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
        narrow_ok.load(Ordering::Relaxed) > MIN_NARROW_OK,
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
    // THE GATE'S OWN LIVENESS. The two counts above prove borrows were granted,
    // which is necessary and NOT sufficient: a single-shard instance grants
    // them happily while never entering the code this file is named after. Only
    // the promotion counters can tell the difference, and they exist only under
    // `__probe_wide` — so run this gate with
    // `--features __probe_wide` whenever you need it to prove itself, and treat
    // a plain run as the cheap regression check rather than as evidence.
    //
    // The predicate below MIRRORS `lib.rs`'s re-export of `wide_probe`, which
    // also requires the sharded tracker: the `__probe_*` / `__tracker_legacy`
    // features select the LEGACY tracker, which has no wide path and no
    // counters. Spelling only `__probe_wide` here is why
    // `cargo test --all-features` (which turns on the mutually exclusive
    // tracker selectors at once) failed to COMPILE this test binary -- red on
    // `main` @ ee07b00 too, and it took the whole job's other test binaries
    // down with it.
    #[cfg(all(
        feature = "__probe_wide",
        not(any(
            feature = "__probe_count",
            feature = "__probe_noscan",
            feature = "__probe_lockonly",
            feature = "__tracker_legacy"
        ))
    ))]
    {
        use rav1d_disjoint_mut::wide_probe;
        let promotions = wide_probe::WIDE_SHARDS.load(Ordering::Relaxed)
            + wide_probe::WIDE_BLOCKS.load(Ordering::Relaxed)
            + wide_probe::WIDE_FULL.load(Ordering::Relaxed);
        assert!(
            promotions > 0,
            "VACUOUS GATE: {ROUNDS} whole-buffer borrows produced zero wide \
             promotions, so nothing here exercised the wide path. This is what \
             `SHARDS_SERIAL = 1` (issue #458) did to this file before the \
             `set_parallelism` call at the top existed."
        );
    }
}
