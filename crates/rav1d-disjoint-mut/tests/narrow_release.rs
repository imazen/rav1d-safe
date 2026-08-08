//! Race gate for the LOCK-FREE release path, narrow against narrow.
//!
//! # What this exists to catch
//!
//! Retiring a borrow used to run under the shard lock; it is now one
//! `fetch_and(!bit)` on the shard's atomic occupancy byte. That makes the
//! release a read-modify-write on state that live records depend on, and it
//! opens a failure direction the rest of the suite does not cover:
//!
//! * **A release that clears too LITTLE** (a lost update — `store(load | bit)`
//!   in `publish` racing a `fetch_and`) leaks the slot forever. That is gated
//!   by `tracker_shard::tests::threaded_churn_leaks_no_slots`.
//! * **A release that clears too MUCH** — anything that writes the occupancy
//!   byte rather than clearing one bit of it — silently retires records that
//!   are still live. Every overlap against those records then goes undetected,
//!   and their slots can be handed out and overwritten underneath them.
//!
//! The second direction had no gate. Verified by mutation on 2026-08-08:
//! replacing `Shard::retire`'s `fetch_and(!bit)` with `store(0)` leaves the
//! ENTIRE `cargo test -p rav1d-disjoint-mut` suite green — 23 lib + 25
//! `soundness.rs` + 5 `shard_liveness.rs` + `wide_exclusion.rs`. It passes
//! because nothing else drives many live NARROW records through one shard while
//! also racing an overlapping pair against them: `wide_exclusion.rs` scatters
//! its narrow borrows across the whole shard prefix, and its violation is
//! wide-versus-narrow, which does not use shard slots at all.
//!
//! # How
//!
//! The buffer is deliberately SMALLER than the tracker's `SHARD_MIN_LEN`
//! (64 KiB), so the instance gets exactly ONE shard and every record competes
//! for the same seven slots. Each thread alternates:
//!
//! * a PRIVATE range nobody else touches — pure allocate/release churn, which
//!   is what supplies the concurrent `retire` calls; and
//! * a SHARED range every thread contends for — at most one thread may hold it
//!   mutably at a time, and the tracker is what has to enforce that.
//!
//! A thread granted the shared range publishes a generation, spins, then clears
//! it. Another thread that is ALSO granted the shared range reads the
//! generation before and after its own acquisition and reports a violation only
//! when both reads are the same non-zero value — so a holder that released and
//! re-took between the two observations cannot be mistaken for one that stayed
//! live. That direction of error under-reports and never invents.

use rav1d_disjoint_mut::DisjointMut;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Below `SHARD_MIN_LEN` (64 KiB) so `mask_for` gives the instance one shard
/// and all seven slots are contended. This is the whole point of the test: the
/// over-clear hazard is invisible when records are spread thin.
const LEN: usize = 32 * 1024;

const THREADS: usize = 7;
const ROUNDS: usize = 200_000;

/// The contended interval. Every thread tries to take exactly this range
/// mutably, so any two simultaneous grants are a provable overlap.
const SHARED: core::ops::Range<usize> = 4096..4104;

#[test]
fn a_lock_free_release_never_retires_a_live_neighbour() {
    // Overlap refusals are the DESIRED outcome on one side of each race, so the
    // default hook's backtrace spam is suppressed for the duration.
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let dm = Arc::new(DisjointMut::new(vec![0u8; LEN]));
    // 0 = nobody holds SHARED; non-zero = that generation does.
    let holder = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let witnesses = Arc::new(AtomicUsize::new(0));
    let shared_ok = Arc::new(AtomicUsize::new(0));
    let private_ok = Arc::new(AtomicUsize::new(0));
    let refused = Arc::new(AtomicUsize::new(0));

    let mut hs = Vec::new();
    for t in 0..THREADS {
        let dm = Arc::clone(&dm);
        let holder = Arc::clone(&holder);
        let stop = Arc::clone(&stop);
        let witnesses = Arc::clone(&witnesses);
        let shared_ok = Arc::clone(&shared_ok);
        let private_ok = Arc::clone(&private_ok);
        let refused = Arc::clone(&refused);
        hs.push(std::thread::spawn(move || {
            // Disjoint from SHARED and from every other thread's slice, so a
            // conflict here would be a bug in the test, not the tracker.
            let private = (t * 64)..(t * 64 + 8);
            // Generations are per-thread and never zero, so `holder` can always
            // be attributed and 0 unambiguously means "free".
            let mut generation = t + 1;
            let mut round = 0usize;
            while !stop.load(Ordering::Relaxed) {
                round += 1;
                // Three private acquire/release pairs per shared attempt: the
                // over-clear hazard needs a NEIGHBOUR release to fire while the
                // shared record is live, so churn has to outnumber contention.
                for _ in 0..3 {
                    if panic::catch_unwind(AssertUnwindSafe(|| {
                        let mut g = dm.index_mut(private.clone());
                        g[0] = g[0].wrapping_add(1);
                        drop(g);
                    }))
                    .is_ok()
                    {
                        private_ok.fetch_add(1, Ordering::Relaxed);
                    } else {
                        refused.fetch_add(1, Ordering::Relaxed);
                    }
                }

                generation += THREADS;
                let mine = generation;
                let granted = panic::catch_unwind(AssertUnwindSafe(|| {
                    let g = dm.index_mut(SHARED);
                    // Read BEFORE claiming: a non-zero value here is another
                    // thread's live claim over the same bytes.
                    let before = holder.load(Ordering::Acquire);
                    holder.store(mine, Ordering::Release);
                    std::hint::spin_loop();
                    let after = holder.load(Ordering::Acquire);
                    // `after != gen` means someone else claimed it while we
                    // held it; `before == after != 0` means someone else's
                    // claim spanned our whole grant. Either is a missed
                    // overlap; the second cannot be a re-take.
                    if after != mine || (before != 0 && before == after) {
                        witnesses.fetch_add(1, Ordering::Relaxed);
                    }
                    holder.store(0, Ordering::Release);
                    drop(g);
                }))
                .is_ok();
                if granted {
                    shared_ok.fetch_add(1, Ordering::Relaxed);
                } else {
                    refused.fetch_add(1, Ordering::Relaxed);
                }
                if round >= ROUNDS {
                    break;
                }
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    panic::set_hook(prev);

    // Liveness. A tracker that refused everything, or one whose threads never
    // met, would report zero witnesses for the wrong reason.
    assert!(
        private_ok.load(Ordering::Relaxed) > 10_000,
        "private churn barely ran ({}); the release path was not exercised",
        private_ok.load(Ordering::Relaxed)
    );
    assert!(
        shared_ok.load(Ordering::Relaxed) > 1_000,
        "the shared range was granted almost never ({}); the race window was \
         never exercised",
        shared_ok.load(Ordering::Relaxed)
    );
    assert!(
        refused.load(Ordering::Relaxed) > 0,
        "not one acquisition was ever refused; the {THREADS} threads never \
         actually contended for the shared range"
    );

    assert_eq!(
        witnesses.load(Ordering::Relaxed),
        0,
        "two mutable borrows of the same bytes were live at once — a release \
         retired a record that was still held (private_ok={}, shared_ok={}, \
         refused={})",
        private_ok.load(Ordering::Relaxed),
        shared_ok.load(Ordering::Relaxed),
        refused.load(Ordering::Relaxed)
    );
}
