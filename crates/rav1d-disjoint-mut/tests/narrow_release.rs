//! Race gate for the LOCK-FREE release path, narrow against narrow.
//!
//! # What this exists to catch
//!
//! Retiring a borrow used to run under the shard lock; it is now one plain
//! `Release` store to this slot's OWN `Shard::live` byte. Release therefore
//! writes state that live records depend on without holding anything, and that
//! opens a failure direction the rest of the suite does not cover:
//!
//! * **A release that clears too LITTLE** (a lost update) leaks the slot
//!   forever. That is gated by
//!   `tracker_shard::tests::threaded_churn_leaks_no_slots`.
//! * **A release that clears too MUCH** — anything that retires more than the
//!   one slot it owns — silently retires records that are still live. Every
//!   overlap against those records then goes undetected, and their slots can be
//!   handed out and overwritten underneath them.
//!
//! The second direction had no gate. Verified by mutation on 2026-08-08,
//! against the then-current SHARED-bitmap representation: replacing
//! `Shard::retire`'s `fetch_and(!bit)` with a whole-byte `store(0)` left the
//! ENTIRE `cargo test -p rav1d-disjoint-mut` suite green — 23 lib + 25
//! `soundness.rs` + 5 `shard_liveness.rs` + `wide_exclusion.rs`. It passed
//! because nothing else drives many live NARROW records through one shard while
//! also racing an overlapping pair against them: `wide_exclusion.rs` scatters
//! its narrow borrows across the whole shard prefix, and its violation is
//! wide-versus-narrow, which does not use shard slots at all.
//!
//! THE REPRESENTATION HAS SINCE CHANGED and the mutation above no longer
//! describes a defect: liveness is one atomic byte PER SLOT (`Shard::live`),
//! so a correct `retire` IS `store(0)` — of that slot's own byte. This file
//! still gates the same direction on the new shape; verified 2026-08-08 by
//! widening `Shard::live_mask`'s `allocated <= 1` fast path to `<= 2`, which
//! makes the mask a SUBSET of the live set and drops live neighbours: this
//! test, and only this test, FAILS. RE-VERIFIED in full on 2026-08-09, both
//! halves: with `<= 2` planted, this test fails (`witnesses` = 1, in 2.52 s, at
//! private_ok 4_200_000 / shared_ok 1_200_246 / refused 199_754) and
//! `--no-fail-fast` over the rest of the suite — 26 lib + 25 `soundness` + 5
//! `shard_liveness` + `wide_exclusion` + 2 `guard_move_release` — is GREEN.
//! The paragraph above is a live record, not a stale one.
//!
//! # It is also the crate's aliasing gate, and that is why it runs under Miri
//!
//! The tracker cannot tell you whether the REFERENCES it hands out obey Rust's
//! aliasing model — only Miri can, and this is the only test in the suite that
//! drives a release and an overlapping acquisition of the same bytes hard
//! enough for Miri to schedule one inside the other. It is what found the
//! guards' by-value-move defect (2026-08-09, `tests/guard_move_release.rs`
//! now gates that specific shape). Do not drop it from the Miri leg.
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

/// Native rounds. Under Miri, four orders of magnitude fewer — see below.
const ROUNDS: usize = if cfg!(miri) { 400 } else { 200_000 };

/// Liveness floors.
///
/// The native pair (10_000 / 1_000) is unchanged. The Miri pair is NOT a
/// relaxation of it: at 400 rounds the arms attempt 8_400 private and 2_800
/// shared acquisitions, so 4_000 / 700 are floors at ~48% and ~25% of
/// attempts, where the native floors sit at ~0.24% and ~0.07%. The Miri run is
/// held to a proportionally TIGHTER standard, not a looser one. Measured
/// 2026-08-09, M4 Pro, `--test narrow_release` at 400 Miri rounds:
///
/// ```text
///   Stacked Borrows   private_ok 8400   shared_ok 1846   refused 954
///   Tree Borrows      private_ok 8400   shared_ok 1862   refused 938
/// ```
///
/// Why the round count moves at all: Miri interprets. Two-point fit of this
/// test's own wall time under Stacked Borrows, `total = a + b * rounds`, from
/// 23.12 s at 100 rounds and 92.19 s at 400 (same host, same day):
/// b = 0.230 s/round, a = 0.10 s. Extended to 200_000 that is **~12.8 hours per
/// memory model** — the Miri leg could never have finished it, and never did:
/// run 31292996318 reached this file only because it ABORTED on UB.
///
/// What the Miri leg is for is the ALIASING MODEL, and that does not need a
/// long race. **What the shorter run costs, stated honestly:** re-run against
/// the pre-fix guards on 2026-08-09, this file at 400 rounds still aborts under
/// Stacked Borrows on the default seed (same tag, `<129070>`, as CI run
/// 31292996318) — but under Tree Borrows it did NOT reproduce at either of the
/// two seeds tried (0 and 1). Tree-Borrows coverage of that defect rests on
/// `tests/guard_move_release.rs`, which contends the same range harder and
/// aborts under BOTH models on the default seed. The native run here keeps the
/// full 200_000, which is what gates the tracker's own race behaviour.
const MIN_PRIVATE_OK: usize = if cfg!(miri) { 4_000 } else { 10_000 };
const MIN_SHARED_OK: usize = if cfg!(miri) { 700 } else { 1_000 };

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
        private_ok.load(Ordering::Relaxed) > MIN_PRIVATE_OK,
        "private churn barely ran ({}); the release path was not exercised",
        private_ok.load(Ordering::Relaxed)
    );
    assert!(
        shared_ok.load(Ordering::Relaxed) > MIN_SHARED_OK,
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
