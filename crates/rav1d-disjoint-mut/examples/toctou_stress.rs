//! Stress harness for the wide/narrow lock-ordering TOCTOU fixed in 4af62ae.
//!
//! # CORRECTION (2026-08-08, independent verifier): the bug IS gated
//!
//! An earlier revision of this header claimed `cargo test --release -p
//! rav1d-disjoint-mut` passes everything with the TOCTOU planted, three runs
//! out of three, and that no in-tree gate exists. **That is wrong at the
//! suite level and the harness below was built on it.** Re-measured by
//! mutation — delete the in-lock `state` re-read from
//! `BorrowTracker::add`'s single-block fast path, exactly the named hazard:
//!
//!   tests/soundness.rs        25 passed  (incl. `concurrent_overlaps_are_caught`)
//!   tests/shard_liveness.rs    5 passed
//!   lib unit tests            23 passed
//!   tests/wide_exclusion.rs   ***FAILED*** in 0.03 s, every run
//!
//! `a_wide_borrow_excludes_every_narrow_shard` fails with
//! "a narrow mutable borrow was granted inside a wide mutable borrow that was
//! provably still held" — **5 of 5 runs** on the composed tree and **3 of 3**
//! on the unmodified base (`fix/aarch64-mc-itx` @ 2c7c082), so it is neither
//! flaky nor an artifact of any later tracker change. Restoring the re-read
//! returns it to green.
//!
//! What survives of the original claim is only its narrow half: the
//! *`soundness.rs`* tests do not catch it, for the structural reason below.
//! `wide_exclusion.rs` does, because unlike them it takes and DROPS the wide
//! borrow 60,000 times, so contenders repeatedly enter the window where
//! `state` reads 0 pre-lock.
//!
//! The structural reason the `soundness.rs` tests miss it: they hold a wide
//! borrow for the whole attempt window, so every contender's *pre-lock*
//! `state` load already reads non-zero and takes the slow path — the in-lock
//! re-read is never the thing that saves it. The bug lives in a window a few
//! nanoseconds wide: a narrow registrant loads `state == 0`, a wide registrant
//! then publishes into `self.wide` and bumps `state`, and only then does the
//! narrow take its shard lock and scan a shard that legitimately holds no
//! record of the wide borrow. Hitting it needs *churn* — which is what
//! `wide_exclusion.rs` supplies and what this harness was written to supply.
//!
//! # Why this is not a `#[test]`
//!
//! Because `wide_exclusion.rs` already is one and is faster at it. This file
//! stays as a dev tool for the steady-state directions and for its explicit
//! iteration budget, not as the gate — see the honest result below.
//!
//! # HONEST RESULT (2026-08-07): this harness does NOT detect that bug either
//!
//! Measured on an Apple M4 Pro, `--release`, 8 s, 7 narrow threads:
//!
//!   fixed tracker    granted 77,568,436  refused 1,045,068  violations 0
//!   TOCTOU planted   granted 94,270,522  refused 1,048,518  violations 0
//!
//! The two sides genuinely race — a million refusals per run says so — and the
//! run is within a factor of 15 of the acquisition count that produced 115
//! violations in the original report. It still reports nothing, and the reason
//! is a blind spot in the *detector*, not in the stress:
//!
//! A narrow borrow that slips through does so **while the wide borrow is being
//! registered** — that is the entire window. `live` cannot have been published
//! yet, because the wide thread only publishes it after `index_mut` returns.
//! So every violation this bug produces lands exactly where the flag says "no
//! wide borrow", and the conservative rule below discards it. Any purely
//! external, flag-based observer has the same hole: the fact being observed is
//! "the wide record is in `self.wide`", and that is only visible from inside
//! the tracker.
//!
//! **So: `violations 0` from this harness is NOT evidence the tracker is
//! sound.** What it is good for is the steady-state directions — wide held vs
//! narrow arriving, and narrow vs narrow — where the refusal count proves the
//! two sides met.
//!
//! The "closing the real gap needs a test-only in-tracker hook" conclusion
//! that used to end this section was drawn from the false premise corrected
//! at the top: `wide_exclusion.rs` closes it already, from *inside* the
//! tracker's own panic path rather than from an external flag. It gets to
//! observe the miss because the tracker itself is what refuses — the narrow
//! side's `catch_unwind` distinguishes "refused" from "granted", and the flag
//! is only used to decide whether a GRANT was concurrent, which for a
//! 60,000-round drop/retake loop it reliably is. An in-tracker query hook
//! would still be a stronger detector; it is an improvement, not a
//! prerequisite.
//!
//! # Detection rule (conservative in the direction that hides the bug)
//!
//! A narrow borrow succeeding at the same moment a wide borrow is live is the
//! violation. "At the same moment" is decided by a generation counter, not a
//! flag, so a wide guard that dropped and restarted between the two
//! observations cannot be mistaken for one that stayed live:
//!
//!   * the wide thread bumps `gen`, acquires, publishes `live = gen`, and
//!     stores `live = 0` before dropping;
//!   * a narrow thread reads `live` before and after its own successful
//!     acquisition and reports a violation only when both reads are the same
//!     NON-ZERO generation.
//!
//! That direction of error is conservative: `live` is published *after* the
//! wide guard is registered, so a genuine miss inside that publish gap goes
//! uncounted. The harness under-reports and never invents.
//!
//! # Usage
//!
//!   cargo run --release -p rav1d-disjoint-mut --example toctou_stress -- [seconds] [threads]
//!
//! Exit status 1 and a non-zero violation count means the tracker missed an
//! overlap. Exit 0 with `violations 0` is evidence of absence only in
//! proportion to the acquisition count it prints — read that number.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rav1d_disjoint_mut::DisjointMut;

/// Big enough to span many blocks at every supported `BLOCK_SHIFT`.
const LEN: usize = 1 << 20;
/// The byte both sides fight over. Odd offset so it is not block-aligned.
const PIVOT: usize = (LEN / 2) + 37;

fn main() {
    let secs: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let narrow_threads: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    let dm = Arc::new(DisjointMut::new(vec![0u8; LEN]));
    let live = Arc::new(AtomicU64::new(0));
    let generation = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let violations = Arc::new(AtomicU64::new(0));
    let acquisitions = Arc::new(AtomicU64::new(0));
    let refusals = Arc::new(AtomicU64::new(0));

    // Overlap panics are the expected outcome on most iterations; silence the
    // hook so the run is readable.
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let mut handles = Vec::new();

    // The wide side: a borrow spanning every block, taken and dropped as fast
    // as possible so the publish-then-bump window comes round constantly.
    {
        let dm = Arc::clone(&dm);
        let live = Arc::clone(&live);
        let generation = Arc::clone(&generation);
        let stop = Arc::clone(&stop);
        handles.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let g = generation.fetch_add(1, Ordering::Relaxed) + 1;
                if let Ok(guard) = panic::catch_unwind(AssertUnwindSafe(|| dm.index_mut(0..LEN))) {
                    live.store(g, Ordering::SeqCst);
                    std::hint::spin_loop();
                    live.store(0, Ordering::SeqCst);
                    drop(guard);
                }
            }
        }));
    }

    // The narrow side: single-block borrows on the contested byte.
    for _ in 0..narrow_threads {
        let dm = Arc::clone(&dm);
        let live = Arc::clone(&live);
        let stop = Arc::clone(&stop);
        let violations = Arc::clone(&violations);
        let acquisitions = Arc::clone(&acquisitions);
        let refusals = Arc::clone(&refusals);
        handles.push(std::thread::spawn(move || {
            let mut local_acq = 0u64;
            let mut local_ref = 0u64;
            let mut local_bad = 0u64;
            while !stop.load(Ordering::Relaxed) {
                for _ in 0..1024 {
                    let before = live.load(Ordering::SeqCst);
                    let got = panic::catch_unwind(AssertUnwindSafe(|| {
                        let guard = dm.index_mut(PIVOT..PIVOT + 1);
                        // Observed while the narrow guard is provably alive.
                        let after = live.load(Ordering::SeqCst);
                        drop(guard);
                        after
                    }));
                    match got {
                        Ok(after) => {
                            local_acq += 1;
                            if before != 0 && before == after {
                                local_bad += 1;
                            }
                        }
                        Err(_) => local_ref += 1,
                    }
                }
            }
            acquisitions.fetch_add(local_acq, Ordering::Relaxed);
            refusals.fetch_add(local_ref, Ordering::Relaxed);
            violations.fetch_add(local_bad, Ordering::Relaxed);
        }));
    }

    let t0 = Instant::now();
    std::thread::sleep(Duration::from_secs(secs));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
    panic::set_hook(prev);

    let acq = acquisitions.load(Ordering::Relaxed);
    let refu = refusals.load(Ordering::Relaxed);
    let bad = violations.load(Ordering::Relaxed);
    println!(
        "elapsed {:.1}s  narrow_threads {narrow_threads}  granted {acq}  refused {refu}  \
         violations {bad}",
        t0.elapsed().as_secs_f64()
    );
    // A run in which the narrow side was never refused proves nothing about
    // the tracker: the two sides never met. Say so rather than exit 0.
    if refu == 0 {
        println!("INCONCLUSIVE: the narrow side was never refused, so the two sides never raced");
        std::process::exit(2);
    }
    if bad != 0 {
        println!("FAIL: {bad} narrow borrows were granted while a wide borrow was live");
        std::process::exit(1);
    }
    println!("OK — steady-state only; see the module header for why this cannot");
    println!("     rule out the 4af62ae registration-window TOCTOU");
}
