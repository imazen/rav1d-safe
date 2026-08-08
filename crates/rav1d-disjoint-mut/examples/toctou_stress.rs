//! Stress harness for the wide/narrow lock-ordering TOCTOU fixed in 4af62ae.
//!
//! # Why this is not a `#[test]`
//!
//! The in-tree soundness suite does **not** detect that bug. Verified by
//! mutation on 2026-08-07: delete the in-lock `state` re-read from
//! `BorrowTracker::add`'s single-block fast path and
//! `cargo test --release -p rav1d-disjoint-mut` still passes everything,
//! including `concurrent_overlaps_are_caught`, three runs out of three.
//!
//! The reason is structural, not an oversight in that test. It has one thread
//! hold a wide borrow for the whole attempt window, so every contender's
//! *pre-lock* `state` load already reads non-zero and takes the slow path — the
//! in-lock re-read is never the thing that saves it. The bug lives in a window
//! a few nanoseconds wide: a narrow registrant loads `state == 0`, a wide
//! registrant then publishes into `self.wide` and bumps `state`, and only then
//! does the narrow take its shard lock and scan a shard that legitimately holds
//! no record of the wide borrow. Hitting it needs *churn* — wide borrows being
//! taken and dropped continuously against narrow borrows on the same byte — and
//! the original report saw 115 / 18 / 22 violations across three runs of
//! ~1.4e9 acquisitions. That is minutes of wall clock, which is why it is a
//! committed dev tool with an explicit iteration budget rather than something
//! that runs on every `cargo test`.
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
//! two sides met. Closing the real gap needs a test-only hook that lets a
//! granted narrow borrow ask the tracker whether an overlapping record exists;
//! that hook does not exist yet and is named as remaining work.
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
