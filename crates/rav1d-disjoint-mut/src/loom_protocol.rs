//! Models the production algorithm, with both metadata and payload accesses
//! checked by Loom. Run via `scripts/review.sh loom`; never in a normal build.

use super::*;
use loom::cell::UnsafeCell;
use loom::sync::Arc;
use loom::thread;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::string::String;

fn model(f: impl Fn() + Send + Sync + 'static) {
    // Expected admission refusals can occur on thousands of schedules. Keep
    // diagnostics for every other panic, including all Loom violations.
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let original = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info
                .payload()
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| info.payload().downcast_ref::<&str>().copied())
                .unwrap_or("");
            if !msg.contains("overlapping DisjointMut") {
                original(info);
            }
        }));
    });
    let mut builder = loom::model::Builder::new();
    assert!(
        builder.max_permutations.is_none() && builder.max_duration.is_none(),
        "release models must not silently stop at a permutation or time cap"
    );
    // Exhaust schedules up to this preemption bound; no permutation/time cap.
    // Increasing LOOM_MAX_PREEMPTIONS extends the same production models.
    if std::env::var_os("LOOM_MAX_PREEMPTIONS").is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = std::env::var("LOOM_MAX_BRANCHES")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(20_000);
    builder.check(f);
}

fn b(range: Range<usize>) -> Bounds {
    Bounds { range }
}

struct Lease<'a>(&'a BorrowTracker, BorrowId);

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.0.remove(self.1);
    }
}

fn attempt(t: &BorrowTracker, range: Range<usize>, mutable: bool) -> Option<Lease<'_>> {
    // Catch only the specified rejection, not Loom failures or unrelated
    // panics. An unexpected panic must fail the model, never become a retry.
    match catch_unwind(AssertUnwindSafe(|| {
        if mutable {
            t.add_mut(&b(range))
        } else {
            t.add_immut(&b(range))
        }
    })) {
        Ok(id) => Some(Lease(t, id)),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(
                msg.contains("overlapping DisjointMut"),
                "unexpected panic: {msg}"
            );
            None
        }
    }
}

struct Buffer {
    tracker: BorrowTracker,
    payload: UnsafeCell<usize>,
}

// SAFETY: payload access is confined to a production-tracker lease below.
// Loom checks this very claim on every explored execution.
unsafe impl Sync for Buffer {}

fn race_ranges(
    ranges: impl Fn(&BorrowTracker) -> (Range<usize>, Range<usize>) + Send + Sync + 'static,
) {
    model(move || {
        set_parallelism(2);
        let buffer = Arc::new(Buffer {
            tracker: BorrowTracker::new(1 << 24),
            payload: UnsafeCell::new(0),
        });
        let (left, right) = ranges(&buffer.tracker);
        let threads: Vec<_> = [left, right]
            .into_iter()
            .map(|range| {
                let buffer = buffer.clone();
                thread::spawn(move || {
                    // Reuse is essential: a single admission cannot check the
                    // Release/Acquire handoff of writes from a previous owner.
                    for _ in 0..2 {
                        if let Some(lease) = attempt(&buffer.tracker, range.clone(), true) {
                            let ptr = buffer.payload.get_mut();
                            unsafe {
                                *ptr.deref() += 1;
                            }
                            thread::yield_now();
                            drop(ptr); // end the reference before retiring its record
                            drop(lease);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert!(unsafe { *buffer.payload.get().deref() } > 0);
    });
}

#[test]
fn narrow_exclusion_and_retirement_handoff() {
    race_ranges(|_| (0..1, 0..1));
}

#[test]
fn wide_narrow_exclusion_and_handoff() {
    // The entire allocation exceeds MAX_BLOCKS_SCAN at the derived shift.
    // Explicitly assert coverage rather than silently testing two narrows.
    model(|| {
        set_parallelism(2);
        let t = BorrowTracker::new(1 << 24);
        let id = t.add_mut(&b(0..1 << 24));
        assert_eq!(id.kind(), KIND_WIDE);
        t.remove(id);
    });
    race_ranges(|_| (0..1 << 24, 0..1));
}

#[test]
fn multi_shard_exclusion_and_handoff() {
    model(|| {
        set_parallelism(2);
        let t = BorrowTracker::new(1 << 20);
        let boundary = 1 << t.block_shift();
        let id = t.add_mut(&b(boundary - 1..boundary + 1));
        assert_eq!(id.kind(), KIND_NARROW);
        assert!(id.pairs() > 1);
        t.remove(id);
    });
    race_ranges(|t| {
        let boundary = 1 << t.block_shift();
        (boundary - 1..boundary + 1, boundary..boundary + 1)
    });
}

#[test]
fn last_reader_and_full_shard_fallback() {
    model(|| {
        let t = Arc::new(BorrowTracker::new(32));
        // Fill the real slot count and force the eighth reader to the wide
        // list. Dropping that wide record must not retire any narrow reader.
        let readers: Vec<_> = (0..SLOTS + 1)
            .map(|_| Lease(&t, t.add_immut(&b(0..1))))
            .collect();
        assert_eq!(readers.last().unwrap().1.kind(), KIND_WIDE);
        let mut readers = readers;
        let wide = readers.pop().unwrap();
        let last = readers.remove(0);
        drop(readers);
        // Refresh the conservative allocation bitmap before spawning. Racing
        // seven already-dead flags adds schedules, but no new protocol state.
        drop(Lease(&t, t.add_immut(&b(2..3))));
        let tc = t.clone();
        let writer = thread::spawn(move || {
            assert!(attempt(&tc, 0..1, true).is_none());
        });
        drop(wide);
        writer.join().unwrap();
        assert!(attempt(&t, 0..1, true).is_none());
        drop(last);
        assert!(attempt(&t, 0..1, true).is_some());
    });
}

#[test]
fn rectangle_rows_exclude_but_gaps_remain_available() {
    model(|| {
        let mut tracker = BorrowTracker::new(32);
        tracker.set_row_stride(32, 8);
        let t = Arc::new(tracker);
        let rect = Lease(&t, t.add_rect_mut(0, 2, 2, 8).expect("rectangle coverage"));
        let tc = t.clone();
        let other = thread::spawn(move || {
            assert!(attempt(&tc, 8..10, false).is_none());
            assert!(attempt(&tc, 2..8, true).is_some());
        });
        other.join().unwrap();
        drop(rect);
        assert!(attempt(&t, 8..10, true).is_some());
    });
}

#[test]
fn global_hints_cannot_remap_a_live_tracker() {
    model(|| {
        let t = Arc::new(BorrowTracker::new(1 << 20));
        let mapping = (t.mask, t.shift);
        let lease = Lease(&t, t.add_mut(&b(17..33)));
        let tc = t.clone();
        let update = thread::spawn(move || {
            set_parallelism(8);
            set_tile_concurrency(16);
            set_parallelism(1);
            set_tile_concurrency(1);
            let fresh = BorrowTracker::new(1 << 20);
            let own = Lease(&fresh, fresh.add_mut(&b(17..33)));
            assert!(attempt(&tc, 20..21, false).is_none());
            drop(own);
        });
        update.join().unwrap();
        assert_eq!((t.mask, t.shift), mapping);
        drop(lease);
        assert!(attempt(&t, 17..33, true).is_some());
    });
}

#[test]
fn local_policy_survives_concurrent_global_changes_and_retirement() {
    for threads in [1, 8] {
        model(move || {
            let mut tracker = BorrowTracker::new(1 << 20);
            tracker.configure_parallelism(1 << 20, threads, 4);
            let mapping = (tracker.mask, tracker.shift);
            let t = Arc::new(tracker);
            let lease = Lease(&t, t.add_mut(&b(17..33)));
            let tc = t.clone();
            let update = thread::spawn(move || {
                set_parallelism(24);
                set_tile_concurrency(32);
                assert!(attempt(&tc, 20..21, false).is_none());
            });
            update.join().unwrap();
            assert_eq!((t.mask, t.shift), mapping);
            drop(lease);
            let mut t = match Arc::try_unwrap(t) {
                Ok(t) => t,
                Err(_) => panic!("live owner"),
            };
            t.configure_parallelism(1 << 20, if threads == 1 { 8 } else { 1 }, 1);
            assert!(attempt(&t, 17..33, true).is_some());
        });
    }
}
