//! The published 0.3 tracker, instrumented with Loom metadata and payload cells.
use super::*;
use loom::{cell::UnsafeCell, sync::Arc, thread};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::string::String;

fn model(f: impl Fn() + Sync + Send + 'static) {
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
    if std::env::var_os("LOOM_MAX_PREEMPTIONS").is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 20_000;
    builder.check(f);
}

struct Lease<'a>(&'a BorrowTracker, BorrowId);
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.0.remove(self.1);
    }
}

fn attempt(t: &BorrowTracker, mutable: bool) -> Option<Lease<'_>> {
    let bounds = Bounds { range: 0..1 };
    match catch_unwind(AssertUnwindSafe(|| {
        if mutable {
            t.add_mut(&bounds)
        } else {
            t.add_immut(&bounds)
        }
    })) {
        Ok(id) => Some(Lease(t, id)),
        Err(error) => {
            let msg = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied())
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
// SAFETY: every payload reference below is confined to a tracker reservation.
// Loom checks the same claim through the instrumented payload and metadata.
unsafe impl Sync for Buffer {}

fn race(overflow: bool, readers: bool) {
    model(move || {
        let buffer = Arc::new(Buffer {
            tracker: BorrowTracker::new(),
            payload: UnsafeCell::new(0),
        });
        let seeds: Vec<_> = if overflow {
            (1..=INLINE_SLOTS)
                .map(|i| buffer.tracker.add_immut(&Bounds { range: i..i + 1 }))
                .collect()
        } else {
            Vec::new()
        };
        // Prove the selected path actually occurs, with the same slot capacity
        // as production. No reduced-capacity model hides the overflow boundary.
        let probe = attempt(&buffer.tracker, true).unwrap();
        assert_eq!(probe.1.0 as usize >= INLINE_SLOTS, overflow);
        drop(probe);
        let roles: &[bool] = if readers {
            &[false, false, true]
        } else {
            &[true, true]
        };
        let threads: Vec<_> = roles
            .iter()
            .map(|&mutable| {
                let b = buffer.clone();
                thread::spawn(move || {
                    for _ in 0..2 {
                        if let Some(lease) = attempt(&b.tracker, mutable) {
                            if mutable {
                                let ptr = b.payload.get_mut();
                                unsafe {
                                    *ptr.deref() += 1;
                                }
                                thread::yield_now();
                                drop(ptr);
                            } else {
                                let ptr = b.payload.get();
                                let observed = unsafe { *ptr.deref() };
                                thread::yield_now();
                                assert_eq!(unsafe { *ptr.deref() }, observed);
                                drop(ptr);
                            }
                            drop(lease); // references end before retirement
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        for id in seeds {
            buffer.tracker.remove(id);
        }
        let final_lease = attempt(&buffer.tracker, true).unwrap();
        buffer.payload.with_mut(|ptr| unsafe { *ptr += 1 });
        drop(final_lease);
    });
}

#[test]
fn inline_exclusion_reuse_and_payload_handoff() {
    race(false, false);
}
#[test]
fn overflow_exclusion_reuse_and_payload_handoff() {
    race(true, false);
}
#[test]
fn shared_reader_retirement_and_writer_exclusion() {
    race(false, true);
}
