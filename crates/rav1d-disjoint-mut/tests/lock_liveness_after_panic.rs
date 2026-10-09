//! A panic inside the tracker's all-shard section must not leave any shard locked.
//!
//! Found by the independent audit of 0.4.0 (finding A-03). Two safe sequences made the wide
//! path panic while it held every shard lock, and the lock was never released, so every later
//! borrow of the instance (even of disjoint bytes) spun forever:
//!
//! 1. forget a rectangle guard, `resize` (which drops the stride hint), then admit a wide
//!    range: the stale rectangle record was decoded with stride 0 (division by zero);
//! 2. exhaust the `u16` wide-record ids: the capacity assert ran after the record was pushed.
//!
//! Records of forgotten guards stay reserved by design (conservative), so the first borrow in
//! sequence 1 is still refused as an overlap; what must hold is that the refusal is an
//! ordinary one and that the instance remains usable afterwards.
#![cfg(all(not(feature = "untracked"), not(feature = "__probe_count"), not(miri)))]
#![forbid(unsafe_code)]

use rav1d_disjoint_mut::DisjointMut;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;
use std::time::Duration;

fn message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// Run `f` on another thread and require it to finish: a stranded lock shows as a timeout.
fn finishes_within<R: Send + 'static>(what: &str, f: impl FnOnce() -> R + Send + 'static) -> R {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(5))
        .unwrap_or_else(|_| panic!("{what}: did not complete, a panic left shard locks held"))
}

#[test]
fn stale_rectangle_after_resize_is_a_conservative_hull_not_a_crash() {
    let mut dm = DisjointMut::new(vec![0u8; 65536]);
    dm.configure_parallelism(2, 1);
    dm.declare_row_stride(8);
    std::mem::forget(
        dm.index_rect_mut(0, 2, 2, 8)
            .expect("rectangle must be admitted"),
    );
    dm.resize(65536, 0); // drops the stride hint; the leaked record stays
    let err = catch_unwind(AssertUnwindSafe(|| drop(dm.index_mut(..))))
        .expect_err("the whole-buffer borrow overlaps the leaked record's hull");
    let msg = message(&err);
    assert!(
        msg.contains("overlap"),
        "refusal must be an ordinary overlap panic, got: {msg}"
    );
    assert!(
        !msg.contains("divide"),
        "must not be a division by zero: {msg}"
    );
}

#[test]
fn leaked_rect_resize_panic_does_not_strand_all_locks() {
    let mut dm = DisjointMut::new(vec![0u8; 65536]);
    dm.configure_parallelism(2, 1);
    dm.declare_row_stride(8);
    std::mem::forget(
        dm.index_rect_mut(0, 2, 2, 8)
            .expect("rectangle must be admitted"),
    );
    dm.resize(65536, 0);
    let _ = catch_unwind(AssertUnwindSafe(|| drop(dm.index_mut(..))));
    // A later borrow of bytes far from the leaked record must still be admitted.
    let ok = finishes_within(
        "disjoint admission after the refused wide borrow",
        move || catch_unwind(AssertUnwindSafe(|| drop(dm.index(32768..32769)))).is_ok(),
    );
    assert!(ok, "a disjoint borrow must be admitted");
}

#[test]
fn wide_record_limit_panic_does_not_strand_all_locks() {
    let dm = DisjointMut::new(vec![0u8; 8]);
    let exhaustion = catch_unwind(AssertUnwindSafe(|| {
        for _ in 0..65544 {
            std::mem::forget(dm.index(0..1));
        }
    }));
    let err = exhaustion.expect_err("expected a finite wide-record limit");
    assert!(
        message(&err).contains("too many concurrent wide borrows"),
        "unexpected panic: {}",
        message(&err)
    );
    // The table is still full of leaked records, so a borrow landing in the full shard also
    // needs a wide id and is refused at capacity. That refusal must be PROMPT and clean: a
    // stranded lock would make this call spin until the deadline instead.
    let outcome = finishes_within("admission after wide-record exhaustion", move || {
        catch_unwind(AssertUnwindSafe(|| drop(dm.index_mut(2..3))))
            .map_err(|payload| message(&payload))
    });
    if let Err(msg) = outcome {
        assert!(
            msg.contains("too many concurrent wide borrows"),
            "a refusal must be the capacity refusal, got: {msg}"
        );
    }
}
