//! A poisoned `DisjointMut` must refuse EVERY later borrow, including an empty
//! range (which touches no bytes). Regression test for an adversarial review
//! finding: the empty-range early return in the tracker skipped the poison check.
// Observes the overlap tracker's poisoning, which `untracked` removes by design.
#![cfg(not(feature = "untracked"))]

use rav1d_disjoint_mut::DisjointMut;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

#[test]
fn empty_ranges_are_refused_after_poison() {
    let dm = DisjointMut::new_eager(vec![0u8; 64]);
    // An out-of-bounds borrow panics and poisons the whole structure.
    assert!(catch_unwind(AssertUnwindSafe(|| drop(dm.index_mut(64)))).is_err());
    // Everything after that must panic, however small, including empty ranges.
    for (name, f) in [
        ("index 0", Box::new(|| drop(dm.index(0))) as Box<dyn Fn()>),
        ("index_mut 0..0", Box::new(|| drop(dm.index_mut(0..0)))),
        ("index 64..64", Box::new(|| drop(dm.index(64..64)))),
        ("index_mut 10..10", Box::new(|| drop(dm.index_mut(10..10)))),
        ("index 5..3 (inverted)", Box::new(|| drop(dm.index(5..3)))),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(f)).is_err(),
            "poisoned structure admitted: {name}"
        );
    }
}

#[test]
fn empty_ranges_work_when_not_poisoned() {
    let dm = DisjointMut::new_eager(vec![0u8; 64]);
    drop(dm.index_mut(0..0));
    drop(dm.index(64..64));
    drop(dm.index_mut(10..10));
}
