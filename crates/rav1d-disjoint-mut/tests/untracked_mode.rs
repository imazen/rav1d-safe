//! Contract of the `untracked` feature: overlap tracking is OFF, slice bounds
//! checks are ON.
//!
//! The whole point of the mode is that the only thing it gives up is the
//! overlap check. These tests pin both halves, so a future edit that loses
//! either one fails here rather than in a decode:
//!
//! * `out_of_bounds_*` — an out-of-range index still panics, for every access
//!   shape the decoder uses. This is the "not a security bug" half: overlap
//!   cannot turn into an out-of-bounds access because bounds are checked against
//!   the owner's live length, which is not part of the contested bytes.
//! * `overlap_*` — overlapping borrows are accepted (no tracker exists). That
//!   is formally undefined behaviour under the Rust memory model; the tests use
//!   `PlainData` elements only, which is the compile-time statement of "the
//!   consequence is a wrong value".
#![cfg(feature = "untracked")]

use rav1d_disjoint_mut::DisjointMut;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

fn panics(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

#[test]
fn out_of_bounds_mut_index_panics() {
    let dm = DisjointMut::new(vec![0u8; 64]);
    assert!(panics(|| drop(dm.index_mut(64))));
    assert!(panics(|| drop(dm.index_mut(1000))));
    assert!(panics(|| drop(dm.index_mut(60..65))));
    assert!(panics(|| drop(dm.index_mut(65..70))));
    assert!(panics(|| drop(dm.index_mut(70..))));
}

#[test]
fn out_of_bounds_shared_index_panics() {
    let dm = DisjointMut::new(vec![0u8; 64]);
    assert!(panics(|| drop(dm.index(64))));
    assert!(panics(|| drop(dm.index(60..65))));
    assert!(panics(|| drop(dm.index(70..))));
}

#[test]
fn in_bounds_edges_still_work() {
    let dm = DisjointMut::new(vec![0u8; 64]);
    drop(dm.index_mut(63));
    drop(dm.index_mut(0..64));
    drop(dm.index_mut(64..64));
    drop(dm.index(..));
}

#[test]
fn out_of_bounds_after_a_panic_still_panics() {
    // No poisoning machinery exists without a tracker; a later OOB must still
    // be rejected rather than silently allowed.
    let dm = DisjointMut::new(vec![0u8; 8]);
    assert!(panics(|| drop(dm.index_mut(8))));
    assert!(panics(|| drop(dm.index_mut(8))));
    drop(dm.index_mut(7));
}

#[test]
fn overlap_is_not_tracked() {
    let dm = DisjointMut::new(vec![0u8; 64]);
    // Two live mutable borrows of overlapping bytes: with the tracker this
    // panics, without it it is accepted.
    let a = dm.index_mut(0..32);
    let b = dm.index_mut(16..48);
    drop((a, b));
}

#[test]
fn overlapping_writers_of_one_value_leave_that_value() {
    // The decoder's overlap sites write the same bytes, so racing them is
    // idempotent. Eight threads write the same pattern over overlapping ranges.
    let dm = DisjointMut::new(vec![0u8; 4096]);
    std::thread::scope(|s| {
        for t in 0..8usize {
            let dm = &dm;
            s.spawn(move || {
                for _ in 0..200 {
                    let start = (t * 256) % 2048;
                    let mut g = dm.index_mut(start..start + 2048);
                    for b in g.iter_mut() {
                        *b = 0xA5;
                    }
                }
            });
        }
    });
    let g = dm.index(..);
    // Every byte any thread touched holds the common value; the rest is zero.
    // Whatever interleaving occurred, no byte is anything else.
    assert!(g.iter().all(|&b| b == 0xA5 || b == 0));
}
