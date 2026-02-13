//! Red-team tests for DisjointMut soundness.
//! Run under: cargo +nightly miri test
//! And: MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test

use rav1d_disjoint_mut::DisjointMut;
use std::sync::Arc;
use std::thread;

/// CRITICAL TEST: The core use case — one thread writes [0..50], another writes [50..100]
/// through a shared &DisjointMut. This is the whole point of the crate.
#[test]
fn test_concurrent_disjoint_mut_access() {
    let dm = Arc::new(DisjointMut::new(vec![0u8; 100]));

    let dm1 = dm.clone();
    let dm2 = dm.clone();

    let t1 = thread::spawn(move || {
        let mut guard = dm1.index_mut(0..50);
        for i in 0..50 {
            guard[i] = 1;
        }
    });

    let t2 = thread::spawn(move || {
        let mut guard = dm2.index_mut(50..100);
        for i in 0..50 {
            guard[i] = 2;
        }
    });

    t1.join().unwrap();
    t2.join().unwrap();

    // Verify results
    let guard = dm.index(0..100);
    for i in 0..50 {
        assert_eq!(guard[i], 1);
    }
    for i in 50..100 {
        assert_eq!(guard[i], 2);
    }
}

/// Test: immut guard on [0..50] while mut guard on [50..100] — single threaded.
/// This exercises the as_mut_ptr path creating &mut Vec while &[u8] exists.
#[test]
fn test_disjoint_immut_and_mut_single_thread() {
    let dm = DisjointMut::new(vec![0u8; 100]);
    let immut_guard = dm.index(0..50);
    let mut mut_guard = dm.index_mut(50..100);
    mut_guard[0] = 42;
    assert_eq!(immut_guard[0], 0);
    assert_eq!(mut_guard[0], 42);
}

/// Test: two disjoint mutable guards simultaneously — single threaded.
#[test]
fn test_two_disjoint_mut_guards_single_thread() {
    let dm = DisjointMut::new(vec![0u8; 100]);
    let mut g1 = dm.index_mut(0..50);
    let mut g2 = dm.index_mut(50..100);
    g1[49] = 1;
    g2[0] = 2;
    assert_eq!(g1[49], 1);
    assert_eq!(g2[0], 2);
}

/// Test: immut guard coexists with another immut guard on overlapping range.
#[test]
fn test_overlapping_immut_guards() {
    let dm = DisjointMut::new(vec![42u8; 100]);
    let g1 = dm.index(0..60);
    let g2 = dm.index(40..100);
    assert_eq!(g1[50], g2[10]);
}

/// Test: concurrent reads from multiple threads.
#[test]
fn test_concurrent_reads() {
    let dm = Arc::new(DisjointMut::new(vec![42u8; 100]));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let dm = dm.clone();
            thread::spawn(move || {
                let guard = dm.index(0..100);
                assert_eq!(guard[50], 42);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

/// Test: mut + immut on disjoint ranges across threads.
#[test]
fn test_cross_thread_mut_immut_disjoint() {
    let dm = Arc::new(DisjointMut::new(vec![0u8; 100]));

    let dm1 = dm.clone();
    let dm2 = dm.clone();

    let t1 = thread::spawn(move || {
        let guard = dm1.index(0..50);
        assert_eq!(guard[0], 0);
    });

    let t2 = thread::spawn(move || {
        let mut guard = dm2.index_mut(50..100);
        guard[0] = 99;
    });

    t1.join().unwrap();
    t2.join().unwrap();
}

/// Test: guard drop deregisters correctly, allowing re-borrow.
#[test]
fn test_guard_drop_enables_reborrow() {
    let dm = DisjointMut::new(vec![0u8; 100]);
    {
        let mut g = dm.index_mut(0..100);
        g[0] = 1;
    }
    // Guard dropped — should be able to borrow again
    let g = dm.index(0..100);
    assert_eq!(g[0], 1);
}

/// Test: overlapping mut panics (only when tracking is active).
#[test]
#[should_panic(expected = "overlapping")]
fn test_overlapping_mut_panics() {
    let dm = DisjointMut::new(vec![0u8; 100]);
    let _g1 = dm.index_mut(0..60);
    let _g2 = dm.index_mut(40..100);
}

/// Test: mut overlapping with immut panics (only when tracking is active).
#[test]
#[should_panic(expected = "overlapping")]
fn test_mut_overlapping_immut_panics() {
    let dm = DisjointMut::new(vec![0u8; 100]);
    let _g1 = dm.index(0..60);
    let _g2 = dm.index_mut(40..100);
}

/// Test: immut overlapping with existing mut panics (only when tracking is active).
#[test]
#[should_panic(expected = "overlapping")]
fn test_immut_overlapping_mut_panics() {
    let dm = DisjointMut::new(vec![0u8; 100]);
    let _g1 = dm.index_mut(0..60);
    let _g2 = dm.index(40..100);
}

/// Test: single-element mut guards at adjacent indices.
#[test]
fn test_adjacent_element_guards() {
    let dm = DisjointMut::new(vec![0u8; 10]);
    let mut g1 = dm.index_mut(5usize);
    let mut g2 = dm.index_mut(6usize);
    *g1 = 5;
    *g2 = 6;
    assert_eq!(*g1, 5);
    assert_eq!(*g2, 6);
}

/// Empty ranges borrow zero bytes and should never conflict with anything.
#[test]
fn test_empty_range_no_conflict() {
    let dm = DisjointMut::new(vec![0u8; 100]);
    let _g1 = dm.index_mut(50..50); // empty range
    let mut g2 = dm.index_mut(0..100); // entire buffer — no conflict
    g2[50] = 42;
    assert_eq!(g2[50], 42);
}

/// Two empty ranges at the same position don't conflict.
#[test]
fn test_two_empty_ranges_same_position() {
    let dm = DisjointMut::new(vec![0u8; 100]);
    let _g1 = dm.index_mut(50..50);
    let _g2 = dm.index_mut(50..50);
}

/// OOB panic poisons the data structure. After recovery, all borrows fail.
/// This follows std::sync::Mutex semantics — after a panic, the data may be
/// in an inconsistent state, so we fail loudly rather than silently allowing
/// access to potentially corrupted data.
#[test]
fn test_oob_panic_poisons() {
    let dm = DisjointMut::new(vec![0u8; 10]);

    // This should panic because index is OOB
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = dm.index_mut(0..100); // OOB
    }));
    assert!(result.is_err());

    // The DisjointMut is now poisoned — all future borrows should panic.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = dm.index_mut(0..5);
    }));
    assert!(result.is_err());
}

/// Panic while holding a mutable guard poisons the data structure.
/// Requires `std` feature (poisoning uses `thread::panicking()`).
#[test]
#[cfg(feature = "std")]
fn test_panic_during_mut_guard_poisons() {
    let dm = DisjointMut::new(vec![0u8; 100]);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut g = dm.index_mut(0..50);
        g[0] = 42;
        panic!("simulated write failure");
    }));
    assert!(result.is_err());

    // Poisoned — even non-overlapping borrows fail.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = dm.index(50..100);
    }));
    assert!(result.is_err());
}

/// Panic while holding an immutable guard does NOT poison.
/// Immutable guards don't modify data, so no inconsistency is possible.
#[test]
fn test_panic_during_immut_guard_no_poison() {
    let dm = DisjointMut::new(vec![42u8; 100]);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let g = dm.index(0..50);
        assert_eq!(g[0], 42);
        panic!("simulated read failure");
    }));
    assert!(result.is_err());

    // NOT poisoned — immutable guards don't corrupt data.
    let g = dm.index(0..50);
    assert_eq!(g[0], 42);
}

/// Test: concurrent disjoint access with Box<[u8]> backing.
/// Probes whether addr_of_mut!(**ptr) creates conflicting retags.
#[test]
fn test_concurrent_disjoint_box_slice() {
    use rav1d_disjoint_mut::DisjointMutSlice;

    let dm: DisjointMutSlice<u8> = DisjointMut::new(vec![0u8; 100].into_boxed_slice());
    let dm = Arc::new(dm);

    let dm1 = dm.clone();
    let dm2 = dm.clone();

    let t1 = thread::spawn(move || {
        let mut guard = dm1.index_mut(0..50);
        for i in 0..50 {
            guard[i] = 1;
        }
    });

    let t2 = thread::spawn(move || {
        let mut guard = dm2.index_mut(50..100);
        for i in 0..50 {
            guard[i] = 2;
        }
    });

    t1.join().unwrap();
    t2.join().unwrap();

    let guard = dm.index(0..100);
    for i in 0..50 {
        assert_eq!(guard[i], 1);
    }
    for i in 50..100 {
        assert_eq!(guard[i], 2);
    }
}

/// Test: concurrent disjoint access with array backing.
#[test]
fn test_concurrent_disjoint_array() {
    let dm = Arc::new(DisjointMut::new([0u8; 100]));

    let dm1 = dm.clone();
    let dm2 = dm.clone();

    let t1 = thread::spawn(move || {
        let mut guard = dm1.index_mut(0..50);
        for i in 0..50 {
            guard[i] = 1;
        }
    });

    let t2 = thread::spawn(move || {
        let mut guard = dm2.index_mut(50..100);
        for i in 0..50 {
            guard[i] = 2;
        }
    });

    t1.join().unwrap();
    t2.join().unwrap();

    let guard = dm.index(0..100);
    assert_eq!(guard[49], 1);
    assert_eq!(guard[50], 2);
}

/// Test: zerocopy cast — u8 buffer accessed as [u8; 4] elements.
/// Uses [u8; 4] instead of u32 to avoid alignment requirements that
/// Miri may not satisfy for Vec<u8>.
#[cfg(feature = "zerocopy")]
#[test]
fn test_zerocopy_cast_disjoint() {
    let dm = DisjointMut::new(vec![0u8; 16]);
    let mut g1 = dm.mut_slice_as::<_, [u8; 4]>(0..2);
    let mut g2 = dm.mut_slice_as::<_, [u8; 4]>(2..4);
    g1[0] = [1, 2, 3, 4];
    g2[0] = [5, 6, 7, 8];
    assert_eq!(g1[0], [1, 2, 3, 4]);
    assert_eq!(g2[0], [5, 6, 7, 8]);
}

/// Test: DisjointMutArcSlice concurrent access.
#[test]
fn test_arc_slice_concurrent() {
    use rav1d_disjoint_mut::DisjointMutArcSlice;

    let dm: DisjointMutArcSlice<u8> = (0..100u8).collect();
    let dm1 = dm.clone();
    let dm2 = dm.clone();

    let t1 = thread::spawn(move || {
        let guard = dm1.inner.index(0..50);
        assert_eq!(guard[0], 0);
    });

    let t2 = thread::spawn(move || {
        let guard = dm2.inner.index(50..100);
        assert_eq!(guard[0], 50);
    });

    t1.join().unwrap();
    t2.join().unwrap();
}
