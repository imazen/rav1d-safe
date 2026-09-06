//! Compatibility and adversarial gates for the published 0.3 API only.
use rav1d_disjoint_mut::{DisjointImmutGuard, DisjointMut, DisjointMutGuard};
use std::panic::{AssertUnwindSafe, catch_unwind};

const fn array_buffer() -> DisjointMut<[u32; 8]> {
    DisjointMut::new([0; 8])
}
static BUFFER: DisjointMut<[u32; 8]> = array_buffer();
static EMPTY: DisjointMut<Vec<u8>> = DisjointMut::new(Vec::new());
static CHECKED: bool = BUFFER.is_checked();

fn refuses(f: impl FnOnce(), reason: &str) {
    let error = catch_unwind(AssertUnwindSafe(f)).expect_err("invalid access was accepted");
    let text = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(text.contains(reason), "unexpected rejection: {text}");
}

#[test]
fn const_and_static_construction_stays_checked_across_threads() {
    assert!(CHECKED && EMPTY.is_checked());
    assert!(EMPTY.index(..).is_empty());
    let mut left = BUFFER.index_mut(0..4);
    std::thread::scope(|s| {
        s.spawn(|| {
            refuses(
                || {
                    drop(BUFFER.index_mut(2..3));
                },
                "overlapping DisjointMut",
            );
            BUFFER.index_mut(4..8).fill(2);
        })
        .join()
        .unwrap();
        left.fill(1);
    });
    drop(left);
    assert_eq!(&*BUFFER.index(..), &[1, 1, 1, 1, 2, 2, 2, 2]);
}

#[test]
fn existing_guard_send_sync_bounds_and_moves_compile() {
    fn both<T: Send + Sync>() {}
    both::<DisjointMutGuard<'static, Vec<u8>, [u8]>>();
    both::<DisjointImmutGuard<'static, Vec<u8>, [u8]>>();
    let dm = DisjointMut::new(vec![0u8; 8]);
    let guard = dm.index_mut(..);
    std::thread::scope(|s| {
        s.spawn(move || {
            let mut moved = Box::new(guard);
            moved.fill(9);
        })
        .join()
        .unwrap();
    });
    let shared = dm.index(..);
    std::thread::scope(|s| {
        s.spawn(move || assert_eq!(&*shared, &[9; 8]))
            .join()
            .unwrap();
    });
    dm.index_mut(..).fill(3);
}

#[test]
fn interval_admission_matches_an_independent_element_set() {
    let ranges = [0..0, 0..1, 0..3, 1..2, 1..4, 3..4, 4..4];
    let mut accepted = 0;
    let mut rejected = 0;
    for a in &ranges {
        for b in &ranges {
            for (a_mut, b_mut) in [(true, true), (true, false), (false, true), (false, false)] {
                let conflict = (a_mut || b_mut) && a.clone().any(|i| b.clone().any(|j| i == j));
                let dm = DisjointMut::new(vec![0u16; 4]);
                let try_second = || {
                    if b_mut {
                        dm.index_mut(b.clone()).fill(2);
                    } else {
                        assert_eq!(dm.index(b.clone()).len(), b.len());
                    }
                };
                let run = || {
                    if conflict {
                        refuses(try_second, "overlapping DisjointMut");
                    } else {
                        try_second();
                    }
                };
                if a_mut {
                    let mut first = dm.index_mut(a.clone());
                    run();
                    first.fill(1); // challenge temporary invalidation under Miri
                } else {
                    let first = dm.index(a.clone());
                    run();
                    assert!(first.iter().all(|&v| v == 0));
                }
                if conflict {
                    rejected += 1;
                } else {
                    accepted += 1;
                }
            }
        }
    }
    assert!(accepted > 0 && rejected > 0);
}

#[test]
fn slot_exhaustion_and_reuse_never_release_a_forgotten_guard() {
    let dm = DisjointMut::new(vec![0u8; 256]);
    std::mem::forget(dm.index_mut(0));
    let mut guards: Vec<_> = (1..254).map(|i| dm.index_mut(i)).collect();
    refuses(
        || {
            drop(dm.index_mut(254));
        },
        "too many concurrent borrows",
    );
    for g in &mut guards {
        **g = 7;
    }
    drop(guards.pop());
    dm.index_mut(254..256).fill(9);
    refuses(
        || {
            drop(dm.index(0));
        },
        "overlapping DisjointMut",
    );
    assert!(guards.iter().all(|g| **g == 7));
    drop(guards);
    for _ in 0..3 {
        let all: Vec<_> = (1..254).map(|i| dm.index(i)).collect();
        refuses(
            || {
                drop(dm.index_mut(0));
            },
            "overlapping DisjointMut",
        );
        drop(all);
    }
}

#[test]
fn zst_and_reversed_ranges_do_not_corrupt_other_records() {
    let dm = DisjointMut::new([(); 4]);
    let left = dm.index_mut(0..2);
    let right = dm.index_mut(2..4);
    let empty = dm.index_mut(1..1);
    assert!(empty.is_empty());
    refuses(
        || {
            drop(dm.index(1));
        },
        "overlapping DisjointMut",
    );
    drop((left, right, empty));
    let start = 3;
    let end = 2;
    assert!(catch_unwind(AssertUnwindSafe(|| dm.index_mut(start..end))).is_err());
    refuses(
        || {
            drop(dm.index(0));
        },
        "poisoned",
    );
}

#[cfg(all(feature = "aligned", feature = "zerocopy"))]
#[test]
fn all_four_cast_paths_keep_the_original_reservation() {
    let dm = DisjointMut::new(aligned::Aligned::<aligned::A16, _>([0u8; 64]));
    let mut slice = Box::new(dm.mut_slice_as::<_, u32>(1..3));
    let mut element = dm.mut_element_as::<u32>(3);
    refuses(
        || {
            drop(dm.index(5));
        },
        "overlapping DisjointMut",
    );
    slice.fill(7);
    *element = 9;
    drop((slice, element));
    let slice = dm.slice_as::<_, u32>(1..3);
    let element = Box::new(dm.element_as::<u32>(3));
    refuses(
        || {
            drop(dm.index_mut(5));
        },
        "overlapping DisjointMut",
    );
    assert_eq!(&*slice, &[7, 7]);
    assert_eq!(**element, 9);
    drop((slice, element));
    dm.index_mut(..).fill(1);
}
