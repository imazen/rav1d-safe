//! The 0.3 constructor contract, including concurrent first use and mutation
//! before/after initialization. Run with std, no_std, and both Miri models.
use rav1d_disjoint_mut::DisjointMut;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Barrier;

const fn buffer() -> DisjointMut<[u32; 8]> {
    DisjointMut::new([0; 8])
}
static BUFFER: DisjointMut<[u32; 8]> = buffer();
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
fn const_static_concurrent_first_use_stays_checked() {
    assert!(CHECKED && EMPTY.is_checked());
    assert!(EMPTY.index(..).is_empty());
    let start = Barrier::new(4);
    let held = Barrier::new(4);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..4)
            .map(|i| {
                let start = &start;
                let held = &held;
                scope.spawn(move || {
                    start.wait();
                    let mut own = BUFFER.index_mut(i * 2..i * 2 + 2);
                    held.wait();
                    refuses(|| drop(BUFFER.index_mut(i * 2)), "overlapping DisjointMut");
                    own.fill(i as u32 + 1);
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    });
    assert_eq!(&*BUFFER.index(..), &[1, 1, 2, 2, 3, 3, 4, 4]);
}

#[test]
fn both_constructors_preserve_live_leases_when_global_hints_change() {
    for new in [DisjointMut::new, DisjointMut::new_eager] {
        let dm = new(vec![0u8; 8192]);
        let mut held = dm.index_mut(100..200);
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    rav1d_disjoint_mut::set_parallelism(64);
                    rav1d_disjoint_mut::set_tile_concurrency(64);
                    refuses(|| drop(dm.index(120..160)), "overlapping DisjointMut");
                    dm.index_mut(200..300).fill(7);
                })
                .join()
                .unwrap();
            held.fill(3);
        });
        drop(held);
        assert!(dm.index(100..200).iter().all(|&b| b == 3));
        assert!(dm.index(200..300).iter().all(|&b| b == 7));
    }
}

#[test]
fn resize_and_stride_before_and_after_first_borrow() {
    for new in [DisjointMut::new, DisjointMut::new_eager] {
        let mut dm = new(Vec::<u16>::new());
        dm.resize(32, 0);
        dm.declare_row_stride(8);
        {
            let mut rect = dm.index_rect_mut(0, 2, 2, 8).expect("rectangle");
            refuses(|| drop(dm.index_mut(8..10)), "overlapping DisjointMut");
            dm.index_mut(2..8).fill(2);
            rect.row_mut(1).fill(1);
        }
        dm.resize(64, 3);
        dm.declare_row_stride(16);
        let rect = dm.index_rect(32, 2, 2, 16).expect("resized rectangle");
        assert_eq!(rect.row(1), &[3, 3]);
        refuses(|| drop(dm.index_mut(48..50)), "overlapping DisjointMut");
    }
}

#[test]
fn moved_initialized_buffer_preserves_poison_and_leaked_reservations() {
    for new in [DisjointMut::new, DisjointMut::new_eager] {
        let dm = new(vec![0u8; 16]);
        core::mem::forget(dm.index_mut(0..4));
        let moved = Box::new(dm);
        refuses(|| drop(moved.index(0..1)), "overlapping DisjointMut");
        moved.index_mut(4..8).fill(9);
        refuses(|| drop(moved.index_mut(20..21)), "out of range");
        refuses(|| drop(moved.index(8..9)), "poisoned");
    }
}

#[test]
fn exclusive_storage_mutation_before_initialization_uses_current_length() {
    let mut dm = DisjointMut::new(Vec::<u8>::new());
    dm.get_mut().resize(4096, 5);
    let mut moved = Box::new(dm);
    assert_eq!(*moved.index(4095), 5);
    moved.get_mut().resize(8192, 6);
    assert_eq!(*moved.index(8191), 6);
    assert_eq!(moved.into_inner().len(), 8192);
    assert!(DisjointMut::new(Vec::<u8>::new()).into_inner().is_empty());
}
