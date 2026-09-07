//! Shared scratch sizes below the former 64K-element sharding threshold.
//! Keep the public API tests independent of the tracker's placement formula.

use rav1d_disjoint_mut::DisjointMut;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Barrier;

fn overlap(f: impl FnOnce()) {
    let error = catch_unwind(AssertUnwindSafe(f)).expect_err("overlap was admitted");
    let message = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(message.contains("overlapping DisjointMut"), "{message}");
}

#[test]
fn disjoint_lanes_survive_simultaneous_guards_and_slot_reuse() {
    // Twelve-byte elements also exercise element coordinates, not byte counts.
    for len in [1023, 1024, 1536, 6721, 11520] {
        let mut storage = DisjointMut::new(vec![[0u32; 3]; len]);
        storage.configure_parallelism(8, 4);
        let barrier = Barrier::new(8);
        std::thread::scope(|scope| {
            for lane in 0..8 {
                let storage = &storage;
                let barrier = &barrier;
                scope.spawn(move || {
                    for round in 1..=if cfg!(miri) { 2 } else { 64 } {
                        // Include windows that cross the ordinary 64-element block.
                        let start = lane * 128 + 60 + round % 8;
                        let value = [lane as u32, round as u32, 0xfeedbeef];
                        let mut held = storage.index_mut(start..start + 8);
                        held.fill(value);
                        barrier.wait();
                        if lane == 0 {
                            // A broad reader must meet every shard/overflow record.
                            overlap(|| drop(storage.index(..)));
                            rav1d_disjoint_mut::set_parallelism(24);
                            rav1d_disjoint_mut::set_tile_concurrency(32);
                        }
                        barrier.wait();
                        assert!(held.iter().all(|&v| v == value));
                        drop(held);
                        assert!(storage.index(start..start + 8).iter().all(|&v| v == value));
                    }
                });
            }
        });
    }
}

#[test]
fn broad_reader_excludes_narrow_writers_across_policy_and_size_changes() {
    let mut storage = DisjointMut::new(vec![7u16; 1024]);
    for (threads, len) in [(8, 1024), (1, 1536), (24, 6721), (1, 1023), (8, 11520)] {
        storage.configure_parallelism(threads, 4);
        storage.resize(len, 7);
        let held = storage.index(..);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for index in [0, 63, 64, len / 2, len - 1] {
                    overlap(|| *storage.index_mut(index) = 9);
                }
            });
        });
        assert!(held.iter().all(|&v| v == 7));
        drop(held);
        storage.index_mut(..).fill(7);
    }
}
