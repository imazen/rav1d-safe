//! Benchmark the borrow-tracking lock overhead in DisjointMut.
//!
//! Measures the cost of checked borrow/release cycles, which is dominated
//! by the lock acquisition in BorrowTracker. Useful for comparing spinlock
//! vs Mutex implementations.

use criterion::{Criterion, criterion_group, criterion_main};
use rav1d_disjoint_mut::DisjointMut;
use std::hint::black_box;

/// Single-threaded borrow/release: measures uncontended lock overhead.
fn bench_uncontended_mut_borrow(c: &mut Criterion) {
    let buf = DisjointMut::new(vec![0u8; 4096]);

    c.bench_function("uncontended_mut_borrow", |b| {
        b.iter(|| {
            let guard = buf.index_mut(black_box(0usize)..black_box(32usize));
            drop(black_box(guard));
        });
    });
}

/// Single-threaded immutable borrow/release.
fn bench_uncontended_immut_borrow(c: &mut Criterion) {
    let buf = DisjointMut::new(vec![0u8; 4096]);

    c.bench_function("uncontended_immut_borrow", |b| {
        b.iter(|| {
            let guard = buf.index(black_box(0usize)..black_box(32usize));
            drop(black_box(guard));
        });
    });
}

/// Alternating disjoint mutable borrows (simulates rav1d tile pattern).
fn bench_disjoint_mut_cycle(c: &mut Criterion) {
    let buf = DisjointMut::new(vec![0u8; 4096]);

    c.bench_function("disjoint_mut_4_regions", |b| {
        b.iter(|| {
            for i in 0..4 {
                let start = i * 1024;
                let guard = buf.index_mut(black_box(start)..black_box(start + 1024));
                drop(black_box(guard));
            }
        });
    });
}

/// Multiple concurrent immutable borrows (reader pattern).
fn bench_concurrent_immut_borrows(c: &mut Criterion) {
    let buf = DisjointMut::new(vec![0u8; 4096]);

    c.bench_function("concurrent_4_immut_borrows", |b| {
        b.iter(|| {
            let g1 = buf.index(black_box(0usize)..black_box(1024usize));
            let g2 = buf.index(black_box(1024usize)..black_box(2048usize));
            let g3 = buf.index(black_box(2048usize)..black_box(3072usize));
            let g4 = buf.index(black_box(3072usize)..black_box(4096usize));
            drop(black_box(g4));
            drop(black_box(g3));
            drop(black_box(g2));
            drop(black_box(g1));
        });
    });
}

/// Two threads doing disjoint mutable borrows (contention test).
fn bench_two_thread_contention(c: &mut Criterion) {
    use std::sync::Arc;

    let buf = Arc::new(DisjointMut::new(vec![0u8; 4096]));

    c.bench_function("two_thread_disjoint_mut", |b| {
        b.iter(|| {
            let buf2 = Arc::clone(&buf);
            let handle = std::thread::spawn(move || {
                for _ in 0..100 {
                    let guard = buf2.index_mut(black_box(2048usize)..black_box(4096usize));
                    drop(black_box(guard));
                }
            });
            for _ in 0..100 {
                let guard = buf.index_mut(black_box(0usize)..black_box(2048usize));
                drop(black_box(guard));
            }
            handle.join().unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_uncontended_mut_borrow,
    bench_uncontended_immut_borrow,
    bench_disjoint_mut_cycle,
    bench_concurrent_immut_borrows,
    bench_two_thread_contention,
);
criterion_main!(benches);
