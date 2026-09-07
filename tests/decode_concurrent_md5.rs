//! Same-process decoder concurrency with pinned visible-pixel hashes.
//!
//! CI runs this binary in release and debug with the regular checked and asm
//! configurations. The separate `__simd_test` whole-plane comparison protocol
//! is single-threaded; that gate selects `decode_md5_committed` instead.

#[path = "common/committed_vectors.rs"]
mod committed_vectors;
use committed_vectors::{VECTORS, decode_md5_with_threads};

/// Same-process clients are essential here: nextest normally isolates test
/// cases, which cannot expose one decoder clobbering another decoder's globals.
#[test]
fn simultaneous_single_and_multi_thread_decoders_match_reference_md5() {
    assert!(
        !cfg!(feature = "__simd_test"),
        "__simd_test saves/restores entire planes and requires one decoder worker; \
         run decode_concurrent_md5 with the regular checked or asm features, and \
         run decode_md5_committed for the __simd_test differential gate"
    );
    let start = std::sync::Barrier::new(3);
    std::thread::scope(|scope| {
        let workers: Vec<_> = [1, 2, 4]
            .into_iter()
            .map(|threads| {
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    for _ in 0..3 {
                        for &(label, data, expected) in &VECTORS[..4] {
                            assert_eq!(
                                decode_md5_with_threads(data, threads),
                                expected,
                                "{label}, threads={threads}, same-process concurrent decoders"
                            );
                        }
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}
