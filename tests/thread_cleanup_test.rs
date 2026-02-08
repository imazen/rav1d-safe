//! Thread cleanup tests - verify worker threads are properly joined
//!
//! These tests ensure that when a multi-threaded decoder is dropped,
//! all worker threads are properly joined and don't leak.
//!
//! All tests are serialized via a static mutex, and thread counting
//! only looks at rav1d-worker-* threads to avoid test harness noise.

use rav1d_safe::src::managed::{Decoder, Settings};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

/// Mutex to serialize ALL thread tests. Any test that creates a Decoder
/// must hold this lock to avoid thread count interference.
static THREAD_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the serialization lock, recovering from poison if a previous test panicked.
fn lock() -> MutexGuard<'static, ()> {
    THREAD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Count rav1d worker threads in the current process (Linux only).
/// Only counts threads whose comm name starts with "rav1d-worker",
/// avoiding interference from the test harness threads.
#[cfg(target_os = "linux")]
fn count_worker_threads() -> usize {
    let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            let comm_path = e.path().join("comm");
            std::fs::read_to_string(comm_path)
                .map(|name| name.trim().starts_with("rav1d-worker"))
                .unwrap_or(false)
        })
        .count()
}

#[test]
#[cfg(target_os = "linux")]
fn test_single_threaded_no_leak() {
    let _lock = lock();

    // Single-threaded decoder should not spawn workers
    assert_eq!(
        count_worker_threads(),
        0,
        "Stale worker threads from previous test"
    );

    {
        let mut decoder = Decoder::new().unwrap();
        let _ = decoder.decode(&[]);

        // Should have zero rav1d workers (single-threaded mode)
        assert_eq!(
            count_worker_threads(),
            0,
            "Single-threaded decoder should not spawn workers"
        );
    }

    // After drop, still zero
    assert_eq!(count_worker_threads(), 0);
}

#[test]
#[cfg(target_os = "linux")]
fn test_multi_threaded_cleanup() {
    let _lock = lock();

    assert_eq!(count_worker_threads(), 0, "Stale workers before test");

    {
        let mut decoder = Decoder::with_settings(Settings {
            threads: 4, // Spawn 4 worker threads
            ..Default::default()
        })
        .unwrap();

        let _ = decoder.decode(&[]);

        // Should have spawned workers
        let workers = count_worker_threads();
        assert!(workers >= 4, "Expected at least 4 workers, got {}", workers);
    }

    // Threads should be joined synchronously by Decoder::drop
    std::thread::sleep(Duration::from_millis(50));

    assert_eq!(
        count_worker_threads(),
        0,
        "Worker threads leaked after drop"
    );
}

#[test]
#[cfg(target_os = "linux")]
fn test_auto_detect_threads_cleanup() {
    let _lock = lock();

    assert_eq!(count_worker_threads(), 0, "Stale workers before test");

    {
        let mut decoder = Decoder::with_settings(Settings {
            threads: 0, // Auto-detect (will spawn workers)
            ..Default::default()
        })
        .unwrap();

        let _ = decoder.decode(&[]);

        // Should have spawned workers
        let workers = count_worker_threads();
        assert!(
            workers > 0,
            "Auto-detect should spawn worker threads, got {}",
            workers
        );
    }

    std::thread::sleep(Duration::from_millis(50));

    assert_eq!(
        count_worker_threads(),
        0,
        "Worker threads leaked with auto-detect"
    );
}

#[test]
#[cfg(target_os = "linux")]
fn test_multiple_decoder_cycles() {
    let _lock = lock();

    assert_eq!(count_worker_threads(), 0, "Stale workers before test");

    // Create and drop multiple decoders to ensure no accumulation of leaked threads
    for i in 0..5 {
        let mut decoder = Decoder::with_settings(Settings {
            threads: 2,
            ..Default::default()
        })
        .unwrap();

        let _ = decoder.decode(&[]);
        drop(decoder);

        std::thread::sleep(Duration::from_millis(20));

        assert_eq!(
            count_worker_threads(),
            0,
            "Workers leaked after cycle {}",
            i
        );
    }
}

#[test]
fn test_drop_without_decode() {
    let _lock = lock();

    // Ensure dropping a decoder that never decoded anything still cleans up properly
    let decoder = Decoder::with_settings(Settings {
        threads: 4,
        ..Default::default()
    })
    .unwrap();

    // Drop immediately without decoding
    drop(decoder);

    // If this doesn't hang, the test passes
}

#[test]
fn test_multiple_decoders_simultaneous() {
    let _lock = lock();

    // Test that multiple decoders can coexist without interfering
    let decoder1 = Decoder::with_settings(Settings {
        threads: 2,
        ..Default::default()
    })
    .unwrap();

    let decoder2 = Decoder::with_settings(Settings {
        threads: 2,
        ..Default::default()
    })
    .unwrap();

    drop(decoder1);
    drop(decoder2);

    // If this doesn't hang or crash, the test passes
}
