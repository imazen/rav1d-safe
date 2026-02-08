//! Thread cleanup tests - verify worker threads are properly joined
//!
//! These tests ensure that when a multi-threaded decoder is dropped,
//! all worker threads are properly joined and don't leak.
//!
//! IMPORTANT: Run these tests serially to avoid thread count interference:
//! `cargo test --test thread_cleanup_test -- --test-threads=1`

use rav1d_safe::src::managed::{Decoder, Settings};
use std::time::Duration;

/// Count threads in the current process (Linux only)
#[cfg(target_os = "linux")]
fn count_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("Failed to read /proc/self/task")
        .count()
}

#[test]
#[cfg(target_os = "linux")]
fn test_single_threaded_no_leak() {
    // Single-threaded decoder should not spawn workers
    let baseline = count_threads();
    
    {
        let mut decoder = Decoder::new().unwrap();
        let _ = decoder.decode(&[]);
        
        // Should still be at baseline (no workers spawned)
        let with_decoder = count_threads();
        assert_eq!(baseline, with_decoder, 
            "Single-threaded decoder should not spawn workers");
    }
    
    // After drop, still at baseline
    let after_drop = count_threads();
    assert_eq!(baseline, after_drop);
}

#[test]
#[cfg(target_os = "linux")]
fn test_multi_threaded_cleanup() {
    let baseline = count_threads();
    
    {
        let mut decoder = Decoder::with_settings(Settings {
            threads: 4,  // Spawn 4 worker threads
            ..Default::default()
        }).unwrap();
        
        let _ = decoder.decode(&[]);
        
        // Should have spawned workers
        let with_workers = count_threads();
        assert!(with_workers >= baseline + 4, 
            "Expected at least {} threads (baseline {} + 4 workers), got {}",
            baseline + 4, baseline, with_workers);
    }
    
    // Give threads a moment to exit (they should join synchronously, but allow OS scheduling)
    std::thread::sleep(Duration::from_millis(50));
    
    let after_drop = count_threads();
    
    // Critical test: threads should be joined, not leaked
    assert_eq!(baseline, after_drop,
        "Worker threads leaked! Baseline: {}, After drop: {}", 
        baseline, after_drop);
}

#[test]
#[cfg(target_os = "linux")]
fn test_auto_detect_threads_cleanup() {
    let baseline = count_threads();
    
    {
        let mut decoder = Decoder::with_settings(Settings {
            threads: 0,  // Auto-detect (will spawn workers)
            ..Default::default()
        }).unwrap();
        
        let _ = decoder.decode(&[]);
        
        // Should have spawned workers
        let with_workers = count_threads();
        assert!(with_workers > baseline, 
            "Auto-detect should spawn worker threads");
    }
    
    std::thread::sleep(Duration::from_millis(50));
    
    let after_drop = count_threads();
    assert_eq!(baseline, after_drop,
        "Worker threads leaked with auto-detect! Baseline: {}, After drop: {}", 
        baseline, after_drop);
}

#[test]
#[cfg(target_os = "linux")]
fn test_multiple_decoder_cycles() {
    // Create and drop multiple decoders to ensure no accumulation of leaked threads
    let baseline = count_threads();
    
    for _ in 0..5 {
        let mut decoder = Decoder::with_settings(Settings {
            threads: 2,
            ..Default::default()
        }).unwrap();
        
        let _ = decoder.decode(&[]);
        drop(decoder);
        
        std::thread::sleep(Duration::from_millis(20));
    }
    
    let final_count = count_threads();
    assert_eq!(baseline, final_count,
        "Threads accumulated after multiple decoder cycles! Baseline: {}, Final: {}", 
        baseline, final_count);
}

#[test]
fn test_drop_without_decode() {
    // Ensure dropping a decoder that never decoded anything still cleans up properly
    let decoder = Decoder::with_settings(Settings {
        threads: 4,
        ..Default::default()
    }).unwrap();
    
    // Drop immediately without decoding
    drop(decoder);
    
    // If this doesn't hang, the test passes
}

#[test]
fn test_multiple_decoders_simultaneous() {
    // Test that multiple decoders can coexist without interfering
    let decoder1 = Decoder::with_settings(Settings {
        threads: 2,
        ..Default::default()
    }).unwrap();
    
    let decoder2 = Decoder::with_settings(Settings {
        threads: 2,
        ..Default::default()
    }).unwrap();
    
    drop(decoder1);
    drop(decoder2);
    
    // If this doesn't hang or crash, the test passes
}
