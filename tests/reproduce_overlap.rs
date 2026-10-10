//! Reproducer for DisjointMut overlap panic with frame threading.
//! Run: cargo test --release -p rav1d-safe --test reproduce_overlap -- --ignored --nocapture

use rav1d_safe::{Decoder, Settings};
use std::fs::File;
use std::io::BufReader;

mod ivf_parser;

fn decode_with_threads(path: &str, threads: u32, max_frame_delay: u32) -> (usize, usize) {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Test vector not found: {}", path);
            return (0, 0);
        }
    };

    let mut reader = BufReader::new(file);
    let frames = ivf_parser::parse_all_frames(&mut reader).expect("Failed to parse IVF");
    let total = frames.len();
    eprintln!(
        "Decoding {} frames with threads={}, max_frame_delay={}",
        total, threads, max_frame_delay
    );

    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = max_frame_delay;
    let mut decoder = Decoder::with_settings(settings).expect("Failed to create decoder");
    let mut decoded = 0;

    for (i, frame) in frames.iter().enumerate() {
        match decoder.decode(&frame.data) {
            Ok(Some(f)) => {
                decoded += 1;
                if decoded <= 3 || decoded == total {
                    eprintln!("  Frame {}: decoded {}x{}", i, f.width(), f.height());
                }
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("  Frame {}: error {:?}", i, e);
            }
        }
    }

    // Drain buffered frames — frame threading needs multiple drain calls.
    // Feed empty data to pump the decoder, then flush.
    eprintln!("  Buffered — draining...");
    for _ in 0..total + 16 {
        match decoder.decode(&[]) {
            Ok(Some(f)) => {
                decoded += 1;
                eprintln!("  Drain: decoded {}x{}", f.width(), f.height());
            }
            Ok(None) => {}
            Err(_) => break,
        }
    }
    match decoder.flush() {
        Ok(flushed) => {
            decoded += flushed.len();
            if !flushed.is_empty() {
                eprintln!("  Flushed {} frames", flushed.len());
            }
        }
        Err(e) => {
            eprintln!("  Flush error: {:?}", e);
        }
    }

    eprintln!("  Result: {}/{} decoded", decoded, total);
    (decoded, total)
}

/// Test frame threading with allintra content (all keyframes, no inter-frame deps).
#[test]
#[ignore]
fn test_frame_threading_allintra() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-vectors/dav1d-test-data/8-bit/intra/av1-1-b8-02-allintra.ivf"
    );

    // Frame threading enabled: threads=8, auto frame delay (n_fc = min(sqrt(8), 8) = 3)
    let (decoded, total) = decode_with_threads(path, 8, 0);
    assert!(decoded > 0, "Should decode at least some frames");
    eprintln!("allintra: {}/{}", decoded, total);
}

/// Test frame threading with inter-frame content (P/B frames, motion compensation).
/// This exercises CDEF/loopfilter on frames that reference other frames.
#[test]
#[ignore]
fn test_frame_threading_inter() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-vectors/dav1d-test-data/10-bit/quantizer/av1-1-b10-00-quantizer-00.ivf"
    );

    // Tile threading with 4 threads (max_frame_delay=1 prevents frame-level parallelism)
    let (decoded, total) = decode_with_threads(path, 4, 1);
    assert!(
        decoded > 0,
        "Should decode at least some frames with tile threading"
    );
    eprintln!("inter tile: {}/{}", decoded, total);
}

/// Test true frame threading (max_frame_delay=0 = auto) with inter content.
#[test]
#[ignore]
fn test_true_frame_threading() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-vectors/dav1d-test-data/10-bit/quantizer/av1-1-b10-00-quantizer-00.ivf"
    );

    let (decoded, total) = decode_with_threads(path, 4, 0);
    eprintln!("frame threading inter: {}/{}", decoded, total);
    assert!(
        decoded > 0,
        "Should decode at least some frames with frame threading"
    );
}

/// Stress test: many threads, many different test vectors.
#[test]
#[ignore]
fn test_frame_threading_stress() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/test-vectors/dav1d-test-data/");

    let vectors = [
        "8-bit/intra/av1-1-b8-02-allintra.ivf",
        "10-bit/quantizer/av1-1-b10-00-quantizer-00.ivf",
        "10-bit/quantizer/av1-1-b10-00-quantizer-03.ivf",
    ];

    for threads in [2, 4, 8, 16] {
        for vec in &vectors {
            let path = format!("{}{}", base, vec);
            eprintln!("\n=== threads={}, vector={} ===", threads, vec);
            let (decoded, _total) = decode_with_threads(&path, threads, 0);
            assert!(
                decoded > 0,
                "Failed with threads={}, vector={}",
                threads,
                vec
            );
        }
    }
}

/// Test single-frame OBU decode with frame threading enabled.
/// AVIF decoding scenario: one frame, multiple frame contexts.
#[test]
#[ignore]
fn test_single_obu_frame_threading() {
    let obu_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/crash_vectors/kodim03_yuv420_8bpc.obu"
    );

    let obu = match std::fs::read(obu_path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("OBU not found: {}", obu_path);
            return;
        }
    };
    eprintln!("OBU size: {} bytes", obu.len());

    // Aggressive frame threading: 16 threads, auto frame delay (n_fc = 4)
    for threads in [2, 4, 8, 16] {
        eprintln!("\n--- threads={}, max_frame_delay=0 ---", threads);
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 0;
        let mut decoder = Decoder::with_settings(settings).expect("Failed to create decoder");

        match decoder.decode(&obu) {
            Ok(Some(f)) => eprintln!(
                "  Decoded: {}x{} @ {}bpc",
                f.width(),
                f.height(),
                f.bit_depth()
            ),
            Ok(None) => {
                eprintln!("  Buffered — flushing...");
                match decoder.flush() {
                    Ok(frames) => eprintln!("  Flushed {} frames", frames.len()),
                    Err(e) => eprintln!("  Flush error: {:?}", e),
                }
            }
            Err(e) => eprintln!("  Error: {:?}", e),
        }
    }
}

/// Verify max_frame_delay=1 actually works for single-frame decode.
#[test]
#[ignore]
fn test_single_obu_tile_threading() {
    let obu_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/crash_vectors/kodim03_yuv420_8bpc.obu"
    );

    let obu = match std::fs::read(obu_path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("OBU not found: {}", obu_path);
            return;
        }
    };
    eprintln!("OBU size: {} bytes", obu.len());

    // Tile threading: multiple threads, single frame context
    for threads in [0, 2, 4, 8] {
        eprintln!("\n--- threads={}, max_frame_delay=1 ---", threads);
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 1;
        let mut decoder = Decoder::with_settings(settings).expect("Failed to create decoder");

        match decoder.decode(&obu) {
            Ok(Some(f)) => {
                eprintln!(
                    "  Decoded: {}x{} @ {}bpc",
                    f.width(),
                    f.height(),
                    f.bit_depth()
                );
                assert!(f.width() > 0);
            }
            Ok(None) => {
                eprintln!("  Buffered — flushing...");
                match decoder.flush() {
                    Ok(frames) => {
                        eprintln!("  Flushed {} frames", frames.len());
                        assert!(!frames.is_empty(), "Frame lost with threads={}", threads);
                    }
                    Err(e) => panic!("  Flush error with threads={}: {:?}", threads, e),
                }
            }
            Err(e) => panic!("  Decode error with threads={}: {:?}", threads, e),
        }
    }
}
