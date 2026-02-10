use rav1d_safe::src::managed::{Decoder, Planes};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

mod ivf_parser;

fn test_vectors_dir() -> PathBuf {
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    PathBuf::from(target_dir).join("test-vectors")
}

fn find_test_ivf() -> Option<PathBuf> {
    let dav1d_data = test_vectors_dir().join("dav1d-test-data");

    // Look for small IVF test files
    let candidates = [
        "10-bit/film_grain/clip_0.ivf",
        "10-bit/film_grain/clip_1.ivf",
        "8-bit/hdr/hdr10plus_metadata.ivf",
    ];

    for candidate in &candidates {
        let path = dav1d_data.join(candidate);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

#[test]
#[ignore] // Only run when test vectors are available
fn test_decode_real_bitstream() {
    let vector_path = match find_test_ivf() {
        Some(path) => path,
        None => {
            eprintln!("No test vectors found. Run: bash scripts/download-test-vectors.sh");
            return;
        }
    };

    eprintln!("Testing with: {}", vector_path.display());

    let file = File::open(&vector_path).expect("Failed to open test vector");
    let mut reader = BufReader::new(file);

    // Parse IVF file to extract raw OBU frames
    let frames = ivf_parser::parse_all_frames(&mut reader).expect("Failed to parse IVF file");

    eprintln!("IVF file contains {} frames", frames.len());

    let mut decoder = Decoder::new().expect("Failed to create decoder");
    let mut frames_decoded = 0;

    // Feed each frame's OBU data to the decoder
    for (i, ivf_frame) in frames.iter().enumerate() {
        eprintln!(
            "Processing IVF frame {} ({} bytes)",
            i,
            ivf_frame.data.len()
        );

        match decoder.decode(&ivf_frame.data) {
            Ok(Some(frame)) => {
                frames_decoded += 1;
                eprintln!(
                    "  Decoded frame {}: {}x{} @ {}-bit",
                    frames_decoded,
                    frame.width(),
                    frame.height(),
                    frame.bit_depth()
                );

                // Verify we can access pixel data
                match frame.planes() {
                    Planes::Depth8(planes) => {
                        let y = planes.y();
                        assert!(y.width() > 0);
                        assert!(y.height() > 0);
                        assert!(y.stride() >= y.width());

                        // Read first pixel
                        let _first_pixel = y.pixel(0, 0);
                    }
                    Planes::Depth16(planes) => {
                        let y = planes.y();
                        assert!(y.width() > 0);
                        assert!(y.height() > 0);
                        assert!(y.stride() >= y.width());

                        // Read first pixel
                        let _first_pixel = y.pixel(0, 0);
                    }
                }
            }
            Ok(None) => {
                eprintln!("  Need more data for frame {}", i);
            }
            Err(e) => {
                eprintln!("  Decode error on frame {}: {}", i, e);
            }
        }
    }

    // Flush any remaining frames
    match decoder.flush() {
        Ok(remaining) => {
            frames_decoded += remaining.len();
            eprintln!("Flushed {} additional frames", remaining.len());
        }
        Err(e) => {
            eprintln!("Flush error: {}", e);
        }
    }

    eprintln!("Total frames decoded: {}", frames_decoded);
    assert!(frames_decoded > 0, "Should have decoded at least one frame");
}

#[test]
#[ignore]
fn test_decode_hdr_metadata() {
    let dav1d_data = test_vectors_dir().join("dav1d-test-data");

    // Look for HDR test vectors
    let hdr_candidates = [
        "10-bit/film_grain/clip_0.ivf",
        "8-bit/hdr/hdr10plus_metadata.ivf",
    ];

    for candidate in &hdr_candidates {
        let path = dav1d_data.join(candidate);
        if !path.exists() {
            continue;
        }

        eprintln!("Testing HDR with: {}", path.display());

        let file = File::open(&path).expect("Failed to open test vector");
        let mut reader = BufReader::new(file);

        let frames = ivf_parser::parse_all_frames(&mut reader).expect("Failed to parse IVF file");

        let mut decoder = Decoder::new().expect("Failed to create decoder");

        // Decode first frame
        for ivf_frame in &frames {
            if let Ok(Some(frame)) = decoder.decode(&ivf_frame.data) {
                // Check color info
                let color = frame.color_info();
                eprintln!("  Primaries: {:?}", color.primaries);
                eprintln!("  Transfer: {:?}", color.transfer_characteristics);
                eprintln!("  Matrix: {:?}", color.matrix_coefficients);
                eprintln!("  Range: {:?}", color.color_range);

                // Check HDR metadata if present
                if let Some(cll) = frame.content_light() {
                    eprintln!("  Max CLL: {} nits", cll.max_content_light_level);
                    eprintln!("  Max FALL: {} nits", cll.max_frame_average_light_level);
                }

                if let Some(md) = frame.mastering_display() {
                    eprintln!("  Max luminance: {:.2} nits", md.max_luminance_nits());
                    eprintln!("  Min luminance: {:.4} nits", md.min_luminance_nits());
                }

                break;
            }
        }

        return;
    }

    eprintln!("No HDR test vectors found");
}

/// Decode all frames from an IVF file, asserting no panics and at least one frame produced.
fn decode_ivf_file(path: &std::path::Path) {
    let file =
        File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let mut reader = BufReader::new(file);
    let frames = ivf_parser::parse_all_frames(&mut reader)
        .unwrap_or_else(|e| panic!("Failed to parse IVF {}: {e}", path.display()));

    let mut decoder = Decoder::new().expect("Failed to create decoder");
    let mut decoded = 0usize;

    for ivf_frame in &frames {
        match decoder.decode(&ivf_frame.data) {
            Ok(Some(_frame)) => decoded += 1,
            Ok(None) => {}
            Err(_) => {}
        }
    }

    if let Ok(remaining) = decoder.flush() {
        decoded += remaining.len();
    }

    assert!(decoded > 0, "No frames decoded from {}", path.display());
}

/// Regression test: 00000315.ivf triggered a panic in blend_v (OBMC)
/// before the mask axis and 3/4 reduction fix.
#[test]
#[ignore] // requires test vectors
fn test_obmc_blend_v_regression_00000315() {
    let path = test_vectors_dir().join("dav1d-test-data/8-bit/data/00000315.ivf");
    if !path.exists() {
        eprintln!("Skipping: test vector not found at {}", path.display());
        return;
    }
    decode_ivf_file(&path);
}

/// Regression test: 00000327.ivf triggered a panic in blend_h (OBMC)
/// before the mask axis and 3/4 reduction fix.
#[test]
#[ignore] // requires test vectors
fn test_obmc_blend_h_regression_00000327() {
    let path = test_vectors_dir().join("dav1d-test-data/8-bit/data/00000327.ivf");
    if !path.exists() {
        eprintln!("Skipping: test vector not found at {}", path.display());
        return;
    }
    decode_ivf_file(&path);
}

/// Decode every IVF test vector under `max_bytes` in `subdir`.
/// Catches regressions across the dav1d test suite.
fn sweep_vectors(subdir: &str, max_bytes: u64) {
    let dir = test_vectors_dir().join(subdir);
    if !dir.exists() {
        eprintln!("Skipping: vectors not found at {}", dir.display());
        return;
    }

    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("Failed to read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "ivf"))
        .filter(|e| e.metadata().map(|m| m.len() <= max_bytes).unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut passed = 0;
    let mut failed = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        // Catch panics so one bad vector doesn't abort the whole test
        let path_clone = path.clone();
        let result = std::panic::catch_unwind(|| {
            decode_ivf_file(&path_clone);
        });

        match result {
            Ok(()) => passed += 1,
            Err(_) => {
                eprintln!("FAILED: {name}");
                failed.push(name);
            }
        }
    }

    eprintln!(
        "{subdir}: {passed}/{} vectors decoded successfully",
        entries.len()
    );
    assert!(
        failed.is_empty(),
        "{subdir}: these vectors failed to decode: {failed:?}"
    );
}

/// Decode every 8-bit IVF test vector under 100KB.
#[test]
#[ignore] // requires test vectors, slow
fn test_decode_all_8bit_vectors() {
    sweep_vectors("dav1d-test-data/8-bit/data", 100_000);
}

/// Decode every 10-bit IVF test vector under 100KB.
#[test]
#[ignore] // requires test vectors, slow
fn test_decode_all_10bit_vectors() {
    sweep_vectors("dav1d-test-data/10-bit/data", 100_000);
}

/// Decode every 12-bit IVF test vector under 100KB.
#[test]
#[ignore] // requires test vectors, slow
fn test_decode_all_12bit_vectors() {
    sweep_vectors("dav1d-test-data/12-bit/data", 100_000);
}

/// Decode ALL IVF test vectors across every subdirectory (under 100KB each).
#[test]
#[ignore] // requires test vectors, very slow
fn test_decode_all_vectors_comprehensive() {
    let subdirs = [
        "dav1d-test-data/8-bit/data",
        "dav1d-test-data/8-bit/cdfupdate",
        "dav1d-test-data/8-bit/features",
        "dav1d-test-data/8-bit/film_grain",
        "dav1d-test-data/8-bit/issues",
        "dav1d-test-data/8-bit/mfmv",
        "dav1d-test-data/8-bit/mv",
        "dav1d-test-data/8-bit/quantizer",
        "dav1d-test-data/8-bit/resize",
        "dav1d-test-data/8-bit/size",
        "dav1d-test-data/8-bit/svc",
        "dav1d-test-data/10-bit/data",
        "dav1d-test-data/10-bit/features",
        "dav1d-test-data/10-bit/film_grain",
        "dav1d-test-data/10-bit/quantizer",
        "dav1d-test-data/12-bit/data",
        "dav1d-test-data/12-bit/features",
    ];
    for subdir in &subdirs {
        sweep_vectors(subdir, 100_000);
    }
}

#[test]
fn test_ivf_parser() {
    // Test IVF parser with a real file if available
    if let Some(path) = find_test_ivf() {
        let file = File::open(&path).expect("Failed to open test vector");
        let mut reader = BufReader::new(file);

        let header = ivf_parser::parse_ivf_header(&mut reader).expect("Failed to parse IVF header");

        eprintln!("IVF header: {:?}", header);
        assert!(header.width > 0);
        assert!(header.height > 0);
    }
}
