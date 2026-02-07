use rav1d_safe::src::managed::{Decoder, Planes};
use std::fs;
use std::path::PathBuf;

fn test_vectors_dir() -> PathBuf {
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| "target".to_string());
    PathBuf::from(target_dir).join("test-vectors")
}

fn find_small_test_vector() -> Option<PathBuf> {
    let dav1d_data = test_vectors_dir().join("dav1d-test-data");

    // Try to find a small OBU test vector (raw OBU data, no container)
    let candidates = [
        "10-bit/argon/test185_302.obu",
        "10-bit/argon/test5606.obu",
        "12-bit/argon/test16153.obu",
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
    let vector_path = match find_small_test_vector() {
        Some(path) => path,
        None => {
            eprintln!("No test vectors found. Run: bash scripts/download-test-vectors.sh");
            return;
        }
    };

    eprintln!("Testing with: {}", vector_path.display());

    let data = fs::read(&vector_path).expect("Failed to read test vector");

    // OBU format contains raw AV1 bitstream data
    let mut decoder = Decoder::new().expect("Failed to create decoder");

    // Feed the data in chunks to test streaming
    const CHUNK_SIZE: usize = 4096;
    let mut frames_decoded = 0;

    for chunk in data.chunks(CHUNK_SIZE) {
        match decoder.decode(chunk) {
            Ok(Some(frame)) => {
                frames_decoded += 1;
                eprintln!("Frame {}: {}x{} @ {}-bit",
                         frames_decoded,
                         frame.width(),
                         frame.height(),
                         frame.bit_depth());

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
                // Need more data
            }
            Err(e) => {
                eprintln!("Decode error: {}", e);
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

    // Try to find an HDR test vector (OBU format)
    let hdr_candidates = [
        "10-bit/argon/test185_302.obu",
        "12-bit/argon/test16153.obu",
    ];

    for candidate in &hdr_candidates {
        let path = dav1d_data.join(candidate);
        if !path.exists() {
            continue;
        }

        eprintln!("Testing HDR with: {}", path.display());

        let data = fs::read(&path).expect("Failed to read test vector");
        let mut decoder = Decoder::new().expect("Failed to create decoder");

        // Decode first frame
        for chunk in data.chunks(8192) {
            if let Ok(Some(frame)) = decoder.decode(chunk) {
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

#[test]
fn test_row_iteration() {
    // This test doesn't need real bitstream - just verify the API compiles
    // Real test would use a decoded frame
}
