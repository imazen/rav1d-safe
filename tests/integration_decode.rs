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
