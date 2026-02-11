//! Pixel-perfect decode verification using MD5 checksums.
//!
//! Parses reference MD5 hashes from dav1d-test-data meson.build files
//! and verifies that rav1d-safe produces bit-identical output.
//!
//! The MD5 is computed identically to dav1d: for each decoded frame,
//! hash Y plane rows (width pixels), then U rows, then V rows.
//! 16-bit pixels are hashed as little-endian bytes.
//! The hash accumulates across ALL frames in the file.

use rav1d_safe::src::managed::{Decoder, Frame, Planes};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

mod ivf_parser;

fn test_vectors_dir() -> PathBuf {
    // Test vectors live at project_root/test-vectors/ (gitignored)
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("test-vectors")
}

fn dav1d_test_data() -> PathBuf {
    test_vectors_dir().join("dav1d-test-data")
}

/// A test vector with its expected MD5 hash.
struct TestVector {
    name: String,
    ivf_path: PathBuf,
    expected_md5: String,
}

/// Parse meson.build to extract test vector entries.
///
/// Looks for lines like:
///   ['00000001', files('00000001.ivf'), '98b8c18a74a27a4fc1436f08b73b9270'],
fn parse_meson_build(meson_path: &Path) -> Vec<TestVector> {
    let content = match std::fs::read_to_string(meson_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let dir = meson_path.parent().unwrap();
    let mut vectors = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        // Match lines like: ['name', files('file.ivf'), 'md5hash'],
        if !trimmed.starts_with("['") && !trimmed.starts_with("[\"") {
            continue;
        }

        // Extract all single-quoted strings from the line
        let mut quoted = Vec::new();
        let mut chars = trimmed.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\'' {
                let s: String = chars.by_ref().take_while(|&c| c != '\'').collect();
                if !s.is_empty() {
                    quoted.push(s);
                }
            }
        }

        // We need at least: name, filename, md5
        if quoted.len() < 3 {
            continue;
        }

        let name = quoted[0].clone();

        // The filename is the quoted string that ends in .ivf or .obu
        let filename = match quoted.iter().find(|s| s.ends_with(".ivf") || s.ends_with(".obu")) {
            Some(f) => f.as_str(),
            None => continue,
        };

        // Find the MD5 hash (last quoted string, 32 hex chars)
        let md5 = quoted
            .iter()
            .rev()
            .find(|s| s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()));

        let md5 = match md5 {
            Some(m) => m.clone(),
            None => continue,
        };

        let ivf_path = dir.join(filename);
        vectors.push(TestVector {
            name,
            ivf_path,
            expected_md5: md5,
        });
    }

    vectors
}

/// Compute dav1d-compatible MD5 over all decoded frames.
///
/// Algorithm: for each frame, feed Y rows then U rows then V rows
/// into the MD5 hasher. For 8-bit, each row is `width` bytes.
/// For 16-bit, each row is `width * 2` bytes in little-endian order.
fn compute_decode_md5(ivf_path: &Path) -> Result<(String, usize), String> {
    let file = File::open(ivf_path)
        .map_err(|e| format!("Failed to open {}: {e}", ivf_path.display()))?;
    let mut reader = BufReader::new(file);

    let frames = ivf_parser::parse_all_frames(&mut reader)
        .map_err(|e| format!("Failed to parse IVF {}: {e}", ivf_path.display()))?;

    let mut decoder = Decoder::new().map_err(|e| format!("Failed to create decoder: {e}"))?;
    let mut ctx = md5::Context::new();
    let mut frame_count = 0usize;

    // Decode each IVF packet
    for ivf_frame in &frames {
        match decoder.decode(&ivf_frame.data) {
            Ok(Some(frame)) => {
                hash_frame(&frame, &mut ctx);
                frame_count += 1;
            }
            Ok(None) => {}
            Err(e) => {
                return Err(format!(
                    "Decode error on frame {} of {}: {e}",
                    frame_count,
                    ivf_path.display()
                ));
            }
        }
    }

    // Flush remaining frames
    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                hash_frame(frame, &mut ctx);
                frame_count += 1;
            }
        }
        Err(e) => {
            return Err(format!("Flush error for {}: {e}", ivf_path.display()));
        }
    }

    let digest = ctx.finalize();
    Ok((format!("{:x}", digest), frame_count))
}

/// Hash a single decoded frame into the MD5 context.
/// Matches dav1d's md5_write() exactly.
fn hash_frame(frame: &Frame, ctx: &mut md5::Context) {
    match frame.planes() {
        Planes::Depth8(planes) => {
            // Y plane
            for row in planes.y().rows() {
                ctx.consume(row);
            }
            // U plane
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    ctx.consume(row);
                }
            }
            // V plane
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    ctx.consume(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            // Y plane - little-endian bytes
            for row in planes.y().rows() {
                for &pixel in row {
                    ctx.consume(pixel.to_le_bytes());
                }
            }
            // U plane
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &pixel in row {
                        ctx.consume(pixel.to_le_bytes());
                    }
                }
            }
            // V plane
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    for &pixel in row {
                        ctx.consume(pixel.to_le_bytes());
                    }
                }
            }
        }
    }
}

/// Collect all test vectors from a meson.build file and verify MD5s.
/// Returns (passed, failed_names, skipped).
fn verify_meson_vectors(meson_path: &Path) -> (usize, Vec<String>, usize) {
    let vectors = parse_meson_build(meson_path);
    let mut passed = 0;
    let mut failed = Vec::new();
    let mut skipped = 0;

    for vector in &vectors {
        if !vector.ivf_path.exists() {
            skipped += 1;
            continue;
        }

        match compute_decode_md5(&vector.ivf_path) {
            Ok((actual_md5, frame_count)) => {
                if actual_md5 == vector.expected_md5 {
                    passed += 1;
                } else {
                    eprintln!(
                        "MISMATCH: {} ({} frames)\n  expected: {}\n  actual:   {}",
                        vector.name, frame_count, vector.expected_md5, actual_md5
                    );
                    failed.push(vector.name.clone());
                }
            }
            Err(e) => {
                eprintln!("ERROR: {}: {e}", vector.name);
                failed.push(format!("{} (error: {e})", vector.name));
            }
        }
    }

    (passed, failed, skipped)
}

// ============================================================================
// Tests
// ============================================================================

/// Verify first vector to smoke-test the infrastructure.
#[test]
fn test_md5_verify_first_8bit() {
    let meson = dav1d_test_data().join("8-bit/data/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found. Run: bash scripts/download-test-vectors.sh");
        return;
    }

    let vectors = parse_meson_build(&meson);
    assert!(!vectors.is_empty(), "No test vectors found in meson.build");

    let first = &vectors[0];
    if !first.ivf_path.exists() {
        eprintln!("Skipping: {} not found", first.ivf_path.display());
        return;
    }

    let (actual_md5, frame_count) = compute_decode_md5(&first.ivf_path)
        .unwrap_or_else(|e| panic!("Failed to decode {}: {e}", first.name));

    eprintln!(
        "{}: {} frames, md5={}",
        first.name, frame_count, actual_md5
    );

    assert_eq!(
        actual_md5, first.expected_md5,
        "MD5 mismatch for {} — expected {}, got {}",
        first.name, first.expected_md5, actual_md5
    );
}

/// Verify ALL 8-bit/data test vectors.
#[test]
fn test_md5_verify_all_8bit_data() {
    let meson = dav1d_test_data().join("8-bit/data/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("8-bit/data: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "8-bit/data: {} vectors failed MD5 verification: {:?}",
        failed.len(),
        failed
    );
}

/// Verify ALL 10-bit/data test vectors.
#[test]
fn test_md5_verify_all_10bit_data() {
    let meson = dav1d_test_data().join("10-bit/data/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("10-bit/data: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "10-bit/data: {} vectors failed MD5 verification: {:?}",
        failed.len(),
        failed
    );
}

/// Verify ALL 12-bit/data test vectors.
#[test]
fn test_md5_verify_all_12bit_data() {
    let meson = dav1d_test_data().join("12-bit/data/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("12-bit/data: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "12-bit/data: {} vectors failed MD5 verification: {:?}",
        failed.len(),
        failed
    );
}

/// Verify 8-bit feature test vectors.
#[test]
fn test_md5_verify_8bit_features() {
    let meson = dav1d_test_data().join("8-bit/features/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("8-bit/features: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "8-bit/features: {} vectors failed: {:?}",
        failed.len(),
        failed
    );
}

/// Verify 8-bit quantizer test vectors.
#[test]
fn test_md5_verify_8bit_quantizer() {
    let meson = dav1d_test_data().join("8-bit/quantizer/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("8-bit/quantizer: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "8-bit/quantizer: {} vectors failed: {:?}",
        failed.len(),
        failed
    );
}

/// Verify 8-bit size test vectors.
#[test]
fn test_md5_verify_8bit_size() {
    let meson = dav1d_test_data().join("8-bit/size/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("8-bit/size: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "8-bit/size: {} vectors failed: {:?}",
        failed.len(),
        failed
    );
}

/// Verify 8-bit issue regression test vectors.
#[test]
fn test_md5_verify_8bit_issues() {
    let meson = dav1d_test_data().join("8-bit/issues/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("8-bit/issues: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "8-bit/issues: {} vectors failed: {:?}",
        failed.len(),
        failed
    );
}

/// Verify 10-bit quantizer test vectors.
#[test]
fn test_md5_verify_10bit_quantizer() {
    let meson = dav1d_test_data().join("10-bit/quantizer/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("10-bit/quantizer: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "10-bit/quantizer: {} vectors failed: {:?}",
        failed.len(),
        failed
    );
}

/// Verify 8-bit film grain test vectors (with film grain applied).
#[test]
fn test_md5_verify_8bit_film_grain() {
    let meson = dav1d_test_data().join("8-bit/film_grain/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    // Film grain tests use a different variable name (fg_tests) but same format
    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("8-bit/film_grain: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "8-bit/film_grain: {} vectors failed: {:?}",
        failed.len(),
        failed
    );
}

/// Verify 8-bit cdf update test vectors.
#[test]
fn test_md5_verify_8bit_cdfupdate() {
    let meson = dav1d_test_data().join("8-bit/cdfupdate/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("8-bit/cdfupdate: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "8-bit/cdfupdate: {} vectors failed: {:?}",
        failed.len(),
        failed
    );
}

/// Verify 8-bit vq_suite test vectors.
#[test]
fn test_md5_verify_8bit_vq_suite() {
    let meson = dav1d_test_data().join("8-bit/vq_suite/meson.build");
    if !meson.exists() {
        eprintln!("Skipping: test vectors not found");
        return;
    }

    let (passed, failed, skipped) = verify_meson_vectors(&meson);
    eprintln!("8-bit/vq_suite: {passed} passed, {} failed, {skipped} skipped", failed.len());

    assert!(
        failed.is_empty(),
        "8-bit/vq_suite: {} vectors failed: {:?}",
        failed.len(),
        failed
    );
}

/// Comprehensive: verify ALL meson.build files with reference MD5s.
#[test]
fn test_md5_verify_comprehensive() {
    let base = dav1d_test_data();
    if !base.exists() {
        eprintln!("Skipping: dav1d-test-data not found");
        return;
    }

    let meson_files = [
        "8-bit/data/meson.build",
        "8-bit/features/meson.build",
        "8-bit/issues/meson.build",
        "8-bit/quantizer/meson.build",
        "8-bit/size/meson.build",
        "8-bit/cdfupdate/meson.build",
        "8-bit/vq_suite/meson.build",
        "8-bit/intra/meson.build",
        "8-bit/mfmv/meson.build",
        "8-bit/mv/meson.build",
        "8-bit/resize/meson.build",
        "8-bit/film_grain/meson.build",
        "10-bit/data/meson.build",
        "10-bit/features/meson.build",
        "10-bit/quantizer/meson.build",
        "10-bit/issues/meson.build",
        "12-bit/data/meson.build",
        "12-bit/features/meson.build",
    ];

    let mut total_passed = 0;
    let mut total_failed = Vec::new();
    let mut total_skipped = 0;

    for meson_rel in &meson_files {
        let meson_path = base.join(meson_rel);
        if !meson_path.exists() {
            continue;
        }

        let (passed, failed, skipped) = verify_meson_vectors(&meson_path);
        eprintln!("{meson_rel}: {passed} passed, {} failed, {skipped} skipped", failed.len());

        total_passed += passed;
        for f in failed {
            total_failed.push(format!("{meson_rel}: {f}"));
        }
        total_skipped += skipped;
    }

    eprintln!(
        "\nTOTAL: {total_passed} passed, {} failed, {total_skipped} skipped",
        total_failed.len()
    );

    assert!(
        total_failed.is_empty(),
        "Comprehensive MD5 verification: {} vectors failed:\n{}",
        total_failed.len(),
        total_failed.join("\n")
    );
}
