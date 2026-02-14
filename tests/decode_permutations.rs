//! E2E decode parity across all archmage token permutations.
//!
//! For each test vector, decode once at full CPU capability to get a reference
//! MD5, then re-decode inside `for_each_token_permutation` — verifying every
//! token combination produces bit-identical output.
//!
//! This exercises every dispatch gate in every SIMD module (itx, mc, cdef,
//! loopfilter, looprestoration, ipred, filmgrain, pal, refmvs) at every
//! tier the CPU supports (e.g. AVX-512, AVX2, SSE4, SSE2, scalar).
//!
//! Token state is process-wide, so this MUST run with `--test-threads=1`.
//!
//! **Requires `--release`** — debug mode is 50-100x slower.

#[cfg(debug_assertions)]
compile_error!("decode_permutations tests require release mode: cargo test --release");

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

mod ivf_parser;
mod test_vectors;

fn dav1d_test_data() -> PathBuf {
    test_vectors::ensure_dav1d_test_data()
}

// ---------------------------------------------------------------------------
// Helpers (shared with decode_md5_verify)
// ---------------------------------------------------------------------------

struct TestVector {
    name: String,
    ivf_path: PathBuf,
    expected_md5: String,
}

fn parse_meson_build(meson_path: &Path) -> Vec<TestVector> {
    let content = match std::fs::read_to_string(meson_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let dir = meson_path.parent().unwrap();
    let mut vectors = Vec::new();

    let mut entries = Vec::new();
    let mut current_entry = String::new();
    let mut in_entry = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if !in_entry {
            if trimmed.starts_with('[') && trimmed.contains('\'') {
                current_entry = trimmed.to_string();
                if trimmed.contains("],") || trimmed.ends_with(']') {
                    entries.push(current_entry.clone());
                    current_entry.clear();
                } else {
                    in_entry = true;
                }
            }
        } else {
            current_entry.push(' ');
            current_entry.push_str(trimmed);
            if trimmed.contains("],") || trimmed.ends_with(']') {
                entries.push(current_entry.clone());
                current_entry.clear();
                in_entry = false;
            }
        }
    }

    for entry in &entries {
        if !entry.contains("files(") {
            continue;
        }

        let mut quoted = Vec::new();
        let mut chars = entry.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\'' {
                let s: String = chars.by_ref().take_while(|&c| c != '\'').collect();
                if !s.is_empty() {
                    quoted.push(s);
                }
            }
        }

        if quoted.len() < 3 {
            continue;
        }

        let name = quoted[0].clone();
        let filename = match quoted
            .iter()
            .find(|s| s.ends_with(".ivf") || s.ends_with(".obu"))
        {
            Some(f) => f.as_str(),
            None => continue,
        };
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

fn hash_frame(frame: &Frame, ctx: &mut md5::Context) {
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                ctx.consume(row);
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    ctx.consume(row);
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    ctx.consume(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            for row in planes.y().rows() {
                for &pixel in row {
                    ctx.consume(pixel.to_le_bytes());
                }
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &pixel in row {
                        ctx.consume(pixel.to_le_bytes());
                    }
                }
            }
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

fn decode_ivf(ivf_path: &Path, apply_grain: bool) -> Result<(String, usize), String> {
    let file = File::open(ivf_path).map_err(|e| format!("open {}: {e}", ivf_path.display()))?;
    let mut reader = BufReader::new(file);
    let frames = ivf_parser::parse_all_frames(&mut reader)
        .map_err(|e| format!("parse IVF {}: {e}", ivf_path.display()))?;

    let settings = Settings {
        apply_grain,
        ..Default::default()
    };
    let mut decoder = Decoder::with_settings(settings).map_err(|e| format!("init decoder: {e}"))?;
    let mut ctx = md5::Context::new();
    let mut frame_count = 0usize;

    for ivf_frame in &frames {
        match decoder.decode(&ivf_frame.data) {
            Ok(Some(frame)) => {
                hash_frame(&frame, &mut ctx);
                frame_count += 1;
            }
            Ok(None) => {}
            Err(e) => {
                return Err(format!(
                    "decode error frame {} of {}: {e}",
                    frame_count,
                    ivf_path.display()
                ));
            }
        }
    }

    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                hash_frame(frame, &mut ctx);
                frame_count += 1;
            }
        }
        Err(e) => return Err(format!("flush {}: {e}", ivf_path.display())),
    }

    let digest = ctx.finalize();
    Ok((format!("{:x}", digest), frame_count))
}

// ---------------------------------------------------------------------------
// Permutation test core
// ---------------------------------------------------------------------------

/// Decode a single vector at every token permutation, verify MD5 parity.
fn verify_permutations(
    ivf_path: &Path,
    name: &str,
    expected_md5: &str,
    apply_grain: bool,
) -> Result<usize, String> {
    use archmage::testing::{for_each_token_permutation, CompileTimePolicy};

    let mut failures: Vec<String> = Vec::new();
    let mut permutations_run = 0;

    let report = for_each_token_permutation(CompileTimePolicy::WarnStderr, |perm| {
        permutations_run += 1;
        match decode_ivf(ivf_path, apply_grain) {
            Ok((md5, _count)) => {
                if md5 != expected_md5 {
                    failures.push(format!(
                        "{name}: md5 mismatch at {perm} — expected {expected_md5}, got {md5}"
                    ));
                }
            }
            Err(e) => {
                failures.push(format!("{name}: decode error at {perm} — {e}"));
            }
        }
    });

    if !failures.is_empty() {
        return Err(format!(
            "{} failures across {} permutations:\n  {}",
            failures.len(),
            report.permutations_run,
            failures.join("\n  ")
        ));
    }

    Ok(report.permutations_run)
}

/// Run permutations for all IVF vectors in a meson.build file.
/// Returns (passed, failed_messages, skipped, total_permutations).
fn verify_meson_permutations(
    meson_path: &Path,
    apply_grain: bool,
) -> (usize, Vec<String>, usize, usize) {
    let vectors = parse_meson_build(meson_path);
    let mut passed = 0;
    let mut failed = Vec::new();
    let mut skipped = 0;
    let mut total_perms = 0;

    for vector in &vectors {
        if !vector.ivf_path.exists() {
            skipped += 1;
            continue;
        }
        let ext = vector
            .ivf_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        if ext != "ivf" {
            skipped += 1;
            continue;
        }

        match verify_permutations(&vector.ivf_path, &vector.name, &vector.expected_md5, apply_grain)
        {
            Ok(perms) => {
                passed += 1;
                total_perms += perms;
            }
            Err(msg) => {
                failed.push(msg);
            }
        }
    }

    (passed, failed, skipped, total_perms)
}

// ---------------------------------------------------------------------------
// Tests — one per meson.build directory
// ---------------------------------------------------------------------------

/// Smoke test: single vector, all permutations.
#[test]
fn test_permutations_smoke() {
    let meson = dav1d_test_data().join("8-bit/data/meson.build");
    let vectors = parse_meson_build(&meson);
    let first = vectors
        .first()
        .expect("no vectors in 8-bit/data/meson.build");
    assert!(
        first.ivf_path.exists(),
        "missing: {}",
        first.ivf_path.display()
    );

    let perms = verify_permutations(
        &first.ivf_path,
        &first.name,
        &first.expected_md5,
        false,
    )
    .unwrap_or_else(|e| panic!("{e}"));

    eprintln!(
        "Smoke: {} — {} permutations, all matched md5={}",
        first.name, perms, first.expected_md5
    );
    assert!(perms >= 2, "expected at least 2 permutations, got {perms}");
}

#[test]
fn test_permutations_8bit_data() {
    let meson = dav1d_test_data().join("8-bit/data/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/data: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_features() {
    let meson = dav1d_test_data().join("8-bit/features/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/features: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_issues() {
    let meson = dav1d_test_data().join("8-bit/issues/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/issues: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_quantizer() {
    let meson = dav1d_test_data().join("8-bit/quantizer/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/quantizer: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_size() {
    let meson = dav1d_test_data().join("8-bit/size/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/size: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_intra() {
    let meson = dav1d_test_data().join("8-bit/intra/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/intra: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_film_grain() {
    let meson = dav1d_test_data().join("8-bit/film_grain/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, true);
    eprintln!("8-bit/film_grain: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_cdfupdate() {
    let meson = dav1d_test_data().join("8-bit/cdfupdate/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/cdfupdate: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_mfmv() {
    let meson = dav1d_test_data().join("8-bit/mfmv/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/mfmv: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_mv() {
    let meson = dav1d_test_data().join("8-bit/mv/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/mv: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_8bit_resize() {
    let meson = dav1d_test_data().join("8-bit/resize/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("8-bit/resize: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_10bit_data() {
    let meson = dav1d_test_data().join("10-bit/data/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("10-bit/data: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_10bit_features() {
    let meson = dav1d_test_data().join("10-bit/features/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("10-bit/features: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_10bit_quantizer() {
    let meson = dav1d_test_data().join("10-bit/quantizer/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("10-bit/quantizer: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_10bit_issues() {
    let meson = dav1d_test_data().join("10-bit/issues/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("10-bit/issues: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_10bit_film_grain() {
    let meson = dav1d_test_data().join("10-bit/film_grain/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, true);
    eprintln!("10-bit/film_grain: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_12bit_data() {
    let meson = dav1d_test_data().join("12-bit/data/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("12-bit/data: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}

#[test]
fn test_permutations_12bit_features() {
    let meson = dav1d_test_data().join("12-bit/features/meson.build");
    let (passed, failed, skipped, perms) = verify_meson_permutations(&meson, false);
    eprintln!("12-bit/features: {passed} passed, {} failed, {skipped} skipped, {perms} total permutations", failed.len());
    assert!(failed.is_empty(), "failures:\n{}", failed.join("\n"));
}
