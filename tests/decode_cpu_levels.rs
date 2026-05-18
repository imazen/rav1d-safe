//! Test decode correctness across all CPU feature levels.
//!
//! For each platform-relevant level (scalar, x86-64-v2, v3, v4, etc.),
//! decode the same test vectors and verify that MD5 hashes match the
//! dav1d reference values.
//!
//! Since `rav1d_set_cpu_flags_mask` is global state, all levels are tested
//! sequentially within each test function. Do NOT split levels into separate
//! `#[test]` functions — they would race in parallel test execution.
//!
//! **Requires `--release`** — debug mode is 50-100x slower and will time out.

#[cfg(debug_assertions)]
compile_error!("decode_cpu_levels tests require release mode: cargo test --release");

use rav1d_safe::src::managed::{CpuLevel, Decoder, DecodeFrameType, Frame, Planes, Settings};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

mod ivf_parser;
mod test_vectors;

fn dav1d_test_data() -> PathBuf {
    test_vectors::ensure_dav1d_test_data()
}

/// Decoder-flag overrides parsed from a standalone `test()` call in a
/// dav1d meson.build (e.g., `--filmgrain 1`, `--oppoint N`, `--alllayers N`,
/// `--decodeframetype intra|reference|key`). `None` ⇒ use defaults.
#[derive(Clone, Default)]
struct VectorArgs {
    apply_grain: Option<bool>,
    operating_point: Option<u8>,
    all_layers: Option<bool>,
    decode_frame_type: Option<DecodeFrameType>,
}

/// A test vector with expected MD5.
struct TestVector {
    name: String,
    ivf_path: PathBuf,
    expected_md5: String,
    /// Set by standalone `test('name', dav1d, args: [...])` entries that
    /// need non-default decoder flags. `None` for `tests += [...]` entries.
    args: Option<VectorArgs>,
}

/// Parse meson.build for test vector entries.
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
            args: None,
        });
    }

    // Also parse standalone `test('name', dav1d, suite: [...], args: [...])`
    // calls. These carry decoder-flag overrides via the args array (e.g.,
    // `--filmgrain`, `--oppoint`, `--alllayers`, `--decodeframetype`) and
    // an expected MD5 via `--verify`. Used by vq_suite to exercise modes
    // the array-add form cannot express.
    vectors.extend(parse_standalone_tests(&content, dir));

    vectors
}

/// Parse `test('name', dav1d, suite: [...], args: dav1d_test_args + [...])`
/// invocations. Multi-line: collect from `test(` through the matching `)`.
fn parse_standalone_tests(content: &str, dir: &Path) -> Vec<TestVector> {
    let mut out = Vec::new();
    let mut iter = content.lines();
    let mut buf = String::new();
    let mut depth: i32 = 0;
    let mut active = false;
    while let Some(line) = iter.next() {
        let trimmed = line.trim();
        if !active {
            if trimmed.starts_with("test(") {
                buf.clear();
                buf.push_str(trimmed);
                depth = 1;
                // Account for any closing parens on the same line.
                for ch in trimmed[5..].chars() {
                    match ch {
                        '(' => depth += 1,
                        ')' => depth -= 1,
                        _ => {}
                    }
                }
                if depth == 0 {
                    if let Some(v) = parse_one_standalone_test(&buf, dir) {
                        out.push(v);
                    }
                    buf.clear();
                } else {
                    active = true;
                }
            }
        } else {
            buf.push(' ');
            buf.push_str(trimmed);
            for ch in trimmed.chars() {
                match ch {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
            }
            if depth == 0 {
                if let Some(v) = parse_one_standalone_test(&buf, dir) {
                    out.push(v);
                }
                buf.clear();
                active = false;
            }
        }
    }
    out
}

fn parse_one_standalone_test(entry: &str, dir: &Path) -> Option<TestVector> {
    // Extract every quoted string in order — preserves the args array
    // structure for flag/value pairs.
    let mut quoted = Vec::new();
    let mut chars = entry.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' {
            let s: String = chars.by_ref().take_while(|&c| c != '\'').collect();
            quoted.push(s);
        }
    }
    if quoted.is_empty() {
        return None;
    }
    let name = quoted[0].clone();
    let filename = quoted
        .iter()
        .find(|s| s.ends_with(".ivf") || s.ends_with(".obu"))?
        .clone();
    let expected_md5 = quoted
        .iter()
        .rev()
        .find(|s| s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()))?
        .clone();

    let mut args = VectorArgs::default();
    let mut i = 0;
    while i + 1 < quoted.len() {
        match quoted[i].as_str() {
            "--filmgrain" => {
                args.apply_grain = Some(quoted[i + 1] != "0");
                i += 2;
            }
            "--oppoint" => {
                args.operating_point = quoted[i + 1].parse().ok();
                i += 2;
            }
            "--alllayers" => {
                args.all_layers = Some(quoted[i + 1] != "0");
                i += 2;
            }
            "--decodeframetype" => {
                args.decode_frame_type = match quoted[i + 1].as_str() {
                    "all" => Some(DecodeFrameType::All),
                    "reference" => Some(DecodeFrameType::Reference),
                    "intra" => Some(DecodeFrameType::Intra),
                    "key" => Some(DecodeFrameType::Key),
                    _ => None,
                };
                i += 2;
            }
            _ => i += 1,
        }
    }

    Some(TestVector {
        name,
        ivf_path: dir.join(filename),
        expected_md5,
        args: Some(args),
    })
}

/// Hash a decoded frame into MD5 context (matches dav1d's md5_write).
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

/// Decode an IVF file at a specific CPU level, return MD5 and frame count.
fn decode_at_level(
    ivf_path: &Path,
    level: CpuLevel,
    apply_grain: bool,
    overrides: Option<&VectorArgs>,
) -> Result<(String, usize), String> {
    let file = File::open(ivf_path).map_err(|e| format!("open {}: {e}", ivf_path.display()))?;
    let mut reader = BufReader::new(file);
    let frames = ivf_parser::parse_all_frames(&mut reader)
        .map_err(|e| format!("parse IVF {}: {e}", ivf_path.display()))?;

    let mut settings = Settings::default();
    settings.cpu_level = level;
    settings.apply_grain = apply_grain;
    if let Some(args) = overrides {
        if let Some(v) = args.apply_grain {
            settings.apply_grain = v;
        }
        if let Some(v) = args.operating_point {
            settings.operating_point = v;
        }
        if let Some(v) = args.all_layers {
            settings.all_layers = v;
        }
        if let Some(v) = args.decode_frame_type {
            settings.decode_frame_type = v;
        }
    }
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

/// Run a meson.build's vectors at a given CPU level.
/// Returns (passed, failed_names, skipped).
fn verify_at_level(
    meson_path: &Path,
    level: CpuLevel,
    apply_grain: bool,
) -> (usize, Vec<String>, usize) {
    let vectors = parse_meson_build(meson_path);
    let mut passed = 0;
    let mut failed = Vec::new();
    let mut skipped = 0;

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

        match decode_at_level(&vector.ivf_path, level, apply_grain, vector.args.as_ref()) {
            Ok((actual_md5, _frame_count)) => {
                if actual_md5 == vector.expected_md5 {
                    passed += 1;
                } else {
                    eprintln!(
                        "  MISMATCH [{}]: {} expected={} actual={}",
                        level, vector.name, vector.expected_md5, actual_md5
                    );
                    failed.push(vector.name.clone());
                }
            }
            Err(e) => {
                eprintln!("  ERROR [{}]: {}: {e}", level, vector.name);
                failed.push(format!("{} (error)", vector.name));
            }
        }
    }

    (passed, failed, skipped)
}

// ============================================================================
// Tests
// ============================================================================

/// Verify 8-bit/data vectors at ALL platform CPU levels.
///
/// This is the most important correctness test: every CPU level must produce
/// bit-identical output (matching the dav1d reference MD5).
#[test]
fn test_cpu_levels_8bit_data() {
    let meson = dav1d_test_data().join("8-bit/data/meson.build");

    let levels = CpuLevel::platform_levels();
    let mut all_failures = Vec::new();

    for &level in levels {
        let (passed, failed, skipped) = verify_at_level(&meson, level, false);
        eprintln!(
            "[{}] 8-bit/data: {passed} passed, {} failed, {skipped} skipped",
            level,
            failed.len()
        );
        for f in &failed {
            all_failures.push(format!("[{}] {f}", level));
        }
    }

    assert!(
        all_failures.is_empty(),
        "CPU level parity failures:\n{}",
        all_failures.join("\n")
    );
}

/// Verify 10-bit/data vectors at all platform CPU levels.
#[test]
fn test_cpu_levels_10bit_data() {
    let meson = dav1d_test_data().join("10-bit/data/meson.build");

    let levels = CpuLevel::platform_levels();
    let mut all_failures = Vec::new();

    for &level in levels {
        let (passed, failed, skipped) = verify_at_level(&meson, level, false);
        eprintln!(
            "[{}] 10-bit/data: {passed} passed, {} failed, {skipped} skipped",
            level,
            failed.len()
        );
        for f in &failed {
            all_failures.push(format!("[{}] {f}", level));
        }
    }

    assert!(
        all_failures.is_empty(),
        "CPU level parity failures:\n{}",
        all_failures.join("\n")
    );
}

/// Verify 12-bit/data vectors at all platform CPU levels.
#[test]
fn test_cpu_levels_12bit_data() {
    let meson = dav1d_test_data().join("12-bit/data/meson.build");

    let levels = CpuLevel::platform_levels();
    let mut all_failures = Vec::new();

    for &level in levels {
        let (passed, failed, skipped) = verify_at_level(&meson, level, false);
        eprintln!(
            "[{}] 12-bit/data: {passed} passed, {} failed, {skipped} skipped",
            level,
            failed.len()
        );
        for f in &failed {
            all_failures.push(format!("[{}] {f}", level));
        }
    }

    assert!(
        all_failures.is_empty(),
        "CPU level parity failures:\n{}",
        all_failures.join("\n")
    );
}

/// Quick smoke test: decode one 8-bit vector at every level.
/// Faster than the full suite, good for rapid iteration.
#[test]
fn test_cpu_levels_smoke() {
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

    let levels = CpuLevel::platform_levels();
    for &level in levels {
        let (actual_md5, frame_count) = decode_at_level(&first.ivf_path, level, false, None)
            .unwrap_or_else(|e| panic!("[{}] decode failed: {e}", level));
        eprintln!(
            "[{}] {}: {frame_count} frames, md5={actual_md5}",
            level, first.name
        );
        assert_eq!(
            actual_md5, first.expected_md5,
            "[{}] MD5 mismatch for {} — expected {}, got {actual_md5}",
            level, first.name, first.expected_md5
        );
    }
}

/// Comprehensive: all meson.build directories, all CPU levels.
/// This is the full matrix test. Takes a while.
#[test]
fn test_cpu_levels_comprehensive() {
    let base = dav1d_test_data();

    let meson_files: &[(&str, bool)] = &[
        ("8-bit/data/meson.build", false),
        ("8-bit/features/meson.build", false),
        ("8-bit/issues/meson.build", false),
        ("8-bit/quantizer/meson.build", false),
        ("8-bit/size/meson.build", false),
        ("8-bit/cdfupdate/meson.build", false),
        ("8-bit/vq_suite/meson.build", false),
        ("8-bit/intra/meson.build", false),
        ("8-bit/mfmv/meson.build", false),
        ("8-bit/mv/meson.build", false),
        ("8-bit/resize/meson.build", false),
        ("8-bit/film_grain/meson.build", true),
        ("10-bit/data/meson.build", false),
        ("10-bit/features/meson.build", false),
        ("10-bit/quantizer/meson.build", false),
        ("10-bit/issues/meson.build", false),
        ("10-bit/film_grain/meson.build", true),
        ("12-bit/data/meson.build", false),
        ("12-bit/features/meson.build", false),
    ];

    let levels = CpuLevel::platform_levels();
    let mut all_failures = Vec::new();

    for &level in levels {
        let mut level_passed = 0;
        let mut level_failed = 0;
        let mut level_skipped = 0;

        for &(meson_rel, grain) in meson_files {
            let meson_path = base.join(meson_rel);
            assert!(meson_path.exists(), "missing: {}", meson_path.display());

            let (passed, failed, skipped) = verify_at_level(&meson_path, level, grain);
            level_passed += passed;
            level_failed += failed.len();
            level_skipped += skipped;

            for f in &failed {
                all_failures.push(format!("[{}] {meson_rel}: {f}", level));
            }
        }

        eprintln!(
            "[{}] TOTAL: {level_passed} passed, {level_failed} failed, {level_skipped} skipped",
            level
        );
    }

    assert!(
        all_failures.is_empty(),
        "CPU level comprehensive failures ({}):\n{}",
        all_failures.len(),
        all_failures.join("\n")
    );
}

/// Verify the vq_suite vectors that the meson.build expresses via standalone
/// `test()` calls (filmgrain / oppoint / alllayers / decodeframetype). These
/// previously fell through the parser. Smoke test for both the parse + the
/// per-vector Settings dispatch.
#[test]
fn test_vq_suite_standalone() {
    let meson = dav1d_test_data().join("8-bit/vq_suite/meson.build");
    let vectors = parse_meson_build(&meson);
    let standalone: Vec<&TestVector> =
        vectors.iter().filter(|v| v.args.is_some()).collect();

    assert!(
        !standalone.is_empty(),
        "no standalone test() entries parsed from {}",
        meson.display()
    );
    eprintln!(
        "vq_suite: parsed {} standalone test() entries (out of {} total)",
        standalone.len(),
        vectors.len()
    );

    let levels = CpuLevel::platform_levels();
    let mut failures = Vec::new();

    for &level in levels {
        for vector in &standalone {
            if !vector.ivf_path.exists() {
                continue;
            }
            match decode_at_level(&vector.ivf_path, level, false, vector.args.as_ref()) {
                Ok((actual_md5, _)) => {
                    if actual_md5 != vector.expected_md5 {
                        failures.push(format!(
                            "[{level}] {}: expected={} actual={actual_md5}",
                            vector.name, vector.expected_md5
                        ));
                    }
                }
                Err(e) => failures.push(format!("[{level}] {}: {e}", vector.name)),
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "vq_suite standalone failures ({}):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
