//! Decode AV1 bitstreams and compute MD5 of decoded pixel data.
//!
//! Supports IVF, raw OBU (Section 5), and Annex B container formats.
//! Produces MD5 hashes compatible with dav1d's --verify / aomdec --md5 format.
//!
//! Usage:
//!   cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example decode_md5
//!   ./target/release/examples/decode_md5 [--filmgrain] [-q] [--threads N] <input> [expected_md5]

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::env;
use std::fs;
use std::io::Cursor;

#[path = "helpers/annexb_parser.rs"]
mod annexb_parser;
#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

fn hash_frame(frame: &Frame, hasher: &mut md5::Context, verbose: bool) {
    if verbose {
        eprintln!(
            "  Frame: {}x{} bpc={} layout={:?}",
            frame.width(),
            frame.height(),
            frame.bit_depth(),
            frame.pixel_layout()
        );
    }
    match frame.planes() {
        Planes::Depth8(planes) => {
            let y = planes.y();
            if verbose {
                eprintln!(
                    "  Y plane: {}x{} stride={}",
                    y.width(),
                    y.height(),
                    y.stride()
                );
            }
            for row in y.rows() {
                hasher.consume(row);
            }
            if let Some(u) = planes.u() {
                if verbose {
                    eprintln!(
                        "  U plane: {}x{} stride={}",
                        u.width(),
                        u.height(),
                        u.stride()
                    );
                }
                for row in u.rows() {
                    hasher.consume(row);
                }
            }
            if let Some(v) = planes.v() {
                if verbose {
                    eprintln!(
                        "  V plane: {}x{} stride={}",
                        v.width(),
                        v.height(),
                        v.stride()
                    );
                }
                for row in v.rows() {
                    hasher.consume(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            let y = planes.y();
            for row in y.rows() {
                for &pixel in row {
                    hasher.consume(pixel.to_le_bytes());
                }
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &pixel in row {
                        hasher.consume(pixel.to_le_bytes());
                    }
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    for &pixel in row {
                        hasher.consume(pixel.to_le_bytes());
                    }
                }
            }
        }
    }
}

/// Detect input format from file contents.
enum Format {
    Ivf,
    AnnexB,
    RawObu,
}

fn detect_format(data: &[u8]) -> Format {
    if data.len() >= 4 && &data[0..4] == b"DKIF" {
        return Format::Ivf;
    }

    if data.is_empty() {
        return Format::RawObu;
    }

    // Check if first byte is a valid OBU header:
    // Bit 7: forbidden (must be 0)
    // Bits 6-3: obu_type (valid: 1-8, 15)
    // Bit 1: obu_has_size_field
    let first = data[0];
    let forbidden = (first >> 7) & 1;
    let obu_type = (first >> 3) & 0xF;
    let has_size = (first >> 1) & 1;

    if forbidden == 0 && matches!(obu_type, 1..=8 | 15) && has_size == 1 {
        Format::RawObu
    } else {
        // Not a valid OBU header — assume Annex B (LEB128 temporal unit sizes)
        Format::AnnexB
    }
}

fn process_frame(
    frame: &Frame,
    hasher: &mut md5::Context,
    frame_count: &mut u32,
    verbose: bool,
    per_frame: bool,
    limit: Option<u32>,
) {
    if limit.is_some_and(|l| *frame_count >= l) {
        return;
    }
    if per_frame {
        let mut frame_hasher = md5::Context::new();
        hash_frame(frame, &mut frame_hasher, false);
        hash_frame(frame, hasher, false);
        #[allow(deprecated)]
        let frame_digest = frame_hasher.compute();
        eprintln!("frame {} md5={:x}", *frame_count, frame_digest);
    } else {
        hash_frame(frame, hasher, verbose);
    }
    *frame_count += 1;
}

fn decode_frames(
    decoder: &mut Decoder,
    data: &[u8],
    hasher: &mut md5::Context,
    frame_count: &mut u32,
    verbose: bool,
    per_frame: bool,
    limit: Option<u32>,
) {
    match decoder.decode(data) {
        Ok(Some(frame)) => {
            process_frame(&frame, hasher, frame_count, verbose, per_frame, limit);
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("Decode error: {:?} ({})", e, e);
            return;
        }
    }
    // Drain additional frames from buffered data
    loop {
        match decoder.get_frame() {
            Ok(Some(frame)) => {
                process_frame(&frame, hasher, frame_count, verbose, per_frame, limit);
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("Decode error draining frames: {}", e);
                break;
            }
        }
    }
}

// ARM's CPU mask currently leaves several baseline NEON dispatchers active.
// The sidecar oracle must select their real scalar fallback as well. Example
// builds enable archmage's testable_dispatch dev dependency; keep its lock and
// disable state alive until every decoder worker has joined.
#[cfg(target_arch = "aarch64")]
struct ArmScalarGuard {
    _lock: archmage::testing::TokenTestGuard,
}

#[cfg(target_arch = "aarch64")]
impl ArmScalarGuard {
    fn new() -> Self {
        let lock = archmage::testing::lock_token_testing();
        archmage::Arm64::dangerously_disable_token_process_wide(true)
            .expect("ARM scalar oracle needs testable_dispatch");
        Self { _lock: lock }
    }
}

#[cfg(target_arch = "aarch64")]
impl Drop for ArmScalarGuard {
    fn drop(&mut self) {
        archmage::Arm64::dangerously_disable_token_process_wide(false)
            .expect("restore ARM tokens after scalar oracle");
    }
}

// The x86 mask gates pixel DSP, but plain msac incants and autoversioned
// coefficient decoding select tokens independently. Cap those tokens too so
// the sidecar oracle exercises the requested fallback tier throughout decoding.
#[cfg(target_arch = "x86_64")]
struct X86TierGuard {
    level: rav1d_safe::src::managed::CpuLevel,
    _lock: archmage::testing::TokenTestGuard,
}

#[cfg(target_arch = "x86_64")]
impl X86TierGuard {
    fn for_level(level: rav1d_safe::src::managed::CpuLevel) -> Option<Self> {
        use rav1d_safe::src::managed::CpuLevel as L;
        let level = match level {
            L::Native => return None,
            L::X86V2 | L::X86V3 | L::X86V4 => level,
            _ => L::Scalar,
        };
        let lock = archmage::testing::lock_token_testing();
        Self::set_disabled(level, true);
        Some(Self { level, _lock: lock })
    }

    fn set_disabled(level: rav1d_safe::src::managed::CpuLevel, disabled: bool) {
        use rav1d_safe::src::managed::CpuLevel as L;
        let result = match level {
            L::X86V2 => archmage::X64V3Token::dangerously_disable_token_process_wide(disabled),
            L::X86V3 => archmage::X64V4Token::dangerously_disable_token_process_wide(disabled),
            L::X86V4 => {
                archmage::Avx512Fp16Token::dangerously_disable_token_process_wide(disabled)
                    .expect("x86 tier oracle needs testable_dispatch");
                archmage::X64V4xToken::dangerously_disable_token_process_wide(disabled)
            }
            _ => archmage::X64V1Token::dangerously_disable_token_process_wide(disabled),
        };
        result.expect("x86 tier oracle needs testable_dispatch");
    }
}

#[cfg(target_arch = "x86_64")]
impl Drop for X86TierGuard {
    fn drop(&mut self) {
        Self::set_disabled(self.level, false);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod x86_tier_tests {
    use super::X86TierGuard;
    use archmage::SimdToken;
    use rav1d_safe::src::managed::CpuLevel as L;

    #[test]
    fn tier_caps_apply_to_workers_and_restore_baseline_tokens() {
        for level in [L::Scalar, L::X86V2, L::X86V3, L::X86V4] {
            let guard = X86TierGuard::for_level(level).unwrap();
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    std::thread::spawn(move || {
                        for _ in 0..1000 {
                            if level == L::Scalar {
                                assert!(archmage::X64V1Token::summon().is_none());
                            } else {
                                assert!(archmage::X64V1Token::summon().is_some());
                            }
                            if matches!(level, L::Scalar | L::X86V2) {
                                assert!(archmage::X64V3Token::summon().is_none());
                            }
                            if level != L::X86V4 {
                                assert!(archmage::X64V4Token::summon().is_none());
                            }
                            assert!(archmage::X64V4xToken::summon().is_none());
                            assert!(archmage::Avx512Fp16Token::summon().is_none());
                        }
                    })
                })
                .collect();
            for worker in workers {
                worker.join().unwrap();
            }
            drop(guard);
            assert!(archmage::X64V1Token::summon().is_some());
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod arm_scalar_tests {
    use super::ArmScalarGuard;
    use archmage::SimdToken;

    #[test]
    fn scalar_guard_disables_worker_tokens_and_restores_neon() {
        let guard = ArmScalarGuard::new();
        let workers: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..1000 {
                        assert!(archmage::Arm64::summon().is_none());
                        assert!(archmage::Arm64V2Token::summon().is_none());
                        assert!(archmage::Arm64V3Token::summon().is_none());
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        drop(guard);
        assert!(archmage::Arm64::summon().is_some());
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut filmgrain = false;
    let mut quiet = false;
    let mut per_frame = false;
    let mut level: Option<rav1d_safe::src::managed::CpuLevel> = None;
    let mut settings_strictness: Option<rav1d_safe::src::managed::Strictness> = None;
    // Default 1 to keep every existing invocation byte-identical. `--threads 8`
    // exists because a single-threaded-only identity check cannot see a
    // tile-threading or borrow-tracker defect at all, and ad-hoc vectors (a
    // size sweep, a forced-tile grid) are not in the corpus that
    // `md5_inventory --threads` covers.
    let mut threads: u32 = 1;
    // Frames in flight (`--delay N`, 0 = auto). Defaults to the library default, which
    // in untracked builds already frame-threads when `--threads` > 1.
    let mut max_frame_delay = Settings::default().max_frame_delay;
    let mut limit: Option<u32> = None;
    // dav1d-test-data standalone test() args — see tests/decode_cpu_levels.rs
    // for the same mapping applied via Settings.
    let mut operating_point: Option<u8> = None;
    let mut all_layers: Option<bool> = None;
    let mut decode_frame_type: Option<rav1d_safe::src::managed::DecodeFrameType> = None;
    let mut positional: Vec<String> = Vec::new();

    // meson/dav1d args use both `--flag value` and `--flag=value` forms.
    let mut it = args[1..].iter().flat_map(|a| match a.split_once('=') {
        Some((k, v)) if k.starts_with("--") => vec![k.to_string(), v.to_string()],
        _ => vec![a.clone()],
    });
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--filmgrain" => filmgrain = true,
            "-q" | "--quiet" => quiet = true,
            "--per-frame" => per_frame = true,
            "--scalar" => level = Some(rav1d_safe::src::managed::CpuLevel::Scalar),
            "--lenient" => {
                settings_strictness = Some(rav1d_safe::src::managed::Strictness::Lenient)
            }
            "--level" => {
                use rav1d_safe::src::managed::CpuLevel as L;
                level = Some(match it.next().as_deref() {
                    Some("scalar") => L::Scalar,
                    Some("v2") => L::X86V2,
                    Some("v3") => L::X86V3,
                    Some("v4") => L::X86V4,
                    Some("neon") => L::Neon,
                    Some("neon-dotprod") => L::NeonDotprod,
                    Some("neon-i8mm") => L::NeonI8mm,
                    Some("native") => L::Native,
                    other => panic!(
                        "--level needs scalar|v2|v3|v4|neon|neon-dotprod|neon-i8mm|native, got {other:?}"
                    ),
                });
            }
            "--delay" => {
                max_frame_delay = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--delay needs a number (frames in flight; 0 = auto)");
            }
            "--threads" => {
                threads = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--threads needs a number");
            }
            "--limit" => {
                limit = Some(
                    it.next()
                        .and_then(|v| v.parse().ok())
                        .expect("--limit needs a frame count"),
                );
            }
            "--oppoint" => {
                operating_point = Some(
                    it.next()
                        .and_then(|v| v.parse().ok())
                        .expect("--oppoint needs a number"),
                );
            }
            "--alllayers" => {
                all_layers = Some(it.next().as_deref() != Some("0"));
            }
            "--decodeframetype" => {
                use rav1d_safe::src::managed::DecodeFrameType as FT;
                decode_frame_type = Some(match it.next().as_deref() {
                    Some("all") => FT::All,
                    Some("reference") => FT::Reference,
                    Some("intra") => FT::Intra,
                    Some("key") => FT::Key,
                    other => {
                        panic!("--decodeframetype needs all|reference|intra|key, got {other:?}")
                    }
                });
            }
            _ => positional.push(arg),
        }
    }

    if positional.is_empty() {
        eprintln!(
            "Usage: {} [--filmgrain] [-q] [--per-frame] [--threads N] [--limit N] [--oppoint N] [--alllayers 0|1] [--decodeframetype all|reference|intra|key] <input> [expected_md5]",
            args[0]
        );
        std::process::exit(1);
    }

    let input_path = &positional[0];
    let expected_md5 = positional.get(1).map(String::as_str);
    let data = fs::read(input_path).expect("Failed to read input");
    let verbose = !quiet;

    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = max_frame_delay;
    settings.apply_grain = filmgrain;
    if let Some(l) = level {
        settings.cpu_level = l;
    }
    if let Some(s) = settings_strictness {
        settings.strictness = s;
    }
    if let Some(op) = operating_point {
        settings.operating_point = op;
    }
    if let Some(al) = all_layers {
        settings.all_layers = al;
    }
    if let Some(ft) = decode_frame_type {
        settings.decode_frame_type = ft;
    }
    #[cfg(target_arch = "aarch64")]
    let _arm_scalar_guard = (settings.cpu_level == rav1d_safe::src::managed::CpuLevel::Scalar)
        .then(ArmScalarGuard::new);
    #[cfg(target_arch = "x86_64")]
    let _x86_tier_guard = X86TierGuard::for_level(settings.cpu_level);
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        use rav1d_safe::src::managed::CpuLevel as L;
        match settings.cpu_level {
            L::Native => {}
            L::X86V4 => {
                assert!(archmage::X64V4xToken::summon().is_none());
                assert!(archmage::Avx512Fp16Token::summon().is_none());
            }
            L::X86V3 => assert!(archmage::X64V4Token::summon().is_none()),
            L::X86V2 => assert!(archmage::X64V3Token::summon().is_none()),
            _ => assert!(archmage::X64V1Token::summon().is_none()),
        }
    }
    #[cfg(target_arch = "aarch64")]
    if settings.cpu_level == rav1d_safe::src::managed::CpuLevel::Scalar {
        use archmage::SimdToken;
        assert!(archmage::Arm64::summon().is_none());
        assert!(archmage::Arm64V2Token::summon().is_none());
        assert!(archmage::Arm64V3Token::summon().is_none());
    }
    let mut decoder = Decoder::with_settings(settings).expect("decoder creation failed");
    let mut hasher = md5::Context::new();
    let mut frame_count = 0u32;

    match detect_format(&data) {
        Format::Ivf => {
            let mut cursor = Cursor::new(&data);
            let frames = ivf_parser::parse_all_frames(&mut cursor).expect("IVF parse failed");
            for ivf_frame in &frames {
                decode_frames(
                    &mut decoder,
                    &ivf_frame.data,
                    &mut hasher,
                    &mut frame_count,
                    verbose,
                    per_frame,
                    limit,
                );
            }
        }
        Format::AnnexB => match annexb_parser::parse_annexb(&data) {
            Ok(units) => {
                if verbose {
                    eprintln!("Annex B: {} temporal units", units.len());
                }
                for (tu_idx, unit) in units.iter().enumerate() {
                    if verbose {
                        eprintln!("  TU {tu_idx}: {} bytes", unit.data.len());
                    }
                    decode_frames(
                        &mut decoder,
                        &unit.data,
                        &mut hasher,
                        &mut frame_count,
                        verbose,
                        per_frame,
                        limit,
                    );
                }
            }
            Err(e) => {
                eprintln!("Annex B parse error: {}", e);
            }
        },
        Format::RawObu => {
            decode_frames(
                &mut decoder,
                &data,
                &mut hasher,
                &mut frame_count,
                verbose,
                per_frame,
                limit,
            );
        }
    }

    // Flush remaining frames
    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                process_frame(
                    frame,
                    &mut hasher,
                    &mut frame_count,
                    verbose,
                    per_frame,
                    limit,
                );
            }
        }
        Err(e) => {
            eprintln!("Flush error: {}", e);
        }
    }

    #[allow(deprecated)]
    let digest = hasher.compute();
    let md5_hex = format!("{:x}", digest);

    println!("{}", md5_hex);
    if verbose {
        eprintln!("Frames: {}", frame_count);
    }

    if let Some(expected) = expected_md5 {
        if md5_hex == expected {
            if verbose {
                eprintln!("MATCH");
            }
        } else {
            eprintln!("MISMATCH: expected {} got {}", expected, md5_hex);
            std::process::exit(1);
        }
    }
}
