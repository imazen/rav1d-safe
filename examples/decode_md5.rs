//! Decode AV1 bitstreams and compute MD5 of decoded pixel data.
//!
//! Supports IVF, raw OBU (Section 5), and Annex B container formats.
//! Produces MD5 hashes compatible with dav1d's --verify / aomdec --md5 format.
//!
//! Usage:
//!   cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example decode_md5
//!   ./target/release/examples/decode_md5 [--filmgrain] [-q] <input> [expected_md5]

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::env;
use std::fs;
use std::io::Cursor;

mod annexb_parser;
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
) {
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
) {
    match decoder.decode(data) {
        Ok(Some(frame)) => {
            process_frame(&frame, hasher, frame_count, verbose, per_frame);
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("Decode error: {}", e);
            return;
        }
    }
    // Drain additional frames from buffered data
    loop {
        match decoder.get_frame() {
            Ok(Some(frame)) => {
                process_frame(&frame, hasher, frame_count, verbose, per_frame);
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("Decode error draining frames: {}", e);
                break;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut filmgrain = false;
    let mut quiet = false;
    let mut per_frame = false;
    let mut positional = Vec::new();

    for arg in &args[1..] {
        match arg.as_str() {
            "--filmgrain" => filmgrain = true,
            "-q" | "--quiet" => quiet = true,
            "--per-frame" => per_frame = true,
            _ => positional.push(arg.as_str()),
        }
    }

    if positional.is_empty() {
        eprintln!(
            "Usage: {} [--filmgrain] [-q] <input> [expected_md5]",
            args[0]
        );
        std::process::exit(1);
    }

    let input_path = positional[0];
    let expected_md5 = positional.get(1).copied();
    let data = fs::read(input_path).expect("Failed to read input");
    let verbose = !quiet;

    let settings = Settings {
        threads: 1,
        apply_grain: filmgrain,
        ..Default::default()
    };
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
                );
            }
        }
        Format::AnnexB => match annexb_parser::parse_annexb(&data) {
            Ok(units) => {
                if verbose {
                    eprintln!("Annex B: {} temporal units", units.len());
                }
                for unit in &units {
                    decode_frames(
                        &mut decoder,
                        &unit.data,
                        &mut hasher,
                        &mut frame_count,
                        verbose,
                        per_frame,
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
            );
        }
    }

    // Flush remaining frames
    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                process_frame(frame, &mut hasher, &mut frame_count, verbose, per_frame);
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
