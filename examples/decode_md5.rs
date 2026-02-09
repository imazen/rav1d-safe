//! Decode IVF files and compute MD5 of decoded pixel data.
//!
//! Produces MD5 hashes compatible with dav1d's --verify / aomdec --md5 format.
//!
//! Usage:
//!   cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example decode_md5
//!   ./target/release/examples/decode_md5 <input.ivf>
//!
//! Or with ASM:
//!   cargo build --release --features "asm,bitdepth_8,bitdepth_16" --example decode_md5
//!   ./target/release/examples/decode_md5 <input.ivf>

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::env;
use std::fs;
use std::io::Cursor;

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
            // Y plane
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
            // U plane
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
            // V plane
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
            // Y plane - hash u16 as little-endian bytes
            let y = planes.y();
            for row in y.rows() {
                for &pixel in row {
                    hasher.consume(pixel.to_le_bytes());
                }
            }
            // U plane
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &pixel in row {
                        hasher.consume(pixel.to_le_bytes());
                    }
                }
            }
            // V plane
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.ivf> [expected_md5]", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let expected_md5 = args.get(2).map(|s| s.as_str());
    let data = fs::read(input_path).expect("Failed to read input");

    // Detect format: IVF starts with "DKIF", OBU doesn't
    let is_ivf = data.len() >= 4 && &data[0..4] == b"DKIF";

    let settings = Settings {
        threads: 1,
        apply_grain: false,
        ..Default::default()
    };
    let mut decoder = Decoder::with_settings(settings).expect("decoder creation failed");
    let mut hasher = md5::Context::new();
    let mut frame_count = 0u32;

    if is_ivf {
        let mut cursor = Cursor::new(&data);
        let frames = ivf_parser::parse_all_frames(&mut cursor).expect("IVF parse failed");

        for ivf_frame in &frames {
            match decoder.decode(&ivf_frame.data) {
                Ok(Some(frame)) => {
                    hash_frame(&frame, &mut hasher, true);
                    frame_count += 1;
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("Decode error on frame: {}", e);
                }
            }
        }
    } else {
        // Raw OBU
        match decoder.decode(&data) {
            Ok(Some(frame)) => {
                hash_frame(&frame, &mut hasher, true);
                frame_count += 1;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Decode error: {}", e);
            }
        }
    }

    // Flush remaining frames
    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                hash_frame(frame, &mut hasher, true);
                frame_count += 1;
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
    eprintln!("Frames: {}", frame_count);

    if let Some(expected) = expected_md5 {
        if md5_hex == expected {
            eprintln!("MATCH");
        } else {
            eprintln!("MISMATCH: expected {} got {}", expected, md5_hex);
            std::process::exit(1);
        }
    }
}
