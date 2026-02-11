//! Per-frame MD5 for comparison with dav1d reference.

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::env;
use std::fs;
use std::io::Cursor;

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

fn hash_frame(frame: &Frame) -> String {
    let mut hasher = md5::Context::new();
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                hasher.consume(row);
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    hasher.consume(row);
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    hasher.consume(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            for row in planes.y().rows() {
                for &px in row {
                    hasher.consume(px.to_le_bytes());
                }
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &px in row {
                        hasher.consume(px.to_le_bytes());
                    }
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    for &px in row {
                        hasher.consume(px.to_le_bytes());
                    }
                }
            }
        }
    }
    #[allow(deprecated)]
    let digest = hasher.compute();
    format!("{:x}", digest)
}

fn print_frame_info(frame: &Frame, frame_num: usize, md5: &str) {
    println!(
        "Frame {}: {} ({}x{} bpc={})",
        frame_num,
        md5,
        frame.width(),
        frame.height(),
        frame.bit_depth()
    );
    match frame.planes() {
        Planes::Depth8(planes) => {
            let y_plane = planes.y();
            let row = y_plane.row(0);
            let n = 16.min(row.len());
            let pixels: Vec<String> = row[..n].iter().map(|p| format!("{:3}", p)).collect();
            println!("  Y[0..15]: {}", pixels.join(" "));
        }
        Planes::Depth16(planes) => {
            let y_plane = planes.y();
            let row = y_plane.row(0);
            let n = 16.min(row.len());
            let pixels: Vec<String> = row[..n].iter().map(|p| format!("{:3}", p)).collect();
            println!("  Y[0..15]: {}", pixels.join(" "));
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.ivf>", args[0]);
        std::process::exit(1);
    }

    let data = fs::read(&args[1]).expect("Failed to read input");
    let mut cursor = Cursor::new(&data);
    let frames = ivf_parser::parse_all_frames(&mut cursor).expect("IVF parse failed");

    let settings = Settings {
        threads: 1,
        apply_grain: false,
        ..Default::default()
    };
    let mut decoder = Decoder::with_settings(settings).expect("decoder creation failed");
    let mut frame_num = 0;

    for ivf_frame in &frames {
        match decoder.decode(&ivf_frame.data) {
            Ok(Some(frame)) => {
                let md5 = hash_frame(&frame);
                // Print first 16 Y pixels
                print_frame_info(&frame, frame_num, &md5);
                frame_num += 1;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Decode error: {}", e);
            }
        }
    }

    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                let md5 = hash_frame(frame);
                print_frame_info(frame, frame_num, &md5);
                frame_num += 1;
            }
        }
        Err(e) => {
            eprintln!("Flush error: {}", e);
        }
    }
}
