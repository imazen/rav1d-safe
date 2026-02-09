//! Dump raw decoded pixels to stdout for comparison.
//!
//! Usage:
//!   ./target/release/examples/dump_pixels <input.ivf> > output.yuv

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::env;
use std::fs;
use std::io::{self, Cursor, Write};

mod ivf_parser;

fn dump_frame(frame: &Frame, out: &mut impl Write) -> io::Result<()> {
    match frame.planes() {
        Planes::Depth8(planes) => {
            let y = planes.y();
            for row in y.rows() {
                out.write_all(row)?;
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    out.write_all(row)?;
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    out.write_all(row)?;
                }
            }
        }
        Planes::Depth16(planes) => {
            let y = planes.y();
            for row in y.rows() {
                for &pixel in row {
                    out.write_all(&pixel.to_le_bytes())?;
                }
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &pixel in row {
                        out.write_all(&pixel.to_le_bytes())?;
                    }
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    for &pixel in row {
                        out.write_all(&pixel.to_le_bytes())?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.ivf>", args[0]);
        std::process::exit(1);
    }

    let data = fs::read(&args[1]).expect("Failed to read input");
    let is_ivf = data.len() >= 4 && &data[0..4] == b"DKIF";

    let settings = Settings {
        threads: 1,
        apply_grain: false,
        ..Default::default()
    };
    let mut decoder = Decoder::with_settings(settings).expect("decoder creation failed");
    let mut out = io::stdout().lock();
    let mut frame_count = 0u32;

    if is_ivf {
        let mut cursor = Cursor::new(&data);
        let frames = ivf_parser::parse_all_frames(&mut cursor).expect("IVF parse failed");
        for ivf_frame in &frames {
            match decoder.decode(&ivf_frame.data) {
                Ok(Some(frame)) => {
                    dump_frame(&frame, &mut out).unwrap();
                    frame_count += 1;
                }
                Ok(None) => {}
                Err(e) => eprintln!("Decode error: {}", e),
            }
        }
    } else {
        match decoder.decode(&data) {
            Ok(Some(frame)) => {
                dump_frame(&frame, &mut out).unwrap();
                frame_count += 1;
            }
            Ok(None) => {}
            Err(e) => eprintln!("Decode error: {}", e),
        }
    }

    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                dump_frame(frame, &mut out).unwrap();
                frame_count += 1;
            }
        }
        Err(e) => eprintln!("Flush error: {}", e),
    }

    eprintln!("Frames: {}", frame_count);
}
