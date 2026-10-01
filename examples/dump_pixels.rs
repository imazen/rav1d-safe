//! Dump raw decoded pixels to stdout for comparison.
//!
//! Usage:
//!   ./target/release/examples/dump_pixels <input.ivf> > output.yuv

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::env;
use std::fs;
use std::io::{self, Cursor, Write};

#[path = "helpers/annexb_parser.rs"]
mod annexb_parser;
#[path = "helpers/ivf_parser.rs"]
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
    let mut it = args.iter().skip(1);
    let mut level: Option<rav1d_safe::src::managed::CpuLevel> = None;
    let mut input = None;
    while let Some(a) = it.next() {
        if a == "--level" {
            use rav1d_safe::src::managed::CpuLevel as L;
            level = Some(match it.next().map(|s| s.as_str()) {
                Some("scalar") => L::Scalar,
                Some("v2") => L::X86V2,
                Some("v3") => L::X86V3,
                Some("v4") => L::X86V4,
                Some("native") => L::Native,
                other => panic!("--level needs scalar|v2|v3|v4|native, got {other:?}"),
            });
        } else {
            input = Some(a.clone());
        }
    }
    let Some(input) = input else {
        eprintln!("Usage: {} [--level L] <input.ivf>", args[0]);
        std::process::exit(1);
    };

    let data = fs::read(&input).expect("Failed to read input");
    let is_ivf = data.len() >= 4 && &data[0..4] == b"DKIF";
    let is_annexb = !is_ivf && {
        let f = data.first().copied().unwrap_or(0);
        !(f >> 7 == 0 && matches!((f >> 3) & 0xF, 1..=8 | 15) && (f >> 1) & 1 == 1)
    };

    let mut settings = Settings::default();
    settings.threads = 1;
    settings.apply_grain = false;
    if let Some(l) = level {
        settings.cpu_level = l;
    }
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
    } else if is_annexb {
        for unit in annexb_parser::parse_annexb(&data).expect("annexb parse failed") {
            match decoder.decode(&unit.data) {
                Ok(Some(frame)) => {
                    dump_frame(&frame, &mut out).unwrap();
                    frame_count += 1;
                }
                Ok(None) => {}
                Err(e) => eprintln!("Decode error: {}", e),
            }
            loop {
                match decoder.get_frame() {
                    Ok(Some(frame)) => {
                        dump_frame(&frame, &mut out).unwrap();
                        frame_count += 1;
                    }
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("Decode error draining frames: {}", e);
                        break;
                    }
                }
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
        loop {
            match decoder.get_frame() {
                Ok(Some(frame)) => {
                    dump_frame(&frame, &mut out).unwrap();
                    frame_count += 1;
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("Decode error draining frames: {}", e);
                    break;
                }
            }
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
