//! Profile AV1 decode performance.
//!
//! Usage:
//!   # Safe-SIMD:
//!   cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example profile_decode
//!   # ASM:
//!   cargo build --release --features "asm,bitdepth_8,bitdepth_16" --example profile_decode
//!
//!   ./target/release/examples/profile_decode <input.ivf> [iterations]

use rav1d_safe::src::managed::{Decoder, Settings};
use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::{self, Read};
use std::time::Instant;

/// Parse IVF header (32 bytes), return frame count.
fn parse_ivf_header<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut header = [0u8; 32];
    reader.read_exact(&mut header)?;
    if &header[0..4] != b"DKIF" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not an IVF file"));
    }
    Ok(u32::from_le_bytes([header[24], header[25], header[26], header[27]]))
}

/// Parse one IVF frame, return None on EOF.
fn parse_ivf_frame<R: Read>(reader: &mut R) -> io::Result<Option<Vec<u8>>> {
    let mut fh = [0u8; 12];
    match reader.read_exact(&mut fh) {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let size = u32::from_le_bytes([fh[0], fh[1], fh[2], fh[3]]) as usize;
    let mut data = vec![0u8; size];
    reader.read_exact(&mut data)?;
    Ok(Some(data))
}

/// Extract all OBU frames from an IVF file.
fn read_ivf_frames(path: &str) -> Vec<Vec<u8>> {
    let mut f = File::open(path).expect("Failed to open input");
    parse_ivf_header(&mut f).expect("Invalid IVF header");
    let mut frames = Vec::new();
    while let Some(data) = parse_ivf_frame(&mut f).expect("IVF parse error") {
        frames.push(data);
    }
    frames
}

fn decode_once(obu_frames: &[Vec<u8>]) -> usize {
    let settings = Settings {
        threads: 1,
        ..Default::default()
    };
    let mut decoder = Decoder::with_settings(settings).expect("decoder creation failed");
    let mut count = 0;

    for obu in obu_frames {
        match decoder.decode(obu) {
            Ok(Some(frame)) => {
                black_box(&frame);
                count += 1;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Decode error: {e}");
            }
        }
    }

    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                black_box(frame);
                count += 1;
            }
        }
        Err(e) => {
            eprintln!("Flush error: {e}");
        }
    }

    count
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.ivf> [iterations]", args[0]);
        std::process::exit(1);
    }

    let obu_frames = read_ivf_frames(&args[1]);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let total_bytes: usize = obu_frames.iter().map(|f| f.len()).sum();

    eprintln!("Input: {} ({} frames, {} bytes)", args[1], obu_frames.len(), total_bytes);
    eprintln!("Iterations: {iterations}");

    // Warmup
    let frame_count = decode_once(&obu_frames);
    eprintln!("Decoded frames per iteration: {frame_count}");

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        let f = decode_once(black_box(&obu_frames));
        black_box(f);
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed / iterations as u32;
    let per_frame = elapsed / (iterations * frame_count) as u32;
    eprintln!(
        "Total: {:.3}s ({} iters, {:.3}ms/iter, {:.3}ms/frame)",
        elapsed.as_secs_f64(),
        iterations,
        per_iter.as_secs_f64() * 1000.0,
        per_frame.as_secs_f64() * 1000.0,
    );
}
