//! Profile AV1 decode performance from IVF files.
//!
//! Usage:
//!   cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example profile_ivf
//!   ./target/release/examples/profile_ivf <input.ivf> [iterations]
//!
//!   # With perf:
//!   perf record -g ./target/release/examples/profile_ivf <input.ivf> 200
//!   perf report

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

use rav1d_safe::src::managed::{Decoder, Settings};
use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::time::Instant;

fn decode_ivf_frames(frames: &[ivf_parser::IvfFrame]) -> usize {
    let settings = Settings {
        threads: 1,
        ..Default::default()
    };
    let mut decoder = Decoder::with_settings(settings).expect("decoder creation failed");
    let mut decoded = 0;

    for ivf_frame in frames {
        match decoder.decode(&ivf_frame.data) {
            Ok(Some(frame)) => {
                black_box(&frame);
                decoded += 1;
            }
            Ok(None) => {}
            Err(_) => {}
        }
    }

    if let Ok(remaining) = decoder.flush() {
        for frame in &remaining {
            black_box(frame);
        }
        decoded += remaining.len();
    }

    decoded
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.ivf> [iterations]", args[0]);
        std::process::exit(1);
    }

    let file = File::open(&args[1]).expect("Failed to open input");
    let mut reader = BufReader::new(file);
    let frames = ivf_parser::parse_all_frames(&mut reader).expect("Failed to parse IVF");

    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    eprintln!("Input: {} ({} IVF frames)", args[1], frames.len());
    eprintln!("Iterations: {}", iterations);

    // Warmup
    let decoded = decode_ivf_frames(&frames);
    eprintln!("Frames decoded per iteration: {}", decoded);

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        let d = decode_ivf_frames(black_box(&frames));
        black_box(d);
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed / iterations as u32;
    eprintln!(
        "Total: {:.3}s ({} iters, {:.3}ms/iter)",
        elapsed.as_secs_f64(),
        iterations,
        per_iter.as_secs_f64() * 1000.0,
    );
}
