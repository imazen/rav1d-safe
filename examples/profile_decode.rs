//! Profile AV1 decode performance.
//!
//! Usage:
//!   # Safe-SIMD:
//!   cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example profile_decode
//!   # ASM:
//!   cargo build --release --features "asm,bitdepth_8,bitdepth_16" --example profile_decode
//!
//!   ./target/release/examples/profile_decode <input.obu> [iterations]

use rav1d_safe::src::managed::{Decoder, Settings};
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

fn decode_once(data: &[u8]) -> usize {
    let settings = Settings {
        threads: 1,
        ..Default::default()
    };
    let mut decoder = Decoder::with_settings(settings).expect("decoder creation failed");
    let mut frames = 0;

    match decoder.decode(data) {
        Ok(Some(frame)) => {
            black_box(&frame);
            frames += 1;
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("Decode error: {}", e);
        }
    }

    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                black_box(frame);
            }
            frames += remaining.len();
        }
        Err(e) => {
            eprintln!("Flush error: {}", e);
        }
    }

    frames
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.obu> [iterations]", args[0]);
        std::process::exit(1);
    }

    let data = fs::read(&args[1]).expect("Failed to read input");
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    eprintln!("Input: {} ({} bytes)", args[1], data.len());
    eprintln!("Iterations: {}", iterations);

    // Warmup
    let frames = decode_once(&data);
    eprintln!("Frames per decode: {}", frames);

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        let f = decode_once(black_box(&data));
        black_box(f);
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed / iterations as u32;
    eprintln!(
        "Total: {:.3}s ({} iters, {:.3}ms/iter)",
        elapsed.as_secs_f64(),
        iterations,
        per_iter.as_secs_f64() * 1000.0
    );
}
