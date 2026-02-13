//! Profile AVIF decode performance.
//!
//! Usage:
//!   ./target/release/examples/profile_avif <input.avif> [iterations]

use rav1d_safe::src::managed::{CpuLevel, Decoder, Settings};
use std::env;
use std::hint::black_box;
use std::time::Instant;

fn extract_obu(avif_bytes: &[u8]) -> Vec<u8> {
    let parser = zenavif_parse::AvifParser::from_bytes(avif_bytes)
        .expect("Failed to parse AVIF");
    parser.primary_data()
        .expect("Failed to extract primary item")
        .into_owned()
}

fn decode_once(obu: &[u8]) -> usize {
    let mut decoder = Decoder::with_settings(Settings {
        threads: 1,
        frame_size_limit: 8192 * 8192,
        cpu_level: CpuLevel::Native,
        ..Default::default()
    }).expect("decoder creation failed");

    let mut count = 0;
    match decoder.decode(obu) {
        Ok(Some(frame)) => { black_box(&frame); count += 1; }
        Ok(None) => {}
        Err(e) => eprintln!("Decode error: {e}"),
    }
    if let Ok(remaining) = decoder.flush() {
        for frame in &remaining {
            black_box(frame);
            count += 1;
        }
    }
    count
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.avif> [iterations]", args[0]);
        std::process::exit(1);
    }

    let avif_bytes = std::fs::read(&args[1]).expect("Failed to read file");
    let obu = extract_obu(&avif_bytes);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    eprintln!("Input: {} ({:.1} KB AVIF, {:.1} KB OBU)", args[1],
        avif_bytes.len() as f64 / 1024.0, obu.len() as f64 / 1024.0);
    eprintln!("Iterations: {iterations}");

    // Warmup
    let frame_count = decode_once(&obu);
    eprintln!("Frames per decode: {frame_count}");

    let start = Instant::now();
    for _ in 0..iterations {
        let f = decode_once(black_box(&obu));
        black_box(f);
    }
    let elapsed = start.elapsed();

    eprintln!("Total: {:.3}s ({} iters, {:.1}ms/iter)",
        elapsed.as_secs_f64(), iterations,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64);
}
