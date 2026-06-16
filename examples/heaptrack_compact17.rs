//! Allocation-count harness for issue #17 (compact_read_per_row churn).
//!
//! Decodes an AVIF/OBU with tile threading ON (`threads >= 2` → `n_tc > 1` →
//! the per-edge / per-block compact buffer path), so heaptrack captures the
//! `compact_read_per_row` allocation site.
//!
//! Usage:
//!   heaptrack ./target/release/examples/heaptrack_compact17 <input.avif> [threads] [iters]

use rav1d_safe::src::managed::{Decoder, Settings};
use std::env;
use std::hint::black_box;

fn extract_obu(bytes: &[u8]) -> Vec<u8> {
    // Raw OBU files start with a temporal-delimiter OBU (0x12 0x00); AVIF starts
    // with an ISOBMFF box. Try AVIF parse first, fall back to raw OBU.
    match zenavif_parse::AvifParser::from_bytes(bytes) {
        Ok(parser) => parser
            .primary_data()
            .expect("extract primary item")
            .into_owned(),
        Err(_) => bytes.to_vec(),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.avif|.obu> [threads] [iters]", args[0]);
        std::process::exit(1);
    }
    let bytes = std::fs::read(&args[1]).expect("read input");
    let obu = extract_obu(&bytes);
    let threads: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    eprintln!(
        "input={} obu={:.1}KB threads={threads} iters={iters}",
        args[1],
        obu.len() as f64 / 1024.0
    );

    let mut frames = 0usize;
    for _ in 0..iters {
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 1; // pure tile threading (n_fc = 1)
        settings.frame_size_limit = 8192 * 8192;
        let mut decoder = Decoder::with_settings(settings).expect("decoder");
        if let Ok(Some(frame)) = decoder.decode(black_box(&obu)) {
            black_box(&frame);
            frames += 1;
        }
        if let Ok(remaining) = decoder.flush() {
            frames += remaining.len();
        }
    }
    eprintln!("decoded {frames} frame(s)");
}
