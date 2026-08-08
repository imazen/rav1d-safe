//! A/B decode-throughput harness for cross-commit comparison.
//!
//! Unlike the divan / zenbench benches, this is a *single-config-per-process*
//! runner: one input, one thread count, N timed reps. That shape exists so an
//! external driver can interleave two BUILDS (two commits) at process
//! granularity — alternating A,B,A,B,... — which is the only way to compare
//! commits without letting thermal drift masquerade as a delta.
//!
//! The global tile-threading flag (`set_tile_threading`, driven by `n_tc > 1`)
//! is process-wide and latched at decoder creation, so one config per process
//! is also the only way to measure a threads=1 control that is genuinely
//! uncontaminated by a previous multithreaded decoder in the same process.
//!
//! Usage:
//!   bench_ab_decode <input.avif> <threads> <iters> <reps> <label>
//!
//! Emits to stdout, tab-separated:
//!   RESULT   label file threads rep iters ms_total ms_per_frame
//!   CHECKSUM label file threads md5-of-all-planes
//!
//! The checksum lets the same run double as a bit-identity check between arms.

use rav1d_safe::src::managed::{CpuLevel, Decoder, Frame, Planes, Settings};
use std::hint::black_box;
use std::time::Instant;

fn extract_obu(avif_bytes: &[u8]) -> Vec<u8> {
    let parser = zenavif_parse::AvifParser::from_bytes(avif_bytes).expect("avif parse");
    parser
        .primary_data()
        .expect("avif primary item")
        .into_owned()
}

/// md5 over every plane's visible pixels (row-by-row, so stride padding never
/// enters the hash).
fn frame_md5(frame: &Frame) -> String {
    let mut ctx = md5::Context::new();
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                ctx.consume(row);
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    ctx.consume(row);
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    ctx.consume(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            for row in planes.y().rows() {
                for &px in row {
                    ctx.consume(px.to_le_bytes());
                }
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
        }
    }
    format!("{:x}", ctx.finalize())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "Usage: {} <input.avif> <threads> <iters> <reps> <label>",
            args[0]
        );
        std::process::exit(2);
    }
    let path = &args[1];
    let threads: u32 = args[2].parse().expect("threads");
    let iters: usize = args[3].parse().expect("iters");
    let reps: usize = args[4].parse().expect("reps");
    let label = &args[5];
    let file = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string();

    let avif = std::fs::read(path).expect("read input");
    let obu = extract_obu(&avif);

    let mut settings = Settings::default();
    settings.threads = threads;
    settings.frame_size_limit = 8192 * 8192;
    // Pin tile threading only. The checked build already forces n_fc = 1
    // (frame threading needs `unchecked`), so this is a no-op there — but it
    // keeps an `unchecked` build on the SAME threading model instead of
    // silently gaining frame threading and making the arms incomparable.
    settings.max_frame_delay = 1;
    // Optional 6th arg: CPU dispatch tier. Only affects the flag-gated DSP
    // tables; the unconditional `Arm64::summon()` sites stay NEON regardless.
    if let Some(level) = args.get(6) {
        settings.cpu_level = match level.as_str() {
            "native" => CpuLevel::Native,
            "scalar" => CpuLevel::Scalar,
            "neon" => CpuLevel::Neon,
            other => panic!("unknown cpu level {other}"),
        };
    }
    let mut dec = Decoder::with_settings(settings).expect("decoder");

    // Warmup + checksum + geometry report (one decode, not timed).
    let warm = dec.decode(&obu).expect("decode").expect("frame");
    let md5 = frame_md5(&warm);
    let (w, h, bpc) = (warm.width(), warm.height(), warm.bit_depth());
    drop(warm);
    let _ = dec.flush();
    println!("CHECKSUM\t{label}\t{file}\t{threads}\t{md5}");
    println!("GEOM\t{label}\t{file}\t{threads}\t{w}x{h}\t{bpc}bpc");

    #[cfg(feature = "probe-tasktime")]
    {
        rav1d_safe::src::probe_tasktime::reset();
        rav1d_safe::src::probe_tasktime::start_monitor();
    }

    for rep in 0..reps {
        let t0 = Instant::now();
        for _ in 0..iters {
            let f = dec.decode(black_box(&obu)).expect("decode");
            black_box(&f);
            drop(f);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "RESULT\t{label}\t{file}\t{threads}\t{rep}\t{iters}\t{ms:.4}\t{:.6}",
            ms / iters as f64
        );
    }
    #[cfg(feature = "probe-tasktime")]
    rav1d_safe::src::probe_tasktime::report((reps * iters) as u64);
    let _ = dec.flush();
}
