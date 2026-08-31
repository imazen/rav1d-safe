//! THROWAWAY driver for the DisjointMut contention probe.
//!
//! Build with `--features probe-count`; decodes one AVIF `iters` times at a
//! given thread count and dumps the per-tracker-instance / per-thread counters.
//! Counters are reset after a warmup decode so the report covers timed work.
//!
//! Usage: probe_tracker <input.avif> <threads> <iters>

use rav1d_safe::src::managed::{Decoder, Settings};
use std::hint::black_box;
use std::time::Instant;

fn extract_obu(avif_bytes: &[u8]) -> Vec<u8> {
    let parser = zenavif_parse::AvifParser::from_bytes(avif_bytes).expect("avif parse");
    parser
        .primary_data()
        .expect("avif primary item")
        .into_owned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <input.avif> <threads> <iters>", args[0]);
        std::process::exit(2);
    }
    let path = &args[1];
    let threads: u32 = args[2].parse().expect("threads");
    let iters: u64 = args[3].parse().expect("iters");
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
    let mut dec = Decoder::with_settings(settings).expect("decoder");

    // Warmup (allocates every buffer, so no first-touch cost lands in the report).
    let warm = dec.decode(&obu).expect("decode").expect("frame");
    let (w, h, bpc) = (warm.width(), warm.height(), warm.bit_depth());
    drop(warm);
    let _ = dec.flush();

    #[cfg(feature = "probe-count")]
    rav1d_disjoint_mut::probe::reset();
    #[cfg(feature = "probe-wide")]
    rav1d_disjoint_mut::wide_probe::reset();
    #[cfg(feature = "probe-lockstats")]
    rav1d_disjoint_mut::lock_probe::reset();
    #[cfg(feature = "probe-shardsim")]
    rav1d_disjoint_mut::probe::shard_reset();
    #[cfg(feature = "probe-sites")]
    rav1d_disjoint_mut::site_probe::reset();

    let t0 = Instant::now();
    for _ in 0..iters {
        let f = dec.decode(black_box(&obu)).expect("decode");
        black_box(&f);
        drop(f);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    println!(
        "RUN\t{file}\t{w}x{h}\t{bpc}bpc\tthreads={threads}\titers={iters}\tms_total={ms:.2}\tms_per_frame={:.3}",
        ms / iters as f64
    );

    #[cfg(feature = "probe-wide")]
    print!("{}", rav1d_disjoint_mut::wide_probe::report());
    #[cfg(feature = "probe-lockstats")]
    print!("{}", rav1d_disjoint_mut::lock_probe::report());
    #[cfg(feature = "probe-count")]
    {
        print!("{}", rav1d_disjoint_mut::probe::report(iters));
    }
    #[cfg(feature = "probe-shardsim")]
    {
        print!("{}", rav1d_disjoint_mut::probe::shard_report());
    }
    #[cfg(feature = "probe-sites")]
    {
        print!("{}", rav1d_disjoint_mut::site_probe::report(iters));
    }
    #[cfg(not(any(feature = "probe-count", feature = "probe-wide")))]
    eprintln!("(built without --features probe-count / probe-wide; no counters)");

    let _ = dec.flush();
}
