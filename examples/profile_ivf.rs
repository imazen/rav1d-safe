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

use rav1d_safe::src::managed::{Decoder, InloopFilters, Settings};
use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::time::Instant;

/// `RAV1D_THREADS` (default 1) and `RAV1D_INLOOP` (default `all`).
///
/// Threads default to 1 so every number recorded by this example before the
/// knob existed stays comparable. It is needed because the t=1 -> t=8 slowdown
/// on real multi-frame content (1.14x-3.44x across the LR ladder, 2026-08-10)
/// could not be PROFILED at all otherwise: `bench_ivf_limit` is thread-capable
/// but exits after one pass, too short for `/usr/bin/sample` to attach to.
///
/// `RAV1D_INLOOP` mirrors `dav1d --inloopfilters` — see `bench_ivf_limit`. It
/// changes output pixels; attribution only.
fn decode_ivf_frames(frames: &[ivf_parser::IvfFrame]) -> usize {
    let mut settings = Settings::default();
    settings.threads = std::env::var("RAV1D_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    // Tile threading only, matching `dav1d --framedelay 1` and the rest of the
    // gap campaign. A no-op in the checked build, which pins n_fc = 1 anyway.
    settings.max_frame_delay = 1;
    settings.inloop_filters = match std::env::var("RAV1D_INLOOP").as_deref().unwrap_or("all") {
        "all" => InloopFilters::all(),
        "none" => InloopFilters::none(),
        "nodeblock" => InloopFilters::CDEF.union(InloopFilters::RESTORATION),
        "nocdef" => InloopFilters::DEBLOCK.union(InloopFilters::RESTORATION),
        "norestoration" => InloopFilters::DEBLOCK.union(InloopFilters::CDEF),
        other => panic!(
            "RAV1D_INLOOP={other}: expected all|none|nodeblock|nocdef|norestoration \
             (dav1d --inloopfilters spelling)"
        ),
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

    // `RAV1D_ABLATE=looprestoration,cdef` switches those families' dispatchers
    // to the generic scalar reference. Needs `--features __ablate`; asserting
    // that here is what stops an un-ablated build from quietly reporting two
    // identical arms as an A/B result.
    if let Ok(list) = std::env::var("RAV1D_ABLATE") {
        assert!(
            rav1d_safe::src::ablate::ENABLED,
            "RAV1D_ABLATE set but built without --features __ablate"
        );
        let fams: Vec<_> = list
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| {
                rav1d_safe::src::ablate::Family::from_name(s)
                    .unwrap_or_else(|| panic!("unknown family {s}"))
            })
            .collect();
        eprintln!(
            "Ablated: {:?}",
            fams.iter().map(|f| f.name()).collect::<Vec<_>>()
        );
        rav1d_safe::src::ablate::set_disabled(&fams);
    }

    // Per-iteration timing to stdout, so an external driver can take a median
    // instead of trusting one mean.
    let reps: usize = std::env::var("RAV1D_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Warmup
    let decoded = decode_ivf_frames(&frames);
    eprintln!("Frames decoded per iteration: {}", decoded);

    // Timed runs
    let label = std::env::var("RAV1D_LABEL").unwrap_or_else(|_| "run".into());
    let mut last = 0.0f64;
    for rep in 0..reps {
        let start = Instant::now();
        for _ in 0..iterations {
            let d = decode_ivf_frames(black_box(&frames));
            black_box(d);
        }
        let elapsed = start.elapsed();
        let per_frame =
            elapsed.as_secs_f64() * 1000.0 / (iterations as f64 * decoded.max(1) as f64);
        last = per_frame;
        println!("RESULT\t{label}\t{rep}\t{iterations}\t{decoded}\t{per_frame:.6}");
    }
    eprintln!("{label}: {last:.4} ms/frame");
}
