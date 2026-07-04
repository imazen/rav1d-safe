//! Tile-threading race stress loop (zenavif#30 forensics).
//!
//! Repeatedly decodes one raw AV1 OBU stream with a FRESH decoder per
//! iteration (threads=0 -> one worker per logical CPU, tile threading,
//! n_fc=1), the exact per-image call shape zenavif uses. Under heavy system
//! load (several parallel instances of this loop), a missing task-ordering
//! edge lets CDEF's bottom-edge padding read picture rows another worker is
//! still writing -- DisjointMut catches the overlap and panics the worker,
//! which (pre-fix) wedged the calling thread in `rav1d_decode_frame`'s
//! condvar forever.
//!
//! Usage: decode_stress <input.obu> <iters> [threads]
//!
//! Prints one heartbeat line per iteration; the driving harness treats a
//! silent, 0-CPU process as the hang and a "panicked" line in the log as
//! the race firing.

use rav1d_safe::src::managed::{Decoder, Settings};
use std::io::Write as _;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: decode_stress <input.obu> <iters> [threads]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).expect("read input");
    let iters: u64 = args[2].parse().expect("iters");
    let threads: u32 = args
        .get(3)
        .map(|s| s.parse().expect("threads"))
        .unwrap_or(0);

    for i in 1..=iters {
        let mut settings = Settings::default();
        settings.threads = threads;
        let mut decoder = Decoder::with_settings(settings).expect("create decoder");
        let mut frames = 0u32;
        if let Some(_f) = decoder.decode(&data).expect("decode") {
            frames += 1;
        }
        while let Some(_f) = decoder.get_frame().expect("drain") {
            frames += 1;
        }
        if frames == 0 {
            for _f in decoder.flush().expect("flush") {
                frames += 1;
            }
        }
        assert!(frames > 0, "no frames decoded");
        println!("iter {i} ok frames={frames}");
        std::io::stdout().flush().unwrap();
    }
    println!("done");
}
