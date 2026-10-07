//! THROWAWAY driver: per-call-site DisjointMut registration counts on an IVF.
//!
//! Same idea as `probe_tracker` (which takes an AVIF) but for the stills
//! corpus, which ships as IVF. Decodes the first frame `iters` times at the
//! given thread count, then dumps `site_probe::report` — the count-and-extent
//! table the "fewer registrations" work is aimed at.
//!
//! Build with `--features __probe_sites`; compiles (and does nothing) without.
//!
//! Usage: probe_sites_ivf <input.ivf> <threads> <iters> [all [delay]]
//!
//! By default only the FIRST packet is decoded (a still). With `all`, every packet is
//! pumped (decode, then drain `get_frame`) so inter frames are measured too; the report
//! is then per frame of the whole stream. `delay` sets `max_frame_delay` (default 1).
#![cfg_attr(not(feature = "__probe_sites"), allow(unused))]

use rav1d_safe::src::managed::{Decoder, Settings};
use std::hint::black_box;

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <input.ivf> <threads> <iters>", args[0]);
        std::process::exit(2);
    }
    let path = &args[1];
    let threads: u32 = args[2].parse().expect("threads");
    let iters: u64 = args[3].parse().expect("iters");

    let file = std::fs::File::open(path).expect("open ivf");
    let frames =
        ivf_parser::parse_all_frames(&mut std::io::BufReader::new(file)).expect("parse ivf");
    let first = &frames[0].data;

    let all = args.get(4).is_some_and(|a| a == "all");
    let delay: u32 = args.get(5).map_or(1, |d| d.parse().expect("delay"));
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = delay;
    settings.frame_size_limit = 8192 * 8192;
    let mut dec = Decoder::with_settings(settings).expect("decoder");

    let pump = |dec: &mut Decoder, packets: &[&[u8]]| -> u64 {
        let mut n = 0u64;
        for data in packets {
            if let Some(f) = dec.decode(black_box(data)).expect("decode") {
                black_box(&f);
                n += 1;
            }
            while let Some(f) = dec.get_frame().expect("get_frame") {
                black_box(&f);
                n += 1;
            }
        }
        for f in dec.flush().expect("flush") {
            black_box(&f);
            n += 1;
        }
        n
    };
    let packets: Vec<&[u8]> = if all {
        frames.iter().map(|f| &f.data[..]).collect()
    } else {
        vec![&first[..]]
    };

    // Warmup: allocates every buffer, so first-touch doesn't land in counts.
    let _ = pump(&mut dec, &packets);

    #[cfg(feature = "__probe_sites")]
    rav1d_disjoint_mut::site_probe::reset();
    #[cfg(feature = "__probe_wide")]
    rav1d_disjoint_mut::wide_probe::reset();

    let mut decoded = 0u64;
    for _ in 0..iters {
        decoded += pump(&mut dec, &packets);
    }
    eprintln!("decoded {decoded} frames");
    let iters = decoded.max(1);

    #[cfg(feature = "__probe_sites")]
    print!("{}", rav1d_disjoint_mut::site_probe::report(iters));
    #[cfg(feature = "__probe_wide")]
    print!("{}", rav1d_disjoint_mut::wide_probe::report());
    #[cfg(not(any(feature = "__probe_sites", feature = "__probe_wide")))]
    eprintln!("(built without --features __probe_sites / __probe_wide; no report)");
}
