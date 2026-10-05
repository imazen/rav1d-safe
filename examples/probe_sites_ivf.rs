//! THROWAWAY driver: per-call-site DisjointMut registration counts on an IVF.
//!
//! Same idea as `probe_tracker` (which takes an AVIF) but for the stills
//! corpus, which ships as IVF. Decodes the first frame `iters` times at the
//! given thread count, then dumps `site_probe::report` — the count-and-extent
//! table the "fewer registrations" work is aimed at.
//!
//! Build with `--features __probe_sites`; compiles (and does nothing) without.
//!
//! Usage: probe_sites_ivf <input.ivf> <threads> <iters>
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

    let mut settings = Settings::default();
    settings.threads = threads;
    settings.frame_size_limit = 8192 * 8192;
    let mut dec = Decoder::with_settings(settings).expect("decoder");

    // Warmup: allocates every buffer, so first-touch doesn't land in counts.
    let _ = dec.decode(first).expect("warmup").expect("frame");
    let _ = dec.flush();

    #[cfg(feature = "__probe_sites")]
    rav1d_disjoint_mut::site_probe::reset();
    #[cfg(feature = "__probe_wide")]
    rav1d_disjoint_mut::wide_probe::reset();

    for _ in 0..iters {
        let f = dec.decode(black_box(first)).expect("decode");
        black_box(&f);
        drop(f);
        let _ = dec.flush();
    }

    #[cfg(feature = "__probe_sites")]
    print!("{}", rav1d_disjoint_mut::site_probe::report(iters));
    #[cfg(feature = "__probe_wide")]
    print!("{}", rav1d_disjoint_mut::wide_probe::report());
    #[cfg(not(any(feature = "__probe_sites", feature = "__probe_wide")))]
    eprintln!("(built without --features __probe_sites / __probe_wide; no report)");
}
