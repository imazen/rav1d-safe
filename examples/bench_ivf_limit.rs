//! Whole-process IVF decode runner, shaped so `dav1d --limit N` is a
//! drop-in comparison arm.
//!
//! `bench_ab_decode` re-decodes ONE key-frame OBU lifted out of an AVIF, which
//! is the right shape for the 4K gap grid and the wrong shape for a real
//! multi-frame stream: it never exercises inter prediction, reference
//! management, or (the reason this file exists) a bitstream that actually
//! switches loop restoration on. The standing gap grid is blind to loop
//! restoration for exactly that reason — `v4k_8tile{,_10b}` do zero LR.
//!
//! So: decode the first `limit` frames of an IVF at `threads`, print a
//! checksum, exit. The driver times the *whole process* at two frame counts
//! and fits `total = alpha + beta * frames`, which cancels binary load, IVF
//! parse and decoder construction — the same instrument, on the same input
//! file, that `verify_gap.sh` already points at `dav1d --limit`.
//!
//! Usage:
//!   bench_ivf_limit <input.ivf> <threads> <limit> [label]
//!
//! Emits to stdout, tab-separated:
//!   CHECKSUM  label  file  threads  limit  md5-of-all-decoded-planes
//!
//! NO timing is printed: the point is that the *driver's* clock is the only
//! instrument, so both arms are measured by the same one.

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;

fn hash_frame(ctx: &mut md5::Context, frame: &Frame) {
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                ctx.consume(row);
            }
            for p in [planes.u(), planes.v()].into_iter().flatten() {
                for row in p.rows() {
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
            for p in [planes.u(), planes.v()].into_iter().flatten() {
                for row in p.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <input.ivf> <threads> <limit> [label]", args[0]);
        std::process::exit(2);
    }
    let path = &args[1];
    let threads: u32 = args[2].parse().expect("threads");
    let limit: usize = args[3].parse().expect("limit");
    let label = args.get(4).map(String::as_str).unwrap_or("rav1d");
    let file = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string();

    let f = File::open(path).expect("open input");
    let mut reader = BufReader::new(f);
    let frames = ivf_parser::parse_all_frames(&mut reader).expect("parse IVF");

    let mut settings = Settings::default();
    settings.threads = threads;
    settings.frame_size_limit = 8192 * 8192;
    // Tile threading only, matching `dav1d --framedelay 1` and the rest of the
    // gap campaign. The checked build already forces n_fc = 1, so this only
    // matters if someone points an `unchecked` build at the same script.
    settings.max_frame_delay = 1;
    let mut dec = Decoder::with_settings(settings).expect("decoder");

    let mut ctx = md5::Context::new();
    let mut decoded = 0usize;
    // `limit` counts OUTPUT frames, exactly like dav1d's `--limit`, so the two
    // arms do the same amount of work when the stream has show-existing frames.
    for ivf_frame in &frames {
        if decoded >= limit {
            break;
        }
        match dec.decode(&ivf_frame.data) {
            Ok(Some(frame)) => {
                hash_frame(&mut ctx, &frame);
                black_box(&frame);
                decoded += 1;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("decode error: {e:?}");
                std::process::exit(3);
            }
        }
    }
    if decoded < limit {
        if let Ok(rest) = dec.flush() {
            for frame in &rest {
                if decoded >= limit {
                    break;
                }
                hash_frame(&mut ctx, frame);
                black_box(frame);
                decoded += 1;
            }
        }
    }
    println!(
        "CHECKSUM\t{label}\t{file}\t{threads}\t{decoded}\t{:x}",
        ctx.finalize()
    );
}
