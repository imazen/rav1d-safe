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
//!
//! **`RAV1D_MD5=1` OPTS IN to hashing, and a timed run must NOT set it.**
//! Hashing is a per-pixel `md5::Context::consume`, and at 16bpc it is a
//! per-*pixel* `to_le_bytes` feed — on `10-bit/issues/318_tx_4x4` that alone
//! was 8.5 of 17.0 ms/frame, i.e. HALF the measured cost, against a
//! `dav1d --muxer null` arm that hashes nothing. Measured that way the gap to
//! dav1d reads 8.5x when it is 4.2x. Run the identity check as its own
//! (untimed) pass.

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

use rav1d_safe::src::managed::{Decoder, Frame, InloopFilters, Planes, Settings};
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;

/// `RAV1D_INLOOP` — the same axis as `dav1d --inloopfilters <str>`, so a
/// filter's cost can be attributed on BOTH decoders with one instrument
/// instead of profiling ours and arguing about theirs.
///
/// Accepts `all` (default), `none`, `nodeblock`, `nocdef`, `norestoration`.
/// The names are dav1d's on purpose: a cell is then literally the same string
/// on both arms, which is one less place for the two sides to drift apart.
///
/// It CHANGES OUTPUT PIXELS — this is an attribution arm, never a correctness
/// one. Do not compare an md5 across two values of it.
fn inloop_from_env() -> InloopFilters {
    let all = InloopFilters::all();
    match std::env::var("RAV1D_INLOOP").as_deref().unwrap_or("all") {
        "all" => all,
        "none" => InloopFilters::none(),
        "nodeblock" => InloopFilters::CDEF.union(InloopFilters::RESTORATION),
        "nocdef" => InloopFilters::DEBLOCK.union(InloopFilters::RESTORATION),
        "norestoration" => InloopFilters::DEBLOCK.union(InloopFilters::CDEF),
        other => panic!(
            "RAV1D_INLOOP={other}: expected all|none|nodeblock|nocdef|norestoration \
             (dav1d --inloopfilters spelling)"
        ),
    }
}

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
    settings.inloop_filters = inloop_from_env();
    let mut dec = Decoder::with_settings(settings).expect("decoder");

    // Off by default — see the module docs. `dav1d --muxer null` hashes
    // nothing, so hashing here would put a per-pixel cost on one arm only.
    let want_md5 = std::env::var("RAV1D_MD5").as_deref() == Ok("1");
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
                if want_md5 {
                    hash_frame(&mut ctx, &frame);
                }
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
                if want_md5 {
                    hash_frame(&mut ctx, frame);
                }
                black_box(frame);
                decoded += 1;
            }
        }
    }
    let digest = if want_md5 {
        format!("{:x}", ctx.finalize())
    } else {
        "nohash".to_string()
    };
    println!("CHECKSUM\t{label}\t{file}\t{threads}\t{decoded}\t{digest}");
}
