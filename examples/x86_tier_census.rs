//! Which x86 SIMD tier did this run actually execute? (`--features __probe_x86tier`)
//!
//! A 766/766 corpus PASS is worthless as evidence about the x86 vector kernels
//! if the run never entered them — the same "permission is not execution" trap
//! that made `enable_cdef = 1` read as "CDEF ran" while it filtered zero
//! blocks. On x86_64 every `safe_simd` dispatcher funnels through
//! `cpu::summon_avx2` / `summon_avx512` / `summon_avx512x`, so counting grants
//! and refusals there is a direct census of the executed tier.
//!
//! It also settles `docs/X64_APPLICABILITY.md` A6 for x86: those three gates
//! consult `rav1d_cpu_flags_mask` BEFORE summoning, so `CpuLevel::Scalar`
//! should collapse `*_grant` to zero here — unlike aarch64, where the
//! dispatchers call `Arm64::summon()` with no mask in the path and
//! `CpuLevel::Scalar` silently does nothing.
//!
//! Usage:
//!   cargo build --release --features __probe_x86tier --example x86_tier_census
//!   ./x86_tier_census <file.ivf> [--threads N]
//!
//! Prints one TSV row per `CpuLevel`, with the decoded md5 so a tier that
//! changes the pixels is impossible to miss.

use rav1d_safe::src::managed::{CpuLevel, Decoder, Frame, Planes, Settings};

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

fn hash_frame(frame: &Frame, ctx: &mut md5::Context) {
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
                for &v in row {
                    ctx.consume(v.to_le_bytes());
                }
            }
            for p in [planes.u(), planes.v()].into_iter().flatten() {
                for row in p.rows() {
                    for &v in row {
                        ctx.consume(v.to_le_bytes());
                    }
                }
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: x86_tier_census <file.ivf> [--threads N]");
    let mut threads = 1u32;
    let rest: Vec<String> = args.collect();
    let mut i = 0;
    while i < rest.len() {
        if rest[i] == "--threads" {
            threads = rest[i + 1].parse().expect("--threads needs a number");
            i += 2;
        } else {
            panic!("unknown arg: {}", rest[i]);
        }
    }

    let file = std::fs::File::open(&path).expect("open ivf");
    let mut reader = std::io::BufReader::new(file);
    let frames = ivf_parser::parse_all_frames(&mut reader).expect("parse ivf");

    #[cfg(all(target_arch = "x86_64", feature = "__probe_x86tier"))]
    let labels = rav1d_safe::src::cpu::tier_census::LABELS;
    #[cfg(not(all(target_arch = "x86_64", feature = "__probe_x86tier")))]
    let labels: [&str; 6] = ["-", "-", "-", "-", "-", "-"];

    println!("arch\tlevel\tthreads\tframes\tmd5\t{}", labels.join("\t"));

    for &level in CpuLevel::platform_levels() {
        #[cfg(all(target_arch = "x86_64", feature = "__probe_x86tier"))]
        rav1d_safe::src::cpu::tier_census::reset();

        let mut settings = Settings::default();
        settings.threads = threads;
        settings.cpu_level = level;
        let mut decoder = Decoder::with_settings(settings).expect("decoder");
        let mut ctx = md5::Context::new();
        let mut n = 0usize;
        for f in &frames {
            match decoder.decode(&f.data) {
                Ok(Some(frame)) => {
                    hash_frame(&frame, &mut ctx);
                    n += 1;
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("decode error at frame {n} level {level:?}: {e}");
                    std::process::exit(1);
                }
            }
        }
        for frame in &decoder.flush().expect("flush") {
            hash_frame(frame, &mut ctx);
            n += 1;
        }
        drop(decoder);

        #[cfg(all(target_arch = "x86_64", feature = "__probe_x86tier"))]
        let counts = rav1d_safe::src::cpu::tier_census::snapshot();
        #[cfg(not(all(target_arch = "x86_64", feature = "__probe_x86tier")))]
        let counts = [0u64; 6];

        let cols: Vec<String> = counts.iter().map(|c| c.to_string()).collect();
        println!(
            "{}\t{level:?}\t{threads}\t{n}\t{:x}\t{}",
            std::env::consts::ARCH,
            ctx.finalize(),
            cols.join("\t")
        );
    }
}
