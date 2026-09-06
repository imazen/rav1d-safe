//! Interleaved managed AV1 decode with native SIMD and forced scalar.
//!
//! Fixtures are explicit and every decode error fails the benchmark. All YUV
//! planes are compared exactly before timing, including high-bit-depth pixels.
//! Run: cargo bench --bench tier_isolation -- --format=llm

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use zenbench::prelude::*;

#[path = "../tests/ivf_parser.rs"]
mod ivf_parser;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) {
    TierToken::dangerously_disable_token_process_wide(!enabled)
        .expect("benchmark needs testable runtime dispatch; omit target-cpu=native");
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_: bool) {
    panic!("this tier benchmark requires ARM64 or x86-64");
}

fn decode_all(frames: &[ivf_parser::IvfFrame], mut visit: impl FnMut(&Frame)) -> usize {
    let mut settings = Settings::default();
    settings.threads = 1;
    settings.apply_grain = true;
    let mut decoder = Decoder::with_settings(settings).expect("decoder creation");
    let mut count = 0;
    for obu in frames {
        if let Some(frame) = decoder.decode(&obu.data).expect("decode OBU") {
            visit(&frame);
            count += 1;
        }
    }
    for frame in decoder.flush().expect("flush decoder") {
        visit(&frame);
        count += 1;
    }
    assert!(count > 0, "fixture produced no decoded frames");
    count
}

fn snapshot(frame: &Frame) -> (usize, usize, u8, Vec<u16>) {
    let mut samples = Vec::new();
    match frame.planes() {
        Planes::Depth8(p) => {
            for row in p.y().rows() {
                samples.extend(row.iter().map(|&v| u16::from(v)));
            }
            if let Some(u) = p.u() {
                for row in u.rows() {
                    samples.extend(row.iter().map(|&v| u16::from(v)));
                }
            }
            if let Some(v) = p.v() {
                for row in v.rows() {
                    samples.extend(row.iter().map(|&v| u16::from(v)));
                }
            }
        }
        Planes::Depth16(p) => {
            for row in p.y().rows() {
                samples.extend_from_slice(row);
            }
            if let Some(u) = p.u() {
                for row in u.rows() {
                    samples.extend_from_slice(row);
                }
            }
            if let Some(v) = p.v() {
                for row in v.rows() {
                    samples.extend_from_slice(row);
                }
            }
        }
    }
    (
        frame.width() as usize,
        frame.height() as usize,
        frame.bit_depth(),
        samples,
    )
}

fn bench_tiers(suite: &mut Suite) {
    let root = std::env::var_os("RAV1D_BENCH_VECTORS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-vectors/dav1d-test-data")
        });
    for name in [
        "8-bit/data/00000795.ivf",
        "10-bit/data/00000775.ivf",
        "12-bit/data/00000790.ivf",
        "8-bit/film_grain/av1-1-b8-23-film_grain-50.ivf",
        "10-bit/film_grain/av1-1-b10-23-film_grain-50.ivf",
    ] {
        let path = root.join(name);
        let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let frames =
            ivf_parser::parse_all_frames(&mut std::io::Cursor::new(&data)).expect("parse IVF");
        set_simd(true);
        let mut native = Vec::new();
        let count = decode_all(&frames, |frame| native.push(snapshot(frame)));
        set_simd(false);
        let mut scalar = Vec::new();
        assert_eq!(
            decode_all(&frames, |frame| scalar.push(snapshot(frame))),
            count
        );
        assert_eq!(native, scalar, "decoded YUV parity: {name}");
        set_simd(true);
        eprintln!(
            "fixture {name}: {} bytes, {count} decoded frames, exact tier parity passed",
            data.len()
        );
        let frames = std::sync::Arc::new(frames);
        suite.compare(format!("decode_tiers/{name}"), |g| {
            g.throughput(Throughput::Elements(count as u64));
            for (label, enabled) in [("native_simd", true), ("forced_scalar", false)] {
                let frames = frames.clone();
                g.bench(label, move |b| {
                    b.with_input(move || set_simd(enabled))
                        .run(|_| decode_all(black_box(&frames), |_| {}))
                });
            }
        });
    }
}

zenbench::main!(bench_tiers);
