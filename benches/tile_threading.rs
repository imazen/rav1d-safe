//! Tile threading benchmark using zenbench interleaved measurement.
//!
//! Compares 1-thread vs 2-thread vs 4-thread tile decode on real AVIF photos.
//! Interleaved execution ensures thermal/load parity between configurations.
//!
//! Tile threading works in every build; `untracked` removes the tracker and the
//! copy path and is much faster:
//!   cargo bench --bench tile_threading --no-default-features --features "bitdepth_8,bitdepth_16,untracked"

use rav1d_safe::src::managed::{CpuLevel, Decoder, Settings};
use std::path::PathBuf;
use std::sync::OnceLock;
use zenbench::prelude::*;

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors")
        .join("bench")
}

struct TestVector {
    name: &'static str,
    obu: Vec<u8>,
}

fn load_vector(filename: &'static str) -> Option<TestVector> {
    let path = bench_dir().join(filename);
    let data = std::fs::read(&path).ok()?;
    let parser = zenavif_parse::AvifParser::from_bytes(&data).ok()?;
    let obu = parser.primary_data().ok()?.into_owned();

    // Trial decode to verify
    let mut settings = Settings::default();
    settings.frame_size_limit = 8192 * 8192;
    let mut dec = Decoder::with_settings(settings).ok()?;
    dec.decode(&obu).ok()?.as_ref()?;
    Some(TestVector {
        name: filename.split('.').next().unwrap_or(filename),
        obu,
    })
}

fn vectors() -> &'static [TestVector] {
    static CACHE: OnceLock<Vec<TestVector>> = OnceLock::new();
    CACHE.get_or_init(|| {
        ["photo_4k.avif", "photo_8k.avif"]
            .iter()
            .filter_map(|f| {
                let v = load_vector(f);
                if v.is_none() {
                    eprintln!("warning: {f} not found in test-vectors/bench/");
                }
                v
            })
            .collect()
    })
}

fn decode_obu(obu: &[u8], threads: u32) -> usize {
    decode_obu_cpu(obu, threads, CpuLevel::Native)
}

fn decode_obu_scalar(obu: &[u8], threads: u32) -> usize {
    decode_obu_cpu(obu, threads, CpuLevel::Scalar)
}

fn decode_obu_cpu(obu: &[u8], threads: u32, cpu: CpuLevel) -> usize {
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = 1; // tile threading only
    settings.frame_size_limit = 8192 * 8192;
    settings.cpu_level = cpu;

    let mut dec = Decoder::with_settings(settings).expect("decoder creation failed");
    let mut n = 0;
    match dec.decode(obu) {
        Ok(Some(_)) => n += 1,
        Ok(None) => {}
        Err(_) => return 0,
    }
    if let Ok(frames) = dec.flush() {
        n += frames.len();
    }
    n
}

fn bench_tile_threading(suite: &mut Suite) {
    let vecs = vectors();
    if vecs.is_empty() {
        eprintln!("No test vectors found. Run: just generate-bench-avif");
        return;
    }

    for tv in vecs {
        suite.compare(format!("tile_{}", tv.name), |group| {
            group.throughput(Throughput::Bytes(tv.obu.len() as u64));
            group.config().max_time(std::time::Duration::from_secs(30));

            group.bench("1t", |b| {
                let obu = &tv.obu;
                b.iter(|| {
                    assert_eq!(decode_obu(obu, 1), 1);
                })
            });

            group.bench("2t", |b| {
                let obu = &tv.obu;
                b.iter(|| {
                    assert_eq!(decode_obu(obu, 2), 1);
                })
            });

            group.bench("4t", |b| {
                let obu = &tv.obu;
                b.iter(|| {
                    assert_eq!(decode_obu(obu, 4), 1);
                })
            });

            group.bench("scalar_1t", |b| {
                let obu = &tv.obu;
                b.iter(|| {
                    assert_eq!(decode_obu_scalar(obu, 1), 1);
                })
            });

            group.bench("scalar_4t", |b| {
                let obu = &tv.obu;
                b.iter(|| {
                    assert_eq!(decode_obu_scalar(obu, 4), 1);
                })
            });
        });
    }
}

zenbench::main!(bench_tile_threading);
