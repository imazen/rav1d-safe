//! Benchmark: SIMD vs Scalar, single-threaded vs tile-threaded.
//!
//! Measures decode time for real AVIF photos under different configurations:
//! - SIMD (Native CPU features) vs Scalar (no SIMD)
//! - 1 thread vs auto threads (tile threading, max_frame_delay=1)
//!
//! Run: cargo bench --bench threading_bench --no-default-features --features "bitdepth_8,bitdepth_16"
//! Save: cargo bench --bench threading_bench --no-default-features --features "bitdepth_8,bitdepth_16" 2>&1 | tee benchmarks/threading_$(date +%Y%m%d_%H%M%S).txt

use divan::Bencher;
use rav1d_safe::{CpuLevel, Decoder, Settings};
use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;
use zenavif_parse::AvifParser;

fn main() {
    divan::main();
}

// ---------------------------------------------------------------------------
// Config combinations
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct DecodeConfig {
    threads: u32,
    max_frame_delay: u32,
    cpu_level: CpuLevel,
    label: &'static str,
}

impl fmt::Display for DecodeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label)
    }
}

const CONFIGS: &[DecodeConfig] = &[
    DecodeConfig {
        threads: 1,
        max_frame_delay: 1,
        cpu_level: CpuLevel::Native,
        label: "simd_1t",
    },
    DecodeConfig {
        threads: 2,
        max_frame_delay: 1,
        cpu_level: CpuLevel::Native,
        label: "simd_2t",
    },
    DecodeConfig {
        threads: 4,
        max_frame_delay: 1,
        cpu_level: CpuLevel::Native,
        label: "simd_4t",
    },
    DecodeConfig {
        threads: 0,
        max_frame_delay: 1,
        cpu_level: CpuLevel::Native,
        label: "simd_auto",
    },
];

fn configs() -> &'static [DecodeConfig] {
    CONFIGS
}

// ---------------------------------------------------------------------------
// AVIF → OBU extraction
// ---------------------------------------------------------------------------

fn extract_obu(avif_bytes: &[u8]) -> Option<Vec<u8>> {
    let parser = AvifParser::from_bytes(avif_bytes).ok()?;
    let data = parser.primary_data().ok()?;
    Some(data.into_owned())
}

fn decode_obu_with(obu: &[u8], cfg: &DecodeConfig) -> usize {
    let mut settings = Settings::default();
    settings.threads = cfg.threads;
    settings.max_frame_delay = cfg.max_frame_delay;
    settings.cpu_level = cfg.cpu_level;
    settings.frame_size_limit = 8192 * 8192;
    let mut dec = Decoder::with_settings(settings).expect("decoder creation failed");

    let mut n = 0;
    match dec.decode(obu) {
        Ok(Some(_)) => n += 1,
        Ok(None) => {}
        Err(_) => return 0,
    }
    if let Ok(remaining) = dec.flush() {
        n += remaining.len();
    }
    n
}

// ---------------------------------------------------------------------------
// Test vector loading
// ---------------------------------------------------------------------------

struct AvifVector {
    name: String,
    obu: Vec<u8>,
    file_bytes: usize,
}

impl fmt::Display for AvifVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors")
        .join("bench")
}

fn load_avif_vector(filename: &str) -> Option<AvifVector> {
    let path = bench_dir().join(filename);
    let data = std::fs::read(&path).ok()?;
    let file_bytes = data.len();
    let obu = extract_obu(&data)?;

    // Trial decode single-threaded to verify
    let cfg = DecodeConfig {
        threads: 1,
        max_frame_delay: 1,
        cpu_level: CpuLevel::Native,
        label: "trial",
    };
    if decode_obu_with(&obu, &cfg) == 0 {
        eprintln!("warning: {filename} decoded 0 frames, skipping");
        return None;
    }

    Some(AvifVector {
        name: filename.to_string(),
        obu,
        file_bytes,
    })
}

fn vectors() -> &'static [AvifVector] {
    static CACHE: OnceLock<Vec<AvifVector>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let files = ["photo_2k.avif", "photo_4k.avif", "photo_8k.avif"];
        let mut v = Vec::new();
        for f in &files {
            match load_avif_vector(f) {
                Some(tv) => {
                    eprintln!(
                        "loaded {} ({:.1} KB OBU from {:.1} KB AVIF)",
                        tv.name,
                        tv.obu.len() as f64 / 1024.0,
                        tv.file_bytes as f64 / 1024.0,
                    );
                    v.push(tv);
                }
                None => eprintln!("warning: {f} not found. Run: just generate-bench-avif"),
            }
        }
        v
    })
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

#[divan::bench_group(sample_count = 10, sample_size = 1)]
mod decode_threading {
    use super::*;

    #[divan::bench(
        args = vectors(),
        consts = [0, 1],
        ignore = vectors().is_empty(),
    )]
    fn decode<const CFG_IDX: usize>(bencher: Bencher, tv: &AvifVector) {
        let cfg = &configs()[CFG_IDX];
        bencher
            .counter(divan::counter::BytesCount::new(tv.obu.len()))
            .with_inputs(|| cfg)
            .bench_values(|cfg| {
                assert_eq!(
                    decode_obu_with(&tv.obu, cfg),
                    1,
                    "decode failed for {}",
                    cfg.label
                );
            });
    }
}
