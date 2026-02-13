//! Tango A/B decode benchmarks for comparing across commits.
//!
//! Uses paired benchmarking methodology for high-sensitivity change detection.
//! Detects 1% performance changes within ~1 second.
//!
//! ## Workflow
//!
//! 1. Export baseline from current commit:
//!    ```bash
//!    just tango-export
//!    ```
//!
//! 2. Make changes, then compare:
//!    ```bash
//!    just tango-compare
//!    ```

use rav1d_safe::src::managed::{Decoder, Settings};
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::OnceLock;
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main, IntoBenchmarks};

// ---------------------------------------------------------------------------
// IVF parser
// ---------------------------------------------------------------------------

fn parse_ivf_frames(mut data: &[u8]) -> Vec<Vec<u8>> {
    if data.len() < 32 || &data[..4] != b"DKIF" {
        return Vec::new();
    }
    data = &data[32..];
    let mut frames = Vec::new();
    while data.len() >= 12 {
        let size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        data = &data[12..];
        if data.len() < size {
            break;
        }
        frames.push(data[..size].to_vec());
        data = &data[size..];
    }
    frames
}

// ---------------------------------------------------------------------------
// AVIF OBU extraction
// ---------------------------------------------------------------------------

fn extract_obu(avif_bytes: &[u8]) -> Option<Vec<u8>> {
    let parser = zenavif_parse::AvifParser::from_bytes(avif_bytes).ok()?;
    let data = parser.primary_data().ok()?;
    Some(data.into_owned())
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

fn decode_ivf_frames(obu_frames: &[Vec<u8>]) -> usize {
    let mut dec = Decoder::with_settings(Settings {
        threads: 1,
        ..Default::default()
    })
    .expect("decoder creation");
    let mut n = 0;
    for obu in obu_frames {
        if let Ok(Some(_)) = dec.decode(obu) {
            n += 1;
        }
    }
    if let Ok(remaining) = dec.flush() {
        n += remaining.len();
    }
    n
}

fn decode_obu(obu: &[u8]) -> usize {
    let mut dec = Decoder::with_settings(Settings {
        threads: 1,
        frame_size_limit: 8192 * 8192,
        ..Default::default()
    })
    .expect("decoder creation");
    let mut n = 0;
    if let Ok(Some(_)) = dec.decode(obu) {
        n += 1;
    }
    if let Ok(remaining) = dec.flush() {
        n += remaining.len();
    }
    n
}

// ---------------------------------------------------------------------------
// Vector loading (cached, &'static via OnceLock)
// ---------------------------------------------------------------------------

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors")
        .join("dav1d-test-data")
}

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors")
        .join("bench")
}

fn allintra_frames() -> &'static [Vec<u8>] {
    static CACHE: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let path = vectors_dir().join("8-bit/intra/av1-1-b8-02-allintra.ivf");
        let data = std::fs::read(&path).unwrap_or_default();
        let frames = parse_ivf_frames(&data);
        if frames.is_empty() {
            eprintln!("warning: allintra vector not found at {}", path.display());
        }
        frames
    })
}

fn filmgrain_frames() -> &'static [Vec<u8>] {
    static CACHE: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let path = vectors_dir().join("10-bit/film_grain/av1-1-b10-23-film_grain-50.ivf");
        let data = std::fs::read(&path).unwrap_or_default();
        let frames = parse_ivf_frames(&data);
        if frames.is_empty() {
            eprintln!("warning: filmgrain vector not found at {}", path.display());
        }
        frames
    })
}

fn photo_2k_obu() -> &'static [u8] {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let path = bench_dir().join("photo_2k.avif");
        std::fs::read(&path)
            .ok()
            .and_then(|d| extract_obu(&d))
            .unwrap_or_else(|| {
                eprintln!("warning: photo_2k.avif not found. Run: just generate-bench-avif");
                Vec::new()
            })
    })
}

fn photo_4k_obu() -> &'static [u8] {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let path = bench_dir().join("photo_4k.avif");
        std::fs::read(&path)
            .ok()
            .and_then(|d| extract_obu(&d))
            .unwrap_or_else(|| {
                eprintln!("warning: photo_4k.avif not found. Run: just generate-bench-avif");
                Vec::new()
            })
    })
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn benchmarks() -> impl IntoBenchmarks {
    let mut v = Vec::new();

    if !allintra_frames().is_empty() {
        v.push(benchmark_fn("allintra_8bpc", |b| {
            b.iter(|| black_box(decode_ivf_frames(allintra_frames())))
        }));
    }

    if !filmgrain_frames().is_empty() {
        v.push(benchmark_fn("filmgrain_10bpc", |b| {
            b.iter(|| black_box(decode_ivf_frames(filmgrain_frames())))
        }));
    }

    if !photo_2k_obu().is_empty() {
        v.push(benchmark_fn("photo_2k", |b| {
            b.iter(|| black_box(decode_obu(photo_2k_obu())))
        }));
    }

    if !photo_4k_obu().is_empty() {
        v.push(benchmark_fn("photo_4k", |b| {
            b.iter(|| black_box(decode_obu(photo_4k_obu())))
        }));
    }

    v
}

tango_benchmarks!(benchmarks());
tango_main!();
