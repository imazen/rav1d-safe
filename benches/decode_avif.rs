//! AVIF decode benchmarks using real photographs at 2K/4K/8K resolution.
//!
//! Requires AVIF test files in `test-vectors/bench/`.
//! Generate them with: `just generate-bench-avif`
//!
//! Compare safety levels:
//! ```bash
//! cargo bench --bench decode_avif                              # checked (default)
//! cargo bench --bench decode_avif --features unchecked         # unchecked indexing
//! cargo bench --bench decode_avif --features asm               # hand-written asm
//! ```

use divan::Bencher;
use rav1d_safe::src::managed::{Decoder, Settings};
use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;
use zenavif_parse::AvifParser;

fn main() {
    divan::main();
}

// ---------------------------------------------------------------------------
// AVIF → OBU extraction
// ---------------------------------------------------------------------------

/// Extract raw AV1 OBU data from an AVIF file.
fn extract_obu(avif_bytes: &[u8]) -> Option<Vec<u8>> {
    let parser = AvifParser::from_bytes(avif_bytes).ok()?;
    let data = parser.primary_data().ok()?;
    Some(data.into_owned())
}

/// Decode OBU data through the managed API. Returns frame count.
fn decode_obu(obu: &[u8]) -> usize {
    let mut dec = Decoder::with_settings(Settings {
        threads: 1,
        frame_size_limit: 8192 * 8192, // Allow up to 8K square
        ..Default::default()
    })
    .expect("decoder creation failed");

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

    // Trial decode to make sure it works
    if decode_obu(&obu) == 0 {
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
                None => eprintln!("warning: {f} not found or decode failed. Run: just generate-bench-avif"),
            }
        }
        if v.is_empty() {
            eprintln!(
                "warning: no AVIF bench vectors found in {}",
                bench_dir().display()
            );
        }
        v
    })
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

#[divan::bench_group(sample_count = 10, sample_size = 1)]
mod decode_avif {
    use super::*;

    #[divan::bench(
        args = vectors(),
        ignore = vectors().is_empty(),
    )]
    fn decode(bencher: Bencher, tv: &AvifVector) {
        bencher
            .counter(divan::counter::BytesCount::new(tv.obu.len()))
            .bench(|| decode_obu(&tv.obu));
    }
}
