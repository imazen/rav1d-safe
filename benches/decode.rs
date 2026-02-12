//! Decode benchmarks using real AV1 test vectors.
//!
//! Discovers IVF files from `target/test-vectors/dav1d-test-data/` at runtime.
//! If vectors are missing, benchmarks are skipped with a warning.
//!
//! Compare safety levels:
//! ```bash
//! cargo bench --bench decode                              # checked (default)
//! cargo bench --bench decode --features unchecked          # unchecked indexing
//! cargo bench --bench decode --features asm                # hand-written asm
//! ```

use divan::Bencher;
use rav1d_safe::src::managed::{Decoder, Settings};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn main() {
    divan::main();
}

// ---------------------------------------------------------------------------
// IVF parser (inlined from tests/ivf_parser.rs — benches can't use test modules)
// ---------------------------------------------------------------------------

fn parse_ivf_header(r: &mut &[u8]) -> Option<()> {
    if r.len() < 32 {
        return None;
    }
    let hdr = &r[..32];
    *r = &r[32..];
    if &hdr[0..4] != b"DKIF" || &hdr[8..12] != b"AV01" {
        return None;
    }
    Some(())
}

fn parse_ivf_frames(mut data: &[u8]) -> Vec<Vec<u8>> {
    if parse_ivf_header(&mut data).is_none() {
        return Vec::new();
    }
    let mut frames = Vec::new();
    while data.len() >= 12 {
        let size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        // skip 12-byte frame header
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
// Test vector discovery and caching
// ---------------------------------------------------------------------------

/// A pre-parsed test vector: name + OBU frames ready to feed the decoder.
struct TestVector {
    name: String,
    frames: Vec<Vec<u8>>,
    total_bytes: usize,
}

/// Display impl so divan can label benchmark args.
impl fmt::Display for TestVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

fn vectors_dir() -> PathBuf {
    // Try crate root first, then target/ for backwards compatibility
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_path = crate_root.join("test-vectors").join("dav1d-test-data");
    if root_path.exists() {
        return root_path;
    }
    let target = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| crate_root.join("target"));
    target.join("test-vectors").join("dav1d-test-data")
}

/// Max IVF file size to benchmark (100 KB). Vectors larger than this
/// often take many seconds per decode, making the benchmark suite too slow.
const MAX_VECTOR_BYTES: u64 = 100_000;

/// Discover IVF files from a subdirectory, sorted largest-first, capped at `limit`.
fn discover_vectors(subdir: &str, limit: usize) -> Vec<TestVector> {
    let dir = vectors_dir().join(subdir);
    if !dir.exists() {
        return Vec::new();
    }

    // Collect all .ivf files with their sizes
    let mut entries: Vec<(PathBuf, u64)> = Vec::new();
    collect_ivf_files(&dir, &mut entries);

    // Filter out huge vectors, then sort largest-first.
    // We try more candidates than `limit` because some vectors may fail validation.
    entries.retain(|(_, size)| *size <= MAX_VECTOR_BYTES);
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.truncate(limit * 10);

    let mut result = Vec::with_capacity(limit);
    for (path, _size) in entries {
        if result.len() >= limit {
            break;
        }
        let Ok(data) = std::fs::read(&path) else {
            continue;
        };
        let frames = parse_ivf_frames(&data);
        if frames.is_empty() {
            continue;
        }
        // Trial decode to filter out vectors that panic or produce no frames
        if !validate_vector(&frames) {
            let rel = path.strip_prefix(&vectors_dir()).unwrap_or(&path).display();
            eprintln!("skipping {rel} (decode failed or panicked)");
            continue;
        }
        let total_bytes: usize = frames.iter().map(|f| f.len()).sum();
        let name = path
            .strip_prefix(&vectors_dir())
            .unwrap_or(&path)
            .display()
            .to_string()
            .replace('\\', "/");
        result.push(TestVector {
            name,
            frames,
            total_bytes,
        });
    }
    result
}

fn collect_ivf_files(dir: &Path, out: &mut Vec<(PathBuf, u64)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_ivf_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "ivf") {
            if let Ok(meta) = path.metadata() {
                out.push((path, meta.len()));
            }
        }
    }
}

/// Decode all OBU frames through the managed API. Returns frame count.
fn decode_all(obu_frames: &[Vec<u8>]) -> usize {
    let mut dec = Decoder::with_settings(Settings {
        threads: 1,
        ..Default::default()
    })
    .expect("decoder creation failed");

    let mut n = 0;
    for obu in obu_frames {
        match dec.decode(obu) {
            Ok(Some(_frame)) => n += 1,
            Ok(None) => {}
            Err(_) => {} // skip decode errors on malformed vectors
        }
    }
    if let Ok(remaining) = dec.flush() {
        n += remaining.len();
    }
    n
}

/// Try to decode a vector in a separate thread so panics don't kill the process.
/// Returns false if decoding panics or fails to produce any frames.
fn validate_vector(obu_frames: &[Vec<u8>]) -> bool {
    let frames = obu_frames.to_vec();
    let result = std::thread::spawn(move || decode_all(&frames)).join();
    matches!(result, Ok(n) if n > 0)
}

// ---------------------------------------------------------------------------
// Cached vector sets (loaded once per process)
// ---------------------------------------------------------------------------

const MAX_VECTORS_PER_GROUP: usize = 5;

fn vectors_8bit() -> &'static [TestVector] {
    static CACHE: OnceLock<Vec<TestVector>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let v = discover_vectors("8-bit/data", MAX_VECTORS_PER_GROUP);
        if v.is_empty() {
            eprintln!(
                "warning: no 8-bit test vectors found. Run: bash scripts/download-test-vectors.sh"
            );
        }
        v
    })
}

fn vectors_10bit() -> &'static [TestVector] {
    static CACHE: OnceLock<Vec<TestVector>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let v = discover_vectors("10-bit/data", MAX_VECTORS_PER_GROUP);
        if v.is_empty() {
            eprintln!(
                "warning: no 10-bit test vectors found. Run: bash scripts/download-test-vectors.sh"
            );
        }
        v
    })
}

fn vectors_filmgrain() -> &'static [TestVector] {
    static CACHE: OnceLock<Vec<TestVector>> = OnceLock::new();
    CACHE.get_or_init(|| {
        // Collect from both 8-bit and 10-bit film_grain dirs
        let mut v = discover_vectors("8-bit/film_grain", MAX_VECTORS_PER_GROUP);
        let remaining = MAX_VECTORS_PER_GROUP.saturating_sub(v.len());
        if remaining > 0 {
            v.extend(discover_vectors("10-bit/film_grain", remaining));
        }
        if v.is_empty() {
            eprintln!(
                "warning: no film grain test vectors found. Run: bash scripts/download-test-vectors.sh"
            );
        }
        v
    })
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

#[divan::bench_group(sample_count = 20, sample_size = 1)]
mod decode_8bit {
    use super::*;

    #[divan::bench(
        args = vectors_8bit(),
        ignore = vectors_8bit().is_empty(),
    )]
    fn decode(bencher: Bencher, tv: &TestVector) {
        bencher
            .counter(divan::counter::BytesCount::new(tv.total_bytes))
            .bench(|| decode_all(&tv.frames));
    }
}

#[divan::bench_group(sample_count = 20, sample_size = 1)]
mod decode_10bit {
    use super::*;

    #[divan::bench(
        args = vectors_10bit(),
        ignore = vectors_10bit().is_empty(),
    )]
    fn decode(bencher: Bencher, tv: &TestVector) {
        bencher
            .counter(divan::counter::BytesCount::new(tv.total_bytes))
            .bench(|| decode_all(&tv.frames));
    }
}

#[divan::bench_group(sample_count = 20, sample_size = 1)]
mod decode_filmgrain {
    use super::*;

    #[divan::bench(
        args = vectors_filmgrain(),
        ignore = vectors_filmgrain().is_empty(),
    )]
    fn decode(bencher: Bencher, tv: &TestVector) {
        bencher
            .counter(divan::counter::BytesCount::new(tv.total_bytes))
            .bench(|| decode_all(&tv.frames));
    }
}
