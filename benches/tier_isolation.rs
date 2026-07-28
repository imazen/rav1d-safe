//! SIMD-tier isolation: the native top tier vs the same decoder forced to scalar.
//!
//! `decode.rs` measures absolute decode throughput and `checked_vs_unchecked.rs`
//! compares indexing modes. Neither can tell you whether the ~31 NEON kernel
//! files are earning their keep — a kernel slower than its own scalar fallback
//! is invisible in both. This bench decodes the same vectors twice, once with
//! the native SIMD token disabled. (The same gap in linear-srgb was hiding a
//! real regression.)
//!
//! Run: `bash scripts/download-test-vectors.sh && cargo bench --bench tier_isolation`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

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
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));
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
            let rel = path.strip_prefix(vectors_dir()).unwrap_or(&path).display();
            eprintln!("skipping {rel} (decode failed or panicked)");
            continue;
        }
        let total_bytes: usize = frames.iter().map(|f| f.len()).sum();
        let name = path
            .strip_prefix(vectors_dir())
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
        } else if path.extension().is_some_and(|e| e == "ivf")
            && let Ok(meta) = path.metadata()
        {
            out.push((path, meta.len()));
        }
    }
}

/// Decode all OBU frames through the managed API. Returns frame count.
fn decode_all(obu_frames: &[Vec<u8>]) -> usize {
    let mut settings = Settings::default();
    settings.threads = 1;
    let mut dec = Decoder::with_settings(settings).expect("decoder creation failed");

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
// SIMD tier control
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

/// True when the tier can actually be turned off AND the decoder has a path to
/// fall back to. If not, the "scalar" arm would either re-measure the SIMD path
/// or panic, so both arms are skipped instead.
fn toggleable() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        // aarch64 has NO scalar fallback in this decoder, by design.
        // `archmage::Arm64` is a type alias for `NeonToken`, and AArch64
        // guarantees NEON, so src/safe_simd/{mc,filmgrain}_arm.rs call
        // `Arm64::summon().unwrap()` unconditionally (9 sites). That unwrap is
        // sound in production — NEON is mandatory on every AArch64 CPU — but it
        // means disabling the token leaves no code path to run: the decoder
        // panics rather than degrading. So NEON's contribution here cannot be
        // measured by tier isolation; the ARM build is 100% NEON by
        // construction, which is the intended design, not a gap.
        //
        // On x86_64 the comparison is meaningful: AVX2 is genuinely optional
        // and a scalar path exists for CPUs without it.
        if cfg!(target_arch = "aarch64") {
            eprintln!(
                "[tier_isolation] aarch64: this decoder has no scalar fallback \
                 (Arm64 = NeonToken, unwrapped unconditionally), so NEON cannot \
                 be isolated — every ARM path is already NEON. Skipping. Run \
                 this bench on x86_64 to compare AVX2 against scalar."
            );
            return false;
        }
        let ok = set_simd(true) && set_simd(false);
        set_simd(true);
        if !ok {
            eprintln!(
                "[tier_isolation] SIMD tier is not toggleable here (compile-time \
                 guaranteed). Drop -C target-cpu=native and ensure \
                 archmage/testable_dispatch. Skipping."
            );
        }
        ok
    })
}

// ---------------------------------------------------------------------------
// Benchmark groups — each vector decoded with SIMD on, then forced scalar
// ---------------------------------------------------------------------------

#[divan::bench_group(sample_count = 10, sample_size = 1)]
mod tier_8bit {
    use super::*;

    #[divan::bench(args = vectors_8bit(), ignore = vectors_8bit().is_empty() || !toggleable())]
    fn simd(bencher: Bencher, tv: &TestVector) {
        set_simd(true);
        bencher
            .counter(divan::counter::BytesCount::new(tv.total_bytes))
            .bench(|| decode_all(&tv.frames));
    }

    #[divan::bench(args = vectors_8bit(), ignore = vectors_8bit().is_empty() || !toggleable())]
    fn scalar(bencher: Bencher, tv: &TestVector) {
        set_simd(false);
        bencher
            .counter(divan::counter::BytesCount::new(tv.total_bytes))
            .bench(|| decode_all(&tv.frames));
        set_simd(true);
    }
}

#[divan::bench_group(sample_count = 10, sample_size = 1)]
mod tier_10bit {
    use super::*;

    #[divan::bench(args = vectors_10bit(), ignore = vectors_10bit().is_empty() || !toggleable())]
    fn simd(bencher: Bencher, tv: &TestVector) {
        set_simd(true);
        bencher
            .counter(divan::counter::BytesCount::new(tv.total_bytes))
            .bench(|| decode_all(&tv.frames));
    }

    #[divan::bench(args = vectors_10bit(), ignore = vectors_10bit().is_empty() || !toggleable())]
    fn scalar(bencher: Bencher, tv: &TestVector) {
        set_simd(false);
        bencher
            .counter(divan::counter::BytesCount::new(tv.total_bytes))
            .bench(|| decode_all(&tv.frames));
        set_simd(true);
    }
}
