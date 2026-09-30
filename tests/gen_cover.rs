//! Owned generated-coverage vectors at every CPU level.
//!
//! `tests/gen_vectors/` + `tests/gen_vectors_manifest.tsv` are produced by
//! `scripts/gen_vectors.sh`, which runs zenav1-svt's `gen_cover` example: a
//! fixed matrix of encodes covering the decoder-facing feature space
//! (8/10-bit, 4:2:0/4:4:4/mono, tiles, superres, film grain, sb64/sb128,
//! preset/qp extremes, intra stills and flat low-delay P sequences with
//! synthetic motion). This is the owned, hermetic counterpart of the Argon
//! minimal-cover gate — the streams are generated from code we own and are
//! committed rather than fetched.
//!
//! Pins come from the ASM build's `decode_md5` (dav1d's hand-written
//! assembly), so the manifest is a dav1d-parity oracle: a regression in any
//! tier — scalar included — fails against it. Each stream decodes twice,
//! `apply_grain=false` against `md5_no_film_grain` and `apply_grain=true`
//! against `md5_ref`, at every `CpuLevel::platform_levels()`.
//!
//! Regenerate with `scripts/gen_vectors.sh` after encoder changes; commit
//! streams and manifest together.

#[cfg(debug_assertions)]
compile_error!("gen_cover requires release mode: cargo test --release --test gen_cover");

use rav1d_safe::src::managed::{CpuLevel, Decoder, Settings, Strictness};
use std::path::PathBuf;
use std::sync::Mutex;

#[path = "common/committed_vectors.rs"]
#[allow(dead_code)]
mod committed_vectors;

use committed_vectors::hash_frame;

static CPU_LEVEL_LOCK: Mutex<()> = Mutex::new(());

struct Vector {
    name: String,
    path: PathBuf,
    md5_nofg: String,
    md5_ref: String,
}

fn manifest() -> Vec<Vector> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(root.join("tests/gen_vectors_manifest.tsv"))
        .expect("read gen_vectors_manifest.tsv");
    text.lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("name\t"))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 4, "bad manifest row: {l}");
            Vector {
                name: f[0].to_string(),
                path: root.join("tests/gen_vectors").join(format!("{}.obu", f[0])),
                md5_nofg: f[2].to_string(),
                md5_ref: f[3].to_string(),
            }
        })
        .collect()
}

fn decode_md5_at(data: &[u8], level: CpuLevel, grain: bool) -> (String, Vec<String>) {
    let _guard = CPU_LEVEL_LOCK.lock().unwrap();
    let mut settings = Settings::default();
    settings.strictness = Strictness::Lenient;
    settings.threads = 1;
    settings.max_frame_delay = 1;
    settings.frame_size_limit = 8192 * 8192;
    settings.cpu_level = level;
    settings.apply_grain = grain;
    let mut decoder = Decoder::with_settings(settings).expect("decoder");
    let mut ctx = md5::Context::new();
    let mut errors = Vec::new();

    // Generated streams are section-5 OBU (raw): the decoder consumes a
    // concatenated temporal-unit stream in one call.
    match decoder.decode(data) {
        Ok(Some(frame)) => hash_frame(&frame, &mut ctx),
        Ok(None) => {}
        Err(e) => {
            errors.push(format!("decode error: {e}"));
            return (String::new(), errors);
        }
    }
    loop {
        match decoder.get_frame() {
            Ok(Some(frame)) => hash_frame(&frame, &mut ctx),
            Ok(None) => break,
            Err(e) => {
                errors.push(format!("drain error: {e}"));
                break;
            }
        }
    }
    match decoder.flush() {
        Ok(remaining) => {
            for f in &remaining {
                hash_frame(f, &mut ctx);
            }
        }
        Err(e) => errors.push(format!("flush error: {e}")),
    }
    (format!("{:x}", ctx.finalize()), errors)
}

#[test]
fn gen_cover_streams_bit_exact_across_levels() {
    let vectors = manifest();
    let levels = CpuLevel::platform_levels();
    eprintln!(
        "gen cover: {} vectors × {} levels × 2 grain passes",
        vectors.len(),
        levels.len()
    );

    let mut failures = Vec::new();
    for v in &vectors {
        let data = std::fs::read(&v.path).unwrap_or_else(|e| panic!("read {:?}: {e}", v.path));
        for &level in levels {
            for (grain, expected) in [(false, &v.md5_nofg), (true, &v.md5_ref)] {
                let (got, errors) = decode_md5_at(&data, level, grain);
                if !errors.is_empty() {
                    failures.push(format!(
                        "{} [{level} grain={grain}]: {}",
                        v.name,
                        errors.join("; ")
                    ));
                } else if got != *expected {
                    failures.push(format!(
                        "{} [{level} grain={grain}]: expected {expected}, got {got}",
                        v.name
                    ));
                }
            }
        }
        eprintln!("ok {}", v.name);
    }
    assert!(
        failures.is_empty(),
        "{} generated-vector divergences:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
