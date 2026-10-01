//! Decode performance benchmarks.
//!
//! Scalar vs SIMD (interleaved, same binary — fair comparison):
//!   cargo bench --bench checked_vs_untracked --no-default-features --features "bitdepth_8,bitdepth_16"
//!
//! Tracked vs untracked DisjointMut (cross-build baseline comparison):
//!   cargo clean && cargo bench --bench checked_vs_untracked \
//!     --no-default-features --features "bitdepth_8,bitdepth_16" \
//!     -- --save-baseline=checked
//!   cargo clean && cargo bench --bench checked_vs_untracked \
//!     --no-default-features --features "bitdepth_8,bitdepth_16,untracked" \
//!     -- --baseline=checked

use rav1d_safe::src::managed::{self, CpuLevel, Decoder, Settings};
use std::path::PathBuf;
use std::sync::OnceLock;
use zenbench::prelude::*;

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors")
        .join("bench")
}

fn load_obu(filename: &str) -> Option<Vec<u8>> {
    let path = bench_dir().join(filename);
    let data = std::fs::read(&path).ok()?;
    let parser = zenavif_parse::AvifParser::from_bytes(&data).ok()?;
    let obu = parser.primary_data().ok()?.into_owned();
    let mut s = Settings::default();
    s.frame_size_limit = 8192 * 8192;
    let mut dec = Decoder::with_settings(s).ok()?;
    dec.decode(&obu).ok()?.map(|_| obu)
}

fn obu_4k() -> &'static [u8] {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE.get_or_init(|| load_obu("photo_4k.avif").expect("photo_4k.avif missing"))
}

fn decode(obu: &[u8], cpu: CpuLevel) {
    let mut s = Settings::default();
    s.frame_size_limit = 8192 * 8192;
    s.cpu_level = cpu;
    let mut dec = Decoder::with_settings(s).unwrap();
    dec.decode(obu).unwrap().unwrap();
}

fn bench_decode(suite: &mut Suite) {
    let mode = if managed::is_untracked() {
        "untracked"
    } else {
        "tracked"
    };
    eprintln!("rav1d-safe build mode: {mode}");

    let obu = obu_4k();

    // Scalar vs SIMD — interleaved in the same binary, fair comparison
    suite.compare("decode_4k_scalar_vs_simd", |g| {
        g.throughput(Throughput::Bytes(obu.len() as u64));
        g.config().max_time(std::time::Duration::from_secs(60));

        g.bench("scalar", |b| b.iter(|| decode(obu, CpuLevel::Scalar)));

        g.bench("simd", |b| b.iter(|| decode(obu, CpuLevel::Native)));
    });

    // Single bench for cross-build baseline comparison (tracked vs untracked)
    suite.group("decode_4k_1t", |g| {
        g.throughput(Throughput::Bytes(obu.len() as u64));
        g.config().max_time(std::time::Duration::from_secs(30));

        g.bench("decode", |b| b.iter(|| decode(obu, CpuLevel::Native)));
    });
}

zenbench::main!(bench_decode);
