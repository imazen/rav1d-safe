//! Per-vector inverse-transform shape census (`__ablate` feature).
//!
//! "Which transform shapes does this bitstream actually ask for, and which of
//! them did a SIMD kernel take?" A profiler cannot answer that: an inlined
//! scalar fallback and an inlined SIMD dispatcher land on the same symbol, and
//! a shape that is never coded costs nothing and shows up nowhere.
//!
//! This is the tool that refuted issue #455's open item 5 (16bpc itx above
//! 16x16). On `L3840x2160_420_10b` the >16x16 fallback is 20 calls out of
//! 272,949 — 0.15% of coefficient area — and 0 on `v4k_8tile_10b`.
//!
//! Usage:
//!   cargo build --release --features __ablate --example itx_shape_census
//!   ./target/release/examples/itx_shape_census <in.avif> [more.avif ...]
//!
//! Emits TSV to stdout: `vector  depth  path  shape  calls  coeff_area`.

use rav1d_safe::src::ablate;
use rav1d_safe::src::managed::{Decoder, Settings};

fn main() {
    // A harness MUST assert this: without `__ablate` every counter stays zero
    // and the run reads as "this bitstream codes no transforms".
    assert!(
        ablate::ENABLED,
        "build with --features __ablate; without it the census is all zeros"
    );
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: itx_shape_census <in.avif> ...");
    println!("vector\tdepth\tpath\tshape\tcalls\tcoeff_area");
    for path in &args {
        let name = std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        let bytes = std::fs::read(path).expect("read input");
        let parser = zenavif_parse::AvifParser::from_bytes(&bytes).expect("avif parse");
        let obu = parser
            .primary_data()
            .expect("avif primary item")
            .into_owned();

        ablate::itx_shape_reset();
        let mut settings = Settings::default();
        settings.threads = 1;
        settings.frame_size_limit = 8192 * 8192;
        let mut decoder = Decoder::with_settings(settings).expect("decoder");
        decoder.decode(&obu).expect("decode");
        let _ = decoder.flush();

        for line in ablate::itx_shape_report().lines().skip(1) {
            println!("{name}\t{line}");
        }
    }
}
