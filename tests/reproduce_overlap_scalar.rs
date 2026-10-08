//! Test tile threading with zero pixel-dispatch flags; token SIMD can remain active.
//! Run: cargo test --release --test reproduce_overlap_scalar -- --ignored --nocapture

use rav1d_safe::src::managed::{CpuLevel, Decoder, Settings};

mod ivf_parser;

#[test]
#[ignore]
fn test_tile_threading_scalar() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/crash_vectors/kodim03_yuv420_8bpc.obu"
    );

    let obu = match std::fs::read(path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("OBU not found: {}", path);
            return;
        }
    };
    eprintln!("OBU size: {} bytes", obu.len());

    for threads in [0, 2, 4, 8] {
        eprintln!("\n--- threads={}, max_frame_delay=1, SCALAR ---", threads);
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 1;
        settings.cpu_level = CpuLevel::Scalar;
        let mut decoder = Decoder::with_settings(settings).expect("create");

        match decoder.decode(&obu) {
            Ok(Some(f)) => {
                eprintln!(
                    "  Decoded: {}x{} @ {}bpc",
                    f.width(),
                    f.height(),
                    f.bit_depth()
                );
            }
            Ok(None) => {
                eprintln!("  Buffered");
                match decoder.flush() {
                    Ok(frames) => {
                        eprintln!("  Flushed {} frames", frames.len());
                        for f in &frames {
                            eprintln!("    {}x{}", f.width(), f.height());
                        }
                    }
                    Err(e) => eprintln!("  Flush err: {:?}", e),
                }
            }
            Err(e) => eprintln!("  Decode err: {:?}", e),
        }
    }
}
