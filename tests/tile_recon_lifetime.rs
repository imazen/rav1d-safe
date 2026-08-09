//! Lifetime policy for the private tile-reconstruction planes.
//!
//! The policy has two halves and BOTH are load-bearing:
//!
//! 1. **Reused while the decoder is decoding.** Dropping them at every frame
//!    exit is the obvious fix for #474's retention and it measures WORSE on the
//!    exact number it is meant to improve: peak RSS +205.7 MB releasing vs
//!    +100.8 MB reusing on `v4k_8tile` 8bpc t=8 over 20 decodes, because macOS
//!    does not return the freed large regions before the next frame faults a
//!    fresh set in. `RAV1D_TILE_OWNED_RELEASE=1` restores that arm.
//! 2. **Dropped the moment it stops.** `rav1d_flush` — which `Decoder::flush()`,
//!    a seek and an end-of-stream all reach — releases every frame context's
//!    planes, so an idle decoder holds nothing.
//!
//! Neither half is visible in the pixels: a decoder that caches its tile
//! buffers forever decodes byte-identically to one that never does. The corpus
//! gate, the frame-md5 gate and the parity gate are all structurally blind to
//! it, which is how #474 shipped holding `tile_columns x plane_bytes` for the
//! decoder's whole life. So this asserts on the MECHANISM —
//! `tile_recon::accounting`'s live / peak byte counters — and asserts LIVENESS
//! from `peak` first, so a run in which the feature silently declined (single
//! tile, one worker, `allow_intrabc`, over the memory ceiling, …) can never be
//! mistaken for a run in which it was released.
//!
//! Run: `cargo test --release --features tile-owned-recon --test tile_recon_lifetime`

#![cfg(feature = "tile-owned-recon")]
#![cfg(not(debug_assertions))]

use rav1d_safe::src::managed::{Decoder, Settings};
use rav1d_safe::src::tile_recon::accounting;

fn obu_4k() -> Vec<u8> {
    let data = std::fs::read("test-vectors/bench/photo_4k.avif")
        .expect("test-vectors/bench/photo_4k.avif is gitignored; copy it in from another checkout");
    let parser = zenavif_parse::AvifParser::from_bytes(&data).expect("avif parse");
    parser.primary_data().expect("primary").into_owned()
}

fn decoder(threads: u32) -> Decoder {
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.frame_size_limit = 8192 * 8192;
    // Pure tile threading: `n_fc = 1`, which is also what `tile_recon::setup`
    // requires (it declines under frame threading).
    settings.max_frame_delay = 1;
    Decoder::with_settings(settings).expect("decoder")
}

/// ONE test, because both counters are process-global: two `#[test]`s would run
/// on two threads of one process and read each other's peak.
#[test]
fn private_tile_planes_are_reused_across_frames_and_dropped_on_flush() {
    let obu = obu_4k();

    // -- threads=1: the feature declines, so nothing is allocated at all.
    // First, while `peak` is still provably 0.
    assert_eq!(accounting::peak_bytes(), 0, "counters must start clean");
    {
        let mut d = decoder(1);
        assert!(d.decode(&obu).expect("decode").is_some());
        let _ = d.flush();
    }
    assert_eq!(
        accounting::peak_bytes(),
        0,
        "threads=1 must not allocate private tile planes"
    );

    // -- threads=8: allocated, held across the decode, dropped on flush.
    let mut d = decoder(8);
    assert!(d.decode(&obu).expect("decode").is_some());

    let held = accounting::live_bytes();
    let peak = accounting::peak_bytes();
    // LIVENESS. Without this every assertion below passes trivially on a build
    // where the feature never engaged.
    assert!(
        peak > 0,
        "the feature never allocated a private plane set at threads=8; this \
         test proves nothing until it does — is the vector single-tile, or is \
         it over RAV1D_TILE_OWNED_MAX_MB?"
    );
    assert_eq!(
        held, peak,
        "planes must survive frame exit so the next frame reuses them instead \
         of faulting a second generation in beside the first"
    );

    let _ = d.flush();
    assert_eq!(
        accounting::live_bytes(),
        0,
        "a flushed decoder is still holding private tile planes"
    );

    drop(d);
    assert_eq!(
        accounting::live_bytes(),
        0,
        "bytes still live after the decoder was dropped"
    );
    eprintln!("peak private tile planes at threads=8: {peak} bytes");
}
