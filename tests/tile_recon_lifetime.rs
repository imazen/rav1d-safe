//! The private tile-reconstruction planes must not outlive the frame.
//!
//! This property is INVISIBLE IN THE PIXELS: a decoder that caches its tile
//! buffers for its whole life decodes byte-identically to one that releases
//! them at frame exit. The corpus gate, the md5 gate and the parity gate are
//! all structurally blind to it, which is exactly how it shipped in #474 as a
//! `tile_columns x plane_bytes` retention (96 MB on `v4k_8tile` 8bpc).
//!
//! So assert on the MECHANISM — `tile_recon::accounting`'s live/peak byte
//! counters — and assert LIVENESS from `peak`, so a run in which the feature
//! silently declined (single tile, one worker, `allow_intrabc`, …) can never
//! be mistaken for a run in which it was released.
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

/// Decode `obu` with `threads` workers and return `live_bytes()` sampled after
/// the decoder has produced its frame but BEFORE it is dropped — the state a
/// long-lived decoder sits in between frames.
fn live_after_frame(obu: &[u8], threads: u32) -> usize {
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.frame_size_limit = 8192 * 8192;
    // Pure tile threading: `n_fc = 1`, which is also what `tile_recon::setup`
    // requires (it declines under frame threading).
    settings.max_frame_delay = 1;
    let mut decoder = Decoder::with_settings(settings).expect("decoder");
    let frame = decoder.decode(obu).expect("decode");
    assert!(frame.is_some(), "vector produced no frame");
    let _ = decoder.flush();
    accounting::live_bytes()
}

/// ONE test, because both counters are process-global: two `#[test]`s would
/// run on two threads of one process and read each other's peak.
#[test]
fn private_tile_planes_are_released_at_frame_exit() {
    let obu = obu_4k();

    // The feature declines below two workers, so nothing is allocated and
    // nothing has to be released. Do this arm FIRST, while `peak` is still 0.
    assert_eq!(accounting::peak_bytes(), 0, "counters must start clean");
    assert_eq!(live_after_frame(&obu, 1), 0);
    assert_eq!(
        accounting::peak_bytes(),
        0,
        "threads=1 must not allocate private tile planes"
    );

    // Liveness: without this the release assertion below passes trivially on a
    // build where the feature never engages.
    let live = live_after_frame(&obu, 8);
    let peak = accounting::peak_bytes();
    assert!(
        peak > 0,
        "the feature never allocated a private plane set at threads=8; this \
         test proves nothing about release until it does — is the vector \
         single-tile?"
    );

    assert_eq!(
        live, 0,
        "a decoder that has decoded one multi-tile frame is still holding \
         {live} bytes of private tile planes (peak {peak})"
    );
    assert_eq!(
        accounting::live_bytes(),
        0,
        "bytes still live after the decoder was dropped"
    );
    eprintln!("peak private tile planes at threads=8: {peak} bytes");
}
