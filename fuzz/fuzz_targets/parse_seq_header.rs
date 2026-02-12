//! Fuzz target: Feed arbitrary bytes through OBU parsing via the managed API.
//!
//! Uses a minimal decoder configuration to exercise the OBU/sequence header
//! parsing paths with minimal overhead. Unlike decode_obu, this target
//! focuses on parser correctness rather than full reconstruction.
//!
//! Seed corpus: test-vectors/dav1d-test-data/oss-fuzz/asan/
#![no_main]

use libfuzzer_sys::fuzz_target;
use rav1d_safe::{Decoder, Settings, InloopFilters, DecodeFrameType};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Disable all optional processing to focus fuzzing on the parser.
    let settings = Settings {
        threads: 1,
        inloop_filters: InloopFilters::none(),
        decode_frame_type: DecodeFrameType::All,
        ..Settings::default()
    };

    let mut decoder = match Decoder::with_settings(settings) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Try decoding — we only care that invalid data doesn't panic.
    let _ = decoder.decode(data);
    let _ = decoder.flush();
});
