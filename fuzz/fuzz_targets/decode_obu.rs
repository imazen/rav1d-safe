//! Fuzz target: Feed arbitrary bytes through the managed safe Rust decoder.
//!
//! Exercises the full decode pipeline: OBU parsing, entropy decoding,
//! reconstruction, loop filters, and film grain application.
//!
//! Seed corpus: test-vectors/dav1d-test-data/oss-fuzz/asan/
#![no_main]

use libfuzzer_sys::fuzz_target;
use rav1d_safe::src::managed::{Decoder, Settings};

fuzz_target!(|data: &[u8]| {
    // Skip empty inputs — nothing to decode.
    if data.is_empty() {
        return;
    }

    // Create a single-threaded decoder with tight size limits for fuzzing.
    // 256×256 = 65536 pixels is plenty for exercising all code paths
    // without triggering OOM from adversarial frame dimensions.
    let mut decoder = match Decoder::with_settings(Settings {
        frame_size_limit: 256 * 256,
        ..Settings::default()
    }) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Feed the fuzzed data as a single OBU chunk.
    // The decoder should handle malformed data gracefully (return Err, not panic).
    let _ = decoder.decode(data);

    // Drain any buffered frames.
    let _ = decoder.flush();
});
