//! Fuzz target: Feed arbitrary bytes through the managed safe Rust decoder.
//!
//! Exercises the full decode pipeline: OBU parsing, entropy decoding,
//! reconstruction, loop filters, and film grain application.
//!
//! Seed corpus: test-vectors/dav1d-test-data/oss-fuzz/asan/
#![no_main]

use libfuzzer_sys::fuzz_target;
use rav1d_safe::Decoder;

fuzz_target!(|data: &[u8]| {
    // Skip empty inputs — nothing to decode.
    if data.is_empty() {
        return;
    }

    // Create a single-threaded decoder (deterministic, no thread pool overhead).
    let mut decoder = match Decoder::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    // Feed the fuzzed data as a single OBU chunk.
    // The decoder should handle malformed data gracefully (return Err, not panic).
    let _ = decoder.decode(data);

    // Drain any buffered frames.
    let _ = decoder.flush();
});
