//! #423 regression: `Decoder::flush()` is a DRAIN, then a reset — it must return
//! every frame the decoder still owes, never discard it.
//!
//! `flush()` used to call `rav1d_flush` (dav1d reset semantics: drop pending
//! input, drop the ready output picture, drop every frame-threading
//! `out_delayed` slot) and only then loop `rav1d_get_picture`, which by that
//! point had nothing left to return. Two consequences, both silent frame loss:
//!
//! * With frame threading (`threads >= 2`, `max_frame_delay != 1` — needs the
//!   `unchecked` or `asm` feature, the default build clamps `n_fc` to 1),
//!   `decode()` legitimately returns `Ok(None)` with the frame in flight, and the
//!   following `flush()` destroyed it. Whether a consumer lost its last frame
//!   depended on scheduling — the `asm` CI flavour hashed 0 frames on every
//!   committed vector in `lr_sgr_vectors_threaded_match_reference_md5`.
//! * With any thread count, a chunk holding several temporal units: `decode()`
//!   returns the first frame, the rest of the chunk is still queued as pending
//!   input, and `flush()` threw it away instead of parsing it.
//!
//! The second case is what gives this file teeth in the default build: the
//! vector is fed twice in ONE `decode()` call (temporal delimiter + sequence
//! header + key frame, twice — a legal two-frame stream), and `flush()` must
//! produce the second frame. Reverting `flush()` to reset-then-drain fails
//! `flush_returns_frames_still_queued_in_the_last_chunk` at `threads = 1`.
//! The first case is asserted by the threaded tests, which are only a real gate
//! where frame threading exists (`--features unchecked`, or the `asm` CI leg);
//! elsewhere they degrade to the single-thread contract, which is stated in
//! each assertion message.

#![forbid(unsafe_code)]

use rav1d_safe::src::managed::{Decoder, Frame, Settings};
use std::path::PathBuf;

/// A single-temporal-unit 8-bit stream: temporal delimiter, sequence header,
/// one key frame. 25 KB, decodes to one 384x256 frame at production defaults.
fn single_tu_stream() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/crash_vectors/kodim03_yuv420_8bpc.obu");
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    // Precondition for the concatenation trick below: the stream begins with a
    // temporal delimiter OBU (type 2, has_size), so `A ++ A` is two temporal
    // units and not one malformed one.
    assert_eq!(
        &data[..2],
        &[0x12, 0x00],
        "vector must start with a temporal delimiter"
    );
    data
}

fn decoder(threads: u32, max_frame_delay: u32) -> Decoder {
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = max_frame_delay;
    Decoder::with_settings(settings).expect("decoder")
}

/// `decode(chunk)` then `flush()`, the pump every consumer writes. Returns
/// every frame that came out of either call, in order.
fn pump(dec: &mut Decoder, chunk: &[u8]) -> Vec<Frame> {
    let mut frames = Vec::new();
    if let Some(f) = dec.decode(chunk).expect("decode") {
        frames.push(f);
    }
    frames.extend(dec.flush().expect("flush"));
    frames
}

fn dims(f: &Frame) -> (u32, u32) {
    (f.width(), f.height())
}

/// The reference: at `threads = 1` a single TU decodes synchronously to exactly
/// one frame, and `flush()` afterwards has nothing to add.
#[test]
fn single_tu_at_one_thread_yields_one_frame() {
    let stream = single_tu_stream();
    let mut dec = decoder(1, 1);
    let first = dec.decode(&stream).expect("decode");
    assert!(
        first.is_some(),
        "threads = 1 decodes a single TU synchronously"
    );
    let rest = dec.flush().expect("flush");
    assert!(
        rest.is_empty(),
        "nothing left to drain after a synchronous decode"
    );
}

/// Two temporal units in one chunk: `decode()` returns the first frame and the
/// second is still pending input. `flush()` must parse and return it, not drop
/// it. This holds at every thread count, so it is the gate that has teeth in
/// the default (frame-threading-less) build.
#[test]
fn flush_returns_frames_still_queued_in_the_last_chunk() {
    let single = single_tu_stream();
    let mut double = single.clone();
    double.extend_from_slice(&single);

    let expected = {
        let mut dec = decoder(1, 1);
        let f = dec.decode(&single).expect("decode").expect("frame");
        dims(&f)
    };

    for threads in [1u32, 2, 4] {
        let mut dec = decoder(threads, 0);
        let frames = pump(&mut dec, &double);
        assert_eq!(
            frames.len(),
            2,
            "threads = {threads}: two temporal units were fed in one decode() call, \
             so decode() + flush() must yield two frames — flush() discarded the \
             temporal unit still queued as pending input"
        );
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(dims(f), expected, "threads = {threads}: frame {i} geometry");
        }
    }
}

/// With frame threading, `decode()` may return `None` while the frame is still
/// in flight; `flush()` must wait for it and return it. Where frame threading
/// is compiled out (`n_fc` clamped to 1 without `unchecked`/`asm`), this is the
/// same single-thread contract as above — one frame, never zero.
#[test]
fn flush_returns_in_flight_frames_under_frame_threading() {
    let stream = single_tu_stream();
    let mut saw_deferred = false;
    for threads in [2u32, 4, 8] {
        let mut dec = decoder(threads, 0);
        let first = dec.decode(&stream).expect("decode");
        saw_deferred |= first.is_none();
        let mut frames: Vec<Frame> = first.into_iter().collect();
        frames.extend(dec.flush().expect("flush"));
        assert_eq!(
            frames.len(),
            1,
            "threads = {threads}, max_frame_delay = auto: one temporal unit in, so \
             decode() + flush() must yield exactly one frame — with frame threading \
             the frame was in flight when flush() ran and flush() threw it away"
        );
    }
    // Not asserted, reported: whether this build actually deferred the frame
    // (frame threading compiled in) or decoded it synchronously. The assertion
    // above is the contract either way; this line says which half it tested.
    eprintln!(
        "frame threading {} on this build: decode() {} the single-TU frame at some \
         thread count",
        if saw_deferred { "ACTIVE" } else { "INACTIVE" },
        if saw_deferred {
            "deferred"
        } else {
            "never deferred"
        },
    );
}

/// `flush()` is still a reset: after draining, the decoder accepts a brand new
/// stream (fresh sequence header, fresh key frame) and decodes it in full.
#[test]
fn flush_then_decode_again_starts_a_fresh_stream() {
    let stream = single_tu_stream();
    for threads in [1u32, 4] {
        let mut dec = decoder(threads, 0);
        assert_eq!(
            pump(&mut dec, &stream).len(),
            1,
            "threads = {threads}: first stream"
        );
        assert_eq!(
            pump(&mut dec, &stream).len(),
            1,
            "threads = {threads}: second stream"
        );
        assert_eq!(
            dec.flush().expect("flush").len(),
            0,
            "threads = {threads}: nothing owed after two complete pumps"
        );
    }
}
