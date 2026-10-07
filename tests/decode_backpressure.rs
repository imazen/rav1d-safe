//! `Decoder::decode` must not lose a packet to decoder backpressure.
//!
//! The decoder refuses new input (EAGAIN) while input from an earlier call is still
//! queued behind a finished picture. `decode()` used to turn that into
//! `Err(NeedMoreData)` and drop the packet, so a caller that never drained
//! `get_frame()` between calls silently lost frames (found when a benchmark pump
//! counted 2 of 192). The documented pump polls `get_frame()`; this test is the one
//! that does not, at several thread / frame-delay combinations.

use rav1d_safe::src::managed::{Decoder, Frame, Settings};

#[path = "common/committed_vectors.rs"]
#[allow(dead_code)]
mod committed_vectors;

fn frame_md5(frame: &Frame) -> String {
    let mut ctx = md5::Context::new();
    committed_vectors::hash_frame(frame, &mut ctx);
    format!("{:x}", ctx.finalize())
}

fn decoder(threads: u32, delay: u32) -> Decoder {
    let mut s = Settings::default();
    s.threads = threads;
    s.max_frame_delay = delay;
    Decoder::with_settings(s).unwrap()
}

fn stream(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/gen_vectors/{name}.obu",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(path).unwrap()
}

/// Reference: the documented pump (decode, then drain `get_frame`), single-threaded.
fn reference(data: &[u8]) -> Vec<String> {
    let mut d = decoder(1, 1);
    let mut out = Vec::new();
    if let Some(f) = d.decode(data).unwrap() {
        out.push(frame_md5(&f));
    }
    while let Some(f) = d.get_frame().unwrap() {
        out.push(frame_md5(&f));
    }
    out.extend(d.flush().unwrap().iter().map(frame_md5));
    out
}

#[test]
fn decode_without_draining_loses_no_frames() {
    for name in [
        "v8_420_shift2_p0_96x96",
        "v8_420_shift4_128x128",
        "i8_420_tiles2x2_128x128",
    ] {
        let data = stream(name);
        let one = reference(&data);
        assert!(!one.is_empty(), "{name}: reference decoded nothing");
        // The same sequence twice in a row is a valid stream (a second coded video
        // sequence). One `decode()` call returns only the first frame and leaves the
        // rest of its input queued, so the second call meets backpressure.
        let expected: Vec<String> = one.iter().chain(one.iter()).cloned().collect();
        for (threads, delay) in [(1, 1), (2, 2), (4, 1), (4, 2), (8, 0)] {
            eprintln!("COMBO {name} t={threads} d={delay}");
            let mut d = decoder(threads, delay);
            let mut got = Vec::new();
            for call in 0..2 {
                match d.decode(&data) {
                    Ok(Some(f)) => got.push(frame_md5(&f)),
                    Ok(None) => {}
                    Err(e) => panic!("{name} t={threads} d={delay} call {call}: {e}"),
                }
            }
            got.extend(d.flush().unwrap().iter().map(frame_md5));
            assert_eq!(
                got, expected,
                "{name} t={threads} d={delay}: frames lost or reordered"
            );
        }
    }
}

#[test]
fn reset_discards_frames_held_back_by_backpressure() {
    let data = stream("v8_420_shift2_p0_96x96");
    let one = reference(&data);
    let mut d = decoder(1, 1);
    let _ = d.decode(&data).unwrap();
    let _ = d.decode(&data).unwrap(); // leaves a held-back picture behind
    d.reset();
    // After a reset nothing from before it may come out, and decoding works again.
    assert!(d.get_frame().unwrap().is_none());
    let mut got = Vec::new();
    if let Some(f) = d.decode(&data).unwrap() {
        got.push(frame_md5(&f));
    }
    got.extend(d.flush().unwrap().iter().map(frame_md5));
    assert_eq!(got, one);
}
