//! Cooperative in-flight decode cancellation (issue #412).
//!
//! A [`Stop`] token set via [`Decoder::set_stop`] is polled at sbrow boundaries
//! in the single-threaded decode loop; when it fires the in-flight frame is
//! aborted and the decode call returns [`Error::Cancelled`] instead of running
//! the (possibly crafted-but-legal, unbounded-time) frame to completion.
//!
//! **Requires `--release`** — debug mode is too slow for full-frame decode.
//!
//! Run: cargo test --release --test cancellation

#[cfg(debug_assertions)]
compile_error!("cancellation tests require release mode: cargo test --release");

use rav1d_safe::src::managed::{Decoder, Error, Stop, StopReason, Unstoppable};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 8bpc 4:2:0 photo — several superblock rows, so the per-sbrow checkpoint fires
/// multiple times during one decode.
const KODIM03: &[u8] = include_bytes!("crash_vectors/kodim03_yuv420_8bpc.obu");

/// A `Stop` that permits `allow` checks, then signals `Cancelled` on every
/// subsequent check. `allow = 0` stops at the very first sbrow.
struct CountdownStop {
    remaining: AtomicUsize,
}

impl CountdownStop {
    fn new(allow: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(allow),
        }
    }
}

impl Stop for CountdownStop {
    fn check(&self) -> Result<(), StopReason> {
        // Decrement while positive; once it hits zero, always stop.
        loop {
            let cur = self.remaining.load(Ordering::Relaxed);
            if cur == 0 {
                return Err(StopReason::Cancelled);
            }
            if self
                .remaining
                .compare_exchange(cur, cur - 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(());
            }
        }
    }
}

/// Decode `data` to completion, returning the frame count or the (unwrapped)
/// error. Drains via `flush()` so a `None` first return still counts frames.
fn decode_full(d: &mut Decoder, data: &[u8]) -> Result<usize, Error> {
    let mut n = 0;
    match d.decode(data) {
        Ok(Some(_)) => n += 1,
        Ok(None) => {}
        Err(e) => return Err(e.error().clone()),
    }
    match d.flush() {
        Ok(frames) => n += frames.len(),
        Err(e) => return Err(e.error().clone()),
    }
    Ok(n)
}

/// Decode and assert the call is cancelled. Avoids `expect_err` because
/// `Frame` is not `Debug`.
fn assert_cancelled(d: &mut Decoder, data: &[u8]) {
    match d.decode(data) {
        Err(e) => assert_eq!(*e.error(), Error::Cancelled, "got {:?}", e.error()),
        Ok(_) => panic!("decode must be cancelled, but it completed"),
    }
}

#[test]
fn cancel_at_first_sbrow_yields_cancelled() {
    let mut d = Decoder::new().unwrap();
    d.set_stop(Some(Arc::new(CountdownStop::new(0))));
    assert_cancelled(&mut d, KODIM03);
}

#[test]
fn cancel_mid_frame_yields_cancelled() {
    let mut d = Decoder::new().unwrap();
    // Allow a few sbrows to decode, then stop — exercises mid-frame abort.
    d.set_stop(Some(Arc::new(CountdownStop::new(2))));
    assert_cancelled(&mut d, KODIM03);
}

#[test]
fn unstoppable_token_decodes_normally() {
    let mut d = Decoder::new().unwrap();
    d.set_stop(Some(Arc::new(Unstoppable)));
    let frames = decode_full(&mut d, KODIM03).expect("Unstoppable must not block decode");
    assert!(frames > 0, "expected at least one decoded frame");
}

#[test]
fn no_token_decodes_normally() {
    let mut d = Decoder::new().unwrap();
    let frames = decode_full(&mut d, KODIM03).expect("decode without a token must succeed");
    assert!(frames > 0, "expected at least one decoded frame");
}

#[test]
fn cancel_tile_threaded_yields_cancelled() {
    // threads=4, max_frame_delay=1 → n_tc=4, n_fc=1 → tile-threaded workers
    // (the path that does NOT go through rav1d_decode_frame_main). The token is
    // checked per task in each worker and aborts the frame via the flush error
    // path. allow=0 fires on the first worker task.
    let mut settings = rav1d_safe::src::managed::Settings::default();
    settings.threads = 4;
    settings.max_frame_delay = 1;
    let mut d = Decoder::with_settings(settings).unwrap();
    d.set_stop(Some(Arc::new(CountdownStop::new(0))));
    // With frame threading off (n_fc=1) the decode is synchronous, so the abort
    // surfaces on this decode call (or, defensively, on the flush drain).
    match decode_full(&mut d, KODIM03) {
        Err(Error::Cancelled) => {}
        other => panic!("tile-threaded decode must cancel, got {other:?}"),
    }
}

#[test]
fn clearing_token_restores_normal_decode() {
    // A token that would cancel, then cleared before decoding: decode succeeds.
    let mut d = Decoder::new().unwrap();
    d.set_stop(Some(Arc::new(CountdownStop::new(0))));
    d.set_stop(None);
    let frames = decode_full(&mut d, KODIM03).expect("cleared token must not cancel");
    assert!(frames > 0, "expected at least one decoded frame");
}
