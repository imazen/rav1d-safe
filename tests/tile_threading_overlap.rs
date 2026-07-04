//! Regression test for DisjointMut overlap panic in tile threading.
//!
//! Crafted AV1 bitstream (extracted from fuzz corpus AVIF container) with
//! dimensions that cause tile thread loopfilter to access overlapping regions
//! of the pixel buffer, triggering DisjointMut's runtime borrow checker.
//!
//! The overlap is non-deterministic (~40% repro rate per attempt) due to thread
//! scheduling, so we run multiple attempts to increase the chance of triggering.
//!
//! The panic occurs in a rav1d-worker thread, not the calling thread, so
//! `catch_unwind` cannot intercept it. Instead we spawn decode in a child
//! thread and join with a timeout to detect panics or deadlocks.
//!
//! **Requires `--release`** — debug mode is too slow for decode tests.
//!
//! Run: cargo test --release --test tile_threading_overlap -- --ignored --nocapture

#[cfg(debug_assertions)]
compile_error!("tile_threading_overlap tests require release mode: cargo test --release");

use rav1d_safe::src::managed::{Decoder, Settings};
use std::time::Duration;

const OBU: &[u8] = include_bytes!("crash_vectors/disjoint_mut_tile_overlap.obu");

/// Outcome of a single decode attempt.
#[derive(Debug)]
enum DecodeOutcome {
    /// Decode completed (success or graceful error).
    Ok,
    /// Worker thread panicked (DisjointMut overlap) causing deadlock or join failure.
    WorkerPanic,
    /// Decode did not complete within the timeout (likely deadlocked after worker panic).
    Timeout,
}

/// Attempt a single decode with the given thread count.
///
/// The entire decode runs in a spawned thread so that worker-thread panics
/// can be detected via join timeout rather than hanging the test runner.
fn try_decode_with_threads(threads: u32, timeout: Duration) -> DecodeOutcome {
    let handle = std::thread::spawn(move || {
        let mut settings = Settings::default();
        settings.threads = threads;
        // max_frame_delay=1 disables frame threading, isolating tile threading.
        settings.max_frame_delay = 1;
        let mut decoder = Decoder::with_settings(settings).expect("create decoder");

        match decoder.decode(OBU) {
            Ok(Some(frame)) => {
                eprintln!(
                    "  threads={threads}: decoded {}x{} @ {}bpc",
                    frame.width(),
                    frame.height(),
                    frame.bit_depth()
                );
            }
            Ok(None) => match decoder.flush() {
                Ok(frames) => {
                    eprintln!("  threads={threads}: flushed {} frames", frames.len());
                }
                Err(e) => {
                    eprintln!("  threads={threads}: flush error: {e:?}");
                }
            },
            Err(e) => {
                eprintln!("  threads={threads}: decode error: {e:?}");
            }
        }
        // Explicit drop to join worker threads — this is where we detect worker panics.
        drop(decoder);
    });

    // Poll the join handle with a timeout.
    let start = std::time::Instant::now();
    loop {
        if handle.is_finished() {
            return match handle.join() {
                Ok(()) => DecodeOutcome::Ok,
                Err(_) => DecodeOutcome::WorkerPanic,
            };
        }
        if start.elapsed() >= timeout {
            return DecodeOutcome::Timeout;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Single-threaded decode should complete without panic or deadlock.
#[test]
#[ignore]
fn single_threaded_no_panic() {
    eprintln!("OBU size: {} bytes", OBU.len());
    let outcome = try_decode_with_threads(1, Duration::from_secs(30));
    match outcome {
        DecodeOutcome::Ok => eprintln!("Single-threaded decode completed without panic."),
        other => panic!("Single-threaded decode failed unexpectedly: {other:?}"),
    }
}

/// Multi-threaded tile decode must not trigger DisjointMut overlap panic.
///
/// Previously, the loop filter V-pass at the bottom of sbrow N would
/// read/write pixels extending into the top rows of sbrow N+1. This
/// conflicted with concurrent TileReconstruction for sbrow N+1, causing
/// DisjointMut to (correctly) detect overlapping borrows and panic on a
/// worker thread, which deadlocked the decoder.
///
/// Fixed by adding a deblock progress barrier in check_tile: reconstruction
/// of sbrow N now waits until DeblockRows for sbrow N-1 completes.
#[test]
#[ignore]
fn multi_threaded_tile_overlap() {
    eprintln!("OBU size: {} bytes", OBU.len());

    let mut panic_or_timeout = 0;
    let mut ok_count = 0;
    let attempts = 10;
    // Per-attempt timeout: the decode itself is fast (~10ms for 700x400),
    // but a worker panic can deadlock the decoder drop, so we allow 10s.
    let timeout = Duration::from_secs(10);

    for attempt in 0..attempts {
        for threads in [4, 8] {
            eprintln!("Attempt {}/{attempts}, threads={threads}", attempt + 1);
            match try_decode_with_threads(threads, timeout) {
                DecodeOutcome::Ok => {
                    ok_count += 1;
                    eprintln!("  ok");
                }
                DecodeOutcome::WorkerPanic => {
                    panic_or_timeout += 1;
                    eprintln!("  WORKER PANIC (DisjointMut overlap)");
                }
                DecodeOutcome::Timeout => {
                    panic_or_timeout += 1;
                    eprintln!("  TIMEOUT (likely deadlocked after worker panic)");
                }
            }
        }
        // If we see any failures, stop early — the fix has regressed.
        if panic_or_timeout > 0 {
            break;
        }
    }

    let total = panic_or_timeout + ok_count;
    eprintln!("\nResults: {panic_or_timeout} failures, {ok_count} ok out of {total} attempts");

    assert_eq!(
        panic_or_timeout, 0,
        "DisjointMut overlap should not occur — deblock progress barrier prevents \
         concurrent reconstruction and loop filter V-pass on the same pixel rows"
    );
}

// ============================================================================
// zenavif#30: CDEF padding vs loop-filter compact-COW guards
// ============================================================================

/// Trigger stream for the second overlap class (found 2026-07-03, zenavif#30):
/// a rav1e-encoded 1024×1024 4:2:0 still whose tile-threaded decode raced the
/// loop filter's compact-buffer guards against CDEF.
const CDEF_LPF_OBU: &[u8] = include_bytes!("crash_vectors/tile_threading_cdef_lpf_race.obu");

/// Loop-filter compact-COW guards must not touch pixels dav1d never writes.
///
/// Two defects composed (both fixed in the same change):
/// 1. `compact_write_back_per_row` rewrote — and mutably guarded — every
///    pixel of the loop filter's read window, including the 7 tap rows/cols
///    the filter only READS. dav1d's CDEF task legitimately reads (bottom-edge
///    padding) and writes (its own blocks) inside that zone concurrently:
///    dav1d's CDEF lag ahead of deblock is exactly 2 pad rows + max modified
///    rows. Fixed by diffing against a pristine copy and writing back only
///    modified spans (`compact_write_back_per_row_diff`).
/// 2. The read window used the LUMA tap reach (7) for chroma too; chroma
///    deblock reads at most 3 rows/cols beyond the edge (wd6), and rows 4..=7
///    above a chroma edge belong to the previous sbrow's CDEF writes
///    (4-chroma-row lag). Fixed by plane-accurate `tap_before`.
///
/// Either defect makes a worker panic with `overlapping DisjointMut`
/// (`cdef.rs` padding / block IO vs `loopfilter.rs` compact guards); in
/// `unchecked` builds the write-back variant could instead silently clobber
/// concurrent CDEF output with stale bytes. Pre-fix the panic also wedged the
/// decode wait forever — see `tests/worker_panic_recovery.rs` for that half.
///
/// The race needs scheduling pressure: several concurrent tile-threaded
/// decoders in-process. Pre-fix this configuration fires within the first few
/// iterations (~100% of runs); post-fix it must stay silent.
#[test]
#[ignore]
fn multi_threaded_cdef_lpf_race() {
    const PAR: usize = 6;
    const ITERS: usize = 25;

    let workers: Vec<_> = (0..PAR)
        .map(|w| {
            std::thread::spawn(move || {
                for i in 0..ITERS {
                    let mut settings = Settings::default();
                    settings.threads = 8;
                    let mut decoder = Decoder::with_settings(settings).expect("create decoder");
                    match decoder.decode(CDEF_LPF_OBU) {
                        Ok(Some(_frame)) => {}
                        Ok(None) => {
                            let frames = decoder.flush().expect("flush");
                            assert!(!frames.is_empty(), "no frame decoded");
                        }
                        // With the worker panic guard, a racing worker panic
                        // surfaces as a decode error here (not a hang).
                        Err(e) => panic!("worker {w} iter {i}: decode failed: {e:?}"),
                    }
                }
            })
        })
        .collect();

    // Join with a global timeout so any residual wedge fails loudly instead
    // of hanging the test runner.
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(240);
    let mut workers: Vec<_> = workers.into_iter().map(Some).collect();
    loop {
        let mut all_done = true;
        for slot in workers.iter_mut() {
            if let Some(h) = slot {
                if h.is_finished() {
                    slot.take()
                        .unwrap()
                        .join()
                        .expect("decode loop hit the CDEF/LPF race (worker panicked)");
                } else {
                    all_done = false;
                }
            }
        }
        if all_done {
            break;
        }
        assert!(
            start.elapsed() < timeout,
            "decode loops wedged (>240s): the zenavif#30 hang is back"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}
