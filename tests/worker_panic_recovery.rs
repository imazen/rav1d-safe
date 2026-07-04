//! Worker-thread death must surface as a decode ERROR, never a hang.
//!
//! Regression test for the zenavif#30 wedge: a tile worker that dies by
//! panic (originally a `DisjointMut` overlap panic in the loop filter's
//! compact write-back racing CDEF padding) can never complete its claimed
//! task, so `fc.task_thread.task_counter` never reaches 0 and — pre-fix —
//! `rav1d_decode_frame`'s completion wait blocked forever: four zenavif
//! conformance cells sat 76-90 minutes at 0 CPU in `futex_` before being
//! killed. The fix (`TaskThreadData::panicked` + the worker unwind guard)
//! turns that into an `EGeneric` decode error and keeps `Decoder::drop`
//! (join) and `rav1d_flush` from waiting on the dead worker.
//!
//! Uses the private `__test_induce_worker_panic` feature to make the next
//! task-claiming worker panic — the exact shape of a real worker bug.
//!
//! **Requires `--release`** — debug decode is too slow for decode tests.
//!
//! Run: cargo test --release --features __test_induce_worker_panic \
//!        --test worker_panic_recovery

#![cfg(feature = "__test_induce_worker_panic")]

#[cfg(debug_assertions)]
compile_error!("worker_panic_recovery tests require release mode: cargo test --release");

use rav1d_safe::src::managed::{Decoder, Settings};
use rav1d_safe::src::thread_task::TEST_INDUCE_WORKER_PANIC;
use std::sync::atomic::Ordering;
use std::time::Duration;

const OBU: &[u8] = include_bytes!("crash_vectors/tile_threading_cdef_lpf_race.obu");

/// A worker panic mid-frame must produce a decode error within the timeout —
/// not a forever-wedge — and the decoder must still drop cleanly (the drop
/// joins worker threads; the dead one must not stall `rav1d_flush`-style
/// waits either).
#[test]
#[ignore]
fn worker_panic_fails_decode_instead_of_hanging() {
    let done = std::thread::spawn(|| {
        let mut settings = Settings::default();
        settings.threads = 8;
        let mut decoder = Decoder::with_settings(settings).expect("create decoder");

        TEST_INDUCE_WORKER_PANIC.store(true, Ordering::SeqCst);
        let result = decoder.decode(OBU);
        // Disarm in case the frame finished before any worker claimed a task
        // while armed (should not happen with threads=8, but never leave a
        // global armed).
        let armed_unused = TEST_INDUCE_WORKER_PANIC.swap(false, Ordering::SeqCst);
        assert!(
            !armed_unused,
            "no worker ever claimed a task while armed — test exercised nothing"
        );
        assert!(
            result.is_err(),
            "decode with a dead worker must fail, got {:?}",
            result.map(|f| f.is_some())
        );
        // Drop must join the remaining workers without waiting on the dead one.
        drop(decoder);
    });

    // The whole sequence (decode error + drop) must complete promptly. The
    // pre-fix behavior waits forever on fc.task_thread.cond.
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(30);
    while !done.is_finished() {
        assert!(
            start.elapsed() < timeout,
            "decode+drop did not complete within {timeout:?}: the worker-panic \
             wedge is back (zenavif#30)"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    done.join().expect("test thread panicked");
}

/// After a worker died, later decodes on the same decoder must keep failing
/// fast (the pool is poisoned — a dead worker means suspect shared state),
/// and creating a FRESH decoder must work.
#[test]
#[ignore]
fn worker_panic_poisons_decoder_but_not_process() {
    let done = std::thread::spawn(|| {
        let mut settings = Settings::default();
        settings.threads = 8;
        let mut decoder = Decoder::with_settings(settings).expect("create decoder");

        TEST_INDUCE_WORKER_PANIC.store(true, Ordering::SeqCst);
        assert!(decoder.decode(OBU).is_err(), "first decode must fail");
        let _ = TEST_INDUCE_WORKER_PANIC.swap(false, Ordering::SeqCst);
        assert!(
            decoder.decode(OBU).is_err(),
            "decoder with a dead worker must stay failed"
        );
        drop(decoder);

        // A fresh decoder in the same process decodes fine.
        let mut settings = Settings::default();
        settings.threads = 8;
        let mut fresh = Decoder::with_settings(settings).expect("create fresh decoder");
        let frame = fresh.decode(OBU).expect("fresh decoder decodes");
        assert!(frame.is_some(), "fresh decoder must produce a frame");
    });

    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(30);
    while !done.is_finished() {
        assert!(
            start.elapsed() < timeout,
            "poisoned-decoder sequence did not complete within {timeout:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    done.join().expect("test thread panicked");
}
