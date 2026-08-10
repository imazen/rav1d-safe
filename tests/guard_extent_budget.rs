//! Standing gate for the bounds map's extent invariant (`docs/BOUNDS_MAP.md`).
//!
//! > While tile threading is active, a **partial** reservation against a
//! > picture plane may not exceed
//! > [`TILE_THREADED_PIC_EXTENT_MAX_BYTES`](rav1d_safe::include::dav1d::picture::TILE_THREADED_PIC_EXTENT_MAX_BYTES).
//!
//! # Why this exists
//!
//! "Widen a guard's reservation to cut the registration count" has been built
//! and refuted three times — #469's strided rectangle, #475's hull, #485's
//! read band — and each refutation cost a full round because the failure is
//! RARE and DATA-DEPENDENT: the bounds map measured 2-16 collisions across
//! 1406 corpus frames for a ~124 byte widening, and **zero** on either 4K gap
//! vector. A decode test cannot reliably catch that. So this gate does not try
//! to catch the collision; it catches **the widening**, deterministically, on
//! the first frame, at the moment it is introduced.
//!
//! The invariant itself lives in
//! `include/dav1d/picture.rs::note_pic_extent`, at the single funnel every
//! tracked picture-plane reservation passes through. It is compiled under
//! `debug_assertions` (so plain `cargo test` catches a widening) or under
//! `--features probe-sites` (so this gate can run it in RELEASE, over real
//! decodes, at the speed that needs). The default release build has no
//! counter, no atomic load and no branch there.
//!
//! # Non-vacuity
//!
//! Three liveness assertions, all of which have failed on purpose during
//! development:
//!
//! 1. the invariant was EVALUATED (`checks > 0`),
//! 2. it was evaluated **while tile threading was active** (`checks_tt > 0`) —
//!    without this the antecedent is false and the gate can never fail,
//! 3. the whole-component exemption was exercised (`whole > 0`), so a future
//!    change that routes every reservation through the exemption is visible
//!    rather than silently disarming the gate.
//!
//! # Running
//!
//! ```text
//! cargo test --release --features probe-sites --test guard_extent_budget -- --nocapture
//! RAV1D_EXTENT_GATE_CORPUS=1 cargo test --release --features probe-sites \
//!     --test guard_extent_budget -- --nocapture     # + the dav1d corpus leg
//! ```
//!
//! The corpus leg is opt-in **from the caller** (CI sets it), never decided
//! inside the test body.

#![cfg(feature = "probe-sites")]

use rav1d_safe::include::dav1d::picture::{
    TILE_THREADED_PIC_EXTENT_MAX_BYTES, extent_budget, pic_extent_ceiling_const,
};
use rav1d_safe::src::managed::{Decoder, Settings};
use std::path::{Path, PathBuf};

mod ivf_parser;
mod test_vectors;

/// Decode every frame of `data` at `threads`, returning the frame count.
///
/// `max_frame_delay = 1` pins `n_fc = 1`, so this is pure TILE threading — the
/// configuration the invariant is about.
fn decode_all(data: &[u8], threads: u32) -> Result<usize, String> {
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = 1;
    let mut decoder = Decoder::with_settings(settings).map_err(|e| format!("create: {e:?}"))?;
    let mut frames = 0usize;
    match decoder.decode(data) {
        Ok(Some(_)) => frames += 1,
        Ok(None) => {}
        Err(e) => return Err(format!("decode: {e:?}")),
    }
    match decoder.flush() {
        Ok(rest) => frames += rest.len(),
        Err(e) => return Err(format!("flush: {e:?}")),
    }
    Ok(frames)
}

fn committed_vectors() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/crash_vectors");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("crash_vectors unreadable at {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "obu" || x == "ivf"))
        .collect();
    v.sort();
    assert!(
        !v.is_empty(),
        "no committed vectors in {} — the gate would be vacuous",
        dir.display()
    );
    v
}

/// The dav1d corpus leg. Caller-gated (`RAV1D_EXTENT_GATE_CORPUS=1`), never
/// decided inside the test body — a test that silently decides not to test is
/// worse than one that loudly fails.
fn corpus_vectors() -> Vec<PathBuf> {
    let root = test_vectors::ensure_dav1d_test_data();
    let mut v = Vec::new();
    for group in ["8-bit/data", "10-bit/data"] {
        let dir = root.join(group);
        let Ok(rd) = std::fs::read_dir(&dir) else {
            panic!("corpus group missing: {}", dir.display());
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "ivf") {
                v.push(p);
            }
        }
    }
    v.sort();
    assert!(
        !v.is_empty(),
        "corpus leg selected but no .ivf vectors found"
    );
    v
}

#[test]
fn picture_reservations_stay_inside_the_measured_ceiling() {
    // Threads > 1 latches `tile_threading_active()`, which is the invariant's
    // antecedent. 8 matches the campaign's standing cell.
    const THREADS: u32 = 8;

    let mut decoded = 0usize;
    let mut failed = Vec::new();

    for p in committed_vectors() {
        let data = std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        // Crash vectors are deliberately malformed; a decode ERROR is fine and
        // expected. What must not happen is the extent panic, which aborts the
        // test process rather than returning `Err`.
        match decode_all(&data, THREADS) {
            Ok(n) => decoded += n,
            Err(_) => failed.push(p),
        }
    }

    if std::env::var_os("RAV1D_EXTENT_GATE_CORPUS").is_some() {
        for p in corpus_vectors() {
            let f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
            let mut reader = std::io::BufReader::new(f);
            let frames = ivf_parser::parse_all_frames(&mut reader)
                .unwrap_or_else(|e| panic!("ivf {}: {e:?}", p.display()));
            let mut settings = Settings::default();
            settings.threads = THREADS;
            settings.max_frame_delay = 1;
            let mut dec = Decoder::with_settings(settings).expect("decoder");
            for frame in frames.iter().take(4) {
                let _ = dec.decode(&frame.data);
            }
            let _ = dec.flush();
            decoded += 4;
        }
    }

    let (checks, checks_tt, max_bytes, max_tt, whole) = extent_budget::report();
    println!(
        "extent budget: checks={checks} checks_tile_threaded={checks_tt} \
         max={max_bytes} B max_under_tile_threading={max_tt} B \
         whole_component_exempt={whole} fallback_ceiling={TILE_THREADED_PIC_EXTENT_MAX_BYTES} B \
         frames={decoded} undecodable_vectors={}",
        failed.len()
    );
    println!(
        "{:<38} {:>8} {:>9} {:>5} {:>12}  site of max",
        "file", "max B", "ceiling", "rows", "n"
    );
    let mut over = Vec::new();
    for (file, bytes, site, n, rows) in extent_budget::per_file() {
        // A file with a tight `PIC_EXTENT_CEILINGS` entry is held to it exactly,
        // which is what the in-decoder check does. A file WITHOUT one is held to
        // one picture row, which this per-file summary cannot know (it is a
        // property of the plane, not the file) — those are covered by the
        // `MAX_ROWS_TT == 1` assertion below, which is the same bound.
        let tight = pic_extent_ceiling_const(&file);
        let over_here = tight.is_some_and(|c| bytes > c);
        let c = match tight {
            Some(c) => c.to_string(),
            None => "1 row".to_string(),
        };
        println!(
            "{file:<38} {bytes:>8} {c:>9} {rows:>5} {n:>12}  {site}{}",
            if over_here { "  <-- OVER" } else { "" }
        );
        if over_here {
            over.push(format!("{site}: {bytes} B > the {c} B ceiling for {file}"));
        }
    }
    assert_eq!(
        extent_budget::MAX_ROWS_TT.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "some picture-plane reservation spanned more than one row while tile \
         threading was active. That is exactly #469's strided rectangle and \
         #475's hull: the extra bytes are the inter-row gaps, which belong to \
         other columns of the same rows, and AV1 tiles partition a frame BY \
         COLUMN."
    );

    // --- liveness: the gate must be able to fail ---
    assert!(
        checks > 0,
        "the extent invariant was never evaluated — no picture-plane slice \
         reservation reached `note_pic_extent`. Either the funnel moved or the \
         decodes did nothing; either way this gate is vacuous."
    );
    assert!(
        checks_tt > 100_000,
        "only {checks_tt} reservations were taken while tile threading was \
         active. The invariant's antecedent is `tile_threading_active()`, so \
         below that it can never fail and the gate proves nothing."
    );
    assert!(
        whole > 0,
        "the whole-component exemption was never taken. It is meant to be the \
         narrow, deliberate escape for `full_guard`; if nothing uses it the \
         exemption is dead code and should be removed rather than left as an \
         unexercised hole."
    );
    assert!(
        decoded > 0,
        "no frames decoded — the vectors did not exercise the decoder"
    );

    // --- the invariant itself ---
    //
    // Redundant with the in-decoder panic (which fires first, inside whichever
    // worker took the guard), and deliberately so: this restates it as a test
    // assertion so a `--features probe-sites` run that somehow swallowed the
    // panic still reports.
    assert!(
        over.is_empty(),
        "picture-plane reservations exceeded their measured per-file ceiling \
         while tile threading was active:\n  {}\n\
         See docs/BOUNDS_MAP.md — price the widening against the budget table \
         and raise PIC_EXTENT_CEILINGS WITH the measurement, or narrow the guard.",
        over.join("\n  ")
    );
    let _ = max_tt;
}
