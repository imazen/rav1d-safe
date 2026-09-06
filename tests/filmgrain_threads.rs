//! #479 regression: film grain must decode at `threads > 1`.
//!
//! Film grain is applied by N worker threads that each `fetch_add` a *different*
//! `FG_BLOCK_SIZE` row band off `TaskThreadData::delayed_fg_progress[0]`
//! (`src/thread_task.rs`, `TaskType::FgApply`). The four aarch64 film-grain
//! guards in `src/safe_simd/filmgrain_arm.rs` used to reserve the WHOLE plane
//! per band (`full_guard_mut` / `full_guard`), so above one thread the workers
//! collided on the first band each:
//!
//! ```text
//!  current: &mut _[0..147456] on ThreadId(6) at include/dav1d/picture.rs:736
//! existing: &mut _[0..147456]                at include/dav1d/picture.rs:736
//! ```
//!
//! and the dead worker then wedged the main thread on `thread_task.rs`'s
//! `unwrap()` of a `None`. 13 of 768 corpus vectors could not be decoded **at
//! all** above one thread, so every corpus run of the 2026-08 campaign passed
//! `--skip-group film_grain`.
//!
//! Why the existing gate could not see it: `decode_md5_verify`'s film-grain
//! tests use `Settings::default()`, which is `threads: 1`. Single-threaded
//! decode never starts a second grain worker, so no amount of MD5 checking
//! there can fail on a whole-plane reservation.
//!
//! **Teeth.** Reverting any one of the four guards to its `full_guard*` form
//! makes `film_grain_md5_matches_at_1_2_4_8_threads` abort. Verified by
//! planting exactly that mutation; see the PR body for the transcript.
//!
//! **Liveness.** `grain_actually_changes_pixels` fails if grain synthesis never
//! ran (a decode that silently skipped it would otherwise pass every MD5 check
//! against a grain-off reference), and `some_vector_spans_multiple_row_bands`
//! fails if no vector is tall enough for two workers to contend in the first
//! place. Both are preconditions for this test to mean anything.
//!
//! **What this does NOT cover.** The corpus has 13 film-grain vectors and that
//! is the whole of the coverage; census (`examples/decode_md5`, 2026-08-10):
//!
//! ```text
//!  8-bit  I420  64x63 (odd height), 352x288
//!  8-bit  I422  64x63 (odd height), 63x64 (odd WIDTH)
//!  8-bit  I444  64x64 x3
//! 10-bit  I420  352x288
//! 10-bit  I422  64x63 (odd height), 63x64 (odd WIDTH)
//! 10-bit  I444  64x64 x3
//! ```
//!
//! So all three layouts and both bit depths are exercised, and the odd-width
//! `ss_x` path — the one that makes `rav1d_apply_grain_row` write two padding
//! pixels into the INPUT luma plane while other workers read it — is covered by
//! the two `422_oddwidth` vectors. Missing, in the order it would matter:
//! **4:2:0 at odd width** (nothing exercises `ss_x` and `ss_y` together with the
//! padding write), **12-bit film grain** (no vector at all), and **any grain
//! frame larger than 352x288** — so the narrowed guard extents are only ever
//! checked at small strides, and 352x288 gives at most 9 row bands.

#![forbid(unsafe_code)]

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::path::{Path, PathBuf};

mod ivf_parser;
mod test_vectors;

/// `FG_BLOCK_SIZE` — the row-band height each grain worker claims.
/// Not re-exported from the crate, so it is restated here; if it ever changes,
/// `some_vector_spans_multiple_row_bands` gets *more* conservative, not less.
const FG_BLOCK_SIZE: usize = 32;

/// The two meson groups whose vectors dav1d decodes with `--filmgrain 1`.
const GRAIN_GROUPS: &[&str] = &["8-bit/film_grain", "10-bit/film_grain"];

struct Vector {
    group: &'static str,
    name: String,
    ivf: PathBuf,
    expected_md5: String,
}

/// Minimal `['name', files('x.ivf'), 'md5']` reader — the same shape
/// `decode_md5_verify` and `examples/md5_inventory` parse.
fn parse_meson(group: &'static str, meson: &Path) -> Vec<Vector> {
    let text = std::fs::read_to_string(meson).unwrap_or_default();
    let dir = meson.parent().unwrap();
    let mut entries = Vec::new();
    let mut cur = String::new();
    let mut open = false;
    for line in text.lines() {
        let t = line.trim();
        if !open {
            if t.starts_with('[') && t.contains('\'') {
                cur = t.to_string();
                if t.contains("],") || t.ends_with(']') {
                    entries.push(std::mem::take(&mut cur));
                } else {
                    open = true;
                }
            }
        } else {
            cur.push(' ');
            cur.push_str(t);
            if t.contains("],") || t.ends_with(']') {
                entries.push(std::mem::take(&mut cur));
                open = false;
            }
        }
    }

    let mut out = Vec::new();
    for e in &entries {
        if !e.contains("files(") {
            continue;
        }
        let mut quoted = Vec::new();
        let mut chars = e.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\'' {
                let s: String = chars.by_ref().take_while(|&c| c != '\'').collect();
                if !s.is_empty() {
                    quoted.push(s);
                }
            }
        }
        if quoted.len() < 3 {
            continue;
        }
        let Some(file) = quoted
            .iter()
            .find(|s| s.ends_with(".ivf") || s.ends_with(".obu"))
        else {
            continue;
        };
        let Some(md5) = quoted
            .iter()
            .rev()
            .find(|s| s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()))
        else {
            continue;
        };
        out.push(Vector {
            group,
            name: quoted[0].clone(),
            ivf: dir.join(file),
            expected_md5: md5.clone(),
        });
    }
    out
}

fn grain_vectors() -> Vec<Vector> {
    let base = test_vectors::ensure_dav1d_test_data();
    let mut all = Vec::new();
    for g in GRAIN_GROUPS {
        let meson = base.join(g).join("meson.build");
        assert!(
            meson.exists(),
            "missing {} — the film-grain groups are what this test is for",
            meson.display()
        );
        all.extend(parse_meson(g, &meson));
    }
    assert!(
        all.len() >= 13,
        "expected >= 13 film-grain vectors, parsed {} — a parser regression \
         would make every assertion below vacuous",
        all.len()
    );
    all
}

fn hash_frame(frame: &Frame, ctx: &mut md5::Context) {
    match frame.planes() {
        Planes::Depth8(p) => {
            for row in p.y().rows() {
                ctx.consume(row);
            }
            for plane in [p.u(), p.v()].into_iter().flatten() {
                for row in plane.rows() {
                    ctx.consume(row);
                }
            }
        }
        Planes::Depth16(p) => {
            for row in p.y().rows() {
                for &px in row {
                    ctx.consume(px.to_le_bytes());
                }
            }
            for plane in [p.u(), p.v()].into_iter().flatten() {
                for row in plane.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
        }
    }
}

/// Decode every packet and return `(md5, frames, max_height)`.
fn decode(ivf: &Path, threads: u32, apply_grain: bool) -> Result<(String, usize, usize), String> {
    let (hash, frames, height, _) = decode_with_delay(ivf, threads, 1, apply_grain)?;
    Ok((hash, frames, height))
}

fn decode_with_delay(
    ivf: &Path,
    threads: u32,
    max_frame_delay: u32,
    apply_grain: bool,
) -> Result<(String, usize, usize, usize), String> {
    let file = std::fs::File::open(ivf).map_err(|e| format!("open {}: {e}", ivf.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let packets = ivf_parser::parse_all_frames(&mut reader).map_err(|e| format!("ivf: {e}"))?;

    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = max_frame_delay;
    settings.apply_grain = apply_grain;
    let mut decoder = Decoder::with_settings(settings).map_err(|e| format!("decoder: {e}"))?;

    let mut ctx = md5::Context::new();
    let mut n = 0usize;
    let mut deferred = 0usize;
    let mut max_h = 0usize;
    let note = |frame: &Frame, ctx: &mut md5::Context, n: &mut usize, max_h: &mut usize| {
        let h = match frame.planes() {
            Planes::Depth8(p) => p.y().height(),
            Planes::Depth16(p) => p.y().height(),
        };
        *max_h = (*max_h).max(h);
        hash_frame(frame, ctx);
        *n += 1;
    };

    for p in &packets {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "decoder made no progress on input backpressure"
            );
            match decoder.decode(&p.data) {
                Ok(Some(frame)) => {
                    note(&frame, &mut ctx, &mut n, &mut max_h);
                    break;
                }
                Ok(None) => {
                    deferred += 1;
                    break;
                }
                Err(e) if matches!(e.error(), rav1d_safe::src::managed::Error::NeedMoreData) => {
                    // send_data rejected this packet before consuming it.
                    // Retrieve output from the previous packet, then retry
                    // the SAME bytes; dropping them would hide frame loss.
                    if let Some(frame) = decoder
                        .get_frame()
                        .map_err(|e| format!("backpressure: {e}"))?
                    {
                        note(&frame, &mut ctx, &mut n, &mut max_h);
                    }
                    // None can still mean pending input was parsed into a
                    // frame context. Retry submission before forcing a drain.
                }
                Err(e) => return Err(format!("decode packet {n}: {e}")),
            }
        }
    }
    for frame in decoder.flush().map_err(|e| format!("flush: {e}"))? {
        note(&frame, &mut ctx, &mut n, &mut max_h);
    }
    Ok((format!("{:x}", ctx.finalize()), n, max_h, deferred))
}

/// The gate: every film-grain vector must produce the reference MD5 at 1, 2, 4
/// and 8 threads. Repeated, because a guard collision is a race — on `main`
/// it fired on the first try every time, but a *future* narrowing bug might
/// not, and a flaky abort is still an abort.
#[test]
fn film_grain_md5_matches_at_1_2_4_8_threads() {
    let vectors = grain_vectors();
    let mut failures = Vec::new();

    for v in &vectors {
        for threads in [1u32, 2, 4, 8] {
            let reps = if threads == 1 { 1 } else { 3 };
            for rep in 0..reps {
                match decode(&v.ivf, threads, true) {
                    Ok((md5, frames, _)) => {
                        if md5 != v.expected_md5 {
                            failures.push(format!(
                                "{}/{} t={threads} rep={rep}: md5 {md5} != {} ({frames} frames)",
                                v.group, v.name, v.expected_md5
                            ));
                        }
                    }
                    Err(e) => {
                        failures.push(format!("{}/{} t={threads} rep={rep}: {e}", v.group, v.name))
                    }
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} film-grain vector/thread cells failed:\n{}",
        failures.len(),
        vectors.len(),
        failures.join("\n")
    );
}

/// Liveness 1: grain synthesis must actually run. Without this, a decode that
/// silently skipped grain would satisfy every MD5 comparison above against a
/// grain-free reference and the test would prove nothing about the guards.
#[test]
fn grain_actually_changes_pixels() {
    let vectors = grain_vectors();
    let mut changed = 0usize;
    for v in &vectors {
        let on = decode(&v.ivf, 1, true).expect("grain-on decode");
        let off = decode(&v.ivf, 1, false).expect("grain-off decode");
        if on.0 != off.0 {
            changed += 1;
        }
    }
    assert!(
        changed >= vectors.len() / 2,
        "grain changed pixels in only {changed} of {} vectors — the film-grain \
         kernels are not running, so the thread test above is vacuous",
        vectors.len()
    );
}

/// Liveness 2: at least one vector must be taller than one row band, otherwise
/// there is only ever one band to hand out and two workers can never contend.
#[test]
fn some_vector_spans_multiple_row_bands() {
    let vectors = grain_vectors();
    let mut best = 0usize;
    let mut best_name = String::new();
    for v in &vectors {
        let (_, _, h) = decode(&v.ivf, 1, true).expect("decode");
        if h > best {
            best = h;
            best_name = format!("{}/{}", v.group, v.name);
        }
    }
    let bands = best.div_ceil(FG_BLOCK_SIZE);
    assert!(
        bands >= 2,
        "tallest film-grain vector is {best} px ({best_name}) = {bands} row band(s); \
         two grain workers can never contend, so the thread test is vacuous"
    );
    eprintln!("tallest grain vector {best_name}: {best} px = {bands} row bands");
}

/// Frame contexts must actually be enabled; checked builds clamp them to one.
#[cfg(feature = "unchecked")]
#[test]
fn film_grain_frame_and_tile_threads_match_reference() {
    let vectors = grain_vectors();
    let mut deferred = 0;
    let mut multi_frame_vectors = 0;
    for v in &vectors {
        let baseline = decode(&v.ivf, 1, true).expect("serial reference");
        assert_eq!(baseline.0, v.expected_md5, "serial {}", v.name);
        multi_frame_vectors += usize::from(baseline.1 > 1);
        for (threads, delay) in [(4, 2), (8, 2), (8, 4)] {
            for rep in 0..3 {
                let result = decode_with_delay(&v.ivf, threads, delay, true).unwrap_or_else(|e| {
                    panic!("{} t={threads} delay={delay} rep={rep}: {e}", v.name)
                });
                assert_eq!(
                    result.0, v.expected_md5,
                    "{} t={threads} delay={delay} rep={rep}",
                    v.name
                );
                assert_eq!(result.1, baseline.1, "lost/duplicated frames: {}", v.name);
                deferred += result.3;
            }
        }
    }
    assert!(
        multi_frame_vectors > 0,
        "no multi-frame input exercised the pipeline"
    );
    assert!(deferred > 0, "frame decoding never deferred output");
    eprintln!(
        "{} vectors, {} multi-frame vectors, {} deferred packet outputs, {} threaded decode runs",
        vectors.len(),
        multi_frame_vectors,
        deferred,
        vectors.len() * 9
    );
}

/// Separate decoders have separate picture buffers but share dispatch state.
#[test]
fn film_grain_independent_decoders_match_reference() {
    let vectors = grain_vectors();
    let barrier = std::sync::Barrier::new(3);
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for worker in 0..3 {
            let vectors = &vectors;
            let barrier = &barrier;
            workers.push(scope.spawn(move || {
                // One start barrier: a decode failure must not leave sibling
                // workers blocked at a later barrier.
                barrier.wait();
                for v in vectors {
                    let delay = if cfg!(feature = "unchecked") && worker != 0 {
                        2
                    } else {
                        1
                    };
                    let result = decode_with_delay(&v.ivf, 4, delay, true).unwrap_or_else(|e| {
                        panic!("{} decoder={worker} delay={delay}: {e}", v.name)
                    });
                    assert_eq!(
                        result.0, v.expected_md5,
                        "{} decoder={worker} delay={delay}",
                        v.name
                    );
                    assert!(result.1 > 0, "no frames: {}", v.name);
                }
            }));
        }
        for worker in workers {
            worker.join().expect("parallel decoder panicked");
        }
    });
}
