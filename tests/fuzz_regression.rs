//! Fuzz crash regression suite — the standing gate against a fixed crash coming back.
//!
//! Runs **every committed crash repro in the repo** through **every fuzz target's
//! entry point**, on the **stable** toolchain: no nightly, no `cargo-fuzz`, no
//! libFuzzer, no sanitizer. A panic from any (seed, entry point) pair fails the
//! test and names both.
//!
//! Why this exists: the continuous fuzz farm files a crash, someone fixes it and
//! closes the issue, and nothing in-repo stops the next refactor from
//! re-introducing it. That is the mechanism behind the ~13 open issues titled
//! `RECURRED after <link>` — the repro existed, but only in the issue body and in
//! block storage, never in a test. A repro that is not in a test is not a gate.
//!
//! # What it walks
//!
//! Two roots, both required to be non-empty:
//!
//! * `fuzz/regression/` — minimised libFuzzer inputs, in a per-target subdirectory
//!   (`fuzz/regression/<fuzz-target-name>/`). Loose files at the root are run too.
//! * `tests/crash_vectors/` — raw AV1 OBU repros extracted from real files and from
//!   farm crashes. `tests/safe_simd_crashes.rs` runs these through the *default*
//!   decoder; this suite additionally runs them through the three fuzz targets'
//!   **tighter, different** configurations (frame-size limit, in-loop filters off,
//!   `max_frame_delay = 1`, film grain off), which reach different code.
//!
//! A seed's directory records which target found it; it does **not** restrict where
//! it runs. Every seed goes through every entry point, because a bug found by one
//! target is routinely reachable from another (the farm's own issues show the same
//! panic signature arriving on `decode_obu`, `parse_seq_header` *and*
//! `differential_dav1d`).
//!
//! # Adding a seed
//!
//! Drop the minimised file into `fuzz/regression/<target>/`, add a [`GUARDED`] row
//! if it guards a filed issue, and bump [`MIN_SEEDS`]. Hard ceiling **8 KB** per
//! global policy (`cargo fuzz tmin` first); anything larger stays in block storage
//! and is referenced by path from the issue.
//!
//! # Anti-vacuity
//!
//! A regression suite that finds zero seeds, or whose seeds have quietly stopped
//! reaching the decoder, passes while proving nothing. Four guards:
//!
//! 1. Missing or empty seed root → fail.
//! 2. Fewer than [`MIN_SEEDS`] files → fail (a deleted seed cannot pass silently).
//! 3. Every [`GUARDED`] row's file must exist → fail (a closed issue keeps its guard).
//! 4. At least [`MIN_SEEDS_REACHING_RECON`] seeds must decode to a frame → fail. A
//!    seed that no longer parses is not exercising the kernel it was filed against.
//!
//! Plus [`fuzz_regression_corpus_exercises_instrumented_kernels`], which under
//! `--features __ablate` asserts the corpus actually reaches inverse transforms,
//! CDEF and loop restoration — the three families with activity counters.
//!
//! Run: `cargo test --release --test fuzz_regression`
//! Liveness: `cargo test --release --features __ablate --test fuzz_regression`

// Release-only, like `tests/safe_simd_crashes.rs`: a debug decode of the larger
// vectors is minutes, not milliseconds. A `compile_error!` rather than a
// `#![cfg(not(debug_assertions))]` on purpose — the cfg form would make a debug
// `cargo test` silently run zero tests, which is the vacuous pass this file exists
// to prevent.
#[cfg(debug_assertions)]
compile_error!(
    "fuzz_regression requires release mode: cargo test --release --test fuzz_regression"
);

use std::fs;
use std::path::{Path, PathBuf};

use rav1d_safe::src::managed::{DecodeFrameType, Decoder, InloopFilters, Settings};

/// Matches `frame_size_limit` in all three fuzz targets.
const FRAME_SIZE_LIMIT_PIXELS: u32 = 256 * 256;

/// Seed roots, relative to the crate root. Both must exist and be non-empty.
const SEED_ROOTS: &[&str] = &["fuzz/regression", "tests/crash_vectors"];

/// Floor on the total seed count across [`SEED_ROOTS`].
///
/// 29 = the corpus as of 2026-08-14 (4 in `fuzz/regression`, 25 in
/// `tests/crash_vectors`). Raise it when you add seeds. Its only job is to make a
/// *deletion* loud: without it, `rm -r fuzz/regression` leaves a green suite.
const MIN_SEEDS: usize = 29;

/// Floor on how many seeds decode to at least one frame under some entry point.
///
/// Measured at 21 of 29 on 2026-08-14 (aarch64, release, default features). Of the
/// other 8, six still reach reconstruction and loop restoration before the decoder
/// errors out (proved by the `__ablate` activity table in
/// `benchmarks/fuzz_regression_2026-08-14.meta`) — "no frame" is not the same as
/// "no kernel ran". This number is what makes a slide toward vacuity visible.
/// Do not lower it to make a run pass; a drop means seeds stopped reaching the
/// decoder, which is the vacuity this suite exists to prevent.
const MIN_SEEDS_REACHING_RECON: usize = 21;

/// Per-seed size ceiling for `fuzz/regression/` (global fuzz-storage policy):
/// tiny minimised POCs only, target < 1 KB, hard ceiling 8 KB. Anything bigger
/// belongs in block storage, referenced by path from the issue.
///
/// Deliberately **not** applied to `tests/crash_vectors/`: those are hand-extracted
/// AV1 OBU streams (up to 30 KB) that predate this suite and are the corpus for
/// `tests/safe_simd_crashes.rs`, `tests/tile_threading_overlap.rs` and the
/// threading-race gates. The policy names `fuzz/regression/`; enforcing it on the
/// other root would mean deleting existing gates.
const MAX_SEED_BYTES: usize = 8 * 1024;

/// Does the size ceiling apply to this seed?
fn ceiling_applies(rel: &str) -> bool {
    rel.replace('\\', "/").starts_with("fuzz/regression/")
}

/// Seeds that guard a filed fuzz issue: `(path, issues, signature)`.
///
/// The point of the table is deletion-resistance: when an issue is closed as fixed
/// because one of these seeds now decodes cleanly, the seed must stay. A missing
/// file fails [`guarded_seeds_all_present`] with the issue numbers that lose their
/// gate.
///
/// `signature` is the panic location the farm recorded **at filing time**. Source
/// lines have since moved (both `mc_arm.rs` and `looprestoration_arm.rs` were
/// rewritten), so it is provenance, not an assertion target.
const GUARDED: &[(&str, &str, &str)] = &[
    (
        "tests/crash_vectors/arm_boxsum3_oob_8bpc.obu",
        "#427, #434",
        "looprestoration_arm.rs:399:13 boxsum3_8bpc vertical-pass OOB",
    ),
    (
        "tests/crash_vectors/arm_boxsum3_oob_16bpc.obu",
        "#429, #432",
        "looprestoration_arm.rs:935:13 boxsum3_16bpc vertical-pass OOB",
    ),
    (
        "tests/crash_vectors/arm_aa_base_underflow_8bpc.obu",
        "#428, #435",
        "looprestoration_arm.rs:465:30 selfguided_filter_8bpc aa_base underflow",
    ),
    (
        "tests/crash_vectors/arm_aa_base_underflow_16bpc.obu",
        "#431, #437",
        "looprestoration_arm.rs:999:30 selfguided_filter_16bpc aa_base underflow",
    ),
    (
        "tests/crash_vectors/arm_mc16_overshoot.obu",
        "#430, #436, #439, #444",
        "mc_arm.rs:5930/5937:61 16bpc mc_put dst slice overshoot",
    ),
    (
        "tests/crash_vectors/arm_mc16_avg_overshoot.obu",
        "#442",
        "mc_arm.rs:4772:61 16bpc avg dst slice overshoot",
    ),
    (
        "tests/crash_vectors/arm_mc16_mask_overshoot.obu",
        "#438",
        "mc_arm.rs:5003:61 16bpc mask dst slice overshoot",
    ),
    (
        "tests/crash_vectors/arm_mc16_blend_overshoot.obu",
        "#440",
        "mc_arm.rs:5112:61 16bpc blend dst slice overshoot",
    ),
    (
        "tests/crash_vectors/arm_mc16_blend_dir_overshoot.obu",
        "#443",
        "mc_arm.rs:5235:61 16bpc blend_dir dst slice overshoot",
    ),
    (
        "tests/crash_vectors/arm_mc16_w_mask_overshoot.obu",
        "#441",
        "mc_arm.rs:5465:61 16bpc w_mask dst slice overshoot",
    ),
    (
        "fuzz/regression/differential_dav1d/crash-itx-16x64-dc-rect2",
        "#433 (differential_dav1d.rs:192:13)",
        "itx 16x64 DC rect2 divergence vs dav1d",
    ),
    (
        "fuzz/regression/parse_seq_header/crash-mv-add-overflow",
        "-",
        "mv add overflow in the sequence-header target",
    ),
    (
        "fuzz/regression/cdef_tile_race/crash-cdef-tile-overlap.avif",
        "#30 (zenavif)",
        "CDEF/loop-filter tile-threading DisjointMut overlap",
    ),
];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every regular file under `dir`, recursively, sorted for a stable report order.
fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let rd = match fs::read_dir(&d) {
            Ok(rd) => rd,
            Err(e) => panic!("cannot read seed directory {}: {e}", d.display()),
        };
        for entry in rd.flatten() {
            let p = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(p),
                Ok(t) if t.is_file() => {
                    // Skip editor/OS droppings, never skip a real seed.
                    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if name != ".gitignore" && name != ".DS_Store" {
                        out.push(p);
                    }
                }
                _ => {}
            }
        }
    }
    out.sort();
    out
}

/// Every seed in the repo, with its path relative to the crate root.
fn all_seeds() -> Vec<(PathBuf, String)> {
    let root = crate_root();
    let mut seeds = Vec::new();
    for rel in SEED_ROOTS {
        let dir = root.join(rel);
        assert!(
            dir.is_dir(),
            "seed root {} is missing — the regression corpus is the gate; \
             a suite with no corpus passes while proving nothing",
            dir.display()
        );
        let files = collect_files(&dir);
        assert!(
            !files.is_empty(),
            "seed root {} is empty — refusing to pass vacuously",
            dir.display()
        );
        for f in files {
            let disp = f
                .strip_prefix(&root)
                .unwrap_or(&f)
                .to_string_lossy()
                .into_owned();
            seeds.push((f, disp));
        }
    }
    seeds
}

/// Did this entry point get far enough to reconstruct a frame?
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reach {
    /// A frame came out: parsing, entropy decode and reconstruction all ran.
    Frame,
    /// Accepted the bytes but produced nothing (buffering, or a truncated stream).
    NoFrame,
    /// Rejected. The seed exercises the parser only under this configuration.
    Rejected,
}

fn drive(mut decoder: Decoder, data: &[u8]) -> Reach {
    match decoder.decode(data) {
        Ok(Some(_frame)) => Reach::Frame,
        Ok(None) => match decoder.flush() {
            Ok(frames) if !frames.is_empty() => Reach::Frame,
            _ => Reach::NoFrame,
        },
        Err(_) => {
            // A rejected chunk can still have queued work; drain it, as the fuzz
            // targets do (they call `flush()` unconditionally).
            match decoder.flush() {
                Ok(frames) if !frames.is_empty() => Reach::Frame,
                _ => Reach::Rejected,
            }
        }
    }
}

/// Mirrors `fuzz/fuzz_targets/decode_obu.rs`.
fn run_decode_obu(data: &[u8]) -> Reach {
    let mut settings = Settings::default();
    settings.frame_size_limit = FRAME_SIZE_LIMIT_PIXELS;
    match Decoder::with_settings(settings) {
        Ok(d) => drive(d, data),
        Err(_) => Reach::Rejected,
    }
}

/// Mirrors `fuzz/fuzz_targets/parse_seq_header.rs`.
fn run_parse_seq_header(data: &[u8]) -> Reach {
    let mut settings = Settings::default();
    settings.threads = 1;
    settings.frame_size_limit = FRAME_SIZE_LIMIT_PIXELS;
    settings.inloop_filters = InloopFilters::none();
    settings.decode_frame_type = DecodeFrameType::All;
    match Decoder::with_settings(settings) {
        Ok(d) => drive(d, data),
        Err(_) => Reach::Rejected,
    }
}

/// Mirrors the rav1d-safe half of `fuzz/fuzz_targets/differential_dav1d.rs`.
///
/// The dav1d comparison half is deliberately absent: it needs a `pkg-config`
/// libdav1d ≥ 1.3 that CI's stable job does not have, and the divergence check is
/// not what this suite gates. What it gates is that the seed does not make
/// **rav1d-safe** panic under the differential target's settings.
fn run_differential_rav1d_half(data: &[u8]) -> Reach {
    let mut settings = Settings::default();
    settings.threads = 1;
    settings.max_frame_delay = 1;
    settings.frame_size_limit = FRAME_SIZE_LIMIT_PIXELS;
    settings.apply_grain = false;
    match Decoder::with_settings(settings) {
        Ok(d) => drive(d, data),
        Err(_) => Reach::Rejected,
    }
}

/// Production defaults: no frame-size limit, all in-loop filters, film grain on.
///
/// Not a fuzz target, and that is the point. All three targets cap
/// `frame_size_limit` at 65 536 pixels, so every seed larger than 256×256 is
/// rejected by them at the header and reaches no kernel at all. Ten of the
/// `tests/crash_vectors` repros are larger than that; without this entry point the
/// suite would run them vacuously.
fn run_default_settings(data: &[u8]) -> Reach {
    match Decoder::new() {
        Ok(d) => drive(d, data),
        Err(_) => Reach::Rejected,
    }
}

/// The AV1 payload inside an AVIF container, if these bytes are one.
///
/// `fuzz/regression/cdef_tile_race/crash-cdef-tile-overlap.avif` is a container,
/// not a raw OBU stream: fed to the decoder verbatim it is rejected at the first
/// byte and performs 0 units of work in every instrumented family — a seed that
/// guards nothing. Unwrapping it is what makes it a gate. Kept alongside the raw
/// bytes rather than replacing them, because the fuzz targets themselves receive
/// raw bytes and that path must stay covered.
fn avif_payload(data: &[u8]) -> Option<Vec<u8>> {
    let parser = zenavif_parse::AvifParser::from_bytes(data).ok()?;
    Some(parser.primary_data().ok()?.into_owned())
}

/// One fuzz-target entry point: its name, and the function that drives it.
type EntryPoint = (&'static str, fn(&[u8]) -> Reach);

const ENTRY_POINTS: &[EntryPoint] = &[
    ("decode_obu", run_decode_obu),
    ("parse_seq_header", run_parse_seq_header),
    (
        "differential_dav1d[rav1d-half]",
        run_differential_rav1d_half,
    ),
    ("default_settings", run_default_settings),
];

/// The suite: every seed through every entry point, no panics.
#[test]
fn fuzz_regression_seeds_do_not_panic() {
    let seeds = all_seeds();
    assert!(
        seeds.len() >= MIN_SEEDS,
        "found {} seeds, expected at least {MIN_SEEDS} — a committed crash repro \
         was deleted. Restore it, or lower MIN_SEEDS in the same commit that \
         explains why the repro is no longer needed.",
        seeds.len()
    );

    let mut reaching_recon = 0usize;
    let mut runs = 0usize;
    let mut oversized: Vec<String> = Vec::new();
    // Collected, not fail-fast: a set-diff BY NAME beats a count, and one run
    // should name every crashing (seed, entry point) pair rather than the first.
    let mut crashed: Vec<String> = Vec::new();

    for (path, rel) in &seeds {
        let input = fs::read(path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
        if input.len() > MAX_SEED_BYTES && ceiling_applies(rel) {
            oversized.push(format!("{rel} ({} bytes)", input.len()));
        }

        let mut forms: Vec<(&str, Vec<u8>)> = vec![("raw", input.clone())];
        if let Some(payload) = avif_payload(&input) {
            forms.push(("avif-payload", payload));
        }

        let mut reached = false;
        let mut marks = String::new();
        for (form, bytes) in &forms {
            if forms.len() > 1 {
                marks.push('[');
            }
            for (name, f) in ENTRY_POINTS {
                runs += 1;
                match std::panic::catch_unwind(|| f(bytes)) {
                    Ok(reach) => {
                        marks.push_str(match reach {
                            Reach::Frame => "F",
                            Reach::NoFrame => "-",
                            Reach::Rejected => "x",
                        });
                        reached |= reach == Reach::Frame;
                    }
                    Err(payload) => {
                        // The default panic hook has already printed the message
                        // and location to stderr; record which pair produced it.
                        let msg = payload
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                            .unwrap_or_else(|| "<non-string panic payload>".to_owned());
                        marks.push('P');
                        crashed.push(format!("{rel} ({form}) via {name}: {msg}"));
                    }
                }
            }
            if forms.len() > 1 {
                marks.push(']');
            }
        }
        if reached {
            reaching_recon += 1;
        }
        eprintln!("{marks}  {rel} ({} bytes)", input.len());
    }

    eprintln!(
        "\n{} seeds, {} entry points, {runs} runs; {reaching_recon} seeds reached \
         reconstruction (F = frame, - = accepted/no frame, x = rejected, \
         P = PANIC; one bracketed group per input form, columns in ENTRY_POINTS \
         order: {})",
        seeds.len(),
        ENTRY_POINTS.len(),
        ENTRY_POINTS
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    );

    assert!(
        crashed.is_empty(),
        "{} regression seed/entry-point pairs PANICKED — a previously-fixed crash \
         is back: {crashed:#?}",
        crashed.len()
    );

    assert!(
        oversized.is_empty(),
        "seeds over the {MAX_SEED_BYTES}-byte policy ceiling: {oversized:?} — \
         minimise with `cargo fuzz tmin`, or keep the raw artifact in block \
         storage and reference it from the issue"
    );

    assert!(
        reaching_recon >= MIN_SEEDS_REACHING_RECON,
        "only {reaching_recon} of {} seeds decoded a frame, expected at least \
         {MIN_SEEDS_REACHING_RECON}. Seeds that no longer parse cannot guard the \
         kernel they were filed against — this suite would be passing vacuously.",
        seeds.len()
    );
}

/// Every seed named in [`GUARDED`] still exists.
///
/// This is what lets an issue be closed as fixed: the guard cannot silently
/// evaporate afterwards.
#[test]
fn guarded_seeds_all_present() {
    let root = crate_root();
    let mut missing = Vec::new();
    for (rel, issues, sig) in GUARDED {
        if !root.join(rel).is_file() {
            missing.push(format!("{rel} (guards {issues}: {sig})"));
        }
    }
    assert!(
        missing.is_empty(),
        "regression seeds are missing, so these issues have no gate: {missing:#?}"
    );
}

/// Liveness: the corpus must actually reach the instrumented kernel families.
///
/// Without this, "15 crash tests pass" can mean "15 inputs were rejected at the
/// header". Only three families carry activity counters (`itx`, `cdef`,
/// `looprestoration` — the `ablate::note()` sites), so this proves reconstruction
/// and loop restoration ran; there is **no counter for motion compensation**, so
/// the `mc_arm` seeds' liveness is not covered here (see the report in
/// `benchmarks/fuzz_regression_2026-08-14.meta`).
///
/// Requires `--features __ablate`; without it the counters are compile-time no-ops
/// and the test says so and stops rather than asserting on zeros.
#[test]
fn fuzz_regression_corpus_exercises_instrumented_kernels() {
    use rav1d_safe::src::ablate::{self, Family};

    if !ablate::ENABLED {
        eprintln!(
            "SKIP-REPORT: activity counters are compiled out. This is not a runtime \
             self-skip of a gate — the counters do not exist in this build. Run \
             `cargo test --release --features __ablate --test fuzz_regression` for \
             the liveness assertion."
        );
        return;
    }

    // Per-seed first (this is the table that says which seed guards which kernel),
    // then the corpus total.
    let mut total = [0u64; 9];
    eprintln!("seed\titx\tcdef\tlooprestoration");
    for (path, rel) in all_seeds() {
        let input = fs::read(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
        ablate::activity_reset();
        let mut forms: Vec<Vec<u8>> = vec![input.clone()];
        if let Some(payload) = avif_payload(&input) {
            forms.push(payload);
        }
        for bytes in &forms {
            for (_, f) in ENTRY_POINTS {
                let _ = std::panic::catch_unwind(|| f(bytes));
            }
        }
        let per = ablate::activity_snapshot();
        for (t, p) in total.iter_mut().zip(per.iter()) {
            *t += *p;
        }
        eprintln!(
            "{rel}\t{}\t{}\t{}",
            per[Family::Itx as usize],
            per[Family::Cdef as usize],
            per[Family::LoopRestoration as usize]
        );
    }
    let snap = total;
    for f in [Family::Itx, Family::Cdef, Family::LoopRestoration] {
        let units = snap[f as usize];
        eprintln!("activity {}: {units} units", f.name());
        assert!(
            units > 0,
            "the whole regression corpus performed 0 units of {} work — the seeds \
             are not reaching the kernels they were filed against",
            f.name()
        );
    }
}
