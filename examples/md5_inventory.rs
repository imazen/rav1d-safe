//! Per-vector MD5 inventory over the whole dav1d-test-data corpus.
//!
//! The `decode_md5_verify` integration test asserts pass/fail per meson group
//! and prints the failing names only inside a panic message. That is fine as a
//! gate and useless as a *record*: every attribution question ("did this fix
//! repair the same vectors it broke?") needs the failing set BY NAME, and needs
//! it in a form you can `comm`/`join` against a later run.
//!
//! This emits one TSV row per vector — group, name, status, expected, actual,
//! frames, wall_ms — so a before/after comparison is a set-diff, never a count
//! comparison. A change that repairs 50 vectors and breaks 50 others is
//! invisible in a count and obvious in a set-diff.
//!
//! Usage:
//!   cargo build --release --example md5_inventory
//!   ./target/release/examples/md5_inventory > ~/tmp/md5_inventory.tsv
//!
//! Optional args:
//!   --group <substr>   only run meson groups whose relative path contains this
//!   --name <substr>    only run vectors whose name contains this
//!   --threads <n>      decoder thread count (default 1). Run the gate at 1
//!                      AND at 8: at 1 the tile workers never start, so a
//!                      single-threaded-only gate cannot see a borrow-tracker
//!                      or threading regression at all.

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

struct TestVector {
    name: String,
    ivf_path: PathBuf,
    expected_md5: String,
}

/// (relative meson.build path, apply_grain)
///
/// Mirrors `tests/decode_md5_verify.rs::test_md5_verify_comprehensive` exactly:
/// dav1d's md5 muxer defaults to film grain OFF, so only the film_grain/ groups
/// decode with grain applied.
const MESON_GROUPS: &[(&str, bool)] = &[
    ("8-bit/data/meson.build", false),
    ("8-bit/features/meson.build", false),
    ("8-bit/issues/meson.build", false),
    ("8-bit/quantizer/meson.build", false),
    ("8-bit/size/meson.build", false),
    ("8-bit/cdfupdate/meson.build", false),
    ("8-bit/vq_suite/meson.build", false),
    ("8-bit/intra/meson.build", false),
    ("8-bit/mfmv/meson.build", false),
    ("8-bit/mv/meson.build", false),
    ("8-bit/resize/meson.build", false),
    ("8-bit/film_grain/meson.build", true),
    ("10-bit/data/meson.build", false),
    ("10-bit/features/meson.build", false),
    ("10-bit/quantizer/meson.build", false),
    ("10-bit/issues/meson.build", false),
    ("10-bit/film_grain/meson.build", true),
    ("12-bit/data/meson.build", false),
    ("12-bit/features/meson.build", false),
];

fn parse_meson_build(meson_path: &Path) -> Vec<TestVector> {
    let content = match std::fs::read_to_string(meson_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let dir = meson_path.parent().unwrap();
    let mut entries = Vec::new();
    let mut current = String::new();
    let mut in_entry = false;
    for line in content.lines() {
        let t = line.trim();
        if !in_entry {
            if t.starts_with('[') && t.contains('\'') {
                current = t.to_string();
                if t.contains("],") || t.ends_with(']') {
                    entries.push(current.clone());
                    current.clear();
                } else {
                    in_entry = true;
                }
            }
        } else {
            current.push(' ');
            current.push_str(t);
            if t.contains("],") || t.ends_with(']') {
                entries.push(current.clone());
                current.clear();
                in_entry = false;
            }
        }
    }

    let mut vectors = Vec::new();
    for entry in &entries {
        if !entry.contains("files(") {
            continue;
        }
        let mut quoted = Vec::new();
        let mut chars = entry.chars().peekable();
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
        let name = quoted[0].clone();
        let filename = match quoted
            .iter()
            .find(|s| s.ends_with(".ivf") || s.ends_with(".obu"))
        {
            Some(f) => f.as_str(),
            None => continue,
        };
        let md5 = match quoted
            .iter()
            .rev()
            .find(|s| s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()))
        {
            Some(m) => m.clone(),
            None => continue,
        };
        vectors.push(TestVector {
            name,
            ivf_path: dir.join(filename),
            expected_md5: md5,
        });
    }
    vectors
}

fn hash_frame(frame: &Frame, ctx: &mut md5::Context) {
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                ctx.consume(row);
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    ctx.consume(row);
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    ctx.consume(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            for row in planes.y().rows() {
                for &p in row {
                    ctx.consume(p.to_le_bytes());
                }
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &p in row {
                        ctx.consume(p.to_le_bytes());
                    }
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    for &p in row {
                        ctx.consume(p.to_le_bytes());
                    }
                }
            }
        }
    }
}

fn decode_md5(ivf_path: &Path, apply_grain: bool, threads: u32) -> Result<(String, usize), String> {
    let file = std::fs::File::open(ivf_path).map_err(|e| format!("open: {e}"))?;
    let mut reader = std::io::BufReader::new(file);
    let frames = ivf_parser::parse_all_frames(&mut reader).map_err(|e| format!("ivf: {e}"))?;

    let mut settings = Settings::default();
    settings.apply_grain = apply_grain;
    settings.threads = threads;
    let mut decoder = Decoder::with_settings(settings).map_err(|e| format!("decoder: {e}"))?;
    let mut ctx = md5::Context::new();
    let mut n = 0usize;

    for f in &frames {
        match decoder.decode(&f.data) {
            Ok(Some(frame)) => {
                hash_frame(&frame, &mut ctx);
                n += 1;
            }
            Ok(None) => {}
            Err(e) => return Err(format!("decode frame {n}: {e}")),
        }
    }
    match decoder.flush() {
        Ok(rest) => {
            for frame in &rest {
                hash_frame(frame, &mut ctx);
                n += 1;
            }
        }
        Err(e) => return Err(format!("flush: {e}")),
    }
    Ok((format!("{:x}", ctx.finalize()), n))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut group_filter: Option<String> = None;
    // Groups to DROP. Issue #479: `rav1d_apply_grain_row` bands rows per worker
    // while `filmgrain_arm.rs` takes a whole-plane mutable guard, which aborts
    // 13 of 768 vectors at `threads > 1` on unmodified `main`. Dropping them
    // from BOTH sides of a set-diff keeps the comparison honest; the TOTAL line
    // prints what was skipped so an empty filter cannot pass as a full run.
    let mut skip_groups: Vec<String> = Vec::new();
    let mut name_filter: Option<String> = None;
    let mut activity = false;
    // The corpus gate defaulted to one thread, which made it structurally
    // blind to anything the tile workers do differently — including every
    // change to the borrow tracker, whose whole reason to exist is
    // multi-threaded decode. Run it at 1 AND at 8 and set-diff the two.
    let mut threads = 1u32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--group" => {
                group_filter = args.get(i + 1).cloned();
                i += 2;
            }
            "--skip-group" => {
                skip_groups.push(
                    args.get(i + 1)
                        .cloned()
                        .expect("--skip-group needs a value"),
                );
                i += 2;
            }
            "--name" => {
                name_filter = args.get(i + 1).cloned();
                i += 2;
            }
            // Per-family work counts alongside each vector. Answers "which
            // vectors even exercise family X?" — the question that has to be
            // settled BEFORE budgeting a kernel port, because a profiler
            // reports "never called" and "free" as the same 0.0 ms. Needs
            // `--features __ablate`; without it every count is 0 and the
            // assert below refuses to emit a misleading all-zero column.
            "--threads" => {
                threads = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .expect("--threads needs a number");
                i += 2;
            }
            "--activity" => {
                activity = true;
                i += 1;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/dav1d-test-data");
    assert!(
        base.join("8-bit/data/meson.build").exists(),
        "dav1d-test-data missing at {}",
        base.display()
    );

    assert!(
        !activity || rav1d_safe::src::ablate::ENABLED,
        "--activity needs --features __ablate; without it every count is 0 \
         and the output would read as 'no family does any work'"
    );

    // The `__simd_test` differential harness saves, restores and re-writes the
    // WHOLE picture component around each loopfilter / looprestoration call
    // (`src/loopfilter.rs:140,182`, `src/looprestoration.rs:212,258`). That is
    // sound only single-threaded: measured at `--threads 8 --group 8-bit/data`
    // it produced 313 errors in 358 vectors, e.g.
    //
    //      current: &mut _[163840..163968]   <- a concurrent 128-px row write
    //     existing:    & _[0..983040]        <- the harness's whole-plane save
    //
    // and the restore would clobber other workers' output even with tracking
    // off. Narrowing the guards cannot fix it — the harness *semantically*
    // needs the whole plane. Fail loud rather than emit a TSV of mystery
    // errors that reads as a decoder regression (#479 audit).
    assert!(
        !(cfg!(feature = "__simd_test") && threads > 1),
        "--threads {threads} with --features __simd_test: the differential \
         harness is single-thread-only by construction. Use --threads 1, or \
         drop __simd_test."
    );

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let head = "group\tname\tstatus\texpected\tactual\tframes\twall_ms";
    if activity {
        let fams: Vec<&str> = rav1d_safe::src::ablate::Family::ALL
            .iter()
            .map(|f| f.name())
            .collect();
        writeln!(out, "{head}\t{}", fams.join("\t")).unwrap();
    } else {
        writeln!(out, "{head}").unwrap();
    }

    let (mut pass, mut fail, mut err, mut skip) = (0usize, 0usize, 0usize, 0usize);

    for &(group, grain) in MESON_GROUPS {
        if let Some(g) = &group_filter {
            if !group.contains(g.as_str()) {
                continue;
            }
        }
        if skip_groups.iter().any(|g| group.contains(g.as_str())) {
            continue;
        }
        let meson = base.join(group);
        if !meson.exists() {
            eprintln!("MISSING GROUP {group}");
            continue;
        }
        let group_key = group.trim_end_matches("/meson.build");
        for v in parse_meson_build(&meson) {
            if let Some(nf) = &name_filter {
                if !v.name.contains(nf.as_str()) {
                    continue;
                }
            }
            let ext = v
                .ivf_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            if !v.ivf_path.exists() || ext != "ivf" {
                skip += 1;
                writeln!(
                    out,
                    "{group_key}\t{}\tSKIP\t{}\t-\t0\t0",
                    v.name, v.expected_md5
                )
                .unwrap();
                continue;
            }
            // Marker so a `__simd_test_log` build's per-call `*_MISMATCH` lines
            // (which go to stderr) can be attributed to the vector that
            // produced them. Sound at `--threads 1`, and still sound above
            // it because `decode_md5` joins before returning, so no worker
            // line can straddle two vectors (they may interleave WITHIN one).
            eprintln!("VECTOR\t{group_key}\t{}", v.name);
            rav1d_safe::src::ablate::activity_reset();
            let t0 = Instant::now();
            let res = decode_md5(&v.ivf_path, grain, threads);
            let ms = t0.elapsed().as_millis();
            let act = if activity {
                let counts = rav1d_safe::src::ablate::activity_snapshot();
                let cols: Vec<String> = counts.iter().map(|c| c.to_string()).collect();
                format!("\t{}", cols.join("\t"))
            } else {
                String::new()
            };
            match res {
                Ok((actual, n)) => {
                    let status = if actual == v.expected_md5 {
                        pass += 1;
                        "PASS"
                    } else {
                        fail += 1;
                        "MISMATCH"
                    };
                    writeln!(
                        out,
                        "{group_key}\t{}\t{status}\t{}\t{actual}\t{n}\t{ms}{act}",
                        v.name, v.expected_md5
                    )
                    .unwrap();
                }
                Err(e) => {
                    err += 1;
                    let e = e.replace('\t', " ").replace('\n', " ");
                    writeln!(
                        out,
                        "{group_key}\t{}\tERROR\t{}\tERR:{e}\t0\t{ms}{act}",
                        v.name, v.expected_md5
                    )
                    .unwrap();
                }
            }
            out.flush().unwrap();
        }
        eprintln!("done {group_key}");
    }

    let skipped = if skip_groups.is_empty() {
        "-".to_string()
    } else {
        skip_groups.join(",")
    };
    eprintln!(
        "TOTAL threads={threads} pass={pass} mismatch={fail} error={err} skip={skip} skipped_groups={skipped}"
    );
}
