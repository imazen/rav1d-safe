//! Causal kernel attribution for MD5 conformance failures (`__ablate` feature).
//!
//! Runs the dav1d-test-data corpus once per ablation arm — `none` (the
//! baseline), one arm per SIMD kernel family with that family forced to the
//! generic scalar reference, and `all` with every family scalar — and emits one
//! TSV row per (arm, vector).
//!
//! The reason to ablate rather than read the `__simd_test_log` mismatch counts:
//! a logged mismatch says a kernel *diverges from scalar somewhere in this
//! stream*, not that the divergence is what corrupted the final MD5. A kernel
//! can diverge on a block that is later fully overwritten, and a stream can
//! fail for a reason no harness covers. Ablation answers the question the fix
//! agent actually has — "if I make this kernel bit-exact, which vectors turn
//! green?" — because a vector that flips PASS under `--ablate itx` fails *only*
//! because of itx.
//!
//! Usage:
//!   cargo build --release --features __ablate --example md5_ablate
//!   ./target/release/examples/md5_ablate > ~/tmp/ablate.tsv
//!
//! Args:
//!   --arms a,b,c   only run these arms (names from `Family::name`, plus
//!                  `none` and `all`); default: all arms
//!   --only-failing <baseline.tsv>
//!                  restrict to vectors marked MISMATCH in a prior inventory
//!                  TSV (cheap: the passing vectors cannot flip to PASS)

use rav1d_safe::src::ablate::Family;
use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

struct TestVector {
    name: String,
    ivf_path: PathBuf,
    expected_md5: String,
}

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

fn decode_md5(ivf_path: &Path, apply_grain: bool) -> Result<String, String> {
    let file = std::fs::File::open(ivf_path).map_err(|e| format!("open: {e}"))?;
    let mut reader = std::io::BufReader::new(file);
    let frames = ivf_parser::parse_all_frames(&mut reader).map_err(|e| format!("ivf: {e}"))?;
    let mut settings = Settings::default();
    settings.apply_grain = apply_grain;
    let mut decoder = Decoder::with_settings(settings).map_err(|e| format!("decoder: {e}"))?;
    let mut ctx = md5::Context::new();
    for f in &frames {
        match decoder.decode(&f.data) {
            Ok(Some(frame)) => hash_frame(&frame, &mut ctx),
            Ok(None) => {}
            Err(e) => return Err(format!("decode: {e}")),
        }
    }
    match decoder.flush() {
        Ok(rest) => {
            for frame in &rest {
                hash_frame(frame, &mut ctx);
            }
        }
        Err(e) => return Err(format!("flush: {e}")),
    }
    Ok(format!("{:x}", ctx.finalize()))
}

fn main() {
    // Liveness gate. Without `__ablate` every dispatcher's guard folds to a
    // constant `false`, so all arms would decode identically and the run would
    // report "no kernel is responsible for anything" — a green-looking result
    // that measures nothing. Refuse rather than emit it.
    assert!(
        rav1d_safe::src::ablate::ENABLED,
        "build with --features __ablate; without it every arm is the same decoder"
    );

    let args: Vec<String> = std::env::args().collect();
    let mut arm_filter: Option<Vec<String>> = None;
    let mut only_failing: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--arms" => {
                arm_filter = Some(
                    args[i + 1]
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect(),
                );
                i += 2;
            }
            "--only-failing" => {
                only_failing = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    // Restrict to the baseline's failing set when asked: a PASS vector cannot
    // become "more PASS", so ablating on it wastes wall time. Note the arm still
    // has to be checked for *regressions* on passing vectors if a fix ships —
    // that is the fix agent's gate, not this attribution pass.
    let failing: Option<HashSet<String>> = only_failing.map(|p| {
        let text =
            std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        text.lines()
            .skip(1)
            .filter_map(|l| {
                let mut f = l.split('\t');
                let group = f.next()?;
                let name = f.next()?;
                let status = f.next()?;
                (status == "MISMATCH").then(|| format!("{group}\t{name}"))
            })
            .collect()
    });

    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/dav1d-test-data");

    // Arm list, three shapes:
    //   `none`        — baseline, every kernel NEON.
    //   `<fam>`       — that family scalar, the rest NEON. A vector that flips
    //                   to PASS is broken *only* by that family (necessary).
    //   `only_<fam>`  — that family NEON, every other family scalar. A vector
    //                   that MISMATCHes is broken by that family *on its own*
    //                   (sufficient). This is the dual, and it is what
    //                   disambiguates a vector that two families each corrupt:
    //                   neither single ablation repairs it, so it is invisible
    //                   in the `<fam>` arms alone.
    //   `all`         — every kernel scalar; the conformance floor.
    let mut arms: Vec<(String, Vec<Family>)> = vec![("none".into(), vec![])];
    for f in Family::ALL {
        arms.push((f.name().to_string(), vec![*f]));
    }
    for f in Family::ALL {
        let others: Vec<Family> = Family::ALL.iter().copied().filter(|g| g != f).collect();
        arms.push((format!("only_{}", f.name()), others));
    }
    arms.push(("all".into(), Family::ALL.to_vec()));
    if let Some(want) = &arm_filter {
        arms.retain(|(n, _)| want.iter().any(|w| w == n));
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "arm\tgroup\tname\tstatus\texpected\tactual").unwrap();

    for (arm_name, fams) in &arms {
        rav1d_safe::src::ablate::set_disabled(fams);
        let (mut pass, mut fail, mut err) = (0usize, 0usize, 0usize);
        for &(group, grain) in MESON_GROUPS {
            let meson = base.join(group);
            if !meson.exists() {
                continue;
            }
            let group_key = group.trim_end_matches("/meson.build");
            for v in parse_meson_build(&meson) {
                if let Some(set) = &failing {
                    if !set.contains(&format!("{group_key}\t{}", v.name)) {
                        continue;
                    }
                }
                let ext = v
                    .ivf_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                if !v.ivf_path.exists() || ext != "ivf" {
                    continue;
                }
                match decode_md5(&v.ivf_path, grain) {
                    Ok(actual) => {
                        let status = if actual == v.expected_md5 {
                            pass += 1;
                            "PASS"
                        } else {
                            fail += 1;
                            "MISMATCH"
                        };
                        writeln!(
                            out,
                            "{arm_name}\t{group_key}\t{}\t{status}\t{}\t{actual}",
                            v.name, v.expected_md5
                        )
                        .unwrap();
                    }
                    Err(e) => {
                        err += 1;
                        let e = e.replace('\t', " ").replace('\n', " ");
                        writeln!(
                            out,
                            "{arm_name}\t{group_key}\t{}\tERROR\t{}\tERR:{e}",
                            v.name, v.expected_md5
                        )
                        .unwrap();
                    }
                }
            }
        }
        out.flush().unwrap();
        eprintln!("arm={arm_name} pass={pass} mismatch={fail} error={err}");
    }
    rav1d_safe::src::ablate::set_disabled(&[]);
}
