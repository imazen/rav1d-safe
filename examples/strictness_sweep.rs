//! Decode every AV1 stream under the given paths with `Strictness::Lenient` and
//! `Strictness::Strict` and report where the two verdicts differ.
//!
//! Input forms: raw OBU streams (`*.obu`, `crash-*`), AVIF containers (the
//! primary item's AV1 payload, via `zenavif-parse`), and IVF files (frames
//! concatenated into one chunk). The question it answers is what a stricter
//! default costs on real content: a disagreement is either a non-conforming file
//! that `Strict` correctly rejects, or a false positive that must be fixed before
//! `Strict` can be the default. `benchmarks/strictness_2026-08-28.meta` holds the
//! numbers this produced for the corpora on hand when the default was chosen.
//!
//! ```text
//! cargo run --release --example strictness_sweep -- [--jobs N] [--quiet] <file-or-dir>...
//! ```
//!
//! Exit status is always 0 — this is a survey, not a gate. Use `--quiet` to print
//! only the disagreements and the summary.
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rav1d_safe::src::managed::{Decoder, Settings, Strictness};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Frame,
    NoFrame,
    Rejected,
}

impl Verdict {
    fn tag(self) -> &'static str {
        match self {
            Verdict::Frame => "frame",
            Verdict::NoFrame => "nofrm",
            Verdict::Rejected => "ERROR",
        }
    }
}

fn decode(strictness: Strictness, data: &[u8]) -> Verdict {
    let mut settings = Settings::default();
    settings.strictness = strictness;
    let Ok(mut decoder) = Decoder::with_settings(settings) else {
        return Verdict::Rejected;
    };
    match decoder.decode(data) {
        Ok(Some(_)) => Verdict::Frame,
        Ok(None) => match decoder.flush() {
            Ok(frames) if !frames.is_empty() => Verdict::Frame,
            _ => Verdict::NoFrame,
        },
        Err(_) => match decoder.flush() {
            Ok(frames) if !frames.is_empty() => Verdict::Frame,
            _ => Verdict::Rejected,
        },
    }
}

/// Reduce a container to the raw OBU bytes the decoder wants.
fn payload(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() >= 12 && &data[4..8] == b"ftyp" {
        let parser = zenavif_parse::AvifParser::from_bytes(data).ok()?;
        return Some(parser.primary_data().ok()?.into_owned());
    }
    if data.len() >= 32 && &data[..4] == b"DKIF" {
        let mut out = Vec::new();
        let mut pos = u16::from_le_bytes([data[6], data[7]]) as usize;
        while pos + 12 <= data.len() {
            let size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 12;
            let end = pos.checked_add(size)?.min(data.len());
            out.extend_from_slice(&data[pos..end]);
            pos = end;
        }
        return Some(out);
    }
    Some(data.to_vec())
}

fn wanted(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    name.starts_with("crash-")
        || lower.ends_with(".obu")
        || lower.ends_with(".avif")
        || lower.ends_with(".ivf")
        || lower.ends_with(".av1")
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        out.push(path.to_path_buf());
        return;
    }
    let Ok(rd) = fs::read_dir(path) else {
        eprintln!("cannot read {}", path.display());
        return;
    };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect(&p, out);
        } else if p.file_name().and_then(|n| n.to_str()).is_some_and(wanted) {
            out.push(p);
        }
    }
}

fn main() {
    let mut jobs = 4usize;
    let mut quiet = false;
    let mut roots = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--jobs" => jobs = args.next().and_then(|v| v.parse().ok()).unwrap_or(4),
            "--quiet" => quiet = true,
            _ => roots.push(a),
        }
    }
    if roots.is_empty() {
        eprintln!("usage: strictness_sweep [--jobs N] [--quiet] <file-or-dir>...");
        std::process::exit(2);
    }
    let mut files = Vec::new();
    for r in &roots {
        collect(Path::new(r), &mut files);
    }
    if files.is_empty() {
        eprintln!("no .obu/.avif/.ivf/crash-* files under {roots:?}");
        std::process::exit(2);
    }

    let next = AtomicUsize::new(0);
    let counts = Mutex::new([[0usize; 3]; 3]); // [lenient][strict]
    let disagreements = Mutex::new(Vec::new());
    let unreadable = AtomicUsize::new(0);
    let idx = |v: Verdict| match v {
        Verdict::Frame => 0,
        Verdict::NoFrame => 1,
        Verdict::Rejected => 2,
    };

    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = files.get(i) else { break };
                    let Some(data) = fs::read(path).ok().and_then(|d| payload(&d)) else {
                        unreadable.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let lenient = decode(Strictness::Lenient, &data);
                    let strict = decode(Strictness::Strict, &data);
                    counts.lock().unwrap()[idx(lenient)][idx(strict)] += 1;
                    let line = format!(
                        "lenient={} strict={}  {}",
                        lenient.tag(),
                        strict.tag(),
                        path.display()
                    );
                    if lenient != strict {
                        disagreements.lock().unwrap().push(line.clone());
                        println!("DISAGREE {line}");
                    } else if !quiet {
                        println!("{line}");
                    }
                }
            });
        }
    });

    let c = counts.into_inner().unwrap();
    let d = disagreements.into_inner().unwrap();
    let tags = ["frame", "nofrm", "error"];
    println!();
    println!(
        "{} files ({} unreadable), {} disagreements",
        files.len(),
        unreadable.load(Ordering::Relaxed),
        d.len()
    );
    println!(
        "{:>16} | {:>7} {:>7} {:>7}  <- strict",
        "lenient", tags[0], tags[1], tags[2]
    );
    for (i, row) in c.iter().enumerate() {
        println!(
            "{:>16} | {:>7} {:>7} {:>7}",
            tags[i], row[0], row[1], row[2]
        );
    }
    for line in &d {
        println!("  {line}");
    }
}
