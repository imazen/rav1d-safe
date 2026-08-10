//! THROWAWAY driver: run the `__probe_bounds` map over dav1d-test-data.
//!
//! The 4K gap vectors this campaign measures on have loop restoration switched
//! OFF, so a bounds map built only from them says nothing about the LR path —
//! which is live in most of the corpus. This driver decodes a named subset of
//! the corpus in ONE process, so all the vectors' acquisitions land in one
//! concurrency map, and prints a per-vector liveness line first so that
//! "LR was exercised" is a measurement rather than an assumption.
//!
//! Usage:
//!   probe_bounds_corpus <threads> [--group <substr>] [--name <substr>]
//!                       [--limit <n>] [--lr-only] [--frames <n>]
//!
//! `--lr-only` runs every selected vector once as a PROBE (counting only), keeps
//! the ones whose census contains a `looprestoration`/`lr_apply` site, resets,
//! and then runs the map over just those.
//!
//! Half of this file's bindings only exist for the `__probe_bounds` build; the
//! example still has to COMPILE in the default one (it is an `--all-targets`
//! clippy target), and adding a warning to the default build is not acceptable
//! for a throwaway.
#![cfg_attr(not(feature = "__probe_bounds"), allow(unused))]

use rav1d_safe::src::managed::{Decoder, Settings};
use std::path::{Path, PathBuf};

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

const MESON_GROUPS: &[&str] = &[
    "8-bit/data/meson.build",
    "8-bit/features/meson.build",
    "8-bit/issues/meson.build",
    "8-bit/quantizer/meson.build",
    "8-bit/size/meson.build",
    "8-bit/cdfupdate/meson.build",
    "8-bit/vq_suite/meson.build",
    "8-bit/intra/meson.build",
    "8-bit/mfmv/meson.build",
    "8-bit/mv/meson.build",
    "8-bit/resize/meson.build",
    "10-bit/data/meson.build",
    "10-bit/features/meson.build",
    "10-bit/quantizer/meson.build",
    "10-bit/issues/meson.build",
    "12-bit/data/meson.build",
    "12-bit/features/meson.build",
];

fn corpus_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/dav1d-test-data")
}

fn parse_meson(meson: &Path) -> Vec<(String, PathBuf)> {
    // Same shape as `examples/md5_inventory.rs`: entries are `[ 'name', ...,
    // files('x.ivf'), ..., '<md5>' ]`, possibly wrapped over several lines.
    let Ok(content) = std::fs::read_to_string(meson) else {
        return Vec::new();
    };
    let dir = meson.parent().unwrap();
    let mut entries = Vec::new();
    let mut current = String::new();
    let mut in_entry = false;
    for line in content.lines() {
        let t = line.trim();
        if !in_entry {
            if t.starts_with('[') && t.contains('\'') {
                current = t.to_string();
                if t.contains("],") || t.ends_with(']') {
                    entries.push(std::mem::take(&mut current));
                } else {
                    in_entry = true;
                }
            }
        } else {
            current.push(' ');
            current.push_str(t);
            if t.contains("],") || t.ends_with(']') {
                entries.push(std::mem::take(&mut current));
                in_entry = false;
            }
        }
    }
    let mut out = Vec::new();
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
        let Some(file) = quoted
            .iter()
            .find(|s| s.ends_with(".ivf") || s.ends_with(".obu"))
        else {
            continue;
        };
        out.push((quoted[0].clone(), dir.join(file)));
    }
    out
}

fn decode(path: &Path, threads: u32, max_frames: usize) -> Option<usize> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let frames = ivf_parser::parse_all_frames(&mut reader).ok()?;
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.frame_size_limit = 8192 * 8192;
    settings.apply_grain = false;
    let mut dec = Decoder::with_settings(settings).ok()?;
    let mut n = 0usize;
    for f in frames.iter().take(max_frames) {
        match dec.decode(&f.data) {
            Ok(Some(_)) => n += 1,
            Ok(None) => {}
            Err(_) => return Some(n),
        }
    }
    if let Ok(rest) = dec.flush() {
        n += rest.len();
    }
    Some(n)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut group = String::new();
    let mut name = String::new();
    let mut limit = usize::MAX;
    let mut frames = 8usize;
    let mut lr_only = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--group" => {
                group = args[i + 1].clone();
                i += 2;
            }
            "--name" => {
                name = args[i + 1].clone();
                i += 2;
            }
            "--limit" => {
                limit = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--frames" => {
                frames = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--lr-only" => {
                lr_only = true;
                i += 1;
            }
            other => panic!("unknown arg {other}"),
        }
    }

    let root = corpus_root();
    let mut vectors: Vec<(String, String, PathBuf)> = Vec::new();
    for g in MESON_GROUPS {
        if !group.is_empty() && !g.contains(&group) {
            continue;
        }
        for (n, p) in parse_meson(&root.join(g)) {
            if !name.is_empty() && !n.contains(&name) {
                continue;
            }
            vectors.push((g.to_string(), n, p));
        }
    }
    vectors.sort();
    eprintln!("selected {} vectors", vectors.len());

    // Pass 1 (optional): find which vectors exercise loop restoration at all.
    if lr_only {
        let mut keep = Vec::new();
        for (g, n, p) in vectors.iter() {
            #[cfg(feature = "__probe_bounds")]
            rav1d_disjoint_mut::bounds_probe::reset();
            let f = decode(p, 1, frames).unwrap_or(0);
            #[cfg(feature = "__probe_bounds")]
            {
                let (_tot, lr) = rav1d_disjoint_mut::bounds_probe::regs_matching("looprestoration");
                let (_t2, lra) = rav1d_disjoint_mut::bounds_probe::regs_matching("lr_apply");
                if f > 0 && lr + lra > 0 {
                    keep.push((g.clone(), n.clone(), p.clone()));
                }
            }
        }
        eprintln!("{} of {} vectors have LR live", keep.len(), vectors.len());
        vectors = keep;
    }
    vectors.truncate(limit);

    #[cfg(feature = "probe-sites")]
    rav1d_disjoint_mut::site_probe::reset();
    #[cfg(feature = "__probe_bounds")]
    rav1d_disjoint_mut::bounds_probe::reset();

    let mut total_frames = 0usize;
    let mut prev_total = 0u64;
    let mut prev_lr = 0u64;
    println!("#vec\tgroup\tname\tframes\tregs\tlr_regs");
    for (g, n, p) in vectors.iter() {
        let f = decode(p, threads, frames).unwrap_or(0);
        total_frames += f;
        #[cfg(feature = "__probe_bounds")]
        {
            let (tot, lr) = rav1d_disjoint_mut::bounds_probe::regs_matching("looprestoration");
            let (_t, lra) = rav1d_disjoint_mut::bounds_probe::regs_matching("lr_apply");
            let lr = lr + lra;
            println!(
                "VEC\t{}\t{}\t{}\t{}\t{}",
                g.trim_end_matches("/meson.build"),
                n,
                f,
                tot - prev_total,
                lr - prev_lr
            );
            prev_total = tot;
            prev_lr = lr;
        }
        #[cfg(not(feature = "__probe_bounds"))]
        println!("VEC\t{}\t{}\t{}\t-\t-", g, n, f);
    }
    println!(
        "CORPUS\tvectors={}\tframes={total_frames}\tthreads={threads}",
        vectors.len()
    );

    #[cfg(feature = "probe-sites")]
    print!("{}", rav1d_disjoint_mut::site_probe::report(1));
    #[cfg(feature = "__probe_bounds")]
    print!("{}", rav1d_disjoint_mut::bounds_probe::report(1));
}
