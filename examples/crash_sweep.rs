//! Sweep a directory of fuzz-farm crash artifacts through every fuzz target's
//! entry point and report which ones still panic on this tree.
//!
//! The continuous fuzz farm files one issue per panic *signature* but keeps
//! every artifact that produced it (hundreds per signature in
//! `s3://zenfuzz/crashes/rav1d-safe/<target>/<arch>/<sig_hash>/`). Triaging an
//! issue means running all of them, not the one named in the issue body — a
//! signature that no longer reproduces from 700 inputs is stale; one that
//! reproduces from 2 is live. `tests/fuzz_regression.rs` gates the committed
//! seeds; this tool answers the question for a directory you just synced.
//!
//! Entry points mirror `tests/fuzz_regression.rs` exactly (the three fuzz
//! targets' settings plus production defaults). A panic is caught per
//! (file, entry point) pair and reported with its location, so one run over a
//! directory names every live crash and its panic site.
//!
//! ```text
//! cargo run --release --example crash_sweep -- ~/tmp/crashes/439 ~/tmp/crashes/444
//! ```
//!
//! Exit status is non-zero when any pair panicked.

use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rav1d_safe::src::managed::{DecodeFrameType, Decoder, InloopFilters, Settings};

/// Matches `frame_size_limit` in all three fuzz targets.
const FRAME_SIZE_LIMIT_PIXELS: u32 = 256 * 256;

/// Location + message of the most recent panic, recorded by the hook.
static LAST_PANIC: Mutex<Option<String>> = Mutex::new(None);

fn drive(mut decoder: Decoder, data: &[u8]) -> bool {
    match decoder.decode(data) {
        Ok(Some(_)) => true,
        Ok(None) | Err(_) => matches!(decoder.flush(), Ok(f) if !f.is_empty()),
    }
}

fn run_decode_obu(data: &[u8]) -> bool {
    let mut settings = Settings::default();
    settings.frame_size_limit = FRAME_SIZE_LIMIT_PIXELS;
    Decoder::with_settings(settings).is_ok_and(|d| drive(d, data))
}

fn run_parse_seq_header(data: &[u8]) -> bool {
    let mut settings = Settings::default();
    settings.threads = 1;
    settings.frame_size_limit = FRAME_SIZE_LIMIT_PIXELS;
    settings.inloop_filters = InloopFilters::none();
    settings.decode_frame_type = DecodeFrameType::All;
    Decoder::with_settings(settings).is_ok_and(|d| drive(d, data))
}

fn run_differential_rav1d_half(data: &[u8]) -> bool {
    let mut settings = Settings::default();
    settings.threads = 1;
    settings.max_frame_delay = 1;
    settings.frame_size_limit = FRAME_SIZE_LIMIT_PIXELS;
    settings.apply_grain = false;
    Decoder::with_settings(settings).is_ok_and(|d| drive(d, data))
}

fn run_default_settings(data: &[u8]) -> bool {
    Decoder::new().is_ok_and(|d| drive(d, data))
}

type EntryPoint = (&'static str, fn(&[u8]) -> bool);

const ENTRY_POINTS: &[EntryPoint] = &[
    ("decode_obu", run_decode_obu),
    ("parse_seq_header", run_parse_seq_header),
    (
        "differential_dav1d[rav1d-half]",
        run_differential_rav1d_half,
    ),
    ("default_settings", run_default_settings),
];

fn collect_files(path: &Path, out: &mut Vec<PathBuf>) {
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
            collect_files(&p, out);
        } else if p
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("crash-") || n.ends_with(".obu"))
        {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: crash_sweep <file-or-dir>...");
        std::process::exit(2);
    }
    let mut files = Vec::new();
    for a in &args {
        collect_files(Path::new(a), &mut files);
    }
    if files.is_empty() {
        eprintln!("no crash-* / *.obu files found under {:?}", args);
        std::process::exit(2);
    }

    // Record the panic site instead of letting the default hook spray the
    // backtrace for every pair; the summary at the end is the report.
    panic::set_hook(Box::new(|info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| info.payload().downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        *LAST_PANIC.lock().unwrap() = Some(format!("{loc}: {msg}"));
    }));

    let mut panicked: Vec<(String, &str, String)> = Vec::new();
    let mut runs = 0usize;
    let mut frames = 0usize;
    for f in &files {
        let data = match fs::read(f) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("read {}: {e}", f.display());
                continue;
            }
        };
        for (name, ep) in ENTRY_POINTS {
            runs += 1;
            match panic::catch_unwind(AssertUnwindSafe(|| ep(&data))) {
                Ok(true) => frames += 1,
                Ok(false) => {}
                Err(_) => {
                    let site = LAST_PANIC.lock().unwrap().take().unwrap_or_default();
                    panicked.push((f.display().to_string(), name, site));
                }
            }
        }
    }

    let _ = panic::take_hook();
    println!(
        "{} files, {} runs, {} decoded a frame, {} panicked",
        files.len(),
        runs,
        frames,
        panicked.len()
    );
    // Group by panic site so a 700-artifact sweep reads as a handful of lines.
    let mut by_site: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for (file, ep, site) in &panicked {
        by_site
            .entry(site.clone())
            .or_default()
            .push(format!("{file} via {ep}"));
    }
    for (site, pairs) in &by_site {
        println!("\n{} pair(s) at {site}", pairs.len());
        for p in pairs.iter().take(5) {
            println!("  {p}");
        }
        if pairs.len() > 5 {
            println!("  ... {} more", pairs.len() - 5);
        }
    }
    if !panicked.is_empty() {
        std::process::exit(1);
    }
}
