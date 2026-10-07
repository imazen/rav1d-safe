//! Every committed generated stream must decode identically at any thread count and
//! frame delay, in whichever tracking mode this is built with.
//!
//! This is the gate that was missing when element-granularity above-context tracking
//! (`f.a`) assumed tile workers never share a slot: with 64x64 superblocks and 64-pixel
//! tile columns two workers legitimately use the two halves of one 128-pixel slot, and
//! the tracked build panicked ("overlapping DisjointMut") on every such stream at 2+
//! threads. The generated corpus has exactly those streams (`tiles1x4`, `tiles2x2`).

use rav1d_safe::src::managed::{Decoder, Frame, Settings};

#[path = "common/committed_vectors.rs"]
#[allow(dead_code)]
mod committed_vectors;

fn decode(data: &[u8], threads: u32, delay: u32) -> Result<String, String> {
    let mut s = Settings::default();
    s.threads = threads;
    s.max_frame_delay = delay;
    let mut d = Decoder::with_settings(s).map_err(|e| e.to_string())?;
    let mut ctx = md5::Context::new();
    let mut hash = |f: &Frame| committed_vectors::hash_frame(f, &mut ctx);
    if let Some(f) = d.decode(data).map_err(|e| e.to_string())? {
        hash(&f);
    }
    while let Some(f) = d.get_frame().map_err(|e| e.to_string())? {
        hash(&f);
    }
    for f in d.flush().map_err(|e| e.to_string())? {
        hash(&f);
    }
    Ok(format!("{:x}", ctx.finalize()))
}

#[test]
fn committed_streams_are_thread_and_delay_invariant() {
    let dir = format!("{}/tests/gen_vectors", env!("CARGO_MANIFEST_DIR"));
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "obu"))
        .collect();
    names.sort();
    assert!(names.len() >= 20, "corpus missing: {}", names.len());
    let mut failures = Vec::new();
    for path in &names {
        let data = std::fs::read(path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let reference = decode(&data, 1, 1).unwrap_or_else(|e| panic!("{name} t=1: {e}"));
        for (threads, delay) in [(2, 1), (4, 1), (8, 1), (2, 2), (4, 0), (8, 3)] {
            // A tracker overlap panics a worker; that surfaces as an error or a panic.
            let got = std::panic::catch_unwind(|| decode(&data, threads, delay));
            match got {
                Ok(Ok(h)) if h == reference => {}
                Ok(Ok(h)) => failures.push(format!(
                    "{name} t={threads} d={delay}: md5 {h} != {reference}"
                )),
                Ok(Err(e)) => failures.push(format!("{name} t={threads} d={delay}: error {e}")),
                Err(_) => failures.push(format!("{name} t={threads} d={delay}: panic")),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
