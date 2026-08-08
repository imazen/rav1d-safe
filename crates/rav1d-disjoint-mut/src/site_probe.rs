//! THROWAWAY per-call-site borrow counter (feature `__probe_sites`).
//!
//! Never merge into a shipping configuration. It exists to answer one question
//! that `sample` cannot: *how many* borrow registrations does each source line
//! take, as opposed to how much time the tracker spends on that line's behalf.
//! Self-time attribution folds the tracker into whichever caller LLVM inlined
//! it into and says nothing about counts.
//!
//! The whole `index`/`index_mut` -> `add_mut`/`add_immut` -> `add` chain is
//! already `#[track_caller]` in the shipping build, so `Location::caller()`
//! inside `add` is the real borrow site with no extra plumbing.
//!
//! Storage is a fixed open-addressed table keyed on the `&'static Location`
//! POINTER (not its contents) — one `Location` per source site, so pointer
//! identity is site identity and the hash is free.

use core::panic::Location;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering::Relaxed;
use std::string::String;
use std::vec::Vec;

/// Power of two. 4 K sites is far more than the decoder has.
const CAP: usize = 4096;

pub struct Site {
    key: AtomicUsize,
    n_mut: AtomicU64,
    n_immut: AtomicU64,
    /// Sum of `end - start` over every registration, so a site's mean guard
    /// extent is derivable (the coarsening question is "few big" vs "many
    /// small", which a bare count cannot answer).
    bytes: AtomicU64,
}

impl Site {
    const fn new() -> Self {
        Self {
            key: AtomicUsize::new(0),
            n_mut: AtomicU64::new(0),
            n_immut: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }
}

pub static SITES: [Site; CAP] = [const { Site::new() }; CAP];
/// Registrations dropped because the table filled (should stay 0).
pub static LOST: AtomicU64 = AtomicU64::new(0);
/// key -> `Location`, appended under a lock the first time a key is claimed, so
/// `report` can name a site without turning a `usize` back into a reference.
/// Off the hot path: one push per distinct source line, ever.
static NAMES: std::sync::Mutex<Vec<(usize, &'static Location<'static>)>> =
    std::sync::Mutex::new(Vec::new());

#[inline]
fn slot_for(key: usize, loc: &'static Location<'static>) -> Option<usize> {
    // Fibonacci hash of the pointer; `Location`s are statics so the low bits
    // are not uniformly distributed.
    let mut h = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize & (CAP - 1);
    for _ in 0..64 {
        let cur = SITES[h].key.load(Relaxed);
        if cur == key {
            return Some(h);
        }
        if cur == 0
            && SITES[h]
                .key
                .compare_exchange(0, key, Relaxed, Relaxed)
                .is_ok()
        {
            if let Ok(mut n) = NAMES.lock() {
                n.push((key, loc));
            }
            return Some(h);
        }
        h = (h + 1) & (CAP - 1);
    }
    None
}

#[inline]
pub fn record(loc: &'static Location<'static>, is_mut: bool, len: usize) {
    let key = loc as *const Location<'static> as usize;
    match slot_for(key, loc) {
        Some(h) => {
            if is_mut {
                SITES[h].n_mut.fetch_add(1, Relaxed);
            } else {
                SITES[h].n_immut.fetch_add(1, Relaxed);
            }
            SITES[h].bytes.fetch_add(len as u64, Relaxed);
        }
        None => {
            LOST.fetch_add(1, Relaxed);
        }
    }
}

/// Zeroes the COUNTERS but keeps the key→slot assignment, so a reset between a
/// warmup decode and the timed decodes does not re-race the CAS.
pub fn reset() {
    for s in SITES.iter() {
        s.n_mut.store(0, Relaxed);
        s.n_immut.store(0, Relaxed);
        s.bytes.store(0, Relaxed);
    }
    LOST.store(0, Relaxed);
    for n in crate::site_class::NULLED.iter() {
        n.store(0, Relaxed);
    }
    for c in crate::site_class::BY_SIZE.iter() {
        for n in c.iter() {
            n.store(0, Relaxed);
        }
    }
}

/// TSV: `SITE <n_total_per_frame> <n_mut> <n_immut> <mean_bytes> <file:line:col>`,
/// descending by count. `frames` divides the counters so the numbers are
/// per-frame.
pub fn report(frames: u64) -> String {
    use std::fmt::Write as _;
    let f = frames.max(1) as f64;
    let mut rows: Vec<(u64, u64, u64, String)> = Vec::new();
    let mut total = 0u64;
    let names = NAMES.lock().ok();
    for s in SITES.iter() {
        let key = s.key.load(Relaxed);
        if key == 0 {
            continue;
        }
        let m = s.n_mut.load(Relaxed);
        let i = s.n_immut.load(Relaxed);
        if m + i == 0 {
            continue;
        }
        total += m + i;
        let where_ = names
            .as_ref()
            .and_then(|n| n.iter().find(|(k, _)| *k == key))
            .map(|(_, l)| std::format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| std::format!("?{key:#x}"));
        rows.push((m + i, m, s.bytes.load(Relaxed), where_));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = String::new();
    let _ = writeln!(
        out,
        "SITES\ttotal_per_frame={:.0}\tdistinct={}\tlost={}\tmemo_misses={}",
        total as f64 / f,
        rows.len(),
        LOST.load(Relaxed),
        crate::site_class::MEMO_MISSES.load(Relaxed),
    );
    // Per-class totals. This is the census half of the site-class instrument:
    // it proves WHICH class each site landed in, and it is the only thing that
    // can catch a `#[track_caller]` chain that stops at a pass-through wrapper
    // (borrows then pile up on `picture.rs` and classify as `other`, which
    // would silently make the recon/filter split meaningless).
    let mut per_class = [0u64; crate::site_class::N_CLASSES];
    for (n, _, _, w) in rows.iter() {
        let file = w.split(':').next().unwrap_or("");
        per_class[crate::site_class::class_for_file_pub(file) as usize] += *n;
    }
    for (i, name) in crate::site_class::CLASS_NAMES.iter().enumerate() {
        let _ = writeln!(
            out,
            "CLASS\t{}\t{:.0}\t{:.1}%\tnulled={:.0}\tbig={:.0}\tsmall={:.0}",
            name,
            per_class[i] as f64 / f,
            100.0 * per_class[i] as f64 / total.max(1) as f64,
            crate::site_class::NULLED[i].load(Relaxed) as f64 / f,
            crate::site_class::BY_SIZE[i][1].load(Relaxed) as f64 / f,
            crate::site_class::BY_SIZE[i][0].load(Relaxed) as f64 / f,
        );
    }
    let _ = writeln!(
        out,
        "#site\tper_frame\tmut\timmut\tmean_bytes\tcum_pct\tclass\twhere"
    );
    let mut cum = 0u64;
    for (n, m, b, w) in rows.iter() {
        cum += n;
        let file = w.split(':').next().unwrap_or("");
        let _ = writeln!(
            out,
            "SITE\t{:.0}\t{:.0}\t{:.0}\t{:.1}\t{:.1}%\t{}\t{}",
            *n as f64 / f,
            *m as f64 / f,
            (*n - *m) as f64 / f,
            *b as f64 / *n as f64,
            100.0 * cum as f64 / total as f64,
            crate::site_class::CLASS_NAMES[crate::site_class::class_for_file_pub(file) as usize],
            w
        );
    }
    out
}
