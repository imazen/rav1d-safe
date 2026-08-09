//! THROWAWAY measurement probe. Never merge.
//!
//! Answers the P1 (scaling-plateau) questions that cannot be answered by
//! wall-clock A/B alone:
//!
//! 1. **Is work distributed N ways?** Per-worker busy nanoseconds, per task
//!    stage. A straggler shows up as one worker with far more busy time than
//!    the rest; an under-fed pool shows up as low aggregate busy against
//!    `wall * threads`.
//! 2. **Are the post-tile stages serial?** The five filter stages
//!    (deblock-cols / deblock-rows / cdef / super-res / loop-restoration) are
//!    driven by ONE task per frame sbrow that falls through all of them, so
//!    their summed time is a candidate Amdahl term. Measured directly.
//! 3. **How much concurrency is actually realised?** A sampling monitor reads
//!    the count of workers inside a stage body every `SAMPLE_US` and builds a
//!    time-weighted histogram. `mean(active)` IS the achieved parallelism.
//!
//! Cost: two `Instant::now()` per stage execution. A 4K frame runs ~170 stage
//! executions (136 tile-sbrow + 34 filter-sbrow), so ~340 clock reads per
//! frame against a ~330 ms frame — unmeasurable. This is deliberately unlike
//! the tracker probes, which sit on a 50-million-per-frame path.
//!
//! Counters are process-global and never reset between reps; the driver prints
//! and divides by the frame count.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub const MAX_WORKERS: usize = 72;
pub const N_STAGE: usize = 8;

pub const STAGE_NAMES: [&str; N_STAGE] = [
    "tile_entropy",
    "tile_recon",
    "deblock_cols",
    "deblock_rows",
    "cdef",
    "superres",
    "loop_restore",
    "other",
];

/// Which stages make up the single-task-at-a-time post-filter chain.
pub const FILTER_STAGES: [usize; 5] = [2, 3, 4, 5, 6];

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);

/// Flat `[worker][stage]` tables; nested const-repeat needs `Copy`, which
/// atomics are not, so the index is computed as `w * N_STAGE + s`.
static BUSY_NS: [AtomicU64; N_STAGE * MAX_WORKERS] = [ZERO; N_STAGE * MAX_WORKERS];
static BUSY_CNT: [AtomicU64; N_STAGE * MAX_WORKERS] = [ZERO; N_STAGE * MAX_WORKERS];

#[inline]
const fn ix(w: usize, s: usize) -> usize {
    w * N_STAGE + s
}
static PARK_NS: [AtomicU64; MAX_WORKERS] = [ZERO; MAX_WORKERS];
static PARK_CNT: [AtomicU64; MAX_WORKERS] = [ZERO; MAX_WORKERS];

/// Number of workers currently executing a stage body. Sampled by the monitor.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);
/// Number of workers currently inside one of the five FILTER stages. If the
/// post-tile chain really is one task at a time, this never exceeds 1 — and
/// `FILT_CONC[k]` for `k >= 2` stays at zero.
static FILT_ACTIVE: AtomicUsize = AtomicUsize::new(0);
static FILT_CONC: [AtomicU64; MAX_WORKERS + 1] = [ZERO; MAX_WORKERS + 1];
/// Workers currently inside a TILE stage (entropy or reconstruction).
///
/// Added for the t4->t8 scaling question: the concurrency histogram alone
/// cannot say WHERE the low-occupancy samples sit. `TAIL_CONC` is the same
/// histogram restricted to the samples where no worker is in a tile stage but
/// at least one is in a filter stage -- i.e. the post-tile chain running on
/// its own. If the scaling loss is a serial filter TAIL, that is where it is;
/// if it is mid-frame dependency jitter, `TAIL_CONC` stays near empty while
/// `CONC` still shows low buckets.
static TILE_ACTIVE: AtomicUsize = AtomicUsize::new(0);
static TAIL_CONC: [AtomicU64; MAX_WORKERS + 1] = [ZERO; MAX_WORKERS + 1];
static TAIL_SAMPLES: AtomicU64 = AtomicU64::new(0);
/// `check_tile` deferral causes: 0 = this tile's own sbrow progress, 1 = the
/// second-pass progress gate, 2 = THE DEBLOCK BARRIER (rav1d-safe-only),
/// 3 = reference-frame progress. Index 4 counts admissions.
pub const N_DEFER: usize = 5;
pub const DEFER_NAMES: [&str; N_DEFER] = [
    "own_progress",
    "pass2_progress",
    "deblock_barrier",
    "ref_progress",
    "admitted",
];
static DEFER: [AtomicU64; N_DEFER] = [ZERO; N_DEFER];

#[inline]
pub fn defer(kind: usize) {
    DEFER[kind].fetch_add(1, Ordering::Relaxed);
}
/// Time-weighted concurrency histogram: `CONC[k]` counts samples that saw
/// exactly `k` workers inside a stage body.
static CONC: [AtomicU64; MAX_WORKERS + 1] = [ZERO; MAX_WORKERS + 1];
static SAMPLES: AtomicU64 = AtomicU64::new(0);

static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static SLOT: usize = NEXT_SLOT.fetch_add(1, Ordering::Relaxed).min(MAX_WORKERS - 1);
}

#[inline]
pub fn slot() -> usize {
    SLOT.with(|s| *s)
}

/// Enter a stage body. Returns the start instant for [`stage_end`].
#[inline]
const fn is_filter(stage: usize) -> bool {
    stage >= 2 && stage <= 6
}

#[inline]
pub fn stage_begin_of(stage: usize) -> Instant {
    ACTIVE.fetch_add(1, Ordering::Relaxed);
    if is_filter(stage) {
        FILT_ACTIVE.fetch_add(1, Ordering::Relaxed);
    } else if stage < 2 {
        TILE_ACTIVE.fetch_add(1, Ordering::Relaxed);
    }
    Instant::now()
}

#[inline]
pub fn stage_end(t0: Instant, stage: usize) {
    let now = Instant::now();
    let ns = now.duration_since(t0).as_nanos() as u64;
    ACTIVE.fetch_sub(1, Ordering::Relaxed);
    if is_filter(stage) {
        FILT_ACTIVE.fetch_sub(1, Ordering::Relaxed);
    } else if stage < 2 {
        TILE_ACTIVE.fetch_sub(1, Ordering::Relaxed);
    }
    let w = slot();
    BUSY_NS[ix(w, stage)].fetch_add(ns, Ordering::Relaxed);
    BUSY_CNT[ix(w, stage)].fetch_add(1, Ordering::Relaxed);
    log_event(t0, ns, w, stage);
}

#[inline]
pub fn park_begin() -> Instant {
    Instant::now()
}

#[inline]
pub fn park_end(t0: Instant) {
    let now = Instant::now();
    let ns = now.duration_since(t0).as_nanos() as u64;
    let w = slot();
    PARK_NS[w].fetch_add(ns, Ordering::Relaxed);
    PARK_CNT[w].fetch_add(1, Ordering::Relaxed);
    log_event(t0, ns, w, STAGE_PARK);
}

// ---------------------------------------------------------------------------
// Exact interval log.
//
// The sampling monitor above answers "what is the mean concurrency"; it cannot
// answer "what is running during the last 20% of THIS frame", because a
// 50 us sampler has no frame boundaries in it and a mean hides a bimodal
// distribution. This log records the exact [start, start+dur) interval of every
// stage body, every park, and every frame, so occupancy-over-time, the tail
// composition and the critical path are all derived offline from one run
// instead of estimated from three histograms.
//
// Cost: the same two clock reads the counters already took (`stage_end` now
// reuses one `Instant::now()` for both the duration and the log), plus one
// relaxed fetch_add and two relaxed stores. ~170 stage + ~250 park events per
// 4K frame against a ~65 ms frame.
// ---------------------------------------------------------------------------

/// Synthetic stage ids that are not task stages.
pub const STAGE_PARK: usize = 8;
pub const STAGE_FRAME: usize = 9;
pub const N_EVSTAGE: usize = 10;
pub const EVSTAGE_NAMES: [&str; N_EVSTAGE] = [
    "tile_entropy",
    "tile_recon",
    "deblock_cols",
    "deblock_rows",
    "cdef",
    "superres",
    "loop_restore",
    "other",
    "park",
    "frame",
];

/// 1 M events is ~40 frames of a 4K decode at t=8 including parks. Overflow is
/// counted and reported rather than silently wrapping, so a truncated log can
/// never be mistaken for a complete one.
const EV_CAP: usize = 1 << 20;
static EV_T0: [AtomicU64; EV_CAP] = [ZERO; EV_CAP];
static EV_PACK: [AtomicU64; EV_CAP] = [ZERO; EV_CAP];
static EV_N: AtomicUsize = AtomicUsize::new(0);
static EV_LOST: AtomicU64 = AtomicU64::new(0);

/// Time origin for the log. Set by [`reset`]; every `t0` is ns since then.
static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

#[inline]
fn origin() -> Instant {
    *ORIGIN.get_or_init(Instant::now)
}

#[inline]
fn log_event(t0: Instant, dur_ns: u64, worker: usize, stage: usize) {
    let i = EV_N.fetch_add(1, Ordering::Relaxed);
    if i >= EV_CAP {
        EV_LOST.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let rel = t0.saturating_duration_since(origin()).as_nanos() as u64;
    EV_T0[i].store(rel, Ordering::Relaxed);
    EV_PACK[i].store(
        (dur_ns.min(u32::MAX as u64) << 32) | ((worker as u64 & 0xffff) << 16) | stage as u64,
        Ordering::Relaxed,
    );
}

/// Mark the start of one driver-level decode call. Pair with [`frame_end`].
pub fn frame_begin() -> Instant {
    Instant::now()
}

pub fn frame_end(t0: Instant) {
    let ns = Instant::now().duration_since(t0).as_nanos() as u64;
    log_event(t0, ns, MAX_WORKERS - 1, STAGE_FRAME);
}

/// Write the log as a TSV: `t0_ns  dur_ns  worker  stage`.
pub fn dump_events(path: &str) {
    use std::io::Write as _;
    let n = EV_N.load(Ordering::Relaxed).min(EV_CAP);
    let lost = EV_LOST.load(Ordering::Relaxed);
    let f = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("PROBE evlog_error {path} {e}");
            return;
        }
    };
    let mut w = std::io::BufWriter::new(f);
    let _ = writeln!(w, "t0_ns\tdur_ns\tworker\tstage");
    for i in 0..n {
        let t0 = EV_T0[i].load(Ordering::Relaxed);
        let p = EV_PACK[i].load(Ordering::Relaxed);
        let _ = writeln!(
            w,
            "{}\t{}\t{}\t{}",
            t0,
            p >> 32,
            (p >> 16) & 0xffff,
            EVSTAGE_NAMES[(p & 0xffff) as usize % N_EVSTAGE]
        );
    }
    let _ = w.flush();
    println!("PROBE evlog {path} events {n} lost {lost}");
}

const SAMPLE_US: u64 = 50;

/// Start the concurrency sampler. Idempotent-ish; call once from the driver.
pub fn start_monitor() {
    std::thread::spawn(|| {
        loop {
            let a = ACTIVE.load(Ordering::Relaxed).min(MAX_WORKERS);
            CONC[a].fetch_add(1, Ordering::Relaxed);
            let fa = FILT_ACTIVE.load(Ordering::Relaxed).min(MAX_WORKERS);
            FILT_CONC[fa].fetch_add(1, Ordering::Relaxed);
            if TILE_ACTIVE.load(Ordering::Relaxed) == 0 && fa > 0 {
                TAIL_CONC[a].fetch_add(1, Ordering::Relaxed);
                TAIL_SAMPLES.fetch_add(1, Ordering::Relaxed);
            }
            SAMPLES.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_micros(SAMPLE_US));
        }
    });
}

/// Zero every counter. Call after warmup, before the timed reps, so the
/// warmup decode and the thread-pool spin-up do not enter the numbers.
pub fn reset() {
    // Latch the log's time origin here, so `t0` is ns since the start of the
    // timed reps rather than since an arbitrary earlier point.
    let _ = origin();
    EV_N.store(0, Ordering::Relaxed);
    EV_LOST.store(0, Ordering::Relaxed);
    for w in 0..MAX_WORKERS {
        for s in 0..N_STAGE {
            BUSY_NS[ix(w, s)].store(0, Ordering::Relaxed);
            BUSY_CNT[ix(w, s)].store(0, Ordering::Relaxed);
        }
        PARK_NS[w].store(0, Ordering::Relaxed);
        PARK_CNT[w].store(0, Ordering::Relaxed);
    }
    for k in 0..=MAX_WORKERS {
        CONC[k].store(0, Ordering::Relaxed);
        FILT_CONC[k].store(0, Ordering::Relaxed);
        TAIL_CONC[k].store(0, Ordering::Relaxed);
    }
    for k in 0..N_DEFER {
        DEFER[k].store(0, Ordering::Relaxed);
    }
    SAMPLES.store(0, Ordering::Relaxed);
    TAIL_SAMPLES.store(0, Ordering::Relaxed);
}

/// Dump every counter as `PROBE <key> <value>` lines on stdout.
pub fn report(frames: u64) {
    let f = frames.max(1) as f64;
    let mut per_stage_total = [0u64; N_STAGE];
    let mut per_worker_total = [0u64; MAX_WORKERS];
    let mut used = 0usize;

    for w in 0..MAX_WORKERS {
        let mut wt = 0u64;
        for s in 0..N_STAGE {
            let ns = BUSY_NS[ix(w, s)].load(Ordering::Relaxed);
            per_stage_total[s] += ns;
            wt += ns;
        }
        per_worker_total[w] = wt;
        if wt > 0 || PARK_NS[w].load(Ordering::Relaxed) > 0 {
            used = used.max(w + 1);
        }
    }

    println!("PROBE frames {frames}");
    for s in 0..N_STAGE {
        let cnt: u64 = (0..MAX_WORKERS)
            .map(|w| BUSY_CNT[ix(w, s)].load(Ordering::Relaxed))
            .sum();
        println!(
            "PROBE stage_ms_per_frame {} {:.3} count_per_frame {:.2}",
            STAGE_NAMES[s],
            per_stage_total[s] as f64 / 1e6 / f,
            cnt as f64 / f
        );
    }

    let filter_ns: u64 = FILTER_STAGES.iter().map(|&s| per_stage_total[s]).sum();
    let tile_ns: u64 = per_stage_total[0] + per_stage_total[1];
    println!(
        "PROBE filter_chain_ms_per_frame {:.3}",
        filter_ns as f64 / 1e6 / f
    );
    println!("PROBE tile_ms_per_frame {:.3}", tile_ns as f64 / 1e6 / f);

    for w in 0..used {
        println!(
            "PROBE worker {w} busy_ms_per_frame {:.3} park_ms_per_frame {:.3} park_count_per_frame {:.2}",
            per_worker_total[w] as f64 / 1e6 / f,
            PARK_NS[w].load(Ordering::Relaxed) as f64 / 1e6 / f,
            PARK_CNT[w].load(Ordering::Relaxed) as f64 / f
        );
    }

    let samples = SAMPLES.load(Ordering::Relaxed).max(1);
    let mut weighted = 0f64;
    for k in 0..=MAX_WORKERS {
        let c = CONC[k].load(Ordering::Relaxed);
        if c > 0 {
            println!(
                "PROBE conc {k} samples {c} frac {:.4}",
                c as f64 / samples as f64
            );
            weighted += (k * c as usize) as f64;
        }
    }
    println!("PROBE mean_active {:.3}", weighted / samples as f64);
    for k in 0..=MAX_WORKERS {
        let c = FILT_CONC[k].load(Ordering::Relaxed);
        if c > 0 {
            println!(
                "PROBE filtconc {k} samples {c} frac {:.4}",
                c as f64 / samples as f64
            );
        }
    }
    let tail = TAIL_SAMPLES.load(Ordering::Relaxed);
    let mut tail_weighted = 0f64;
    for k in 0..=MAX_WORKERS {
        let c = TAIL_CONC[k].load(Ordering::Relaxed);
        if c > 0 {
            println!(
                "PROBE tailconc {k} samples {c} frac_of_all {:.4}",
                c as f64 / samples as f64
            );
            tail_weighted += (k * c as usize) as f64;
        }
    }
    println!(
        "PROBE tail_frac_of_wall {:.4} tail_mean_active {:.3}",
        tail as f64 / samples as f64,
        if tail > 0 {
            tail_weighted / tail as f64
        } else {
            0.0
        }
    );
    for k in 0..N_DEFER {
        println!(
            "PROBE defer {} per_frame {:.2}",
            DEFER_NAMES[k],
            DEFER[k].load(Ordering::Relaxed) as f64 / f
        );
    }
    // Mean active over the samples where ANY worker was busy — i.e. achieved
    // parallelism while the decoder is actually decoding, excluding the gaps
    // between reps and the driver's own setup.
    let busy_samples: u64 = (1..=MAX_WORKERS)
        .map(|k| CONC[k].load(Ordering::Relaxed))
        .sum();
    if busy_samples > 0 {
        println!(
            "PROBE mean_active_when_busy {:.3}",
            weighted / busy_samples as f64
        );
    }
}
