//! Is the rectangle overlap predicate EXACT, against a brute-force oracle?
//!
//! The tracker's exactness argument is what makes the crate's 65 `unsafe` uses
//! sound, and the rectangle primitive rests it on two cheap comparisons (hull
//! overlap AND column overlap) instead of any row arithmetic. The argument is
//! written out on `ShardRecs::c0`; this checks it by experiment.
//!
//! The oracle is a literal byte set: paint every byte shape A covers into a
//! plane, then ask whether any byte shape B covers is already painted. No
//! geometry reasoning is transcribed from the implementation — if the two ever
//! disagree, one of them is wrong and the test says which direction:
//!
//! * oracle says overlap, tracker says no  → **MISSED OVERLAP**, i.e. two
//!   aliasing `&mut` handed out. Unsound.
//! * oracle says disjoint, tracker says yes → **FALSE POSITIVE**, i.e. the
//!   spurious `overlapping DisjointMut` panic the per-row split existed to
//!   avoid (PR #467 measured 8-9 of them in 24 concurrent runs).
//!
//! Iterations default low so this stays a fast gate; set `RECT_ORACLE_ITERS`
//! for a soak.

use rav1d_disjoint_mut::{DisjointMut, StridedRows, set_parallelism};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// `SHARD_MIN_LEN`: below this the tracker does not shard, and the row map is
/// not consulted at all.
const MIN_LEN: usize = 64 * 1024;

/// SplitMix64. Deterministic and seeded, so a failure is reproducible from the
/// seed printed in the panic message.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    /// Uniform in `lo..=hi`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Rect { start: usize, w: usize, h: usize },
    Interval { start: usize, end: usize },
}

impl Shape {
    /// THE ORACLE. Every byte this shape covers, enumerated one at a time.
    ///
    /// Deliberately the dumbest possible definition — a rectangle is its rows,
    /// an interval is its range. Nothing here knows about hulls, columns,
    /// bands, groups or shards.
    fn for_each_byte(&self, stride: usize, mut f: impl FnMut(usize)) {
        match *self {
            Shape::Rect { start, w, h } => {
                for row in 0..h {
                    for col in 0..w {
                        f(start + row * stride + col);
                    }
                }
            }
            Shape::Interval { start, end } => {
                for b in start..end {
                    f(b);
                }
            }
        }
    }

    /// Does this shape span more than one row?
    ///
    /// A rectangle never does *in the column sense* — its columns are the same
    /// in every row. An INTERVAL that runs off the end of its row does, and
    /// that is the one shape the row map cannot describe: it registers
    /// `0..=COL_ANY` and falls back to the plain hull test, which is
    /// conservative. This classifier is what separates "known degradation"
    /// from "new bug".
    fn crosses_row(&self, stride: usize) -> bool {
        match *self {
            Shape::Rect { .. } => false,
            Shape::Interval { start, end } => start / stride != (end - 1) / stride,
        }
    }

    /// Highest byte touched, +1 — for the in-bounds check.
    fn limit(&self, stride: usize) -> usize {
        match *self {
            Shape::Rect { start, w, h } => start + (h - 1) * stride + w,
            Shape::Interval { end, .. } => end,
        }
    }
}

/// Random shape. Biased on purpose towards the geometries the decoder makes:
/// small aligned blocks, one-pixel edge columns, full rows, and the
/// deliberately-unaligned reads (`ipred`'s x-1, CDEF's x-2, MC's x-3).
fn gen_shape(rng: &mut Rng, stride: usize, rows: usize) -> Shape {
    let kind = rng.below(10);
    if kind < 7 {
        // A rectangle.
        let w = match rng.below(8) {
            0 => 1,
            1 => rng.range(1, 4),
            2 => stride.min(rng.range(1, 64)),
            3 => stride,                           // a full row
            _ => stride.min(1 << rng.range(2, 6)), // 4..32, the aligned sizes
        };
        let h = match rng.below(4) {
            0 => 1,
            1 => rng.range(1, 4),
            _ => rng.range(1, 32.min(rows)),
        };
        let h = h.min(rows);
        let col = rng.below(stride - w + 1);
        let row = rng.below(rows - h + 1);
        Shape::Rect {
            start: row * stride + col,
            w,
            h,
        }
    } else {
        // A plain interval, often crossing a row boundary.
        let len = match rng.below(4) {
            0 => rng.range(1, 8),
            1 => rng.range(1, 64),
            _ => rng.range(1, 2 * stride),
        };
        let max_start = stride * rows - len;
        Shape::Interval {
            start: rng.below(max_start + 1),
            end: 0,
        }
        .with_len(len)
    }
}

impl Shape {
    fn with_len(self, len: usize) -> Self {
        match self {
            Shape::Interval { start, .. } => Shape::Interval {
                start,
                end: start + len,
            },
            other => other,
        }
    }
}

/// Take the borrow the shape describes, mutably. Returns `true` if the tracker
/// REFUSED it (panicked with an overlap).
fn conflicts(dm: &DisjointMut<Vec<u8>>, shape: Shape, stride: usize) -> bool {
    catch_unwind(AssertUnwindSafe(|| match shape {
        Shape::Rect { start, w, h } => {
            let g = dm.index_rect_mut(StridedRows {
                start,
                w,
                h,
                stride,
            });
            drop(g);
        }
        Shape::Interval { start, end } => {
            let g = dm.index_mut(start..end);
            drop(g);
        }
    }))
    .is_err()
}

/// Hold the borrow open, so the second one sees it.
fn hold<'a>(
    dm: &'a DisjointMut<Vec<u8>>,
    shape: Shape,
    stride: usize,
) -> rav1d_disjoint_mut::DisjointMutGuard<'a, Vec<u8>, [u8]> {
    match shape {
        Shape::Rect { start, w, h } => dm.index_rect_mut(StridedRows {
            start,
            w,
            h,
            stride,
        }),
        Shape::Interval { start, end } => dm.index_mut(start..end),
    }
}

fn iters() -> usize {
    std::env::var("RECT_ORACLE_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500)
}

/// The property, over one plane shape.
fn run_stride(
    stride: usize,
    rows: usize,
    seed: u64,
    n: usize,
) -> Result<(usize, usize, usize), String> {
    let len = stride * rows;
    assert!(
        len >= MIN_LEN,
        "stride {stride} x {rows} rows is not sharded"
    );

    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; len]);
    dm.set_row_stride(stride);

    let mut painted = vec![false; len];
    let mut rng = Rng(seed);
    let (mut n_overlap, mut n_disjoint) = (0usize, 0usize);
    let mut n_false_pos = 0usize;

    for i in 0..n {
        let a = gen_shape(&mut rng, stride, rows);
        let b = gen_shape(&mut rng, stride, rows);
        if a.limit(stride) > len || b.limit(stride) > len {
            continue;
        }

        // ORACLE: paint A, then ask whether B lands on any painted byte.
        a.for_each_byte(stride, |o| painted[o] = true);
        let mut truth = false;
        b.for_each_byte(stride, |o| truth |= painted[o]);

        // TRACKER: hold A, try B.
        let held = hold(&dm, a, stride);
        let observed = conflicts(&dm, b, stride);
        drop(held);

        // Clean up the oracle plane for the next round.
        a.for_each_byte(stride, |o| painted[o] = false);

        if truth && !observed {
            // SOUNDNESS. No tolerance, no classification, no excuse: the
            // tracker handed out two aliasing `&mut`.
            return Err(format!(
                "MISSED OVERLAP (UNSOUND)\n  \
                 stride={stride} rows={rows} seed={seed} iter={i}\n  \
                 A = {a:?}\n  B = {b:?}"
            ));
        }
        if !truth && observed {
            // A spurious panic. Tolerated ONLY where the design documents it:
            // an interval that runs off the end of its row cannot be described
            // as a rectangle, registers `0..=COL_ANY`, and degrades to the
            // plain hull test. Between two shapes the map CAN describe, the
            // predicate is supposed to be exact — so that case is a new bug.
            n_false_pos += 1;
            if !(a.crosses_row(stride) || b.crosses_row(stride)) {
                return Err(format!(
                    "FALSE POSITIVE between two describable shapes\n  \
                     stride={stride} rows={rows} seed={seed} iter={i}\n  \
                     A = {a:?}\n  B = {b:?}\n  \
                     neither crosses a row, so both have real column ranges \
                     and the pair should have been exact"
                ));
            }
        }

        if truth {
            n_overlap += 1;
        } else {
            n_disjoint += 1;
        }

        // Hygiene: nothing may be left registered. A partially-published
        // borrow surviving a caught panic would show up here and nowhere else.
        drop(dm.index_mut(0..len));
    }
    Ok((n_overlap, n_disjoint, n_false_pos))
}

/// Silence the panic hook for the duration: the tracker panics thousands of
/// times on purpose here, and each one prints a multi-line message.
fn quietly<R>(f: impl FnOnce() -> R) -> R {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(prev);
    match r {
        Ok(v) => v,
        Err(e) => std::panic::resume_unwind(e),
    }
}

#[test]
fn rect_versus_rect_and_rect_versus_interval_match_a_byte_set_oracle() {
    // Plane shapes: the row-map floor (16), non-powers-of-two, a real 4K luma
    // row, and its 10-bit twin.
    let planes: &[(usize, usize)] = &[
        (16, 4096),
        (64, 1024),
        (100, 800),
        (128, 512),
        (256, 256),
        (257, 256),
        (3840, 32),
        (4096, 32),
    ];
    let n = iters();
    let mut tot_overlap = 0usize;
    let mut tot_disjoint = 0usize;
    let mut tot_false_pos = 0usize;
    let mut failures: Vec<String> = Vec::new();
    quietly(|| {
        for (i, &(stride, rows)) in planes.iter().enumerate() {
            match run_stride(stride, rows, 0xC0FFEE ^ (i as u64), n) {
                Ok((o, d, fp)) => {
                    tot_overlap += o;
                    tot_disjoint += d;
                    tot_false_pos += fp;
                }
                Err(e) => failures.push(e),
            }
        }
    });
    if !failures.is_empty() {
        panic!(
            "{} plane shape(s) disagreed with the oracle:\n\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }

    // LIVENESS: a run that never produced both answers proves nothing. The
    // false-positive direction needs disjoint pairs and the missed-overlap
    // direction needs overlapping ones.
    assert!(
        tot_overlap > 100 && tot_disjoint > 100,
        "degenerate generator: {tot_overlap} overlapping / {tot_disjoint} disjoint pairs"
    );
    eprintln!(
        "rect oracle: {tot_overlap} overlapping / {tot_disjoint} disjoint pairs; \
         0 missed overlaps; {tot_false_pos} conservative false positives \
         ({:.2}% of disjoint pairs), all involving a row-crossing interval",
        100.0 * tot_false_pos as f64 / tot_disjoint.max(1) as f64
    );
}

/// Two IMMUTABLE borrows must never conflict, whatever their geometry.
#[test]
fn two_immutable_rectangles_never_conflict() {
    const STRIDE: usize = 256;
    const ROWS: usize = 256;
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; STRIDE * ROWS]);
    dm.set_row_stride(STRIDE);
    let mut rng = Rng(0xBEEF);
    for _ in 0..500 {
        let a = gen_shape(&mut rng, STRIDE, ROWS);
        let b = gen_shape(&mut rng, STRIDE, ROWS);
        if a.limit(STRIDE) > STRIDE * ROWS || b.limit(STRIDE) > STRIDE * ROWS {
            continue;
        }
        // Immutable pair: take both at once, no panic is permitted.
        let _ia = match a {
            Shape::Rect { start, w, h } => Some(dm.index_rect(StridedRows {
                start,
                w,
                h,
                stride: STRIDE,
            })),
            Shape::Interval { .. } => None,
        };
        let _ib = match b {
            Shape::Rect { start, w, h } => Some(dm.index_rect(StridedRows {
                start,
                w,
                h,
                stride: STRIDE,
            })),
            Shape::Interval { .. } => None,
        };
    }
}
