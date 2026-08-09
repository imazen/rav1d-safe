//! Price Variant 1 of the tile-keyed borrow design: **separate owned buffer per
//! tile, stitched into the picture**.
//!
//! Variant 1's premise is that a tile task writes reconstruction into a private
//! `w_tile x h_tile` buffer it owns outright (plain `&mut`, no tracker, no
//! `unsafe`) and the picture is touched only by a stitch. The stitch is claimed
//! to be "one frame copy — ~12 MB at 4K 8bpc, sub-millisecond". This example
//! MEASURES that instead of assuming it, because the copy is *strided* on the
//! destination side and that is not the same thing as a 12 MB `memcpy`.
//!
//! What is measured, for the real `v4k_8tile` geometry (3840x2160, 4:2:0, so
//! three planes and 12.4 MB per frame):
//!
//! * `stitch_whole` — one pass over every tile at the end of the frame.
//! * `stitch_sbrow` — the shape a real decoder needs: the stitch for superblock
//!   row `N` must complete before the filter chain consumes row `N`, so it
//!   happens 34 times per frame in 64-pixel bands, not once.
//! * `memcpy_flat` — a contiguous 12.4 MB copy, as the lower bound the "one
//!   frame copy" framing appeals to.
//! * `recon_write` — writing the same bytes into a private tile buffer versus
//!   into the shared picture plane, which is the *other* half of Variant 1's
//!   claim ("recon may get FASTER from cache locality"). Both arms write the
//!   same number of bytes in the same 4x4-transform-block order.
//!
//! Every arm is interleaved with a rotating start position and the median of
//! `reps` is reported with min/max, per `docs/AGENT_BRIEF.md` section 2.
//!
//! Run WITHOUT `nice` (Darwin maps it to E-cores).
//!
//! ```text
//! cargo run --release --example tile_stitch_cost -- [reps] [threads]
//! ```

use std::hint::black_box;
use std::time::Instant;

/// `v4k_8tile.avif`: 3840x2160 4:2:0 8bpc, 4 tile columns x 2 tile rows.
const W: usize = 3840;
const H: usize = 2160;
const TILE_COLS: usize = 4;
const TILE_ROWS: usize = 2;
const SB: usize = 64;

/// One plane's geometry.
#[derive(Clone, Copy)]
struct Plane {
    w: usize,
    h: usize,
    stride: usize,
}

fn planes() -> [Plane; 3] {
    // dav1d aligns the stride up; 4K luma lands on 3840 exactly, chroma on 1920.
    [
        Plane {
            w: W,
            h: H,
            stride: W,
        },
        Plane {
            w: W / 2,
            h: H / 2,
            stride: W / 2,
        },
        Plane {
            w: W / 2,
            h: H / 2,
            stride: W / 2,
        },
    ]
}

/// Tile column boundaries in pixels for a plane, snapped to superblocks.
fn tile_x_bounds(p: &Plane, ss_hor: usize) -> Vec<usize> {
    let sb = SB >> ss_hor;
    let sbw = p.w.div_ceil(sb);
    (0..=TILE_COLS)
        .map(|i| (sbw * i / TILE_COLS * sb).min(p.w))
        .collect()
}

fn tile_y_bounds(p: &Plane, ss_ver: usize) -> Vec<usize> {
    let sb = SB >> ss_ver;
    let sbh = p.h.div_ceil(sb);
    (0..=TILE_ROWS)
        .map(|i| (sbh * i / TILE_ROWS * sb).min(p.h))
        .collect()
}

struct Frame {
    planes: [Plane; 3],
    /// The shared picture: one `Vec` per plane.
    pic: Vec<Vec<u8>>,
    /// Per-tile private buffers: `[plane][tile]`, each `tw x th` COMPACT
    /// (its own tight stride), which is the point of Variant 1 — the tile
    /// buffer is smaller and contiguous.
    tiles: Vec<Vec<Vec<u8>>>,
    /// `[plane] -> (x0, x1, y0, y1)` per tile.
    geom: Vec<Vec<(usize, usize, usize, usize)>>,
}

impl Frame {
    fn new() -> Self {
        let planes = planes();
        let pic = planes
            .iter()
            .map(|p| vec![0u8; p.stride * p.h])
            .collect::<Vec<_>>();
        let mut tiles = Vec::new();
        let mut geom = Vec::new();
        for (pl, p) in planes.iter().enumerate() {
            let ss = if pl == 0 { 0 } else { 1 };
            let xb = tile_x_bounds(p, ss);
            let yb = tile_y_bounds(p, ss);
            let mut tb = Vec::new();
            let mut tg = Vec::new();
            for ty in 0..TILE_ROWS {
                for tx in 0..TILE_COLS {
                    let (x0, x1, y0, y1) = (xb[tx], xb[tx + 1], yb[ty], yb[ty + 1]);
                    tb.push(vec![0u8; (x1 - x0) * (y1 - y0)]);
                    tg.push((x0, x1, y0, y1));
                }
            }
            tiles.push(tb);
            geom.push(tg);
        }
        Self {
            planes,
            pic,
            tiles,
            geom,
        }
    }

    fn frame_bytes(&self) -> usize {
        self.planes.iter().map(|p| p.w * p.h).sum()
    }

    /// One stitch of every tile into the picture, whole-frame.
    fn stitch_whole(&mut self) {
        for pl in 0..3 {
            let stride = self.planes[pl].stride;
            for (t, &(x0, x1, y0, y1)) in self.geom[pl].iter().enumerate() {
                let tw = x1 - x0;
                let src = &self.tiles[pl][t];
                let dst = &mut self.pic[pl];
                for y in y0..y1 {
                    let s = (y - y0) * tw;
                    let d = y * stride + x0;
                    dst[d..d + tw].copy_from_slice(&src[s..s + tw]);
                }
            }
        }
    }

    /// The shape a real decoder needs: stitch superblock row `sby` only.
    ///
    /// The filter chain for sbrow N reads reconstruction for row N, so the
    /// stitch cannot be deferred to the end of the frame — it happens once per
    /// sbrow per tile column, in 64-pixel bands.
    fn stitch_sbrow(&mut self, sby: usize) {
        for pl in 0..3 {
            let ss = if pl == 0 { 0 } else { 1 };
            let sb = SB >> ss;
            let stride = self.planes[pl].stride;
            let ph = self.planes[pl].h;
            let by0 = sby * sb;
            if by0 >= ph {
                continue;
            }
            let by1 = (by0 + sb).min(ph);
            for (t, &(x0, x1, y0, y1)) in self.geom[pl].iter().enumerate() {
                // Only the tile row containing this sbrow participates.
                if by0 >= y1 || by1 <= y0 {
                    continue;
                }
                let tw = x1 - x0;
                let src = &self.tiles[pl][t];
                let dst = &mut self.pic[pl];
                for y in by0.max(y0)..by1.min(y1) {
                    let s = (y - y0) * tw;
                    let d = y * stride + x0;
                    dst[d..d + tw].copy_from_slice(&src[s..s + tw]);
                }
            }
        }
    }

    fn n_sbrows(&self) -> usize {
        self.planes[0].h.div_ceil(SB)
    }
}

/// Write `bytes` worth of 4x4 transform blocks into `buf` at `stride`,
/// covering the `w x h` region — the access pattern reconstruction has.
#[inline(never)]
fn write_blocks(buf: &mut [u8], stride: usize, w: usize, h: usize, seed: u8) {
    let mut v = seed;
    let mut by = 0;
    while by < h {
        let mut bx = 0;
        while bx < w {
            for y in 0..4.min(h - by) {
                let row = (by + y) * stride + bx;
                let n = 4.min(w - bx);
                for x in 0..n {
                    buf[row + x] = v;
                }
            }
            v = v.wrapping_add(1);
            bx += 4;
        }
        by += 4;
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn report(name: &str, mut xs: Vec<f64>, note: &str) {
    let lo = xs.iter().cloned().fold(f64::MAX, f64::min);
    let hi = xs.iter().cloned().fold(f64::MIN, f64::max);
    let m = median(&mut xs);
    println!(
        "{name:<16} {m:8.3} ms  [{lo:7.3}..{hi:7.3}]  n={:<3} {note}",
        xs.len()
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(9);
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut f = Frame::new();
    let fb = f.frame_bytes();
    let nsb = f.n_sbrows();
    println!(
        "geometry: {W}x{H} 4:2:0, {TILE_COLS}x{TILE_ROWS} tiles, {} sbrows, {:.2} MB/frame",
        nsb,
        fb as f64 / 1e6
    );
    println!("threads={threads} reps={reps}  (this arm set is single-threaded unless noted)");
    println!();

    // Fill tile buffers with something so the copies are not from zero pages.
    for pl in 0..3 {
        for (t, b) in f.tiles[pl].iter_mut().enumerate() {
            for (i, x) in b.iter_mut().enumerate() {
                *x = (i as u8).wrapping_add(t as u8);
            }
        }
    }

    let mut flat_src = vec![7u8; fb];
    let mut flat_dst = vec![0u8; fb];

    let mut a_whole = Vec::new();
    let mut a_sbrow = Vec::new();
    let mut a_flat = Vec::new();

    for round in 0..reps {
        for slot in 0..3 {
            match (round + slot) % 3 {
                0 => {
                    let t = Instant::now();
                    f.stitch_whole();
                    a_whole.push(t.elapsed().as_secs_f64() * 1e3);
                    black_box(f.pic[0][0]);
                }
                1 => {
                    let t = Instant::now();
                    for sby in 0..nsb {
                        f.stitch_sbrow(sby);
                    }
                    a_sbrow.push(t.elapsed().as_secs_f64() * 1e3);
                    black_box(f.pic[0][0]);
                }
                _ => {
                    let t = Instant::now();
                    flat_dst.copy_from_slice(black_box(&flat_src));
                    a_flat.push(t.elapsed().as_secs_f64() * 1e3);
                    black_box(flat_dst[0]);
                    black_box(&mut flat_src);
                }
            }
        }
    }

    println!("--- Variant 1 stitch, per frame ---");
    report("memcpy_flat", a_flat, "contiguous 12.4 MB, the lower bound");
    report("stitch_whole", a_whole, "all tiles, once at end of frame");
    report(
        "stitch_sbrow",
        a_sbrow,
        "34 sbrow bands x tiles, the shape a decoder needs",
    );

    // --- recon write locality: private compact tile buffer vs shared plane ---
    let mut a_tile = Vec::new();
    let mut a_pic = Vec::new();
    for round in 0..reps {
        for slot in 0..2 {
            if (round + slot) % 2 == 0 {
                let t = Instant::now();
                for pl in 0..3 {
                    for (ti, &(x0, x1, y0, y1)) in f.geom[pl].iter().enumerate() {
                        let (tw, th) = (x1 - x0, y1 - y0);
                        write_blocks(&mut f.tiles[pl][ti], tw, tw, th, ti as u8);
                    }
                }
                a_tile.push(t.elapsed().as_secs_f64() * 1e3);
                black_box(f.tiles[0][0][0]);
            } else {
                let t = Instant::now();
                for pl in 0..3 {
                    let stride = f.planes[pl].stride;
                    for (ti, &(x0, x1, y0, y1)) in f.geom[pl].iter().enumerate() {
                        let (tw, th) = (x1 - x0, y1 - y0);
                        let base = y0 * stride + x0;
                        write_blocks(&mut f.pic[pl][base..], stride, tw, th, ti as u8);
                    }
                }
                a_pic.push(t.elapsed().as_secs_f64() * 1e3);
                black_box(f.pic[0][0]);
            }
        }
    }
    println!();
    println!("--- recon write target, per frame (same bytes, same 4x4 order) ---");
    report("write_into_pic", a_pic, "shared plane, picture stride");
    report(
        "write_into_tile",
        a_tile,
        "private compact tile buffer, tile stride",
    );
    println!();
    println!(
        "NOTE: the stitch is pure extra work — Variant 1 pays write_into_tile + stitch_sbrow\n\
         where the current design pays write_into_pic alone."
    );
    let _ = threads;
}
