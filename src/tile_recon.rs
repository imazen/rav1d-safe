//! Owned per-tile reconstruction buffers ("Variant 1" of issue #455's
//! tile-keyed borrow design).
//!
//! # Why
//!
//! The borrow tracker's whole cost at `t > 1` exists because two tile workers
//! legitimately write the SAME picture rows at different columns. A strided
//! `w x h` block borrow therefore cannot be taken as one contiguous hull — the
//! hull spans inter-row gaps that belong to the next tile column — so
//! [`block_mut`] splits into `h` per-row borrows plus a compact copy and its
//! write-back. That split is the 7.9 M -> 22.7 M registrations/frame explosion
//! measured in #455, and the per-registration cost also rises 79% from `t = 2`
//! to `t = 8` through shard contention.
//!
//! AV1 guarantees tile regions do not overlap during reconstruction. So give
//! each tile its OWN buffer: the collision that forced the split cannot happen,
//! and reconstruction keeps the single-guard, no-copy path at every thread
//! count. Ownership is static, so nothing has to be proved at run time.
//!
//! # The geometry trick
//!
//! Each tile's buffer has the **same byte length and the same stride** as the
//! picture plane it mirrors. Every frame-coordinate offset reconstruction
//! computes — `4 * (t.b.y * pixel_stride + t.b.x)` at 22 sites, plus ~10 direct
//! reads of `f.cur.stride[..]` — therefore indexes the same pixel in the tile
//! buffer as it would in the picture, unchanged. That is what keeps this a
//! three-line seam in `recon.rs` instead of a rewrite of every offset
//! computation.
//!
//! The cost is memory, and it is real. Allocation is `alloc_zeroed`, so
//! untouched pages are never faulted in — but a tile writes its own columns
//! across every row of its tile ROW, and a page spans whole rows, so the
//! resident set grows by roughly `tile_columns x plane_bytes`. MEASURED on
//! v4k_8tile 8bpc t=8: peak RSS 106.3 -> 202.4 MB (+96.1 MB, 4 tile columns x
//! 24.0 MB of planes); t=1 is untouched at 99.5 -> 99.6 MB because the feature
//! declines below two workers. An earlier draft that filled the buffer with
//! `Vec::resize` instead measured +192.5 MB. See
//! `benchmarks/tile_owned_recon_2026-08-09.meta`.
//!
//! # What is NOT covered
//!
//! * `allow_intrabc` frames — intra block copy reads the *current* picture as a
//!   reference, which would be stale in a tile buffer until the stitch. Those
//!   frames keep the shared picture.
//! * The filter chain (deblock / CDEF / loop restoration / superres). It
//!   genuinely crosses tile boundaries (`src/lf_apply.rs:563`, `:608`) and runs
//!   after the stitch on the unified picture, exactly as before and with
//!   exactly today's tracking.
//!
//! [`block_mut`]: crate::src::with_offset::WithOffset::block_mut

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::headers::Rav1dPixelLayout;
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::src::internal::Rav1dFrameData;
use crate::src::strided::Strided as _;
use std::ffi::c_int;

/// Geometry a [`TileReconBufs`] was built for. A mismatch forces a rebuild;
/// an exact match lets the buffers survive across frames, so the pages are
/// faulted in once for a whole sequence rather than once per frame.
#[derive(PartialEq, Eq, Clone, Copy)]
struct Geometry {
    n_tiles: usize,
    /// Byte length of each plane's buffer.
    byte_len: [usize; 3],
    /// Byte stride of each plane.
    stride: [isize; 3],
}

pub(crate) struct TileReconBufs {
    /// `planes[tile_idx]` — one full-geometry plane set per tile.
    planes: Vec<[Rav1dPictureDataComponent; 3]>,
    geometry: Geometry,
}

impl TileReconBufs {
    #[inline]
    pub(crate) fn planes(&self, tile_idx: usize) -> &[Rav1dPictureDataComponent; 3] {
        &self.planes[tile_idx]
    }
}

/// Is the runtime switch on?
///
/// Deliberately an env var rather than a `const`, so both arms of an A/B are
/// the same binary and an inter-arm delta cannot be a codegen artifact
/// (#455's `probe-*` convention).
#[cfg(feature = "tile-owned-recon")]
fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("RAV1D_TILE_OWNED").as_deref(), Ok("0")))
}

/// Allow private per-tile buffers with a SINGLE worker thread.
///
/// #474 declines below two workers because there is nothing to win there: with
/// one worker the shared picture already takes one hull guard per block and the
/// stitch is pure added cost. But the zero-tracker CEILING arm has to be priced
/// at t=1 too — that is the cell the campaign's bar is closest at, and the
/// ownership argument does not depend on the thread count. Off unless asked.
#[cfg(feature = "tile-owned-recon")]
fn allow_single_thread() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("RAV1D_TILE_OWNED_T1").as_deref(), Ok("1")))
}

/// Allocate (or reuse) one owned plane set per tile, if this frame qualifies.
///
/// Sets `f.tile_recon` to `None` when it does not, which restores the shared
/// picture at every seam with no other branch.
#[cfg(feature = "tile-owned-recon")]
pub(crate) fn setup(c: &crate::src::internal::Rav1dContext, f: &mut Rav1dFrameData) {
    if !enabled() {
        f.tile_recon = None;
        return;
    }
    let disqualified = {
        // Only worth anything when tile workers actually run concurrently.
        let Some(frame_hdr) = f.frame_hdr.as_ref() else {
            return;
        };
        let frame_hdr = &***frame_hdr;
        let n_tiles = frame_hdr.tiling.cols as usize * frame_hdr.tiling.rows as usize;
        // `allow_intrabc` reads the current picture as an MC reference.
        // Frame threading (`n_fc > 1`) splits a tile into entropy and
        // reconstruction passes; out of scope for this round.
        (c.tc.len() < 2 && !allow_single_thread())
            || n_tiles < 2
            || frame_hdr.allow_intrabc
            || c.fc.len() > 1
            || f.cur.data.is_none()
    };
    if disqualified {
        f.tile_recon = None;
        return;
    }

    let n_tiles = {
        let frame_hdr = f.frame_hdr();
        frame_hdr.tiling.cols as usize * frame_hdr.tiling.rows as usize
    };
    let model = &f.cur.data.as_ref().unwrap().data;
    let has_chroma = f.cur.p.layout != Rav1dPixelLayout::I400;
    let n_planes = if has_chroma { 3 } else { 1 };

    // Negative strides are legal in the picture API and would make the
    // row-range arithmetic below wrong; decline rather than get it subtly
    // wrong. Also decline if the tile grid's last pixel does not provably fit
    // in the plane, so the stitch can never clamp silently.
    for pl in 0..n_planes {
        if model[pl].stride() <= 0 || model[pl].byte_len() == 0 {
            f.tile_recon = None;
            return;
        }
    }
    if !last_pixel_fits(f, n_planes) {
        f.tile_recon = None;
        return;
    }

    let geometry = Geometry {
        n_tiles,
        byte_len: [0, 1, 2].map(|pl| model[pl].byte_len()),
        stride: [0, 1, 2].map(|pl| model[pl].stride()),
    };
    if let Some(existing) = &f.tile_recon {
        if existing.geometry == geometry {
            return; // Reuse: pages already resident.
        }
    }

    let mut planes = Vec::new();
    if planes.try_reserve_exact(n_tiles).is_err() {
        f.tile_recon = None;
        return;
    }
    for _ in 0..n_tiles {
        let Some(y) = Rav1dPictureDataComponent::new_private_like(&model[0]) else {
            f.tile_recon = None;
            return;
        };
        let (u, v) = if has_chroma {
            let (Some(u), Some(v)) = (
                Rav1dPictureDataComponent::new_private_like(&model[1]),
                Rav1dPictureDataComponent::new_private_like(&model[2]),
            ) else {
                f.tile_recon = None;
                return;
            };
            (u, v)
        } else {
            // I400: chroma planes are zero-length in the picture too; mirror
            // them so indexing shapes match, but they are never touched.
            let (Some(u), Some(v)) = (
                Rav1dPictureDataComponent::new_private_like(&model[0]),
                Rav1dPictureDataComponent::new_private_like(&model[0]),
            ) else {
                f.tile_recon = None;
                return;
            };
            (u, v)
        };
        planes.push([y, u, v]);
    }
    f.tile_recon = Some(TileReconBufs { planes, geometry });
}

/// Prove the bottom-right pixel of every tile lands inside every plane, so
/// [`stitch_sbrow`] can never index out of range or silently clamp.
#[cfg(feature = "tile-owned-recon")]
fn last_pixel_fits(f: &Rav1dFrameData, n_planes: usize) -> bool {
    let layout = f.cur.p.layout;
    let ss_ver = (layout == Rav1dPixelLayout::I420) as c_int;
    let ss_hor = (layout != Rav1dPixelLayout::I444) as c_int;
    let model = &f.cur.data.as_ref().unwrap().data;
    let pixel_size = if f.cur.p.bpc > 8 { 2usize } else { 1 };
    for pl in 0..n_planes {
        let (sv, sh) = if pl == 0 { (0, 0) } else { (ss_ver, ss_hor) };
        let stride_px = model[pl].stride() as usize / pixel_size;
        let last_row = ((f.bh * 4) >> sv) as usize;
        let last_col = ((f.bw * 4) >> sh) as usize;
        if last_row == 0 || last_col > stride_px {
            return false;
        }
        let need = (last_row - 1) * stride_px + last_col;
        if need > model[pl].byte_len() / pixel_size {
            return false;
        }
    }
    true
}

/// Copy one tile's slice of one superblock row from its private buffer into
/// the shared picture.
///
/// Called at the end of `rav1d_decode_tile_sbrow`, i.e. BEFORE the tile's
/// progress counter is published, so the filter task for this superblock row
/// — which is gated on every tile having published it — always sees complete
/// pixels.
///
/// Every row borrow taken here covers exactly the tile's own columns, so two
/// tile columns stitching the same rows never overlap: `w` pixels reserved,
/// `w` pixels written, no inter-row gap. At 4K with 4x2 tiles that is 64 row
/// borrows per tile per superblock row — 17,408 per frame against the
/// 22,700,725 the split reconstruction path costs.
#[cfg(feature = "tile-owned-recon")]
pub(crate) fn stitch_sbrow<BD: BitDepth>(
    f: &Rav1dFrameData,
    t: &crate::src::internal::Rav1dTaskContext,
) {
    let Some(tr) = &f.tile_recon else { return };
    let ts = &f.ts[t.ts];
    let dst_planes = &f.cur.data.as_ref().unwrap().data;
    let src_planes = tr.planes(t.ts);

    let layout = f.cur.p.layout;
    let n_planes = if layout == Rav1dPixelLayout::I400 {
        1
    } else {
        3
    };
    let luma = match luma_rect(
        t.b.y,
        f.sb_step,
        ts.tiling.row_end,
        ts.tiling.col_start,
        ts.tiling.col_end,
    ) {
        Some(r) => r,
        None => return,
    };

    for pl in 0..n_planes {
        let Rect {
            row0,
            row1,
            col0,
            col1,
        } = plane_rect(luma, pl, layout);
        let w = col1 - col0;
        let stride_px = src_planes[pl].pixel_stride::<BD>() as usize;
        for row in row0..row1 {
            let off = row * stride_px + col0;
            let src = src_planes[pl].slice::<BD, _>((off.., ..w));
            let mut dst = dst_planes[pl].slice_mut::<BD, _>((off.., ..w));
            BD::pixel_copy(&mut dst, &src, w);
        }
    }
}

/// A half-open pixel rectangle `[row0, row1) x [col0, col1)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Rect {
    pub row0: usize,
    pub row1: usize,
    pub col0: usize,
    pub col1: usize,
}

/// The LUMA pixel rectangle one `(tile, superblock row)` task reconstructs.
///
/// Pure so the partition property — every rectangle disjoint, their union the
/// whole block grid — can be asserted directly. A stitch that drops or
/// duplicates a row is the failure mode this design invites, and identical
/// frame md5 is the end-to-end detector; this is the unit-level one.
///
/// Bounds are the tile's BLOCK grid, not the visible frame: reconstruction
/// writes whole blocks, so the padded columns/rows past `w`/`h` are written
/// too and must be carried across or the filter chain reads stale bytes.
/// All inputs are in 4-pixel units, as `Rav1dTileStateTiling` stores them.
#[cfg(feature = "tile-owned-recon")]
pub(crate) fn luma_rect(
    by: c_int,
    sb_step: c_int,
    row_end: c_int,
    col_start: c_int,
    col_end: c_int,
) -> Option<Rect> {
    let y1 = std::cmp::min(by + sb_step, row_end);
    if y1 <= by || col_end <= col_start || by < 0 || col_start < 0 {
        return None;
    }
    Some(Rect {
        row0: (by * 4) as usize,
        row1: (y1 * 4) as usize,
        col0: (col_start * 4) as usize,
        col1: (col_end * 4) as usize,
    })
}

/// Subsample a luma rectangle for plane `pl`.
#[cfg(feature = "tile-owned-recon")]
pub(crate) fn plane_rect(luma: Rect, pl: usize, layout: Rav1dPixelLayout) -> Rect {
    if pl == 0 {
        return luma;
    }
    let ss_ver = (layout == Rav1dPixelLayout::I420) as usize;
    let ss_hor = (layout != Rav1dPixelLayout::I444) as usize;
    Rect {
        row0: luma.row0 >> ss_ver,
        row1: luma.row1 >> ss_ver,
        col0: luma.col0 >> ss_hor,
        col1: luma.col1 >> ss_hor,
    }
}

#[cfg(all(test, feature = "tile-owned-recon"))]
mod tests {
    use super::*;

    /// The v4k_8tile geometry: 3840x2160, 4:2:0, 4x2 tiles, 64-pixel
    /// superblocks. `bw = 960` and `bh = 540` in 4-pixel units.
    const BW4: c_int = 960;
    const BH4: c_int = 540;
    const SB_STEP: c_int = 16; // 64 pixels
    const TILE_COLS: usize = 4;
    const TILE_ROWS: usize = 2;

    fn tile_bounds(tc: usize, tr: usize) -> (c_int, c_int, c_int, c_int) {
        // Superblock-aligned splits, exactly as the frame header stores them.
        let sbw = (BW4 + SB_STEP - 1) / SB_STEP;
        let sbh = (BH4 + SB_STEP - 1) / SB_STEP;
        let cs = (sbw * tc as c_int / TILE_COLS as c_int) * SB_STEP;
        let ce = std::cmp::min(
            (sbw * (tc + 1) as c_int / TILE_COLS as c_int) * SB_STEP,
            BW4,
        );
        let rs = (sbh * tr as c_int / TILE_ROWS as c_int) * SB_STEP;
        let re = std::cmp::min(
            (sbh * (tr + 1) as c_int / TILE_ROWS as c_int) * SB_STEP,
            BH4,
        );
        (cs, ce, rs, re)
    }

    /// The load-bearing property: over every tile and every superblock row,
    /// the stitched rectangles TILE the block grid — no pixel written twice
    /// (two tiles racing on one byte), none left behind (the filter chain
    /// reading a stale row).
    #[test]
    fn stitch_rectangles_exactly_partition_the_block_grid() {
        let mut seen = vec![0u8; (BW4 * 4 * BH4 * 4) as usize];
        for tr in 0..TILE_ROWS {
            for tc in 0..TILE_COLS {
                let (cs, ce, rs, re) = tile_bounds(tc, tr);
                let mut by = rs;
                while by < re {
                    let r = luma_rect(by, SB_STEP, re, cs, ce).expect("non-empty sbrow");
                    for y in r.row0..r.row1 {
                        for x in r.col0..r.col1 {
                            seen[y * (BW4 * 4) as usize + x] += 1;
                        }
                    }
                    by += SB_STEP;
                }
            }
        }
        let twice = seen.iter().filter(|&&c| c > 1).count();
        let never = seen.iter().filter(|&&c| c == 0).count();
        assert_eq!((twice, never), (0, 0), "duplicated / dropped pixels");
    }

    /// The last superblock row of a tile is short whenever the tile height is
    /// not a whole number of superblocks — 540 is not a multiple of 16 — and
    /// clamping it to `row_end` is what keeps the partition exact.
    #[test]
    fn a_short_final_sbrow_is_clamped_to_the_tile_and_not_rounded_up() {
        let (cs, ce, _rs, re) = tile_bounds(0, TILE_ROWS - 1);
        assert_eq!(re, BH4, "bottom tile row ends at the block grid");
        let last_by = re - (re % SB_STEP);
        assert_ne!(re % SB_STEP, 0, "this geometry must exercise a short sbrow");
        let r = luma_rect(last_by, SB_STEP, re, cs, ce).unwrap();
        assert_eq!(r.row1, (re * 4) as usize);
        assert!(r.row1 < ((last_by + SB_STEP) * 4) as usize);
    }

    #[test]
    fn chroma_rects_subsample_and_stay_disjoint_across_tile_columns() {
        let a = luma_rect(0, SB_STEP, BH4, 0, 240).unwrap();
        let b = luma_rect(0, SB_STEP, BH4, 240, 480).unwrap();
        for layout in [
            Rav1dPixelLayout::I420,
            Rav1dPixelLayout::I422,
            Rav1dPixelLayout::I444,
        ] {
            let ca = plane_rect(a, 1, layout);
            let cb = plane_rect(b, 1, layout);
            assert_eq!(ca.col1, cb.col0, "chroma columns must abut for this layout");
            assert!(ca.col1 > ca.col0 && cb.col1 > cb.col0);
            let ss_ver = (layout == Rav1dPixelLayout::I420) as usize;
            assert_eq!(ca.row1, a.row1 >> ss_ver);
        }
    }

    #[test]
    fn an_empty_or_inverted_range_produces_no_rectangle() {
        assert!(luma_rect(16, SB_STEP, 16, 0, 240).is_none());
        assert!(luma_rect(0, SB_STEP, BH4, 240, 240).is_none());
    }
}
