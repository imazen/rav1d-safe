//! A per-superblock read cache for the loop filter's COLUMN pass.
//!
//! # What this is for
//!
//! `LfBlock::fill` is the largest single borrow-registration site in the
//! decoder: 3,835,042 per frame on `v4k_8tile` 8bpc at `--threads 8`, 33.6% of
//! the whole decoder's 11,401,399 (`docs/MUT_RECON_KERNELS.md` §11a). Measured
//! here with `--features __probe_lf_hist`, that population splits
//! **69.3% / 30.7%** between the two passes, and the two halves have completely
//! different shapes:
//!
//! | pass | rectangle | registrations |
//! |---|---|---|
//! | H — `filter_plane_cols_*`, vertical edges | `4 * groups` rows x `2 * reach` px | `4 * groups` |
//! | V — `filter_plane_rows_*`, horizontal edges | `2 * reach` rows x `4 * groups` px | `2 * reach` |
//!
//! Fusing `n` groups divides the V pass's cost by `n` and does nothing at all
//! for the H pass, whose rectangle grows in the ROW direction — one guard per
//! picture row per group, no matter how the run fuses. That is measured, not
//! argued: with the [`LF_BATCH_MAX`](super::loopfilter::LF_BATCH_MAX) cap
//! lifted to 32 the V pass's registrations fall by **1.971x** and the H pass's
//! by **1.000x**.
//!
//! So the H pass's 2.66 M/frame can only be cut by fusing ACROSS the caller's
//! `x` loop, and that is what this does.
//!
//! # The shape, and why it needs no halo handshake
//!
//! `docs/MUT_RECON_KERNELS.md` §11b asked whether a filter band could own a
//! row band with a HALO shared with its neighbour, and answered: not in safe
//! Rust, because `&mut` exclusion is a static fact and a run-time ownership
//! handoff is a borrow tracker wearing a different hat.
//!
//! This band sidesteps that question entirely, because **it is a read cache,
//! not an owner.** `LfBlock::close_band` writes each changed span to the
//! PICTURE exactly as `LfBlock::close` always has — same mutable guard, same
//! extent, same 17,852 write registrations per frame — and additionally
//! mirrors it into the band so the next `x` in the same superblock sees it.
//! The band is therefore never authoritative and never diverges from the
//! picture:
//!
//! * nothing is ever "handed over", so no halo protocol is needed;
//! * any call may fall back to the picture path at any point and read the same
//!   bytes — which is what makes `open_band` returning `None` (band too small,
//!   rectangle off the plane) simply correct rather than a special case;
//! * the write population, the only MUTABLE one, is bit-for-bit unchanged.
//!
//! The one reservation this widens is the IMMUTABLE read: one guard per
//! picture row over `4 * w + 14` contiguous pixels, instead of one guard per
//! picture row per filtered column over `2 * reach`. Contiguous within a row,
//! ~150 bytes — nowhere near the 50-60 KB strided hull that measured 2.65x
//! SLOWER at t=8 (§11c). The hull's cost was its EXTENT crossing the tracker's
//! wide path; this pays no extent for the count it buys, which is the gap
//! §11d/§11f named.
//!
//! # Ordering
//!
//! The H pass is sequentially dependent along `x`: filtering at `4x` writes
//! pixels that the window at `4(x+1)` reads. The band preserves that exactly —
//! reads and writes go through it in the same order they went through the
//! picture. Across superblocks the dependency also holds, because each band is
//! filled at the start of its own superblock, after the previous superblock's
//! writes have already landed in the picture.

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::PicOffset;
use crate::src::loopfilter::LF_TAP_REACH;

/// Columns of margin on each side of the superblock, in pixels.
///
/// The widest tap window is `+-LF_TAP_REACH`, and the leftmost filtered edge
/// sits at the superblock's own left column, so the band starts `LF_TAP_REACH`
/// pixels before it and ends the same distance after the last edge.
pub(crate) const LF_BAND_PAD: usize = LF_TAP_REACH as usize;

/// Runtime A/B switch, default **ON**: `RAV1D_LF_BAND=0` makes every column
/// call take the picture path.
///
/// One binary, two arms — the `RAV1D_OWNED_RECON` convention this campaign
/// uses so that an A/B pair cannot differ in anything but the arm. Disarmed,
/// the only residual is the `Option` argument threaded through the dispatch,
/// which is what the `bandoff`-vs-`main` column of the measurement prices.
pub(crate) fn lf_band_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("RAV1D_LF_BAND").as_deref(), Ok("0")))
}

/// Widest band either pass needs, on either axis.
///
/// A superblock is at most `4 * 32` pixels on a side, and the padded axis adds
/// [`LF_BAND_PAD`] at each end — columns for the H pass, rows for the V pass.
/// One square capacity covers both orientations, so the same buffer can be
/// reused by either without a re-check that depends on which pass has it.
pub(crate) const LF_BAND_MAX_DIM: usize = 4 * 32 + 2 * LF_BAND_PAD;

/// Reusable scratch for one superblock's column pass.
///
/// Allocated once per `rav1d_loopfilter_sbrow_cols` call (17 per frame at 4K)
/// and reused across every superblock and both chroma planes in it, so the
/// decode path takes no per-superblock allocation.
pub(crate) struct LfBand<BD: BitDepth> {
    buf: Vec<BD::Pixel>,
    /// Live row stride, in pixels. `0` when the band is not filled.
    stride: usize,
    /// Live row count. `0` when the band is not filled.
    rows: usize,
}

impl<BD: BitDepth> LfBand<BD> {
    /// Capacity for `max_rows x max_cols`, zero-filled once.
    pub(crate) fn with_capacity(max_rows: usize, max_cols: usize) -> Self {
        Self {
            buf: vec![0u8.into(); max_rows * max_cols],
            stride: 0,
            rows: 0,
        }
    }

    /// Fill from the picture: `rows` picture rows starting `row_pad` rows
    /// ABOVE `dst`'s row, each `cols` pixels wide starting `col_pad` pixels
    /// left of `dst`.
    ///
    /// The two passes pad on different axes — the H pass reads `+-reach`
    /// columns at the rows it filters, the V pass reads `+-reach` rows at the
    /// columns it filters — so the pad is a parameter rather than a constant
    /// on one axis.
    ///
    /// ONE immutable guard per picture row, over a CONTIGUOUS span — this is
    /// the whole registration cost of the pass that replaces
    /// `4 * groups` guards per filtered column.
    ///
    /// Returns `false` — leaving the band disarmed, so callers use the picture
    /// path — when the rectangle would leave the plane or exceed the capacity
    /// this band was built with. That is a correctness-free decision: the band
    /// is only ever a cache of what the picture already holds.
    pub(crate) fn fill_from(
        &mut self,
        dst: PicOffset,
        pic_stride: isize,
        rows: usize,
        cols: usize,
        row_pad: usize,
        col_pad: usize,
    ) -> bool {
        self.stride = 0;
        self.rows = 0;
        if rows == 0 || cols == 0 || rows * cols > self.buf.len() {
            return false;
        }
        let first = dst.offset as isize - row_pad as isize * pic_stride - col_pad as isize;
        if first < 0 {
            return false;
        }
        // Row offsets are monotone in `r`, so checking the two ends bounds all
        // of them — including a negative picture stride.
        let last = first + (rows as isize - 1) * pic_stride;
        let plane = dst.data.pixel_len::<BD>() as isize;
        if first.min(last) < 0 || first.max(last) + cols as isize > plane {
            return false;
        }
        for r in 0..rows {
            let off = (first + r as isize * pic_stride) as usize;
            let guard = PicOffset {
                data: dst.data,
                offset: off,
            }
            .slice::<BD>(cols);
            self.buf[r * cols..][..cols].copy_from_slice(&guard);
        }
        self.stride = cols;
        self.rows = rows;
        true
    }

    /// Disarm without filling, so every call takes the picture path.
    pub(crate) fn disarm(&mut self) {
        self.stride = 0;
        self.rows = 0;
    }

    /// The live band, or `None` when disarmed.
    #[inline(always)]
    pub(crate) fn view(&mut self) -> Option<LfBandView<'_, BD>> {
        if self.stride == 0 {
            return None;
        }
        Some(LfBandView {
            stride: self.stride,
            rows: self.rows,
            buf: &mut self.buf[..self.rows * self.stride],
        })
    }
}

/// A filled band, handed to one `loop_filter_sb128` call.
pub(crate) struct LfBandView<'c, BD: BitDepth> {
    pub(crate) buf: &'c mut [BD::Pixel],
    /// Row stride in pixels — also the band's column count.
    pub(crate) stride: usize,
    pub(crate) rows: usize,
}

/// One `loop_filter_sb128` call's window on a band.
///
/// The anchor is carried as `(row, col)` rather than a flat offset on purpose:
/// the V pass's rectangle grows along a row, and a flat
/// `offset + len <= buf.len()` bound cannot tell a column overrun (which would
/// silently read the NEXT band row) from a row overrun. Two axis bounds can.
pub(crate) struct LfBandCursor<'c, BD: BitDepth> {
    pub(crate) buf: &'c mut [BD::Pixel],
    /// Band row/column of the pixel the picture call names as `dst`.
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) stride: usize,
    pub(crate) rows: usize,
}

