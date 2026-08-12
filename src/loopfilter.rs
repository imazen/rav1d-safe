#![cfg_attr(not(asm_loopfilter), forbid(unsafe_code))]
use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::Align16;
use crate::src::cpu::CpuFlags;
use crate::src::ffi_safe::FFISafe;
use crate::src::internal::Rav1dFrameData;
use crate::src::lf_mask::Av1FilterLUT;
use crate::src::strided::Strided as _;
use crate::src::with_offset::WithOffset;
use crate::src::wrap_fn_ptr::wrap_fn_ptr;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering::Relaxed;
#[allow(non_camel_case_types)]
type ptrdiff_t = isize;
use std::cmp;
use std::ffi::c_int;
use strum::FromRepr;

#[cfg(all(
    asm_loopfilter,
    not(any(target_arch = "riscv64", target_arch = "riscv32"))
))]
use crate::include::common::bitdepth::bd_fn;

#[cfg(not(asm_loopfilter))]
use crate::src::enum_map::DefaultValue;

wrap_fn_ptr!(pub unsafe extern "C" fn loopfilter_sb(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) -> ());

/// Direct dispatch for loopfilter_sb - bypasses function pointer table.
///
/// Selects optimal SIMD implementation at runtime based on CPU features.
/// Used when ASM loopfilter is disabled for zero-overhead direct calls.
#[cfg(not(asm_loopfilter))]
fn loopfilter_sb_scalar<BD: BitDepth>(
    dst: PicOffset,
    mask: &[u32; 3],
    lvl: WithOffset<&[AtomicU8]>,
    b4_stride: usize,
    lut: &Align16<Av1FilterLUT>,
    wh: c_int,
    bd: BD,
    is_y: bool,
    is_v: bool,
) {
    match (is_y, is_v) {
        (true, false) => loop_filter_sb128_rust::<BD, { HV::H as usize }, { YUV::Y as usize }>(
            dst, mask, lvl, b4_stride, lut, wh, bd,
        ),
        (true, true) => loop_filter_sb128_rust::<BD, { HV::V as usize }, { YUV::Y as usize }>(
            dst, mask, lvl, b4_stride, lut, wh, bd,
        ),
        (false, false) => loop_filter_sb128_rust::<BD, { HV::H as usize }, { YUV::UV as usize }>(
            dst, mask, lvl, b4_stride, lut, wh, bd,
        ),
        (false, true) => loop_filter_sb128_rust::<BD, { HV::V as usize }, { YUV::UV as usize }>(
            dst, mask, lvl, b4_stride, lut, wh, bd,
        ),
    }
}

#[cfg(not(asm_loopfilter))]
fn loopfilter_sb_direct<BD: BitDepth>(
    f: &Rav1dFrameData,
    dst: PicOffset,
    mask: &[u32; 3],
    lvl: WithOffset<&[AtomicU8]>,
    w: usize,
    is_y: bool,
    is_v: bool,
) {
    // The invariant every filter-side guard policy rests on, checked where the
    // superblock geometry is in scope (#494).
    //
    // `src/thread_task.rs::check_tile` runs reconstruction of superblock row
    // N+1 concurrently with the deblock tasks of row N — sound only because a
    // horizontal-edge filter cannot read or write past the bottom of its own
    // superblock row. That is a property of the MASK, not of the plane: see
    // [`lf_run_reach`]. Assert it directly, so a window sized from anything
    // wider than the mask fails loudly and deterministically at `t=1` instead
    // of racing a tile worker one run in a thousand at `t=8`.
    //
    // Downward only: the upward reach deliberately enters the previous
    // superblock row (dav1d's CDEF task lags 8 luma / 4 chroma rows behind
    // exactly so that it can).
    //
    // Negative strides are skipped: the row index would need the plane's
    // bottom-up base, and no filter path constructs one.
    #[cfg(debug_assertions)]
    if is_v && (mask[0] | mask[1] | mask[2]) != 0 {
        use crate::include::dav1d::headers::Rav1dPixelLayout;
        let pxstride = dst.pixel_stride::<BD>();
        if pxstride > 0 {
            let pxstride = pxstride as usize;
            let base = dst.data.with_offset::<BD>().offset;
            let row = (dst.offset - base) / pxstride;
            let ss_ver = (!is_y && f.cur.p.layout == Rav1dPixelLayout::I420) as u8;
            let sb_h = ((f.sb_step as usize) * 4) >> ss_ver;
            let reach = lf_run_reach(is_y, mask);
            debug_assert!(
                row % sb_h + reach <= sb_h,
                "V-run window leaves the superblock row: row {row} (+{reach}) \
                 in a {sb_h}-row superblock row, is_y={is_y}, mask={mask:08x?}"
            );
        }
    }

    let stride = dst.stride();
    let b4_stride = f.b4_stride;
    let lut = &f.lf.lim_lut;
    let wh = w as c_int;
    let bd_max = f.bitdepth_max;

    // Save pre-SIMD state for comparison testing
    #[cfg(feature = "__simd_test")]
    let saved_buf = {
        let (guard, _) = dst.full_guard::<BD>();
        guard.to_vec()
    };

    let simd_handled = {
        #[cfg(target_arch = "x86_64")]
        {
            crate::src::safe_simd::loopfilter::loopfilter_sb_dispatch::<BD>(
                dst, stride, mask, lvl, b4_stride, lut, wh, bd_max, is_y, is_v,
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            crate::src::safe_simd::loopfilter_arm::loopfilter_sb_dispatch::<BD>(
                dst, stride, mask, lvl, b4_stride, lut, wh, bd_max, is_y, is_v,
            )
        }
        #[cfg(target_arch = "wasm32")]
        {
            crate::src::safe_simd::loopfilter::loopfilter_sb_dispatch::<BD>(
                dst, stride, mask, lvl, b4_stride, lut, wh, bd_max, is_y, is_v,
            )
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )))]
        {
            let _ = (stride, &mask, &lvl, b4_stride, lut, wh, bd_max, is_y, is_v);
            false
        }
    };

    if simd_handled {
        #[cfg(feature = "__simd_test")]
        {
            // Save SIMD output
            let (guard, _) = dst.full_guard::<BD>();
            let simd_buf = guard.to_vec();
            drop(guard);

            // Restore pre-SIMD state
            {
                let (mut guard, _) = dst.full_guard_mut::<BD>();
                guard.copy_from_slice(&saved_buf);
            }

            // Run scalar on restored state
            let bd = BD::from_c(bd_max);
            loopfilter_sb_scalar::<BD>(dst, mask, lvl, b4_stride as usize, lut, wh, bd, is_y, is_v);

            // Compare SIMD vs scalar
            let (guard, _) = dst.full_guard::<BD>();
            let scalar_buf = guard.to_vec();
            drop(guard);

            let pxstride = dst.pixel_stride::<BD>().unsigned_abs();
            let mut diffs = 0u32;
            let mut first_diff = None;
            for (i, (&sv, &rv)) in simd_buf.iter().zip(scalar_buf.iter()).enumerate() {
                let sv = sv.as_::<i32>();
                let rv = rv.as_::<i32>();
                if sv != rv {
                    diffs += 1;
                    if first_diff.is_none() {
                        let y = i / pxstride;
                        let x = i % pxstride;
                        first_diff = Some((i, x, y, sv, rv));
                    }
                }
            }
            if let Some((idx, x, y, sv, rv)) = first_diff {
                let msg = format!(
                    "LF_MISMATCH diffs={} first=({},{}) idx={} simd={} scalar={} is_y={} is_v={} w={}",
                    diffs, x, y, idx, sv, rv, is_y, is_v, w
                );
                if cfg!(feature = "__simd_test_log") {
                    eprintln!("{msg}");
                } else {
                    panic!("{msg}");
                }
            }

            // Restore SIMD output so decoder proceeds correctly
            {
                let (mut guard, _) = dst.full_guard_mut::<BD>();
                guard.copy_from_slice(&simd_buf);
            }
        }
        return;
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    {
        let b4_stride = b4_stride as usize;
        let bd = BD::from_c(bd_max);
        loopfilter_sb_scalar::<BD>(dst, mask, lvl, b4_stride, lut, wh, bd, is_y, is_v);
    }
}

impl loopfilter_sb::Fn {
    #[allow(dead_code)]
    pub fn call<BD: BitDepth>(
        &self,
        f: &Rav1dFrameData,
        dst: PicOffset,
        mask: &[u32; 3],
        lvl: WithOffset<&[AtomicU8]>,
        w: usize,
        is_y: bool,
        is_v: bool,
    ) {
        cfg_if::cfg_if! {
            if #[cfg(asm_loopfilter)] {
                let _ = (is_y, is_v);
                let dst_ptr = dst.as_mut_ptr::<BD>().cast();
                let stride = dst.stride();
                assert!(lvl.offset <= lvl.data.len());
                // SAFETY: `lvl.offset` is in bounds, just checked above.
                // AtomicU8 has the same size/alignment as u8.
                let lvl_ptr = unsafe { (lvl.data.as_ptr() as *const u8).add(lvl.offset) };
                let lvl_ptr = lvl_ptr.cast::<[u8; 4]>();
                let b4_stride = f.b4_stride;
                let lut = &f.lf.lim_lut;
                let w = w as c_int;
                let bd = f.bitdepth_max;
                let dst = FFISafe::new(&dst);
                let lvl = FFISafe::new(&lvl);
                // SAFETY: Fallback `fn loop_filter_sb128_rust` is safe; asm is supposed to do the same.
                unsafe {
                    self.get()(
                        dst_ptr, stride, mask, lvl_ptr, b4_stride, lut, w, bd, dst, lvl,
                    )
                }
            } else {
                // Direct dispatch: no function pointers, no extern "C" ABI overhead
                loopfilter_sb_direct::<BD>(f, dst, mask, lvl, w, is_y, is_v)
            }
        }
    }

    #[cfg(asm_loopfilter)]
    const fn default<BD: BitDepth, const HV: usize, const YUV: usize>() -> Self {
        Self::new(loop_filter_sb128_c_erased::<BD, { HV }, { YUV }>)
    }
}

pub struct LoopFilterHVDSPContext {
    pub h: loopfilter_sb::Fn,
    pub v: loopfilter_sb::Fn,
}

pub struct LoopFilterYUVDSPContext {
    pub y: LoopFilterHVDSPContext,
    pub uv: LoopFilterHVDSPContext,
}

pub struct Rav1dLoopFilterDSPContext {
    pub loop_filter_sb: LoopFilterYUVDSPContext,
}

/// How the filter reaches the pixels of one tap line.
///
/// Two implementations, and the difference is the point of
/// [`LF_TAP_REACH`]: [`DirectTaps`] registers a fresh one-pixel `DisjointMut`
/// borrow for *every* tap read and write — ~26 of them per column at `wd = 16`,
/// which measured as 44.6% of all borrow-tracker CPU at 8 threads — while
/// [`CompactTaps`] works on a scratch copy of the whole superblock edge that
/// was read in under one guard per row.
trait LfTaps<BD: BitDepth> {
    /// Tap `k` of column `idx`. `k` is in `-LF_TAP_REACH ..= LF_TAP_REACH - 1`
    /// and `idx` in `0..4`.
    fn get(&self, idx: isize, k: isize) -> i32;
    fn set(&mut self, idx: isize, k: isize, px: BD::Pixel);
}

/// Taps read straight from the picture, one tracked borrow each.
struct DirectTaps<'a> {
    dst: PicOffset<'a>,
    stridea: ptrdiff_t,
    strideb: ptrdiff_t,
}

impl<BD: BitDepth> LfTaps<BD> for DirectTaps<'_> {
    #[inline(always)]
    fn get(&self, idx: isize, k: isize) -> i32 {
        (*(self.dst + (self.stridea * idx + self.strideb * k)).index_mut::<BD>()).as_::<i32>()
    }
    #[inline(always)]
    fn set(&mut self, idx: isize, k: isize, px: BD::Pixel) {
        *(self.dst + (self.stridea * idx + self.strideb * k)).index_mut::<BD>() = px;
    }
}

/// Taps in a compact scratch buffer — no borrow tracking at all in here.
///
/// The buffer is a fixed-size array, not a slice, and `at` masks the index:
/// that is what lets LLVM prove every access in-range and drop the bounds
/// check. `loop_filter` reads and writes ~26 taps per column x 4 columns, so
/// the checks are not incidental. The mask cannot hide a real out-of-range
/// index — `open` sizes the rectangle from `lf_reach(wd)` and `loop_filter`
/// never reads outside `+-reach`, which the `debug_assert` below pins.
struct CompactTaps<'a, BD: BitDepth> {
    buf: &'a mut [BD::Pixel; LF_BLOCK_LEN],
    /// Index of `(idx, k) = (0, 0)` within `buf`.
    base: usize,
    stridea: isize,
    strideb: isize,
    /// Live prefix of `buf`; the `debug_assert` bound, not a runtime limit.
    len: usize,
}

impl<BD: BitDepth> CompactTaps<'_, BD> {
    #[inline(always)]
    fn at(&self, idx: isize, k: isize) -> usize {
        let raw = self
            .base
            .wrapping_add_signed(self.stridea * idx + self.strideb * k);
        debug_assert!(raw < self.len, "tap ({idx},{k}) outside the opened block");
        raw & (LF_BLOCK_LEN - 1)
    }
}

impl<BD: BitDepth> LfTaps<BD> for CompactTaps<'_, BD> {
    #[inline(always)]
    fn get(&self, idx: isize, k: isize) -> i32 {
        self.buf[self.at(idx, k)].as_::<i32>()
    }
    #[inline(always)]
    fn set(&mut self, idx: isize, k: isize, px: BD::Pixel) {
        let i = self.at(idx, k);
        self.buf[i] = px;
    }
}

/// Widest tap window any `wd` reads: `p6` at `-7` through `q6` at `+6`.
const LF_TAP_REACH: isize = 7;

/// Widest `2 * reach x 4` (or `4 x 2 * reach`) tap block.
const LF_BLOCK_MAX: usize = 4 * 2 * LF_TAP_REACH as usize;

/// Consecutive 4-pixel groups fused into one [`LfBlock`].
///
/// Fusing is what makes the guard count drop: a `V` edge's rectangle is 4
/// pixels wide and `2 * reach` rows tall, so an unfused call takes up to 14
/// guards to cover 4 pixels each. Fusing `n` groups keeps the same rows but
/// makes them `4 * n` wide — same pixels, `n`x fewer guards.
pub(crate) const LF_BATCH_MAX: usize = 4;

/// Allocation for a fused block: `4 * LF_BATCH_MAX` columns x `2 * reach`
/// taps, rounded up to a power of two so `& (LEN - 1)` proves every
/// [`CompactTaps`] index in-range to LLVM.
pub(crate) const LF_BLOCK_LEN: usize = (LF_BLOCK_MAX * LF_BATCH_MAX).next_power_of_two();

/// Row stride of the scratch rectangle, in pixels.
///
/// The rectangle read from the picture is `w x h` with `w, h <= 16`, but the
/// scratch stores each row at a FIXED stride so a vector kernel can address
/// tap `k` of lane `j` without a runtime multiply and — for the H direction —
/// so a whole row is one aligned 16-lane load. `w` (what is guarded and
/// written back) is unchanged by the padding; the pad columns are never read
/// by `close` and never reach the picture.
pub(crate) const LF_BW: usize = 16;

/// The padded rectangle is exactly `LF_BW x LF_BW`, and `LF_BLOCK_LEN` is a
/// power of two, so `& (LF_BLOCK_LEN - 1)` in [`CompactTaps::at`] cannot hide
/// an out-of-range tap.
const _: () = assert!(LF_BW * LF_BW == LF_BLOCK_LEN);

/// Reusable scratch for [`LfBlock`], owned by one `loop_filter_sb128_rust`
/// call so the zero-init is paid once per superblock edge instead of once per
/// 4-pixel group.
///
/// Stored at the PICTURE's pixel width, so `open` and `close` stay plain
/// `copy_from_slice`. A `u16`-at-both-depths scratch was tried and measured
/// WORSE: it turns those two memcpys into element-wise widen/narrow loops over
/// a runtime-variable length, which cost more (+146 `sample` leaves at t=1,
/// 8bpc, v4k_8tile) than making one kernel serve both depths saved. The kernel
/// splits by bit depth at its own seam instead.
struct LfScratch<BD: BitDepth> {
    buf: [BD::Pixel; LF_BLOCK_LEN],
    pristine: [BD::Pixel; LF_BLOCK_LEN],
}

impl<BD: BitDepth> LfScratch<BD> {
    fn new() -> Self {
        Self {
            buf: [BD::Pixel::from(0u8); LF_BLOCK_LEN],
            pristine: [BD::Pixel::from(0u8); LF_BLOCK_LEN],
        }
    }
}

/// Exactly what one `loop_filter` call can read, expressed as a rectangle.
///
/// `wd` fixes the tap window before any pixel is touched: `+-7` at 16, `+-4` at
/// 8, `+-3` at 6, `+-2` at 4. Four columns share that window, so the call's
/// whole read set is a `2*reach x 4` rectangle (H) or `4 x 2*reach` (V) — read
/// in with ONE guard per picture row instead of [`DirectTaps`]'s fresh
/// one-pixel borrow per tap, of which there are up to 26 per column.
///
/// The rectangle is never wider than the direct path's own read set for the
/// same `wd`, which is what keeps this from inventing false overlaps: any
/// concurrent writer inside it would already be tripping the direct path.
/// The write-back goes further and only touches pixels that actually changed,
/// so it never takes a mutable guard on a tap the filter merely read — the
/// rule `compact_write_back_per_row_diff` exists for (zenavif#30).
/// Runtime switch: take the strided HULL for [`LfBlock::fill`]'s reads even
/// when tile threading is active. **Default off. Measurement arm, not a
/// shipping policy** — both A/B arms are therefore the same binary, the
/// `RAV1D_OWNED_RECON` convention.
///
/// # What this is for
///
/// `LfBlock::fill` is the largest single borrow-registration site left in the
/// decoder: measured on this branch, `v4k_8tile` 8bpc, `--features
/// probe-sites`, registrations per frame —
///
/// | | t=1 | t=8 |
/// |---|---|---|
/// | whole decoder | 6,005,602 | 11,401,399 |
/// | whole filter chain | 995,665 (16.6%) | 6,391,462 (56.1%) |
/// | `fill` alone | (hull path) | 3,835,042 (33.6%) |
///
/// The filter chain's population is **6.4x larger at t=8 than at t=1**, and
/// essentially all of the growth is this one policy branch: at t=1 `fill`
/// takes the hull and costs ~0.47M, at t=8 it takes `h` guards per open. So
/// the filter chain's cost is not intrinsic to what it reads — it is the
/// per-row split, the same shape #482 removed from reconstruction.
///
/// # What is established, and what is NOT
///
/// ESTABLISHED — widening an **immutable** reservation cannot weaken
/// detection. The hull is a superset of the `h` narrow rows, so any overlap
/// the narrow set would have caught, the hull catches too. The only failure
/// mode it can introduce is a FALSE POSITIVE — a loud `overlapping
/// DisjointMut` panic — never a silent wrong pixel. That is what makes this
/// arm safe to run and measure at all, unlike a `tracker: None` probe.
///
/// ESTABLISHED — `LfBlock::close`, the WRITE side, is untouched and still
/// takes one narrow mutable guard per changed row span (17,852 per frame at
/// t=8, `src/loopfilter.rs:739`). Nothing here widens a mutable reservation.
///
/// NOT ESTABLISHED — that no concurrent writer ever touches the gap columns.
/// The pipeline argument is *suggestive*: `rav1d_filter_sbrow_cdef`
/// (`src/recon.rs:3776-3789`) deliberately lags, filtering
/// `[sby*sbsz - 2, sby*sbsz)` and stopping 2 block rows short of the end of
/// its own superblock row, so CDEF(N-1)'s last written luma row is
/// `N*sbsz*4 - 9` while the deepest row `rav1d_loopfilter_sbrow_rows(N)`
/// reaches is `N*sbsz*4 - 7` (`lf_reach` maxes at 7) — disjoint, but by ONE
/// row. And the tile-recon gate (`recon_progress = sby + 2`,
/// `src/thread_task.rs:1030-1036`) is checked over one tile row's tiles, which
/// this round did not chase to the tile-ROW boundary case.
///
/// Until that argument is closed, this stays default-off: a rarely-firing
/// false positive is a decode failure, and "766 vectors passed" is evidence,
/// not a proof.
///
/// # And it does not matter, because it MEASURED 1.98x SLOWER
///
/// See `docs/MUT_RECON_KERNELS.md` §10. Removing 3,463,025 registrations per
/// frame made `v4k_8tile` 8bpc t=8 take 1.98x the user CPU (3.12x with the
/// recon band also disarmed). The hull's extent is 14-16 picture rows, tens of
/// KB, which is far past the sharded tracker's block size and lands on the wide
/// path; fourteen 8-byte registrations are much cheaper than one 50 KB one.
/// The switch is kept, behind a feature so the default build folds it away, as
/// the reproduction for that negative.
#[cfg(feature = "__probe_lf_hull")]
fn lf_hull_reads() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("RAV1D_LF_HULL").as_deref(), Ok("1")))
}

/// The default build: no env read, no atomic, no branch — the constant folds
/// the hull arm out of `fill` entirely.
#[cfg(not(feature = "__probe_lf_hull"))]
#[inline(always)]
fn lf_hull_reads() -> bool {
    false
}

/// The OTHER half of the same ablation: force the PER-ROW read path even
/// without tile threading, i.e. at `t=1`, where it is unconditionally sound
/// (no second thread exists) and uncontended.
///
/// This is what separates the two candidate explanations for the filter
/// chain's t=8 registration population. `RAV1D_LF_HULL=1` says what one wide
/// registration costs against `h` narrow ones WITH contention;
/// `RAV1D_LF_PERROW=1` says what `h` narrow registrations cost against one
/// wide one WITHOUT it. Together they price count against extent, which is the
/// question a next attempt at converting the filter chain has to answer before
/// it starts: a band removes the COUNT, and only pays off if the count is what
/// costs.
#[cfg(feature = "__probe_lf_hull")]
fn lf_force_per_row() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("RAV1D_LF_PERROW").as_deref(), Ok("1")))
}

#[cfg(not(feature = "__probe_lf_hull"))]
#[inline(always)]
fn lf_force_per_row() -> bool {
    false
}

/// The third arm: take EACH per-row read guard TWICE, doubling `LfBlock::fill`'s
/// population without changing any extent, any output, or any other cost.
///
/// This is the only sound way to price a filter-chain registration **at t=8, on
/// the contended path**, and it is what decides whether an owned filter band is
/// worth building. `RAV1D_LF_PERROW` prices one uncontended at t=1;
/// `RAV1D_LF_HULL` cannot price anything because it substitutes a much worse
/// extent (§11c). Doubling changes nothing but the count.
///
/// Sound by construction: the extra guard is IMMUTABLE and covers exactly the
/// same bytes as the one that follows it, and two immutable reservations never
/// conflict — so this cannot invent an overlap that the single guard would not
/// already have found. It is a marginal-cost probe, and the marginal cost of
/// the second 3.46 M is an upper bound on the first's only if the tracker is
/// linear in population; it is reported as the marginal number it is.
#[cfg(feature = "__probe_lf_hull")]
fn lf_double_reads() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("RAV1D_LF_DOUBLE").as_deref(), Ok("1")))
}

#[cfg(not(feature = "__probe_lf_hull"))]
#[inline(always)]
fn lf_double_reads() -> bool {
    false
}

/// LAYOUT-NOISE CONTROL, measurement only: N KiB of `__text` that is never
/// executed, emitted from THIS module so the linker places it where
/// [`LfBlock::fill_rect`]'s monomorphisations go.
///
/// # Why the campaign needed this and did not have it
///
/// `docs/RECT_RECORDS.md` §5d measured a **+1.0% to +1.3% wall at t=1 on
/// `v4k8tile`, 0 of 11 rounds below 1.000 in two sessions**, in an arm whose new
/// code CANNOT execute at t=1 (the hull path returns first). It attributed that
/// to code size by elimination, and its layout control (`plainC`, the same
/// source built in a second worktree) read within ±0.1% — so the band looked
/// tight and the effect looked real and specific to the mechanism.
///
/// That control is too weak to support the conclusion. Two builds of the same
/// source differ only in embedded path strings; they do not MOVE any hot symbol
/// by kilobytes. This one does exactly that and nothing else: with the feature
/// on, `scripts/perf/text_layout_diff.py` reports **zero** symbols resized and
/// every hot loop-filter symbol keeps a byte-identical instruction stream
/// (`scripts/perf/text_symbol_diff.sh`) — only its ADDRESS changes. Whatever
/// this arm measures is the price of moving code, and it is the band any claim
/// about a t=1 regression from added text has to clear. Measured: it clears
/// nothing — see `docs/RECT_SHIP.md` §3.
///
/// The far counterpart is [`crate::src::text_pad`]. The functions are
/// `#[used]`-anchored through a static table rather than called, so nothing can
/// execute them and nothing can eliminate them.
#[cfg(feature = "__pad_text")]
pub(crate) mod text_pad {
    /// One ~600-byte unit of dead text. `K` only forces a distinct
    /// monomorphisation per slot.
    #[inline(never)]
    pub(crate) extern "C" fn unit<const K: usize>(x: &mut [u64; 32]) -> u64 {
        let mut acc = K as u64;
        for i in 0..32 {
            acc = acc
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(x[i] ^ (i as u64));
            x[i] = acc;
            acc ^= acc >> 29;
        }
        acc
    }

    #[cfg_attr(feature = "__pad_small", allow(unused_macros))]
    macro_rules! rung {
        ($name:ident, $base:expr) => {
            #[used]
            static $name: [extern "C" fn(&mut [u64; 32]) -> u64; 8] = [
                unit::<{ $base }>,
                unit::<{ $base + 1 }>,
                unit::<{ $base + 2 }>,
                unit::<{ $base + 3 }>,
                unit::<{ $base + 4 }>,
                unit::<{ $base + 5 }>,
                unit::<{ $base + 6 }>,
                unit::<{ $base + 7 }>,
            ];
        };
    }

    /// The SMALL rung: two units, ~1.2 KiB, for the low end of the ladder.
    #[cfg(feature = "__pad_small")]
    #[used]
    static PAD_S: [extern "C" fn(&mut [u64; 32]) -> u64; 2] = [unit::<900>, unit::<901>];

    #[cfg(not(feature = "__pad_small"))]
    rung!(PAD1, 0);
    #[cfg(feature = "__pad2")]
    rung!(PAD2, 100);
    #[cfg(feature = "__pad3")]
    rung!(PAD3, 200);
    #[cfg(feature = "__pad4")]
    rung!(PAD4, 300);
}

struct LfBlock<'a, 'b, BD: BitDepth> {
    scratch: &'b mut LfScratch<BD>,
    /// Top-left of the rectangle in picture coordinates.
    origin: PicOffset<'a>,
    stride: isize,
    w: usize,
    h: usize,
    /// `buf` index of group 0's `(idx, k) = (0, 0)`.
    base: usize,
    stridea: isize,
    strideb: isize,
    /// `true` for `HV::V` — taps run down the picture and the scratch is
    /// tap-major, which is what lets the vector kernel skip the transpose.
    ///
    /// Read by exactly one call site, `filter_run`'s NEON arm, which is
    /// `#[cfg(target_arch = "aarch64")]`. On every other target the field is
    /// genuinely never read and `clippy -D warnings` (the CI job, on
    /// ubuntu-latest x86_64) rejects it. Narrow the allow to the targets where
    /// the statement is true instead of cfg-ing the field, which would also
    /// have to cfg the struct literal in `open`.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    is_v: bool,
}

/// Tap reach of a filter width. Mirrors the `wd > 4` / `wd > 6` / `wd >= 16`
/// ladder in [`loop_filter`]; keep the two in step.
#[inline(always)]
fn lf_reach(wd: c_int) -> isize {
    if wd >= 16 {
        7
    } else if wd > 6 {
        4
    } else if wd > 4 {
        3
    } else {
        2
    }
}

/// The filter width one 4-pixel group takes, from its bit in the mask.
///
/// `wd = 4 << idx` for luma with `idx` = 2 when `vmask[2]` has the bit, else 1
/// when `vmask[1]` does, else 0; `wd = 4 + 2 * idx` for chroma with `idx` from
/// `vmask[1]`. The level itself is `min(log2(tx) either side of the edge)`
/// capped at 2 — see `src/lf_mask.rs`.
///
/// Extracted so [`lf_run_reach`] and its test share the ladder with the driver
/// instead of transcribing it.
#[inline(always)]
fn lf_group_wd(is_y: bool, vmask: &[u32; 3], xy: u32) -> c_int {
    if is_y {
        let idx = if vmask[2] & xy != 0 {
            2
        } else {
            (vmask[1] & xy != 0) as c_int
        };
        4 << idx
    } else {
        4 + 2 * (vmask[1] & xy != 0) as c_int
    }
}

/// The deepest [`lf_reach`] any group of ONE `loopfilter_sb` run can need,
/// derived from that run's mask.
///
/// Takes [`lf_group_wd`]'s widest answer over the whole mask: [`lf_reach`] is
/// monotone in `wd`, so the reach of the widest level PRESENT ANYWHERE in the
/// mask bounds every group in the run.
///
/// A group whose level resolves to zero does not filter at all and a group can
/// carry a lower level than the run's widest, so this is an upper bound on the
/// run's reach — never an under-estimate.
///
/// # Why a caller must use this instead of the plane's worst case
///
/// This is the bound the whole no-deblock-barrier argument rests on
/// (`src/thread_task.rs::check_tile`, `054e2ed`): the mask level at a
/// horizontal edge is `min(log2(tx_h) above, log2(tx_h) below)` capped at 2
/// (`src/lf_mask.rs`, `masks[1][by4 + y][cmp::min(ttx, btx)]`), and a
/// transform block never crosses a superblock boundary — so level 2 (reach 7)
/// needs >= 16 rows below the edge inside the superblock row, level 1
/// (reach 4) >= 8, level 0 (reach 2) >= 4. **A window sized from the mask
/// therefore cannot read past the superblock row**, which is what makes the
/// reads orderable against concurrent reconstruction of the next superblock
/// row. A window sized from the plane's worst case (7 luma) reads 3 rows past
/// the bottom of the superblock row at every level-0 edge in the last 4-row
/// band, and those rows belong to whoever is reconstructing or column-filtering
/// the NEXT superblock row. That was the x86_64 t=8 race in #494.
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "wasm32", test)),
    allow(dead_code)
)]
#[inline]
pub(crate) fn lf_run_reach(is_y: bool, vmask: &[u32; 3]) -> usize {
    let wd = if is_y {
        if vmask[2] != 0 {
            16
        } else if vmask[1] != 0 {
            8
        } else {
            4
        }
    } else if vmask[1] != 0 {
        6
    } else {
        4
    };
    lf_reach(wd) as usize
}

impl<'a, 'b, BD: BitDepth> LfBlock<'a, 'b, BD> {
    /// Open the rectangle covering `groups` CONSECUTIVE 4-pixel groups that
    /// all filter with the same `wd`.
    ///
    /// The fused rectangle is exactly the union of the rectangles the
    /// `groups` individual calls would have read — the groups are adjacent
    /// and each contributes its own 4 columns (V) or 4 rows (H), so no pixel
    /// enters the guard that an unfused pass would not have guarded. That is
    /// the whole soundness argument, and it is why a group that does NOT
    /// filter must break the run rather than be spanned: spanning it would
    /// guard columns no call reads, which is what collides with a concurrent
    /// tile worker.
    ///
    /// `None` when the rectangle would leave the plane; the caller then
    /// retries per group and finally falls back to [`DirectTaps`].
    #[inline]
    fn open(
        scratch: &'b mut LfScratch<BD>,
        dst: PicOffset<'a>,
        is_v: bool,
        stride: isize,
        wd: c_int,
        groups: usize,
    ) -> Option<Self> {
        let reach = lf_reach(wd);
        // `w`/`h` are the PICTURE rectangle — what gets guarded and written
        // back. The scratch row stride is `LF_BW` regardless, so `w` may be
        // narrower than a scratch row; see [`LF_BW`].
        let (w, h, origin_delta, stridea, strideb, base) = if is_v {
            // Taps run down the picture; each group's four columns run along x.
            (
                4 * groups,
                2 * reach as usize,
                -reach * stride,
                1isize,
                LF_BW as isize,
                reach as usize * LF_BW,
            )
        } else {
            // Taps run along x; each group's four columns run down the picture.
            (
                2 * reach as usize,
                4 * groups,
                -reach,
                LF_BW as isize,
                1isize,
                reach as usize,
            )
        };
        let first = dst.offset as isize + origin_delta;
        let last = first + (h as isize - 1) * stride;
        if first < 0 || last < 0 {
            return None;
        }
        if first.max(last) as usize + w > dst.data.pixel_len::<BD>() {
            return None;
        }
        let origin = PicOffset {
            data: dst.data,
            offset: first as usize,
        };
        // `w` is one of six values, so the row copy is monomorphized on it:
        // `copy_from_slice` at a RUNTIME length is a `memmove` CALL per row,
        // and at up to 16 rows per open that call overhead measured as ~1.5%
        // of a t=1 8bpc frame. Filling `pristine` in the same pass also
        // retires the separate whole-rectangle memcpy.
        match w {
            4 => Self::fill::<4>(scratch, origin, stride, h),
            6 => Self::fill::<6>(scratch, origin, stride, h),
            8 => Self::fill::<8>(scratch, origin, stride, h),
            12 => Self::fill::<12>(scratch, origin, stride, h),
            14 => Self::fill::<14>(scratch, origin, stride, h),
            16 => Self::fill::<16>(scratch, origin, stride, h),
            _ => {
                for row in 0..h {
                    let off = origin.offset.wrapping_add_signed(row as isize * stride);
                    let guard = PicOffset {
                        data: origin.data,
                        offset: off,
                    }
                    .slice::<BD>(w);
                    scratch.buf[row * LF_BW..][..w].copy_from_slice(&guard);
                    scratch.pristine[row * LF_BW..][..w].copy_from_slice(&guard);
                }
            }
        }
        Some(Self {
            scratch,
            origin,
            stride,
            w,
            h,
            base,
            stridea,
            strideb,
            is_v,
        })
    }

    /// Read `h` picture rows of exactly `W` pixels into the scratch and its
    /// pristine copy.
    ///
    /// Under tile threading: ONE immutable guard per picture row, exactly as
    /// before — the constant `W` changes only the copy's codegen, never the
    /// guard's extent. Without tile threading: ONE guard over the strided hull
    /// for the whole rectangle; see [`Self::fill_hull`].
    ///
    /// [`lf_hull_reads`] takes the hull under tile threading too. It is
    /// default-OFF and measurement-only; see that function for what is and is
    /// not established about it.
    #[inline(always)]
    fn fill<const W: usize>(
        scratch: &mut LfScratch<BD>,
        origin: PicOffset,
        stride: isize,
        h: usize,
    ) {
        if (!crate::include::dav1d::picture::tile_threading_active() || lf_hull_reads())
            && !lf_force_per_row()
        {
            return Self::fill_hull::<W>(scratch, origin, stride, h);
        }
        // Everything reachable only under tile threading lives in ONE
        // out-of-line function, so `open` — which is what `fill` is inlined
        // into, twelve times — carries only the hull path plus a call.
        //
        // This is not tidiness; it is the measured fix for
        // `docs/RECT_RECORDS.md` §5d. The rectangle arm cost **+1.0..+1.3% wall
        // at t=1 on `v4k8tile`, 0 of 11 rounds below 1.000 in two sessions**,
        // where the hull path returns above and no rectangle is ever
        // registered. `scripts/perf/text_layout_diff.py` +
        // `text_symbol_diff.sh` located it: every hot loop-filter symbol kept a
        // BYTE-IDENTICAL instruction stream, and the only executed function
        // whose codegen changed at all was `LfBlock::open` — +2 instructions and
        // a register-allocation churn from the twelve cold `fill_rect` calls it
        // had inlined into it. Out-of-lining the whole tile-threading tail makes
        // `open`'s t=1 body SMALLER than base's (base inlines the per-row loop
        // there too) rather than larger.
        //
        // At t=8 it costs one call per `fill` — once per `h` rows, not once per
        // row — on a path that already takes a borrow registration.
        Self::fill_threaded::<W>(scratch, origin, stride, h)
    }

    /// The tile-threading half of [`Self::fill`], out of line: try the exact
    /// strided-RECTANGLE record, and fall back to `h` per-row guards when the
    /// geometry is not representable as one.
    ///
    /// # Why the extent may be one record here and must not be a hull
    ///
    /// The record covers **only** the `h` row segments — `rect_hit_range` walks
    /// rows and knows nothing of the inter-row gaps — so a concurrent writer in a
    /// gap (another tile column of the same picture rows, which is the routine
    /// case under tile threading) is not reported. That is exactly the false
    /// positive that confines [`Self::fill_hull`] to `!tile_threading_active()`,
    /// and it cannot arise here. Nor is any reference wider than one row created:
    /// `DisjointImmutRectGuard` has no `Deref` and derives each row from the
    /// buffer's own pointer, which is what the March-2026 strided tracker got
    /// wrong (exact record, hull-wide reference: UB under both aliasing models).
    ///
    /// `None` from `index_rect_as` is a REFUSAL, never an approximation — no
    /// declared stride, a stride mismatch, `W > stride`, `h > MAX_RECT_ROWS`, a
    /// hull spanning more than `MAX_SHARDS_PER_BORROW` blocks, or a full shard —
    /// and then the per-row loop below runs exactly as it did before rectangles
    /// existed.
    /// `#[inline(never)]`, so `open` carries only the hull path plus a call and
    /// the twelve `fill_rect` monomorphisations stay out of it. (Until the
    /// default flip this was `inline(always)` whenever the rectangle was off,
    /// to keep the default build's codegen bit-for-bit what `main`'s was; the
    /// rectangle is now unconditional, so there is no such arm to preserve.)
    #[inline(never)]
    fn fill_threaded<const W: usize>(
        scratch: &mut LfScratch<BD>,
        origin: PicOffset,
        stride: isize,
        h: usize,
    ) {
        // THROWAWAY (`__probe_bounds`): price the strided-rectangle
        // counterfactual against the live set BEFORE taking the `h` per-row
        // guards, i.e. at the instant a single 2-D registration would have
        // been made. Registers nothing; absent from codegen without the
        // feature. See `bounds_probe::eval_rect`.
        {
            let ps = core::mem::size_of::<BD::Pixel>();
            origin.data.dm().probe_eval_rect(
                core::panic::Location::caller(),
                false,
                origin.offset * ps,
                W * ps,
                h,
                stride * ps as isize,
            );
        }
        // The strided-RECTANGLE record: ONE registration describing the `h`
        // exact row segments, reserving no inter-row gap. This is the third
        // path, not a replacement for either of the other two — `None` means the
        // geometry is not representable as one record (no declared stride, a
        // hull spanning more than `MAX_SHARDS_PER_BORROW` blocks, a full shard)
        // and the per-row loop below runs exactly as before.
        //
        // It is sound at t=8 for the reason the hull is not: the record covers
        // only the segments, so a concurrent writer in a GAP — another tile
        // column of the same picture rows, which is the routine case — is not
        // reported. `fill_hull` cannot be used here for exactly that reason; see
        // its doc comment.
        if Self::fill_rect::<W>(scratch, origin, stride, h) {
            return;
        }
        for row in 0..h {
            let off = origin.offset.wrapping_add_signed(row as isize * stride);
            // The MARGINAL price of one filter-chain registration, measured on
            // the real contended path. See [`lf_double_reads`].
            if lf_double_reads() {
                let extra = PicOffset {
                    data: origin.data,
                    offset: off,
                }
                .slice::<BD>(W);
                core::hint::black_box(&extra[0]);
            }
            let guard = PicOffset {
                data: origin.data,
                offset: off,
            }
            .slice::<BD>(W);
            let src: &[BD::Pixel; W] = (&guard[..W]).try_into().expect("guard is W long");
            let dst: &mut [BD::Pixel; W] = (&mut scratch.buf[row * LF_BW..][..W])
                .try_into()
                .expect("scratch row is LF_BW >= W long");
            *dst = *src;
            let pri: &mut [BD::Pixel; W] = (&mut scratch.pristine[row * LF_BW..][..W])
                .try_into()
                .expect("scratch row is LF_BW >= W long");
            *pri = *src;
        }
    }

    /// The strided-RECTANGLE attempt. Returns `true` when it registered and
    /// copied; `false` when the geometry is not representable as one record, in
    /// which case [`Self::fill_threaded`]'s per-row loop runs unchanged.
    ///
    /// Inlined into its single caller, which is itself `#[inline(never)]`; the
    /// soundness argument and the refusal list are on [`Self::fill_threaded`].
    ///
    /// # Why the extent may be one record here and must not be a hull
    ///
    /// The record covers **only** the `h` row segments — `rect_hit_range` walks
    /// rows and knows nothing of the inter-row gaps — so a concurrent writer in a
    /// gap (another tile column of the same picture rows, which is the routine
    /// case under tile threading) is not reported. That is exactly the false
    /// positive that confines [`Self::fill_hull`] to `!tile_threading_active()`,
    /// and it cannot arise here. Nor is any reference wider than one row created:
    /// `DisjointImmutRectGuard` has no `Deref` and derives each row from the
    /// buffer's own pointer, which is what the March-2026 strided tracker got
    /// wrong (exact record, hull-wide reference: UB under both aliasing models).
    ///
    /// `None` from `index_rect_as` is a REFUSAL, never an approximation — no
    /// declared stride, a stride mismatch, `W > stride`, `h > MAX_RECT_ROWS`, a
    /// hull spanning more than `MAX_SHARDS_PER_BORROW` blocks, or a full shard.
    #[inline(always)]
    fn fill_rect<const W: usize>(
        scratch: &mut LfScratch<BD>,
        origin: PicOffset,
        stride: isize,
        h: usize,
    ) -> bool {
        let Some(rect) = origin
            .data
            .dm()
            .index_rect_as::<BD::Pixel>(origin.offset, W, h, stride)
        else {
            return false;
        };
        for row in 0..h {
            let src: &[BD::Pixel; W] = rect.row(row).try_into().expect("row is W long");
            let dst: &mut [BD::Pixel; W] = (&mut scratch.buf[row * LF_BW..][..W])
                .try_into()
                .expect("scratch row is LF_BW >= W long");
            *dst = *src;
            let pri: &mut [BD::Pixel; W] = (&mut scratch.pristine[row * LF_BW..][..W])
                .try_into()
                .expect("scratch row is LF_BW >= W long");
            *pri = *src;
        }
        true
    }

    /// [`Self::fill`] with ONE registration instead of `h`.
    ///
    /// # Why the extent may widen here and nowhere else
    ///
    /// The hull is `[lo, lo + (h-1)*|stride| + W)` — the per-row set PLUS the
    /// inter-row gaps, which belong to other columns of the same picture rows.
    /// Reserving those gaps is what `block_mut`'s doc calls "correct
    /// single-threaded and wrong under tile threading": two tile workers
    /// routinely write the same rows at different columns, and the gap
    /// reservation turns that genuinely-disjoint pair into a false positive.
    ///
    /// So this path is taken ONLY when `tile_threading_active()` is false,
    /// which is the same latch — process-global, monotone, never stores
    /// `false` — that already selects the identical wide-vs-per-row policy in
    /// [`with_pixel_guard_immut`](crate::include::dav1d::picture::with_pixel_guard_immut),
    /// `block_mut` and `compact_read`. This adds a fourth caller to an existing
    /// policy; it does not invent one.
    ///
    /// Detection is never weakened in either direction: a superset reservation
    /// cannot MISS an overlap the `h` narrow ones would have caught, and with
    /// no tile worker alive there is no second writer for the gaps to collide
    /// with.
    ///
    /// # Why it is worth a branch
    ///
    /// This is the single largest borrow-registration site in the decoder:
    /// 3,835,042 of 15,646,727 registrations per frame (24.5%) on `v4k_8tile`
    /// 8bpc at t=1, measured with `--features probe-sites`. Collapsing `h` rows
    /// to one leaves ~0.47M.
    #[inline(always)]
    fn fill_hull<const W: usize>(
        scratch: &mut LfScratch<BD>,
        origin: PicOffset,
        stride: isize,
        h: usize,
    ) {
        debug_assert!(h > 0);
        let astride = stride.unsigned_abs();
        // `open` proved both endpoints are in the plane, so the hull between
        // them is too.
        let lo = if stride >= 0 {
            origin.offset
        } else {
            origin.offset - (h - 1) * astride
        };
        let total = (h - 1) * astride + W;
        let guard = origin.data.slice::<BD, _>((lo.., ..total));
        {
            let ps = core::mem::size_of::<BD::Pixel>();
            guard.probe_declare_rows(lo * ps, W * ps, h, stride * ps as isize);
        }
        for row in 0..h {
            let idx = if stride >= 0 {
                row * astride
            } else {
                (h - 1 - row) * astride
            };
            let src: &[BD::Pixel; W] = (&guard[idx..][..W])
                .try_into()
                .expect("the hull covers W pixels at every row offset");
            let dst: &mut [BD::Pixel; W] = (&mut scratch.buf[row * LF_BW..][..W])
                .try_into()
                .expect("scratch row is LF_BW >= W long");
            *dst = *src;
            let pri: &mut [BD::Pixel; W] = (&mut scratch.pristine[row * LF_BW..][..W])
                .try_into()
                .expect("scratch row is LF_BW >= W long");
            *pri = *src;
        }
    }

    /// Taps of fused group `g`. Group `g` sits `4 * g` columns (V) or rows (H)
    /// along, i.e. `4 * g * stridea` into the buffer either way, and the
    /// groups' tap windows are disjoint — which is why filtering them all
    /// before any write-back is bit-identical to interleaving.
    #[inline]
    fn taps(&mut self, g: usize) -> CompactTaps<'_, BD> {
        CompactTaps {
            len: (self.h - 1) * LF_BW + self.w,
            buf: &mut self.scratch.buf,
            base: self.base.wrapping_add_signed(4 * g as isize * self.stridea),
            stridea: self.stridea,
            strideb: self.strideb,
        }
    }

    /// Filter every group of the fused run.
    ///
    /// The vector kernel and the scalar reference operate on the SAME scratch
    /// rectangle with the same layout, so this seam changes nothing about
    /// which picture pixels are guarded or written — [`Self::open`] and
    /// [`Self::close`] are untouched by the choice.
    #[inline]
    fn filter_run(&mut self, params: &[(u8, u8, u8, c_int)], wd: c_int, bd: BD) {
        #[cfg(target_arch = "aarch64")]
        {
            use zerocopy::IntoBytes as _;
            if crate::src::safe_simd::loopfilter_arm::lf_compact_run_neon(
                BD::BPC,
                self.scratch.buf.as_mut_bytes(),
                self.base,
                self.is_v,
                4 * params.len(),
                params,
                wd,
                bd.bitdepth() - 8,
                bd.bitdepth_max().into(),
            ) {
                return;
            }
        }
        for (j, &(e, i, h, _)) in params.iter().enumerate() {
            loop_filter::<BD, _>(&mut self.taps(j), e, i, h, wd, bd);
        }
    }

    /// The changed span of scratch row `row`, or `None` if it is untouched.
    ///
    /// This is what keeps `close` from taking a mutable guard on a tap the
    /// filter only READ (zenavif#30), so it must stay exact — the vector form
    /// answers the same question, not a looser one.
    #[inline(always)]
    fn changed_span(&self, row: usize) -> Option<(usize, usize)> {
        #[cfg(target_arch = "aarch64")]
        {
            use zerocopy::IntoBytes as _;
            if let Some(span) = crate::src::safe_simd::loopfilter_arm::lf_diff_span(
                BD::BPC,
                self.scratch.buf.as_bytes(),
                self.scratch.pristine.as_bytes(),
                row,
                self.w,
            ) {
                return span;
            }
        }
        let work = &self.scratch.buf[row * LF_BW..][..self.w];
        let orig = &self.scratch.pristine[row * LF_BW..][..self.w];
        let first = work.iter().zip(orig).position(|(a, b)| a != b)?;
        let last = work
            .iter()
            .zip(orig)
            .rposition(|(a, b)| a != b)
            .expect("a differing pixel exists, so rposition finds one");
        Some((first, last))
    }

    /// Write back only the pixels that changed, one row at a time.
    #[inline]
    fn close(self) {
        for row in 0..self.h {
            let Some((first, last)) = self.changed_span(row) else {
                continue; // row untouched: no write, no mutable guard
            };
            let work = &self.scratch.buf[row * LF_BW..][..self.w];
            let off = self
                .origin
                .offset
                .wrapping_add_signed(row as isize * self.stride)
                + first;
            let mut guard = PicOffset {
                data: self.origin.data,
                offset: off,
            }
            .slice_mut::<BD>(last + 1 - first);
            guard.copy_from_slice(&work[first..=last]);
        }
    }
}

#[inline(never)]
fn loop_filter<BD: BitDepth, T: LfTaps<BD>>(taps: &mut T, e: u8, i: u8, h: u8, wd: c_int, bd: BD) {
    let bitdepth_min_8 = bd.bitdepth() - 8;
    let [f, e, i, h] = [1, e, i, h].map(|n| (n as i32) << bitdepth_min_8);

    for idx in 0..4 {
        // Every read happens before every write, so the shared borrow this
        // closure holds on `taps` has ended by the time the writers below are
        // created. (NLL, not a coincidence — keep it that way.)
        let get_dst = |stride_index: isize| T::get(&*taps, idx, stride_index);

        let mut p6 = 0;
        let mut p5 = 0;
        let mut p4 = 0;
        let mut p3 = 0;
        let mut p2 = 0;
        let p1 = get_dst(-2);
        let p0 = get_dst(-1);
        let q0 = get_dst(0);
        let q1 = get_dst(1);
        let mut q2 = 0;
        let mut q3 = 0;
        let mut q4 = 0;
        let mut q5 = 0;
        let mut q6 = 0;
        let mut flat8out = false;
        let mut flat8in = false;

        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;

        if wd > 4 {
            p2 = get_dst(-3);
            q2 = get_dst(2);

            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;

            if wd > 6 {
                p3 = get_dst(-4);
                q3 = get_dst(3);

                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }
        if !fm {
            continue;
        }

        if wd >= 16 {
            p6 = get_dst(-7);
            p5 = get_dst(-6);
            p4 = get_dst(-5);
            q4 = get_dst(4);
            q5 = get_dst(5);
            q6 = get_dst(6);

            flat8out = (p6 - p0).abs() <= f
                && (p5 - p0).abs() <= f
                && (p4 - p0).abs() <= f
                && (q4 - q0).abs() <= f
                && (q5 - q0).abs() <= f
                && (q6 - q0).abs() <= f;
        }

        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f
                && (p1 - p0).abs() <= f
                && (q1 - q0).abs() <= f
                && (q2 - q0).abs() <= f;
        }

        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f && (q3 - q0).abs() <= f;
        }

        // Last read is above; the writers may take `dst` mutably from here on.
        // Macros rather than closures only because two closures cannot both
        // hold `&mut dst`.
        macro_rules! set_dst {
            ($k:expr, $v:expr $(,)?) => {
                T::set(&mut *taps, idx, $k, ($v).as_::<BD::Pixel>())
            };
        }
        macro_rules! set_dst_clipped {
            ($k:expr, $v:expr $(,)?) => {
                T::set(&mut *taps, idx, $k, bd.iclip_pixel($v))
            };
        }

        if wd >= 16 && flat8out && flat8in {
            set_dst!(
                -6,
                p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8 >> 4,
            );
            set_dst!(
                -5,
                p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8 >> 4,
            );
            set_dst!(
                -4,
                p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8 >> 4,
            );
            set_dst!(
                -3,
                p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8 >> 4,
            );
            set_dst!(
                -2,
                p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8 >> 4,
            );
            set_dst!(
                -1,
                p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8 >> 4,
            );
            set_dst!(
                0,
                p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8 >> 4,
            );
            set_dst!(
                1,
                p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8 >> 4,
            );
            set_dst!(
                2,
                p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8 >> 4,
            );
            set_dst!(
                3,
                p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8 >> 4,
            );
            set_dst!(
                4,
                p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8 >> 4,
            );
            set_dst!(
                5,
                p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8 >> 4,
            );
        } else if wd >= 8 && flat8in {
            set_dst!(-3, p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4 >> 3);
            set_dst!(-2, p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4 >> 3);
            set_dst!(-1, p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4 >> 3);
            set_dst!(0, p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4 >> 3);
            set_dst!(1, p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4 >> 3);
            set_dst!(2, p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4 >> 3);
        } else if wd == 6 && flat8in {
            set_dst!(-2, p2 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + 4 >> 3);
            set_dst!(-1, p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4 >> 3);
            set_dst!(0, p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4 >> 3);
            set_dst!(1, p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4 >> 3);
        } else {
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            fn iclip_diff(v: c_int, bitdepth_min_8: u8) -> i32 {
                iclip(
                    v,
                    -128 * (1 << bitdepth_min_8),
                    128 * (1 << bitdepth_min_8) - 1,
                )
            }

            if hev {
                let f = iclip_diff(p1 - q1, bitdepth_min_8);
                let f = iclip_diff(3 * (q0 - p0) + f, bitdepth_min_8);

                let f1 = cmp::min(f + 4, (128 << bitdepth_min_8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bitdepth_min_8) - 1) >> 3;

                set_dst_clipped!(-1, p0 + f2);
                set_dst_clipped!(0, q0 - f1);
            } else {
                let f = iclip_diff(3 * (q0 - p0), bitdepth_min_8);

                let f1 = cmp::min(f + 4, (128 << bitdepth_min_8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bitdepth_min_8) - 1) >> 3;

                set_dst_clipped!(-1, p0 + f2);
                set_dst_clipped!(0, q0 - f1);

                let f = (f1 + 1) >> 1;
                set_dst_clipped!(-2, p1 + f);
                set_dst_clipped!(1, q1 - f);
            }
        }
    }
}

#[derive(FromRepr)]
enum HV {
    H,
    V,
}

#[derive(FromRepr)]
enum YUV {
    Y,
    UV,
}

fn loop_filter_sb128_rust<BD: BitDepth, const HV: usize, const YUV: usize>(
    dst: PicOffset,
    vmask: &[u32; 3],
    lvl: WithOffset<&[AtomicU8]>,
    b4_stride: usize,
    lut: &Align16<Av1FilterLUT>,
    _wh: c_int,
    bd: BD,
) {
    let hv = HV::from_repr(HV).unwrap();
    let yuv = YUV::from_repr(YUV).unwrap();

    let stride = dst.pixel_stride::<BD>();
    let (stridea, strideb) = match hv {
        HV::H => (stride, 1),
        HV::V => (1, stride),
    };
    let (b4_stridea, b4_strideb) = match hv {
        HV::H => (b4_stride, 1),
        HV::V => (1, b4_stride),
    };

    let vm = match yuv {
        YUV::Y => vmask[0] | vmask[1] | vmask[2],
        YUV::UV => vmask[0] | vmask[1],
    };
    let is_v = matches!(hv, HV::V);

    // Resolve every 4-pixel group's filter parameters before touching a pixel,
    // so consecutive groups that actually filter can be fused into one
    // `LfBlock` (see its `open`). `vm` is a `u32`, so there are at most 32.
    let mut params = [(0u8, 0u8, 0u8, 0 as c_int); 32];
    let mut filters = [false; 32];
    let mut n_groups = 0usize;
    {
        let mut xy = 1u32;
        let mut lvl = lvl;
        while vm & !xy.wrapping_sub(1) != 0 {
            if vm & xy != 0 {
                let l = lvl.data[lvl.offset].load(Relaxed);
                let l = if l != 0 {
                    l
                } else {
                    let lvl = lvl - 4 * b4_strideb;
                    lvl.data[lvl.offset].load(Relaxed)
                };
                if l != 0 {
                    let idx = lf_group_wd(matches!(yuv, YUV::Y), vmask, xy);
                    params[n_groups] = (e_of(lut, l), i_of(lut, l), l >> 4, idx);
                    filters[n_groups] = true;
                }
            }
            n_groups += 1;
            xy <<= 1;
            lvl += 4 * b4_stridea;
        }
    }

    let mut scratch = LfScratch::new();
    let group_step = 4 * stridea;
    let mut g = 0usize;
    while g < n_groups {
        if !filters[g] {
            g += 1;
            continue;
        }
        let wd = params[g].3;
        // Extend the run over consecutive groups that also filter and share
        // `wd` — a non-filtering group or a different tap reach ends it, so
        // the fused rectangle never covers a column no call would read.
        let mut n = 1;
        while n < LF_BATCH_MAX && g + n < n_groups && filters[g + n] && params[g + n].3 == wd {
            n += 1;
        }
        let run_dst = dst + group_step * g as isize;
        match LfBlock::<BD>::open(&mut scratch, run_dst, is_v, stride, wd, n) {
            Some(mut block) => {
                block.filter_run(&params[g..g + n], wd, bd);
                block.close();
            }
            None => {
                // The fused rectangle left the plane. Retry each group on its
                // own, then fall back to the per-tap direct path.
                for j in 0..n {
                    let (e, i, h, _) = params[g + j];
                    let one_dst = dst + group_step * (g + j) as isize;
                    match LfBlock::<BD>::open(&mut scratch, one_dst, is_v, stride, wd, 1) {
                        Some(mut block) => {
                            block.filter_run(&params[g + j..g + j + 1], wd, bd);
                            block.close();
                        }
                        None => {
                            let mut taps = DirectTaps {
                                dst: one_dst,
                                stridea,
                                strideb,
                            };
                            loop_filter::<BD, _>(&mut taps, e, i, h, wd, bd);
                        }
                    }
                }
            }
        }
        g += n;
    }
}

#[inline(always)]
fn e_of(lut: &Align16<Av1FilterLUT>, l: u8) -> u8 {
    lut.e[l as usize]
}

#[inline(always)]
fn i_of(lut: &Align16<Av1FilterLUT>, l: u8) -> u8 {
    lut.i[l as usize]
}

/// # Safety
///
/// Must be called by [`loopfilter_sb::Fn::call`].
#[cfg(asm_loopfilter)]
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn loop_filter_sb128_c_erased<BD: BitDepth, const HV: usize, const YUV: usize>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    vmask: &[u32; 3],
    _lvl_ptr: *const [u8; 4],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    wh: c_int,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) {
    // SAFETY: Was passed as `FFISafe::new(_)` in `loopfilter_sb::Fn::call`.
    let dst = *unsafe { FFISafe::get(dst) };
    // SAFETY: Was passed as `FFISafe::new(_)` in `loopfilter_sb::Fn::call`.
    let lvl = *unsafe { FFISafe::get(lvl) };
    let b4_stride = b4_stride as usize;
    let bd = BD::from_c(bitdepth_max);
    loop_filter_sb128_rust::<BD, { HV }, { YUV }>(dst, vmask, lvl, b4_stride, lut, wh, bd)
}

impl Rav1dLoopFilterDSPContext {
    pub const fn default<BD: BitDepth>() -> Self {
        cfg_if::cfg_if! {
            if #[cfg(asm_loopfilter)] {
                use HV::*;
                use YUV::*;
                Self {
                    loop_filter_sb: LoopFilterYUVDSPContext {
                        y: LoopFilterHVDSPContext {
                            h: loopfilter_sb::Fn::default::<BD, { H as _ }, { Y as _ }>(),
                            v: loopfilter_sb::Fn::default::<BD, { V as _ }, { Y as _ }>(),
                        },
                        uv: LoopFilterHVDSPContext {
                            h: loopfilter_sb::Fn::default::<BD, { H as _ }, { UV as _ }>(),
                            v: loopfilter_sb::Fn::default::<BD, { V as _ }, { UV as _ }>(),
                        },
                    },
                }
            } else {
                Self {
                    loop_filter_sb: LoopFilterYUVDSPContext {
                        y: LoopFilterHVDSPContext {
                            h: loopfilter_sb::Fn::DEFAULT,
                            v: loopfilter_sb::Fn::DEFAULT,
                        },
                        uv: LoopFilterHVDSPContext {
                            h: loopfilter_sb::Fn::DEFAULT,
                            v: loopfilter_sb::Fn::DEFAULT,
                        },
                    },
                }
            }
        }
    }

    #[cfg(all(asm_loopfilter, any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline(always)]
    const fn init_x86<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        if !flags.contains(CpuFlags::SSSE3) {
            return self;
        }

        self.loop_filter_sb.y.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_y, ssse3);
        self.loop_filter_sb.y.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_y, ssse3);
        self.loop_filter_sb.uv.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_uv, ssse3);
        self.loop_filter_sb.uv.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_uv, ssse3);

        #[cfg(target_arch = "x86_64")]
        {
            if !flags.contains(CpuFlags::AVX2) {
                return self;
            }

            self.loop_filter_sb.y.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_y, avx2);
            self.loop_filter_sb.y.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_y, avx2);
            self.loop_filter_sb.uv.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_uv, avx2);
            self.loop_filter_sb.uv.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_uv, avx2);

            if !flags.contains(CpuFlags::AVX512ICL) {
                return self;
            }

            self.loop_filter_sb.y.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_y, avx512icl);
            self.loop_filter_sb.uv.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_uv, avx512icl);

            if !flags.contains(CpuFlags::SLOW_GATHER) {
                self.loop_filter_sb.y.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_y, avx512icl);
                self.loop_filter_sb.uv.h =
                    bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_uv, avx512icl);
            }
        }

        self
    }

    #[cfg(all(asm_loopfilter, any(target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    const fn init_arm<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        if !flags.contains(CpuFlags::NEON) {
            return self;
        }

        self.loop_filter_sb.y.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_y, neon);
        self.loop_filter_sb.y.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_y, neon);
        self.loop_filter_sb.uv.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_uv, neon);
        self.loop_filter_sb.uv.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_uv, neon);

        self
    }

    #[inline(always)]
    const fn init<BD: BitDepth>(self, flags: CpuFlags) -> Self {
        #[cfg(asm_loopfilter)]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                return self.init_x86::<BD>(flags);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            {
                return self.init_arm::<BD>(flags);
            }
        }

        #[allow(unreachable_code)] // Reachable on some #[cfg]s.
        {
            let _ = flags;
            self
        }
    }

    pub const fn new<BD: BitDepth>(flags: CpuFlags) -> Self {
        Self::default::<BD>().init::<BD>(flags)
    }
}

// ============================================================================
// TESTS
// ============================================================================

/// [`lf_run_reach`] against the driver's own per-group ladder (#494).
///
/// The oracle is [`lf_group_wd`] — the function `loop_filter_sb128_rust` itself
/// calls — walked over every group the run would visit, so this cannot drift
/// from the filter the way a transcribed ladder would.
#[cfg(test)]
mod run_reach {
    use super::*;

    /// Bit patterns that place levels on different groups, including the first
    /// and last of the 32 a `u32` mask can hold.
    const PATS: [u32; 8] = [
        0,
        1,
        2,
        0x8000_0000,
        0xffff_ffff,
        0x5555_5555,
        0xaaaa_aaaa,
        0x0001_0000,
    ];

    /// The deepest reach any group of this run actually needs, by walking the
    /// groups exactly as the driver does.
    fn oracle(is_y: bool, vmask: &[u32; 3]) -> usize {
        let vm = if is_y {
            vmask[0] | vmask[1] | vmask[2]
        } else {
            vmask[0] | vmask[1]
        };
        let mut want = 0usize;
        for bit in 0..32 {
            let xy = 1u32 << bit;
            if vm & xy != 0 {
                want = want.max(lf_reach(lf_group_wd(is_y, vmask, xy)) as usize);
            }
        }
        want
    }

    #[test]
    fn run_reach_equals_the_widest_group_it_can_meet() {
        let mut covered = [false; 8];
        for &m0 in &PATS {
            for &m1 in &PATS {
                for &m2 in &PATS {
                    let vmask = [m0, m1, m2];
                    for is_y in [true, false] {
                        let want = oracle(is_y, &vmask);
                        if want == 0 {
                            // No group filters: the run returns early before
                            // any window is sized.
                            continue;
                        }
                        assert_eq!(
                            lf_run_reach(is_y, &vmask),
                            want,
                            "is_y={is_y} mask={vmask:08x?}"
                        );
                        covered[want] = true;
                    }
                }
            }
        }
        // Liveness: every reach the ladder can produce was actually reached,
        // so a mutation to any rung has a cell that fails.
        assert!(
            covered[2] && covered[3] && covered[4] && covered[7],
            "reaches exercised: {covered:?}"
        );
    }

    /// The property the guard policy depends on: the run's reach never exceeds
    /// the rows a transform of that level leaves below the edge inside a
    /// superblock row. Level 2 needs 16 rows, level 1 needs 8, level 0 needs 4.
    #[test]
    fn run_reach_fits_the_transform_that_selected_it() {
        for (wd, rows_below) in [(4, 4usize), (6, 4), (8, 8), (16, 16)] {
            assert!(
                lf_reach(wd) as usize <= rows_below,
                "wd {wd} reaches {} into {rows_below} rows",
                lf_reach(wd)
            );
        }
    }
}

/// Per-variant bit-identity gate for the aarch64 NEON deblocking kernels.
///
/// The kernels and the scalar `loop_filter` are driven over the SAME scratch
/// rectangle with the same layout, so a mismatch here is an arithmetic
/// divergence and nothing else. Every (direction x width x bit depth x fused
/// group count) cell is covered, and the pixel generator sweeps noise
/// amplitude so each cell exercises the no-filter, narrow, hev, flat and wide
/// branches rather than only whichever one random data happens to hit.
#[cfg(all(test, target_arch = "aarch64"))]
mod neon_parity {
    use super::*;
    use crate::include::common::bitdepth::BitDepth8;
    use crate::include::common::bitdepth::BitDepth16;

    /// xorshift; a fixed generator keeps a failure reproducible from its seed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    /// Which branch of the ladder a lane took, so the test can prove it
    /// actually reached each one instead of silently testing only `!fm`.
    #[derive(Default, Debug)]
    struct Coverage {
        nofilter: u32,
        narrow_hev: u32,
        narrow_flat: u32,
        flat6: u32,
        flat8: u32,
        wide: u32,
    }

    fn classify<BD: BitDepth>(
        buf: &[BD::Pixel; LF_BLOCK_LEN],
        base: usize,
        stridea: isize,
        strideb: isize,
        lane: usize,
        e: u8,
        i: u8,
        h: u8,
        wd: c_int,
        bd: BD,
        cov: &mut Coverage,
    ) {
        let bd8 = bd.bitdepth() - 8;
        let [f, e, i, h] = [1, e, i, h].map(|n| (n as i32) << bd8);
        let at = |k: isize| -> i32 {
            buf[base.wrapping_add_signed(stridea * lane as isize + strideb * k)
                & (LF_BLOCK_LEN - 1)]
                .as_::<i32>()
        };
        let (p6, p5, p4, p3, p2, p1, p0) = (at(-7), at(-6), at(-5), at(-4), at(-3), at(-2), at(-1));
        let (q0, q1, q2, q3, q4, q5, q6) = (at(0), at(1), at(2), at(3), at(4), at(5), at(6));
        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;
        if wd > 4 {
            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;
            if wd > 6 {
                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }
        if !fm {
            cov.nofilter += 1;
            return;
        }
        let mut flat8in = false;
        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f
                && (p1 - p0).abs() <= f
                && (q1 - q0).abs() <= f
                && (q2 - q0).abs() <= f;
        }
        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f && (q3 - q0).abs() <= f;
        }
        let flat8out = wd >= 16
            && (p6 - p0).abs() <= f
            && (p5 - p0).abs() <= f
            && (p4 - p0).abs() <= f
            && (q4 - q0).abs() <= f
            && (q5 - q0).abs() <= f
            && (q6 - q0).abs() <= f;
        if wd >= 16 && flat8out && flat8in {
            cov.wide += 1;
        } else if wd >= 8 && flat8in {
            cov.flat8 += 1;
        } else if wd == 6 && flat8in {
            cov.flat6 += 1;
        } else if (p1 - p0).abs() > h || (q1 - q0).abs() > h {
            cov.narrow_hev += 1;
        } else {
            cov.narrow_flat += 1;
        }
    }

    fn one_cell<BD: BitDepth>(bd: BD, wd: c_int, is_v: bool, groups: usize, cov: &mut Coverage) {
        let bd_max: u16 = bd.bitdepth_max().into();
        let reach = lf_reach(wd) as usize;
        let (w, h, stridea, strideb, base) = if is_v {
            (4 * groups, 2 * reach, 1isize, LF_BW as isize, reach * LF_BW)
        } else {
            (2 * reach, 4 * groups, LF_BW as isize, 1isize, reach)
        };

        let mut rng = Rng(0x9E37_79B9_7F4A_7C15
            ^ ((wd as u64) << 40)
            ^ ((groups as u64) << 20)
            ^ (is_v as u64)
            ^ ((bd_max as u64) << 8));

        for trial in 0..3000u32 {
            // Sweep the noise amplitude so flat/wide branches are reachable:
            // a plateau plus +-amp noise, with amp from 0 (perfectly flat) up
            // past the largest threshold.
            let amp = 1u32 + (trial % 24) * (bd_max as u32 + 1) / 24;
            let plateau = rng.below(bd_max as u32 + 1);
            let mut buf = [BD::Pixel::from(0u8); LF_BLOCK_LEN];
            for (idx, px) in buf.iter_mut().enumerate() {
                // Every 8th trial is fully random, to hit the mask edges the
                // plateau generator never reaches.
                let _ = idx;
                let v: u16 = if trial % 8 == 7 {
                    rng.below(bd_max as u32 + 1) as u16
                } else {
                    (plateau as i64 + rng.below(2 * amp + 1) as i64 - amp as i64)
                        .clamp(0, bd_max as i64) as u16
                };
                *px = v.as_::<BD::Pixel>();
            }

            let mut params = [(0u8, 0u8, 0u8, wd); LF_BATCH_MAX];
            for p in params.iter_mut().take(groups) {
                // The LUT's `e` runs to 255 and `i`/`h` to 63/15 in practice,
                // but nothing downstream relies on that, so sweep wider.
                *p = (
                    rng.below(256) as u8,
                    rng.below(64) as u8,
                    rng.below(16) as u8,
                    wd,
                );
            }

            for lane in 0..4 * groups {
                let (e, i, h, _) = params[lane / 4];
                classify::<BD>(&buf, base, stridea, strideb, lane, e, i, h, wd, bd, cov);
            }

            let mut simd = buf;
            use zerocopy::IntoBytes as _;
            let ok = crate::src::safe_simd::loopfilter_arm::lf_compact_run_neon(
                BD::BPC,
                simd.as_mut_bytes(),
                base,
                is_v,
                4 * groups,
                &params[..groups],
                wd,
                bd.bitdepth() - 8,
                bd_max,
            );
            assert!(ok, "kernel refused wd={wd} is_v={is_v} groups={groups}");

            let mut reference = buf;
            for (g, &(e, i, hh, _)) in params[..groups].iter().enumerate() {
                let mut taps = CompactTaps {
                    len: (h - 1) * LF_BW + w,
                    buf: &mut reference,
                    base: base.wrapping_add_signed(4 * g as isize * stridea),
                    stridea,
                    strideb,
                };
                loop_filter::<BD, _>(&mut taps, e, i, hh, wd, bd);
            }

            for row in 0..h {
                for col in 0..w {
                    let idx = row * LF_BW + col;
                    assert_eq!(
                        simd[idx].as_::<i32>(),
                        reference[idx].as_::<i32>(),
                        "bd={} wd={wd} is_v={is_v} groups={groups} trial={trial} \
                         row={row} col={col} params={:?}",
                        bd.bitdepth(),
                        &params[..groups],
                    );
                }
            }
        }
    }

    fn sweep<BD: BitDepth>(bd: BD, widths: &[c_int]) {
        // `archmage::testing`'s token switch is process-global and another
        // test flips it; without this the kernel gets refused a summon
        // mid-sweep. See `safe_simd::token_test_lock`.
        let _guard = crate::src::safe_simd::token_test_lock();
        for &wd in widths {
            for &is_v in &[false, true] {
                for groups in 1..=LF_BATCH_MAX {
                    let mut cov = Coverage::default();
                    one_cell::<BD>(bd, wd, is_v, groups, &mut cov);
                    // Liveness: a cell that never filters proves nothing.
                    assert!(
                        cov.narrow_hev > 0 && cov.narrow_flat > 0,
                        "wd={wd} is_v={is_v} groups={groups} never took a narrow branch: {cov:?}"
                    );
                    match wd {
                        6 => assert!(cov.flat6 > 0, "wd=6 never flat: {cov:?}"),
                        8 => assert!(cov.flat8 > 0, "wd=8 never flat: {cov:?}"),
                        16 => assert!(
                            cov.wide > 0 && cov.flat8 > 0,
                            "wd=16 missed a flat branch: {cov:?}"
                        ),
                        _ => {}
                    }
                    assert!(cov.nofilter > 0, "wd={wd} never skipped a lane: {cov:?}");
                }
            }
        }
    }

    // Luma reaches wd 4/8/16, chroma 4/6 — but the kernels are indexed by `wd`
    // alone, so every width is swept against both.
    const ALL_WD: [c_int; 4] = [4, 6, 8, 16];

    /// `close`'s vector diff scan must answer EXACTLY the scalar
    /// `position`/`rposition` pair, including ignoring the pad columns beyond
    /// `w` — a looser answer would widen a mutable guard onto a merely-read
    /// tap, which is the zenavif#30 conflict.
    fn diff_span_cell<BD: BitDepth>(bd: BD) {
        use zerocopy::IntoBytes as _;
        let _guard = crate::src::safe_simd::token_test_lock();
        let bd_max: u16 = bd.bitdepth_max().into();
        let mut rng = Rng(0xDEAD_BEEF_1234_5678 ^ bd_max as u64);
        let mut fired = 0u32;
        let mut empty = 0u32;
        for _ in 0..20000u32 {
            let mut work = [BD::Pixel::from(0u8); LF_BLOCK_LEN];
            let mut pristine = [BD::Pixel::from(0u8); LF_BLOCK_LEN];
            for (a, b) in work.iter_mut().zip(pristine.iter_mut()) {
                let v = rng.below(bd_max as u32 + 1) as u16;
                *b = v.as_::<BD::Pixel>();
                // Mostly equal, so the "row untouched" early-out is exercised.
                *a = if rng.below(4) == 0 {
                    (rng.below(bd_max as u32 + 1) as u16).as_::<BD::Pixel>()
                } else {
                    v.as_::<BD::Pixel>()
                };
            }
            let w = 1 + rng.below(LF_BW as u32) as usize;
            let row = rng.below(LF_BW as u32) as usize;

            let a = &work[row * LF_BW..][..w];
            let b = &pristine[row * LF_BW..][..w];
            let want = a
                .iter()
                .zip(b)
                .position(|(x, y)| x != y)
                .map(|first| (first, a.iter().zip(b).rposition(|(x, y)| x != y).unwrap()));
            let got = crate::src::safe_simd::loopfilter_arm::lf_diff_span(
                BD::BPC,
                work.as_bytes(),
                pristine.as_bytes(),
                row,
                w,
            )
            .expect("aarch64 always has NEON");
            assert_eq!(got, want, "bd={} row={row} w={w}", bd.bitdepth());
            if want.is_some() {
                fired += 1;
            } else {
                empty += 1;
            }
        }
        assert!(
            fired > 100 && empty > 100,
            "diff span not exercised both ways"
        );
    }

    #[test]
    fn neon_diff_span_matches_scalar() {
        diff_span_cell::<BitDepth8>(BitDepth8::new(()));
        diff_span_cell::<BitDepth16>(BitDepth16::new(1023));
        diff_span_cell::<BitDepth16>(BitDepth16::new(4095));
    }

    #[test]
    fn neon_matches_scalar_8bpc() {
        sweep::<BitDepth8>(BitDepth8::new(()), &ALL_WD);
    }

    #[test]
    fn neon_matches_scalar_10bpc() {
        sweep::<BitDepth16>(BitDepth16::new(1023), &ALL_WD);
    }

    #[test]
    fn neon_matches_scalar_12bpc() {
        sweep::<BitDepth16>(BitDepth16::new(4095), &ALL_WD);
    }
}
