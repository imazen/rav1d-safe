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
struct CompactTaps<'a, BD: BitDepth> {
    buf: &'a mut [BD::Pixel],
    /// Index of `(idx, k) = (0, 0)` within `buf`.
    base: usize,
    stridea: isize,
    strideb: isize,
}

impl<BD: BitDepth> CompactTaps<'_, BD> {
    #[inline(always)]
    fn at(&self, idx: isize, k: isize) -> usize {
        self.base
            .wrapping_add_signed(self.stridea * idx + self.strideb * k)
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
struct LfBlock<'a, BD: BitDepth> {
    /// Top-left of the rectangle in picture coordinates.
    origin: PicOffset<'a>,
    stride: isize,
    w: usize,
    h: usize,
    buf: [BD::Pixel; LF_BLOCK_MAX],
    pristine: [BD::Pixel; LF_BLOCK_MAX],
    /// `buf` index of `(idx, k) = (0, 0)`.
    base: usize,
    stridea: isize,
    strideb: isize,
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

impl<'a, BD: BitDepth> LfBlock<'a, BD> {
    /// `None` when the rectangle would leave the plane; the caller then falls
    /// back to [`DirectTaps`], which is what this replaced.
    #[inline]
    fn open(dst: PicOffset<'a>, is_v: bool, stride: isize, wd: c_int) -> Option<Self> {
        let reach = lf_reach(wd);
        let (w, h, origin_delta, stridea, strideb, base) = if is_v {
            // Taps run down the picture; the four columns run along x.
            let w = 4usize;
            (
                w,
                2 * reach as usize,
                -reach * stride,
                1isize,
                w as isize,
                reach as usize * w,
            )
        } else {
            // Taps run along x; the four columns run down the picture.
            let w = 2 * reach as usize;
            (w, 4usize, -reach, w as isize, 1isize, reach as usize)
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
        let mut buf = [BD::Pixel::from(0u8); LF_BLOCK_MAX];
        for row in 0..h {
            let off = origin.offset.wrapping_add_signed(row as isize * stride);
            let guard = PicOffset {
                data: origin.data,
                offset: off,
            }
            .slice::<BD>(w);
            buf[row * w..][..w].copy_from_slice(&guard);
        }
        Some(Self {
            origin,
            stride,
            w,
            h,
            pristine: buf,
            buf,
            base,
            stridea,
            strideb,
        })
    }

    #[inline]
    fn taps(&mut self) -> CompactTaps<'_, BD> {
        CompactTaps {
            buf: &mut self.buf[..self.w * self.h],
            base: self.base,
            stridea: self.stridea,
            strideb: self.strideb,
        }
    }

    /// Write back only the pixels that changed, one row at a time.
    #[inline]
    fn close(self) {
        for row in 0..self.h {
            let work = &self.buf[row * self.w..][..self.w];
            let orig = &self.pristine[row * self.w..][..self.w];
            let Some(first) = work.iter().zip(orig).position(|(a, b)| a != b) else {
                continue; // row untouched: no write, no mutable guard
            };
            let last = work
                .iter()
                .zip(orig)
                .rposition(|(a, b)| a != b)
                .expect("a differing pixel exists, so rposition finds one");
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
    mut dst: PicOffset,
    vmask: &[u32; 3],
    mut lvl: WithOffset<&[AtomicU8]>,
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
    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        'block: {
            if vm & xy == 0 {
                break 'block;
            }
            let l = lvl.data[lvl.offset].load(Relaxed);
            let l = if l != 0 {
                l
            } else {
                let lvl = lvl - 4 * b4_strideb;
                lvl.data[lvl.offset].load(Relaxed)
            };
            if l == 0 {
                break 'block;
            }
            let h = l >> 4;
            let e = lut.e[l as usize];
            let i = lut.i[l as usize];
            let idx = match yuv {
                YUV::Y => {
                    let idx = if vmask[2] & xy != 0 {
                        2
                    } else {
                        (vmask[1] & xy != 0) as c_int
                    };
                    4 << idx
                }
                YUV::UV => {
                    let idx = (vmask[1] & xy != 0) as c_int;
                    4 + 2 * idx
                }
            };
            // One guard per picture row over this call's tap rectangle, rather
            // than one per tap. See `LfBlock`.
            match LfBlock::<BD>::open(dst, is_v, stride, idx) {
                Some(mut block) => {
                    loop_filter::<BD, _>(&mut block.taps(), e, i, h, idx, bd);
                    block.close();
                }
                None => {
                    let mut taps = DirectTaps {
                        dst,
                        stridea,
                        strideb,
                    };
                    loop_filter::<BD, _>(&mut taps, e, i, h, idx, bd);
                }
            }
        }
        xy <<= 1;
        dst += 4 * stridea;
        lvl += 4 * b4_stridea;
    }
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
