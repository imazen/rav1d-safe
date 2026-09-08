//! Bounded source reservations for the x86 interpolation kernels.
//!
//! These replace whole-component read guards on reference pictures / private
//! edge-emulation buffers. Unlike reconstruction writes, they may cover row
//! gaps: the complete contiguous slice is registered with DisjointMut, so any
//! concurrent write in a gap is still rejected. This is a narrowing of the
//! existing whole-component exemption in `note_pic_extent`, not permission to
//! widen a reconstruction guard. No mutable reference is created here.

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::{PicOffset, Rav1dPictureDataComponentInner};
use crate::src::disjoint_mut::DisjointImmutGuard;
use crate::src::levels::Filter2d;
use crate::src::strided::Strided as _;

type SourceGuard<'a, BD> =
    DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [<BD as BitDepth>::Pixel]>;

/// Reserve the hull of the output block plus its horizontal / vertical taps.
/// Bounds are in pixels; the returned base still names the unpadded origin.
#[inline]
#[cfg_attr(
    any(debug_assertions, feature = "__probe_sites", feature = "__probe_usage"),
    track_caller
)]
pub(super) fn read_guard<BD: BitDepth>(
    src: PicOffset<'_>,
    w: usize,
    h: usize,
    padding: [(usize, usize); 2],
) -> (SourceGuard<'_, BD>, usize) {
    let [(left, right), (top, bottom)] = padding;
    // AV1 interpolation is at most 128x128 with eight taps per axis. Keep
    // this new exemption bounded independently of allocation size and stride.
    assert!((1..=128).contains(&w) && (1..=128).contains(&h));
    assert!(left <= 3 && right <= 4 && top <= 3 && bottom <= 4);
    let stride = src.data.pixel_stride::<BD>();
    let first = src
        .offset
        .checked_add_signed(
            stride
                .checked_mul(-(top as isize))
                .expect("MC top offset overflow"),
        )
        .and_then(|x| x.checked_sub(left))
        .expect("MC window starts outside picture");
    let rows = h + top + bottom;
    let width = w + left + right;
    let span = (rows - 1)
        .checked_mul(stride.unsigned_abs())
        .expect("MC row span overflow");
    let start = if stride >= 0 {
        first
    } else {
        first
            .checked_sub(span)
            .expect("MC window starts below negative-stride picture")
    };
    let end = start
        .checked_add(span)
        .and_then(|x| x.checked_add(width))
        .expect("MC window end overflow");
    // Register EXACTLY the full Rust slice, including inter-row gaps. Going
    // through dm() deliberately uses this bounded reference-read exemption;
    // the ordinary partial reconstruction extent ceiling remains unchanged.
    let guard = src.data.dm().slice_as::<_, BD::Pixel>(start..end);
    let ps = core::mem::size_of::<BD::Pixel>();
    guard.probe_declare_rows(first * ps, width * ps, rows, stride * ps as isize);
    (guard, src.offset - start)
}

#[inline]
#[cfg_attr(
    any(debug_assertions, feature = "__probe_sites", feature = "__probe_usage"),
    track_caller
)]
pub(super) fn filter_guard<BD: BitDepth>(
    src: PicOffset<'_>,
    filter: Filter2d,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) -> (SourceGuard<'_, BD>, usize) {
    let pad = |phase| {
        if phase == 0 {
            (0, 0)
        } else if filter == Filter2d::Bilinear {
            (0, 1)
        } else {
            (3, 4)
        }
    };
    read_guard::<BD>(src, w as usize, h as usize, [pad(mx), pad(my)])
}

#[cfg(all(test, not(feature = "c-ffi")))]
mod tests {
    use super::*;
    use crate::include::common::bitdepth::{BitDepth8, BitDepth16};
    use crate::include::dav1d::picture::Rav1dPictureDataComponent;
    use crate::src::with_offset::WithOffset;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use zerocopy::IntoBytes;

    #[test]
    fn reference_window_reserves_its_full_hull_including_gaps() {
        for stride in [64isize, -64] {
            let inner = Rav1dPictureDataComponentInner::from_slice_copy(&[0; 64 * 32]);
            let pic = Rav1dPictureDataComponent::from_parts(inner, stride);
            let src = WithOffset {
                data: &pic,
                offset: 16 * 64 + 16,
            };
            let (guard, base) = read_guard::<BitDepth8>(src, 8, 8, [(3, 4), (3, 4)]);
            let start = src.offset - base;
            let end = start + guard.len();
            assert_eq!(guard.len(), 14 * 64 + 15);
            for y in -3isize..12 {
                for x in -3isize..12 {
                    let index = (src.offset as isize + y * stride + x) as usize;
                    assert!(start <= index && index < end);
                }
            }
            #[cfg(not(feature = "unchecked"))]
            for probe in [start, start + 32, end - 1] {
                assert!(
                    catch_unwind(AssertUnwindSafe(|| pic.index_mut::<BitDepth8>(probe))).is_err()
                );
            }
            // Adjacent storage is available while the window stays borrowed.
            *pic.index_mut::<BitDepth8>(start - 1) = 1;
            *pic.index_mut::<BitDepth8>(end) = 2;
            drop(guard);
            *pic.index_mut::<BitDepth8>(start) = 3;
        }
    }

    #[test]
    fn warp_reference_windows_cover_both_strides_and_all_depths() {
        let _lock = archmage::testing::lock_token_testing();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            return;
        };
        let mut cases = 0;
        for bits in [8, 10, 12] {
            let ps = if bits == 8 { 1 } else { 2 };
            let pixels: Vec<u16> = (0..64 * 32)
                .map(|i| ((i * 73 + i / 7) & ((1 << bits) - 1)) as u16)
                .collect();
            let bytes: Vec<u8> = if bits == 8 {
                pixels.iter().map(|&x| x as u8).collect()
            } else {
                pixels.as_bytes().to_vec()
            };
            for sign in [1, -1] {
                let stride = sign * 64 * ps as isize;
                let pic = Rav1dPictureDataComponent::from_parts(
                    Rav1dPictureDataComponentInner::from_slice_copy(&bytes),
                    stride,
                );
                let src = WithOffset {
                    data: &pic,
                    offset: 16 * 64 + 16,
                };
                for abcd in [[0; 4], [64, -64, 32, -32]] {
                    for mx in [-1024, 0, 1024] {
                        for my in [-1024, 0, 1024] {
                            let mut put = [vec![0u8; 64 * ps], vec![0; 64 * ps]];
                            let mut prep = [[0i16; 64]; 2];
                            if bits == 8 {
                                let (tight, tb) =
                                    read_guard::<BitDepth8>(src, 8, 8, [(3, 4), (3, 4)]);
                                let (full, fb) = src.full_guard::<BitDepth8>();
                                for (i, (data, base)) in
                                    [(&*tight, tb), (&*full, fb)].into_iter().enumerate()
                                {
                                    super::super::warp_affine_8x8_8bpc_avx2(
                                        token,
                                        &mut put[i],
                                        8,
                                        data,
                                        base,
                                        stride,
                                        &abcd,
                                        mx,
                                        my,
                                    );
                                    super::super::warp_affine_8x8t_8bpc_avx2(
                                        token,
                                        &mut prep[i],
                                        8,
                                        data,
                                        base,
                                        stride,
                                        &abcd,
                                        mx,
                                        my,
                                    );
                                }
                            } else {
                                let (tight, tb) =
                                    read_guard::<BitDepth16>(src, 8, 8, [(3, 4), (3, 4)]);
                                let (full, fb) = src.full_guard::<BitDepth16>();
                                for (i, (data, base)) in
                                    [(tight.as_bytes(), tb), (full.as_bytes(), fb)]
                                        .into_iter()
                                        .enumerate()
                                {
                                    let intermediate = if bits == 12 { 2 } else { 4 };
                                    super::super::warp_affine_8x8_16bpc_avx2(
                                        token,
                                        &mut put[i],
                                        16,
                                        data,
                                        base * ps,
                                        stride,
                                        &abcd,
                                        mx,
                                        my,
                                        intermediate,
                                        (1 << bits) - 1,
                                    );
                                    super::super::warp_affine_8x8t_16bpc_avx2(
                                        token,
                                        &mut prep[i],
                                        8,
                                        data,
                                        base * ps,
                                        stride,
                                        &abcd,
                                        mx,
                                        my,
                                        intermediate,
                                    );
                                }
                            }
                            assert_eq!(put[0], put[1]);
                            assert_eq!(prep[0], prep[1]);
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 108);
    }

    /// Compare each dispatch on its exact tap window with the same dispatch on
    /// a generously padded source. Any SIMD over-read must fail the tight arm.
    #[test]
    fn mc_reference_windows_cover_all_filters_phases_widths_and_depths() {
        let _lock = archmage::testing::lock_token_testing();
        if crate::src::cpu::summon_avx2().is_none() {
            return;
        }
        let mut modes = std::collections::BTreeSet::new();
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::WarnStderr,
            |_| {
                let Some(token) = crate::src::cpu::summon_avx2() else {
                    return;
                };
                let wide = crate::src::cpu::summon_avx512().is_some();
                if !modes.insert(wide) {
                    return;
                }
                let mut samples = 0usize;
                for (w, h) in [
                    (2, 2),
                    (4, 4),
                    (8, 16),
                    (16, 8),
                    (32, 32),
                    (64, 4),
                    (128, 128),
                ] {
                    for filter in [
                        Filter2d::Regular8Tap,
                        Filter2d::RegularSmooth8Tap,
                        Filter2d::RegularSharp8Tap,
                        Filter2d::SharpRegular8Tap,
                        Filter2d::SharpSmooth8Tap,
                        Filter2d::Sharp8Tap,
                        Filter2d::SmoothRegular8Tap,
                        Filter2d::Smooth8Tap,
                        Filter2d::SmoothSharp8Tap,
                        Filter2d::Bilinear,
                    ] {
                        for mx in 0..16 {
                            for my in 0..16 {
                                check_filter::<BitDepth8>(
                                    token,
                                    filter,
                                    w,
                                    h,
                                    mx,
                                    my,
                                    BitDepth8::new(()),
                                );
                                for max in [1023, 4095] {
                                    check_filter::<BitDepth16>(
                                        token,
                                        filter,
                                        w,
                                        h,
                                        mx,
                                        my,
                                        BitDepth16::new(max),
                                    );
                                }
                                samples += 3;
                            }
                        }
                    }
                }
                assert_eq!(samples, 7 * 10 * 16 * 16 * 3);
            },
        );
        assert!(report.permutations_run > 0 && !modes.is_empty());
        eprintln!("MC exact-window modes (AVX-512 enabled): {modes:?}; 53760 cases per mode");
    }

    fn check_filter<BD: BitDepth>(
        token: archmage::Desktop64,
        filter: Filter2d,
        w: i32,
        h: i32,
        mx: i32,
        my: i32,
        bd: BD,
    ) {
        use crate::include::common::bitdepth::{AsPrimitive, BPC};
        let ps = core::mem::size_of::<BD::Pixel>();
        const STRIDE: usize = 192;
        let mut pixels: Vec<BD::Pixel> = (0..STRIDE * 144)
            .map(|i| (((i * 73 + i / 7) as i32) & bd.bitdepth_max().as_::<i32>()).as_())
            .collect();
        let pic = Rav1dPictureDataComponent::wrap_buf::<BD>(&mut pixels, STRIDE);
        let src = WithOffset {
            data: &pic,
            offset: 4 * STRIDE + 8,
        };
        let (tight, tb) = filter_guard::<BD>(src, filter, w, h, mx, my);
        let (full, fb) = src.full_guard::<BD>();
        let mut put = [
            vec![0u8; w as usize * h as usize * ps],
            vec![0; w as usize * h as usize * ps],
        ];
        let mut prep = [
            vec![0i16; w as usize * h as usize],
            vec![0; w as usize * h as usize],
        ];
        for (i, (data, base)) in [(tight.as_bytes(), tb), (full.as_bytes(), fb)]
            .into_iter()
            .enumerate()
        {
            let (hf, vf) = filter.hv();
            match BD::BPC {
                BPC::BPC8 => {
                    if filter == Filter2d::Bilinear {
                        super::super::put_bilin_8bpc_dispatch_inner(
                            token,
                            &mut put[i],
                            w as isize,
                            &data[base..],
                            STRIDE as isize,
                            w,
                            h,
                            mx,
                            my,
                        );
                        super::super::prep_bilin_8bpc_dispatch_inner(
                            token,
                            &mut prep[i],
                            &data[base..],
                            STRIDE as isize,
                            w,
                            h,
                            mx,
                            my,
                        );
                    } else {
                        super::super::put_8tap_8bpc_dispatch_inner(
                            token,
                            &mut put[i],
                            w as isize,
                            data,
                            base,
                            STRIDE as isize,
                            w,
                            h,
                            mx,
                            my,
                            hf,
                            vf,
                        );
                        super::super::prep_8tap_8bpc_dispatch_inner(
                            token,
                            &mut prep[i],
                            data,
                            base,
                            STRIDE as isize,
                            w,
                            h,
                            mx,
                            my,
                            hf,
                            vf,
                        );
                    }
                }
                BPC::BPC16 => {
                    let data = zerocopy::Ref::<_, [u16]>::new_slice(data)
                        .unwrap()
                        .into_slice();
                    let dst = zerocopy::Ref::<_, [u16]>::new_slice(&mut put[i][..])
                        .unwrap()
                        .into_mut_slice();
                    let max = bd.into_c();
                    if filter == Filter2d::Bilinear {
                        if let Some(wide) = crate::src::cpu::summon_avx512() {
                            super::super::put_bilin_16bpc_avx512_impl_inner(
                                wide,
                                dst,
                                w as isize,
                                &data[base..],
                                STRIDE as isize,
                                w,
                                h,
                                mx,
                                my,
                                max,
                            );
                            super::super::prep_bilin_16bpc_avx512_impl_inner(
                                wide,
                                &mut prep[i],
                                &data[base..],
                                STRIDE as isize,
                                w,
                                h,
                                mx,
                                my,
                                max,
                            );
                        } else {
                            super::super::put_bilin_16bpc_avx2_impl_inner_safe(
                                token,
                                dst,
                                w as isize,
                                &data[base..],
                                STRIDE as isize,
                                w,
                                h,
                                mx,
                                my,
                                max,
                            );
                            super::super::prep_bilin_16bpc_avx2_impl_inner_safe(
                                token,
                                &mut prep[i],
                                &data[base..],
                                STRIDE as isize,
                                w,
                                h,
                                mx,
                                my,
                                max,
                            );
                        }
                    } else {
                        super::super::put_8tap_16bpc_dispatch_inner(
                            token,
                            dst,
                            2 * w as isize,
                            data,
                            base,
                            2 * STRIDE as isize,
                            w,
                            h,
                            mx,
                            my,
                            max,
                            hf,
                            vf,
                        );
                        super::super::prep_8tap_16bpc_dispatch_inner(
                            token,
                            &mut prep[i],
                            data,
                            base,
                            2 * STRIDE as isize,
                            w,
                            h,
                            mx,
                            my,
                            max,
                            hf,
                            vf,
                        );
                    }
                }
            }
        }
        let filter = filter as u8;
        assert_eq!(put[0], put[1], "put: {w}x{h} filter={filter} ({mx},{my})");
        assert_eq!(prep[0], prep[1], "prep: {w}x{h} {filter} ({mx},{my})");
    }
}
