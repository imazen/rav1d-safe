#![cfg(all(
    test,
    target_arch = "x86_64",
    not(feature = "asm"),
    not(feature = "c-ffi")
))]
//! Source-window parity for backwards row walks, before MC optimization.
use super::*;
use crate::include::common::bitdepth::{BitDepth, BitDepth8};
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::src::levels::Filter2d;
use crate::src::safe_simd::aligned_plane;
const fn wrap_len(n: usize) -> usize {
    n.div_ceil(64) * 64
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
}

const SEED: u64 = 0x8b5a_cdef_91a3_7257;

const FILTERS: &[Filter2d] = &[
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
];

/// Every source row must stay within its complete bounded window; a suffix
/// starting at row zero cannot represent a backwards walk.
#[test]
#[cfg(not(feature = "c-ffi"))]
fn reversed_source_rows_match_scalar_for_all_8tap_filters() {
    use crate::include::dav1d::picture::Rav1dPictureDataComponentInner;
    use crate::src::with_offset::WithOffset;

    let _lock = crate::src::safe_simd::token_test_lock();
    let token = crate::src::cpu::summon_avx2().expect("native x86 MC requires AVX2");
    let token512 = crate::src::cpu::summon_avx512();
    let bd = BitDepth8::new(());
    const STRIDE: usize = 192;
    const ROWS: usize = 144;
    let mut rng = Rng(SEED);
    let pixels: Vec<u8> = (0..STRIDE * ROWS)
        .map(|i| match i % 19 {
            0 => 0,
            1 => 255,
            _ => rng.byte(),
        })
        .collect();
    let source_picture = Rav1dPictureDataComponent::from_parts(
        Rav1dPictureDataComponentInner::from_slice_copy(&pixels),
        -(STRIDE as isize),
    );
    let source = WithOffset {
        data: &source_picture,
        offset: 139 * STRIDE + 8,
    };
    let _outside = source_picture.index_mut::<BitDepth8>(143 * STRIDE);
    let mut cases = 0;
    for (w, h) in [(2, 2), (7, 9), (17, 5), (32, 16), (128, 128)] {
        let dst_stride = w + 3;
        let dst_len = wrap_len(dst_stride * h);
        for &filter in FILTERS.iter().filter(|&&f| f != Filter2d::Bilinear) {
            let (hf, vf) = filter.hv();
            for mx in 0..16 {
                for my in 0..16 {
                    let label = format!("negative f{} {w}x{h} ({mx},{my})", filter as u8);
                    let (guard, base) =
                        crate::src::safe_simd::mc::reference::filter_guard::<BitDepth8>(
                            source, filter, w as i32, h as i32, mx, my,
                        );
                    let mut stage = aligned_plane(&vec![0u8; dst_len]);
                    let expected_picture =
                        Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut stage, dst_stride);
                    crate::src::mc::put_8tap_rust::<BitDepth8>(
                        expected_picture.with_offset::<BitDepth8>(),
                        source,
                        w,
                        h,
                        mx as usize,
                        my as usize,
                        (hf, vf),
                        bd,
                    );
                    let mut expected_put = vec![0u8; dst_len];
                    expected_picture.copy_pixels_to::<BitDepth8>(&mut expected_put);
                    let mut expected_prep = vec![0i16; w * h];
                    crate::src::mc::prep_8tap_rust::<BitDepth8>(
                        &mut expected_prep,
                        source,
                        w,
                        h,
                        mx as usize,
                        my as usize,
                        (hf, vf),
                        bd,
                    );
                    let mut put = vec![0u8; dst_len];
                    put_8tap_8bpc_avx2_impl_testable(
                        token,
                        &mut put,
                        dst_stride as isize,
                        &guard,
                        base,
                        -(STRIDE as isize),
                        w as i32,
                        h as i32,
                        mx,
                        my,
                        hf,
                        vf,
                    );
                    assert_eq!(put, expected_put, "AVX2 put: {label}");
                    let mut prep = vec![0i16; w * h];
                    prep_8tap_8bpc_avx2_impl_testable(
                        token,
                        &mut prep,
                        &guard,
                        base,
                        -(STRIDE as isize),
                        w as i32,
                        h as i32,
                        mx,
                        my,
                        hf,
                        vf,
                    );
                    assert_eq!(prep, expected_prep, "AVX2 prep: {label}");
                    if let Some(token512) = token512 {
                        put.fill(0);
                        put_8tap_8bpc_avx512_impl_inner(
                            token512,
                            &mut put,
                            dst_stride as isize,
                            &guard,
                            base,
                            -(STRIDE as isize),
                            w as i32,
                            h as i32,
                            mx,
                            my,
                            hf,
                            vf,
                        );
                        assert_eq!(put, expected_put, "AVX-512 put: {label}");
                        prep.fill(0);
                        prep_8tap_8bpc_avx512_impl_inner(
                            token512,
                            &mut prep,
                            &guard,
                            base,
                            -(STRIDE as isize),
                            w as i32,
                            h as i32,
                            mx,
                            my,
                            hf,
                            vf,
                        );
                        assert_eq!(prep, expected_prep, "AVX-512 prep: {label}");
                    }
                    let mut dispatch_stage = aligned_plane(&vec![0u8; dst_len]);
                    let dispatch_picture = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(
                        &mut dispatch_stage,
                        dst_stride,
                    );
                    assert!(crate::src::safe_simd::mc::mc_put_dispatch::<BitDepth8>(
                        filter,
                        dispatch_picture.with_offset::<BitDepth8>(),
                        source,
                        w as i32,
                        h as i32,
                        mx,
                        my,
                        bd,
                    ));
                    let mut dispatch_put = vec![0u8; dst_len];
                    dispatch_picture.copy_pixels_to::<BitDepth8>(&mut dispatch_put);
                    assert_eq!(dispatch_put, expected_put, "dispatch put: {label}");
                    let mut dispatch_prep = vec![0i16; w * h];
                    assert!(crate::src::safe_simd::mc::mct_prep_dispatch::<BitDepth8>(
                        filter,
                        &mut dispatch_prep,
                        source,
                        w as i32,
                        h as i32,
                        mx,
                        my,
                        bd,
                    ));
                    assert_eq!(dispatch_prep, expected_prep, "dispatch prep: {label}");
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 5 * 9 * 256);
}

#[arcane]
fn put_8tap_8bpc_avx2_impl_testable(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    put_8tap_8bpc_avx2_impl_inner(
        _token, dst, dst_stride, src, src_base, src_stride, w, h, mx, my, h_filter, v_filter,
    );
}

/// Drives `prep_8tap_8bpc_avx2_impl_inner` (rite) from test code.
#[cfg(all(test, target_arch = "x86_64"))]
#[arcane]
fn prep_8tap_8bpc_avx2_impl_testable(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    prep_8tap_8bpc_avx2_impl_inner(
        _token, tmp, src, src_base, src_stride, w, h, mx, my, h_filter, v_filter,
    );
}
