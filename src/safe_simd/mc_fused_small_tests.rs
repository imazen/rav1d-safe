//! Independent two-dimensional arithmetic with a tight 7x7 source footprint.

use super::*;

#[arcane]
fn fused_rows(
    token: Desktop64,
    source: &[u8],
    origin: usize,
    stride: isize,
    horizontal: &[i8; 8],
    vertical: &[i8; 8],
) -> ([[i32; 4]; 4], [[i32; 4]; 4]) {
    let put = fused_hv_4x4_8bpc::<10>(token, source, origin, stride, horizontal, vertical);
    let prep = fused_hv_4x4_8bpc::<6>(token, source, origin, stride, horizontal, vertical);
    let mut put_lanes = [[0; 4]; 4];
    let mut prep_lanes = [[0; 4]; 4];
    for y in 0..4 {
        storeu_128!(&mut put_lanes[y], put[y]);
        storeu_128!(&mut prep_lanes[y], prep[y]);
    }
    (put_lanes, prep_lanes)
}

fn scalar(
    source: &[u8],
    origin: usize,
    stride: isize,
    horizontal: &[i8; 8],
    vertical: &[i8; 8],
) -> ([[i32; 4]; 4], [[i32; 4]; 4]) {
    let mut put = [[0; 4]; 4];
    let mut prep = [[0; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            let mut sum = 0i32;
            for v in 0..4 {
                let mut horizontal_sum = 0i32;
                for h in 0..4 {
                    let at = (origin as isize
                        + (y as isize + v as isize - 1) * stride
                        + x as isize
                        + h as isize
                        - 1) as usize;
                    horizontal_sum += i32::from(source[at]) * i32::from(horizontal[h + 2]);
                }
                let intermediate = (horizontal_sum + 2) >> 2;
                assert!(i16::try_from(intermediate).is_ok());
                sum += intermediate * i32::from(vertical[v + 2]);
            }
            put[y][x] = (sum + 512) >> 10;
            prep[y][x] = (sum + 32) >> 6;
            assert!(i16::try_from(prep[y][x]).is_ok());
        }
    }
    (put, prep)
}

#[test]
fn fused_four_tap_matches_scalar_with_only_active_source_bytes() {
    let _lock = crate::src::safe_simd::token_test_lock();
    let token = crate::src::cpu::summon_avx2().expect("native fused MC oracle needs AVX2");
    let filters: Vec<_> = crate::src::tables::dav1d_mc_subpel_filters
        .iter()
        .flat_map(|family| family.iter())
        .filter(|filter| tap_base_8tap(filter) == 2)
        .collect();
    assert!(!filters.is_empty());
    let mut comparisons = 0;
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    for horizontal in &filters {
        for vertical in &filters {
            // Independent linear bounds include every possible input byte,
            // not only the pixel patterns sampled below. Horizontal rounding
            // is monotone; vertical rows have independent source pixels.
            let hmin =
                (255 * horizontal.iter().map(|&f| i32::from(f).min(0)).sum::<i32>() + 2) >> 2;
            let hmax =
                (255 * horizontal.iter().map(|&f| i32::from(f).max(0)).sum::<i32>() + 2) >> 2;
            assert!(i16::try_from(hmin).is_ok() && i16::try_from(hmax).is_ok());
            let vmin: i32 = vertical
                .iter()
                .map(|&f| {
                    let f = i32::from(f);
                    f * if f >= 0 { hmin } else { hmax }
                })
                .sum();
            let vmax: i32 = vertical
                .iter()
                .map(|&f| {
                    let f = i32::from(f);
                    f * if f >= 0 { hmax } else { hmin }
                })
                .sum();
            assert!(i16::try_from((vmin + 32) >> 6).is_ok());
            assert!(i16::try_from((vmax + 32) >> 6).is_ok());
            for pattern in 0..8 {
                let mut source = [0u8; 49];
                for (at, pixel) in source.iter_mut().enumerate() {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let h = i32::from(horizontal[2 + (at % 7).min(3)]);
                    let v = i32::from(vertical[2 + (at / 7).min(3)]);
                    *pixel = match pattern {
                        0 => 0,
                        1 => 255,
                        2 => {
                            if at % 2 == 0 {
                                255
                            } else {
                                0
                            }
                        }
                        3 => (at * 255 / 48) as u8,
                        4 => {
                            if h > 0 {
                                255
                            } else {
                                0
                            }
                        }
                        5 => {
                            if h < 0 {
                                255
                            } else {
                                0
                            }
                        }
                        6 => {
                            if h * v > 0 {
                                255
                            } else {
                                0
                            }
                        }
                        _ => rng as u8,
                    };
                }
                let unchanged = source;
                for (origin, stride) in [(8, 7), (36, -7)] {
                    assert_eq!(
                        fused_rows(token, &source, origin, stride, horizontal, vertical),
                        scalar(&source, origin, stride, horizontal, vertical),
                        "horizontal={horizontal:?} vertical={vertical:?} pattern={pattern} stride={stride}",
                    );
                    assert_eq!(source, unchanged);
                    comparisons += 1;
                }
            }
        }
    }
    assert_eq!(comparisons, filters.len() * filters.len() * 8 * 2);
}
