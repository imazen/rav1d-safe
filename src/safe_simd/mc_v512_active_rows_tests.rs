//! Active vertical rows and full signed intermediate values against scalar.

use super::*;

#[arcane]
fn put_row(
    token: Server64,
    mid: &[[i16; MID_STRIDE]],
    width: usize,
    filter: &[i8; 8],
    dst: &mut [u8],
) {
    v_filter_8tap_8bpc_avx512_inner(token, dst, mid, width, filter, 10, 255);
}

#[arcane]
fn prep_row(
    token: Server64,
    mid: &[[i16; MID_STRIDE]],
    width: usize,
    filter: &[i8; 8],
    dst: &mut [i16],
) {
    v_filter_8tap_to_i16_avx512_inner(token, mid, dst, width, filter, 6);
}

#[test]
fn avx512_vertical_prep_reads_only_active_rows_at_legal_intermediate_extrema() {
    let _lock = crate::src::safe_simd::token_test_lock();
    let Some(token) = crate::src::cpu::summon_avx512() else {
        return;
    };
    // Every rounded horizontal output lies in this table-wide interval.
    let endpoints = [-1785i16, 0, 1, 4080, 5865];
    let mut tap_cases = [0; 3];
    for family in crate::src::tables::dav1d_mc_subpel_filters.iter() {
        for filter in family {
            let start = tap_base_8tap(filter);
            let count = 8 - 2 * start;
            let mut mid = vec![[0i16; MID_STRIDE]; start + count];
            for pattern in 0..4 {
                for (y, row) in mid.iter_mut().enumerate() {
                    for (x, value) in row.iter_mut().enumerate() {
                        *value = match pattern {
                            0 => endpoints[0],
                            1 => endpoints[4],
                            2 => {
                                if filter[y] >= 0 {
                                    endpoints[4]
                                } else {
                                    endpoints[0]
                                }
                            }
                            _ => endpoints[(x + y) % endpoints.len()],
                        };
                    }
                }
                for width in [4, 8, 15, 16, 17, 31, 32, 65, 128] {
                    let mut expected = vec![1234; width + 7];
                    let mut actual = expected.clone();
                    for x in 0..width {
                        let sum: i32 = (start..start + count)
                            .map(|y| i32::from(filter[y]) * i32::from(mid[y][x]))
                            .sum();
                        let value = (sum + 32) >> 6;
                        expected[x] = i16::try_from(value).expect("legal prep result must fit");
                    }
                    prep_row(token, &mid, width, filter, &mut actual);
                    assert_eq!(
                        actual, expected,
                        "filter={filter:?} pattern={pattern} width={width}"
                    );
                    tap_cases[(count - 4) / 2] += 1;
                }
            }
        }
    }
    assert!(tap_cases.iter().all(|&count| count > 0));
    println!("AVX512 prep scalar comparisons by four/six/eight taps: {tap_cases:?}");
}

#[test]
fn avx512_vertical_put_reads_only_active_rows_at_full_i16_range() {
    let _lock = crate::src::safe_simd::token_test_lock();
    let Some(token) = crate::src::cpu::summon_avx512() else {
        return;
    };
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut tap_cases = [0; 3];
    for family in crate::src::tables::dav1d_mc_subpel_filters.iter() {
        for filter in family {
            let start = tap_base_8tap(filter);
            let count = 8 - 2 * start;
            assert!([4, 6, 8].contains(&count));
            // The inactive trailing rows are absent, not zero-filled.
            let mut mid = vec![[0i16; MID_STRIDE]; start + count];
            for pattern in 0..6 {
                for (y, row) in mid.iter_mut().enumerate() {
                    for (x, value) in row.iter_mut().enumerate() {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        *value = match pattern {
                            0 => i16::MIN,
                            1 => i16::MAX,
                            2 => {
                                if (x + y) % 2 == 0 {
                                    i16::MIN
                                } else {
                                    i16::MAX
                                }
                            }
                            3 => {
                                if filter[y] >= 0 {
                                    i16::MAX
                                } else {
                                    i16::MIN
                                }
                            }
                            4 => {
                                if filter[y] >= 0 {
                                    i16::MIN
                                } else {
                                    i16::MAX
                                }
                            }
                            _ => state as i16,
                        };
                    }
                }
                for width in [4, 8, 15, 16, 17, 31, 32, 65, 128] {
                    let mut expected = vec![0xa5; width + 7];
                    let mut actual = expected.clone();
                    for x in 0..width {
                        let sum: i32 = (start..start + count)
                            .map(|y| i32::from(filter[y]) * i32::from(mid[y][x]))
                            .sum();
                        expected[x] = ((sum + 512) >> 10).clamp(0, 255) as u8;
                    }
                    put_row(token, &mid, width, filter, &mut actual);
                    assert_eq!(
                        actual, expected,
                        "filter={filter:?} pattern={pattern} width={width}"
                    );
                    tap_cases[(count - 4) / 2] += 1;
                }
            }
        }
    }
    assert!(
        tap_cases.iter().all(|&count| count > 0),
        "every tap count must run"
    );
    println!("AVX512 put scalar comparisons by four/six/eight taps: {tap_cases:?}");
}
