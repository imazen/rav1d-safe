//! Differential parity for the aarch64 16bpc `prep` (compound-prediction
//! motion compensation) kernels, against the scalar reference in `src/mc.rs`.
//!
//! ## Why 16bpc specifically
//!
//! `mct_prep_direct` already dual-computes NEON against
//! `prep_8tap_rust` / `prep_bilin_rust` under `__simd_test`, and a full-corpus
//! decode logs 132,144 `MC_PREP_MISMATCH` calls across 91 vectors, every one
//! of them 10- or 12-bit (record:
//! `benchmarks/aarch64_md5_attribution_2026-08-07.meta`). Zero at 8bpc. This
//! module reproduces that in a second, per (filter, mx, my, size, bitdepth)
//! cell, so a fix can be attributed rather than hoped at.
//!
//! ## The convention this module pins
//!
//! `prep` output is the compound-prediction intermediate that `avg` / `w_avg`
//! / `mask` / `w_mask` then consume. At 16bpc the reference subtracts
//! `BitDepth16::PREP_BIAS` (8192) from every value so it fits `i16`, and the
//! four consumers add `PREP_BIAS * k` back inside their rounding constant.
//!
//! The aarch64 side used to run an UNBIASED variant of that convention in
//! three of the five kernels — `prep_8tap_16bpc_inner` omitted the
//! subtraction, `avg`/`w_avg`/`mask` omitted the addition — while
//! `prep_bilin_16bpc_inner` and `w_mask_16bpc_inner` used the biased one. Two
//! conventions in one seam is not survivable in a decoder that falls back to
//! the scalar reference for any shape its SIMD does not cover: the `tmp`
//! buffer is shared, so a scalar-produced `tmp` read by a NEON `avg` (or the
//! reverse) is off by exactly one `PREP_BIAS`. This module asserts the SCALAR
//! convention, because that is the one the fallback path always speaks.

#![cfg(all(test, target_arch = "aarch64", not(feature = "asm")))]

use crate::include::common::bitdepth::{BitDepth, BitDepth16};
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::src::levels::Filter2d;

/// xorshift64*, so a failure reproduces from its seed.
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
    fn in_range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i32
    }
}

/// Result accumulator: which parameter cells ran, which diverged.
#[derive(Default)]
struct Report {
    cells: usize,
    live: usize,
    bad: Vec<String>,
    first: Option<String>,
}

impl Report {
    fn record(&mut self, label: &str, live: bool, ok: bool, detail: impl FnOnce() -> String) {
        self.cells += 1;
        if live {
            self.live += 1;
        }
        if !ok {
            if self.first.is_none() {
                self.first = Some(format!("{label}: {}", detail()));
            }
            self.bad.push(label.to_string());
        }
    }

    fn finish(self, what: &str, min_cells: usize) {
        assert!(
            self.cells >= min_cells,
            "{what}: only {} parameter cells ran (expected >= {min_cells}) — \
             the sweep is not reaching the kernel",
            self.cells
        );
        assert_eq!(
            self.live,
            self.cells,
            "{what}: {} of {} cells did NOT take the NEON path, so they compared \
             the scalar reference against itself and proved nothing",
            self.cells - self.live,
            self.cells
        );
        assert!(
            self.bad.is_empty(),
            "{what}: {} of {} parameter cells diverge from the scalar reference.\n  \
             first: {}\n  cells: {:?}",
            self.bad.len(),
            self.cells,
            self.first.unwrap_or_default(),
            self.bad
        );
    }
}

/// Padding around the block, in pixels. The 8-tap reads `[-3, +4]` in both
/// directions; 8 is comfortably clear of that on every side.
const PAD: usize = 8;

/// `Rav1dPictureDataComponent::wrap_buf` asserts the buffer is a multiple of
/// 64 BYTES, so round the pixel count up to keep the harness legal.
///
/// The other half of that contract is ALIGNMENT: the buffer must also START on
/// a 64-byte boundary, which a `Vec<BD::Pixel>` does not. `aligned_plane`
/// supplies the storage; see its doc comment.
fn plane_len<T>(stride: usize, rows: usize) -> usize {
    let px = stride * rows;
    let per = 64 / core::mem::size_of::<T>();
    px.next_multiple_of(per)
}

/// Run one prep cell through NEON and through the scalar reference.
///
/// Returns `(neon_tmp, scalar_tmp, neon_took_the_simd_path)`.
fn prep_cell(
    filter: Filter2d,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth_max: u16,
    src_plane: &[u16],
    stride: usize,
) -> (Vec<i16>, Vec<i16>, bool) {
    let bd = BitDepth16::new(bitdepth_max);
    let base = PAD * stride + PAD;

    let mut neon = vec![0i16; w * h];
    let mut px = crate::src::safe_simd::aligned_plane(src_plane);
    let live = {
        let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth16>(&mut px, stride);
        let src = comp.with_offset::<BitDepth16>() + base;
        crate::src::safe_simd::mc_arm::mct_prep_dispatch::<BitDepth16>(
            filter, &mut neon, src, w as i32, h as i32, mx, my, bd,
        )
    };

    let mut scalar = vec![0i16; w * h];
    let mut px2 = crate::src::safe_simd::aligned_plane(src_plane);
    {
        let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth16>(&mut px2, stride);
        let src = comp.with_offset::<BitDepth16>() + base;
        match filter {
            Filter2d::Bilinear => crate::src::mc::prep_bilin_rust::<BitDepth16>(
                &mut scalar,
                src,
                w,
                h,
                mx as usize,
                my as usize,
                bd,
            ),
            _ => crate::src::mc::prep_8tap_rust::<BitDepth16>(
                &mut scalar,
                src,
                w,
                h,
                mx as usize,
                my as usize,
                filter.hv(),
                bd,
            ),
        }
    }

    (neon, scalar, live)
}

/// Every AV1 inter block size that `mct_prep` can be asked for.
const SIZES: &[(usize, usize)] = &[
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 32),
    (64, 64),
    (128, 128),
];

/// The four subpel branches: (0,0) pure copy, H only, V only, H+V. Each is a
/// separate arm of `prep_8tap_16bpc_inner` with its own shift and its own
/// `PREP_BIAS` handling, so a sweep that only tries H+V misses three
/// independent code paths.
const SUBPEL: &[(i32, i32)] = &[(0, 0), (5, 0), (0, 11), (5, 11), (1, 1), (15, 15)];

#[test]
fn prep_16bpc_matches_scalar() {
    let _lock = crate::src::safe_simd::token_test_lock();
    let mut rep = Report::default();

    for &bitdepth in &[10u8, 12] {
        let bd_max = ((1u32 << bitdepth) - 1) as u16;
        for &filter in &[
            Filter2d::Regular8Tap,
            Filter2d::Smooth8Tap,
            Filter2d::Sharp8Tap,
            Filter2d::RegularSmooth8Tap,
            Filter2d::Bilinear,
        ] {
            for &(w, h) in SIZES {
                let stride = w + 2 * PAD;
                let mut rng = Rng(0x5EED_0000_0000_0001
                    ^ ((bitdepth as u64) << 48)
                    ^ ((filter as u64) << 40)
                    ^ ((w * h) as u64));
                let plane: Vec<u16> = (0..plane_len::<u16>(stride, h + 2 * PAD))
                    .map(|_| rng.in_range(0, bd_max as i32) as u16)
                    .collect();

                for &(mx, my) in SUBPEL {
                    let (neon, scalar, live) =
                        prep_cell(filter, w, h, mx, my, bd_max, &plane, stride);
                    let bad = (0..w * h).find(|&i| neon[i] != scalar[i]);
                    rep.record(
                        &format!(
                            "prep bd={bitdepth} filter={} {w}x{h} mx={mx} my={my}",
                            filter as u32
                        ),
                        live,
                        bad.is_none(),
                        || {
                            let i = bad.unwrap();
                            format!(
                                "at ({},{}) neon={} scalar={} (diff {})",
                                i % w,
                                i / w,
                                neon[i],
                                scalar[i],
                                neon[i] as i32 - scalar[i] as i32
                            )
                        },
                    );
                }
            }
        }
    }
    rep.finish("aarch64 16bpc prep", 600);
}

/// 8bpc prep is already bit-exact (zero `MC_PREP_MISMATCH` across all 358
/// 8-bit/data vectors). This arm is the control: if it ever goes red, the
/// harness itself is wrong, not the 16bpc kernel.
#[test]
fn prep_8bpc_matches_scalar_control() {
    use crate::include::common::bitdepth::BitDepth8;
    let _lock = crate::src::safe_simd::token_test_lock();
    let mut rep = Report::default();
    let bd = BitDepth8::new(());

    for &filter in &[
        Filter2d::Regular8Tap,
        Filter2d::Sharp8Tap,
        Filter2d::Bilinear,
    ] {
        for &(w, h) in SIZES {
            let stride = w + 2 * PAD;
            let base = PAD * stride + PAD;
            let mut rng = Rng(0xC0FF_EE00_0000_0002 ^ ((filter as u64) << 40) ^ ((w * h) as u64));
            let plane: Vec<u8> = (0..plane_len::<u8>(stride, h + 2 * PAD))
                .map(|_| rng.in_range(0, 255) as u8)
                .collect();

            for &(mx, my) in SUBPEL {
                let mut neon = vec![0i16; w * h];
                let mut px = crate::src::safe_simd::aligned_plane(&plane);
                let live = {
                    let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut px, stride);
                    let src = comp.with_offset::<BitDepth8>() + base;
                    crate::src::safe_simd::mc_arm::mct_prep_dispatch::<BitDepth8>(
                        filter, &mut neon, src, w as i32, h as i32, mx, my, bd,
                    )
                };
                let mut scalar = vec![0i16; w * h];
                let mut px2 = crate::src::safe_simd::aligned_plane(&plane);
                {
                    let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut px2, stride);
                    let src = comp.with_offset::<BitDepth8>() + base;
                    match filter {
                        Filter2d::Bilinear => crate::src::mc::prep_bilin_rust::<BitDepth8>(
                            &mut scalar,
                            src,
                            w,
                            h,
                            mx as usize,
                            my as usize,
                            bd,
                        ),
                        _ => crate::src::mc::prep_8tap_rust::<BitDepth8>(
                            &mut scalar,
                            src,
                            w,
                            h,
                            mx as usize,
                            my as usize,
                            filter.hv(),
                            bd,
                        ),
                    }
                }
                let bad = (0..w * h).find(|&i| neon[i] != scalar[i]);
                rep.record(
                    &format!("prep bd=8 filter={} {w}x{h} mx={mx} my={my}", filter as u32),
                    live,
                    bad.is_none(),
                    || {
                        let i = bad.unwrap();
                        format!(
                            "at ({},{}) neon={} scalar={}",
                            i % w,
                            i / w,
                            neon[i],
                            scalar[i]
                        )
                    },
                );
            }
        }
    }
    rep.finish("aarch64 8bpc prep (control)", 200);
}
