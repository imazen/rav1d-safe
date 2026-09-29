//! Differential parity for the x86_64 16bpc `prep` (compound-prediction
//! motion compensation) kernels, against the scalar reference in `src/mc.rs`.
//!
//! Twin of `mc_arm_prep_parity.rs` — same cells, same convention assertion
//! (the scalar `PREP_BIAS` convention). The aarch64 version documents why the
//! bias seam is fragile; on x86 the argon conformance corpus catches the same
//! class of divergence via `decode_md5`.

#![cfg(all(test, target_arch = "x86_64", not(feature = "asm")))]

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
            "{what}: {} of {} cells did NOT take the SIMD path, so they compared \
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
fn plane_len<T>(stride: usize, rows: usize) -> usize {
    let px = stride * rows;
    let per = 64 / core::mem::size_of::<T>();
    px.next_multiple_of(per)
}

/// Run one prep cell through AVX2 and through the scalar reference.
///
/// Returns `(simd_tmp, scalar_tmp, simd_took_the_simd_path)`.
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

    let mut simd = vec![0i16; w * h];
    let mut px = crate::src::safe_simd::aligned_plane(src_plane);
    let live = {
        let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth16>(&mut px, stride);
        let src = comp.with_offset::<BitDepth16>() + base;
        crate::src::safe_simd::mc::mct_prep_dispatch::<BitDepth16>(
            filter, &mut simd, src, w as i32, h as i32, mx, my, bd,
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

    (simd, scalar, live)
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
                    let (simd, scalar, live) =
                        prep_cell(filter, w, h, mx, my, bd_max, &plane, stride);
                    let bad = (0..w * h).find(|&i| simd[i] != scalar[i]);
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
                                "at ({},{}) simd={} scalar={} (diff {})",
                                i % w,
                                i / w,
                                simd[i],
                                scalar[i],
                                simd[i] as i32 - scalar[i] as i32
                            )
                        },
                    );
                }
            }
        }
    }
    rep.finish("x86 16bpc prep", 600);
}

/// Run one put cell through AVX2 and through the scalar reference.
///
/// Returns `(simd_dst_row0..h, scalar_dst_row0..h, simd_took_the_simd_path)`
/// with each output cropped to the `w * h` block.
fn put_cell(
    filter: Filter2d,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth_max: u16,
    src_plane: &[u16],
    stride: usize,
) -> (Vec<u16>, Vec<u16>, bool) {
    let bd = BitDepth16::new(bitdepth_max);
    let base = PAD * stride + PAD;
    let dst_stride = w;
    let dst_len = plane_len::<u16>(dst_stride, h);

    let mut simd = vec![0u16; dst_len];
    let mut px = crate::src::safe_simd::aligned_plane(src_plane);
    let live = {
        let mut dpx = crate::src::safe_simd::aligned_plane(&simd);
        let src_comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth16>(&mut px, stride);
        let dst_comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth16>(&mut dpx, dst_stride);
        let src = src_comp.with_offset::<BitDepth16>() + base;
        let dst = dst_comp.with_offset::<BitDepth16>() + 0usize;
        let took = crate::src::safe_simd::mc::mc_put_dispatch::<BitDepth16>(
            filter, dst, src, w as i32, h as i32, mx, my, bd,
        );
        simd.copy_from_slice(&dpx[..dst_len]);
        took
    };

    let mut scalar = vec![0u16; dst_len];
    let mut px2 = crate::src::safe_simd::aligned_plane(src_plane);
    {
        let mut dpx2 = crate::src::safe_simd::aligned_plane(&scalar);
        let src_comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth16>(&mut px2, stride);
        let dst_comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth16>(&mut dpx2, dst_stride);
        let src = src_comp.with_offset::<BitDepth16>() + base;
        let dst = dst_comp.with_offset::<BitDepth16>() + 0usize;
        match filter {
            Filter2d::Bilinear => crate::src::mc::put_bilin_rust::<BitDepth16>(
                dst,
                src,
                w,
                h,
                mx as usize,
                my as usize,
                bd,
            ),
            _ => crate::src::mc::put_8tap_rust::<BitDepth16>(
                dst,
                src,
                w,
                h,
                mx as usize,
                my as usize,
                filter.hv(),
                bd,
            ),
        }
        scalar.copy_from_slice(&dpx2[..dst_len]);
    }

    (simd, scalar, live)
}

/// 16bpc put sweep — twin of the prep sweep. The put bilinear kernels share
/// `h_bilin_16bpc_*_inner` (mid pass) and carry their own `4 + ib` V-pass and
/// two-stage H-only rounding; argon coverage showed this family drifting the
/// same way prep did.
#[test]
fn put_16bpc_matches_scalar() {
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
                let mut rng = Rng(0x5EED_0000_0000_0002
                    ^ ((bitdepth as u64) << 48)
                    ^ ((filter as u64) << 40)
                    ^ ((w * h) as u64));
                let plane: Vec<u16> = (0..plane_len::<u16>(stride, h + 2 * PAD))
                    .map(|_| rng.in_range(0, bd_max as i32) as u16)
                    .collect();

                for &(mx, my) in SUBPEL {
                    let (simd, scalar, live) =
                        put_cell(filter, w, h, mx, my, bd_max, &plane, stride);
                    let bad = (0..w * h).find(|&i| simd[i] != scalar[i]);
                    rep.record(
                        &format!(
                            "put bd={bitdepth} filter={} {w}x{h} mx={mx} my={my}",
                            filter as u32
                        ),
                        live,
                        bad.is_none(),
                        || {
                            let i = bad.unwrap();
                            format!(
                                "at ({},{}) simd={} scalar={} (diff {})",
                                i % w,
                                i / w,
                                simd[i],
                                scalar[i],
                                simd[i] as i32 - scalar[i] as i32
                            )
                        },
                    );
                }
            }
        }
    }
    rep.finish("x86 16bpc put", 600);
}

/// 8bpc prep control: if this ever goes red, the harness itself is wrong.
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
                let mut simd = vec![0i16; w * h];
                let mut px = crate::src::safe_simd::aligned_plane(&plane);
                let live = {
                    let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut px, stride);
                    let src = comp.with_offset::<BitDepth8>() + base;
                    crate::src::safe_simd::mc::mct_prep_dispatch::<BitDepth8>(
                        filter, &mut simd, src, w as i32, h as i32, mx, my, bd,
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
                let bad = (0..w * h).find(|&i| simd[i] != scalar[i]);
                rep.record(
                    &format!("prep bd=8 filter={} {w}x{h} mx={mx} my={my}", filter as u32),
                    live,
                    bad.is_none(),
                    || {
                        let i = bad.unwrap();
                        format!(
                            "at ({},{}) simd={} scalar={}",
                            i % w,
                            i / w,
                            simd[i],
                            scalar[i]
                        )
                    },
                );
            }
        }
    }
    rep.finish("x86 8bpc prep (control)", 200);
}
