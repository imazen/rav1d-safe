#![cfg(all(test, target_arch = "x86_64", not(feature = "asm")))]

//! Full-domain parity for the safe 8bpc MC kernels on x86_64: every AV1 inter
//! block size, every non-bilinear 8tap filter combination plus bilinear, and
//! the complete 16x16 phase grid (integer, h-only, v-only, all hv pairs).
//! Each cell compares the scalar reference against the dispatch result, the
//! AVX2 impl called directly, and the AVX-512 impl when the CPU has it — so
//! all tiers are covered regardless of which one dispatch picks.

use crate::include::common::bitdepth::{BitDepth, BitDepth8};
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::src::levels::Filter2d;
use crate::src::safe_simd::aligned_plane;

const PAD: usize = 8;

/// `Rav1dPictureDataComponent::wrap_buf` copies into a `PicBuf` whose length
/// must be a multiple of the guaranteed 64-byte unit.
const fn wrap_len(n: usize) -> usize {
    n.div_ceil(64) * 64
}

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
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
}

const SEED: u64 = 0x8b5a_cdef_91a3_7257;

/// Every AV1 inter block size `put_8tap`/`prep_8tap` can be asked for.
const BLOCK_SIZES: &[(usize, usize)] = &[
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (4, 16),
    (16, 4),
    (8, 16),
    (16, 8),
    (8, 32),
    (32, 8),
    (16, 32),
    (32, 16),
    (16, 16),
    (16, 64),
    (64, 16),
    (32, 64),
    (64, 32),
    (32, 32),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
];

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

/// The reference-window pad a filter+phase needs, matching `filter_guard`:
/// `(0,0)` when the axis is integer, `(0,1)` for bilinear, else `(3,4)`.
fn pad(filter: Filter2d, phase: i32) -> (usize, usize) {
    if phase == 0 {
        (0, 0)
    } else if filter == Filter2d::Bilinear {
        (0, 1)
    } else {
        (3, 4)
    }
}

/// The contiguous guard window and the offset of the (0,0) pixel within it,
/// as `filter_guard`/`read_guard` would produce.
fn src_window(
    filter: Filter2d,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    plane: &[u8],
    stride: usize,
) -> (&[u8], usize) {
    let (l, r) = pad(filter, mx);
    let (t, b) = pad(filter, my);
    let base = PAD * stride + PAD;
    let origin = base - t * stride - l;
    let end = base + (h - 1 + b) * stride + w + r;
    (&plane[origin..end], t * stride + l)
}

struct PutOut {
    dispatch: Vec<u8>,
    avx2: Vec<u8>,
    avx512: Option<Vec<u8>>,
    scalar: Vec<u8>,
    live: bool,
}

fn put_cell(
    filter: Filter2d,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    plane: &[u8],
    stride: usize,
) -> PutOut {
    let bd = BitDepth8::new(());
    let base = PAD * stride + PAD;

    // Default wrap_buf copies into PicBuf; C-FFI uses the aligned caller
    // storage directly. Read back through the corresponding storage model.
    let mut dispatch_px = vec![0u8; wrap_len(h * stride)];
    let mut px = aligned_plane(plane);
    let live = {
        let src_comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut px, stride);
        let mut stage = aligned_plane(&vec![0u8; wrap_len(h * stride)]);
        let dst_comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut stage, stride);
        let live = crate::src::safe_simd::mc::mc_put_dispatch::<BitDepth8>(
            filter,
            dst_comp.with_offset::<BitDepth8>(),
            src_comp.with_offset::<BitDepth8>() + base,
            w as i32,
            h as i32,
            mx,
            my,
            bd,
        );
        #[cfg(not(feature = "c-ffi"))]
        dst_comp.copy_pixels_to::<BitDepth8>(&mut dispatch_px);
        #[cfg(feature = "c-ffi")]
        {
            drop(dst_comp);
            dispatch_px.copy_from_slice(&stage);
        }
        live
    };

    let (window, src_base) = src_window(filter, w, h, mx, my, plane, stride);
    let (hf, vf) = filter.hv();
    let (mut avx2_px, mut avx512_px) = (
        vec![0u8; wrap_len(h * stride)],
        vec![0u8; wrap_len(h * stride)],
    );
    let token = crate::src::cpu::summon_avx2().unwrap();
    let has512 = crate::src::cpu::summon_avx512().is_some();
    match filter {
        Filter2d::Bilinear => {
            crate::src::safe_simd::mc::put_bilin_8bpc_avx2_impl_testable(
                token,
                &mut avx2_px,
                stride as isize,
                &window[src_base..],
                stride as isize,
                w as i32,
                h as i32,
                mx,
                my,
            );
            if has512 && w >= 64 {
                let t512 = crate::src::cpu::summon_avx512().unwrap();
                crate::src::safe_simd::mc::put_bilin_8bpc_avx512_impl_inner(
                    t512,
                    &mut avx512_px,
                    0,
                    stride as isize,
                    &window[src_base..],
                    0,
                    stride as isize,
                    w as i32,
                    h as i32,
                    mx,
                    my,
                );
            } else {
                avx512_px.clone_from(&avx2_px);
            }
        }
        _ => {
            crate::src::safe_simd::mc::put_8tap_8bpc_avx2_impl_testable(
                token,
                &mut avx2_px,
                stride as isize,
                window,
                src_base,
                stride as isize,
                w as i32,
                h as i32,
                mx,
                my,
                hf,
                vf,
            );
            if has512 {
                let t512 = crate::src::cpu::summon_avx512().unwrap();
                crate::src::safe_simd::mc::put_8tap_8bpc_avx512_impl_inner(
                    t512,
                    &mut avx512_px,
                    0,
                    stride as isize,
                    window,
                    src_base,
                    stride as isize,
                    w as i32,
                    h as i32,
                    mx,
                    my,
                    hf,
                    vf,
                );
            } else {
                avx512_px.clone_from(&avx2_px);
            }
        }
    }

    let mut scalar_px = vec![0u8; wrap_len(h * stride)];
    let mut px2 = aligned_plane(plane);
    {
        let src_comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut px2, stride);
        let mut stage = aligned_plane(&vec![0u8; wrap_len(h * stride)]);
        let dst_comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut stage, stride);
        match filter {
            Filter2d::Bilinear => crate::src::mc::put_bilin_rust::<BitDepth8>(
                dst_comp.with_offset::<BitDepth8>(),
                src_comp.with_offset::<BitDepth8>() + base,
                w,
                h,
                mx as usize,
                my as usize,
                bd,
            ),
            _ => crate::src::mc::put_8tap_rust::<BitDepth8>(
                dst_comp.with_offset::<BitDepth8>(),
                src_comp.with_offset::<BitDepth8>() + base,
                w,
                h,
                mx as usize,
                my as usize,
                filter.hv(),
                bd,
            ),
        }
        #[cfg(not(feature = "c-ffi"))]
        dst_comp.copy_pixels_to::<BitDepth8>(&mut scalar_px);
        #[cfg(feature = "c-ffi")]
        {
            drop(dst_comp);
            scalar_px.copy_from_slice(&stage);
        }
    }

    PutOut {
        dispatch: dispatch_px,
        avx2: avx2_px,
        avx512: has512.then_some(avx512_px),
        scalar: scalar_px,
        live,
    }
}

struct PrepOut {
    dispatch: Vec<i16>,
    avx2: Vec<i16>,
    avx512: Option<Vec<i16>>,
    scalar: Vec<i16>,
    live: bool,
}

fn prep_cell(
    filter: Filter2d,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    plane: &[u8],
    stride: usize,
) -> PrepOut {
    let bd = BitDepth8::new(());
    let base = PAD * stride + PAD;

    let mut dispatch = vec![0i16; w * h];
    let mut px = aligned_plane(plane);
    let live = {
        let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut px, stride);
        let src = comp.with_offset::<BitDepth8>() + base;
        crate::src::safe_simd::mc::mct_prep_dispatch::<BitDepth8>(
            filter,
            &mut dispatch,
            src,
            w as i32,
            h as i32,
            mx,
            my,
            bd,
        )
    };

    let (window, src_base) = src_window(filter, w, h, mx, my, plane, stride);
    let (hf, vf) = filter.hv();
    let (mut avx2, mut avx512) = (vec![0i16; w * h], vec![0i16; w * h]);
    let token = crate::src::cpu::summon_avx2().unwrap();
    let has512 = crate::src::cpu::summon_avx512().is_some();
    match filter {
        Filter2d::Bilinear => {
            crate::src::safe_simd::mc::prep_bilin_8bpc_avx2_impl_testable(
                token,
                &mut avx2,
                &window[src_base..],
                stride as isize,
                w as i32,
                h as i32,
                mx,
                my,
            );
            avx512.clone_from(&avx2);
        }
        _ => {
            crate::src::safe_simd::mc::prep_8tap_8bpc_avx2_impl_testable(
                token,
                &mut avx2,
                window,
                src_base,
                stride as isize,
                w as i32,
                h as i32,
                mx,
                my,
                hf,
                vf,
            );
            if has512 {
                let t512 = crate::src::cpu::summon_avx512().unwrap();
                crate::src::safe_simd::mc::prep_8tap_8bpc_avx512_impl_inner(
                    t512,
                    &mut avx512,
                    window,
                    src_base,
                    stride as isize,
                    w as i32,
                    h as i32,
                    mx,
                    my,
                    hf,
                    vf,
                );
            } else {
                avx512.clone_from(&avx2);
            }
        }
    }

    let mut scalar = vec![0i16; w * h];
    let mut px2 = aligned_plane(plane);
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

    PrepOut {
        dispatch,
        avx2,
        avx512: has512.then_some(avx512),
        scalar,
        live,
    }
}

#[test]
fn put_prep_8bpc_full_domain_parity() {
    // Token permutations in sibling tests mask AVX2/AVX-512 globally; serialize.
    let _lock = crate::src::safe_simd::token_test_lock();
    if crate::src::cpu::summon_avx2().is_none() {
        eprintln!("Skipping x86 8bpc MC parity: AVX2 token unavailable");
        return;
    }
    let mut rng = Rng(SEED);
    let mut labels = Vec::new();
    let mut cells = 0u64;
    for &(w, h) in BLOCK_SIZES {
        let stride = w + 2 * PAD;
        let rows = h + 2 * PAD;
        let mut plane = vec![0u8; wrap_len(rows * stride)];
        for p in &mut plane {
            *p = rng.byte();
        }
        for &filter in FILTERS {
            for mx in 0i32..=15 {
                for my in 0i32..=15 {
                    let label = format!("f{} {w}x{h} mx={mx} my={my}", filter as u8);

                    let o = put_cell(filter, w, h, mx, my, &plane, stride);
                    assert!(o.live, "mc_put_dispatch not taken ({label})");
                    if o.dispatch != o.scalar {
                        labels.push(format!("put dispatch {label}"));
                    }
                    if o.avx2 != o.scalar {
                        let i = (0..o.avx2.len().min(o.scalar.len()))
                            .find(|&i| o.avx2[i] != o.scalar[i]);
                        labels.push(format!(
                            "put avx2 {label} first-diff@{i:?} simd={:?} scalar={:?} disp={:?}",
                            i.map(|i| &o.avx2[i.saturating_sub(4)..(i + 8).min(o.avx2.len())]),
                            i.map(|i| &o.scalar[i.saturating_sub(4)..(i + 8).min(o.scalar.len())]),
                            i.map(
                                |i| &o.dispatch[i.saturating_sub(4)..(i + 8).min(o.dispatch.len())]
                            ),
                        ));
                    }
                    if let Some(v) = &o.avx512 {
                        if *v != o.scalar {
                            labels.push(format!("put avx512 {label}"));
                        }
                    }

                    let p = prep_cell(filter, w, h, mx, my, &plane, stride);
                    assert!(p.live, "mct_prep_dispatch not taken ({label})");
                    if p.dispatch != p.scalar {
                        labels.push(format!("prep dispatch {label}"));
                    }
                    if p.avx2 != p.scalar {
                        labels.push(format!("prep avx2 {label}"));
                    }
                    if let Some(v) = &p.avx512 {
                        if *v != p.scalar {
                            labels.push(format!("prep avx512 {label}"));
                        }
                    }

                    cells += 2;
                }
            }
        }
    }
    assert!(
        labels.is_empty(),
        "{} mismatches, first 10: {:?}",
        labels.len(),
        &labels[..labels.len().min(10)]
    );
    eprintln!(
        "x86 8bpc parity: {cells} cells across {} block sizes and {} filters",
        BLOCK_SIZES.len(),
        FILTERS.len()
    );
}

/// Exercise two-pixel chroma blocks and each vector/scalar tail boundary with
/// endpoint-valued inputs, including the largest supported block.
#[test]
fn put_prep_8bpc_endpoint_and_tail_parity() {
    let _lock = crate::src::safe_simd::token_test_lock();
    crate::src::cpu::summon_avx2().expect("native x86 MC regression requires AVX2");
    for (w, h) in [
        (2, 2),
        (2, 4),
        (4, 2),
        (2, 8),
        (8, 2),
        (3, 3),
        (6, 8),
        (7, 9),
        (15, 17),
        (17, 15),
        (31, 33),
        (63, 65),
        (127, 128),
        (128, 128),
    ] {
        let stride = w + 2 * PAD;
        let mut plane = vec![0u8; wrap_len((h + 2 * PAD) * stride)];
        for (i, p) in plane.iter_mut().enumerate() {
            *p = if ((i % stride) % 8 < 4) ^ ((i / stride) % 8 < 4) {
                255
            } else {
                0
            };
        }
        assert!(plane.contains(&0) && plane.contains(&255));
        for &filter in FILTERS {
            for mx in 0..16 {
                for my in 0..16 {
                    let label = format!("endpoint f{} {w}x{h} mx={mx} my={my}", filter as u8);
                    let put = put_cell(filter, w, h, mx, my, &plane, stride);
                    assert!(put.live, "put dispatch missing: {label}");
                    assert_eq!(put.dispatch, put.scalar, "put dispatch: {label}");
                    assert_eq!(put.avx2, put.scalar, "put AVX2: {label}");
                    if let Some(avx512) = put.avx512 {
                        assert_eq!(avx512, put.scalar, "put AVX-512: {label}");
                    }
                    let prep = prep_cell(filter, w, h, mx, my, &plane, stride);
                    assert!(prep.live, "prep dispatch missing: {label}");
                    assert_eq!(prep.dispatch, prep.scalar, "prep dispatch: {label}");
                    assert_eq!(prep.avx2, prep.scalar, "prep AVX2: {label}");
                    if let Some(avx512) = prep.avx512 {
                        assert_eq!(avx512, prep.scalar, "prep AVX-512: {label}");
                    }
                }
            }
        }
    }
}

/// maddubs saturates each pair and epi16 additions wrap. Linear extrema on
/// [0,255] bound every pair and every subset of the current filter rows.
#[test]
fn pair_window_arithmetic_fits_i16_for_all_filter_rows() {
    let mut pair_bounds = (i32::MAX, i32::MIN);
    let mut sum_bounds = (i32::MAX, i32::MIN);
    for family in crate::src::tables::dav1d_mc_subpel_filters.iter() {
        for filter in family {
            for pair in filter.windows(2) {
                let low: i32 = pair.iter().map(|&c| i32::from(c).min(0) * 255).sum();
                let high: i32 = pair.iter().map(|&c| i32::from(c).max(0) * 255).sum();
                assert!(low >= i32::from(i16::MIN) && high <= i32::from(i16::MAX));
                pair_bounds.0 = pair_bounds.0.min(low);
                pair_bounds.1 = pair_bounds.1.max(high);
            }
            let low: i32 = filter.iter().map(|&c| i32::from(c).min(0) * 255).sum();
            let high: i32 = filter.iter().map(|&c| i32::from(c).max(0) * 255).sum();
            // 34 is the largest rounding addition in the pair-window paths.
            assert!(low >= i32::from(i16::MIN) && high + 34 <= i32::from(i16::MAX));
            sum_bounds.0 = sum_bounds.0.min(low);
            sum_bounds.1 = sum_bounds.1.max(high);
        }
    }
    eprintln!(
        "8bpc filter pair bounds={pair_bounds:?}, subset bounds={sum_bounds:?}, max round=34"
    );
}

/// The row-batched optimization must also walk backwards within the complete
/// bounded source window; a suffix starting at row zero cannot do that.
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
                    crate::src::safe_simd::mc::put_8tap_8bpc_avx2_impl_testable(
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
                    crate::src::safe_simd::mc::prep_8tap_8bpc_avx2_impl_testable(
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
                        crate::src::safe_simd::mc::put_8tap_8bpc_avx512_impl_inner(
                            token512,
                            &mut put,
                            0,
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
                        crate::src::safe_simd::mc::prep_8tap_8bpc_avx512_impl_inner(
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
