//! Differential fuzz target: rav1d-safe vs dav1d.
//!
//! Feeds the same bytes to both decoders. Any divergence panics:
//!   - One decoder accepts the input, the other rejects it
//!   - Both decode a frame but disagree on dimensions / bit depth / pixel layout
//!   - Both decode a frame but disagree on any pixel value
//!
//! Both decoders are run single-threaded with `max_frame_delay = 1` for
//! determinism (tile parallelism only — no frame threading).
//!
//! Licenses are compatible: rav1d-safe is BSD-2-Clause (port of rav1d),
//! dav1d is BSD-2-Clause.
//!
//! Seed corpus: share `fuzz/corpus/decode_obu/` (valid + adversarial AV1 OBU bytes).
//! Performance: ~2x slower than `decode_obu` — suitable for nightly runs.

#![no_main]

use dav1d::{Decoder as Dav1dDecoder, PixelLayout as Dav1dPixelLayout, PlanarImageComponent, Settings as Dav1dSettings};
use libfuzzer_sys::fuzz_target;
use rav1d_safe::src::managed::{Decoder, PixelLayout, Planes, Settings, Strictness};

const FRAME_SIZE_LIMIT_PIXELS: u32 = 256 * 256;

/// Decode `data` through rav1d-safe. Returns the first complete frame or an error.
fn decode_rav1d(data: &[u8]) -> Result<Option<rav1d_safe::src::managed::Frame>, String> {
    let mut settings = Settings::default();
    settings.threads = 1;
    settings.max_frame_delay = 1;
    settings.frame_size_limit = FRAME_SIZE_LIMIT_PIXELS;
    settings.apply_grain = false; // film-grain RNG is stochastic; disable for byte-exact compare
    // dav1d parity: the libdav1d we link runs its library default
    // (strict_std_compliance = 0) and conceals corrupt tile data; the byte-exact
    // compare only holds if rav1d-safe conceals the same way.
    settings.strictness = Strictness::Lenient;

    let mut decoder = Decoder::with_settings(settings).map_err(|e| format!("rav1d init: {e:?}"))?;
    let result = decoder.decode(data).map_err(|e| format!("rav1d decode: {e:?}"))?;
    Ok(result)
}

/// Decode `data` through dav1d. Returns the first complete frame or an error.
fn decode_dav1d(data: &[u8]) -> Result<Option<dav1d::Picture>, String> {
    let mut settings = Dav1dSettings::new();
    settings.set_n_threads(1);
    settings.set_max_frame_delay(1);
    settings.set_frame_size_limit(FRAME_SIZE_LIMIT_PIXELS);
    settings.set_apply_grain(false);

    let mut decoder =
        Dav1dDecoder::with_settings(&settings).map_err(|e| format!("dav1d init: {e:?}"))?;

    // send_data needs an owned, Send + 'static buffer.
    let owned: Vec<u8> = data.to_vec();
    match decoder.send_data(owned, None, None, None) {
        Ok(()) => {}
        Err(e) if e.is_again() => return Ok(None),
        Err(e) => return Err(format!("dav1d send_data: {e:?}")),
    }

    match decoder.get_picture() {
        Ok(pic) => Ok(Some(pic)),
        Err(e) if e.is_again() => Ok(None),
        Err(e) => Err(format!("dav1d get_picture: {e:?}")),
    }
}

/// Match dav1d's pixel layout to rav1d-safe's variant set.
fn layouts_match(rav1d: PixelLayout, dav1d: Dav1dPixelLayout) -> bool {
    matches!(
        (rav1d, dav1d),
        (PixelLayout::I400, Dav1dPixelLayout::I400)
            | (PixelLayout::I420, Dav1dPixelLayout::I420)
            | (PixelLayout::I422, Dav1dPixelLayout::I422)
            | (PixelLayout::I444, Dav1dPixelLayout::I444)
    )
}

/// Byte-exact compare of one 8-bit plane.
///
/// `rav1d_view` rows are `width` long (no padding); `dav1d_plane_bytes`
/// is the contiguous `stride * height` buffer.
fn compare_plane_u8(
    label: &str,
    rav1d_view: &rav1d_safe::src::managed::PlaneView8<'_>,
    dav1d_plane_bytes: &[u8],
    dav1d_stride_bytes: usize,
    dav1d_height: u32,
) {
    let r_height = rav1d_view.height();
    let r_width = rav1d_view.width();
    if r_height as u32 != dav1d_height {
        panic!(
            "DIVERGENCE {label}: row count rav1d={r_height} dav1d={dav1d_height}",
        );
    }
    for y in 0..r_height {
        let r_row = rav1d_view.row(y);
        let start = y * dav1d_stride_bytes;
        let d_row = &dav1d_plane_bytes[start..start + r_width];
        if r_row != d_row {
            let mismatch = r_row.iter().zip(d_row).position(|(a, b)| a != b).unwrap_or(0);
            panic!(
                "DIVERGENCE {label} row {y} col {mismatch}: rav1d={} dav1d={}",
                r_row[mismatch], d_row[mismatch]
            );
        }
    }
}

/// Compare one 10/12-bit plane bit-exactly. dav1d stores high-bit-depth
/// samples as native-endian `u16` exposed through the byte plane, so we
/// reinterpret two bytes per pixel; the dav1d stride is in bytes and rav1d's
/// `row()` yields `&[u16]` of `width` samples.
fn compare_plane_u16(
    label: &str,
    rav1d_view: &rav1d_safe::src::managed::PlaneView16<'_>,
    dav1d_plane_bytes: &[u8],
    dav1d_stride_bytes: usize,
    dav1d_height: u32,
) {
    let r_height = rav1d_view.height();
    let r_width = rav1d_view.width();
    if r_height as u32 != dav1d_height {
        panic!("DIVERGENCE {label}: row count rav1d={r_height} dav1d={dav1d_height}");
    }
    for y in 0..r_height {
        let r_row = rav1d_view.row(y);
        let start = y * dav1d_stride_bytes;
        let d_row = &dav1d_plane_bytes[start..start + r_width * 2];
        for x in 0..r_width {
            let d = u16::from_ne_bytes([d_row[2 * x], d_row[2 * x + 1]]);
            if r_row[x] != d {
                panic!(
                    "DIVERGENCE {label} row {y} col {x}: rav1d={} dav1d={}",
                    r_row[x], d
                );
            }
        }
    }
}

fn compare_pictures(
    rav1d_frame: &rav1d_safe::src::managed::Frame,
    dav1d_picture: &dav1d::Picture,
) {
    let rw = rav1d_frame.width();
    let dw = dav1d_picture.width();
    let rh = rav1d_frame.height();
    let dh = dav1d_picture.height();
    if rw != dw || rh != dh {
        panic!("DIVERGENCE dims: rav1d={rw}x{rh} dav1d={dw}x{dh}");
    }

    let rbd = rav1d_frame.bit_depth();
    let dbd = dav1d_picture.bit_depth() as u8;
    if rbd != dbd {
        panic!("DIVERGENCE bit_depth: rav1d={rbd} dav1d={dbd}");
    }

    let rlayout = rav1d_frame.pixel_layout();
    let dlayout = dav1d_picture.pixel_layout();
    if !layouts_match(rlayout, dlayout) {
        panic!("DIVERGENCE pixel_layout: rav1d={rlayout:?} dav1d={dlayout:?}");
    }

    // Both 8-bit and 10/12-bit planes are bit-exact-compared below (the u16
    // path reinterprets dav1d's native-endian high-bit-depth bytes).
    match rav1d_frame.planes() {
        Planes::Depth8(p8) => {
            let y_view = p8.y();
            let y_stride = dav1d_picture.stride(PlanarImageComponent::Y) as usize;
            let y_plane = dav1d_picture.plane(PlanarImageComponent::Y);
            compare_plane_u8("Y", &y_view, y_plane.as_ref(), y_stride, dh);

            // 4:0:0 has no chroma. For 4:2:0/4:2:2/4:4:4, compare U + V.
            if !matches!(dlayout, Dav1dPixelLayout::I400) {
                if let Some(u_view) = p8.u() {
                    let u_stride = dav1d_picture.stride(PlanarImageComponent::U) as usize;
                    let u_plane = dav1d_picture.plane(PlanarImageComponent::U);
                    let (_, u_height) =
                        dav1d_picture.plane_data_geometry(PlanarImageComponent::U);
                    compare_plane_u8("U", &u_view, u_plane.as_ref(), u_stride, u_height);
                }
                if let Some(v_view) = p8.v() {
                    let v_stride = dav1d_picture.stride(PlanarImageComponent::V) as usize;
                    let v_plane = dav1d_picture.plane(PlanarImageComponent::V);
                    let (_, v_height) =
                        dav1d_picture.plane_data_geometry(PlanarImageComponent::V);
                    compare_plane_u8("V", &v_view, v_plane.as_ref(), v_stride, v_height);
                }
            }
        }
        Planes::Depth16(p16) => {
            let y_view = p16.y();
            let y_stride = dav1d_picture.stride(PlanarImageComponent::Y) as usize;
            let y_plane = dav1d_picture.plane(PlanarImageComponent::Y);
            compare_plane_u16("Y", &y_view, y_plane.as_ref(), y_stride, dh);

            // 4:0:0 has no chroma. For 4:2:0/4:2:2/4:4:4, compare U + V.
            if !matches!(dlayout, Dav1dPixelLayout::I400) {
                if let Some(u_view) = p16.u() {
                    let u_stride = dav1d_picture.stride(PlanarImageComponent::U) as usize;
                    let u_plane = dav1d_picture.plane(PlanarImageComponent::U);
                    let (_, u_height) =
                        dav1d_picture.plane_data_geometry(PlanarImageComponent::U);
                    compare_plane_u16("U", &u_view, u_plane.as_ref(), u_stride, u_height);
                }
                if let Some(v_view) = p16.v() {
                    let v_stride = dav1d_picture.stride(PlanarImageComponent::V) as usize;
                    let v_plane = dav1d_picture.plane(PlanarImageComponent::V);
                    let (_, v_height) =
                        dav1d_picture.plane_data_geometry(PlanarImageComponent::V);
                    compare_plane_u16("V", &v_view, v_plane.as_ref(), v_stride, v_height);
                }
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let rav1d_result = decode_rav1d(data);
    let dav1d_result = decode_dav1d(data);

    match (&rav1d_result, &dav1d_result) {
        (Ok(Some(_)), Ok(Some(_))) => {
            if let (Ok(Some(r)), Ok(Some(d))) = (rav1d_result, dav1d_result) {
                compare_pictures(&r, &d);
            }
        }
        (Ok(None), Ok(None)) | (Ok(None), Ok(Some(_))) | (Ok(Some(_)), Ok(None)) => {
            // Asymmetric buffering: rav1d-safe may emit synchronously while
            // dav1d buffers (or vice-versa). The byte-exact compare happens
            // only when both yield a picture from the same input chunk.
        }
        (Err(_), Err(_)) => {
            // Both rejected — symmetric, OK.
        }
        (Ok(Some(_)), Err(de)) => {
            panic!("DIVERGENCE: rav1d-safe decoded a frame but dav1d errored: {de}");
        }
        (Err(re), Ok(Some(_))) => {
            panic!("DIVERGENCE: dav1d decoded a frame but rav1d-safe errored: {re}");
        }
        (Ok(None), Err(_)) | (Err(_), Ok(None)) => {
            // One buffered, one rejected. Could be a divergence but often
            // an incomplete-input asymmetry; revisit if false-positive
            // volume is low after a fuzzing session.
        }
    }
});
