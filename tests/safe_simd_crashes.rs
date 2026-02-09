//! Reproduction tests for safe_simd panics found via zenavif AVIF corpus.
//!
//! Each test decodes a raw AV1 OBU bitstream extracted from a real AVIF file.
//! These should NOT panic — panics indicate bounds-check bugs in safe SIMD code.
//!
//! Run: cargo test --test safe_simd_crashes

use rav1d_safe::src::managed::Decoder;

fn decode_obu(data: &[u8]) -> Result<(), String> {
    let mut decoder = Decoder::new().map_err(|e| format!("create: {e:?}"))?;
    match decoder.decode(data) {
        Ok(Some(frame)) => {
            eprintln!(
                "  Decoded: {}x{} @ {}bpc",
                frame.width(),
                frame.height(),
                frame.bit_depth()
            );
            Ok(())
        }
        Ok(None) => {
            // Flush to get any remaining frames
            match decoder.flush() {
                Ok(frames) if !frames.is_empty() => {
                    for frame in &frames {
                        eprintln!(
                            "  Flushed: {}x{} @ {}bpc",
                            frame.width(),
                            frame.height(),
                            frame.bit_depth()
                        );
                    }
                    Ok(())
                }
                Ok(_) => Err("no frame produced".into()),
                Err(e) => Err(format!("flush: {e:?}")),
            }
        }
        Err(e) => Err(format!("decode: {e:?}")),
    }
}

/// loopfilter.rs:1345 - slice_as byte-vs-element offset bug
/// "range end index 4108 out of range for slice of length 4099"
#[test]
fn loopfilter_slice_bounds_alpha_noispe() {
    let data = include_bytes!("crash_vectors/alpha_noispe.obu");
    let _ = decode_obu(data);
}

/// ipred.rs:1473 - topleft index out of bounds in z2 prediction
/// "index out of bounds: the len is 257 but the index is 257"
#[test]
fn ipred_z2_bounds_kodim03() {
    let data = include_bytes!("crash_vectors/kodim03_yuv420_8bpc.obu");
    let _ = decode_obu(data);
}

/// ipred.rs:1473 - topleft index out of bounds in z2 prediction
/// "index out of bounds: the len is 257 but the index is 264"
#[test]
fn ipred_z2_bounds_circle() {
    let data = include_bytes!("crash_vectors/circle_custom_properties.obu");
    let _ = decode_obu(data);
}

/// range start index out of range (likely ipred or itx)
/// "range start index 528 out of range for slice of length 514"
#[test]
fn range_start_bounds_colors_hdr() {
    let data = include_bytes!("crash_vectors/colors_hdr_rec2020.obu");
    let _ = decode_obu(data);
}

/// rav1d-disjoint-mut/src/lib.rs:1315 - PicBuf never allocated
/// "PicBuf: aligned region (63 + 0) exceeds Vec length (0)"
/// Affects 45/3261 real-world AVIF files (google-native + unsplash corpus)
#[test]
fn picbuf_empty_vec() {
    let data = include_bytes!("crash_vectors/picbuf_empty_vec.obu");
    let _ = decode_obu(data);
}
