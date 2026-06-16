//! Reproduction tests for safe_simd panics found via zenavif AVIF corpus.
//!
//! Each test decodes a raw AV1 OBU bitstream extracted from a real AVIF file.
//! These should NOT panic — panics indicate bounds-check bugs in safe SIMD code.
//!
//! **Requires `--release`** — debug mode is too slow for decode tests.
//!
//! Run: cargo test --release --test safe_simd_crashes

#[cfg(debug_assertions)]
compile_error!("safe_simd_crashes tests require release mode: cargo test --release");

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

// ----------------------------------------------------------------------------
// ARM aarch64 NEON-path SGR/MC panics (found via the Hetzner fuzz farm, arm64).
//
// These ONLY panic when the decoder runs the aarch64 safe_simd paths
// (looprestoration_arm.rs / mc_arm.rs). On x86_64 the equivalent x86 scalar
// bodies are in-bounds, so these vectors decode cleanly there — verify the fix
// by cross-compiling and running under qemu-aarch64-static (see
// `just test-aarch64` / the CARGO_TARGET_AARCH64_*_RUNNER env).
// ----------------------------------------------------------------------------

/// looprestoration_arm.rs:399 - boxsum3_8bpc vertical-pass OOB write.
/// The ARM scalar port wrote `out_idx = y*REST_UNIT_STRIDE + x` for y up to
/// h-2 = 68, reaching index 68*390 = 26520, past the 26520-element buffer.
/// "index out of bounds: the len is 26520 but the index is 26520"
/// Fixed by replacing the ARM body with the in-bounds x86 scalar boxsum3_8bpc.
#[test]
fn arm_boxsum3_oob_8bpc() {
    let data = include_bytes!("crash_vectors/arm_boxsum3_oob_8bpc.obu");
    let _ = decode_obu(data);
}

/// looprestoration_arm.rs:935 - boxsum3_16bpc vertical-pass OOB write.
/// Same structural bug as the 8bpc variant, in the 16bpc box-sum.
/// "index out of bounds: the len is 26520 but the index is 26520"
/// Fixed by replacing the ARM body with the in-bounds x86 scalar boxsum3_16bpc.
#[test]
fn arm_boxsum3_oob_16bpc() {
    let data = include_bytes!("crash_vectors/arm_boxsum3_oob_16bpc.obu");
    let _ = decode_obu(data);
}

/// looprestoration_arm.rs:465 - selfguided_filter_8bpc aa_base underflow.
/// At row_offset==0, `(0 as isize - 1) as usize` is usize::MAX; the subsequent
/// `row_start * REST_UNIT_STRIDE` multiply overflows in checked arithmetic.
/// "attempt to multiply with overflow"
/// Fixed by using the x86 aa_base = (row_offset + 1)*REST_UNIT_STRIDE + 2 form.
#[test]
fn arm_aa_base_underflow_8bpc() {
    let data = include_bytes!("crash_vectors/arm_aa_base_underflow_8bpc.obu");
    let _ = decode_obu(data);
}

/// looprestoration_arm.rs:999 - selfguided_filter_16bpc aa_base underflow.
/// Same underflow as the 8bpc variant, in the 16bpc self-guided filter.
/// "attempt to multiply with overflow"
/// Fixed by using the x86 aa_base = (row_offset + 1)*REST_UNIT_STRIDE + 2 form.
#[test]
fn arm_aa_base_underflow_16bpc() {
    let data = include_bytes!("crash_vectors/arm_aa_base_underflow_16bpc.obu");
    let _ = decode_obu(data);
}

/// mc_arm.rs:5930 - 16bpc MC-put dst slice overshoot.
/// The dst guard is sized (h-1)*stride + w (last row needs only w), but the
/// 16bpc branch sliced h*stride + w, overshooting by stride - w.
/// "range end index 10784 out of range for slice of length 8096"
/// Fixed by computing dst_byte_len from h.saturating_sub(1).
#[test]
fn arm_mc16_overshoot() {
    let data = include_bytes!("crash_vectors/arm_mc16_overshoot.obu");
    let _ = decode_obu(data);
}
