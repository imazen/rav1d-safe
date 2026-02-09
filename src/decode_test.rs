//! Decode parity test — decodes AV1 bitstreams and produces a hash
//! of the pixel output for parity verification between asm and safe-simd.
//!
//! Usage:
//!   RAV1D_TEST_IVF=/path/to/file.ivf cargo test --lib --release -- decode_test_ivf --nocapture
//!
//! Set RAV1D_TEST_EXPECTED_HASH to verify against a known-good hash.
//!
//! Note: Requires feature "c-ffi" to access the dav1d_* API, OR uses the
//! internal rav1d_* API when c-ffi is not enabled.

#[cfg(test)]
mod tests {
    use crate::include::common::bitdepth::BitDepth8;
    use crate::include::dav1d::data::Rav1dData;
    use crate::include::dav1d::dav1d::Rav1dSettings;
    use crate::include::dav1d::headers::Rav1dPixelLayout;
    use crate::include::dav1d::picture::Rav1dPicture;
    use crate::src::c_arc::CArc;
    use crate::src::c_box::CBox;
    use crate::src::error::Rav1dError;
    use crate::src::lib::rav1d_close;
    use crate::src::lib::rav1d_get_picture;
    use crate::src::lib::rav1d_open;
    use crate::src::lib::rav1d_send_data;
    use std::env;
    use std::fs;

    /// Parse an IVF file and return a Vec of frame data payloads.
    fn parse_ivf(data: &[u8]) -> Vec<Vec<u8>> {
        assert!(data.len() >= 32, "IVF file too short for header");
        assert_eq!(&data[0..4], b"DKIF", "Not an IVF file");
        assert_eq!(&data[8..12], b"AV01", "Not an AV1 IVF file");

        let mut frames = Vec::new();
        let mut pos = 32;

        while pos + 12 <= data.len() {
            let frame_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 12; // skip size (4) + timestamp (8)

            if pos + frame_size > data.len() {
                break;
            }

            frames.push(data[pos..pos + frame_size].to_vec());
            pos += frame_size;
        }

        frames
    }

    /// FNV-1a 128-bit hash for fast pixel output comparison.
    fn hash_bytes(data: &[u8]) -> String {
        let mut h1: u64 = 0xcbf29ce484222325;
        let mut h2: u64 = 0x6c62272e07bb0142;
        for &b in data {
            h1 ^= b as u64;
            h1 = h1.wrapping_mul(0x01000193);
            h2 ^= b as u64;
            h2 = h2.wrapping_mul(0x00000100000001b3);
        }
        format!("{:016x}{:016x}", h1, h2)
    }

    fn decode_ivf_to_hash(ivf_path: &str) -> (String, usize) {
        let ivf_data = fs::read(ivf_path)
            .unwrap_or_else(|e| panic!("Failed to read IVF file '{}': {}", ivf_path, e));
        let frames = parse_ivf(&ivf_data);
        assert!(!frames.is_empty(), "No frames found in IVF file");

        // Use single-threaded mode to avoid deadlocks in test context
        // and ensure deterministic output
        let settings = Rav1dSettings {
            n_threads: 1,
            max_frame_delay: 1,
            apply_grain: false, // Disable grain for deterministic pixel comparison
            ..Default::default()
        };
        let (ctx, _handles) = rav1d_open(&settings).expect("Failed to open decoder");

        let mut pixel_data: Vec<u8> = Vec::new();
        let mut frame_count = 0usize;

        for obu_data in &frames {
            let boxed: Box<[u8]> = obu_data.clone().into_boxed_slice();
            let carc = CArc::wrap(CBox::from_box(boxed)).expect("CArc::wrap failed");
            let mut data = Rav1dData::from(carc);

            // Send data, draining pictures as needed
            loop {
                match rav1d_send_data(&ctx, &mut data) {
                    Ok(()) => break,
                    Err(Rav1dError::EAGAIN) => {
                        // Drain a picture before retrying
                        let mut pic = Rav1dPicture::default();
                        match rav1d_get_picture(&ctx, &mut pic) {
                            Ok(()) => {
                                extract_pixels(&pic, &mut pixel_data);
                                frame_count += 1;
                            }
                            Err(Rav1dError::EAGAIN) => {}
                            Err(e) => panic!("rav1d_get_picture error during drain: {:?}", e),
                        }
                    }
                    Err(e) => panic!("rav1d_send_data error: {:?}", e),
                }
            }

            // Try to get any ready pictures
            loop {
                let mut pic = Rav1dPicture::default();
                match rav1d_get_picture(&ctx, &mut pic) {
                    Ok(()) => {
                        extract_pixels(&pic, &mut pixel_data);
                        frame_count += 1;
                    }
                    Err(Rav1dError::EAGAIN) => break,
                    Err(e) => panic!("rav1d_get_picture error: {:?}", e),
                }
            }
        }

        // Flush: send empty data to signal end-of-stream, then drain
        {
            let mut empty = Rav1dData::default();
            let _ = rav1d_send_data(&ctx, &mut empty);
        }
        loop {
            let mut pic = Rav1dPicture::default();
            match rav1d_get_picture(&ctx, &mut pic) {
                Ok(()) => {
                    extract_pixels(&pic, &mut pixel_data);
                    frame_count += 1;
                }
                Err(Rav1dError::EAGAIN) => break,
                Err(e) => panic!("rav1d_get_picture flush error: {:?}", e),
            }
        }

        rav1d_close(ctx);

        let hash = hash_bytes(&pixel_data);
        (hash, frame_count)
    }

    fn extract_pixels(pic: &Rav1dPicture, output: &mut Vec<u8>) {
        let w = pic.p.w as usize;
        let h = pic.p.h as usize;
        let bpc = pic.p.bpc;

        let data = pic.data.as_ref().expect("No picture data");
        let strides = pic.stride;

        let n_planes: usize = match pic.p.layout {
            Rav1dPixelLayout::I400 => 1,
            _ => 3,
        };

        for plane in 0..n_planes {
            let component = &data.data[plane];
            let stride = if plane == 0 { strides[0] } else { strides[1] };
            let abs_stride = stride.unsigned_abs();

            let (plane_w, plane_h) = if plane == 0 {
                (w, h)
            } else {
                match pic.p.layout {
                    Rav1dPixelLayout::I420 => ((w + 1) / 2, (h + 1) / 2),
                    Rav1dPixelLayout::I422 => ((w + 1) / 2, h),
                    Rav1dPixelLayout::I444 => (w, h),
                    Rav1dPixelLayout::I400 => unreachable!(),
                }
            };

            let bytes_per_pixel = if bpc > 8 { 2usize } else { 1 };
            let row_bytes = plane_w * bytes_per_pixel;

            let byte_len = component.byte_len();
            let guard = component.slice::<BitDepth8, _>((0.., ..byte_len));
            let all_bytes: &[u8] = &guard;

            let start = if stride < 0 { byte_len - abs_stride } else { 0 };

            for row in 0..plane_h {
                let row_start = if stride < 0 {
                    start.wrapping_sub(row * abs_stride)
                } else {
                    start + row * abs_stride
                };

                let row_slice = &all_bytes[row_start..row_start + row_bytes];
                output.extend_from_slice(row_slice);
            }
        }
    }

    #[test]
    fn decode_test_ivf() {
        let ivf_path = match env::var("RAV1D_TEST_IVF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("RAV1D_TEST_IVF not set, skipping decode test");
                return;
            }
        };

        let (hash, frame_count) = decode_ivf_to_hash(&ivf_path);

        #[cfg(feature = "asm")]
        let mode = "asm";
        #[cfg(not(feature = "asm"))]
        let mode = "safe-simd";

        eprintln!("Decoded {} frames in {} mode", frame_count, mode);
        eprintln!("Pixel hash: {}", hash);

        if let Ok(expected) = env::var("RAV1D_TEST_EXPECTED_HASH") {
            assert_eq!(
                hash, expected,
                "Parity mismatch! {} mode produced different output.\n\
                 Expected: {}\n\
                 Got:      {}",
                mode, expected, hash
            );
            eprintln!("Parity verified: {} matches expected hash", mode);
        }
    }
}
