use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};

fn hash_frame(frame: &Frame, ctx: &mut md5::Context) {
    match frame.planes() {
        Planes::Depth8(p) => {
            for row in p.y().rows() {
                ctx.consume(row);
            }
            if let Some(u) = p.u() {
                for row in u.rows() {
                    ctx.consume(row);
                }
            }
            if let Some(v) = p.v() {
                for row in v.rows() {
                    ctx.consume(row);
                }
            }
        }
        Planes::Depth16(p) => {
            for row in p.y().rows() {
                for &px in row {
                    ctx.consume(px.to_le_bytes());
                }
            }
            if let Some(u) = p.u() {
                for row in u.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
            if let Some(v) = p.v() {
                for row in v.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
        }
    }
}

#[test]
fn published_decoder_accepts_the_patch_and_matches_reference_pixels() {
    let vectors: &[(&str, &[u8], &str)] = &[
        (
            "kodim03",
            include_bytes!("../../../../tests/crash_vectors/kodim03_yuv420_8bpc.obu"),
            "f7de1083a1166170f8ae1f79328f275a",
        ),
        (
            "alpha",
            include_bytes!("../../../../tests/crash_vectors/alpha_noispe.obu"),
            "c8863ea13a56b1ae731cdd23bcef40c8",
        ),
        (
            "hdr",
            include_bytes!("../../../../tests/crash_vectors/colors_hdr_rec2020.obu"),
            "d9c0ea6b0213b64132a65d3a7e76edf4",
        ),
        (
            "circle",
            include_bytes!("../../../../tests/crash_vectors/circle_custom_properties.obu"),
            "bd06968f3606982bb9c398ad6f7f41c2",
        ),
    ];
    for threads in [1, 4] {
        for &(name, bytes, reference) in vectors {
            let mut settings = Settings::default();
            settings.threads = threads;
            settings.max_frame_delay = 1;
            let mut decoder = Decoder::with_settings(settings).unwrap();
            let mut hash = md5::Context::new();
            let mut count = 0;
            if let Some(frame) = decoder.decode(bytes).unwrap() {
                hash_frame(&frame, &mut hash);
                count += 1;
            }
            for frame in decoder.flush().unwrap() {
                hash_frame(&frame, &mut hash);
                count += 1;
            }
            assert!(count > 0, "{name} t={threads}: no frames decoded");
            assert_eq!(
                format!("{:x}", hash.finalize()),
                reference,
                "{name} t={threads}"
            );
        }
    }
}
