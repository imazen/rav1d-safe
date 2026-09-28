//! Exercise AV1 metadata parsing and the managed physical-unit accessors together.

use rav1d_safe::Decoder;

#[test]
fn parsed_av1_mastering_display_uses_av1_fixed_point_units() {
    // This committed stream starts with a temporal delimiter (2 bytes), then
    // a sequence header (11 bytes). Insert metadata before its frame OBU.
    let source = include_bytes!("crash_vectors/colors_hdr_rec2020.obu");
    assert_eq!(&source[..3], &[0x12, 0, 0x0a]);
    assert_eq!(source[3], 9);
    assert_eq!(source[13] >> 3, 6);

    // metadata_type = HDR_MDCV (2); the AV1 payload uses big-endian bit fields:
    // chromaticities are unsigned 0.16, max luminance 24.8, min luminance 18.14.
    // Values are chosen to have exactly representable physical interpretations.
    let mut metadata = vec![0x2a, 26, 2];
    for component in [32768_u16, 16384, 8192, 49152, 4096, 2048, 20480, 21504] {
        metadata.extend_from_slice(&component.to_be_bytes());
    }
    metadata.extend_from_slice(&256000_u32.to_be_bytes());
    metadata.extend_from_slice(&16384_u32.to_be_bytes());
    metadata.push(0x80); // trailing_one_bit and byte alignment

    let mut stream = source[..13].to_vec();
    stream.extend_from_slice(&metadata);
    stream.extend_from_slice(&source[13..]);
    let mut decoder = Decoder::new().unwrap();
    let frame = decoder.decode(&stream).unwrap().unwrap();
    let mastering = frame
        .mastering_display()
        .expect("metadata reaches the frame");

    // Raw fields remain AV1 integers; conversion is performed only by helpers.
    assert_eq!(mastering.max_luminance, 256000);
    assert_eq!(mastering.min_luminance, 16384);
    assert_eq!(mastering.primaries[0], [32768, 16384]);
    assert_eq!(mastering.max_luminance_nits(), 1000.0);
    assert_eq!(mastering.min_luminance_nits(), 1.0);
    assert_eq!(mastering.primary_chromaticity(0), [0.5, 0.25]);
    assert_eq!(mastering.primary_chromaticity(1), [0.125, 0.75]);
    assert_eq!(mastering.primary_chromaticity(2), [0.0625, 0.03125]);
    assert_eq!(mastering.white_point_chromaticity(), [0.3125, 0.328125]);
}
