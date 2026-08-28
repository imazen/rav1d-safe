//! `Settings::strictness` — the decoder's conformance policy (#422, #424).
//!
//! Two real non-conforming streams that the AV1 reference decoder (libaom's
//! `aomdec`) rejects and that dav1d's library default decodes to garbage:
//!
//! * `segment_id_desync_zenrav1e35.obu` — zenrav1e#35 (found via zenavif#29): the
//!   encoder omitted the chroma TUs of `BLOCK_16X4` blocks in 4:2:0, so the symbol
//!   stream desynchronises at the first such block. aomdec: "Corrupted
//!   segment_ids". 192x256 4:2:0 8-bit, one frame, 4267 bytes. rav1d-safe#422.
//! * `tile_padding_desync_fuzz424.obu` — the `differential_dav1d` farm artifact
//!   behind rav1d-safe#424 (sha256 `d204fcaa…` in block storage), cut after its
//!   frame OBU (the original carries a garbage fourth OBU that even lenient
//!   dav1d rejects). aomdec: "Failed to decode tile data". 10-bit; dav1d and
//!   rav1d-safe decode it to *different* pixels because both are concealing
//!   corrupt tile data whose decoding the spec leaves undefined.
//!
//! Both verified 2026-08-28 against aomdec (libaom 632172a) and the dav1d 1.5.3
//! CLI: `--strict 0` decodes 1/1 frames, `--strict 1` rejects. The record is
//! `benchmarks/strictness_2026-08-28.meta`.
use rav1d_safe::src::managed::{Decoder, Error, Frame, Planes, Settings, Strictness};

const SEG_ID_DESYNC: &[u8] = include_bytes!("strictness_vectors/segment_id_desync_zenrav1e35.obu");
const TILE_PADDING_DESYNC: &[u8] =
    include_bytes!("strictness_vectors/tile_padding_desync_fuzz424.obu");
/// A conforming encode (rav1e output of kodim03), used as the control.
const CONFORMING: &[u8] = include_bytes!("crash_vectors/kodim03_yuv420_8bpc.obu");

/// Decode a one-frame stream to its frame, or the error the decoder reported.
fn decode_first(settings: Settings, data: &[u8]) -> Result<Frame, Error> {
    let mut decoder = Decoder::with_settings(settings).expect("decoder init");
    match decoder.decode(data) {
        Ok(Some(frame)) => Ok(frame),
        Ok(None) => match decoder.flush() {
            Ok(mut frames) if !frames.is_empty() => Ok(frames.remove(0)),
            Ok(_) => panic!("stream produced no frame and no error"),
            Err(e) => Err(e.decompose().0),
        },
        Err(e) => Err(e.decompose().0),
    }
}

fn with(strictness: Strictness) -> Settings {
    let mut settings = Settings::default();
    settings.strictness = strictness;
    settings
}

fn luma_bytes(frame: &Frame) -> Vec<u8> {
    match frame.planes() {
        Planes::Depth8(p) => p.y().as_slice().to_vec(),
        Planes::Depth16(p) => p
            .y()
            .as_slice()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    }
}

#[test]
fn lenient_is_below_strict_and_strict_is_the_default() {
    assert!(Strictness::Lenient < Strictness::Strict);
    assert_eq!(Strictness::default(), Strictness::Strict);
    assert_eq!(Settings::default().strictness, Strictness::Strict);
}

#[test]
fn desynced_segment_ids_strict_rejects_lenient_conceals() {
    let frame = decode_first(with(Strictness::Lenient), SEG_ID_DESYNC)
        .expect("Lenient must decode the #422 stream like dav1d does");
    assert_eq!((frame.width(), frame.height()), (192, 256));

    let err = decode_first(with(Strictness::Strict), SEG_ID_DESYNC)
        .err()
        .expect("Strict must reject a segment_id outside 0..=LastActiveSegId");
    assert!(matches!(err, Error::InvalidData), "got {err:?}");

    // The default settings are Strict, so `Decoder::new()` rejects it too.
    let err = decode_first(Settings::default(), SEG_ID_DESYNC)
        .err()
        .expect("default settings must reject the #422 stream");
    assert!(matches!(err, Error::InvalidData), "got {err:?}");
}

#[test]
fn corrupt_tile_padding_strict_rejects_lenient_conceals() {
    let frame = decode_first(with(Strictness::Lenient), TILE_PADDING_DESYNC)
        .expect("Lenient must decode the #424 stream like dav1d does");
    assert_eq!(frame.bit_depth(), 10);

    let err = decode_first(with(Strictness::Strict), TILE_PADDING_DESYNC)
        .err()
        .expect("Strict must reject tile data the reference decoder rejects");
    assert!(matches!(err, Error::InvalidData), "got {err:?}");
}

#[test]
fn conforming_stream_is_identical_under_both() {
    let lenient = decode_first(with(Strictness::Lenient), CONFORMING).expect("lenient decode");
    let strict = decode_first(with(Strictness::Strict), CONFORMING).expect("strict decode");
    assert_eq!(
        (lenient.width(), lenient.height()),
        (strict.width(), strict.height())
    );
    assert_eq!(luma_bytes(&lenient), luma_bytes(&strict));
}

#[test]
#[allow(deprecated)]
fn deprecated_strict_std_compliance_still_means_strict() {
    let mut settings = with(Strictness::Lenient);
    settings.strict_std_compliance = true;
    let err = decode_first(settings, SEG_ID_DESYNC)
        .err()
        .expect("strict_std_compliance = true must behave as Strictness::Strict");
    assert!(matches!(err, Error::InvalidData), "got {err:?}");
}
