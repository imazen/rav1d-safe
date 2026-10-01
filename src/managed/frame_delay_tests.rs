use super::*;

// Committed 1024x1024 still with a 4x8 tile grid (below the large-frame threshold).
const STREAM: &[u8] = include_bytes!("../../tests/crash_vectors/tile_threading_cdef_lpf_race.obu");

fn hash(frame: &Frame) -> String {
    let mut hash = md5::Context::new();
    match frame.planes() {
        Planes::Depth8(p) => {
            for row in p.y().rows() {
                hash.consume(row);
            }
            for plane in [p.u(), p.v()].into_iter().flatten() {
                for row in plane.rows() {
                    hash.consume(row);
                }
            }
        }
        Planes::Depth16(p) => {
            for row in p.y().rows() {
                for px in row {
                    hash.consume(px.to_le_bytes());
                }
            }
            for plane in [p.u(), p.v()].into_iter().flatten() {
                for row in plane.rows() {
                    for px in row {
                        hash.consume(px.to_le_bytes());
                    }
                }
            }
        }
    }
    format!("{:x}", hash.finalize())
}

#[test]
fn auto_frame_delay_follows_frame_size() {
    use crate::src::lib::size_aware_frame_delay as delay;
    // One thread never frame-threads.
    assert_eq!(delay(1, 640, 480), 1);
    assert_eq!(delay(1, 3840, 2160), 1);
    // Frames below ~6 M pixels: two in flight at any thread count (measured to 8).
    for t in [2, 4, 8, 16] {
        assert_eq!(delay(t, 640, 480), 2, "480p t={t}");
        assert_eq!(delay(t, 1920, 1080), 2, "1080p t={t}");
        assert_eq!(delay(t, 2560, 1440), 2, "1440p t={t}");
    }
    // 4K: tile threading alone through 4 threads, a second frame only when it is
    // untracked and tile threading has stalled.
    for t in [2, 4] {
        assert_eq!(delay(t, 3840, 2160), 1, "4K t={t}");
        assert_eq!(delay(t, 3840, 2560), 1, "4K photo t={t}");
    }
    let big = if is_untracked() { 2 } else { 1 };
    assert_eq!(delay(8, 3840, 2160), big);
    assert_eq!(delay(8, 3840, 2560), big);
    // The pixel product must not overflow for the largest legal frames.
    assert_eq!(delay(4, 65536, 65536), 1);
}

#[test]
fn deferred_open_sizes_frame_delay_from_first_sequence_header() {
    // threads > 1 with the delay on auto: the open waits for the frame size.
    let mut settings = Settings::default();
    settings.threads = 4;
    settings.max_frame_delay = 0;
    let mut decoder = Decoder::with_settings(settings).unwrap();
    assert!(decoder.ctx.is_none(), "open is deferred until data arrives");
    // Nothing sent yet: no frames, and the usual calls are harmless.
    assert!(decoder.get_frame().unwrap().is_none());
    decoder.reset();
    assert!(decoder.flush().unwrap().is_empty());
    assert!(decoder.ctx.is_none());

    // The first still is 1024x1024, below the large-frame threshold: two contexts.
    let mut frames: Vec<Frame> = decoder.decode(STREAM).unwrap().into_iter().collect();
    frames.extend(decoder.flush().unwrap());
    assert_eq!(decoder.ctx.as_ref().unwrap().fc.len(), 2);
    assert_eq!(frames.len(), 1);
    assert_eq!(hash(&frames[0]), "51b9c3ab246fda65e2c0a2155588e9a5");

    // A stop token set before the open is applied when it happens.
    let mut settings = Settings::default();
    settings.threads = 4;
    let mut decoder = Decoder::with_settings(settings).unwrap();
    decoder.set_stop(Some(Arc::new(Unstoppable)));
    let mut frames: Vec<Frame> = decoder.decode(STREAM).unwrap().into_iter().collect();
    frames.extend(decoder.flush().unwrap());
    assert_eq!(frames.len(), 1);

    // Data without a readable sequence header falls back to the core's own auto
    // choice instead of failing the open.
    let mut settings = Settings::default();
    settings.threads = 4;
    let mut decoder = Decoder::with_settings(settings).unwrap();
    let _ = decoder.decode(&[0x12, 0x00]);
    assert!(decoder.ctx.is_some());
}

#[test]
fn explicit_or_single_thread_settings_open_immediately() {
    let mut settings = Settings::default();
    settings.threads = 1;
    let decoder = Decoder::with_settings(settings).unwrap();
    assert_eq!(decoder.ctx.as_ref().unwrap().fc.len(), 1);

    let mut settings = Settings::default();
    settings.threads = 4;
    settings.max_frame_delay = 3;
    let decoder = Decoder::with_settings(settings).unwrap();
    assert_eq!(decoder.ctx.as_ref().unwrap().fc.len(), 3);

    let mut settings = Settings::default();
    settings.threads = 4;
    settings.max_frame_delay = 1;
    let decoder = Decoder::with_settings(settings).unwrap();
    assert_eq!(decoder.ctx.as_ref().unwrap().fc.len(), 1);
}
