use super::*;

// Committed 1024x1024 still with a 4x8 tile grid.
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
fn parallel_frame_tile_contexts_preserve_frames() {
    let mut serial = Decoder::new().unwrap();
    let reference = serial.decode(STREAM).unwrap().expect("single still frame");
    let header = &reference.inner.frame_hdr.as_ref().unwrap().tiling;
    eprintln!("fixture tiles={}x{}", header.cols, header.rows);
    assert!(
        header.cols as usize * header.rows as usize > 1,
        "fixture must have multiple tiles"
    );
    let reference = hash(&reference);
    assert_eq!(
        &STREAM[..2],
        &[0x12, 0],
        "temporal delimiter required for repetition"
    );
    let mut modes = vec![(8, 1)];
    if cfg!(feature = "unchecked") {
        modes.extend([(8, 2), (8, 4)]);
    }
    for (threads, max_frame_delay) in modes {
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = max_frame_delay;
        let mut decoder = Decoder::with_settings(settings).unwrap();
        assert_eq!(decoder.ctx.fc.len(), max_frame_delay as usize);
        assert_eq!(decoder.ctx.tc.len(), threads as usize);
        let mut output = Vec::new();
        for _ in 0..12 {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                assert!(
                    std::time::Instant::now() < deadline,
                    "input backpressure did not clear"
                );
                match decoder.decode(STREAM) {
                    Ok(frame) => {
                        output.extend(frame);
                        break;
                    }
                    Err(e) if matches!(e.error(), Error::NeedMoreData) => {
                        output.extend(decoder.get_frame().unwrap());
                    }
                    Err(e) => panic!("decode: {e}"),
                }
            }
        }
        output.extend(decoder.flush().unwrap());
        assert_eq!(output.len(), 12, "frame contexts={max_frame_delay}");
        for frame in &output {
            assert_eq!(hash(frame), reference, "frame contexts={max_frame_delay}");
        }
        eprintln!("workers={threads} frame_contexts={max_frame_delay}: all 12 frame hashes match");
    }
}
