use super::*;

// Committed 1024x1024 still with a 4x8 tile grid.
const STREAM: &[u8] = include_bytes!("../../tests/crash_vectors/tile_threading_cdef_lpf_race.obu");

fn assert_picture_policy(frame: &Frame, threads: u32) {
    for plane in &frame.inner.data.as_ref().unwrap().data {
        let policy = plane
            .threading_policy()
            .expect("decoder-owned picture policy");
        assert_eq!(policy.parallel, threads > 1);
        assert_eq!(plane.uses_row_guards(), threads > 1);
    }
}

#[test]
fn picture_policy_is_local_and_survives_decoder_lifetimes() {
    let mut retained = Vec::new();
    for order in [[8, 1], [1, 8]] {
        let mut decoders: Vec<_> = order
            .into_iter()
            .map(|threads| {
                let mut settings = Settings::default();
                settings.threads = threads;
                settings.max_frame_delay = 1;
                Decoder::with_settings(settings).unwrap()
            })
            .collect();
        for _ in 0..2 {
            for (decoder, threads) in decoders.iter_mut().zip(order) {
                let frame = decoder.decode(STREAM).unwrap().expect("still frame");
                assert_picture_policy(&frame, threads);
                assert_eq!(hash(&frame), "51b9c3ab246fda65e2c0a2155588e9a5");
                retained.push((frame, threads));
            }
        }
        drop(decoders);
        for (frame, threads) in &retained {
            assert_picture_policy(frame, *threads);
            let mut copy = Rav1dPicture::default();
            crate::src::picture::rav1d_picture_alloc_copy(
                &None,
                &mut copy,
                frame.inner.p.w,
                &frame.inner,
            )
            .unwrap();
            assert_picture_policy(&Frame { inner: copy }, *threads);
        }
    }
}

#[test]
fn copied_picture_can_allocate_after_decoder_and_source_drop() {
    let mut settings = Settings::default();
    settings.threads = 4;
    settings.max_frame_delay = 1;
    let mut decoder = Decoder::with_settings(settings).unwrap();
    let mut frame = decoder.decode(STREAM).unwrap().expect("still frame");
    drop(decoder);

    // Each generation must own enough allocator state to allocate the next.
    // Retaining a pointer into either the decoder or the previous picture
    // would leave a dangling cookie after these explicit drops.
    for _ in 0..16 {
        let mut copy = Rav1dPicture::default();
        crate::src::picture::rav1d_picture_alloc_copy(
            &None,
            &mut copy,
            frame.inner.p.w,
            &frame.inner,
        )
        .unwrap();
        drop(frame);
        frame = Frame { inner: copy };
        assert_picture_policy(&frame, 4);
    }
}

#[test]
fn serial_and_threaded_decoders_run_concurrently_with_local_policies() {
    let (a_tx, a_rx) = std::sync::mpsc::sync_channel(1);
    let (b_tx, b_rx) = std::sync::mpsc::sync_channel(1);
    let retained = std::thread::scope(|scope| {
        let handles: Vec<_> = [(1, a_tx, b_rx), (8, b_tx, a_rx)]
            .into_iter()
            .map(|(threads, tx, rx)| {
                scope.spawn(move || {
                    let mut settings = Settings::default();
                    settings.threads = threads;
                    settings.max_frame_delay = 1;
                    let mut decoder = Decoder::with_settings(settings).unwrap();
                    let mut retained = Vec::new();
                    for _ in 0..8 {
                        tx.send(()).unwrap();
                        rx.recv_timeout(std::time::Duration::from_secs(10))
                            .expect("peer decoder made no progress");
                        let frame = decoder.decode(STREAM).unwrap().expect("still frame");
                        assert_picture_policy(&frame, threads);
                        assert_eq!(hash(&frame), "51b9c3ab246fda65e2c0a2155588e9a5");
                        retained.push((frame, threads));
                    }
                    retained
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(retained.len(), 16);
    for (frame, threads) in &retained {
        assert_picture_policy(frame, *threads);
    }
}

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
