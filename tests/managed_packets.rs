//! Streaming ownership and timing, exercised against original lossless AV1.
use rav1d_safe::{Decoder, Frame, Packet, Planes, ReceiveStatus, SendStatus, Settings};
use std::sync::Arc;

#[test]
fn packet_retains_its_public_auto_traits() {
    fn require<T: Send + Sync + std::panic::RefUnwindSafe + std::panic::UnwindSafe>() {}
    require::<Packet>();
}

fn temporal_units() -> Vec<Vec<u8>> {
    let data = include_bytes!("media_vectors/av1-10-420-full-17x13.ivf");
    assert_eq!(&data[..4], b"DKIF");
    let mut cursor = 32;
    let mut units = Vec::new();
    while cursor < data.len() {
        let len = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 12;
        units.push(data[cursor..cursor + len].to_vec());
        cursor += len;
    }
    assert_eq!(units.len(), 4);
    units
}

fn available(decoder: &mut Decoder, frames: &mut Vec<Frame>) {
    for _ in 0..16 {
        match decoder.receive().unwrap() {
            ReceiveStatus::Frame(frame) => frames.push(frame),
            ReceiveStatus::NeedInput => return,
            _ => panic!("unexpected terminal state before end_input"),
        }
    }
    panic!("unbounded output from four-frame fixture");
}

fn finish(decoder: &mut Decoder, frames: &mut Vec<Frame>) {
    decoder.end_input();
    decoder.end_input();
    for _ in 0..16 {
        match decoder.receive().unwrap() {
            ReceiveStatus::Frame(frame) => frames.push(frame),
            ReceiveStatus::EndOfStream => {
                assert!(matches!(
                    decoder.receive().unwrap(),
                    ReceiveStatus::EndOfStream
                ));
                return;
            }
            _ => panic!("ended input must not request more packets"),
        }
    }
    panic!("unbounded output from four-frame fixture");
}

fn check_pixels(frame: &Frame, index: usize) {
    assert_eq!((frame.width(), frame.height()), (17, 13));
    assert_eq!(frame.render_size(), (17, 13));
    assert_eq!(frame.bit_depth(), 10);
    let Planes::Depth16(planes) = frame.planes() else {
        panic!("wrong storage")
    };
    for (component, plane) in [planes.y(), planes.u().unwrap(), planes.v().unwrap()]
        .iter()
        .enumerate()
    {
        for (y, row) in plane.rows().enumerate() {
            for (x, &actual) in row.iter().enumerate() {
                let level = if x == 0 && y == 0 {
                    if index.is_multiple_of(2) { 0 } else { 256 }
                } else {
                    (17 * x + 29 * y + 43 * index + 71 * component) % 257
                };
                assert_eq!(
                    actual,
                    ((1023 * level + 128) / 256) as u16,
                    "frame {index}, component {component}, ({x},{y})"
                );
            }
        }
    }
}

#[test]
fn packets_preserve_timing_and_native_pixels_through_decoder_drop() {
    for threads in [1, 4] {
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 4;
        settings.apply_grain = false;
        let mut decoder = Decoder::with_settings(settings).unwrap();
        assert!(matches!(
            decoder.receive().unwrap(),
            ReceiveStatus::NeedInput
        ));
        let mut frames = Vec::new();
        for (i, unit) in temporal_units().into_iter().enumerate() {
            let mut packet = Packet::new(unit)
                .unwrap()
                .with_timestamp(-100 + 7 * i as i64)
                .with_duration(7)
                .with_offset(1000 + i as i64);
            for attempt in 0..16 {
                match decoder.send_packet(&mut packet).unwrap() {
                    SendStatus::Accepted => break,
                    SendStatus::ReceivePending => available(&mut decoder, &mut frames),
                    _ => panic!("unrecognized submit result"),
                }
                assert!(attempt < 15, "no submission progress");
            }
            assert!(packet.is_empty());
            assert_eq!(packet.len(), 0);
            assert!(decoder.send_packet(&mut packet).is_err());
            available(&mut decoder, &mut frames);
        }
        finish(&mut decoder, &mut frames);
        let mut late = Packet::new(temporal_units().remove(0)).unwrap();
        let late_len = late.len();
        assert!(decoder.send_packet(&mut late).is_err());
        assert_eq!(late.len(), late_len);
        assert!(decoder.decode(&[]).is_err());
        drop(decoder);
        assert_eq!(frames.len(), 4);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.timestamp(), -100 + 7 * i as i64);
            assert_eq!(frame.duration(), 7);
            assert_eq!(frame.input_offset(), 1000 + i as i64);
            let color = frame.raw_color_info();
            assert_eq!(
                (
                    color.primaries,
                    color.transfer_characteristics,
                    color.matrix_coefficients
                ),
                (1, 1, 1)
            );
            assert!(color.full_range);
            assert_eq!(color.chroma_sample_position, 0);
            check_pixels(frame, i);
        }
    }
}

#[test]
fn backpressure_keeps_packet_and_reset_discards_pending_output() {
    let units = temporal_units();
    let mut decoder = Decoder::new().unwrap();
    let mut joined = Packet::new(units.concat()).unwrap().with_timestamp(123);
    assert_eq!(
        decoder.send_packet(&mut joined).unwrap(),
        SendStatus::Accepted
    );
    let mut retry = Packet::new(units[0].clone()).unwrap().with_timestamp(456);
    let size = retry.len();
    assert_eq!(
        decoder.send_packet(&mut retry).unwrap(),
        SendStatus::ReceivePending
    );
    assert_eq!(retry.len(), size);
    let retained = match decoder.receive().unwrap() {
        ReceiveStatus::Frame(frame) => frame,
        _ => panic!("expected first picture"),
    };
    decoder.reset();
    assert!(matches!(
        decoder.receive().unwrap(),
        ReceiveStatus::NeedInput
    ));
    assert_eq!(
        decoder.send_packet(&mut retry).unwrap(),
        SendStatus::Accepted
    );
    let mut frames = Vec::new();
    finish(&mut decoder, &mut frames);
    assert_eq!(
        frames.len(),
        1,
        "reset must discard the earlier pending pictures"
    );
    assert_eq!(retained.timestamp(), 123);
    assert_eq!(frames[0].timestamp(), 456);
    check_pixels(&retained, 0);
    check_pixels(&frames[0], 0);
    decoder.reset();
    assert!(matches!(
        decoder.receive().unwrap(),
        ReceiveStatus::NeedInput
    ));
}

#[test]
fn empty_stream_ends_without_a_frame_and_empty_packets_are_rejected() {
    assert!(Packet::new(Vec::new()).is_err());
    let mut decoder = Decoder::new().unwrap();
    let mut frames = Vec::new();
    finish(&mut decoder, &mut frames);
    assert!(frames.is_empty());
    decoder.reset();
    assert!(decoder.decode(&[]).unwrap().is_none());
}

#[cfg(not(feature = "c-ffi"))]
#[test]
fn picture_budget_applies_to_native_decode_before_publication() {
    for threads in [1, 4] {
        for limit in [0, 1, 128] {
            let mut settings = Settings::default();
            settings.threads = threads;
            settings.max_frame_delay = 4;
            settings.max_picture_bytes = Some(limit);
            let mut decoder = Decoder::with_settings(settings).unwrap();
            let result = decoder.decode(&temporal_units().concat());
            let error = match result {
                Err(error) => error,
                Ok(None) => match decoder.flush() {
                    Err(error) => error,
                    Ok(_) => panic!("native admission did not report refusal"),
                },
                Ok(Some(_)) => panic!("picture escaped before native admission"),
            };
            assert!(
                matches!(error.error(), rav1d_safe::Error::Other(message) if message == "native picture storage limit exceeded"),
                "{error:?}"
            );
        }
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 4;
        settings.apply_grain = false;
        settings.max_picture_bytes = Some(1 << 20);
        let mut decoder = Decoder::with_settings(settings).unwrap();
        let mut frames = Vec::new();
        if let Some(frame) = decoder.decode(&temporal_units().concat()).unwrap() {
            frames.push(frame);
        }
        frames.extend(decoder.flush().unwrap());
        drop(decoder);
        assert_eq!(frames.len(), 4);
        for (index, frame) in frames.iter().enumerate() {
            check_pixels(frame, index);
        }
    }
}

#[cfg(feature = "c-ffi")]
#[test]
fn picture_budget_refuses_an_unqualified_ffi_allocator_at_construction() {
    let mut settings = Settings::default();
    settings.max_picture_bytes = Some(0);
    assert!(
        matches!(Decoder::with_settings(settings), Err(error) if matches!(error.error(), rav1d_safe::Error::InvalidSettings(_)))
    );
}

#[test]
fn end_input_waits_for_delayed_first_frame_without_an_earlier_poll() {
    for threads in [1, 4] {
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 4;
        let mut decoder = Decoder::with_settings(settings).unwrap();
        let mut packet = Packet::new(temporal_units().remove(0))
            .unwrap()
            .with_timestamp(-33);
        assert_eq!(
            decoder.send_packet(&mut packet).unwrap(),
            SendStatus::Accepted
        );
        let mut frames = Vec::new();
        finish(&mut decoder, &mut frames);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].timestamp(), -33);
        check_pixels(&frames[0], 0);
    }
}

#[test]
fn retry_after_receiving_preserves_all_frames_and_packet_provenance() {
    let units = temporal_units();
    let mut decoder = Decoder::new().unwrap();
    let mut first = Packet::new(units.concat()).unwrap().with_timestamp(11);
    let mut retry = Packet::new(units[0].clone()).unwrap().with_timestamp(22);
    assert_eq!(
        decoder.send_packet(&mut first).unwrap(),
        SendStatus::Accepted
    );
    assert_eq!(
        decoder.send_packet(&mut retry).unwrap(),
        SendStatus::ReceivePending
    );
    let mut frames = Vec::new();
    available(&mut decoder, &mut frames);
    assert_eq!(
        decoder.send_packet(&mut retry).unwrap(),
        SendStatus::Accepted
    );
    finish(&mut decoder, &mut frames);
    assert_eq!(frames.len(), 5);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.timestamp(), if index < 4 { 11 } else { 22 });
        check_pixels(frame, index % 4);
    }
}

#[test]
fn mapped_old_pixels_remain_borrowed_while_inter_frames_decode() {
    let mut units = temporal_units();
    let mut decoder = Decoder::new().unwrap();
    let old = decoder.decode(&units.remove(0)).unwrap().unwrap();
    let Planes::Depth16(planes) = old.planes() else {
        panic!("wrong storage")
    };
    let guard = planes.y();
    let address = guard.as_slice().as_ptr();
    let before = guard.as_slice().to_vec();
    for unit in units {
        decoder.decode(&unit).unwrap();
    }
    decoder.reset();
    drop(decoder);
    assert_eq!(address, guard.as_slice().as_ptr());
    assert_eq!(before, guard.as_slice());
    check_pixels(&old, 0);
}

#[test]
fn raw_color_codes_and_chroma_position_survive_actual_sequence_parsing() {
    let mut unit = temporal_units().remove(0);
    let mut cursor = 0;
    let (start, end) = loop {
        let kind = (unit[cursor] >> 3) & 15;
        assert_eq!(unit[cursor] & 4, 0, "fixture has no OBU extension header");
        cursor += 1;
        let mut len = 0_usize;
        let mut shift = 0;
        loop {
            let byte = unit[cursor];
            cursor += 1;
            len |= usize::from(byte & 127) << shift;
            if byte & 128 == 0 {
                break;
            }
            shift += 7;
        }
        if kind == 1 {
            break (cursor, cursor + len);
        }
        cursor += len;
    };
    let bits: String = unit[start..end]
        .iter()
        .map(|v| format!("{v:08b}"))
        .collect();
    // This fixture explicitly stores primaries=1, transfer=1, matrix=1. Locate
    // that unique record only inside its sequence header, then replace the first
    // two codes by values outside the managed enums. The surrounding profile-0
    // ten-bit 420 syntax puts range then two chroma-position bits immediately
    // after the record (subsampling is implicit for profile 0).
    let records: Vec<_> = bits.match_indices("000000010000000100000001").collect();
    assert_eq!(records.len(), 1);
    let color_start = records[0].0;
    for (offset, value, width) in [(0, 222_u32, 8), (8, 223, 8), (25, 2, 2)] {
        for bit in 0..width {
            let position = color_start + offset + bit;
            let byte = &mut unit[start + position / 8];
            let mask = 1 << (7 - position % 8);
            *byte = (*byte & !mask) | (((value >> (width - 1 - bit)) as u8 & 1) * mask);
        }
    }
    let mut decoder = Decoder::new().unwrap();
    let frame = decoder.decode(&unit).unwrap().unwrap();
    let raw = frame.raw_color_info();
    assert_eq!(raw.primaries, 222);
    assert_eq!(raw.transfer_characteristics, 223);
    assert_eq!(raw.matrix_coefficients, 1);
    assert_eq!(raw.chroma_sample_position, 2);
    assert!(raw.full_range);
    check_pixels(&frame, 0);
}

#[test]
fn packet_context_lives_with_all_presentations_and_their_clones() {
    for threads in [1, 4] {
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 4;
        let mut decoder = Decoder::with_settings(settings).unwrap();
        let context = Arc::new((7_u32, "old configuration"));
        let weak = Arc::downgrade(&context);
        let mut packet = Packet::new(temporal_units().concat())
            .unwrap()
            .with_context(context);
        assert_eq!(
            decoder.send_packet(&mut packet).unwrap(),
            SendStatus::Accepted
        );
        drop(packet);
        let mut frames = Vec::new();
        finish(&mut decoder, &mut frames);
        assert_eq!(frames.len(), 4);
        for (i, frame) in frames.iter().enumerate() {
            check_pixels(frame, i);
            let value = frame.context().unwrap();
            assert_eq!(
                value.downcast_ref::<(u32, &str)>(),
                Some(&(7, "old configuration"))
            );
            assert!(value.downcast_ref::<String>().is_none());
            assert!(std::ptr::eq(value, frames[0].context().unwrap()));
        }
        let retained = frames[0].clone();
        decoder.reset();
        drop(decoder);
        drop(frames);
        assert!(weak.upgrade().is_some());
        check_pixels(&retained, 0);
        drop(retained);
        assert!(
            weak.upgrade().is_none(),
            "native owners leaked the packet context"
        );
    }
}

#[test]
fn packet_context_survives_backpressure_and_error_without_leaking() {
    let context = Arc::new(42_u32);
    let weak = Arc::downgrade(&context);
    let mut decoder = Decoder::new().unwrap();
    let mut joined = Packet::new(temporal_units().concat()).unwrap();
    assert_eq!(
        decoder.send_packet(&mut joined).unwrap(),
        SendStatus::Accepted
    );
    let mut retry = Packet::new(temporal_units().remove(0))
        .unwrap()
        .with_context(context);
    assert_eq!(
        decoder.send_packet(&mut retry).unwrap(),
        SendStatus::ReceivePending
    );
    decoder.end_input();
    assert!(decoder.send_packet(&mut retry).is_err());
    decoder.reset();
    assert!(
        weak.upgrade().is_some(),
        "rejected packet must retain its owner"
    );
    assert_eq!(
        decoder.send_packet(&mut retry).unwrap(),
        SendStatus::Accepted
    );
    drop(retry);
    let mut frames = Vec::new();
    finish(&mut decoder, &mut frames);
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].context().unwrap().downcast_ref::<u32>(),
        Some(&42)
    );
    drop(frames);
    drop(decoder);
    assert!(weak.upgrade().is_none());
}

#[test]
fn packet_context_reset_releases_unpublished_presentations() {
    for threads in [1, 4] {
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 4;
        let mut decoder = Decoder::with_settings(settings).unwrap();
        let context = Arc::new(7_u32);
        let weak = Arc::downgrade(&context);
        let mut packet = Packet::new(temporal_units().concat())
            .unwrap()
            .with_context(context);
        assert_eq!(
            decoder.send_packet(&mut packet).unwrap(),
            SendStatus::Accepted
        );
        drop(packet);
        decoder.reset();
        assert!(
            weak.upgrade().is_none(),
            "reset retained unpublished context"
        );
    }
}

#[test]
fn packet_context_survives_film_grain_picture_copy() {
    for bytes in [
        include_bytes!("gen_vectors/i8_420_grain_128x128.obu").as_slice(),
        include_bytes!("gen_vectors/i10_420_grain_128x128.obu").as_slice(),
    ] {
        let context = Arc::new(9_u32);
        let weak = Arc::downgrade(&context);
        let mut packet = Packet::new(bytes.to_vec()).unwrap().with_context(context);
        let mut decoder = Decoder::new().unwrap();
        assert_eq!(
            decoder.send_packet(&mut packet).unwrap(),
            SendStatus::Accepted
        );
        let mut frames = Vec::new();
        finish(&mut decoder, &mut frames);
        assert!(!frames.is_empty());
        for frame in &frames {
            assert_eq!(frame.context().unwrap().downcast_ref::<u32>(), Some(&9));
        }
        drop(decoder);
        drop(frames);
        drop(packet);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn packet_context_parse_error_retains_input_until_caller_releases_it() {
    let context = Arc::new(17_u32);
    let weak = Arc::downgrade(&context);
    let mut packet = Packet::new(vec![0xff]).unwrap().with_context(context);
    let mut decoder = Decoder::new().unwrap();
    let error = decoder.send_packet(&mut packet).unwrap_err();
    assert_eq!(error.error(), &rav1d_safe::Error::InvalidData);
    assert_eq!(packet.len(), 1);
    decoder.reset();
    assert!(weak.upgrade().is_some());
    drop(packet);
    assert!(weak.upgrade().is_none());
}

#[test]
fn packet_context_cancelled_decode_releases_owners_on_reset_and_drop() {
    struct Cancel;
    impl enough::Stop for Cancel {
        fn check(&self) -> Result<(), enough::StopReason> {
            Err(enough::StopReason::Cancelled)
        }
    }
    for threads in [1, 4] {
        let context = Arc::new(17_u32);
        let weak = Arc::downgrade(&context);
        let mut packet = Packet::new(temporal_units().concat())
            .unwrap()
            .with_context(context);
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 4;
        let mut decoder = Decoder::with_settings(settings).unwrap();
        decoder.set_stop(Some(Arc::new(Cancel)));
        match decoder.send_packet(&mut packet) {
            Err(error) => assert_eq!(error.error(), &rav1d_safe::Error::Cancelled),
            Ok(SendStatus::Accepted) => {
                decoder.end_input();
                match decoder.receive() {
                    Err(error) => assert_eq!(error.error(), &rav1d_safe::Error::Cancelled),
                    Ok(_) => panic!("cancelled input produced a successful receive"),
                }
            }
            Ok(_) => panic!("first input cannot be backpressured"),
        }
        decoder.reset();
        drop(packet);
        assert!(weak.upgrade().is_none());
    }
}
