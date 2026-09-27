//! Streaming ownership and timing, exercised against original lossless AV1.
use rav1d_safe::{Decoder, Frame, Packet, Planes, ReceiveStatus, SendStatus, Settings};

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
                    if index % 2 == 0 { 0 } else { 256 }
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
