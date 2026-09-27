//! Output provenance must not be inferred from a retained reference header.
use rav1d_safe::{Decoder, Frame, Packet, Planes, ReceiveStatus, SendStatus, Settings, Strictness};

fn units() -> Vec<(usize, Vec<u8>)> {
    let data = include_bytes!("media_vectors/native-reordered-17x17-444.obu");
    let mut cursor = 0;
    let mut starts = Vec::new();
    while cursor < data.len() {
        let start = cursor;
        let header = data[cursor];
        cursor += 1;
        assert_eq!(header & 4, 0, "no extension header in this fixture");
        assert_ne!(header & 2, 0);
        if (header >> 3) & 15 == 2 {
            starts.push(start);
        }
        let mut len = 0;
        for shift in (0..56).step_by(7) {
            let byte = data[cursor];
            cursor += 1;
            len |= usize::from(byte & 127) << shift;
            if byte & 128 == 0 {
                break;
            }
        }
        cursor += len;
    }
    assert_eq!(cursor, data.len());
    starts.push(data.len());
    starts
        .windows(2)
        .map(|s| (s[0], data[s[0]..s[1]].to_vec()))
        .collect()
}

fn available(decoder: &mut Decoder, output: &mut Vec<Frame>, ended: bool) {
    for _ in 0..16 {
        match decoder.receive().unwrap() {
            ReceiveStatus::Frame(frame) => output.push(frame),
            ReceiveStatus::NeedInput if !ended => return,
            ReceiveStatus::EndOfStream if ended => return,
            _ => panic!("unexpected receive state"),
        }
    }
    panic!("unbounded output");
}

#[test]
fn reordered_presentations_keep_their_own_flags_and_packet_offsets() {
    let units = units();
    assert_eq!(units.len(), 5);
    for threads in [1, 4] {
        let mut settings = Settings::default();
        settings.threads = threads;
        settings.max_frame_delay = 4;
        let mut decoder = Decoder::with_settings(settings).unwrap();
        let mut frames = Vec::new();
        for (i, (offset, bytes)) in units.iter().enumerate() {
            let mut packet = Packet::new(bytes.clone())
                .unwrap()
                .with_timestamp(i as i64 * 1001)
                .with_offset(*offset as i64);
            loop {
                match decoder.send_packet(&mut packet).unwrap() {
                    SendStatus::Accepted => break,
                    SendStatus::ReceivePending => available(&mut decoder, &mut frames, false),
                    _ => panic!("unexpected submit state"),
                }
            }
            available(&mut decoder, &mut frames, false);
        }
        decoder.end_input();
        available(&mut decoder, &mut frames, true);
        drop(decoder);
        assert_eq!(frames.len(), 5);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.is_keyframe(), i == 0);
            assert_eq!(frame.is_show_existing(), matches!(i, 2 | 4));
            assert_eq!(frame.timestamp(), i as i64 * 1001);
            assert_eq!(frame.input_offset(), units[i].0 as i64);
            let Planes::Depth8(planes) = frame.planes() else {
                panic!("storage")
            };
            for (p, plane) in [planes.y(), planes.u().unwrap(), planes.v().unwrap()]
                .iter()
                .enumerate()
            {
                for (y, row) in plane.rows().enumerate() {
                    for (x, &sample) in row.iter().enumerate() {
                        assert_eq!(sample, ((17 * x + 29 * y + 43 * i + 71 * p) % 256) as u8);
                    }
                }
            }
        }
    }
}

#[test]
fn showing_an_older_key_picture_is_not_a_new_keyframe() {
    // Deliberately non-conforming: show an already visible key frame. Strict
    // decoding rejects a non-showable reference; lenient playback accepts it.
    // Even there, the retained key-frame header must not label a new seek point.
    let show_key = [0x12, 0, 0x1a, 1, 0x88];
    for strictness in [Strictness::Strict, Strictness::Lenient] {
        let mut settings = Settings::default();
        settings.strictness = strictness;
        let mut decoder = Decoder::with_settings(settings).unwrap();
        let key = decoder.decode(&units()[0].1).unwrap().unwrap();
        assert!(key.is_keyframe());
        assert!(!key.is_show_existing());
        let shown = decoder.decode(&show_key);
        if strictness == Strictness::Strict {
            assert!(shown.is_err());
        } else {
            let shown = shown.unwrap().unwrap();
            assert!(shown.is_show_existing());
            assert!(!shown.is_keyframe());
            assert!(key.is_keyframe(), "reference provenance was mutated");
        }
    }
}
