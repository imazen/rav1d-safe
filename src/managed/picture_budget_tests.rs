use super::*;
use crate::include::dav1d::headers::{DRav1d, Rav1dSequenceHeader};
use crate::include::dav1d::picture::Rav1dPicAllocator;

fn sequence_header() -> Rav1dSequenceHeader {
    let mut decoder = Decoder::new().unwrap();
    let frame = decoder
        .decode(include_bytes!(
            "../../tests/crash_vectors/tile_threading_cdef_lpf_race.obu"
        ))
        .unwrap()
        .unwrap();
    frame.inner.seq_hdr.as_ref().unwrap().rav1d.clone()
}

#[test]
fn picture_budget_tracks_padding_and_last_native_owner() {
    let sequence = sequence_header();
    for layout in [
        Rav1dPixelLayout::I400,
        Rav1dPixelLayout::I420,
        Rav1dPixelLayout::I422,
        Rav1dPixelLayout::I444,
    ] {
        for hbd in [0, 1, 2] {
            for (width, height) in [(17, 13), (128, 128), (513, 9)] {
                let mut sequence = sequence.clone();
                sequence.layout = layout;
                sequence.hbd = hbd;
                let sequence = Arc::new(DRav1d::from_rav1d(sequence));
                let allocator = Rav1dPicAllocator::with_limit(Some(usize::MAX));
                let picture = allocator
                    .alloc_picture_data(width, height, sequence.clone(), None)
                    .unwrap();
                let bytes = allocator.budget.as_ref().unwrap().used();
                assert!(bytes > 0);
                assert!(bytes >= (width * height * if hbd == 0 { 1 } else { 2 }) as usize);
                drop(picture);
                assert_eq!(allocator.budget.as_ref().unwrap().used(), 0);

                let tight = Rav1dPicAllocator::with_limit(Some(bytes - 1));
                assert!(matches!(
                    tight.alloc_picture_data(width, height, sequence.clone(), None),
                    Err(Rav1dError::EPICLIMIT)
                ));
                assert_eq!(tight.budget.as_ref().unwrap().used(), 0);
                let exact = Rav1dPicAllocator::with_limit(Some(bytes));
                let picture = exact
                    .alloc_picture_data(width, height, sequence.clone(), None)
                    .unwrap();
                let alias = picture.clone();
                drop(picture);
                assert_eq!(exact.budget.as_ref().unwrap().used(), bytes);
                assert!(matches!(
                    exact.alloc_picture_data(width, height, sequence.clone(), None),
                    Err(Rav1dError::EPICLIMIT)
                ));
                drop(alias);
                assert_eq!(exact.budget.as_ref().unwrap().used(), 0);
                drop(
                    exact
                        .alloc_picture_data(width, height, sequence, None)
                        .unwrap(),
                );
                assert_eq!(exact.budget.as_ref().unwrap().used(), 0);
            }
        }
    }
}

#[test]
fn picture_budget_follows_copy_allocator_after_its_creator_drops() {
    let sequence = sequence_header();
    let allocator = Rav1dPicAllocator::with_limit(Some(usize::MAX));
    let original = allocator
        .alloc_picture_data(17, 13, Arc::new(DRav1d::from_rav1d(sequence)), None)
        .unwrap();
    let bytes = allocator.budget.as_ref().unwrap().used();
    drop(original);
    let allocator = Rav1dPicAllocator::with_limit(Some(bytes * 2));
    let sequence = sequence_header();
    let original = allocator
        .alloc_picture_data(17, 13, Arc::new(DRav1d::from_rav1d(sequence)), None)
        .unwrap();
    let budget = allocator.budget.clone().unwrap();
    drop(allocator);
    let mut copy = Rav1dPicture::default();
    crate::src::picture::rav1d_picture_alloc_copy(&None, &mut copy, 17, &original).unwrap();
    assert_eq!(budget.used(), bytes * 2);
    let mut refused = Rav1dPicture::default();
    assert!(matches!(
        crate::src::picture::rav1d_picture_alloc_copy(&None, &mut refused, 17, &copy),
        Err(Rav1dError::EPICLIMIT)
    ));
    drop(original);
    assert_eq!(budget.used(), bytes);
    crate::src::picture::rav1d_picture_alloc_copy(&None, &mut refused, 17, &copy).unwrap();
    drop(copy);
    drop(refused);
    assert_eq!(budget.used(), 0);
}

#[test]
fn picture_budget_native_grain_refusal_reset_and_retained_output() {
    fn one(decoder: &mut Decoder, input: &[u8]) -> Result<Frame> {
        let mut frames = Vec::new();
        if let Some(frame) = decoder.decode(input)? {
            frames.push(frame);
        }
        decoder.end_input();
        loop {
            match decoder.receive()? {
                ReceiveStatus::Frame(frame) => frames.push(frame),
                ReceiveStatus::EndOfStream => break,
                ReceiveStatus::NeedInput => panic!("ended input requested more packets"),
            }
        }
        assert_eq!(frames.len(), 1);
        Ok(frames.pop().unwrap())
    }
    for input in [
        include_bytes!("../../tests/gen_vectors/i8_420_grain_128x128.obu").as_slice(),
        include_bytes!("../../tests/gen_vectors/i10_420_grain_128x128.obu").as_slice(),
    ] {
        for threads in [1, 4] {
            let configured = |limit, grain| {
                let mut settings = Settings::default();
                settings.threads = threads;
                settings.max_frame_delay = 1;
                settings.apply_grain = grain;
                settings.max_picture_bytes = Some(limit);
                Decoder::with_settings(settings).unwrap()
            };
            // Observe the real native allocation, rather than duplicate its
            // geometry arithmetic. No-grain output shares reconstruction.
            let mut probe = configured(usize::MAX, false);
            let budget = probe.ctx.allocator.budget.clone().unwrap();
            let frame = one(&mut probe, input).unwrap();
            let bytes = budget.used();
            assert!(bytes > 0);
            drop(probe);
            assert_eq!(budget.used(), bytes);
            drop(frame);
            assert_eq!(budget.used(), 0);

            // Reconstruction fits; the separate grain output cannot fit.
            let mut refused = configured(bytes, true);
            let budget = refused.ctx.allocator.budget.clone().unwrap();
            let Err(error) = one(&mut refused, input) else {
                panic!("grain copy escaped its native picture allowance")
            };
            assert!(matches!(error.error(), Error::Other(message)
                if message == "native picture storage limit exceeded"));
            assert!(budget.used() <= bytes);
            refused.reset();
            assert_eq!(budget.used(), 0);
            drop(refused);

            let mut decoder = configured(bytes * 2, true);
            let budget = decoder.ctx.allocator.budget.clone().unwrap();
            let retained = one(&mut decoder, input).unwrap();
            assert_eq!(budget.used(), bytes * 2);
            decoder.reset();
            assert_eq!(budget.used(), bytes);
            assert!(one(&mut decoder, input).is_err());
            decoder.reset();
            assert_eq!(budget.used(), bytes);
            drop(retained);
            assert_eq!(budget.used(), 0);
            let retained = one(&mut decoder, input).unwrap();
            drop(decoder);
            assert_eq!(budget.used(), bytes);
            drop(retained);
            assert_eq!(budget.used(), 0);
        }
    }
}
