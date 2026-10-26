//! Scoped allocation-mode cost on committed small AV1 streams and a tiled still.
//! Each retained decoder resets between passes, exposing pooled-buffer reuse.
//! This is a control workload, not a claim about production video throughput.
use rav1d_safe::{Decoder, Frame, Settings};
use zenbench::prelude::*;

#[path = "../tests/common/committed_vectors.rs"]
#[allow(dead_code)]
mod committed_vectors;

fn decode(decoder: &mut Decoder, data: &[u8], mut visit: impl FnMut(&Frame)) -> usize {
    decoder.reset();
    let mut count = 0;
    if let Some(frame) = decoder.decode(data).expect("native decode") {
        visit(&frame);
        count += 1;
    }
    while let Some(frame) = decoder.get_frame().expect("native receive") {
        visit(&frame);
        count += 1;
        assert!(count < 256, "committed fixture produced unbounded output");
    }
    for frame in decoder.flush().expect("native drain") {
        visit(&frame);
        count += 1;
    }
    assert!(count > 0);
    count
}

fn configured(threads: u32, bounded: bool) -> Decoder {
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = 2;
    settings.max_picture_bytes = bounded.then_some(64 * 1024 * 1024);
    Decoder::with_settings(settings).unwrap()
}

fn picture_budget(suite: &mut Suite) {
    let fixtures = [
        (
            "inter8_128x128",
            include_bytes!("../tests/gen_vectors/v8_420_shift4_128x128.obu").as_slice(),
        ),
        (
            "inter10_96x96",
            include_bytes!("../tests/gen_vectors/v10_420_shift3_96x96.obu").as_slice(),
        ),
        (
            "grain8_128x128",
            include_bytes!("../tests/gen_vectors/i8_420_grain_128x128.obu").as_slice(),
        ),
        (
            "grain10_128x128",
            include_bytes!("../tests/gen_vectors/i10_420_grain_128x128.obu").as_slice(),
        ),
        (
            "tiled_1024x1024",
            include_bytes!("../tests/crash_vectors/tile_threading_cdef_lpf_race.obu").as_slice(),
        ),
    ];
    for (name, data) in fixtures {
        for threads in [1, 4] {
            let mut expected = None;
            let mut count = 0;
            for bounded in [false, true] {
                let mut decoder = configured(threads, bounded);
                for _ in 0..2 {
                    let mut hash = md5::Context::new();
                    count = decode(&mut decoder, data, |frame| {
                        committed_vectors::hash_frame(frame, &mut hash)
                    });
                    let result = (count, hash.finalize());
                    if let Some(expected) = expected {
                        assert_eq!(result, expected, "{name} t={threads} bounded={bounded}");
                    } else {
                        expected = Some(result);
                    }
                }
            }
            suite.compare(format!("picture_storage/{name}/{threads}t"), |group| {
                group.throughput(Throughput::Elements(count as u64));
                group.config().max_time(std::time::Duration::from_secs(3));
                for (label, bounded) in [
                    ("pooled", false),
                    ("pooled_control", false),
                    ("bounded", true),
                ] {
                    group.bench(label, move |b| {
                        let mut decoder = configured(threads, bounded);
                        b.iter(|| assert_eq!(decode(&mut decoder, black_box(data), |_| {}), count));
                    });
                }
            });
        }
    }
}

zenbench::main!(picture_budget);
