//! MT stress test
use rav1d_safe::src::managed::{Decoder, Settings};
use std::sync::OnceLock;

fn obu_4k() -> &'static [u8] {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let data = std::fs::read("test-vectors/bench/photo_4k.avif").expect("photo_4k.avif");
        let parser = zenavif_parse::AvifParser::from_bytes(&data).expect("avif parse");
        parser.primary_data().expect("primary").into_owned()
    })
}

#[test]
fn mt_stress_4k() {
    let obu = obu_4k();

    for n_threads in [1, 2, 4, 8, 16] {
        for trial in 0..5 {
            let mut settings = Settings::default();
            settings.threads = n_threads;
            settings.frame_size_limit = 8192 * 8192;
            let mut decoder = Decoder::with_settings(settings).expect("decoder");

            let result = decoder.decode(obu);
            assert!(
                result.is_ok(),
                "threads={n_threads} trial={trial}: error {:?}",
                result.err()
            );
            let _ = decoder.flush();
            eprintln!("threads={n_threads} trial={trial}: ok");
        }
    }
}
