//! Wrap an AVIF still's AV1 payload into an N-frame IVF so `dav1d` can be
//! measured on exactly the bitstream `bench_ab_decode` feeds our decoder.
//!
//! Every IVF frame is the SAME complete OBU stream the AVIF carries (temporal
//! delimiter + sequence header + key frame), which is what makes the two arms
//! comparable: both sides re-parse the sequence header once per frame, and
//! neither has any inter-frame prediction to amortise.
//!
//! Usage: avif_to_ivf <input.avif> <frames> <out.ivf>

fn extract_obu(avif_bytes: &[u8]) -> Vec<u8> {
    let parser = zenavif_parse::AvifParser::from_bytes(avif_bytes).expect("avif parse");
    parser
        .primary_data()
        .expect("avif primary item")
        .into_owned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <input.avif> <frames> <out.ivf>", args[0]);
        std::process::exit(2);
    }
    let avif = std::fs::read(&args[1]).expect("read input");
    let frames: u32 = args[2].parse().expect("frames");
    let obu = extract_obu(&avif);

    // Geometry comes from our own decoder rather than from a hand-rolled
    // sequence-header parse, so the IVF header cannot disagree with the
    // stream it wraps.
    let mut settings = rav1d_safe::src::managed::Settings::default();
    settings.threads = 1;
    settings.frame_size_limit = 8192 * 8192;
    let mut dec = rav1d_safe::src::managed::Decoder::with_settings(settings).expect("decoder");
    let frame = dec.decode(&obu).expect("decode").expect("frame");
    let (w, h) = (frame.width() as u16, frame.height() as u16);
    drop(frame);

    let mut out: Vec<u8> = Vec::with_capacity(32 + frames as usize * (12 + obu.len()));
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes()); // version
    out.extend_from_slice(&32u16.to_le_bytes()); // header length
    out.extend_from_slice(b"AV01");
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes()); // rate numerator
    out.extend_from_slice(&1u32.to_le_bytes()); // rate denominator
    out.extend_from_slice(&frames.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // unused
    for pts in 0..frames {
        out.extend_from_slice(&(obu.len() as u32).to_le_bytes());
        out.extend_from_slice(&(pts as u64).to_le_bytes());
        out.extend_from_slice(&obu);
    }
    std::fs::write(&args[3], &out).expect("write ivf");
    println!(
        "{}\t{w}x{h}\t{frames}\tobu={}\tivf={}",
        args[3],
        obu.len(),
        out.len()
    );
}
