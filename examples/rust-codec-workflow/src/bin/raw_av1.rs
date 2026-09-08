#![forbid(unsafe_code)]

use rav1d_safe::{Decoder, PixelLayout, Planes};
use zenrav1e::prelude::{
    ChromaSampling, Config, Context, EncoderConfig, EncoderStatus, SpeedSettings,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const WIDTH: usize = 128;
    const HEIGHT: usize = 128;

    let mut enc = EncoderConfig::default();
    enc.width = WIDTH;
    enc.height = HEIGHT;
    enc.still_picture = true;
    enc.chroma_sampling = ChromaSampling::Cs420;
    enc.bit_depth = 8;
    enc.quantizer = 80;
    enc.speed_settings = SpeedSettings::from_preset(10);
    let mut encoder: Context<u8> = Config::new().with_encoder_config(enc).new_context()?;

    // zenrav1e takes planar YCbCr. This is a luma ramp with neutral chroma.
    let y: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|i| (16 + (i % WIDTH)) as u8)
        .collect();
    let u = vec![128; WIDTH / 2 * (HEIGHT / 2)];
    let v = u.clone();
    let mut input = encoder.new_frame();
    for (plane, samples) in input.planes.iter_mut().zip([&y, &u, &v]) {
        let stride = WIDTH.div_ceil(1 << plane.cfg.xdec);
        plane.copy_from_raw_u8(samples, stride, 1);
    }
    encoder.send_frame(input)?;
    encoder.flush();

    let mut obu = Vec::new();
    loop {
        match encoder.receive_packet() {
            Ok(packet) => obu.extend_from_slice(&packet.data),
            Err(EncoderStatus::Encoded) => continue,
            Err(EncoderStatus::LimitReached) => break,
            Err(error) => return Err(error.into()),
        }
    }
    assert!(!obu.is_empty());

    // Packet data is raw AV1 OBU data, accepted directly by the Rust decoder.
    // It is not an AVIF or IVF file. No C API or container stripping is involved.
    let mut decoder = Decoder::new()?;
    let mut frames = Vec::new();
    if let Some(frame) = decoder.decode(&obu)? {
        frames.push(frame);
    }
    frames.extend(decoder.flush()?);
    assert_eq!(frames.len(), 1);
    let frame = &frames[0];
    assert_eq!(
        (frame.width(), frame.height()),
        (WIDTH as u32, HEIGHT as u32)
    );
    assert_eq!(frame.bit_depth(), 8);
    assert_eq!(frame.pixel_layout(), PixelLayout::I420);
    let Planes::Depth8(planes) = frame.planes() else {
        return Err("expected 8-bit decoded planes".into());
    };
    assert_eq!(planes.y().rows().count(), HEIGHT);
    println!(
        "{} OBU bytes -> {}x{} {}-bit YUV420",
        obu.len(),
        frame.width(),
        frame.height(),
        frame.bit_depth()
    );
    // This encode is lossy, so pixel equality with the input is not expected.
    Ok(())
}
