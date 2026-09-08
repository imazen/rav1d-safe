#![forbid(unsafe_code)]

use rgb::Rgb;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = (128_u32, 128_u32);
    let pixels: Vec<Rgb<u8>> = (0..width * height)
        .map(|i| Rgb::new((i % width) as u8, (i / width) as u8, 96))
        .collect();
    let input = zenavif::PixelBuffer::from_pixels_erased(pixels, width, height)?;

    // The encode feature uses zenravif/zenrav1e and wraps AV1 in an AVIF file.
    let encoded = zenavif::encode(&input)?;
    // The default AVIF decoder handles the container and color conversion,
    // using rav1d-safe's Rust API for the AV1 payload.
    let decoded = zenavif::decode(&encoded.avif_file)?;
    assert_eq!((decoded.width(), decoded.height()), (width, height));
    println!(
        "{} AVIF bytes -> {}x{} pixel buffer",
        encoded.avif_file.len(),
        decoded.width(),
        decoded.height()
    );
    Ok(())
}
