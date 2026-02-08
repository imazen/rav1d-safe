use rav1d_safe::src::managed::{Decoder, Planes};
use std::fs::File;
use std::io::{BufReader, Read};

// Minimal IVF parser for demo
fn parse_ivf_frames(reader: &mut impl Read) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut header = [0u8; 32];
    reader.read_exact(&mut header)?;

    if &header[0..4] != b"DKIF" {
        return Err("Not an IVF file".into());
    }

    let mut frames = Vec::new();
    loop {
        let mut frame_header = [0u8; 12];
        if reader.read_exact(&mut frame_header).is_err() {
            break;
        }

        let frame_size = u32::from_le_bytes([
            frame_header[0],
            frame_header[1],
            frame_header[2],
            frame_header[3],
        ]) as usize;

        let mut frame_data = vec![0u8; frame_size];
        reader.read_exact(&mut frame_data)?;
        frames.push(frame_data);
    }

    Ok(frames)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Find a test vector
    let test_file = "target/test-vectors/dav1d-test-data/10-bit/film_grain/clip_0.ivf";

    println!("=== rav1d-safe Managed API Demo ===\n");
    println!("Decoding: {}\n", test_file);

    // Parse IVF to extract OBU frames
    let file = File::open(test_file)?;
    let mut reader = BufReader::new(file);
    let frames = parse_ivf_frames(&mut reader)?;
    println!("IVF file contains {} frames\n", frames.len());

    // Create decoder
    let mut decoder = Decoder::new()?;
    println!("✓ Decoder created\n");

    // Decode frames
    for (i, frame_data) in frames.iter().enumerate() {
        match decoder.decode(frame_data)? {
            Some(frame) => {
                println!("Frame {}:", i + 1);
                println!("  Size: {}x{}", frame.width(), frame.height());
                println!("  Bit depth: {}", frame.bit_depth());
                println!("  Layout: {:?}", frame.pixel_layout());

                // Color info
                let color = frame.color_info();
                println!("  Color primaries: {:?}", color.primaries);
                println!("  Transfer: {:?}", color.transfer_characteristics);

                // HDR metadata
                if let Some(cll) = frame.content_light() {
                    println!("  HDR Max CLL: {} nits", cll.max_content_light_level);
                }

                // Zero-copy pixel access
                match frame.planes() {
                    Planes::Depth8(planes) => {
                        let y = planes.y();
                        println!(
                            "  Y plane: {}x{} (stride={})",
                            y.width(),
                            y.height(),
                            y.stride()
                        );
                        println!("  First pixel: {}", y.pixel(0, 0));
                    }
                    Planes::Depth16(planes) => {
                        let y = planes.y();
                        println!(
                            "  Y plane: {}x{} (stride={})",
                            y.width(),
                            y.height(),
                            y.stride()
                        );
                        println!("  First pixel: {}", y.pixel(0, 0));
                    }
                }
                println!();
            }
            None => {}
        }
    }

    // Flush remaining
    let remaining = decoder.flush()?;
    if !remaining.is_empty() {
        println!("Flushed {} additional frames", remaining.len());
    }

    println!("\n✓ Decoding complete!");
    Ok(())
}
