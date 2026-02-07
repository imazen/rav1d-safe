use rav1d_safe::src::managed::{Decoder, Settings, Planes};
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.ivf>", args[0]);
        eprintln!("\nExample AV1 files:");
        eprintln!("  - Download from: https://storage.googleapis.com/aom-test-data/");
        eprintln!("  - Or run: bash scripts/download-test-vectors.sh");
        return Ok(());
    }

    let input_path = &args[1];
    println!("Decoding: {}", input_path);

    // Read input file
    let data = fs::read(input_path)?;
    println!("File size: {} bytes", data.len());

    // Create decoder with default settings
    let settings = Settings {
        threads: 0, // Auto-detect
        apply_grain: true,
        ..Default::default()
    };

    let mut decoder = Decoder::with_settings(settings)?;
    println!("Decoder created");

    // Decode frames
    let mut frame_count = 0;
    let chunk_size = 8192;

    for chunk in data.chunks(chunk_size) {
        match decoder.decode(chunk) {
            Ok(Some(frame)) => {
                frame_count += 1;

                println!("\n=== Frame {} ===", frame_count);
                println!("  Dimensions: {}x{}", frame.width(), frame.height());
                println!("  Bit depth: {}", frame.bit_depth());
                println!("  Layout: {:?}", frame.pixel_layout());

                // Color metadata
                let color = frame.color_info();
                println!("  Primaries: {:?}", color.primaries);
                println!("  Transfer: {:?}", color.transfer_characteristics);
                println!("  Matrix: {:?}", color.matrix_coefficients);
                println!("  Range: {:?}", color.color_range);

                // HDR metadata if present
                if let Some(cll) = frame.content_light() {
                    println!("  HDR Content Light:");
                    println!("    Max CLL: {} nits", cll.max_content_light_level);
                    println!("    Max FALL: {} nits", cll.max_frame_average_light_level);
                }

                if let Some(md) = frame.mastering_display() {
                    println!("  HDR Mastering Display:");
                    println!("    Max luminance: {:.2} nits", md.max_luminance_nits());
                    println!("    Min luminance: {:.6} nits", md.min_luminance_nits());

                    let [r_x, r_y] = md.primary_chromaticity(0);
                    let [g_x, g_y] = md.primary_chromaticity(1);
                    let [b_x, b_y] = md.primary_chromaticity(2);
                    let [w_x, w_y] = md.white_point_chromaticity();

                    println!("    Red primary: ({:.4}, {:.4})", r_x, r_y);
                    println!("    Green primary: ({:.4}, {:.4})", g_x, g_y);
                    println!("    Blue primary: ({:.4}, {:.4})", b_x, b_y);
                    println!("    White point: ({:.4}, {:.4})", w_x, w_y);
                }

                // Access pixel data
                match frame.planes() {
                    Planes::Depth8(planes) => {
                        let y_plane = planes.y();
                        println!("  Y plane: {}x{} stride={}",
                                 y_plane.width(), y_plane.height(), y_plane.stride());

                        // Sample some pixels
                        if y_plane.width() > 0 && y_plane.height() > 0 {
                            let sample_pixels = [
                                (0, 0),
                                (y_plane.width() / 2, y_plane.height() / 2),
                                (y_plane.width() - 1, y_plane.height() - 1),
                            ];

                            print!("  Sample pixels: ");
                            for (x, y) in &sample_pixels {
                                if *x < y_plane.width() && *y < y_plane.height() {
                                    print!("({},{})={} ", x, y, y_plane.pixel(*x, *y));
                                }
                            }
                            println!();
                        }
                    }
                    Planes::Depth16(planes) => {
                        let y_plane = planes.y();
                        println!("  Y plane (16-bit): {}x{} stride={}",
                                 y_plane.width(), y_plane.height(), y_plane.stride());

                        // Sample some pixels
                        if y_plane.width() > 0 && y_plane.height() > 0 {
                            let sample_pixels = [
                                (0, 0),
                                (y_plane.width() / 2, y_plane.height() / 2),
                                (y_plane.width() - 1, y_plane.height() - 1),
                            ];

                            print!("  Sample pixels: ");
                            for (x, y) in &sample_pixels {
                                if *x < y_plane.width() && *y < y_plane.height() {
                                    print!("({},{})={} ", x, y, y_plane.pixel(*x, *y));
                                }
                            }
                            println!();
                        }
                    }
                }

                // Only decode first frame for demo
                if frame_count >= 1 {
                    break;
                }
            }
            Ok(None) => {
                // Need more data
            }
            Err(e) => {
                eprintln!("Decode error: {}", e);
                break;
            }
        }
    }

    // Flush remaining frames
    println!("\nFlushing decoder...");
    match decoder.flush() {
        Ok(remaining) => {
            if !remaining.is_empty() {
                println!("Flushed {} additional frame(s)", remaining.len());
                frame_count += remaining.len();
            }
        }
        Err(e) => {
            eprintln!("Flush error: {}", e);
        }
    }

    println!("\n=== Summary ===");
    println!("Total frames decoded: {}", frame_count);

    Ok(())
}
