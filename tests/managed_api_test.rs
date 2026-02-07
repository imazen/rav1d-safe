use rav1d_safe::src::managed::{Decoder, Planes, Settings};

#[test]
fn test_decoder_creation() {
    // Test that we can create a decoder with default settings
    let decoder = Decoder::new();
    assert!(decoder.is_ok(), "Failed to create decoder");
}

#[test]
fn test_decoder_with_custom_settings() {
    // Test decoder with custom settings
    let settings = Settings {
        threads: 4,
        apply_grain: false,
        ..Default::default()
    };

    let decoder = Decoder::with_settings(settings);
    assert!(
        decoder.is_ok(),
        "Failed to create decoder with custom settings"
    );
}

#[test]
fn test_decode_empty_data() {
    // Test that decoding empty data returns NeedMoreData
    let mut decoder = Decoder::new().expect("Failed to create decoder");

    match decoder.decode(&[]) {
        Ok(None) => {
            // Expected: no frame available with empty data
        }
        Ok(Some(_)) => panic!("Unexpected frame from empty data"),
        Err(e) => {
            // Also acceptable: error on empty data
            eprintln!("Error on empty data (acceptable): {}", e);
        }
    }
}

// Note: Full decode test would require valid AV1 bitstream data
// This would be better as an integration test with actual test vectors
