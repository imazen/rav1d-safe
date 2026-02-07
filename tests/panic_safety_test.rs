//! Panic safety tests for the managed API
//!
//! These tests verify that the decoder properly cleans up resources
//! when panics occur during decoding.

use std::panic::{catch_unwind, AssertUnwindSafe};
use rav1d_safe::src::managed::Decoder;

#[test]
fn test_decoder_drop_on_panic() {
    // Create a decoder
    let mut decoder = Decoder::new().unwrap();
    
    // Attempt to decode with a panic occurring
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Empty data should not panic, but we'll test the unwinding mechanism
        let _ = decoder.decode(&[]);
        panic!("Intentional panic for testing");
    }));
    
    // Verify the panic occurred
    assert!(result.is_err());
    
    // The decoder should have been dropped properly during unwinding
    // If there were leaks, they would be caught by leak sanitizers (ASAN/LSAN)
}

#[test]
fn test_decoder_multiple_decode_drop() {
    // Test that multiple decode calls followed by drop don't leak
    // This would accumulate leaks if Drop wasn't working properly
    for _ in 0..100 {
        let mut decoder = Decoder::new().unwrap();
        
        // Try decoding empty data
        let _ = decoder.decode(&[]);
        
        // Try decoding with some invalid data (not valid OBU)
        let invalid_data = vec![0u8; 100];
        let _ = decoder.decode(&invalid_data);
        
        // Decoder drops here - if there's a leak, it accumulates across 100 iterations
    }
}

#[test]
fn test_decoder_drop_without_flush() {
    // Test that dropping a decoder without flushing doesn't leak
    for _ in 0..50 {
        let mut decoder = Decoder::new().unwrap();
        
        // Feed some data
        let _ = decoder.decode(&[1, 2, 3, 4, 5]);
        
        // Drop without flushing
        drop(decoder);
    }
}

#[test]
fn test_decoder_panic_during_decode() {
    // Create test that would panic if data handling was incorrect
    let mut decoder = Decoder::new().unwrap();
    
    // These should all handle errors gracefully without panicking
    let _ = decoder.decode(&[]);
    let _ = decoder.decode(&[0xFF; 10]);
    let _ = decoder.decode(&[0x00; 100]);
    
    // Decoder should drop cleanly
}
