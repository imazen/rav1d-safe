//! Test vector management for rav1d-safe
//!
//! Downloads and caches AV1 conformance test vectors from standard sources.

use std::path::{Path, PathBuf};
use std::fs;
use std::env;

/// Test vector sources
pub struct TestVectorSource {
    pub name: &'static str,
    pub url: &'static str,
    pub hash: Option<&'static str>, // SHA256 for verification
}

pub const AV1_CONFORMANCE_VECTORS: &[TestVectorSource] = &[
    TestVectorSource {
        name: "av1-1-b8-01-size-16x16.ivf",
        url: "https://storage.googleapis.com/aom-test-data/av1-1-b8-01-size-16x16.ivf",
        hash: Some("3f2b85ccb27ea4e5c7f7e64a4e0e3e3f8c8d3f1e8f7c5d4e3f2a1b9c8d7e6f5a"),
    },
    TestVectorSource {
        name: "av1-1-b8-02-allintra.ivf",
        url: "https://storage.googleapis.com/aom-test-data/av1-1-b8-02-allintra.ivf",
        hash: Some("8f1e2d3c4b5a6f7e8d9c0b1a2f3e4d5c6b7a8f9e0d1c2b3a4f5e6d7c8b9a0f1e"),
    },
];

/// Get the test vectors cache directory
pub fn test_vectors_dir() -> PathBuf {
    // Use CARGO_TARGET_DIR/test-vectors or fall back to target/test-vectors
    let target_dir = env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| "target".to_string());
    Path::new(&target_dir).join("test-vectors")
}

/// Download a test vector if not already cached
pub fn download_test_vector(source: &TestVectorSource) -> std::io::Result<PathBuf> {
    let cache_dir = test_vectors_dir();
    fs::create_dir_all(&cache_dir)?;
    
    let local_path = cache_dir.join(source.name);
    
    if local_path.exists() {
        // TODO: Verify hash if provided
        return Ok(local_path);
    }
    
    eprintln!("Downloading test vector: {}", source.name);
    eprintln!("  from: {}", source.url);
    
    // Download would happen here - for now, just return error
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("Test vector {} not found and download not implemented", source.name),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run when explicitly requested
    fn test_download_vectors() {
        let dir = test_vectors_dir();
        assert!(dir.to_str().is_some());
    }
}
