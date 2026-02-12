//! Test vector management — auto-download and cache dav1d-test-data.
//!
//! Call `ensure_dav1d_test_data()` from any test that needs vectors.
//! It clones the repo on first use and panics if it can't.

use std::path::PathBuf;
use std::sync::Once;

static DOWNLOAD: Once = Once::new();

/// Returns the path to `test-vectors/dav1d-test-data/`, cloning the repo if needed.
///
/// Panics if the clone fails (network error, git not installed, etc).
/// The clone is shallow (`--depth 1`) and cached across test runs.
pub fn ensure_dav1d_test_data() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/dav1d-test-data");

    DOWNLOAD.call_once(|| {
        if dir.join("8-bit/data/meson.build").exists() {
            return; // already have it
        }

        eprintln!("Cloning dav1d-test-data (one-time download, ~109 MB)...");
        let parent = dir.parent().unwrap();
        std::fs::create_dir_all(parent).expect("failed to create test-vectors/");

        let status = std::process::Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "https://code.videolan.org/videolan/dav1d-test-data.git",
                dir.to_str().unwrap(),
            ])
            .status()
            .expect("failed to run git — is git installed?");

        assert!(
            status.success(),
            "git clone dav1d-test-data failed (exit {}). Check network connectivity.",
            status
        );
    });

    assert!(
        dir.join("8-bit/data/meson.build").exists(),
        "dav1d-test-data missing after download attempt. \
         Delete test-vectors/dav1d-test-data/ and re-run to retry."
    );

    dir
}
