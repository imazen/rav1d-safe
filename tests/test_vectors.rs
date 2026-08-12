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

        // Clone to a PID-unique staging directory and rename into place.
        //
        // `DOWNLOAD: Once` only serialises within one process, and nextest runs
        // every test in its OWN process — so all 13 conformance tests reached
        // this clone concurrently. The first created `dir`; the other twelve
        // hit `git clone`'s "destination path already exists" and died with
        // exit 128. That is the failure CI reported, and it could not
        // reproduce under `cargo test`, which is single-process and where the
        // `Once` does work.
        //
        // Staging + rename makes the race benign: every process does its own
        // clone, exactly one rename wins, and the losers see the sentinel and
        // discard their copy. Wasteful under a cold cache, correct always —
        // and in CI the fetch script populates this path first, so normally no
        // process clones at all.
        let staging = parent.join(format!("dav1d-test-data.staging.{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&staging);

        let status = std::process::Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "https://code.videolan.org/videolan/dav1d-test-data.git",
                staging.to_str().unwrap(),
            ])
            .status()
            .expect("failed to run git — is git installed?");

        assert!(
            status.success(),
            "git clone dav1d-test-data failed (exit {status}). Check network connectivity."
        );

        // Someone else may have won while we cloned; that is fine and expected.
        if std::fs::rename(&staging, &dir).is_err() {
            let _ = std::fs::remove_dir_all(&staging);
            assert!(
                dir.join("8-bit/data/meson.build").exists(),
                "could not move the clone into {} and no other process supplied it",
                dir.display()
            );
        }
    });

    assert!(
        dir.join("8-bit/data/meson.build").exists(),
        "dav1d-test-data missing after download attempt. \
         Delete test-vectors/dav1d-test-data/ and re-run to retry."
    );

    dir
}
