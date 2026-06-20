//! Cross-architecture decode-correctness regression (issue #400).
//!
//! AV1 decode is bit-exact: a given stream must decode to identical pixels on
//! every architecture and CPU level. These committed OBU vectors are pinned to
//! their reference YUV MD5 (the x86_64 output, which passes the dav1d
//! conformance suite). The test runs on x86_64 (`cargo test`) AND on aarch64
//! under qemu (`cross test` in CI), so if an arch-specific kernel stops being
//! bit-exact — e.g. the aarch64 NEON inverse transforms that issue #400 gated
//! off for being off-by-up-to-15 — the aarch64 run diverges from these MD5s and
//! fails here instead of silently shipping wrong pixels.
//!
//! **Requires `--release`** — debug decode is too slow.
//!
//! Run: cargo test --release --test decode_md5_committed

#[cfg(debug_assertions)]
compile_error!("decode_md5_committed tests require release mode: cargo test --release");

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};

fn hash_frame(frame: &Frame, ctx: &mut md5::Context) {
    match frame.planes() {
        Planes::Depth8(p) => {
            for row in p.y().rows() {
                ctx.consume(row);
            }
            if let Some(u) = p.u() {
                for row in u.rows() {
                    ctx.consume(row);
                }
            }
            if let Some(v) = p.v() {
                for row in v.rows() {
                    ctx.consume(row);
                }
            }
        }
        Planes::Depth16(p) => {
            for row in p.y().rows() {
                for &px in row {
                    ctx.consume(px.to_le_bytes());
                }
            }
            if let Some(u) = p.u() {
                for row in u.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
            if let Some(v) = p.v() {
                for row in v.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
        }
    }
}

/// Decode `data` single-threaded and return the YUV MD5 of all decoded frames.
fn decode_md5(data: &[u8]) -> String {
    let mut settings = Settings::default();
    settings.threads = 1;
    settings.frame_size_limit = 8192 * 8192;
    let mut d = Decoder::with_settings(settings).expect("decoder");
    let mut ctx = md5::Context::new();
    if let Ok(Some(f)) = d.decode(data) {
        hash_frame(&f, &mut ctx);
    }
    if let Ok(rem) = d.flush() {
        for f in &rem {
            hash_frame(f, &mut ctx);
        }
    }
    format!("{:x}", ctx.finalize())
}

/// `(label, committed OBU, reference YUV MD5)`. Reference MD5s captured from the
/// x86_64 scalar+native decode (identical) on 2026-06-17.
const VECTORS: &[(&str, &[u8], &str)] = &[
    (
        "kodim03_yuv420_8bpc",
        include_bytes!("crash_vectors/kodim03_yuv420_8bpc.obu"),
        "f7de1083a1166170f8ae1f79328f275a",
    ),
    (
        "alpha_noispe",
        include_bytes!("crash_vectors/alpha_noispe.obu"),
        "c8863ea13a56b1ae731cdd23bcef40c8",
    ),
    (
        "colors_hdr_rec2020_16bpc",
        include_bytes!("crash_vectors/colors_hdr_rec2020.obu"),
        "d9c0ea6b0213b64132a65d3a7e76edf4",
    ),
    (
        "circle_custom_properties",
        include_bytes!("crash_vectors/circle_custom_properties.obu"),
        "bd06968f3606982bb9c398ad6f7f41c2",
    ),
    // Issue #400: 121x33 I400 stream whose top-left uses a 16x64 DCT_DCT DC-only
    // block. The aarch64 NEON `dc_only_rect64` helper applied the rect2 sqrt2
    // scaling unconditionally and used shift=1 instead of 2, biasing the DC by 1
    // (NEON decoded 126 where dav1d/scalar give 127). With __simd_test enabled
    // this vector also runs the per-transform NEON-vs-scalar bit-exactness gate.
    (
        "arm_itx_16x64_dc_rect2",
        include_bytes!("crash_vectors/arm_itx_16x64_dc_rect2.obu"),
        "ecc2a091a9f40fb0d126e5bb087e2c49",
    ),
];

#[test]
fn committed_vectors_match_reference_md5() {
    let arch = std::env::consts::ARCH;
    let mut failures = Vec::new();
    for (label, data, expected) in VECTORS {
        let actual = decode_md5(data);
        if actual != *expected {
            failures.push(format!(
                "{label} on {arch}: expected {expected}, got {actual}"
            ));
        } else {
            eprintln!("{label} on {arch}: md5={actual} OK");
        }
    }
    assert!(
        failures.is_empty(),
        "decode diverged from the reference (arch-specific non-bit-exact kernel?):\n{}",
        failures.join("\n")
    );
}
