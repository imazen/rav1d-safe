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

/// Decode `data` with `threads` worker threads and return the YUV MD5 of all
/// decoded frames. Output must not depend on the thread count — AV1 decode is
/// deterministic — so every caller asserts against the same reference MD5.
///
/// `max_frame_delay = 1` pins `n_fc = 1` (pure tile threading, synchronous
/// decode) like every other threaded test in this suite: with frame threading
/// enabled instead, `decode()` may legitimately return `None` with the frame
/// still in flight, and the managed `flush()` — which has `rav1d_flush`
/// reset-and-discard semantics — then DROPS that frame rather than draining
/// it (observed on the `asm` CI flavor: 0 frames hashed). That drain footgun
/// is tracked separately; these committed-vector tests must not depend on it.
/// Decode errors panic — a committed conformant vector failing to decode is a
/// bug, never something to hash around.
fn decode_md5_with_threads(data: &[u8], threads: u32) -> String {
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = 1;
    settings.frame_size_limit = 8192 * 8192;
    let mut d = Decoder::with_settings(settings).expect("decoder");
    let mut ctx = md5::Context::new();
    if let Some(f) = d.decode(data).expect("decode error on committed vector") {
        hash_frame(&f, &mut ctx);
    }
    for f in &d.flush().expect("flush error on committed vector") {
        hash_frame(f, &mut ctx);
    }
    format!("{:x}", ctx.finalize())
}

/// Decode `data` single-threaded and return the YUV MD5 of all decoded frames.
fn decode_md5(data: &[u8]) -> String {
    decode_md5_with_threads(data, 1)
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

// ----------------------------------------------------------------------------
// Issue #14: aarch64 loop-restoration (SGR/wiener) regression vectors.
//
// Conformant still-picture streams whose loop-restoration selection exercises
// every looprestoration_arm code path that was broken in releases <= 0.5.7.
// Generated with zenrav1e @ ac8c4ef3 (still_picture, threads(1), 256x256 or
// 640x256 synthetic content — the exact encoder + content shape of zenrav1e's
// `intrabc_fires_and_roundtrips_both_samplings` CI test that first hit the
// panic in CI). Reference MD5s are the x86_64 decode, byte-identical to
// `aomdec --rawvideo` output for every vector (verified 2026-07-04 on a
// Neoverse-N1 vs x86_64: see rav1d-safe#14 for the full matrix).
//
// What each vector proved against registry 0.5.7 on native aarch64:
//   * lr_sgr_8bpc_glyph_s2 / lr_sgr_444_8bpc_glyph_s2 (SGR 3x3+5x5, 8bpc):
//     debug builds panic `attempt to multiply with overflow` at
//     looprestoration_arm.rs:465:30 (selfguided aa_base underflow — the
//     zenrav1e ARM CI failure); release builds panic `index out of bounds:
//     the len is 26520 but the index is 26520` at :399:13 (boxsum3_8bpc).
//   * lr_sgr_10bpc_glyph_nocdef / lr_sgr_10bpc_wide_nocdef (SGR 3x3, 10-bit):
//     the 16bpc twins — :999:30 (debug) / :935:13 (release, the exact panic
//     reported in issue #14 from Apple-Silicon AVIF decodes).
//   * lr_wiener5_8bpc_intrabc_s2 (wiener5): decoded WITHOUT error on 0.5.7
//     but with wrong pixels (mis-centered 5-tap window) — the MD5 pin is what
//     catches it.
//   * lr_sgr_10bpc_noisy_nocdef (SGR 5x5-only, 10-bit): also silently wrong
//     on 0.5.7 (untruncated-vs-rounded coefficient scaling), no panic.
//
// The 10-bit vectors are encoded with CDEF disabled so their whole-frame MD5
// isolates loop restoration: the aarch64 16bpc CDEF divergence tracked in
// issue #414 is still open and would otherwise mask/fail these pins on arm64.
// ----------------------------------------------------------------------------
const LR_SGR_VECTORS: &[(&str, &[u8], &str)] = &[
    (
        "lr_sgr_8bpc_glyph_s2",
        include_bytes!("crash_vectors/lr_sgr_8bpc_glyph_s2.obu"),
        "cb6f47fede190e68f39bbb688281d6a1",
    ),
    (
        "lr_wiener5_8bpc_intrabc_s2",
        include_bytes!("crash_vectors/lr_wiener5_8bpc_intrabc_s2.obu"),
        "c34a47ad7fc9795cb0145d68c69d586a",
    ),
    (
        "lr_sgr_444_8bpc_glyph_s2",
        include_bytes!("crash_vectors/lr_sgr_444_8bpc_glyph_s2.obu"),
        "d5373e2a15659807e7c11f15978b51ca",
    ),
    (
        "lr_sgr_10bpc_glyph_nocdef",
        include_bytes!("crash_vectors/lr_sgr_10bpc_glyph_nocdef.obu"),
        "18ded6a8434cbece969316339fb58c50",
    ),
    (
        "lr_sgr_10bpc_noisy_nocdef",
        include_bytes!("crash_vectors/lr_sgr_10bpc_noisy_nocdef.obu"),
        "bafc939f19adc1451c122baa7a8c828d",
    ),
    (
        "lr_sgr_10bpc_wide_nocdef",
        include_bytes!("crash_vectors/lr_sgr_10bpc_wide_nocdef.obu"),
        "f4b701d7130860b210f3448002fe466b",
    ),
];

/// Issue #14 (single-threaded): the LR vectors decode without panicking and
/// bit-identical to the x86/aomdec reference on every arch.
#[test]
fn lr_sgr_vectors_match_reference_md5() {
    let arch = std::env::consts::ARCH;
    let mut failures = Vec::new();
    for (label, data, expected) in LR_SGR_VECTORS {
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
        "LR decode diverged from the reference (issue #14 regression):\n{}",
        failures.join("\n")
    );
}

/// Issue #14 (worker-threaded): same streams through an 8-worker decode — the
/// exact shape of the original report (`rav1d-worker-N` threads panicking in
/// `selfguided_filter_16bpc` on a threaded AVIF decode). Output must be
/// bit-identical to the single-threaded/x86/aomdec reference.
///
/// Not compiled under `__simd_test`: the dual-compute harness takes a
/// full-plane guard around each filter call, which (correctly) trips the
/// DisjointMut overlap detector against concurrent worker writes — the
/// harness is single-thread-only by design.
#[cfg(not(feature = "__simd_test"))]
#[test]
fn lr_sgr_vectors_threaded_match_reference_md5() {
    let arch = std::env::consts::ARCH;
    let mut failures = Vec::new();
    for (label, data, expected) in LR_SGR_VECTORS {
        let actual = decode_md5_with_threads(data, 8);
        if actual != *expected {
            failures.push(format!(
                "{label} threaded on {arch}: expected {expected}, got {actual}"
            ));
        } else {
            eprintln!("{label} threaded on {arch}: md5={actual} OK");
        }
    }
    assert!(
        failures.is_empty(),
        "threaded LR decode diverged from the reference (issue #14 regression):\n{}",
        failures.join("\n")
    );
}

// ----------------------------------------------------------------------------
// Issue #446: aarch64 12bpc CDEF primary-tap selection.
//
// The aarch64 CDEF port (`src/safe_simd/cdef_arm.rs`) took over every CDEF
// call on ARM and, in its 16bpc filter, selected `pri_tap` from bit 0 of
// `pri_strength` instead of bit `bitdepth_min_8`. The caller has already
// scaled the level (`y_pri_lvl = (y_lvl >> 2) << bitdepth_min_8`), so at 10/12
// bits that read the wrong bit and filtered with tap 4 where dav1d/libaom use
// tap 3 — wrong pixels on every high-bit-depth frame whose CDEF fired, intra
// and inter alike. dav1d does the shift in the matching NEON path
// (`src/arm/64/cdef_tmpl.S`, `lsr w9, w3, w9` under `.if \bpc == 16`).
//
// Vector: the color track of libavif's
// `colors-animated-12bpc-keyframes-0-2-3.avif` (profile 2, 64x64, 12bpc,
// 4:2:2, SB128, frames KEY/INTER/KEY/KEY/INTER). The per-shown-frame MD5s
// below are the C oracle's, not ours: `aomdec 3.14.1 --rawvideo` and
// `dav1d 1.5.4 --muxer yuv` both produce exactly these bytes.
const CDEF_12BPC_OBU: &[u8] = include_bytes!("crash_vectors/arm_cdef_12bpc_422_pri_tap.obu");

/// One MD5 per shown frame over Y,U,V at cropped dims, 16-bit little-endian.
const CDEF_12BPC_FRAME_MD5S: &[&str] = &[
    "ef8b977a24cd428ae6a7626968b694c4",
    "9f2291bf3f75dd440bc1d64ae26e0ac8",
    "d4c7026d6bf22d5b973d59adbe73043d",
    "484033dc917f8c49aba051070733e3ce",
    "af113a069ef6a383e80522ce50943a99",
];

#[test]
fn cdef_12bpc_422_matches_c_oracle_per_frame() {
    let mut settings = Settings::default();
    settings.threads = 1;
    settings.apply_grain = false;
    let mut d = Decoder::with_settings(settings).expect("decoder");

    // A multi-temporal-unit raw-OBU stream: send once, then drain every
    // buffered shown frame (`flush()` would discard them).
    let mut got = Vec::new();
    let push = |f: &Frame, got: &mut Vec<String>| {
        let mut ctx = md5::Context::new();
        hash_frame(f, &mut ctx);
        got.push(format!("{:x}", ctx.finalize()));
    };
    if let Some(f) = d.decode(CDEF_12BPC_OBU).expect("decode error") {
        push(&f, &mut got);
    }
    while let Some(f) = d.get_frame().expect("drain error") {
        push(&f, &mut got);
    }

    assert_eq!(
        got.len(),
        CDEF_12BPC_FRAME_MD5S.len(),
        "expected {} shown frames, decoded {}",
        CDEF_12BPC_FRAME_MD5S.len(),
        got.len()
    );
    let mut failures = Vec::new();
    for (i, (a, e)) in got.iter().zip(CDEF_12BPC_FRAME_MD5S).enumerate() {
        if a != e {
            failures.push(format!("frame {i}: expected {e}, got {a}"));
        }
    }
    assert!(
        failures.is_empty(),
        "12bpc 4:2:2 decode diverged from the aomdec/dav1d oracle on {} (issue #446):\n{}",
        std::env::consts::ARCH,
        failures.join("\n")
    );
}
