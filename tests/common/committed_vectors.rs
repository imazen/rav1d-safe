//! Shared committed vectors and visible-plane hashing for decode gates.

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings, Strictness};

pub(super) fn hash_frame(frame: &Frame, ctx: &mut md5::Context) {
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
pub(super) fn decode_md5_with_threads(data: &[u8], threads: u32) -> String {
    let mut settings = Settings::default();
    // These MD5s pin what every architecture must produce for the *same*
    // stream, conforming or not: `arm_itx_16x64_dc_rect2.obu` is a
    // fuzz-derived 37-byte stream the reference decoder rejects, and its MD5
    // documents dav1d-parity concealment through the 16x64 DC-only itx on
    // every arch. The production default is `Strict` since 0.6.0, which
    // refuses such streams up front; parity references need Lenient.
    settings.strictness = Strictness::Lenient;
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

/// `(label, committed OBU, reference YUV MD5)`. Reference MD5s captured from the
/// x86_64 scalar+native decode (identical) on 2026-06-17.
pub(super) const VECTORS: &[(&str, &[u8], &str)] = &[
    (
        "kodim03_yuv420_8bpc",
        include_bytes!("../crash_vectors/kodim03_yuv420_8bpc.obu"),
        "f7de1083a1166170f8ae1f79328f275a",
    ),
    (
        "alpha_noispe",
        include_bytes!("../crash_vectors/alpha_noispe.obu"),
        "c8863ea13a56b1ae731cdd23bcef40c8",
    ),
    (
        "colors_hdr_rec2020_16bpc",
        include_bytes!("../crash_vectors/colors_hdr_rec2020.obu"),
        "d9c0ea6b0213b64132a65d3a7e76edf4",
    ),
    (
        "circle_custom_properties",
        include_bytes!("../crash_vectors/circle_custom_properties.obu"),
        "bd06968f3606982bb9c398ad6f7f41c2",
    ),
    // Issue #400: 121x33 I400 stream whose top-left uses a 16x64 DCT_DCT DC-only
    // block. The aarch64 NEON `dc_only_rect64` helper applied the rect2 sqrt2
    // scaling unconditionally and used shift=1 instead of 2, biasing the DC by 1
    // (NEON decoded 126 where dav1d/scalar give 127). With __simd_test enabled
    // this vector also runs the per-transform NEON-vs-scalar bit-exactness gate.
    (
        "arm_itx_16x64_dc_rect2",
        include_bytes!("../crash_vectors/arm_itx_16x64_dc_rect2.obu"),
        "ecc2a091a9f40fb0d126e5bb087e2c49",
    ),
];
