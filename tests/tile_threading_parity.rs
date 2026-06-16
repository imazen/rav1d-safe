//! Pixel-parity regression test for the thread-local compact scratch pool
//! (issue #17).
//!
//! Under tile threading (`n_tc > 1`) the loop filter, ipred CFL, and
//! `with_pixel_guard_*` paths materialize each filtered edge / predicted block
//! into a compact per-row buffer. Issue #17 replaced the per-call
//! `vec![0u8; …]` in `compact_read_per_row` with a reused thread-local scratch
//! buffer to kill ~3M alloc+free pairs per 8K frame. Buffer reuse is only sound
//! if every read fully overwrites the region it hands out — a stale-data leak
//! would silently corrupt pixels.
//!
//! AV1 decoding is deterministic: tile-threaded output MUST be byte-identical to
//! single-threaded output. So we decode each committed vector both ways and
//! assert the decoded frames hash identically. `threads = 1` keeps tile
//! threading OFF (zero-copy `narrow_guard` path); `threads = 4` turns it ON
//! (`n_tc = 4` → `set_tile_threading(true)` → pooled compact path). Any
//! divergence is a corruption bug in the pooled buffer.
//!
//! **Requires `--release`** — debug mode is too slow for decode tests.
//!
//! Run: cargo test --release --test tile_threading_parity

#[cfg(debug_assertions)]
compile_error!("tile_threading_parity tests require release mode: cargo test --release");

use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};

/// Decode every frame in `data` and return the MD5 of all plane bytes, plus the
/// number of frames hashed. `threads = 1` exercises the single-threaded
/// zero-copy path; `threads >= 2` forces `n_tc > 1`, enabling the tile-threaded
/// pooled compact path. `max_frame_delay = 1` pins `n_fc = 1` so decoding stays
/// synchronous (pure tile threading, no frame threading).
fn decode_md5(data: &[u8], threads: u32) -> Result<(String, usize), String> {
    let mut settings = Settings::default();
    settings.threads = threads;
    // Pin n_fc = 1 so decoding stays synchronous (pure tile threading).
    settings.max_frame_delay = 1;
    let mut decoder = Decoder::with_settings(settings).map_err(|e| format!("create: {e:?}"))?;
    let mut ctx = md5::Context::new();
    let mut frames = 0usize;

    match decoder.decode(data) {
        Ok(Some(frame)) => {
            hash_frame(&frame, &mut ctx);
            frames += 1;
        }
        Ok(None) => {}
        Err(e) => return Err(format!("decode: {e:?}")),
    }
    match decoder.flush() {
        Ok(remaining) => {
            for frame in &remaining {
                hash_frame(frame, &mut ctx);
                frames += 1;
            }
        }
        Err(e) => return Err(format!("flush: {e:?}")),
    }
    Ok((format!("{:x}", ctx.finalize()), frames))
}

/// Hash all active plane pixels of a frame. Mirrors the plane walk in
/// `decode_md5_verify::hash_frame`.
fn hash_frame(frame: &Frame, ctx: &mut md5::Context) {
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                ctx.consume(row);
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    ctx.consume(row);
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    ctx.consume(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            for row in planes.y().rows() {
                for &px in row {
                    ctx.consume(px.to_le_bytes());
                }
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
        }
    }
}

/// Assert single-threaded and tile-threaded decode of `data` produce identical
/// frames. `label` is for diagnostics.
fn assert_parity(label: &str, data: &[u8]) {
    let (st_md5, st_frames) = decode_md5(data, 1)
        .unwrap_or_else(|e| panic!("{label}: single-threaded decode failed: {e}"));
    let (mt_md5, mt_frames) =
        decode_md5(data, 4).unwrap_or_else(|e| panic!("{label}: tile-threaded decode failed: {e}"));

    assert!(st_frames > 0, "{label}: single-threaded produced no frames");
    assert_eq!(
        st_frames, mt_frames,
        "{label}: frame count differs ST={st_frames} MT={mt_frames}"
    );
    assert_eq!(
        st_md5, mt_md5,
        "{label}: tile-threaded output diverged from single-threaded \
         (pooled compact buffer corrupted pixels) — ST={st_md5} MT={mt_md5}"
    );
    eprintln!("{label}: ST==MT md5={st_md5} ({st_frames} frame(s))");
}

/// 8bpc 4:2:0 real photo — exercises the 8bpc loop-filter compact path and
/// ipred CFL (chroma-from-luma) compact read.
#[test]
fn parity_kodim03_8bpc_420() {
    assert_parity(
        "kodim03_8bpc_420",
        include_bytes!("crash_vectors/kodim03_yuv420_8bpc.obu"),
    );
}

/// 10/12bpc HDR — exercises the 16bpc loop-filter compact path.
#[test]
fn parity_colors_hdr_rec2020_16bpc() {
    assert_parity(
        "colors_hdr_rec2020_16bpc",
        include_bytes!("crash_vectors/colors_hdr_rec2020.obu"),
    );
}

/// Crafted stream whose tile-thread loop filter accesses overlapping pixel
/// regions — the original motivation for the per-row compact decomposition.
#[test]
fn parity_disjoint_mut_tile_overlap() {
    assert_parity(
        "disjoint_mut_tile_overlap",
        include_bytes!("crash_vectors/disjoint_mut_tile_overlap.obu"),
    );
}
