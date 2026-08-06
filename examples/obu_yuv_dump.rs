//! Decode a Section-5 raw-OBU stream and dump cropped planar YUV, plus a
//! per-shown-frame md5, in the exact layout `aomdec --rawvideo` writes.
//!
//! This is the differential harness for cross-decoder settling work: the
//! output file is byte-comparable with `aomdec --rawvideo -o out.yuv in.obu`,
//! and the per-frame md5s are comparable with libavif-style `<vector>.md5`
//! goldens (one md5 per shown frame over Y[,U,V], cropped dims, 1 byte/sample
//! at bd8 and 2 bytes little-endian above).
//!
//! Accepts a Section-5 raw-OBU stream or an IVF file (auto-detected), so the
//! same explicit settings (grain off, single-threaded, selectable in-loop
//! filters and CPU level) apply across the whole test corpus.
//!
//! Usage:
//!   cargo build --release --example obu_yuv_dump
//!   ./target/release/examples/obu_yuv_dump <input> <output.yuv> [--tu-split] [--threads N]
//!
//! `--tu-split` feeds one temporal unit per `decode()` call (split at
//! OBU_TEMPORAL_DELIMITER) instead of handing the whole stream to the decoder
//! at once; both must produce identical output.

use rav1d_safe::src::managed::{CpuLevel, Decoder, Frame, InloopFilters, Planes, Settings};
use std::env;
use std::fs;
use std::io::{BufWriter, Cursor, Write};

#[path = "helpers/ivf_parser.rs"]
mod ivf_parser;

/// Split a Section-5 raw-OBU stream into temporal units at each
/// OBU_TEMPORAL_DELIMITER. Returns byte ranges covering the whole input.
fn split_temporal_units(data: &[u8]) -> Result<Vec<std::ops::Range<usize>>, String> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        let hdr = data[i];
        if hdr & 0x80 != 0 {
            return Err(format!("forbidden bit set in OBU header at {i}"));
        }
        let obu_type = (hdr >> 3) & 0xF;
        let has_ext = (hdr >> 2) & 1 == 1;
        let has_size = (hdr >> 1) & 1 == 1;
        if !has_size {
            return Err(format!("OBU at {i} has no size field (not Section 5)"));
        }
        let mut j = i + 1 + usize::from(has_ext);
        let mut size = 0usize;
        let mut shift = 0u32;
        loop {
            let b = *data
                .get(j)
                .ok_or_else(|| format!("truncated leb128 at {j}"))?;
            j += 1;
            size |= ((b & 0x7f) as usize) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
            if shift > 56 {
                return Err(format!("leb128 too long at {i}"));
            }
        }
        if obu_type == 2 {
            starts.push(i);
        }
        i = j
            .checked_add(size)
            .filter(|&e| e <= data.len())
            .ok_or_else(|| format!("OBU at {i} overruns input"))?;
    }
    if starts.is_empty() {
        starts.push(0);
    }
    if starts[0] != 0 {
        starts.insert(0, 0);
    }
    let mut ranges = Vec::with_capacity(starts.len());
    for k in 0..starts.len() {
        let end = starts.get(k + 1).copied().unwrap_or(data.len());
        ranges.push(starts[k]..end);
    }
    Ok(ranges)
}

fn frame_bytes(frame: &Frame) -> Vec<u8> {
    let mut out = Vec::new();
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                out.extend_from_slice(row);
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    out.extend_from_slice(row);
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    out.extend_from_slice(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            for row in planes.y().rows() {
                for &px in row {
                    out.extend_from_slice(&px.to_le_bytes());
                }
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &px in row {
                        out.extend_from_slice(&px.to_le_bytes());
                    }
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    for &px in row {
                        out.extend_from_slice(&px.to_le_bytes());
                    }
                }
            }
        }
    }
    out
}

fn emit(frame: &Frame, idx: &mut usize, out: &mut impl Write) {
    let bytes = frame_bytes(frame);
    let mut h = md5::Context::new();
    h.consume(&bytes);
    #[allow(deprecated)]
    let digest = h.compute();
    eprintln!(
        "frame {} md5={:x} {}x{} bpc={} layout={:?} bytes={}",
        *idx,
        digest,
        frame.width(),
        frame.height(),
        frame.bit_depth(),
        frame.pixel_layout(),
        bytes.len()
    );
    out.write_all(&bytes).expect("write failed");
    *idx += 1;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut positional = Vec::new();
    let mut tu_split = false;
    let mut threads = 1u32;
    let mut filters = InloopFilters::all();
    let mut cpu_level = CpuLevel::Native;
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tu-split" => tu_split = true,
            "--threads" => threads = it.next().expect("--threads needs a value").parse().unwrap(),
            // Mirrors dav1d's `--inloopfilters none|deblock|cdef|restoration|all`
            // so both decoders can be gated identically for stage bisection.
            "--inloopfilters" => {
                let v = it.next().expect("--inloopfilters needs a value");
                filters = InloopFilters::none();
                for part in v.split('+') {
                    filters = match part {
                        "none" => InloopFilters::none(),
                        "all" => InloopFilters::all(),
                        "deblock" => filters.union(InloopFilters::DEBLOCK),
                        "cdef" => filters.union(InloopFilters::CDEF),
                        "restoration" => filters.union(InloopFilters::RESTORATION),
                        other => panic!("unknown inloop filter {other}"),
                    };
                }
            }
            "--cpu-level" => {
                let v = it.next().expect("--cpu-level needs a value");
                cpu_level = *CpuLevel::platform_levels()
                    .iter()
                    .find(|l| l.name() == v)
                    .unwrap_or_else(|| panic!("unknown cpu level {v}"));
            }
            other => positional.push(other.to_string()),
        }
    }
    if positional.len() < 2 {
        eprintln!(
            "Usage: obu_yuv_dump <input.obu> <output.yuv> [--tu-split] [--threads N] \
             [--inloopfilters none|deblock|cdef|restoration|all(+combined)] [--cpu-level NAME]"
        );
        std::process::exit(2);
    }

    let data = fs::read(&positional[0]).expect("read input");
    let mut out = BufWriter::new(fs::File::create(&positional[1]).expect("create output"));

    let mut settings = Settings::default();
    settings.threads = threads;
    settings.apply_grain = false;
    settings.inloop_filters = filters;
    settings.cpu_level = cpu_level;
    let mut dec = Decoder::with_settings(settings).expect("decoder");
    let mut idx = 0usize;

    // IVF is fed frame by frame; a raw-OBU stream is fed whole (or split at
    // temporal-delimiter boundaries with --tu-split).
    let is_ivf = data.len() >= 4 && &data[0..4] == b"DKIF";
    let chunks: Vec<std::ops::Range<usize>> = if is_ivf {
        let mut cursor = Cursor::new(&data);
        let frames = ivf_parser::parse_all_frames(&mut cursor).expect("IVF parse");
        eprintln!("fed as {} IVF frames", frames.len());
        // The parser copies frame payloads; rebuild ranges by re-scanning is
        // unnecessary — decode them directly here and skip the range path.
        for fr in &frames {
            match dec.decode(&fr.data) {
                Ok(Some(f)) => emit(&f, &mut idx, &mut out),
                Ok(None) => {}
                Err(e) => {
                    eprintln!("decode error: {e}");
                    std::process::exit(1);
                }
            }
            loop {
                match dec.get_frame() {
                    Ok(Some(f)) => emit(&f, &mut idx, &mut out),
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("drain error: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
        Vec::new()
    } else if tu_split {
        let r = split_temporal_units(&data).expect("TU split");
        eprintln!("fed as {} temporal units", r.len());
        r
    } else {
        eprintln!("fed as one buffer");
        // One chunk covering the whole stream. Spelled with `once` because a
        // single-range vec literal reads to clippy as a botched
        // `(0..n).collect()`.
        std::iter::once(0..data.len()).collect()
    };

    for r in chunks {
        match dec.decode(&data[r]) {
            Ok(Some(f)) => emit(&f, &mut idx, &mut out),
            Ok(None) => {}
            Err(e) => {
                eprintln!("decode error: {e}");
                std::process::exit(1);
            }
        }
        loop {
            match dec.get_frame() {
                Ok(Some(f)) => emit(&f, &mut idx, &mut out),
                Ok(None) => break,
                Err(e) => {
                    eprintln!("drain error: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
    match dec.flush() {
        Ok(rem) => {
            for f in &rem {
                emit(f, &mut idx, &mut out);
            }
        }
        Err(e) => eprintln!("flush error: {e}"),
    }
    out.flush().expect("flush output");
    eprintln!("frames: {idx}");
}
