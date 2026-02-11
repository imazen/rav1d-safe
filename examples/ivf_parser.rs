//! Simple IVF container parser for test infrastructure
//!
//! IVF format: https://wiki.multimedia.cx/index.php/IVF

use std::io::{self, Read};

/// IVF file header (32 bytes)
#[derive(Debug)]
#[allow(dead_code)]
pub struct IvfHeader {
    pub width: u16,
    pub height: u16,
    pub timebase_num: u32,
    pub timebase_den: u32,
    pub num_frames: u32,
}

/// IVF frame header (12 bytes)
#[derive(Debug)]
#[allow(dead_code)]
pub struct IvfFrame {
    pub data: Vec<u8>,
    pub timestamp: u64,
}

pub fn parse_ivf_header<R: Read>(reader: &mut R) -> io::Result<IvfHeader> {
    let mut header = [0u8; 32];
    reader.read_exact(&mut header)?;

    // Check signature "DKIF"
    if &header[0..4] != b"DKIF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not an IVF file",
        ));
    }

    // Check version (should be 0)
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unsupported IVF version",
        ));
    }

    // Check codec "AV01"
    if &header[8..12] != b"AV01" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not an AV1 IVF file",
        ));
    }

    Ok(IvfHeader {
        width: u16::from_le_bytes([header[12], header[13]]),
        height: u16::from_le_bytes([header[14], header[15]]),
        timebase_num: u32::from_le_bytes([header[16], header[17], header[18], header[19]]),
        timebase_den: u32::from_le_bytes([header[20], header[21], header[22], header[23]]),
        num_frames: u32::from_le_bytes([header[24], header[25], header[26], header[27]]),
    })
}

pub fn parse_ivf_frame<R: Read>(reader: &mut R) -> io::Result<Option<IvfFrame>> {
    let mut frame_header = [0u8; 12];

    // Try to read frame header, return None on EOF
    match reader.read_exact(&mut frame_header) {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }

    let frame_size = u32::from_le_bytes([
        frame_header[0],
        frame_header[1],
        frame_header[2],
        frame_header[3],
    ]) as usize;

    let timestamp = u64::from_le_bytes([
        frame_header[4],
        frame_header[5],
        frame_header[6],
        frame_header[7],
        frame_header[8],
        frame_header[9],
        frame_header[10],
        frame_header[11],
    ]);

    let mut data = vec![0u8; frame_size];
    reader.read_exact(&mut data)?;

    Ok(Some(IvfFrame { data, timestamp }))
}

pub fn parse_all_frames<R: Read>(reader: &mut R) -> io::Result<Vec<IvfFrame>> {
    // Skip IVF header
    parse_ivf_header(reader)?;

    let mut frames = Vec::new();
    while let Some(frame) = parse_ivf_frame(reader)? {
        frames.push(frame);
    }

    Ok(frames)
}
