//! AV1 Annex B demuxer for test infrastructure
//!
//! Annex B format wraps raw OBUs with LEB128-encoded size fields at three levels:
//! temporal_unit_size → frame_unit_size → obu_length.
//!
//! OBUs in Annex B typically have `obu_has_size_field = 0` (sizes come from the
//! wrapper). The rav1d decoder expects Section 5 (low-overhead) format where
//! `obu_has_size_field = 1`. This parser converts OBUs to Section 5 format.

/// Parse a LEB128-encoded unsigned integer from a byte slice.
/// Returns (value, bytes_consumed) or None if invalid/truncated.
fn parse_leb128(data: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    for i in 0..8 {
        if i >= data.len() {
            return None;
        }
        let byte = data[i];
        value |= ((byte & 0x7F) as u64) << (i * 7);
        if (byte & 0x80) == 0 {
            return Some((value, i + 1));
        }
    }
    None // More than 8 bytes — invalid
}

/// Encode a value as LEB128 and append to buf.
fn encode_leb128(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// A temporal unit extracted from an Annex B stream.
/// Contains OBU data converted to Section 5 (low-overhead) format.
pub struct AnnexBTemporalUnit {
    pub data: Vec<u8>,
}

/// Parse an Annex B stream into temporal units with Section 5 formatted OBUs.
///
/// Each temporal unit is converted so every OBU has `obu_has_size_field = 1`
/// with the payload size encoded as LEB128 after the header.
pub fn parse_annexb(data: &[u8]) -> Result<Vec<AnnexBTemporalUnit>, String> {
    let mut units = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        // Parse temporal_unit_size
        let (tu_size, tu_leb_len) =
            parse_leb128(&data[pos..]).ok_or("truncated temporal_unit_size")?;
        pos += tu_leb_len;
        let tu_size = tu_size as usize;

        if pos + tu_size > data.len() {
            return Err(format!(
                "temporal unit size {} exceeds remaining data {} at offset {}",
                tu_size,
                data.len() - pos,
                pos
            ));
        }

        let tu_end = pos + tu_size;
        let mut tu_data = Vec::new();

        while pos < tu_end {
            // Parse frame_unit_size
            let (fu_size, fu_leb_len) =
                parse_leb128(&data[pos..]).ok_or("truncated frame_unit_size")?;
            pos += fu_leb_len;
            let fu_size = fu_size as usize;

            if pos + fu_size > tu_end {
                return Err(format!(
                    "frame unit size {} exceeds temporal unit bounds at offset {}",
                    fu_size, pos
                ));
            }

            let fu_end = pos + fu_size;

            while pos < fu_end {
                // Parse obu_length (total OBU size: header + payload)
                let (obu_len, obu_leb_len) =
                    parse_leb128(&data[pos..]).ok_or("truncated obu_length")?;
                pos += obu_leb_len;
                let obu_len = obu_len as usize;

                if pos + obu_len > fu_end {
                    return Err(format!(
                        "OBU length {} exceeds frame unit bounds at offset {}",
                        obu_len, pos
                    ));
                }

                if obu_len == 0 {
                    continue;
                }

                // Read OBU header byte
                let header_byte = data[pos];
                let has_extension = (header_byte >> 2) & 1;
                let has_size = (header_byte >> 1) & 1;
                let header_size = 1 + has_extension as usize;

                if obu_len < header_size {
                    return Err(format!(
                        "OBU length {} too small for header at offset {}",
                        obu_len, pos
                    ));
                }

                let payload_start = pos + header_size;
                let payload_size = obu_len - header_size;

                if has_size == 1 {
                    // OBU already has size field — pass through as-is
                    tu_data.extend_from_slice(&data[pos..pos + obu_len]);
                } else {
                    // Set obu_has_size_field = 1 (bit 1)
                    let modified_header = header_byte | 0x02;
                    tu_data.push(modified_header);

                    // Copy extension byte if present
                    if has_extension == 1 {
                        tu_data.push(data[pos + 1]);
                    }

                    // Encode payload size as LEB128
                    encode_leb128(payload_size as u64, &mut tu_data);

                    // Copy payload
                    tu_data.extend_from_slice(&data[payload_start..payload_start + payload_size]);
                }

                pos += obu_len;
            }
        }

        if !tu_data.is_empty() {
            units.push(AnnexBTemporalUnit { data: tu_data });
        }
    }

    Ok(units)
}
