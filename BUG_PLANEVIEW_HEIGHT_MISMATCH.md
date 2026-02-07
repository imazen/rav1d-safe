# Bug Report: PlaneView Height Mismatch

**Date:** 2026-02-07  
**Severity:** High - Causes decoding failures and panics  
**Component:** `src/managed.rs` - PlaneView8/PlaneView16 construction

## Summary

`PlaneView16` (and possibly `PlaneView8`) reports a `height` value that doesn't match the actual buffer size, causing "out of bounds" panics and size validation failures in downstream code.

## Symptoms

When decoding certain AVIF files, the PlaneView reports metadata that is inconsistent with the actual buffer:

- PlaneView.height() returns value H
- PlaneView.as_slice().len() returns buffer of size S
- But S != stride × H (the expected invariant)

This causes:
1. **Panic in PlaneView::row()**: `range end index X out of range for slice of length Y`
2. **Size validation failures**: Libraries expecting `stride × height` bytes get smaller buffers

## Reproduction

### Test Files

Located in: `/home/lilith/work/zenavif/tests/vectors/libavif/`

**Primary test case:**
- `color_nogrid_alpha_nogrid_gainmap_grid.avif`

**Other affected files (10 total):**
1. `color_nogrid_alpha_nogrid_gainmap_grid.avif`
2. `cosmos1650_yuv444_10bpc_p3pq.avif`
3. `seine_hdr_gainmap_small_srgb.avif`
4. `seine_hdr_gainmap_srgb.avif`
5. `seine_hdr_gainmap_wrongaltr.avif`
6. `supported_gainmap_writer_version_with_extra_bytes.avif`
7. `unsupported_gainmap_minimum_version.avif`
8. `unsupported_gainmap_version.avif`
9. `unsupported_gainmap_writer_version_with_extra_bytes.avif`
10. `weld_sato_12B_8B_q0.avif`

**Pattern:** Many affected files are gainmap-related, suggesting the bug may be triggered by specific AV1 features or metadata.

### Reproduction Code

```rust
use rav1d_safe::src::managed::Decoder;
use std::fs;

fn main() {
    let data = fs::read("/home/lilith/work/zenavif/tests/vectors/libavif/color_nogrid_alpha_nogrid_gainmap_grid.avif").unwrap();
    
    // Parse AVIF to get raw AV1 bitstream
    let parsed = avif_parse::read_avif(&mut std::io::Cursor::new(&data)).unwrap();
    
    // Decode with rav1d-safe
    let mut decoder = Decoder::new().unwrap();
    let frame = decoder.decode(&parsed.primary_item).unwrap().unwrap();
    
    // Check for height mismatch
    use rav1d_safe::src::managed::Planes;
    if let Planes::Depth16(planes) = frame.planes() {
        let y = planes.y();
        let expected_buffer_size = y.height() * y.stride();
        let actual_buffer_size = y.as_slice().len();
        
        println!("Width: {}, Height: {}, Stride: {}", 
            y.width(), y.height(), y.stride());
        println!("Expected buffer size: {} (height × stride)", expected_buffer_size);
        println!("Actual buffer size: {}", actual_buffer_size);
        println!("Mismatch: {} bytes", expected_buffer_size as isize - actual_buffer_size as isize);
    }
}
```

### Expected Output

```
Width: 128, Height: 200, Stride: 256
Expected buffer size: 51200 (height × stride)
Actual buffer size: 32768
Mismatch: 18432 bytes
```

### Actual Behavior

The PlaneView reports:
- `height = 200`
- `stride = 256`  
- `buffer.len() = 32768`

But `32768 / 256 = 128 rows`, not 200 rows!

The buffer actually contains **128 rows**, but the metadata claims **200 rows**.

## Detailed Evidence

Debug output showing the mismatch:

```
DEBUG planar setup: width=128 height=200 sampling=Cs444
  Y: 128x200 stride=256 buffer_len=32768
  U: 128x200 stride=256 buffer_len=32768
  V: 128x200 stride=256 buffer_len=32768
```

Analysis:
- Reported: width=128, height=200, stride=256
- Buffer: 32,768 elements
- Expected: 256 × 200 = 51,200 elements
- Actual rows: 32,768 / 256 = **128 rows** (not 200!)

## Root Cause Location

**File:** `src/managed.rs`

The bug is in how `PlaneView8`/`PlaneView16` are constructed. Likely locations:

1. **Frame::planes() implementation** (lines ~850-950)
   - Constructs PlaneView from Rav1dPictureDataComponent
   - May be reading wrong height field from frame metadata

2. **PlaneView construction** (lines ~700-760)
   - Takes width, height, stride, and DisjointImmutGuard
   - Should validate: `height * stride <= buffer.len()`

3. **DisjointImmutGuard creation** (called from planes())
   - May be creating guard with wrong slice bounds

### Suspected Issue

The frame's `height` metadata may include padding or total allocation size, but the actual pixel data buffer is smaller. The PlaneView constructor should be using the **buffer's actual row count** instead of the metadata height.

Possible fix location (pseudocode):
```rust
// In PlaneView construction
let actual_height = buffer.len() / stride;
if actual_height != reported_height {
    // Use actual_height or return error
}
```

## Impact

This bug affects **10 out of 55 test files** (18.2%) in the zenavif test suite, all causing decode failures.

### Downstream Impact

Any library using rav1d-safe's managed API with these files will:
1. Get size validation errors from YUV conversion libraries
2. Panic when iterating past the actual buffer bounds
3. Silently produce incorrect output if bounds checks are disabled

## Test Infrastructure

### Running the Test Suite

From `/home/lilith/work/zenavif`:

```bash
# Download test vectors (if not already present)
bash scripts/download-avif-test-vectors.sh

# Run integration tests
cargo test --release --test integration_corpus test_decode_all_vectors -- --ignored --nocapture
```

Expected failures: 10 files with "Luma plane have invalid size" errors.

### Minimal Reproduction

```bash
# Clone zenavif for test files
git clone /home/lilith/work/zenavif /tmp/zenavif-test
cd /tmp/zenavif-test

# Build minimal reproducer
cat > test_height_bug.rs << 'RUST'
use rav1d_safe::src::managed::Decoder;

fn main() {
    let data = std::fs::read("tests/vectors/libavif/color_nogrid_alpha_nogrid_gainmap_grid.avif").unwrap();
    let parsed = avif_parse::read_avif(&mut std::io::Cursor::new(&data)).unwrap();
    
    let mut decoder = Decoder::new().unwrap();
    if let Some(frame) = decoder.decode(&parsed.primary_item).unwrap() {
        use rav1d_safe::src::managed::Planes;
        if let Planes::Depth16(planes) = frame.planes() {
            let y = planes.y();
            assert_eq!(y.as_slice().len(), y.height() * y.stride(), 
                "Height mismatch: buffer has {} bytes but height {} × stride {} = {} bytes",
                y.as_slice().len(), y.height(), y.stride(), y.height() * y.stride());
        }
    }
}
RUST

# Run (will panic with assertion failure)
cargo run --release
```

## Workarounds

Until fixed, downstream code can:

1. **Validate before use:**
   ```rust
   let actual_height = plane.as_slice().len() / plane.stride();
   if actual_height != plane.height() {
       // Use actual_height instead
   }
   ```

2. **Use single-threaded decoding:**
   ```rust
   let settings = Settings { threads: 1, ..Default::default() };
   ```
   (Reduces but doesn't eliminate the issue)

## Expected Behavior

PlaneView should maintain the invariant:
```rust
assert!(plane.height() * plane.stride() <= plane.as_slice().len());
```

Either:
- Fix the height to match actual buffer rows, OR
- Extend the buffer to match the reported height, OR  
- Return an error during construction if the invariant is violated

## Additional Notes

- All affected files decode successfully with libavif's dav1d decoder
- The bug appears related to gainmap/HDR metadata handling
- May be specific to certain AV1 encoding parameters or frame types
- Affects both primary color frames and alpha frames

## References

- Investigation details: `/home/lilith/work/zenavif/CLAUDE.md` (Investigation Notes section)
- Session summary: `/home/lilith/work/zenavif/SESSION_SUMMARY.md`
- Test infrastructure: `/home/lilith/work/zenavif/tests/integration_corpus.rs`
- Debug tool: `/home/lilith/work/zenavif/examples/debug_bounds.rs`

## Contact

This bug was discovered during automated testing of the zenavif AVIF decoder.
For questions or additional test cases, see: https://github.com/imazen/zenavif
