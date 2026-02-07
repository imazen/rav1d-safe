# Managed Minimal Safe Rust API for rav1d-safe

**Status:** Design Proposal  
**Goal:** 100% safe Rust API with zero-copy output support for 8/10/12-bit, HDR metadata

## Design Principles

1. **100% Safe Rust** - No `unsafe` in public API surface
2. **Zero-Copy** - Return views/references to decoded data, not copies
3. **Reference Counting** - Use `Arc` for shared ownership of decoded frames
4. **Lifetime Safety** - Rust's borrow checker ensures data validity
5. **Type Safety** - Use enums and strong types, not raw integers
6. **Minimal Surface** - Small, focused API for common use cases

## Core Types

### Decoder Context

```rust
/// Safe AV1 decoder instance
pub struct Decoder {
    inner: Arc<Rav1dContext>,
}

impl Decoder {
    /// Create a new decoder with default settings
    pub fn new() -> Result<Self, Error> {
        Self::with_settings(Settings::default())
    }
    
    /// Create decoder with custom settings
    pub fn with_settings(settings: Settings) -> Result<Self, Error> {
        // Internally calls rav1d_open, wrapped safely
    }
    
    /// Decode AV1 OBU data from a byte slice
    /// 
    /// Returns `Ok(None)` if more data is needed (EAGAIN)
    /// Returns `Ok(Some(frame))` when a frame is ready
    pub fn decode(&mut self, data: &[u8]) -> Result<Option<Frame>, Error> {
        // Internally calls rav1d_send_data + rav1d_get_picture
        // Uses null free callback pattern like zenavif
        // Returns Frame which holds Arc to decoded data
    }
    
    /// Flush decoder and get any remaining frames
    pub fn flush(&mut self) -> Result<Vec<Frame>, Error> {
        // Internally calls rav1d_flush + rav1d_get_picture loop
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // Internally calls rav1d_close
    }
}
```

### Settings

```rust
/// Decoder configuration
#[derive(Clone, Debug)]
pub struct Settings {
    /// Number of threads (0 = auto-detect)
    pub threads: u32,
    
    /// Apply film grain synthesis
    pub apply_grain: bool,
    
    /// Maximum frame size in pixels (0 = unlimited)
    pub frame_size_limit: u32,
    
    /// Decode all layers or just the selected operating point
    pub all_layers: bool,
    
    /// Operating point to decode (0-31)
    pub operating_point: u8,
    
    /// Output invisible frames
    pub output_invisible_frames: bool,
    
    /// Inloop filters to apply
    pub inloop_filters: InloopFilters,
    
    /// Which frame types to decode
    pub decode_frame_type: DecodeFrameType,
    
    /// Strict standard compliance
    pub strict_std_compliance: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            threads: 0,
            apply_grain: true,
            frame_size_limit: 0,
            all_layers: true,
            operating_point: 0,
            output_invisible_frames: false,
            inloop_filters: InloopFilters::all(),
            decode_frame_type: DecodeFrameType::All,
            strict_std_compliance: false,
        }
    }
}

bitflags::bitflags! {
    /// Inloop filter flags
    pub struct InloopFilters: u8 {
        const DEBLOCK = 1 << 0;
        const CDEF = 1 << 1;
        const RESTORATION = 1 << 2;
    }
}

impl InloopFilters {
    pub fn all() -> Self {
        Self::DEBLOCK | Self::CDEF | Self::RESTORATION
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeFrameType {
    /// Decode all frame types
    All,
    /// Decode only reference frames
    Reference,
    /// Decode only intra frames
    Intra,
    /// Decode only key frames
    Key,
}
```

### Decoded Frame

```rust
/// A decoded AV1 frame with zero-copy access to pixel data
/// 
/// Holds an `Arc` to the underlying picture data, so cloning is cheap.
/// Multiple `Frame` instances can safely reference the same decoded data.
#[derive(Clone)]
pub struct Frame {
    inner: Arc<Rav1dPicture>,
}

impl Frame {
    /// Frame width in pixels
    pub fn width(&self) -> u32 {
        self.inner.p.w as u32
    }
    
    /// Frame height in pixels
    pub fn height(&self) -> u32 {
        self.inner.p.h as u32
    }
    
    /// Bit depth (8, 10, or 12)
    pub fn bit_depth(&self) -> u8 {
        self.inner.p.bpc
    }
    
    /// Pixel layout (chroma subsampling)
    pub fn pixel_layout(&self) -> PixelLayout {
        self.inner.p.layout
    }
    
    /// Access pixel data according to bit depth
    pub fn planes(&self) -> Planes<'_> {
        match self.bit_depth() {
            8 => Planes::Depth8(Planes8 { frame: self }),
            10 | 12 => Planes::Depth16(Planes16 { frame: self }),
            _ => unreachable!("invalid bit depth"),
        }
    }
    
    /// Color metadata
    pub fn color_info(&self) -> ColorInfo {
        ColorInfo {
            primaries: self.seq_hdr().pri,
            transfer_characteristics: self.seq_hdr().trc,
            matrix_coefficients: self.seq_hdr().mtrx,
            color_range: if self.seq_hdr().color_range() { 
                ColorRange::Full 
            } else { 
                ColorRange::Limited 
            },
        }
    }
    
    /// HDR content light level metadata, if present
    pub fn content_light(&self) -> Option<ContentLightLevel> {
        self.inner.content_light.as_ref().map(|arc| {
            ContentLightLevel {
                max_content_light_level: arc.max_content_light_level,
                max_frame_average_light_level: arc.max_frame_average_light_level,
            }
        })
    }
    
    /// HDR mastering display metadata, if present
    pub fn mastering_display(&self) -> Option<MasteringDisplay> {
        self.inner.mastering_display.as_ref().map(|arc| {
            MasteringDisplay {
                primaries: arc.primaries,
                white_point: arc.white_point,
                max_luminance: arc.max_luminance,
                min_luminance: arc.min_luminance,
            }
        })
    }
    
    /// Timestamp from input data
    pub fn timestamp(&self) -> i64 {
        self.inner.m.timestamp
    }
    
    /// Duration from input data
    pub fn duration(&self) -> i64 {
        self.inner.m.duration
    }
    
    // Internal helper
    fn seq_hdr(&self) -> &Rav1dSequenceHeader {
        &self.inner.seq_hdr.as_ref().unwrap().rav1d
    }
}
```

### Pixel Data Access (Zero-Copy)

```rust
/// Zero-copy access to pixel planes
/// 
/// Enum dispatches on bit depth for type safety
pub enum Planes<'a> {
    Depth8(Planes8<'a>),
    Depth16(Planes16<'a>),
}

/// 8-bit pixel plane access
pub struct Planes8<'a> {
    frame: &'a Frame,
}

impl<'a> Planes8<'a> {
    /// Y (luma) plane as a 2D strided view
    pub fn y(&self) -> PlaneView8<'a> {
        PlaneView8 {
            data: self.frame.inner.data.as_ref().unwrap().data[0]
                .slice::<BitDepth8, _>(..)  // DisjointImmutGuard
                .into(),  // Convert to safe slice reference
            stride: self.frame.inner.stride[0] as usize,
            width: self.frame.width() as usize,
            height: self.frame.height() as usize,
        }
    }
    
    /// U (chroma) plane, if present (None for grayscale)
    pub fn u(&self) -> Option<PlaneView8<'a>> {
        if self.frame.pixel_layout() == PixelLayout::I400 {
            return None;
        }
        let (w, h) = self.chroma_dimensions();
        Some(PlaneView8 {
            data: self.frame.inner.data.as_ref().unwrap().data[1]
                .slice::<BitDepth8, _>(..)
                .into(),
            stride: self.frame.inner.stride[1] as usize,
            width: w,
            height: h,
        })
    }
    
    /// V (chroma) plane, if present (None for grayscale)
    pub fn v(&self) -> Option<PlaneView8<'a>> {
        if self.frame.pixel_layout() == PixelLayout::I400 {
            return None;
        }
        let (w, h) = self.chroma_dimensions();
        Some(PlaneView8 {
            data: self.frame.inner.data.as_ref().unwrap().data[2]
                .slice::<BitDepth8, _>(..)
                .into(),
            stride: self.frame.inner.stride[1] as usize,
            width: w,
            height: h,
        })
    }
    
    fn chroma_dimensions(&self) -> (usize, usize) {
        let w = self.frame.width() as usize;
        let h = self.frame.height() as usize;
        match self.frame.pixel_layout() {
            PixelLayout::I420 => (w / 2, h / 2),
            PixelLayout::I422 => (w / 2, h),
            PixelLayout::I444 => (w, h),
            PixelLayout::I400 => (0, 0),
        }
    }
}

/// 10/12-bit pixel plane access
pub struct Planes16<'a> {
    frame: &'a Frame,
}

impl<'a> Planes16<'a> {
    /// Y (luma) plane as a 2D strided view
    pub fn y(&self) -> PlaneView16<'a> {
        PlaneView16 {
            data: self.frame.inner.data.as_ref().unwrap().data[0]
                .slice::<BitDepth16, _>(..)
                .into(),
            stride: self.frame.inner.stride[0] as usize,
            width: self.frame.width() as usize,
            height: self.frame.height() as usize,
        }
    }
    
    /// U (chroma) plane, if present
    pub fn u(&self) -> Option<PlaneView16<'a>> {
        if self.frame.pixel_layout() == PixelLayout::I400 {
            return None;
        }
        let (w, h) = self.chroma_dimensions();
        Some(PlaneView16 {
            data: self.frame.inner.data.as_ref().unwrap().data[1]
                .slice::<BitDepth16, _>(..)
                .into(),
            stride: self.frame.inner.stride[1] as usize,
            width: w,
            height: h,
        })
    }
    
    /// V (chroma) plane, if present
    pub fn v(&self) -> Option<PlaneView16<'a>> {
        if self.frame.pixel_layout() == PixelLayout::I400 {
            return None;
        }
        let (w, h) = self.chroma_dimensions();
        Some(PlaneView16 {
            data: self.frame.inner.data.as_ref().unwrap().data[2]
                .slice::<BitDepth16, _>(..)
                .into(),
            stride: self.frame.inner.stride[1] as usize,
            width: w,
            height: h,
        })
    }
    
    fn chroma_dimensions(&self) -> (usize, usize) {
        // Same logic as Planes8
    }
}

/// Zero-copy view of an 8-bit plane
pub struct PlaneView8<'a> {
    data: &'a [u8],  // From DisjointImmutGuard
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> PlaneView8<'a> {
    /// Get a row by index (0-based)
    pub fn row(&self, y: usize) -> &[u8] {
        assert!(y < self.height, "row index out of bounds");
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }
    
    /// Get a single pixel value
    pub fn pixel(&self, x: usize, y: usize) -> u8 {
        assert!(x < self.width && y < self.height, "pixel coordinates out of bounds");
        self.data[y * self.stride + x]
    }
    
    /// Iterate over rows
    pub fn rows(&self) -> impl Iterator<Item = &[u8]> + 'a {
        (0..self.height).map(move |y| self.row(y))
    }
    
    /// Raw slice (includes padding, use stride for 2D indexing)
    pub fn as_slice(&self) -> &[u8] {
        self.data
    }
    
    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn stride(&self) -> usize { self.stride }
}

/// Zero-copy view of a 10/12-bit plane
pub struct PlaneView16<'a> {
    data: &'a [u16],  // From DisjointImmutGuard
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> PlaneView16<'a> {
    /// Get a row by index (0-based)
    pub fn row(&self, y: usize) -> &[u16] {
        assert!(y < self.height, "row index out of bounds");
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }
    
    /// Get a single pixel value
    pub fn pixel(&self, x: usize, y: usize) -> u16 {
        assert!(x < self.width && y < self.height, "pixel coordinates out of bounds");
        self.data[y * self.stride + x]
    }
    
    /// Iterate over rows
    pub fn rows(&self) -> impl Iterator<Item = &[u16]> + 'a {
        (0..self.height).map(move |y| self.row(y))
    }
    
    /// Raw slice (includes padding, use stride for 2D indexing)
    pub fn as_slice(&self) -> &[u16] {
        self.data
    }
    
    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn stride(&self) -> usize { self.stride }
}
```

### Color and HDR Metadata

```rust
/// Pixel layout (chroma subsampling)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelLayout {
    /// 4:0:0 (grayscale, no chroma)
    I400,
    /// 4:2:0 (most common, half-resolution chroma)
    I420,
    /// 4:2:2 (half horizontal chroma)
    I422,
    /// 4:4:4 (full resolution chroma)
    I444,
}

impl From<Rav1dPixelLayout> for PixelLayout {
    fn from(layout: Rav1dPixelLayout) -> Self {
        match layout {
            Rav1dPixelLayout::I400 => Self::I400,
            Rav1dPixelLayout::I420 => Self::I420,
            Rav1dPixelLayout::I422 => Self::I422,
            Rav1dPixelLayout::I444 => Self::I444,
        }
    }
}

/// Color information
#[derive(Clone, Copy, Debug)]
pub struct ColorInfo {
    pub primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,
    pub color_range: ColorRange,
}

/// Color primaries (CIE 1931 xy chromaticity coordinates)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ColorPrimaries {
    BT709 = 1,
    Unspecified = 2,
    BT470M = 4,
    BT470BG = 5,
    BT601 = 6,
    SMPTE240 = 7,
    Film = 8,
    BT2020 = 9,
    XYZ = 10,
    SMPTE431 = 11,
    SMPTE432 = 12,
    EBU3213 = 22,
}

impl From<Rav1dColorPrimaries> for ColorPrimaries {
    fn from(pri: Rav1dColorPrimaries) -> Self {
        match pri {
            Rav1dColorPrimaries::BT709 => Self::BT709,
            Rav1dColorPrimaries::BT2020 => Self::BT2020,
            // ... map all variants
            _ => Self::Unspecified,
        }
    }
}

/// Transfer characteristics (EOTF)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TransferCharacteristics {
    BT709 = 1,
    Unspecified = 2,
    BT470M = 4,
    BT470BG = 5,
    BT601 = 6,
    SMPTE240 = 7,
    Linear = 8,
    Log100 = 9,
    Log100Sqrt10 = 10,
    IEC61966 = 11,
    BT1361 = 12,
    SRGB = 13,
    BT2020_10bit = 14,
    BT2020_12bit = 15,
    SMPTE2084 = 16,  // PQ for HDR10
    SMPTE428 = 17,
    HLG = 18,        // Hybrid Log-Gamma for HDR
}

impl From<Rav1dTransferCharacteristics> for TransferCharacteristics {
    fn from(trc: Rav1dTransferCharacteristics) -> Self {
        // Map all variants
    }
}

/// Matrix coefficients (YUV to RGB conversion)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Identity = 0,
    BT709 = 1,
    Unspecified = 2,
    FCC = 4,
    BT470BG = 5,
    BT601 = 6,
    SMPTE240 = 7,
    YCgCo = 8,
    BT2020NCL = 9,
    BT2020CL = 10,
    SMPTE2085 = 11,
    ChromaDerivedNCL = 12,
    ChromaDerivedCL = 13,
    ICtCp = 14,
}

impl From<Rav1dMatrixCoefficients> for MatrixCoefficients {
    fn from(mtrx: Rav1dMatrixCoefficients) -> Self {
        // Map all variants
    }
}

/// Color range
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorRange {
    /// Limited/studio range (Y: 16-235, UV: 16-240)
    Limited,
    /// Full range (0-255 or 0-1023/4095)
    Full,
}

/// HDR content light level (SMPTE 2086 / CTA-861.3)
#[derive(Clone, Copy, Debug)]
pub struct ContentLightLevel {
    /// Maximum content light level in cd/m² (nits)
    pub max_content_light_level: u16,
    /// Maximum frame-average light level in cd/m² (nits)
    pub max_frame_average_light_level: u16,
}

/// HDR mastering display color volume (SMPTE 2086)
#[derive(Clone, Copy, Debug)]
pub struct MasteringDisplay {
    /// RGB primaries in 0.00002 increments [[R], [G], [B]]
    /// Each is [x, y] chromaticity coordinate
    pub primaries: [[u16; 2]; 3],
    /// White point [x, y] in 0.00002 increments
    pub white_point: [u16; 2],
    /// Maximum luminance in 0.0001 cd/m² increments
    pub max_luminance: u32,
    /// Minimum luminance in 0.0001 cd/m² increments
    pub min_luminance: u32,
}

impl MasteringDisplay {
    /// Get max luminance in nits (cd/m²)
    pub fn max_luminance_nits(&self) -> f64 {
        self.max_luminance as f64 / 10000.0
    }
    
    /// Get min luminance in nits (cd/m²)
    pub fn min_luminance_nits(&self) -> f64 {
        self.min_luminance as f64 / 10000.0
    }
    
    /// Get primary chromaticity as normalized floats [0.0, 1.0]
    pub fn primary_chromaticity(&self, index: usize) -> [f64; 2] {
        assert!(index < 3, "primary index must be 0-2");
        [
            self.primaries[index][0] as f64 / 50000.0,
            self.primaries[index][1] as f64 / 50000.0,
        ]
    }
    
    /// Get white point as normalized floats [0.0, 1.0]
    pub fn white_point_chromaticity(&self) -> [f64; 2] {
        [
            self.white_point[0] as f64 / 50000.0,
            self.white_point[1] as f64 / 50000.0,
        ]
    }
}
```

### Error Handling

```rust
/// Decoder errors
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid settings: {0}")]
    InvalidSettings(&'static str),
    
    #[error("decoder initialization failed")]
    InitFailed,
    
    #[error("decode error: {0}")]
    DecodeFailed(i32),  // errno-style error code
    
    #[error("out of memory")]
    OutOfMemory,
    
    #[error("invalid data")]
    InvalidData,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
```

## Usage Examples

### Basic Decoding

```rust
use rav1d_safe::managed::{Decoder, Settings};

fn decode_av1(obu_data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut decoder = Decoder::new()?;
    
    if let Some(frame) = decoder.decode(obu_data)? {
        println!("Decoded {}x{} frame at {}-bit", 
                 frame.width(), frame.height(), frame.bit_depth());
        
        // Access pixels (zero-copy)
        match frame.planes() {
            Planes::Depth8(planes) => {
                let y_plane = planes.y();
                for (i, row) in y_plane.rows().enumerate() {
                    println!("Row {}: {} pixels", i, row.len());
                }
            }
            Planes::Depth16(planes) => {
                let y_plane = planes.y();
                let pixel = y_plane.pixel(0, 0);
                println!("Top-left pixel value: {}", pixel);
            }
        }
    }
    
    Ok(())
}
```

### HDR Metadata

```rust
fn check_hdr_metadata(frame: &Frame) {
    if let Some(cll) = frame.content_light() {
        println!("Max content light: {} nits", cll.max_content_light_level);
        println!("Max average light: {} nits", cll.max_frame_average_light_level);
    }
    
    if let Some(md) = frame.mastering_display() {
        println!("Peak luminance: {:.2} nits", md.max_luminance_nits());
        println!("Min luminance: {:.4} nits", md.min_luminance_nits());
        
        let [rx, ry] = md.primary_chromaticity(0);
        println!("Red primary: ({:.4}, {:.4})", rx, ry);
    }
    
    let color = frame.color_info();
    if color.transfer_characteristics == TransferCharacteristics::SMPTE2084 {
        println!("HDR10 (PQ) content detected");
    } else if color.transfer_characteristics == TransferCharacteristics::HLG {
        println!("HLG HDR content detected");
    }
}
```

### Multi-threaded Decoding

```rust
fn decode_with_threads(obu_data: &[u8]) -> Result<Frame, Error> {
    let settings = Settings {
        threads: 8,  // Use 8 threads
        apply_grain: true,
        ..Default::default()
    };
    
    let mut decoder = Decoder::with_settings(settings)?;
    
    decoder.decode(obu_data)?
        .ok_or(Error::InvalidData)
}
```

### Zero-Copy Image Proxy

```rust
/// Zero-copy wrapper for serving decoded frames
struct ImageProxy {
    frame: Frame,  // Cheap to clone (Arc inside)
}

impl ImageProxy {
    fn new(obu_data: &[u8]) -> Result<Self, Error> {
        let mut decoder = Decoder::new()?;
        let frame = decoder.decode(obu_data)?
            .ok_or(Error::InvalidData)?;
        Ok(Self { frame })
    }
    
    /// Get Y plane for serving (zero-copy)
    fn y_plane(&self) -> &[u8] {
        match self.frame.planes() {
            Planes::Depth8(planes) => planes.y().as_slice(),
            Planes::Depth16(_) => panic!("16-bit not supported in this proxy"),
        }
    }
    
    /// Clone is cheap (Arc inside Frame)
    fn clone_frame(&self) -> Frame {
        self.frame.clone()
    }
}
```

## Implementation Plan

### Phase 1: Core API Module
- [ ] Create `src/managed.rs` module
- [ ] Implement `Decoder` wrapper (safe wrapper around `rav1d_open`, `send_data`, `get_picture`)
- [ ] Implement `Settings` with conversions to/from `Rav1dSettings`
- [ ] Implement `Error` type with conversions from `Rav1dError`
- [ ] Add `pub mod managed` to `lib.rs` with `#![deny(unsafe_code)]` attribute

### Phase 2: Frame and Metadata Types
- [ ] Implement `Frame` wrapper around `Arc<Rav1dPicture>`
- [ ] Implement `ColorInfo`, `ContentLightLevel`, `MasteringDisplay`
- [ ] Implement `PixelLayout`, `ColorPrimaries`, `TransferCharacteristics`, `MatrixCoefficients`
- [ ] Add conversion traits from internal `Rav1d*` types

### Phase 3: Zero-Copy Pixel Access
- [ ] Implement `Planes`, `Planes8`, `Planes16` enums
- [ ] Implement `PlaneView8` and `PlaneView16` with safe slice conversion from `DisjointImmutGuard`
- [ ] Add row iterator, pixel access methods
- [ ] Ensure all pixel access is bounds-checked and lifetime-safe

### Phase 4: Documentation and Tests
- [ ] Add comprehensive rustdoc with examples
- [ ] Add unit tests for Settings conversions
- [ ] Add integration tests decoding sample files (8/10/12-bit, HDR)
- [ ] Add safety documentation explaining zero-copy guarantees
- [ ] Add README section on managed API

### Phase 5: Advanced Features (Optional)
- [ ] Add `SendDataBuilder` for setting timestamp/duration/offset
- [ ] Add streaming API for multi-frame sequences
- [ ] Add `FramePool` for reusing allocations
- [ ] Add `DecoderBuilder` for more ergonomic construction

## Safety Guarantees

1. **No Unsafe in Public API** - All `unsafe` is internal to the wrapper module
2. **Lifetime Safety** - `PlaneView` borrows from `Frame`, which holds `Arc` to data
3. **Thread Safety** - `Frame` is `Send + Sync` because it holds `Arc`
4. **Bounds Checking** - All pixel access is bounds-checked
5. **Memory Safety** - `Arc` ensures data lives as long as any `Frame` exists
6. **ABI Safety** - Internal FFI calls isolated from public API

## Performance Characteristics

1. **Zero-Copy Decode** - `Frame` holds `Arc`, no pixel copies
2. **Cheap Cloning** - `Frame::clone()` only clones the `Arc`
3. **Lazy Access** - Planes accessed on-demand, no upfront conversion
4. **Inline Small Methods** - `width()`, `height()`, etc. inline to const reads
5. **Iterator Efficiency** - Row iterators compile to tight loops

## Open Questions

1. **Should we expose mutable access?** - Current design is read-only. For film grain application or custom filters, we might want `PlaneViewMut`. This requires more careful lifetime management.

2. **Builder pattern for Settings?** - Current design uses struct literal. Could add `SettingsBuilder` for more ergonomic construction:
   ```rust
   let settings = Settings::builder()
       .threads(8)
       .apply_grain(false)
       .build()?;
   ```

3. **Streaming API design?** - Current API is single-frame focused. For video sequences, we might want:
   ```rust
   let mut stream = decoder.stream();
   for frame in stream.decode_iter(data_source)? {
       // Process frame
   }
   ```

4. **Custom allocators?** - Current design uses default allocator. For performance, might want to expose `Rav1dAllocator` in managed API.

5. **Conversion helpers?** - Should we provide built-in YUV→RGB conversion, or leave that to external crates (yuv, etc.)?

## Rationale

**Why not just make internal Rav1d* types public?**
- Internal types have complex lifetimes and raw pointers
- They expose implementation details (Arc wrapper types, etc.)
- Managed API provides a clean, safe boundary

**Why use DisjointImmutGuard internally?**
- Already provides safe, bounds-checked slice access
- Designed for the exact use case (multiple references to different planes)
- Converting to `&[T]` is zero-cost

**Why separate Depth8/Depth16 types?**
- Type safety - prevent mixing 8-bit and 16-bit access
- Better ergonomics - users don't need to cast/reinterpret
- Matches common usage patterns (AVIF images know bit depth upfront)

**Why Arc<Rav1dPicture> instead of Box?**
- Allows cheap cloning for image proxies
- Multiple consumers can hold references to same decoded frame
- Matches internal rav1d-safe reference counting

## Future Considerations

- **Async Support** - Add `async fn decode_async()` using `tokio::task::spawn_blocking`
- **WASM Target** - Ensure API works in wasm32-unknown-unknown (no threads)
- **Embedded Support** - Consider `no_std` variant with `alloc` only
- **Benches** - Compare performance of managed API vs direct FFI
