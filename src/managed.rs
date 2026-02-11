//! 100% Safe Rust API for rav1d-safe decoder
//!
//! This module provides a fully safe, zero-copy API for decoding AV1 video.
//! It wraps the internal `rav1d` decoder with type-safe, lifetime-safe abstractions.
//!
//! # Features
//!
//! - **100% Safe Rust** - No `unsafe` code in this module
//! - **Zero-Copy** - Direct access to decoded pixel data without copying
//! - **Type Safety** - Enums and strong types instead of raw integers
//! - **HDR Support** - Full access to HDR10, HLG metadata
//! - **Multi-threaded** - Configurable thread pool for parallel decoding
//!
//! # Example
//!
//! ```no_run
//! use rav1d_safe::src::managed::{Decoder, Settings, Planes};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut decoder = Decoder::new()?;
//! let obu_data = b"..."; // AV1 OBU bitstream data
//!
//! if let Some(frame) = decoder.decode(obu_data)? {
//!     println!("Decoded {}x{} frame at {}-bit",
//!              frame.width(), frame.height(), frame.bit_depth());
//!
//!     // Zero-copy access to pixel data
//!     match frame.planes() {
//!         Planes::Depth8(planes) => {
//!             let y_plane = planes.y();
//!             for row in y_plane.rows() {
//!                 // Process 8-bit row data
//!             }
//!         }
//!         Planes::Depth16(planes) => {
//!             let y_plane = planes.y();
//!             let pixel = y_plane.pixel(0, 0);
//!             println!("Top-left pixel: {}", pixel);
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use crate::include::common::bitdepth::{BitDepth16, BitDepth8};
use crate::include::dav1d::data::Rav1dData;
use crate::include::dav1d::dav1d::{Rav1dDecodeFrameType, Rav1dInloopFilterType, Rav1dSettings};
use crate::include::dav1d::headers::{
    Rav1dColorPrimaries, Rav1dMatrixCoefficients, Rav1dPixelLayout, Rav1dTransferCharacteristics,
};
use crate::include::dav1d::picture::{Rav1dPicture, Rav1dPictureDataComponentInner};
use crate::src::c_arc::CArc;
use crate::src::disjoint_mut::DisjointImmutGuard;
use crate::src::error::Rav1dError;
use crate::src::internal::Rav1dContext;
use std::ops::Deref;
use std::sync::Arc;

/// Decoder errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidSettings(&'static str),
    InitFailed,
    OutOfMemory,
    InvalidData,
    NeedMoreData,
    Other(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSettings(msg) => write!(f, "invalid settings: {}", msg),
            Self::InitFailed => write!(f, "decoder initialization failed"),
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::InvalidData => write!(f, "invalid data"),
            Self::NeedMoreData => write!(f, "need more data"),
            Self::Other(msg) => write!(f, "decode error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<Rav1dError> for Error {
    fn from(err: Rav1dError) -> Self {
        match err {
            Rav1dError::EAGAIN => Self::NeedMoreData,
            Rav1dError::ENOMEM => Self::OutOfMemory,
            Rav1dError::EINVAL => Self::InvalidData,
            Rav1dError::EGeneric => Self::Other("generic error".to_string()),
            _ => Self::Other(format!("{:?}", err)),
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Decoder configuration settings
#[derive(Clone, Debug)]
pub struct Settings {
    /// Number of threads for decoding
    ///
    /// * `0` = auto-detect (enables frame threading for better performance but asynchronous behavior)
    /// * `1` = single-threaded (default, simpler synchronous behavior)
    /// * `2+` = multi-threaded with frame threading
    ///
    /// With frame threading enabled (threads >= 2 or threads == 0), `decode()` may return `None`
    /// even when complete frame data is provided, as frames are processed asynchronously.
    /// Call `decode()` or `flush()` multiple times to drain buffered frames.
    pub threads: u32,

    /// Apply film grain synthesis during decoding
    pub apply_grain: bool,

    /// Maximum frame size in pixels (0 = unlimited)
    pub frame_size_limit: u32,

    /// Decode all layers or just the selected operating point
    pub all_layers: bool,

    /// Operating point to decode (0-31)
    pub operating_point: u8,

    /// Output invisible frames (frames not meant for display)
    pub output_invisible_frames: bool,

    /// Inloop filters to apply during decoding
    pub inloop_filters: InloopFilters,

    /// Which frame types to decode
    pub decode_frame_type: DecodeFrameType,

    /// Enforce strict standard compliance
    pub strict_std_compliance: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            // Use single-threaded decoding by default for simpler, deterministic behavior.
            // With threads=0 (auto), frame threading is enabled which causes decode() to
            // return None even when a complete frame has been provided, because frame threads
            // need time to process. Users can set threads=0 for better performance but must
            // handle the asynchronous behavior by calling decode() or flush() multiple times.
            threads: 1,
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

impl From<Settings> for Rav1dSettings {
    fn from(settings: Settings) -> Self {
        Self {
            n_threads: settings.threads as i32,
            max_frame_delay: 0,
            apply_grain: settings.apply_grain,
            operating_point: settings.operating_point,
            all_layers: settings.all_layers,
            frame_size_limit: settings.frame_size_limit,
            allocator: Default::default(),
            logger: None,
            strict_std_compliance: settings.strict_std_compliance,
            output_invisible_frames: settings.output_invisible_frames,
            inloop_filters: settings.inloop_filters.into(),
            decode_frame_type: settings.decode_frame_type.into(),
        }
    }
}

/// Inloop filter flags
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InloopFilters {
    bits: u8,
}

impl InloopFilters {
    pub const DEBLOCK: Self = Self {
        bits: Rav1dInloopFilterType::DEBLOCK.bits(),
    };
    pub const CDEF: Self = Self {
        bits: Rav1dInloopFilterType::CDEF.bits(),
    };
    pub const RESTORATION: Self = Self {
        bits: Rav1dInloopFilterType::RESTORATION.bits(),
    };

    /// Enable all inloop filters
    pub const fn all() -> Self {
        Self {
            bits: Rav1dInloopFilterType::DEBLOCK.bits()
                | Rav1dInloopFilterType::CDEF.bits()
                | Rav1dInloopFilterType::RESTORATION.bits(),
        }
    }

    /// Disable all inloop filters
    pub const fn none() -> Self {
        Self { bits: 0 }
    }

    pub const fn contains(&self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }
}

impl From<InloopFilters> for Rav1dInloopFilterType {
    fn from(filters: InloopFilters) -> Self {
        Self::from_bits_retain(filters.bits)
    }
}

/// Which frame types to decode
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

impl From<DecodeFrameType> for Rav1dDecodeFrameType {
    fn from(ft: DecodeFrameType) -> Self {
        match ft {
            DecodeFrameType::All => Self::All,
            DecodeFrameType::Reference => Self::Reference,
            DecodeFrameType::Intra => Self::Intra,
            DecodeFrameType::Key => Self::Key,
        }
    }
}

/// Safe AV1 decoder instance
///
/// This is the main entry point for decoding AV1 video. It wraps the internal
/// `rav1d` decoder with a safe, type-safe interface.
///
/// # Example
///
/// ```no_run
/// use rav1d_safe::src::managed::Decoder;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut decoder = Decoder::new()?;
/// let obu_data = b"...";
///
/// if let Some(frame) = decoder.decode(obu_data)? {
///     println!("Decoded {}x{}", frame.width(), frame.height());
/// }
/// # Ok(())
/// # }
/// ```
pub struct Decoder {
    ctx: Arc<Rav1dContext>,
    worker_handles: Vec<std::thread::JoinHandle<()>>,
}

impl Decoder {
    /// Create a new decoder with default settings
    pub fn new() -> Result<Self> {
        Self::with_settings(Settings::default())
    }

    /// Create a decoder with custom settings
    pub fn with_settings(settings: Settings) -> Result<Self> {
        let rav1d_settings: Rav1dSettings = settings.into();
        let (ctx, worker_handles) =
            crate::src::lib::rav1d_open(&rav1d_settings).map_err(|_| Error::InitFailed)?;
        Ok(Self {
            ctx,
            worker_handles,
        })
    }

    /// Decode AV1 OBU data from a byte slice
    ///
    /// Returns `Ok(None)` if more data is needed (the decoder is waiting for more input).
    /// Returns `Ok(Some(frame))` when a frame is successfully decoded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rav1d_safe::src::managed::Decoder;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut decoder = Decoder::new()?;
    /// let data = b"..."; // AV1 OBU data
    ///
    /// match decoder.decode(data)? {
    ///     Some(frame) => {
    ///         println!("Got frame: {}x{}", frame.width(), frame.height());
    ///     }
    ///     None => {
    ///         println!("Need more data");
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn decode(&mut self, data: &[u8]) -> Result<Option<Frame>> {
        // Create Rav1dData from slice by copying to a CArc-owned buffer
        let mut rav1d_data = if !data.is_empty() {
            // Allocate and copy in one go: convert to Vec, then Box, then CBox, then CArc
            let owned = data.to_vec().into_boxed_slice();
            let cbox = crate::src::c_box::CBox::from_box(owned);
            let carc = CArc::wrap(cbox).map_err(|_| Error::OutOfMemory)?;
            Rav1dData {
                data: Some(carc),
                m: Default::default(),
            }
        } else {
            Rav1dData {
                data: None,
                m: Default::default(),
            }
        };

        // Send data to decoder
        crate::src::lib::rav1d_send_data(&self.ctx, &mut rav1d_data)?;

        // Try to get a picture
        let mut pic = Rav1dPicture::default();
        match crate::src::lib::rav1d_get_picture(&self.ctx, &mut pic) {
            Ok(()) => Ok(Some(Frame { inner: pic })),
            Err(Rav1dError::EAGAIN) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Try to get a decoded frame without sending new data.
    ///
    /// After calling [`decode()`](Self::decode), the decoder may have buffered
    /// additional frames (e.g. from a raw OBU stream containing multiple temporal
    /// units). Call this in a loop until it returns `Ok(None)` to drain them.
    pub fn get_frame(&mut self) -> Result<Option<Frame>> {
        let mut pic = Rav1dPicture::default();
        match crate::src::lib::rav1d_get_picture(&self.ctx, &mut pic) {
            Ok(()) => Ok(Some(Frame { inner: pic })),
            Err(Rav1dError::EAGAIN) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Flush the decoder and return all remaining frames
    ///
    /// This should be called after all input data has been fed to the decoder
    /// to retrieve any buffered frames.
    pub fn flush(&mut self) -> Result<Vec<Frame>> {
        crate::src::lib::rav1d_flush(&self.ctx);

        let mut frames = Vec::new();
        loop {
            let mut pic = Rav1dPicture::default();
            match crate::src::lib::rav1d_get_picture(&self.ctx, &mut pic) {
                Ok(()) => frames.push(Frame { inner: pic }),
                Err(Rav1dError::EAGAIN) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(frames)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // Signal worker threads to exit
        self.ctx.tell_worker_threads_to_die();

        // Join all worker threads synchronously
        // This is safe because:
        // 1. We're on the main thread (Decoder is not Send)
        // 2. Workers have been signaled to exit via tell_worker_threads_to_die
        // 3. We own the JoinHandles, not the workers themselves
        for handle in self.worker_handles.drain(..) {
            match handle.join() {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("rav1d worker thread panicked during shutdown: {:?}", e);
                }
            }
        }

        // Now drop the Arc<Rav1dContext>
        // Workers have exited, so we're likely the last Arc holder
    }
}

/// A decoded AV1 frame with zero-copy access to pixel data
///
/// Frames are cheap to clone (they use `Arc` internally).
/// Multiple `Frame` instances can safely reference the same decoded data.
#[derive(Clone)]
pub struct Frame {
    inner: Rav1dPicture,
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
        self.inner.p.layout.into()
    }

    /// Access pixel data according to bit depth
    ///
    /// Returns an enum that dispatches to either 8-bit or 16-bit plane access.
    pub fn planes(&self) -> Planes<'_> {
        match self.bit_depth() {
            8 => Planes::Depth8(Planes8 { frame: self }),
            10 | 12 => Planes::Depth16(Planes16 { frame: self }),
            _ => unreachable!("invalid bit depth: {}", self.bit_depth()),
        }
    }

    /// Color metadata
    pub fn color_info(&self) -> ColorInfo {
        let seq_hdr = self.inner.seq_hdr.as_ref().expect("missing seq_hdr");
        ColorInfo {
            primaries: seq_hdr.rav1d.pri.into(),
            transfer_characteristics: seq_hdr.rav1d.trc.into(),
            matrix_coefficients: seq_hdr.rav1d.mtrx.into(),
            color_range: if seq_hdr.rav1d.color_range != 0 {
                ColorRange::Full
            } else {
                ColorRange::Limited
            },
        }
    }

    /// HDR content light level metadata, if present
    pub fn content_light(&self) -> Option<ContentLightLevel> {
        self.inner
            .content_light
            .as_ref()
            .map(|arc| ContentLightLevel {
                max_content_light_level: arc.max_content_light_level,
                max_frame_average_light_level: arc.max_frame_average_light_level,
            })
    }

    /// HDR mastering display metadata, if present
    pub fn mastering_display(&self) -> Option<MasteringDisplay> {
        self.inner
            .mastering_display
            .as_ref()
            .map(|arc| MasteringDisplay {
                primaries: arc.primaries,
                white_point: arc.white_point,
                max_luminance: arc.max_luminance,
                min_luminance: arc.min_luminance,
            })
    }

    /// Timestamp from input data (arbitrary units)
    pub fn timestamp(&self) -> i64 {
        self.inner.m.timestamp
    }

    /// Duration from input data (arbitrary units)
    pub fn duration(&self) -> i64 {
        self.inner.m.duration
    }
}

/// Pixel layout (chroma subsampling)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelLayout {
    /// 4:0:0 (grayscale, no chroma)
    I400,
    /// 4:2:0 (most common, half-resolution chroma horizontally and vertically)
    I420,
    /// 4:2:2 (half horizontal chroma resolution)
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

/// Zero-copy access to pixel planes
///
/// Dispatches on bit depth for type safety.
pub enum Planes<'a> {
    Depth8(Planes8<'a>),
    Depth16(Planes16<'a>),
}

/// 8-bit pixel plane accessor
pub struct Planes8<'a> {
    frame: &'a Frame,
}

impl<'a> Planes8<'a> {
    /// Y (luma) plane as a 2D strided view
    pub fn y(&self) -> PlaneView8<'a> {
        let data = self
            .frame
            .inner
            .data
            .as_ref()
            .expect("missing picture data");
        let guard = data.data[0].slice::<BitDepth8, _>(..);

        let stride = self.frame.inner.stride[0] as usize;
        let buffer_height = if stride > 0 { guard.len() / stride } else { 0 };
        // Use frame's reported height, capped at buffer capacity
        let height = (self.frame.height() as usize).min(buffer_height);

        PlaneView8 {
            guard,
            stride,
            width: self.frame.width() as usize,
            height,
        }
    }

    /// U (chroma) plane, if present (None for grayscale)
    pub fn u(&self) -> Option<PlaneView8<'a>> {
        if self.frame.pixel_layout() == PixelLayout::I400 {
            return None;
        }

        let (w, h) = self.chroma_dimensions();
        let data = self
            .frame
            .inner
            .data
            .as_ref()
            .expect("missing picture data");
        let guard = data.data[1].slice::<BitDepth8, _>(..);

        let stride = self.frame.inner.stride[1] as usize;
        let buffer_height = if stride > 0 { guard.len() / stride } else { 0 };
        let height = h.min(buffer_height);

        Some(PlaneView8 {
            guard,
            stride,
            width: w,
            height,
        })
    }

    /// V (chroma) plane, if present (None for grayscale)
    pub fn v(&self) -> Option<PlaneView8<'a>> {
        if self.frame.pixel_layout() == PixelLayout::I400 {
            return None;
        }

        let (w, h) = self.chroma_dimensions();
        let data = self
            .frame
            .inner
            .data
            .as_ref()
            .expect("missing picture data");
        let guard = data.data[2].slice::<BitDepth8, _>(..);

        let stride = self.frame.inner.stride[1] as usize;
        let buffer_height = if stride > 0 { guard.len() / stride } else { 0 };
        let height = h.min(buffer_height);

        Some(PlaneView8 {
            guard,
            stride,
            width: w,
            height,
        })
    }

    fn chroma_dimensions(&self) -> (usize, usize) {
        let w = self.frame.width() as usize;
        let h = self.frame.height() as usize;
        match self.frame.pixel_layout() {
            PixelLayout::I420 => ((w + 1) / 2, (h + 1) / 2),
            PixelLayout::I422 => ((w + 1) / 2, h),
            PixelLayout::I444 => (w, h),
            PixelLayout::I400 => (0, 0),
        }
    }
}

/// 10/12-bit pixel plane accessor
pub struct Planes16<'a> {
    frame: &'a Frame,
}

impl<'a> Planes16<'a> {
    /// Y (luma) plane as a 2D strided view
    pub fn y(&self) -> PlaneView16<'a> {
        let data = self
            .frame
            .inner
            .data
            .as_ref()
            .expect("missing picture data");
        let guard = data.data[0].slice::<BitDepth16, _>(..);

        // stride[0] is in bytes; divide by 2 for u16 element stride
        let stride = self.frame.inner.stride[0] as usize / 2;
        let buffer_height = if stride > 0 { guard.len() / stride } else { 0 };
        let height = (self.frame.height() as usize).min(buffer_height);

        PlaneView16 {
            guard,
            stride,
            width: self.frame.width() as usize,
            height,
        }
    }

    /// U (chroma) plane, if present
    pub fn u(&self) -> Option<PlaneView16<'a>> {
        if self.frame.pixel_layout() == PixelLayout::I400 {
            return None;
        }

        let (w, h) = self.chroma_dimensions();
        let data = self
            .frame
            .inner
            .data
            .as_ref()
            .expect("missing picture data");
        let guard = data.data[1].slice::<BitDepth16, _>(..);

        // stride[1] is in bytes; divide by 2 for u16 element stride
        let stride = self.frame.inner.stride[1] as usize / 2;
        let buffer_height = if stride > 0 { guard.len() / stride } else { 0 };
        let height = h.min(buffer_height);

        Some(PlaneView16 {
            guard,
            stride,
            width: w,
            height,
        })
    }

    /// V (chroma) plane, if present
    pub fn v(&self) -> Option<PlaneView16<'a>> {
        if self.frame.pixel_layout() == PixelLayout::I400 {
            return None;
        }

        let (w, h) = self.chroma_dimensions();
        let data = self
            .frame
            .inner
            .data
            .as_ref()
            .expect("missing picture data");
        let guard = data.data[2].slice::<BitDepth16, _>(..);

        // stride[1] is in bytes; divide by 2 for u16 element stride
        let stride = self.frame.inner.stride[1] as usize / 2;
        let buffer_height = if stride > 0 { guard.len() / stride } else { 0 };
        let height = h.min(buffer_height);

        Some(PlaneView16 {
            guard,
            stride,
            width: w,
            height,
        })
    }

    fn chroma_dimensions(&self) -> (usize, usize) {
        let w = self.frame.width() as usize;
        let h = self.frame.height() as usize;
        match self.frame.pixel_layout() {
            PixelLayout::I420 => ((w + 1) / 2, (h + 1) / 2),
            PixelLayout::I422 => ((w + 1) / 2, h),
            PixelLayout::I444 => (w, h),
            PixelLayout::I400 => (0, 0),
        }
    }
}

/// Zero-copy view of an 8-bit plane
///
/// Provides row-based and pixel-based access to decoded data.
pub struct PlaneView8<'a> {
    guard: DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [u8]>,
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> PlaneView8<'a> {
    /// Get a row by index (0-based)
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn row(&self, y: usize) -> &[u8] {
        assert!(
            y < self.height,
            "row index {} out of bounds (height: {})",
            y,
            self.height
        );
        let start = y * self.stride;
        &self.guard[start..start + self.width]
    }

    /// Get a single pixel value
    ///
    /// # Panics
    ///
    /// Panics if coordinates are out of bounds.
    pub fn pixel(&self, x: usize, y: usize) -> u8 {
        assert!(
            x < self.width && y < self.height,
            "pixel coordinates ({}, {}) out of bounds ({}x{})",
            x,
            y,
            self.width,
            self.height
        );
        self.guard[y * self.stride + x]
    }

    /// Iterate over rows
    pub fn rows(&'a self) -> impl Iterator<Item = &'a [u8]> + 'a {
        (0..self.height).map(move |y| self.row(y))
    }

    /// Raw slice (includes padding, use stride for 2D indexing)
    pub fn as_slice(&self) -> &[u8] {
        self.guard.deref()
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn stride(&self) -> usize {
        self.stride
    }
}

/// Zero-copy view of a 10/12-bit plane
pub struct PlaneView16<'a> {
    guard: DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [u16]>,
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> PlaneView16<'a> {
    /// Get a row by index (0-based)
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn row(&self, y: usize) -> &[u16] {
        assert!(
            y < self.height,
            "row index {} out of bounds (height: {})",
            y,
            self.height
        );
        let start = y * self.stride;
        &self.guard[start..start + self.width]
    }

    /// Get a single pixel value
    ///
    /// # Panics
    ///
    /// Panics if coordinates are out of bounds.
    pub fn pixel(&self, x: usize, y: usize) -> u16 {
        assert!(
            x < self.width && y < self.height,
            "pixel coordinates ({}, {}) out of bounds ({}x{})",
            x,
            y,
            self.width,
            self.height
        );
        self.guard[y * self.stride + x]
    }

    /// Iterate over rows
    pub fn rows(&'a self) -> impl Iterator<Item = &'a [u16]> + 'a {
        (0..self.height).map(move |y| self.row(y))
    }

    /// Raw slice (includes padding, use stride for 2D indexing)
    pub fn as_slice(&self) -> &[u16] {
        self.guard.deref()
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn stride(&self) -> usize {
        self.stride
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
    Unknown = 0,
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
        match pri.0 {
            1 => Self::BT709,
            4 => Self::BT470M,
            5 => Self::BT470BG,
            6 => Self::BT601,
            7 => Self::SMPTE240,
            8 => Self::Film,
            9 => Self::BT2020,
            10 => Self::XYZ,
            11 => Self::SMPTE431,
            12 => Self::SMPTE432,
            22 => Self::EBU3213,
            _ => Self::Unspecified,
        }
    }
}

/// Transfer characteristics (EOTF / gamma curve)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TransferCharacteristics {
    Reserved = 0,
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
    /// SMPTE 2084 - Perceptual Quantizer for HDR10
    SMPTE2084 = 16,
    SMPTE428 = 17,
    /// Hybrid Log-Gamma for HLG HDR
    HLG = 18,
}

impl From<Rav1dTransferCharacteristics> for TransferCharacteristics {
    fn from(trc: Rav1dTransferCharacteristics) -> Self {
        match trc.0 {
            1 => Self::BT709,
            4 => Self::BT470M,
            5 => Self::BT470BG,
            6 => Self::BT601,
            7 => Self::SMPTE240,
            8 => Self::Linear,
            9 => Self::Log100,
            10 => Self::Log100Sqrt10,
            11 => Self::IEC61966,
            12 => Self::BT1361,
            13 => Self::SRGB,
            14 => Self::BT2020_10bit,
            15 => Self::BT2020_12bit,
            16 => Self::SMPTE2084,
            17 => Self::SMPTE428,
            18 => Self::HLG,
            _ => Self::Unspecified,
        }
    }
}

/// Matrix coefficients (YUV to RGB conversion)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Identity = 0,
    BT709 = 1,
    Unspecified = 2,
    Reserved = 3,
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
        match mtrx.0 {
            0 => Self::Identity,
            1 => Self::BT709,
            4 => Self::FCC,
            5 => Self::BT470BG,
            6 => Self::BT601,
            7 => Self::SMPTE240,
            8 => Self::YCgCo,
            9 => Self::BT2020NCL,
            10 => Self::BT2020CL,
            11 => Self::SMPTE2085,
            12 => Self::ChromaDerivedNCL,
            13 => Self::ChromaDerivedCL,
            14 => Self::ICtCp,
            _ => Self::Unspecified,
        }
    }
}

/// Color range
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorRange {
    /// Limited/studio range (Y: 16-235, UV: 16-240 for 8-bit)
    Limited,
    /// Full range (0-255 for 8-bit, 0-1023 for 10-bit, 0-4095 for 12-bit)
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
    ///
    /// `index` should be 0 (red), 1 (green), or 2 (blue).
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

/// Returns a comma-delimited string of enabled compile-time feature flags.
///
/// Useful for runtime verification of which safety level and capabilities
/// were compiled in.
///
/// ```
/// let features = rav1d_safe::enabled_features();
/// assert!(features.contains("bitdepth_8"));
/// ```
pub fn enabled_features() -> String {
    let mut features = Vec::new();

    if cfg!(feature = "asm") {
        features.push("asm");
    }
    if cfg!(feature = "c-ffi") {
        features.push("c-ffi");
    }
    if cfg!(feature = "unchecked") {
        features.push("unchecked");
    }
    if cfg!(feature = "bitdepth_8") {
        features.push("bitdepth_8");
    }
    if cfg!(feature = "bitdepth_16") {
        features.push("bitdepth_16");
    }

    // Safety level summary
    if cfg!(feature = "asm") {
        features.push("safety:asm");
    } else if cfg!(feature = "c-ffi") {
        features.push("safety:c-ffi");
    } else if cfg!(feature = "unchecked") {
        features.push("safety:unchecked");
    } else {
        features.push("safety:forbid-unsafe");
    }

    features.join(", ")
}
