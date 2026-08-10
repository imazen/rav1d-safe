//! Reconstruction against an EXCLUSIVELY-OWNED, column-compact tile band.
//!
//! # What this is
//!
//! PR #474 gave each tile a private full-plane [`Rav1dPictureDataComponent`]
//! but kept the existing borrow primitives on it, so the tracker still ran on
//! every reconstruction access — just uncontended. PR #481 priced removing it
//! outright with an `tracker: None` probe: **1,970,944 registrations per frame
//! across 21 sites**, worth 0.8996 / 0.8955 / 0.9071 of #474's wall at 8bpc
//! t=2/4/8.
//!
//! This module earns those numbers SOUNDLY. The private band is a plain
//! `Vec<Chunk>` owned by [`Rav1dTaskContext`], the worker's own struct, and it
//! is handed to the reconstruction kernels as a `&mut [u8]`. Uniqueness is
//! therefore proved by **borrowck**, statically, with no run-time record, no
//! lock, and no `unsafe` — `#![forbid(unsafe_code)]` is untouched.
//!
//! # The two changes are one change
//!
//! #474's buffers took the picture's byte length AND stride so that every
//! frame-coordinate offset reconstruction computes indexes the same pixel — a
//! three-line seam, at the cost of `tile_columns × plane` residency
//! (+96.3 MB at 8bpc, +191.0 MB at 10bpc, measured). A kernel signature that
//! takes `(&mut [u8], base, stride)` already carries the origin and stride a
//! compact buffer needs, so the coordinate translation is free once the
//! signature change is being made anyway. Doing them in sequence pays the
//! translation cost twice.
//!
//! So the band here is **column-compact and one superblock row tall**, with its
//! OWN stride, one per WORKER rather than one per tile:
//! `n_workers × max_tile_width × sb_height × planes`.
//!
//! # Why one superblock row is enough
//!
//! Reconstruction never reads a picture row above the current superblock row:
//! `src/recon.rs:2243` and `:3138` gate on `t.b.y & f.sb_step - 1 == 0` and, at
//! a superblock-row boundary, source the top edge from `f.ipred_edge` instead
//! of the plane ([`crate::src::ipred_prepare::rav1d_prepare_intra_edges`] takes
//! it as `prefilter_toplevel_sb_edge`). Left, top-right and bottom-left reads
//! are all clamped to the TILE (`ts.tiling.col_start/col_end/row_end`), which
//! is what AV1 tile independence guarantees.
//!
//! This is load-bearing, so it is enforced rather than assumed: [`Band::at`]
//! translates plane coordinates into band coordinates by subtracting the band
//! origin, and an access above the band underflows into a slice index that is
//! out of bounds. The failure mode is a panic, never a silent wrong pixel.
//!
//! # Scope, stated before the wins
//!
//! Armed only for frames whose reconstruction is entirely INTRA (`KEY` /
//! `INTRA_ONLY`), because every writer of a recon plane has to move together —
//! intra prediction, palette, CfL and the inverse transform all write the same
//! pixels, so converting a subset leaves the tracker on the path. Inter frames,
//! `allow_intrabc`, frame threading (`n_fc > 1`), the `c-ffi` allocator,
//! negative strides and single-tile frames all decline to the shared picture
//! and behave exactly as today.

// The recon conversion is INTRA-ONLY today: inter and the entire filter chain
// still write the shared picture through the tracker, so a number of items here
// have no caller yet. The live subset also DIFFERS BY ARCHITECTURE -- x86's
// ipred/itx dispatchers use `DstBlock` and `as_mut_bytes`/`base`/`byte_stride`
// where aarch64's do not -- so an aarch64 dev box and x86 CI disagree about
// which items are dead, and per-item `cfg_attr`s would rot on the next
// dispatcher change.
//
// Module scope rather than per-item, deliberately: the alternative is a
// scattering of attributes that each look like a suppressed bug. REMOVE THIS
// when the inter and filter conversions land and every item has a caller.
#![allow(dead_code)]

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::PicOffset;
use crate::src::strided::Strided as _;
use std::ops::Deref;
use std::ops::DerefMut;

/// 64-byte allocation unit, so every band row starts
/// [`RAV1D_PICTURE_ALIGNMENT`]-aligned and `zerocopy`'s pixel reinterpretation
/// of a row can never fail on alignment.
///
/// [`RAV1D_PICTURE_ALIGNMENT`]: crate::include::dav1d::picture::RAV1D_PICTURE_ALIGNMENT
#[repr(C, align(64))]
#[derive(Clone, Copy, zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable)]
pub(crate) struct Chunk([u8; 64]);

const CHUNK: usize = 64;

/// The per-worker reconstruction band.
///
/// One allocation per plane, reused across superblock rows and across frames;
/// pages fault in once for a whole sequence.
pub(crate) struct ReconBand {
    /// `planes[pl]` — `rows[pl] * stride[pl]` bytes, column-compact.
    planes: [Vec<Chunk>; 3],
    /// Byte stride of each band (a multiple of [`CHUNK`]).
    stride: [usize; 3],
    /// Rows allocated per plane.
    rows: [usize; 3],
    /// Plane-coordinate origin of the band: `(row0, col0)` in PIXELS.
    origin: [(usize, usize); 3],
    /// Rows and columns of the band that hold live reconstruction, i.e. what
    /// [`stitch`] copies out.
    live: [(usize, usize); 3],
    /// Number of planes actually in use (1 for I400, else 3).
    n_planes: usize,
    /// Armed for the CURRENT superblock-row task?
    armed: bool,
}

impl Default for ReconBand {
    fn default() -> Self {
        Self {
            planes: [const { Vec::new() }; 3],
            stride: [0; 3],
            rows: [0; 3],
            origin: [(0, 0); 3],
            live: [(0, 0); 3],
            n_planes: 0,
            armed: false,
        }
    }
}

impl ReconBand {
    #[inline]
    pub(crate) fn armed(&self) -> bool {
        self.armed
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }

    /// Size (growing only) and arm the band for one `(tile, superblock row)`.
    ///
    /// `geom[pl]` is `(row0, col0, rows, cols, pixel_size)` in that plane's own
    /// pixel coordinates. Returns `false` — leaving the band disarmed, i.e. the
    /// shared picture in use — if any allocation fails.
    pub(crate) fn arm(&mut self, n_planes: usize, geom: &[(usize, usize, usize, usize, usize); 3]) {
        self.armed = false;
        self.n_planes = n_planes;
        for pl in 0..n_planes {
            let (row0, col0, rows, cols, pixel_size) = geom[pl];
            let want_stride = (cols * pixel_size).next_multiple_of(CHUNK);
            let want_chunks = rows * (want_stride / CHUNK);
            if self.planes[pl].len() < want_chunks {
                // `try_reserve` + zeroed extend: a decoder must not abort the
                // process on a failed allocation, so a failure declines to the
                // shared picture instead.
                let extra = want_chunks - self.planes[pl].len();
                if self.planes[pl].try_reserve(extra).is_err() {
                    return;
                }
                self.planes[pl].resize(want_chunks, Chunk([0; CHUNK]));
            }
            self.stride[pl] = want_stride;
            self.rows[pl] = rows;
            self.origin[pl] = (row0, col0);
            self.live[pl] = (rows, cols);
        }
        self.armed = true;
    }

    /// Clamp the live row count of every plane, for the final (partial)
    /// superblock row of a frame.
    pub(crate) fn set_live_rows(&mut self, live_rows: [usize; 3]) {
        for pl in 0..self.n_planes {
            self.live[pl].0 = live_rows[pl].min(self.rows[pl]);
        }
    }

    #[inline]
    pub(crate) fn n_planes(&self) -> usize {
        self.n_planes
    }

    #[inline]
    pub(crate) fn plane_geometry(&self, pl: usize) -> (usize, usize, usize, usize) {
        let (row0, col0) = self.origin[pl];
        let (rows, cols) = self.live[pl];
        (row0, col0, rows, cols)
    }

    /// Read one live row of plane `pl`, for [`stitch`].
    #[inline]
    pub(crate) fn row_bytes(&self, pl: usize, row: usize, len: usize) -> &[u8] {
        let stride = self.stride[pl];
        let all: &[u8] = zerocopy::IntoBytes::as_bytes(&self.planes[pl][..]);
        &all[row * stride..][..len]
    }

    /// An exclusive, strided view of plane `pl` positioned at PLANE pixel
    /// coordinates `(row, col)`.
    ///
    /// Panics (never corrupts) if `(row, col)` is outside the band: the
    /// subtraction underflows and the resulting index is out of bounds.
    #[inline]
    pub(crate) fn at<BD: BitDepth>(&mut self, pl: usize, row: usize, col: usize) -> Band<'_> {
        let (row0, col0) = self.origin[pl];
        let stride = self.stride[pl];
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        // `checked_sub`, not `-`: a row above the band must fail the SAME way in
        // every build profile. With plain subtraction, release wraps to a huge
        // index and the assert below catches it, but a debug-assertions build
        // (coverage, for one) panics earlier with "attempt to subtract with
        // overflow" -- same safety outcome, different message, and the
        // `#[should_panic(expected = ...)]` gate pins the message. Relying on
        // wrapping to produce an out-of-range value is indirect anyway; this
        // states the bound directly.
        let brow = row.checked_sub(row0).expect("recon band row out of range");
        let bcol = col.checked_sub(col0).expect("recon band col out of range");
        assert!(brow < self.rows[pl], "recon band row out of range");
        let bytes: &mut [u8] = zerocopy::IntoBytes::as_mut_bytes(&mut self.planes[pl][..]);
        Band {
            offset: (brow * stride) / pixel_size + bcol,
            bytes,
            stride: stride as isize,
        }
    }
}

/// A strided pixel region whose uniqueness is a static fact.
///
/// The `&mut [u8]` inside is the whole band; `offset` is a PIXEL index into it
/// and `stride` a BYTE stride, matching [`PicOffset`]'s conventions so the
/// kernels' arithmetic is unchanged.
pub(crate) struct Band<'a> {
    bytes: &'a mut [u8],
    offset: usize,
    stride: isize,
}

/// [`Band`], shared.
#[derive(Clone, Copy)]
pub(crate) struct BandRef<'a> {
    bytes: &'a [u8],
    offset: usize,
    stride: isize,
}

/// Where a reconstruction kernel writes: either the shared, borrow-TRACKED
/// picture, or a band this worker owns outright.
///
/// Kernels take `&mut ReconDst<'_>`, which Rust reborrows implicitly in
/// argument position, so passing it down a dispatch chain costs no syntax.
pub(crate) enum ReconDst<'a> {
    Pic(PicOffset<'a>),
    Own(Band<'a>),
}

/// [`ReconDst`], shared.
#[derive(Clone, Copy)]
pub(crate) enum ReconSrc<'a> {
    Pic(PicOffset<'a>),
    Own(BandRef<'a>),
}

/// A mutable pixel slice from either backing.
pub(crate) enum PxMut<'a, BD: BitDepth> {
    Pic(
        crate::src::disjoint_mut::DisjointMutGuard<
            'a,
            crate::include::dav1d::picture::Rav1dPictureDataComponentInner,
            [BD::Pixel],
        >,
    ),
    Own(&'a mut [BD::Pixel]),
}

impl<BD: BitDepth> Deref for PxMut<'_, BD> {
    type Target = [BD::Pixel];
    #[inline]
    fn deref(&self) -> &[BD::Pixel] {
        match self {
            Self::Pic(g) => g,
            Self::Own(s) => s,
        }
    }
}

impl<BD: BitDepth> DerefMut for PxMut<'_, BD> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [BD::Pixel] {
        match self {
            Self::Pic(g) => g,
            Self::Own(s) => s,
        }
    }
}

/// An immutable pixel slice from either backing.
pub(crate) enum Px<'a, BD: BitDepth> {
    Pic(
        crate::src::disjoint_mut::DisjointImmutGuard<
            'a,
            crate::include::dav1d::picture::Rav1dPictureDataComponentInner,
            [BD::Pixel],
        >,
    ),
    Own(&'a [BD::Pixel]),
}

impl<BD: BitDepth> Deref for Px<'_, BD> {
    type Target = [BD::Pixel];
    #[inline]
    fn deref(&self) -> &[BD::Pixel] {
        match self {
            Self::Pic(g) => g,
            Self::Own(s) => s,
        }
    }
}

/// The `(bytes, base, byte_stride)` triple the NEON/AVX transform kernels take.
///
/// On the picture this is [`WithOffset::block_mut`], which under tile threading
/// copies the block into a compact scratch and writes it back per row. On an
/// owned band it is the band itself, so there is no copy, no write-back, and no
/// registration.
///
/// [`WithOffset::block_mut`]: crate::src::with_offset::WithOffset::block_mut
pub(crate) enum DstBlock<'a, BD: BitDepth> {
    Pic(crate::include::dav1d::picture::BlockMut<'a, BD>),
    Own {
        bytes: &'a mut [u8],
        base: usize,
        stride: isize,
    },
}

impl<BD: BitDepth> DstBlock<'_, BD> {
    #[inline]
    pub(crate) fn as_mut_bytes(&mut self) -> &mut [u8] {
        match self {
            Self::Pic(b) => b.as_mut_bytes(),
            Self::Own { bytes, .. } => bytes,
        }
    }

    #[inline]
    pub(crate) fn base(&self) -> usize {
        match self {
            Self::Pic(b) => b.base(),
            Self::Own { base, .. } => *base,
        }
    }

    #[inline]
    pub(crate) fn byte_stride(&self) -> isize {
        match self {
            Self::Pic(b) => b.byte_stride(),
            Self::Own { stride, .. } => *stride,
        }
    }
}

#[inline]
fn px<BD: BitDepth>(bytes: &[u8], off: usize, len: usize) -> &[BD::Pixel] {
    let size = core::mem::size_of::<BD::Pixel>();
    let s = &bytes[off * size..][..len * size];
    zerocopy::FromBytes::ref_from_bytes(s).expect("band row pixel reinterpretation")
}

#[inline]
fn px_mut<BD: BitDepth>(bytes: &mut [u8], off: usize, len: usize) -> &mut [BD::Pixel] {
    let size = core::mem::size_of::<BD::Pixel>();
    let s = &mut bytes[off * size..][..len * size];
    zerocopy::FromBytes::mut_from_bytes(s).expect("band row pixel reinterpretation")
}

impl<'a> ReconDst<'a> {
    /// Byte stride between rows.
    #[inline]
    pub(crate) fn stride(&self) -> isize {
        match self {
            Self::Pic(p) => p.stride(),
            Self::Own(b) => b.stride,
        }
    }

    /// Pixel stride between rows.
    #[inline]
    pub(crate) fn pixel_stride<BD: BitDepth>(&self) -> isize {
        match self {
            Self::Pic(p) => p.pixel_stride::<BD>(),
            Self::Own(b) => b.stride / core::mem::size_of::<BD::Pixel>() as isize,
        }
    }

    /// The same region, origin moved by `delta` PIXELS. Reborrows, so the
    /// result cannot outlive `self` — which is exactly the exclusion proof.
    #[inline]
    pub(crate) fn at(&mut self, delta: isize) -> ReconDst<'_> {
        match self {
            Self::Pic(p) => ReconDst::Pic(*p + delta),
            Self::Own(b) => ReconDst::Own(Band {
                offset: b.offset.wrapping_add_signed(delta),
                bytes: b.bytes,
                stride: b.stride,
            }),
        }
    }

    /// A shared view of the same region.
    #[inline]
    pub(crate) fn as_src(&self) -> ReconSrc<'_> {
        match self {
            Self::Pic(p) => ReconSrc::Pic(*p),
            Self::Own(b) => ReconSrc::Own(BandRef {
                bytes: b.bytes,
                offset: b.offset,
                stride: b.stride,
            }),
        }
    }

    /// The tracked picture offset, if that is what this is. `None` on an owned
    /// band — used by the `asm` FFI paths, which take a raw picture pointer and
    /// therefore fall through to the Rust reference when the band is armed.
    #[inline]
    #[cfg_attr(not(feature = "asm"), allow(dead_code))]
    pub(crate) fn as_pic(&self) -> Option<PicOffset<'a>> {
        match self {
            Self::Pic(p) => Some(*p),
            Self::Own(_) => None,
        }
    }

    /// `len` pixels starting at the origin.
    #[inline]
    pub(crate) fn slice_mut<BD: BitDepth>(&mut self, len: usize) -> PxMut<'_, BD> {
        match self {
            Self::Pic(p) => PxMut::Pic(p.slice_mut::<BD>(len)),
            Self::Own(b) => PxMut::Own(px_mut::<BD>(b.bytes, b.offset, len)),
        }
    }

    /// Copy out a `w`x`h` pixel rectangle, row-major, skipping stride gaps.
    ///
    /// Only the `__simd_test` harness needs this: it snapshots the destination
    /// before a SIMD kernel, re-runs the scalar reference over the same block,
    /// and compares. `PicOffset::strided_slice` served that on the tracked
    /// path, but an owned band has no tracker and hands out no guard, so the
    /// two variants need one shape in common. Row-at-a-time rather than one
    /// span because the band's stride is its own tile width, not the picture's.
    #[cfg(feature = "__simd_test")]
    pub(crate) fn copy_out<BD: BitDepth>(&self, w: usize, h: usize) -> Vec<BD::Pixel> {
        let ps = self.pixel_stride::<BD>();
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            let row = self.as_src().at(ps * y as isize);
            out.extend_from_slice(&row.slice::<BD>(w));
        }
        out
    }

    /// Write back a `w`x`h` pixel rectangle produced by [`Self::copy_out`].
    #[cfg(feature = "__simd_test")]
    pub(crate) fn copy_in<BD: BitDepth>(&mut self, w: usize, h: usize, src: &[BD::Pixel]) {
        assert_eq!(src.len(), w * h, "copy_in expects exactly w*h pixels");
        let ps = self.pixel_stride::<BD>();
        for y in 0..h {
            let mut row = self.at(ps * y as isize);
            row.slice_mut::<BD>(w)
                .copy_from_slice(&src[y * w..(y + 1) * w]);
        }
    }

    /// `len` pixels starting at the origin, read-only.
    #[inline]
    pub(crate) fn slice<BD: BitDepth>(&self, len: usize) -> Px<'_, BD> {
        match self {
            Self::Pic(p) => Px::Pic(p.slice::<BD>(len)),
            Self::Own(b) => Px::Own(px::<BD>(b.bytes, b.offset, len)),
        }
    }

    /// The pixel at the origin.
    #[inline]
    pub(crate) fn get<BD: BitDepth>(&self) -> BD::Pixel {
        match self {
            Self::Pic(p) => *p.index::<BD>(),
            Self::Own(b) => px::<BD>(b.bytes, b.offset, 1)[0],
        }
    }

    /// Write the pixel at the origin.
    #[inline]
    pub(crate) fn set<BD: BitDepth>(&mut self, v: BD::Pixel) {
        match self {
            Self::Pic(p) => *p.index_mut::<BD>() = v,
            Self::Own(b) => px_mut::<BD>(b.bytes, b.offset, 1)[0] = v,
        }
    }

    /// Iterate `h` rows of `w` pixels, mutably.
    #[inline]
    pub(crate) fn for_rows_mut<BD: BitDepth, F: FnMut(usize, &mut [BD::Pixel])>(
        &mut self,
        w: usize,
        h: usize,
        mut f: F,
    ) {
        match self {
            Self::Pic(p) => p.for_rows_mut::<BD, F>(w, h, f),
            Self::Own(b) => {
                if w == 0 || h == 0 {
                    return;
                }
                let pxstride = (b.stride / core::mem::size_of::<BD::Pixel>() as isize) as usize;
                for row in 0..h {
                    f(row, px_mut::<BD>(b.bytes, b.offset + row * pxstride, w));
                }
            }
        }
    }

    /// Iterate `h` rows of `w` pixels, read-only.
    #[inline]
    pub(crate) fn for_rows<BD: BitDepth, F: FnMut(usize, &[BD::Pixel])>(
        &self,
        w: usize,
        h: usize,
        mut f: F,
    ) {
        match self {
            Self::Pic(p) => p.for_rows::<BD, F>(w, h, f),
            Self::Own(b) => {
                if w == 0 || h == 0 {
                    return;
                }
                let pxstride = (b.stride / core::mem::size_of::<BD::Pixel>() as isize) as usize;
                for row in 0..h {
                    f(row, px::<BD>(b.bytes, b.offset + row * pxstride, w));
                }
            }
        }
    }

    /// A `w × h` block as `(bytes, byte_offset, byte_stride)`, closure-shaped.
    ///
    /// The picture arm is [`with_pixel_guard_mut`]; the owned arm is the band
    /// itself, so there is no compact copy, no write-back and no registration.
    ///
    /// [`with_pixel_guard_mut`]: crate::include::dav1d::picture::with_pixel_guard_mut
    #[inline]
    pub(crate) fn with_block_mut<BD: BitDepth, R>(
        &mut self,
        w: usize,
        h: usize,
        f: impl FnOnce(&mut [u8], usize, isize) -> R,
    ) -> R {
        match self {
            Self::Pic(p) => {
                crate::include::dav1d::picture::with_pixel_guard_mut::<BD, R>(p, w, h, f)
            }
            Self::Own(b) => {
                let _ = (w, h);
                let off = b.offset * core::mem::size_of::<BD::Pixel>();
                let stride = b.stride;
                f(b.bytes, off, stride)
            }
        }
    }

    /// A `w × h` block as `(bytes, base, byte_stride)`.
    #[inline]
    pub(crate) fn block_mut<BD: BitDepth>(&mut self, w: usize, h: usize) -> DstBlock<'_, BD> {
        match self {
            Self::Pic(p) => DstBlock::Pic(p.block_mut::<BD>(w, h)),
            Self::Own(b) => {
                let _ = (w, h);
                // Slice FROM the origin with `base = 0`, matching what the
                // picture arm hands out (a compact buffer under tile threading;
                // `narrow_guard_mut`'s base is also 0 at a positive stride). A
                // kernel that ignores `base` — the wasm ones do — is then
                // correct on both backings.
                let off = b.offset * core::mem::size_of::<BD::Pixel>();
                DstBlock::Own {
                    bytes: &mut b.bytes[off..],
                    base: 0,
                    stride: b.stride,
                }
            }
        }
    }
}

impl<'a> ReconSrc<'a> {
    #[inline]
    pub(crate) fn stride(&self) -> isize {
        match self {
            Self::Pic(p) => p.stride(),
            Self::Own(b) => b.stride,
        }
    }

    #[inline]
    pub(crate) fn pixel_stride<BD: BitDepth>(&self) -> isize {
        match self {
            Self::Pic(p) => p.pixel_stride::<BD>(),
            Self::Own(b) => b.stride / core::mem::size_of::<BD::Pixel>() as isize,
        }
    }

    #[inline]
    pub(crate) fn at(&self, delta: isize) -> ReconSrc<'a> {
        match self {
            Self::Pic(p) => ReconSrc::Pic(*p + delta),
            Self::Own(b) => ReconSrc::Own(BandRef {
                bytes: b.bytes,
                offset: b.offset.wrapping_add_signed(delta),
                stride: b.stride,
            }),
        }
    }

    #[inline]
    #[cfg_attr(not(feature = "asm"), allow(dead_code))]
    pub(crate) fn as_pic(&self) -> Option<PicOffset<'a>> {
        match self {
            Self::Pic(p) => Some(*p),
            Self::Own(_) => None,
        }
    }

    #[inline]
    pub(crate) fn slice<BD: BitDepth>(&self, len: usize) -> Px<'a, BD> {
        match self {
            Self::Pic(p) => Px::Pic(p.slice::<BD>(len)),
            Self::Own(b) => Px::Own(px::<BD>(b.bytes, b.offset, len)),
        }
    }

    #[inline]
    pub(crate) fn get<BD: BitDepth>(&self) -> BD::Pixel {
        match self {
            Self::Pic(p) => *p.index::<BD>(),
            Self::Own(b) => px::<BD>(b.bytes, b.offset, 1)[0],
        }
    }

    /// Read a `w × h` block as `(bytes, byte_offset, byte_stride)`.
    ///
    /// The picture arm is [`with_pixel_guard_immut`], i.e. a compact per-row
    /// copy under tile threading; the owned arm is the band itself, zero-copy.
    ///
    /// [`with_pixel_guard_immut`]: crate::include::dav1d::picture::with_pixel_guard_immut
    #[inline]
    pub(crate) fn with_block<BD: BitDepth, R>(
        &self,
        w: usize,
        h: usize,
        f: impl FnOnce(&[u8], usize, isize) -> R,
    ) -> R {
        match self {
            Self::Pic(p) => {
                crate::include::dav1d::picture::with_pixel_guard_immut::<BD, R>(p, w, h, f)
            }
            Self::Own(b) => {
                let _ = (w, h);
                f(
                    b.bytes,
                    b.offset * core::mem::size_of::<BD::Pixel>(),
                    b.stride,
                )
            }
        }
    }

    #[inline]
    pub(crate) fn for_rows<BD: BitDepth, F: FnMut(usize, &[BD::Pixel])>(
        &self,
        w: usize,
        h: usize,
        mut f: F,
    ) {
        match self {
            Self::Pic(p) => p.for_rows::<BD, F>(w, h, f),
            Self::Own(b) => {
                if w == 0 || h == 0 {
                    return;
                }
                let pxstride = (b.stride / core::mem::size_of::<BD::Pixel>() as isize) as usize;
                for row in 0..h {
                    f(row, px::<BD>(b.bytes, b.offset + row * pxstride, w));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::include::common::bitdepth::BitDepth8;

    fn band_of(rows: usize, cols: usize) -> ReconBand {
        let mut b = ReconBand::default();
        b.arm(
            1,
            &[(16, 32, rows, cols, 1), (0, 0, 0, 0, 1), (0, 0, 0, 0, 1)],
        );
        b
    }

    fn dst_at(b: &mut ReconBand, row: usize, col: usize) -> ReconDst<'_> {
        ReconDst::Own(b.at::<BitDepth8>(0, row, col))
    }

    #[test]
    fn translation_is_plane_coordinates_minus_origin() {
        let mut b = band_of(8, 96);
        // Plane (16, 32) is band (0, 0).
        dst_at(&mut b, 16, 32).set::<BitDepth8>(7);
        // Plane (17, 34) is band (1, 2).
        dst_at(&mut b, 17, 34).set::<BitDepth8>(9);
        let stride = b.stride[0];
        assert_eq!(b.row_bytes(0, 0, stride)[0], 7);
        assert_eq!(b.row_bytes(0, 1, stride)[2], 9);
    }

    #[test]
    #[should_panic(expected = "recon band row out of range")]
    fn a_row_above_the_band_panics_it_does_not_alias() {
        let mut b = band_of(8, 96);
        dst_at(&mut b, 15, 32).set::<BitDepth8>(1);
    }

    #[test]
    #[should_panic]
    fn a_row_below_the_band_panics() {
        let mut b = band_of(8, 96);
        dst_at(&mut b, 24, 32).set::<BitDepth8>(1);
    }

    #[test]
    fn for_rows_mut_walks_the_bands_own_stride_not_the_pictures() {
        let mut b = band_of(4, 96);
        assert_eq!(b.stride[0], 128); // 96 rounded up to a 64-byte multiple.
        dst_at(&mut b, 16, 32).for_rows_mut::<BitDepth8, _>(96, 4, |y, row| {
            row.fill(y as u8 + 1);
        });
        for y in 0..4 {
            let r = b.row_bytes(0, y, 96);
            assert!(r.iter().all(|&v| v == y as u8 + 1), "row {y}");
        }
    }

    // The exclusion property itself is a COMPILE-time fact and cannot be
    // asserted at run time, so there is deliberately no test for it here — a
    // test that "passes" because a string matches would prove nothing. It is
    // proved the same way `forbid(unsafe_code)` is: by planting
    //
    //     let mut a = dst_at(&mut b, 16, 32);
    //     let mut c = dst_at(&mut b, 17, 32);
    //     a.set::<BitDepth8>(1);
    //     c.set::<BitDepth8>(2);
    //
    // and observing `error[E0499]: cannot borrow 'b' as mutable more than once
    // at a time`. See docs/MUT_RECON_KERNELS.md §6 for the recorded run.
}

/// The plane set one reconstruction call writes into: the shared picture, or
/// this worker's owned band.
///
/// This is the ENTIRE seam. `recon.rs` asks for a destination by PLANE pixel
/// coordinates rather than by a flat frame offset, which is what lets the band
/// carry its own compact stride — and is the coordinate translation #473
/// costed at 22 sites. It is paid once, here, in the same change that makes the
/// kernels take `&mut`.
pub(crate) enum ReconPlanes<'a> {
    Pic(&'a [crate::include::dav1d::picture::Rav1dPictureDataComponent; 3]),
    Own(&'a mut ReconBand),
}

impl<'a> ReconPlanes<'a> {
    /// Bind to the owned band if it is armed for this task, else to the shared
    /// picture.
    #[inline]
    pub(crate) fn bind(
        pic: &'a [crate::include::dav1d::picture::Rav1dPictureDataComponent; 3],
        band: &'a mut ReconBand,
    ) -> Self {
        if band.armed {
            Self::Own(band)
        } else {
            Self::Pic(pic)
        }
    }

    #[inline]
    pub(crate) fn is_owned(&self) -> bool {
        matches!(self, Self::Own(_))
    }

    /// Pixel stride of plane `pl` — the BAND's stride when owned, which is why
    /// no caller may read `f.cur.stride[..]` for a reconstruction offset.
    #[inline]
    pub(crate) fn pixel_stride<BD: BitDepth>(&self, pl: usize) -> isize {
        match self {
            Self::Pic(p) => p[pl].pixel_stride::<BD>(),
            Self::Own(b) => (b.stride[pl] / core::mem::size_of::<BD::Pixel>()) as isize,
        }
    }

    /// A writable region at plane pixel coordinates `(row, col)`.
    #[inline]
    pub(crate) fn dst<BD: BitDepth>(&mut self, pl: usize, row: usize, col: usize) -> ReconDst<'_> {
        match self {
            Self::Pic(p) => {
                let d = &p[pl];
                ReconDst::Pic(
                    d.with_offset::<BD>() + (row as isize * d.pixel_stride::<BD>() + col as isize),
                )
            }
            Self::Own(b) => ReconDst::Own(b.at::<BD>(pl, row, col)),
        }
    }

    /// A readable region at plane pixel coordinates `(row, col)`.
    #[inline]
    pub(crate) fn src<BD: BitDepth>(&self, pl: usize, row: usize, col: usize) -> ReconSrc<'_> {
        match self {
            Self::Pic(p) => {
                let d = &p[pl];
                ReconSrc::Pic(
                    d.with_offset::<BD>() + (row as isize * d.pixel_stride::<BD>() + col as isize),
                )
            }
            Self::Own(b) => {
                let (row0, col0) = b.origin[pl];
                let stride = b.stride[pl];
                let brow = row - row0;
                assert!(brow < b.rows[pl], "recon band row out of range");
                ReconSrc::Own(BandRef {
                    bytes: zerocopy::IntoBytes::as_bytes(&b.planes[pl][..]),
                    offset: (brow * stride) / core::mem::size_of::<BD::Pixel>() + (col - col0),
                    stride: stride as isize,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Frame- and superblock-row-level policy.
// ---------------------------------------------------------------------------

use crate::include::dav1d::headers::Rav1dPixelLayout;
use crate::src::internal::Rav1dContext;
use crate::src::internal::Rav1dFrameData;
use crate::src::internal::Rav1dTaskContext;

/// Runtime switch, so both arms of an A/B are the SAME BINARY and an inter-arm
/// delta cannot be a codegen artefact (#455's `probe-*` convention).
fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("RAV1D_OWNED_RECON").as_deref(), Ok("0")))
}

/// Decide, once per frame, whether reconstruction may run on owned bands.
///
/// Every decline leaves `f.owned_recon = false`, i.e. exactly today's
/// shared-picture path with exactly today's tracking.
pub(crate) fn frame_setup(c: &Rav1dContext, f: &mut Rav1dFrameData) {
    f.owned_recon = false;
    if !enabled() {
        return;
    }
    // The `asm`/`c-ffi` configuration keeps raw picture pointers across the
    // kernel boundary, which an owned band cannot supply; those dispatch arms
    // recover the tracked `PicOffset` and would panic on a band.
    if cfg!(feature = "c-ffi") {
        return;
    }
    // `__simd_test` re-reads and restores the destination through
    // picture-only primitives.
    if cfg!(feature = "__simd_test") {
        return;
    }
    let Some(frame_hdr) = f.frame_hdr.as_ref() else {
        return;
    };
    let frame_hdr = &***frame_hdr;
    // SCOPE: intra-only frames. Every writer of a recon plane must move
    // together, and the inter path (`mc`, OBMC, inter-intra blending, warp)
    // is not converted. See the module docs.
    if !frame_hdr.frame_type.is_key_or_intra() {
        return;
    }
    // Intra block copy reads the CURRENT picture as an MC reference, which
    // would be stale in a band until the stitch.
    if frame_hdr.allow_intrabc {
        return;
    }
    // Frame threading splits a tile into entropy and reconstruction passes.
    if c.fc.len() > 1 {
        return;
    }
    if f.cur.data.is_none() {
        return;
    }
    // Negative strides make the row arithmetic in `stitch` wrong; decline
    // rather than get it subtly wrong.
    let model = &f.cur.data.as_ref().unwrap().data;
    let n_planes = if f.cur.p.layout == Rav1dPixelLayout::I400 {
        1
    } else {
        3
    };
    for pl in 0..n_planes {
        if model[pl].stride() <= 0 || model[pl].byte_len() == 0 {
            return;
        }
    }
    f.owned_recon = true;
}

/// Geometry of the band for one `(tile, superblock row)` task, in each plane's
/// own pixel coordinates: `(row0, col0, rows, cols, pixel_size)`.
fn band_geometry(
    f: &Rav1dFrameData,
    t: &Rav1dTaskContext,
) -> (usize, [(usize, usize, usize, usize, usize); 3]) {
    let ts = &f.ts[t.ts];
    let layout = f.cur.p.layout;
    let n_planes = if layout == Rav1dPixelLayout::I400 {
        1
    } else {
        3
    };
    let ss_ver = (layout == Rav1dPixelLayout::I420) as usize;
    let ss_hor = (layout != Rav1dPixelLayout::I444) as usize;
    let pixel_size = if f.cur.p.bpc > 8 { 2 } else { 1 };

    let row0 = (t.b.y * 4) as usize;
    let col0 = (ts.tiling.col_start * 4) as usize;
    let rows = (f.sb_step * 4) as usize;
    let cols = ((ts.tiling.col_end - ts.tiling.col_start) * 4) as usize;

    let mut geom = [(0, 0, 0, 0, pixel_size); 3];
    geom[0] = (row0, col0, rows, cols, pixel_size);
    for pl in 1..n_planes {
        geom[pl] = (
            row0 >> ss_ver,
            col0 >> ss_hor,
            rows >> ss_ver,
            cols >> ss_hor,
            pixel_size,
        );
    }
    (n_planes, geom)
}

/// Arm this worker's band for one superblock row of one tile.
pub(crate) fn arm_sbrow(f: &Rav1dFrameData, t: &mut Rav1dTaskContext) {
    t.recon_band.disarm();
    if !f.owned_recon {
        return;
    }
    let ts = &f.ts[t.ts];
    if ts.tiling.col_end <= ts.tiling.col_start || t.b.y < 0 {
        return;
    }
    let (n_planes, geom) = band_geometry(f, t);
    t.recon_band.arm(n_planes, &geom);
    if !t.recon_band.armed() {
        return;
    }
    // The last superblock row of the frame is partial: only the rows the tile
    // actually reconstructs may be copied out.
    let y1 = std::cmp::min(t.b.y + f.sb_step, ts.tiling.row_end);
    if y1 <= t.b.y {
        t.recon_band.disarm();
        return;
    }
    let live = ((y1 - t.b.y) * 4) as usize;
    let layout = f.cur.p.layout;
    let ss_ver = (layout == Rav1dPixelLayout::I420) as usize;
    t.recon_band
        .set_live_rows([live, live >> ss_ver, live >> ss_ver]);
}

/// Copy this tile's superblock row out of the band and into the shared picture.
///
/// Called at the end of `rav1d_decode_tile_sbrow`, i.e. BEFORE the tile's
/// progress counter is published, so the filter task for this superblock row —
/// which waits on every tile having published it — always sees complete pixels.
///
/// Every row borrow taken here covers exactly the tile's own columns, so two
/// tile columns stitching the same rows never overlap. At 4K with 4x2 tiles
/// that is 64 row borrows per tile per superblock row.
pub(crate) fn stitch_sbrow<BD: BitDepth>(f: &Rav1dFrameData, t: &mut Rav1dTaskContext) {
    if !t.recon_band.armed() {
        return;
    }
    let dst_planes = &f.cur.data.as_ref().unwrap().data;
    let pixel_size = core::mem::size_of::<BD::Pixel>();
    for pl in 0..t.recon_band.n_planes() {
        let (row0, col0, rows, cols) = t.recon_band.plane_geometry(pl);
        let dst_stride = dst_planes[pl].pixel_stride::<BD>() as usize;
        let len = cols * pixel_size;
        for row in 0..rows {
            let src = t.recon_band.row_bytes(pl, row, len);
            let off = (row0 + row) * dst_stride + col0;
            let mut dst = dst_planes[pl].slice_mut::<BD, _>((off.., ..cols));
            zerocopy::IntoBytes::as_mut_bytes(&mut *dst).copy_from_slice(src);
        }
    }
    t.recon_band.disarm();
}
