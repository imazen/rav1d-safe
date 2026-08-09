#![cfg_attr(not(asm_fn_ptrs), forbid(unsafe_code))]
#[cfg(asm_fn_ptrs)]
use crate::include::common::bitdepth::BitDepth;
#[cfg(asm_fn_ptrs)]
use crate::src::pixels::Pixels;
use crate::src::strided::Strided;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;

/// Identity of the AV1 TILE a picture reference belongs to, or [`TILE_ANY`].
///
/// # Why the borrow tracker wants this
///
/// AV1 guarantees that tile regions do not overlap during reconstruction —
/// intra prediction, MV prediction and entropy contexts all reset at tile
/// boundaries — so two reconstruction accesses in DIFFERENT tiles are disjoint
/// by the format, whatever their byte intervals look like. The tracker cannot
/// see that today: it hashes ADDRESSES into shards, so two tile columns on the
/// same picture rows collide on a shard, and one strided access spans many
/// shards. Keying on tile identity makes both problems structural rather than
/// statistical.
///
/// # Why it rides on `WithOffset` rather than a thread-local
///
/// MEASURED on this box (`~/tmp/tilekey-micro`, best-of-9 interleaved, an
/// out-of-line callee so the read cannot be hoisted out of a caller's loop):
/// reading the key from a `thread_local!` costs **+0.502 ns per call** — macOS
/// compiles `TLS.with(..)` to an indirect `blr` into `_tlv_get_addr` plus the
/// stack frame that call forces. Reading it from an ARGUMENT costs
/// **+0.000 ns** (as does a global atomic). At the 22,700,725 registrations per
/// 4K frame this decoder makes at t=8, the thread-local channel would cost
/// 11.4 ms/frame against a whole tracker that costs 19.7 — i.e. it would eat
/// the win before the win existed. So the key travels on the reference.
pub type TileKey = u16;

/// "Not attributable to a single tile" — the conservative key.
///
/// Any reference that has not been given a tile carries this, and a borrow
/// registered under it behaves exactly as it does today. It is the SAFE
/// direction to be wrong in: an unkeyed borrow keeps being compared against
/// everything, a wrongly-keyed one would not be.
pub const TILE_ANY: TileKey = u16::MAX;

#[derive(Clone, Copy)]
pub struct WithOffset<T> {
    pub data: T,
    pub offset: usize,
    /// See [`TileKey`]. Preserved by every offset arithmetic op below, because
    /// moving within a tile's region does not change which tile you are in —
    /// the caller that KNOWS the tile stamps it once, at the top of
    /// reconstruction, and every derived reference inherits it.
    pub key: TileKey,
}

impl<T> WithOffset<T> {
    /// A reference with no tile attribution.
    #[inline(always)]
    pub fn any(data: T, offset: usize) -> Self {
        Self {
            data,
            offset,
            key: TILE_ANY,
        }
    }

    /// Stamp this reference (and everything derived from it) with a tile.
    ///
    /// The caller asserts that every byte reachable through the returned
    /// reference lies inside that tile's pixel rectangle. See the module docs
    /// on [`TileKey`] for what the tracker then does with it.
    #[inline(always)]
    pub fn keyed(mut self, key: TileKey) -> Self {
        self.key = key;
        self
    }
}

impl<T> AddAssign<usize> for WithOffset<T> {
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn add_assign(&mut self, rhs: usize) {
        self.offset += rhs;
    }
}

impl<T> SubAssign<usize> for WithOffset<T> {
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn sub_assign(&mut self, rhs: usize) {
        self.offset -= rhs;
    }
}

impl<T> AddAssign<isize> for WithOffset<T> {
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn add_assign(&mut self, rhs: isize) {
        self.offset = self.offset.wrapping_add_signed(rhs);
    }
}

impl<T> SubAssign<isize> for WithOffset<T> {
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn sub_assign(&mut self, rhs: isize) {
        self.offset = self.offset.wrapping_add_signed(-rhs);
    }
}

impl<T> Add<usize> for WithOffset<T> {
    type Output = Self;

    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn add(mut self, rhs: usize) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T> Sub<usize> for WithOffset<T> {
    type Output = Self;

    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn sub(mut self, rhs: usize) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T> Add<isize> for WithOffset<T> {
    type Output = Self;

    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn add(mut self, rhs: isize) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T> Sub<isize> for WithOffset<T> {
    type Output = Self;

    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn sub(mut self, rhs: isize) -> Self::Output {
        self -= rhs;
        self
    }
}

#[cfg(asm_fn_ptrs)]
impl<P: Pixels> WithOffset<P> {
    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn as_ptr<BD: BitDepth>(&self) -> *const BD::Pixel {
        self.data.as_ptr_at::<BD>(self.offset)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn as_mut_ptr<BD: BitDepth>(&self) -> *mut BD::Pixel {
        self.data.as_mut_ptr_at::<BD>(self.offset)
    }

    pub fn wrapping_as_ptr<BD: BitDepth>(&self) -> *const BD::Pixel {
        self.data.wrapping_as_ptr_at::<BD>(self.offset)
    }

    pub fn wrapping_as_mut_ptr<BD: BitDepth>(&self) -> *const BD::Pixel {
        self.data.wrapping_as_mut_ptr_at::<BD>(self.offset)
    }
}

impl<S: Strided> Strided for WithOffset<S> {
    fn stride(&self) -> isize {
        self.data.stride()
    }
}
