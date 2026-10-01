//! Total byte encodings ("plain data") of enums that live in
//! [`DisjointMut`](crate::src::disjoint_mut::DisjointMut) buffers.
//!
//! A [`DisjointMut`] element type must be [`PlainData`]
//! (`Copy + zerocopy::FromBytes`): every bit pattern is a valid value, and there
//! are no pointers or niches. A Rust `enum` is not -- an out-of-range
//! discriminant is instant UB -- so enum-valued context arrays store one of the
//! `*Byte` newtypes below instead and decode on read.
//!
//! Decoding is TOTAL: a byte that no variant encodes maps to a fixed in-range
//! fallback, so a torn, stale or aliased read can only ever produce a wrong but
//! valid value. Writers only ever store [`new`](TxfmSizeByte::new)'s output, so
//! the fallback is unreachable in a correct decode; it exists to make the
//! "overlapping access yields wrong values at worst" claim a compile-time fact
//! (see `rav1d_disjoint_mut::PlainData`).
//!
//! [`PlainData`]: crate::src::disjoint_mut::PlainData

use crate::include::dav1d::headers::Rav1dFilterMode;
use crate::src::align::ArrayDefault;
use crate::src::levels::BlockSize;
use crate::src::levels::CompInterType;
use crate::src::levels::TxfmSize;
use zerocopy::FromBytes;
use zerocopy::Immutable;
use zerocopy::IntoBytes;
use zerocopy::KnownLayout;

macro_rules! plain_byte {
    (
        $(#[$meta:meta])*
        $name:ident($ty:ty),
        default: $default:expr,
        decode: |$v:ident| $decode:expr,
        encode: |$e:ident| $encode:expr $(,)?
    ) => {
        $(#[$meta])*
        #[derive(
            Clone, Copy, PartialEq, Eq, Debug, FromBytes, IntoBytes, KnownLayout, Immutable,
        )]
        #[repr(transparent)]
        pub struct $name(u8);

        impl $name {
            #[inline(always)]
            pub fn new($e: $ty) -> Self {
                Self($encode)
            }

            /// Total: every byte decodes to a valid value.
            #[inline(always)]
            pub fn get(self) -> $ty {
                let $v = self.0;
                $decode
            }
        }

        impl Default for $name {
            #[inline(always)]
            fn default() -> Self {
                Self::new($default)
            }
        }

        impl ArrayDefault for $name {
            #[inline(always)]
            fn default() -> Self {
                <Self as Default>::default()
            }
        }

        impl From<$ty> for $name {
            #[inline(always)]
            fn from(value: $ty) -> Self {
                Self::new(value)
            }
        }
    };
}

plain_byte!(
    /// A [`TxfmSize`] stored as a byte; out-of-range bytes decode to the default.
    TxfmSizeByte(TxfmSize),
    default: <TxfmSize as Default>::default(),
    decode: |v| TxfmSize::from_repr(v as usize).unwrap_or_default(),
    encode: |e| e as u8,
);

plain_byte!(
    /// A [`Rav1dFilterMode`] stored as a byte; out-of-range bytes decode to the default.
    FilterModeByte(Rav1dFilterMode),
    default: <Rav1dFilterMode as Default>::default(),
    decode: |v| Rav1dFilterMode::from_repr(v as usize).unwrap_or_default(),
    encode: |e| e as u8,
);

plain_byte!(
    /// A [`BlockSize`] stored as a byte; out-of-range bytes decode to `Bs128x128`.
    BlockSizeByte(BlockSize),
    default: BlockSize::Bs128x128,
    decode: |v| BlockSize::from_repr(v).unwrap_or(BlockSize::Bs128x128),
    encode: |e| e as u8,
);

plain_byte!(
    /// An `Option<`[`CompInterType`]`>` stored as a byte: `0` is `None`,
    /// otherwise the variant's discriminant. Unencoded bytes decode to `None`.
    CompTypeByte(Option<CompInterType>),
    default: None,
    decode: |v| match v {
        1 => Some(CompInterType::WeightedAvg),
        2 => Some(CompInterType::Avg),
        3 => Some(CompInterType::Seg),
        4 => Some(CompInterType::Wedge),
        _ => None,
    },
    encode: |e| match e {
        None => 0,
        Some(c) => c as u8,
    },
);

impl BlockSizeByte {
    #[inline(always)]
    pub fn dimensions(self) -> &'static [u8; 4] {
        self.get().dimensions()
    }
}

impl CompTypeByte {
    #[inline(always)]
    pub fn is_some(self) -> bool {
        self.get().is_some()
    }

    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.get().is_none()
    }
}
