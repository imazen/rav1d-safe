#![forbid(unsafe_code)]
//! The [`CaseSet`] API below is a safe and simplified version of the `case_set*` macros in `ctx.h`.
//!
//! The `case_set*` macros themselves replaced `memset`s in order to further optimize them
//! (in e3b5d4d044506f9e0e95e79b3de42fd94386cc61,
//! which performed the following optimizations:
//! 1. larger, inlinable writes for small power of 2 lengths
//! 2. aligned writes for the above
//! 3. 8-byte aligned fields [`BlockContext`]
//!   * allows 8-byte aligned writes
//!   * better cache boundary alignment
//!
//! (1) is easy to preserve in Rust, but (2) is difficult to do so without overhead,
//! as unaligned writes are UB, and so we'd need to check at runtime if they're aligned
//! (a runtime-determined `off`set is used, so we can't reasonably ensure this at compile-time).
//!
//! To more thoroughly check this, I ran the same benchmarks done in
//! e3b5d4d044506f9e0e95e79b3de42fd94386cc61, which introduced the `case_set*` macros:
//!
//! ```sh
//! cargo build --release && hyperfine './target/release/dav1d -i ./tests/large/chimera_8b_1080p.ivf -l 1000 -o /dev/null'
//! ```
//!
//! for 3 implementations:
//! 1. the original `case_set*` macros translated directly to `unsafe` Rust `fn`s
//! 2. the safe [`CaseSet`] implementation below using [`small_memset`] with its small powers of 2 optimization
//! 3. a safe [`CaseSet`] implementation using [`slice::fill`]/`memset` only
//!
//! The [`small_memset`] version was ~1.27% faster than the `case_set*` one,
//! and ~3.26% faster than the `memset` one.
//! The `case_set*` macros were also faster than `memset` in C by a similar margin,
//! meaning the `memset` option is the slowest in both C and Rust,
//! and since it was replaced with `case_set*` in C, we shouldn't use it in Rust.
//! Thus, the [`small_memset`] implementation seems optimal, as it:
//! * is the fastest of the Rust implementations
//! * is completely safe
//! * employs the same small powers of 2 optimization the `case_set*` implementation did
//! * is far simpler than the `case_set*` implementation, consisting of a `match` and array writes
//!
//! [`BlockContext`]: crate::src::env::BlockContext
use std::iter::zip;

/// Perform a `memset` optimized for lengths that are small powers of 2.
///
/// For power of 2 lengths `<= UP_TO`,
/// the `memset` is done as an array write of that exactly (compile-time known) length.
/// If the length is not a power of 2 or `> UP_TO`,
/// then the `memset` is done by [`slice::fill`] (a `memset` call) if `WITH_DEFAULT` is `true`,
/// or else skipped if `WITH_DEFAULT` is `false`.
///
/// This optimizes for the common cases where `buf.len()` is a small power of 2,
/// where the array write is optimized as few and large stores as possible.
#[inline]
pub fn small_memset<T: Clone + Copy, const UP_TO: usize, const WITH_DEFAULT: bool>(
    buf: &mut [T],
    val: T,
) {
    fn as_array<T: Clone + Copy, const N: usize>(buf: &mut [T]) -> &mut [T; N] {
        buf.try_into().unwrap()
    }
    match buf.len() {
        01 if UP_TO >= 01 => *as_array(buf) = [val; 01],
        02 if UP_TO >= 02 => *as_array(buf) = [val; 02],
        04 if UP_TO >= 04 => *as_array(buf) = [val; 04],
        08 if UP_TO >= 08 => *as_array(buf) = [val; 08],
        16 if UP_TO >= 16 => *as_array(buf) = [val; 16],
        32 if UP_TO >= 32 => *as_array(buf) = [val; 32],
        _ => {
            if WITH_DEFAULT {
                buf.fill(val)
            }
        }
    }
}

pub struct CaseSetter<const UP_TO: usize, const WITH_DEFAULT: bool> {
    offset: usize,
    len: usize,
}

impl<const UP_TO: usize, const WITH_DEFAULT: bool> CaseSetter<UP_TO, WITH_DEFAULT> {
    #[inline]
    pub fn set<T: Clone + Copy>(&self, buf: &mut [T], val: T) {
        small_memset::<T, UP_TO, WITH_DEFAULT>(&mut buf[self.offset..][..self.len], val);
    }
}

/// The entrypoint to the [`CaseSet`] API.
///
/// `UP_TO` and `WITH_DEFAULT` are made const generic parameters rather than have multiple `case_set*` `fn`s,
/// and these are put in a separate `struct` so that these 2 generic parameters
/// can be manually specified while the ones on the methods are inferred.
pub struct CaseSet<const UP_TO: usize, const WITH_DEFAULT: bool>;

impl<const UP_TO: usize, const WITH_DEFAULT: bool> CaseSet<UP_TO, WITH_DEFAULT> {
    /// Perform one case set.
    ///
    /// This API is generic over the element type (`T`) rather than hardcoding `u8`,
    /// as sometimes other types are used, though only `i8` is used currently.
    ///
    /// The `len` and `offset` are supplied here and
    /// applied to each `buf` passed to [`CaseSetter::set`] in `set_ctx`.
    #[inline]
    pub fn one<T, F>(ctx: T, len: usize, offset: usize, mut set_ctx: F)
    where
        F: FnMut(&CaseSetter<UP_TO, WITH_DEFAULT>, T),
    {
        set_ctx(&CaseSetter { offset, len }, ctx);
    }

    /// Perform many case sets in one call.
    ///
    /// This allows specifying the `set_ctx` closure inline easily,
    /// and also allows you to group the same args together.
    ///
    /// The `lens`, `offsets`, and `dirs` are zipped and passed to [`CaseSet::one`],
    /// where `dirs` can be an array of any type and whose elements are passed back to the `set_ctx` closure.
    #[inline]
    pub fn many<T, F, const N: usize>(
        dirs: [T; N],
        lens: [usize; N],
        offsets: [usize; N],
        mut set_ctx: F,
    ) where
        F: FnMut(&CaseSetter<UP_TO, WITH_DEFAULT>, T),
    {
        for (dir, (len, offset)) in zip(dirs, zip(lens, offsets)) {
            Self::one(dir, len, offset, &mut set_ctx);
        }
    }
}

/// Splat one per-block context update into BOTH directions of the neighbour
/// context: the worker's own **left** context (`l`) and the frame's shared
/// **above** context (`a`).
///
/// # Why this is not [`CaseSet::many`]
///
/// `many` takes a homogeneous `[T; N]`, so both directions must arrive as the
/// same reference type. `t.l` and `f.a[t.a]` are both `&BlockContext` today, so
/// they do — and that is precisely the problem: it forces the LEFT direction
/// through [`CaseSetter::set_disjoint`], i.e. through the borrow tracker, even
/// though `t.l` is a field of `Rav1dTaskContext` that no other worker can name.
/// Half of `ctx.rs`'s registrations were bought for nothing.
///
/// Splitting the two directions lets the left one take
/// [`CaseSetter::set_exclusive`] (`&mut`, no record) while the above one keeps
/// [`CaseSetter::set_disjoint`] (`&`, tracked, genuinely shared across tile
/// workers).
///
/// # Why a macro
///
/// The two directions need different *reference types*, which no closure
/// parameter can abstract over without either a runtime branch per field or a
/// per-field accessor trait. Two hand-written bodies would work and would drift:
/// these lists run to twelve fields, and a field updated on one side only is a
/// silent bitstream bug that no type checks. One field list, two expansions.
///
/// # Shape
///
/// ```ignore
/// case_set_al! {
///     <32, false>
///     l: (&mut t.l, bh4 as usize, by4 as usize),
///     a: (ta,       bw4 as usize, bx4 as usize),
///     // field = (value written LEFT, value written ABOVE)
///     tx_intra = (t_dim.lh as i8, t_dim.lw as i8),
///     mode     = (y_mode_nofilt,  y_mode_nofilt),
///     r#ref[0] = (-1, -1),
/// }
/// ```
///
/// The value is always a pair, even where both directions write the same thing.
/// That is deliberate: the asymmetric sites (`t_dim.lh`/`lw`, `b_dim[3]`/`[2]`,
/// `dav1d_al_part_ctx[1]`/`[0]`) are the ones a reader must not miss, and a
/// single-value shorthand would hide which is which.
///
/// `also |case| { .. }` on either direction appends statements to that
/// direction's `CaseSet::one` body, for the few sites that also splat a
/// non-`BlockContext` array (`t.pal_sz_uv`) with the same offset and length.
/// The binder name is supplied by the caller so it resolves under macro
/// hygiene.
macro_rules! case_set_al {
    (
        <$UP:literal, $WD:literal>
        l: ($l:expr, $llen:expr, $loff:expr) $(also |$lc:ident| $lextra:block)?,
        a: ($a:expr, $alen:expr, $aoff:expr) $(also |$ac:ident| $aextra:block)?,
        $( $f:ident $([$fi:expr])? = ($lv:expr, $av:expr) ),* $(,)?
    ) => {{
        // LEFT: exclusive, untracked. `&mut` is the whole point.
        $crate::src::ctx::CaseSet::<$UP, $WD>::one((), $llen, $loff, |case, ()| {
            let l = &mut *$l;
            $( case.set(&mut l.$f $([$fi])? [..], $lv); )*
            $( { let $lc = case; $lextra } )?
        });
        // ABOVE: `a` is a `&mut BlockContext` borrowed from the per-block
        // `f.a` ELEMENT guard (tile workers own disjoint `t.a` slots), so
        // these writes need no per-field tracker record either.
        $crate::src::ctx::CaseSet::<$UP, $WD>::one((), $alen, $aoff, |case, ()| {
            let a = &mut *$a;
            $( case.set(&mut a.$f $([$fi])? [..], $av); )*
            $( { let $ac = case; $aextra } )?
        });
    }};
}
pub(crate) use case_set_al;
