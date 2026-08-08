//! THROWAWAY **site-selective** borrow-tracker nulling (feature `__probe_class`).
//!
//! # This is UNSOUND and must never reach a shipping build
//!
//! When a call site's class is selected, [`nulled`] makes `add` return
//! [`BorrowId::UNCHECKED`](crate::tracker_shard) **without registering the
//! borrow**, so overlapping mutable access from that site is no longer
//! detected. That is the whole point — it prices what the check costs — but it
//! means the feature is a measurement instrument, not a configuration. It is
//! `__`-prefixed, absent from `default`, absent from every published feature,
//! and the only crate feature that turns it on
//! (`rav1d-safe/probe-class`) is documented THROWAWAY alongside the rest of the
//! `probe-*` family.
//!
//! # The question it answers
//!
//! The global `__probe_addnop` arm prices the whole tracker at once: keep the
//! call, delete the work. That tells you the tracker's total, and nothing about
//! WHERE it is. Two candidate designs need to divide that total differently:
//!
//! * a static row-split of the picture buffer at tile setup removes the tracker
//!   from **tile reconstruction** only — AV1 tiles are independently
//!   reconstructable, so per-tile row segments are provably disjoint in safe
//!   Rust. It cannot cover the filter chain, which reads across tile edges.
//! * a non-atomic serial tracker helps at t=1 only, whatever the class mix.
//!
//! So the decision needs ms/frame attributable to *tile reconstruction* versus
//! *the filter chain* versus *everything else*, at each thread count. Nulling
//! one class at a time is the only way to get that: a profiler attributes
//! `add`/`remove` to whichever caller LLVM inlined it into, and a per-site
//! COUNT does not convert to time (the classes have different guard extents,
//! different shard-collision behaviour and different call-site register
//! pressure).
//!
//! # Why the selector is a runtime mask and not a `cfg`
//!
//! Every arm is then **the same binary**, so an inter-arm delta cannot be a
//! codegen or layout difference. A `const` mask would additionally let LLVM
//! prove the memo lookup dead in the null-nothing arm and delete it, making the
//! baseline arm cheaper than the arms it is the baseline for, which would
//! inflate every class's attributed cost by the instrument's own overhead.
//! `probe-class` with `RAV1D_CLS_NULL=none` versus a plain `base` build is what
//! prices that overhead, and it is reported next to the results.
//!
//! Cost of being a runtime mask: the nulled class's `add` body is still
//! *present* in the binary, just branched over. A design that deleted the
//! tracker outright would also recover its I-cache footprint, so every number
//! here is a LOWER bound on that design's saving. The global compile-time
//! `__probe_addnop` arm bounds the same quantity from the other side.

use core::panic::Location;
use core::sync::atomic::AtomicU8;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering::Relaxed;

/// Tile reconstruction: intra prediction, inverse transform, motion
/// compensation, palette — the writes a per-tile row split would cover.
pub const RECON: u8 = 0;
/// The filter chain: deblock, CDEF, loop restoration. Reads across tile
/// boundaries, so a per-tile split provably cannot cover it.
pub const FILTER: u8 = 1;
/// `decode.rs` itself: the block-level control plane (loop-filter masks,
/// coefficient/block info, per-block context) rather than picture pixels.
/// Split out from [`RECON`] because a picture-buffer row split does NOT cover
/// it, even though it runs inside tile reconstruction.
pub const DECODE: u8 = 2;
/// Everything else: `ctx.rs` / `env.rs` block contexts, CDFs, refmvs, and any
/// borrow whose site did not classify.
pub const OTHER: u8 = 3;
/// `include/dav1d/picture.rs`'s own loop bodies — in practice exactly
/// `BlockMut::drop`'s per-row compact write-back, which only exists under tile
/// threading. Its true class is that of the `block_mut` call that created the
/// block, which `#[track_caller]` cannot forward (the borrow is taken in
/// picture.rs's loop, not passed through from the caller). Rather than guess,
/// it gets its own class and its own arm, and the census proves who its
/// creators are: in these vectors `picture.rs:1889`'s count is EXACTLY equal to
/// the sum of the `itx_arm.rs` `block_mut` sites and `mc_arm.rs` /
/// `looprestoration*.rs` register nothing at all, so here it is 100% recon —
/// but on an inter frame or with loop restoration on it would be mixed, and a
/// hardcoded assignment would be silently wrong there.
pub const PICWB: u8 = 4;
pub const N_CLASSES: usize = 5;

pub const CLASS_NAMES: [&str; N_CLASSES] = ["recon", "filter", "decode", "other", "picwb"];

/// Modifier: null only borrows on instances at least `SHARD_MIN_LEN` (64 KiB)
/// long. That is the frame-sized-buffer population — the picture planes and
/// their kin — and it is the population a static per-tile ROW SPLIT of the
/// picture buffer could actually cover. Without this modifier the `recon` arm
/// also nulls `recon.rs`'s coefficient-context borrows and `ipred_prepare`'s
/// `BlockContext::intra` reads, which live on 32-byte instances that no picture
/// row split touches: 1.26 M of the class's 5.53 M registrations per frame at
/// t=1. Attributing those to a picture-buffer design would overstate it.
pub const ONLY_BIG: u8 = 1 << 5;
/// The complement, so the two halves can be priced separately and checked
/// against the un-modified arm.
pub const ONLY_SMALL: u8 = 1 << 6;

/// Bit `i` set => class `i` is NOT tracked.
static NULL_MASK: AtomicU8 = AtomicU8::new(0);

/// How many times the memo missed. Equals the number of distinct call sites
/// when the table is behaving; anything larger means two hot sites are
/// thrashing one slot and the fast path is not fast. Reported by the census.
pub static MEMO_MISSES: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn set_null_mask(m: u8) {
    NULL_MASK.store(m, Relaxed);
}

#[inline]
pub fn get_null_mask() -> u8 {
    NULL_MASK.load(Relaxed)
}

/// `"none"` | `"all"` | comma-separated class names. Returns `None` on an
/// unknown name so a typo in a sweep script fails loudly instead of silently
/// measuring the baseline arm five times.
pub fn mask_from_str(s: &str) -> Option<u8> {
    let s = s.trim();
    if s.is_empty() || s == "none" {
        return Some(0);
    }
    if s == "all" {
        return Some((1 << N_CLASSES) - 1);
    }
    let mut m = 0u8;
    for part in s.split(',') {
        let part = part.trim();
        match part {
            "big" => m |= ONLY_BIG,
            "small" => m |= ONLY_SMALL,
            _ => m |= 1 << CLASS_NAMES.iter().position(|n| *n == part)?,
        }
    }
    // A size modifier with no class is a no-op arm that would silently measure
    // the baseline under a different name.
    if m & ((1 << N_CLASSES) - 1) == 0 {
        return None;
    }
    Some(m)
}

const CAP: usize = 8192;

/// Memo entry: `(loc_ptr & !7) | (class + 1)`. `Location<'static>` is
/// `{ &'static str, u32, u32 }`, so it is 8-aligned and its low three bits are
/// free; storing `class + 1` (1..=5) keeps 0 meaning "empty" without a separate
/// occupancy bit. One `AtomicUsize` load and two masks is the whole fast path.
static MEMO: [AtomicUsize; CAP] = [const { AtomicUsize::new(0) }; CAP];

/// Two-way: `h` and `h ^ 1` are adjacent 8-byte entries, i.e. the same cache
/// line, so the second probe costs no extra miss. With ~70 distinct sites in
/// 8192 slots a two-way table makes thrashing effectively impossible, and
/// [`MEMO_MISSES`] proves it rather than assuming it.
#[inline(always)]
fn slot(p: usize) -> usize {
    ((p.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 50) & (CAP as usize - 1)) & !1
}

#[inline(always)]
pub fn class_of(loc: &'static Location<'static>) -> u8 {
    let p = loc as *const Location<'static> as usize;
    let h = slot(p);
    let e0 = MEMO[h].load(Relaxed);
    if e0 & !7 == p {
        return (e0 & 7) as u8 - 1;
    }
    let e1 = MEMO[h + 1].load(Relaxed);
    if e1 & !7 == p {
        return (e1 & 7) as u8 - 1;
    }
    class_slow(loc, p, h)
}

#[cold]
#[inline(never)]
fn class_slow(loc: &'static Location<'static>, p: usize, h: usize) -> u8 {
    MEMO_MISSES.fetch_add(1, Relaxed);
    let c = class_for_file(loc.file());
    let packed = (p & !7) | (c as usize + 1);
    // Racing writers store the same value for the same key, so a plain store is
    // enough; the only thing worth avoiding is evicting a live entry, hence the
    // free-slot preference.
    if MEMO[h].load(Relaxed) == 0 {
        MEMO[h].store(packed, Relaxed);
    } else {
        MEMO[h + 1].store(packed, Relaxed);
    }
    c
}

/// Classification is by SOURCE FILE of the borrow site, which is only the right
/// key because the decoder's files are already split along exactly this seam —
/// `ipred*`/`itx*`/`mc*`/`recon` reconstruct, `loopfilter*`/`cdef*`/`l[rf]_*`
/// filter. It is only accurate if the `#[track_caller]` chain reaches the real
/// caller: the pass-through wrappers in `picture.rs` / `with_offset.rs` /
/// `pixels.rs` carry `#[cfg_attr(debug_assertions, track_caller)]`, which is
/// OFF in release, and the census (`__probe_sites`) is what proves the widening
/// took — if borrows still land on `picture.rs`, the classification is blind.
fn class_for_file(f: &str) -> u8 {
    // Everything the decoder registers from `include/` is picture.rs's own loop
    // bodies. Keyed on the DIRECTORY rather than the file so a future helper
    // there lands in the same explicitly-unattributed bucket instead of being
    // silently counted as `other`.
    if f.starts_with("include/") {
        return PICWB;
    }
    let b = match f.rfind('/') {
        Some(i) => &f[i + 1..],
        None => f,
    };
    match b {
        "recon.rs" | "ipred.rs" | "ipred_prepare.rs" | "itx.rs" | "mc.rs" | "pal.rs"
        | "ipred_arm.rs" | "mc_arm.rs" | "pal_arm.rs" | "itx_arm.rs" => RECON,
        "loopfilter.rs" | "loopfilter_arm.rs" | "lf_apply.rs" | "lf_mask.rs" | "cdef.rs"
        | "cdef_apply.rs" | "cdef_arm.rs" | "lr_apply.rs" | "looprestoration.rs"
        | "looprestoration_arm.rs" => FILTER,
        "decode.rs" => DECODE,
        // `itx_arm_neon_8x8.rs`, `itx_arm_neon_rect_large.rs`, ...
        _ if b.starts_with("itx_arm") => RECON,
        _ => OTHER,
    }
}

/// Public for the census only.
pub fn class_for_file_pub(f: &str) -> u8 {
    class_for_file(f)
}

/// How many registrations were actually dropped, per class. Only compiled into
/// the CENSUS binary (`__probe_sites`), never into the timing binary, so the
/// liveness proof costs the measurement nothing: run the census with each
/// `RAV1D_CLS_NULL` value and check `NULLED[c]` equals that class's `CLASS`
/// count. An arm that silently nulls nothing is the failure mode this catches,
/// and it is invisible in both the pixels and the wall clock.
#[cfg(feature = "__probe_sites")]
pub static NULLED: [AtomicU64; N_CLASSES] = [const { AtomicU64::new(0) }; N_CLASSES];
/// Registrations per (class, big) so the census can say exactly what an arm
/// with a size modifier covers.
#[cfg(feature = "__probe_sites")]
pub static BY_SIZE: [[AtomicU64; 2]; N_CLASSES] =
    [const { [const { AtomicU64::new(0) }; 2] }; N_CLASSES];

/// `big` is whether the instance is at least `SHARD_MIN_LEN`; the caller passes
/// it because only the tracker knows its own length.
#[inline(always)]
pub fn nulled(loc: &'static Location<'static>, big: bool) -> bool {
    let c = class_of(loc);
    let m = NULL_MASK.load(Relaxed);
    let size_ok = if big {
        m & ONLY_SMALL == 0
    } else {
        m & ONLY_BIG == 0
    };
    let hit = (m >> c) & 1 != 0 && size_ok;
    #[cfg(feature = "__probe_sites")]
    {
        BY_SIZE[c as usize][big as usize].fetch_add(1, Relaxed);
        if hit {
            NULLED[c as usize].fetch_add(1, Relaxed);
        }
    }
    hit
}
