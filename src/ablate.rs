//! Per-kernel-family SIMD ablation switch (measurement only, `__ablate` feature).
//!
//! # Why this exists
//!
//! `CpuLevel::Scalar` / `rav1d_set_cpu_flags_mask(0)` does **not** disable the
//! safe-SIMD kernels on aarch64: every `safe_simd/*_arm.rs` dispatcher gates on
//! `archmage::Arm64::summon()` (a compile/run-time CPU token) and never reads
//! `rav1d_cpu_flags_mask`. So there was no way to ask "which kernel family is
//! responsible for this failing conformance vector?".
//!
//! This module adds one: a process-global bitmask of *disabled* families. A
//! disabled dispatcher returns `false` ("not handled") at its entry, so the
//! caller falls through to the generic scalar reference. Ablating family X and
//! re-running the corpus turns correlational evidence ("this kernel logs
//! mismatches") into causal evidence ("these N vectors' MD5s are wrong
//! *because of* X").
//!
//! # Cost when the feature is off
//!
//! [`is_off`] is `#[inline(always)]` and returns a `const false` when
//! `__ablate` is not enabled, so every dispatcher's guard folds away entirely.
//! This is compile-time-gated for the same reason `__simd_test` is: no
//! per-call atomic load on the hot path in a production build.
//!
//! # Scope
//!
//! Measurement infrastructure. Never enable `__ablate` in a shipped build —
//! ablating a family makes the decoder slower, not wrong (the scalar path is
//! the reference), but the switch is process-global and racy by design.

/// A dispatchable SIMD kernel family, at the granularity a fix would target.
///
/// `MC_PUT` and `MC_PREP` are split because they are separate kernels with
/// separate known drift classes (`MC_PUT_MISMATCH` vs `MC_PREP_MISMATCH`), and
/// `MC_OTHER` collects the compound-prediction/blend/warp/scale helpers that
/// share `mc_arm.rs` but not the 8-tap filter code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Family {
    Itx = 0,
    McPut = 1,
    McPrep = 2,
    McOther = 3,
    Cdef = 4,
    LoopFilter = 5,
    LoopRestoration = 6,
    IntraPred = 7,
    FilmGrain = 8,
}

impl Family {
    pub const ALL: &'static [Family] = &[
        Family::Itx,
        Family::McPut,
        Family::McPrep,
        Family::McOther,
        Family::Cdef,
        Family::LoopFilter,
        Family::LoopRestoration,
        Family::IntraPred,
        Family::FilmGrain,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Family::Itx => "itx",
            Family::McPut => "mc_put",
            Family::McPrep => "mc_prep",
            Family::McOther => "mc_other",
            Family::Cdef => "cdef",
            Family::LoopFilter => "loopfilter",
            Family::LoopRestoration => "looprestoration",
            Family::IntraPred => "ipred",
            Family::FilmGrain => "filmgrain",
        }
    }

    /// Parse a family from its [`Family::name`]; `"all"` is handled by callers.
    pub fn from_name(s: &str) -> Option<Family> {
        Family::ALL.iter().copied().find(|f| f.name() == s)
    }

    /// Only the two `__ablate` call sites (`set_disabled`, `is_off`) use this,
    /// and both are cfg-gated — so without the feature this is dead code and
    /// `clippy -D warnings` (the CI job) rejects it. Gate it to match its
    /// callers rather than silencing the lint.
    #[cfg(feature = "__ablate")]
    const fn bit(self) -> u32 {
        1u32 << (self as u32)
    }
}

/// Whether the ablation switch is compiled in.
///
/// A harness MUST assert this before reporting numbers. Without it every arm
/// silently measures the same unablated decoder and the whole run reads as
/// "no kernel is responsible for anything" — the exact false-negative this
/// module exists to avoid.
pub const ENABLED: bool = cfg!(feature = "__ablate");

#[cfg(feature = "__ablate")]
static DISABLED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Disable every family in `families` (and only those).
///
/// A no-op without `__ablate`; guard callers on [`ENABLED`].
pub fn set_disabled(families: &[Family]) {
    #[cfg(feature = "__ablate")]
    {
        let mut mask = 0u32;
        for f in families {
            mask |= f.bit();
        }
        DISABLED.store(mask, std::sync::atomic::Ordering::SeqCst);
    }
    #[cfg(not(feature = "__ablate"))]
    let _ = families;
}

/// True when this family's SIMD dispatch has been ablated to scalar.
#[cfg(feature = "__ablate")]
#[inline(always)]
pub fn is_off(f: Family) -> bool {
    DISABLED.load(std::sync::atomic::Ordering::Relaxed) & f.bit() != 0
}

/// Compiles to a constant `false` — every dispatcher's guard folds away.
#[cfg(not(feature = "__ablate"))]
#[inline(always)]
pub fn is_off(_f: Family) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Activity counters
// ---------------------------------------------------------------------------
//
// "Which vectors exercise family X, and by how much?" is a question that has
// cost this project real time twice: `ROADMAP_SIMD_PORTING.md` records loop
// restoration measuring **0.0 ms/frame** on the 4K vectors — not because it is
// fast but because loop restoration is *off* in those bitstreams — and the P2
// profile only discovered that after budgeting a port. A profiler cannot tell
// "cheap" from "never called"; a counter can.
//
// Same compile-time gating as `is_off`: without `__ablate` these are no-ops and
// the call sites vanish.

#[cfg(feature = "__ablate")]
static ACTIVITY: [std::sync::atomic::AtomicU64; 9] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 9];

/// Record that `f` processed `units` of work (pixels, unless noted).
///
/// Called from each family's dispatcher *before* its `is_off` early-return, so
/// the count reflects what the bitstream asks for, not what SIMD handled.
#[inline(always)]
pub fn note(f: Family, units: u64) {
    #[cfg(feature = "__ablate")]
    ACTIVITY[f as usize].fetch_add(units, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "__ablate"))]
    let _ = (f, units);
}

/// Per-family counts in [`Family::ALL`] order. All zero without `__ablate`.
pub fn activity_snapshot() -> [u64; 9] {
    #[cfg(feature = "__ablate")]
    {
        let mut out = [0u64; 9];
        for (i, slot) in ACTIVITY.iter().enumerate() {
            out[i] = slot.load(std::sync::atomic::Ordering::Relaxed);
        }
        out
    }
    #[cfg(not(feature = "__ablate"))]
    [0u64; 9]
}

/// Zero every counter (call between vectors to get per-vector numbers).
pub fn activity_reset() {
    #[cfg(feature = "__ablate")]
    for slot in ACTIVITY.iter() {
        slot.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Inverse-transform shape census
// ---------------------------------------------------------------------------
//
// The activity counters above answer "was this family called at all?". This
// answers the next question down, for the one family where it changed a
// decision: **which transform SHAPES does a bitstream actually ask for, and
// which of them did SIMD handle?**
//
// It exists because a profile alone got the ranking wrong. Issue #455 open item
// 5 ("16bpc itx above 16x16 is still scalar") looked like the top 10bpc target:
// `<itx::itxfm::Fn>::call` carries the largest itx self-time share at 4K 10bpc,
// and the scalar reference is inlined into it. The census says the fallback is
// **20 calls out of 272,949** on that vector — 0.15% of coefficient area — and
// 0 on `v4k_8tile_10b`. What `Fn::call` actually holds is the *hbd dispatch and
// driver*, also inlined into it. Porting 32/64-point 16bpc kernels would have
// been days of work for nothing measurable on any vector this campaign has.
//
// Sixteen counters and two atomics per transform call, all behind `__ablate`;
// without the feature `note_itx_shape` is a no-op and the call site vanishes.

#[cfg(feature = "__ablate")]
const ITX_SIZES: usize = 19;

/// `[bpc16 as usize * 2 + handled as usize][tx_size]`.
#[cfg(feature = "__ablate")]
static ITX_SHAPES: [[std::sync::atomic::AtomicU64; ITX_SIZES]; 4] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; ITX_SIZES] }; 4];

/// Record one inverse-transform call: its size index, its bit depth, and
/// whether a SIMD kernel took it (`false` = fell through to the reference).
#[inline(always)]
pub fn note_itx_shape(tx_size: usize, bitdepth: u8, handled: bool) {
    #[cfg(feature = "__ablate")]
    {
        let row = (bitdepth != 8) as usize * 2 + handled as usize;
        if tx_size < ITX_SIZES {
            ITX_SHAPES[row][tx_size].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    #[cfg(not(feature = "__ablate"))]
    let _ = (tx_size, bitdepth, handled);
}

/// The census as TSV: `depth  path  shape  calls  coeff_area`.
///
/// Header only without `__ablate` — assert [`ENABLED`] in the harness.
#[cfg(feature = "__ablate")]
pub fn itx_shape_report() -> String {
    use crate::src::levels::TxfmSize;
    use core::fmt::Write as _;
    let mut out = String::from(ITX_CENSUS_HEADER);
    for row in 0..4 {
        for i in 0..ITX_SIZES {
            let n = ITX_SHAPES[row][i].load(std::sync::atomic::Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            let (w, h) = match TxfmSize::from_repr(i) {
                Some(t) => t.to_wh(),
                None => continue,
            };
            let _ = writeln!(
                out,
                "{}\t{}\t{}x{}\t{}\t{}",
                if row >= 2 { "16bpc" } else { "8bpc" },
                if row % 2 == 1 { "simd" } else { "SCALAR" },
                w,
                h,
                n,
                n * (w * h) as u64,
            );
        }
    }
    out
}

/// Header only: the counters do not exist without `__ablate`.
#[cfg(not(feature = "__ablate"))]
pub fn itx_shape_report() -> String {
    String::from(ITX_CENSUS_HEADER)
}

const ITX_CENSUS_HEADER: &str = "depth\tpath\tshape\tcalls\tcoeff_area\n";

/// Zero the census (call between vectors to get per-vector numbers).
pub fn itx_shape_reset() {
    #[cfg(feature = "__ablate")]
    for row in ITX_SHAPES.iter() {
        for slot in row.iter() {
            slot.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}
