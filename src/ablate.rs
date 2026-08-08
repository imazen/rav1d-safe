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
