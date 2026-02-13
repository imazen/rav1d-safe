#![deny(unsafe_op_in_unsafe_fn)]

use crate::include::common::attributes::clz;
use crate::include::common::intops::inv_recenter;
use crate::include::common::intops::ulog2;
use crate::src::c_arc::CArc;
use crate::src::cpu::CpuFlags;
use cfg_if::cfg_if;
use std::ffi::c_int;
use std::ffi::c_uint;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
#[cfg(asm_msac)]
use std::ops::Range;
#[cfg(asm_msac)]
use std::ptr;
#[cfg(asm_msac)]
use std::slice;

// SSE2 intrinsics for unchecked-only msac SIMD (not archmage — direct #[target_feature])
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
use core::arch::x86_64::*;
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
use safe_unaligned_simd::x86_64 as safe_simd;

/// Helper macros wrapping SSE2 intrinsic calls in `unsafe {}` blocks.
/// SSE2 is baseline on x86_64 (always available), so this is sound.
/// Using `unsafe` instead of `#[target_feature]` allows `#[inline(always)]`
/// and avoids the non-inlineable function boundary that `#[target_feature]` creates.
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
macro_rules! loadu_si64 {
    ($arr:expr) => {
        // SAFETY: SSE2 is baseline x86_64, always available
        unsafe { safe_simd::_mm_loadu_si64($arr) }
    };
}
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
macro_rules! loadu_si128 {
    ($arr:expr) => {
        // SAFETY: SSE2 is baseline x86_64, always available
        unsafe { safe_simd::_mm_loadu_si128($arr) }
    };
}
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
macro_rules! storeu_si128 {
    ($arr:expr, $v:expr) => {
        // SAFETY: SSE2 is baseline x86_64, always available
        unsafe { safe_simd::_mm_storeu_si128($arr, $v) }
    };
}
/// Wrap an SSE2 intrinsic call in unsafe. SSE2 is baseline x86_64.
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
macro_rules! sse2 {
    ($e:expr) => {
        // SAFETY: SSE2 is baseline x86_64, always available
        unsafe { $e }
    };
}

#[cfg(all(asm_msac, target_feature = "sse2"))]
extern "C" {
    fn dav1d_msac_decode_hi_tok_sse2(s: *mut MsacAsmContext, cdf: *mut u16) -> c_uint;
    fn dav1d_msac_decode_bool_sse2(s: *mut MsacAsmContext, f: c_uint) -> c_uint;
    fn dav1d_msac_decode_bool_equi_sse2(s: *mut MsacAsmContext) -> c_uint;
    fn dav1d_msac_decode_bool_adapt_sse2(s: *mut MsacAsmContext, cdf: *mut u16) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt16_sse2(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt8_sse2(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt4_sse2(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
}

#[cfg(all(asm_msac, target_arch = "x86_64"))]
extern "C" {
    fn dav1d_msac_decode_symbol_adapt16_avx2(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint;
}

#[cfg(all(asm_msac, target_feature = "neon"))]
extern "C" {
    fn dav1d_msac_decode_hi_tok_neon(s: *mut MsacAsmContext, cdf: *mut u16) -> c_uint;
    fn dav1d_msac_decode_bool_neon(s: *mut MsacAsmContext, f: c_uint) -> c_uint;
    fn dav1d_msac_decode_bool_equi_neon(s: *mut MsacAsmContext) -> c_uint;
    fn dav1d_msac_decode_bool_adapt_neon(s: *mut MsacAsmContext, cdf: *mut u16) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt16_neon(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt8_neon(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt4_neon(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
}

pub struct Rav1dMsacDSPContext {
    #[cfg(asm_msac)]
    symbol_adapt16: unsafe extern "C" fn(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint,
}

impl Rav1dMsacDSPContext {
    pub const fn default() -> Self {
        cfg_if! {
            if #[cfg(asm_msac)] {
                Self {
                    symbol_adapt16: rav1d_msac_decode_symbol_adapt_c,
                }
            } else {
                Self {}
            }
        }
    }

    #[cfg(all(asm_msac, any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline(always)]
    const fn init_x86(mut self, flags: CpuFlags) -> Self {
        if !flags.contains(CpuFlags::SSE2) {
            return self;
        }

        self.symbol_adapt16 = dav1d_msac_decode_symbol_adapt16_sse2;

        #[cfg(target_arch = "x86_64")]
        {
            if !flags.contains(CpuFlags::AVX2) {
                return self;
            }

            self.symbol_adapt16 = dav1d_msac_decode_symbol_adapt16_avx2;
        }

        self
    }

    #[cfg(all(asm_msac, any(target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    const fn init_arm(self, _flags: CpuFlags) -> Self {
        self
    }

    #[inline(always)]
    const fn init(self, flags: CpuFlags) -> Self {
        #[cfg(asm_msac)]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                return self.init_x86(flags);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            {
                return self.init_arm(flags);
            }
        }

        #[allow(unreachable_code)] // Reachable on some #[cfg]s.
        {
            let _ = flags;
            self
        }
    }

    pub const fn new(flags: CpuFlags) -> Self {
        Self::default().init(flags)
    }
}

impl Default for Rav1dMsacDSPContext {
    fn default() -> Self {
        Self::default()
    }
}

pub type EcWin = usize;

/// For asm builds: raw pointers into `MsacContext::data` for C FFI layout.
///
/// # Safety
///
/// [`Self`] must be the first field of [`MsacAsmContext`] for asm layout purposes,
/// and that [`MsacAsmContext`] must be a field of [`MsacContext`].
/// And [`Self::pos`] and [`Self::end`] must be either [`ptr::null`],
/// or [`Self::pos`] must point into (or the end of) [`MsacContext::data`],
/// and [`Self::end`] must point to the end of [`MsacContext::data`],
/// where [`MsacContext::data`] is part of the [`MsacContext`]
/// containing [`MsacAsmContext`] and thus also [`Self`].
#[cfg(asm_msac)]
#[repr(C)]
struct MsacAsmContextBuf {
    pos: *const u8,
    end: *const u8,
}

#[cfg(asm_msac)]
/// SAFETY: [`MsacAsmContextBuf`] is always contained in [`MsacAsmContext::buf`],
/// which is always contained in [`MsacContext::asm`], whose [`MsacContext::data`] field
/// is what is stored in [`MsacAsmContextBuf::pos`] and [`MsacAsmContextBuf::end`].
/// Since [`MsacContext::data`] is [`Send`], [`MsacAsmContextBuf`] is also [`Send`].
#[allow(unsafe_code)]
unsafe impl Send for MsacAsmContextBuf {}

#[cfg(asm_msac)]
/// SAFETY: [`MsacAsmContextBuf`] is always contained in [`MsacAsmContext::buf`],
/// which is always contained in [`MsacContext::asm`], whose [`MsacContext::data`] field
/// is what is stored in [`MsacAsmContextBuf::pos`] and [`MsacAsmContextBuf::end`].
/// Since [`MsacContext::data`] is [`Sync`], [`MsacAsmContextBuf`] is also [`Sync`].
#[allow(unsafe_code)]
unsafe impl Sync for MsacAsmContextBuf {}

#[cfg(asm_msac)]
impl Default for MsacAsmContextBuf {
    fn default() -> Self {
        Self {
            pos: ptr::null(),
            end: ptr::null(),
        }
    }
}

#[cfg(asm_msac)]
impl From<&[u8]> for MsacAsmContextBuf {
    fn from(value: &[u8]) -> Self {
        let Range { start, end } = value.as_ptr_range();
        Self { pos: start, end }
    }
}

/// For non-asm builds: byte indices into `MsacContext::data`.
///
/// Uses `usize` indices instead of raw pointers, so this type is
/// automatically `Send + Sync` without any unsafe impls.
#[cfg(not(asm_msac))]
struct MsacAsmContextBuf {
    /// Byte index of current position within `MsacContext::data`.
    pos: usize,
    /// Byte index of end (always `data.len()`).
    /// Maintained for layout parity with asm but not read in non-asm path.
    #[allow(dead_code)]
    end: usize,
}

#[cfg(not(asm_msac))]
impl Default for MsacAsmContextBuf {
    fn default() -> Self {
        Self { pos: 0, end: 0 }
    }
}

#[cfg_attr(asm_msac, repr(C))]
pub struct MsacAsmContext {
    buf: MsacAsmContextBuf,
    pub dif: EcWin,
    pub rng: c_uint,
    pub cnt: c_int,
    allow_update_cdf: c_int,
    #[cfg(all(asm_msac, target_arch = "x86_64"))]
    symbol_adapt16: unsafe extern "C" fn(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint,
}

impl Default for MsacAsmContext {
    fn default() -> Self {
        Self {
            buf: Default::default(),
            dif: Default::default(),
            rng: Default::default(),
            cnt: Default::default(),
            allow_update_cdf: Default::default(),

            #[cfg(all(asm_msac, target_arch = "x86_64"))]
            symbol_adapt16: Rav1dMsacDSPContext::default().symbol_adapt16,
        }
    }
}

impl MsacAsmContext {
    fn allow_update_cdf(&self) -> bool {
        self.allow_update_cdf != 0
    }
}

#[derive(Default)]
pub struct MsacContext {
    asm: MsacAsmContext,
    data: Option<CArc<[u8]>>,
}

impl Deref for MsacContext {
    type Target = MsacAsmContext;

    fn deref(&self) -> &Self::Target {
        &self.asm
    }
}

impl DerefMut for MsacContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.asm
    }
}

impl MsacContext {
    pub fn data(&self) -> &[u8] {
        &**self.data.as_ref().unwrap()
    }

    pub fn buf_index(&self) -> usize {
        cfg_if! {
            if #[cfg(asm_msac)] {
                // We safely subtract instead of unsafely use `ptr::offset_from`
                // as asm sets `buf_pos`, so we don't need to rely on its safety,
                // and because codegen is no less optimal this way.
                self.buf.pos as usize - self.data().as_ptr() as usize
            } else {
                // Without asm, pos is already a byte index.
                self.buf.pos
            }
        }
    }

    fn with_buf(&mut self, mut f: impl FnMut(&[u8]) -> &[u8]) {
        let data = &**self.data.as_ref().unwrap();
        let buf = &data[self.buf_index()..];
        let buf = f(buf);
        cfg_if! {
            if #[cfg(asm_msac)] {
                self.buf.pos = buf.as_ptr();
            } else {
                // Compute the new byte index from the returned sub-slice.
                self.buf.pos = buf.as_ptr() as usize - data.as_ptr() as usize;
            }
        }
        // We don't actually need to set `self.buf_end` since it has not changed.
    }
}

/// Return value uses `n` bits.
#[inline]
pub fn rav1d_msac_decode_bools(s: &mut MsacContext, n: u8) -> c_uint {
    let mut v = 0;
    for _ in 0..n {
        v = v << 1 | rav1d_msac_decode_bool_equi(s) as c_uint;
    }
    v
}

#[inline]
pub fn rav1d_msac_decode_uniform(s: &mut MsacContext, n: c_uint) -> c_int {
    assert!(n > 0);
    let l = ulog2(n) as u8 + 1;
    assert!(l > 1);
    let m = (1 << l) - n;
    let v = rav1d_msac_decode_bools(s, l - 1);
    (if v < m {
        v
    } else {
        (v << 1) - m + rav1d_msac_decode_bool_equi(s) as c_uint
    }) as c_int
}

const EC_PROB_SHIFT: c_uint = 6;
const EC_MIN_PROB: c_uint = 4;
const _: () = assert!(EC_MIN_PROB <= (1 << EC_PROB_SHIFT) / 16);

const EC_WIN_SIZE: usize = mem::size_of::<EcWin>() << 3;

/// Branchless CDF update after symbol decode.
///
/// For i < val: cdf[i] += (32768 - cdf[i]) >> rate (probability increases)
/// For i >= val: cdf[i] -= cdf[i] >> rate (probability decreases)
///
/// Uses mask-select to avoid branches on the val boundary.
#[inline(always)]
fn update_cdf(cdf: &mut [u16], n: usize, val: usize, rate: u16, count: u16) {
    for i in 0..n {
        let mask = ((i < val) as u16).wrapping_neg(); // 0xFFFF if below val, 0 otherwise
        let delta_up = (32768u16.wrapping_sub(cdf[i])) >> rate;
        let delta_dn = cdf[i] >> rate;
        // Apply increase (delta_up) if below val, decrease (delta_dn) if at/above val
        cdf[i] = cdf[i]
            .wrapping_add(delta_up & mask)
            .wrapping_sub(delta_dn & !mask);
    }
    cdf[n] = count + (count < 32) as u16;
}

#[inline]
#[cold]
fn ctx_refill(s: &mut MsacContext) {
    let mut c = (EC_WIN_SIZE as c_int) - 24 - s.cnt;
    let mut dif = s.dif;
    s.with_buf(|mut buf| {
        loop {
            if buf.is_empty() {
                // set remaining bits to 1;
                dif |= !(!(0xff as EcWin) << c);
                break;
            }
            dif |= ((buf[0] ^ 0xff) as EcWin) << c;
            buf = &buf[1..];
            c -= 8;
            if c < 0 {
                break;
            }
        }
        buf
    });
    s.dif = dif;
    s.cnt = (EC_WIN_SIZE as c_int) - 24 - c;
}

#[inline(always)]
fn ctx_norm(s: &mut MsacContext, dif: EcWin, rng: c_uint) {
    let d = 15 ^ (31 ^ clz(rng));
    let cnt = s.cnt;
    debug_assert!(rng <= 65535);
    s.dif = dif << d;
    s.rng = rng << d;
    s.cnt = cnt - d;
    // unsigned compare avoids redundant refills at eob
    if (cnt as u32) < (d as u32) {
        ctx_refill(s);
    }
}

#[inline(always)]
#[cfg_attr(
    all(asm_msac, any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_bool_equi_rust(s: &mut MsacContext) -> bool {
    let r = s.rng;
    let mut dif = s.dif;
    debug_assert!(dif >> (EC_WIN_SIZE - 16) < r as EcWin);
    let mut v = (r >> 8 << 7) + EC_MIN_PROB;
    let vw = (v as EcWin) << (EC_WIN_SIZE - 16);
    let ret = dif >= vw;
    dif -= (ret as EcWin) * vw;
    v = v.wrapping_add((ret as c_uint) * (r.wrapping_sub(2 * v)));
    ctx_norm(s, dif, v);
    !ret
}

#[inline(always)]
#[cfg_attr(
    all(asm_msac, any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_bool_rust(s: &mut MsacContext, f: c_uint) -> bool {
    let r = s.rng;
    let mut dif = s.dif;
    debug_assert!(dif >> (EC_WIN_SIZE - 16) < r as EcWin);
    let mut v = ((r >> 8) * (f >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB;
    let vw = (v as EcWin) << (EC_WIN_SIZE - 16);
    let ret = dif >= vw;
    dif -= (ret as EcWin) * vw;
    v = v.wrapping_add((ret as c_uint) * (r.wrapping_sub(2 * v)));
    ctx_norm(s, dif, v);
    !ret
}

pub fn rav1d_msac_decode_subexp(s: &mut MsacContext, r#ref: c_uint, n: c_uint, mut k: u8) -> c_int {
    assert!(n >> k == 8);
    let mut a = 0;
    if rav1d_msac_decode_bool_equi(s) {
        if rav1d_msac_decode_bool_equi(s) {
            k += rav1d_msac_decode_bool_equi(s) as u8 + 1;
        }
        a = 1 << k;
    }
    let v = rav1d_msac_decode_bools(s, k) + a;
    (if r#ref * 2 <= n {
        inv_recenter(r#ref, v)
    } else {
        n - 1 - inv_recenter(n - 1 - r#ref, v)
    }) as c_int
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..16`, so it is really a `u4`.
fn rav1d_msac_decode_symbol_adapt_rust(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    let c = (s.dif >> (EC_WIN_SIZE - 16)) as c_uint;
    let r = s.rng >> 8;
    let mut u;
    let mut v = s.rng;
    let mut val = 0;
    debug_assert!(n_symbols < 16);
    debug_assert!(cdf[n_symbols as usize] <= 32);
    loop {
        u = v;
        v = r * ((cdf[val as usize] >> EC_PROB_SHIFT) as c_uint);
        v >>= 7 - EC_PROB_SHIFT;
        v += EC_MIN_PROB * ((n_symbols as c_uint) - val);
        if !(c < v) {
            break;
        }
        val += 1;
    }
    debug_assert!(u <= s.rng);
    ctx_norm(
        s,
        s.dif.wrapping_sub((v as EcWin) << (EC_WIN_SIZE - 16)),
        u - v,
    );
    if s.allow_update_cdf() {
        let n_usize = n_symbols as usize;
        let count = cdf[n_usize];
        let rate = 4 + (count >> 4) + (n_symbols > 2) as u16;
        let val = val as usize;
        update_cdf(cdf, n_usize, val, rate, count);
    }
    debug_assert!(val <= n_symbols as _);
    val as u8
}

/// # Safety
///
/// Must be called through [`Rav1dMsacDSPContext::symbol_adapt16`]
/// in [`rav1d_msac_decode_symbol_adapt16`].
#[cfg(asm_msac)]
#[cfg_attr(not(all(asm_msac, target_arch = "x86_64")), allow(dead_code))]
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn rav1d_msac_decode_symbol_adapt_c(
    s: &mut MsacAsmContext,
    cdf: *mut u16,
    n_symbols: usize,
    cdf_len: usize,
) -> c_uint {
    // SAFETY: In the `rav1d_msac_decode_symbol_adapt16` caller,
    // `&mut s.asm` is passed, so we can reverse this to get back `s`.
    // The `.sub` is safe since were are subtracting the offset of `asm` within `s`,
    // so that will stay in bounds of the `s: MsacContext` allocated object.
    let s = unsafe {
        &mut *ptr::from_mut(s)
            .sub(mem::offset_of!(MsacContext, asm))
            .cast::<MsacContext>()
    };

    // SAFETY: This is only called from [`dav1d_msac_decode_symbol_adapt16`],
    // where it comes from `cdf.len()`.
    let cdf = unsafe { slice::from_raw_parts_mut(cdf, cdf_len) };

    rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols as u8) as c_uint
}

#[inline(always)]
#[cfg_attr(
    all(asm_msac, any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_bool_adapt_rust(s: &mut MsacContext, cdf: &mut [u16; 2]) -> bool {
    let bit = rav1d_msac_decode_bool(s, cdf[0] as c_uint);
    if s.allow_update_cdf() {
        let count = cdf[1];
        let rate = 4 + (count >> 4);
        update_cdf(cdf, 1, bit as usize, rate, count);
    }
    bit
}

/// Return value is in the range `0..=15`.
#[inline(always)]
#[cfg_attr(
    any(
        all(asm_msac, any(target_feature = "sse2", target_feature = "neon")),
        all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"),
    ),
    allow(dead_code)
)]
fn rav1d_msac_decode_hi_tok_rust(s: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
    let mut tok_br = rav1d_msac_decode_symbol_adapt4(s, cdf, 3);
    let mut tok = 3 + tok_br;
    if tok_br == 3 {
        tok_br = rav1d_msac_decode_symbol_adapt4(s, cdf, 3);
        tok = 6 + tok_br;
        if tok_br == 3 {
            tok_br = rav1d_msac_decode_symbol_adapt4(s, cdf, 3);
            tok = 9 + tok_br;
            if tok_br == 3 {
                tok = 12 + rav1d_msac_decode_symbol_adapt4(s, cdf, 3);
            }
        }
    }
    tok
}

// ============================================================================
// SSE2 implementations (unchecked-only, x86_64)
//
// SSE2 intrinsics called via sse2!() macro wrapping each call in unsafe{}.
// This avoids #[target_feature] (which prevents #[inline(always)]) and
// archmage #[arcane]/#[rite] (which created non-inlineable boundaries).
// SSE2 is baseline x86_64 — always available, so the unsafe is sound.
//
// Gated behind `unchecked` feature since intrinsic calls require unsafe.
// Default build keeps branchless scalar under forbid(unsafe_code).
// ============================================================================

/// min_prob table: EC_MIN_PROB * (n - i) for symbol_adapt functions.
/// Padded to 31 entries so any offset in 0..16 can safely load 16 values via SIMD.
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
static MIN_PROB_16: [u16; 31] = [
    60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0, // valid entries
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // padding
];

/// SSE2 implementation of symbol_adapt4 (n_symbols <= 3).
///
/// Uses SSE2 SIMD for parallel CDF probability computation, comparison,
/// and CDF update. Fully inlines into callers — no `#[target_feature]` boundary.
/// SSE2 intrinsics are called via `sse2!()` macro (safe: SSE2 is baseline x86_64).
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
#[allow(unsafe_code)]
#[inline(always)]
fn adapt4_sse2(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols > 0 && n_symbols <= 3);
    let c = (s.dif >> (EC_WIN_SIZE - 16)) as u16;
    let n = n_symbols as usize;

    let cdf_arr: &[u16; 4] = cdf[..4].try_into().unwrap();
    let cdf_vec = loadu_si64!(cdf_arr);

    let rng_masked = (s.rng & 0xff00) as i16;
    let rng_vec = sse2!(_mm_set1_epi16(rng_masked));

    let cdf_shifted = sse2!(_mm_slli_epi16::<7>(_mm_srli_epi16::<6>(cdf_vec)));
    let prod = sse2!(_mm_mulhi_epu16(cdf_shifted, rng_vec));

    let min_prob_offset = 15 - n;
    let min_prob_arr: &[u16; 4] = MIN_PROB_16[min_prob_offset..min_prob_offset + 4]
        .try_into()
        .unwrap();
    let min_prob = loadu_si64!(min_prob_arr);

    let v = sse2!(_mm_add_epi16(prod, min_prob));

    let mut v_arr = [0u16; 8];
    storeu_si128!(&mut v_arr, v);

    let c_vec = sse2!(_mm_set1_epi16(c as i16));
    let zero = sse2!(_mm_setzero_si128());
    let sub = sse2!(_mm_subs_epu16(v, c_vec));
    let cmp = sse2!(_mm_cmpeq_epi16(sub, zero));

    let mask = sse2!(_mm_movemask_epi8(cmp)) as u32;
    let val = std::cmp::min((mask.trailing_zeros() >> 1) as u8, n_symbols);

    let u = if val == 0 {
        s.rng
    } else {
        v_arr[val as usize - 1] as u32
    };
    let v_val = if val >= n_symbols {
        0u32
    } else {
        v_arr[val as usize] as u32
    };

    ctx_norm(
        s,
        s.dif.wrapping_sub((v_val as EcWin) << (EC_WIN_SIZE - 16)),
        u - v_val,
    );

    if s.allow_update_cdf() {
        let count = cdf[n];
        let rate_val = 4 + (count >> 4) + (n_symbols > 2) as u16;

        let all_ones = sse2!(_mm_cmpeq_epi16(zero, zero));
        let avg = sse2!(_mm_avg_epu16(all_ones, cmp));
        let delta = sse2!(_mm_sub_epi16(avg, cdf_vec));
        let base = sse2!(_mm_sub_epi16(cdf_vec, cmp));
        let rate_vec = sse2!(_mm_cvtsi32_si128(rate_val as i32));
        let shifted = sse2!(_mm_sra_epi16(delta, rate_vec));
        let updated = sse2!(_mm_add_epi16(base, shifted));

        let mut cdf_out = [0u16; 8];
        storeu_si128!(&mut cdf_out, updated);
        cdf[0] = cdf_out[0];
        cdf[1] = cdf_out[1];
        cdf[2] = cdf_out[2];
        cdf[n] = count + (count < 32) as u16;
    }

    val
}

/// SSE2 implementation of symbol_adapt8 (n_symbols <= 7).
///
/// Same strategy as adapt4 but handles up to 7 symbols with 128-bit loads.
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
#[allow(unsafe_code)]
#[inline(always)]
fn adapt8_sse2(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols > 0 && n_symbols <= 7);
    let c = (s.dif >> (EC_WIN_SIZE - 16)) as u16;
    let n = n_symbols as usize;

    let cdf_arr: &[u16; 8] = cdf[..8].try_into().unwrap();
    let cdf_vec = loadu_si128!(cdf_arr);

    let rng_masked = (s.rng & 0xff00) as i16;
    let rng_vec = sse2!(_mm_set1_epi16(rng_masked));

    let cdf_shifted = sse2!(_mm_slli_epi16::<7>(_mm_srli_epi16::<6>(cdf_vec)));
    let prod = sse2!(_mm_mulhi_epu16(cdf_shifted, rng_vec));

    let min_prob_offset = 15 - n;
    let min_prob_arr: &[u16; 8] = MIN_PROB_16[min_prob_offset..min_prob_offset + 8]
        .try_into()
        .unwrap();
    let min_prob = loadu_si128!(min_prob_arr);

    let v = sse2!(_mm_add_epi16(prod, min_prob));

    let mut v_arr = [0u16; 8];
    storeu_si128!(&mut v_arr, v);

    let c_vec = sse2!(_mm_set1_epi16(c as i16));
    let zero = sse2!(_mm_setzero_si128());
    let sub = sse2!(_mm_subs_epu16(v, c_vec));
    let cmp = sse2!(_mm_cmpeq_epi16(sub, zero));

    let mask = sse2!(_mm_movemask_epi8(cmp)) as u32;
    let val = std::cmp::min((mask.trailing_zeros() >> 1) as u8, n_symbols);

    let u = if val == 0 {
        s.rng
    } else {
        v_arr[val as usize - 1] as u32
    };
    let v_val = if val >= n_symbols {
        0u32
    } else {
        v_arr[val as usize] as u32
    };

    ctx_norm(
        s,
        s.dif.wrapping_sub((v_val as EcWin) << (EC_WIN_SIZE - 16)),
        u - v_val,
    );

    if s.allow_update_cdf() {
        let count = cdf[n];
        let rate_val = 4 + (count >> 4) + (n_symbols > 2) as u16;

        let all_ones = sse2!(_mm_cmpeq_epi16(zero, zero));
        let avg = sse2!(_mm_avg_epu16(all_ones, cmp));
        let delta = sse2!(_mm_sub_epi16(avg, cdf_vec));
        let base = sse2!(_mm_sub_epi16(cdf_vec, cmp));
        let rate_vec = sse2!(_mm_cvtsi32_si128(rate_val as i32));
        let shifted = sse2!(_mm_sra_epi16(delta, rate_vec));
        let updated = sse2!(_mm_add_epi16(base, shifted));

        let mut cdf_out = [0u16; 8];
        storeu_si128!(&mut cdf_out, updated);
        for i in 0..n {
            cdf[i] = cdf_out[i];
        }
        cdf[n] = count + (count < 32) as u16;
    }

    val
}

/// SSE2 implementation of hi_tok.
///
/// Inlines adapt4 + CDF update + renorm for all 4 iterations,
/// avoiding per-call overhead. Fully inlined (no `#[target_feature]` boundary).
#[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))]
#[allow(unsafe_code)]
#[inline(always)]
fn hi_tok_sse2(s: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
    const N: usize = 3;

    let min_prob_arr: &[u16; 4] = MIN_PROB_16[12..16].try_into().unwrap();
    let min_prob = loadu_si64!(min_prob_arr);

    let update = s.allow_update_cdf();
    let mut tok_br;
    let mut tok: u8 = 0;

    for iter in 0..4u8 {
        let c = (s.dif >> (EC_WIN_SIZE - 16)) as u16;

        let cdf_load: &[u16; 4] = cdf[..4].try_into().unwrap();
        let cdf_vec = loadu_si64!(cdf_load);

        let rng_masked = (s.rng & 0xff00) as i16;
        let rng_vec = sse2!(_mm_set1_epi16(rng_masked));

        let cdf_shifted = sse2!(_mm_slli_epi16::<7>(_mm_srli_epi16::<6>(cdf_vec)));
        let prod = sse2!(_mm_mulhi_epu16(cdf_shifted, rng_vec));
        let v = sse2!(_mm_add_epi16(prod, min_prob));

        let mut v_arr = [0u16; 8];
        storeu_si128!(&mut v_arr, v);

        let c_vec = sse2!(_mm_set1_epi16(c as i16));
        let zero = sse2!(_mm_setzero_si128());
        let sub = sse2!(_mm_subs_epu16(v, c_vec));
        let cmp = sse2!(_mm_cmpeq_epi16(sub, zero));
        let mask = sse2!(_mm_movemask_epi8(cmp)) as u32;
        tok_br = std::cmp::min((mask.trailing_zeros() >> 1) as u8, N as u8);

        let u = if tok_br == 0 {
            s.rng
        } else {
            v_arr[tok_br as usize - 1] as u32
        };
        let v_val = if tok_br >= N as u8 {
            0u32
        } else {
            v_arr[tok_br as usize] as u32
        };

        if update {
            let count = cdf[N];
            let rate_val = 4 + (count >> 4) + 1;

            let all_ones = sse2!(_mm_cmpeq_epi16(zero, zero));
            let avg = sse2!(_mm_avg_epu16(all_ones, cmp));
            let delta = sse2!(_mm_sub_epi16(avg, cdf_vec));
            let base = sse2!(_mm_sub_epi16(cdf_vec, cmp));
            let rate_vec = sse2!(_mm_cvtsi32_si128(rate_val as i32));
            let shifted = sse2!(_mm_sra_epi16(delta, rate_vec));
            let updated = sse2!(_mm_add_epi16(base, shifted));

            let mut cdf_out = [0u16; 8];
            storeu_si128!(&mut cdf_out, updated);
            cdf[0] = cdf_out[0];
            cdf[1] = cdf_out[1];
            cdf[2] = cdf_out[2];
            cdf[N] = count + (count < 32) as u16;
        }

        let new_rng = u - v_val;
        let d = 15 ^ (31 ^ clz(new_rng));
        let new_dif = s.dif.wrapping_sub((v_val as EcWin) << (EC_WIN_SIZE - 16));
        let cnt = s.cnt;
        s.dif = new_dif << d;
        s.rng = new_rng << d;
        s.cnt = cnt - d;

        if (cnt as u32) < (d as u32) {
            ctx_refill(s);
        }

        tok = iter * 3 + 3 + tok_br;
        if tok_br < 3 {
            break;
        }
    }

    tok
}

// ============================================================================
// Branchless scalar implementations (used when asm and SSE2 are both disabled)
// ============================================================================

/// Branchless implementation of symbol_adapt for n_symbols <= 3 (adapt4).
///
/// Eliminates branch misprediction from the serial comparison loop by
/// computing all v values and counting matches branchlessly.
#[cfg(all(not(asm_msac), not(all(target_arch = "x86_64", feature = "unchecked"))))]
#[inline(always)]
fn rav1d_msac_decode_symbol_adapt4_branchless(
    s: &mut MsacContext,
    cdf: &mut [u16],
    n_symbols: u8,
) -> u8 {
    debug_assert!(n_symbols > 0 && n_symbols <= 3);
    let c = (s.dif >> (EC_WIN_SIZE - 16)) as c_uint;
    let r = s.rng >> 8;
    let n = n_symbols as c_uint;

    // Compute all v values (at most 3 for adapt4)
    // NB: >> has lower precedence than + in Rust, so parens are required
    let v0 = (r * ((cdf[0] >> EC_PROB_SHIFT) as c_uint) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * n;
    let v1 = if n > 1 {
        (r * ((cdf[1] >> EC_PROB_SHIFT) as c_uint) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - 1)
    } else {
        0
    };
    let v2 = if n > 2 {
        (r * ((cdf[2] >> EC_PROB_SHIFT) as c_uint) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - 2)
    } else {
        0
    };

    // Branchless: count how many v[i] have c < v[i]
    // Since v is monotonically decreasing, this gives the first index where c >= v
    let val = (c < v0) as u32 + (c < v1) as u32 + (c < v2) as u32;
    debug_assert!(val <= n);

    // v_arr[0..3] for indexed access, with sentinel for boundary
    let v_arr = [v0, v1, v2, 0];
    let u = if val == 0 {
        s.rng
    } else {
        v_arr[val as usize - 1]
    };
    let v_val = v_arr[val as usize];

    ctx_norm(
        s,
        s.dif.wrapping_sub((v_val as EcWin) << (EC_WIN_SIZE - 16)),
        u - v_val,
    );

    if s.allow_update_cdf() {
        let n_usize = n_symbols as usize;
        let count = cdf[n_usize];
        let rate = 4 + (count >> 4) + (n_symbols > 2) as u16;
        update_cdf(cdf, n_usize, val as usize, rate, count);
    }

    val as u8
}

/// Branchless implementation of symbol_adapt for n_symbols <= 7 (adapt8).
///
/// Same strategy as adapt4 but handles up to 7 symbols.
#[cfg(all(not(asm_msac), not(all(target_arch = "x86_64", feature = "unchecked"))))]
#[inline(always)]
fn rav1d_msac_decode_symbol_adapt8_branchless(
    s: &mut MsacContext,
    cdf: &mut [u16],
    n_symbols: u8,
) -> u8 {
    debug_assert!(n_symbols > 0 && n_symbols <= 7);
    let c = (s.dif >> (EC_WIN_SIZE - 16)) as c_uint;
    let r = s.rng >> 8;
    let n = n_symbols as c_uint;

    // Compute v[i] = r * (cdf[i] >> 6) >> 1 + 4 * (n - i)
    // Use a fixed-size array, compute only valid entries
    // NB: >> has lower precedence than + in Rust, so parens are required
    let mut v = [0u32; 8]; // v[7] stays 0 as sentinel
    for i in 0..n_symbols as usize {
        v[i] = (r * ((cdf[i] >> EC_PROB_SHIFT) as c_uint) >> (7 - EC_PROB_SHIFT))
            + EC_MIN_PROB * (n - i as c_uint);
    }

    // Branchless: count how many v[i] have c < v[i]
    let mut val = 0u32;
    for i in 0..n_symbols as usize {
        val += (c < v[i]) as u32;
    }
    debug_assert!(val <= n);

    let u = if val == 0 { s.rng } else { v[val as usize - 1] };
    let v_val = v[val as usize];

    ctx_norm(
        s,
        s.dif.wrapping_sub((v_val as EcWin) << (EC_WIN_SIZE - 16)),
        u - v_val,
    );

    if s.allow_update_cdf() {
        let n_usize = n_symbols as usize;
        let count = cdf[n_usize];
        let rate = 4 + (count >> 4) + (n_symbols > 2) as u16;
        update_cdf(cdf, n_usize, val as usize, rate, count);
    }

    val as u8
}

impl MsacContext {
    pub fn new(data: CArc<[u8]>, disable_cdf_update_flag: bool, dsp: &Rav1dMsacDSPContext) -> Self {
        let buf = {
            cfg_if! {
                if #[cfg(asm_msac)] {
                    MsacAsmContextBuf::from(data.as_ref())
                } else {
                    MsacAsmContextBuf { pos: 0, end: data.as_ref().len() }
                }
            }
        };
        let asm = MsacAsmContext {
            buf,
            dif: 0,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: (!disable_cdf_update_flag).into(),
            #[cfg(all(asm_msac, target_arch = "x86_64"))]
            symbol_adapt16: dsp.symbol_adapt16,
        };
        let mut s = Self {
            asm,
            data: Some(data),
        };
        let _ = dsp; // Silence unused warnings when asm is off.
        ctx_refill(&mut s);
        s
    }
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..4`.
#[inline(always)]
pub fn rav1d_msac_decode_symbol_adapt4(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 4);
    let ret;
    cfg_if! {
        if #[cfg(all(asm_msac, target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt4_sse2(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(all(asm_msac, target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt4_neon(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))] {
            ret = adapt4_sse2(s, cdf, n_symbols);
        } else if #[cfg(not(asm_msac))] {
            ret = rav1d_msac_decode_symbol_adapt4_branchless(s, cdf, n_symbols);
        } else {
            ret = rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols);
        }
    }
    debug_assert!(ret < 4);
    ret as u8 % 4
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..8`.
#[inline(always)]
pub fn rav1d_msac_decode_symbol_adapt8(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 8);
    let ret;
    cfg_if! {
        if #[cfg(all(asm_msac, target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt8_sse2(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(all(asm_msac, target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt8_neon(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))] {
            ret = adapt8_sse2(s, cdf, n_symbols);
        } else if #[cfg(not(asm_msac))] {
            ret = rav1d_msac_decode_symbol_adapt8_branchless(s, cdf, n_symbols);
        } else {
            ret = rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols);
        }
    }
    debug_assert!(ret < 8);
    ret as u8 % 8
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..16`.
#[inline(always)]
#[cfg_attr(asm_msac, allow(unsafe_code))]
pub fn rav1d_msac_decode_symbol_adapt16(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 16);
    let ret;
    cfg_if! {
        if #[cfg(all(asm_msac, target_arch = "x86_64"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                (s.symbol_adapt16)(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize, cdf.len())
            };
        } else if #[cfg(all(asm_msac, target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt16_sse2(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize, cdf.len())
            };
        } else if #[cfg(all(asm_msac, target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt16_neon(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(not(asm_msac))] {
            // Serial loop is faster than branchless for adapt16: typical AV1 distributions
            // exit early (3-5 iterations), while branchless always computes all n_symbols values.
            ret = rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols) as c_uint;
        } else {
            ret = rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols) as c_uint;
        }
    }
    debug_assert!(ret < 16);
    ret as u8 % 16
}

#[inline(always)]
pub fn rav1d_msac_decode_bool_adapt(s: &mut MsacContext, cdf: &mut [u16; 2]) -> bool {
    cfg_if! {
        if #[cfg(all(asm_msac, target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_adapt_rust`].
            unsafe {
                dav1d_msac_decode_bool_adapt_sse2(&mut s.asm, cdf.as_mut_ptr()) != 0
            }
        } else if #[cfg(all(asm_msac, target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_adapt_rust`].
            unsafe {
                dav1d_msac_decode_bool_adapt_neon(&mut s.asm, cdf.as_mut_ptr()) != 0
            }
        } else {
            rav1d_msac_decode_bool_adapt_rust(s, cdf)
        }
    }
}

#[inline(always)]
pub fn rav1d_msac_decode_bool_equi(s: &mut MsacContext) -> bool {
    cfg_if! {
        if #[cfg(all(asm_msac, target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_equi_rust`].
            unsafe {
                dav1d_msac_decode_bool_equi_sse2(&mut s.asm) != 0
            }
        } else if #[cfg(all(asm_msac, target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_equi_rust`].
            unsafe {
                dav1d_msac_decode_bool_equi_neon(&mut s.asm) != 0
            }
        } else {
            rav1d_msac_decode_bool_equi_rust(s)
        }
    }
}

#[inline(always)]
pub fn rav1d_msac_decode_bool(s: &mut MsacContext, f: c_uint) -> bool {
    cfg_if! {
        if #[cfg(all(asm_msac, target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_rust`].
            unsafe {
                dav1d_msac_decode_bool_sse2(&mut s.asm, f) != 0
            }
        } else if #[cfg(all(asm_msac, target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_rust`].
            unsafe {
                dav1d_msac_decode_bool_neon(&mut s.asm, f) != 0
            }
        } else {
            rav1d_msac_decode_bool_rust(s, f)
        }
    }
}

/// Return value is in the range `0..16`.
#[inline(always)]
pub fn rav1d_msac_decode_hi_tok(s: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
    let ret;
    cfg_if! {
        if #[cfg(all(asm_msac, target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_hi_tok_rust`].
            ret = (unsafe {
                dav1d_msac_decode_hi_tok_sse2(&mut s.asm, cdf.as_mut_ptr())
            }) as u8;
        } else if #[cfg(all(asm_msac, target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_hi_tok_rust`].
            ret = unsafe {
                dav1d_msac_decode_hi_tok_neon(&mut s.asm, cdf.as_mut_ptr())
            } as u8;
        } else if #[cfg(all(not(asm_msac), target_arch = "x86_64", feature = "unchecked"))] {
            ret = hi_tok_sse2(s, cdf);
        } else if #[cfg(not(asm_msac))] {
            ret = rav1d_msac_decode_hi_tok_rust(s, cdf);
        }
    }
    debug_assert!(ret < 16);
    ret % 16
}
