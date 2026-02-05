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
use std::ops::Range;
use std::ptr;
use std::slice;

// x86_64 SIMD intrinsics for safe_simd implementations
#[cfg(all(not(feature = "asm"), target_arch = "x86_64"))]
use std::arch::x86_64::*;

// aarch64 SIMD intrinsics for safe_simd implementations
#[cfg(all(not(feature = "asm"), target_arch = "aarch64"))]
use std::arch::aarch64::*;

#[cfg(all(feature = "asm", target_feature = "sse2"))]
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
extern "C" {
    fn dav1d_msac_decode_symbol_adapt16_avx2(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint;
}

#[cfg(all(feature = "asm", target_feature = "neon"))]
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
    symbol_adapt16: unsafe extern "C" fn(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint,
}

impl Rav1dMsacDSPContext {
    pub const fn default() -> Self {
        Self {
            symbol_adapt16: rav1d_msac_decode_symbol_adapt_c,
        }
    }

    #[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
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

    #[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    const fn init_arm(self, _flags: CpuFlags) -> Self {
        self
    }

    #[inline(always)]
    const fn init(self, flags: CpuFlags) -> Self {
        #[cfg(feature = "asm")]
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

/// # Safety
///
/// [`Self`] must be the first field of [`MsacAsmContext`] for asm layout purposes,
/// and that [`MsacAsmContext`] must be a field of [`MsacContext`].
/// And [`Self::pos`] and [`Self::end`] must be either [`ptr::null`],
/// or [`Self::pos`] must point into (or the end of) [`MsacContext::data`],
/// and [`Self::end`] must point to the end of [`MsacContext::data`],
/// where [`MsacContext::data`] is part of the [`MsacContext`]
/// containing [`MsacAsmContext`] and thus also [`Self`].
#[repr(C)]
struct MsacAsmContextBuf {
    pos: *const u8,
    end: *const u8,
}

/// SAFETY: [`MsacAsmContextBuf`] is always contained in [`MsacAsmContext::buf`],
/// which is always contained in [`MsacContext::asm`], whose [`MsacContext::data`] field
/// is what is stored in [`MsacAsmContextBuf::pos`] and [`MsacAsmContextBuf::end`].
/// Since [`MsacContext::data`] is [`Send`], [`MsacAsmContextBuf`] is also [`Send`].
unsafe impl Send for MsacAsmContextBuf {}

/// SAFETY: [`MsacAsmContextBuf`] is always contained in [`MsacAsmContext::buf`],
/// which is always contained in [`MsacContext::asm`], whose [`MsacContext::data`] field
/// is what is stored in [`MsacAsmContextBuf::pos`] and [`MsacAsmContextBuf::end`].
/// Since [`MsacContext::data`] is [`Sync`], [`MsacAsmContextBuf`] is also [`Sync`].
unsafe impl Sync for MsacAsmContextBuf {}

impl Default for MsacAsmContextBuf {
    fn default() -> Self {
        Self {
            pos: ptr::null(),
            end: ptr::null(),
        }
    }
}

impl From<&[u8]> for MsacAsmContextBuf {
    fn from(value: &[u8]) -> Self {
        let Range { start, end } = value.as_ptr_range();
        Self { pos: start, end }
    }
}

#[repr(C)]
pub struct MsacAsmContext {
    buf: MsacAsmContextBuf,
    pub dif: EcWin,
    pub rng: c_uint,
    pub cnt: c_int,
    allow_update_cdf: c_int,
    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
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

            #[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
        // We safely subtract instead of unsafely use `ptr::offset_from`
        // as asm sets `buf_pos`, so we don't need to rely on its safety,
        // and because codegen is no less optimal this way.
        self.buf.pos as usize - self.data().as_ptr() as usize
    }

    fn with_buf(&mut self, mut f: impl FnMut(&[u8]) -> &[u8]) {
        let data = &**self.data.as_ref().unwrap();
        let buf = &data[self.buf_index()..];
        let buf = f(buf);
        self.buf.pos = buf.as_ptr();
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

#[inline]
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

#[inline]
fn ctx_norm(s: &mut MsacContext, dif: EcWin, rng: c_uint) {
    let d = 15 ^ (31 ^ clz(rng));
    let cnt = s.cnt;
    assert!(rng <= 65535);
    s.dif = dif << d;
    s.rng = rng << d;
    s.cnt = cnt - d;
    // unsigned compare avoids redundant refills at eob
    if (cnt as u32) < (d as u32) {
        ctx_refill(s);
    }
}

#[cfg_attr(
    all(feature = "asm", any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_bool_equi_rust(s: &mut MsacContext) -> bool {
    let r = s.rng;
    let mut dif = s.dif;
    assert!(dif >> (EC_WIN_SIZE - 16) < r as EcWin);
    let mut v = (r >> 8 << 7) + EC_MIN_PROB;
    let vw = (v as EcWin) << (EC_WIN_SIZE - 16);
    let ret = dif >= vw;
    dif -= (ret as EcWin) * vw;
    v = v.wrapping_add((ret as c_uint) * (r.wrapping_sub(2 * v)));
    ctx_norm(s, dif, v);
    !ret
}

#[cfg_attr(
    all(feature = "asm", any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_bool_rust(s: &mut MsacContext, f: c_uint) -> bool {
    let r = s.rng;
    let mut dif = s.dif;
    assert!(dif >> (EC_WIN_SIZE - 16) < r as EcWin);
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
    assert!(n_symbols < 16);
    assert!(cdf[n_symbols as usize] <= 32);
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
    assert!(u <= s.rng);
    ctx_norm(
        s,
        s.dif.wrapping_sub((v as EcWin) << (EC_WIN_SIZE - 16)),
        u - v,
    );
    if s.allow_update_cdf() {
        let count = cdf[n_symbols as usize];
        let rate = 4 + (count >> 4) + (n_symbols > 2) as u16;
        let val = val as usize;
        for cdf in &mut cdf[..val] {
            *cdf += (1 << 15) - *cdf >> rate;
        }
        for cdf in &mut cdf[val..n_symbols as usize] {
            *cdf -= *cdf >> rate;
        }
        cdf[n_symbols as usize] = count + (count < 32) as u16;
    }
    debug_assert!(val <= n_symbols as _);
    val as u8
}

/// # Safety
///
/// Must be called through [`Rav1dMsacDSPContext::symbol_adapt16`]
/// in [`rav1d_msac_decode_symbol_adapt16`].
#[cfg_attr(not(all(feature = "asm", target_arch = "x86_64")), allow(dead_code))]
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

#[cfg_attr(
    all(feature = "asm", any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_bool_adapt_rust(s: &mut MsacContext, cdf: &mut [u16; 2]) -> bool {
    let bit = rav1d_msac_decode_bool(s, cdf[0] as c_uint);
    if s.allow_update_cdf() {
        let count = cdf[1];
        let rate = 4 + (count >> 4);
        if bit {
            cdf[0] += (1 << 15) - cdf[0] >> rate;
        } else {
            cdf[0] -= cdf[0] >> rate;
        }
        cdf[1] = count + (count < 32) as u16;
    }
    bit
}

/// Return value is in the range `0..=15`.
#[cfg_attr(
    all(feature = "asm", any(target_feature = "sse2", target_feature = "neon")),
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
// Safe SIMD implementations (used when asm feature is disabled)
// ============================================================================

/// min_prob table: EC_MIN_PROB * (n - i - 1) for symbol_adapt functions
#[cfg(all(not(feature = "asm"), target_arch = "x86_64"))]
static MIN_PROB_16: [u16; 16] = [60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0];

/// AVX2 implementation of symbol_adapt16
/// Uses parallel CDF probability calculation and comparison
#[cfg(all(not(feature = "asm"), target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn rav1d_msac_decode_symbol_adapt16_avx2(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    let c = (s.dif >> (EC_WIN_SIZE - 16)) as u32;
    let n = n_symbols as usize;

    // Load CDF values (16 values = 256 bits)
    let cdf_vec = unsafe { _mm256_loadu_si256(cdf.as_ptr() as *const __m256i) };

    // Broadcast rng masked with 0xff00
    let rng_masked = (s.rng & 0xff00) as i16;
    let rng_vec = unsafe { _mm256_set1_epi16(rng_masked) };

    // Calculate (cdf >> 6) << 7 then pmulhuw
    // This computes: ((cdf >> 6) * (rng >> 8)) >> 1
    let cdf_shifted = unsafe { _mm256_slli_epi16(_mm256_srli_epi16(cdf_vec, 6), 7) };
    let prod = unsafe { _mm256_mulhi_epu16(cdf_shifted, rng_vec) };

    // Load min_prob values offset by (15 - n_symbols)
    let min_prob_offset = 15 - n;
    let min_prob = unsafe {
        _mm256_loadu_si256(MIN_PROB_16.as_ptr().add(min_prob_offset) as *const __m256i)
    };

    // v = prod + min_prob
    let v = unsafe { _mm256_add_epi16(prod, min_prob) };

    // Store v for indexed access
    let mut v_arr = [0u16; 16];
    unsafe { _mm256_storeu_si256(v_arr.as_mut_ptr() as *mut __m256i, v) };

    // Compare using pmaxuw then equality: c >= v[i] iff max(c, v) == c
    let c_vec = unsafe { _mm256_set1_epi16(c as i16) };
    let max_cv = unsafe { _mm256_max_epu16(c_vec, v) };
    let cmp = unsafe { _mm256_cmpeq_epi16(max_cv, c_vec) };

    // Get mask and count trailing zeros to find first symbol where c < v
    let mask = unsafe { _mm256_movemask_epi8(cmp) as u32 };
    let val = (mask.trailing_zeros() >> 1) as u8;

    // Get u (previous v) and current v values for renormalization
    let u = if val == 0 { s.rng } else { v_arr[val as usize - 1] as u32 };
    let v_val = v_arr[val as usize] as u32;

    // Update CDF if enabled
    if s.allow_update_cdf() {
        let count = cdf[n];
        let rate = 4 + (count >> 4) + 1; // n_symbols > 2 always true for adapt16
        let val_usize = val as usize;

        for i in 0..val_usize {
            cdf[i] = cdf[i].wrapping_add(((1u16 << 15).wrapping_sub(cdf[i])) >> rate);
        }
        for i in val_usize..n {
            cdf[i] = cdf[i].wrapping_sub(cdf[i] >> rate);
        }
        cdf[n] = count + (count < 32) as u16;
    }

    // Renormalize
    ctx_norm(
        s,
        s.dif.wrapping_sub((v_val as EcWin) << (EC_WIN_SIZE - 16)),
        u - v_val,
    );

    val
}

/// SSE2 implementation of symbol_adapt8
#[cfg(all(not(feature = "asm"), target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn rav1d_msac_decode_symbol_adapt8_sse2(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    // For 8 symbols, SSE2 is sufficient
    // The overhead of SIMD setup may not be worth it for small n, use scalar
    rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols)
}

/// SSE2 implementation of symbol_adapt4
#[cfg(all(not(feature = "asm"), target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn rav1d_msac_decode_symbol_adapt4_sse2(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    // For 4 symbols, scalar is likely faster due to SIMD setup overhead
    rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols)
}

// NEON implementations for aarch64
#[cfg(all(not(feature = "asm"), target_arch = "aarch64"))]
static MIN_PROB_16_ARM: [u16; 16] = [60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0];

/// NEON implementation of symbol_adapt16
#[cfg(all(not(feature = "asm"), target_arch = "aarch64"))]
unsafe fn rav1d_msac_decode_symbol_adapt16_neon(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    let c = (s.dif >> (EC_WIN_SIZE - 16)) as u16;
    let n = n_symbols as usize;

    // Load CDF (16 values = 2 x uint16x8)
    let cdf_lo = unsafe { vld1q_u16(cdf.as_ptr()) };
    let cdf_hi = unsafe { vld1q_u16(cdf.as_ptr().add(8)) };

    // rng masked and broadcast
    let rng_masked = (s.rng & 0xff00) as u16;
    let rng_vec = unsafe { vdupq_n_u16(rng_masked) };

    // Process (cdf >> 6) << 7 for both halves
    let cdf_shifted_lo = unsafe { vshlq_n_u16(vshrq_n_u16(cdf_lo, 6), 7) };
    let cdf_shifted_hi = unsafe { vshlq_n_u16(vshrq_n_u16(cdf_hi, 6), 7) };

    // High multiply using vmull + shrn
    let rng_lo = unsafe { vget_low_u16(rng_vec) };
    let rng_hi = unsafe { vget_high_u16(rng_vec) };

    // Low 8
    let prod_lo_a = unsafe { vshrn_n_u32(vmull_u16(vget_low_u16(cdf_shifted_lo), rng_lo), 16) };
    let prod_lo_b = unsafe { vshrn_n_u32(vmull_u16(vget_high_u16(cdf_shifted_lo), rng_hi), 16) };
    let prod_lo = unsafe { vcombine_u16(prod_lo_a, prod_lo_b) };

    // High 8
    let prod_hi_a = unsafe { vshrn_n_u32(vmull_u16(vget_low_u16(cdf_shifted_hi), rng_lo), 16) };
    let prod_hi_b = unsafe { vshrn_n_u32(vmull_u16(vget_high_u16(cdf_shifted_hi), rng_hi), 16) };
    let prod_hi = unsafe { vcombine_u16(prod_hi_a, prod_hi_b) };

    // Load min_prob
    let min_prob_offset = 15 - n;
    let min_prob_lo = unsafe { vld1q_u16(MIN_PROB_16_ARM.as_ptr().add(min_prob_offset)) };
    let min_prob_hi = unsafe { vld1q_u16(MIN_PROB_16_ARM.as_ptr().add(min_prob_offset + 8)) };

    // v = prod + min_prob
    let v_lo = unsafe { vaddq_u16(prod_lo, min_prob_lo) };
    let v_hi = unsafe { vaddq_u16(prod_hi, min_prob_hi) };

    // Store v
    let mut v_arr = [0u16; 16];
    unsafe { vst1q_u16(v_arr.as_mut_ptr(), v_lo) };
    unsafe { vst1q_u16(v_arr.as_mut_ptr().add(8), v_hi) };

    // Compare c >= v[i]
    let c_vec = unsafe { vdupq_n_u16(c) };
    let cmp_lo = unsafe { vcgeq_u16(c_vec, v_lo) };
    let cmp_hi = unsafe { vcgeq_u16(c_vec, v_hi) };

    // Find symbol by counting consecutive true comparisons
    let mut mask_arr = [0u16; 16];
    unsafe { vst1q_u16(mask_arr.as_mut_ptr(), cmp_lo) };
    unsafe { vst1q_u16(mask_arr.as_mut_ptr().add(8), cmp_hi) };

    let mut val = 0u8;
    for i in 0..n {
        if mask_arr[i] != 0 {
            val = (i + 1) as u8;
        } else {
            break;
        }
    }

    // Get u and v values
    let u = if val == 0 { s.rng } else { v_arr[val as usize - 1] as u32 };
    let v_val = v_arr[val as usize] as u32;

    // Update CDF if enabled
    if s.allow_update_cdf() {
        let count = cdf[n];
        let rate = 4 + (count >> 4) + 1;
        let val_usize = val as usize;

        for i in 0..val_usize {
            cdf[i] = cdf[i].wrapping_add(((1u16 << 15).wrapping_sub(cdf[i])) >> rate);
        }
        for i in val_usize..n {
            cdf[i] = cdf[i].wrapping_sub(cdf[i] >> rate);
        }
        cdf[n] = count + (count < 32) as u16;
    }

    // Renormalize
    ctx_norm(
        s,
        s.dif.wrapping_sub((v_val as EcWin) << (EC_WIN_SIZE - 16)),
        u - v_val,
    );

    val
}

/// NEON implementation of symbol_adapt8 - uses scalar for simplicity
#[cfg(all(not(feature = "asm"), target_arch = "aarch64"))]
unsafe fn rav1d_msac_decode_symbol_adapt8_neon(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols)
}

/// NEON implementation of symbol_adapt4 - uses scalar for simplicity
#[cfg(all(not(feature = "asm"), target_arch = "aarch64"))]
unsafe fn rav1d_msac_decode_symbol_adapt4_neon(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols)
}

impl MsacContext {
    pub fn new(data: CArc<[u8]>, disable_cdf_update_flag: bool, dsp: &Rav1dMsacDSPContext) -> Self {
        let asm = MsacAsmContext {
            buf: data.as_ref().into(),
            dif: 0,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: (!disable_cdf_update_flag).into(),
            #[cfg(all(feature = "asm", target_arch = "x86_64"))]
            symbol_adapt16: dsp.symbol_adapt16,
        };
        let mut s = Self {
            asm,
            data: Some(data),
        };
        let _ = dsp.symbol_adapt16; // Silence unused warnings.
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
        if #[cfg(all(feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt4_sse2(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(all(feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt4_neon(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
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
        if #[cfg(all(feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt8_sse2(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(all(feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt8_neon(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
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
pub fn rav1d_msac_decode_symbol_adapt16(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 16);
    let ret;
    cfg_if! {
        if #[cfg(all(feature = "asm", target_arch = "x86_64"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                (s.symbol_adapt16)(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize, cdf.len())
            };
        } else if #[cfg(all(feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt16_sse2(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize, cdf.len())
            };
        } else if #[cfg(all(feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt16_neon(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(all(not(feature = "asm"), target_arch = "x86_64"))] {
            // Safe SIMD AVX2 implementation
            ret = unsafe {
                rav1d_msac_decode_symbol_adapt16_avx2(s, cdf, n_symbols)
            } as c_uint;
        } else if #[cfg(all(not(feature = "asm"), target_arch = "aarch64"))] {
            // Safe SIMD NEON implementation
            ret = unsafe {
                rav1d_msac_decode_symbol_adapt16_neon(s, cdf, n_symbols)
            } as c_uint;
        } else {
            ret = rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols);
        }
    }
    debug_assert!(ret < 16);
    ret as u8 % 16
}

pub fn rav1d_msac_decode_bool_adapt(s: &mut MsacContext, cdf: &mut [u16; 2]) -> bool {
    cfg_if! {
        if #[cfg(all(feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_adapt_rust`].
            unsafe {
                dav1d_msac_decode_bool_adapt_sse2(&mut s.asm, cdf.as_mut_ptr()) != 0
            }
        } else if #[cfg(all(feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_adapt_rust`].
            unsafe {
                dav1d_msac_decode_bool_adapt_neon(&mut s.asm, cdf.as_mut_ptr()) != 0
            }
        } else {
            rav1d_msac_decode_bool_adapt_rust(s, cdf)
        }
    }
}

pub fn rav1d_msac_decode_bool_equi(s: &mut MsacContext) -> bool {
    cfg_if! {
        if #[cfg(all(feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_equi_rust`].
            unsafe {
                dav1d_msac_decode_bool_equi_sse2(&mut s.asm) != 0
            }
        } else if #[cfg(all(feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_equi_rust`].
            unsafe {
                dav1d_msac_decode_bool_equi_neon(&mut s.asm) != 0
            }
        } else {
            rav1d_msac_decode_bool_equi_rust(s)
        }
    }
}

pub fn rav1d_msac_decode_bool(s: &mut MsacContext, f: c_uint) -> bool {
    cfg_if! {
        if #[cfg(all(feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_rust`].
            unsafe {
                dav1d_msac_decode_bool_sse2(&mut s.asm, f) != 0
            }
        } else if #[cfg(all(feature = "asm", target_feature = "neon"))] {
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
        if #[cfg(all(feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_hi_tok_rust`].
            ret = (unsafe {
                dav1d_msac_decode_hi_tok_sse2(&mut s.asm, cdf.as_mut_ptr())
            }) as u8;
        } else if #[cfg(all(feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_hi_tok_rust`].
            ret = unsafe {
                dav1d_msac_decode_hi_tok_neon(&mut s.asm, cdf.as_mut_ptr())
            } as u8;
        } else {
            ret = rav1d_msac_decode_hi_tok_rust(s, cdf);
        }
    }
    debug_assert!(ret < 16);
    ret % 16
}
