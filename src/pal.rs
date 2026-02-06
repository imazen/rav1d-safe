#![deny(unsafe_op_in_unsafe_fn)]

use crate::src::cpu::CpuFlags;
use crate::src::wrap_fn_ptr::wrap_fn_ptr;
use std::ffi::c_int;
use std::slice;

wrap_fn_ptr!(pub unsafe extern "C" fn pal_idx_finish(
    dst: *mut u8,
    src: *const u8,
    bw: c_int,
    bh: c_int,
    w: c_int,
    h: c_int,
) -> ());

/// Direct dispatch for pal_idx_finish - bypasses function pointer table.
///
/// Checks CPU flags at runtime and dispatches to the optimal SIMD implementation.
/// Used when `feature = "asm"` is disabled for zero-overhead direct calls.
#[cfg(not(feature = "asm"))]
fn pal_idx_finish_direct(
    dst: Option<&mut [u8]>,
    tmp: &mut [u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) {
    let dst = dst.map(|dst| &mut dst[..(bw / 2) * bh]);
    let tmp = &mut tmp[..bw * bh];
    // SAFETY: Note that `dst` and `src` may be the same.
    // This is safe because they are raw ptrs for now,
    // and in the fallback `fn pal_idx_finish_rust`, this is checked for
    // before creating `&mut`s from them.
    let dst = dst.unwrap_or(tmp).as_mut_ptr();
    let src = tmp.as_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        use crate::src::cpu::CpuFlags;
        if crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
            let [bw, bh, w, h] = [bw, bh, w, h].map(|it| it as c_int);
            // SAFETY: AVX2 verified by CpuFlags check. Pointers derived from valid slices above.
            unsafe {
                crate::src::safe_simd::pal::pal_idx_finish_avx2(dst, src, bw, bh, w, h);
            }
            return;
        }
    }

    // Scalar fallback (also used on aarch64 where no NEON pal implementation exists)
    #[allow(unreachable_code)]
    {
        let [bw, bh, w, h] = [bw, bh, w, h].map(|it| it as c_int);
        // SAFETY: Pointers derived from valid slices above. pal_idx_finish_c is safe.
        unsafe {
            pal_idx_finish_c(dst, src, bw, bh, w, h);
        }
    }
}

impl pal_idx_finish::Fn {
    /// If `dst` is [`None`], `tmp` is used as `dst`.
    /// This is why `tmp` must be `&mut`, too.
    /// `tmp` is always used as `src`.
    pub fn call(
        &self,
        dst: Option<&mut [u8]>,
        tmp: &mut [u8],
        bw: usize,
        bh: usize,
        w: usize,
        h: usize,
    ) {
        cfg_if::cfg_if! {
            if #[cfg(feature = "asm")] {
                let dst = dst.map(|dst| &mut dst[..(bw / 2) * bh]);
                let tmp = &mut tmp[..bw * bh];
                // SAFETY: Note that `dst` and `src` may be the same.
                // This is safe because they are raw ptrs for now,
                // and in the fallback `fn pal_idx_finish_rust`, this is checked for
                // before creating `&mut`s from them.
                let dst = dst.unwrap_or(tmp).as_mut_ptr();
                let src = tmp.as_ptr();
                let [bw, bh, w, h] = [bw, bh, w, h].map(|it| it as c_int);
                // SAFETY: Fallback `fn pal_idx_finish_rust` is safe; asm is supposed to do the same.
                unsafe { self.get()(dst, src, bw, bh, w, h) }
            } else {
                pal_idx_finish_direct(dst, tmp, bw, bh, w, h)
            }
        }
    }
}

pub struct Rav1dPalDSPContext {
    pub pal_idx_finish: pal_idx_finish::Fn,
}

enum PalIdx<'a> {
    Idx { dst: &'a mut [u8], src: &'a [u8] },
    Tmp(&'a mut [u8]),
}

/// # Safety
///
/// Must be called by [`pal_idx_finish::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn pal_idx_finish_c(
    dst: *mut u8,
    src: *const u8,
    bw: c_int,
    bh: c_int,
    w: c_int,
    h: c_int,
) {
    let [bw, bh, w, h] = [bw, bh, w, h].map(|it| it as usize);

    assert!(bw >= 4 && bw <= 64 && bw.is_power_of_two());
    assert!(bh >= 4 && bh <= 64 && bh.is_power_of_two());
    assert!(w >= 4 && w <= bw && (w & 3) == 0);
    assert!(h >= 4 && h <= bh && (h & 3) == 0);

    let idx = if src == dst {
        // SAFETY: `src` length sliced in `pal_idx_finish::Fn::call` and `src == dst`.
        let tmp = unsafe { slice::from_raw_parts_mut(dst, bw * bh) };
        PalIdx::Tmp(tmp)
    } else {
        // SAFETY: `src` length sliced in `pal_idx_finish::Fn::call`.
        let src = unsafe { slice::from_raw_parts(src, bw * bh) };
        // SAFETY: `dst` length sliced in `pal_idx_finish::Fn::call`.
        let dst = unsafe { slice::from_raw_parts_mut(dst, (bw / 2) * bh) };
        PalIdx::Idx { dst, src }
    };

    pal_idx_finish_rust(idx, bw, bh, w, h)
}

/// Fill invisible edges and pack to 4-bit (2 pixels per byte).
fn pal_idx_finish_rust(idx: PalIdx, bw: usize, bh: usize, w: usize, h: usize) {
    let dst_w = w / 2;
    let dst_bw = bw / 2;

    let dst = match idx {
        PalIdx::Tmp(tmp) => {
            for y in 0..h {
                let src = y * bw;
                let dst = y * dst_bw;
                for x in 0..dst_w {
                    let src = &tmp[src + 2 * x..][..2];
                    tmp[dst + x] = src[0] | (src[1] << 4)
                }
                if dst_w < dst_bw {
                    let src = tmp[src + w];
                    tmp[dst..][dst_w..dst_bw].fill(0x11 * src);
                }
            }

            &mut tmp[..dst_bw * bh]
        }
        PalIdx::Idx { dst, src } => {
            for y in 0..h {
                let src = &src[y * bw..];
                let dst = &mut dst[y * dst_bw..];
                for x in 0..dst_w {
                    dst[x] = src[2 * x] | (src[2 * x + 1] << 4)
                }
                if dst_w < dst_bw {
                    dst[dst_w..dst_bw].fill(0x11 * src[w]);
                }
            }

            dst
        }
    };

    if h < bh {
        let (last_row, dst) = dst[(h - 1) * dst_bw..].split_at_mut(dst_bw);

        for row in dst.chunks_exact_mut(dst_bw) {
            row.copy_from_slice(last_row);
        }
    }
}

impl Rav1dPalDSPContext {
    pub const fn default() -> Self {
        Self {
            pal_idx_finish: pal_idx_finish::Fn::new(pal_idx_finish_c),
        }
    }

    #[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline(always)]
    const fn init_x86(mut self, flags: CpuFlags) -> Self {
        if !flags.contains(CpuFlags::SSSE3) {
            return self;
        }

        self.pal_idx_finish = pal_idx_finish::decl_fn!(fn dav1d_pal_idx_finish_ssse3);

        #[cfg(target_arch = "x86_64")]
        {
            if !flags.contains(CpuFlags::AVX2) {
                return self;
            }

            self.pal_idx_finish = pal_idx_finish::decl_fn!(fn dav1d_pal_idx_finish_avx2);

            if !flags.contains(CpuFlags::AVX512ICL) {
                return self;
            }

            self.pal_idx_finish = pal_idx_finish::decl_fn!(fn dav1d_pal_idx_finish_avx512icl);
        }

        self
    }

    #[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    const fn init_arm(self, _flags: CpuFlags) -> Self {
        self
    }

    #[cfg(all(not(feature = "asm"), feature = "c-ffi", target_arch = "x86_64"))]
    #[inline(always)]
    const fn init_x86_safe_simd(mut self, _flags: CpuFlags) -> Self {
        self.pal_idx_finish =
            pal_idx_finish::Fn::new(crate::src::safe_simd::pal::pal_idx_finish_avx2);
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

        #[cfg(all(not(feature = "asm"), feature = "c-ffi"))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                return self.init_x86_safe_simd(flags);
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

impl Default for Rav1dPalDSPContext {
    fn default() -> Self {
        Self::default()
    }
}
