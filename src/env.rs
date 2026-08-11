#![forbid(unsafe_code)]
use crate::include::common::intops::apply_sign;
use crate::include::dav1d::headers::Rav1dFilterMode;
use crate::include::dav1d::headers::Rav1dFrameHeader;
use crate::include::dav1d::headers::Rav1dWarpedMotionParams;
use crate::include::dav1d::headers::Rav1dWarpedMotionType;
use crate::src::align::Align8;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::disjoint_mut::DisjointMutSlice;
use crate::src::internal::Bxy;
use crate::src::levels::BlockLevel;
use crate::src::levels::BlockPartition;
use crate::src::levels::CompInterType;
use crate::src::levels::DCT_DCT;
use crate::src::levels::H_ADST;
use crate::src::levels::H_FLIPADST;
use crate::src::levels::IDTX;
use crate::src::levels::Mv;
use crate::src::levels::SegmentId;
use crate::src::levels::TxfmSize;
use crate::src::levels::TxfmType;
use crate::src::levels::V_ADST;
use crate::src::levels::V_FLIPADST;
use crate::src::refmvs::RefMvsCandidate;
use crate::src::tables::TxfmInfo;
use std::cmp;
use std::cmp::Ordering;
use std::ffi::c_int;
use std::ffi::c_uint;

/// Read one element of the LEFT neighbour context, exclusively.
///
/// `t.l` is a [`BlockContext`] field of `Rav1dTaskContext` — the worker's own
/// struct — so a `&mut BlockContext` proves single-consumer exclusion by
/// borrowck and [`DisjointMut::get_mut`] is a plain field access that registers
/// nothing. Its sibling `f.a[t.a]` hangs off `Rav1dFrameData`, which tile tasks
/// reach through a SHARED `fc.data.try_read()` guard, so the ABOVE reads in the
/// same helpers keep their tracked `index()`. Same struct type, opposite
/// verdicts — `docs/OWNERSHIP_MODELS.md` §7e.
///
/// This is PR #492's `CaseSetter::set_exclusive` applied to the READ side: the
/// helpers below took both directions as one reference type, which dragged the
/// worker-local one through the tracker with the shared one.
///
/// The extent is unchanged — this removes a record, never widens one.
macro_rules! lread {
    ($l:expr, $field:ident [$k:expr], $i:expr) => {
        $l.$field[$k].get_mut()[$i as usize]
    };
    ($l:expr, $field:ident, $i:expr) => {
        $l.$field.get_mut()[$i as usize]
    };
}

#[derive(Default)]
pub struct BlockContext {
    pub mode: DisjointMut<Align8<[u8; 32]>>,
    pub lcoef: DisjointMut<Align8<[u8; 32]>>,
    pub ccoef: [DisjointMut<Align8<[u8; 32]>>; 2],
    pub seg_pred: DisjointMut<Align8<[u8; 32]>>,
    pub skip: DisjointMut<Align8<[u8; 32]>>,
    pub skip_mode: DisjointMut<Align8<[u8; 32]>>,
    pub intra: DisjointMut<Align8<[u8; 32]>>,
    pub comp_type: DisjointMut<Align8<[Option<CompInterType>; 32]>>,
    pub r#ref: [DisjointMut<Align8<[i8; 32]>>; 2],

    /// No [`Rav1dFilterMode::Switchable`]s here.
    /// TODO(kkysen) split [`Rav1dFilterMode`] into a version without [`Rav1dFilterMode::Switchable`].
    pub filter: [DisjointMut<Align8<[Rav1dFilterMode; 32]>>; 2],

    pub tx_intra: DisjointMut<Align8<[i8; 32]>>,
    pub tx: DisjointMut<Align8<[TxfmSize; 32]>>,
    pub tx_lpf_y: DisjointMut<Align8<[u8; 32]>>,
    pub tx_lpf_uv: DisjointMut<Align8<[u8; 32]>>,
    pub partition: DisjointMut<Align8<[u8; 16]>>,
    pub uvmode: DisjointMut<Align8<[u8; 32]>>,
    pub pal_sz: DisjointMut<Align8<[u8; 32]>>,
}

#[inline]
pub fn get_intra_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    if have_left {
        if have_top {
            let ctx = lread!(l, intra, yb4) + *a.intra.index(xb4 as usize);
            ctx + (ctx == 2) as u8
        } else {
            lread!(l, intra, yb4) * 2
        }
    } else {
        if have_top {
            *a.intra.index(xb4 as usize) * 2
        } else {
            0
        }
    }
}

#[inline]
pub fn get_tx_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    max_tx: &TxfmInfo,
    yb4: c_int,
    xb4: c_int,
) -> u8 {
    (lread!(l, tx_intra, yb4) as i32 >= max_tx.lh as i32) as u8
        + (*a.tx_intra.index(xb4 as usize) as i32 >= max_tx.lw as i32) as u8
}

#[inline]
pub fn get_partition_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    bl: BlockLevel,
    yb8: c_int,
    xb8: c_int,
) -> u8 {
    // the right-most ("index zero") bit of the partition represents the 8x8 block level,
    // but the BlockLevel enum represents the variants numerically in the opposite order
    // (128x128 = 0, 8x8 = 4). The shift reverses the ordering.
    let has_bl = |x| (x >> (4 - bl as u8)) & 1;
    has_bl(*a.partition.index(xb8 as usize)) + 2 * has_bl(lread!(l, partition, yb8))
}

#[inline]
pub fn gather_left_partition_prob(r#in: &[u16; 16], bl: BlockLevel) -> u32 {
    let mut out =
        r#in[BlockPartition::H as usize - 1] as i32 - r#in[BlockPartition::H as usize] as i32;
    // Exploit the fact that cdfs for BlockPartition::Split, BlockPartition::TopSplit,
    // BlockPartition::BottomSplit and BlockPartition::LeftSplit are neighbors.
    out += r#in[BlockPartition::Split as usize - 1] as i32
        - r#in[BlockPartition::LeftSplit as usize] as i32;
    if bl != BlockLevel::Bl128x128 {
        out +=
            r#in[BlockPartition::H4 as usize - 1] as i32 - r#in[BlockPartition::H4 as usize] as i32;
    }
    out as u32
}

#[inline]
pub fn gather_top_partition_prob(r#in: &[u16; 16], bl: BlockLevel) -> u32 {
    // Exploit the fact that cdfs for BlockPartition::V, BlockPartition::Split and
    // BlockPartition::TopSplit are neighbors.
    let mut out = r#in[BlockPartition::V as usize - 1] as i32
        - r#in[BlockPartition::TopSplit as usize] as i32;
    // Exploit the facts that cdfs for BlockPartition::LeftSplit and
    // BlockPartition::RightSplit are neighbors, the probability for
    // BlockPartition::V4 is always zero, and the probability for
    // BlockPartition::RightSplit is zero in 128x128 blocks.
    out += r#in[BlockPartition::LeftSplit as usize - 1] as i32;
    if bl != BlockLevel::Bl128x128 {
        out += r#in[BlockPartition::V4 as usize - 1] as i32
            - r#in[BlockPartition::RightSplit as usize] as i32;
    }
    out as u32
}

#[inline]
pub fn get_uv_inter_txtp(uvt_dim: &TxfmInfo, ytxtp: TxfmType) -> TxfmType {
    if uvt_dim.max == TxfmSize::S32x32 as _ {
        return if ytxtp == IDTX { IDTX } else { DCT_DCT };
    }
    if uvt_dim.min == TxfmSize::S16x16 as _
        && ((1 << ytxtp as u8)
            & ((1 << H_FLIPADST) | (1 << V_FLIPADST) | (1 << H_ADST) | (1 << V_ADST)))
            != 0
    {
        return DCT_DCT;
    }

    ytxtp
}

#[inline]
pub fn get_filter_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    comp: bool,
    dir: bool,
    r#ref: i8,
    yb4: c_int,
    xb4: c_int,
) -> u8 {
    // The two directions are no longer the same reference type, so the
    // homogeneous `[(a, xb4), (l, yb4)].map(..)` cannot carry them both. A
    // macro expands the identical body once per direction; substitution keeps
    // the reads exactly as lazy as the closure's were, so the ABOVE side's
    // registration count is unchanged.
    macro_rules! filter_of {
        ($ref0:expr, $ref1:expr, $filt:expr) => {
            if $ref0 == r#ref || $ref1 == r#ref {
                $filt
            } else {
                Rav1dFilterMode::N_SWITCHABLE_FILTERS
            }
        };
    }
    let a_filter = filter_of!(
        *a.r#ref[0].index(xb4 as usize),
        *a.r#ref[1].index(xb4 as usize),
        *a.filter[dir as usize].index(xb4 as usize)
    );
    let l_filter = filter_of!(
        lread!(l, r#ref[0], yb4),
        lread!(l, r#ref[1], yb4),
        lread!(l, filter[dir as usize], yb4)
    );

    (comp as u8) * 4
        + (if a_filter == l_filter {
            a_filter
        } else if a_filter == Rav1dFilterMode::N_SWITCHABLE_FILTERS {
            l_filter
        } else if l_filter == Rav1dFilterMode::N_SWITCHABLE_FILTERS {
            a_filter
        } else {
            Rav1dFilterMode::N_SWITCHABLE_FILTERS
        } as u8)
}

#[inline]
pub fn get_comp_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    if have_top {
        if have_left {
            if a.comp_type.index(xb4 as usize).is_some() {
                if lread!(l, comp_type, yb4).is_some() {
                    4
                } else {
                    // 4U means intra (-1) or bwd (>= 4)
                    2 + (lread!(l, r#ref[0], yb4) as c_uint >= 4) as u8
                }
            } else if lread!(l, comp_type, yb4).is_some() {
                // 4U means intra (-1) or bwd (>= 4)
                2 + (*a.r#ref[0].index(xb4 as usize) as c_uint >= 4) as u8
            } else {
                ((lread!(l, r#ref[0], yb4) >= 4) ^ (*a.r#ref[0].index(xb4 as usize) >= 4)) as u8
            }
        } else {
            if a.comp_type.index(xb4 as usize).is_some() {
                3
            } else {
                (*a.r#ref[0].index(xb4 as usize) >= 4) as u8
            }
        }
    } else if have_left {
        if lread!(l, comp_type, yb4).is_some() {
            3
        } else {
            (lread!(l, r#ref[0], yb4) >= 4) as u8
        }
    } else {
        1
    }
}

#[inline]
pub fn get_comp_dir_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    // `has_uni_comp` was one closure over `&BlockContext` and the `edge`
    // bindings below chose a direction at run time. Neither survives the split,
    // so the body is expanded per direction and every `edge` choice becomes a
    // branch on the same condition. Value-for-value identical to the closure
    // form; only the LEFT arm's reads change from `index()` to `get_mut()`.
    macro_rules! uni_comp_a {
        ($off:expr) => {
            (*a.r#ref[0].index($off as usize) < 4) == (*a.r#ref[1].index($off as usize) < 4)
        };
    }
    macro_rules! uni_comp_l {
        ($off:expr) => {
            (lread!(l, r#ref[0], $off) < 4) == (lread!(l, r#ref[1], $off) < 4)
        };
    }

    if have_top && have_left {
        let a_intra = *a.intra.index(xb4 as usize) != 0;
        let l_intra = lread!(l, intra, yb4) != 0;

        if a_intra && l_intra {
            return 2;
        }
        if a_intra || l_intra {
            // The edge examined is the side that is NOT intra.
            if a_intra {
                if lread!(l, comp_type, yb4).is_none() {
                    return 2;
                }
                return 1 + 2 * uni_comp_l!(yb4) as u8;
            } else {
                if a.comp_type.index(xb4 as usize).is_none() {
                    return 2;
                }
                return 1 + 2 * uni_comp_a!(xb4) as u8;
            }
        }

        let a_comp = a.comp_type.index(xb4 as usize).is_some();
        let l_comp = lread!(l, comp_type, yb4).is_some();
        let a_ref0 = *a.r#ref[0].index(xb4 as usize);
        let l_ref0 = lread!(l, r#ref[0], yb4);

        if !a_comp && !l_comp {
            return 1 + 2 * ((a_ref0 >= 4) == (l_ref0 >= 4)) as u8;
        } else if !a_comp || !l_comp {
            // The edge examined is the compound side.
            let uni = if a_comp {
                uni_comp_a!(xb4)
            } else {
                uni_comp_l!(yb4)
            };
            if !uni {
                return 1;
            }
            return 3 + ((a_ref0 >= 4) == (l_ref0 >= 4)) as u8;
        } else {
            let a_uni = uni_comp_a!(xb4);
            let l_uni = uni_comp_l!(yb4);

            if !a_uni && !l_uni {
                return 0;
            }
            if !a_uni || !l_uni {
                return 2;
            }
            return 3 + ((a_ref0 == 4) == (l_ref0 == 4)) as u8;
        }
    } else if have_left {
        if lread!(l, intra, yb4) != 0 {
            return 2;
        }
        if lread!(l, comp_type, yb4).is_none() {
            return 2;
        }
        return 4 * uni_comp_l!(yb4) as u8;
    } else if have_top {
        if *a.intra.index(xb4 as usize) != 0 {
            return 2;
        }
        if a.comp_type.index(xb4 as usize).is_none() {
            return 2;
        }
        return 4 * uni_comp_a!(xb4) as u8;
    } else {
        return 2;
    };
}

#[inline]
pub fn get_poc_diff(order_hint_n_bits: u8, poc0: c_int, poc1: c_int) -> c_int {
    if order_hint_n_bits == 0 {
        return 0;
    }
    let mask = 1 << order_hint_n_bits - 1;
    let diff = poc0 - poc1;
    return (diff & mask - 1) - (diff & mask);
}

#[inline]
pub fn get_jnt_comp_ctx(
    order_hint_n_bits: u8,
    poc: c_uint,
    ref0poc: c_uint,
    ref1poc: c_uint,
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
) -> u8 {
    let d0 = get_poc_diff(order_hint_n_bits, ref0poc as c_int, poc as c_int).abs();
    let d1 = get_poc_diff(order_hint_n_bits, poc as c_int, ref1poc as c_int).abs();
    let offset = (d0 == d1) as u8;
    macro_rules! jnt_ctx {
        ($comp:expr, $ref0:expr) => {
            ($comp >= Some(CompInterType::Avg) || $ref0 == 6) as u8
        };
    }
    let a_ctx = jnt_ctx!(
        *a.comp_type.index(xb4 as usize),
        *a.r#ref[0].index(xb4 as usize)
    );
    let l_ctx = jnt_ctx!(lread!(l, comp_type, yb4), lread!(l, r#ref[0], yb4));

    3 * offset + a_ctx + l_ctx
}

#[inline]
pub fn get_mask_comp_ctx(a: &BlockContext, l: &mut BlockContext, yb4: c_int, xb4: c_int) -> u8 {
    macro_rules! mask_ctx {
        ($comp:expr, $ref0:expr) => {
            if $comp >= Some(CompInterType::Seg) {
                1
            } else if $ref0 == 6 {
                3
            } else {
                0
            }
        };
    }
    let a_ctx = mask_ctx!(
        *a.comp_type.index(xb4 as usize),
        *a.r#ref[0].index(xb4 as usize)
    );
    let l_ctx = mask_ctx!(lread!(l, comp_type, yb4), lread!(l, r#ref[0], yb4));

    cmp::min(a_ctx + l_ctx, 5)
}

fn cmp_counts(c1: u8, c2: u8) -> u8 {
    use Ordering::*;
    match c1.cmp(&c2) {
        Less => 0,
        Equal => 1,
        Greater => 2,
    }
}

#[inline]
pub fn av1_get_ref_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    let mut cnt = [0; 2];

    if have_top && *a.intra.index(xb4 as usize) == 0 {
        cnt[(*a.r#ref[0].index(xb4 as usize) >= 4) as usize] += 1;
        if a.comp_type.index(xb4 as usize).is_some() {
            cnt[(*a.r#ref[1].index(xb4 as usize) >= 4) as usize] += 1;
        }
    }

    if have_left && lread!(l, intra, yb4) == 0 {
        cnt[(lread!(l, r#ref[0], yb4) >= 4) as usize] += 1;
        if lread!(l, comp_type, yb4).is_some() {
            cnt[(lread!(l, r#ref[1], yb4) >= 4) as usize] += 1;
        }
    }

    cmp_counts(cnt[0], cnt[1])
}

#[inline]
pub fn av1_get_fwd_ref_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    let mut cnt = [0; 4];

    if have_top && *a.intra.index(xb4 as usize) == 0 {
        let ref0 = *a.r#ref[0].index(xb4 as usize);
        if ref0 < 4 {
            cnt[ref0 as usize] += 1;
        }
        let ref1 = *a.r#ref[1].index(xb4 as usize);
        if a.comp_type.index(xb4 as usize).is_some() && ref1 < 4 {
            cnt[ref1 as usize] += 1;
        }
    }

    if have_left && lread!(l, intra, yb4) == 0 {
        let ref0 = lread!(l, r#ref[0], yb4);
        if ref0 < 4 {
            cnt[ref0 as usize] += 1;
        }
        let ref1 = lread!(l, r#ref[1], yb4);
        if lread!(l, comp_type, yb4).is_some() && ref1 < 4 {
            cnt[ref1 as usize] += 1;
        }
    }

    cnt[0] += cnt[1];
    cnt[2] += cnt[3];

    cmp_counts(cnt[0], cnt[2])
}

#[inline]
pub fn av1_get_fwd_ref_1_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    let mut cnt = [0; 2];

    if have_top && *a.intra.index(xb4 as usize) == 0 {
        let ref0 = *a.r#ref[0].index(xb4 as usize);
        if ref0 < 2 {
            cnt[ref0 as usize] += 1;
        }
        let ref1 = *a.r#ref[1].index(xb4 as usize);
        if a.comp_type.index(xb4 as usize).is_some() && ref1 < 2 {
            cnt[ref1 as usize] += 1;
        }
    }

    if have_left && lread!(l, intra, yb4) == 0 {
        let ref0 = lread!(l, r#ref[0], yb4);
        if ref0 < 2 {
            cnt[ref0 as usize] += 1;
        }
        let ref1 = lread!(l, r#ref[1], yb4);
        if lread!(l, comp_type, yb4).is_some() && ref1 < 2 {
            cnt[ref1 as usize] += 1;
        }
    }

    cmp_counts(cnt[0], cnt[1])
}

#[inline]
pub fn av1_get_fwd_ref_2_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    let mut cnt = [0; 2];

    if have_top && *a.intra.index(xb4 as usize) == 0 {
        let ref0 = *a.r#ref[0].index(xb4 as usize);
        if (ref0 ^ 2) < 2 {
            cnt[(ref0 - 2) as usize] += 1;
        }
        let ref1 = *a.r#ref[1].index(xb4 as usize);
        if a.comp_type.index(xb4 as usize).is_some() && (ref1 ^ 2) < 2 {
            cnt[(ref1 - 2) as usize] += 1;
        }
    }

    if have_left && lread!(l, intra, yb4) == 0 {
        let ref0 = lread!(l, r#ref[0], yb4);
        if (ref0 ^ 2) < 2 {
            cnt[(ref0 - 2) as usize] += 1;
        }
        let ref1 = lread!(l, r#ref[1], yb4);
        if lread!(l, comp_type, yb4).is_some() && (ref1 ^ 2) < 2 {
            cnt[(ref1 - 2) as usize] += 1;
        }
    }

    cmp_counts(cnt[0], cnt[1])
}

#[inline]
pub fn av1_get_bwd_ref_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    let mut cnt = [0; 3];

    if have_top && *a.intra.index(xb4 as usize) == 0 {
        let ref0 = *a.r#ref[0].index(xb4 as usize);
        if ref0 >= 4 {
            cnt[(ref0 - 4) as usize] += 1;
        }
        let ref1 = *a.r#ref[1].index(xb4 as usize);
        if a.comp_type.index(xb4 as usize).is_some() && ref1 >= 4 {
            cnt[(ref1 - 4) as usize] += 1;
        }
    }

    if have_left && lread!(l, intra, yb4) == 0 {
        let ref0 = lread!(l, r#ref[0], yb4);
        if ref0 >= 4 {
            cnt[(ref0 - 4) as usize] += 1;
        }
        let ref1 = lread!(l, r#ref[1], yb4);
        if lread!(l, comp_type, yb4).is_some() && ref1 >= 4 {
            cnt[(ref1 - 4) as usize] += 1;
        }
    }

    cnt[1] += cnt[0];

    cmp_counts(cnt[1], cnt[2])
}

#[inline]
pub fn av1_get_bwd_ref_1_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    let mut cnt = [0; 3];

    if have_top && *a.intra.index(xb4 as usize) == 0 {
        let ref0 = *a.r#ref[0].index(xb4 as usize);
        if ref0 >= 4 {
            cnt[(ref0 - 4) as usize] += 1;
        }
        let ref1 = *a.r#ref[1].index(xb4 as usize);
        if a.comp_type.index(xb4 as usize).is_some() && ref1 >= 4 {
            cnt[(ref1 - 4) as usize] += 1;
        }
    }

    if have_left && lread!(l, intra, yb4) == 0 {
        let ref0 = lread!(l, r#ref[0], yb4);
        if ref0 >= 4 {
            cnt[(ref0 - 4) as usize] += 1;
        }
        let ref1 = lread!(l, r#ref[1], yb4);
        if lread!(l, comp_type, yb4).is_some() && ref1 >= 4 {
            cnt[(ref1 - 4) as usize] += 1;
        }
    }

    cmp_counts(cnt[0], cnt[1])
}

#[inline]
pub fn av1_get_uni_p1_ctx(
    a: &BlockContext,
    l: &mut BlockContext,
    yb4: c_int,
    xb4: c_int,
    have_top: bool,
    have_left: bool,
) -> u8 {
    let mut cnt = [0; 3];

    if have_top && *a.intra.index(xb4 as usize) == 0 {
        if let Some(cnt) = cnt.get_mut((*a.r#ref[0].index(xb4 as usize) - 1) as usize) {
            *cnt += 1;
        }
        if a.comp_type.index(xb4 as usize).is_some() {
            if let Some(cnt) = cnt.get_mut((*a.r#ref[1].index(xb4 as usize) - 1) as usize) {
                *cnt += 1;
            }
        }
    }

    if have_left && lread!(l, intra, yb4) == 0 {
        if let Some(cnt) = cnt.get_mut((lread!(l, r#ref[0], yb4) - 1) as usize) {
            *cnt += 1;
        }
        if lread!(l, comp_type, yb4).is_some() {
            if let Some(cnt) = cnt.get_mut((lread!(l, r#ref[1], yb4) - 1) as usize) {
                *cnt += 1;
            }
        }
    }

    cnt[1] += cnt[2];

    cmp_counts(cnt[0], cnt[1])
}

#[inline]
pub fn get_drl_context(ref_mv_stack: &[RefMvsCandidate; 8], ref_idx: usize) -> c_int {
    if ref_mv_stack[ref_idx].weight >= 640 {
        (ref_mv_stack[ref_idx + 1].weight < 640) as c_int
    } else if ref_mv_stack[ref_idx + 1].weight < 640 {
        2
    } else {
        0
    }
}

#[inline]
pub fn get_cur_frame_segid(
    b: Bxy,
    have_top: bool,
    have_left: bool,
    cur_seg_map: &DisjointMutSlice<SegmentId>,
    stride: usize,
) -> (SegmentId, u8) {
    let negative_adjustment = have_left as usize + have_top as usize * stride;
    let offset = b.x as usize + b.y as usize * stride - negative_adjustment;
    match (have_left, have_top) {
        (true, true) => {
            let l = *cur_seg_map.index(offset + stride);
            let a = *cur_seg_map.index(offset + 1);
            let al = *cur_seg_map.index(offset);
            let seg_ctx = if l == a && al == l {
                2
            } else if l == a || al == l || a == al {
                1
            } else {
                0
            };
            let seg_id = if a == al { a } else { l };
            (seg_id, seg_ctx)
        }
        (true, false) | (false, true) => (*cur_seg_map.index(offset), 0),
        (false, false) => (Default::default(), 0),
    }
}

#[inline]
fn fix_int_mv_precision(mv: &mut Mv) {
    mv.x = (mv.x - (mv.x >> 15) + 3) & !7;
    mv.y = (mv.y - (mv.y >> 15) + 3) & !7;
}

#[inline]
pub(crate) fn fix_mv_precision(hdr: &Rav1dFrameHeader, mv: &mut Mv) {
    if hdr.force_integer_mv {
        fix_int_mv_precision(mv);
    } else if !(*hdr).hp {
        mv.x = (mv.x - (mv.x >> 15)) & !1;
        mv.y = (mv.y - (mv.y >> 15)) & !1;
    }
}

#[inline]
pub(crate) fn get_gmv_2d(
    gmv: &Rav1dWarpedMotionParams,
    bx4: c_int,
    by4: c_int,
    bw4: c_int,
    bh4: c_int,
    hdr: &Rav1dFrameHeader,
) -> Mv {
    match gmv.r#type {
        Rav1dWarpedMotionType::RotZoom => {
            assert!(gmv.matrix[5] == gmv.matrix[2]);
            assert!(gmv.matrix[4] == -gmv.matrix[3]);
        }
        Rav1dWarpedMotionType::Translation => {
            let mut res = Mv {
                y: (gmv.matrix[0] >> 13) as i16,
                x: (gmv.matrix[1] >> 13) as i16,
            };
            if hdr.force_integer_mv {
                fix_int_mv_precision(&mut res);
            }
            return res;
        }
        Rav1dWarpedMotionType::Identity => {
            return Mv::ZERO;
        }
        Rav1dWarpedMotionType::Affine => {}
    }
    let x = bx4 * 4 + bw4 * 2 - 1;
    let y = by4 * 4 + bh4 * 2 - 1;
    let xc = (gmv.matrix[2] - (1 << 16)) * x + gmv.matrix[3] * y + gmv.matrix[0];
    let yc = (gmv.matrix[5] - (1 << 16)) * y + gmv.matrix[4] * x + gmv.matrix[1];
    let shift = 16 - (3 - !hdr.hp as c_int);
    let round = 1 << shift >> 1;
    let mut res = Mv {
        y: apply_sign(yc.abs() + round >> shift << !hdr.hp as c_int, yc) as i16,
        x: apply_sign(xc.abs() + round >> shift << !hdr.hp as c_int, xc) as i16,
    };
    if hdr.force_integer_mv {
        fix_int_mv_precision(&mut res);
    }
    return res;
}

/// The LEFT/ABOVE split must not change one context value.
///
/// Oracle: the pre-split implementations, transcribed VERBATIM from
/// `414515c:src/env.rs` — the commit this branch is based on — so the refactor
/// is differenced against the code it replaces rather than against a paraphrase
/// of the spec. Eleven of the fifteen only changed `index()` to `get_mut()` and
/// could still have had an index transposed by the scripted rewrite; the other
/// four were RESTRUCTURED and are where a real bug would hide:
/// `get_filter_ctx`, `get_jnt_comp_ctx` and `get_mask_comp_ctx` lost a
/// homogeneous `[(a, xb4), (l, yb4)].map(..)` (the two directions are no longer
/// one type), and `get_comp_dir_ctx` lost a closure plus two run-time `edge`
/// selections.
#[cfg(test)]
mod left_split_parity {
    use super::*;
    use crate::src::levels::N_INTRA_PRED_MODES;
    use crate::src::levels::N_UV_INTRA_PRED_MODES;
    use crate::src::tables::dav1d_txfm_dimensions;
    use strum::EnumCount as _;

    // ---- oracles: verbatim from 414515c ----------------------------------

    pub fn base_get_intra_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        if have_left {
            if have_top {
                let ctx = *l.intra.index(yb4 as usize) + *a.intra.index(xb4 as usize);
                ctx + (ctx == 2) as u8
            } else {
                *l.intra.index(yb4 as usize) * 2
            }
        } else {
            if have_top {
                *a.intra.index(xb4 as usize) * 2
            } else {
                0
            }
        }
    }

    pub fn base_get_tx_ctx(
        a: &BlockContext,
        l: &BlockContext,
        max_tx: &TxfmInfo,
        yb4: c_int,
        xb4: c_int,
    ) -> u8 {
        (*l.tx_intra.index(yb4 as usize) as i32 >= max_tx.lh as i32) as u8
            + (*a.tx_intra.index(xb4 as usize) as i32 >= max_tx.lw as i32) as u8
    }

    pub fn base_get_partition_ctx(
        a: &BlockContext,
        l: &BlockContext,
        bl: BlockLevel,
        yb8: c_int,
        xb8: c_int,
    ) -> u8 {
        // the right-most ("index zero") bit of the partition represents the 8x8 block level,
        // but the BlockLevel enum represents the variants numerically in the opposite order
        // (128x128 = 0, 8x8 = 4). The shift reverses the ordering.
        let has_bl = |x| (x >> (4 - bl as u8)) & 1;
        has_bl(*a.partition.index(xb8 as usize)) + 2 * has_bl(*l.partition.index(yb8 as usize))
    }

    pub fn base_get_filter_ctx(
        a: &BlockContext,
        l: &BlockContext,
        comp: bool,
        dir: bool,
        r#ref: i8,
        yb4: c_int,
        xb4: c_int,
    ) -> u8 {
        let [a_filter, l_filter] = [(a, xb4), (l, yb4)].map(|(al, b4)| {
            if *al.r#ref[0].index(b4 as usize) == r#ref || *al.r#ref[1].index(b4 as usize) == r#ref
            {
                *al.filter[dir as usize].index(b4 as usize)
            } else {
                Rav1dFilterMode::N_SWITCHABLE_FILTERS
            }
        });

        (comp as u8) * 4
            + (if a_filter == l_filter {
                a_filter
            } else if a_filter == Rav1dFilterMode::N_SWITCHABLE_FILTERS {
                l_filter
            } else if l_filter == Rav1dFilterMode::N_SWITCHABLE_FILTERS {
                a_filter
            } else {
                Rav1dFilterMode::N_SWITCHABLE_FILTERS
            } as u8)
    }

    pub fn base_get_comp_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        if have_top {
            if have_left {
                if a.comp_type.index(xb4 as usize).is_some() {
                    if l.comp_type.index(yb4 as usize).is_some() {
                        4
                    } else {
                        // 4U means intra (-1) or bwd (>= 4)
                        2 + (*l.r#ref[0].index(yb4 as usize) as c_uint >= 4) as u8
                    }
                } else if l.comp_type.index(yb4 as usize).is_some() {
                    // 4U means intra (-1) or bwd (>= 4)
                    2 + (*a.r#ref[0].index(xb4 as usize) as c_uint >= 4) as u8
                } else {
                    ((*l.r#ref[0].index(yb4 as usize) >= 4)
                        ^ (*a.r#ref[0].index(xb4 as usize) >= 4)) as u8
                }
            } else {
                if a.comp_type.index(xb4 as usize).is_some() {
                    3
                } else {
                    (*a.r#ref[0].index(xb4 as usize) >= 4) as u8
                }
            }
        } else if have_left {
            if l.comp_type.index(yb4 as usize).is_some() {
                3
            } else {
                (*l.r#ref[0].index(yb4 as usize) >= 4) as u8
            }
        } else {
            1
        }
    }

    pub fn base_get_comp_dir_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        let has_uni_comp = |edge: &BlockContext, off| {
            (*edge.r#ref[0].index(off as usize) < 4) == (*edge.r#ref[1].index(off as usize) < 4)
        };

        if have_top && have_left {
            let a_intra = *a.intra.index(xb4 as usize) != 0;
            let l_intra = *l.intra.index(yb4 as usize) != 0;

            if a_intra && l_intra {
                return 2;
            }
            if a_intra || l_intra {
                let edge = if a_intra { &l } else { &a };
                let off = if a_intra { yb4 } else { xb4 };

                if edge.comp_type.index(off as usize).is_none() {
                    return 2;
                }
                return 1 + 2 * has_uni_comp(edge, off) as u8;
            }

            let a_comp = a.comp_type.index(xb4 as usize).is_some();
            let l_comp = l.comp_type.index(yb4 as usize).is_some();
            let a_ref0 = *a.r#ref[0].index(xb4 as usize);
            let l_ref0 = *l.r#ref[0].index(yb4 as usize);

            if !a_comp && !l_comp {
                return 1 + 2 * ((a_ref0 >= 4) == (l_ref0 >= 4)) as u8;
            } else if !a_comp || !l_comp {
                let edge = if a_comp { &a } else { &l };
                let off = if a_comp { xb4 } else { yb4 };

                if !has_uni_comp(edge, off) {
                    return 1;
                }
                return 3 + ((a_ref0 >= 4) == (l_ref0 >= 4)) as u8;
            } else {
                let a_uni = has_uni_comp(&a, xb4);
                let l_uni = has_uni_comp(&l, yb4);

                if !a_uni && !l_uni {
                    return 0;
                }
                if !a_uni || !l_uni {
                    return 2;
                }
                return 3 + ((a_ref0 == 4) == (l_ref0 == 4)) as u8;
            }
        } else if have_top || have_left {
            let edge = if have_left { l } else { a };
            let off = if have_left { yb4 } else { xb4 };

            if *edge.intra.index(off as usize) != 0 {
                return 2;
            }
            if edge.comp_type.index(off as usize).is_none() {
                return 2;
            }
            return 4 * has_uni_comp(&edge, off) as u8;
        } else {
            return 2;
        };
    }

    pub fn base_get_jnt_comp_ctx(
        order_hint_n_bits: u8,
        poc: c_uint,
        ref0poc: c_uint,
        ref1poc: c_uint,
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
    ) -> u8 {
        let d0 = get_poc_diff(order_hint_n_bits, ref0poc as c_int, poc as c_int).abs();
        let d1 = get_poc_diff(order_hint_n_bits, poc as c_int, ref1poc as c_int).abs();
        let offset = (d0 == d1) as u8;
        let [a_ctx, l_ctx] = [(a, xb4), (l, yb4)].map(|(al, b4)| {
            (*al.comp_type.index(b4 as usize) >= Some(CompInterType::Avg)
                || *al.r#ref[0].index(b4 as usize) == 6) as u8
        });

        3 * offset + a_ctx + l_ctx
    }

    pub fn base_get_mask_comp_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
    ) -> u8 {
        let [a_ctx, l_ctx] = [(a, xb4), (l, yb4)].map(|(al, b4)| {
            if *al.comp_type.index(b4 as usize) >= Some(CompInterType::Seg) {
                1
            } else if *al.r#ref[0].index(b4 as usize) == 6 {
                3
            } else {
                0
            }
        });

        cmp::min(a_ctx + l_ctx, 5)
    }

    pub fn base_av1_get_ref_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        let mut cnt = [0; 2];

        if have_top && *a.intra.index(xb4 as usize) == 0 {
            cnt[(*a.r#ref[0].index(xb4 as usize) >= 4) as usize] += 1;
            if a.comp_type.index(xb4 as usize).is_some() {
                cnt[(*a.r#ref[1].index(xb4 as usize) >= 4) as usize] += 1;
            }
        }

        if have_left && *l.intra.index(yb4 as usize) == 0 {
            cnt[(*l.r#ref[0].index(yb4 as usize) >= 4) as usize] += 1;
            if l.comp_type.index(yb4 as usize).is_some() {
                cnt[(*l.r#ref[1].index(yb4 as usize) >= 4) as usize] += 1;
            }
        }

        cmp_counts(cnt[0], cnt[1])
    }

    pub fn base_av1_get_fwd_ref_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        let mut cnt = [0; 4];

        if have_top && *a.intra.index(xb4 as usize) == 0 {
            let ref0 = *a.r#ref[0].index(xb4 as usize);
            if ref0 < 4 {
                cnt[ref0 as usize] += 1;
            }
            let ref1 = *a.r#ref[1].index(xb4 as usize);
            if a.comp_type.index(xb4 as usize).is_some() && ref1 < 4 {
                cnt[ref1 as usize] += 1;
            }
        }

        if have_left && *l.intra.index(yb4 as usize) == 0 {
            let ref0 = *l.r#ref[0].index(yb4 as usize);
            if ref0 < 4 {
                cnt[ref0 as usize] += 1;
            }
            let ref1 = *l.r#ref[1].index(yb4 as usize);
            if l.comp_type.index(yb4 as usize).is_some() && ref1 < 4 {
                cnt[ref1 as usize] += 1;
            }
        }

        cnt[0] += cnt[1];
        cnt[2] += cnt[3];

        cmp_counts(cnt[0], cnt[2])
    }

    pub fn base_av1_get_fwd_ref_1_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        let mut cnt = [0; 2];

        if have_top && *a.intra.index(xb4 as usize) == 0 {
            let ref0 = *a.r#ref[0].index(xb4 as usize);
            if ref0 < 2 {
                cnt[ref0 as usize] += 1;
            }
            let ref1 = *a.r#ref[1].index(xb4 as usize);
            if a.comp_type.index(xb4 as usize).is_some() && ref1 < 2 {
                cnt[ref1 as usize] += 1;
            }
        }

        if have_left && *l.intra.index(yb4 as usize) == 0 {
            let ref0 = *l.r#ref[0].index(yb4 as usize);
            if ref0 < 2 {
                cnt[ref0 as usize] += 1;
            }
            let ref1 = *l.r#ref[1].index(yb4 as usize);
            if l.comp_type.index(yb4 as usize).is_some() && ref1 < 2 {
                cnt[ref1 as usize] += 1;
            }
        }

        cmp_counts(cnt[0], cnt[1])
    }

    pub fn base_av1_get_fwd_ref_2_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        let mut cnt = [0; 2];

        if have_top && *a.intra.index(xb4 as usize) == 0 {
            let ref0 = *a.r#ref[0].index(xb4 as usize);
            if (ref0 ^ 2) < 2 {
                cnt[(ref0 - 2) as usize] += 1;
            }
            let ref1 = *a.r#ref[1].index(xb4 as usize);
            if a.comp_type.index(xb4 as usize).is_some() && (ref1 ^ 2) < 2 {
                cnt[(ref1 - 2) as usize] += 1;
            }
        }

        if have_left && *l.intra.index(yb4 as usize) == 0 {
            let ref0 = *l.r#ref[0].index(yb4 as usize);
            if (ref0 ^ 2) < 2 {
                cnt[(ref0 - 2) as usize] += 1;
            }
            let ref1 = *l.r#ref[1].index(yb4 as usize);
            if l.comp_type.index(yb4 as usize).is_some() && (ref1 ^ 2) < 2 {
                cnt[(ref1 - 2) as usize] += 1;
            }
        }

        cmp_counts(cnt[0], cnt[1])
    }

    pub fn base_av1_get_bwd_ref_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        let mut cnt = [0; 3];

        if have_top && *a.intra.index(xb4 as usize) == 0 {
            let ref0 = *a.r#ref[0].index(xb4 as usize);
            if ref0 >= 4 {
                cnt[(ref0 - 4) as usize] += 1;
            }
            let ref1 = *a.r#ref[1].index(xb4 as usize);
            if a.comp_type.index(xb4 as usize).is_some() && ref1 >= 4 {
                cnt[(ref1 - 4) as usize] += 1;
            }
        }

        if have_left && *l.intra.index(yb4 as usize) == 0 {
            let ref0 = *l.r#ref[0].index(yb4 as usize);
            if ref0 >= 4 {
                cnt[(ref0 - 4) as usize] += 1;
            }
            let ref1 = *l.r#ref[1].index(yb4 as usize);
            if l.comp_type.index(yb4 as usize).is_some() && ref1 >= 4 {
                cnt[(ref1 - 4) as usize] += 1;
            }
        }

        cnt[1] += cnt[0];

        cmp_counts(cnt[1], cnt[2])
    }

    pub fn base_av1_get_bwd_ref_1_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        let mut cnt = [0; 3];

        if have_top && *a.intra.index(xb4 as usize) == 0 {
            let ref0 = *a.r#ref[0].index(xb4 as usize);
            if ref0 >= 4 {
                cnt[(ref0 - 4) as usize] += 1;
            }
            let ref1 = *a.r#ref[1].index(xb4 as usize);
            if a.comp_type.index(xb4 as usize).is_some() && ref1 >= 4 {
                cnt[(ref1 - 4) as usize] += 1;
            }
        }

        if have_left && *l.intra.index(yb4 as usize) == 0 {
            let ref0 = *l.r#ref[0].index(yb4 as usize);
            if ref0 >= 4 {
                cnt[(ref0 - 4) as usize] += 1;
            }
            let ref1 = *l.r#ref[1].index(yb4 as usize);
            if l.comp_type.index(yb4 as usize).is_some() && ref1 >= 4 {
                cnt[(ref1 - 4) as usize] += 1;
            }
        }

        cmp_counts(cnt[0], cnt[1])
    }

    pub fn base_av1_get_uni_p1_ctx(
        a: &BlockContext,
        l: &BlockContext,
        yb4: c_int,
        xb4: c_int,
        have_top: bool,
        have_left: bool,
    ) -> u8 {
        let mut cnt = [0; 3];

        if have_top && *a.intra.index(xb4 as usize) == 0 {
            if let Some(cnt) = cnt.get_mut((*a.r#ref[0].index(xb4 as usize) - 1) as usize) {
                *cnt += 1;
            }
            if a.comp_type.index(xb4 as usize).is_some() {
                if let Some(cnt) = cnt.get_mut((*a.r#ref[1].index(xb4 as usize) - 1) as usize) {
                    *cnt += 1;
                }
            }
        }

        if have_left && *l.intra.index(yb4 as usize) == 0 {
            if let Some(cnt) = cnt.get_mut((*l.r#ref[0].index(yb4 as usize) - 1) as usize) {
                *cnt += 1;
            }
            if l.comp_type.index(yb4 as usize).is_some() {
                if let Some(cnt) = cnt.get_mut((*l.r#ref[1].index(yb4 as usize) - 1) as usize) {
                    *cnt += 1;
                }
            }
        }

        cnt[1] += cnt[2];

        cmp_counts(cnt[0], cnt[1])
    }

    // ---- harness ---------------------------------------------------------

    /// xorshift; a fixed seed keeps a failure reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    fn comp_type_of(v: u32) -> Option<CompInterType> {
        match v {
            0 => None,
            1 => Some(CompInterType::WeightedAvg),
            2 => Some(CompInterType::Avg),
            3 => Some(CompInterType::Seg),
            _ => Some(CompInterType::Wedge),
        }
    }

    fn filter_of(v: u32) -> Rav1dFilterMode {
        match v {
            0 => Rav1dFilterMode::Regular8Tap,
            1 => Rav1dFilterMode::Smooth8Tap,
            2 => Rav1dFilterMode::Sharp8Tap,
            3 => Rav1dFilterMode::Bilinear,
            _ => Rav1dFilterMode::Switchable,
        }
    }

    fn block_level_of(v: u32) -> BlockLevel {
        match v {
            0 => BlockLevel::Bl128x128,
            1 => BlockLevel::Bl64x64,
            2 => BlockLevel::Bl32x32,
            3 => BlockLevel::Bl16x16,
            _ => BlockLevel::Bl8x8,
        }
    }

    /// Every field any of the fifteen helpers reads, filled over its real
    /// value range AND under the decoder's own coupling between them.
    ///
    /// The coupling is not decoration: the pre-split ladders index `cnt` by a
    /// reference slot only after testing `intra == 0`, so `intra == 0` must
    /// imply `ref[0] >= 0`, and they read `ref[1]` only under
    /// `comp_type.is_some()`. Filling the fields independently panics the
    /// ORACLE (`cnt[(-1) as usize]`) — which is how this invariant was found,
    /// and why it is stated here instead of clamped away.
    fn random_ctx(rng: &mut Rng) -> BlockContext {
        let mut b = BlockContext::default();
        for i in 0..32 {
            let is_intra = rng.below(2) == 1;
            let comp = if is_intra {
                None
            } else {
                comp_type_of(rng.below(5))
            };
            b.intra.get_mut()[i] = is_intra as u8;
            b.comp_type.get_mut()[i] = comp;
            // -1 is intra; 0..=6 are the seven reference slots.
            b.r#ref[0].get_mut()[i] = if is_intra { -1 } else { rng.below(7) as i8 };
            b.r#ref[1].get_mut()[i] = if comp.is_some() {
                rng.below(7) as i8
            } else {
                -1
            };
            b.mode.get_mut()[i] = rng.below(N_INTRA_PRED_MODES as u32) as u8;
            b.uvmode.get_mut()[i] = rng.below(N_UV_INTRA_PRED_MODES as u32) as u8;
            b.tx_intra.get_mut()[i] = rng.below(9) as i8 - 1;
            b.filter[0].get_mut()[i] = filter_of(rng.below(5));
            b.filter[1].get_mut()[i] = filter_of(rng.below(5));
        }
        for i in 0..16 {
            b.partition.get_mut()[i] = rng.below(32) as u8;
        }
        b
    }

    /// Distinct outputs seen per helper. A helper that returned one constant
    /// over 20,000 trials would make its parity assertion vacuous, so the test
    /// fails on that too — the "assert liveness" half of the gate.
    #[derive(Default)]
    struct Seen([u64; 16]);
    impl Seen {
        fn note(&mut self, v: u8) {
            self.0[(v & 15) as usize] |= 1 << (v >> 4).min(63);
        }
        fn distinct(&self) -> u32 {
            self.0.iter().map(|w| w.count_ones()).sum()
        }
    }

    #[test]
    fn all_fifteen_helpers_match_the_pre_split_implementation() {
        // Under Miri the point is the ALIASING of `&mut BlockContext` +
        // `get_mut()`, not coverage, and 20,000 trials do not finish inside a
        // sane timeout. 200 is not a relaxation, and the reason is that the
        // liveness assertions at the bottom of this function are
        // UNCONDITIONAL: if 200 trials failed to reach all four
        // `have_top`/`have_left` combinations or all five `get_comp_dir_ctx`
        // outputs, the Miri run itself would FAIL rather than pass with less
        // coverage. `trial_floor_is_not_vacuous` is the early warning for the
        // same thing in a normal build, so a regression does not need a Miri
        // run to surface.
        let trials = if cfg!(miri) { 200 } else { 20_000 };
        let mut rng = Rng(0x5eed_1eaf_c0ff_ee01);
        let mut seen: [Seen; 15] = Default::default();
        let mut tl_combos = 0u8;
        for _ in 0..trials {
            let a = random_ctx(&mut rng);
            let mut l = random_ctx(&mut rng);
            // A second copy of the LEFT state, because the oracle needs a
            // shared `&BlockContext` and the split needs `&mut`.
            let l_ref = {
                let mut c = BlockContext::default();
                for i in 0..32 {
                    c.intra.get_mut()[i] = l.intra.get_mut()[i];
                    c.mode.get_mut()[i] = l.mode.get_mut()[i];
                    c.uvmode.get_mut()[i] = l.uvmode.get_mut()[i];
                    c.tx_intra.get_mut()[i] = l.tx_intra.get_mut()[i];
                    c.comp_type.get_mut()[i] = l.comp_type.get_mut()[i];
                    c.r#ref[0].get_mut()[i] = l.r#ref[0].get_mut()[i];
                    c.r#ref[1].get_mut()[i] = l.r#ref[1].get_mut()[i];
                    c.filter[0].get_mut()[i] = l.filter[0].get_mut()[i];
                    c.filter[1].get_mut()[i] = l.filter[1].get_mut()[i];
                }
                for i in 0..16 {
                    c.partition.get_mut()[i] = l.partition.get_mut()[i];
                }
                c
            };
            let yb4 = rng.below(32) as c_int;
            let xb4 = rng.below(32) as c_int;
            let yb8 = rng.below(16) as c_int;
            let xb8 = rng.below(16) as c_int;
            let have_top = rng.below(2) == 1;
            let have_left = rng.below(2) == 1;
            tl_combos |= 1 << (have_top as u8 * 2 + have_left as u8);
            let bl = block_level_of(rng.below(5));
            let t_dim = &dav1d_txfm_dimensions[rng.below(TxfmSize::COUNT as u32) as usize];
            let comp = rng.below(2) == 1;
            let dir = rng.below(2) == 1;
            let r#ref = rng.below(8) as i8 - 1;

            macro_rules! check {
                ($i:expr, $f:ident ( $($arg:expr),* $(,)? )) => {{
                    let want = paste_base!($f)(&a, &l_ref, $($arg),*);
                    let got = $f(&a, &mut l, $($arg),*);
                    assert_eq!(got, want, concat!(stringify!($f), " diverged"));
                    seen[$i].note(got);
                }};
            }
            // `concat_idents!` is unstable, so the oracle name is spelled out.
            macro_rules! paste_base {
                (get_intra_ctx) => {
                    base_get_intra_ctx
                };
                (get_tx_ctx) => {
                    base_get_tx_ctx
                };
                (get_partition_ctx) => {
                    base_get_partition_ctx
                };
                (get_filter_ctx) => {
                    base_get_filter_ctx
                };
                (get_comp_ctx) => {
                    base_get_comp_ctx
                };
                (get_comp_dir_ctx) => {
                    base_get_comp_dir_ctx
                };
                (get_mask_comp_ctx) => {
                    base_get_mask_comp_ctx
                };
                (av1_get_ref_ctx) => {
                    base_av1_get_ref_ctx
                };
                (av1_get_fwd_ref_ctx) => {
                    base_av1_get_fwd_ref_ctx
                };
                (av1_get_fwd_ref_1_ctx) => {
                    base_av1_get_fwd_ref_1_ctx
                };
                (av1_get_fwd_ref_2_ctx) => {
                    base_av1_get_fwd_ref_2_ctx
                };
                (av1_get_bwd_ref_ctx) => {
                    base_av1_get_bwd_ref_ctx
                };
                (av1_get_bwd_ref_1_ctx) => {
                    base_av1_get_bwd_ref_1_ctx
                };
                (av1_get_uni_p1_ctx) => {
                    base_av1_get_uni_p1_ctx
                };
            }

            check!(0, get_intra_ctx(yb4, xb4, have_top, have_left));
            check!(1, get_tx_ctx(t_dim, yb4, xb4));
            check!(2, get_partition_ctx(bl, yb8, xb8));
            check!(3, get_filter_ctx(comp, dir, r#ref, yb4, xb4));
            check!(4, get_comp_ctx(yb4, xb4, have_top, have_left));
            check!(5, get_comp_dir_ctx(yb4, xb4, have_top, have_left));
            check!(6, get_mask_comp_ctx(yb4, xb4));
            check!(7, av1_get_ref_ctx(yb4, xb4, have_top, have_left));
            check!(8, av1_get_fwd_ref_ctx(yb4, xb4, have_top, have_left));
            check!(9, av1_get_fwd_ref_1_ctx(yb4, xb4, have_top, have_left));
            check!(10, av1_get_fwd_ref_2_ctx(yb4, xb4, have_top, have_left));
            check!(11, av1_get_bwd_ref_ctx(yb4, xb4, have_top, have_left));
            check!(12, av1_get_bwd_ref_1_ctx(yb4, xb4, have_top, have_left));
            check!(13, av1_get_uni_p1_ctx(yb4, xb4, have_top, have_left));

            // `get_jnt_comp_ctx`'s extra arguments are pure arithmetic on POCs.
            let ohn = rng.below(9) as u8;
            let poc = rng.below(256);
            let r0 = rng.below(256);
            let r1 = rng.below(256);
            let want = base_get_jnt_comp_ctx(ohn, poc, r0, r1, &a, &l_ref, yb4, xb4);
            let got = get_jnt_comp_ctx(ohn, poc, r0, r1, &a, &mut l, yb4, xb4);
            assert_eq!(got, want, "get_jnt_comp_ctx diverged");
            seen[14].note(got);
        }

        assert_eq!(
            tl_combos, 0b1111,
            "not all have_top/have_left combinations ran"
        );
        for (i, s) in seen.iter().enumerate() {
            assert!(
                s.distinct() >= 2,
                "helper {i} returned one constant over {trials} trials — its parity \
                 assertion proves nothing"
            );
        }
        // `get_comp_dir_ctx` is the restructured one with the most branches;
        // require its whole 0..=4 range so no `edge` arm is silently unreached.
        assert!(
            seen[5].distinct() >= 5,
            "get_comp_dir_ctx reached only {} of its 5 outputs",
            seen[5].distinct()
        );
    }

    /// The Miri trial floor is not vacuous.
    ///
    /// `all_fifteen_helpers_…` runs 200 trials under Miri instead of 20,000.
    /// Its own liveness assertions are unconditional, so a 200-trial run that
    /// lost coverage would FAIL under Miri rather than pass — that is the
    /// actual guarantee, and the Miri leg passing 3/3 is the evidence.
    ///
    /// This is the early warning for the same thing in a normal build: 200
    /// trials of the SAME generator (a different draw sequence, since it does
    /// not consume the other helpers' arguments) still reach all four
    /// `have_top`/`have_left` combinations and all five `get_comp_dir_ctx`
    /// outputs. If a future edit to `random_ctx` makes 200 too few, this fails
    /// on every build instead of only when someone runs Miri.
    #[test]
    fn trial_floor_is_not_vacuous() {
        let mut rng = Rng(0x5eed_1eaf_c0ff_ee01);
        let mut tl_combos = 0u8;
        let mut seen5 = Seen::default();
        for _ in 0..200 {
            let a = random_ctx(&mut rng);
            let mut l = random_ctx(&mut rng);
            let yb4 = rng.below(32) as c_int;
            let xb4 = rng.below(32) as c_int;
            let have_top = rng.below(2) == 1;
            let have_left = rng.below(2) == 1;
            tl_combos |= 1 << (have_top as u8 * 2 + have_left as u8);
            seen5.note(get_comp_dir_ctx(&a, &mut l, yb4, xb4, have_top, have_left));
        }
        assert_eq!(
            tl_combos, 0b1111,
            "200 trials miss a have_top/have_left combination"
        );
        assert!(
            seen5.distinct() >= 5,
            "200 trials reach only {} of get_comp_dir_ctx's 5 outputs",
            seen5.distinct()
        );
    }

    /// The LEFT smooth-flag helpers must agree with the ABOVE ones they were
    /// split from — same body, different reference type.
    #[test]
    fn sm_flag_left_matches_sm_flag() {
        use crate::src::ipred_prepare::sm_flag;
        use crate::src::ipred_prepare::sm_flag_left;
        use crate::src::ipred_prepare::sm_uv_flag;
        use crate::src::ipred_prepare::sm_uv_flag_left;
        let mut rng = Rng(0x5eed_1eaf_c0ff_ee02);
        let mut nonzero = 0;
        let mut zero = 0;
        for _ in 0..2_000 {
            let mut b = random_ctx(&mut rng);
            for i in 0..32 {
                let want = sm_flag(&b, i);
                assert_eq!(sm_flag_left(&mut b, i), want);
                let want_uv = sm_uv_flag(&b, i);
                assert_eq!(sm_uv_flag_left(&mut b, i), want_uv);
                if want | want_uv != 0 {
                    nonzero += 1;
                } else {
                    zero += 1;
                }
            }
        }
        assert!(
            nonzero > 0 && zero > 0,
            "only one branch ran: {nonzero} / {zero}"
        );
    }
}
