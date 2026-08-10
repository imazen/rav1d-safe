#!/usr/bin/env python3
"""Per-KERNEL-FAMILY self time, in ms/frame, plus a liveness column.

`bucket_selftime.py` answers "how much is kernels?"; this answers "which
kernel?", at the granularity `src/ablate.rs::Family` uses, which is the
granularity a fix would target. The liveness column is the point: a family with
zero leaf samples did not run in this stream, and per docs/AGENT_BRIEF.md a null
from a vector that never runs the code is not a result. Loop restoration is
switched OFF in both of the campaign's 4K gap vectors while it is active in 696
of 768 corpus vectors, so "did this family execute here?" has to be printed, not
assumed.

Usage: family_selftime.py <selftime.tsv> <ms_per_frame> [label]
"""

import re
import sys

# Most specific first; a symbol lands in the first family that matches.
FAMILIES = [
    ("tracker", r"DisjointMut|BorrowTracker|TinyLock|ShardGuard|disjoint_mut|"
                r"borrow_id|ShardRecs|tracker_shard|wide_probe"),
    ("cast_u16", r"cast_slice|slice_as|zerocopy|try_cast|from_bytes"),
    ("entropy", r"msac|decode_coefs|cdf|read_coef|decode_symbol|bool_adapt|golomb|read_mv"),
    ("itx", r"itx|inv_txfm|inv_dct|inv_adst|inv_identity|inv_wht"),
    ("cdef", r"cdef"),
    ("loopfilter", r"loopfilter|loop_filter|\blpf\b|lf_apply|filter_run"),
    ("looprestor", r"looprestor|selfguided|wiener|\bsgr\b|lr_"),
    ("ipred", r"ipred|intra_pred|cfl|pal_pred|filter_edge|splat_dc"),
    ("mc", r"mc_arm|\bmc_|put_8tap|prep_8tap|blend|warp|emu_edge|avg_|w_mask"),
    ("filmgrain", r"filmgrain|film_grain|grain_"),
    ("recon", r"recon_b|decode_b|read_coef_tree|ipred_prepare|lf_mask|CaseSet"),
    ("runtime", r"libsystem|malloc|free|memcpy|memset|bzero|_platform|nanov2|"
                r"pthread|kernel|mach_|szone|calloc|realloc|madvise|munmap|mmap"),
]
RX = [(n, re.compile(p, re.I)) for n, p in FAMILIES]


def fam(sym):
    for n, rx in RX:
        if rx.search(sym):
            return n
    return "other"


def main():
    path, ms = sys.argv[1], float(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else path
    tot = 0.0
    per = {}
    top = {}
    for line in open(path, errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 2:
            continue
        try:
            n = float(f[0].rstrip("%"))
        except ValueError:
            continue
        sym = f[-1].strip()
        if not sym:
            continue
        k = fam(sym)
        per[k] = per.get(k, 0.0) + n
        top.setdefault(k, []).append((n, sym))
        tot += n
    if tot == 0:
        print(f"{label}: no samples")
        return
    print(f"== {label}   leaves={tot:.0f}   ms/frame={ms:.4f}")
    print(f"   {'family':<12} {'ms/frame':>9} {'%':>7}   live?   top leaf")
    order = sorted(per.items(), key=lambda kv: -kv[1])
    for k, v in order:
        big = sorted(top[k], reverse=True)[0][1][:70]
        print(f"   {k:<12} {v/tot*ms:9.4f} {100*v/tot:6.2f}%   "
              f"{'yes' if v > 0 else 'NO ':<5}   {big}")
    for k, _ in FAMILIES:
        if k not in per:
            print(f"   {k:<12} {0.0:9.4f} {0.0:6.2f}%   NO      <no leaf sample in this stream>")


if __name__ == "__main__":
    main()
