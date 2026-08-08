#!/usr/bin/env python3
"""Bucket a `sample_selftime.py` table into entropy / tracker / kernels / other.

Wall clock alone cannot answer "did the tracker's share actually move?" -- a
change that makes the whole decode 6% faster by shrinking one bucket looks
identical, from the outside, to one that shrank a different bucket. This groups
self-time leaves by what they are, so the two are distinguishable.

Buckets, most specific rule first:
  tracker   the borrow tracker and its lock (DisjointMut, BorrowTracker, add /
            remove / add_wide / TinyLock, and the zerocopy guard casts that
            exist only to wrap it)
  entropy   the symbol-decoder side: msac, decode_coefs, CDF work
  kernels   the pixel work: itx, cdef, loopfilter, mc, ipred, lr, filmgrain,
            and the safe_simd modules
  runtime   allocator / libsystem / kernel
  other     everything left, printed individually above a threshold so a
            mis-bucketed hot symbol cannot hide inside a total

Usage: bucket_selftime.py <selftime.tsv> [label]
Input is `sample_selftime.py`'s output: "<samples>\\t<symbol>" lines (a leading
percentage column is tolerated).
"""

import re
import sys

RULES = [
    ("tracker", re.compile(
        r"DisjointMut|BorrowTracker|TinyLock|ShardGuard|disjoint_mut|"
        r"cast_slice|slice_as|borrow_id|ShardRecs|wide_probe", re.I)),
    ("entropy", re.compile(
        r"msac|decode_coefs|cdf|read_coef|decode_symbol|bool_adapt|"
        r"golomb|decode_b\b|read_mv", re.I)),
    ("kernels", re.compile(
        r"itx|inv_txfm|cdef|loopfilter|lpf|loop_filter|\bmc_|put_|prep_|"
        r"ipred|intra_pred|lr_|looprestor|selfguided|wiener|filmgrain|"
        r"film_grain|safe_simd|recon_b|blend|warp|emu_edge|sgr", re.I)),
    ("runtime", re.compile(
        r"libsystem|malloc|free|memcpy|memset|bzero|_platform|nanov2|"
        r"pthread|kernel|mach_|szone|calloc|realloc", re.I)),
]


def bucket(sym):
    for name, rx in RULES:
        if rx.search(sym):
            return name
    return "other"


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    tot = 0
    per = {}
    syms = {}
    for line in open(path, errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 2:
            continue
        # tolerate a leading pct column
        try:
            n = float(f[0].rstrip("%"))
        except ValueError:
            continue
        sym = f[-1].strip()
        if not sym:
            continue
        b = bucket(sym)
        per[b] = per.get(b, 0.0) + n
        syms.setdefault(b, []).append((n, sym))
        tot += n
    if tot == 0:
        print(f"{label}: no samples parsed from {path}")
        return
    print(f"== {label}   total self samples {tot:.0f}")
    for b in ["entropy", "tracker", "kernels", "runtime", "other"]:
        v = per.get(b, 0.0)
        print(f"   {b:<9} {v:>10.0f}  {100.0*v/tot:>6.2f}%")
    # Show the biggest 'other' leaves so a mis-bucket cannot hide in a total.
    for n, s in sorted(syms.get("other", []), reverse=True)[:8]:
        if 100.0 * n / tot >= 0.5:
            print(f"      other>0.5%: {100.0*n/tot:>5.2f}%  {s[:96]}")


if __name__ == "__main__":
    main()
