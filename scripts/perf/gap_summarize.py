#!/usr/bin/env python3
"""Summarise a verify_gap.sh / verify_gap_ivf.sh TSV into ms/frame per cell.

Each row is one (round, arm, vec, threads) two-point timing: `nlo` frames took
`ms_lo`, `nhi` took `ms_hi`. `beta = (ms_hi - ms_lo) / (nhi - nlo)` is ms/frame
with process startup fitted out. Per cell we report the MEDIAN beta over rounds
plus the min/max, because an overlapping [min,max] between two arms means the
delta is not resolvable at this round count and must not be claimed.

Usage: gap_summarize.py <tsv> [--ref <arm>] [--base <arm>] [--head <arm>]
"""

import sys
from collections import defaultdict


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    path = sys.argv[1]
    args = sys.argv[2:]

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    ref = opt("--ref", "dav1d_fd1")
    base = opt("--base", "base")
    head = opt("--head", "comp")

    betas = defaultdict(list)
    order = []
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 8:
            continue
        _round, arm, vec, threads, nlo, ms_lo, nhi, ms_hi = f[:8]
        beta = (float(ms_hi) - float(ms_lo)) / (int(nhi) - int(nlo))
        key = (vec, int(threads))
        if key not in order:
            order.append(key)
        betas[(key, arm)].append(beta)

    arms = sorted({a for (_k, a) in betas})
    print(f"# {path}")
    print(f"# arms: {' '.join(arms)}   n per cell: "
          f"{min(len(v) for v in betas.values())}..{max(len(v) for v in betas.values())}")
    hdr = f"{'vector':<26}{'t':>3}  " + "".join(f"{a:>12}" for a in arms)
    hdr += f"{head+'/'+base:>14}{head+'/'+ref:>14}{base+'/'+ref:>14}"
    print(hdr)
    print("-" * len(hdr))
    for key in order:
        vec, t = key
        row = f"{vec:<26}{t:>3}  "
        med = {}
        for a in arms:
            v = betas.get((key, a))
            if not v:
                row += f"{'-':>12}"
                continue
            med[a] = median(v)
            row += f"{med[a]:>12.2f}"
        def rat(x, y):
            return f"{med[x]/med[y]:>14.3f}" if x in med and y in med and med[y] else f"{'-':>14}"
        row += rat(head, base) + rat(head, ref) + rat(base, ref)
        print(row)

    print()
    print("Per-cell ranges (min..max over rounds) — overlapping ranges between two")
    print("arms mean the delta is NOT resolvable and must not be claimed.")
    print(f"{'vector':<26}{'t':>3}  {'arm':<12}{'median':>10}{'min':>10}{'max':>10}{'n':>4}")
    for key in order:
        for a in arms:
            v = betas.get((key, a))
            if not v:
                continue
            print(f"{key[0]:<26}{key[1]:>3}  {a:<12}{median(v):>10.2f}"
                  f"{min(v):>10.2f}{max(v):>10.2f}{len(v):>4}")


if __name__ == "__main__":
    main()
