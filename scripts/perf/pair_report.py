#!/usr/bin/env python3
"""Paired base-vs-head report over a `content_sweep.sh` TSV.

Prints, per cell: each arm's median ms/frame WITH its min/max band, the
per-round paired ratio (median + band), how many rounds head won, a two-sided
sign-test p, and whether the two arms' ms/frame bands are DISJOINT.

Three rules this encodes, each of which cost the campaign a round:

* the disjointness tick must compare the arms the CLAIM compares (base vs head),
  never ours-vs-dav1d, which for two different decoders can never fail;
* a median without a band is not a result on a loaded box;
* below n=5 nothing is called disjoint at all, because a "band" over two points
  is a line segment and prints as disjoint by accident.

Usage: pair_report.py <sweep.tsv> [base_arm] [head_arm]
Columns: round arm vec threads nlo ms_lo nhi ms_hi f_arm f_grp
"""

import sys
from collections import defaultdict
from math import comb
from statistics import median

N_FLOOR = 5


def sign_p(wins, n):
    """Two-sided sign test over the n rounds that were not exact ties."""
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "base"
    head = sys.argv[3] if len(sys.argv) > 3 else "head"

    b = {}
    foreign = defaultdict(int)
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 10:
            continue
        rnd, arm, vec, _t, nlo, mlo, nhi, mhi, f_arm, f_grp = f[:10]
        if int(nhi) == int(nlo):
            continue
        b[(int(rnd), arm, vec)] = (int(mhi) - int(mlo)) / (int(nhi) - int(nlo))
        foreign[vec] = max(foreign[vec], int(f_arm), int(f_grp))

    vecs = sorted({v for (_r, _a, v) in b})
    print(f"{'cell':24s} {'n':>2s} {'base ms/f':>10s} {'base band':>19s} "
          f"{'head ms/f':>10s} {'head band':>19s} {'head/base':>9s} "
          f"{'ratio band':>17s} {'win':>5s} {'p':>6s} {'DJ':>3s} {'f':>2s}")
    for v in vecs:
        rs = sorted({r for (r, a, vv) in b if vv == v and a in (base, head)
                     if (r, base, v) in b and (r, head, v) in b})
        if not rs:
            continue
        bb = [b[(r, base, v)] for r in rs]
        hh = [b[(r, head, v)] for r in rs]
        ratios = [h / x for h, x in zip(hh, bb) if x > 0]
        wins = sum(1 for h, x in zip(hh, bb) if h < x)
        ties = sum(1 for h, x in zip(hh, bb) if h == x)
        n = len(rs)
        dj = "-"
        if n >= N_FLOOR:
            dj = "DJ" if (max(hh) < min(bb) or max(bb) < min(hh)) else "no"
        print(f"{v:24s} {n:2d} {median(bb):10.4f} [{min(bb):8.4f}..{max(bb):8.4f}] "
              f"{median(hh):10.4f} [{min(hh):8.4f}..{max(hh):8.4f}] "
              f"{median(ratios):9.4f} [{min(ratios):6.4f}..{max(ratios):6.4f}] "
              f"{wins:2d}/{n:<2d} {sign_p(wins, n - ties):6.3f} {dj:>3s} {foreign[v]:2d}")
    if any(len({r for (r, a, vv) in b if vv == v}) < N_FLOOR for v in vecs):
        print(f"\nNOTE: cells with fewer than n={N_FLOOR} rounds print DJ as '-' "
              f"and no disjointness is claimed for them.")


if __name__ == "__main__":
    main()
