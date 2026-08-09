#!/usr/bin/env python3
"""Tail-oriented A/B report over a verify_gap.sh TSV.

The question this answers is NOT "did the median move". A contended-lock change
whose contention rate is ~0.02% cannot move a median; what it can do is remove
the rare catastrophic episode that makes a fat tail. So every cell is reported
as median / p90 / max / band width, per arm, plus the head/base ratio of each.

Columns of the input (verify_gap.sh):
  round arm vec threads nlo ms_lo nhi ms_hi foreign_max

Two derived series per cell:
  beta   = (ms_hi - ms_lo) / (nhi - nlo)   ms/frame, startup removed
  raw_hi = ms_hi                            wall of the NHI-frame run

beta is the headline instrument, but a tail event landing in the 2-frame `lo`
run DEPRESSES beta, so raw_hi is reported alongside as the un-differenced view.
"""

import sys
from collections import defaultdict


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main(path, arms=("base", "head")):
    cells = defaultdict(lambda: defaultdict(list))  # (vec,t) -> arm -> [(beta, hi)]
    foreign = 0
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        _r, arm, vec, t, nlo, lo, nhi, hi, fm = f[:9]
        nlo, lo, nhi, hi = int(nlo), int(lo), int(nhi), int(hi)
        foreign = max(foreign, int(fm))
        beta = (hi - lo) / (nhi - nlo)
        cells[(vec, int(t))][arm].append((beta, hi))

    print(f"# rows with foreign_max>0 anywhere: max foreign seen = {foreign}")
    print()
    hdr = (
        "vec\tt\tarm\tn\tmed\tp90\tmax\tmin\tband\t"
        "band%med\trawhi_med\trawhi_p90\trawhi_max"
    )
    print(hdr)
    rows = {}
    for (vec, t) in sorted(cells, key=lambda k: (k[0], k[1])):
        for arm in arms:
            v = cells[(vec, t)].get(arm, [])
            if not v:
                continue
            b = [x[0] for x in v]
            h = [float(x[1]) for x in v]
            med, p90, mx, mn = pct(b, 0.5), pct(b, 0.9), max(b), min(b)
            rows[(vec, t, arm)] = (med, p90, mx, mn, len(b))
            print(
                f"{vec}\t{t}\t{arm}\t{len(b)}\t{med:.2f}\t{p90:.2f}\t{mx:.2f}\t{mn:.2f}\t"
                f"{mx-mn:.2f}\t{100*(mx-mn)/med:.1f}%\t"
                f"{pct(h,0.5):.0f}\t{pct(h,0.9):.0f}\t{max(h):.0f}"
            )
    print()
    print("vec\tt\tn\tmed_b\tmed_h\tr_med\tp90_b\tp90_h\tr_p90\tmax_b\tmax_h\tr_max\tband_b\tband_h\tr_band")
    for (vec, t) in sorted(cells, key=lambda k: (k[0], k[1])):
        if (vec, t, "base") not in rows or (vec, t, "head") not in rows:
            continue
        bm, bp, bx, bn, n = rows[(vec, t, "base")]
        hm, hp, hx, hn, _ = rows[(vec, t, "head")]
        bb, hb = bx - bn, hx - hn
        print(
            f"{vec}\t{t}\t{n}\t{bm:.2f}\t{hm:.2f}\t{hm/bm:.3f}\t"
            f"{bp:.2f}\t{hp:.2f}\t{hp/bp:.3f}\t"
            f"{bx:.2f}\t{hx:.2f}\t{hx/bx:.3f}\t"
            f"{bb:.2f}\t{hb:.2f}\t{(hb/bb if bb else float('nan')):.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
