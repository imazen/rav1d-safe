#!/usr/bin/env python3
"""Distribution report over tail_sweep.sh output (per-decode samples).

Columns in: round arm vec threads rep ms foreign

For each (vec, threads) cell and arm: n, median, p90, p99, max, min, band
width, and the share of samples above a per-cell outlier threshold defined as
1.25 x the POOLED median of both arms (so the threshold cannot be moved by the
arm under test).
"""

import sys
from collections import defaultdict


def pct(xs, p):
    s = sorted(xs)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main(path):
    cells = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 7:
            continue
        _r, arm, vec, t, _rep, ms, _fm = f[:7]
        cells[(vec, int(t))][arm].append(float(ms))

    print("vec\tt\tarm\tn\tmed\tp90\tp99\tmax\tmin\tband\tband%\t>1.25x\t>1.5x")
    summ = {}
    for key in sorted(cells, key=lambda k: (k[0], k[1])):
        vec, t = key
        pooled = pct([x for a in cells[key].values() for x in a], 0.5)
        thr125, thr150 = 1.25 * pooled, 1.50 * pooled
        for arm in ("base", "head"):
            v = cells[key].get(arm)
            if not v:
                continue
            med, p90, p99 = pct(v, 0.5), pct(v, 0.9), pct(v, 0.99)
            mx, mn = max(v), min(v)
            o125 = sum(1 for x in v if x > thr125)
            o150 = sum(1 for x in v if x > thr150)
            summ[(vec, t, arm)] = (len(v), med, p90, p99, mx, mn, o125, o150)
            print(
                f"{vec}\t{t}\t{arm}\t{len(v)}\t{med:.2f}\t{p90:.2f}\t{p99:.2f}\t{mx:.2f}\t"
                f"{mn:.2f}\t{mx-mn:.2f}\t{100*(mx-mn)/med:.0f}%\t"
                f"{o125} ({100*o125/len(v):.2f}%)\t{o150} ({100*o150/len(v):.2f}%)"
            )
    print()
    print("vec\tt\tn/arm\tr_med\tr_p90\tr_p99\tr_max\tr_band\tout125 base->head\tout150 base->head")
    for key in sorted(cells, key=lambda k: (k[0], k[1])):
        vec, t = key
        if (vec, t, "base") not in summ or (vec, t, "head") not in summ:
            continue
        nb, bm, bp, b99, bx, bn, bo, bo5 = summ[(vec, t, "base")]
        nh, hm, hp, h99, hx, hn, ho, ho5 = summ[(vec, t, "head")]
        bb, hb = bx - bn, hx - hn
        print(
            f"{vec}\t{t}\t{nb}/{nh}\t{hm/bm:.3f}\t{hp/bp:.3f}\t{h99/b99:.3f}\t"
            f"{hx/bx:.3f}\t{hb/bb:.3f}\t{bo}->{ho}\t{bo5}->{ho5}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
