#!/usr/bin/env python3
"""Aggregate decodes/second from concurrent_throughput.sh.

throughput(N) = N * (n_hi - n_lo) / (ms_hi - ms_lo) * 1000

The two-point difference removes the fork/exec storm and per-process startup,
so what is left is steady-state aggregate decode rate. Ratios are paired within
a round; min/max bands are printed so a small difference can be checked against
its own spread.

Usage: concurrent_throughput_report.py <out.tsv>
"""

import sys
from collections import defaultdict


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    rows = []
    for line in open(sys.argv[1]):
        p = line.rstrip("\n").split("\t")
        if len(p) < 9:
            continue
        rows.append(dict(round=int(p[0]), arm=p[1], vec=p[2], np=int(p[3]),
                         nlo=int(p[4]), lo=float(p[5]), nhi=int(p[6]),
                         hi=float(p[7]), foreign=int(p[8])))
    loaded = sum(1 for r in rows if r["foreign"] > 0)
    print(f"rows={len(rows)}  rows_under_foreign_load={loaded}  vec={rows[0]['vec']}")
    tp = defaultdict(float)
    for r in rows:
        dt = r["hi"] - r["lo"]
        tp[(r["arm"], r["np"], r["round"])] = r["np"] * (r["nhi"] - r["nlo"]) / dt * 1000.0
    nps = sorted({np for (_, np, _) in tp})
    rounds = sorted({rd for (_, _, rd) in tp})
    arms = sorted({a for (a, _, _) in tp})
    ours = "rs"
    ref = "dav1d_fd1" if "dav1d_fd1" in arms else [a for a in arms if a != ours][0]
    base_o = base_d = None
    print(f"\n{'N':>3} | {'ours dec/s':>11} {'[min..max]':>19} | {'dav1d dec/s':>11} "
          f"{'[min..max]':>19} | {'ratio d/o':>9} | {'ours scal':>9} {'dav1d scal':>10}")
    for np in nps:
        o = [tp[(ours, np, rd)] for rd in rounds if (ours, np, rd) in tp]
        d = [tp[(ref, np, rd)] for rd in rounds if (ref, np, rd) in tp]
        pr = [tp[(ref, np, rd)] / tp[(ours, np, rd)] for rd in rounds
              if (ours, np, rd) in tp and (ref, np, rd) in tp]
        om, dm = median(o), median(d)
        if base_o is None:
            base_o, base_d = om, dm
        print(f"{np:>3} | {om:11.2f} [{min(o):8.2f}..{max(o):8.2f}] | {dm:11.2f} "
              f"[{min(d):8.2f}..{max(d):8.2f}] | {median(pr):9.3f} | "
              f"{om/base_o:9.2f}x {dm/base_d:9.2f}x")


if __name__ == "__main__":
    main()
