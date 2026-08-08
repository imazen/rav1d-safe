#!/usr/bin/env python3
"""Summarise scaling_ab.sh output: median/min/max per (vector, threads, arm),
ratio to a baseline arm, and the t1->tN scaling factor per arm.

Prints min/max as well as the median because a sub-3% delta whose arms' raw
ranges overlap is not a result -- that mistake has already been made twice in
this campaign.

Usage: scaling_report.py <tsv> [baseline_arm]
"""
import sys
from collections import defaultdict
from statistics import median

path = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "plain"

cells = defaultdict(list)
order = []
for line in open(path):
    p = line.rstrip("\n").split("\t")
    if len(p) < 6 or not p[5]:
        continue
    rnd, arm, vec, t, iters, ms = p[0], p[1], p[2], int(p[3]), p[4], float(p[5])
    foreign = p[6] if len(p) > 6 else "0"
    cells[(vec, t, arm)].append(ms)
    if arm not in order:
        order.append(arm)

vecs = sorted({k[0] for k in cells})
threads = sorted({k[1] for k in cells})

print(f"{'vec':<16}{'t':>3}  {'arm':<8}{'n':>3}{'median':>10}{'min':>10}{'max':>10}"
      f"{'vs '+base:>10}")
for v in vecs:
    for t in threads:
        b = cells.get((v, t, base))
        bmed = median(b) if b else None
        for arm in order:
            r = cells.get((v, t, arm))
            if not r:
                continue
            m = median(r)
            rel = f"{m/bmed:.4f}" if bmed else "-"
            print(f"{v:<16}{t:>3}  {arm:<8}{len(r):>3}{m:>10.2f}{min(r):>10.2f}"
                  f"{max(r):>10.2f}{rel:>10}")
        print()

print(f"{'vec':<16}{'arm':<8}" + "".join(f"{'t='+str(t):>10}" for t in threads)
      + f"{'scaling':>10}")
for v in vecs:
    for arm in order:
        row = []
        for t in threads:
            r = cells.get((v, t, arm))
            row.append(median(r) if r else None)
        if row[0] and row[-1]:
            sc = f"{row[0]/row[-1]:.3f}x"
        else:
            sc = "-"
        print(f"{v:<16}{arm:<8}"
              + "".join(f"{x:>10.2f}" if x else f"{'-':>10}" for x in row)
              + f"{sc:>10}")
