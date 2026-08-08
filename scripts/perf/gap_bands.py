#!/usr/bin/env python3
"""Per-arm min/median/max bands from a verify_gap.sh TSV.

`verify_gap_report.py` prints medians and ratios. That is not enough to believe
a small delta: one campaign reported `88.0 -> 85.6` (2.7%) whose own raw rows
were base [85.50..91.11] against head [84.89..91.50] at n=5 — fully
overlapping, and a re-measure said null. So this prints the BAND next to every
median and says, per cell, whether the two arms' ranges are disjoint.

Usage: gap_bands.py <tsv> [base_arm] [head_arm] [ref_arm]
Rows whose foreign-load column is non-zero are counted and reported separately;
pass ONLY_IDLE=1 in the environment to drop them.
"""
import os
import sys
from collections import defaultdict
from statistics import median

path = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "base"
head = sys.argv[3] if len(sys.argv) > 3 else "head"
ref = sys.argv[4] if len(sys.argv) > 4 else "dav1d_fd1"
only_idle = os.environ.get("ONLY_IDLE") == "1"

betas = defaultdict(list)
loaded = defaultdict(int)
for line in open(path):
    f = line.rstrip("\n").split("\t")
    if len(f) < 8:
        continue
    rnd, arm, vec, t, nlo, lo, nhi, hi = f[:8]
    fmax = int(f[8]) if len(f) > 8 else 0
    if fmax:
        loaded[(vec, int(t))] += 1
        if only_idle:
            continue
    betas[(vec, int(t), arm)].append((int(hi) - int(lo)) / (int(nhi) - int(nlo)))

cells = sorted({(k[0], k[1]) for k in betas})
arms = sorted({k[2] for k in betas}, key=lambda a: (a.startswith("dav1d"), a))

print("ms/frame, median [min..max] over rounds (two-point wall fit)\n")
hdr = f"{'vector':16} {'t':>2}  " + "".join(f"{a:>26}" for a in arms) + f" {'n':>3} {'ld':>3}"
print(hdr)
print("-" * len(hdr))
for vec, t in cells:
    row = f"{vec:16} {t:>2}  "
    n = 0
    for a in arms:
        v = betas.get((vec, t, a))
        if v:
            row += f"{median(v):>10.1f} [{min(v):6.1f}..{max(v):6.1f}]"
            n = max(n, len(v))
        else:
            row += f"{'-':>26}"
    print(row + f" {n:>3} {loaded.get((vec, t), 0):>3}")

print(f"\n{head} vs {base}, and both vs {ref}\n")
hdr2 = (
    f"{'vector':16} {'t':>2}  {'head/base':>9} {'bands':>9}  "
    f"{'base/ref':>8} {'head/ref':>8}  {'ms short of 1.30x ref':>22}"
)
print(hdr2)
print("-" * len(hdr2))
for vec, t in cells:
    b = betas.get((vec, t, base))
    h = betas.get((vec, t, head))
    r = betas.get((vec, t, ref))
    if not (b and h):
        continue
    bm, hm = median(b), median(h)
    disjoint = "disjoint" if (max(h) < min(b) or max(b) < min(h)) else "OVERLAP"
    line = f"{vec:16} {t:>2}  {hm / bm:>9.4f} {disjoint:>9}  "
    if r:
        rm = median(r)
        line += f"{bm / rm:>8.3f} {hm / rm:>8.3f}  {hm - 1.30 * rm:>21.1f}"
    print(line)
