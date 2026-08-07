#!/usr/bin/env python3
"""Report ms/frame and the ratio to dav1d from a verify_gap.sh TSV.

Per round and cell, `beta = (ms_hi - ms_lo) / (nhi - nlo)` fits out the process
startup intercept; the reported figure is the MEDIAN beta across rounds, so one
slow round cannot move it. `n` is the round count that actually committed.

Usage: verify_gap_report.py <tsv> [ref_arm]
"""
import sys
from collections import defaultdict
from statistics import median

path = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else "dav1d_fd1"

betas = defaultdict(list)
foreign = defaultdict(list)
for line in open(path):
    f = line.rstrip("\n").split("\t")
    if len(f) < 8:
        continue
    rnd, arm, vec, t, nlo, lo, nhi, hi = f[:8]
    fmax = int(f[8]) if len(f) > 8 else 0
    b = (int(hi) - int(lo)) / (int(nhi) - int(nlo))
    betas[(vec, int(t), arm)].append(b)
    foreign[(vec, int(t))].append(fmax)

arms = sorted({k[2] for k in betas}, key=lambda a: (a.startswith("dav1d"), a))
cells = sorted({(k[0], k[1]) for k in betas})

w = max(len(a) for a in arms) + 2
print(f"ms/frame (median of n rounds; alpha+beta*frames fitted, wall clock)\n")
hdr = f"{'vector':16} {'t':>3} " + "".join(f"{a:>{w}}" for a in arms) + f"  {'n':>3} {'foreign':>7}"
print(hdr)
print("-" * len(hdr))
for vec, t in cells:
    row = f"{vec:16} {t:>3} "
    n = 0
    for a in arms:
        v = betas.get((vec, t, a))
        row += f"{median(v):>{w}.1f}" if v else f"{'-':>{w}}"
        n = max(n, len(v or []))
    print(row + f"  {n:>3} {max(foreign[(vec,t)]):>7}")

print(f"\nratio to {ref}\n")
print(hdr)
print("-" * len(hdr))
for vec, t in cells:
    r = betas.get((vec, t, ref))
    if not r:
        continue
    rm = median(r)
    row = f"{vec:16} {t:>3} "
    for a in arms:
        v = betas.get((vec, t, a))
        row += f"{median(v)/rm:>{w}.2f}" if v else f"{'-':>{w}}"
    print(row + f"  {len(r):>3}")

print("\nt=1 -> t=8 scaling\n")
for vec in sorted({c[0] for c in cells}):
    parts = []
    for a in arms:
        v1 = betas.get((vec, 1, a))
        v8 = betas.get((vec, 8, a))
        if v1 and v8:
            parts.append(f"{a} {median(v1)/median(v8):.2f}x")
    print(f"  {vec:16} " + "   ".join(parts))

print("\nt=4 -> t=8 (a value > 1.0 means t=8 is SLOWER than t=4: the inversion)\n")
for vec in sorted({c[0] for c in cells}):
    parts = []
    for a in arms:
        v4 = betas.get((vec, 4, a))
        v8 = betas.get((vec, 8, a))
        if v4 and v8:
            parts.append(f"{a} {median(v8)/median(v4):.3f}")
    print(f"  {vec:16} " + "   ".join(parts))
