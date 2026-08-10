#!/usr/bin/env python3
"""Three-arm comparison: main @ b0a00c3 vs main @ 2fae4fe (#482) vs dav1d.

Rows are LOAD-TAGGED (foreign_max is printed). Absolute ms are inflated by the
other agent's job; the paired within-round ratios are the usable statistic and
are printed with their bands.
"""
import sys
from collections import defaultdict
rows = []
for line in open(sys.argv[1]):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 9:
        rows.append(p)
per = defaultdict(list)
for p in rows:
    per[(p[1], p[2], int(p[0]))].append((float(p[7]) - float(p[5])) / (int(p[6]) - int(p[4])))
def med(x):
    s = sorted(x); n = len(s)
    return s[n//2] if n % 2 else .5*(s[n//2-1]+s[n//2])
cell = {k: med(v) for k, v in per.items()}
rounds = sorted({r for (_, _, r) in cell})
fmax = max(int(p[8]) for p in rows)
vecs = ["L512x288_420_8b", "L1024x576_420_8b", "L2048x1152_420_8b",
        "L3840x2160_420_8b", "L1024x576_420_10b", "L3840x2160_420_10b"]
print(f"rows={len(rows)}  foreign_max={fmax}  LOAD-TAGGED (absolutes inflated; ratios are paired within a round)")
print(f"{'cell':<22}{'rs ms':>9}{'rs2 ms':>9}{'dav1d':>9} | {'rs/dav':>7}{'rs2/dav':>8} | "
      f"{'rs2/rs med':>11} {'[min..max]':>17} n")
for v in vecs:
    def g(a):
        return [cell[(a, v, r)] for r in rounds if (a, v, r) in cell]
    A, B, D = g('rs'), g('rs2'), g('dav1d_fd1')
    if not A or not B or not D:
        continue
    pr = [cell[('rs2', v, r)] / cell[('rs', v, r)] for r in rounds
          if ('rs2', v, r) in cell and ('rs', v, r) in cell]
    ra = [cell[('rs', v, r)] / cell[('dav1d_fd1', v, r)] for r in rounds
          if ('rs', v, r) in cell and ('dav1d_fd1', v, r) in cell]
    rb = [cell[('rs2', v, r)] / cell[('dav1d_fd1', v, r)] for r in rounds
          if ('rs2', v, r) in cell and ('dav1d_fd1', v, r) in cell]
    print(f"{v:<22}{med(A):9.3f}{med(B):9.3f}{med(D):9.3f} | {med(ra):7.3f}{med(rb):8.3f} | "
          f"{med(pr):11.4f} [{min(pr):.4f}..{max(pr):.4f}] {len(pr)}")
