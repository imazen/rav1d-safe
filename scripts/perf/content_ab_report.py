#!/usr/bin/env python3
"""Report content_ab.sh: per-cell paired ratio with the noise band printed.

Prints median AND [min..max] for both arms, the paired per-round ratio median,
how many rounds moved which way, and a two-sided sign-test p. A sub-3% headline
whose arms' [min..max] bands OVERLAP is not a result — that mistake was made
once in this campaign already, so the bands are not optional output.

Also cross-checks the md5 column: every arm must agree on every cell, or the
timing is comparing two different decodes.

Usage: content_ab_report.py <ab.tsv> [base_arm] [head_arm]
"""
import sys
import csv
from collections import defaultdict
from math import comb

path = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "base"
head = sys.argv[3] if len(sys.argv) > 3 else "head"

rows = list(csv.DictReader(open(path), delimiter="\t"))
by = defaultdict(dict)          # (vec,t) -> arm -> {round: ms}
md5 = defaultdict(set)          # (vec,t) -> {md5}
foreign_max = 0
for r in rows:
    key = (r["vec"], int(r["threads"]))
    by[key].setdefault(r["arm"], {})[int(r["round"])] = float(r["ms_per_frame"])
    md5[key].add(r["md5"])
    foreign_max = max(foreign_max, int(r["foreign"]))

bad = {k: v for k, v in md5.items() if len(v) != 1}
if bad:
    print("MD5 DISAGREEMENT — arms are not decoding the same thing:")
    for k, v in bad.items():
        print("   ", k, sorted(v))
    sys.exit(2)


def sign_p(better, worse):
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2 * tail)


print(f"file={path}  foreign_max={foreign_max}  "
      f"{'LOADED — ratios only' if foreign_max else 'box idle (no foreign >25% CPU seen)'}")
print()
hdr = (f"{'vec':<22}{'t':>2}  {base+' med':>9}{'[min..max]':>18}  "
       f"{head+' med':>9}{'[min..max]':>18}  {'ratio':>7} {'faster':>7} {'p':>6}")
print(hdr)
print("-" * len(hdr))
for (vec, t), arms in sorted(by.items()):
    if base not in arms or head not in arms:
        continue
    b, h = arms[base], arms[head]
    common = sorted(set(b) & set(h))
    bs = [b[r] for r in common]
    hs = [h[r] for r in common]
    ratios = sorted(h[r] / b[r] for r in common)
    n = len(common)
    med = lambda xs: sorted(xs)[len(xs) // 2] if len(xs) % 2 else \
        (sorted(xs)[len(xs) // 2 - 1] + sorted(xs)[len(xs) // 2]) / 2
    faster = sum(1 for r in common if h[r] < b[r])
    print(f"{vec:<22}{t:>2}  {med(bs):>9.3f}[{min(bs):>7.3f}..{max(bs):>7.3f}]  "
          f"{med(hs):>9.3f}[{min(hs):>7.3f}..{max(hs):>7.3f}]  "
          f"{med(ratios):>7.4f} {faster:>3}/{n:<3} {sign_p(faster, n - faster):>6.3f}")
print()
print("ratio < 1 = head faster. `faster` counts rounds where head beat base on")
print("that cell. Overlapping [min..max] bands with a ratio inside ~1% of 1.0")
print("should be read as null regardless of the sign test.")
