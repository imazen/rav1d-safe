#!/usr/bin/env python3
"""Summarise tracker_ab.sh output: per (vec, threads, arm) median + min/max.

Prints the ratio of every arm against the FIRST arm named on the command line
(or `base` if present), plus the raw band so a sub-3% claim can be checked
against its own noise -- the brief's rule.

Usage: tracker_ab_report.py <tsv> [ref_arm]
"""
import sys
from collections import defaultdict


def med(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    path = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else "base"
    cells = defaultdict(lambda: defaultdict(list))
    md5s = defaultdict(set)
    order = []
    busy_max = 0
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 6 or not f[4]:
            continue
        _r, arm, vec, t, ms, busy = f[0], f[1], f[2], f[3], float(f[4]), int(f[5])
        md5 = f[6] if len(f) > 6 else ""
        busy_max = max(busy_max, busy)
        cells[(vec, t)][arm].append(ms)
        if md5:
            md5s[(vec, t)].add((arm, md5))
        if arm not in order:
            order.append(arm)
    print(f"# foreign_busy_max={busy_max}")
    print("vec\tthreads\tarm\tn\tmedian\tmin\tmax\tratio_vs_" + ref)
    for (vec, t), arms in sorted(cells.items()):
        base = med(arms[ref]) if ref in arms else None
        for arm in order:
            if arm not in arms:
                continue
            v = arms[arm]
            m = med(v)
            rat = f"{m / base:.4f}" if base else "-"
            print(f"{vec}\t{t}\t{arm}\t{len(v)}\t{m:.2f}\t{min(v):.2f}\t{max(v):.2f}\t{rat}")
    # md5 set-diff BY NAME
    for k, s in sorted(md5s.items()):
        byarm = defaultdict(set)
        for arm, h in s:
            byarm[arm].add(h)
        allh = {h for _a, h in s}
        if len(allh) > 1:
            print(f"!! MD5 MISMATCH {k}: " + "; ".join(f"{a}={sorted(h)}" for a, h in sorted(byarm.items())))


main()
