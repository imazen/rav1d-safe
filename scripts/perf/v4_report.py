#!/usr/bin/env python3
"""Paired per-round ratio report for scripts/perf/v4_ab.sh.

A median of two independent per-arm distributions is the wrong statistic on a
box that drifts: it lets the drift into the numerator and the denominator
separately. This pairs WITHIN a round -- both arms ran back-to-back under the
same load -- and reports the median of the per-round ratios plus their min/max,
which is the spread that decides whether a sub-3% claim is real.

Also set-diffs the md5 by value per cell, so a timed row that changed output
cannot be reported as a speedup.

Usage: v4_report.py <tsv> [baseline_arm]
"""

import sys
from collections import defaultdict


def med(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    path = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "base"
    # (vec, t, round, arm) -> ms ; (vec, t) -> {arm: set(md5)}
    ms = {}
    md5s = defaultdict(lambda: defaultdict(set))
    arms = []
    cells = []
    foreign = defaultdict(int)
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 7:
            continue
        rnd, arm, vec, t, v, h, fo = f[0], f[1], f[2], f[3], f[4], f[5], f[6]
        try:
            v = float(v)
        except ValueError:
            continue
        ms[(vec, t, rnd, arm)] = v
        md5s[(vec, t)][arm].add(h)
        foreign[(vec, t)] = max(foreign[(vec, t)], int(fo))
        if arm not in arms:
            arms.append(arm)
        if (vec, t) not in cells:
            cells.append((vec, t))

    others = [a for a in arms if a != base]
    print(f"{'vector':<18}{'t':>3}  {'arm':<10}{'n':>3} {'ratio':>8} "
          f"{'[min':>8}{'max]':>9}  {'base_ms':>9} {'arm_ms':>9}  md5  foreign")
    for (vec, t) in cells:
        rounds = sorted({r for (v_, t_, r, a) in ms if v_ == vec and t_ == t})
        for arm in others:
            ratios, bs, hs = [], [], []
            for r in rounds:
                b = ms.get((vec, t, r, base))
                h = ms.get((vec, t, r, arm))
                if b and h:
                    ratios.append(h / b)
                    bs.append(b)
                    hs.append(h)
            if not ratios:
                continue
            allh = md5s[(vec, t)][base] | md5s[(vec, t)][arm]
            tag = "IDENTICAL" if len(allh) == 1 else f"DIFFER({len(allh)})"
            print(f"{vec:<18}{t:>3}  {arm:<10}{len(ratios):>3} {med(ratios):>8.4f} "
                  f"{min(ratios):>8.4f}{max(ratios):>9.4f}  {med(bs):>9.2f} {med(hs):>9.2f}"
                  f"  {tag}  {foreign[(vec, t)]}")


if __name__ == "__main__":
    main()
