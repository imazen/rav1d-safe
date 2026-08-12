#!/usr/bin/env python3
"""Reduce a `tiled_wallcpu.sh` TSV to the quantity the alignment round is about:
the SPREAD of the code-placement lottery inside each alignment family.

`docs/RECT_SHIP.md` §3 established that a binary's t=1 wall on `v4k8tile` moves
+1.1%..+1.6% for ANY perturbation of its text layout — dead code, a refactor
that shrinks the hot function, a far module — while a byte-identical copy reads
1.0006. So a t=1 delta attributed to a code change is, by default, a draw from a
±1.5% lottery.

The question here is NOT "which binary is fastest". It is whether forcing
function alignment SHRINKS that spread. So for each family (a0 = no alignment,
a4/a5/a6 = 16/32/64-byte) this prints:

  * every rung's paired ratio against THAT FAMILY's own unpadded build, so the
    families are compared on spread, not on level;
  * `spread` = max(rung medians) − min(rung medians) inside the family, which is
    the number that has to fall for alignment to be worth anything;
  * the family's absolute cost, `a*plain / a0plain`, because a stabiliser that
    costs more than the lottery is not a stabiliser worth having.

Round 0 is discarded (first touch of each (arm, cell) pair is cold) and any
round in which ANY arm saw a foreign process above 25% CPU is dropped whole,
exactly as `rect_report.py` does.

Usage: layout_spread.py <tsv> [--families a0,a4,a5,a6] [--rungs plain,pad1,pad2,pad4]
                        [--drop-round N] [--keep-loaded] [--metric wall|cpu]
"""

import sys
from collections import defaultdict
from statistics import median


def load(path, drop_round, drop_loaded):
    cells = defaultdict(lambda: defaultdict(dict))
    foreign = defaultdict(int)
    loaded = defaultdict(set)
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 13:
            continue
        rnd, arm, vec, thr = int(f[0]), f[1], f[2], f[3]
        nlo, wlo, ulo, slo = int(f[4]), int(f[5]), int(f[6]), int(f[7])
        nhi, whi, uhi, shi = int(f[8]), int(f[9]), int(f[10]), int(f[11])
        fgn = int(f[12])
        if rnd < drop_round:
            continue
        cell = f"{vec}:t{thr}"
        if fgn > 0:
            loaded[cell].add(rnd)
        foreign[cell] = max(foreign[cell], fgn)
        d = nhi - nlo
        cells[cell][arm][rnd] = ((whi - wlo) / d, ((uhi + shi) - (ulo + slo)) / d)
    if drop_loaded:
        for cell in cells:
            for arm in cells[cell]:
                for r in list(cells[cell][arm]):
                    if r in loaded[cell]:
                        del cells[cell][arm][r]
    return cells, foreign, loaded


def paired(cells, cell, arm, base, k):
    rs = sorted(set(cells[cell][arm]) & set(cells[cell][base]))
    return [cells[cell][arm][r][k] / cells[cell][base][r][k] for r in rs]


def main():
    a = sys.argv[1:]
    path = a[0]
    fams = (a[a.index("--families") + 1] if "--families" in a else "a0,a4,a5,a6").split(",")
    rungs = (a[a.index("--rungs") + 1] if "--rungs" in a else "plain,pad1,pad2,pad4").split(",")
    drop = int(a[a.index("--drop-round") + 1]) if "--drop-round" in a else 1
    k = 1 if ("--metric" in a and a[a.index("--metric") + 1] == "cpu") else 0
    cells, foreign, loaded = load(path, drop, "--keep-loaded" not in a)
    metric = "cpu" if k else "wall"

    for cell in sorted(cells):
        present = set(cells[cell])
        print(f"\n## {cell}   metric={metric}  (round<{drop} dropped, loaded rounds "
              f"{sorted(loaded[cell])} dropped, foreign_max={foreign[cell]})")
        print(f"{'family':7} {'vs own plain: ' + ' '.join(f'{r:>9}' for r in rungs)}"
              f" {'SPREAD':>8} {'plain/a0plain':>14} {'n':>4}")
        for fam in fams:
            base = f"{fam}plain"
            if base not in present:
                continue
            meds, cols = [], []
            for r in rungs:
                arm = f"{fam}{r}"
                if arm not in present:
                    cols.append(f"{'—':>9}")
                    continue
                v = paired(cells, cell, arm, base, k)
                if not v:
                    cols.append(f"{'—':>9}")
                    continue
                m = median(v)
                meds.append(m)
                cols.append(f"{m:9.4f}")
            spread = (max(meds) - min(meds)) if len(meds) > 1 else float("nan")
            abs_v = paired(cells, cell, base, f"{fams[0]}plain", k)
            abs_m = median(abs_v) if abs_v else float("nan")
            n = len(cells[cell][base])
            print(f"{fam:7} {'              ' + ' '.join(cols)} {spread * 100:7.2f}% "
                  f"{abs_m:14.4f} {n:4}")

        # Per-arm detail, so a spread is never quoted without its signs.
        print(f"  {'arm':10} {'ratio/own plain':>16} {'[min..max]':>19} {'sign':>7}")
        for fam in fams:
            base = f"{fam}plain"
            if base not in present:
                continue
            for r in rungs + ["B", "rect"]:
                arm = f"{fam}{r}"
                if arm not in present or arm == base:
                    continue
                v = paired(cells, cell, arm, base, k)
                if not v:
                    continue
                s = sum(1 for x in v if x < 1.0)
                print(f"  {arm:10} {median(v):16.4f} "
                      f"[{min(v):.4f}..{max(v):.4f}] {s:>3}/{len(v):<3}")


if __name__ == "__main__":
    main()
