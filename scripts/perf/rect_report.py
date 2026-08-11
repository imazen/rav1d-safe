#!/usr/bin/env python3
"""Report the strided-rectangle round: paired per-round ratios with bands.

Input is `tiled_wallcpu.sh`'s TSV:
    round arm vec threads nlo wall_lo user_lo sys_lo nhi wall_hi user_hi sys_hi foreign

Differences from `c256_report.py`, which this is otherwise a copy of:

* `--drop-loaded` also discards any ROUND in which ANY arm saw a foreign process
  above 25% CPU. A loaded round is shared by every arm in it, so its paired
  ratios survive; its per-round DRIFT does not, and one such round moved a
  median by 0.7% in the first pass of this grid. Dropping the whole round keeps
  the arms paired.
* the base arm defaults to `plain` (this branch's default codegen), and the
  report prints the ratio against a SECOND reference too, because the question
  here is two questions: does the rectangle pay (rect vs machoff), and what does
  its machinery cost (machoff vs plain).

ROUND 0 IS DISCARDED: the first touch of each (arm, cell) pair is cold.

Usage: rect_report.py <tsv> [--base ARM] [--vs ARM] [--drop-round N] [--keep-loaded]
"""

import sys
from collections import defaultdict
from statistics import median


def load(path, drop_round, drop_loaded):
    cells = defaultdict(lambda: defaultdict(dict))
    foreign = defaultdict(int)
    loaded_rounds = defaultdict(set)
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
            loaded_rounds[cell].add(rnd)
        foreign[cell] = max(foreign[cell], fgn)
        d = nhi - nlo
        wall = (whi - wlo) / d
        cpu = ((uhi + shi) - (ulo + slo)) / d
        cells[cell][arm][rnd] = (wall, cpu)
    if drop_loaded:
        for cell in cells:
            for arm in cells[cell]:
                for r in list(cells[cell][arm]):
                    if r in loaded_rounds[cell]:
                        del cells[cell][arm][r]
    return cells, foreign, loaded_rounds


def ratios(cells, cell, arm, base):
    rs = sorted(set(cells[cell][arm]) & set(cells[cell][base]))
    w = [cells[cell][arm][r][0] / cells[cell][base][r][0] for r in rs]
    c = [cells[cell][arm][r][1] / cells[cell][base][r][1] for r in rs]
    return w, c


def main():
    a = sys.argv[1:]
    path = a[0]
    base = a[a.index("--base") + 1] if "--base" in a else "plain"
    vs = a[a.index("--vs") + 1] if "--vs" in a else None
    drop = int(a[a.index("--drop-round") + 1]) if "--drop-round" in a else 1
    drop_loaded = "--keep-loaded" not in a
    cells, foreign, loaded = load(path, drop, drop_loaded)
    order = ["plain", "plainB", "machoff", "base", "rect", "rect1",
             "dbloff", "dblon", "untracked", "dav1d_fd1"]
    for cell in sorted(cells):
        arms = [x for x in order if x in cells[cell]] + \
               [x for x in sorted(cells[cell]) if x not in order]
        n = len(cells[cell].get(base, {}))
        print(f"\n## {cell}   n={n} rounds (round<{drop} discarded, "
              f"loaded rounds {'dropped' if drop_loaded else 'kept'}: "
              f"{sorted(loaded[cell])}), foreign_max={foreign[cell]}")
        hdr = (f"{'arm':10} {'wall':>7} {'cpu':>7} {'wall/'+base:>12} "
               f"{'[min..max]':>17} {'sign':>6} {'cpu/'+base:>11} "
               f"{'[min..max]':>17} {'sign':>6}")
        if vs:
            hdr += f" {'wall/'+vs:>12} {'cpu/'+vs:>11}"
        print(hdr)
        for arm in arms:
            w, c = ratios(cells, cell, arm, base)
            if not w:
                continue
            wall = median(v[0] for v in cells[cell][arm].values())
            cpu = median(v[1] for v in cells[cell][arm].values())
            sw = sum(1 for x in w if x < 1.0)
            sc = sum(1 for x in c if x < 1.0)
            row = (f"{arm:10} {wall:7.3f} {cpu:7.3f} {median(w):12.4f} "
                   f"[{min(w):.4f}..{max(w):.4f}] {sw:>3}/{len(w):<2} "
                   f"{median(c):11.4f} [{min(c):.4f}..{max(c):.4f}] "
                   f"{sc:>3}/{len(c):<2}")
            if vs and vs in cells[cell]:
                w2, c2 = ratios(cells, cell, arm, vs)
                if w2:
                    row += f" {median(w2):12.4f} {median(c2):11.4f}"
            print(row)


if __name__ == "__main__":
    main()
