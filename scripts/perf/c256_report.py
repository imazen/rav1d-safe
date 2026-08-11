#!/usr/bin/env python3
"""Report the c256x2048 contention round: paired per-round ratios with bands.

Input is `tiled_wallcpu.sh`'s TSV:
    round arm vec threads nlo wall_lo user_lo sys_lo nhi wall_hi user_hi sys_hi foreign

Per (round, arm, cell) a two-point fit `total = a + b*frames` gives ms/frame for
wall and for CPU (user+sys), so process startup, mmap and decoder construction
drop out of both.

Ratios are PAIRED per round against the base arm and only then reduced to a
median, because a round's drift is shared by every arm in it and cancels in a
paired ratio but not in a ratio of medians (that is worth several points of band
width on these cells). `sign` counts how many rounds fall on the same side of
1.000 — a median inside the noise floor with 4/7 rounds is a null however
pretty it looks.

ROUND 0 IS DISCARDED: the first touch of each (arm, cell) pair is cold.

Usage: c256_report.py <tsv> [--base ARM] [--drop-round N]
"""

import sys
from collections import defaultdict
from statistics import median


def load(path, drop_round):
    # (cell, arm) -> {round: (wall_ms_per_frame, cpu_ms_per_frame)}, plus foreign
    cells = defaultdict(lambda: defaultdict(dict))
    foreign = defaultdict(int)
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
        dn = nhi - nlo
        if dn <= 0:
            continue
        wall = (whi - wlo) / dn
        cpu = ((uhi + shi) - (ulo + slo)) / dn
        cell = f"{vec}:t{thr}"
        cells[cell][arm][rnd] = (wall, cpu)
        foreign[cell] = max(foreign[cell], fgn)
    return cells, foreign


def band(xs):
    return min(xs), max(xs)


def main():
    args = sys.argv[1:]
    path = args[0]
    base = "plain"
    drop = 1
    i = 1
    while i < len(args):
        if args[i] == "--base":
            base = args[i + 1]
            i += 2
        elif args[i] == "--drop-round":
            drop = int(args[i + 1])
            i += 2
        else:
            i += 1

    cells, foreign = load(path, drop)
    for cell in sorted(cells):
        arms = cells[cell]
        if base not in arms:
            print(f"## {cell}: no base arm {base!r}", file=sys.stderr)
            continue
        rounds = sorted(arms[base])
        print(f"\n## {cell}   n={len(rounds)} rounds (round<{drop} discarded), "
              f"foreign_max={foreign[cell]}")
        print("arm\twall_ms/f\t[min..max]\tcpu_ms/f\twall_ratio\t[min..max]\t"
              "sign\tcpu_ratio\tcpu_sign")
        for arm in arms:
            common = [r for r in rounds if r in arms[arm]]
            if not common:
                continue
            w = [arms[arm][r][0] for r in common]
            c = [arms[arm][r][1] for r in common]
            wr = [arms[arm][r][0] / arms[base][r][0] for r in common]
            cr = [arms[arm][r][1] / arms[base][r][1] for r in common]
            lo, hi = band(wr)
            below_w = sum(1 for x in wr if x < 1.0)
            below_c = sum(1 for x in cr if x < 1.0)
            sw = f"{max(below_w, len(wr) - below_w)}/{len(wr)}"
            sc = f"{max(below_c, len(cr) - below_c)}/{len(cr)}"
            print(f"{arm}\t{median(w):.3f}\t[{min(w):.3f}..{max(w):.3f}]\t"
                  f"{median(c):.3f}\t{median(wr):.4f}\t[{lo:.4f}..{hi:.4f}]\t"
                  f"{sw}\t{median(cr):.4f}\t{sc}")


if __name__ == "__main__":
    main()
