#!/usr/bin/env python3
"""Turn the measured t>1 speedup on a SINGLE-TILE stream into the stage split it
implies, so the "two concurrently-runnable tasks" reading of the source is
checked against a number instead of asserted.

On a one-tile frame rav1d-safe has exactly two runnable tasks (source:
`rav1d_task_create_tile_sbrow` enqueues `tiling.cols * tiling.rows` = 1, and
`create_filter_sbrow` enqueues 1). That is a two-stage pipeline, not a
data-parallel decomposition, so the model is

    wall(t=1)  = A + B                 (tile decode, then the filter chain)
    wall(t>=2) = max(A, B) + lag       (they overlap; the filter chain trails
                                        the tile task by >= 1 superblock row)

which gives a speedup ceiling of (A+B)/max(A,B) <= 2, REGARDLESS of thread
count. Inverting it against the measured speedup S yields the implied minority
stage share:

    B/(A+B) = 1 - 1/S      (when A >= B)

If that lands near the filter families' measured share of decode time, the
pipeline reading is corroborated by two independent instruments. If it lands far
above, something other than the filter chain is overlapping and the source
reading is incomplete -- which is the outcome worth knowing.

Usage: size_sweep_t8_amdahl.py <report.tsv>   (the --tsv output of the report)
"""

import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    rows = []
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    by = defaultdict(dict)
    for r in rows:
        if int(r["tiles"]) != 1:
            continue
        by[(r["vec"], r["arm"])][int(r["threads"])] = r

    print("Implied two-stage split on the SINGLE-TILE ladder")
    print("  S = wall(t=1)/wall(t), ceiling for a 2-stage pipeline is (A+B)/max(A,B)")
    print(f"{'vector':<22} {'arm':<10} {'t':>2} {'S':>6} {'implied minority stage':>23} "
          f"{'cores busy':>11}")
    for (vec, arm), d in sorted(by.items()):
        for t in sorted(d):
            if t == 1:
                continue
            s = float(d[t]["speedup_vs_t1"])
            if s <= 0 or s != s:
                continue
            share = (1 - 1 / s) if s >= 1 else float("nan")
            w = float(d[t]["wall_ms"]); c = float(d[t]["cpu_ms"])
            print(f"{vec:<22} {arm:<10} {t:>2} {s:6.3f} {share*100:22.1f}% "
                  f"{c/w if w else float('nan'):11.2f}")


if __name__ == "__main__":
    main()
