#!/usr/bin/env python3
"""The one table the attribution rests on: per-stage CPU inflation from a low
thread count to t=8, tracked vs untracked, with the min/max band.

Kept separate from `tiled_taskprobe_report.py` because it enforces ONE summary
convention and prints its own band. In the big report each column is a median
taken independently, so the stage medians need not sum to the total median (they
differ by ~0.05 ms at 1024x576). Here the TOTAL is the median of each run's own
stage sum, and every delta is `median(t8) - median(tLO)` of the same quantity,
so the numbers quoted in docs/TILED_SCALING.md are reproducible from one rule.

Usage: tiled_stage_delta.py <probedir> [lo_threads=4]
"""

import collections
import glob
import os
import statistics
import sys

STAGES = ["tile_recon", "deblock_cols", "deblock_rows", "cdef",
          "superres", "loop_restore"]


def parse(path):
    d = {"stage": {}, "cell": None}
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == "cell":
            d["cell"] = (f[1], f[2], int(f[3]))
        elif f[0] == "RESULT":
            d["wall"] = float(f[7])
        elif f[0] == "foreign_max":
            d["foreign"] = int(f[1])
        elif f[0] == "PROBE" and f[1] == "stage_ms_per_frame":
            d["stage"][f[2]] = float(f[3])
    d["total"] = sum(d["stage"].values())
    return d


def main():
    probedir = sys.argv[1]
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    runs = [parse(p) for p in sorted(glob.glob(os.path.join(probedir, "*.txt")))]
    runs = [r for r in runs if r["cell"]]
    by = collections.defaultdict(list)
    for r in runs:
        by[r["cell"]].append(r)

    def q(cell, key):
        vals = ([r["stage"].get(key, 0.0) for r in by[cell]] if key in STAGES
                else [r[key] for r in by[cell]])
        return statistics.median(vals), min(vals), max(vals), len(vals)

    print(f"foreign_max={max(r.get('foreign', 0) for r in runs)}  "
          f"low_arm=t{lo}  (t=1 unusable: no task worker, all stage counters 0)")
    print()
    for vec in sorted({c[1] for c in by}):
        for t in (lo, 8):
            for arm in ("tt", "ttu"):
                if (arm, vec, t) not in by:
                    print(f"MISSING {arm} {vec} t{t}")
        print("=" * 116)
        print(f"{vec}")
        print("=" * 116)
        print(f"{'quantity':14} {'arm':4} "
              f"{'t' + str(lo):>9} {'band':>19} {'t8':>9} {'band':>19} "
              f"{'delta':>9} {'ratio':>7}")
        deltas = {}
        noise = {}
        for key in ["total"] + STAGES + ["wall"]:
            for arm in ("tt", "ttu"):
                c1, c8 = (arm, vec, lo), (arm, vec, 8)
                if c1 not in by or c8 not in by:
                    continue
                m1, l1, h1, n1 = q(c1, key)
                m8, l8, h8, n8 = q(c8, key)
                if m1 == 0 and m8 == 0:
                    continue
                deltas[(key, arm)] = m8 - m1
                # The widest band either arm shows for this quantity is the
                # floor below which a delta is not a delta -- printing a
                # "tracker share" of a within-noise movement is how a table
                # gets a confident number for nothing (AGENT_BRIEF: check the
                # noise band before believing a small claim).
                noise[(key, arm)] = max(h1 - l1, h8 - l8)
                print(f"{key:14} {arm:4} {m1:9.3f} [{l1:8.3f}..{h1:8.3f}] "
                      f"{m8:9.3f} [{l8:8.3f}..{h8:8.3f}] {m8 - m1:+9.3f} "
                      f"{(m8 / m1 if m1 else float('nan')):7.3f}")
            print()
        print(f"{'TRACKER SHARE of the t' + str(lo) + '->t8 inflation':50}")
        for key in ["total"] + STAGES:
            a, b = deltas.get((key, "tt")), deltas.get((key, "ttu"))
            if a is None or b is None or abs(a) < 1e-9:
                continue
            nz = max(noise.get((key, "tt"), 0.0), noise.get((key, "ttu"), 0.0))
            if abs(a) <= nz:
                print(f"   {key:14} tracked {a:+8.3f}  untracked {b:+8.3f}  "
                      f"tracker   n/a  (|delta| <= widest band {nz:.3f} -- "
                      f"NOT a movement)")
                continue
            print(f"   {key:14} tracked {a:+8.3f}  untracked {b:+8.3f}  "
                  f"tracker {100 * (a - b) / a:6.1f}%")
        # speedup at the two ceilings, from the t=1 wall
        print()
        for arm in ("tt", "ttu"):
            c1, c8 = (arm, vec, 1), (arm, vec, 8)
            if c1 in by and c8 in by:
                w1 = q(c1, "wall")[0]
                w8 = q(c8, "wall")[0]
                print(f"   speedup t1->t8  {arm:4} {w1:8.3f} -> {w8:8.3f} = "
                      f"{w1 / w8:6.3f}x")
        print()


if __name__ == "__main__":
    main()
