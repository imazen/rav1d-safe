#!/usr/bin/env python3
"""Reduce the tiled task/stage/occupancy census to two tables.

Input: the directory `tiled_taskprobe.sh` wrote (one .txt per arm/vector/threads
/round, each holding the `PROBE ...` lines of one process).

Table 1 -- per (arm, vector, threads): total in-stage busy ms/frame, the split
by stage, achieved occupancy (mean and the HISTOGRAM as decile-ish buckets),
the tail fraction, and the deferral counts.

Table 2 -- the t8/t1 ratio of each stage's busy ms/frame, per (arm, vector).
That is the "what appears at t=8 that is absent at t=1" column, in ms/frame,
which is what the attribution needs.

Medians across rounds, with min/max, so a sub-3% claim can be checked against
its own band (AGENT_BRIEF §2).
"""

import collections
import glob
import os
import statistics
import sys

STAGES = [
    "tile_entropy",
    "tile_recon",
    "deblock_cols",
    "deblock_rows",
    "cdef",
    "superres",
    "loop_restore",
    "other",
]
DEFERS = ["own_progress", "pass2_progress", "deblock_barrier", "ref_progress", "admitted"]


def parse(path):
    d = {
        "stage_ms": {},
        "stage_cnt": {},
        "worker_busy": {},
        "worker_park": {},
        "conc": {},
        "filtconc": {},
        "tailconc": {},
        "defer": {},
    }
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == "RESULT":
            d["wall_ms_frame"] = float(f[7])
        elif f[0] == "cell":
            d["arm"], d["vec"], d["threads"], d["iters"], d["round"] = (
                f[1], f[2], int(f[3]), int(f[4]), int(f[5]),
            )
        elif f[0] == "foreign_max":
            d["foreign"] = int(f[1])
        elif f[0] != "PROBE":
            continue
        elif f[1] == "stage_ms_per_frame":
            d["stage_ms"][f[2]] = float(f[3])
            d["stage_cnt"][f[2]] = float(f[5])
        elif f[1] == "filter_chain_ms_per_frame":
            d["filter_ms"] = float(f[2])
        elif f[1] == "tile_ms_per_frame":
            d["tile_ms"] = float(f[2])
        elif f[1] == "worker":
            d["worker_busy"][int(f[2])] = float(f[4])
            d["worker_park"][int(f[2])] = float(f[6])
        elif f[1] == "conc":
            d["conc"][int(f[2])] = int(f[4])
        elif f[1] == "filtconc":
            d["filtconc"][int(f[2])] = int(f[4])
        elif f[1] == "tailconc":
            d["tailconc"][int(f[2])] = int(f[4])
        elif f[1] == "mean_active":
            d["mean_active"] = float(f[2])
        elif f[1] == "mean_active_when_busy":
            d["mean_active_busy"] = float(f[2])
        elif f[1] == "tail_frac_of_wall":
            d["tail_frac"] = float(f[2])
            d["tail_mean"] = float(f[4])
        elif f[1] == "defer":
            d["defer"][f[2]] = float(f[4])
    return d


def med(vals):
    return statistics.median(vals) if vals else float("nan")


def main():
    outdir = sys.argv[1]
    runs = [parse(p) for p in sorted(glob.glob(os.path.join(outdir, "*.txt")))]
    runs = [r for r in runs if "arm" in r]
    by = collections.defaultdict(list)
    for r in runs:
        by[(r["arm"], r["vec"], r["threads"])].append(r)

    print(f"runs={len(runs)}  cells={len(by)}  "
          f"foreign_max={max((r.get('foreign', 0) for r in runs), default=0)}")
    print()

    # ---- Table 1 -------------------------------------------------------
    hdr = (f"{'arm':4} {'vector':26} {'t':>2} {'n':>2} "
           f"{'busy':>7} {'tile':>7} {'dbc':>6} {'dbr':>6} {'cdef':>6} {'lr':>6} "
           f"{'occ':>5} {'occ_b':>5} {'tail%':>6} {'tailocc':>7} "
           f"{'wall':>7} {'defer/adm':>9}")
    print("=" * len(hdr))
    print("TABLE 1 -- in-stage busy ms/frame, stage split, achieved occupancy")
    print("=" * len(hdr))
    print(hdr)
    for key in sorted(by, key=lambda k: (k[1], k[0], k[2])):
        rs = by[key]
        arm, vec, t = key
        tot = [sum(r["stage_ms"].values()) for r in rs]
        print(f"{arm:4} {vec:26} {t:2d} {len(rs):2d} "
              f"{med(tot):7.3f} "
              f"{med([r['stage_ms'].get('tile_recon', 0) for r in rs]):7.3f} "
              f"{med([r['stage_ms'].get('deblock_cols', 0) for r in rs]):6.3f} "
              f"{med([r['stage_ms'].get('deblock_rows', 0) for r in rs]):6.3f} "
              f"{med([r['stage_ms'].get('cdef', 0) for r in rs]):6.3f} "
              f"{med([r['stage_ms'].get('loop_restore', 0) for r in rs]):6.3f} "
              f"{med([r.get('mean_active', 0) for r in rs]):5.2f} "
              f"{med([r.get('mean_active_busy', 0) for r in rs]):5.2f} "
              f"{100 * med([r.get('tail_frac', 0) for r in rs]):6.2f} "
              f"{med([r.get('tail_mean', 0) for r in rs]):7.2f} "
              f"{med([r.get('wall_ms_frame', 0) for r in rs]):7.3f} "
              f"{med([r['defer'].get('own_progress', 0) for r in rs]):4.0f}/"
              f"{med([r['defer'].get('admitted', 0) for r in rs]):-4.0f}")
    print()

    # ---- Table 2: t8/tLO per stage -------------------------------------
    #
    # The low arm is t=2, NOT t=1, and that is forced by the decoder rather than
    # chosen: at `--threads 1` `n_tc == 1`, no task worker exists, and the whole
    # `rav1d_task_run` stage instrumentation is never entered -- every stage
    # counter reads 0.000 in the t=1 rows of Table 1. t=2 is the first cell on
    # the SAME code path as t=8 (task workers, `tile_threading_active()` latched
    # true, narrow guards), so a t8/t2 ratio isolates the effect of ADDING
    # WORKERS from the effect of switching code paths. The t=1 wall in Table 1
    # is still the right denominator for a SPEEDUP; it is the wrong one for a
    # per-stage CPU ratio.
    LO = 2
    print("=" * 110)
    print(f"TABLE 2 -- CPU inflation from t={LO} to t=8, per stage, in ms/frame "
          "(delta) and as a ratio")
    print("   (t=1 is NOT the baseline: at t=1 no task worker runs, so every "
          "stage counter is 0 -- see Table 1)")
    print("=" * 110)
    print(f"{'arm':4} {'vector':26} {'stage':13} "
          f"{'tLO_ms':>8} {'t8_ms':>8} {'delta':>8} {'ratio':>7} {'share%':>7}")
    for arm in sorted({k[0] for k in by}):
        for vec in sorted({k[1] for k in by if k[0] == arm}):
            k1, k8 = (arm, vec, LO), (arm, vec, 8)
            if k1 not in by or k8 not in by:
                continue
            t1 = {s: med([r["stage_ms"].get(s, 0) for r in by[k1]]) for s in STAGES}
            t8 = {s: med([r["stage_ms"].get(s, 0) for r in by[k8]]) for s in STAGES}
            tot_delta = sum(t8.values()) - sum(t1.values())
            for s in STAGES:
                if t1[s] == 0 and t8[s] == 0:
                    continue
                d = t8[s] - t1[s]
                print(f"{arm:4} {vec:26} {s:13} {t1[s]:8.3f} {t8[s]:8.3f} "
                      f"{d:+8.3f} "
                      f"{(t8[s] / t1[s] if t1[s] else float('nan')):7.3f} "
                      f"{(100 * d / tot_delta if tot_delta else 0):7.1f}")
            print(f"{arm:4} {vec:26} {'TOTAL':13} {sum(t1.values()):8.3f} "
                  f"{sum(t8.values()):8.3f} {tot_delta:+8.3f} "
                  f"{sum(t8.values()) / sum(t1.values()):7.3f} {100.0:7.1f}")
            print()

    # ---- Table 3: occupancy histogram ----------------------------------
    print("=" * 110)
    print("TABLE 3 -- occupancy DISTRIBUTION at t=8 (fraction of wall at k "
          "workers inside a stage body)")
    print("=" * 110)
    for key in sorted(by, key=lambda k: (k[1], k[0])):
        if key[2] != 8:
            continue
        rs = by[key]
        arm, vec, t = key
        tot = sum(sum(r["conc"].values()) for r in rs)
        cells = []
        for k in range(0, 9):
            c = sum(r["conc"].get(k, 0) for r in rs)
            cells.append(f"{k}:{100 * c / tot:4.1f}")
        ftot = sum(sum(r["filtconc"].values()) for r in rs)
        fcells = []
        for k in range(0, 9):
            c = sum(r["filtconc"].get(k, 0) for r in rs)
            if c:
                fcells.append(f"{k}:{100 * c / ftot:4.1f}")
        print(f"{arm:4} {vec:26} all   {' '.join(cells)}")
        print(f"{arm:4} {vec:26} filt  {' '.join(fcells)}")
    print()

    # ---- Table 4: worker balance ---------------------------------------
    print("=" * 110)
    print("TABLE 4 -- per-worker busy ms/frame at t=8 (straggler check)")
    print("=" * 110)
    for key in sorted(by, key=lambda k: (k[1], k[0])):
        if key[2] != 8:
            continue
        rs = by[key]
        arm, vec, t = key
        ws = sorted({w for r in rs for w in r["worker_busy"]})
        vals = [med([r["worker_busy"].get(w, 0) for r in rs]) for w in ws]
        parks = [med([r["worker_park"].get(w, 0) for r in rs]) for w in ws]
        print(f"{arm:4} {vec:26} n_workers={len(ws)} "
              f"busy=[{' '.join(f'{v:.2f}' for v in vals)}] "
              f"spread={max(vals) - min(vals):.2f} "
              f"park_med={statistics.median(parks):.2f}")


if __name__ == "__main__":
    main()
