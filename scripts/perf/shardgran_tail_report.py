#!/usr/bin/env python3
"""Score the shard-granularity ladder on TAIL CONCURRENCY as well as on wall.

`BLOCKS_PER_SHARD` was fitted against whole-frame wall. `docs/TILED_SCALING.md`
§4 then measured that 32-42% of the t=8 wall gap is IDLE CORES, most of it in
the post-tile filter tail -- so a shift that minimises mean wall can be wrong
exactly where the cores are idle. This reducer prints BOTH objectives per rung so
a disagreement between them is visible rather than averaged away.

The tail objective is `tail_idle_frac`: the fraction of the whole frame's
CORE-TIME that is idle *because* the tail is running under-subscribed,

    tail_idle_frac = tail_frac_of_wall * (threads - tail_mean_active) / threads

which is the quantity the "upper-bound recoverable" row of that document
normalises. Lower is better. `tail_frac_of_wall` and `tail_mean_active` are
printed beside it because they can move in opposite directions: a rung that
shortens the tail while emptying it further is not obviously a win.

Usage: shardgran_tail_report.py <probedir>
"""

import glob
import os
import statistics
import sys

STAGES = [
    "tile_entropy", "tile_recon", "deblock_cols", "deblock_rows",
    "cdef", "superres", "loop_restore", "other",
]


def parse(path):
    d = {"stage_ms": {}}
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == "RESULT":
            d["wall"] = float(f[7])
        elif f[0] == "cell":
            d["arm"], d["vec"], d["threads"] = f[1], f[2], int(f[3])
        elif f[0] == "foreign_max":
            d["foreign"] = int(f[1])
        elif f[0] != "PROBE":
            continue
        elif f[1] == "stage_ms_per_frame":
            d["stage_ms"][f[2]] = float(f[3])
        elif f[1] == "filter_chain_ms_per_frame":
            d["filter_ms"] = float(f[2])
        elif f[1] == "mean_active":
            d["mean_active"] = float(f[2])
        elif f[1] == "tail_frac_of_wall":
            d["tail_frac"] = float(f[2])
            d["tail_mean"] = float(f[4])
    return d


def med(v):
    return statistics.median(v) if v else float("nan")


def band(v):
    return (min(v), max(v)) if v else (float("nan"), float("nan"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = [parse(p) for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.txt")))]
    rows = [r for r in rows if "arm" in r and "wall" in r]
    if not rows:
        print("no parseable probe dumps in", sys.argv[1])
        return 1

    cells = sorted({(r["vec"], r["threads"]) for r in rows})
    arms = sorted({r["arm"] for r in rows})
    fmax = max(r.get("foreign", 0) for r in rows)
    print(f"# rows {len(rows)}  arms {len(arms)}  cells {len(cells)}  foreign_max {fmax}")
    print()

    for vec, t in cells:
        sub = [r for r in rows if r["vec"] == vec and r["threads"] == t]
        print(f"=== {vec}  t={t} " + "=" * 30)
        print(
            f"{'arm':<12} {'wall':>8} {'[min..max]':>17} {'busy':>8} "
            f"{'tailfrac':>9} {'tailmean':>9} {'tail_idle':>10} "
            f"{'occ':>6} {'dbcols':>8} {'cdef':>7} {'n':>3}"
        )
        base = None
        for arm in arms:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            walls = [r["wall"] for r in a]
            lo, hi = band(walls)
            busy = med([sum(r["stage_ms"].values()) for r in a])
            tf = med([r.get("tail_frac", float("nan")) for r in a])
            tm = med([r.get("tail_mean", float("nan")) for r in a])
            idle = tf * (t - tm) / t if t else float("nan")
            occ = med([r.get("mean_active", float("nan")) for r in a])
            db = med([r["stage_ms"].get("deblock_cols", float("nan")) for r in a])
            cd = med([r["stage_ms"].get("cdef", float("nan")) for r in a])
            w = med(walls)
            if arm in ("tt", "plain"):
                base = (w, idle)
            print(
                f"{arm:<12} {w:8.3f} [{lo:7.3f}..{hi:7.3f}] {busy:8.3f} "
                f"{tf:9.4f} {tm:9.3f} {idle:10.4f} {occ:6.2f} {db:8.3f} {cd:7.3f} {len(a):3d}"
            )
        if base:
            print(f"{'':<12} (ratios vs the default rung: wall / tail_idle)")
            for arm in arms:
                a = [r for r in sub if r["arm"] == arm]
                if not a or arm in ("tt", "plain"):
                    continue
                w = med([r["wall"] for r in a])
                tf = med([r.get("tail_frac", float("nan")) for r in a])
                tm = med([r.get("tail_mean", float("nan")) for r in a])
                idle = tf * (t - tm) / t if t else float("nan")
                rw = w / base[0] if base[0] else float("nan")
                ri = idle / base[1] if base[1] else float("nan")
                flag = ""
                if (rw < 1.0) != (ri < 1.0):
                    flag = "  <-- OBJECTIVES DISAGREE"
                print(f"{arm:<12} wall {rw:6.4f}x   tail_idle {ri:6.4f}x{flag}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
