#!/usr/bin/env python3
"""Reduce the timer-free LEVER-2 counts across the granularity ladder.

Two tables:

  WIDE   -- the SHIPPED wide-path promotion rate per rung, by door
            (w_shards / w_blocks / w_full). The wide path holds every active
            shard, so any rate here is disproportionate; and the three doors move
            in OPPOSITE directions as the block grows, which is why a single
            "wide_total" is not enough to read the ladder.

  RECT   -- `pct_row_wide` (the quantity that refuted the strided-2D record) per
            rung, plus the same fraction at raised caps 5 / 8 / 16, plus
            registrations per frame (which the shift must NOT change -- the shift
            changes the cost of a registration, not the count, so a moving
            SITES total means something else moved too).

The `shifts` column is the LIVENESS proof for a rung: it is the set of block
shifts the tracker actually used, read off the instances themselves. A rung whose
shifts equal the default's did not arm.

Usage: shardgran_counts_report.py <countsdir>
"""

import glob
import os
import sys

RUNG_ORDER = ["bpsq", "bpshalf", "bps1", "plain", "bps4", "bps8"]


def parse(path):
    d = {"rect": [], "sites_total": None, "wide": None, "run": None}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if f[0] == "cell":
            d["rung"], d["probe"], d["vec"], d["threads"] = f[1], f[2], f[3], f[4]
        elif f[0] == "WIDE":
            d["wide"] = f[1:]
        elif f[0] == "SITES":
            for kv in f[1:]:
                if kv.startswith("total_per_frame="):
                    d["sites_total"] = float(kv.split("=")[1])
        elif f[0] == "RECT" and len(f) > 20:
            d["rect"].append(f[1:])
        elif f[0].startswith("RUN"):
            g = line.split()
            for kv in g:
                if kv.startswith("ms_per_frame="):
                    d["run_ms"] = float(kv.split("=")[1])
    return d


def key(r):
    o = RUNG_ORDER.index(r["rung"]) if r["rung"] in RUNG_ORDER else 99
    return (r.get("vec", ""), int(r.get("threads", 0)), o)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = [parse(p) for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.txt")))]
    rows = [r for r in rows if "rung" in r]
    if not rows:
        print("no parseable dumps in", sys.argv[1])
        return 1

    w = sorted([r for r in rows if r["wide"]], key=key)
    if w:
        print("TABLE W -- shipped wide-path promotions, per DECODE RUN (not per frame)")
        print("           w_shards: >MAX_SHARDS_PER_BORROW distinct shards")
        print("           w_blocks: >MAX_BLOCKS_SCAN blocks")
        print("           w_full:   no free slot in a shard (rises with a COARSER block)")
        print()
        print(f"{'vec':<26} {'t':>2} {'rung':<8} {'slow':>12} {'multi':>12} "
              f"{'w_shards':>10} {'w_blocks':>9} {'w_full':>9} {'wide_tot':>9}")
        last = None
        for r in w:
            cell = (r["vec"], r["threads"])
            if last is not None and cell != last:
                print()
            last = cell
            _shift, slow, multi, ws, wb, wf, tot = r["wide"]
            print(f"{r['vec']:<26} {r['threads']:>2} {r['rung']:<8} {int(slow):>12,} "
                  f"{int(multi):>12,} {int(ws):>10,} {int(wb):>9,} {int(wf):>9,} {int(tot):>9,}")
        print()

    rect = sorted([r for r in rows if r["rect"]], key=key)
    if rect:
        print("TABLE R -- strided-2D counterfactual per rung. pct_row_wide is the")
        print("           refuting quantity; c5/c8/c16 are RAISED-cap counterfactuals.")
        print("           `shifts` is the rung's LIVENESS proof.")
        print()
        # column indices into the RECT row (after the leading 'RECT')
        C = {
            "n": 0, "rows_mean": 8, "row_shards_mean": 12, "row_shards_max": 13,
            "pct_hull_wide": 14, "pct_row_wide": 15, "pct_c5": 16, "pct_c8": 17,
            "pct_c16": 18, "pct_perrow_narrow": 19, "shifts": 20, "where": 21,
        }
        sites = {}
        for r in rect:
            for row in r["rect"]:
                sites.setdefault(row[C["where"]], set()).add(r["rung"])
        hot = sorted(sites, key=lambda s: -max(
            int(row[C["n"]]) for r in rect for row in r["rect"] if row[C["where"]] == s))
        for r in rect:
            print(f"--- {r['vec']} t={r['threads']} rung={r['rung']} "
                  f"registrations/frame={r['sites_total']!r}")
            print(f"    {'site':<44} {'n':>10} {'rows':>6} {'shards':>7} {'max':>4} "
                  f"{'row_wide':>9} {'c5':>7} {'c8':>7} {'c16':>7} {'shifts':>8}")
            byname = {row[C["where"]]: row for row in r["rect"]}
            for s in hot:
                row = byname.get(s)
                if row is None:
                    continue
                print(f"    {s[-44:]:<44} {int(row[C['n']]):>10,} "
                      f"{float(row[C['rows_mean']]):>6.2f} "
                      f"{float(row[C['row_shards_mean']]):>7.3f} "
                      f"{int(row[C['row_shards_max']]):>4} "
                      f"{float(row[C['pct_row_wide']]):>8.2f}% "
                      f"{float(row[C['pct_c5']]):>6.2f}% "
                      f"{float(row[C['pct_c8']]):>6.2f}% "
                      f"{float(row[C['pct_c16']]):>6.2f}% "
                      f"{row[C['shifts']]:>8}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
