#!/usr/bin/env python3
"""Reduce tiled_wallcpu.sh rows to per-cell wall/CPU per frame and the deficit
decomposition.

Two-point fit per row: beta = (total_hi - total_lo) / (n_hi - n_lo), separately
for wall and for user+sys, so process startup cancels. Median across rounds with
the min/max band (AGENT_BRIEF: print the band, a sub-3% claim has to be checked
against it).

The decomposition, per (arm, vector), reported in ms/frame so the parts add:

    ideal   = cpu(t=1) / t                    perfect scaling, no added work
    +work   = (cpu(t) - cpu(1)) / t           the added CPU, spread over t cores
    +idle   = wall(t) - cpu(t)/t              cores that were not busy
    -----------------------------------------------------------------
    wall(t) = ideal + work + idle

`+work` and `+idle` are the two halves of the tiled deficit, in the same unit,
so their ratio says which one to attack.
"""

import collections
import statistics
import sys


def main():
    rows = []
    for line in open(sys.argv[1]):
        f = line.rstrip("\n").split("\t")
        if len(f) < 13:
            continue
        rnd, arm, vec, t, nlo, wlo, ulo, slo, nhi, whi, uhi, shi, fo = f[:13]
        t, nlo, nhi = int(t), int(nlo), int(nhi)
        wlo, ulo, slo, whi, uhi, shi = (int(x) for x in (wlo, ulo, slo, whi, uhi, shi))
        dn = nhi - nlo
        rows.append({
            "round": int(rnd), "arm": arm, "vec": vec, "t": t,
            "wall": (whi - wlo) / dn,
            "cpu": ((uhi + shi) - (ulo + slo)) / dn,
            "foreign": int(fo),
        })

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["vec"], r["t"])].append(r)

    def med(k, key):
        return statistics.median([r[key] for r in by[k]])

    def band(k, key):
        v = [r[key] for r in by[k]]
        return min(v), max(v)

    print(f"rows={len(rows)}  cells={len(by)}  "
          f"foreign_max={max(r['foreign'] for r in rows)}  "
          f"rounds={len({r['round'] for r in rows})}")
    print()
    print("=" * 128)
    print("PER CELL -- wall and CPU ms/frame (two-point fit), cores busy, speedup vs own t=1")
    print("=" * 128)
    print(f"{'arm':10} {'vector':24} {'t':>2} {'wall':>9} {'[band]':>19} "
          f"{'cpu':>9} {'[band]':>19} {'cores':>6} {'S':>6} {'cpu/t1':>7}")
    for arm in sorted({k[0] for k in by}):
        for vec in sorted({k[1] for k in by if k[0] == arm}):
            base_w = med((arm, vec, 1), "wall")
            base_c = med((arm, vec, 1), "cpu")
            for t in sorted({k[2] for k in by if k[0] == arm and k[1] == vec}):
                k = (arm, vec, t)
                w, c = med(k, "wall"), med(k, "cpu")
                wl, wh = band(k, "wall")
                cl, ch = band(k, "cpu")
                print(f"{arm:10} {vec:24} {t:2d} {w:9.3f} [{wl:8.3f}..{wh:8.3f}] "
                      f"{c:9.3f} [{cl:8.3f}..{ch:8.3f}] {c / w:6.2f} "
                      f"{base_w / w:6.3f} {c / base_c:7.3f}")
            print()

    print("=" * 128)
    print("DEFICIT DECOMPOSITION -- wall(t) = ideal + added_work + idle_cores, ms/frame")
    print("=" * 128)
    print(f"{'arm':10} {'vector':24} {'t':>2} {'wall':>8} {'ideal':>8} "
          f"{'+work':>8} {'+idle':>8} {'work%':>6} {'idle%':>6} {'S':>6} {'S_ideal':>7}")
    for arm in sorted({k[0] for k in by}):
        for vec in sorted({k[1] for k in by if k[0] == arm}):
            base_c = med((arm, vec, 1), "cpu")
            base_w = med((arm, vec, 1), "wall")
            for t in sorted({k[2] for k in by if k[0] == arm and k[1] == vec}):
                if t == 1:
                    continue
                k = (arm, vec, t)
                w, c = med(k, "wall"), med(k, "cpu")
                ideal = base_c / t
                work = (c - base_c) / t
                idle = w - c / t
                over = w - ideal
                print(f"{arm:10} {vec:24} {t:2d} {w:8.3f} {ideal:8.3f} "
                      f"{work:+8.3f} {idle:+8.3f} "
                      f"{(100 * work / over if over else 0):6.1f} "
                      f"{(100 * idle / over if over else 0):6.1f} "
                      f"{base_w / w:6.3f} {base_w / ideal:7.3f}")
            print()

    print("=" * 128)
    print("OURS / dav1d_fd1, paired within round")
    print("=" * 128)
    for vec in sorted({k[1] for k in by}):
        for t in sorted({k[2] for k in by if k[1] == vec}):
            a, b = ("rs", vec, t), ("dav1d_fd1", vec, t)
            if a not in by or b not in by:
                continue
            ra = {r["round"]: r for r in by[a]}
            rb = {r["round"]: r for r in by[b]}
            common = sorted(set(ra) & set(rb))
            rw = [ra[i]["wall"] / rb[i]["wall"] for i in common]
            rc = [ra[i]["cpu"] / rb[i]["cpu"] for i in common]
            print(f"{vec:24} t={t}  n={len(common)}  "
                  f"wall {statistics.median(rw):6.3f} [{min(rw):.3f}..{max(rw):.3f}]  "
                  f"cpu {statistics.median(rc):6.3f} [{min(rc):.3f}..{max(rc):.3f}]")
        print()


if __name__ == "__main__":
    main()
