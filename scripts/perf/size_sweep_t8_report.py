#!/usr/bin/env python3
"""Latency AND CPU-waste across the size ladder x thread count.

Input: the TSV from scripts/perf/size_sweep_t8.sh
  round arm vec threads nlo wall_lo user_lo sys_lo nhi wall_hi user_hi sys_hi foreign

Both quantities go through the SAME two-point fit, `total = a + b*frames`, so
process startup -- exec, mmap, decoder construction, and thread-pool spin-up,
which is a real per-PROCESS cost that would otherwise inflate the CPU of every
high-thread cell -- drops out of both:

    ms_wall/frame = (wall_hi - wall_lo) / (nhi - nlo)
    ms_cpu /frame = ((user+sys)_hi - (user+sys)_lo) / (nhi - nlo)

`ms_cpu` is core-milliseconds burned per decoded frame: at t=1 it tracks wall,
and above t=1 the gap between them IS the waste.

Everything is PAIRED WITHIN A ROUND before it is reduced -- speedups compare
t and t=1 from the same round, ratios compare arms from the same round -- then
median with the min/max band printed, per docs/AGENT_BRIEF.md.

DECISION RULES, pre-registered here so they are not fitted to the answer:
  * latency-optimal t  = smallest t whose median wall is within 2% of the best
    median wall for that cell (more threads for no measured gain is waste).
  * "stops helping"    = the largest t whose marginal doubling (t/2 -> t) has a
    speedup band strictly above 1.0. Beyond it, threads buy nothing measurable.
  * "stops being free" = the largest t whose CPU-per-decode band overlaps or
    sits below 1.10x of t=1. Beyond it, latency is bought with throughput.

Usage: size_sweep_t8_report.py <tsv> [--tsv out.tsv]
"""

import re
import sys
from collections import defaultdict

VEC_RE = re.compile(r"^L(\d+)x(\d+)_(\d+)_(\d+)b(?:__t(\d+))?$")
OURS = "rs"


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def parse(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 13:
                continue
            rows.append(dict(
                round=int(p[0]), arm=p[1], vec=p[2], threads=int(p[3]),
                nlo=int(p[4]), w_lo=float(p[5]), u_lo=float(p[6]), s_lo=float(p[7]),
                nhi=int(p[8]), w_hi=float(p[9]), u_hi=float(p[10]), s_hi=float(p[11]),
                foreign=int(p[12]),
            ))
    return rows


def geom(vec):
    m = VEC_RE.match(vec)
    if not m:
        return None
    w, h, fmt, depth, tiles = m.groups()
    return dict(w=int(w), h=int(h), fmt=fmt, depth=int(depth),
                luma=int(w) * int(h), tiles=int(tiles) if tiles else 1)


def fmt_band(v, lo, hi, width=8, prec=3):
    return f"{v:{width}.{prec}f} [{lo:.{prec}f}..{hi:.{prec}f}]"


def main():
    path = sys.argv[1]
    out_tsv = sys.argv[sys.argv.index("--tsv") + 1] if "--tsv" in sys.argv else None
    rows = parse(path)
    if not rows:
        sys.exit("no rows")

    loaded = sum(1 for r in rows if r["foreign"] > 0)
    fmax = max(r["foreign"] for r in rows)
    print(f"rows={len(rows)}  rows_under_foreign_load={loaded}  foreign_max={fmax}")
    if loaded:
        print("  LOAD-TAGGED: absolutes are inflated; the usable statistics are the "
              "paired within-round ratios (speedup, cpu multiplier, ours/dav1d).")

    wall = {}
    cpu = {}
    for r in rows:
        dn = r["nhi"] - r["nlo"]
        k = (r["arm"], r["vec"], r["threads"], r["round"])
        wall[k] = (r["w_hi"] - r["w_lo"]) / dn
        cpu[k] = ((r["u_hi"] + r["s_hi"]) - (r["u_lo"] + r["s_lo"])) / dn

    arms = sorted({r["arm"] for r in rows})
    threads = sorted({r["threads"] for r in rows})
    allv = sorted({r["vec"] for r in rows})
    rounds = sorted({r["round"] for r in rows})

    # Drop rounds that did not complete every (arm, vec, threads) triple, so
    # every cell in every table has the same n.
    full = {(a, v, t) for (a, v, t, _) in wall}
    complete = [rd for rd in rounds
                if all((a, v, t, rd) in wall for (a, v, t) in full)]
    dropped = [rd for rd in rounds if rd not in complete]
    if dropped:
        print(f"dropped incomplete rounds: {dropped}  (kept n={len(complete)})")
    rounds = complete
    if not rounds:
        sys.exit("no complete rounds")
    print(f"n={len(rounds)} complete rounds; arms={arms}; threads={threads}")

    def series(d, arm, vec, t):
        return [d[(arm, vec, t, rd)] for rd in rounds if (arm, vec, t, rd) in d]

    def paired(dn, dd, an, vn, tn, ad, vd, td):
        out = []
        for rd in rounds:
            kn, kd = (an, vn, tn, rd), (ad, vd, td, rd)
            if kn in dn and kd in dd and dd[kd]:
                out.append(dn[kn] / dd[kd])
        return out

    ladder = [v for v in allv if geom(v) and geom(v)["tiles"] == 1]
    multi = [v for v in allv if geom(v) and geom(v)["tiles"] > 1]
    ladder.sort(key=lambda v: (geom(v)["fmt"], geom(v)["depth"], geom(v)["luma"]))

    out_lines = ["vec\tfmt\tdepth\tw\th\tluma\ttiles\tarm\tthreads\t"
                 "wall_ms\twall_lo\twall_hi\tcpu_ms\tcpu_lo\tcpu_hi\t"
                 "speedup_vs_t1\tsu_lo\tsu_hi\tcpu_mult_vs_t1\tcm_lo\tcm_hi\t"
                 "eff_su_over_t\tratio_wall_vs_dav1d_fd1\tratio_cpu_vs_dav1d_fd1"]

    ref = "dav1d_fd1" if "dav1d_fd1" in arms else None

    for group, title in ((ladder, "SINGLE-TILE LADDER"), (multi, "FORCED MULTI-TILE")):
        if not group:
            continue
        for vec in group:
            g = geom(vec)
            print()
            print(f"=== {title}: {vec}  ({g['w']}x{g['h']} YUV{g['fmt']} {g['depth']}bpc, "
                  f"{g['tiles']} tile{'s' if g['tiles'] > 1 else ''}) ===")
            hdr = (f"{'arm':<10} {'t':>2} | {'wall ms/frame':>26} | {'cpu ms/frame':>26} | "
                   f"{'cores':>5} | {'speedup':>18} {'S/t':>5} | {'cpu x t1':>18}")
            print(hdr)
            for arm in arms:
                for t in threads:
                    w = series(wall, arm, vec, t)
                    c = series(cpu, arm, vec, t)
                    if not w:
                        continue
                    # speedup vs t=1, paired within round: wall(t=1)/wall(t)
                    su = paired(wall, wall, arm, vec, 1, arm, vec, t)
                    cm = paired(cpu, cpu, arm, vec, t, arm, vec, 1)
                    smed = median(su) if su else float("nan")
                    cmed = median(cm) if cm else float("nan")
                    # cores busy on average during the decode: CPU per frame
                    # divided by wall per frame. 1.0 = one core; anything above
                    # is parallelism, and anything above the speedup is waste.
                    cores = median(paired(cpu, wall, arm, vec, t, arm, vec, t))
                    print(f"{arm:<10} {t:>2} | {fmt_band(median(w), min(w), max(w), 8, 3):>26} | "
                          f"{fmt_band(median(c), min(c), max(c), 8, 3):>26} | "
                          f"{cores:5.2f} | "
                          f"{fmt_band(smed, min(su) if su else 0, max(su) if su else 0, 5, 3):>18} "
                          f"{smed/t:5.2f} | "
                          f"{fmt_band(cmed, min(cm) if cm else 0, max(cm) if cm else 0, 5, 3):>18}")
                    rw = paired(wall, wall, arm, vec, t, ref, vec, t) if ref else []
                    rc = paired(cpu, cpu, arm, vec, t, ref, vec, t) if ref else []
                    out_lines.append(
                        f"{vec}\t{g['fmt']}\t{g['depth']}\t{g['w']}\t{g['h']}\t{g['luma']}\t{g['tiles']}\t"
                        f"{arm}\t{t}\t{median(w):.5f}\t{min(w):.5f}\t{max(w):.5f}\t"
                        f"{median(c):.5f}\t{min(c):.5f}\t{max(c):.5f}\t"
                        f"{smed:.4f}\t{(min(su) if su else float('nan')):.4f}\t{(max(su) if su else float('nan')):.4f}\t"
                        f"{cmed:.4f}\t{(min(cm) if cm else float('nan')):.4f}\t{(max(cm) if cm else float('nan')):.4f}\t"
                        f"{smed/t:.4f}\t{(median(rw) if rw else float('nan')):.4f}\t"
                        f"{(median(rc) if rc else float('nan')):.4f}")
            # ours vs dav1d, per thread count
            if ref and OURS in arms:
                print(f"  -- {OURS} / {ref}, paired within round --")
                for t in threads:
                    rw = paired(wall, wall, OURS, vec, t, ref, vec, t)
                    rc = paired(cpu, cpu, OURS, vec, t, ref, vec, t)
                    if not rw:
                        continue
                    print(f"     t={t}: wall {median(rw):.3f} [{min(rw):.3f}..{max(rw):.3f}]"
                          f"   cpu {median(rc):.3f} [{min(rc):.3f}..{max(rc):.3f}]")

    # ---- the decision table ----
    print()
    print("=" * 118)
    print("DECISION TABLE  (rules pre-registered in this file's docstring)")
    print("=" * 118)
    print(f"{'vector':<26} {'arm':<10} {'t*lat':>6} {'best ms':>9} {'S(t*)':>6} "
          f"{'stops helping':>14} {'stops free':>11} {'cpu@t8/t1':>10} {'S/C@t8':>7}")
    for vec in ladder + multi:
        for arm in arms:
            meds = {t: median(series(wall, arm, vec, t)) for t in threads
                    if series(wall, arm, vec, t)}
            if not meds:
                continue
            best = min(meds.values())
            t_star = min(t for t, v in meds.items() if v <= best * 1.02)
            # stops helping: largest t whose marginal doubling band is > 1.0
            stops_help = 1
            for i in range(1, len(threads)):
                lo, hi = threads[i - 1], threads[i]
                marg = paired(wall, wall, arm, vec, lo, arm, vec, hi)  # lo/hi = speedup
                if marg and min(marg) > 1.0:
                    stops_help = hi
                else:
                    break
            # stops being free: largest t whose cpu multiplier vs t1 <= 1.10
            stops_free = 1
            for t in threads[1:]:
                cm = paired(cpu, cpu, arm, vec, t, arm, vec, 1)
                if cm and median(cm) <= 1.10:
                    stops_free = t
                else:
                    break
            tmax = threads[-1]
            su_max = paired(wall, wall, arm, vec, 1, arm, vec, tmax)
            cm_max = paired(cpu, cpu, arm, vec, tmax, arm, vec, 1)
            s = median(su_max) if su_max else float("nan")
            c = median(cm_max) if cm_max else float("nan")
            s_star = median(paired(wall, wall, arm, vec, 1, arm, vec, t_star)) \
                if paired(wall, wall, arm, vec, 1, arm, vec, t_star) else float("nan")
            print(f"{vec:<26} {arm:<10} {t_star:>6} {best:9.3f} {s_star:6.2f} "
                  f"{stops_help:>14} {stops_free:>11} {c:10.2f} {(s/c if c else float('nan')):7.2f}")

    # ---- the compact matrix: one row per cell, one block per thread count ----
    for arm in arms:
        print()
        print("=" * 118)
        print(f"COMPACT MATRIX -- {arm}:  ms/frame | cores busy | "
              f"{'ratio vs ' + ref if ref else 'n/a'} (wall)")
        print("=" * 118)
        head = f"{'vector':<26}"
        for t in threads:
            head += f" | {'t=' + str(t):>22}"
        print(head)
        for vec in ladder + multi:
            line = f"{vec:<26}"
            for t in threads:
                w = series(wall, arm, vec, t)
                if not w:
                    line += f" | {'-':>22}"
                    continue
                cores = median(paired(cpu, wall, arm, vec, t, arm, vec, t))
                rw = paired(wall, wall, arm, vec, t, ref, vec, t) if ref else []
                rr = f"{median(rw):.2f}x" if rw else "  -  "
                line += f" | {median(w):9.3f} {cores:5.2f}c {rr:>6}"
            print(line)

    if out_tsv:
        with open(out_tsv, "w") as fh:
            fh.write("\n".join(out_lines) + "\n")
        print(f"\nwrote {out_tsv}")


if __name__ == "__main__":
    main()
