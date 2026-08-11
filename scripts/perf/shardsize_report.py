#!/usr/bin/env python3
"""The picture-size sweep of the shard-granularity ladder, reduced to a crossover table.

One row per vector. Ratios are paired within the same interleave, which is the
only thing that survives a busy box (AGENT_BRIEF §2). Bands are printed and the
disjointness of the base/head bands is computed for the arm the CLAIM is about
(head vs plain — NOT ours vs dav1d, which is a tick that can never fail).

    ratio      median(arm) / median(plain), wall and CPU
    disj       1 when min(plain band) > max(arm band), i.e. the bands do not overlap
    to_ceiling (plain - arm) / (plain - untracked): how much of the distance to the
               tracker-free build this rung closes. Undefined when the ceiling is
               not measured.

Usage: shardsize_report.py <wallcpu.tsv> [--base plain] [--tsv out.tsv]
"""
import collections
import re
import statistics
import sys


def dims(name):
    m = re.search(r"[CL](\d+)x(\d+)_", name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def main():
    path = sys.argv[1]
    base = "plain"
    if "--base" in sys.argv:
        base = sys.argv[sys.argv.index("--base") + 1]
    tsv = sys.argv[sys.argv.index("--tsv") + 1] if "--tsv" in sys.argv else None

    rows = []
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 13:
            continue
        rnd, arm, vec, t, nlo, wlo, ulo, slo, nhi, whi, uhi, shi, fo = f[:13]
        nlo, nhi = int(nlo), int(nhi)
        wlo, ulo, slo, whi, uhi, shi = (int(x) for x in (wlo, ulo, slo, whi, uhi, shi))
        dn = nhi - nlo
        rows.append(dict(round=int(rnd), arm=arm, vec=vec, t=int(t),
                         wall=(whi - wlo) / dn,
                         cpu=((uhi + shi) - (ulo + slo)) / dn,
                         foreign=int(fo)))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["vec"], r["t"])].append(r)

    arms = sorted({k[0] for k in by})
    order = [a for a in (base, "bps1", "bpshalf", "bpsq", "bps4", "untracked", "dav1d_fd1")
             if a in arms] + [a for a in arms if a not in
                              (base, "bps1", "bpshalf", "bpsq", "bps4", "untracked", "dav1d_fd1")]
    ts = sorted({k[2] for k in by})
    fmax = max(r["foreign"] for r in rows)
    nrounds = len({r["round"] for r in rows})
    print(f"rows={len(rows)} cells={len(by)} rounds={nrounds} foreign_max={fmax} base={base}")
    if fmax > 0:
        print("!! foreign load observed — report PAIRED RATIOS, tag absolutes")
    print()

    out = ["\t".join("vector w h t base_wall_ms base_cpu_ms arm wall_ratio cpu_ratio "
                     "disj_wall band_lo band_hi to_ceiling".split())]
    for t in ts:
        vecs = sorted({k[1] for k in by if k[2] == t}, key=lambda v: (dims(v)[0], dims(v)[1]))
        print("=" * 118)
        print(f"t={t}   wall ms/frame and RATIO vs {base} (median of {nrounds}); "
              f"disj = base/arm bands disjoint")
        print("=" * 118)
        head = f"{'vector':22}{'w':>5}{'h':>6}{'base_wall':>11}{'base_cpu':>10}"
        for a in order:
            if a == base:
                continue
            head += f"{a:>13}"
        print(head + f"{'ceil%':>8}")
        for v in vecs:
            k0 = (base, v, t)
            if k0 not in by:
                continue
            b_w = statistics.median(r["wall"] for r in by[k0])
            b_c = statistics.median(r["cpu"] for r in by[k0])
            b_lo = min(r["wall"] for r in by[k0])
            w, h = dims(v)
            line = f"{v[:21]:22}{w:>5}{h:>6}{b_w:>11.3f}{b_c:>10.3f}"
            ceil = None
            ku = ("untracked", v, t)
            if ku in by:
                ceil = statistics.median(r["wall"] for r in by[ku])
            for a in order:
                if a == base:
                    continue
                k = (a, v, t)
                if k not in by:
                    line += f"{'-':>13}"
                    continue
                m = statistics.median(r["wall"] for r in by[k])
                mc = statistics.median(r["cpu"] for r in by[k])
                hi = max(r["wall"] for r in by[k])
                lo = min(r["wall"] for r in by[k])
                disj = 1 if (b_lo > hi or lo > max(r["wall"] for r in by[k0])) else 0
                mark = "*" if disj else " "
                line += f"{m / b_w:>12.3f}{mark}"
                frac = ""
                if ceil is not None and abs(b_w - ceil) > 1e-9:
                    frac = f"{100 * (b_w - m) / (b_w - ceil):.1f}"
                out.append("\t".join(str(x) for x in [
                    v, w, h, t, f"{b_w:.4f}", f"{b_c:.4f}", a,
                    f"{m / b_w:.4f}", f"{mc / b_c:.4f}", disj,
                    f"{lo:.4f}", f"{hi:.4f}", frac]))
            # headline: how much of the distance to the tracker-free ceiling the
            # recommended rung closes
            kh = ("bpshalf", v, t)
            if ceil is not None and kh in by and abs(b_w - ceil) > 1e-9:
                mh = statistics.median(r["wall"] for r in by[kh])
                line += f"{100 * (b_w - mh) / (b_w - ceil):>8.1f}"
            print(line)
        print("  * = base and arm wall bands are disjoint")
        print()

    if tsv:
        open(tsv, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
