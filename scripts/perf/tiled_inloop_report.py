#!/usr/bin/env python3
"""Reduce tiled_inloop_ab.sh to each decoder's own deblock cost and its scaling.

Per (arm, vector, threads): wall/CPU per frame at `--inloopfilters all` and at
`nodeblock`, their difference (= that decoder's deblock chain, in ms/frame), and
the ratio of the differences across arms. The question the table answers is
whether OUR deblock chain costs proportionally more than dav1d's at t=8 than it
does at t=1 -- i.e. whether the inflation is threading-dependent.

Two-point fit per row for wall and for user+sys, median across rounds with band.
"""

import collections
import statistics
import sys


def main():
    rows = []
    for line in open(sys.argv[1]):
        f = line.rstrip("\n").split("\t")
        if len(f) < 14:
            continue
        rnd, arm, il, vec, t, nlo, wlo, ulo, slo, nhi, whi, uhi, shi, fo = f[:14]
        t, nlo, nhi = int(t), int(nlo), int(nhi)
        wlo, ulo, slo, whi, uhi, shi = (int(x) for x in (wlo, ulo, slo, whi, uhi, shi))
        dn = nhi - nlo
        rows.append({"round": int(rnd), "arm": arm, "il": il, "vec": vec, "t": t,
                     "wall": (whi - wlo) / dn,
                     "cpu": ((uhi + shi) - (ulo + slo)) / dn,
                     "foreign": int(fo)})
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["il"], r["vec"], r["t"])].append(r)

    print(f"rows={len(rows)}  cells={len(by)}  "
          f"foreign_max={max(r['foreign'] for r in rows)}  "
          f"rounds={len({r['round'] for r in rows})}")
    print("  `nodeblock` CHANGES OUTPUT PIXELS -- attribution only, never an "
          "md5 comparison.")
    print()
    print("=" * 122)
    print("DEBLOCK CHAIN COST BY DECODER -- wall ms/frame at all vs nodeblock, "
          "and the difference")
    print("=" * 122)
    print(f"{'vector':24} {'t':>2} {'arm':10} {'all':>9} {'nodeblk':>9} "
          f"{'deblock':>9} {'db%':>6} {'cpu_all':>9} {'cpu_nodb':>9} "
          f"{'cpu_db':>9}")
    deb = {}
    for vec in sorted({k[2] for k in by}):
        for t in sorted({k[3] for k in by if k[2] == vec}):
            for arm in sorted({k[0] for k in by}):
                ka, kn = (arm, "all", vec, t), (arm, "nodeblock", vec, t)
                if ka not in by or kn not in by:
                    continue
                # Pair within round so drift cancels.
                ra = {r["round"]: r for r in by[ka]}
                rn = {r["round"]: r for r in by[kn]}
                common = sorted(set(ra) & set(rn))
                wa = statistics.median([ra[i]["wall"] for i in common])
                wn = statistics.median([rn[i]["wall"] for i in common])
                ca = statistics.median([ra[i]["cpu"] for i in common])
                cn = statistics.median([rn[i]["cpu"] for i in common])
                dw = statistics.median([ra[i]["wall"] - rn[i]["wall"] for i in common])
                dc = statistics.median([ra[i]["cpu"] - rn[i]["cpu"] for i in common])
                deb[(arm, vec, t)] = (dw, dc, wa, ca)
                print(f"{vec:24} {t:2d} {arm:10} {wa:9.3f} {wn:9.3f} "
                      f"{dw:+9.3f} {100 * dw / wa:6.1f} {ca:9.3f} {cn:9.3f} "
                      f"{dc:+9.3f}")
            print()

    print("=" * 122)
    print("DEBLOCK CPU ms/frame, and how it scales -- the claim is that OURS "
          "grows with threads and dav1d's does not")
    print("=" * 122)
    print(f"{'vector':24} {'arm':10} {'db_cpu@t1':>10} {'db_cpu@t8':>10} "
          f"{'t8/t1':>7} | {'ours/dav1d @t1':>15} {'@t8':>8}")
    for vec in sorted({k[1] for k in deb}):
        base = deb.get(("dav1d", vec, 1)), deb.get(("dav1d", vec, 8))
        for arm in sorted({k[0] for k in deb}):
            a1, a8 = deb.get((arm, vec, 1)), deb.get((arm, vec, 8))
            if not a1 or not a8:
                continue
            r1 = a1[1] / base[0][1] if base[0] and base[0][1] else float("nan")
            r8 = a8[1] / base[1][1] if base[1] and base[1][1] else float("nan")
            print(f"{vec:24} {arm:10} {a1[1]:10.3f} {a8[1]:10.3f} "
                  f"{a8[1] / a1[1] if a1[1] else float('nan'):7.3f} | "
                  f"{r1:15.3f} {r8:8.3f}")
        print()


if __name__ == "__main__":
    main()
