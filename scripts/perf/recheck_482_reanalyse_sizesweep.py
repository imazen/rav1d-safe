#!/usr/bin/env python3
"""Re-derive, and then re-reduce, the size-sweep round's disputed 4K cell.

The size sweep (docs/SIZE_SWEEP.md, "Does #482 close the hump?") reported
`rs2/rs = 1.0854` at `L3840x2160_420_8b` t=1 -- a 9% regression -- from n=5
load-tagged rounds, and flagged it as the one cell it could not settle.

Its raw rows are still on disk. This does two things with them:

  1. REPRODUCES the published median from the raw rows, so the reanalysis is
     demonstrably operating on the same data and the same fit.
  2. RE-REDUCES the same rows with an estimator that suits the noise. The
     per-round paired median assumes the two arms in a round saw the same
     machine. When a run can land in a slower regime -- contention, core
     placement, a frequency state -- the disturbance is one-sided-positive,
     and the MINIMUM over rounds is the least-disturbed observation of each
     arm's fixed cost. The median of a 5-sample paired ratio is not.

No new measurement. Same file, same two-point fit, different reduction.

Usage: recheck_482_reanalyse_sizesweep.py [mainarm_gap.tsv]
"""

import os
import sys
from collections import defaultdict


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/tmp/szsweep/p23/mainarm_gap.tsv")
    cell = sys.argv[2] if len(sys.argv) > 2 else "L3840x2160_420_8b"

    beta = defaultdict(dict)          # arm -> round -> ms/frame
    order = defaultdict(list)         # round -> [arm, ...] in execution order
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != cell or f[3] != "1":
            continue
        rnd, arm = int(f[0]), f[1]
        nlo, elo, nhi, ehi = int(f[4]), float(f[5]), int(f[6]), float(f[7])
        beta[arm][rnd] = (ehi - elo) / (nhi - nlo)
        order[rnd].append(arm)

    rounds = sorted(order)
    print(f"# {path}")
    print(f"# cell {cell} t=1,  rounds {rounds}\n")

    print("## the rows, as fitted ms/frame, in execution order")
    print()
    print(f"{'round':<6} {'execution order':<34} " +
          "  ".join(f"{a:>10}" for a in ("rs", "rs2")))
    print("-" * 78)
    for r in rounds:
        oo = " -> ".join(order[r])
        cells = "  ".join(f"{beta[a].get(r, float('nan')):>10.2f}"
                          for a in ("rs", "rs2"))
        print(f"{r:<6} {oo:<34} {cells}")
    print()

    pr = [beta["rs2"][r] / beta["rs"][r] for r in rounds
          if r in beta["rs"] and r in beta["rs2"]]
    print("## reduction 1 -- paired per-round ratio, then median (as published)")
    print()
    print("   per round: " + "  ".join(f"{v:.4f}" for v in pr))
    print(f"   median {median(pr):.4f}   band [{min(pr):.4f}..{max(pr):.4f}]"
          f"   spread {(max(pr) - min(pr)) * 100:.1f} points")
    print(f"   rounds >= 1.04: {sum(1 for v in pr if v >= 1.04)} of {len(pr)}")
    print()

    print("## reduction 2 -- per-arm MINIMUM over rounds, then ratio")
    print()
    mn = {a: min(beta[a].values()) for a in ("rs", "rs2")}
    md = {a: median(list(beta[a].values())) for a in ("rs", "rs2")}
    for a in ("rs", "rs2"):
        vals = [beta[a][r] for r in rounds]
        print(f"   {a:<4} min {mn[a]:7.2f}   median {md[a]:7.2f}   "
              f"max {max(vals):7.2f}   spread {(max(vals)/min(vals)-1)*100:5.1f}%")
    print()
    print(f"   rs2/rs by MINIMUM : {mn['rs2'] / mn['rs']:.4f}")
    print(f"   rs2/rs by MEDIAN  : {md['rs2'] / md['rs']:.4f}")
    print()
    print("   (#482's own two rounds on v4k_8tile reported 0.9823 and 0.9790.)")
    print()

    # --- is the noise random, or ordered? -----------------------------------
    # If the disturbance were random per run, "which arm ran later" would carry
    # no information. Test it directly: within each round, compare the arm that
    # ran later against the one that ran earlier.
    print("## the noise is not random -- it is ORDERED within the group")
    print()
    print(f"   {'round':<6} {'earlier':<22} {'later':<22} {'later/earlier':>14}")
    print("   " + "-" * 66)
    lat = []
    for r in rounds:
        pos = {a: i for i, a in enumerate(order[r])}
        a, b = ("rs", "rs2") if pos["rs"] < pos["rs2"] else ("rs2", "rs")
        ratio = beta[b][r] / beta[a][r]
        lat.append(ratio)
        sep = pos[b] - pos[a] - 1
        print(f"   {r:<6} {a + ' @' + str(pos[a]) + ' ' + format(beta[a][r], '.2f'):<22} "
              f"{b + ' @' + str(pos[b]) + ' ' + format(beta[b][r], '.2f'):<22} "
              f"{ratio:>13.4f}" + (f"   ({sep} arm between)" if sep else ""))
    slower = sum(1 for v in lat if v > 1.0)
    print()
    print(f"   the LATER arm was slower in {slower} of {len(lat)} rounds; "
          f"median later/earlier = {median(lat):.4f}")
    print("   Position, not code: it lands on whichever arm the rotation put")
    print("   second, and a 3-arm rotation does not put each arm there equally")
    print("   often -- so it does not average out of a 5-round median.")
    print()

    # --- compare the arms at MATCHED position -------------------------------
    # The confound-free comparison. Five rounds over three positions leaves
    # only 1-2 samples per (arm, position) cell, so this is reported WITH its n
    # and is not decisive by itself -- but it is the only reduction of these
    # rows that does not let position masquerade as code.
    print("## the same rows compared at MATCHED position (n is small -- shown)")
    print()
    print(f"   {'position':<10} {'rs':>18} {'rs2':>18} {'rs2/rs':>10}")
    print("   " + "-" * 58)
    npos = max(len(order[r]) for r in rounds)
    for k in range(npos):
        got = {}
        for a in ("rs", "rs2"):
            vals = [beta[a][r] for r in rounds
                    if r in beta[a] and order[r].index(a) == k]
            if vals:
                got[a] = sum(vals) / len(vals), len(vals)
        if len(got) == 2:
            (ra, na), (rb, nb) = got["rs"], got["rs2"]
            print(f"   {k:<10} {format(ra, '.2f') + ' (n=' + str(na) + ')':>18} "
                  f"{format(rb, '.2f') + ' (n=' + str(nb) + ')':>18} "
                  f"{rb / ra:>10.4f}")
        else:
            have = ", ".join(f"{a} n={v[1]}" for a, v in got.items()) or "none"
            print(f"   {k:<10} {'-- only ' + have:>48}")


if __name__ == "__main__":
    main()
