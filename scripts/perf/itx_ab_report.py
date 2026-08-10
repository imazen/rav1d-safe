#!/usr/bin/env python3
"""Per-cell report for the itx-16bpc A/B/C sweep.

Columns: two-point wall fit beta = (ms_hi - ms_lo) / (n_hi - n_lo) per round,
then the median over complete rounds. A round is complete for a cell only if
every arm produced a row, so a partial trailing round is dropped and every cell
carries the same n.

The disjointness tick compares THE ARMS THE CLAIM COMPARES (base vs ABC, and
each incremental step) — not ours-vs-dav1d, which is two different decoders and
would be a tick that can never fail (SIZE_SWEEP.md trap 3).
"""
import collections
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/Users/lilith/tmp/itxwork/sweep1.tsv"
ARMS = ["base", "A", "AB", "ABC", "dav1d_fd1"]

per = collections.defaultdict(dict)   # (vec,t,round) -> arm -> beta
fmax = collections.defaultdict(int)
for line in open(path):
    f = line.rstrip("\n").split("\t")
    if len(f) < 9:
        continue
    rnd, arm, vec, t, nlo, mlo, nhi, mhi, fm = f
    beta = (int(mhi) - int(mlo)) / (int(nhi) - int(nlo))
    per[(vec, t, int(rnd))][arm] = beta
    fmax[(vec, t)] = max(fmax[(vec, t)], int(fm))

cells = collections.defaultdict(lambda: collections.defaultdict(list))
nround = collections.Counter()
for (vec, t, rnd), d in sorted(per.items()):
    if not all(a in d for a in ARMS):
        continue
    nround[(vec, t)] += 1
    for a in ARMS:
        cells[(vec, t)][a].append(d[a])


def band(xs):
    return min(xs), max(xs)


def disjoint(xs, ys):
    return max(xs) < min(ys) or max(ys) < min(xs)


order = [
    "L64x36_420_8b", "L256x144_420_8b", "L512x288_420_8b",
    "L1024x576_420_8b", "L2048x1152_420_8b", "L3840x2160_420_8b",
    "L64x36_420_10b", "L256x144_420_10b", "L512x288_420_10b",
    "L1024x576_420_10b", "L2048x1152_420_10b", "L3840x2160_420_10b",
    "L512x288_444_10b", "L1024x576_444_10b", "L3840x2160_444_10b",
    "v4k_8tile", "v4k_8tile_10b",
]
keys = sorted(cells, key=lambda k: (order.index(k[0]) if k[0] in order else 99, int(k[1])))

print("cell                        t  n  fmax   base_ms    ABC_ms  ABC/base  A/base  AB/A  ABC/AB   dj?"
      "   base/dav1d  ABC/dav1d")
for k in keys:
    vec, t = k
    c = cells[k]
    med = {a: statistics.median(c[a]) for a in ARMS}
    r = {a: [c[a][i] / c["base"][i] for i in range(len(c[a]))] for a in ARMS}
    rd = {a: [c[a][i] / c["dav1d_fd1"][i] for i in range(len(c[a]))] for a in ARMS}
    ab_a = [c["AB"][i] / c["A"][i] for i in range(len(c["A"]))]
    abc_ab = [c["ABC"][i] / c["AB"][i] for i in range(len(c["AB"]))]
    dj = "DJ " if disjoint(c["base"], c["ABC"]) else "  ."
    print(f"{vec:24s} {t:>2s} {nround[k]:>2d} {fmax[k]:>4d} "
          f"{med['base']:9.4f} {med['ABC']:9.4f} "
          f"{statistics.median(r['ABC']):8.4f} {statistics.median(r['A']):7.4f} "
          f"{statistics.median(ab_a):5.4f} {statistics.median(abc_ab):6.4f} {dj}"
          f"  {statistics.median(rd['base']):10.4f} {statistics.median(rd['ABC']):10.4f}")

print()
print("per-arm bands (ms/frame): min..max over complete rounds")
for k in keys:
    vec, t = k
    c = cells[k]
    s = "  ".join(f"{a}=[{min(c[a]):.4f}..{max(c[a]):.4f}]" for a in ARMS)
    print(f"  {vec:24s} t={t}  {s}")

print()
print("paired ratio bands (per-round ABC/base and ABC/dav1d):")
for k in keys:
    vec, t = k
    c = cells[k]
    rr = sorted(c["ABC"][i] / c["base"][i] for i in range(len(c["base"])))
    rdv = sorted(c["ABC"][i] / c["dav1d_fd1"][i] for i in range(len(c["base"])))
    print(f"  {vec:24s} t={t}  ABC/base med={statistics.median(rr):.4f} "
          f"[{rr[0]:.4f}..{rr[-1]:.4f}]   ABC/dav1d med={statistics.median(rdv):.4f} "
          f"[{rdv[0]:.4f}..{rdv[-1]:.4f}]")

print()
print("depth penalty (10bpc ms / 8bpc ms at the same size, 4:2:0), per arm, per-round paired:")
sizes = ["L64x36", "L256x144", "L512x288", "L1024x576", "L2048x1152", "L3840x2160"]
print(f"{'size':14s} {'base':>8s} {'ABC':>8s}   {'dav1d':>8s}")
for s in sizes:
    k8, k10 = (f"{s}_420_8b", "1"), (f"{s}_420_10b", "1")
    if k8 not in cells or k10 not in cells:
        continue
    n = min(len(cells[k8]["base"]), len(cells[k10]["base"]))
    row = []
    for a in ("base", "ABC", "dav1d_fd1"):
        rr = sorted(cells[k10][a][i] / cells[k8][a][i] for i in range(n))
        row.append(f"{statistics.median(rr):8.4f}")
    print(f"{s:14s} " + " ".join(row) + f"   (n={n})")
