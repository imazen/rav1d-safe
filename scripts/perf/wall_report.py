#!/usr/bin/env python3
"""Two-point wall-clock fit: total = alpha + beta*frames, per (vector, threads, arm).

beta is the per-frame decode cost with process startup / mmap / teardown fitted
out; alpha is that fixed cost. Reported as the median beta over all rounds.
"""
import sys, statistics as st
from collections import defaultdict

rows = defaultdict(list)   # (vec, t, arm) -> [(alpha_ms, beta_ms)]
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 9 or f[1] != "WALLFIT":
        continue
    arm, vec, t = f[2], f[3], int(f[4])
    nlo, lo, nhi, hi = int(f[5]), float(f[6]), int(f[7]), float(f[8])
    if nhi == nlo:
        continue
    beta = (hi - lo) / (nhi - nlo)
    alpha = lo - beta * nlo
    rows[(vec, t, arm)].append((alpha, beta))

vecs = ["v256", "v1024", "v1024_10b", "v4k_1tile", "v4k_1tile_10b", "v4k_8tile", "v4k_8tile_10b"]
threads = [1, 2, 4, 8]
arms = ["rs", "dav1d_fd1", "dav1d_fdA"]

def beta(v, t, a):
    xs = rows.get((v, t, a))
    return st.median(b for _, b in xs) if xs else None

def alpha(v, t, a):
    xs = rows.get((v, t, a))
    return st.median(a_ for a_, _ in xs) if xs else None

print("per-frame decode cost beta (ms/frame), wall-clock two-point fit; "
      "alpha = fixed per-process cost (ms)\n")
hdr = (f"{'vector':<16}{'t':>2}  " + "".join(f"{a:>13}" for a in arms) +
       f"{'rs/fd1':>9}{'rs/fdA':>9}   " + "".join(f"{'a:'+a[:6]:>11}" for a in arms))
print(hdr); print("-" * len(hdr))
for v in vecs:
    for t in threads:
        bs = [beta(v, t, a) for a in arms]
        if all(x is None for x in bs):
            continue
        cells = "".join(f"{x:>13.3f}" if x is not None else f"{'-':>13}" for x in bs)
        r1 = f"{bs[0]/bs[1]:>9.2f}" if bs[0] and bs[1] and bs[1] > 0 else f"{'-':>9}"
        rA = f"{bs[0]/bs[2]:>9.2f}" if bs[0] and bs[2] and bs[2] > 0 else f"{'-':>9}"
        al = "".join(f"{alpha(v,t,a):>11.1f}" if alpha(v,t,a) is not None else f"{'-':>11}"
                     for a in arms)
        print(f"{v:<16}{t:>2}  {cells}{r1}{rA}   {al}")
    print()

print("\nthread scaling on beta (speedup vs that arm's own t=1)\n")
h2 = f"{'vector':<16}{'arm':<12}" + "".join(f"{'t='+str(t):>9}" for t in threads)
print(h2); print("-" * len(h2))
for v in vecs:
    for a in arms:
        b1 = beta(v, 1, a)
        if not b1:
            continue
        print(f"{v:<16}{a:<12}" + "".join(
            (f"{b1/beta(v,t,a):>9.2f}" if beta(v, t, a) else f"{'-':>9}") for t in threads))
    print()

ns = {k: len(v) for k, v in rows.items()}
print(f"\nfits per cell: min {min(ns.values())} max {max(ns.values())}")
