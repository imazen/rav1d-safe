#!/usr/bin/env python3
"""Median-per-cell report for the rav1d-safe vs dav1d yardstick sweep."""
import sys, statistics as st
from collections import defaultdict

path = sys.argv[1]
res = defaultdict(list)     # (vec, t, arm) -> [ms_per_frame]
geom, loads, fails = {}, defaultdict(list), []
csums = defaultdict(set)

for line in open(path):
    f = line.rstrip("\n").split("\t")
    if len(f) < 3:
        continue
    kind = f[1]
    if kind == "RESULT" and len(f) >= 9:
        arm, vec, t = f[2], f[3].replace(".avif", ""), int(f[4])
        res[(vec, t, arm)].append(float(f[8]))
    elif kind == "GEOM":
        geom[f[3].replace(".avif", "")] = f"{f[5]} {f[6]}"
    elif kind == "LOAD":
        loads[(f[3].replace('.avif',''), int(f[4]))].append(float(f[5]))
    elif kind == "CHECKSUM":
        csums[f[3].replace(".avif", "")].add(f[5])
    elif kind == "FAIL":
        fails.append(line.rstrip())

vecs = ["v256", "v1024", "v1024_10b", "v4k_1tile", "v4k_1tile_10b", "v4k_8tile", "v4k_8tile_10b"]
threads = [1, 2, 4, 8]
arms = ["rs", "dav1d1", "dav1dA"]

def med(v, t, a):
    xs = res.get((v, t, a))
    return st.median(xs) if xs else None

print("ms/frame (median of all reps/rounds); n = sample count\n")
hdr = f"{'vector':<16}{'geom':<18}{'t':>2}  " + "".join(f"{a:>12}" for a in arms) + \
      f"{'rs/dav1d1':>11}{'rs/dav1dA':>11}   n(rs)"
print(hdr)
print("-" * len(hdr))
for v in vecs:
    for t in threads:
        vals = [med(v, t, a) for a in arms]
        if all(x is None for x in vals):
            continue
        cells = "".join(f"{x:>12.3f}" if x is not None else f"{'-':>12}" for x in vals)
        r1 = f"{vals[0]/vals[1]:>11.2f}" if vals[0] and vals[1] else f"{'-':>11}"
        rA = f"{vals[0]/vals[2]:>11.2f}" if vals[0] and vals[2] else f"{'-':>11}"
        n = len(res.get((v, t, 'rs'), []))
        print(f"{v:<16}{geom.get(v,'?'):<18}{t:>2}  {cells}{r1}{rA}{n:>8}")
    print()

print("\nthread scaling (speedup vs that arm's own t=1)\n")
hdr2 = f"{'vector':<16}{'arm':<8}" + "".join(f"{'t='+str(t):>9}" for t in threads)
print(hdr2); print("-" * len(hdr2))
for v in vecs:
    for a in arms:
        base = med(v, 1, a)
        if base is None:
            continue
        row = "".join(
            (f"{base/med(v,t,a):>9.2f}" if med(v, t, a) else f"{'-':>9}") for t in threads)
        print(f"{v:<16}{a:<8}{row}")
    print()

print("\nchecksums (rav1d-safe, must be one per vector across all thread counts):")
for v in vecs:
    if v in csums:
        print(f"  {v:<16}{'  '.join(sorted(csums[v]))}   [{len(csums[v])} distinct]")

mx = max((max(x) for x in loads.values()), default=0)
print(f"\nmax 1-min load average recorded during the sweep: {mx:.2f}")
if fails:
    print(f"\n{len(fails)} FAIL rows:")
    for f in fails[:20]:
        print("  " + f)
