import sys, statistics, collections
path = sys.argv[1]
rows = collections.defaultdict(list)
md5s = collections.defaultdict(set)
for line in open(path):
    p = line.rstrip("\n").split("\t")
    if len(p) < 6: continue
    _r, vec, t, arm, ms, md5 = p[0], p[1], p[2], p[3], float(p[4]), p[5]
    rows[(vec, int(t), arm)].append(ms)
    md5s[(vec, int(t))].add(md5)
arms = sorted({k[2] for k in rows}, key=lambda a: ["base","itx8","cdef","lfmask","lfbatch","head"].index(a) if a in ["base","itx8","cdef","lfmask","lfbatch","head"] else 99)
cells = sorted({(k[0], k[1]) for k in rows})
print(f"{'vector':<18}{'t':>3}  " + "".join(f"{a:>11}" for a in arms) + "   base/head   md5s")
for (vec, t) in cells:
    line = f"{vec:<18}{t:>3}  "
    meds = {}
    for a in arms:
        v = rows.get((vec,t,a))
        meds[a] = statistics.median(v) if v else float("nan")
        line += f"{meds[a]:>11.1f}"
    last = arms[-1]
    line += f"   {meds['base']/meds[last]:>8.3f}x   {len(md5s[(vec,t)])}"
    print(line)
print()
print("n per cell/arm:", {a: len(rows[(cells[0][0], cells[0][1], a)]) for a in arms})
print("distinct md5 per (vec,t) — must be 1:", {f"{v}:{t}": sorted(s) for (v,t),s in sorted(md5s.items())})
