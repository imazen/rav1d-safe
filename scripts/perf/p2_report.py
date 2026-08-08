"""Report a p2_sweep.sh TSV.

Columns: round, vector, threads, arm, ms_per_frame, frame_md5[, busy]

The trailing `busy` column is written by `STRICT=0` runs (a box shared with
other agents, where a genuinely idle window may never arrive). When any row is
tagged busy the absolute ms/frame is INFLATED by foreign load and must not be
quoted as a clean number — but the arms still run back to back inside a cell
with the order rotating, so the PAIRED per-round ratio is sound. Both are
printed; the `clean` table appears only if clean rows exist.
"""

import sys, statistics, collections

path = sys.argv[1]
rows = collections.defaultdict(list)      # (vec,t,arm) -> [ms]
clean = collections.defaultdict(list)     # (vec,t,arm) -> [ms] with busy==0
per_round = collections.defaultdict(dict) # (vec,t,round) -> {arm: median ms}
md5s = collections.defaultdict(set)
nbusy = collections.Counter()
ntot = collections.Counter()
stage = collections.defaultdict(lambda: collections.defaultdict(list))

for line in open(path):
    p = line.rstrip("\n").split("\t")
    if len(p) < 6:
        continue
    r, vec, t, arm, ms, md5 = p[0], p[1], int(p[2]), p[3], float(p[4]), p[5]
    busy = int(p[6]) if len(p) > 6 and p[6] != "" else 0
    rows[(vec, t, arm)].append(ms)
    ntot[(vec, t)] += 1
    if busy:
        nbusy[(vec, t)] += 1
    else:
        clean[(vec, t, arm)].append(ms)
    stage[(vec, t, r)][arm].append(ms)
    md5s[(vec, t)].add(md5)

for key, per_arm in stage.items():
    vec, t, r = key
    per_round[(vec, t)][r] = {a: statistics.median(v) for a, v in per_arm.items()}

known = ["base", "itx8", "cdef", "lfmask", "lfbatch", "lfneon", "head"]
arms = sorted({k[2] for k in rows}, key=lambda a: known.index(a) if a in known else 99)
cells = sorted({(k[0], k[1]) for k in rows})
last = arms[-1]


def table(src, title):
    have = [c for c in cells if all(src.get((c[0], c[1], a)) for a in arms)]
    if not have:
        return
    print(f"\n## {title}")
    print(f"{'vector':<18}{'t':>3}  " + "".join(f"{a:>11}" for a in arms) + f"   base/{last}   n/arm")
    for vec, t in have:
        med = {a: statistics.median(src[(vec, t, a)]) for a in arms}
        line = f"{vec:<18}{t:>3}  " + "".join(f"{med[a]:>11.1f}" for a in arms)
        print(line + f"   {med['base'] / med[last]:>9.3f}x   {len(src[(vec, t, arms[0])]):>5}")


table(rows, "median of ALL rows")
table(clean, "median of rows the idle guard passed (busy=0)")

print("\n## paired per-round ratios (robust to steady foreign load)")
print(f"{'vector':<18}{'t':>3}  {'median':>9} {'min':>9} {'max':>9}  {'rounds':>7}  {'busy rows':>10}")
for vec, t in cells:
    rr = [
        d["base"] / d[last]
        for d in per_round[(vec, t)].values()
        if "base" in d and last in d
    ]
    if not rr:
        continue
    print(
        f"{vec:<18}{t:>3}  {statistics.median(rr):>8.3f}x {min(rr):>8.3f}x {max(rr):>8.3f}x"
        f"  {len(rr):>7}  {nbusy[(vec, t)]:>4}/{ntot[(vec, t)]:<5}"
    )

bad = {f"{v}:{t}": sorted(s) for (v, t), s in sorted(md5s.items()) if len(s) != 1}
print("\nframe md5 per (vector,threads) — one distinct value per cell means every arm agreed")
print("  DISAGREEMENTS:", bad if bad else "none")
