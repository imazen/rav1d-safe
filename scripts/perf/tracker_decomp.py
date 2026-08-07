#!/usr/bin/env python3
"""Decompose the DisjointMut borrow-tracker's cost from a 4-arm ab_sweep TSV.

The four arms are builds of `examples/bench_ab_decode.rs` that differ only in
which part of `BorrowTracker` is present:

  base       everything (this is what ships)
  noscan     lock + slot bookkeeping, overlap scan removed
  lockonly   lock/unlock only, no scan and no slot bookkeeping
  untracked  no tracker at all (every DisjointMut built untracked)

So, per (vector, threads) cell:

  scan       = base      - noscan     (the overlap scan inside the crit section)
  bookkeep   = noscan    - lockonly   (alloc/free, Location store, BorrowId)
  lock       = lockonly  - untracked  (raw acquire/release traffic, incl. spin)
  tracker    = base      - untracked  (all of it)

Usage: tracker_decomp.py <sweep.tsv> [--out summary.tsv]
"""

import sys
from collections import defaultdict
from statistics import median

ARMS = ["base", "noscan", "lockonly", "untracked"]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    times = defaultdict(list)
    geom = {}
    md5s = defaultdict(set)
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2:
                continue
            if f[1] == "RESULT":
                _, _, arm, vec, threads, _, _, _, per = f[:9]
                times[(vec, int(threads), arm)].append(float(per))
            elif f[1] == "GEOM":
                _, _, _, vec, _, dims, bpc = f[:7]
                geom[vec] = f"{dims} {bpc}"
            elif f[1] == "CHECKSUM":
                _, _, arm, vec, threads, m = f[:6]
                md5s[(vec, int(threads), arm)].add(m)

    keys = sorted({(v, t) for (v, t, _) in times})
    hdr = [
        "vector", "geom", "threads",
        "base_ms", "untracked_ms", "tracker_ms", "tracker_pct",
        "scan_ms", "bookkeep_ms", "lock_ms", "reps", "md5",
    ]
    rows = [hdr]

    def med(vec, t, arm):
        v = times.get((vec, t, arm), [])
        return median(v) if v else None

    for vec, t in keys:
        m = {a: med(vec, t, a) for a in ARMS}
        if any(m[a] is None for a in ARMS):
            rows.append([vec, geom.get(vec, "?"), str(t)]
                        + ["FAIL" if m[a] is None else f"{m[a]:.2f}" for a in ARMS]
                        + ["-"] * 5)
            continue
        tracker = m["base"] - m["untracked"]
        scan = m["base"] - m["noscan"]
        book = m["noscan"] - m["lockonly"]
        lock = m["lockonly"] - m["untracked"]
        nreps = min(len(times[(vec, t, a)]) for a in ARMS)
        hashes = {a: sorted(md5s.get((vec, t, a), set())) for a in ARMS}
        live = {a: h[0] for a, h in hashes.items() if h}
        md5note = "all-agree" if len(set(live.values())) <= 1 else "DIFFER"
        rows.append([
            vec, geom.get(vec, "?"), str(t),
            f"{m['base']:.2f}", f"{m['untracked']:.2f}",
            f"{tracker:.2f}", f"{100.0 * tracker / m['base']:.1f}",
            f"{scan:.2f}", f"{book:.2f}", f"{lock:.2f}",
            str(nreps), md5note,
        ])

    widths = [max(len(r[i]) for r in rows) for i in range(len(hdr))]
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip())

    # Scaling table: ms and speedup vs that arm's own threads=1.
    print()
    print("SCALING (speedup vs the same arm at threads=1)")
    hdr2 = ["vector", "arm", "t1_ms", "t2_ms", "t4_ms", "t8_ms",
            "t2x", "t4x", "t8x"]
    rows2 = [hdr2]
    for vec in sorted({v for v, _ in keys}):
        for arm in ARMS:
            ms = {t: med(vec, t, arm) for t in (1, 2, 4, 8)}
            if ms[1] is None:
                continue
            row = [vec, arm] + [
                f"{ms[t]:.1f}" if ms[t] is not None else "FAIL" for t in (1, 2, 4, 8)
            ]
            row += [
                f"{ms[1] / ms[t]:.2f}" if ms[t] else "-" for t in (2, 4, 8)
            ]
            rows2.append(row)
    widths2 = [max(len(r[i]) for r in rows2) for i in range(len(hdr2))]
    for r in rows2:
        print("  ".join(c.ljust(widths2[i]) for i, c in enumerate(r)).rstrip())

    if out:
        with open(out, "w") as fh:
            for r in rows:
                fh.write("\t".join(r) + "\n")
            fh.write("\n")
            for r in rows2:
                fh.write("\t".join(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
