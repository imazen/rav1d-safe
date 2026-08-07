#!/usr/bin/env python3
"""Summarise an `ab_sweep.sh` TSV into a per-cell median table with ratios.

Usage: ab_report.py <sweep_raw.tsv> [--baseline LABEL] [--out summary.tsv]

Reports, per (vector, threads): the median ms/frame of each arm across every
rep of every round, each arm's ratio against the baseline arm, and the
decoded-frame md5 of each arm (so the same sweep doubles as a bit-identity
check). A cell whose arm failed to decode is reported as FAIL, not omitted.
"""

import sys
from collections import defaultdict
from statistics import median

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    times = defaultdict(list)      # (vec, threads, arm) -> [ms/frame]
    md5s = defaultdict(set)        # (vec, threads, arm) -> {md5}
    geom = {}                      # vec -> "WxH bpc"
    fails = defaultdict(int)       # (vec, threads, arm) -> count

    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2:
                continue
            kind = f[1]
            if kind == "RESULT":
                # round RESULT label file threads rep iters ms_total ms_per_frame
                _, _, arm, vec, threads, _, _, _, per = f[:9]
                times[(vec, int(threads), arm)].append(float(per))
            elif kind == "CHECKSUM":
                _, _, arm, vec, threads, md5 = f[:6]
                md5s[(vec, int(threads), arm)].add(md5)
            elif kind == "GEOM":
                _, _, _, vec, _, dims, bpc = f[:7]
                geom[vec] = f"{dims} {bpc}"
            elif kind == "FAIL":
                _, _, arm, vec, threads, _ = f[:6]
                fails[(vec, int(threads), arm)] += 1

    arms = []
    for (_, _, arm) in list(times) + list(fails):
        if arm not in arms:
            arms.append(arm)
    baseline = "before"
    if "--baseline" in sys.argv:
        baseline = sys.argv[sys.argv.index("--baseline") + 1]
    if baseline not in arms and arms:
        baseline = arms[0]
    others = [a for a in arms if a != baseline]

    keys = sorted({(v, t) for (v, t, _) in list(times) + list(fails)},
                  key=lambda k: (k[0], k[1]))
    hdr = ["vector", "geom", "threads", f"{baseline}_ms"]
    for a in others:
        hdr += [f"{a}_ms", f"{a}/{baseline}"]
    hdr += ["reps", "md5"]
    rows = [hdr]

    def med(vec, t, arm):
        v = times.get((vec, t, arm), [])
        return median(v) if v else None

    for vec, t in keys:
        base = med(vec, t, baseline)
        row = [vec, geom.get(vec, "?"), str(t),
               f"{base:.3f}" if base is not None else "FAIL"]
        for a in others:
            m = med(vec, t, a)
            row.append(f"{m:.3f}" if m is not None else "FAIL")
            row.append(f"{m / base:.4f}" if (m is not None and base) else "-")
        nreps = max((len(times.get((vec, t, a), [])) for a in arms), default=0)
        hashes = {a: sorted(md5s.get((vec, t, a), set())) for a in arms}
        live = {a: h[0] for a, h in hashes.items() if h}
        unstable = any(len(h) > 1 for h in hashes.values())
        if unstable:
            md5note = "UNSTABLE"
        elif len(set(live.values())) <= 1:
            md5note = "all-agree" if live else "-"
        else:
            md5note = "DIFFER " + repr(live)
        row += [str(nreps), md5note]
        rows.append(row)

    widths = [max(len(r[i]) for r in rows) for i in range(len(hdr))]
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip())

    print()
    for (vec, t, arm), n in sorted(fails.items()):
        print(f"FAILED TO DECODE: {arm} {vec} threads={t} ({n} invocations)")

    print()
    for arm in arms:
        bad = []
        for vec in sorted(geom):
            hashes = {t: next(iter(md5s[(vec, t, arm)]))
                      for t in (1, 2, 4, 8) if md5s.get((vec, t, arm))}
            if len(set(hashes.values())) > 1:
                bad.append(f"{vec}: {hashes}")
        print(f"{arm}: thread-count md5 divergence: "
              f"{'NONE' if not bad else '; '.join(bad)}")

    if out:
        with open(out, "w") as fh:
            for r in rows:
                fh.write("\t".join(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
