#!/usr/bin/env python3
"""Reduce recheck_482.sh rows to paired verdicts.

Input columns (TSV, no header):
    round arm vec threads nlo ext_lo nhi ext_hi int_lo int_hi f_arm f_grp

Two independent instruments per (round, cell, arm):
  * ext  -- two-point wall fit, (ext_hi - ext_lo) / (nhi - nlo). Removes exec,
            mmap, container parse and decoder construction. This is the
            instrument BOTH disputed measurements used.
  * int  -- the harness's own in-process timer at the high frame count, which
            brackets exactly the timed decodes. Independent of process startup
            entirely.
Both are reported for every pair. Agreement is evidence; disagreement is a
finding.

Verdicts are PAIRED WITHIN (round, cell): the arms in one group ran
back-to-back and saw the same machine, so their ratio survives a busy box even
though their absolute ms does not. A pair is called only on the ratio band --
`[min..max]` over rounds -- excluding 1.0. That band compares the two arms the
CLAIM compares, so it can actually fail; an arms-vs-dav1d band could not.

Usage: recheck_482_report.py <gap.tsv> [--clean-only] [--pairs a/b,c/d]
"""

import sys
from collections import defaultdict
from math import comb


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def sign_p(wins, n):
    """Exact two-sided sign test, ties already excluded from n."""
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def main():
    args = [a for a in sys.argv[1:]]
    clean_only = "--clean-only" in args
    args = [a for a in args if a != "--clean-only"]
    pairs = ["head/parent", "headoff/parent", "head/headoff",
             "main/parent", "mainoff/parent", "main/head"]
    dump = None
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--pairs" and i + 1 < len(args):
            pairs = args[i + 1].split(",")
            i += 2
        elif a == "--dump" and i + 1 < len(args):   # "<pair>:<vector> t=<n>"
            p, c = args[i + 1].split(":", 1)
            dump = (p, c)
            i += 2
        elif a.startswith("--"):
            i += 1
        else:
            positional.append(a)
            i += 1
    if not positional:
        sys.exit(__doc__)
    path = positional[0]
    per_round = {}

    # (cell, arm, round) -> {"ext": ms/frame, "int": ms/frame, "pos": k}
    #
    # `pos` is the arm's EXECUTION POSITION inside its (round, cell) group.
    # The harness writes rows in execution order, so row order recovers it
    # without re-running anything. It matters: the rotation advances by one
    # arm per round, so with N arms the same PAIR sits adjacent in N-1 of
    # every N rounds and maximally separated in the Nth. If position carries
    # a cost, a paired ratio inherits it asymmetrically.
    data = defaultdict(dict)
    fgrp = {}
    seq = defaultdict(int)
    cells, arms = [], []
    dropped = 0
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 12:
            continue
        rnd, arm, vec, thr = int(f[0]), f[1], f[2], int(f[3])
        nlo, elo, nhi, ehi = int(f[4]), float(f[5]), int(f[6]), float(f[7])
        ilo, ihi, f_grp = f[8], f[9], int(f[11])
        cell = (vec, thr)
        pos = seq[(cell, rnd)]
        seq[(cell, rnd)] += 1
        if clean_only and f_grp != 0:
            dropped += 1
            continue
        ext = (ehi - elo) / (nhi - nlo)
        try:
            itn = float(ihi)
        except ValueError:
            itn = float("nan")
        # A mirrored harness runs each arm twice per group (forward pass and
        # reverse pass). Reduce the pair by MINIMUM: the quantity is fixed and
        # the noise is one-sided-positive, so the smaller observation is the
        # less-disturbed one. Single-pass data is unaffected.
        prev = data[(cell, arm)].get(rnd)
        rec = {"ext": ext, "int": itn, "pos": pos}
        if prev is not None:
            rec = {
                "ext": min(prev["ext"], ext),
                "int": min(x for x in (prev["int"], itn) if x == x)
                if (prev["int"] == prev["int"] or itn == itn) else float("nan"),
                "pos": prev["pos"],
                "pos2": pos,
            }
        data[(cell, arm)][rnd] = rec
        fgrp[(cell, rnd)] = f_grp
        if cell not in cells:
            cells.append(cell)
        if arm not in arms:
            arms.append(arm)

    print(f"# source: {path}")
    print(f"# rows kept: {sum(len(v) for v in data.values())}"
          + (f"  (dropped {dropped} loaded rows)" if clean_only else ""))
    if fgrp:
        loaded = sum(1 for v in fgrp.values() if v != 0)
        print(f"# groups: {len(fgrp)}   under foreign load: {loaded}"
              f"   idle: {len(fgrp) - loaded}")
    print()

    # --- absolutes, per arm, both instruments -------------------------------
    print("## ms/frame per arm  (median [min..max], n)")
    print()
    hdr = f"{'cell':<26} {'arm':<9} {'ext median':>11} {'ext band':>21} "\
          f"{'int median':>11} {'int band':>21} {'n':>3}"
    print(hdr)
    print("-" * len(hdr))
    for cell in cells:
        for arm in arms:
            d = data.get((cell, arm))
            if not d:
                continue
            e = [v["ext"] for v in d.values()]
            i = [v["int"] for v in d.values() if v["int"] == v["int"]]
            cn = f"{cell[0]} t={cell[1]}"
            ib = f"[{min(i):.2f}..{max(i):.2f}]" if i else "-"
            im = f"{median(i):.3f}" if i else "-"
            print(f"{cn:<26} {arm:<9} {median(e):>11.3f} "
                  f"{'[' + format(min(e), '.2f') + '..' + format(max(e), '.2f') + ']':>21} "
                  f"{im:>11} {ib:>21} {len(e):>3}")
        print()

    # --- position effect ----------------------------------------------------
    # Normalise each arm's ms/frame by its own median across rounds, then
    # average by execution position. A flat row means position is free and any
    # paired ratio is clean. A sloped row means the rotation is injecting a
    # bias into every pair whose two arms do not sit adjacent.
    print("## execution-position effect  (per-arm ms/frame / that arm's own median)")
    print("## flat = position is free; sloped = the rotation biases paired ratios")
    print()
    npos = max((v["pos"] for d in data.values() for v in d.values()), default=-1) + 1
    hdr = f"{'cell':<26} " + " ".join(f"{'pos' + str(k):>8}" for k in range(npos)) + f" {'spread':>8}"
    print(hdr)
    print("-" * len(hdr))
    for cell in cells:
        by_pos = defaultdict(list)
        for arm in arms:
            d = data.get((cell, arm))
            if not d:
                continue
            vals = [v["int"] for v in d.values() if v["int"] == v["int"]]
            if not vals:
                continue
            m = median(vals)
            for v in d.values():
                if v["int"] == v["int"] and m:
                    by_pos[v["pos"]].append(v["int"] / m)
        if not by_pos:
            continue
        means = [median(by_pos[k]) if by_pos.get(k) else float("nan")
                 for k in range(npos)]
        fin = [x for x in means if x == x]
        spread = (max(fin) - min(fin)) * 100 if fin else float("nan")
        cn = f"{cell[0]} t={cell[1]}"
        print(f"{cn:<26} " + " ".join(f"{x:>8.4f}" for x in means)
              + f" {spread:>7.2f}%")
    print()

    # --- paired ratios ------------------------------------------------------
    print("## paired within-round ratios  (the arms the claim compares)")
    print("## VERDICT: 'disjoint' means the ratio band excludes 1.0.")
    print()
    hdr = f"{'cell':<26} {'pair':<18} {'inst':<4} {'median':>8} {'band':>20} "\
          f"{'wins':>7} {'p':>7}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for cell in cells:
        for pair in pairs:
            a, b = pair.split("/")
            da, db = data.get((cell, a)), data.get((cell, b))
            if not da or not db:
                continue
            rounds = sorted(set(da) & set(db))
            if not rounds:
                continue
            for inst in ("ext", "int"):
                rs = []
                for r in rounds:
                    x, y = da[r][inst], db[r][inst]
                    if x == x and y == y and y != 0:
                        rs.append(x / y)
                if not rs:
                    continue
                lo, hi, med = min(rs), max(rs), median(rs)
                # wins = rounds where the numerator arm is FASTER
                w = sum(1 for v in rs if v < 1.0)
                ties = sum(1 for v in rs if v == 1.0)
                p = sign_p(w, len(rs) - ties)
                # A band over fewer than 5 rounds is a point, not a band --
                # calling it disjoint would be a tick that cannot fail.
                # Sign convention: the percentage is always the NUMERATOR's
                # change in decode time against the denominator. Negative is
                # faster. No "+2% faster" ambiguity.
                pct = (med - 1) * 100
                if len(rs) < 5:
                    verdict = f"n={len(rs)} TOO FEW to call"
                elif hi < 1.0:
                    verdict = f"{pct:+.2f}% FASTER, disjoint"
                elif lo > 1.0:
                    verdict = f"{pct:+.2f}% SLOWER, disjoint"
                else:
                    verdict = f"{pct:+.2f}%, straddles 1.0"
                cn = f"{cell[0]} t={cell[1]}" if inst == "ext" else ""
                pn = pair if inst == "ext" else ""
                print(f"{cn:<26} {pn:<18} {inst:<4} {med:>8.4f} "
                      f"{'[' + format(lo, '.4f') + '..' + format(hi, '.4f') + ']':>20} "
                      f"{str(w) + '/' + str(len(rs)):>7} {p:>7.3f}  {verdict}")
                if dump and pair == dump[0] and f"{cell[0]} t={cell[1]}" == dump[1]:
                    per_round.setdefault(inst, [
                        (r, da[r]["pos"], db[r]["pos"], da[r][inst] / db[r][inst])
                        for r in rounds if db[r][inst]])
                # Split the same ratios by execution order. If the two arms'
                # positions carry a cost, "numerator first" and "numerator
                # last" disagree -- and the rotation does not visit the two
                # equally often, so the median inherits the imbalance.
                before = [da[r][inst] / db[r][inst] for r in rounds
                          if db[r][inst] and da[r]["pos"] < db[r]["pos"]]
                after = [da[r][inst] / db[r][inst] for r in rounds
                         if db[r][inst] and da[r]["pos"] > db[r]["pos"]]
                if inst == "int" and before and after:
                    print(f"{'':<26} {'':<18} {'  ^ by order':<4} "
                          f"num-first {median(before):.4f} (n={len(before)})   "
                          f"num-last {median(after):.4f} (n={len(after)})   "
                          f"gap {(median(after) - median(before)) * 100:+.2f}%")
        print()

    # --- per-round dump -----------------------------------------------------
    # The disputed measurement was a median over 5 rounds. Printing every round
    # of this one says whether a 5-round window could land where it landed.
    if dump and per_round:
        print(f"## every round of {dump[0]} at {dump[1]}")
        print("## (round, position of numerator, position of denominator, ratio)")
        print()
        for inst, rs in per_round.items():
            vals = [v for _, _, _, v in rs]
            print(f"{inst}:")
            print("   " + "  ".join(f"r{r}:p{pa}/p{pb}={v:.4f}"
                                    for r, pa, pb, v in rs))
            over = [v for v in vals if v >= 1.04]
            print(f"   rounds >= 1.04 (the size sweep's claim level): "
                  f"{len(over)} of {len(vals)}")
        print()


if __name__ == "__main__":
    main()
