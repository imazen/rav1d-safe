#!/usr/bin/env python3
"""Summarise scripts/perf/tshift_ab.sh output.

Prints median AND min/max per arm per cell, because the brief's rule is that a
sub-3% claim is not believable until the two arms' ranges are shown not to
overlap (an 88.0 -> 85.6 headline once had base [85.50..91.11] against head
[84.89..91.50] at n=5, i.e. null).

Also set-diffs the per-arm md5 BY VALUE, so an arm that changed output shows up
here rather than in a later conformance run.

Usage: tshift_report.py <tsv> [baseline_arm]
"""

import sys
from collections import defaultdict


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    path = sys.argv[1]
    base_arm = sys.argv[2] if len(sys.argv) > 2 else None

    ms = defaultdict(list)          # (vec, t, arm) -> [ms]
    md5s = defaultdict(set)         # (vec, t, arm) -> {md5}
    foreign = defaultdict(int)      # (vec, t) -> max foreign
    arms, cells = [], []

    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 7:
            continue
        _, arm, vec, t, msv, md5, frn = f[:7]
        if not msv:
            continue
        key = (vec, int(t))
        ms[(vec, int(t), arm)].append(float(msv))
        md5s[(vec, int(t), arm)].add(md5)
        foreign[key] = max(foreign[key], int(frn))
        if arm not in arms:
            arms.append(arm)
        if key not in cells:
            cells.append(key)

    if base_arm is None:
        base_arm = arms[0]

    print(f"baseline arm: {base_arm}")
    hdr = f"{'vec':<16}{'t':>3}  {'arm':<12}{'n':>3}{'med':>9}{'min':>9}{'max':>9}{'ratio':>8}  md5"
    print(hdr)
    print("-" * len(hdr))
    for vec, t in cells:
        bl = ms.get((vec, t, base_arm))
        bmed = median(bl) if bl else None
        for arm in arms:
            v = ms.get((vec, t, arm))
            if not v:
                continue
            m = median(v)
            ratio = f"{m / bmed:.4f}" if bmed else "-"
            hs = md5s[(vec, t, arm)]
            tag = next(iter(hs))[:8] if len(hs) == 1 else "UNSTABLE:" + ",".join(
                sorted(h[:8] for h in hs)
            )
            print(
                f"{vec:<16}{t:>3}  {arm:<12}{len(v):>3}{m:>9.1f}{min(v):>9.1f}"
                f"{max(v):>9.1f}{ratio:>8}  {tag}"
            )
        # Range-overlap verdict against the baseline arm.
        for arm in arms:
            if arm == base_arm or (vec, t, arm) not in ms:
                continue
            a, b = ms[(vec, t, base_arm)], ms[(vec, t, arm)]
            sep = "SEPARATED" if (max(b) < min(a) or min(b) > max(a)) else "OVERLAP"
            print(f"{'':<16}{'':>3}  {arm} vs {base_arm}: ranges {sep}")
        print(f"{'':<16}{'':>3}  foreign>25%: {foreign[(vec, t)]}")

    # PAIRED per-round ratios. The arms are interleaved back-to-back inside one
    # round, so a round that ran under foreign load inflates BOTH arms and the
    # ratio survives it; a median of two independent per-arm distributions does
    # not. On a box shared with other agents this is the only statistic worth
    # quoting, and the spread of the per-round ratios is the honest error bar.
    print()
    print("paired per-round ratios (arm / baseline, same round)")
    hdr = f"{'vec':<16}{'t':>3}  {'arm':<12}{'n':>3}{'med':>9}{'min':>9}{'max':>9}"
    print(hdr)
    print("-" * len(hdr))
    per_round = defaultdict(dict)   # (vec, t, round) -> {arm: ms}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 7 or not f[4]:
            continue
        per_round[(f[2], int(f[3]), int(f[0]))][f[1]] = float(f[4])
    for vec, t in cells:
        for arm in arms:
            if arm == base_arm:
                continue
            rs = []
            for (v, tt, _r), d in per_round.items():
                if (v, tt) == (vec, t) and arm in d and base_arm in d and d[base_arm]:
                    rs.append(d[arm] / d[base_arm])
            if not rs:
                continue
            print(
                f"{vec:<16}{t:>3}  {arm:<12}{len(rs):>3}{median(rs):>9.4f}"
                f"{min(rs):>9.4f}{max(rs):>9.4f}"
            )

    # Cross-arm md5 identity, by value not by count.
    print()
    allh = defaultdict(set)
    for (vec, t, arm), hs in md5s.items():
        allh[(vec, t)] |= hs
    for k in sorted(allh, key=lambda k: (k[0], k[1])):
        n = len(allh[k])
        print(f"md5 {k[0]} t={k[1]}: {'BIT-IDENTICAL' if n == 1 else f'DIVERGED ({n})'}")


if __name__ == "__main__":
    main()
