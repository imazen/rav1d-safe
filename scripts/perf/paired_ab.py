#!/usr/bin/env python3
"""Paired base-vs-head verdict from a verify_gap.sh TSV.

`verify_gap_report.py` prints each arm's median beta. That is the right shape
for an ours-vs-dav1d gap table and the WRONG one for an A/B claim: two medians
can differ by 1% while every individual round disagrees about the sign.

This pairs the two arms WITHIN each round — the harness runs them back to back
with a rotating order inside one cell, so a round is the unit that saw the same
load — and reports:

  * the median of the per-round ratios (not the ratio of the medians),
  * the min/max band of those ratios,
  * how many rounds the head arm won,
  * whether the two arms' raw beta bands are DISJOINT, which is the only
    honest way to call a sub-3% difference real. Note this compares the arms
    the CLAIM compares; printing ours-vs-dav1d disjointness would be a tick
    that can never fail.
  * a two-sided sign test p-value, exact, so "8 of 9 faster" is not read as
    significance on its own.

Usage: paired_ab.py <tsv> <base_arm> <head_arm>
"""
import sys
from collections import defaultdict
from math import comb
from statistics import median

path, base_arm, head_arm = sys.argv[1], sys.argv[2], sys.argv[3]

beta = defaultdict(dict)  # (vec, t) -> {round: {arm: beta}}
foreign = defaultdict(int)
for line in open(path):
    f = line.rstrip("\n").split("\t")
    if len(f) < 8:
        continue
    rnd, arm, vec, t, nlo, lo, nhi, hi = f[:8]
    fmax = int(f[8]) if len(f) > 8 else 0
    b = (int(hi) - int(lo)) / (int(nhi) - int(nlo))
    beta[(vec, int(t))].setdefault(int(rnd), {})[arm] = b
    foreign[(vec, int(t))] = max(foreign[(vec, int(t))], fmax)


def sign_p(wins, n):
    """Two-sided exact sign test; ties already excluded from n."""
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    tail = sum(comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


hdr = f"{'vector':<16} {'t':>2} {'ratio':>7} {'band':>17} {'faster':>7} {'p':>6} {'disjoint':>9} {'n':>3} {'fgn':>4}"
print(f"paired {head_arm} / {base_arm}  (per-round ratios; <1.000 means {head_arm} is faster)")
print(hdr)
print("-" * len(hdr))
for key in sorted(beta):
    vec, t = key
    rounds = beta[key]
    rs, bb, hh = [], [], []
    for r in sorted(rounds):
        d = rounds[r]
        if base_arm in d and head_arm in d and d[base_arm] > 0:
            rs.append(d[head_arm] / d[base_arm])
            bb.append(d[base_arm])
            hh.append(d[head_arm])
    if not rs:
        continue
    wins = sum(1 for x in rs if x < 1.0)
    ties = sum(1 for x in rs if x == 1.0)
    n_eff = len(rs) - ties
    disjoint = "YES" if (max(hh) < min(bb) or min(hh) > max(bb)) else "no"
    print(
        f"{vec:<16} {t:>2} {median(rs):>7.4f} [{min(rs):>6.4f}..{max(rs):>6.4f}]"
        f" {wins:>3}/{len(rs):<3} {sign_p(wins, n_eff):>6.3f} {disjoint:>9} {len(rs):>3} {foreign[key]:>4}"
    )
print()
print("raw beta bands (ms/frame), for the disjointness column")
print(f"{'vector':<16} {'t':>2} {base_arm + ' band':>26} {head_arm + ' band':>26}")
for key in sorted(beta):
    vec, t = key
    bb = [d[base_arm] for d in beta[key].values() if base_arm in d]
    hh = [d[head_arm] for d in beta[key].values() if head_arm in d]
    if not bb or not hh:
        continue
    print(
        f"{vec:<16} {t:>2} [{min(bb):>10.2f}..{max(bb):>10.2f}] [{min(hh):>10.2f}..{max(hh):>10.2f}]"
    )
