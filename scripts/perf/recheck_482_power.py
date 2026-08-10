#!/usr/bin/env python3
"""How many rounds does the disputed cell actually need?

Two questions the verdict rests on, both answered from the measured rounds
rather than from an assumption about the noise:

  1. Is 1.0854 -- the size sweep's median -- inside the sampling distribution
     of this cell's median at n=20? Percentile bootstrap CI of the median.

  2. Could a FIVE-round window have landed there? Exhaustively (or by
     resampling) draw 5 of the measured rounds without replacement, take the
     median exactly as the size sweep did, and report how often it reaches
     1.04 and 1.0854.

If (2) says a 5-round median reaches the claimed level a non-trivial fraction
of the time, then the disputed number is a sample size result, not a code
result, and no mechanism needs to be invented to explain it.

Usage: recheck_482_power.py <gap.tsv> <pair> <vector> <threads> [inst]
"""

import itertools
import random
import sys
from collections import defaultdict


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    path = sys.argv[1]
    num, den = sys.argv[2].split("/")
    vec, thr = sys.argv[3], sys.argv[4]
    inst = sys.argv[5] if len(sys.argv) > 5 else "int"

    beta = defaultdict(dict)
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 12 or f[2] != vec or f[3] != thr:
            continue
        rnd, arm = int(f[0]), f[1]
        nlo, elo, nhi, ehi = int(f[4]), float(f[5]), int(f[6]), float(f[7])
        v = (ehi - elo) / (nhi - nlo) if inst == "ext" else float(f[9])
        beta[arm][rnd] = v

    rounds = sorted(set(beta[num]) & set(beta[den]))
    rs = [beta[num][r] / beta[den][r] for r in rounds]
    n = len(rs)
    print(f"# {vec} t={thr}   {num}/{den}   instrument={inst}   n={n}")
    print(f"# per-round ratios: " + " ".join(f"{v:.4f}" for v in rs))
    print()

    med = median(rs)
    rng = random.Random(20260810)
    boot = sorted(median([rs[rng.randrange(n)] for _ in range(n)])
                  for _ in range(20000))
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    print(f"median {med:.4f}   95% bootstrap CI [{lo:.4f}..{hi:.4f}]   "
          f"(20,000 resamples, seed 20260810)")
    for claim, label in ((1.0854, "the size sweep's median"),
                         (1.0, "parity")):
        inside = lo <= claim <= hi
        print(f"   {claim:.4f} ({label}): "
              f"{'INSIDE' if inside else 'OUTSIDE'} the CI")
    print()

    # --- what a 5-round window can do ---------------------------------------
    k = 5
    combos = list(itertools.combinations(range(n), k))
    if len(combos) > 200000:
        combos = [tuple(rng.sample(range(n), k)) for _ in range(200000)]
        how = f"200,000 random draws of {k} from {n}"
    else:
        how = f"all {len(combos):,} ways to draw {k} of {n}"
    meds = [median([rs[i] for i in c]) for c in combos]
    meds.sort()
    print(f"## the median of a {k}-round window ({how})")
    print()
    print(f"   range [{meds[0]:.4f}..{meds[-1]:.4f}]   "
          f"p5 {meds[int(0.05*len(meds))]:.4f}   "
          f"p50 {meds[len(meds)//2]:.4f}   "
          f"p95 {meds[int(0.95*len(meds))]:.4f}")
    for thresh in (1.0, 1.04, 1.0854):
        c = sum(1 for m in meds if m >= thresh)
        print(f"   P(5-round median >= {thresh:.4f}) = "
              f"{c / len(meds) * 100:6.2f}%   ({c:,} of {len(meds):,})")


if __name__ == "__main__":
    main()
