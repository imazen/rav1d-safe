#!/usr/bin/env python3
"""Paired A/B verdict for the LEFT-context read split.

A ratio of medians is not a verdict when the effect is ~1-3% and this box's
own null-arm floor is ~1.2% (docs/SHARD_SIZE_SWEEP.md §5). So this reports
three things per cell and lets them disagree in public:

  1. ratio of medians, with each arm's [min..max] band and whether the bands
     are disjoint (the campaign's usual `*`);
  2. the PAIRED per-round ratio — head and base ran inside the same round, in
     rotating order, under the same load, so their per-round quotient cancels
     drift that the pooled medians do not — its median and its sign count;
  3. the two-sided sign-test p for that count, so "sign-consistent across n
     rounds" is a number rather than an adjective.

Round 1 is DISCARDED by default: the first touch of each (arm, cell) pays cold
binary and file-cache cost, which lands almost entirely in `ms_lo` and biases
the two-point fit downward by ~2x (AGENT_BRIEF / census trap 2).

Usage: ctxread_report.py <wall.tsv> [--keep-round-1] [--label ms|cpu]
"""

import sys
from collections import defaultdict
from math import comb
from statistics import median


def two_sided_sign_p(k: int, n: int) -> float:
    """Exact binomial, p=0.5, two-sided: P(|X - n/2| >= |k - n/2|)."""
    if n == 0:
        return 1.0
    d = abs(k - n / 2)
    tot = 0
    for i in range(n + 1):
        if abs(i - n / 2) >= d - 1e-9:
            tot += comb(n, i)
    return min(1.0, tot / 2**n)


def main() -> int:
    args = [a for a in sys.argv[1:]]
    keep1 = "--keep-round-1" in args
    args = [a for a in args if not a.startswith("--")]
    path = args[0]
    unit = "cpu" if "cpu" in path else "ms"

    # (vec, threads, arm) -> {round: per_frame}
    beta = defaultdict(dict)
    foreign = defaultdict(int)
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        lo_i = head.index("ms_lo") if "ms_lo" in head else head.index("cpu_lo")
        hi_i = head.index("ms_hi") if "ms_hi" in head else head.index("cpu_hi")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            r = int(c[0])
            if r == 1 and not keep1:
                continue
            nlo, nhi = int(c[4]), int(c[6])
            b = (int(c[hi_i]) - int(c[lo_i])) / (nhi - nlo)
            key = (c[2], int(c[3]), c[1])
            beta[key][r] = b
            foreign[(c[2], int(c[3]))] = max(foreign[(c[2], int(c[3]))], int(c[8]))

    cells = sorted({(v, t) for (v, t, _) in beta}, key=lambda x: (x[0], -x[1]))
    arms = sorted({a for (_, _, a) in beta})
    print(f"# unit = {unit}/frame, round 1 {'KEPT' if keep1 else 'discarded (cold)'}")
    print(
        f"{'cell':<12}{'t':>2} {'n':>3} {'base':>9} {'head':>9} {'ratio':>7} "
        f"{'disj':>5} {'paired':>7} {'h<b':>6} {'p':>7} {'b/dav':>7} {'h/dav':>7} {'fmax':>5}"
    )
    for vec, t in cells:
        got = {a: beta.get((vec, t, a), {}) for a in arms}
        if not got.get("base") or not got.get("head"):
            continue
        rounds = sorted(set(got["base"]) & set(got["head"]))
        b = [got["base"][r] for r in rounds]
        h = [got["head"][r] for r in rounds]
        pr = [got["head"][r] / got["base"][r] for r in rounds]
        wins = sum(1 for x in pr if x < 1.0)
        # A band of one point is not a band; below n=3 the answer is "n/a", not
        # a free "yes".
        if len(rounds) < 3:
            disj = "n/a"
        else:
            disj = "yes" if (max(h) < min(b) or min(h) > max(b)) else "no"
        dav = got.get("dav1d_fd1", {})
        dm = median(dav.values()) if dav else float("nan")
        print(
            f"{vec:<12}{t:>2} {len(rounds):>3} {median(b):>9.3f} {median(h):>9.3f} "
            f"{median(h) / median(b):>7.4f} {disj:>5} {median(pr):>7.4f} "
            f"{wins:>3}/{len(pr):<2} {two_sided_sign_p(wins, len(pr)):>7.3f} "
            f"{median(b) / dm:>7.3f} {median(h) / dm:>7.3f} {foreign[(vec, t)]:>5}"
        )
        print(
            f"{'':<15}bands base [{min(b):.3f}..{max(b):.3f}]  "
            f"head [{min(h):.3f}..{max(h):.3f}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
