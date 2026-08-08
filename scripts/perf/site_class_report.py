#!/usr/bin/env python3
"""Per-call-site-class tracker attribution from a site_class_sweep.sh TSV.

Prints, per cell:

  1. every arm's ms/frame as median [min..max] over rounds — bands, not bare
     medians, because a 2-3% claim whose arms' ranges overlap is not a result;
  2. per-class attributable ms = median(cls_none) - median(cls_<class>), with a
     disjoint/OVERLAP verdict on the two bands;
  3. an ADDITIVITY check: the per-class deltas must sum to the all-nulled delta.
     They are measured independently, so a large residual means the classes
     interact (shard contention, I-cache) and the split is not a decomposition;
  4. what the gap to dav1d would be if that class's tracker cost went to zero,
     anchored on `base` (the real shipping build), not on the instrumented one.

Usage: site_class_report.py <tsv>
"""
import os
import sys
from collections import defaultdict
from statistics import median

path = sys.argv[1]
only_idle = os.environ.get("ONLY_IDLE") == "1"
CLASSES = ["recon", "recon+big", "filter", "decode", "other", "picwb"]
# `recon+big` is a SUBSET of `recon`, not a peer, so it is excluded from the
# additivity sum and from the disjoint partition of the total.
PARTITION = ["recon", "filter", "decode", "other", "picwb"]

betas = defaultdict(list)
loaded = defaultdict(int)
for line in open(path):
    f = line.rstrip("\n").split("\t")
    if len(f) < 8:
        continue
    _rnd, arm, vec, t, nlo, lo, nhi, hi = f[:8]
    fmax = int(f[8]) if len(f) > 8 else 0
    if fmax:
        loaded[(vec, int(t))] += 1
        if only_idle:
            continue
    betas[(vec, int(t), arm)].append((int(hi) - int(lo)) / (int(nhi) - int(nlo)))

cells = sorted({(k[0], k[1]) for k in betas})
arms = sorted({k[2] for k in betas})


def med(vec, t, arm):
    v = betas.get((vec, t, arm))
    return median(v) if v else None


def band(vec, t, arm):
    v = betas.get((vec, t, arm))
    return (min(v), max(v)) if v else None


print("== ms/frame, median [min..max] over rounds (two-point wall fit, alpha removed) ==\n")
order = [a for a in ("base", "cls_none", "addnop", "untracked", "dav1d_fd1") if a in arms]
order += [f"cls_{c}" for c in CLASSES if f"cls_{c}" in arms]
order += [a for a in ("cls_all",) if a in arms]
hdr = f"{'vector':16} {'t':>2} " + "".join(f"{a:>25}" for a in order) + f" {'n':>3} {'ld':>3}"
print(hdr)
print("-" * len(hdr))
for vec, t in cells:
    row = f"{vec:16} {t:>2} "
    n = 0
    for a in order:
        m, b = med(vec, t, a), band(vec, t, a)
        if m is None:
            row += f"{'-':>25}"
        else:
            row += f"{m:>9.1f} [{b[0]:6.1f}..{b[1]:6.1f}]"
            n = max(n, len(betas[(vec, t, a)]))
    print(row + f" {n:>3} {loaded.get((vec, t), 0):>3}")

print("\n== instrument weight and whole-tracker anchors (ms/frame) ==\n")
h = (
    f"{'vector':16} {'t':>2} {'base':>8} {'untracked':>10} {'tracker':>8} "
    f"{'addnop':>8} {'body':>7} {'call':>6} {'cls_none':>9} {'instr':>7} "
    f"{'cls_all':>8} {'exposed':>8} {'dav1d':>7} {'gap':>6}"
)
print(h)
print("-" * len(h))
for vec, t in cells:
    b, u, an, cn, ca, d = (
        med(vec, t, a)
        for a in ("base", "untracked", "addnop", "cls_none", "cls_all", "dav1d_fd1")
    )
    if None in (b, u, cn, ca, d):
        continue
    print(
        f"{vec:16} {t:>2} {b:>8.1f} {u:>10.1f} {b - u:>8.1f} "
        f"{an:>8.1f} {b - an:>7.1f} {an - u:>6.1f} {cn:>9.1f} {cn - b:>7.1f} "
        f"{ca:>8.1f} {ca - an:>8.1f} {d:>7.1f} {b / d:>6.3f}"
    )
print(
    "\ntracker = base-untracked (all of it) | body = base-addnop (work, call kept)\n"
    "call = addnop-untracked (the call barrier alone) | instr = cls_none-base\n"
    "exposed = cls_all-addnop: the instrument's per-borrow class lookup once there\n"
    "is no tracker latency left to hide it under, PLUS the cost of the tracker body\n"
    "still being present inline and branched over. It is why every per-class number\n"
    "below is a LOWER bound on deleting that class outright."
)

print("\n== per-class attributable ms/frame = cls_none - cls_<class> ==\n")
h2 = f"{'vector':16} {'t':>2} " + "".join(f"{c:>20}" for c in CLASSES) + f"{'SUM':>9}{'all':>9}{'resid':>8}"
print(h2)
print("-" * len(h2))
attrib = {}
overlap = {}
for vec, t in cells:
    cn = med(vec, t, "cls_none")
    cnb = band(vec, t, "cls_none")
    if cn is None:
        continue
    row = f"{vec:16} {t:>2} "
    s = 0.0
    for c in CLASSES:
        m, bb = med(vec, t, f"cls_{c}"), band(vec, t, f"cls_{c}")
        if m is None:
            row += f"{'-':>20}"
            continue
        d = cn - m
        if c in PARTITION:
            s += d
        dj = "d" if (bb[1] < cnb[0] or cnb[1] < bb[0]) else "O"
        # An overlapping band is a NULL for that class, not a small number. The
        # projection table below must not quietly spend it.
        attrib[(vec, t, c)] = d if dj == "d" else 0.0
        overlap[(vec, t, c)] = dj == "O"
        row += f"{d:>16.1f} {dj:>3}"
    ca = med(vec, t, "cls_all")
    tot = cn - ca if ca is not None else float("nan")
    row += f"{s:>9.1f}{tot:>9.1f}{s - tot:>8.1f}"
    print(row)
print("\nd = the class's band is disjoint from cls_none's; O = they overlap, and the")
print("class is then treated as a NULL (0.0) in the projection below, not as its")
print("median. SUM should equal `all`; resid is the interaction term.")

print("\n== gap to dav1d if one class's tracker cost went to zero ==\n")
print("projected = (base - attributable) / dav1d.  base and dav1d are the real")
print("builds; the attributable ms comes from the instrumented family.\n")
h3 = (
    f"{'vector':16} {'t':>2} {'now':>6} " + "".join(f"{c:>11}" for c in CLASSES)
    + f"{'A=rbig+picwb':>13}{'all':>8}{'untracked':>10}"
)
print(h3)
print("-" * len(h3))
for vec, t in cells:
    b, d, cn, ca, u = (
        med(vec, t, a) for a in ("base", "dav1d_fd1", "cls_none", "cls_all", "untracked")
    )
    if None in (b, d, cn):
        continue
    row = f"{vec:16} {t:>2} {b / d:>6.3f} "
    for c in CLASSES:
        a = attrib.get((vec, t, c))
        if a is None:
            row += f"{'-':>11}"
        else:
            mark = "*" if overlap.get((vec, t, c)) else " "
            row += f"{(b - a) / d:>10.3f}{mark}"
    rp = attrib.get((vec, t, "recon+big"), attrib.get((vec, t, "recon"), 0)) + attrib.get(
        (vec, t, "picwb"), 0
    )
    row += f"{(b - rp) / d:>13.3f}"
    row += f"{(b - (cn - ca)) / d:>8.3f}" if ca is not None else f"{'-':>8}"
    row += f"{u / d:>10.3f}"
    print(row)
print("\n* = that class's band overlapped cls_none's, so it is counted as zero and the")
print("cell just repeats `now`.")

# The pre-registered decision rule is stated as a fraction of the EXCESS over
# the 1.30x bar, not as an absolute ms, because 12 ms means something different
# at a cell that is 22 ms over the bar and one that is 58 ms over.
print("\n== share of the excess over the 1.30x bar that each class accounts for ==\n")
print("excess = base - 1.30 * dav1d.  A = recon+big (picture-buffer recon) + picwb,")
print("which is what a static per-tile row split of the picture buffer can reach.\n")
h4 = (
    f"{'vector':16} {'t':>2} {'excess':>7} " + "".join(f"{c:>11}" for c in CLASSES)
    + f"{'A':>8}{'tracker':>9}"
)
print(h4)
print("-" * len(h4))
for vec, t in cells:
    b, d, u = (med(vec, t, a) for a in ("base", "dav1d_fd1", "untracked"))
    if None in (b, d, u):
        continue
    ex = b - 1.30 * d
    row = f"{vec:16} {t:>2} {ex:>7.1f} "
    for c in CLASSES:
        a = attrib.get((vec, t, c))
        row += f"{'-':>11}" if a is None else f"{100 * a / ex:>10.0f}%"
    rp = attrib.get((vec, t, "recon+big"), attrib.get((vec, t, "recon"), 0)) + attrib.get(
        (vec, t, "picwb"), 0
    )
    row += f"{100 * rp / ex:>7.0f}%{100 * (b - u) / ex:>8.0f}%"
    print(row)
print("\ntracker = the whole tracker (base - untracked), i.e. the hard cap on any")
print("tracker-removal design at that cell. >100% means a zero-cost tracker would")
print("clear the 1.30x bar there.")
