#!/usr/bin/env python3
"""Rank what 10bpc costs that 8bpc does not, in ms/frame.

Inputs are two `sample_selftime.py` tables (8bpc and 10bpc) plus the measured
ms/frame for each arm. Shares are converted to ms/frame BEFORE differencing,
because 10bpc's frame is longer: a symbol can shrink in percentage terms while
growing in milliseconds, and milliseconds are what the gap is denominated in.

Symbols are folded to a FAMILY (module + function, generic parameters and the
`::h<hash>` suffix removed) so that the 8bpc and 10bpc monomorphisations of the
same kernel land on the same row. The raw per-symbol tables are printed too, so
a bad fold cannot hide inside a total.

Usage: depth_delta.py <8bpc.selftime> <ms_per_frame_8> <10bpc.selftime> <ms_per_frame_10> [topn]
"""

import re
import sys
from collections import Counter

HASH = re.compile(r"::h[0-9a-f]{16}\b")
GEN = re.compile(r"<[^<>]*>")
BD = re.compile(r"\b(BitDepth8|BitDepth16|bitdepth_8|bitdepth_16|bpc8|bpc16|u8|u16)\b")


def fold(sym):
    s = HASH.sub("", sym)
    prev = None
    while prev != s:  # collapse nested generics
        prev = s
        s = GEN.sub("", s)
    s = BD.sub("BD", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^rav1d_safe::", "", s)
    return s


def load(path):
    tot = 0.0
    per = Counter()
    raw = Counter()
    for line in open(path, errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 2:
            continue
        try:
            n = float(f[0].rstrip("%"))
        except ValueError:
            continue
        sym = f[-1].strip()
        if not sym:
            continue
        per[fold(sym)] += n
        raw[sym] += n
        tot += n
    return per, raw, tot


def main():
    p8, ms8, p10, ms10 = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4])
    topn = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    a, araw, atot = load(p8)
    b, braw, btot = load(p10)
    print(f"8bpc  leaves={atot:.0f}  ms/frame={ms8:.3f}")
    print(f"10bpc leaves={btot:.0f}  ms/frame={ms10:.3f}")
    print(f"delta ms/frame = {ms10-ms8:+.3f}   ({100*(ms10/ms8-1):+.1f}%)\n")

    rows = []
    for k in set(a) | set(b):
        m8 = a.get(k, 0.0) / atot * ms8
        m10 = b.get(k, 0.0) / btot * ms10
        rows.append((m10 - m8, m8, m10, k))
    rows.sort(reverse=True)
    acc = sum(r[0] for r in rows)
    print(f"{'d_ms':>8} {'8bpc_ms':>8} {'10b_ms':>8}  family   "
          f"(sum of all deltas = {acc:+.3f} ms, must equal the frame delta above)")
    for d, m8, m10, k in rows[:topn]:
        print(f"{d:+8.3f} {m8:8.3f} {m10:8.3f}  {k[:110]}")
    print("  ...")
    for d, m8, m10, k in rows[-8:]:
        print(f"{d:+8.3f} {m8:8.3f} {m10:8.3f}  {k[:110]}")

    only10 = sorted(((b[k] / btot * ms10, k) for k in b if k not in a), reverse=True)[:12]
    only8 = sorted(((a[k] / atot * ms8, k) for k in a if k not in b), reverse=True)[:12]
    print("\n-- families present ONLY at 10bpc (ms/frame) --")
    for v, k in only10:
        print(f"{v:8.3f}  {k[:110]}")
    print("\n-- families present ONLY at 8bpc (ms/frame) --")
    for v, k in only8:
        print(f"{v:8.3f}  {k[:110]}")


if __name__ == "__main__":
    main()
