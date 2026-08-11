#!/usr/bin/env python3
"""Reduce `tiled_wallcpu.sh` rows for the "derived rows-per-block rule as the
DEFAULT" round.

Two-point fit per row (`beta = (total_hi - total_lo) / (n_hi - n_lo)`, wall and
user+sys separately) so process startup cancels, then the median across rounds
with the min/max band.

What this prints that the generic reporter does not:

* **ours/dav1d for the DEFAULT build AND for the base arm, side by side.** The
  whole reason for the round is that the campaign has been quoting a gated arm's
  ratio as if it were the shipped decoder's.
* **Ratio bands**, worst case over both arms' [min..max], so a verdict says
  whether it is band-safe rather than just quoting a point estimate.
* **Disjointness against the arm the claim is about** — head vs base, not
  ours vs dav1d, which is two different decoders and can never fail.

`--drop-round N` removes a round (default 0: first touch of each (arm, cell) is
cold). `--factorial` reduces the per-plane shift grid instead, including the
additivity residual that says whether the two planes interact.
"""

import argparse
import collections
import statistics
import sys


def load(paths, drop_round):
    rows = []
    for p in paths:
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if len(f) < 13:
                continue
            rnd, arm, vec, t, nlo, wlo, ulo, slo, nhi, whi, uhi, shi, fo = f[:13]
            if int(rnd) == drop_round:
                continue
            nlo, nhi = int(nlo), int(nhi)
            wlo, ulo, slo, whi, uhi, shi = (int(x) for x in (wlo, ulo, slo, whi, uhi, shi))
            dn = nhi - nlo
            rows.append({
                "round": int(rnd), "arm": arm, "vec": vec, "t": int(t),
                "wall": (whi - wlo) / dn,
                "cpu": ((uhi + shi) - (ulo + slo)) / dn,
                "foreign": int(fo),
            })
    return rows


class Cells:
    def __init__(self, rows):
        self.by = collections.defaultdict(list)
        for r in rows:
            self.by[(r["arm"], r["vec"], r["t"])].append(r)
        self.rows = rows

    def has(self, arm, vec, t):
        return (arm, vec, t) in self.by

    def med(self, arm, vec, t, key="wall"):
        return statistics.median([r[key] for r in self.by[(arm, vec, t)]])

    def band(self, arm, vec, t, key="wall"):
        v = [r[key] for r in self.by[(arm, vec, t)]]
        return min(v), max(v)

    def n(self, arm, vec, t):
        return len(self.by[(arm, vec, t)])


def ratio_band(c, a, b, vec, t, key="wall"):
    """Worst-case [lo..hi] for a/b over both arms' own bands."""
    alo, ahi = c.band(a, vec, t, key)
    blo, bhi = c.band(b, vec, t, key)
    return alo / bhi, ahi / blo


def paired(c, a, b, vec, t, key="wall"):
    """Per-ROUND ratio a/b, which is what the interleave is for.

    The arms rotate within a round, so a drift shared by every arm — and there
    is one: v4k8tile reads 47.5 / 50.6 / 48.9 ms/frame across rounds for ALL
    FIVE arms including dav1d — cancels in the paired ratio and does not cancel
    in a ratio of medians. Returns (median, min, max, n_below_1, n).
    """
    ra = {r["round"]: r[key] for r in c.by[(a, vec, t)]}
    rb = {r["round"]: r[key] for r in c.by[(b, vec, t)]}
    rs = [ra[k] / rb[k] for k in sorted(set(ra) & set(rb)) if rb[k]]
    if not rs:
        return float("nan"), float("nan"), float("nan"), 0, 0
    return statistics.median(rs), min(rs), max(rs), sum(1 for x in rs if x < 1.0), len(rs)


def disjoint(c, a, b, vec, t, key="wall"):
    alo, ahi = c.band(a, vec, t, key)
    blo, bhi = c.band(b, vec, t, key)
    return ahi < blo or bhi < alo


SHORT = {
    "C1024x192_420_8b__t8": "c1024x192",
    "C1024x288_420_8b__t8": "c1024x288",
    "C1024x384_420_8b__t8": "c1024x384",
    "C1024x576_420_8b__t8": "c1024x576",
    "C256x2048_420_8b__t8": "c256x2048",
    "C512x288_420_8b__t8": "c512x288",
    "C512x576_420_8b__t8": "c512x576",
    "C3840x256_420_8b__t8": "c3840x256",
    "C3840x2160_420_8b__t8": "c3840x2160",
    "L1024x576_420_10b__t8": "c1024x576_10b",
    "v4k_8tile": "v4k8tile",
}


def main_table(c, base, head, key):
    order = []
    for r in c.rows:
        k = (r["vec"], r["t"])
        if k not in order:
            order.append(k)
    unit = "wall" if key == "wall" else "CPU"
    print("=" * 132)
    print(f"{unit} ms/frame, two-point fit, median of n rounds. "
          f"head = DEFAULT build (derived rows rule), base = --features bps-blocks")
    print("=" * 132)
    hdr = (f"{'cell':<16}{'t':>2} {'n':>2} {'base':>9} {'head':>9} "
           f"{'head/base':>10} {'dj':>3} {'base/dav1d':>11} {'HEAD/dav1d':>11} "
           f"{'half/base':>10} {'untrk/base':>11}")
    print(hdr)
    for vec, t in order:
        if not (c.has(base, vec, t) and c.has(head, vec, t)):
            continue
        b = c.med(base, vec, t, key)
        h = c.med(head, vec, t, key)
        dv = c.med("dav1d_fd1", vec, t, key) if c.has("dav1d_fd1", vec, t) else float("nan")
        hf = c.med("bpshalf", vec, t, key) if c.has("bpshalf", vec, t) else float("nan")
        un = c.med("untracked", vec, t, key) if c.has("untracked", vec, t) else float("nan")
        dj = "*" if disjoint(c, head, base, vec, t, key) else ""
        print(f"{SHORT.get(vec, vec):<16}{t:>2} {c.n(head, vec, t):>2} "
              f"{b:>9.3f} {h:>9.3f} {h / b:>10.4f} {dj:>3} "
              f"{b / dv:>11.3f} {h / dv:>11.3f} {hf / b:>10.4f} {un / b:>11.4f}")
    print()
    print("PAIRED per-round head/base — the statistic the interleave is for. "
          "A drift shared by all arms cancels here and does not cancel in a "
          "ratio of medians.")
    print(f"{'cell':<16}{'t':>2} {'median':>8} {'[min..max]':>18} {'<1':>6}  "
          f"{'unpaired band':<20}{'verdict':<26}")
    for vec, t in order:
        if not (c.has(base, vec, t) and c.has(head, vec, t)):
            continue
        m, lo, hi, nb, n = paired(c, head, base, vec, t, key)
        ub = ratio_band(c, head, base, vec, t, key)
        # Mechanical, deliberately: "all below 1.0" is a statement about the
        # paired distribution, not a significance claim. Whether a 0.06% offset
        # matters is a question about the NOISE FLOOR, and the floor is read off
        # the identity controls (every cell at t=1; c256x2048 and v4k8tile at
        # t=8, where the rule provably returns the same shift). The doc names it;
        # a script that printed "REGRESSION" for +0.06% on provably identical
        # code would be lying with a straight face.
        if n < 3:
            v = f"n={n}, no verdict"
        elif hi < 1.0:
            v = f"all {n} below 1.0"
        elif lo > 1.0:
            v = f"all {n} above 1.0"
        else:
            v = f"spans 1.0 ({nb}/{n} below)"
        if abs(m - 1.0) < 0.01:
            v += "  |d|<1%"
        print(f"{SHORT.get(vec, vec):<16}{t:>2} {m:>8.4f} {f'[{lo:.4f}..{hi:.4f}]':>18} "
              f"{f'{nb}/{n}':>6}  [{ub[0]:.3f}..{ub[1]:.3f}]{'':<7}{v:<26}")
    print()
    print("PAIRED per-round ours/dav1d — the number the campaign table must quote")
    print(f"{'cell':<16}{'t':>2} {'HEAD/dav1d':>11} {'[min..max]':>18} "
          f"{'base/dav1d':>11} {'[min..max]':>18}")
    for vec, t in order:
        if not (c.has("dav1d_fd1", vec, t) and c.has(head, vec, t)):
            continue
        hm, hlo, hhi, _, _ = paired(c, head, "dav1d_fd1", vec, t, key)
        bm, blo, bhi, _, _ = paired(c, base, "dav1d_fd1", vec, t, key)
        print(f"{SHORT.get(vec, vec):<16}{t:>2} {hm:>11.3f} {f'[{hlo:.3f}..{hhi:.3f}]':>18} "
              f"{bm:>11.3f} {f'[{blo:.3f}..{bhi:.3f}]':>18}")


def factorial(c, key):
    vecs = sorted({r["vec"] for r in c.rows})
    for vec in vecs:
        ts = sorted({r["t"] for r in c.rows if r["vec"] == vec})
        for t in ts:
            arms = sorted({r["arm"] for r in c.rows if r["vec"] == vec and r["t"] == t})
            pins = [a for a in arms if a.startswith("pinL")]
            if not pins:
                continue
            base = "pinL10C8"
            print("=" * 100)
            print(f"PER-PLANE SHIFT FACTORIAL — {SHORT.get(vec, vec)} t={t}, {key} ms/frame, "
                  f"n={c.n(base, vec, t)}; base = {base} (= the block-count rule)")
            print("=" * 100)
            b = c.med(base, vec, t, key)
            print(f"{'arm':<12}{'ms/f':>9}{'[min..max]':>20}{'/base':>9}{'dj':>4}"
                  f"{'paired':>9}{'[min..max]':>18}{'<1':>6}   note")
            notes = {
                "pinL10C8": "block-count rule (bps-blocks)",
                "pinL11C9": "bps-1",
                "pinL12C10": "bps-half",
                "pinL11C10": "THE DERIVED RULE (default)",
                "pinL12C9": "the corner no arm can reach",
                "plain": "unpinned DEFAULT build (cross-check vs pinL11C10)",
            }
            for a in ["plain"] + sorted(pins):
                if not c.has(a, vec, t):
                    continue
                m = c.med(a, vec, t, key)
                lo, hi = c.band(a, vec, t, key)
                dj = "*" if disjoint(c, a, base, vec, t, key) else ""
                pm, plo, phi, nb, n = paired(c, a, base, vec, t, key)
                print(f"{a:<12}{m:>9.3f}{f'[{lo:.3f}..{hi:.3f}]':>20}{m / b:>9.4f}{dj:>4}"
                      f"{pm:>9.4f}{f'[{plo:.4f}..{phi:.4f}]':>18}{f'{nb}/{n}':>6}   "
                      f"{notes.get(a, '')}")
            print()
            # Additivity: is the (L, C) grid separable?
            print("ADDITIVITY — if the planes do not interact, "
                  "r(L,C) = r(L,8) * r(10,C). Residual = measured / predicted.")
            print(f"{'cell':<10}{'measured':>10}{'predicted':>11}{'residual':>10}")
            for L in (11, 12):
                for C in (9, 10):
                    a = f"pinL{L}C{C}"
                    la, ca = f"pinL{L}C8", f"pinL10C{C}"
                    if not all(c.has(x, vec, t) for x in (a, la, ca)):
                        continue
                    m = c.med(a, vec, t, key) / b
                    p = (c.med(la, vec, t, key) / b) * (c.med(ca, vec, t, key) / b)
                    print(f"L{L}C{C:<7}{m:>10.4f}{p:>11.4f}{m / p:>10.4f}")
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+")
    ap.add_argument("--drop-round", type=int, default=0)
    ap.add_argument("--base", default="bpsblocks")
    ap.add_argument("--head", default="plain")
    ap.add_argument("--factorial", action="store_true")
    a = ap.parse_args()

    rows = load(a.tsv, a.drop_round)
    if not rows:
        sys.exit("no rows")
    c = Cells(rows)
    fmax = max(r["foreign"] for r in rows)
    print(f"rows={len(rows)}  cells={len(c.by)}  rounds={sorted({r['round'] for r in rows})}  "
          f"foreign_max={fmax}  (round {a.drop_round} dropped: cold first touch)")
    if fmax > 0:
        print("!! foreign load observed on some rows — report PAIRED RATIOS, tag absolutes")
    print()
    for key in ("wall", "cpu"):
        if a.factorial:
            factorial(c, key)
        else:
            main_table(c, a.base, a.head, key)
        print()


if __name__ == "__main__":
    main()
