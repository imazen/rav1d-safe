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
    print("RATIO BANDS (worst case over both arms' [min..max])")
    print(f"{'cell':<16}{'t':>2}  {'head/base':<22}{'HEAD/dav1d':<22}{'base/dav1d':<22}"
          f"{'verdict (head/base)':<24}")
    for vec, t in order:
        if not (c.has(base, vec, t) and c.has(head, vec, t)):
            continue
        hb = ratio_band(c, head, base, vec, t, key)
        hd = ratio_band(c, head, "dav1d_fd1", vec, t, key) if c.has("dav1d_fd1", vec, t) else (float("nan"),) * 2
        bd = ratio_band(c, base, "dav1d_fd1", vec, t, key) if c.has("dav1d_fd1", vec, t) else (float("nan"),) * 2
        if hb[1] < 1.0:
            v = "WIN, band-safe"
        elif hb[0] > 1.0:
            v = "REGRESSION, band-safe"
        else:
            v = "null (bands span 1.0)"
        print(f"{SHORT.get(vec, vec):<16}{t:>2}  "
              f"[{hb[0]:.4f}..{hb[1]:.4f}]{'':<6}[{hd[0]:.3f}..{hd[1]:.3f}]{'':<8}"
              f"[{bd[0]:.3f}..{bd[1]:.3f}]{'':<8}{v:<24}")


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
            print(f"{'arm':<12}{'ms/f':>9}{'[min..max]':>20}{'/base':>9}{'dj':>4}   note")
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
                print(f"{a:<12}{m:>9.3f}{f'[{lo:.3f}..{hi:.3f}]':>20}{m / b:>9.4f}{dj:>4}   "
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
