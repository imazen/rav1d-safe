#!/usr/bin/env python3
"""Report `content_sweep.sh` rows: per-cell ms/frame, ratio to dav1d, and the
regression of the ratio on borrow registrations per pixel.

The two-point fit `total = a + b*frames` is applied per (round, arm, cell), so
`b` is ms/frame with process startup removed. Ratios are paired WITHIN a round
(both arms saw the same box state) and reduced by median, with min/max printed —
a median without a band is not a result on a loaded box.

Registration counts come from `examples/probe_tracker --features probe-sites`
(counts only; that build's wall clock is perturbed and is never used here).

Usage: content_report.py <sweep.tsv> <sites.txt>...
"""

import statistics
import sys
from collections import defaultdict

PX = {
    "64x36": 64 * 36, "256x144": 256 * 144, "512x288": 512 * 288,
    "1024x576": 1024 * 576, "2048x1152": 2048 * 1152, "3840x2160": 3840 * 2160,
}


def size_of(vec):
    for k in PX:
        if k in vec:
            return k
    raise SystemExit(f"no size in {vec}")


def load_sweep(path):
    """-> {(round, arm, vec): ms_per_frame}, {vec: max foreign}"""
    b = {}
    foreign = defaultdict(int)
    rounds = defaultdict(set)
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 10:
            continue
        rnd, arm, vec, _t, nlo, mlo, nhi, mhi, f_arm, f_grp = f[:10]
        nlo, mlo, nhi, mhi = int(nlo), int(mlo), int(nhi), int(mhi)
        if nhi == nlo:
            continue
        b[(int(rnd), arm, vec)] = (mhi - mlo) / (nhi - nlo)
        foreign[vec] = max(foreign[vec], int(f_arm), int(f_grp))
        rounds[vec].add(int(rnd))
    return b, foreign, rounds


def load_sites(paths):
    """-> {vec: registrations_per_frame}"""
    regs = {}
    for p in paths:
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[1] == "SITES":
                regs[f[0]] = int(f[2].split("=")[1])
    return regs


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    b, foreign, rounds = load_sweep(sys.argv[1])
    regs = load_sites(sys.argv[2:])

    vecs = sorted({v for (_r, _a, v) in b})
    arms = sorted({a for (_r, a, _v) in b})
    ours = [a for a in arms if not a.startswith("dav1d")][0]
    ref = [a for a in arms if a.startswith("dav1d")][0]

    print(f"arms: ours={ours} ref={ref}")
    print(f"{'cell':26s} {'n':>2s} {'px':>9s} {'ours ms/f':>10s} {'ref ms/f':>9s} "
          f"{'ratio':>7s} {'band':>17s} {'regs/f':>10s} {'regs/px':>8s} "
          f"{'ours ms/MP':>10s} {'fmax':>4s}")
    table = []
    for v in vecs:
        rs = sorted(r for r in rounds[v] if (r, ours, v) in b and (r, ref, v) in b)
        if not rs:
            continue
        ratios = [b[(r, ours, v)] / b[(r, ref, v)] for r in rs if b[(r, ref, v)] > 0]
        o = statistics.median([b[(r, ours, v)] for r in rs])
        d = statistics.median([b[(r, ref, v)] for r in rs])
        px = PX[size_of(v)]
        rg = regs.get(v)
        rpp = (rg / px) if rg else float("nan")
        row = dict(cell=v, n=len(ratios), px=px, ours=o, ref=d,
                   ratio=statistics.median(ratios), lo=min(ratios), hi=max(ratios),
                   regs=rg, rpp=rpp, mspmp=o / (px / 1e6), fmax=foreign[v])
        table.append(row)
        print(f"{v:26s} {row['n']:2d} {px:9,d} {o:10.4f} {d:9.4f} "
              f"{row['ratio']:7.4f} [{row['lo']:6.4f}..{row['hi']:6.4f}] "
              f"{(rg if rg else 0):10,d} {rpp:8.3f} {row['mspmp']:10.2f} {row['fmax']:4d}")

    # Which candidate explains the ratio? Pixel count is SIZE_SWEEP.md's axis;
    # regs/pixel is its named mechanism; ms/MP is total decode work per pixel.
    import math
    good = [r for r in table if r["regs"]]
    if len(good) >= 4:
        print()
        for xname, xf in (("regs/pixel", lambda r: r["rpp"]),
                          ("log10(pixels)", lambda r: math.log10(r["px"])),
                          ("log10(ours ms/MP)", lambda r: math.log10(r["mspmp"])),
                          ("log10(dav1d ms/MP)",
                           lambda r: math.log10(r["ref"] / (r["px"] / 1e6)))):
            xs = [xf(r) for r in good]
            ys = [r["ratio"] for r in good]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            syy = sum((y - my) ** 2 for y in ys)
            beta = sxy / sxx if sxx else 0.0
            r2 = (sxy * sxy / (sxx * syy)) if sxx and syy else 0.0
            print(f"ratio ~ {xname:18s}: slope {beta:+.4f}  intercept {my - beta * mx:+.4f}  "
                  f"R^2 {r2:.3f}  n={n}")

    # Per content class, at one size, across quality: does the ratio move
    # monotonically? A 4-of-4 replication is worth more than one regression.
    print()
    print("fixed size 1024x576, ratio vs quality, per content class")
    classes = {}
    for r in table:
        c = r["cell"]
        if "_1024x576_q" not in c:
            continue
        cls, q = c.split("_1024x576_q")
        classes.setdefault(cls, []).append((int(q), r))
    for cls in sorted(classes):
        rows = sorted(classes[cls])
        qs = " ".join(f"q{q}={r['ratio']:.3f}" for q, r in rows)
        mono = all(a[1]["ratio"] >= b[1]["ratio"] for a, b in zip(rows, rows[1:]))
        span = max(r["ratio"] for _q, r in rows) / min(r["ratio"] for _q, r in rows)
        print(f"  {cls:8s} {qs}   monotone_down={mono}  span={span:.2f}x")
    if classes:
        allr = [r["ratio"] for _q, rs in
                [(k, v) for k, v in classes.items()] for _q2, r in rs]
        print(f"  spread across content x quality at ONE size: "
              f"{min(allr):.3f} .. {max(allr):.3f} ({max(allr) / min(allr):.2f}x)")


if __name__ == "__main__":
    main()
