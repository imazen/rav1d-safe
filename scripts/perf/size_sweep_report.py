#!/usr/bin/env python3
"""Fit `ms_per_frame = alpha + beta*pixels` for both decoders and print the
intercept SEPARATELY from the slope.

Input: the TSV written by scripts/perf/size_sweep.sh
       round arm vec threads nlo ms_lo nhi ms_hi foreign_max

Per cell and round, ms/frame is the two-point wall fit (ms_hi - ms_lo) /
(nhi - nlo), which removes process startup. Ratios are PAIRED WITHIN A ROUND
(both arms saw the same box state) and then reduced by median, with the min/max
band printed so a sub-3% claim can be checked against its own noise, per
docs/AGENT_BRIEF.md.

Two fits are printed because they answer different questions:
  * OLS over ms/frame vs pixels -- the standard fit. Its residual is absolute
    ms, so the 4K point dominates and alpha is an extrapolation.
  * A relative-error fit (weights 1/y^2) -- every size contributes equally in
    percentage terms, which is what a fit spanning 3.5 decades of pixel count
    needs if the small end is to constrain anything.
The tiny cell's measured ms/frame is printed too: it is a direct UPPER BOUND on
alpha that needs no model at all.

Usage: size_sweep_report.py <gap.tsv> [--tsv out.tsv]
"""

import re
import sys
from collections import defaultdict

VEC_RE = re.compile(r"^L(\d+)x(\d+)_(\d+)_(\d+)b$")


def parse(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            rows.append(
                dict(
                    round=int(p[0]), arm=p[1], vec=p[2], threads=int(p[3]),
                    nlo=int(p[4]), ms_lo=float(p[5]), nhi=int(p[6]),
                    ms_hi=float(p[7]), foreign=int(p[8]),
                )
            )
    return rows


def geom(vec):
    m = VEC_RE.match(vec)
    if not m:
        return None
    w, h, fmt, depth = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
    luma = w * h
    chroma = {"420": luma // 2, "422": luma, "444": 2 * luma}[fmt]
    return dict(w=w, h=h, fmt=fmt, depth=depth, luma=luma, samples=luma + chroma)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def ols(xs, ys, weights=None):
    """Least squares y = a + b*x. Returns (a, b, r2)."""
    if weights is None:
        weights = [1.0] * len(xs)
    sw = sum(weights)
    mx = sum(w * x for w, x in zip(weights, xs)) / sw
    my = sum(w * y for w, y in zip(weights, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(weights, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(weights, xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_res = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(weights, xs, ys))
    ss_tot = sum(w * (y - my) ** 2 for w, y in zip(weights, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a, b, r2


def main():
    path = sys.argv[1]
    out_tsv = None
    if "--tsv" in sys.argv:
        out_tsv = sys.argv[sys.argv.index("--tsv") + 1]
    rows = parse(path)
    if not rows:
        sys.exit("no rows")

    loaded = sum(1 for r in rows if r["foreign"] > 0)
    print(f"rows={len(rows)}  rows_under_foreign_load={loaded}")

    # per (arm, vec, round) -> ms/frame
    per = defaultdict(list)
    for r in rows:
        mspf = (r["ms_hi"] - r["ms_lo"]) / (r["nhi"] - r["nlo"])
        per[(r["arm"], r["vec"], r["round"])].append(mspf)
    cell = {k: median(v) for k, v in per.items()}

    arms = sorted({a for (a, _, _) in cell})
    allv = {v for (_, v, _) in cell}
    # Vectors whose name is not a ladder cell (the campaign's own v4k_8tile
    # anchors) get a plain table instead of a fit -- a size fit over one size
    # would be a fabrication.
    other_v = sorted(v for v in allv if geom(v) is None)
    vecs = sorted((v for v in allv if geom(v)), key=lambda v: (geom(v)["fmt"], geom(v)["depth"], geom(v)["luma"]))
    rounds = sorted({rd for (_, _, rd) in cell})
    threads = sorted({r["threads"] for r in rows})
    ours = "rs"
    ref = "dav1d_fd1" if "dav1d_fd1" in arms else [a for a in arms if a != ours][0]
    if other_v:
        print(f"\n=== non-ladder vectors (t={threads}) ===")
        for v in other_v:
            o = [cell[(ours, v, rd)] for rd in rounds if (ours, v, rd) in cell]
            d = [cell[(ref, v, rd)] for rd in rounds if (ref, v, rd) in cell]
            pr = [cell[(ours, v, rd)] / cell[(ref, v, rd)] for rd in rounds
                  if (ours, v, rd) in cell and (ref, v, rd) in cell]
            if not pr:
                continue
            print(f"  {v:<20} ours {median(o):9.3f} [{min(o):.3f}..{max(o):.3f}]  "
                  f"dav1d {median(d):9.3f} [{min(d):.3f}..{max(d):.3f}]  "
                  f"ratio {median(pr):.3f} [{min(pr):.3f}..{max(pr):.3f}]")

    out_lines = ["fmt\tdepth\tw\th\tluma_px\tsamples\tours_ms\tours_lo\tours_hi\tdav1d_ms\tdav1d_lo\tdav1d_hi\tratio_med\tratio_lo\tratio_hi\tdisjoint"]

    for fmt in ("420", "444"):
        for depth in (8, 10):
            sel = [v for v in vecs if geom(v)["fmt"] == fmt and geom(v)["depth"] == depth]
            if not sel:
                continue
            print()
            print(f"=== YUV{fmt}  {depth}bpc  t={','.join(map(str,threads))}  (n={len(rounds)} rounds) ===")
            print(f"{'size':>12} {'Mpx':>7} | {'ours ms/f':>10} {'[min..max]':>18} | "
                  f"{'dav1d ms/f':>10} {'[min..max]':>18} | {'ratio':>6} {'[min..max]':>16} band")
            xs_l, xs_s, y_ours, y_dav = [], [], [], []
            prev_pr = None
            for v in sel:
                g = geom(v)
                o = [cell[(ours, v, rd)] for rd in rounds if (ours, v, rd) in cell]
                d = [cell[(ref, v, rd)] for rd in rounds if (ref, v, rd) in cell]
                pr = [cell[(ours, v, rd)] / cell[(ref, v, rd)] for rd in rounds
                      if (ours, v, rd) in cell and (ref, v, rd) in cell]
                om, dm, rm = median(o), median(d), median(pr)
                # The claim this table makes is about how the RATIO moves with
                # size, so the disjointness that matters is this size's ratio
                # band against the PREVIOUS size's -- not ours-vs-dav1d, which
                # is trivially disjoint and would be a vacuous green tick.
                if prev_pr is None:
                    disj = "-"
                elif max(pr) < min(prev_pr) or max(prev_pr) < min(pr):
                    disj = "vs-prev:disjoint"
                else:
                    disj = "vs-prev:OVERLAP"
                prev_pr = pr
                print(f"{g['w']}x{g['h']:<7} {g['luma']/1e6:7.4f} | {om:10.4f} "
                      f"[{min(o):8.4f}..{max(o):8.4f}] | {dm:10.4f} "
                      f"[{min(d):8.4f}..{max(d):8.4f}] | {rm:6.3f} "
                      f"[{min(pr):6.3f}..{max(pr):6.3f}] {disj}")
                xs_l.append(g["luma"]); xs_s.append(g["samples"])
                y_ours.append(om); y_dav.append(dm)
                out_lines.append(
                    f"{fmt}\t{depth}\t{g['w']}\t{g['h']}\t{g['luma']}\t{g['samples']}\t"
                    f"{om:.5f}\t{min(o):.5f}\t{max(o):.5f}\t{dm:.5f}\t{min(d):.5f}\t{max(d):.5f}\t"
                    f"{rm:.4f}\t{min(pr):.4f}\t{max(pr):.4f}\t{disj}")

            # A two-parameter linear model over 3.5 decades of pixel count is
            # misspecified whenever ms/MP is not flat -- and it is not, for
            # either decoder. Print ms/MP per size so the misspecification is
            # visible rather than hidden inside an R^2, and fit the small end
            # separately, where a line IS defensible and the intercept is a
            # physical per-frame fixed cost rather than a 4K extrapolation.
            print("  -- ms per megapixel (the model check: a flat column means alpha ~ 0) --")
            print("     " + "  ".join(f"{g//1000}k:{o/(g/1e6):.1f}/{d/(g/1e6):.1f}"
                                      for g, o, d in zip(xs_l, y_ours, y_dav)))
            # The least model-dependent alpha available: an affine fit through
            # the two SMALLEST cells only. It assumes nothing about the rest of
            # the ladder, and at 2,304 px the pixel term is small enough that
            # the intercept is not an extrapolation from far away.
            def two_pt(ys):
                b = (ys[1] - ys[0]) / (xs_l[1] - xs_l[0])
                return ys[0] - b * xs_l[0], b
            a2o, b2o = two_pt(y_ours)
            a2d, b2d = two_pt(y_dav)
            print(f"  -- two-point alpha ({xs_l[0]} and {xs_l[1]} px only) --")
            print(f"     ours : alpha = {a2o*1000:7.2f} us/frame   beta = {b2o*1e6:6.2f} ms/MP"
                  f"   (alpha is {100*a2o/y_ours[0]:4.1f}% of the tiny frame)")
            print(f"     dav1d: alpha = {a2d*1000:7.2f} us/frame   beta = {b2d*1e6:6.2f} ms/MP"
                  f"   (alpha is {100*a2d/y_dav[0]:4.1f}% of the tiny frame)")
            print(f"     alpha_ours - alpha_dav1d = {(a2o-a2d)*1000:+.2f} us/frame")

            k = min(3, len(xs_l))
            ao, bo, r2o = ols(xs_l[:k], y_ours[:k])
            ad, bd, r2d = ols(xs_l[:k], y_dav[:k])
            print(f"  -- small-end fit ({k} smallest sizes only) --")
            print(f"     ours : alpha = {ao*1000:8.1f} us/frame   beta = {bo*1e6:7.3f} ms/MP  R2={r2o:.5f}")
            print(f"     dav1d: alpha = {ad*1000:8.1f} us/frame   beta = {bd*1e6:7.3f} ms/MP  R2={r2d:.5f}")
            print(f"     alpha_ours - alpha_dav1d = {(ao-ad)*1000:+.1f} us/frame   "
                  f"alpha ratio = {(ao/ad) if ad else float('nan'):.3f}")

            for label, xs, unit in (("luma px", xs_l, "MP"), ("total samples", xs_s, "Msample")):
                print(f"  -- fit vs {label} --")
                for name, wts in (("OLS      ", None),
                                  ("rel-err  ", "rel")):
                    for who, ys in (("ours ", y_ours), ("dav1d", y_dav)):
                        w = [1.0 / (y * y) for y in ys] if wts == "rel" else None
                        a, b, r2 = ols(xs, ys, w)
                        print(f"     {name} {who}: alpha = {a*1000:8.1f} us/frame   "
                              f"beta = {b*1e6:8.3f} ms/{unit}   R2={r2:.5f}")
                    aw = [1.0 / (y * y) for y in y_ours] if wts == "rel" else None
                    dw = [1.0 / (y * y) for y in y_dav] if wts == "rel" else None
                    ao, bo, _ = ols(xs, y_ours, aw)
                    ad, bd, _ = ols(xs, y_dav, dw)
                    print(f"     {name} ratio: alpha_ours/alpha_dav1d = "
                          f"{(ao/ad) if ad else float('nan'):6.3f}   "
                          f"beta_ours/beta_dav1d = {(bo/bd) if bd else float('nan'):6.3f}")
            print(f"  -- tiny cell direct (no model): ours {y_ours[0]*1000:.1f} us  "
                  f"dav1d {y_dav[0]*1000:.1f} us  ratio {y_ours[0]/y_dav[0]:.3f} --")

    if out_tsv:
        with open(out_tsv, "w") as fh:
            fh.write("\n".join(out_lines) + "\n")
        print(f"\nwrote {out_tsv}")


if __name__ == "__main__":
    main()
