#!/usr/bin/env python3
"""The doubling->collapse TRANSFER COEFFICIENT, computed per cell.

`docs/RECT_SHIP.md` §7 priced the CDEF registration population by DOUBLING it in
one binary and called the result an upper bound on what collapsing it could buy,
with a standing caveat: `LfBlock::fill`'s own doubling priced +3.37% wall on
`c256x2048` and its actual collapse delivered ~0 there. A ceiling with an
unknown discount is not a decision.

This computes the discount ON THE CELLS WHERE BOTH NUMBERS EXIST. For each cell:

    ceiling  = (1 - 1/rows) * (doubling wall ratio - 1)     # the collapse's
                                                            # arithmetic best
    tau      = delivered / ceiling                          # measured transfer

`rows` is the per-call row count from `--features __probe_bounds`' RECT rows, so
the fraction removed is measured, not assumed. `delivered` for `fill` is the
same-source-controlled t=8 win. `tau` then prices the OTHER site's collapse
before it is built.

Usage: layout_transfer.py <grid.tsv> [--drop-round N] [--keep-loaded]
"""

import sys
from statistics import median

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from layout_spread import load, paired  # noqa: E402

# rows per call at each site, from `probe_tracker --features __probe_bounds`
# (`counts/pb_<cell>_t8.txt`, RECT rows). Population deltas from `probe-sites`
# with each doubling env var on/off (`counts/populations.tsv`).
ROWS = {
    "c1024x192:t8": {"lf": 9.01, "cdef": 8.00},
    "c1024x384:t8": {"lf": 9.01, "cdef": 8.00},
    "c1024x576:t8": {"lf": 8.92, "cdef": 8.00},
}
POP = {  # base, +cdef, +lf  (registrations/frame)
    "c1024x192:t8": (156777, 33152, 60820),
    "c1024x384:t8": (333863, 76480, 125018),
    "c1024x576:t8": (529092, 121856, 190632),
}
# `fill`'s DELIVERED t=8 win at the shipped configuration, same-source
# controlled (`ship`/`plain2`), from docs/RECT_SHIP.md §6 grid R — cited, not
# re-measured here. Grid M re-measures it in-grid against a pad rung.
DELIVERED_LF = {
    "c1024x192:t8": 0.9851,
    "c1024x384:t8": 0.9762,
    "c1024x576:t8": 0.9826,
}


def main():
    a = sys.argv[1:]
    drop = int(a[a.index("--drop-round") + 1]) if "--drop-round" in a else 1
    cells, foreign, loaded = load(a[0], drop, "--keep-loaded" not in a)
    print(f"{'cell':14} {'site':5} {'+regs/frm':>10} {'dbl wall':>9} {'sign':>6} "
          f"{'dbl cpu':>8} {'ns/reg':>7} {'ceiling':>8} {'delivered':>10} {'tau':>6}")
    for cell in sorted(c for c in cells if c in ROWS):
        base_pop, dcdef, dlf = POP[cell]
        for site, off, on, dpop in (("lf", "lfoff", "lfon", dlf),
                                    ("cdef", "cdefoff", "cdefon", dcdef)):
            if off not in cells[cell] or on not in cells[cell]:
                continue
            w = paired(cells, cell, on, off, 0)
            c = paired(cells, cell, on, off, 1)
            wm, cm = median(w), median(c)
            sign = sum(1 for x in w if x < 1.0)
            # marginal ns per registration, from the CPU delta of the arm pair
            cpu_off = median(v[1] for v in cells[cell][off].values())
            ns = (cm - 1.0) * cpu_off * 1e6 / dpop
            rows = ROWS[cell][site]
            ceiling = (1.0 - 1.0 / rows) * (wm - 1.0)
            deliv = DELIVERED_LF.get(cell) if site == "lf" else None
            tau = ((1.0 - deliv) / ceiling) if (deliv and ceiling) else None
            print(f"{cell:14} {site:5} {dpop:10} {wm:9.4f} {sign:>3}/{len(w):<2} "
                  f"{cm:8.4f} {ns:7.2f} {ceiling * 100:7.2f}% "
                  f"{'' if deliv is None else f'{deliv:10.4f}'}"
                  f"{'' if tau is None else f' {tau:6.3f}'}")
        # what the measured tau predicts for the CDEF collapse on this cell
        if all(x in cells[cell] for x in ("lfoff", "lfon", "cdefoff", "cdefon")):
            wlf = median(paired(cells, cell, "lfon", "lfoff", 0))
            wcd = median(paired(cells, cell, "cdefon", "cdefoff", 0))
            clf = (1 - 1 / ROWS[cell]["lf"]) * (wlf - 1)
            ccd = (1 - 1 / ROWS[cell]["cdef"]) * (wcd - 1)
            tau = (1 - DELIVERED_LF[cell]) / clf
            print(f"{'':14} {'->':5} predicted CDEF collapse = "
                  f"{-100 * ccd * tau:+.2f}% wall (ceiling {-100 * ccd:+.2f}%, "
                  f"tau {tau:.3f})")


if __name__ == "__main__":
    main()
