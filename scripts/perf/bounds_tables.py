#!/usr/bin/env python3
"""Derive the decision tables from a `--features __probe_bounds` report.

Reads the raw `B*` rows one run emitted and prints:

  A. over-reservation per site  — reserved bytes / footprint bytes, and the
     SHAPE of the waste (leading / trailing / inter-row-gap).
  B. the widening budget        — for each site, how many acquisitions had a
     concurrently-live foreign MUTABLE reservation within k bytes, for every
     k in the histogram. Widening that site's extent by k collides exactly
     that often.
  C. concurrent-conflict sets   — per ordered site pair: co-live count, whether
     the RESERVATIONS ever intersected, whether the FOOTPRINTS ever did, and
     the closest approach.

Usage: bounds_tables.py <report.txt> [--top N]
"""

import sys

GAPS = ["0", "4", "16", "64", "256", "1K", "4K", "16K", "64K", "1M", ">1M", "none"]


def read(path):
    site, conc, gapmut, pair, hdr, inst = [], [], [], [], {}, []
    for line in open(path, encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split("\t")
        tag = f[0]
        if tag in ("BOUNDS", "SITES", "CORPUS", "RUN", "BOVLSUM", "BINSTSUM"):
            hdr[tag] = f[1:]
        elif tag == "BSITE":
            site.append(f)
        elif tag == "BCONC":
            conc.append(f)
        elif tag == "BGAPMUT":
            gapmut.append(f)
        elif tag == "BPAIR":
            pair.append(f)
        elif tag == "BINST":
            inst.append(f)
    return site, conc, gapmut, pair, inst, hdr


def short(where):
    return where.replace("src/", "").replace("include/dav1d/", "")


def main():
    path = sys.argv[1]
    top = 14
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    site, conc, gapmut, pair, inst, hdr = read(path)

    print(f"# {path}")
    for k in ("RUN", "CORPUS", "SITES", "BOUNDS", "BOVLSUM", "BINSTSUM"):
        if k in hdr:
            print(f"{k}\t" + "\t".join(hdr[k]))
    print()

    print("## A. over-reservation and the shape of the waste")
    print(
        "site\tn\tres_mean\tfp_mean\tover\tkind\trows\tw\tgap_waste\tlead\ttail\tnever_deref"
    )
    for r in site[:top]:
        (
            _,
            n,
            _mut,
            res,
            fp,
            over,
            kind,
            _nd,
            _nw,
            nnever,
            rows,
            w,
            gap,
            lead,
            tail,
            _nr,
            _nwr,
            _le,
            where,
        ) = r[:19]
        print(
            f"{short(where)}\t{n}\t{res}\t{fp}\t{over}\t{kind}\t{rows}\t{w}\t{gap}\t{lead}\t{tail}\t{nnever}"
        )
    print()

    print("## B. widening budget — acquisitions with a concurrent foreign WRITE within k bytes")
    print("(cumulative: widening the reservation by k collides this many times)")
    print("site\tn\t" + "\t".join("<=" + g for g in GAPS[:-1]) + "\tno_conc_write")
    for r in gapmut[:top]:
        n = r[1]
        h = [int(x) for x in r[2 : 2 + len(GAPS)]]
        where = r[2 + len(GAPS)]
        cum, acc = [], 0
        for i in range(len(GAPS) - 1):
            acc += h[i]
            cum.append(acc)
        print(f"{short(where)}\t{n}\t" + "\t".join(str(c) for c in cum) + f"\t{h[-1]}")
    print()

    print("## C. concurrent-conflict sets (ordered pairs, RAW counts)")
    print(
        "acquiring\tconcurrent\tco_live\tres_intersect\tfp_intersect\trowband_intersect\tforeign_is_write\tmin_gap\tverdict"
    )
    for r in pair[:40]:
        _, n, res, fp, row, fmut, mg, a, b = r[:9]
        writer = int(fmut) > 0
        if int(res) > 0 and writer:
            # A write cannot be concurrent with an overlapping reservation:
            # the tracker panics. Seeing this means the PROBE is wrong.
            v = "IMPOSSIBLE-instrument-error"
        elif int(res) > 0:
            # Both sides immutable. Legal, and the reason a shared read window
            # can never be "owned" by one worker.
            v = "shared reads overlap (legal, no conflict)"
        elif writer:
            v = "no overlap; widening room vs a WRITER = min_gap"
        else:
            v = "no overlap; readers only, widening room = min_gap"
        print(f"{short(a)}\t{short(b)}\t{n}\t{res}\t{fp}\t{row}\t{fmut}\t{mg}\t{v}")


if __name__ == "__main__":
    main()
