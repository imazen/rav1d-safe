#!/usr/bin/env python3
"""Reduce the size-sweep counting dumps to one row per (vector, rung).

Two tables come out, because the size question has two countable halves:

  RECT  — the geometry the height model predicts: rows-per-block, how many shard
          lines one strided access touches (`row_shards_mean/max`), and what
          fraction exceed MAX_SHARDS_PER_BORROW (`pct_row_wide`). Also `shifts`,
          which is the rung's liveness proof: a rung whose shifts equal the
          default's did not arm on that vector.

  WIDE  — the SHIPPED counters (`multi`, `w_shards`), normalised per frame so
          cells with different iteration counts compare.

Usage: shardsize_counts_report.py <countsdir> [--tsv out.tsv]
"""
import re
import sys
from collections import defaultdict

CDEF_SITES = (
    "safe_simd/cdef_arm.rs:622:9",
    "safe_simd/cdef_arm.rs:192:9",
    "safe_simd/cdef_arm.rs:1217:9",
    "cdef_apply.rs:104:32",
)
LF_SITE = "loopfilter.rs:809:17"


def parse(path):
    rect, wide, meta = [], None, {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if p[0] == "RUN":
            meta["vec"] = p[1]
            meta["wh"] = p[2]
            meta["iters"] = int(p[5].split("=")[1])
        elif p[0] == "RECT":
            rect.append(p)
        elif p[0] == "WIDE":
            wide = p
        elif p[0] == "cell":
            meta["tag"], meta["vecname"], meta["threads"] = p[1], p[2], p[3]
    return rect, wide, meta


def dims(name):
    m = re.search(r"C(\d+)x(\d+)_", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"L(\d+)x(\d+)_", name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def main():
    d = sys.argv[1]
    tsv = None
    if "--tsv" in sys.argv:
        tsv = sys.argv[sys.argv.index("--tsv") + 1]
    import os

    rows_rect, rows_wide = {}, {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".txt"):
            continue
        rect, wide, meta = parse(os.path.join(d, fn))
        tag = meta.get("tag", fn)
        rung, probe = tag.split("__", 1)
        vec = meta.get("vecname", fn)
        it = meta.get("iters", 1)
        if probe == "probebounds" and rect:
            by_site = {r[-1].replace("src/", ""): r for r in rect}
            cd = [by_site[s] for s in CDEF_SITES if s in by_site]
            lf = by_site.get(LF_SITE)
            # Field order of the `#rectsite` header, 0 = the literal "RECT":
            # 9 rows_mean, 10 rows_max, 13 row_shards_mean, 14 row_shards_max,
            # 16 pct_row_wide, 17 pct_wide_c5, 21 shifts, 22 where.
            rows_rect[(vec, rung)] = dict(
                cdef_rowsh=max((float(r[13]) for r in cd), default=float("nan")),
                cdef_rowshmax=max((int(r[14]) for r in cd), default=0),
                cdef_pctwide=max((float(r[16]) for r in cd), default=float("nan")),
                lf_rowsh=float(lf[13]) if lf else float("nan"),
                lf_rowshmax=int(lf[14]) if lf else 0,
                lf_pctwide=float(lf[16]) if lf else float("nan"),
                lf_c5=float(lf[17]) if lf else float("nan"),
                lf_rowsmean=float(lf[9]) if lf else float("nan"),
                cdef_rowsmean=max((float(r[9]) for r in cd), default=float("nan")),
                shifts=",".join(sorted({s for r in rect for s in r[21].split(",")})),
            )
        if probe == "probewide" and wide:
            rows_wide[(vec, rung)] = dict(
                const_shift=int(wide[1]),
                slow=int(wide[2]) / it,
                multi=int(wide[3]) / it,
                w_shards=int(wide[4]) / it,
                w_full=int(wide[6]) / it,
            )

    vecs = sorted({v for v, _ in list(rows_rect) + list(rows_wide)},
                  key=lambda v: (dims(v)[0], dims(v)[1]))
    out = []
    hdr = ("vector w h rung shifts cdef_rows cdef_rowsh cdef_rowshmax cdef_pctwide "
           "lf_rows lf_rowsh lf_rowshmax lf_pctwide lf_c5 multi_pf wide_pf wfull_pf").split()
    out.append("\t".join(hdr))
    for v in vecs:
        w, h = dims(v)
        for rung in ("plain", "bps1", "bpshalf", "bpsrows"):
            r = rows_rect.get((v, rung))
            wd = rows_wide.get((v, rung))
            if not r and not wd:
                continue
            r = r or {}
            wd = wd or {}
            out.append("\t".join(str(x) for x in [
                v, w, h, rung,
                r.get("shifts", "-"),
                f"{r.get('cdef_rowsmean', float('nan')):.2f}",
                f"{r.get('cdef_rowsh', float('nan')):.3f}",
                r.get("cdef_rowshmax", "-"),
                f"{r.get('cdef_pctwide', float('nan')):.2f}",
                f"{r.get('lf_rowsmean', float('nan')):.2f}",
                f"{r.get('lf_rowsh', float('nan')):.3f}",
                r.get("lf_rowshmax", "-"),
                f"{r.get('lf_pctwide', float('nan')):.2f}",
                f"{r.get('lf_c5', float('nan')):.2f}",
                f"{wd.get('multi', float('nan')):.0f}",
                f"{wd.get('w_shards', float('nan')):.0f}",
                f"{wd.get('w_full', float('nan')):.0f}",
            ]))
    text = "\n".join(out)
    print(text)
    if tsv:
        open(tsv, "w").write(text + "\n")


if __name__ == "__main__":
    main()
