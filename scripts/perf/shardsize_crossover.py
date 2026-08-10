#!/usr/bin/env python3
"""The size sweep's headline table: geometry, counts and wall ratio in one row.

Plots every cell against ROWS PER BLOCK — `2^shift / stride`, the quantity the
height model says the defect is a function of — rather than against pixels, and
prints the stride-divides-block flag beside it because that is the second
predictor (when it holds, within-row accesses never cross a block boundary and
the shipped `multi` counter reads exactly zero).

The plane geometry is the allocator's own arithmetic
(`Rav1dPicAllocator::alloc_picture_data`), and every shift it predicts was
checked against the shift the tracker reports through `--features
__probe_bounds`. It is a closed form, not a fit.

Usage: shardsize_crossover.py <wallcpu.tsv> <counts_report.tsv> [--tsv out.tsv]
"""
import collections
import re
import statistics
import sys

ARMS = ("bps1", "bpshalf", "bpsq", "bpsrows", "untracked", "dav1d_fd1")


def plane(w, h, hbd=0):
    """(len, stride) of the luma plane, from the allocator's arithmetic."""
    stride = ((w + 127) & ~127) << hbd
    if stride & 1023 == 0:
        stride += 64
    return stride * ((h + 127) & ~127), stride


def shipped_shift(length, target_blocks=256):
    return (length // target_blocks).bit_length() - 1


def dims(name):
    m = re.search(r"[CL](\d+)x(\d+)_", name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def main():
    wall_path, counts_path = sys.argv[1], sys.argv[2]
    tsv = sys.argv[sys.argv.index("--tsv") + 1] if "--tsv" in sys.argv else None

    by = collections.defaultdict(list)
    foreign = 0
    rounds = set()
    for line in open(wall_path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 13:
            continue
        rnd, arm, vec, t, nlo, wlo, ulo, slo, nhi, whi, uhi, shi, fo = f[:13]
        dn = int(nhi) - int(nlo)
        by[(vec, arm)].append((
            (int(whi) - int(wlo)) / dn,
            ((int(uhi) + int(shi)) - (int(ulo) + int(slo))) / dn,
        ))
        foreign = max(foreign, int(fo))
        rounds.add(int(rnd))

    counts = {}
    for line in open(counts_path):
        f = line.rstrip("\n").split("\t")
        if f[0] == "vector":
            continue
        counts[(f[0], f[3])] = f

    vecs = sorted({v for v, _ in by}, key=lambda v: (plane(*dims(v))[1], dims(v)[1]))
    print(f"rounds={len(rounds)} foreign_max={foreign}  "
          f"(t=1 calibration puts this grid's noise floor at ~1.2%)")
    print()
    hdr = (f"{'cell':22}{'w':>5}{'h':>5}{'strd':>6}{'sh':>3}{'rows/blk':>9}{'s|b':>4}"
           f"{'rsmax':>6}{'pctwide':>8}{'wide/f':>7}{'wall_ms':>9}{'cpu_ms':>8}{'cores':>6}")
    for a in ARMS:
        hdr += f"{a:>10}"
    print(hdr + f"{'cpu_rows':>9}{'disjoint':>10}")
    out = ["\t".join(("cell w h stride shift rows_per_block stride_divides "
                      "lf_row_shards_max cdef_pct_row_wide wide_pf base_wall_ms "
                      "base_cpu_ms cores_busy " + " ".join(ARMS)
                      + " cpu_ratio_bpsrows disjoint").split())]
    for v in vecs:
        w, h = dims(v)
        ln, st = plane(w, h)
        sh = shipped_shift(ln)
        rpb = (1 << sh) / st
        sdb = int((1 << sh) % st == 0)
        c = counts.get((v, "plain"))
        wide = c[15] if c else "-"
        rsmax = c[11] if c else "-"      # lf_rowshmax
        pctw = c[8] if c else "-"        # cdef_pctwide
        base = [x[0] for x in by[(v, "plain")]]
        basec = [x[1] for x in by[(v, "plain")]]
        bm = statistics.median(base)
        bc = statistics.median(basec)
        line = (f"{v[:21]:22}{w:>5}{h:>5}{st:>6}{sh:>3}{rpb:>9.2f}{sdb:>4}"
                f"{rsmax:>6}{pctw:>8}{wide:>7}{bm:>9.3f}{bc:>8.2f}{bc / bm:>6.2f}")
        row = [v, w, h, st, sh, f"{rpb:.3f}", sdb, rsmax, pctw, wide, f"{bm:.4f}",
               f"{bc:.4f}", f"{bc / bm:.2f}"]
        disj = []
        for a in ARMS:
            k = (v, a)
            if k not in by:
                line += f"{'-':>10}"
                row.append("")
                continue
            m = [x[0] for x in by[k]]
            r = statistics.median(m) / bm
            line += f"{r:>10.3f}"
            row.append(f"{r:.4f}")
            if a in ("bpshalf", "bpsrows") and (min(base) > max(m) or min(m) > max(base)):
                disj.append(a)
        kr = (v, "bpsrows")
        cpu_rows = (statistics.median(x[1] for x in by[kr]) / bc) if kr in by else float("nan")
        line += f"{cpu_rows:>9.3f}{','.join(disj) if disj else '-':>10}"
        row.append(f"{cpu_rows:.4f}")
        row.append(",".join(disj))
        print(line)
        out.append("\t".join(str(x) for x in row))
    if tsv:
        open(tsv, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
