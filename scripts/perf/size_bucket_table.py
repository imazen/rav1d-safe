#!/usr/bin/env python3
"""ms per MEGAPIXEL per bucket, per profiled cell.

Shares alone cannot answer "what grows with image size" — a bucket can shrink
as a percentage while growing per pixel, and per pixel is the unit the size
question is denominated in. This multiplies each bucket's share by the cell's
measured ms/frame (from the CLEAN sweep, not from the profile) and divides by
megapixels.

Usage: size_bucket_table.py <mspf.tsv> <profdir>
  mspf.tsv: "<vector>\\t<ms_per_frame>" per line; the vector name must be
  L<W>x<H>_<fmt>_<d>b so the pixel count can be recovered from it.
"""

import os
import re
import sys

VEC_RE = re.compile(r"^L(\d+)x(\d+)_(\d+)_(\d+)b$")
BUCKETS = ("entropy", "tracker", "kernels", "runtime", "other")


def main():
    mspf, profdir = sys.argv[1], sys.argv[2]
    rows = []
    for line in open(mspf):
        v, m = line.split()
        g = VEC_RE.match(v)
        if not g:
            continue
        px = int(g.group(1)) * int(g.group(2))
        p = os.path.join(profdir, f"{v}.buckets.txt")
        if not os.path.exists(p):
            print(f"# MISSING profile: {v}")
            continue
        share = {}
        for ln in open(p):
            t = ln.split()
            if len(t) == 3 and t[2].endswith("%"):
                share[t[0]] = float(t[2].rstrip("%"))
        rows.append((v, float(m), px, share))
    rows.sort(key=lambda r: (r[0].split("_")[-1], r[2]))
    hdr = f"{'vector':<22}{'Mpx':>7}{'ms/frame':>10}{'ms/MP':>8}"
    hdr += "".join(f"{b:>9}" for b in BUCKETS)
    print(hdr)
    for v, m, px, share in rows:
        mp = px / 1e6
        line = f"{v:<22}{mp:7.4f}{m:10.4f}{m/mp:8.2f}"
        line += "".join(f"{share.get(b, 0.0)/100*m/mp:9.2f}" for b in BUCKETS)
        print(line)
    print("\n# non-entropy ms/MP (everything the entropy decoder is not)")
    for v, m, px, share in rows:
        mp = px / 1e6
        ne = sum(share.get(b, 0.0) for b in BUCKETS if b != "entropy") / 100 * m / mp
        en = share.get("entropy", 0.0) / 100 * m / mp
        print(f"{v:<22} entropy {en:7.2f}   non-entropy {ne:7.2f}   "
              f"non-entropy share {100*ne/(en+ne):5.1f}%")


if __name__ == "__main__":
    main()
