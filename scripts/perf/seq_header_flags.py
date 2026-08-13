#!/usr/bin/env python3
"""Print the AV1 sequence-header tool flags for each vector in a directory.

One encoder config across a size ladder does NOT mean one TOOL SET: libaom
picks superblock size, CDEF and loop restoration per resolution, so two cells of
the same ladder can be running different kernels. This reads the flags out of
the bitstream rather than guessing, which is the difference between "the ratio
changes with size" and "the ratio changes because the encoder turned a tool on".

Sequence header layout, AV1 spec 5.5.1, for the reduced-still-picture form every
AVIF still uses:

    seq_profile f(3), still_picture f(1), reduced_still_picture_header f(1),
    seq_level_idx[0] f(5),
    frame_width_bits_minus_1 f(4), frame_height_bits_minus_1 f(4),
    max_frame_width_minus_1 f(n+1), max_frame_height_minus_1 f(m+1),
    use_128x128_superblock f(1), enable_filter_intra f(1),
    enable_intra_edge_filter f(1),
    enable_superres f(1), enable_cdef f(1), enable_restoration f(1),
    color_config() ...

NOTE ON SCOPE, so this is not over-read: `enable_cdef`/`enable_restoration` are
SEQUENCE-level permissions. Whether a given frame actually applies CDEF, or
picks anything but RESTORE_NONE per plane, lives in the uncompressed frame
header and is not parsed here. Absence of the permission is conclusive; presence
is not. Use a profile's leaf samples to prove a kernel EXECUTED.

Usage: seq_header_flags.py <dir-of-*.obu-or-*.ivf> ...
"""

import glob
import os
import sys


class Bits:
    def __init__(self, b):
        self.b = b
        self.i = 0

    def f(self, n):
        v = 0
        for _ in range(n):
            byte = self.b[self.i >> 3]
            v = (v << 1) | ((byte >> (7 - (self.i & 7))) & 1)
            self.i += 1
        return v


def leb128(b, i):
    v = 0
    for k in range(8):
        x = b[i]
        i += 1
        v |= (x & 0x7F) << (k * 7)
        if not (x & 0x80):
            break
    return v, i


def find_seq_obu(data):
    i = 0
    while i < len(data):
        h = data[i]
        i += 1
        obu_type = (h >> 3) & 0xF
        ext = (h >> 2) & 1
        has_size = (h >> 1) & 1
        if ext:
            i += 1
        if has_size:
            size, i = leb128(data, i)
        else:
            size = len(data) - i
        if obu_type == 1:
            return data[i:i + size]
        i += size
    return None


def parse(payload):
    r = Bits(payload)
    out = {}
    out["seq_profile"] = r.f(3)
    out["still_picture"] = r.f(1)
    red = r.f(1)
    out["reduced_still"] = red
    if not red:
        return out  # not the AVIF-still shape; do not guess past here
    out["seq_level_idx"] = r.f(5)
    wb = r.f(4) + 1
    hb = r.f(4) + 1
    out["max_w"] = r.f(wb) + 1
    out["max_h"] = r.f(hb) + 1
    out["sb128"] = r.f(1)
    out["filter_intra"] = r.f(1)
    out["intra_edge_filter"] = r.f(1)
    out["superres"] = r.f(1)
    out["cdef"] = r.f(1)
    out["restoration"] = r.f(1)
    return out


def main():
    paths = []
    for a in sys.argv[1:]:
        paths += sorted(glob.glob(os.path.join(a, "*.obu"))) if os.path.isdir(a) else [a]
        if os.path.isdir(a) and not paths:
            paths = sorted(glob.glob(os.path.join(a, "*.ivf")))
    print("vector\tw\th\tsb128\tfilter_intra\tintra_edge\tsuperres\tcdef_seq\trestore_seq")
    for p in paths:
        data = open(p, "rb").read()
        if data[:4] == b"DKIF":
            hdr = int.from_bytes(data[6:8], "little")
            sz = int.from_bytes(data[hdr:hdr + 4], "little")
            data = data[hdr + 12:hdr + 12 + sz]
        payload = find_seq_obu(data)
        if payload is None:
            print(f"{os.path.basename(p)}\t<no sequence header>")
            continue
        f = parse(payload)
        if not f.get("reduced_still"):
            print(f"{os.path.basename(p)}\t<not reduced_still_picture_header>")
            continue
        print(f"{os.path.basename(p)}\t{f['max_w']}\t{f['max_h']}\t{f['sb128']}\t"
              f"{f['filter_intra']}\t{f['intra_edge_filter']}\t{f['superres']}\t"
              f"{f['cdef']}\t{f['restoration']}")


if __name__ == "__main__":
    main()
