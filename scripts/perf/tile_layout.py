#!/usr/bin/env python3
"""Print the AV1 TILE LAYOUT (TileCols x TileRows) for each vector.

Why this exists, ahead of any thread-scaling table: AV1 tile threading cannot
exceed the tile count. A decoder handed `--threads 8` on a ONE-TILE stream has
no tile-level work to split, so whatever t>1 buys there comes from the
post-filter pipeline (deblock/CDEF/LR sb-row tasks), not from tiles. Reporting
`tile_cols x tile_rows` BEFORE the latency tables is the difference between "t=8
did not help" and "t=8 could not have helped".

The tile count lives in the UNCOMPRESSED FRAME HEADER (spec 5.9.15
`tile_info()`), not the sequence header, so `seq_header_flags.py` cannot answer
this. This walks the sequence header far enough to learn the three things the
frame header parse needs (`max_frame_width/height`, `use_128x128_superblock`,
`enable_superres`), then parses the frame header of the FIRST frame up to
`tile_info()`.

SCOPE, stated so this is not over-read: only the `reduced_still_picture_header`
form (every AVIF still, and every vector on this ladder) is parsed. Anything
else prints `<unsupported>` rather than guessing — a mis-parse here would be
worse than a blank.

TEETH: run it on a vector known to be multi-tile. `v4k_8tile` must read 8, and
a ladder vector must read 1; a parser that can only ever print 1 proves nothing.

Usage: tile_layout.py <file-or-dir> ...
"""

import glob
import os
import sys

MAX_TILE_WIDTH = 4096
MAX_TILE_AREA = 4096 * 2304
MAX_TILE_COLS = 64
MAX_TILE_ROWS = 64


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


def obus(data):
    """Yield (obu_type, payload) in order."""
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
        yield obu_type, data[i:i + size]
        i += size


def parse_seq(payload):
    r = Bits(payload)
    s = {}
    s["seq_profile"] = r.f(3)
    s["still_picture"] = r.f(1)
    s["reduced_still"] = r.f(1)
    if not s["reduced_still"]:
        return None
    s["seq_level_idx"] = r.f(5)
    wb = r.f(4) + 1
    hb = r.f(4) + 1
    s["max_w"] = r.f(wb) + 1
    s["max_h"] = r.f(hb) + 1
    s["sb128"] = r.f(1)
    s["filter_intra"] = r.f(1)
    s["intra_edge_filter"] = r.f(1)
    s["superres"] = r.f(1)
    s["cdef"] = r.f(1)
    s["restoration"] = r.f(1)
    return s


def tile_log2(blk, target):
    k = 0
    while (blk << k) < target:
        k += 1
    return k


def parse_frame_tiles(payload, s):
    """Parse uncompressed_header() up to and including tile_info().

    Only the reduced_still_picture_header path. Every branch skipped below is
    skipped because the reduced form fixes its condition, and the reason is
    named inline so the omission can be checked against the spec.
    """
    r = Bits(payload)
    # reduced_still: show_existing_frame=0, frame_type=KEY_FRAME, FrameIsIntra=1,
    # show_frame=1, showable_frame=0, error_resilient_mode=1 -- no bits read.
    r.f(1)  # disable_cdf_update
    # reduced_still fixes seq_force_screen_content_tools = SELECT (2), so the
    # per-frame bit IS present.
    allow_scr = r.f(1)
    if allow_scr:
        # seq_force_integer_mv = SELECT (2) in the reduced form -> bit present.
        r.f(1)  # force_integer_mv
    # frame_id_numbers_present_flag = 0 -> no current_frame_id.
    # frame_size_override_flag = 0 (reduced) -> no frame_size bits.
    # OrderHintBits = 0 (enable_order_hint = 0 in the reduced form) -> f(0).
    # primary_ref_frame = PRIMARY_REF_NONE (FrameIsIntra) -> no bits.
    # decoder_model_info_present_flag = 0 -> no bits.
    # refresh_frame_flags = allFrames (KEY_FRAME && show_frame) -> no bits.
    # frame_size(): frame_size_override_flag = 0 -> dims come from the sequence
    # header; superres_params(): enable_superres gates the only bit.
    if s["superres"]:
        r.f(1)  # use_superres (coded_denom follows only if set; see below)
        # A set use_superres would need SUPERRES_DENOM_BITS more; every vector
        # here has enable_superres = 0, so refuse rather than half-parse.
        return None
    # render_size()
    if r.f(1):  # render_and_frame_size_different
        r.f(16)
        r.f(16)
    if allow_scr:  # && UpscaledWidth == FrameWidth (true: no superres)
        r.f(1)  # allow_intrabc
    # reduced_still -> disable_frame_end_update_cdf = 1, no bit.
    # primary_ref_frame == PRIMARY_REF_NONE -> cdf init, no bits.
    # use_ref_frame_mvs = 0 -> no motion_field_estimation.

    # ---- tile_info() ----
    mi_cols = 2 * ((s["max_w"] + 7) >> 3)
    mi_rows = 2 * ((s["max_h"] + 7) >> 3)
    sb128 = s["sb128"]
    sb_cols = ((mi_cols + 31) >> 5) if sb128 else ((mi_cols + 15) >> 4)
    sb_rows = ((mi_rows + 31) >> 5) if sb128 else ((mi_rows + 15) >> 4)
    sb_shift = 5 if sb128 else 4
    sb_size = sb_shift + 2
    max_tile_width_sb = MAX_TILE_WIDTH >> sb_size
    max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size)
    min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols)
    max_log2_tile_cols = tile_log2(1, min(sb_cols, MAX_TILE_COLS))
    max_log2_tile_rows = tile_log2(1, min(sb_rows, MAX_TILE_ROWS))
    min_log2_tiles = max(min_log2_tile_cols,
                         tile_log2(max_tile_area_sb, sb_rows * sb_cols))

    uniform = r.f(1)
    if uniform:
        cols_log2 = min_log2_tile_cols
        while cols_log2 < max_log2_tile_cols:
            if r.f(1):
                cols_log2 += 1
            else:
                break
        min_log2_tile_rows = max(min_log2_tiles - cols_log2, 0)
        rows_log2 = min_log2_tile_rows
        while rows_log2 < max_log2_tile_rows:
            if r.f(1):
                rows_log2 += 1
            else:
                break
        tile_cols = (sb_cols + (1 << cols_log2) - 1) >> cols_log2
        tile_cols = 1 << cols_log2
        tile_rows = 1 << rows_log2
    else:
        # Non-uniform: widths are coded per tile with ns(). aomenc's
        # --tile-columns path emits the uniform form, so this is not exercised
        # by our vectors; parse it properly rather than assume.
        widest = 0
        start_sb = 0
        i = 0
        while start_sb < sb_cols:
            max_width = min(sb_cols - start_sb, max_tile_width_sb)
            w = ns(r, max_width) + 1
            widest = max(widest, w)
            start_sb += w
            i += 1
        tile_cols = i
        # tile rows need maxTileHeightSb from the widest column; follow spec.
        if min_log2_tiles > 0:
            max_tile_area_sb2 = (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        else:
            max_tile_area_sb2 = sb_rows * sb_cols
        max_tile_height_sb = max(max_tile_area_sb2 // widest, 1)
        start_sb = 0
        i = 0
        while start_sb < sb_rows:
            max_height = min(sb_rows - start_sb, max_tile_height_sb)
            h = ns(r, max_height) + 1
            start_sb += h
            i += 1
        tile_rows = i
    return dict(tile_cols=tile_cols, tile_rows=tile_rows,
                sb_cols=sb_cols, sb_rows=sb_rows, sb128=sb128,
                mi_cols=mi_cols, mi_rows=mi_rows, uniform=uniform)


def ns(r, n):
    w = 0
    x = n
    while x:
        x >>= 1
        w += 1
    m = (1 << w) - n
    v = r.f(w - 1)
    if v < m:
        return v
    extra = r.f(1)
    return (v << 1) - m + extra


def one(path):
    data = open(path, "rb").read()
    if data[:4] == b"DKIF":
        hdr = int.from_bytes(data[6:8], "little")
        sz = int.from_bytes(data[hdr:hdr + 4], "little")
        data = data[hdr + 12:hdr + 12 + sz]
    seq = None
    for t, payload in obus(data):
        if t == 1:
            seq = parse_seq(payload)
            if seq is None:
                return None, "<not reduced_still_picture_header>"
        elif t in (3, 6) and seq is not None:
            try:
                r = parse_frame_tiles(payload, seq)
            except (IndexError, ZeroDivisionError):
                return seq, "<frame header parse failed>"
            if r is None:
                return seq, "<superres present; refused>"
            return seq, r
    return seq, "<no frame header OBU>"


def main():
    paths = []
    for a in sys.argv[1:]:
        if os.path.isdir(a):
            got = sorted(glob.glob(os.path.join(a, "*.obu")))
            got += sorted(glob.glob(os.path.join(a, "*.ivf")))
            got += sorted(glob.glob(os.path.join(a, "*.avif")))
            paths += got
        else:
            paths.append(a)
    print("vector\tw\th\tsb128\tsb_cols\tsb_rows\ttile_cols\ttile_rows\ttiles\tsbrows_per_tile")
    for p in paths:
        name = os.path.basename(p)
        if p.endswith(".avif"):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            data = open(p, "rb").read()
            off = data.find(b"mdat")
            if off < 0:
                print(f"{name}\t<no mdat>")
                continue
            # crude: AVIF primary item payload starts right after the mdat box
            # header. Prefer the .ivf/.obu form when available.
            print(f"{name}\t<use the .ivf; avif item offsets not parsed here>")
            continue
        seq, r = one(p)
        if not isinstance(r, dict):
            print(f"{name}\t{r}")
            continue
        print(f"{name}\t{seq['max_w']}\t{seq['max_h']}\t{r['sb128']}\t"
              f"{r['sb_cols']}\t{r['sb_rows']}\t{r['tile_cols']}\t{r['tile_rows']}\t"
              f"{r['tile_cols'] * r['tile_rows']}\t{r['sb_rows'] // max(r['tile_rows'],1)}")


if __name__ == "__main__":
    main()
