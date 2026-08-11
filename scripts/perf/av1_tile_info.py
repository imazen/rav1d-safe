#!/usr/bin/env python3
"""Exact tile geometry of an AVIF still, parsed from the AV1 bitstream.

WHY THIS EXISTS
---------------
`avifenc --tilecolslog2 2 --tilerowslog2 1` is a REQUEST. libaom clamps it
against the frame's superblock count, so a small picture can silently come back
with fewer tiles than asked — and "the rung only arms on tiled content" makes an
un-noticed clamp a silent VOID cell rather than a wrong number. The encoder log
records what was asked for, not what was emitted, so the answer has to come out
of the bitstream.

Scope, stated so it is not mistaken for a general AV1 parser: it handles the
`reduced_still_picture_header = 1` shape that libaom emits for AVIF stills, and
the ordinary key-frame shape. It refuses anything else loudly rather than
guessing, because a wrong tile count here would be worse than no tile count.

Usage: av1_tile_info.py <file.avif|file.obu> ...
Prints: name  WxH  bitdepth  tile_cols x tile_rows = tiles  sb_cols x sb_rows
"""
import struct
import subprocess
import sys


class BitReader:
    def __init__(self, data):
        self.d = data
        self.pos = 0

    def f(self, n):
        v = 0
        for _ in range(n):
            byte = self.d[self.pos >> 3]
            bit = (byte >> (7 - (self.pos & 7))) & 1
            v = (v << 1) | bit
            self.pos += 1
        return v

    def uvlc(self):
        leading = 0
        while True:
            if self.f(1):
                break
            leading += 1
            if leading >= 32:
                return (1 << 32) - 1
        if leading == 0:
            return 0
        return self.f(leading) + (1 << leading) - 1

    def le(self, n):
        v = 0
        for i in range(n):
            v |= self.f(8) << (i * 8)
        return v

    def ns(self, n):
        # non-symmetric unsigned, AV1 spec 4.10.7
        w = n.bit_length()
        m = (1 << w) - n
        v = self.f(w - 1)
        if v < m:
            return v
        return (v << 1) - m + self.f(1)


def leb128(d, off):
    v = 0
    for i in range(8):
        b = d[off + i]
        v |= (b & 0x7F) << (i * 7)
        if not (b & 0x80):
            return v, off + i + 1
    raise ValueError("leb128 too long")


def obus(data):
    """Yield (obu_type, payload) for each OBU in a raw temporal unit."""
    off = 0
    while off < len(data):
        b = data[off]
        ext = (b >> 2) & 1
        has_size = (b >> 1) & 1
        otype = (b >> 3) & 0xF
        p = off + 1
        if ext:
            p += 1
        if has_size:
            size, p = leb128(data, p)
        else:
            size = len(data) - p
        yield otype, data[p:p + size]
        off = p + size


class Seq:
    pass


def parse_seq(payload):
    r = BitReader(payload)
    s = Seq()
    r.f(3)                       # seq_profile
    r.f(1)                       # still_picture
    s.reduced = r.f(1)           # reduced_still_picture_header
    if s.reduced:
        r.f(5)                   # seq_level_idx[0]
        s.decoder_model_info_present = 0
        s.frame_id_numbers_present = 0
        s.equal_picture_interval = 1
        s.frame_ids = 0
    else:
        timing_info_present = r.f(1)
        decoder_model_info_present = 0
        if timing_info_present:
            r.f(32); r.f(32)      # num_units_in_display_tick, time_scale
            equal_picture_interval = r.f(1)
            if equal_picture_interval:
                r.uvlc()
            decoder_model_info_present = r.f(1)
            if decoder_model_info_present:
                s.buffer_delay_length = r.f(5) + 1
                r.f(32); r.f(5); r.f(5)
        s.decoder_model_info_present = decoder_model_info_present
        initial_display_delay_present = r.f(1)
        operating_points_cnt = r.f(5) + 1
        for i in range(operating_points_cnt):
            r.f(12)               # operating_point_idc
            level = r.f(5)
            if level > 7:
                r.f(1)            # seq_tier
            if decoder_model_info_present:
                if r.f(1):        # decoder_model_present_for_this_op
                    r.f(s.buffer_delay_length); r.f(s.buffer_delay_length); r.f(1)
            if initial_display_delay_present:
                if r.f(1):
                    r.f(4)
        s.frame_ids = 0
    s.w_bits = r.f(4) + 1
    s.h_bits = r.f(4) + 1
    s.max_w = r.f(s.w_bits) + 1
    s.max_h = r.f(s.h_bits) + 1
    if not s.reduced:
        s.frame_ids = r.f(1)
        if s.frame_ids:
            r.f(4); r.f(3)
    s.use_128x128 = r.f(1)
    r.f(1)                        # enable_filter_intra
    r.f(1)                        # enable_intra_edge_filter
    if not s.reduced:
        raise SystemExit("this file is not a reduced_still_picture_header stream; "
                         "the parser refuses to guess the inter-frame shape")
    s.enable_order_hint = 0
    s.enable_superres = r.f(1)
    s.enable_cdef = r.f(1)
    s.enable_restoration = r.f(1)
    # color_config
    high_bd = r.f(1)
    s.bitdepth = 10 if high_bd else 8   # seq_profile 0/1 path
    s.mono = r.f(1)
    if r.f(1):                    # color_description_present
        r.f(8); r.f(8); r.f(8)
    if not s.mono:
        r.f(1)                    # color_range
        s.subx = 1
        s.suby = 1
        r.f(2)                    # chroma_sample_position (subsampled 420)
    r.f(1)                        # separate_uv_delta_q
    r.f(1)                        # film_grain_params_present
    s.r = r
    return s


def parse_frame_header(payload, s):
    r = BitReader(payload)
    # reduced_still_picture_header => show_existing_frame = 0, frame_type = KEY,
    # show_frame = 1, showable_frame = 0, error_resilient_mode = 1.
    disable_cdf_update = r.f(1)
    allow_screen_content_tools = 1 if s.force_screen_content_tools == 2 else s.force_screen_content_tools
    if s.force_screen_content_tools == 2:
        allow_screen_content_tools = r.f(1)
    if allow_screen_content_tools:
        if s.force_integer_mv == 2:
            r.f(1)
    # frame_size(): frame_size_override_flag == 0 under reduced header
    w, h = s.max_w, s.max_h
    if s.enable_superres:
        if r.f(1):
            r.f(3)
    # render_size()
    if r.f(1):
        r.f(16); r.f(16)
    allow_intrabc = 0
    if allow_screen_content_tools:
        allow_intrabc = r.f(1)
    # tile_info()
    mi_cols = 2 * ((w + 7) >> 3)
    mi_rows = 2 * ((h + 7) >> 3)
    sb_shift = 5 if s.use_128x128 else 4
    sb_size = sb_shift + 2
    sb_cols = (mi_cols + 31) >> 5 if s.use_128x128 else (mi_cols + 15) >> 4
    sb_rows = (mi_rows + 31) >> 5 if s.use_128x128 else (mi_rows + 15) >> 4
    max_tile_width_sb = 4096 >> sb_size
    max_tile_area_sb = (4096 * 2304) >> (2 * sb_size)
    min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols)
    max_log2_tile_cols = tile_log2(1, min(sb_cols, 64))
    max_log2_tile_rows = tile_log2(1, min(sb_rows, 64))
    min_log2_tiles = max(min_log2_tile_cols,
                         tile_log2(max_tile_area_sb, sb_rows * sb_cols))
    uniform = r.f(1)
    if uniform:
        log2_cols = min_log2_tile_cols
        while log2_cols < max_log2_tile_cols:
            if r.f(1):
                log2_cols += 1
            else:
                break
        min_log2_tile_rows = max(min_log2_tiles - log2_cols, 0)
        log2_rows = min_log2_tile_rows
        while log2_rows < max_log2_tile_rows:
            if r.f(1):
                log2_rows += 1
            else:
                break
        return w, h, 1 << log2_cols, 1 << log2_rows, sb_cols, sb_rows, True
    # non-uniform spacing
    widest = 0
    start_sb = 0
    cols = 0
    while start_sb < sb_cols:
        max_w_sb = min(sb_cols - start_sb, max_tile_width_sb)
        width_sb = r.ns(max_w_sb) + 1
        widest = max(widest, width_sb)
        start_sb += width_sb
        cols += 1
    max_tile_area_sb2 = (sb_rows * sb_cols)
    max_h_sb = max(max_tile_area_sb2 // widest, 1)
    start_sb = 0
    rows = 0
    while start_sb < sb_rows:
        m = min(sb_rows - start_sb, max_h_sb)
        height_sb = r.ns(m) + 1
        start_sb += height_sb
        rows += 1
    return w, h, cols, rows, sb_cols, sb_rows, False


def tile_log2(blk_size, target):
    k = 0
    while (blk_size << k) < target:
        k += 1
    return k


def primary_obu(path):
    if path.endswith(".obu") or path.endswith(".ivf"):
        d = open(path, "rb").read()
        if path.endswith(".ivf"):
            off = 32
            sz = struct.unpack_from("<I", d, off)[0]
            return d[off + 12:off + 12 + sz]
        return d
    out = subprocess.run(
        [sys.executable, "-c", PRIMARY_SNIPPET, path],
        capture_output=True)
    if out.returncode != 0:
        raise SystemExit(out.stderr.decode())
    return out.stdout


PRIMARY_SNIPPET = r'''
import sys, struct
d = open(sys.argv[1],'rb').read()
# Minimal ISOBMFF walk to the primary item's mdat extent (iloc, construction 0).
def boxes(buf, base=0):
    off = 0
    while off + 8 <= len(buf):
        size = struct.unpack_from('>I', buf, off)[0]
        typ = buf[off+4:off+8]
        hdr = 8
        if size == 1:
            size = struct.unpack_from('>Q', buf, off+8)[0]; hdr = 16
        elif size == 0:
            size = len(buf) - off
        yield typ, buf[off+hdr:off+size], base+off+hdr
        off += size
def find(buf, path, base=0):
    for typ, body, b in boxes(buf, base):
        if typ == path[0]:
            if len(path) == 1: return body, b
            r = find(body, path[1:], b)
            if r: return r
    return None
meta = find(d, [b'meta'])
body, base = meta
body = body[4:]; base += 4   # full box
pitm = find(body, [b'pitm'], base)
ver = pitm[0][0]
item_id = struct.unpack_from('>H' if ver == 0 else '>I', pitm[0], 4)[0]
iloc, ilb = find(body, [b'iloc'], base)
ver = iloc[0]; off = 4
b1 = iloc[off]; b2 = iloc[off+1]
offset_size = b1 >> 4; length_size = b1 & 15
base_offset_size = b2 >> 4
index_size = b2 & 15 if ver in (1,2) else 0
off += 2
if ver < 2:
    count = struct.unpack_from('>H', iloc, off)[0]; off += 2
else:
    count = struct.unpack_from('>I', iloc, off)[0]; off += 4
def rd(n, o):
    v = 0
    for i in range(n): v = (v<<8) | iloc[o+i]
    return v, o+n
for _ in range(count):
    if ver < 2:
        iid = struct.unpack_from('>H', iloc, off)[0]; off += 2
    else:
        iid = struct.unpack_from('>I', iloc, off)[0]; off += 4
    if ver in (1,2):
        off += 2  # construction_method
    off += 2      # data_reference_index
    bo, off = rd(base_offset_size, off)
    ext = struct.unpack_from('>H', iloc, off)[0]; off += 2
    for _ in range(ext):
        if index_size: _, off = rd(index_size, off)
        eo, off = rd(offset_size, off)
        el, off = rd(length_size, off)
        if iid == item_id:
            sys.stdout.buffer.write(d[bo+eo: bo+eo+el])
            sys.exit(0)
sys.exit('primary item not found')
'''


def main():
    print("vector\twxh\tbitdepth\ttile_cols\ttile_rows\ttiles\tsb_cols\tsb_rows\tuniform")
    for path in sys.argv[1:]:
        tu = primary_obu(path)
        seq = None
        res = None
        for otype, payload in obus(tu):
            if otype == 1:            # OBU_SEQUENCE_HEADER
                seq = parse_seq(payload)
                # reduced header implies these defaults
                seq.force_screen_content_tools = 2
                seq.force_integer_mv = 2
            elif otype in (3, 6) and seq is not None:  # FRAME_HEADER / FRAME
                res = parse_frame_header(payload, seq)
                break
        if res is None:
            print(f"{path}\tPARSE_FAILED")
            continue
        w, h, tc, tr, sbc, sbr, uni = res
        name = path.split("/")[-1]
        print(f"{name}\t{w}x{h}\t{seq.bitdepth}\t{tc}\t{tr}\t{tc*tr}\t{sbc}\t{sbr}\t{int(uni)}")


if __name__ == "__main__":
    main()
