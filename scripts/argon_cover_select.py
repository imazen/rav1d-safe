#!/usr/bin/env python3
"""Select a minimal covering set of Argon streams and write a fetch manifest.

The official Argon release ships coverage reports (`prof_av1_profile*.xml`)
where every spec test-point names the streams that cover it
(<zeroCoveringStream>, <nonzeroCoveringStream>, <minCoveringStream>,
<maxCoveringStream>). This script:

  1. range-fetches the zip's central directory + the three XML reports,
  2. runs greedy set-cover (max uncovered items, tie-break smallest stream),
  3. emits tests/argon_cover_manifest.tsv:
       relpath <TAB> data_off <TAB> csize <TAB> usize <TAB> method <TAB> md5_nofg <TAB> md5_ref

The data offsets are valid for the canonical zip only — if ARGON_URL repacks
the suite differently the fetch helper falls back to per-file GET or a local
full extract. Reference MD5s are pulled from the zip's own sidecars.

Usage:
    scripts/argon_cover_select.py [--budget-mb N] [--out PATH]
"""

import re
import struct
import sys
import zlib
import urllib.request
from collections import defaultdict

URL = "https://aom-cwg-av1-argon-streams-public.s3.us-east-1.amazonaws.com/argon_coveragetool_av1_base_and_extended_profiles_v2.1.1.zip"
ROOT = "argon_coveragetool_av1_base_and_extended_profiles_v2.1/"
TAIL = 1 << 23  # last 8MB: EOCD locator is inside


def http_range(start, end):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req).read()


def parse_cd(data):
    """Central directory -> {name: (local_header_off, csize, usize, method)}."""
    entries = {}
    off = 0
    while off < len(data) - 46:
        if data[off : off + 4] != b"PK\x01\x02":
            break
        (sig, ver, ver2, flags, method, mt, md, crc, csize, usize, nlen, elen, clen,
         disk, iattr, eattr, lho) = struct.unpack("<IHHHHHHIIIHHHHHII", data[off : off + 46])
        name = data[off + 46 : off + 46 + nlen].decode("utf-8", "replace")
        extra = data[off + 46 + nlen : off + 46 + nlen + elen]
        if any(v == 0xFFFFFFFF for v in (usize, csize, lho)):
            eo = 0
            while eo + 4 <= len(extra):
                tag, sz = struct.unpack("<HH", extra[eo : eo + 4])
                body = extra[eo + 4 : eo + 4 + sz]
                if tag == 0x0001:
                    bo = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", body[bo : bo + 8])[0]; bo += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", body[bo : bo + 8])[0]; bo += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", body[bo : bo + 8])[0]
                    break
                eo += 4 + sz
        entries[name] = (lho, csize, usize, method)
        off += 46 + nlen + elen + clen
    return entries


def data_offset(entries, name):
    """Resolve a member's data offset from its local header (1 range GET)."""
    lho, _, _, _ = entries[name]
    hdr = http_range(lho, lho + 30)
    assert hdr[:4] == b"PK\x03\x04", f"bad local header for {name}"
    lnlen, lelen = struct.unpack("<HH", hdr[26:30])
    return lho + 30 + lnlen + lelen


def fetch_file(entries, name):
    dstart = data_offset(entries, name)
    _, csize, usize, method = entries[name]
    raw = http_range(dstart, dstart + csize - 1) if csize else b""
    if len(raw) < csize:
        raw = http_range(dstart, dstart + csize + 64)[:csize]
    return zlib.decompress(raw, -15) if method == 8 else raw[:usize]


def main():
    budget_mb = 0.0
    out = "tests/argon_cover_manifest.tsv"
    args = sys.argv[1:]
    while args:
        a = args.pop(0)
        if a == "--budget-mb":
            budget_mb = float(args.pop(0))
        elif a == "--out":
            out = args.pop(0)
        else:
            sys.exit(f"unknown arg {a}")

    size_resp = urllib.request.urlopen(
        urllib.request.Request(URL, method="HEAD")
    )
    total = int(size_resp.headers["Content-Length"])
    tail = http_range(total - TAIL, total - 1)

    eocd = tail.rfind(b"PK\x05\x06")
    assert eocd >= 0
    loc = eocd - 20
    assert tail[loc : loc + 4] == b"PK\x06\x07"
    eocd64_off = struct.unpack("<Q", tail[loc + 8 : loc + 16])[0]
    z64 = tail[eocd64_off - (total - TAIL) :]
    cd_size, cd_off = struct.unpack("<QQ", z64[40:56])
    print(f"zip: {total} bytes, cd {cd_size} @ {cd_off}", file=sys.stderr)
    entries = parse_cd(http_range(cd_off, cd_off + cd_size - 1))
    print(f"entries: {len(entries)}", file=sys.stderr)

    item2streams = defaultdict(set)
    for prof in ("profile0", "profile1", "profile2"):
        xml = fetch_file(entries, ROOT + f"prof_av1_{prof}.xml").decode("utf-8", "replace")
        for m in re.finditer(r"<testPoint>(.*?)</testPoint>", xml, re.S):
            blk = m.group(1)
            nm = re.search(r"<name>(.*?)</name>", blk, re.S)
            name = f"{prof}::{nm.group(1)[:80]}" if nm else prof + "::?"
            for sm in re.finditer(
                r"<(zero|nonzero|min|max)CoveringStream>(.*?)</\1CoveringStream>", blk
            ):
                s = sm.group(2).strip()
                if s.startswith("./"):
                    s = s[2:]
                item2streams[name].add(s)
    print(f"testPoints: {len(item2streams)}", file=sys.stderr)

    s2i = defaultdict(set)
    for i, ss in item2streams.items():
        for s in ss:
            s2i[s].add(i)
    sizes = {s: entries[ROOT + s][1] for s in s2i}

    # *_large_scale_tile streams are light-field tile-list content: their
    # sidecar MD5s are produced by aomdec's lightfield_tile_list_decoder
    # (decodes a SUBSET of tiles), a decode path rav1d-safe does not
    # implement. They can never match a normal full decode — exclude them.
    rem = {s: i for s, i in s2i.items() if "large_scale_tile" not in s}
    rem_i = set()
    for items in rem.values():
        rem_i |= items
    rem_i &= set(item2streams)

    picked = []
    spent = 0
    while rem_i:
        best, bk = None, None
        for s, items in rem.items():
            new = items & rem_i
            if not new:
                continue
            if budget_mb and spent + sizes.get(s, 0) > budget_mb * 1e6:
                continue
            k = (len(new), -sizes.get(s, 0))
            if bk is None or k > bk:
                best, bk = s, k
        if best is None:
            break
        picked.append(best)
        spent += sizes[best]
        rem_i -= rem[best]
        del rem[best]
    print(
        f"cover: {len(picked)} streams, {spent/1e6:.1f} MB, "
        f"uncovered {len(rem_i)}/{len(item2streams)}",
        file=sys.stderr,
    )

    rows = []
    for rel in sorted(picked):
        d_off = data_offset(entries, ROOT + rel)
        _, csize, usize, method = entries[ROOT + rel]
        name = rel.split("/")[-1].rsplit(".obu", 1)[0]
        prof_dir = rel.split("/")[0]
        md5s = []
        for kind in ("md5_no_film_grain", "md5_ref"):
            side = f"{ROOT}{prof_dir}/{kind}/{name}.md5"
            if side in entries:
                md5s.append(fetch_file(entries, side).split()[0].decode())
            else:
                md5s.append("-")
        rows.append((rel, d_off, csize, usize, method, md5s[0], md5s[1]))

    with open(out, "w") as f:
        f.write(
            "# relpath\tdata_off\tcsize\tusize\tmethod\tmd5_no_film_grain\tmd5_ref\n"
        )
        f.write(f"# source: {URL}\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {out}: {len(rows)} streams", file=sys.stderr)


if __name__ == "__main__":
    main()
