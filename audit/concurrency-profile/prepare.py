#!/usr/bin/env python3
"""Extract the first packet without transcoding and record input provenance."""
import hashlib
import json
import pathlib
import struct
import sys

repo = pathlib.Path(__file__).resolve().parents[2]
out = pathlib.Path(sys.argv[1]).resolve()
(out / "inputs").mkdir(parents=True, exist_ok=True)
source = repo / "test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf"
b = source.read_bytes()
assert b[:4] == b"DKIF" and b[8:12] == b"AV01"
header = struct.unpack_from("<H", b, 6)[0]
n = struct.unpack_from("<I", b, header)[0]
first = out / "inputs/non_uniform_first.obu"
first.write_bytes(b[header+12:header+12+n])
paths = [source, first, repo / "tests/crash_vectors/tile_threading_cdef_lpf_race.obu", repo / "tests/crash_vectors/lr_sgr_10bpc_noisy_nocdef.obu"]
records = [dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
(out / "inputs.json").write_text(json.dumps(records, indent=2)+"\n")
