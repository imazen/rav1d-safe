#!/usr/bin/env python3
"""Cross-check every displayed frame against installed dav1d; reject bad input."""
import gzip
import json
import pathlib
import re
import subprocess
import sys

repo = pathlib.Path(__file__).resolve().parents[2]
out = pathlib.Path(sys.argv[1]).resolve()
inputs = {
    "multi": repo / "test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf",
    "tiled_first": out / "inputs/non_uniform_first.obu",
    "tiled_stress": repo / "tests/crash_vectors/tile_threading_cdef_lpf_race.obu",
    "single_10b": repo / "tests/crash_vectors/lr_sgr_10bpc_noisy_nocdef.obu",
}
records = []
for name, path in inputs.items():
    dest = out / "oracle" / name
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["dav1d", "-i", str(path), "--threads", "1", "--framedelay", "1", "--filmgrain", "1", "--muxer", "framemd5", "-o", str(dest / "%n.md5")]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    (dest / "stderr.txt").write_text(p.stderr)
    expected = []
    for f in sorted(dest.glob("*.md5"), key=lambda p: int(p.stem)):
        expected.append(re.search(r"\b[0-9a-f]{32}\b", f.read_text()).group())
    raw = gzip.open(out / "baseline" / f"{name}-t1-i1.log.gz", "rt").read()
    actual = [s.split("\t")[-1] for s in raw.splitlines() if s.startswith("FRAME\t")]
    assert expected and expected == actual, (name, expected, actual)
    records.append(dict(input=name, command=cmd, md5=expected, matched=True))
    print(name, len(expected), "frames match dav1d", flush=True)

valid = inputs["multi"].read_bytes()
for name, data in [("partial_packet_header", valid + b"x"), ("truncated_packet", valid[:-1]), ("invalid_obu", b"\xff\xff\xff\xff")]:
    p = out / f"{name}.bad"
    p.write_bytes(data)
    cmd = [str(out / "bin/base"), str(p), "1", "1", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0, name
    records.append(dict(input=name, rejected=True, returncode=result.returncode, stderr=result.stderr))
(out / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
