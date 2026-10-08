#!/usr/bin/env python3
"""Verify an explicit decode_md5 frame limit stops before a malformed packet."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--profile-binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    name = "i8_420_p8_q63_noise_64x64"
    source = ROOT / "tests/gen_vectors" / (name + ".obu")
    packet = source.read_bytes()
    rows = [line.split("\t") for line in
            (ROOT / "tests/gen_vectors_manifest.tsv").read_text().splitlines()
            if line.startswith(name + "\t")]
    assert len(rows) == 1
    expected = rows[0][2]
    scratch = Path.home() / "tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        vector = Path(directory) / "valid-frame-then-malformed-packet.ivf"
        header = struct.pack("<4sHH4sHHIIII", b"DKIF", 0, 32, b"AV01",
                             64, 64, 1, 1, 2, 0)
        # The second IVF packet has a forbidden-bit OBU header. The container
        # remains valid, so only submitting that packet reaches the error.
        vector.write_bytes(header + struct.pack("<IQ", len(packet), 0) + packet
                           + struct.pack("<IQ", 1, 1) + b"\x80")

        def run(limit):
            command = [str(binary), "--threads", "1", "--delay", "1"]
            if limit is not None:
                command += ["--limit", str(limit)]
            command += [str(vector)]
            result = subprocess.run(command, text=True, capture_output=True,
                                    timeout=120)
            print(json.dumps({"limit": limit, "returncode": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr}),
                  flush=True)
            return result

        control = run(None)
        assert "Decode error:" in control.stderr, "malformed packet must reach decoder"
        for limit, digest in [(0, hashlib.md5(b"").hexdigest()), (1, expected)]:
            result = run(limit)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == digest, result.stdout
            assert f"Frames: {limit}" in result.stderr, result.stderr
            assert "Decode error" not in result.stderr, "packet after limit was decoded"
            assert "Flush error" not in result.stderr, result.stderr
        if args.profile_binary is not None:
            profile = args.profile_binary.resolve(strict=True)
            env = dict(os.environ, RAV1D_THREADS="1", RAV1D_FRAME_DELAY="1",
                       RAV1D_REPS="2", RAV1D_LEVEL="native", RAV1D_INLOOP="all")
            for switch in ("RAV1D_ABLATE", "RAV1D_PPROF"):
                assert switch not in env, f"unset {switch} before benchmark checks"
            bad = subprocess.run([str(profile), str(vector), "1"], env=env,
                                 text=True, capture_output=True, timeout=120)
            print(json.dumps({"profile_control": "malformed", "returncode": bad.returncode,
                              "stdout": bad.stdout, "stderr": bad.stderr}), flush=True)
            assert bad.returncode != 0, "malformed benchmark must fail"
            assert "decode failed during benchmark" in bad.stderr, bad.stderr
            assert not any(line.startswith("RESULT") for line in bad.stdout.splitlines())
            valid = Path(directory) / "valid-frame.ivf"
            header = struct.pack("<4sHH4sHHIIII", b"DKIF", 0, 32, b"AV01",
                                 64, 64, 1, 1, 1, 0)
            valid.write_bytes(header + struct.pack("<IQ", len(packet), 0) + packet)
            good = subprocess.run([str(profile), str(valid), "1"], env=env,
                                  text=True, capture_output=True, timeout=120)
            print(json.dumps({"profile_control": "valid", "returncode": good.returncode,
                              "stdout": good.stdout, "stderr": good.stderr}), flush=True)
            assert good.returncode == 0, good.stderr
            results = [line.split("\t") for line in good.stdout.splitlines()
                       if line.startswith("RESULT\t")]
            assert len(results) == 2, good.stdout
            assert all(row[4] == "1" for row in results), results
    print("PASS: malformed control and both frame-limit gates", flush=True)


if __name__ == "__main__":
    main()
