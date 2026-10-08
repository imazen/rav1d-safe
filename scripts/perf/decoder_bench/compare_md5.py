#!/usr/bin/env python3
"""Check whole-clip benchmark output against dav1d before timing binaries."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from ab_bench import digest


def checked(command, env):
    result = subprocess.run(command, env=env, text=True, capture_output=True,
                            check=False, timeout=120)
    if result.returncode or re.search(r"(?:decode|flush) error", result.stderr, re.I):
        raise RuntimeError(f"{command} failed ({result.returncode}): {result.stderr}")
    md5 = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{32}", md5):
        raise RuntimeError(f"invalid MD5 output from {command}: {md5!r}")
    return md5, result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, action="append")
    parser.add_argument("--stream", required=True, action="append")
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--dav1d", default="dav1d")
    parser.add_argument("--filmgrain", action="store_true",
                        help="apply grain in both decoders (matches profile_ivf's default)")
    args = parser.parse_args()
    threads = [int(value) for value in args.threads.split(",")]
    if not threads or min(threads) < 1 or args.delay < 1:
        parser.error("threads and delay must be positive")
    env = dict(os.environ, RAV1D_FRAME_DELAY=str(args.delay))
    for switch in ("RAV1D_ABLATE", "RAV1D_PPROF"):
        if switch in env:
            parser.error(f"unset {switch} before checking benchmark output")
    binaries = {str(Path(binary).resolve()): digest(binary) for binary in args.binary}
    version = subprocess.run([args.dav1d, "--version"], text=True,
                             capture_output=True, check=True)
    print(json.dumps({"event": "start", "binaries": binaries,
                      "dav1d": (version.stdout + version.stderr).strip(),
                      "threads": threads, "delay": args.delay,
                      "filmgrain": args.filmgrain}), flush=True)
    for stream in args.stream:
        expected, _ = checked([args.dav1d, "-i", stream, "--muxer", "md5",
                               "-o", "-", "-q", "--filmgrain", str(int(args.filmgrain)),
                               "--threads", "1", "--framedelay", str(args.delay)], env)
        for binary in binaries:
            for count in threads:
                grain = ["--filmgrain"] if args.filmgrain else []
                actual, diagnostic = checked([binary, *grain, "--threads", str(count),
                                              "--delay", str(args.delay), stream], env)
                frames = re.findall(r"^Frames: (\d+)$", diagnostic, re.M)
                if len(frames) != 1 or int(frames[0]) <= 0:
                    raise RuntimeError(f"invalid frame count: {diagnostic}")
                if actual != expected:
                    raise RuntimeError(f"MD5 mismatch: {binary} {stream} t={count}: "
                                       f"{actual} != {expected}")
                print(json.dumps({"event": "case", "binary": binary,
                                  "stream": Path(stream).name, "sha256": digest(stream),
                                  "threads": count, "frames": int(frames[0]),
                                  "md5": actual}), flush=True)
    print(json.dumps({"event": "complete"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"benchmark output check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
