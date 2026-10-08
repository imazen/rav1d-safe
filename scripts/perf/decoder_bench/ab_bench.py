#!/usr/bin/env python3
"""Interleave two profile_ivf binaries using the fair_bench.py timing protocol.

Use identical binaries for an A-versus-A control. Decoding must succeed and
every timed pass must return the same nonzero frame count. Output is JSON,
including all observations and executable/stream hashes. Timing is inside
profile_ivf, with one untimed warm-up per invocation.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(binary, stream, threads, args):
    cpus = (str(args.first_cpu) if threads == 1 else
            f"{args.first_cpu}-{args.first_cpu + 2 * threads - 1}")
    env = dict(os.environ, RAV1D_THREADS=str(threads),
               RAV1D_REPS=str(args.passes), RAV1D_FRAME_DELAY=str(args.delay),
               RAV1D_LEVEL="native", RAV1D_INLOOP="all")
    # Inherited diagnostic switches can invalidate a nominally identical A/B.
    for switch in ("RAV1D_ABLATE", "RAV1D_PPROF"):
        if switch in env:
            raise RuntimeError(f"unset {switch} before benchmarking")
    proc = subprocess.run(["taskset", "-c", cpus, binary, stream, "1"],
                          text=True, capture_output=True, env=env, check=False)
    if proc.returncode:
        raise RuntimeError(f"{binary} exited {proc.returncode}:\n{proc.stderr}")
    if "decode error" in proc.stderr.lower() or "flush error" in proc.stderr.lower():
        raise RuntimeError(f"decoder reported an error:\n{proc.stderr}")
    rows = [line.split("\t") for line in proc.stdout.splitlines()
            if line.startswith("RESULT\t")]
    if len(rows) != args.passes:
        raise RuntimeError(f"expected {args.passes} timing rows, got {len(rows)}")
    frames = [int(row[4]) for row in rows]
    if len(set(frames)) != 1 or frames[0] <= 0:
        raise RuntimeError(f"invalid frame counts: {frames}")
    samples = [float(row[5]) for row in rows]
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise RuntimeError(f"invalid timing samples: {samples}")
    return samples, frames[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--before-rev", required=True)
    parser.add_argument("--after-rev", required=True)
    parser.add_argument("--stream", action="append", required=True)
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--first-cpu", type=int, default=8)
    parser.add_argument("--delay", type=int, default=1)
    args = parser.parse_args()
    if args.rounds < 1 or args.passes < 1:
        parser.error("rounds and passes must be positive")
    threads = [int(value) for value in args.threads.split(",")]
    if not threads or min(threads) < 1:
        parser.error("threads must be positive")
    report = {"protocol": "fresh decoder, warm-up, internal ms/frame, alternating A/B",
              "before": {"revision": args.before_rev, "sha256": digest(args.before)},
              "after": {"revision": args.after_rev, "sha256": digest(args.after)},
              "rounds": args.rounds, "passes": args.passes, "delay": args.delay,
              "first_cpu": args.first_cpu, "load_start": os.getloadavg(), "cases": []}
    # Persist each completed case immediately; a later failure must not turn
    # earlier observations into a summary that appears fully successful.
    print(json.dumps({"event": "start", "metadata": report}), flush=True)
    for stream in args.stream:
        for count in threads:
            values = [[], []]
            frames = set()
            for round_index in range(args.rounds):
                for arm in ([0, 1] if round_index % 2 == 0 else [1, 0]):
                    samples, decoded = run([args.before, args.after][arm], stream,
                                           count, args)
                    values[arm].extend(samples)
                    frames.add(decoded)
            if len(frames) != 1:
                raise RuntimeError(f"A/B frame count mismatch: {frames}")
            case = {"stream": Path(stream).name, "sha256": digest(stream),
                    "threads": count, "frames": frames.pop(),
                    "before_ms": values[0], "after_ms": values[1],
                    "before_median": statistics.median(values[0]),
                    "after_median": statistics.median(values[1]),
                    "before_min": min(values[0]), "after_min": min(values[1]),
                    "load": os.getloadavg()}
            case["ratio"] = case["after_median"] / case["before_median"]
            print(json.dumps({"event": "case", **case}), flush=True)
    print(json.dumps({"event": "complete"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1)
