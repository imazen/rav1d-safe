#!/usr/bin/env python3
"""Run sequential A/A controls and A/B comparisons in both decoder modes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-prefix", required=True)
    parser.add_argument("--after-prefix", required=True)
    parser.add_argument("--before-rev", required=True)
    parser.add_argument("--after-rev", required=True)
    parser.add_argument("--stream", required=True, action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--first-cpu", type=int, default=8)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--mode", action="append", choices=("safe", "untracked"),
                        help="select modes; default runs both")
    args = parser.parse_args()
    if args.mode and len(set(args.mode)) != len(args.mode):
        parser.error("--mode cannot repeat")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    common = ["--threads", args.threads, "--rounds", str(args.rounds),
              "--passes", str(args.passes), "--first-cpu", str(args.first_cpu),
              "--delay", str(args.delay)]
    for stream in args.stream:
        common.extend(["--stream", stream])
    script = str(Path(__file__).with_name("ab_bench.py"))
    for mode in args.mode or ("safe", "untracked"):
        before = f"{args.before_prefix}-{mode}/release/examples/profile_ivf"
        after = f"{args.after_prefix}-{mode}/release/examples/profile_ivf"
        for phase in ("aa", "ab"):
            path = output / f"{mode}-{phase}.jsonl"
            command = [sys.executable, script, "--before", before,
                       "--after", before if phase == "aa" else after,
                       "--before-rev", args.before_rev, "--after-rev",
                       args.before_rev if phase == "aa" else args.after_rev, *common]
            print(json.dumps({"event": "phase", "mode": mode, "phase": phase,
                              "output": str(path), "command": command}), flush=True)
            # Exclusive creation protects prior results from accidental reruns.
            with path.open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        text=True, check=False)
            print(path.read_text(), end="", flush=True)
            if result.returncode:
                raise RuntimeError(f"{mode}/{phase} failed; full output: {path}")
    print(json.dumps({"event": "campaign-complete"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"paired campaign failed: {error}", file=sys.stderr)
        raise SystemExit(1)
