#!/usr/bin/env python3
"""Report completed paired-mode measurements alongside their A/A controls."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from ab_bench import digest


def phase(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if (not rows or rows[0].get("event") != "start"
            or rows[-1].get("event") != "complete"
            or any(row.get("event") != "case" for row in rows[1:-1])):
        raise ValueError(f"incomplete phase: {path}")
    meta = rows[0]["metadata"]
    count = meta["rounds"] * meta["passes"]
    if meta["rounds"] < 1 or meta["passes"] < 1:
        raise ValueError(f"nonpositive sample count: {path}")
    cases = {}
    for row in rows[1:-1]:
        key = (row["stream"], row["threads"])
        if key in cases or row["frames"] <= 0:
            raise ValueError(f"duplicate case or nonpositive frame count: {path} {key}")
        for arm in ("before", "after"):
            samples = row[f"{arm}_ms"]
            if (len(samples) != count
                    or any(not math.isfinite(x) or x <= 0 for x in samples)):
                raise ValueError(f"invalid samples: {path} {key} {arm}")
            for stat, value in (("median", statistics.median(samples)),
                                ("min", min(samples))):
                if not math.isclose(value, row[f"{arm}_{stat}"], rel_tol=1e-12):
                    raise ValueError(f"stored statistic differs from samples: {path} {key}")
        cases[key] = row
    if not cases:
        raise ValueError(f"empty phase: {path}")
    return meta, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    comparisons = []
    inputs = {}
    common_cases = None
    for mode in ("safe", "untracked"):
        paths = [args.directory / f"{mode}-{p}.jsonl" for p in ("aa", "ab")]
        (aa_meta, aa), (ab_meta, ab) = [phase(path) for path in paths]
        if (aa_meta["before"] != aa_meta["after"]
                or aa_meta["before"] != ab_meta["before"]):
            raise ValueError(f"A/A does not control the A/B baseline: {mode}")
        for setting in ("rounds", "passes", "delay", "first_cpu", "protocol"):
            if aa_meta[setting] != ab_meta[setting]:
                raise ValueError(f"unmatched {setting}: {mode}")
        if set(aa) != set(ab) or (common_cases is not None and set(ab) != common_cases):
            raise ValueError(f"unmatched cases: {mode}")
        common_cases = set(ab)
        inputs.update({path.name: digest(path) for path in paths})
        for key, case in ab.items():
            control = aa[key]
            if (case["sha256"] != control["sha256"]
                    or case["frames"] != control["frames"]):
                raise ValueError(f"unmatched stream or frames: {mode} {key}")
            comparisons.append({
                "mode": mode, "stream": key[0], "threads": key[1],
                "frames": case["frames"], "samples_per_arm": ab_meta["rounds"] * ab_meta["passes"],
                "before_median_ms": case["before_median"],
                "after_median_ms": case["after_median"],
                "aa_median_ratio": control["after_median"] / control["before_median"],
                "aa_minimum_ratio": control["after_min"] / control["before_min"],
                "ab_median_ratio": case["after_median"] / case["before_median"],
                "ab_minimum_ratio": case["after_min"] / case["before_min"],
                # Three timed passes in one process share scheduling and caches;
                # retain process-round pairs instead of treating them as independent.
                "aa_round_median_ratios": round_ratios(control, aa_meta["passes"]),
                "ab_round_median_ratios": round_ratios(case, ab_meta["passes"]),
            })
    print(json.dumps({"inputs": inputs, "comparisons": comparisons}, indent=2))


def round_ratios(case, passes):
    before, after = case["before_ms"], case["after_ms"]
    return [statistics.median(after[i:i + passes]) / statistics.median(before[i:i + passes])
            for i in range(0, len(before), passes)]


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"paired report failed: {error}", file=sys.stderr)
        raise SystemExit(1)
