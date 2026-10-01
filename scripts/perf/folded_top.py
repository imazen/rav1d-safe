#!/usr/bin/env python3
"""Summarize a folded (stackcollapse-format) file.

Usage: folded_top.py <file.collapsed> [N]

Prints:
  - top-N leaf frames (self time; what sampling attributes directly)
  - top-N inclusive frames (any occurrence in stack)
Percentages are of total samples in the file.
"""
import re
import sys
from collections import Counter


def norm(sym: str) -> str:
    """Collapse demangled Rust symbols to a stable fn-ish key."""
    s = sym.split(", closure")[0]
    # strip generic args / hashes for grouping inline twins
    s = re.sub(r"\.\.\w{16}$", "", s)  # llvm suffix
    s = re.sub(r"::h[0-9a-f]{16}$", "", s)
    return s


def main() -> None:
    path = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    leaf: Counter[str] = Counter()
    incl: Counter[str] = Counter()
    total = 0
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            stack, _, count_s = line.rpartition(" ")
            count = int(count_s)
            total += count
            frames = stack.split(";")
            leaf[norm(frames[-1])] += count
            for fr in set(norm(x) for x in frames):
                incl[fr] += count
    print(f"total samples: {total}\n")
    print(f"== top {topn} leaf (self time) ==")
    for name, c in leaf.most_common(topn):
        print(f"{c:7d} {100.0 * c / total:5.1f}%  {name[:150]}")
    print(f"\n== top {topn} inclusive ==")
    for name, c in incl.most_common(topn):
        print(f"{c:7d} {100.0 * c / total:5.1f}%  {name[:150]}")


if __name__ == "__main__":
    main()
