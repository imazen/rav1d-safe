#!/usr/bin/env python3
"""Compare the __text layout of two Mach-O binaries, symbol by symbol.

Answers three questions that a wall-clock A/B cannot:

  1. Which symbols exist in only one arm (the new code)?
  2. Which symbols present in BOTH changed SIZE (a codegen change, not
     placement)?
  3. Which symbols kept their size but MOVED, and did the move cross a
     64-byte cache line or a 4 KiB page boundary (a placement change)?

Question 2 is the discriminator the rectangle round needed and did not take:
if the hot t=1 loop-filter symbols are byte-for-byte the same size in both
arms, a t=1 cost cannot be their codegen and must be placement or a
neighbouring symbol.

Usage: text_layout_diff.py <base-binary> <head-binary> [--hot REGEX] [--top N]
"""

import re
import subprocess
import sys


def symbols(path):
    """{name: (addr, size)} for every __text symbol, from `nm -n`.

    `nm` on macOS does not print sizes, so sizes are derived from the gap to
    the next symbol address. That is exact for a contiguous __text and is the
    same approximation `size`'s per-section total makes.
    """
    out = subprocess.run(
        ["nm", "-n", "-U", path], capture_output=True, text=True, check=True
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        addr, kind, name = parts[0], parts[1], parts[2]
        if kind not in ("t", "T"):
            continue
        try:
            rows.append((int(addr, 16), name))
        except ValueError:
            continue
    rows.sort()
    syms = {}
    for i, (addr, name) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else addr
        syms[norm(name)] = (addr, max(0, end - addr))
    return syms


# Two builds of the same crate with different feature sets (or different
# --target-dir) get different `-C metadata`, which lands in every v0 mangled
# symbol as a `Cs<base62>_` crate disambiguator. Matching symbols across arms
# requires normalising it away, or EVERY symbol reads as "only in head".
_DISAMB = re.compile(r"Cs[0-9A-Za-z]{10,}_")


def norm(name):
    return _DISAMB.sub("Cs_", name)


def demangle(name):
    return name


def main():
    base_path, head_path = sys.argv[1], sys.argv[2]
    hot = None
    top = 25
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--hot":
            hot = re.compile(args[i + 1])
            i += 2
        elif args[i] == "--top":
            top = int(args[i + 1])
            i += 2
        else:
            i += 1

    b, h = symbols(base_path), symbols(head_path)
    only_h = sorted(set(h) - set(b), key=lambda n: -h[n][1])
    only_b = sorted(set(b) - set(h), key=lambda n: -b[n][1])
    both = set(b) & set(h)

    resized = [(n, b[n][1], h[n][1]) for n in both if b[n][1] != h[n][1]]
    resized.sort(key=lambda r: -abs(r[2] - r[1]))
    moved = [n for n in both if b[n][0] != h[n][0]]

    print(f"# base {base_path}")
    print(f"# head {head_path}")
    print(f"symbols_base\t{len(b)}")
    print(f"symbols_head\t{len(h)}")
    print(f"text_base\t{sum(s for _, s in b.values())}")
    print(f"text_head\t{sum(s for _, s in h.values())}")
    print(f"only_in_head\t{len(only_h)}\tbytes\t{sum(h[n][1] for n in only_h)}")
    print(f"only_in_base\t{len(only_b)}\tbytes\t{sum(b[n][1] for n in only_b)}")
    print(f"resized_in_both\t{len(resized)}\tnet_bytes\t{sum(y - x for _, x, y in resized)}")
    print(f"moved_same_size\t{len(moved) - len(resized)}")

    print("\n## only in head (the new code), top %d by size" % top)
    for n in only_h[:top]:
        print(f"{h[n][1]:8d}\t{demangle(n)}")

    print("\n## only in base, top %d by size" % top)
    for n in only_b[:top]:
        print(f"{b[n][1]:8d}\t{demangle(n)}")

    print("\n## present in BOTH but RESIZED, top %d by |delta|" % top)
    for n, x, y in resized[:top]:
        print(f"{y - x:+8d}\t{x:8d} -> {y:8d}\t{demangle(n)}")

    if hot is not None:
        print("\n## HOT symbols (regex): size and placement")
        print("delta_size\tbase_size\thead_size\tbase_addr\thead_addr\tline_off_base\tline_off_head\tcrosses\tname")
        hits = sorted(n for n in both if hot.search(n))
        for n in hits:
            ba, bs = b[n]
            ha, hs = h[n]
            # 64-byte cache line offset of the entry point, and whether the
            # symbol's body spans a different NUMBER of lines / pages.
            lb, lh = ba % 64, ha % 64
            lines_b = (ba % 64 + bs + 63) // 64
            lines_h = (ha % 64 + hs + 63) // 64
            pages_b = (ba % 4096 + bs + 4095) // 4096
            pages_h = (ha % 4096 + hs + 4095) // 4096
            crosses = []
            if lines_b != lines_h:
                crosses.append(f"lines {lines_b}->{lines_h}")
            if pages_b != pages_h:
                crosses.append(f"pages {pages_b}->{pages_h}")
            print(
                f"{hs - bs:+d}\t{bs}\t{hs}\t0x{ba:x}\t0x{ha:x}\t{lb}\t{lh}\t"
                f"{','.join(crosses) or '-'}\t{demangle(n)}"
            )
        only_hot_h = [n for n in only_h if hot.search(n)]
        only_hot_b = [n for n in only_b if hot.search(n)]
        if only_hot_h:
            print("\n## HOT-matching symbols only in HEAD")
            for n in only_hot_h:
                print(f"{h[n][1]:8d}\t{demangle(n)}")
        if only_hot_b:
            print("\n## HOT-matching symbols only in BASE")
            for n in only_hot_b:
                print(f"{b[n][1]:8d}\t{demangle(n)}")


if __name__ == "__main__":
    main()
