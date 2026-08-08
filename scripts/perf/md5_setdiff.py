#!/usr/bin/env python3
"""Set-diff two md5_inventory TSVs BY NAME, with the hash inside the value.

A count comparison hides a change that repairs five vectors and breaks five.
This keys on (group, name) and compares (status, actual-md5), so a swap shows
up as N entries differing rather than as "766 == 766".

Usage: md5_setdiff.py <baseline.tsv[.zst]> <head.tsv>
Exit 0 only when the two agree on every key.
"""

import subprocess
import sys


def load(path):
    if path.endswith(".zst"):
        text = subprocess.run(["zstd", "-dc", path], capture_output=True,
                              check=True).stdout.decode()
    else:
        text = open(path).read()
    rows = {}
    for i, line in enumerate(text.splitlines()):
        f = line.split("\t")
        if i == 0 and f[0] == "group":
            continue
        if len(f) < 5:
            continue
        rows[(f[0], f[1])] = (f[2], f[4])
    return rows


def main():
    base, head = load(sys.argv[1]), load(sys.argv[2])
    only_b = sorted(set(base) - set(head))
    only_h = sorted(set(head) - set(base))
    diff = sorted(k for k in set(base) & set(head) if base[k] != head[k])
    print(f"baseline entries : {len(base)}  (PASS {sum(1 for v in base.values() if v[0]=='PASS')})")
    print(f"head entries     : {len(head)}  (PASS {sum(1 for v in head.values() if v[0]=='PASS')})")
    print(f"only in baseline : {len(only_b)}")
    print(f"only in head     : {len(only_h)}")
    print(f"differing        : {len(diff)}   <- status OR md5, not a count")
    for k in only_b[:20]:
        print(f"  -only-baseline {k} {base[k]}")
    for k in only_h[:20]:
        print(f"  +only-head     {k} {head[k]}")
    for k in diff[:20]:
        print(f"  !differs       {k} base={base[k]} head={head[k]}")
    ok = not (only_b or only_h or diff)
    print("SETDIFF: CLEAN" if ok else "SETDIFF: DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
