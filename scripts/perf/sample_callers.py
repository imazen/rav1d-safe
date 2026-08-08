#!/usr/bin/env python3
"""Attribute a *subsystem's* samples to the callers that entered it.

`sample_selftime.py` answers "which leaf burns the time"; this answers "who
asked for it".  Given one or more regexes naming a subsystem (e.g. the borrow
tracker, or the DisjointMut slice helpers), it walks the call graph and, for
every maximal subtree rooted at a matching frame, credits that subtree's whole
sample count to the nearest ancestor that does NOT match.  Nested matches are
folded into the outermost one, so nothing is double counted.

Usage:  sample_callers.py <sample.txt> <regex> [<regex> ...]
        sample_callers.py <sample.txt> --self <regex> [...]

`--self` credits only the SELF samples of matching frames rather than whole
subtrees; use it when the subsystem calls back out into code you also want to
see attributed separately.
"""
import re
import subprocess
import sys
import collections

args = [a for a in sys.argv[1:] if a != '--self']
self_only = '--self' in sys.argv[1:]
samplefile = args[0]
pats = [re.compile(p) for p in args[1:]]
if not pats:
    sys.exit("need at least one regex")

lines = open(samplefile, errors='replace').read().split('\n')
start = next(i for i, l in enumerate(lines) if l.startswith('Call graph:'))
try:
    end = next(i for i, l in enumerate(lines) if l.startswith('Total number in stack'))
except StopIteration:
    end = next(i for i, l in enumerate(lines) if l.startswith('Binary Images'))

row = re.compile(r'^([ +!:|]*?)(\d+) (.*)$')
ent = []
base = None
for ln in lines[start + 1:end]:
    m = row.match(ln)
    if not m:
        continue
    plen, c, rest = len(m.group(1)), int(m.group(2)), m.group(3)
    if 'Thread_' in rest:
        base = plen
        ent.append((0, c, '<thread>'))
        continue
    if base is None:
        continue
    nm = re.match(r'^(.*?)\s+\(in ', rest)
    ent.append(((plen - base) // 2, c, (nm.group(1) if nm else rest).strip()))

names = sorted({n for _, _, n in ent})
dem = dict(zip(names, subprocess.run(['rustfilt'], input='\n'.join(names),
                                     capture_output=True, text=True).stdout.split('\n')))
D = [(d, c, dem.get(n, n)) for d, c, n in ent]
total = sum(c for d, c, n in D if d == 0)


def short(n):
    n = n.replace('rav1d_safe::src::', '').replace('rav1d_disjoint_mut::', '')
    n = re.sub(r'::<[^>]*BitDepth(8|16)[^>]*>', '', n)
    return n[:100]


def matches(n):
    return any(p.search(n) for p in pats)


# ancestor stack of (depth, name); walk in order, tracking the innermost
# non-matching ancestor for each frame.
agg = collections.Counter()
grand = 0
stack = []          # list of (depth, name, matching)
for d, c, n in D:
    while stack and stack[-1][0] >= d:
        stack.pop()
    m = matches(n)
    inside = any(s[2] for s in stack)
    if m and not inside:
        # outermost entry into the subsystem: find nearest non-matching ancestor
        caller = next((s[1] for s in reversed(stack) if not s[2]), '<root>')
        if self_only:
            kids = 0
            # self = own count - immediate children's counts (computed below)
            agg[(short(caller), short(n))] += 0  # placeholder, filled later
        else:
            agg[short(caller)] += c
            grand += c
    stack.append((d, n, m or inside))

if self_only:
    # recompute with self semantics: for every matching frame, self samples go
    # to the nearest non-matching ancestor.
    agg = collections.Counter()
    grand = 0
    stack = []
    for i, (d, c, n) in enumerate(D):
        while stack and stack[-1][0] >= d:
            stack.pop()
        if matches(n):
            kids = 0
            j = i + 1
            while j < len(D) and D[j][0] > d:
                if D[j][0] == d + 1:
                    kids += D[j][1]
                j += 1
            slf = c - kids
            caller = next((s[1] for s in reversed(stack) if not matches(s[1])), '<root>')
            agg[short(caller)] += slf
            grand += slf
        stack.append((d, n, False))

print(f"# subsystem {' | '.join(a for a in args[1:])}"
      f"   {'self' if self_only else 'inclusive'} {grand} ({100.0 * grand / total:.2f}% of {total})")
for k, v in agg.most_common(30):
    print(f"{v:7d}\t{100.0 * v / total:6.2f}%\t{k}")
