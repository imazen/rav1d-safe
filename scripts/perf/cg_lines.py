#!/usr/bin/env python3
"""Callgrind line-level analyzer: self cost per source file / per function, with
an optional pass-difference (profile N=2 minus N=1 isolates ONE pass).

  cg_lines.py files  <out> [--minus <out1>] [--frames N] [--top K]
  cg_lines.py std    <out> [--minus <out1>] [--frames N] [--top K]

`files` ranks self cost by the instruction's source file. Callgrind attributes by
the LINE-TABLE file, so inlined `core::cmp`/`option`/`slice::index` code shows up
under the std file, not the Rust function that incurred it: that bucket is the
inlined bounds-check / Option / min-max tax. `std` ranks the functions that incur
it. See docs/ASM_VS_DAV1D.md for how this is used against a source-built dav1d.
"""
import argparse, collections, re, sys

def parse(path):
    """-> (cost[(file,line)], fcost[(fn,file)]), self Ir only (first event)."""
    files, fns = {}, {}
    cost, fcost = collections.Counter(), collections.Counter()
    cur_file = cur_fn = None
    pos = 0
    skip = False

    def name(tok, table):
        m = re.match(r'\((\d+)\)\s*(.*)', tok)
        if not m:
            return tok
        i, rest = m.group(1), m.group(2).strip()
        if rest:
            table[i] = rest
        return table.get(i, i)

    for l in open(path, errors='replace'):
        l = l.rstrip('\n')
        if not l:
            continue
        c = l[0]
        if c in '+-*' or c.isdigit():
            parts = l.split()
            p = parts[0]
            if p != '*':
                pos = pos + int(p) if p[0] in '+-' else int(p)
            if skip:  # the line after `calls=` is the inclusive call cost
                skip = False
                continue
            if len(parts) > 1:
                try:
                    v = int(parts[1])
                except ValueError:
                    continue
                cost[(cur_file, pos)] += v
                fcost[(cur_fn, cur_file)] += v
        elif l.startswith(('fl=', 'fi=', 'fe=')):
            cur_file = name(l[3:], files)
        elif l.startswith('fn='):
            cur_fn = name(l[3:], fns)
        elif l.startswith('calls='):
            skip = True
        elif l.startswith('cfn='):
            name(l[4:], fns)
        elif l.startswith(('cfl=', 'cfi=')):
            name(l[4:], files)
    return cost, fcost

def diff(a, b):
    return collections.Counter({k: a[k] - b.get(k, 0) for k in a if a[k] - b.get(k, 0) > 0})

def is_std(f):
    return bool(f) and ('/rustlib/' in f or '/.cargo/registry' in f)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['files', 'std'])
    ap.add_argument('out')
    ap.add_argument('--minus')
    ap.add_argument('--frames', type=int, default=1, help='divide by frames (reports kIr/frame)')
    ap.add_argument('--top', type=int, default=20)
    a = ap.parse_args()
    cost, fcost = parse(a.out)
    if a.minus:
        c1, f1 = parse(a.minus)
        cost, fcost = diff(cost, c1), diff(fcost, f1)
    k = a.frames * 1e3
    if a.mode == 'files':
        by = collections.Counter()
        for (f, _), v in cost.items():
            by[(f or '?').replace('/home/lilith/work/zen/rav1d-safe/', '')] += v
        print(f"total {sum(by.values())/a.frames/1e6:.2f} MIr/frame")
        for f, v in by.most_common(a.top):
            print(f"{v/k:10.1f} kIr/f  {f[:110]}")
    else:
        by, tot = collections.Counter(), collections.Counter()
        for (fn, f), v in fcost.items():
            tot[fn] += v
            if is_std(f):
                by[fn] += v
        print(f"inlined std/dep code: {sum(by.values())/k:.0f} kIr/frame")
        for fn, v in by.most_common(a.top):
            print(f"{v/k:10.1f} kIr/f of {tot[fn]/k:9.1f}  {fn[:100]}")

if __name__ == '__main__':
    sys.exit(main())
