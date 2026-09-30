#!/usr/bin/env python3
"""Per-function self-Ir diff between two callgrind_annotate outputs.
Usage: cg_kernel_diff.py <safe.cgout> <asm.cgout>   # prints SAFE/ASM/DIFF table
"""
import re, subprocess, sys
from collections import defaultdict

def selfmap(f):
    out = subprocess.run(["callgrind_annotate","--inclusive=no",f],
                         capture_output=True,text=True).stdout
    m = defaultdict(int)
    for line in out.splitlines():
        g = re.match(r"^\s*([\d,]+)\s*\(\s*[\d.]+%\)\s+(.*)",line)
        if not g: continue
        cost = int(g.group(1).replace(",",""))
        func = g.group(2).split(":")[-1]
        func = re.sub(r"\s*\[.*$","",func)
        # normalize: keep last path-ish component, strip generics noise
        func = func.split("/")[-1] if "/" in func else func
        m[func] += cost
    return m

A = selfmap(sys.argv[1]); B = selfmap(sys.argv[2])
keys = set(A)|set(B)
rows = sorted(keys, key=lambda k:-(A.get(k,0)-B.get(k,0)))
print(f"{'SAFE':>12} {'ASM':>12} {'DIFF':>10}  function")
for k in rows[:60]:
    a,b = A.get(k,0), B.get(k,0)
    print(f"{a/1e6:11.1f}M {b/1e6:11.1f}M {(a-b)/1e6:9.1f}M  {k[:110]}")
