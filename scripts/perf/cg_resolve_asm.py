#!/usr/bin/env python3
"""Resolve raw callgrind fn= records (incl. bare 0xADDR nasm bodies) to the
nearest preceding dav1d_* symbol; prints per-function self-Ir table.
Usage: cg_resolve_asm.py <asm-build.cgout>  (run from repo root; needs nm)
"""
import re, subprocess, sys, bisect
from collections import defaultdict

nm = subprocess.run(["nm","-n","target/asm/release/examples/profile_avif"],
                    capture_output=True,text=True).stdout
syms = [(int(p[0],16),p[2]) for l in nm.splitlines()
        if len(p:=l.split())>=3 and p[1] in 'tT' and p[2].startswith('dav1d')]
addrs = [a for a,_ in syms]
def resolve(addr):
    i = bisect.bisect_right(addrs,addr)-1
    return syms[i][1] if i>=0 else None

per_fn = defaultdict(int)
cur = None
for line in open(sys.argv[1]):
    if line.startswith("fn="):
        cur = re.sub(r"^fn=\(\d+\)\s*","",line).strip()
    elif cur and re.match(r"^[\d+*\-]", line):
        cost = line.split()
        if len(cost)>=2 and cost[1].isdigit():
            per_fn[cur] += int(cost[1])

norm = defaultdict(int)
for k,v in per_fn.items():
    if m := re.match(r"^0x([0-9a-f]+)$", k):
        k = resolve(int(m.group(1),16)) or k
    if k.startswith("dav1d"):
        k = re.sub(r"(\.\w+)+$","",k)
    norm[k]+=v

dav = {k:v for k,v in norm.items() if k.startswith("dav1d_")}
print(f"dav1d fn total: {sum(dav.values())/1e6:.1f}M over {len(dav)} fns")
for k,v in sorted(dav.items(),key=lambda kv:-kv[1])[:35]:
    print(f"{v/1e6:9.1f}M  {k}")
