#!/usr/bin/env python3
"""Self time per symbol from a macOS `sample` call graph.

`sample` prints a tree whose depth is encoded purely by prefix WIDTH (two
columns per level, filled with the cycling markers '+ ! : |'), so depth is
(prefix_len - thread_prefix_len)/2.  A node's SELF samples are its own count
minus the sum of its immediate children's counts.

Symbols `sample` refused to name (<deduplicated_symbol> from identical-code
folding, and ???) are resolved here against the image's own local symbol table
via `nm -n`, using the address `sample` recorded.
"""
import re, sys, subprocess, bisect, collections, os, glob

samplefile = sys.argv[1]
demangle = '--demangle' in sys.argv
lines = open(samplefile, errors='replace').read().split('\n')

images = {}
in_bi = False
for ln in lines:
    if ln.startswith('Binary Images:'): in_bi = True; continue
    if in_bi:
        m = re.match(r'\s*(0x[0-9a-f]+)\s*-\s*(0x[0-9a-f]+)\s+\+?(\S+)\s+\(.*?\)\s+<[^>]*>\s+(\S+)', ln)
        if m:
            path = m.group(4)
            if '*' in path:
                cands = glob.glob(path)
                path = cands[0] if cands else None
            images[m.group(3)] = (path, int(m.group(1), 16))

symcache = {}
def symtab(img):
    if img in symcache: return symcache[img]
    path, load = images.get(img, (None, 0))
    ents = []
    if path and os.path.exists(path):
        out = subprocess.run(['nm','-n','-arch','arm64',path],
                             capture_output=True, text=True).stdout
        for l in out.split('\n'):
            p = l.split()
            if len(p) >= 3 and re.fullmatch(r'[0-9a-f]+', p[0]):
                ents.append((int(p[0],16), p[2]))
    ents.sort()
    symcache[img] = (ents, [e[0] for e in ents], load)
    return symcache[img]

def resolve(img, addr):
    ents, keys, load = symtab(img)
    if not ents: return None
    i = bisect.bisect_right(keys, addr - load) - 1
    return ents[i][1] if i >= 0 else None

start = next(i for i,l in enumerate(lines) if l.startswith('Call graph:'))
try:
    end = next(i for i,l in enumerate(lines) if l.startswith('Total number in stack'))
except StopIteration:
    end = next(i for i,l in enumerate(lines) if l.startswith('Binary Images'))

row = re.compile(r'^([ +!:|]*?)(\d+) (.*)$')
entries = []      # (depth, count, name, image, tail)
base = None
for ln in lines[start+1:end]:
    if not ln.strip(): continue
    m = row.match(ln)
    if not m: continue
    plen, cnt, rest = len(m.group(1)), int(m.group(2)), m.group(3)
    if 'Thread_' in rest:
        base = plen; entries.append((0, cnt, '<thread>', '', '')); continue
    if base is None: continue
    depth = (plen - base)//2
    mi = re.match(r'^(.*?)\s+\(in ([^)]+)\)(.*)$', rest)
    if mi: name, img, tail = mi.group(1).strip(), mi.group(2), mi.group(3)
    else:  name, img, tail = rest.strip(), '', rest
    entries.append((depth, cnt, name, img, tail))

childsum = collections.defaultdict(int)
for i,(d,c,_,_,_) in enumerate(entries):
    for j in range(i+1, len(entries)):
        d2 = entries[j][0]
        if d2 <= d: break
        if d2 == d+1: childsum[i] += entries[j][1]

self_s = collections.Counter(); total = 0
for i,(d,c,name,img,tail) in enumerate(entries):
    if name == '<thread>': continue
    s = c - childsum[i]
    if s <= 0: continue
    if name.startswith('<deduplicated') or name.startswith('???'):
        a = re.findall(r'0x[0-9a-f]+', tail)
        imgn = img
        if not imgn:
            mm = re.search(r'\(in ([^)]+)\)', tail); imgn = mm.group(1) if mm else ''
        r = resolve(imgn, int(a[0],16)) if a and imgn else None
        name = r or f'<unresolved:{imgn}>'
    self_s[name] += s; total += s

names = list(self_s)
if demangle:
    out = subprocess.run(['rustfilt'], input='\n'.join(names), capture_output=True, text=True)
    if out.returncode == 0:
        dm = dict(zip(names, out.stdout.split('\n')))
        merged = collections.Counter()
        for n,v in self_s.items(): merged[dm.get(n,n)] += v
        self_s = merged

print(f"# total leaf samples: {total}   (file: {samplefile})")
for n,v in self_s.most_common(int(os.environ.get('TOPN','60'))):
    print(f"{v}\t{100.0*v/total:.2f}%\t{n}")
