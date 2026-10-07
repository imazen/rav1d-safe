#!/usr/bin/env python3
"""tool_census.py DAV1D_BIN STREAM.ivf... : per-frame counts of AV1 coding-tool kernel calls.

Counts hits on dav1d's 8bpc asm kernels with gdb breakpoints (single thread, native dispatch).
Needs a symbolised dav1d (see docs/ASM_VS_DAV1D.md) and gdb.
"""
import re, subprocess, sys, tempfile
D = sys.argv[1]
syms = subprocess.run(['nm', D], capture_output=True, text=True).stdout.split('\n')
names = sorted({l.split()[2] for l in syms if len(l.split()) == 3 and l.split()[1] == 'T'})
GROUPS = {
    'warp':     r'^dav1d_warp_affine_8x8t?_8bpc_',
    'put/prep': r'^dav1d_(put|prep)_(6tap|8tap|bilin)[a-z_]*_8bpc_',
    'compound': r'^dav1d_(avg|w_avg|mask|w_mask_\d+)_8bpc_',
    'blend':    r'^dav1d_blend[_hv]?_8bpc_',
    'palette':  r'^dav1d_pal_idx_finish_',
    'cdef':     r'^dav1d_cdef_filter_\d+x\d+_8bpc_',
    'lr':       r'^dav1d_(wiener|sgr)_filter\d*_8bpc_',
}
for f in sys.argv[2:]:
    sel = {g: [n for n in names if re.match(p, n)] for g, p in GROUPS.items()}
    allsyms = [n for v in sel.values() for n in v]
    cmds = ''.join(f'break {n}\nignore $bpnum 1000000000\n' for n in allsyms)
    cmds += f'run\ninfo breakpoints\nquit\n'
    with tempfile.NamedTemporaryFile('w', suffix='.gdb', delete=False) as t:
        t.write('set pagination off\nset confirm off\n' + cmds); path = t.name
    out = subprocess.run(['gdb', '-batch', '-x', path, '--args', D, '-i', f, '--muxer', 'null', '--threads', '1'],
                         capture_output=True, text=True).stdout
    hits = {}
    for m in re.finditer(r'^\d+\s+breakpoint\s+keep y\s+\S+ in (\S+).*?\n(?:\s+breakpoint already hit (\d+) times?)?', out, re.M | re.S):
        pass
    cur = None
    for line in out.split('\n'):
        m = re.match(r'^\d+\s+breakpoint\s+keep\s+\w+\s+0x\w+\s+<(\w+)', line) or re.match(r'^\d+\s+breakpoint\s+keep\s+\w+\s+0x\w+\s+(?:\S+\s+)?in (\w+)', line)
        if m: cur = m.group(1)
        m = re.search(r'breakpoint already hit (\d+) time', line)
        if m and cur: hits[cur] = hits.get(cur, 0) + int(m.group(1))
    import struct
    d = open(f, 'rb').read(); nfr = struct.unpack('<I', d[24:28])[0]
    row = {g: sum(hits.get(n, 0) for n in v) for g, v in sel.items()}
    print(f"{f.split('/')[-1]:28s} " + ' '.join(f'{g}={row[g] / nfr:8.1f}' for g in GROUPS) + f'   (per frame, {nfr} frames)', flush=True)
