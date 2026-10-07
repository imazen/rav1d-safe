#!/usr/bin/env python3
"""Separate per-decoder setup cost from steady-state cost per frame.

  steady_state.py [fair_bench.py args except --stream] --clip NAME=FILE.ivf:SHORT_FRAMES ...

For each clip it times a SHORT prefix (SHORT_FRAMES frames) and the full stream with the
fair_bench protocol (fresh decoder per pass, so creation and thread spawn count), then
solves  T(n) = setup + n * steady  from the two totals:
  steady = (T_full - T_short) / (N_full - N_short)      [ms/frame, setup removed]
  setup  = T_short - N_short * steady                   [ms per fresh decoder, incl. first-frame costs]
Needs scripts/perf/decoder_bench/ivf_prefix.py. Output: one row per decoder.
"""
import os, re, struct, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]; clips = []
while '--clip' in args:
    i = args.index('--clip'); clips.append(args[i + 1]); del args[i:i + 2]
tmp = tempfile.mkdtemp(); streams = []; meta = {}
for c in clips:
    name, rest = c.split('=', 1); f, short = rest.rsplit(':', 1)
    nfull = struct.unpack('<I', open(f, 'rb').read(28)[24:28])[0]
    pre = f'{tmp}/{name}_short.ivf'
    subprocess.run([sys.executable, f'{here}/ivf_prefix.py', f, pre, short], check=True, capture_output=True)
    streams += ['--stream', f'{name}={f}', '--stream', f'{name}_short={pre}']; meta[name] = (int(short), nfull)
proc = subprocess.run([sys.executable, f'{here}/fair_bench.py'] + args + streams, capture_output=True, text=True)
out = proc.stdout
if proc.returncode:
    sys.exit(f'fair_bench failed ({proc.returncode}):\n' + proc.stderr[-2000:] + '\n--- stdout so far ---\n' + out)
print(out.split('\n')[0])
res = {}; cur = None
for line in out.split('\n'):
    m = re.match(r'## (\S+)\s+threads=(\d+)\s+\((\d+) frames\)', line)
    if m: cur = (m.group(1), int(m.group(2)), int(m.group(3))); continue
    m = re.match(r'\s{2}(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x$', line)
    if m and cur: res[(cur, m.group(1).strip())] = float(m.group(2)) * cur[2]   # total ms for the pass
print(f"{'clip':14s} {'t':>2s} {'decoder':28s} {'full ms/fr':>10s} {'steady':>8s} {'setup ms':>9s} {'setup % of full pass':>20s}")
for (cur, dec), tot in sorted(res.items(), key=lambda kv: (kv[0][0][0].replace('_short', ''), kv[0][0][1], kv[0][1])):
    name, t, n = cur
    if name.endswith('_short'): continue
    sn, fn = meta[name]
    short = [v for (c, d), v in res.items() if c[0] == name + '_short' and c[1] == t and d == dec][0]
    steady = (tot - short) / (fn - sn); setup = short - sn * steady
    print(f"{name:14s} {t:2d} {dec:28s} {tot / fn:10.3f} {steady:8.3f} {setup:9.1f} {100 * setup / tot:19.1f}%", flush=True)
