#!/usr/bin/env python3
"""Fair in-process decoder comparison: dav1d vs libgav1 vs rav1d-safe.

Protocol (identical for every contestant; see docs/DECODER_COMPARISON.md):
  * IVF parsed into memory once; a FRESH decoder per pass (creation and thread
    spawn counted for everyone); every picture drained and released; timing taken
    INSIDE the process around the pass only (no process startup, no file IO);
  * one untimed warm-up pass, then PASSES timed passes per process;
  * contestants interleaved in rotating order over ROUNDS rounds, pinned to fixed
    cores, frame counts asserted equal, median/min/max reported;
  * every contestant is also run capped at AVX2 (libgav1 cannot use AVX-512).

  fair_bench.py --bench-dec BIN --untracked BIN --default BIN \
                --stream photo4k=photo.ivf [--stream ...] [--threads 1,4]
BIN for rav1d is `target/release/examples/profile_ivf`. Env: ROUNDS (default 4),
PASSES (default 5), FIRST_CPU (default 2).
"""
import argparse, os, statistics, subprocess, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--bench-dec', required=True, help='built scripts/perf/decoder_bench/bench_dec.cc')
ap.add_argument('--untracked', required=True, help='profile_ivf built with --features untracked')
ap.add_argument('--default', required=True, help='profile_ivf built with default features')
ap.add_argument('--stream', action='append', required=True, metavar='NAME=FILE.ivf')
ap.add_argument('--threads', default='1,4')
ap.add_argument('--mode', choices=['tile', 'auto'], default='tile',
                help='tile: tile/post-filter threading only (frame delay 1); auto: each decoder default parallelism '
                     '(frame threading where available; rav1d sizes it from the frame size)')
a = ap.parse_args()
ROUNDS = int(os.environ.get('ROUNDS', '4')); PASSES = int(os.environ.get('PASSES', '5'))
FIRST = int(os.environ.get('FIRST_CPU', '2'))
OURS = {'untr': a.untracked, 'safe': a.default}
CONT = [('dav1d', 'native'), ('dav1d', 'avx2'), ('gav1', 'avx2'), ('untr', 'native'),
        ('untr', 'avx2'), ('safe', 'native'), ('safe', 'avx2')]
LABEL = {'dav1d': 'dav1d', 'gav1': 'libgav1', 'untr': 'rav1d untracked', 'safe': 'rav1d default'}

def cpus(t):
    return str(FIRST) if t == 1 else f'{FIRST}-{FIRST + 2 * t - 1}'

def run(c, f, t):
    pin = ['taskset', '-c', cpus(t)]
    kind, lvl = c
    env = dict(os.environ)
    if kind in ('dav1d', 'gav1'):
        cmd = pin + [a.bench_dec, kind, f, str(t), str(PASSES), '1' if (kind == 'dav1d' and lvl == 'avx2') else '0', a.mode]
    else:
        cmd = pin + [OURS[kind], f, '1']
        env.update(RAV1D_THREADS=str(t), RAV1D_REPS=str(PASSES), RAV1D_LEVEL='v3' if lvl == 'avx2' else 'native',
                   RAV1D_FRAME_DELAY='0' if a.mode == 'auto' else '1')
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if kind in ('dav1d', 'gav1'):
        rows = [l.split() for l in p.stdout.splitlines() if l.startswith('PASS')]
        return [float(r[1]) for r in rows], int(rows[0][2])
    rows = [l.split('\t') for l in p.stdout.splitlines() if l.startswith('RESULT')]
    return [float(r[5]) for r in rows], int(rows[0][4])

print(f"load {os.getloadavg()[0]:.2f}; mode={a.mode}; rounds={ROUNDS} passes/round={PASSES} (+1 warm-up each)", flush=True)
for spec in a.stream:
    name, f = spec.split('=', 1)
    for t in [int(x) for x in a.threads.split(',')]:
        res = {c: [] for c in CONT}; frames = {}
        for r in range(ROUNDS):
            for i in range(len(CONT)):
                c = CONT[(i + r) % len(CONT)]
                v, n = run(c, f, t); res[c] += v; frames[c] = n
        assert len(set(frames.values())) == 1, f'frame count mismatch {frames}'
        base = statistics.median(res[CONT[0]])
        print(f"\n## {name}  threads={t}  ({frames[CONT[0]]} frames)")
        print(f"  {'decoder':26s} {'median':>8s} {'min':>8s} {'max':>8s}  {'vs dav1d':>8s}")
        for c in CONT:
            xs = res[c]; m = statistics.median(xs)
            lab = LABEL[c[0]] + ('' if c[0] == 'gav1' else f' ({c[1]})')
            print(f"  {lab:26s} {m:8.3f} {min(xs):8.3f} {max(xs):8.3f}  {m / base:7.2f}x", flush=True)
