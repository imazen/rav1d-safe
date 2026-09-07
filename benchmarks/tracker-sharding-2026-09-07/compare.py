"""Paired, rotated in-memory decode runs; fail on any frame/hash discrepancy."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--name', required=True)
p.add_argument('--arms', nargs='+', required=True, help='label=/absolute/binary')
p.add_argument('--inputs', nargs='+', default=['multi', 'first', 'stress', 'single10'])
p.add_argument('--cells', nargs='+', default=['1:1:0', '8:1:0', '24:1:0', '8:4:0', '1:1:24'])
p.add_argument('--reps', type=int, default=5)
p.add_argument('--passes', type=int, default=32, help='video passes; stills use 4x')
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
a.work_dir.mkdir(exist_ok=True, parents=True)
video = repo / 'test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf'
data = video.read_bytes()
assert data[:4] == b'DKIF' and int.from_bytes(data[6:8], 'little') == 32
assert len(data) >= 44
packet_size = int.from_bytes(data[32:36], 'little')
assert 0 < packet_size <= len(data) - 44
first = a.work_dir / 'non_uniform_first.obu'
first.write_bytes(data[44:44 + packet_size])
inputs = {
    'multi': video,
    'first': first,
    'stress': repo / 'tests/crash_vectors/tile_threading_cdef_lpf_race.obu',
    'single10': repo / 'tests/crash_vectors/lr_sgr_10bpc_noisy_nocdef.obu',
}
reference = json.loads((repo / 'benchmarks/upstream-2026-09-07/frame-reference.json').read_text())
arms = [arm.split('=', 1) for arm in a.arms]
rows = []
a.work_dir.mkdir(exist_ok=True, parents=True)
with (a.work_dir / (a.name + '.jsonl')).open('x') as log:
    for rep in range(a.reps):
        for name in a.inputs:
            for cell in a.cells:
                threads, instances, prime = map(int, cell.split(':'))
                order = arms[rep % len(arms):] + arms[:rep % len(arms)]
                if (rep + threads) % 2:
                    order = order[::-1]
                for arm, binary in order:
                    env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
                    if prime:
                        env['RAV1D_PRIME_THREADS'] = str(prime)
                    passes = a.passes if name == 'multi' else a.passes * 4
                    cmd = [binary, str(inputs[name]), str(threads), str(instances), str(passes), '1']
                    result = subprocess.run(cmd, cwd=repo, env=env, capture_output=True,
                                            text=True, timeout=60)
                    row = dict(rep=rep, input=name, threads=threads, instances=instances,
                               prime=prime, arm=arm, command=cmd, exit=result.returncode,
                               stdout=result.stdout, stderr=result.stderr)
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    assert result.returncode == 0, row
                    lines = result.stdout.splitlines()
                    frames = [line for line in lines if line.startswith('FRAME\t')]
                    assert frames == reference[name], row
                    timed = instances * passes * len(frames)
                    validated = instances * 2 + 1
                    assert f'VALIDATED\t{validated}\tlifetime_frames={timed + validated * len(frames)}\ttimed_frames={timed}' in lines, row
                    times = [line.split('\t') for line in lines if line.startswith('RESULT\t')]
                    assert len(times) == 1 and int(times[0][2]) == timed, row
                    row['ms'] = float(times[0][-1])
                    rows.append(row)
                print(f'{a.name}: round {rep + 1} {name} {cell} validated', flush=True)
summary = []
for name in a.inputs:
    for cell in a.cells:
        threads, instances, prime = map(int, cell.split(':'))
        medians = {}
        for arm, _ in arms:
            values = [r['ms'] for r in rows if
                      (r['input'], r['threads'], r['instances'], r['prime'], r['arm']) ==
                      (name, threads, instances, prime, arm)]
            medians[arm] = statistics.median(values)
        summary.append(dict(input=name, threads=threads, instances=instances, prime=prime,
                            medians=medians))
        print(name, cell, medians, flush=True)
(a.work_dir / (a.name + '-summary.json')).write_text(json.dumps(summary, indent=2) + '\n')
