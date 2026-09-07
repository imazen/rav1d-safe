"""Compare verified releases/features using internal persistent-decoder timers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--reps', type=int, default=7)
a = p.parse_args()
repo, root = a.repo.resolve(), a.work_dir.resolve()
inputs = {
    'multi': repo/'test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf',
    'first': root/'non_uniform_first.obu',
    'stress': repo/'tests/crash_vectors/tile_threading_cdef_lpf_race.obu',
    'single10': repo/'tests/crash_vectors/lr_sgr_10bpc_noisy_nocdef.obu',
}
ivf = inputs['multi'].read_bytes()
pos = int.from_bytes(ivf[6:8], 'little')
size = int.from_bytes(ivf[pos:pos+4], 'little')
assert ivf[:4] == b'DKIF' and size and len(ivf) >= pos+12+size
inputs['first'].write_bytes(ivf[pos+12:pos+12+size])
oracle = json.loads((repo/'benchmarks/upstream-2026-09-07/frame-reference.json').read_text())
passes = {'multi': 16, 'first': 64, 'stress': 32, 'single10': 128}
cells = [(t, 1) for t in [1, 2, 4, 8, 16, 24]] + [(1, 4), (8, 4)]
prior = [json.loads(x) for x in (root/'matrix.jsonl').read_text().splitlines()] if (root/'matrix.jsonl').exists() else []
key = lambda r: (r['input'], r['threads'], r['instances'], r['arm'])
seen = {(r['rep'], *key(r)) for r in prior}
failed = {key(r) for r in prior if not r['valid']}
rows = prior.copy()
env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
lock = open('/home/lilith/tmp/rav1d-benchmark.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
with (root/'matrix.jsonl').open('a') as output:
    for rep in range(a.reps):
        names = list(inputs)
        names = names[rep % 4:] + names[:rep % 4]
        for name in names:
            for cell_index, (threads, instances) in enumerate(cells):
                arms = ['current-checked', 'current-unchecked', 'current-asm', 'upstream']
                if instances == 1 and threads in [1, 8]:
                    arms += ['latest-checked', 'latest-unchecked', 'example-control']
                if instances == 1 and threads == 1:
                    arms += ['first-checked', 'first-unchecked', 'restored-asm']
                if name == 'multi' and instances == 1 and threads in [8, 24]:
                    arms += ['auto-unchecked', 'auto-asm', 'upstream-auto']
                shift = (rep + cell_index) % len(arms)
                arms = arms[shift:] + arms[:shift]
                if rep % 2:
                    arms.reverse()
                for arm in arms:
                    cell = (name, threads, instances, arm)
                    if (rep, *cell) in seen or cell in failed:
                        continue
                    command = [str(root/'bin'/arm), str(inputs[name]), str(threads),
                               str(instances), str(passes[name]), '1']
                    started = time.time()
                    r = subprocess.run(command, cwd=repo, env=env, capture_output=True,
                                       text=True, timeout=90)
                    row = dict(rep=rep, input=name, threads=threads, instances=instances,
                               arm=arm, command=command, exit=r.returncode, stdout=r.stdout,
                               stderr=r.stderr, started_unix=started, load=os.getloadavg())
                    lines = r.stdout.splitlines()
                    frames = [x for x in lines if x.startswith('FRAME\t')]
                    validations = [x.split('\t') for x in lines if x.startswith('VALIDATED\t')]
                    timings = [x.split('\t') for x in lines if x.startswith('RESULT\t')]
                    valid = (r.returncode == 0 and frames == oracle[name] and
                             len(validations) == 1 and int(validations[0][1]) == instances*2+1 and
                             len(timings) == 1 and int(timings[0][2]) == len(frames)*passes[name]*instances)
                    row['valid'] = valid
                    row['ms_per_frame'] = float(timings[0][-1]) if valid else None
                    output.write(json.dumps(row)+'\n')
                    output.flush()
                    try:
                        assert valid, (arm, name, threads, instances, r.stderr)
                    except AssertionError:
                        # Keep the correctness gate strict. A failed cell has NO
                        # numeric result, including from any earlier passing run.
                        failed.add(cell)
                        print('CORRECTNESS FAILURE:', cell, r.stderr[-500:], flush=True)
                    rows.append(row)
        print(f'Completed rotation {rep+1}/{a.reps}: {len(rows)} attempts, {len(failed)} failed cells', flush=True)
summary = []
for name in inputs:
    for threads, instances in cells:
        selected = [r for r in rows if (r['input'], r['threads'], r['instances']) == (name, threads, instances) and key(r) not in failed and r['valid']]
        values = {arm: [r['ms_per_frame'] for r in selected if r['arm'] == arm]
                  for arm in sorted({r['arm'] for r in selected})}
        assert all(len(v) == a.reps for v in values.values()), (name, threads, instances)
        medians = {arm: statistics.median(v) for arm, v in values.items()}
        comparisons = {}
        for arm in values:
            ratios = [x/y for x, y in zip(values[arm], values['upstream'])]
            comparisons[arm] = dict(ratio_of_medians=medians[arm]/medians['upstream'],
                                   paired_median=statistics.median(ratios),
                                   paired_range=[min(ratios), max(ratios)])
        summary.append(dict(input=name, threads=threads, instances=instances,
                            medians=medians, samples=values, vs_upstream=comparisons))
        print(name, f'{threads}x{instances}', medians, flush=True)
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

(root/'failed-cells.json').write_text(json.dumps(sorted(failed), indent=2)+'\n')
