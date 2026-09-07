"""Control in-process timers; process startup, file I/O and hashes are untimed."""
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
repo = a.repo.resolve()
root = a.work_dir.resolve()
inputs = {
    'multi': repo / 'test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf',
    'first': root / 'non_uniform_first.obu',
    'stress': repo / 'tests/crash_vectors/tile_threading_cdef_lpf_race.obu',
    'single10': repo / 'tests/crash_vectors/lr_sgr_10bpc_noisy_nocdef.obu',
}
# The tiled single-frame input is precisely the first packet of the video.
ivf = inputs['multi'].read_bytes()
pos = int.from_bytes(ivf[6:8], 'little')
size = int.from_bytes(ivf[pos:pos+4], 'little')
assert ivf[:4] == b'DKIF' and len(ivf) >= pos + 12 + size
inputs['first'].write_bytes(ivf[pos+12:pos+12+size])
cells = [(t, 1) for t in [1, 2, 4, 8, 16, 24]] + [(1, 4), (8, 4)]
passes = {'multi': 24, 'first': 256, 'stress': 128, 'single10': 256}
references = {}
rows = []
env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
# Shared timing lock; run-heavy is still required for CPU/RAM isolation.
lock = open('/home/lilith/tmp/rav1d-benchmark.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
with (root / 'matrix.jsonl').open('w') as f:
    for rep in range(a.reps):
        names = list(inputs)
        names = names[rep % len(names):] + names[:rep % len(names)]
        for name in names:
            for cell_index, (threads, instances) in enumerate(cells):
                arms = ['checked', 'upstream-fd1']
                if name == 'multi':
                    arms.append('upstream-auto')
                shift = (rep + cell_index) % len(arms)
                arms = arms[shift:] + arms[:shift]
                for arm in arms:
                    cmd = [str(root / 'bin' / arm), str(inputs[name]), str(threads),
                           str(instances), str(passes[name]), '1']
                    started = time.time()
                    result = subprocess.run(cmd, cwd=repo, env=env, capture_output=True,
                                            text=True, timeout=60)
                    row = dict(rep=rep, input=name, threads=threads, instances=instances,
                               arm=arm, command=cmd, exit=result.returncode,
                               stdout=result.stdout, stderr=result.stderr,
                               started_unix=started, load=os.getloadavg())
                    f.write(json.dumps(row) + '\n')
                    f.flush()  # Preserve failed runs too.
                    assert result.returncode == 0, row
                    lines = result.stdout.splitlines()
                    frames = [v for v in lines if v.startswith('FRAME\t')]
                    assert frames and frames == references.setdefault(name, frames), row
                    validations = [v.split('\t') for v in lines if v.startswith('VALIDATED\t')]
                    assert len(validations) == 1 and int(validations[0][1]) == instances * 2 + 1, row
                    values = [v.split('\t') for v in lines if v.startswith('RESULT\t')]
                    assert len(values) == 1 and int(values[0][2]) == len(frames)*passes[name]*instances, row
                    row['ms_per_frame'] = float(values[0][-1])
                    rows.append(row)
        print(f'Completed rotation {rep + 1}/{a.reps}; {len(rows)} runs validated', flush=True)
summary = []
for name in inputs:
    for threads, instances in cells:
        arms = ['checked', 'upstream-fd1'] + (['upstream-auto'] if name == 'multi' else [])
        values = {
            arm: [r['ms_per_frame'] for r in rows if
                  (r['input'], r['threads'], r['instances'], r['arm']) ==
                  (name, threads, instances, arm)] for arm in arms
        }
        medians = {arm: statistics.median(v) for arm, v in values.items()}
        comparisons = {}
        for arm in arms[1:]:
            ratios = [x/y for x, y in zip(values['checked'], values[arm])]
            comparisons[arm] = dict(ratio_of_medians=medians['checked']/medians[arm],
                                   paired_median=statistics.median(ratios),
                                   paired_range=[min(ratios), max(ratios)])
        summary.append(dict(input=name, threads=threads, instances=instances,
                            medians=medians, samples=values, comparisons=comparisons))
        print(name, f'{threads}x{instances}', medians, comparisons, flush=True)
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
(root/'frame-reference.json').write_text(json.dumps(references, indent=2)+'\n')
