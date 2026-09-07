"""Matched in-memory still timings with two independent output checks.

Run under run-heavy, with no other build/encode/profile active. Subprocesses
launch the harness; only its internal decode timer enters the measurements.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--name', required=True)
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--arms', nargs='+', required=True, help='label=/absolute/binary')
p.add_argument('--inputs', nargs='+')
p.add_argument('--cells', nargs='+', default=['1:1:0', '4:1:0', '8:1:0', '24:1:0'])
p.add_argument('--reps', type=int, default=5)
p.add_argument('--target-ms', type=int, default=150)
a = p.parse_args()
assert a.reps >= 5
root = a.work_dir.resolve()
corpus = json.loads((root / 'corpus.json').read_text())
if a.inputs:
    corpus = [c for c in corpus if c['name'] in a.inputs]
    assert len(corpus) == len(a.inputs)
arms = [arm.split('=', 1) for arm in a.arms]
assert len({arm for arm, _ in arms}) == len(arms)
reference = {}
records = []
env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
dav1d_version = subprocess.run(['dav1d', '-v'], capture_output=True, text=True, check=True)
provenance = dict(arms={label: dict(path=binary, sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest())
                        for label, binary in arms},
                  upstream=dict(path=str(a.upstream), sha256=hashlib.sha256(a.upstream.read_bytes()).hexdigest()),
                  corpus_sha256=hashlib.sha256((root / 'corpus.json').read_bytes()).hexdigest(),
                  args=vars(a) | {'work_dir': str(root), 'upstream': str(a.upstream)},
                  dav1d_version=(dav1d_version.stdout + dav1d_version.stderr).strip())
(root / (a.name + '-provenance.json')).write_text(json.dumps(provenance, indent=2) + '\n')


def run(binary, source, threads, instances, prime, passes, rep, arm, log):
    runenv = env.copy()
    if prime:
        runenv['RAV1D_PRIME_THREADS'] = str(prime)
    command = [str(binary), source['path'], str(threads), str(instances), str(passes), '1']
    result = subprocess.run(command, env=runenv, capture_output=True, text=True, timeout=120)
    row = dict(input=source['name'], threads=threads, instances=instances, prime=prime,
               passes=passes, rep=rep, arm=arm, command=command, exit=result.returncode,
               stdout=result.stdout, stderr=result.stderr)
    log.write(json.dumps(row) + '\n')
    log.flush()
    assert result.returncode == 0, row
    lines = result.stdout.splitlines()
    frames = [line for line in lines if line.startswith('FRAME\t')]
    assert len(frames) == 1, row
    assert frames == reference[source['name']], row
    timed = instances * passes
    validated = instances * 2 + 1
    assert f'VALIDATED\t{validated}\tlifetime_frames={timed + validated}\ttimed_frames={timed}' in lines, row
    times = [line.split('\t') for line in lines if line.startswith('RESULT\t')]
    assert len(times) == 1 and int(times[0][2]) == timed, row
    row['ms'] = float(times[0][-1])
    return row


with (root / (a.name + '.jsonl')).open('x') as log:
    for source in corpus:
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
        digest = root / (source['name'] + '.dav1d.md5')
        command = ['dav1d', '-q', '-i', source['path'], '--muxer', 'md5', '--filmgrain', '1',
                   '--threads', '1', '-o', str(digest)]
        subprocess.run(command, check=True, capture_output=True)
        md5 = digest.read_text().split()[0]
        assert len(md5) == 32 and all(c in '0123456789abcdef' for c in md5)
        reference[source['name']] = [f"FRAME\t0\t{source['width']}x{source['height']}\t{source['bit_depth']}\t{md5}"]
        for cell in a.cells:
            threads, instances, prime = map(int, cell.split(':'))
            pilot = run(a.upstream, source, threads, instances, prime, 2, -1, 'upstream-pilot', log)
            passes = max(2, math.ceil(a.target_ms / (pilot['ms'] * instances)))
            for rep in range(a.reps):
                order = arms[rep % len(arms):] + arms[:rep % len(arms)]
                if rep % 2:
                    order = order[::-1]
                for arm, binary in order:
                    records.append(run(binary, source, threads, instances, prime, passes, rep, arm, log))
            print(source['name'], cell, 'validated', flush=True)
    (root / (a.name + '-frame-reference.json')).write_text(json.dumps(reference, indent=2) + '\n')
summary = []
for source in corpus:
    for cell in a.cells:
        threads, instances, prime = map(int, cell.split(':'))
        samples = {arm: [r['ms'] for r in records if
                        (r['input'], r['threads'], r['instances'], r['prime'], r['arm']) ==
                        (source['name'], threads, instances, prime, arm)] for arm, _ in arms}
        summary.append(dict(input=source['name'], threads=threads, instances=instances, prime=prime,
                            medians={arm: statistics.median(values) for arm, values in samples.items()},
                            samples=samples))
(root / (a.name + '-summary.json')).write_text(json.dumps(summary, indent=2) + '\n')
print(a.name, 'complete:', len(records), 'validated measured runs', flush=True)
