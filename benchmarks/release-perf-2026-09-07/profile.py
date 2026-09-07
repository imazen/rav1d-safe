"""Sample self instruction pointers only while the harness's timer is active."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
a = p.parse_args()
root = a.work_dir.resolve()
repo = a.repo.resolve()
out = root / 'profiles'
out.mkdir(exist_ok=True)
records = []
for arm, threads, passes in [('current-unchecked', 8, 96), ('current-asm', 8, 384)]:
    name = f'{arm}-t{threads}'
    ctl, ack = out / (name + '.ctl'), out / (name + '.ack')
    for path in [ctl, ack]:
        if path.exists():
            path.unlink()
        os.mkfifo(path)
    env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
    env['RAV1D_PERF_CONTROL'] = str(ctl) + ',' + str(ack)
    command = ['perf', 'record', '-e', 'cycles:u', '-F', '499', '-o', str(out / (name + '.data')),
               '--delay=-1', '--control=fifo:' + str(ctl) + ',' + str(ack), '--',
               str(root / 'bin' / arm), str(repo / 'test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf'),
               str(threads), '1', str(passes), '1']
    r = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    records.append(dict(name=name, command=command, exit=r.returncode, stdout=r.stdout, stderr=r.stderr))
    (out / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
    assert r.returncode == 0 and 'VALIDATED\t3\t' in r.stdout, records[-1]
    report = ['perf', 'report', '--stdio', '--no-children', '--percent-limit', '0.1',
              '-i', str(out / (name + '.data')), '--sort', 'symbol', '--field-separator', ';']
    r = subprocess.run(report, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    (out / (name + '.txt')).write_text(r.stdout)
    print(name, 'completed and validated', flush=True)
