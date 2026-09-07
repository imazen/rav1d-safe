"""Profile still decode timers; validate output, keep probes out of speed data."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--arms', nargs='+', required=True, help='label=/absolute/binary')
p.add_argument('--cases', nargs='+', required=True, help='input:threads:arm')
p.add_argument('--seconds', type=float, default=2)
p.add_argument('--output-subdir', default='profiles', help='Fresh profile directory inside work-dir')
a = p.parse_args()
root = a.work_dir.resolve()
arms = dict(arm.split('=', 1) for arm in a.arms)
corpus = {c['name']: c for c in json.loads((root / 'corpus.json').read_text())}
assert Path(a.output_subdir).name == a.output_subdir and a.output_subdir.startswith('profiles')
out = root / a.output_subdir
out.mkdir(exist_ok=True)
assert not (out / 'commands.json').exists(), 'refuse to overwrite previous profile evidence'
assert not (out / 'provenance.json').exists(), 'refuse to overwrite previous profile provenance'
(out / 'provenance.json').write_text(json.dumps(dict(
    arms={label: dict(path=binary, sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest())
          for label, binary in arms.items()},
    corpus_sha256=hashlib.sha256((root / 'corpus.json').read_bytes()).hexdigest(),
    cases=a.cases, seconds=a.seconds,
), indent=2) + '\n')
records = []
for case in a.cases:
    name, threads, arm = case.split(':')
    source = corpus[name]
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    md5 = (root / (name + '.dav1d.md5')).read_text().split()[0]
    expected = f"FRAME\t0\t{source['width']}x{source['height']}\t{source['bit_depth']}\t{md5}"
    env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
    pilot_cmd = [arms[arm], source['path'], threads, '1', '2', '1']
    pilot = subprocess.run(pilot_cmd, env=env, capture_output=True, text=True, check=True)
    assert expected in pilot.stdout.splitlines()
    ms = float(next(l for l in pilot.stdout.splitlines() if l.startswith('RESULT\t')).split('\t')[-1])
    passes = max(2, math.ceil(a.seconds * 1000 / ms))
    key = f'{name}-t{threads}-{arm}'
    ctl, ack = out / (key + '.ctl'), out / (key + '.ack')
    for path in [ctl, ack]:
        assert not path.exists(), 'refuse to overwrite profile control files'
        os.mkfifo(path)
    env['RAV1D_PERF_CONTROL'] = str(ctl) + ',' + str(ack)
    command = ['perf', 'record', '-e', 'cycles:u', '-F', '499', '-o', str(out / (key + '.data')),
               '--delay=-1', '--control=fifo:' + str(ctl) + ',' + str(ack), '--',
               arms[arm], source['path'], threads, '1', str(passes), '1']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    record = dict(case=case, command=command, exit=result.returncode,
                  stdout=result.stdout, stderr=result.stderr, pilot_command=pilot_cmd,
                  pilot_stdout=pilot.stdout)
    records.append(record)
    (out / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
    assert result.returncode == 0, record
    lines = result.stdout.splitlines()
    assert expected in lines, record
    assert f'VALIDATED\t3\tlifetime_frames={passes+3}\ttimed_frames={passes}' in lines, record
    report = ['perf', 'report', '--stdio', '--no-children', '--percent-limit', '0.05',
              '-i', str(out / (key + '.data')), '--sort', 'symbol', '--field-separator', ';']
    r = subprocess.run(report, capture_output=True, text=True, check=True)
    (out / (key + '.txt')).write_text(r.stdout)
    print(case, 'profile validated', flush=True)
