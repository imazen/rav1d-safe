"""Collect local MSAC counters from the archived entropy-probe experiment.

Run through run-heavy, sequentially. Never use diagnostic RESULT times as
performance evidence. All frames must still match independent dav1d.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--inputs', nargs='+')
p.add_argument('--threads', type=int, nargs='+', default=[1, 8])
a = p.parse_args()
root = a.work_dir.resolve()
corpus = json.loads((root / 'corpus.json').read_text())
if a.inputs:
    corpus = [c for c in corpus if c['name'] in a.inputs]
    assert len(corpus) == len(a.inputs)
env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
records = []


def add(left, right):
    assert len(left) == len(right)
    return [add(x, y) if isinstance(x, list) else x + y
            for x, y in zip(left, right)]


for source in corpus:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    digest = root / (source['name'] + '.entropy-probe.dav1d.md5')
    reference_command = ['dav1d', '-q', '-i', source['path'], '--muxer', 'md5',
                         '--filmgrain', '1', '--threads', '1', '-o', str(digest)]
    subprocess.run(reference_command, check=True, capture_output=True)
    expected = (f"FRAME\t0\t{source['width']}x{source['height']}\t"
                f"{source['bit_depth']}\t{digest.read_text().split()[0]}")
    first_counters = None
    for threads in a.threads:
        command = [str(a.binary), source['path'], str(threads), '1', '1', '1']
        result = subprocess.run(command, capture_output=True, text=True,
                                env=env, timeout=120)
        raw = dict(command=command, reference_command=reference_command,
                   reference=expected, exit=result.returncode,
                   stdout=result.stdout, stderr=result.stderr)
        path = root / f"entropy-probe-{source['name']}-t{threads}.json"
        with path.open('x') as output:
            json.dump(raw, output, indent=2)
            output.write('\n')
        assert result.returncode == 0, path
        lines = result.stdout.splitlines()
        assert [s for s in lines if s.startswith('FRAME\t')] == [expected], path
        assert 'VALIDATED\t3\tlifetime_frames=4\ttimed_frames=1' in lines, path
        counters = [json.loads(s.split('\t', 1)[1]) for s in result.stderr.splitlines()
                    if s.startswith('MSAC_PROBE\t')]
        assert counters, path
        total = counters[0]
        for row in counters[1:]:
            assert total.keys() == row.keys()
            total = {k: add(total[k], row[k]) for k in total}
        norms = sum(total['norm_shifts'])
        symbols = sum(map(sum, total['symbols']))
        assert norms == symbols + sum(total['bools']), (path, total)
        assert sum(map(sum, total['cdf_count_bins'])) == symbols, path
        if first_counters is None:
            first_counters = total
        else:
            assert total == first_counters, (path, 'worker count changed entropy operations')
        record = dict(input=source['name'], threads=threads, visible_frames=4,
                      contexts=len(counters), total=total, raw=path.name)
        records.append(record)
        n3 = total['symbols'][3]
        print(source['name'], threads, 'norms/frame', norms / 4,
              'n3 share', round(sum(n3) / norms, 4),
              'n3 symbols', [round(x / sum(n3), 4) for x in n3[:4]],
              'refills/norm', round(sum(total['refill_c_octets']) / norms, 4), flush=True)

provenance = dict(binary=str(a.binary),
                  binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                  corpus_sha256=hashlib.sha256((root / 'corpus.json').read_bytes()).hexdigest(),
                  args=vars(a) | dict(binary=str(a.binary), work_dir=str(root)),
                  timing_use='none: diagnostic build, RESULT times must be ignored',
                  records=records)
(root / 'entropy-probe-summary.json').write_text(json.dumps(provenance, indent=2) + '\n')
