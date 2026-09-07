"""Untimed transform-shape census; run sequentially through run-heavy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--threads', type=int, nargs='+', default=[1, 8])
a = p.parse_args()
root = a.work_dir.resolve()
corpus = json.loads((root / 'corpus.json').read_text())
env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
records = []
for source in corpus:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    digest = root / (source['name'] + '.dav1d.md5')
    reference_command = ['dav1d', '-q', '-i', source['path'], '--muxer', 'md5',
                         '--filmgrain', '1', '--threads', '1', '-o', str(digest)]
    subprocess.run(reference_command, check=True, capture_output=True)
    expected = (f"FRAME\t0\t{source['width']}x{source['height']}\t"
                f"{source['bit_depth']}\t{digest.read_text().split()[0]}")
    first_counts = None
    for threads in a.threads:
        command = [str(a.binary), source['path'], str(threads), '1', '1', '1']
        result = subprocess.run(command, env=env, capture_output=True,
                                text=True, timeout=120)
        path = root / f"census-{source['name']}-t{threads}.json"
        with path.open('x') as output:
            json.dump(dict(command=command, reference_command=reference_command,
                           exit=result.returncode, expected=expected,
                           stdout=result.stdout, stderr=result.stderr), output, indent=2)
            output.write('\n')
        assert result.returncode == 0, path
        lines = result.stdout.splitlines()
        assert [s for s in lines if s.startswith('FRAME\t')] == [expected], path
        assert 'VALIDATED\t3\tlifetime_frames=4\ttimed_frames=1' in lines, path
        start = lines.index('ITX_CENSUS_BEGIN')
        end = lines.index('ITX_CENSUS_END')
        assert lines[start + 1] == 'depth\tpath\tshape\tcalls\tcoeff_area'
        counts = []
        for line in lines[start + 2:end]:
            depth, dispatch, shape, calls, area = line.split('\t')
            calls, area = int(calls), int(area)
            w, h = map(int, shape.split('x'))
            assert area == calls * w * h and calls % 4 == 0, line
            assert dispatch in ['SCALAR', 'simd']
            counts.append(dict(depth=depth, dispatch=dispatch, shape=shape,
                               calls_per_frame=calls // 4, coeff_area_per_frame=area // 4))
        assert counts, path
        if first_counts is None:
            first_counts = counts
        else:
            assert counts == first_counts, (path, 'worker count changed transform census')
        records.append(dict(input=source['name'], threads=threads,
                            visible_frames=4, counts=counts, raw=path.name))
        fallback = sum(c['coeff_area_per_frame'] for c in counts if c['dispatch'] == 'SCALAR')
        total = sum(c['coeff_area_per_frame'] for c in counts)
        print(source['name'], threads, 'outer-fallback coefficient area',
              f'{fallback / total:.2%}', flush=True)

provenance = dict(binary=str(a.binary),
                  binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                  corpus_sha256=hashlib.sha256((root / 'corpus.json').read_bytes()).hexdigest(),
                  method='Fresh process per cell; all SIMD families enabled; no counter resets. '
                         'Four visible decodes per invocation. Diagnostic RESULT times excluded. '
                         'SCALAR means outer dispatch declined; fallback can use SIMD internally.',
                  records=records)
(root / 'census-summary.json').write_text(json.dumps(provenance, indent=2) + '\n')
