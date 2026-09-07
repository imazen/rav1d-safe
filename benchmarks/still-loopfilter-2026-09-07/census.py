"""Count loopfilter mask use, with independent visible-frame checks.

Run sequentially through run-heavy. Each fresh process performs one serial
reference decode and three requested-worker decodes. Subtract the serial
contribution using the matching t1 cell; counters themselves never reset.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8, 16, 24])
p.add_argument('--inputs', nargs='+')
p.add_argument('--marker', type=Path)
a = p.parse_args()
assert a.threads[0] == 1 and len(set(a.threads)) == len(a.threads)
root = a.work_dir.resolve()
corpus = json.loads((root / 'corpus.json').read_text())
if a.inputs:
    corpus = [s for s in corpus if s['name'] in a.inputs]
    assert len(corpus) == len(a.inputs)
env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
fields = ['calls', 'lanes', 'passing', 'mid', 'wide', 'narrow', 'none_calls',
          'mid_unused_calls', 'wide_unused_calls', 'narrow_unused_calls']
sites = json.loads((root / 'mask-census-source-audit.json').read_text())['sites']
records = []
for source in corpus:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    digest = root / (source['name'] + '.dav1d.md5')
    reference_command = ['dav1d', '-q', '-i', source['path'], '--muxer', 'md5',
                         '--filmgrain', '1', '--threads', '1', '-o', str(digest)]
    subprocess.run(reference_command, check=True, capture_output=True)
    expected = (f"FRAME\t0\t{source['width']}x{source['height']}\t"
                f"{source['bit_depth']}\t{digest.read_text().split()[0]}")
    serial = None
    for threads in a.threads:
        if a.marker:
            assert 'codex-still-parity' in a.marker.read_text()
            a.marker.write_text(datetime.now(timezone.utc).isoformat()
                + f' codex-still-parity loopfilter mask census {source["name"]} t{threads}\n')
        command = [str(a.binary), source['path'], str(threads), '1', '1', '1']
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
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
        start, end = [lines.index('LF_MASK_CENSUS_' + tag) for tag in ['BEGIN', 'END']]
        assert lines[start + 1].split('\t') == ['kernel', *fields]
        counts = {site['kernel']: {f: 0 for f in fields} for site in sites}
        for line in lines[start + 2:end]:
            name, *values = line.split('\t')
            assert name in counts and len(values) == len(fields)
            counts[name] = dict(zip(fields, map(int, values)))
        assert sum(c['calls'] for c in counts.values()) > 0, path
        for site in sites:
            c = counts[site['kernel']]
            assert c['lanes'] == site['lanes'] * c['calls'], (path, site)
            assert c['passing'] == c['mid'] + c['wide'] + c['narrow'] <= c['lanes']
            assert all(0 <= c[f] <= c['calls'] for f in fields[6:])
        if threads == 1:
            assert all(v % 4 == 0 for c in counts.values() for v in c.values()), path
            serial = {name: {f: v // 4 for f, v in c.items()} for name, c in counts.items()}
        assert serial is not None
        corrected = {}
        for name, c in counts.items():
            values = {f: v - serial[name][f] for f, v in c.items()}
            assert all(v >= 0 for v in values.values()), (path, name, values)
            corrected[name] = {f: v / 3 for f, v in values.items()}
        calls = sum(c['calls'] for c in corrected.values())
        none = sum(c['none_calls'] for c in corrected.values())
        record = dict(input=source['name'], threads=threads, raw=path.name,
                      lifetime_decodes=4, serial_reference_decodes=1, requested_decodes=3,
                      requested_counts_per_decode=corrected,
                      all_requested_counts_integral=all(v.is_integer() for c in corrected.values()
                                                       for v in c.values()),
                      total_kernel_calls_per_requested_decode=calls,
                      no_passing_lane_call_fraction=none / calls)
        records.append(record)
        print(source['name'], threads, f'{calls:.0f} calls/frame, no eligible lanes {none / calls:.2%}', flush=True)
        (root / 'mask-census-summary.json').write_text(json.dumps(dict(
            binary=str(a.binary), binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
            corpus_sha256=hashlib.sha256((root / 'corpus.json').read_bytes()).hexdigest(),
            sites=sites,
            method='Monotonic diagnostic counters, no resets, all SIMD families enabled. '
                   'One serial reference plus three requested-worker decodes per fresh process. '
                   'Subtract matching serial per-frame counts before dividing by three. '
                   'All visible frames checked against dav1d; RESULT times excluded. '
                   'mid/wide/narrow count final selected lanes, none_calls counts fm == 0.',
            records=records), indent=2) + '\n')
