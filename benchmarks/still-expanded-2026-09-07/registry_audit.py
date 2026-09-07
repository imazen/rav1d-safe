"""Audit native-geometry candidates and already-exposed canonical test sources.

Reads a pinned metadata snapshot only. Does not select/freeze final workload
sources, download image bytes, or inspect decode performance.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--registry', type=Path, required=True)
p.add_argument('--seed-sources', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
provenance = json.loads((a.registry / 'provenance.json').read_text())
for entry in provenance['files']:
    data = (a.registry / entry['path']).read_bytes()
    assert len(data) == entry['bytes']
    assert hashlib.sha256(data).hexdigest() == entry['sha256']


def rows(name):
    with (a.registry / name).open() as source:
        return list(csv.DictReader(source, delimiter='\t'))


manifest = rows('CORPUS-MANIFEST.tsv')
assert len(manifest) == 2160
families = {row['id']: row for row in rows('manifests/split_map_family.tsv')}
train = {row['id']: row for row in rows('manifests/train.tsv')}
test = {row['id']: row for row in rows('manifests/test.tsv')}
assert len(train) == 1084 and len(test) == 418
assert train.keys().isdisjoint(test)
eligible = []
for row in manifest:
    width, height = int(row['width']), int(row['height'])
    landscape = width >= 7680 and height >= 4320
    portrait = width >= 4320 and height >= 7680
    if landscape or portrait:
        eligible.append(row | dict(landscape_uhd=landscape, portrait_uhd=portrait,
                                    canonical_family=families[row['number']]['family']))
seeds = json.loads(a.seed_sources.read_text())
exposed = []
for seed in seeds:
    row = test[seed['number']]
    assert seed['path'] == row['path']
    assert seed['sha256'] == row['sha256']
    family = families[seed['number']]['family']
    relatives = [other['id'] for other in families.values()
                 if family and other['family'] == family]
    exposed.append(dict(id=seed['number'], path=seed['path'], canonical_split=row['split'],
                        sha256=row['sha256'], family=family,
                        excluded_holdout_ids=sorted(set(relatives + [seed['number']])),
                        campaign_role='Already-exposed investigation seed; never a holdout result'))
out = dict(registry_revision=provenance['revision'], registry_files=provenance['files'],
           admission_rule='Manifest width>=7680,height>=4320 or width>=4320,height>=7680; header/content verification still required',
           canonical_rows=len(manifest), metadata_geometry_candidates=len(eligible),
           candidates=eligible, exposed_test_sources=exposed,
           source_selection_frozen=False,
           limitation='17 metadata-eligible files cannot provide 12 development plus 12 distinct holdout sources. They also include renders and related maps; this is not a completed natural-image corpus.')
with a.output.open('x') as output:
    output.write(json.dumps(out, indent=2) + '\n')
print(f'{len(eligible)} metadata geometry candidates; {len(exposed)} already-exposed canonical test sources recorded.')
