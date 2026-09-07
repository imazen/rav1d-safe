"""Pin reviewed source membership and balanced splits before AV1 measurement.

This freezes source bytes and center crops, not encodes or the workload matrix.
The specific admission decisions are documented in SOURCE_FREEZE.md.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--crop-audit', type=Path, required=True)
a = p.parse_args()
selected = {}
for version in [1, 2, 3, 5]:
    acquisition = json.loads((a.work_dir / f'acquisition-v{version}/sources.json').read_text())
    validation = {v['id']: v for v in json.loads(
        (a.work_dir / f'validation-v{version}/validation.json').read_text())}
    for row in acquisition:
        selected[row['source']['id']] = (version, row, validation[row['source']['id']])

# Content strata keep a ground photograph, detailed foliage, a map, and
# different document layouts in each split. This is fixed before AV1 work.
strata = [
    ('photograph', ['photo-london-tram', 'texture-apache-winter']),
    ('photograph', ['photo-emi-koussi', 'photo-new-orleans', 'photo-naples',
                    'photo-kinshasa', 'photo-brazil-rivers', 'photo-orenburg-snow']),
    ('texture', ['texture-coleus', 'texture-galium']),
    ('texture', ['texture-armeria', 'texture-wood-wall']),
    ('texture', ['texture-bergen-forest', 'photo-el-salvador']),
    ('map', ['map-choh', 'map-muwo']),
    ('map', ['map-mora', 'map-shen']),
    ('text', ['text-nasa-api', 'text-nist-sha']),
    ('text', ['text-irs-form', 'text-census-income']),
    ('text', ['text-noaa-ian', 'text-cdc-mmwr']),
]
salt = 'rav1d-still-parity-source-split-v1:'
svg_ns = '{http://www.w3.org/2000/svg}'
href_name = '{http://www.w3.org/1999/xlink}href'
sources = []
crop_audits = []


def center_crop(width, height):
    crop_width = min(width, height * 16 / 9)
    crop_height = crop_width * 9 / 16
    return [(width - crop_width) / 2, (height - crop_height) / 2,
            (width + crop_width) / 2, (height + crop_height) / 2]


def audit_svg(path):
    root = ET.parse(path).getroot()
    x0, y0, width, height = map(float, root.attrib['viewBox'].split())
    assert x0 == 0 and y0 == 0
    crop = center_crop(width, height)
    images = {e.get('id'): e for e in root.iter(svg_ns + 'image')}
    image_uses = []

    def visit(element, ancestors):
        ref = element.get(href_name, '').removeprefix('#')
        if ref in images:
            # Reject unsupported ancestor transformations instead of assuming
            # that a nested reference has the same picture coordinates.
            assert all(e.get('transform') is None for e in ancestors)
            transform = element.get('transform', '')
            match = re.fullmatch(r'matrix\(([^)]+)\)', transform)
            assert match, transform
            aa, bb, cc, dd, xx, yy = map(float, match[1].replace(',', ' ').split())
            image = images[ref]
            assert float(image.get('x', '0')) == 0 and float(image.get('y', '0')) == 0
            iw, ih = float(image.get('width')), float(image.get('height'))
            points = [(aa*x + cc*y + xx, bb*x + dd*y + yy)
                      for x, y in [(0, 0), (iw, 0), (0, ih), (iw, ih)]]
            bounds = [min(p[0] for p in points), min(p[1] for p in points),
                      max(p[0] for p in points), max(p[1] for p in points)]
            separated = (bounds[2] <= crop[0] or crop[2] <= bounds[0]
                         or bounds[3] <= crop[1] or crop[3] <= bounds[1])
            assert separated, (path, bounds, crop)
            image_uses.append(dict(id=ref, bounds=bounds, outside_crop=True))
        for child in element:
            visit(child, ancestors + [element])

    visit(root, [])
    assert images and {i['id'] for i in image_uses} == images.keys()
    return dict(svg_path=str(path), svg_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                page_points=[width, height], crop_points=crop,
                raster_uses=image_uses,
                scope='Source-specific Cairo SVG image uses, including mask definitions; '
                      'all ancestor transforms absent. Not a general PDF verifier.')


for category, identifiers in strata:
    assert len(identifiers) % 2 == 0
    ordered = sorted(identifiers, key=lambda value: hashlib.sha256((salt + value).encode()).digest())
    for index, identifier in enumerate(ordered):
        version, acquired, checked = selected[identifier]
        assert checked['status'] != 'failed', identifier
        asset = acquired['asset']
        assert asset['complete']
        path = Path(asset['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256']
        record = acquired['source'].copy()
        record.update(category=category, split='development' if index < len(ordered)//2 else 'holdout',
                      asset_path=str(path), asset_sha256=asset['sha256'], asset_bytes=asset['bytes'],
                      acquisition_version=version)
        if record['kind'] == 'pdf':
            if checked['raster_objects']:
                assert identifier in ['map-shen', 'map-muwo', 'text-noaa-ian']
                audit = audit_svg(a.work_dir / f'validation-v{version}' / (identifier + '.svg'))
                crop_audits.append(dict(id=identifier, **audit))
                record['pdf_crop_points'] = audit['crop_points']
                record['native_detail'] = 'Vector content in selected crop; embedded raster logos outside crop'
            else:
                record['native_detail'] = 'Selected PDF page has no embedded raster objects (pdfimages)'
            record['pdf_raster_objects_before_crop'] = checked['raster_objects']
        else:
            record.update(stored_size=checked['stored_size'], oriented_size=checked['oriented_size'],
                          camera_make=checked.get('camera_make'), camera_model=checked.get('camera_model'),
                          exif_orientation=checked['exif_orientation'],
                          native_detail='Full JPEG decoded; native camera geometry verified; never upscale')
        sources.append(record)

assert len(sources) == 24
assert len({s['asset_sha256'] for s in sources}) == 24
assert len({s['family'] for s in sources}) == 24
assert Counter(s['split'] for s in sources) == {'development': 12, 'holdout': 12}
for split in ['development', 'holdout']:
    assert Counter(s['category'] for s in sources if s['split'] == split) == {
        'photograph': 4, 'texture': 3, 'map': 2, 'text': 3}
out = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_membership_frozen=True,
           source_splits_frozen=True, workload_and_bitstreams_frozen=False,
           content_review='All selected source previews inspected before any AV1 work.',
           primary_crop='Centered 16:9; downsample camera rasters, render PDF vectors at target resolution.',
           split_method='SHA-256 ordering within predeclared content strata; first half development',
           split_salt=salt, strata=strata, sources=sources,
           exposure=dict(av1_decodes=0, av1_performance_measurements=0),
           limitations=['PDFs are native vector render sources, not camera photographs.',
                        'No HDR claim: source rasters are 8-bit; bit-depth edge coverage remains future work.',
                        'Encoder/quality/bit-depth/tile/edge assignments and output hashes still need freezing.',
                        'No verified off-host backup of large local assets.'])
with a.output.open('x') as output:
    output.write(json.dumps(out, indent=2) + '\n')
with a.crop_audit.open('x') as output:
    output.write(json.dumps(crop_audits, indent=2) + '\n')
print('Pinned 24 distinct source hashes/families, 12 development + 12 holdout; 3 raster-logo crop audits pass.')
