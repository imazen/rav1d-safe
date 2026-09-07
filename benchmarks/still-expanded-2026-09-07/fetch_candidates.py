"""Acquire candidate source bytes and geometry, without running an AV1 decoder.

Run with /usr/bin/python3 (system Pillow), sequentially through run-heavy.
Each invocation requires a fresh output
directory; partial downloads and failures remain visible for investigation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from PIL import Image, __version__ as pillow_version

p = argparse.ArgumentParser()
p.add_argument('--plan', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--repo', type=Path, required=True)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
plan_bytes = a.plan.read_bytes()
plan = json.loads(plan_bytes)
(a.output_dir / 'plan.json').write_bytes(plan_bytes)
log = (a.output_dir / 'progress.jsonl').open('x')
requests = []
results = []


def progress(**record):
    record['utc'] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(record)
    print(line, flush=True)
    log.write(line + '\n')
    log.flush()
    marker = a.repo / '.workongoing'
    if marker.exists() and 'codex-still-parity' in marker.read_text():
        marker.write_text(record['utc'] + ' codex-still-parity source acquisition; no AV1 timing\n')


def download(url, destination, limit):
    record = dict(url=url, path=str(destination), maximum_bytes=limit)
    requests.append(record)
    start = time.monotonic()
    digest = hashlib.sha256()
    sha1 = hashlib.sha1()
    total = 0
    try:
        request = Request(url, headers={
            'User-Agent': 'rav1d-safe-corpus/1.0 (https://github.com/imazen/rav1d-safe)',
        })
        with urlopen(request, timeout=20) as response:
            record['status'] = response.status
            assert response.status == 200
            if response.headers.get('Content-Length'):
                assert int(response.headers['Content-Length']) <= limit
            with destination.open('xb') as output:
                while True:
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    total += len(block)
                    assert total <= limit, 'source exceeds acquisition limit'
                    assert time.monotonic() - start < 120, 'source transfer deadline'
                    output.write(block)
                    digest.update(block)
                    sha1.update(block)
            if response.headers.get('Content-Length'):
                assert total == int(response.headers['Content-Length'])
        record.update(bytes=total, sha256=digest.hexdigest(), sha1=sha1.hexdigest(), complete=True)
    except Exception as error:
        record.update(bytes_received=total, complete=False, error=str(error))
        raise
    finally:
        (a.output_dir / 'requests.json').write_text(json.dumps(requests, indent=2) + '\n')
    return record


def command(arguments, destination):
    completed = subprocess.run(arguments, capture_output=True, timeout=90)
    destination.write_bytes(completed.stdout)
    destination.with_suffix(destination.suffix + '.stderr').write_bytes(completed.stderr)
    assert completed.returncode == 0, (arguments, completed.stderr.decode(errors='replace'))
    return completed.stdout


for source in plan['candidates']:
    progress(id=source['id'], event='start')
    directory = a.output_dir / source['id']
    directory.mkdir()
    result = dict(source=source)
    results.append(result)
    try:
        if source['kind'] == 'nasa':
            page = directory / 'source.html'
            result['page'] = download(source['page_url'], page, 2_000_000)
            links = set(re.findall(r'(?:href|src|data-zoom-image)="([^"]+)"', page.read_text()))
            candidates = [urljoin(source['page_url'], link) for link in links
                          if '/DatabaseImages/ESC/large/' in link
                          and link.endswith('/' + source['nasa_id'] + '.JPG')]
            assert len(candidates) == 1, candidates
            asset_url = candidates[0]
        else:
            asset_url = source['asset_url']
        asset = directory / ('source.pdf' if source['kind'] == 'pdf' else 'source.jpg')
        result['asset'] = download(asset_url, asset, 64 * 1024 * 1024)
        if 'expected_bytes' in source:
            assert result['asset']['bytes'] == source['expected_bytes']
        if 'expected_sha1' in source:
            assert result['asset']['sha1'] == source['expected_sha1']
        if source['kind'] == 'pdf':
            assert asset.read_bytes()[:5] == b'%PDF-'
            command(['pdfinfo', str(asset)], directory / 'pdfinfo.txt')
            page = str(source['planned_pdf_page'])
            command(['pdfimages', '-f', page, '-l', page, '-list', str(asset)],
                    directory / 'pdfimages.txt')
            command(['pdftotext', '-f', page, '-l', page, '-layout', str(asset), '-'],
                    directory / 'page-text.txt')
            result['status'] = 'PDF acquired; vector/content review and rasterization outstanding'
        else:
            with Image.open(asset) as image:
                assert image.format == 'JPEG'
                size = list(image.size)
                stream = dict(width=image.width, height=image.height, mode=image.mode,
                              exif_orientation=image.getexif().get(274, 1),
                              pillow_version=pillow_version)
            (directory / 'geometry.json').write_text(json.dumps(stream, indent=2) + '\n')
            assert size == source['expected_size'], (size, source['expected_size'])
            width, height = size
            assert width >= 7680 and height >= 4320 or width >= 4320 and height >= 7680
            result['geometry'] = stream
            result['status'] = 'Native-geometry raster acquired; full pixel/content review outstanding'
        progress(id=source['id'], event='acquired', bytes=result['asset']['bytes'])
    except Exception as error:
        result.update(status='failed', error=str(error))
        progress(id=source['id'], event='failed', error=str(error))
    finally:
        (a.output_dir / 'sources.json').write_text(json.dumps(results, indent=2) + '\n')

summary = dict(plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), sources=len(results),
               acquired=sum(r['status'] != 'failed' for r in results),
               failed=sum(r['status'] == 'failed' for r in results),
               full_pixel_validation=False, final_split_frozen=False,
               av1_decodes=0, av1_performance_measurements=0)
(a.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
progress(event='complete', **summary)
log.close()
