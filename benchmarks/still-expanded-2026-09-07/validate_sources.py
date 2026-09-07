"""Verify acquired source bytes and make previews, without AV1 decoding/timing.

Use /usr/bin/python3 and run-heavy. Source and acquisition records are never
rewritten. A new output directory preserves this independent validation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageOps, __version__ as pillow_version

p = argparse.ArgumentParser()
p.add_argument('--acquisition', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--repo', type=Path, required=True)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
records = json.loads((a.acquisition / 'sources.json').read_text())
results = []
commands = []
log = (a.output_dir / 'progress.jsonl').open('x')


def progress(**record):
    record['utc'] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(record)
    print(line, flush=True)
    log.write(line + '\n')
    log.flush()
    marker = a.repo / '.workongoing'
    if marker.exists() and 'codex-still-parity' in marker.read_text():
        marker.write_text(record['utc'] + ' codex-still-parity verify source pixels; no AV1 timing\n')


for record in records:
    source = record['source']
    result = dict(id=source['id'], acquisition_status=record['status'])
    results.append(result)
    progress(id=source['id'], event='start')
    try:
        asset = record.get('asset', {})
        assert asset.get('complete'), 'no complete acquired asset'
        path = Path(asset['path'])
        data = path.read_bytes()
        assert len(data) == asset['bytes']
        assert hashlib.sha256(data).hexdigest() == asset['sha256']
        if 'expected_sha1' in source:
            assert hashlib.sha1(data).hexdigest() == source['expected_sha1']
        del data
        result.update(path=str(path), sha256=asset['sha256'], bytes=asset['bytes'])
        if source['kind'] == 'pdf':
            prefix = a.output_dir / source['id']
            page = str(source['planned_pdf_page'])
            command = ['pdftocairo', '-f', page, '-l', page, '-scale-to', '960',
                       '-singlefile', '-png', str(path), str(prefix)]
            completed = subprocess.run(command, capture_output=True, timeout=90)
            commands.append(dict(id=source['id'], command=command,
                                 exit=completed.returncode,
                                 stderr=completed.stderr.decode(errors='replace')))
            assert completed.returncode == 0
            preview = prefix.with_suffix('.png')
            assert preview.is_file()
            rows = (path.parent / 'pdfimages.txt').read_text().splitlines()[2:]
            images = [row for row in rows if row.strip()]
            result.update(pdf_page=int(page), raster_objects=len(images),
                          raster_object_records=images,
                          text_characters=len((path.parent / 'page-text.txt').read_text()),
                          preview=str(preview), status='PDF page rendered; content review pending')
        else:
            with Image.open(path) as image:
                assert image.format == 'JPEG'
                assert list(image.size) == source['expected_size']
                image.load()  # Decode every source pixel; truncated input must fail.
                exif = image.getexif()
                result.update(stored_size=list(image.size), mode=image.mode,
                              exif_orientation=exif.get(274, 1),
                              camera_make=exif.get(271), camera_model=exif.get(272),
                              capture_time=exif.get(306), pillow_version=pillow_version)
                oriented = ImageOps.exif_transpose(image)
                result['oriented_size'] = list(oriented.size)
                w, h = oriented.size
                assert w >= 7680 and h >= 4320 or w >= 4320 and h >= 7680
                oriented.thumbnail((960, 640), Image.Resampling.LANCZOS)
                preview = a.output_dir / (source['id'] + '.png')
                oriented.save(preview)
                result.update(preview=str(preview),
                              status='Full JPEG decode and native geometry pass; content review pending')
        result['preview_sha256'] = hashlib.sha256(preview.read_bytes()).hexdigest()
        progress(id=source['id'], event='verified', status=result['status'])
    except Exception as error:
        result.update(status='failed', error=str(error))
        progress(id=source['id'], event='failed', error=str(error))
    finally:
        (a.output_dir / 'validation.json').write_text(json.dumps(results, indent=2) + '\n')
        (a.output_dir / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')

summary = dict(sources=len(results), verified=sum(r['status'] != 'failed' for r in results),
               failed=sum(r['status'] == 'failed' for r in results),
               source_content_review_complete=False, final_split_frozen=False,
               av1_decodes=0, av1_performance_measurements=0)
(a.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
progress(event='complete', **summary)
log.close()
