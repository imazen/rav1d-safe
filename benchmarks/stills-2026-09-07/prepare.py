"""Generate the still investigation corpus. Run under scripts/run-heavy.

Requires system Pillow; large assets stay outside Git. The source manifest pins
public downloads by SHA-256. This is a development set, not a parity holdout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageOps, __version__ as pillow_version

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--sources', type=Path, default=Path(__file__).with_name('sources.json'))
a = p.parse_args()
root = a.work_dir.resolve()
root.mkdir(parents=True, exist_ok=True)
sources = json.loads(a.sources.read_text())
records = []
encoder_help = subprocess.run(['aomenc', '--help'], capture_output=True, text=True, check=True)
assert 'AOMedia Project AV1 Encoder v3.13.1' in encoder_help.stdout, 'use pinned libaom 3.13.1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for source in sources:
    path = root / Path(source['path']).name
    if not path.exists():
        subprocess.run(['curl', '-fLsS', '--max-time', '120', source['url'],
                        '-o', str(path)], check=True)
    assert sha(path) == source['sha256'], 'source changed: ' + str(path)
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    assert image.size == (int(source['width']), int(source['height']))
    kind = {'1407': 'photo', '5017': 'map'}[source['number']]
    for label, width, height in [('2k', 1920, 1080), ('4k', 3840, 2160), ('8k', 7680, 4320)]:
        assert image.width >= width and image.height >= height
        # Center crop to 16:9, then downsample; never upscale. Pillow YCbCr is
        # full-range BT.601. Box subsampling produces centered 4:2:0 chroma.
        rgb = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        y, u, v = rgb.convert('YCbCr').split()
        y4m = root / f'{kind}-{label}.y4m'
        with y4m.open('wb') as out:
            out.write(f'YUV4MPEG2 W{width} H{height} F1:1 Ip A1:1 C420jpeg XCOLORRANGE=FULL\nFRAME\n'.encode())
            out.write(y.tobytes())
            for plane in [u, v]:
                out.write(plane.resize((width // 2, height // 2), Image.Resampling.BOX).tobytes())
        # 8K cannot be one tile: AV1 limits tile width to 4096 and area to
        # 4096*2304. Use the minimum legal 2x2 arrangement there.
        for layout, cols, rows in [('min', int(width > 4096), int(width > 4096)), ('t8', 2, 1)]:
            name = f'{kind}-{label}-{layout}'
            encoded = root / (name + '.ivf')
            command = ['aomenc', '--allintra', '--ivf', '--limit=1', '--passes=1',
                       '--threads=8', '--row-mt=1', '--cpu-used=6', '--end-usage=q',
                       '--cq-level=24', '--bit-depth=8', '--input-bit-depth=8',
                       f'--tile-columns={cols}', f'--tile-rows={rows}',
                       '--test-decode=fatal', '-o', str(encoded), str(y4m)]
            with (root / (name + '-encode.log')).open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            record = dict(name=name, path=str(encoded), sha256=sha(encoded),
                          bytes=encoded.stat().st_size, width=width, height=height,
                          bit_depth=8, chroma='420', tile_cols=1 << cols, tile_rows=1 << rows,
                          source_id=source['number'], source_sha256=source['sha256'],
                          y4m_sha256=sha(y4m), pillow_version=pillow_version,
                          encoder='libaom aomenc 3.13.1', command=command)
            records.append(record)
            (root / 'corpus.json').write_text(json.dumps(records, indent=2) + '\n')
            print(name, record['bytes'], 'encoded', flush=True)
