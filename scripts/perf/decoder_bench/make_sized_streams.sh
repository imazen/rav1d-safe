#!/usr/bin/env bash
# Builds the 720p / 1080p / 1080p-4-tile panning inter clips used by
# docs/DECODER_COMPARISON.md (size-aware frame delay) from one 3840x2160 I420 frame.
#
#   make_sized_streams.sh photo4k.yuv OUT_DIR
#
# photo4k.yuv: e.g. `dav1d -i photo.ivf -o photo4k.yuv --limit 1` (3840x2160 8-bit 4:2:0).
# Needs python3 and aomenc (libaom). Each clip is 30 frames, a window panning 3 px right
# and 2 px down per frame, encoded --cpu-used=6 --end-usage=q --cq-level=32
# --lag-in-frames=0.
set -euo pipefail
src=$1; out=$2; mkdir -p "$out"
python3 - "$src" "$out" <<'PY'
import sys
W0, H0 = 3840, 2160
d = open(sys.argv[1], 'rb').read()
assert len(d) >= W0 * H0 * 3 // 2, 'expected a 3840x2160 I420 frame'
Y = d[:W0 * H0]; U = d[W0 * H0:W0 * H0 * 5 // 4]; V = d[W0 * H0 * 5 // 4:W0 * H0 * 3 // 2]
def crop(p, pw, x, y, w, h):
    return b''.join(p[(y + r) * pw + x:(y + r) * pw + x + w] for r in range(h))
for name, (w, h) in (('720p', (1280, 720)), ('1080p', (1920, 1080))):
    buf = bytearray()
    for i in range(30):
        x, y = (3 * i) & ~1, (2 * i) & ~1
        buf += crop(Y, W0, x, y, w, h) + crop(U, W0 // 2, x // 2, y // 2, w // 2, h // 2) \
            + crop(V, W0 // 2, x // 2, y // 2, w // 2, h // 2)
    open(f'{sys.argv[2]}/{name}.yuv', 'wb').write(buf)
PY
enc() { # name src w h tile_cols_log2
    aomenc "$out/$2.yuv" --i420 -w "$3" -h "$4" --fps=30/1 --limit=30 --cpu-used=6 \
        --end-usage=q --cq-level=32 --lag-in-frames=0 --threads=8 --tile-columns="$5" \
        --ivf -o "$out/$1.ivf" >/dev/null 2>&1
}
enc 720p 720p 1280 720 0
enc 1080p 1080p 1920 1080 0
enc 1080p_4tile 1080p 1920 1080 2
ls -l "$out"/*.ivf
