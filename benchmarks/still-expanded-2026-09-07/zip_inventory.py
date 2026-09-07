"""Read a pinned remote ZIP's directory and selected PNG headers with ranges.

No image extraction or pixel decode. Header CRCs are checked; complete member
and archive hashes are not claimed. Run before choosing benchmark workloads.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--headers', type=int, default=64)
a = p.parse_args()
root = a.work_dir
info = json.loads((root / 'div8k-mirror-info.json').read_text())
entry = next(r for r in json.loads((root / 'div8k-mirror-tree.json').read_text()) if r['path'] == 'DIV8K.zip')
url = f'https://huggingface.co/datasets/Iceclear/DIV8K_TrainingSet/resolve/{info["sha"]}/DIV8K.zip'
out = root / 'div8k-inventory'
out.mkdir()
requests = []


class Ranges(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.size = entry['size']
        tail = (root / 'div8k-zip-tail.bin').read_bytes()
        self.cache = [(self.size - len(tail), tail)]

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = [0, self.position, self.size][whence] + offset
        assert 0 <= self.position <= self.size
        return self.position

    def read(self, n=-1):
        start = self.position
        if n < 0:
            n = self.size - start
        n = min(n, self.size - start)
        assert n <= 2_000_000, 'metadata reader refuses a large transfer'
        if not n:
            return b''
        for offset, data in self.cache:
            if offset <= start and start + n <= offset + len(data):
                self.position += n
                return data[start - offset:start - offset + n]
        end = start + n - 1
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read(n + 1)
            assert response.status == 206
            assert response.headers['Content-Range'] == f'bytes {start}-{end}/{self.size}'
            assert len(data) == n
        name = f'{start}-{end}.bin'
        (out / name).write_bytes(data)
        requests.append(dict(start=start, end=end, bytes=n, file=name,
                             sha256=hashlib.sha256(data).hexdigest()))
        (out / 'ranges.json').write_text(json.dumps(requests, indent=2) + '\n')
        self.cache.append((start, data))
        self.position += n
        return data


with zipfile.ZipFile(Ranges()) as archive:
    members = [dict(name=m.filename, bytes=m.file_size, compressed_bytes=m.compress_size,
                    compression=m.compress_type, crc32=f'{m.CRC:08x}', header_offset=m.header_offset)
               for m in archive.infolist()]
    (out / 'members.json').write_text(json.dumps(members, indent=2) + '\n')
    images = [m for m in archive.infolist() if m.filename.lower().endswith('.png')]
    # Predeclared order depends on source name only, not pixels or timings.
    images.sort(key=lambda m: hashlib.sha256(('rav1d-still-source-screen-v1:' + m.filename).encode()).digest())
    headers = []
    for member in images[:a.headers]:
        with archive.open(member) as source:
            header = source.read(33)
        assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[8:16] == b'\x00\x00\x00\rIHDR'
        assert zlib.crc32(header[12:29]) == struct.unpack('>I', header[29:33])[0]
        width, height = struct.unpack('>II', header[16:24])
        row = dict(name=member.filename, width=width, height=height,
                   bit_depth=header[24], color_type=header[25],
                   header_sha256=hashlib.sha256(header).hexdigest(),
                   landscape_uhd=width >= 7680 and height >= 4320,
                   portrait_uhd=width >= 4320 and height >= 7680)
        headers.append(row)
        (out / 'headers.json').write_text(json.dumps(headers, indent=2) + '\n')
        print(member.filename, f'{width}x{height}', flush=True)

record = dict(url=url, mirror_revision=info['sha'], archive_bytes=entry['size'],
              expected_archive_sha256=entry['lfs']['oid'], complete_archive_hash_verified=False,
              png_members=len(images), header_candidates=len(headers),
              eligible_headers=sum(r['landscape_uhd'] or r['portrait_uhd'] for r in headers),
              transferred_bytes=sum(r['bytes'] for r in requests), seeded_tail_bytes=65536,
              frozen_workload_selection=False,
              scope='Directory and CRC-checked PNG headers only; no pixel or decode-performance inspection')
(out / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
