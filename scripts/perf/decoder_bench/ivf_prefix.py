#!/usr/bin/env python3
"""ivf_prefix.py IN.ivf OUT.ivf N : keep the first N frames of an IVF (fixes the header count)."""
import struct, sys
d = open(sys.argv[1], 'rb').read(); n = int(sys.argv[3])
hl = struct.unpack('<H', d[6:8])[0]; pos = hl; kept = 0
while kept < n and pos + 12 <= len(d):
    pos += 12 + struct.unpack('<I', d[pos:pos + 4])[0]; kept += 1
out = bytearray(d[:pos]); out[24:28] = struct.pack('<I', kept)
open(sys.argv[2], 'wb').write(out); print(f'{sys.argv[2]}: {kept} frames, {len(out)} bytes')
