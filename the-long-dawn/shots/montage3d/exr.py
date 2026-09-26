"""Minimal OpenEXR reader (scanline files, NONE / ZIPS / ZIP compression, HALF or FLOAT channels).

Blender's multilayer EXRs name channels 'ViewLayer.Combined.R', 'ViewLayer.Depth.Z', ...
read_exr(path) -> {channel_name: float32 (H, W) array}.
"""
import struct
import zlib

import numpy as np

_BLOCK_LINES = {0: 1, 2: 1, 3: 16}          # NONE, ZIPS, ZIP


def _header(f):
    if f[:4] != b'v/1\x01':
        raise ValueError('not an EXR file')
    p = 8
    hdr = {}
    while True:
        e = f.index(b'\0', p)
        name = f[p:e].decode()
        p = e + 1
        if not name:
            break
        e = f.index(b'\0', p)
        typ = f[p:e].decode()
        p = e + 1
        sz = struct.unpack('<i', f[p:p + 4])[0]
        p += 4
        val = f[p:p + sz]
        p += sz
        if typ == 'chlist':
            chs = []
            q = 0
            while val[q] != 0:
                e = val.index(b'\0', q)
                n = val[q:e].decode()
                q = e + 1
                pt = struct.unpack('<i', val[q:q + 4])[0]
                q += 16
                chs.append((n, pt))
            hdr[name] = chs
        elif typ == 'compression':
            hdr[name] = val[0]
        elif typ == 'box2i':
            hdr[name] = struct.unpack('<4i', val)
        else:
            hdr[name] = (typ, val)
    return hdr, p


def _unzip(raw, nbytes):
    d = np.frombuffer(zlib.decompress(raw), np.uint8)
    # undo the predictor: running sum of deltas (each stored + 128) mod 256
    t = np.empty_like(d)
    t[0] = d[0]
    t[1:] = (d[1:].astype(np.int32) - 128) & 0xFF
    t = np.cumsum(t, dtype=np.uint64).astype(np.uint8) if False else _cumsum_u8(t)
    # de-interleave: first half = even bytes, second half = odd bytes
    half = (len(t) + 1) // 2
    out = np.empty_like(t)
    out[0::2] = t[:half]
    out[1::2] = t[half:]
    return out


def _cumsum_u8(t):
    return (np.cumsum(t.astype(np.int64)) & 0xFF).astype(np.uint8)


def read_exr(path, want=None):
    """Read all (or the `want`-listed) channels of a scanline EXR as float32 (H, W) arrays."""
    with open(path, 'rb') as fh:
        f = fh.read()
    hdr, p = _header(f)
    comp = hdr['compression']
    if comp not in _BLOCK_LINES:
        raise ValueError(f'unsupported EXR compression {comp}')
    x0, y0, x1, y1 = hdr['dataWindow']
    W, H = x1 - x0 + 1, y1 - y0 + 1
    chs = hdr['channels']                      # already sorted by name in the file
    sizes = [2 if pt == 1 else 4 for _, pt in chs]
    line_bytes = sum(sizes) * W
    bl = _BLOCK_LINES[comp]
    nblocks = (H + bl - 1) // bl
    offs = struct.unpack(f'<{nblocks}Q', f[p:p + 8 * nblocks])
    buf = np.empty((H, line_bytes), np.uint8)
    for off in offs:
        y, n = struct.unpack('<ii', f[off:off + 8])
        raw = f[off + 8:off + 8 + n]
        lines = min(bl, y1 - y + 1)
        expect = lines * line_bytes
        if comp == 0 or n == expect:
            data = np.frombuffer(raw, np.uint8)
        else:
            data = _unzip(raw, expect)
        buf[y - y0:y - y0 + lines] = data.reshape(lines, line_bytes)
    out = {}
    col = 0
    for (name, pt), sz in zip(chs, sizes):
        seg = buf[:, col:col + sz * W]
        col += sz * W
        if want is not None and name not in want:
            continue
        dt = np.float16 if pt == 1 else np.float32
        out[name] = np.ascontiguousarray(seg).view(dt).reshape(H, W).astype(np.float32)
    return out


def layer_rgb(ch, layer='ViewLayer.Combined'):
    """Stack a layer's R, G, B channels into an (H, W, 3) float32 image."""
    return np.dstack([ch[f'{layer}.R'], ch[f'{layer}.G'], ch[f'{layer}.B']]).astype(np.float32)
