"""Default-off entrance for C1680..1687, using the accepted filmed sweep's pixels.

The take starts at C1680 (EDL offset 0); no earlier sweep frame was delivered.
Its reference coverage already equals 0.039861153811 (native book_C_matte/1680,
float32/255 mean, sweepC_before.csv). ftburn.retime seeks that nonzero target
directly, with no entrance interval; its native-flame gate equals one at birth.
The entrance envelope covers the whole baked burn contribution, including
scorch and light, without moving the accepted wipe.

Only the first eight frames are candidates. At C1680 the picture is the exact
held race that ftburn bakes; the envelope reaches the unchanged sweep at C1688.
This is an opacity change, not a measurement or reconstruction of the keyed
hole. It preserves every accepted pixel from C1688, including the exit at1717.
The pinned JPEG inputs are copies of delivered pre-finish frames; do not apply
look.finish here. EDIT applies its normal film finish and captions once.
"""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'assets' / 'burn_footage' / 'sweep_entry'
FIRST, DONE = 1680, 1688
VARIANTS = ('accepted', 'soft-entry')


def entry_opacity(frame):
    """Cubic entrance; zero at1680 and unity at1688, with flat endpoint slopes."""
    if not isinstance(frame, (int, np.integer)):
        raise ValueError('frame must be an integer')
    u = min(1.0, max(0.0, (int(frame) - FIRST) / (DONE - FIRST)))
    return u * u * (3.0 - 2.0 * u)


def composite(rgb, race, frame, candidate='accepted'):
    """Mix the complete baked burn contribution in linear light; never re-time it.

    The exact identity branches avoid transfer-function round trips on accepted
    frames. An opaque output matte is still required: the race is already in RGB.
    """
    if candidate not in VARIANTS:
        raise ValueError('unknown sweep candidate')
    if candidate == 'accepted':
        return rgb
    weight = entry_opacity(frame)
    if frame < FIRST or weight == 1.0:
        return rgb
    for image in (rgb, race):
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3
                or image.dtype != np.float32 or not np.isfinite(image).all()
                or np.any(image < 0) or np.any(image > 1)):
            raise ValueError('sweep inputs must be finite float32 display RGB in [0,1]')
    if rgb.shape != race.shape:
        raise ValueError('sweep and held race must have the same shape')
    if weight == 0.0:
        return race
    # These are display-referred source images, the same inputs ftburn writes.
    # Decode before mixing so the entrance attenuates light, not sRGB code values.
    def linear(image):
        return np.where(image <= .04045, image / 12.92,
                        ((image + .055) / 1.055) ** 2.4)

    mixed = linear(race) * np.float32(1.0 - weight) + linear(rgb) * np.float32(weight)
    out = np.where(mixed <= .0031308, mixed * 12.92,
                   1.055 * np.power(mixed, 1.0 / 2.4) - .055)
    return np.clip(out, 0, 1).astype(np.float32)


def validate_inputs(asset_dir=None):
    """Fail closed on a missing or changed delivered input; no farm cache fallback."""
    directory = Path(asset_dir) if asset_dir is not None else ASSETS
    manifest = json.loads((directory / 'source.json').read_text())
    expected = {'race': ('race_01679.jpg', 'renders/embers_C3/f_01679.jpg')}
    expected.update({str(f): (f'sweep_{f:05d}.jpg', f'renders/book_C_ft/f_{f:05d}.jpg')
                     for f in range(FIRST, DONE)})
    if manifest.get('schema') != 1 or set(manifest.get('inputs', {})) != set(expected):
        raise ValueError('sweep input manifest has the wrong frame set')
    for key, (filename, source) in expected.items():
        row = manifest['inputs'][key]
        if row.get('file') != filename or row.get('source') != source:
            raise ValueError(f'sweep input path mismatch: {key}')
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'sweep input must be a copied regular file: {key}')
        data = path.read_bytes()
        if len(data) != row.get('bytes') or hashlib.sha256(data).hexdigest() != row.get('sha256'):
            raise ValueError(f'sweep input hash mismatch: {key}')
    return manifest


def _scale(scale):
    if not math.isfinite(scale) or not 0 < scale <= 1 or int(804 * scale) < 1:
        raise ValueError('scale must produce positive dimensions, at most native')
    return int(1920 * scale), int(804 * scale)


def _read(path, scale):
    width, height = _scale(scale)
    flag = cv2.IMREAD_REDUCED_COLOR_2 if scale <= .5 else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    expected = (402, 960, 3) if scale <= .5 else (804, 1920, 3)
    if image is None or image.shape != expected:
        raise ValueError(f'sweep source is not a native 1920x804 JPEG: {path}')
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image[..., ::-1].astype(np.float32) / 255.0


def _frame(frame):
    if not isinstance(frame, (int, np.integer)) or not FIRST <= frame < DONE:
        raise ValueError('pinned sweep candidate only supplies C1680..1687')
    return int(frame)


def render_original(frame, scale=1.0):
    """The delivered raw take, with its original opaque matte convention."""
    frame = _frame(frame)
    _scale(scale)
    manifest = validate_inputs()
    rgb = _read(ASSETS / manifest['inputs'][str(frame)]['file'], scale)
    return dict(rgb=rgb, alpha=np.ones(rgb.shape[:2], np.float32))


def make_renderer(candidate='accepted', scale=1.0):
    """Own one held image per callable; accepted never reads that image."""
    if candidate not in VARIANTS:
        raise ValueError('unknown sweep candidate')
    _scale(scale)
    manifest = validate_inputs()
    race = (_read(ASSETS / manifest['inputs']['race']['file'], scale)
            if candidate != 'accepted' else None)

    def render(frame):
        frame = _frame(frame)
        rgb = _read(ASSETS / manifest['inputs'][str(frame)]['file'], scale)
        rgb = composite(rgb, race, frame, candidate)
        return dict(rgb=rgb, alpha=np.ones(rgb.shape[:2], np.float32))

    return render
