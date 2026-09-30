"""Output isolation and atomic frame publication for the explicit D renderer."""
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import openring_d as D


def output_dir(shot, out, root):
    """Validate a new D render directory before a scene can construct its cache.

    Relative outputs are repo-relative and must name this shot's new render
    directory. An absolute directory outside the repository is permitted for
    look development. Neither route may pass through a symlink. The shared
    ``renders`` root is checked even for an external output, because the tower
    constructor uses it for its inherited procedural cache.
    """
    if shot not in D.SHOTS:
        raise ValueError(f'Unknown D shot: {shot}')
    root = Path(root).resolve()
    renders = root / 'renders'
    if renders.is_symlink() or (renders.exists() and not renders.is_dir()):
        raise ValueError('Refusing a linked or non-directory renders root')
    out = Path(out)
    if '..' in out.parts:
        raise ValueError('Output may not contain parent traversal')
    external = out.is_absolute()
    path = out if external else root / out
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Output may not be a symlink or traverse one')
    path = path.absolute()
    inside = path == root or root in path.parents
    if inside or not external:
        if path.parent != renders or path.name != f'embers_D_{shot}':
            raise ValueError(f'Use renders/embers_D_{shot} or an absolute external look-development directory')
    if path.exists() and not path.is_dir():
        raise ValueError('Output must be a directory')
    return path


def publish(path, rgb):
    """Encode a complete RGB frame, then publish without replacing any target.

    Input is finite unit-range RGB (values outside the unit range are clipped).
    The caller creates the already-validated parent directory. A same-directory
    temporary file keeps a watcher from seeing an incomplete PNG or JPEG, and
    the hard link atomically fails if any file or symlink already owns the name.
    """
    path = Path(path)
    formats = {'.png': 'PNG', '.jpg': 'JPEG'}
    if path.suffix.lower() not in formats:
        raise ValueError('Frame output must be PNG or JPG')
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError('Frame must be a nonempty H x W x 3 RGB array')
    if not np.isfinite(array).all():
        raise ValueError('Frame contains non-finite RGB values')
    pixels = np.rint(np.clip(array, 0., 1.) * 255.).astype(np.uint8)
    image_format = formats[path.suffix.lower()]
    options = {'quality': 95, 'subsampling': 0} if image_format == 'JPEG' else {}
    with tempfile.NamedTemporaryFile(prefix='.' + path.name + '.', suffix='.tmp',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            Image.fromarray(pixels).save(stream, format=image_format, **options)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
