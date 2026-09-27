#!/bin/bash
# THE LONG DAWN · FINISH: bake spektrafilm's 3-LUT bundles (L1 ACEScct -> film log E, L2 -> negative density,
# L3 -> 2383/2393 print + scan -> sRGB) into finish/luts/. About 35 s per stock on the Mac; the cubes are ~11 MB each
# and are NOT committed (finish/luts/ is git-ignored; the .npy caches beside them are written on first use).
#
# spektrafilm v0.3.3 (GPLv3; its LUTs are CC BY-SA 4.0, credit in edit/CREDITS.md) needs Python 3.13, so it lives in
# its own venv, ~/.venvs/finish (never in ~/.venvs/longdawn: the render lanes' numba stack stays untouched):
#   uv venv ~/.venvs/finish --python 3.13
#   VIRTUAL_ENV=~/.venvs/finish uv pip install --no-deps "git+https://github.com/andreavolpato/spektrafilm@v0.3.3"
#   VIRTUAL_ENV=~/.venvs/finish uv pip install --only-binary :all: "numpy~=2.4" "scipy~=1.17" "colour-science~=0.4.6" \
#       "scikit-image~=0.26" "matplotlib~=3.10" "opt-einsum~=3.4.0" "numba~=0.64" "Pillow~=12.1" "markdown~=3.7" \
#       opencolorio==2.5.2 "opencv-python-headless==4.10.0.84" "OpenImageIO~=3.1.11"
#   then the shims below (macOS 12 has no wheels for pyfftw/lensfunpy/rawpy/exiv2; only the RAW-file path uses the
#   last three, and pyfftw's numpy_fft interface maps 1:1 onto scipy.fft).
set -euo pipefail
cd "$(dirname "$0")"
PY=~/.venvs/finish/bin/python
SP=$($PY -c 'import site; print(site.getsitepackages()[0])')
if [ ! -f "$SP/pyfftw/interfaces/numpy_fft.py" ]; then
  mkdir -p "$SP/pyfftw/interfaces"; : > "$SP/pyfftw/__init__.py"; : > "$SP/pyfftw/interfaces/__init__.py"
  cat > "$SP/pyfftw/interfaces/numpy_fft.py" <<'PYEOF'
"""Shim: pyfftw.interfaces.numpy_fft -> scipy.fft (drops pyFFTW-only kwargs such as planner_effort/threads)."""
import scipy.fft as _f
_DROP = ('planner_effort', 'threads', 'auto_align_input', 'auto_contiguous', 'overwrite_input')
def _wrap(fn):
    def g(*a, **k):
        for d in _DROP:
            k.pop(d, None)
        return fn(*a, **k)
    return g
for _n in ('fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn', 'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn'):
    globals()[_n] = _wrap(getattr(_f, _n))
fftfreq, rfftfreq, fftshift, ifftshift = _f.fftfreq, _f.rfftfreq, _f.fftshift, _f.ifftshift
PYEOF
  for m in exiv2 lensfunpy rawpy; do echo '"""Shim: only spektrafilm RAW-file loading uses this module."""' > "$SP/$m.py"; done
fi
FILMS=${FILMS:-"kodak_vision3_500t kodak_vision3_250d kodak_vision3_50d"}
for film in $FILMS; do
  ~/.venvs/finish/bin/spektrafilm-lut build --film "$film" --print kodak_2383 --print kodak_2393 \
    --input acescct --output srgb --topology 3lut --resolution 65 --out ./luts/
done
