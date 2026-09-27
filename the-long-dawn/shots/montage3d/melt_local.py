"""MONTAGE-MELT's local test driver: render.py with its own lock file, so a melt test never blocks (or is blocked
by) MONTAGE-3D-4's Ring renders. The Mac's renderq.py still limits heavy work; finals go to the farm.

  python3 ~/mishamisha/_local_logs/renderq.py -- python3 melt_local.py meltc --frames 5420 --scale 0.25 --out t_m0
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402

render.LOCK = os.path.join(HERE, 'cache', 'meltc_blender.lock')
render.other_blenders = lambda: []
if __name__ == '__main__':
    render.main()
