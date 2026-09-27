"""MONTAGE-MELT's render driver: render.py with its own lock file (a melt test never blocks, or is blocked by,
MONTAGE-3D-4's Ring renders), plus an output-existence check. The common driver now propagates Blender and
post-processing failures and requires every requested frame to be posted. Used locally and by cloud/jobs/meltC.json.

  python3 ~/mishamisha/_local_logs/renderq.py -- python3 melt_local.py meltc --frames 5420 --scale 0.25 --out t_m0
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402

render.LOCK = os.path.join(HERE, 'cache', 'meltc_blender.lock')
render.other_blenders = lambda: []


def _wanted(argv):
    fr, out, final, outdir = None, None, False, None
    for i, a in enumerate(argv):
        if a == '--frames':
            fr = [int(x) for x in argv[i + 1].split(',')]
        elif a == '--range':
            s, e = argv[i + 1].split('-')
            fr = list(range(int(s), int(e) + 1))
        elif a == '--out':
            out = argv[i + 1]
        elif a == '--outdir':
            outdir = argv[i + 1]
        elif a == '--final':
            final = True
    if final and outdir:
        d = os.path.join(render.ROOT, outdir)
    elif final:
        d = render.FINAL_DIR
    else:
        d = os.path.join(HERE, 'tests', out or 'meltc_test')
    return fr or [], d


if __name__ == '__main__':
    render.main()
    frames, d = _wanted(sys.argv[1:])
    miss = [f for f in frames if not any(os.path.exists(os.path.join(d, f'f_{f:05d}{e}')) for e in ('.png', '.jpg'))]
    if miss:
        log = os.path.join(HERE, 'cache', 'meltc')
        print(f'MISSING {len(miss)} frame(s): {miss[:12]} (see the blender_*.log in {log})', flush=True)
        sys.exit(1)
