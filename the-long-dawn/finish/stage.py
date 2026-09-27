"""THE LONG DAWN · FINISH, the edit's last per-frame picture stage (called from edit/assemble.py Ctx.picture for the
masters). One look for all three films, chosen by the director; per take, a mode from its source's own tone path:

    film  frames that went through lib/look.finish (the Hill curve): the full finish (negative, halation, grain, print)
    ink   RUN-C's ink pages (runC_*, written straight to sRGB, no Hill curve): the emulsion's grain only
    None  slates, black, EDIT proxies: never touched (the caller skips them)

Grain is seeded per (cut, cut frame): it renews every frame, never loops, and is identical across workers, re-runs and
A's two masters. Exposure is fixed (no auto-exposure anywhere), so the finish cannot pump or flicker.
License: GPL-3.0-or-later (see edit/CREDITS.md).
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(HERE, '.numba'))

LOOK = os.environ.get('FINISH_LOOK', '500T_2383_fire')      # the director's pick goes here
INK_LOOK = 'ink_grain'
INK_PREFIXES = ('runC_',)                                  # stems whose frames are plain sRGB (no Hill curve)


def mode_for(take):
    if take is None:
        return None
    stem = os.path.basename(take.get('stem', ''))
    if any(stem.startswith(p) for p in INK_PREFIXES):
        return 'ink'
    return 'film'


class Finisher:
    def __init__(self, look=None):
        import filmfinish as FF
        import filmfast
        self.name = look or LOOK
        self.film = filmfast.Fast(FF.make(self.name))
        self.ink = filmfast.Fast(FF.make(INK_LOOK))

    def __call__(self, img, take, cut, f):
        m = mode_for(take)
        if m == 'film':
            return self.film(img, cut, f)
        if m == 'ink':
            return self.ink(img, cut, f)
        return img


def code_id(look=None):
    """Everything that decides the finished pixels: the code, the look's parameters, the baked LUT bundle."""
    import filmfinish as FF
    name = look or LOOK
    h = hashlib.sha1()
    for fn in ('filmfinish.py', 'filmfast.py', 'stage.py'):
        h.update(open(os.path.join(HERE, fn), 'rb').read())
    for n in (name, INK_LOOK):
        kw = FF.LOOKS[n]
        h.update(json.dumps([n, kw], sort_keys=True, default=str).encode())
        film = kw.get('film')
        if film:
            b = FF.Bundle(film, kw.get('print_', 'kodak_2383'))       # raises if the LUTs are not baked
            h.update(json.dumps([b.film, b.print_, b.le_min, b.le_max, b.d_min.tolist(), b.d_max.tolist(),
                                 float(b.l1.sum()), float(b.l2.sum()), float(b.l3.sum())]).encode())
    return h.hexdigest()[:12]
