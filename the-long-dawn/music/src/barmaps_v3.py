"""THE LONG DAWN v3 - writes music/v3/barmap_{A,B,C}.json from the revised beat sheets (BIBLE_V3, REVISION 1).

    python barmaps_v3.py B        # write + validate one map
    python barmaps_v3.py          # all that are defined

This file is the single source of the bar maps: sections (whole bars), events (sync points), the
breaths, and the effects (ambience beds + point events) that the fallback masters and the finals use.

Event fields: id, t (s), frame, bar, bar_beat, kind (picture | sfx | music | fire | text | night),
lock (hard: picture and music both hold to it / soft: picture may move it +-0.5 s, effects follow picture),
owner (the lane that must honour it), desc, tags.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timeline_v3 import BAR_S, BEAT_S, FPS, V3, BarMap, bar_beat, tb, validate, table  # noqa: E402


def T(bar, beat=1.0):
    """seconds of (bar, beat), 1-based"""
    return round(((bar - 1) * 4 + (beat - 1)) * BEAT_S, 4)


def F(frame):
    return round(frame / FPS, 4)


class Map:
    def __init__(self, cut, title, bars, source):
        self.d = dict(cut=cut, title=title, fps=24, bpm=72, meter="4/4", beat_frames=20, bar_frames=80, bars=bars,
                      frames=bars * 80, seconds=round(bars * BAR_S, 4), samples_48k=bars * 80 * 2000,
                      status="locked", source=source, sections=[], events=[], breaths=[], ambience=[], sfx=[],
                      notes=[])

    def sec(self, sid, name, b0, b1, tags=(), level="pp", picture="", music="", text=""):
        self.d["sections"].append(dict(id=sid, name=name, bar_start=b0, bar_end=b1, t0=T(b0), t1=T(b1 + 1),
                                       f0=(b0 - 1) * 80, f1=b1 * 80, tags=list(tags), level=level,
                                       picture=picture, music=music, text=text))

    def ev(self, eid, t, kind, desc, lock="hard", owner="", tags=(), **kw):
        b, bb = bar_beat(tb(t))
        fr = t * FPS
        e = dict(id=eid, t=round(t, 4), frame=int(round(fr)) if abs(fr - round(fr)) < 1e-3 else None, bar=b,
                 bar_beat=round(bb, 3), kind=kind, lock=lock, owner=owner, desc=desc, tags=list(tags))
        e.update(kw)
        self.d["events"].append(e)
        return t

    def breath(self, t_down, ms, exempt=(), why=""):
        self.d["breaths"].append(dict(t0=round(t_down - ms / 1000.0, 4), t1=round(t_down, 4), exempt=list(exempt),
                                      why=why))

    def bed(self, bid, fx, t0, t1, gain_db, fade_in=1.5, fade_out=1.5, env=None, **params):
        b = dict(id=bid, fx=fx, t0=round(t0, 4), t1=round(t1, 4), gain_db=gain_db, fade_in=fade_in,
                 fade_out=fade_out, params=params)
        if env:
            b["env"] = env
        self.d["ambience"].append(b)

    def fx(self, fid, fx, t, gain_db, pan=None, dist=None, exempt=False, **params):
        self.d["sfx"].append(dict(id=fid, fx=fx, t=round(t, 4), gain_db=gain_db, pan=pan, dist=dist,
                                  exempt=exempt, params=params))

    def write(self, path=None):
        self.d["events"].sort(key=lambda e: e["t"])
        p = path or os.path.join(V3, f"barmap_{self.d['cut']}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(self.d, open(p, "w"), indent=1)
        bm = BarMap(self.d["cut"], p)
        e, w = validate(bm)
        print(table(bm))
        print(f"{self.d['cut']}: {len(e)} errors, {len(w)} warnings -> {p}")
        return bm


CUTS = {}


def cut(fn):
    CUTS[fn.__name__[-1]] = fn
    return fn


if __name__ == "__main__":
    which = [a.upper() for a in sys.argv[1:]] or sorted(CUTS)
    for c in which:
        if c in CUTS:
            CUTS[c]().write()
        else:
            print(f"{c}: not defined yet")
