"""THE LONG DAWN v3 - the time grid and the bar maps.

One grid for all three films (BIBLE_V3 section 9): 24 fps, 72 BPM, 4/4 with no tempo change.
1 beat = 20 frames = 0.8333 s = 40 000 samples @ 48 kHz; 1 bar = 80 frames = 3.3333 s.

The bar map of each cut (music/v3/barmap_<cut>.json) is the contract between picture and
music.  Canonical time of every event is `frame` (24 fps, exact) when it lies on a frame, and
`t` (seconds, 4 decimals) otherwise (e.g. a breath that starts 275 ms before a downbeat).
Global beats are 0-based quarter notes from the top of the cut:  gb(bar, beat) = (bar-1)*4 + beat-1.

    python timeline_v3.py B          # validate music/v3/barmap_B.json and print it as a table
    python timeline_v3.py --all
"""
import json
import os
import sys

SR = 48000
FPS = 24
FR = SR // FPS                 # 2000 samples per frame
BPM = 72
BEAT_S = 60.0 / BPM            # 0.8333 s
BEAT_N = 40000                 # samples per beat
BEAT_F = 20                    # frames per beat
BAR_F = 80                     # frames per bar
BAR_S = 4 * BEAT_S

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
V3 = os.path.join(MUSIC, "v3")


def gb(bar, beat=1.0):
    """global beat of (1-based) bar and (1-based, fractional) beat"""
    return (bar - 1) * 4 + (beat - 1)


def fb(frame):
    """global beat of a frame"""
    return frame / 20.0


def tb(t):
    """global beat of a time in seconds"""
    return t / BEAT_S


def bt(beat):
    """seconds of a global beat"""
    return beat * BEAT_S


def bar_beat(beat):
    """(bar, beat-in-bar) of a global beat, both 1-based"""
    beat = round(beat, 6)
    return int(beat // 4) + 1, round(beat % 4 + 1, 6)


def frame_of(t):
    return int(round(t * FPS))


def on_frame(t, tol=1e-3):
    return abs(t * FPS - round(t * FPS)) < tol * FPS


class BarMap:
    """A cut's bar map: sections (whole bars) and events (sync points)."""

    def __init__(self, cut, path=None, cues=None):
        self.cut = cut.upper()
        self.path = path or os.path.join(V3, f"barmap_{self.cut}.json")
        d = json.load(open(self.path))
        if "sync" in d and "events" not in d:
            d = _from_showrunner(d, cues or os.path.join(V3, f"cues_{self.cut}.json"))
        self.d = d
        self.bars = int(d["bars"])
        self.frames = self.bars * BAR_F
        self.n = self.frames * FR               # samples, exactly the cut's length
        self.render_n = self.n + SR             # 1 s of tail margin while rendering
        self.seconds = self.frames / FPS
        self.sections = d["sections"]
        for s in self.sections:
            s["b0"] = gb(s["bar_start"])
            s["b1"] = gb(s["bar_end"] + 1)
            s.setdefault("tags", [])
        self.events = d.get("events", [])
        for e in self.events:
            # exact on the frame grid when the event sits on a frame
            e["beat"] = e["frame"] / 20.0 if (e.get("frame") is not None and on_frame(e["t"])) else round(tb(e["t"]), 6)
            e.setdefault("tags", [])
        self.breaths = d.get("breaths", [])

    # -- lookup --------------------------------------------------------------
    def section(self, sid):
        for s in self.sections:
            if s["id"] == sid:
                return s
        raise KeyError(sid)

    def sec_at(self, beat):
        for s in self.sections:
            if s["b0"] <= beat < s["b1"]:
                return s
        return self.sections[-1]

    def event(self, eid):
        for e in self.events:
            if e["id"] == eid:
                return e
        raise KeyError(eid)

    def ev(self, eid):
        """global beat of an event"""
        return self.event(eid)["beat"]

    def find(self, kind=None, tag=None, section=None):
        out = []
        for e in self.events:
            if kind and e.get("kind") != kind:
                continue
            if tag and tag not in e.get("tags", []):
                continue
            if section and e.get("section") != section:
                continue
            out.append(e)
        return sorted(out, key=lambda e: e["t"])

    def with_tag(self, tag):
        return [s for s in self.sections if tag in s.get("tags", [])]

    def breath_beats(self):
        """[(b0, b1, exempt_set)] for the render engine"""
        return [(round(tb(b["t0"]), 6), round(tb(b["t1"]), 6), set(b.get("exempt", []))) for b in self.breaths]


def _resolve_t(x, sync):
    """a time given as seconds, or as {"at": sync_id, "beats": +n, "s": +x, "frames": +f}"""
    if isinstance(x, dict):
        e = sync[x["at"]]
        return e["f"] / FPS + x.get("beats", 0.0) * BEAT_S + x.get("s", 0.0) + x.get("frames", 0) / FPS
    return float(x)


def _from_showrunner(d, cues_path):
    """The locked bar maps (SHOWRUNNER-REV: sections + sync + text) merged with the music cue sheet
    (music/v3/cues_<cut>.json: section tags/levels, event annotations, music-only events, breaths, effects).
    The bar map itself is never modified."""
    cues = json.load(open(cues_path)) if os.path.exists(cues_path) else {}
    sync = {e["id"]: e for e in d["sync"]}
    out = dict(cut=d["cut"], title=d.get("title", ""), fps=d["fps"], bpm=d["bpm"], bars=d["bars"],
               status="locked (SHOWRUNNER-REV)" + (" + cues" if cues else ""), text=d.get("text", []))
    secs = []
    for sec in d["sections"]:
        x = dict(sec)
        x.update(cues.get("sections", {}).get(sec["id"], {}))
        x.setdefault("tags", [])
        secs.append(x)
    out["sections"] = secs
    evs = []
    ann = cues.get("events", {})
    for e in d["sync"]:
        x = dict(id=e["id"], t=e["f"] / FPS, frame=e["f"], bar=e["bar"], bar_beat=e["beat"], desc=e.get("what", ""),
                 kind="picture", lock="hard", tags=[])
        x.update(ann.get(e["id"], {}))
        evs.append(x)
    for e in cues.get("extra_events", []):
        x = dict(e)
        x["t"] = _resolve_t(e["t"], sync)
        fr = x["t"] * FPS
        x["frame"] = int(round(fr)) if abs(fr - round(fr)) < 1e-3 else None
        x.setdefault("lock", "soft")
        x.setdefault("tags", [])
        evs.append(x)
    out["events"] = sorted(evs, key=lambda e: e["t"])
    out["breaths"] = [dict(t0=_resolve_t(b["t0"], sync), t1=_resolve_t(b["t1"], sync), exempt=b.get("exempt", []),
                           why=b.get("why", "")) for b in cues.get("breaths", [])]
    out["ambience"] = [dict(b, t0=_resolve_t(b["t0"], sync), t1=_resolve_t(b["t1"], sync))
                       for b in cues.get("ambience", [])]
    out["sfx"] = [dict(b, t=_resolve_t(b["t"], sync)) for b in cues.get("sfx", [])]
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate(bm, verbose=True):
    errs, warns = [], []
    d = bm.d
    if d.get("fps", FPS) != FPS or d.get("bpm", BPM) != BPM:
        errs.append("fps/bpm must be 24/72")
    # sections: contiguous whole bars from 1 to bars
    nxt = 1
    for s in bm.sections:
        if s["bar_start"] != nxt:
            errs.append(f"section {s['id']} starts at bar {s['bar_start']}, expected {nxt}")
        if s["bar_end"] < s["bar_start"]:
            errs.append(f"section {s['id']} ends before it starts")
        t0, t1 = (s["bar_start"] - 1) * BAR_S, s["bar_end"] * BAR_S
        if abs(s.get("t0", t0) - t0) > 0.051 or abs(s.get("t1", t1) - t1) > 0.051:
            errs.append(f"section {s['id']}: t0/t1 {s.get('t0')}-{s.get('t1')} != bars "
                        f"{s['bar_start']}-{s['bar_end']} ({t0:.3f}-{t1:.3f})")
        nxt = s["bar_end"] + 1
    if nxt != bm.bars + 1:
        errs.append(f"sections end at bar {nxt - 1}, map has {bm.bars} bars")
    ids = set()
    for e in bm.events:
        if e["id"] in ids:
            errs.append(f"duplicate event id {e['id']}")
        ids.add(e["id"])
        t = e["t"]
        if not (0 <= t <= bm.seconds + 1e-6):
            errs.append(f"event {e['id']} at {t} s is outside the cut")
        if "frame" in e and e["frame"] is not None and abs(e["frame"] - t * FPS) > 0.51:
            errs.append(f"event {e['id']}: frame {e['frame']} != t {t} s")
        if "bar" in e:
            b, bb = bar_beat(tb(t))
            if e["bar"] != b or abs(e.get("bar_beat", bb) - bb) > 0.02:
                errs.append(f"event {e['id']}: bar {e['bar']}.{e.get('bar_beat')} != {b}.{bb:.2f}")
        sec = bm.sec_at(e["beat"] - 1e-9 if t >= bm.seconds else e["beat"])
        if e.get("section") and e["section"] != sec["id"] and abs(t - sec["t0"] if "t0" in sec else 0) > 1e-3:
            warns.append(f"event {e['id']} ({t:.3f} s) tagged {e['section']} but falls in {sec['id']}")
        if e.get("lock") == "hard" and not on_frame(t):
            warns.append(f"hard event {e['id']} at {t} s is not on a frame")
    for b in bm.breaths:
        if not (b["t1"] > b["t0"]):
            errs.append(f"breath {b} is empty")
    if verbose:
        for x in errs:
            print("ERROR", x)
        for x in warns:
            print("warn ", x)
    return errs, warns


def table(bm):
    lines = [f"{bm.cut}: {bm.bars} bars = {bm.frames} frames = {bm.seconds:.3f} s = {bm.n} samples"]
    for s in bm.sections:
        lines.append(f"  {s['id']:5s} bars {s['bar_start']:3d}-{s['bar_end']:3d}  {s['t0']:7.3f}-{s['t1']:7.3f} s  "
                     f"{s.get('name', '')[:40]:40s} {','.join(s.get('tags', []))}")
        for e in bm.events:
            if s["b0"] - 1e-6 <= e["beat"] < s["b1"] - 1e-6 or (s is bm.sections[-1] and e["beat"] >= s["b1"] - 1e-6):
                b, bb = bar_beat(e["beat"])
                lines.append(f"        {e['t']:8.3f} s  f{e['t'] * FPS:7.1f}  {b:3d}.{bb:<5.2f} {e.get('kind', ''):8s} "
                             f"{e.get('lock', ''):4s} {e['id']}: {e.get('desc', '')[:70]}")
    return "\n".join(lines)


if __name__ == "__main__":
    cuts = ["A", "B", "C"] if (len(sys.argv) < 2 or sys.argv[1] == "--all") else [sys.argv[1].upper()]
    bad = 0
    for c in cuts:
        p = os.path.join(V3, f"barmap_{c}.json")
        if not os.path.exists(p):
            print(f"{c}: no bar map yet ({p})")
            continue
        bm = BarMap(c)
        e, w = validate(bm)
        print(table(bm))
        print(f"{c}: {len(e)} errors, {len(w)} warnings\n")
        bad += len(e)
    sys.exit(1 if bad else 0)
