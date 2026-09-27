"""THE LONG DAWN v3 - the music cue sheets: music/v3/cues_<cut>.json.

The bar maps (music/v3/barmap_<cut>.json) are LOCKED and owned by SHOWRUNNER-REV ("change both or
neither"); this file never edits them.  A cue sheet annotates them for the score and the effects:
  sections      {section id: {tags, level, lufs [lo, hi] (short-term, final mix), centroid [lo, hi] Hz}}
  events        {sync id: {kind, tags, lock, ... night parameters ...}}
  extra_events  music-only sync points (t in seconds, or {"at": sync id, "beats": +n})
  breaths       silences ending on a sync point
  ambience/sfx  the effects (beds and point events), timed from sync ids
Times given as {"at": id, ...} follow the bar map automatically.

    python cues_v3.py B      # write music/v3/cues_B.json and print the merged map
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timeline_v3 import V3, BarMap, table, validate  # noqa: E402


def at(sid, beats=0.0, s=0.0, frames=0):
    d = {"at": sid}
    if beats:
        d["beats"] = beats
    if s:
        d["s"] = s
    if frames:
        d["frames"] = frames
    return d


# ---------------------------------------------------------------------------
# B . THE KEEPER
# ---------------------------------------------------------------------------
def cues_B():
    C = dict(cut="B", sections={}, events={}, extra_events=[], breaths=[], ambience=[], sfx=[])
    S = C["sections"]

    def sec(sid, tags, level, rel, centroid=None):
        S[sid] = dict(tags=tags, level=level, rel=list(rel))
        if centroid:
            S[sid]["centroid"] = list(centroid)
    # the level map, in LU relative to the film's loudest 3 s (the crane, bar 52): (lo = the section's median
    # may not fall below it, hi = its loudest 3 s may not rise above it).  Relative, so the -16 LUFS master
    # cannot fake it: the quietest score, mf only while the range stands alight, the dawn warm, not loud.
    sec("B1", ["dusk"], "pp", (-15, -8), (150, 700))
    sec("B2", ["climb"], "pp", (-17, -8), (150, 900))
    sec("B3", ["ember", "black"], "niente", (-21, -11))
    sec("B4", ["firstfire"], "pp", (-19, -6))
    sec("B5", ["reveal"], "p", (-10, -2), (150, 800))
    sec("B6", ["firstnight"], "p", (-14, -6), (150, 700))
    sec("B7", ["lifetime"], "p", (-13, -4), (150, 900))
    sec("B8", ["farpeak"], "pp", (-18, -8))
    sec("B9", ["lifetime"], "p", (-11, -3), (150, 1000))
    sec("B10", ["grown"], "p", (-12, -3))
    sec("B11", ["lifetime"], "mp", (-9, -2), (200, 1100))
    sec("B12", ["lifetime", "major"], "mp", (-9, -1.5), (200, 1100))
    sec("B13", ["handback", "dawn", "major"], "mp", (-9, 0), (200, 1500))
    sec("B14", ["title", "major"], "p", (-15, -4))
    E = C["events"]
    for k in range(1, 7):
        E[f"peak_{k}"] = dict(kind="picture", tags=["peak"], owner="RUN-B")
    E["last_red"] = dict(kind="picture", tags=["peak"], owner="RUN-B")
    for k in (1, 2, 3):
        E[f"strike{k}"] = dict(kind="sfx", tags=["strike"], owner="HEROINE")
    E["catch"] = dict(kind="picture", owner="HEROINE")
    E["roar"] = dict(kind="fire", tags=["call"], owner="HEROINE")
    E["call_cello"] = dict(kind="music", lock="hard")
    for sid in ("night_1", "call_2", "call_3"):
        E[sid] = dict(kind="music", tags=["night1_call"])
    # THE VIGIL (director, after the lifetime gate): one longest night in the same frame and the same bars.  The
    # stone_* sync points become the hours: she feeds the fire (a branch on the stack, the flames taking it).
    for sid in ("stone_1", "stone_2", "stone_3", "stone_5", "stone_8", "stone_12", "stone_20", "stone_30", "stone_45"):
        E[sid] = dict(kind="sfx", tags=["feed"], desc="THE VIGIL: an hour gone; she feeds the fire")
    # the chaconne: one two-bar variation per hour.  ground = the tetrachord's first note, dir = down/up.  While
    # nothing answers it steps down a degree each hour (D, C, Bb) and the harmony darkens, over her call alone
    # with the ground (one pad voice).  After the far peak it turns and climbs (G, A, then B: D major arrives with
    # the lights below), and the answers come: ONE far horn (bar 35), again as the traveller carries the flame
    # down; then the torches: the CALL passed from horn to horn down the mountain (whoever lights a fire gets the
    # CALL), with more answers from the far peaks, until the ground arrives on D (bar 46).
    N = [("year_2", "D3", -1, "minor", [], [], "STORM", 0.24, 1),
         ("year_3", "C3", -1, "minor", [], [], "CLEAR", 0.25, 1),
         ("year_5", "Bb2", -1, "minor", [], [], "FOG", 0.23, 1),
         ("year_8", "G2", 1, "minor",
          [dict(at=4.0, horn="hn_farther", mode="minor", rhythm=[1.5, 0.5, 0.5, 1.5], vel=0.3)], [], "CLEAR", 0.26, 2),
         ("year_12", "A2", 1, "minor",
          [dict(at=5.0, horn="hn_far", mode="minor", rhythm=[1, 0.5, 0.5, 1], vel=0.31)],
          [dict(at=4.0, horn="hn2", root="D3", rhythm=[1, 1, 1.5], vel=0.3)], "CLEAR", 0.27, 2),
         ("year_20", "B2", 1, "major",
          [dict(at=6.0, horn="hn_farther", mode="major", rhythm=[0.75, 0.25, 0.25, 0.75], vel=0.28)],
          [dict(at=4.0, horn="hn2", root="D3", rhythm=[1, 1, 1.5], vel=0.32),
           dict(at=5.0, horn="hn_far", root="A2", rhythm=[1, 1, 1.5], vel=0.3)], "CLEAR", 0.28, 2),
         ("year_30", "E2", 1, "major",
          [dict(at=3.0, horn="hn_farther", mode="major", rhythm=[1, 0.5, 0.5, 2], vel=0.31),
           dict(at=5.5, horn="hn5", mode="major", rhythm=[1, 0.5, 0.5, 1], vel=0.28)],
          [dict(at=2.5, horn="hn2", root="D3", rhythm=[1, 1, 1.5], vel=0.34),
           dict(at=4.0, horn="hn", root="A2", rhythm=[1, 1, 1.5], vel=0.33),
           dict(at=5.0, horn="hn_far", root="D3", rhythm=[1, 1, 1.5], vel=0.3)], "CLEAR", 0.3, 3),
         ("year_45", "A2", 1, "major",
          [dict(at=2.5, horn="hn_far", mode="major", rhythm=[1, 0.5, 0.5, 2], vel=0.33),
           dict(at=4.5, horn="hn_farther", mode="major", rhythm=[1, 0.5, 0.5, 1.5], vel=0.3),
           dict(at=5.5, horn="hn6", mode="major", rhythm=[1, 0.5, 0.5, 1], vel=0.27)],
          [dict(at=2.0, horn="hn", root="D3", rhythm=[1, 1, 1.5], vel=0.35),
           dict(at=3.5, horn="hn2", root="A2", rhythm=[1, 1, 1.5], vel=0.33),
           dict(at=5.0, horn="hn5", root="D3", rhythm=[1, 1, 1.5], vel=0.29)], "FOG", 0.31, 3)]
    for sid, g, d, mode, ans, calls, w, lv, npad in N:
        E[sid] = dict(kind="night", len_beats=8, ground=g, dir=d, mode=mode, answers=ans, calls=calls, weather=w,
                      level=lv, call_at=1.0, call_rh={"year_20": [1, 2, 2], "year_45": [1, 1, 1.0]}.get(sid, [1, 1, 1.5]),
                      first_answer=(sid == "year_8"), major_v=(mode == "minor" and d < 0), pad=npad,
                      desc=f"THE VIGIL: hour {N.index((sid, g, d, mode, ans, calls, w, lv, npad)) + 2}; chaconne variation")
    E["year_60"] = dict(kind="music", tags=["arrival"])
    E["y60_strikes"] = dict(kind="sfx", tags=["feed"], desc="THE VIGIL: she feeds the fire; the traveller's child sits by her")
    E["far_peak_1"] = dict(kind="picture", tags=["harmonic"])
    E["first_answer"] = dict(kind="music", tags=["answer"])
    E["his_fire"] = dict(kind="fire", tags=["answer"])
    for k in range(1, 7):
        E[f"pale_{k}"] = dict(kind="picture", tags=["pale"], owner="RUN-B")
    for sid in ("sunrise", "call_late", "home_phrase", "child_wakes", "title", "east_greys", "crane", "range_alight"):
        E.setdefault(sid, dict(kind="music"))
    E["birdsong"] = dict(kind="sfx")
    C["extra_events"] += [
        dict(id="m.dusk_voice", t=at("wind", beats=2.0), kind="music", desc="her cello enters on D4 (the light on every peak)"),
        dict(id="m.climb_out", t=at("climb_close", beats=2.5), kind="music", desc="her held D3 gone under the wind"),
        dict(id="m.flare", t=at("far_peak_2", beats=1.0), kind="music", desc="her light flares: her call, from far"),
        dict(id="m.y60_call", t=at("y60_sits"), kind="music", desc="year 60: her call as she sits; the answers overlap"),
    ]
    C["breaths"].append(dict(t0=at("sunrise", s=-0.275), t1=at("sunrise"), exempt=[], why="275 ms of silence ending on the sunrise"))
    # ---- effects: the wind is the floor (beds overlap by their fades)
    A = C["ambience"]

    def bed(bid, fx, t0, t1, gain_db, fi=1.5, fo=1.5, env=None, **p):
        b = dict(id=bid, fx=fx, t0=t0, t1=t1, gain_db=gain_db, fade_in=fi, fade_out=fo, params=p)
        if env:
            b["env"] = env
        A.append(b)
    bed("B.wind.dusk", "wind", 0.0, at("climb", beats=1), -20, 3.0, 2.0,
        env=[[0, 0], [5, 0], [12, -4], [24, -3]], strength=0.42, howl=0.14, hiss=0.07, cut=560, gust_rate=0.16)
    bed("B.wind.climb", "wind", at("last_red", beats=-2), at("summit", beats=1), -26, 3.0, 2.5,
        strength=0.75, howl=0.38, hiss=0.3, cut=820, gust_rate=0.3)
    bed("B.spindrift", "spindrift", at("climb"), at("summit"), -33, 2.0, 2.0, rate=0.4)
    bed("B.wind.ember", "wind", at("summit", beats=-1), at("roar", beats=1), -30, 2.0, 1.2,
        strength=0.34, howl=0.16, hiss=0.05, cut=500, gust_rate=0.14)
    bed("B.fire.reveal", "fire", at("catch"), at("night_1", beats=1), -27, 0.6, 2.5, rate=11, level=0.8, breath=0.7)
    bed("B.wind.night1", "wind", at("roar", beats=-1), at("year_2", beats=1), -27, 2.5, 2.0,
        strength=0.33, howl=0.1, hiss=0.05, cut=520, gust_rate=0.12)
    # THE VIGIL: her fire burns all night (one bed from the first night to the hand-back), fed on the hours
    bed("B.fire.vigil", "fire", at("night_1"), at("her_light", beats=3), -31, 1.5, 6.0,
        env=[[0, 0], [60, -1], [96, 0]], rate=8, level=0.7, breath=0.8)
    # one wind for the night, with a squall in the second hour (the flame nearly dies) and the fog of the small hours
    bed("B.wind.life", "wind", at("stone_1", beats=1), at("crane", beats=1), -30, 2.0, 2.0,
        strength=0.4, howl=0.16, hiss=0.1, cut=560, gust_rate=0.18)
    for sid, nxt in (("year_2", "year_3"),):
        bed(f"B.storm.{sid}", "wind", at(sid, beats=-0.5), at(nxt, beats=0.5), -21, 0.8, 1.2,
            strength=0.9, howl=0.5, hiss=0.45, cut=1100, gust_rate=0.6)
        bed(f"B.snow.{sid}", "spindrift", at(sid), at(nxt), -26, 0.6, 1.0, rate=0.9)
    for sid in ("far_peak_1", "far_peak_2"):
        bed(f"B.wind.{sid}", "wind", at(sid), at(sid, beats=8), -31, 0.6, 0.8, strength=0.5, howl=0.22, hiss=0.12,
            cut=640, gust_rate=0.2)
    bed("B.wind.high", "wind", at("crane", beats=-1), at("sunrise", beats=0.5), -25, 2.5, 1.5,
        strength=0.5, howl=0.08, hiss=0.22, cut=950, gust_rate=0.1)
    bed("B.air.dawn", "wind", at("sunrise"), at("title", beats=12), -31, 4.0, 5.0,
        env=[[0, 0], [30, -3], [50, -7]], strength=0.22, howl=0.02, hiss=0.02, cut=380, gust_rate=0.08)
    X = C["sfx"]

    def fx(fid, kind, t, gain_db, pan=None, dist=None, exempt=False, **p):
        X.append(dict(id=fid, fx=kind, t=t, gain_db=gain_db, pan=pan, dist=dist, exempt=exempt, params=p))
    fx("B.lid", "knock_lid", at("lid"), -21, pan=-0.05)
    fx("B.blow_ember", "blow", at("blow_ember"), -25, dur=3.2, glow=0.5)
    fx("B.ember_hiss", "hiss", at("ember_last", beats=-1.5), -29, dur=2.2)
    for k in (1, 2, 3):
        fx(f"B.strike{k}", "strike", at(f"strike{k}"), -13 - (k - 1) * 0.5, pan=0.05, bright=1.0 + 0.05 * k)
    fx("B.blow_tinder", "blow", at("blow"), -24, dur=4.0, glow=1.0)
    fx("B.catch", "fire", at("catch"), -26, rate=26, level=1.0, breath=0.3, dur=1.6)
    fx("B.roar", "roar", at("roar"), -5, dur=4.5, lead=0.3, size=1.0, sustain=0.45)
    hours = ["stone_1", "stone_2", "stone_3", "stone_5", "stone_8", "stone_12", "stone_20", "stone_30", "stone_45",
             "y60_strikes"]
    for k, sid in enumerate(hours):
        fx(f"B.feed.{sid}", "feed", at(sid), -19 - (1.0 if k % 2 else 0), pan=0.1)
    fx("B.traveller", "fire", at("traveller"), -28, dur=2.0, rate=14, level=0.9, breath=0.5)
    fx("B.his_fire", "roar", at("his_fire"), -20, pan=0.35, dist=0.6, dur=3.5, lead=0.2, size=0.8, sustain=0.35)
    fx("B.birds", "blackbird", at("birdsong"), -17, phrases=3, dist=0.4)
    fx("B.birds2", "blackbird", at("birdsong", beats=9.5), -23, phrases=2, dist=0.7)
    return C


def _base(cut):
    return dict(cut=cut, sections={}, events={}, extra_events=[], breaths=[], ambience=[], sfx=[])


def _helpers(C):
    def sec(sid, tags, level, rel=None):
        C["sections"][sid] = dict(tags=tags, level=level)
        if rel:
            C["sections"][sid]["rel"] = list(rel)

    def bed(bid, fx, t0, t1, gain_db, fi=1.5, fo=1.5, env=None, **p):
        b = dict(id=bid, fx=fx, t0=t0, t1=t1, gain_db=gain_db, fade_in=fi, fade_out=fo, params=p)
        if env:
            b["env"] = env
        C["ambience"].append(b)

    def fx(fid, kind, t, gain_db, pan=None, dist=None, exempt=False, **p):
        C["sfx"].append(dict(id=fid, fx=kind, t=t, gain_db=gain_db, pan=pan, dist=dist, exempt=exempt, params=p))
    return sec, bed, fx


# ---------------------------------------------------------------------------
# A . EVERY STEP CLOSER  (for now: what the fallback master and the effects floor need; A's score comes next)
# ---------------------------------------------------------------------------
def cues_A():
    C = _base("A")
    sec, bed, fx = _helpers(C)
    for sid, tags, lvl, rel in (
            ("A1", ["black"], "niente", (-40, -14)), ("A2", ["night"], "pp", (-24, -8)),
            ("A3", ["glow"], "p", (-20, -5)), ("A4", ["glow"], "p", (-20, -4)), ("A5", ["lydian"], "mp", (-16, -3)),
            ("A6", ["race"], "p", (-16, -2)), ("A7", ["race", "edge"], "mp", (-12, -1)),
            ("A8", ["race", "brink"], "mf", (-10, 0)), ("A9", ["black"], "niente", (-30, -8)),
            ("A10", ["silence"], "pp", (-40, -10)), ("A11", ["night"], "pp", (-26, -10)),
            ("A12", ["fire"], "pp", (-26, -6)), ("A13", ["night"], "p", (-16, -3)), ("A14", ["night"], "mp", (-14, -2)),
            ("A15", ["night"], "p", (-18, -5)), ("A16", ["night"], "p", (-18, -5)), ("A17", ["night"], "p", (-20, -6)),
            ("A18", ["night"], "p", (-18, -5)), ("A19", ["dawn", "major"], "mp", (-14, -2)),
            ("A20", ["title", "major"], "p", (-20, -5))):
        sec(sid, tags, lvl, rel)
    E = C["events"]
    for sid in ("strike1", "strike2", "strike3"):
        E[sid] = dict(kind="sfx", tags=["strike"], owner="HEROINE")
    E["roar"] = dict(kind="fire", tags=["call"], owner="HEROINE")
    for sid in ("karst_flare", "desert_fire", "run_1", "run_3", "run_5", "run_7"):
        E[sid] = dict(kind="fire", tags=["call"])
    E["run_2"] = dict(kind="fire", tags=["answer"], desc="the ANSWER first heard (the second fire)")
    for sid in ("run_4", "run_6"):
        E[sid] = dict(kind="fire")
    E["watchers"] = dict(kind="music", tags=["answer"])
    for sid in ("watchfire_1", "watchfire_2", "watchfire_3", "watchfire_4"):
        E[sid] = dict(kind="picture", tags=["watchfire"])
    for sid, why, ms in (("ignition", "one beat of breath before the ignition", 833.333),
                         ("edge", "200 ms before the race", 200), ("white_impact", "the suck before the IMPACT", 125),
                         ("set_down", "275 ms before the blue hour", 275)):
        C["breaths"].append(dict(t0=at(sid, s=-ms / 1000), t1=at(sid), exempt=[], why=why))
    bed("A.wind.open", "wind", 0.0, at("white", beats=1), -20, 2.5, 2.0, env=[[0, 0], [8, -2], [20, -8]],
        strength=0.45, howl=0.14, hiss=0.08, cut=560, gust_rate=0.14)
    bed("A.fire.edge", "fire", at("towers"), at("white_impact"), -24, 4.0, 0.2, env=[[0, -8], [24, 0], [44, 4]],
        rate=18, level=1.0, breath=1.2)
    bed("A.storm.brink", "wind", at("vortex"), at("white_impact"), -16, 2.0, 0.1, strength=1.0, howl=0.6, hiss=0.5,
        cut=1300, gust_rate=0.7)
    fx("A.impact", "roar", at("white_impact"), -6, dur=5.0, lead=0.05, size=1.6, sustain=0.2, bright=0.6)
    bed("A.wind.ash", "wind", at("dead_valley", beats=-1), at("silence"), -24, 1.0, 0.3, strength=0.35, howl=0.1,
        hiss=0.3, cut=500)
    bed("A.wind.night", "wind", at("silence", s=2.0), at("roar", beats=1), -24, 3.0, 1.0,
        strength=0.35, howl=0.12, hiss=0.06, cut=520, gust_rate=0.12)
    for k in (1, 2, 3):
        fx(f"A.strike{k}", "strike", at(f"strike{k}"), -12 - (k - 1) * 0.5, pan=0.05)
    fx("A.blow", "blow", at("blow"), -22, dur=4.0, glow=1.0)
    fx("A.roar", "roar", at("roar"), -6, dur=4.5, lead=0.3)
    for sid, dist in (("karst_flare", 0.55), ("desert_fire", 0.5), ("run_1", 0.2), ("run_2", 0.3), ("run_3", 0.35),
                      ("run_4", 0.45), ("run_5", 0.5), ("run_6", 0.6), ("run_7", 0.65)):
        fx(f"A.{sid}", "roar", at(sid), -12 - 6 * dist, dist=dist, dur=3.0, lead=0.15, size=0.8, sustain=0.3)
    bed("A.wind.watch", "wind", at("roar"), at("lantern", beats=1), -26, 2.0, 2.0, strength=0.4, howl=0.14, hiss=0.08)
    bed("A.wind.crossing", "wind", at("lantern"), at("set_down", s=-0.3), -24, 2.0, 0.3, strength=0.5, howl=0.2,
        hiss=0.12, cut=620, gust_rate=0.16)
    for k in range(1, 5):
        fx(f"A.watchfire{k}", "fire", at(f"watchfire_{k}", beats=-2), -26, dur=5.0, rate=10, level=0.8, breath=0.6)
    bed("A.air.blue", "wind", at("set_down"), at("end_fade", beats=1), -30, 3.0, 3.0, strength=0.22, howl=0.02,
        hiss=0.02, cut=380, gust_rate=0.08)
    bed("A.fire.blue", "fire", at("set_down"), at("end_fade", beats=1), -30, 2.0, 3.0, rate=8, level=0.7, breath=0.7)
    return C


# ---------------------------------------------------------------------------
# C . THE LAST PAGES  (for now: what the fallback master and the effects floor need; C's score comes last)
# ---------------------------------------------------------------------------
def cues_C():
    C = _base("C")
    sec, bed, fx = _helpers(C)
    for sid, tags, lvl, rel in (
            ("C1", ["book"], "pp", (-40, -12)), ("C2", ["book"], "pp", (-24, -8)), ("C3", ["book"], "pp", (-24, -8)),
            ("C4", ["book"], "p", (-22, -6)), ("C5", ["lydian"], "p", (-18, -4)), ("C6", ["race"], "p", (-16, -3)),
            ("C7", ["race"], "mp", (-14, -1)), ("C8", ["race"], "p", (-16, -2)), ("C9", ["race", "brink"], "mf", (-12, 0)),
            ("C10", ["night"], "pp", (-24, -8)), ("C11", ["race"], "mp", (-16, -2)), ("C12", ["silence"], "pp", (-40, -10)),
            ("C13", ["night"], "pp", (-24, -8)), ("C14", ["fire"], "pp", (-26, -6)), ("C15", ["night"], "p", (-16, -3)),
            ("C16", ["night"], "p", (-18, -4)), ("C17", ["night"], "mp", (-14, -1)), ("C18", ["major"], "p", (-16, -3)),
            ("C19", ["book"], "p", (-20, -5)), ("C20", ["book"], "p", (-20, -5)), ("C21", ["race"], "mp", (-14, -1)),
            ("C22", ["major"], "mp", (-14, -1)), ("C23", ["major"], "p", (-18, -4)), ("C24", ["dawn", "major"], "mp", (-12, -1)),
            ("C25", ["book", "major"], "p", (-20, -5)), ("C26", ["book"], "p", (-20, -5)), ("C27", ["book"], "pp", (-24, -8)),
            ("C28", ["title", "major"], "p", (-22, -6))):
        sec(sid, tags, lvl, rel)
    E = C["events"]
    E["book"] = dict(kind="music", tags=["call"], desc="the storyteller's CALL (in the finished score: the cor anglais)")
    for sid in ("strike1", "strike2", "strike3"):
        E[sid] = dict(kind="sfx", tags=["strike"], owner="HEROINE")
    E["roar"] = dict(kind="fire", tags=["call"], owner="HEROINE")
    for sid in ("beacon_1", "beacon_3", "beacon_5", "beacon_7"):
        E[sid] = dict(kind="fire", tags=["call"])
    E["beacon_2"] = dict(kind="fire", tags=["answer"], desc="the ANSWER first heard")
    for sid in ("beacon_4", "beacon_6"):
        E[sid] = dict(kind="fire")
    for sid in ("coast_1", "coast_2", "coast_3"):
        E[sid] = dict(kind="fire", tags=["answer"])
    for sid, why, ms in (("race", "200 ms before the race", 200), ("unmade", "275 ms before the Ring loop cadences", 275),
                         ("sunrise", "275 ms before the sunrise", 275)):
        C["breaths"].append(dict(t0=at(sid, s=-ms / 1000), t1=at(sid), exempt=[], why=why))
    bed("C.hearth.open", "hearth", 0.0, at("page_turn", beats=2), -22, 1.0, 2.0)
    for sid in ("page_turn_0", "riffle", "page_turn"):
        fx(f"C.{sid}", "page", at(sid), -16)
    bed("C.fire.born", "fire", at("fire_catches"), at("towers_rise", beats=2), -24, 0.6, 2.0, rate=12, level=0.9, breath=0.8)
    bed("C.fire.forge", "fire", at("towers_rise"), at("deep"), -26, 3.0, 0.5, env=[[0, -6], [20, 0], [26, 3]],
        rate=16, level=1.0, breath=1.2)
    bed("C.storm.eye", "wind", at("eye_burn"), at("mirror"), -18, 2.0, 1.0, strength=0.9, howl=0.5, hiss=0.4, cut=1100,
        gust_rate=0.5)
    bed("C.wind.fall", "wind", at("ring_falls"), at("reveal", beats=4), -25, 2.0, 2.0, strength=0.4, howl=0.14,
        hiss=0.07, cut=520)
    fx("C.snow_hiss", "hiss", at("snow"), -14, dur=2.5)
    for k in (1, 2, 3):
        fx(f"C.strike{k}", "strike", at(f"strike{k}"), -12 - (k - 1) * 0.5, pan=0.05)
    fx("C.blow", "blow", at("blow"), -22, dur=4.0, glow=1.0)
    fx("C.roar", "roar", at("roar"), -6, dur=4.5, lead=0.3)
    for k in range(1, 8):
        d = 0.15 + 0.08 * k
        fx(f"C.beacon{k}", "roar", at(f"beacon_{k}"), -12 - 5 * d, dist=d, dur=3.0, lead=0.15, size=0.8, sustain=0.3)
    bed("C.hearth.council", "fire", at("council"), at("white_heart"), -28, 2.0, 0.3, env=[[0, -4], [30, 0], [36, 5]],
        rate=10, level=0.8, breath=0.9)
    bed("C.air.dawn", "wind", at("sunrise"), at("plenty"), -30, 3.0, 2.0, strength=0.22, howl=0.02, hiss=0.02, cut=380)
    bed("C.hearth.end", "hearth", at("plenty"), at("plagal", beats=4), -26, 2.0, 3.0)
    bed("C.sea", "sea", at("havens"), at("blank"), -24, 2.0, 2.0)
    for sid in ("plenty", "havens", "blank", "blank_2"):
        fx(f"C.page.{sid}", "page", at(sid), -18)
    return C


CUES = {"A": cues_A, "B": cues_B, "C": cues_C}


def write(cut):
    C = CUES[cut]()
    p = os.path.join(V3, f"cues_{cut}.json")
    json.dump(C, open(p, "w"), indent=1)
    bm = BarMap(cut)
    e, w = validate(bm, verbose=True)
    return bm, p, e, w


if __name__ == "__main__":
    for c in [a.upper() for a in sys.argv[1:]] or sorted(CUES):
        bm, p, e, w = write(c)
        print(table(bm))
        print(f"{c}: cue sheet {p}: {len(e)} errors, {len(w)} warnings; {len(bm.d['ambience'])} beds, "
              f"{len(bm.d['sfx'])} effects, {len(bm.breaths)} breaths")
