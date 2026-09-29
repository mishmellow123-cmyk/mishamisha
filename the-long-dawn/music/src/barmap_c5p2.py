"""THE LONG DAWN C5, score PASS 2: the bar map and cue sheet re-timed to the MEASURED picture (SOUND-SCORE-C, 29 Sep).

    python barmap_c5p2.py            # -> music/v3/barmap_C5P2.json + music/v3/cues_C5P2.json (then validates them)
    python barmap_c5p2.py --check    # exit 1 if the committed files differ from what the sources now generate

Pass 1 (score_v3_C5.py on barmap_C5.json / cues_C5.json) stays exactly as it was: those files are read, never
written. Pass 2 is a separate cut id, C5P2, so `python render_v3.py C5P2` renders it next to final_C5 without touching
it. Its bar map is barmap_C5 with every picture event re-timed from music/v3/events_C5_measured.json (the delivered
frames, measured). Each sync entry says where its frame came from:

    source = "measured"   the frame IS a measured picture frame (measured_ref names the event and field)
             "verified"   pass 1's frame lies inside the measured window of that event, so it is kept
             "derived"    a musical placement computed from measured frames (derived_from says how)
             "pass 1"     not measurable on this Mac (FLINT, the BEACON RUN, the second half's dawn...): unchanged

Nothing here invents a frame: an event the picture cannot place keeps pass 1's value and says so.
"""
import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timeline_v3 import BAR_F, BEAT_F, FPS, V3, BarMap, validate  # noqa: E402

SRC_MAP = os.path.join(V3, "barmap_C5.json")
SRC_CUES = os.path.join(V3, "cues_C5.json")
MEASURED = os.path.join(V3, "events_C5_measured.json")
OUT_MAP = os.path.join(V3, "barmap_C5P2.json")
OUT_CUES = os.path.join(V3, "cues_C5P2.json")

# pass 1's (est.) events re-timed to a measured frame: sync id -> (measured event id, frames field, what it is now)
MEASURED_MOVES = {
    "low_fire": ("trap.low_forge_sinks", "first", "one forge's flame sinks almost out: it tries to stop alone"),
    "surge": ("trap.forges_surge", "first", "at once every other forge surges up"),
    "flares_back": ("trap.low_forge_returns", "first", "the low forge flares back and races after them"),
    "pulls_ahead": ("trap.leader_left_pulls_ahead", "first", "the two front-runners (same form, uncoded) climb "
                    "together toward the Ring"),
    "one_dark": ("map.one_dark", "first", "every beacon lit but one: the long pause begins (the 7th flame full)"),
    "last_beacon": ("map.last_catch", "first", "the holdout's beacon catches (first visible; full on all_lit)"),
    "forges_cold": ("cold.forges_off", "frame", "every forge goes dark at the same instant: HARD SILENCE to 3999, "
                    "the hammering cut mid-stroke"),
    "storm_gone": ("unfinished.storm_thins", "full", "the storm has thinned to nothing; the Ring hangs grey"),
    "reveal": ("reveal.both_fires_ignite", "frame", "THE REVEAL: two first fires ignite together, hers near and a "
               "rival's on a far peak"),
}
# pass 1's frames that the measured picture confirms (the frame lies inside the event's measured window)
VERIFIED = {
    "old_story": ("refusal.offering_hand", "the offering hand is being drawn when the clarinet's Ring enters (the "
                  "Ring itself appears on the palm 2138-2150, under the motif's Ab-D)"),
    "turns_away": ("refusal.raised_hand", "the refusal gesture, the raised hand, is drawn 2195-2202; pass 1's 2200 "
                   "lies inside it and keeps the cor anglais' V-i landing on 2280"),
}
# new sync entries, measured (id, measured event id, field, what)
NEW_MEASURED = [
    ("leaders_reach", "trap.leader_left_pulls_ahead", "full", "the two front-runners reach the Ring's height"),
    ("map_beacon_2", "map.beacon_2", "first", "a beacon catches on the map (2 of 8)"),
    ("map_beacon_3", "map.beacon_3", "first", "a beacon catches on the map (3 of 8)"),
    ("map_beacon_4", "map.beacon_4", "first", "a beacon catches on the map (4 of 8)"),
    ("map_beacon_5", "map.beacon_5", "first", "a beacon catches on the map (5 of 8)"),
    ("map_beacon_6", "map.beacon_6", "first", "a beacon catches on the map (6 of 8)"),
    ("map_beacon_7", "map.beacon_7", "first", "a beacon catches on the map (7 of 8: then one kingdom is dark)"),
    ("all_lit", "map.last_catch", "full", "every beacon lit: the holdout's flame is full"),
    ("ring_grey", "unfinished.ring_drains", "full", "the Ring's gold has drained to grey"),
]
# musical placements computed from measured frames
DERIVED = {
    "hammer_alone": ("one_dark", BEAT_F, "one beat into the measured pause: the orchestra's thinnest point, under "
                     "the lone hammer (no hammer is drawn: SOUND places it)"),
}
RELABEL = {
    "trap": "THE TRAP: burn-through to the forge-towers under the storm (no hammer strike is drawn: the hammers are "
            "SOUND's, on the music's pulse)",
    "smoke": "(no discrete picture event: smoke rises continuously from the shutdown at 3848)",
    "promise": "'It was a promise to stop, if all the others would.' (caption: EDIT's timing)",
}


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def bar_beat(f):
    return f // BAR_F + 1, round(1 + (f % BAR_F) / BEAT_F, 3)


def build():
    bm = _load(SRC_MAP)
    cues = _load(SRC_CUES)
    ms = _load(MEASURED)
    by = {e["id"]: e for e in ms["events"]}
    out = copy.deepcopy(bm)
    out["cut"] = "C5P2"
    out["title"] = "THE LAST PAGES (v5.2), score pass 2 on the measured picture"
    out["measured_table"] = dict(file="music/v3/events_C5_measured.json", generator_sha256=ms["generator_sha256"],
                                 shots={k: v["frame_set_sha256"] for k, v in ms["shots"].items()})
    out["conventions"] = bm["conventions"] + (" | C5P2: barmap_C5 re-timed from events_C5_measured.json by "
                                              "music/src/barmap_c5p2.py; each sync entry's `source` says whether its "
                                              "frame is measured, verified, derived or pass 1's.")
    sync = {e["id"]: e for e in out["sync"]}

    def put(e, f, **kw):
        e["was_f"] = e.get("f")
        e["f"] = int(f)
        e["bar"], e["beat"] = bar_beat(int(f))
        e["t"] = round(f / FPS, 3)
        e.update(kw)

    for e in out["sync"]:
        e["source"] = "pass 1"
    for sid, (mid, field, what) in MEASURED_MOVES.items():
        f = by[mid]["frames"][field]
        if sid == "pulls_ahead":            # the earlier of the two front-runners
            f = min(f, by["trap.leader_right_pulls_ahead"]["frames"]["first"])
        put(sync[sid], f, what=what + " (measured)", source="measured", measured_ref=f"{mid}.{field}")
    for sid, (mid, why) in VERIFIED.items():
        w = by[mid]["frames"]
        if not w["first"] <= sync[sid]["f"] <= w["last"]:
            raise SystemExit(f"{sid}: pass 1's frame {sync[sid]['f']} is outside {mid} {w['first']}-{w['last']}: "
                             "re-time it instead of calling it verified")
        sync[sid].update(source="verified", verified_by=f"{mid} {w['first']}-{w['last']}", why=why)
    for sid, mid, field, what in NEW_MEASURED:
        f = by[mid]["frames"][field]
        if sid == "leaders_reach":
            f = round((f + by["trap.leader_right_pulls_ahead"]["frames"]["full"]) / 2)
        e = dict(id=sid, what=what + " (measured)", source="measured", measured_ref=f"{mid}.{field}")
        put(e, f)
        e.pop("was_f", None)
        out["sync"].append(e)
        sync[sid] = e
    for sid, (base, df, why) in DERIVED.items():
        put(sync[sid], sync[base]["f"] + df, what=why, source="derived",
            derived_from=f"{base} + {df} frames")
    for sid, what in RELABEL.items():
        sync[sid]["what"] = what
    out["sync"].sort(key=lambda e: (e["f"], e["id"]))
    c = copy.deepcopy(cues)
    c["cut"] = "C5P2"
    c["note"] = ("C5P2 (score pass 2): cues_C5 with the same section levels and breaths (the breath at forges_cold "
                 "now lands on the MEASURED 3848, which it already was); the measured map catches are fire events. "
                 "Effects remain SOUND's (sound/c5_sound_events.json).")
    for k in range(2, 8):
        c["events"][f"map_beacon_{k}"] = dict(kind="fire", tags=["map"])
    c["events"]["last_beacon"] = dict(kind="fire", tags=["map", "holdout"])
    c["events"]["reveal"] = dict(kind="fire", tags=["call", "first_fires"],
                                 desc="two first fires, together: two voices, one gesture")
    return out, c


def dumps(d):
    return json.dumps(d, indent=1) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    a = ap.parse_args()
    m, c = build()
    if a.check:
        bad = [p for p, d in ((OUT_MAP, m), (OUT_CUES, c)) if not os.path.exists(p) or open(p).read() != dumps(d)]
        for p in bad:
            print(f"STALE: {os.path.relpath(p, os.path.dirname(V3))} differs from what barmap_c5p2.py generates now")
        return 1 if bad else 0
    for p, d in ((OUT_MAP, m), (OUT_CUES, c)):
        with open(p, "w") as fh:
            fh.write(dumps(d))
    errs, warns = validate(BarMap("C5P2"))
    print(f"wrote {OUT_MAP} + {OUT_CUES}: {len(errs)} errors, {len(warns)} warnings")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
