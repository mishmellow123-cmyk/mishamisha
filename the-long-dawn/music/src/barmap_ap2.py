"""THE LONG DAWN A, score PASS 2 (AP2): A's bar map and cue sheet plus the crossing's MEASURED set-off (29 Sep).

    python barmap_ap2.py            # -> music/v3/barmap_AP2.json + music/v3/cues_AP2.json (then validates them)
    python barmap_ap2.py --check    # exit 1 if the committed files differ from what the sources now generate

Pass 1 (score_v3_A.py on barmap_A.json / cues_A.json) stays exactly as it was: those files are read, never written.
AP2 is a separate cut id, so `python render_v3.py AP2` renders next to final_A without touching it.

AP2's map is A's map with two NEW sync entries from music/v3/events_AP2_measured.json (walk_setoff, walk_full: the
restaged crossing's set-off and the front's arrival at walking pace, measured by measure_ap2_crossing.py). No
existing entry is moved: A's effects (cues_A.json, sound_recipes_A.py, picture_sync_A.json) are anchored on these
ids, and AP2 plays A's effects, so every effect must resolve to the frame it has in A (the asound/syncA lanes re-time
A's effects in A's own files; regenerate this map after merging them). An entry the picture re-measures but the
score keeps says so in `measured` / `why_kept`, and its `what` is relabelled where the old cause is contradicted.

    source = "measured"   the frame IS a measured picture frame (measured_ref names the table entry)
             "pass 1"     A's frame, unchanged (a crossing entry re-measured by this pass carries `measured`)
"""
import argparse
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timeline_v3 import BAR_F, BEAT_F, FPS, V3, BarMap, validate  # noqa: E402

SRC_MAP = os.path.join(V3, "barmap_A.json")
SRC_CUES = os.path.join(V3, "cues_A.json")
MEASURED = os.path.join(V3, "events_AP2_measured.json")
OUT_MAP = os.path.join(V3, "barmap_AP2.json")
OUT_CUES = os.path.join(V3, "cues_AP2.json")

# a relabel where the restaged picture contradicts pass 1's stated cause (the frame and the music stay)
RELABEL = {
    "lantern": "match cut to the lantern: THE CROSSING; the line stands at the first watch-fire (the walk waits for "
               "walk_setoff)",
    "watchfire_1": "the line STANDS at watch-fire 1 (it has since 4880; the bearers pass just behind it only after "
                   "setting off: front bearer 5304, lantern 5368, measured): harmony change, the violas' CALL",
}


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def _sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def bar_beat(f):
    return f // BAR_F + 1, round(1 + (f % BAR_F) / BEAT_F, 3)


def build():
    bm = _load(SRC_MAP)
    cues = _load(SRC_CUES)
    ms = _load(MEASURED)
    out = copy.deepcopy(bm)
    out["cut"] = "AP2"
    out["title"] = "EVERY STEP CLOSER, score pass 2 on the delivered crossing (the walk waits for the set-off)"
    out["measured_table"] = dict(file="music/v3/events_AP2_measured.json", sha256=_sha(MEASURED),
                                 measured_by=ms["measured_by"])
    out["conventions"] = bm["conventions"] + (" | AP2: barmap_A plus the measured entries of events_AP2_measured.json "
                                              "(music/src/barmap_ap2.py); no entry of A is moved (A's effects are "
                                              "anchored on them); each entry's `source` says measured or pass 1.")
    sync = {e["id"]: e for e in out["sync"]}
    for e in out["sync"]:
        e["source"] = "pass 1"
    for sid, what in RELABEL.items():
        sync[sid]["was_what"] = sync[sid]["what"]
        sync[sid]["what"] = what
    for row in ms["kept"]:
        e = sync[row["id"]]
        if e["f"] != row["frame"]:
            raise ValueError(f"{row['id']}: the measured table keeps {row['frame']}, A's map says {e['f']}")
        e["measured"] = row["measured"]
        e["why_kept"] = row["why_kept"]
    for row in ms["new_sync"]:
        sid, f = row["id"], row["frame"]
        if sid in sync:
            raise ValueError(f"{sid}: AP2 adds entries, it never moves one of A's (A's effects are anchored on them)")
        if not isinstance(f, int) or isinstance(f, bool) or not 0 <= f < bm["bars"] * BAR_F:
            raise ValueError(f"{sid}: a measured frame must be an integer inside the cut")
        b, bb = bar_beat(f)
        e = dict(id=sid, bar=b, beat=bb, t=round(f / FPS, 3), f=f, what=row["what"] + " (measured)",
                 source="measured", measured_ref=f"events_AP2_measured.json:new_sync.{sid}")
        out["sync"].append(e)
        sync[sid] = e
    out["sync"].sort(key=lambda e: (e["f"], e["id"]))
    c = copy.deepcopy(cues)
    c["cut"] = "AP2"
    c["note"] = ("AP2: cues_A.json unchanged (A's effects, levels and breaths) plus the annotations of AP2's measured "
                 "entries; regenerate with barmap_ap2.py after any change to cues_A.json")
    for row in ms["new_sync"]:
        c["events"][row["id"]] = dict(kind="picture", tags=["crossing", "walk"])
    return out, c


def _dumps(d):
    return json.dumps(d, indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    m_, c_ = build()
    if a.check:
        stale = [p for p, d in ((OUT_MAP, m_), (OUT_CUES, c_)) if not os.path.exists(p) or open(p).read() != _dumps(d)]
        for p in stale:
            print("STALE:", os.path.relpath(p, os.path.dirname(V3)))
        print("AP2 map/cues current" if not stale else "regenerate: python barmap_ap2.py")
        return 1 if stale else 0
    for p, d in ((OUT_MAP, m_), (OUT_CUES, c_)):
        with open(p, "w") as fh:
            fh.write(_dumps(d))
    errs, warns = validate(BarMap("AP2"))
    print(f"wrote {os.path.basename(OUT_MAP)} ({len(m_['sync'])} sync entries) and {os.path.basename(OUT_CUES)}: "
          f"{len(errs)} errors, {len(warns)} warnings")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
