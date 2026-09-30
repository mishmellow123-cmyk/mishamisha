"""THE LONG DAWN A: the SOUND table for the second half (3600-6479), built from the MEASURED picture (29 Sep 2026).

    python sound_a_table.py            # -> music/sound/a_sound_events.json and the second half of picture_sync_A.json
    python sound_a_table.py --check    # exit 1 if either committed file is not what build() gives now

Why. A's effects after the first fire were laid on the bar grid while their shots were slates and were never moved
onto the delivered picture: nothing was heard while fires caught on every ridge (3686-3738 measured), the beacon run's
seven flares receded while its fires came nearer, the watcher's near fire (A15) was silent, and watch-fire 1, burning
large beside the line from the crossing's first frame, was heard from 4940 at -61 LUFS. This table says, for every
effect of 3600-6479 that the picture drives, the frame its hit (or its bed's first and last frame) lands on, the
measured event that frame comes from, which of A's approved recordings plays it and at which approved level.
sound_recipes_A turns its rows into sound_v3's EXTRA events and beds and retires the bar-grid cues it replaces
(REPLACED), so `python sound_v3.py A` (and AP2, which forwards to sound_recipes_A) renders them in one step.

Where each frame comes from (`sync`): measured:<event id>.<field> in music/v3/events_A_measured.json (the delivered
frames, one at a time through EDIT; every frame also checked by eye on stills). build() reads the frame from there, and
problems() refuses a row whose frame no longer equals its measurement.

Levels: every `level` is a loudness an approved render gave the same kind of sound (pre-master LUFS: the loudest 400 ms
for events, integrated for beds): SOUND-C's (sound/events_C.json) or A's own (sound/events_A.json, SOUND's last A
render). A row whose level is that reference plus a choice made here carries design=True, the rule in `level_rule`,
and status "ready (design: listen)". Nothing here has been heard: the owner renders and listens.
"""
import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
MEASURED = os.path.join(MUSIC, "v3", "events_A_measured.json")
LEV_C = os.path.join(MUSIC, "sound", "events_C.json")
LEV_A = os.path.join(MUSIC, "sound", "events_A.json")
OUT = os.path.join(MUSIC, "sound", "a_sound_events.json")
PSYNC = os.path.join(MUSIC, "sound", "picture_sync_A.json")
FPS = 24
FRAMES = 6480
HALF_W = 960                                   # the measured positions are half-size pixels

# the bar-grid cues this table replaces: sound_recipes_A keeps their recipes out of the render (skip) with these reasons
REPLACED = {
    **{f"A.run_{k}": "laid on the bar grid while A14 was a slate, distanced 0.2 -> 0.65 as the picture's fires came "
                     "nearer: replaced by A.run.link_* on the measured catches of the delivered run" for k in range(1, 8)},
    "A.watchfire1": "a 5 s pass heard from 4940 (-61 LUFS): watch-fire 1 burns large from the cut at 4880; replaced by "
                    "the bed A.watchfire_1 on its measured footprint",
    **{f"A.watchfire{k}": "a pass on the old staging's bar (5280/5520/5760): in the restaged crossing the far fires are "
                          "one small fire visible 4880-5160 (A.watchfire_far), then hidden behind the great lantern"
       for k in (2, 3, 4)},
    "A.fire.blue": "the same recordings and level, now framed on the measured ending: A.fire.blue.measured pales with "
                   "the four watch-fires EDIT pales (measured) and ends in the picture's fade",
}
# SOUND-C's seven approved beacon flares (events_C.json), quietest first: the run's links take them far to near
RUN_LEVELS = ("C.beacon5", "C.beacon6", "C.beacon3", "C.beacon2", "C.beacon4", "C.beacon7", "C.beacon1")
RUN_DIST = (0.65, 0.6, 0.5, 0.45, 0.35, 0.3, 0.2)   # A's retired run ramp (0.2 .. 0.65), reversed: far to near
# the same four flare recordings recur 11 times in 2.2 s of ridge catches: each plays at a slightly different tape
# speed (pitch follows), within the range SOUND's own run used (0.96-1.05), so no two neighbours are the same sound
STRETCH = (1.0, 0.97, 1.03, 0.96, 1.05, 0.99, 1.02, 0.98, 1.04)
RIDGE_TARGET = "C.beacon5"         # the quietest approved beacon flare: the ridge catches' loudest 400 ms together
RIDGE_WINDOW_F = 10                # frames: the 400 ms over which concurrent catches add up
# A level is match_gain's target BEFORE the outdoor send, so every event renders louder than its level: four ridge rows
# rendered alone (sound_v3.py A --only <row> --solo, 22:10; asound-evidence/claude/ridge_single_offsets.jsonl) came
# out +1.19..+1.78 dB over theirs (mean +1.51), link 1 +1.55 (solo_measure_2144.json), and C.beacon5's own level
# means the same pre-send number. The catches together rendered 1.89 dB over their summed levels (-35.21 LUFS against
# -37.10 with the rows as they were at 21:44, solo_measure_2144.json); only what the ENSEMBLE adds beyond that
# per-event gain (overlapping tails and their reverb) is taken off, so the ridges render as loud as one C.beacon5
# flare renders (link 1: -33.65)
RIDGE_RENDER_EXCESS_DB = 1.89
SINGLE_RENDER_OFFSET_DB = 1.51
RIDGE_EXCESS_DB = round(RIDGE_RENDER_EXCESS_DB - SINGLE_RENDER_OFFSET_DB, 1)
CLOSE_FIRE = "A.x.take"            # A's approved close wood-fire bed (-36.0 LUFS): the fire in her hands
FIRE_BED = "A.fire.blue"           # A's approved distant fire bed (-69.8 LUFS): the blue hour's fires
# A's approved levels, as SOUND's last A render reported them (sound/events_A.json at ea7cdf3). Pinned here, not read:
# every `sound_v3.py A` rewrites events_A.json with the new render's levels (and this table retires A.fire.blue's bar
# cue), so reading it would make the references move with each render. test_a_sound checks them against git HEAD.
A_APPROVED = {"A.x.take": -36.0, "A.fire.blue": -69.8}


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def pan_of(x):
    return round(0.8 * (2 * x / HALF_W - 1), 2)


def levels():
    lv = {e["id"]: e["level"] for e in _load(LEV_C) if e.get("level") is not None}
    lv.update(A_APPROVED)
    return lv


def resolve(ms, sync):
    """'measured:<event id>.<field>' -> the frame in the measured file (None if absent)"""
    kind, ref = sync.split(":", 1)
    if kind != "measured":
        return None
    eid, field = ref.rsplit(".", 1)
    ev = next((e for e in ms["events"] if e["id"] == eid), None)
    return None if ev is None else ev["frames"].get(field)


def build(measured=None):
    ms = measured if measured is not None else _load(MEASURED)
    lv = levels()
    E = {e["id"]: e for e in ms["events"]}
    rows = []

    def event(rid, eid, field, recipe, level, level_from, **kw):
        rows.append(dict(id=rid, kind="event", hit_f=E[eid]["frames"][field], sync=f"measured:{eid}.{field}",
                         recipe=recipe, level=round(level, 1), level_from=level_from, **kw))

    def bed(rid, f0, s0, f1, s1, recipe, level, level_from, **kw):
        rows.append(dict(id=rid, kind="bed", f0=f0, f1=f1, sync=f"{s0} .. {s1}", recipe=recipe,
                         level=round(level, 1), level_from=level_from, **kw))

    # A13 EVERY RIDGE (3660-3799): a short far flare on every measured catch (45 of the 83 source fires catch in view,
    # 3686-3738; the 12 already burning that enter or are uncovered get none). Each catch's weight is its brightness
    # (the luma top-hat's median from the catch on, 15..132: light at the eye, which falls with distance as sound does)
    # and the whole set is scaled so the loudest 400 ms of catches together sits at SOUND-C's quietest approved beacon
    # flare: individually they are faint, together they are the ranges answering.
    catches = sorted((e for e in ms["events"] if e["id"].startswith("ridges.fire_") and e.get("kind") == "catch"),
                     key=lambda e: (e["frames"]["first"], e["id"]))
    bmax = max(e["size"]["tophat_median_after"] for e in catches)
    bmin = min(e["size"]["tophat_median_after"] for e in catches)
    w = {e["id"]: 10 * math.log10(e["size"]["tophat_median_after"] / bmax) for e in catches}
    hits = [(e["frames"]["first"], w[e["id"]]) for e in catches]
    dense = max(10 * math.log10(sum(10 ** (wi / 10) for fi, wi in hits if f0 <= fi < f0 + RIDGE_WINDOW_F))
                for f0, _ in hits)
    top = lv[RIDGE_TARGET] - RIDGE_EXCESS_DB - dense
    for k, e in enumerate(catches):
        b = e["size"]["tophat_median_after"]
        event(f"A.ridge.{e['id'].split('.', 1)[1]}", e["id"], "first", "flare", top + w[e["id"]],
              f"sound/events_C.json {RIDGE_TARGET} (all catches' loudest 400 ms)", flare_index=k % 4,
              stretch=STRETCH[k % len(STRETCH)],
              dist=round(0.85 - 0.2 * (b - bmin) / (bmax - bmin), 2), pan=pan_of(e["region"]["x"]),
              pre=0.0, fi=0.005, post=0.9, fo=0.5, design=True, status="ready (design: listen)",
              level_rule=f"{lv[RIDGE_TARGET]} - {RIDGE_EXCESS_DB} (what the ensemble adds in the render beyond one "
                         f"event: {RIDGE_RENDER_EXCESS_DB} over the summed levels less {SINGLE_RENDER_OFFSET_DB} every "
                         f"event gains, both measured) - {dense:.2f} (the densest 400 ms of weights) + 10 log10("
                         f"brightness / {bmax}); brightness = measured luma top-hat median after the catch")
    # A13 KARST and DESERT stay SOUND's bar-map cues: their measured flares are on the frames they already play
    # (karst 3815, desert 3880); picture_sync pins them there (PICTURE), so no score re-timing can move them off
    # A14 THE BEACON RUN (the linked-fires take): a flare on each link's measured catch; SOUND-C's approved beacon
    # levels, quietest first, far to near, so the last link is the loudest: link 7 is the watchers' own hearth, its
    # warm footprint 6 frames on 12,861 px against 1,690-3,242 px for links 1-6 (run.link_*.size); the links walk down
    # and left towards it (x 596 -> 160, y 109 -> 308); A's retired distance ramp reversed
    links = sorted((e for e in ms["events"] if e["id"].startswith("run.link_")), key=lambda e: int(e["id"][9:]))
    if links:
        for k, e in enumerate(links):
            d = RUN_DIST[k]
            event(f"A.run.link_{k + 1}", e["id"], "first", "flare", lv[RUN_LEVELS[k]],
                  f"sound/events_C.json {RUN_LEVELS[k]}", flare_index=k % 4, dist=d, pan=pan_of(e["region"]["x"]),
                  pre=0.0, fi=0.005, post=2.8, fo=1.4, status="ready",
                  note="SOUND-C's approved beacon levels, reordered quietest first (the far links first)")
            if e["frames"].get("flare") and e["frames"]["flare"] - e["frames"]["first"] >= 12:
                # the link catches early (tinder) and flares on the score's run_1: the flare gets its own burst
                event(f"A.run.link_{k + 1}_flare", e["id"], "flare", "flare", lv[RUN_LEVELS[k]],
                      f"sound/events_C.json {RUN_LEVELS[k]}", flare_index=(k + 2) % 4, dist=d,
                      pan=pan_of(e["region"]["x"]), pre=0.0, fi=0.005, post=2.8, fo=1.4, status="ready")
    # A14-A15: the watchers' hearth, from its catch (link 7) through the watchers' shot to the cut to A16 (4400)
    if "watchers.near_fire" in E:
        nf = E["watchers.near_fire"]
        f0 = E["run.link_7"]["frames"]["first"] if "run.link_7" in E else nf["frames"]["first"]
        s0 = "measured:run.link_7.first" if "run.link_7" in E else "measured:watchers.near_fire.first"
        bed("A.near_fire", f0, s0, nf["frames"]["last"] + 1, "measured:watchers.near_fire.last (+1: to the cut)",
            "fire_bed", lv[CLOSE_FIRE], f"sound/events_A.json {CLOSE_FIRE}", fade_in=1.0, fade_out=0.12,
            lp=9000, width=0.9, pan=pan_of(nf["region"]["x"]), status="ready",
            note="the flare on link 7 is its catch; the bed rises under that flare's 2.8 s tail and holds the "
                 "crackle of the fire the watcher sits by until the cut")
    # A18 THE CROSSING (restaged): watch-fire 1 burns large from the cut, at A's close-fire level, then recedes with
    # its measured footprint as the camera draws back (-6 dB by 5173, -20 dB by 5399), swelling with the three
    # feedings seen in the wide (5422, 5572, 5728: +6 dB); it never leaves the frame, so it runs to the cut to A19
    wf = E["crossing.watchfire_1"]
    samples = sorted((int(f), v[1]) for f, v in wf["size"]["area_px_and_db"].items())
    env = [[f, round(max(-24.0, min(0.0, db)), 1)] for f, db in samples]
    env[0][1] = 0.0                                     # full level on the cut: the fire is burning when we arrive
    bed("A.watchfire_1", wf["frames"]["first"], "measured:crossing.watchfire_1.first", wf["frames"]["last"] + 1,
        "measured:crossing.watchfire_1.last (+1: the cut to A19)", "fire_bed", lv[CLOSE_FIRE],
        f"sound/events_A.json {CLOSE_FIRE}", fade_in=0.0, fade_out=0.3, lp=9000, width=0.9,
        pan=pan_of(wf["region"]["x"]), env_f=env, design=True, status="ready (design: listen)",
        level_rule="full level = A's close fire; then 10 log10 of the measured footprint over the close shot's "
                   "(every 10 frames, 9-frame median, clamped -24..0 dB): on-screen area and sound power both fall "
                   "as 1/distance^2")
    far = E["crossing.watchfire_far"]
    ratio = far["size"]["ratio_to_watchfire_1_db"]
    bed("A.watchfire_far", far["frames"]["first"], "measured:crossing.watchfire_far.first", far["frames"]["last"] + 1,
        "measured:crossing.watchfire_far.last (+1)", "fire_bed", lv[CLOSE_FIRE] + ratio,
        f"sound/events_A.json {CLOSE_FIRE}", trim_db=ratio, fade_in=0.0, fade_out=1.0, lp=2500, width=0.35,
        pan=pan_of(far["region"]["x"]), design=True, status="ready (design: listen)",
        level_rule=f"A's close fire {ratio:+.1f} dB: the far fire's measured footprint over watch-fire 1's in the "
                   "close shot; under watch-fire 1 it will hardly be heard, which is what the picture shows")
    # A19-A20: the blue hour's fire bed at A's approved level, over the measured ending: it pales with each watch-fire
    # EDIT pales (measured 10-90% of its fall) and goes with the picture into the last fade (6456-6479)
    fires = sorted((e for e in ms["events"] if e["id"].startswith("ending.fire_")), key=lambda e: e["id"])
    kept = [1.0] * len(fires)
    pts = [[5840, 0.0]]
    for k, e in enumerate(fires):
        a, b = e["frames"]["pale_start"], e["frames"]["pale_end"]
        pts.append([a, round(10 * math.log10(sum(kept) / len(fires)), 1)])
        kept[k] = e["evidence"]["pale_fraction"]
        pts.append([b, round(10 * math.log10(sum(kept) / len(fires)), 1)])
    end = E["ending.final_fade"]["frames"]
    pts.append([end["first"], pts[-1][1]])
    pts.append([end["last"] + 1, -60.0])
    bed("A.fire.blue.measured", E["ending.fire_1"]["frames"]["first"], "measured:ending.fire_1.first", end["last"] + 1,
        "measured:ending.final_fade.last (+1: black)", "fire_bed", lv[FIRE_BED], f"sound/events_A.json {FIRE_BED}",
        fade_in=0.1, fade_out=0.0, lp=8000, width=0.8, env_f=pts, design=True, status="ready (design: listen)",
        level_rule="A.fire.blue's approved level; each pale takes the fire bed down by the measured light that fire "
                   "keeps (its top-hat after / before, 0.26-0.32), as a share of the four fires")
    out = dict(schema="long-dawn/a-sound-events/2", cut="A", fps=FPS, frames=FRAMES,
               measured="music/v3/events_A_measured.json",
               measured_sha256=sha(MEASURED) if measured is None else None,
               levels=dict(events_C="music/sound/events_C.json (read)",
                           events_A="music/sound/events_A.json at ea7cdf3 (pinned: A_APPROVED)"),
               replaced=REPLACED, events=rows)
    return out


def problems(t, measured=None):
    """the table's contract: every frame is its measurement's (drift fails), inside the film and A's second half;
    approved level provenance; the run grows louder and nearer; the ridge catches together at the approved flare"""
    ms = measured if measured is not None else _load(MEASURED)
    lv = levels()
    out = []
    ids = [r["id"] for r in t["events"]]
    if len(ids) != len(set(ids)):
        out.append("duplicate ids")
    for r in t["events"]:
        rid = r["id"]
        if r["kind"] == "event":
            want = resolve(ms, r["sync"])
            if want is None or r["hit_f"] != want:
                out.append(f"{rid}: hit {r['hit_f']} drifted from {r['sync']} = {want}")
            if not 3600 <= r["hit_f"] < FRAMES:
                out.append(f"{rid}: hit {r['hit_f']} outside A's second half")
            if r.get("pre", 1) != 0.0:
                out.append(f"{rid}: pre-roll before the measured first light")
        else:
            s0, s1 = r["sync"].split(" .. ")
            w0 = resolve(ms, s0)
            w1 = resolve(ms, s1.split(" ")[0])
            if w0 is not None and r["f0"] != w0:
                out.append(f"{rid}: f0 {r['f0']} drifted from {s0} = {w0}")
            if w1 is not None and r["f1"] != w1 + 1:
                out.append(f"{rid}: f1 {r['f1']} drifted from {s1} = {w1} + 1")
            if not 3600 <= r["f0"] < r["f1"] <= FRAMES:
                out.append(f"{rid}: bed {r['f0']}-{r['f1']} outside A's second half or empty")
            for f, _ in r.get("env_f", []):
                if not r["f0"] <= f <= r["f1"]:
                    out.append(f"{rid}: envelope point {f} outside the bed")
        ref = r["level_from"].split()[1] if r.get("level_from") else None
        if ref not in lv:
            out.append(f"{rid}: level without an approved reference")
        elif not r.get("design") and abs(r["level"] - lv[ref]) > 1e-9:
            out.append(f"{rid}: level {r['level']} is not its reference {ref} = {lv[ref]} and is not marked design")
        if r.get("design") and not r.get("level_rule"):
            out.append(f"{rid}: a design level without its rule")
    run = [r for r in t["events"] if r["id"].startswith("A.run.link_") and not r["id"].endswith("_flare")]
    run.sort(key=lambda r: int(r["id"].rsplit("_", 1)[1]))
    for a, b in zip(run, run[1:]):
        if not (b["level"] > a["level"] and b["dist"] <= a["dist"] and b["hit_f"] > a["hit_f"]):
            out.append(f"the run must come nearer and louder at every link ({a['id']} -> {b['id']})")
    ridge = [r for r in t["events"] if r["id"].startswith("A.ridge.")]
    if ridge:
        dense = max(10 * math.log10(sum(10 ** (q["level"] / 10) for q in ridge
                                        if p["hit_f"] <= q["hit_f"] < p["hit_f"] + RIDGE_WINDOW_F)) for p in ridge)
        if abs(dense + RIDGE_EXCESS_DB - lv[RIDGE_TARGET]) > 0.3:
            out.append(f"ridge catches together {dense + RIDGE_EXCESS_DB:.1f} LUFS, not {RIDGE_TARGET} "
                       f"{lv[RIDGE_TARGET]}")
    return out


def picture_sync(t, current):
    """picture_sync_A.json with A's second half measured: the kept bar-map cues pinned on their measured frames
    (PICTURE), every measured row listed, nothing left pending or slate. Every other key is kept as it is."""
    ms = _load(MEASURED)
    E = {e["id"]: e for e in ms["events"]}
    d = json.loads(json.dumps(current))
    for key in ("pending", "not_measurable"):
        d.pop(key, None)
    old_wf = "; watch-fire 1 4990-5040"               # the old crossing's pass: not the restaged picture's
    if isinstance(d.get("on_frame"), str) and old_wf in d["on_frame"]:
        d["on_frame"] = d["on_frame"].replace(old_wf, "; A 3600-6479 re-measured 29 Sep night: see second_half")
    # the numbers in `what` are read from the measured file (its own area definitions), never typed here
    k, s = E["karst.flare"], E["desert.catch"]
    kf, sf_ = k["frames"]["first"], s["frames"]["first"]
    ka, sa = k["evidence"]["warm_area"], s["evidence"]["dim_area"]
    d["t"]["A.karst_flare"] = dict(f=kf, cue=3815,
                                   what=f"measured: whole-frame warm area {ka[str(kf - 1)]} -> {ka[str(kf)]} px on {kf} "
                                        f"(the rock floods), fastest {k['frames']['fastest']}")
    d["t"]["A.desert_fire"] = dict(f=sf_, cue=3880,
                                   what=f"measured: the first warm light on the sand, dim-warm area {sa[str(sf_ - 1)]} "
                                        f"-> {sa[str(sf_)]} px on {sf_}, fastest {s['frames']['fastest']}")
    d["second_half"] = dict(
        _doc="A 3600-6479, measured on the delivered frames (music/v3/events_A_measured.json; ridges reveal_A, "
             "karst/desert montage3d_v3, A14/A15 the linked-fires takes, A18 the restaged crossing, A19-A20 dawnrev_A "
             "+ EDIT's watch-fires); the effects are sound/a_sound_events.json's rows (sound_recipes_A plays them)",
        measured_sha256=sha(MEASURED),
        rows={r["id"]: ({"f": r["hit_f"]} if r["kind"] == "event" else {"f0": r["f0"], "f1": r["f1"]}) |
              {"sync": r["sync"]} for r in t["events"]},
        retired=sorted(t["replaced"]))
    return d


def dumps(d):
    return json.dumps(d, indent=1) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    t = build()
    bad = problems(t)
    if bad:
        print("\n".join("PROBLEM: " + x for x in bad))
        return 1
    ps = picture_sync(t, _load(PSYNC))
    if a.check:
        ok = os.path.exists(OUT) and open(OUT).read() == dumps(t)
        ok2 = open(PSYNC).read() == dumps(ps)
        print("a_sound_events.json is current" if ok else "STALE: a_sound_events.json differs from build()")
        print("picture_sync_A.json is current" if ok2 else "STALE: picture_sync_A.json differs from build()")
        return 0 if ok and ok2 else 1
    with open(OUT, "w") as fh:
        fh.write(dumps(t))
    with open(PSYNC, "w") as fh:
        fh.write(dumps(ps))
    design = [r for r in t["events"] if r.get("design")]
    print(f"wrote {OUT}: {len(t['events'])} rows ({len(design)} with an unheard design level); "
          f"{len(t['replaced'])} bar-grid cues retired; picture_sync_A.json second half")
    return 0


if __name__ == "__main__":
    sys.exit(main())
