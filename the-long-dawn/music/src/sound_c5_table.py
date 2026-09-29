"""THE LONG DAWN C5: the SOUND event table, built from the MEASURED picture (SOUND-SCORE-C, 29 Sep 2026).

    python sound_c5_table.py            # -> music/sound/c5_sound_events.json (+ a summary of what is not ready)
    python sound_c5_table.py --check    # exit 1 if the committed table is not what the sources now give

Why. sound_C.wav belongs to the 7,200-frame cut: its effects sit on the old picture and cannot be offset into the
5,920-frame v5.2 film. This table says, for every effect of C5, the frame its hit (or its bed's start and end) must land
on, where that frame comes from, which of SOUND-C's approved recordings plays it and at which approved level. It is the
single input of sound_recipes_C5P2 (and _C5): once the recordings (music/cache/sound/) are back on a machine,
`python sound_v3.py C5P2` renders the whole effects stem in sync in one step.

Where each frame comes from (`sync`):
    measured:<event>.<field>   music/v3/events_C5_measured.json (the delivered frames; checked by eye on stills)
    score:<barmap event>       the music's pulse in v3/barmap_C5P2.json, or barmap_C5.json for pass 1 (hammers: no
                               strike is drawn anywhere)
    inherited:<...>            a v1 measurement (sound/picture_sync_C.json) moved by the v5 timeline's re-use offset,
                               for shots not on this Mac: verify on the C5 assembly
    edl:<stem>@<source frame>  a v1 source frame that becomes a C5 frame only through EDIT's C5 EDL (FLINT)
    edit:<what>                a frame only EDIT can give (transitions, caption write-ons)

Levels: `level` is the loudness SOUND-C's approved C render gave the same recording (sound/events_C.json, pre-master
LUFS: loudest 400 ms for events, integrated for beds). Rows marked design=True carry a choice made here, unheard (a
trim, the hammers' placement): they render, and are listed for a listening check. A row without a recording or a frame
has a status other than "ready...": sound_recipes_C5P2 refuses to render while any exists, unless told to skip them
(LD_SOUND_ALLOW_UNRESOLVED=1, which prints every skipped row).

THE HARD SILENCE is a contract, not an absence: every effect ends by the shutdown (3848) and none begins before the
Ring's cut (4000); build() refuses a table that breaks it, and sound_recipes_C5P2 mutes 3848..3999 besides.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
ROOT = os.path.dirname(MUSIC)
V3 = os.path.join(MUSIC, "v3")
OUT = os.path.join(MUSIC, "sound", "c5_sound_events.json")
FPS = 24
FRAMES = 5920
SHUTDOWN, RING_CUT = 3848, 4000             # the hard silence: [3848, 4000)
FLINT_SOURCE = {                            # SOUND-C's frames of the flint take in the ring stem (v1 EDL: stem `ring`,
    "strike1": 2980, "strike3": 3178,       # off 0; EDIT-C5 names it `ring_C`), measured on the v1 master
    "blow": 3204, "catch": 3316}            # (sound/picture_sync_C.json `on_frame`)
FLINT_STEMS = ("ring", "ring_C")
RETIRED_SOURCE = (3000, 3150)               # ring_C's find/vision: no C5 FLINT selection may read it (EDIT-C5 975d644,
                                            # edl_v3.FLINT["retired"])
DROPPED = [dict(id="C5.flint.strike2", source_frame=3009,
                why="SOUND-C's second strike (flash on v1 3009) lies inside ring_C's retired find/vision 3000-3149, "
                    "which no C5 FLINT candidate may read (EDIT-C5 975d644 edl_v3.FLINT['retired']): C5 has two "
                    "strikes, not three")]


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def c5_edl(path=None):
    """EDIT's C5 EDL if it exists AND is the 5,920-frame cut; the committed edl_C.json of the old 7,200-frame cut is
    refused by its length (a stale file with a plausible name). -> (edl or None, note)"""
    p = path or os.path.join(ROOT, "edit", "edl", "edl_C.json")
    if not os.path.exists(p):
        return None, "no edit/edl/edl_C.json"
    d = _load(p)
    if d.get("frames") != FRAMES:
        return None, f"edit/edl/edl_C.json is the {d.get('frames')}-frame cut, not C5's {FRAMES}: not used"
    note = "edit/edl/edl_C.json (5,920 frames)"
    for dec in d.get("decisions", []):
        if dec.get("id") == "C12_FLINT" and dec.get("choice") is None:
            note += ": its FLINT is still a decision (C12_FLINT decision_required)"
    return d, note


def flint_frame(edl, key):
    """the C5 frame showing the ring stem's source frame FLINT_SOURCE[key], via the EDL's take offsets (source frame
    = cut frame + off, half-open shot ranges); None when the EDL does not show it"""
    if edl is None:
        return None
    src = FLINT_SOURCE[key]
    if RETIRED_SOURCE[0] <= src < RETIRED_SOURCE[1]:
        return None
    for s in edl["shots"]:
        for t in s.get("takes", [])[:1]:        # the shot's first (chosen) take
            if t.get("stem") in FLINT_STEMS:
                f = src - int(t.get("off", 0))
                if s["f0"] <= f < s["f1"]:
                    return f
    return None


def build(pulse_cut="C5P2", edl_path=None):
    """the table. `pulse_cut` names the bar map whose score pulse places the hammers (the music's, no strike is drawn):
    C5P2 for score pass 2 (the committed snapshot), C5 for pass 1, whose ostinato sits on the retired frames. Every
    other row comes from the picture and is the same under both passes."""
    ms = _load(os.path.join(V3, "events_C5_measured.json"))
    bm = _load(os.path.join(V3, f"barmap_{pulse_cut}.json"))
    lv = {e["id"]: e for e in _load(os.path.join(MUSIC, "sound", "events_C.json")) if "level" in e}
    E = {e["id"]: e for e in ms["events"]}
    B = {e["id"]: e for e in bm["sync"]}
    edl, edl_note = c5_edl(edl_path)

    def M(eid, field):
        return E[eid]["frames"][field]

    def pan_of(eid):
        r = E[eid]["region"]
        return round(0.8 * (2 * r["x"] / 1920 - 1), 2)
    rows = []

    def event(eid, hit, sync, recipe, level_of, status="ready", level_db=None, level_note=None, **kw):
        """level_of: SOUND-C's approved level for that kind of event; else level_db with level_note saying its basis"""
        rows.append(dict(id=eid, kind="event", hit_f=hit, sync=sync, recipe=recipe,
                         level=lv[level_of]["level"] if level_of else level_db,
                         level_from=f"sound/events_C.json {level_of}" if level_of else level_note, status=status, **kw))

    def bed(eid, f0, f1, sync, recipe, level_of, status="ready", **kw):
        rows.append(dict(id=eid, kind="bed", f0=f0, f1=f1, sync=sync, recipe=recipe,
                         level=lv[level_of]["level"] if level_of else None,
                         level_from=f"sound/events_C.json {level_of}" if level_of else None, status=status, **kw))

    # C1-C9 THE BOOK, THE FORGE, THE DEEP, THE EYE (0-2079): the 7,200-frame cut's picture, frame for frame. edl_C of
    # the old cut (0536a3e) and C5's give every row in 0-2079 the same stems at the same offsets, but for the filmed burns
    # (book_C_ft) now on the three burn-throughs at the same offsets. SOUND-C's approved effects there therefore land on
    # the same frames, with SOUND-C's recordings, recipes and levels; where C5 has MEASURED the event itself (the riffle,
    # the page turn, both burn-throughs), the hit moves onto the measurement.
    def first_half(eid, f, sync_note):
        return f"inherited:SOUND-C sound/events_C.json {eid} (frame {f}; the same picture frame in C5){sync_note}"
    bed("C5.hearth.open", 0, 600, first_half("C.hearth.open", 0, ": from black to 26 frames after the page turn"),
        "C.hearth.open", "C.hearth.open")
    event("C5.page.turn_0", 40, first_half("C.page_turn_0", 40, ": over EDIT black, heard, not seen"),
          "C.page_turn_0", "C.page_turn_0")
    event("C5.riffle", M("opening.riffle", "first"), "measured:opening.riffle.first (SOUND-C had 320, the camera's "
          "first move; the leaves sweep from 326)", "C.riffle", "C.riffle")
    event("C5.pen.mountain", 355, first_half("C+.pen.mountain", 355, ""), "C+.pen.mountain", "C+.pen.mountain")
    event("C5.pen.T1", 400, first_half("C+.pen.T1", 400, ": R02's write-on, in_picture from 400"), "C+.pen.T1",
          "C+.pen.T1")
    event("C5.page.turn", M("letters.page_turn", "first"), "measured:letters.page_turn.first (SOUND-C had 575)",
          "C.page_turn", "C.page_turn")
    bed("C5.fire.born", 800, 1080, first_half("C.fire.born", 800, ": the fire catches at 801 (measured) .. into the "
        "forge's bed"), "C.fire.born", "C.fire.born")
    event("C5.burn.letters", M("letters.burn", "onset"), "measured:letters.burn.onset, the page's first see-through "
          "pixel (SOUND-C had 842, on the old page's burn; C5 plays the filmed burn book_C_ft)", "C+.burn.letters",
          "C+.burn.letters")
    bed("C5.fire.forge", 1040, 1680, first_half("C.fire.forge", 1040, ": the cut to the forge .. the Deep"),
        "C.fire.forge", "C.fire.forge")
    event("C5.burn.deep", 1699, first_half("C+.burn.deep", 1699, "; checked on C5's filmed burn-in (1682-1703): "
          "inside its closing, the hole area 80k -> 7k px over 1697-1702"), "C+.burn.deep", "C+.burn.deep")
    event("C5.pen.deep", 1708, first_half("C+.pen.deep", 1708, ": the mine drawn after the burn-in"), "C+.pen.deep",
          "C+.pen.deep")
    event("C5.pen.T7", 1710, first_half("C+.pen.T7", 1710, ": R08's write-on from 1710"), "C+.pen.T7", "C+.pen.T7")
    bed("C5.storm.eye", M("eye.burn", "onset"), 2080, "measured:eye.burn.onset (SOUND-C had 1920) .. the cut to the "
        "Refusal", "C.storm.eye", "C.storm.eye")
    event("C5.burn.eye", M("eye.burn", "onset"), "measured:eye.burn.onset, the scorch's sudden landing (SOUND-C had "
          "1922)", "C+.burn.eye", "C+.burn.eye")
    # C10 THE REFUSAL: the book by the hearth; a quill follows the drawing (no pen is seen: the lines draw themselves)
    bed("C5.hearth.refusal", 2080, 2320, "measured:shot refusal (2080-2319)", "C.hearth.open", "C.hearth.open",
        fade_in=0.5, fade_out=0.3)
    event("C5.pen.refusal_offer", M("refusal.ink_begins", "first"), "measured:refusal.ink_begins.first",
          "pen", "C+.pen.deep", src_start=17.3, dur_f=M("refusal.ring_on_palm", "last") - M("refusal.ink_begins", "first") + 1,
          note="the sleeve, the open hand and the Ring on it (2089-2150); silent through the pause 2151-2155")
    event("C5.pen.refusal_figure", M("refusal.figure_begins", "first"), "measured:refusal.figure_begins.first",
          "pen", "C+.pen.deep", src_start=20.3, dur_f=M("refusal.ink_ends", "last") - M("refusal.figure_begins", "first") + 1,
          note="the figure, its raised hand (2195-2202) and the cloak's hatching, to the last stroke")
    # the built x1burn (29 Sep): C+.burn.map is the recipe written for x1burn's opening ("the crackle rises from its
    # first frame, peaks as half the sheet is open"), so it syncs on the half-open frame, measured from _cover
    event("C5.burn.to_trap", 2330, "measured:the built burn renders/x1_refusal_C5_cover: 1% open 2316, half open 2330 "
          "(52%), full 2340 (x1burn v2, t_open 2312, born at the drawn ring)", "C+.burn.map", "C+.burn.map")
    # C11 THE TRAP: the forges; the surge, the low forge's return and the front-runners' climb are measured; the
    # hammers are the music's (no strike is drawn), on the ostinato's accents, silent while the one forge is low
    bed("C5.forge.trap", 2320, 2640, "measured:shot trap (2320-2639); cut to FLINT's black at 2640",
        "C.fire.forge", "C.fire.forge", fade_in=0.05, fade_out=0.01)
    event("C5.forge.surge", M("trap.forges_surge", "first"), "measured:trap.forges_surge.first", "C.roar", "C.roar")
    event("C5.forge.low_returns", M("trap.low_forge_returns", "first"), "measured:trap.low_forge_returns.first",
          "C.beacon1", "C.beacon1", pan=pan_of("trap.low_forge_returns"))
    bed("C5.forge.leaders", M("trap.leader_left_pulls_ahead", "first"), 2640,
        "measured:trap.leader_left_pulls_ahead.first (to the black)", "C+.fire.rises", "C+.fire.rises",
        fade_in=0.2, fade_out=0.01)
    t0, low, surge, black = (B[k]["f"] for k in ("trap", "low_fire", "surge", "flint_black"))
    accents = [f for a, b in ((t0, low), (surge, black)) for f in range(a, b, 40)]
    for k, f in enumerate(accents):
        event(f"C5.hammer.trap_{k + 1}", f, f"score:barmap_{pulse_cut}'s trap ostinato accent (every 2 beats from trap "
              "and from surge; none while the one forge is low)", "C5.hammer", None, status="ready (design: listen)",
              level_db=HAMMER_DB, level_note=HAMMER_NOTE, design=True)
    # C12 FLINT (not on this Mac): SOUND-C's shared flint take, through EDIT's C5 EDL (strike 2 is DROPPED)
    for key, rid, lid in (("strike1", "C.strike1", "C.strike1"), ("strike3", "C.strike3", "C.strike3"),
                          ("blow", "C.blow", "C.blow"), ("catch", "C.x.catch", "C.x.catch")):
        f = flint_frame(edl, key)
        event(f"C5.flint.{key}", f, f"edl:ring_C@{FLINT_SOURCE[key]} ({edl_note})", rid, lid,
              status="ready" if f is not None else "needs frame (EDIT's C5 EDL for FLINT)")
    bed("C5.wind.flint_to_run", 2640, 3440, "inherited:C.wind.fall (the ranges' wind, v1 from the flint to the run)",
        "C.wind.fall", "C.wind.fall", fade_in=3.0, fade_out=1.5)
    # C13 THE REVEAL: both first fires, together (measured 2880); hers near and centred, the rival's far and right
    rev = M("reveal.both_fires_ignite", "frame")
    # the recordings are the Beacon Run's, at its approved levels; the Reveal is a night (band -24..-6 LU, pp) where the
    # run is the film's fire (-16..-2). Render 3 of pass 2 measured C13's loudest 3 s at -3.2 LU (score -16.0 LUFS,
    # these two -14.8): -8 dB gives the two ignitions the Reveal's level, not the run's
    REVEAL_TRIM = dict(trim_db=-8.0, design=True, status="ready (design trim: listen)")
    event("C5.fire.first_near", rev, "measured:reveal.both_fires_ignite.frame", "C.beacon1", "C.beacon1",
          pan=round(0.8 * (2 * 950 / 1920 - 1), 2), **REVEAL_TRIM)
    event("C5.fire.first_far", rev, "measured:reveal.both_fires_ignite.frame", "C.beacon7", "C.beacon7",
          pan=round(0.8 * (2 * 1390 / 1920 - 1), 2), **REVEAL_TRIM)
    # C14 THE BEACON RUN (not on this Mac): v1's measured catches, moved by the run's re-use offset. The offset is
    # exact: the retired EDL shows runC_scroll 0-319 at 3840 (off -3840), EDIT-C5's at 3120 (off -3120)
    for k in range(1, 8):
        event(f"C5.run.beacon{k}", B[f"beacon_{k}"]["f"], f"inherited:v1 beacon {k} (on the v1 picture within 2 f, "
              f"sound/picture_sync_C.json) - 720 = barmap beacon_{k}", f"C.beacon{k}", f"C.beacon{k}",
              status="ready", verify="re-measure on the C5 assembly (runC_scroll is not on this Mac)")
    event("C5.burn.to_map", 3449, "measured:the built burn renders/x1_map_C5_cover: 1% open 3436, half open 3449 (51%), "
          "full 3460 (x1burn v2, t_open 3432: the 7,200-frame cut's #17 at -720)", "C+.burn.map", "C+.burn.map")
    # C15 THE LAST BEACON: every catch measured; the holdout nearest; a lone hammer in the pause (the music's)
    rec = {2: "C.beacon2", 3: "C.beacon3", 4: "C.beacon4", 5: "C.beacon5", 6: "C.beacon6", 7: "C.beacon7",
           8: "C.beacon1"}
    for k in range(2, 9):
        eid = f"map.beacon_{k}"
        event(f"C5.map.beacon{k}", M(eid, "first"), f"measured:{eid}.first", rec[k], rec[k], pan=pan_of(eid),
              **({"note": "the holdout: SOUND-C's nearest flare"} if k == 8 else {}))
    ham = B["hammer_alone"]["f"]
    for k, f in enumerate((ham, ham + 20)):
        event(f"C5.hammer.alone_{k + 1}", f, f"score:barmap_{pulse_cut} hammer_alone (C5P2: one beat into the measured "
              "pause) + one beat", "C5.hammer.faint", None, status="ready (design: listen)", level_db=HAMMER_FAINT_DB,
              level_note=HAMMER_NOTE, design=True, note="faint; the last stroke ends before the catch at 3786")
    # C16 THE FORGES GO COLD: the forge from the cut that shows them lit (EDIT's COLD_CUT: 3840, or 3816 with the
    # adopted lit lead-in), a hammer 8 frames before the measured shutdown, cut mid-stroke on it; then nothing
    off = M("cold.forges_off", "frame")
    lit = [r["f0"] for r in (edl or {}).get("shots", []) if r.get("name", "").startswith("THE FORGES GO COLD")]
    cold_cut = min(lit) if lit else 3840
    bed("C5.forge.cold", cold_cut, off, f"edl:the forges' first lit frame ({cold_cut}) .. measured:cold.forges_off"
        ".frame (the bed is cut on it)", "C.fire.forge", "C.fire.forge", fade_in=0.01, fade_out=0.005)
    event("C5.hammer.cut", 3840, "measured:shot cold (3840) .. cold.forges_off.frame: cut mid-stroke", "C5.hammer", None,
          status="ready (design: listen)", level_db=HAMMER_DB, level_note=HAMMER_NOTE, design=True,
          post_max_f=off - 3840)
    rows.append(dict(id="C5.silence", kind="silence", f0=off, f1=RING_CUT, sync="measured:cold.forges_off.frame .. "
                     "the Ring's cut", status="ready", note="no effect sounds here; the renderer mutes it besides"))
    # C17 THE RING, UNFINISHED: the storm's wind thins with the measured storm
    s0, sh, s1 = (M("unfinished.storm_thins", k) for k in ("first", "half", "full"))
    bed("C5.storm.unfinished", RING_CUT, s1, "measured:unfinished.storm_thins (first/half/full)", "C.storm.eye",
        "C.storm.eye", fade_in=0.02, fade_out=0.5, trim_db=-9.0, design=True, status="ready (design trim: listen)",
        env_f=[[RING_CUT, 0.0], [s0, 0.0], [sh, -6.0], [s1, -30.0]])
    # C18 THE DEEP (still), C19 THE WATCH (every beacon burns on: no hit), the second half (not on this Mac)
    bed("C5.hearth.deep", 4240, 4480, "measured:deep.no_discrete_event (a still page)", "C.hearth.open",
        "C.hearth.open", fade_in=0.5, fade_out=0.5)
    bed("C5.wind.watch", 4480, 4720, "measured:watch.beacons_burn_on (no catch in shot)", "C.wind.fall",
        "C.wind.fall", fade_in=1.0, fade_out=2.0)
    bed("C5.air.dawn", 4720, 5200, "inherited:C.air.dawn (v1 236.667-256.667 s) - 960 frames", "C.air.dawn",
        "C.air.dawn", fade_in=1.5, fade_out=1.5, verify="the ILLUMINATION is not on this Mac")
    bed("C5.hearth.end", 5200, FRAMES, "inherited:C.hearth.end (from PLENTY to the end)", "C.hearth.end",
        "C.hearth.end", fade_in=1.5, fade_out=1.5)
    event("C5.page.to_blank", 5440, "edit:the page turn into the PEN shot (kind page_turn, 5430-5452, cut 5440): its fold "
          "crosses the frame's centre at eased progress 0.445, frame 5439.7 (assemble.page_turn's geometry)",
          "C.page.blank", "C.page.blank")
    event("C5.pen.caption_22", 5462, "edit:caption R22 is an ink write-on (titles.py set='ink': a pen-shaped wipe over its "
          "first 24 frames from f_in 5462)", "pen", "C+.pen.T14", src_start=40.6, dur_f=24)
    event("C5.burn.title", 5715, "inherited:C+.burn.title (v1 6995, first spark 6982) - 1280", "C+.burn.title",
          "C+.burn.title", verify="the TITLE is not on this Mac")
    for r in rows:
        r.setdefault("design", False)
    return dict(schema="long-dawn/c5-sound-events/1", cut=f"C5 (the picture; hammers on barmap_{pulse_cut}'s pulse)",
                fps=FPS, frames=FRAMES, hard_silence=dict(from_f=SHUTDOWN, to_f=RING_CUT - 1), dropped=DROPPED,
                sources={"music/v3/events_C5_measured.json": sha(os.path.join(V3, "events_C5_measured.json")),
                         f"music/v3/barmap_{pulse_cut}.json": sha(os.path.join(V3, f"barmap_{pulse_cut}.json")),
                         "music/sound/events_C.json": sha(os.path.join(MUSIC, "sound", "events_C.json"))},
                edl=edl_note, events=rows)


# The forge hammers had no approved recording or level (SOUND-C's library has no anvil). Recording: VSCO-2-CE's anvil
# (sound_recipes_C "C5.hammer"), the score's own forging anvil. Level, a design choice between SOUND-C's approved
# neighbours: the Trap's strikes 0.4 dB under the x1burn (C+.burn.map -32.6) and 3.6 dB over the loudest flint strike
# (C.strike3 -36.6); the lone hammer under the map's pause at the quietest strike (C.strike1 -39.0), as its note asks.
HAMMER_DB, HAMMER_FAINT_DB = -33.0, -39.0
HAMMER_NOTE = "design (29 Sep): between SOUND-C's approved C+.burn.map -32.6 and C.strike1..3 -39.0..-36.6"


def problems(t):
    """the table's own contract: frames inside the film, recipes and levels where ready, and the hard silence"""
    out = []
    for r in t["events"]:
        ready = r["status"].startswith("ready")
        if r["kind"] == "event":
            f = r["hit_f"]
            if ready and (f is None or r.get("recipe") is None or r.get("level") is None):
                out.append(f"{r['id']}: ready without a frame, recipe or level")
            if f is not None and not 0 <= f < t["frames"]:
                out.append(f"{r['id']}: frame {f} outside the film")
            if f is not None and SHUTDOWN <= f < RING_CUT:
                out.append(f"{r['id']}: a hit inside the hard silence ({f})")
            if f is not None and f < SHUTDOWN and r.get("post_max_f") is not None and f + r["post_max_f"] > SHUTDOWN:
                out.append(f"{r['id']}: rings past the shutdown")
        elif r["kind"] == "bed":
            if r["f0"] < RING_CUT and r["f1"] > SHUTDOWN:
                out.append(f"{r['id']}: bed {r['f0']}-{r['f1']} crosses the hard silence")
            if not 0 <= r["f0"] < r["f1"] <= t["frames"]:
                out.append(f"{r['id']}: bed {r['f0']}-{r['f1']} outside the film or empty")
            if ready and (r.get("recipe") is None or r.get("level") is None):
                out.append(f"{r['id']}: ready without a recipe or level")
    ids = [r["id"] for r in t["events"]]
    if len(ids) != len(set(ids)):
        out.append("duplicate ids")
    return out


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
    if a.check:
        ok = os.path.exists(OUT) and open(OUT).read() == dumps(t)
        print("c5_sound_events.json is current" if ok else "STALE: c5_sound_events.json differs from build()")
        return 0 if ok else 1
    with open(OUT, "w") as fh:
        fh.write(dumps(t))
    blocked = [r for r in t["events"] if not r["status"].startswith("ready")]
    design = [r for r in t["events"] if r["status"].startswith("ready") and r.get("design")]
    print(f"wrote {OUT}: {len(t['events'])} rows; {len(blocked)} blocked, {len(design)} ready with an unheard design "
          f"choice, {len(t['dropped'])} dropped")
    for r in blocked + design:
        print(f"  {r['id']:26s} {r['status']}")
    for r in t["dropped"]:
        print(f"  {r['id']:26s} DROPPED: {r['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
