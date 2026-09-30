"""D phase 1c score events on treatment v2's fixed 115-bar grid.

This is an opt-in score adapter, not the editorial bar map. It never reads or
writes barmap_D.json or cues_D.json. Every destination event is PROVISIONAL:
prescribed treatment cuts and composed interior placements are not picture
measurements. Phase 2 may supply measured overrides inside the original bar,
with provenance, without changing this composition's section grid.
"""
from copy import deepcopy
import json

from timeline_v3 import BAR_F, BEAT_F, FPS, FR, SR

SECTION_TABLE = (
    ("D01", "BLACK", 0, 80),
    ("D02", "FALSE DAWN", 80, 560),
    ("D03", "GLYPHS", 560, 960),
    ("D04", "POINT", 960, 1040),
    ("D05", "IGNITION AND PROMISE", 1040, 1440),
    ("D06", "MOUNTAIN", 1440, 1680),
    ("D07", "BURN", 1680, 1760),
    ("D08", "INSCRIPTION", 1760, 2080),
    ("D09", "FORGING", 2080, 2400),
    ("D10", "RACE", 2400, 2720),
    ("D11", "DEEP", 2720, 2960),
    ("D12", "BRINK", 2960, 3200),
    ("D13", "VISION", 3200, 3440),
    ("D14", "GAP", 3440, 3520),
    ("D15", "REFUSAL", 3520, 3760),
    ("D16", "TRAP", 3760, 4080),
    ("D17", "DISTANT GLOW", 4080, 4240),
    ("D18", "CROWNS", 4240, 4560),
    ("D19", "TWO FIRES", 4560, 5040),
    ("D20", "EVERY RIDGE", 5040, 5180),
    ("D21", "LAST KINGDOM", 5180, 5680),
    ("D22", "IN STEP", 5680, 5840),
    ("D23", "OLD FIRE", 5840, 6080),
    ("D24", "UNFINISHED", 6080, 6400),
    ("D25", "ABANDONED DEEP", 6400, 6640),
    ("D26", "CROSSING", 6640, 7040),
    ("D27", "WATCH", 7040, 7360),
    ("D28", "TRUE DAWN", 7360, 7840),
    ("D29", "TERRACES", 7840, 8080),
    ("D30", "PLENTY", 8080, 8320),
    ("D31", "LAST LEAF", 8320, 8640),
    ("D32", "LAST PAGES", 8640, 8880),
    ("D33", "TITLE", 8880, 9120),
    ("D34", "HEARTH OUT", 9120, 9200),
)


def _event(name, frame, source, tags=(), fixed=False):
    return dict(id=name, frame=frame, status="provisional", source=source,
                tags=list(tags), fixed=fixed)


# ONE event table. Explicit musical proposals are labelled as such; their
# numerical precision must not be mistaken for a measured picture frame.
EVENT_TABLE = [
    _event("opening", 0, "treatment v2 opening", fixed=True),
    _event("false_dawn", 80, "treatment v2 shot 2", fixed=True),
    _event("glyphs", 560, "treatment v2 shot 3", fixed=True),
    _event("point", 960, "treatment v2 shot 4", fixed=True),
    _event("ignition", 1040, "treatment v2 shot 5", fixed=True),
    _event("mountain", 1440, "treatment v2 shot 6", fixed=True),
    _event("mountain_ring", 1480, "C mountain motif moved into D19; musical placement"),
    _event("old_burn", 1680, "treatment v2 shot 7", fixed=True),
    _event("inscription", 1760, "treatment v2 shot 8", fixed=True),
    _event("inscription_stop", 2040, "composed mid-letter stop; picture timing pending"),
    _event("forging", 2080, "treatment v2 shot 9", fixed=True),
    _event("race", 2400, "treatment v2 shot 10", fixed=True),
    _event("deep", 2720, "treatment v2 shot 11", fixed=True),
    _event("tick_8", 2800, "second Deep bar, eighth-note division"),
    _event("tick_16", 2880, "third Deep bar, sixteenth-note division"),
    _event("brink", 2960, "treatment v2 shot 12", fixed=True),
    _event("vision", 3200, "treatment v2 shot 13", fixed=True),
    _event("vision_complete", 3380, "composed withheld note inside vision, before silence"),
    _event("silence_before_gap", 3400, "last two beats before treatment gap", fixed=True),
    _event("gap", 3440, "treatment v2 shot 14; single hammer", fixed=True),
    _event("refusal", 3520, "treatment v2 shot 15", fixed=True),
    _event("refusal_turn", 3640, "C5 refusal turn at 2200, plus 1440; donor timing"),
    _event("refusal_cadence", 3720, "C5 refusal cadence at 2280, plus 1440; donor timing"),
    _event("trap", 3760, "treatment v2 shot 16", fixed=True),
    _event("trap_drop", 3840, "composed one-voice dropout; picture timing pending"),
    _event("trap_surge", 3880, "composed remaining-voices surge; picture timing pending"),
    _event("trap_return", 3960, "composed dropped-voice return; picture timing pending"),
    _event("leaders_climb", 4000, "composed leaders' ascent; picture timing pending"),
    _event("glow", 4080, "treatment v2 shot 17", fixed=True),
    _event("crowns", 4240, "treatment v2 shot 18 held D", fixed=True),
    _event("crowns_kindle", 4320, "treatment v2 explicit simultaneous kindle", ("call_pair",)),
    _event("two_fires", 4560, "treatment v2 shot 19 paired CALL and WATCH", ("call_pair",), True),
    _event("ridge", 5040, "treatment v2 shot 20", fixed=True),
    _event("map", 5180, "treatment v2 shot 21, bar 65 beat 4", fixed=True),
    _event("map_beacon_2", 5208, "C5P2 measured C3468 + 1740 from cutd Round 3 provisional edit", ("map_beacon_catch",)),
    _event("map_beacon_3", 5263, "C5P2 measured C3523 + 1740 from cutd Round 3 provisional edit", ("map_beacon_catch",)),
    _event("map_beacon_4", 5324, "C5P2 measured C3584 + 1740 from cutd Round 3 provisional edit", ("map_beacon_catch",)),
    _event("map_beacon_5", 5387, "C5P2 measured C3647 + 1740 from cutd Round 3 provisional edit", ("map_beacon_catch",)),
    _event("map_beacon_6", 5420, "C5P2 measured C3680 + 1740 from cutd Round 3 provisional edit", ("map_beacon_catch",)),
    _event("map_beacon_7", 5458, "C5P2 measured C3718 + 1740 from cutd Round 3 provisional edit", ("map_beacon_catch",)),
    _event("map_dark", 5462, "C5P2 measured C3722 + 1740 from cutd Round 3 provisional edit"),
    _event("holdout_hammer", 5496, "cutd Round 3 planned first contact in holdout insert; not measured"),
    _event("last_beacon_catch", 5600, "cutd Round 3: C3786 catch after insert and hold; D picture not measured", ("map_beacon_catch",)),
    _event("all_lit", 5605, "C5P2 measured C3791 + 1814 after cutd insert and hold; D picture not measured"),
    _event("in_step", 5680, "treatment v2 shot 22", fixed=True),
    _event("old_fire", 5840, "treatment v2 shot 23", fixed=True),
    _event("unfinished", 6080, "treatment v2 shot 24", fixed=True),
    _event("lamps", 6240, "composed lamps entry within unfinished; picture timing pending"),
    _event("deep_abandoned", 6400, "treatment v2 shot 25", fixed=True),
    _event("deep_cadence", 6600, "C5 abandoned Deep cadence at 4440, plus 2160"),
    _event("crossing_lantern", 6640, "cutd Round 3 crossing entry: A5180; eight-frame dissolve, no spatial lantern match", fixed=True),
    _event("crossing_walk_setoff", 6640, "AP2 A5180 + 1460 from cutd Round 3 source selection"),
    _event("crossing_walk_second", 6700, "AP2 A5240 + 1460 from cutd Round 3 source selection"),
    _event("crossing_walk_full1", 6740, "AP2 A5280 + 1460 from cutd Round 3 source selection"),
    _event("crossing_walk_full2", 6780, "AP2 A5320 + 1460 from cutd Round 3 source selection"),
    _event("crossing_narrow_start", 6820, "AP2 A5360 + 1460; narrow-path WALK rest starts"),
    _event("crossing_walk_resume", 6980, "AP2 A5520 + 1460; WALK resumes after narrow-path rest"),
    _event("crossing_resume_second", 7020, "AP2 A5560 + 1460 from cutd Round 3 source selection"),
    _event("watch", 7040, "treatment v2 shot 27", fixed=True),
    _event("watch_feed_1", 7120, "composed chorale change on first feed; picture timing pending"),
    _event("watch_feed_2", 7200, "composed chorale change on second feed; picture timing pending"),
    _event("watch_feed_3", 7280, "composed chorale change on third feed; picture timing pending"),
    _event("sunrise", 7360, "treatment v2 shot 28", fixed=True),
    _event("dawn_call", 7400, "C5P2 illumination CALL at 4760 plus 2640"),
    _event("dawn_answer", 7520, "C5P2 illumination ANSWER at 4880 plus 2640"),
    _event("dawn_home", 7680, "C5P2 illumination HOME at 5040 plus 2640"),
    _event("dawn_tonic", 7800, "C5P2 illumination tonic at 5160 plus 2640"),
    _event("terraces", 7840, "treatment v2 shot 29; A6000 plus 1840", fixed=True),
    _event("terraces_home", 7920, "A6080 HOME plus 1840"),
    _event("terraces_cadence", 8000, "A6160 tonic plus 1840"),
    _event("plenty", 8080, "C5P2 MUSIC 5200 plus 2880, not raw book picture 6160", fixed=True),
    _event("last_leaf", 8320, "treatment v2 shot 31", fixed=True),
    _event("last_leaf_break", 8520, "composed phrase rest; picture mid-word timing pending"),
    _event("last_pages", 8640, "treatment v2 shot 32", fixed=True),
    _event("voice_start", 8660, "C5P2 inherited clean-air start; owner voice not supplied"),
    _event("voice_end", 8740, "80-frame clean-air reservation, 3.333 seconds"),
    _event("blank", 8740, "end of provisional inherited voice reservation"),
    _event("title", 8880, "treatment v2 shot 33", fixed=True),
    _event("title_letters", 8902, "C5P2 title first lettering at 5702 plus 3200"),
    _event("plagal_prepare", 8960, "composed G/D before title close"),
    _event("plagal", 9040, "composed title D cadence"),
    _event("score_end", 9120, "treatment v2 bar 115, score silence", fixed=True),
    _event("film_end", 9200, "treatment v2 exact 115-bar endpoint", fixed=True),
]
# Provisional giant-strike grid and antiphonal fire chain: individual names
# allow measured bindings to replace each hit, without modifying the score.
EVENT_TABLE += [
    _event(f"giant_stroke_{i:02d}", f,
           "composed 40-frame giant stroke grid; actual strikes pending",
           ("giant_stroke",))
    for i, f in enumerate([f for f in range(2080, 3200, 40) if not 2720 <= f < 2960], 1)
]
EVENT_TABLE += [
    _event(f"beacon_catch_{i:02d}", f,
           "glowvars Round 2 six-fire chain retimed +1040 into v2; D picture not measured",
           ("beacon_catch",))
    for i, f in enumerate(range(4760, 5000, 40), 1)
]


class DraftMap:
    """BarMap-compatible read-only adapter; overrides require honest provenance."""

    def __init__(self, overrides=None):
        self.cut, self.bars, self.frames = "D", 115, 9200
        self.n, self.render_n = self.frames * FR, self.frames * FR + SR
        self.seconds = self.frames / FPS
        self.sections = [dict(id=i, name=n, f0=a, f1=b, b0=a / BEAT_F,
                              b1=b / BEAT_F, t0=a / FPS, t1=b / FPS,
                              bar_start=a // BAR_F + 1, bar_end=(b - 1) // BAR_F + 1,
                              tags=[]) for i, n, a, b in SECTION_TABLE]
        self.events = deepcopy(EVENT_TABLE)
        self.breaths = []
        by_id = {e["id"]: e for e in self.events}
        if isinstance(overrides, (str, bytes)):
            with open(overrides) as handle:
                overrides = json.load(handle)
        for name, update in (overrides or {}).items():
            if name not in by_id:
                raise ValueError(f"unknown D event: {name}")
            if not isinstance(update, dict) or not update.get("source"):
                raise ValueError(f"{name}: override needs frame, status, and source")
            old = by_id[name]
            f = update.get("frame")
            if type(f) is not int or not 0 <= f <= self.frames:
                raise ValueError(f"{name}: frame must be an integer within D")
            if old["fixed"] and f != old["frame"]:
                raise ValueError(f"{name}: treatment cut is fixed")
            if f // BAR_F != old["frame"] // BAR_F:
                raise ValueError(f"{name}: phase 2 may only move within its bar")
            if update.get("status") not in ("provisional", "measured"):
                raise ValueError(f"{name}: status must be provisional or measured")
            if update["status"] == "measured" and not update.get("measured_ref"):
                raise ValueError(f"{name}: measured binding needs measured_ref")
            old.update({k: update[k] for k in ("frame", "status", "source", "measured_ref") if k in update})
        for e in self.events:
            e.update(beat=e["frame"] / BEAT_F, t=e["frame"] / FPS,
                     bar=e["frame"] // BAR_F + 1,
                     bar_beat=1 + e["frame"] % BAR_F / BEAT_F, kind="picture")
            e["section"] = self.sec_at(e["beat"])["id"]
        self.events.sort(key=lambda e: (e["frame"], e["id"]))
        self.d = dict(cut="D", bars=self.bars, frames=self.frames, fps=FPS, bpm=72,
                      status="phase 1c provisional", sections=self.sections, events=self.events,
                      breaths=[], ambience=[], sfx=[])

    def section(self, sid):
        return next(s for s in self.sections if s["id"] == sid)

    def sec_at(self, beat):
        return next((s for s in self.sections if s["b0"] <= beat < s["b1"]), self.sections[-1])

    def event(self, name):
        try:
            return next(e for e in self.events if e["id"] == name)
        except StopIteration:
            raise KeyError(name) from None

    def ev(self, name):
        return self.event(name)["beat"]

    def find(self, kind=None, tag=None, section=None):
        return [e for e in self.events if (kind is None or e["kind"] == kind)
                and (tag is None or tag in e["tags"])
                and (section is None or e["section"] == section)]

    def with_tag(self, tag):
        return [s for s in self.sections if tag in s["tags"]]

    def breath_beats(self):
        return []
