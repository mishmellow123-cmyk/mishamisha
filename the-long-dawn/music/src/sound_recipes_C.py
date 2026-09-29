"""THE LONG DAWN v3 - SOUND lane: C (THE LAST PAGES), real recordings for the cue sheet AND for COMPOSER-C's own
effects (score_v3_C.extra_effects: the pen, the burn-throughs, the drop, the cock, the cold metal's tick, the seethe,
the torches' dip, the roads, the fire that remains, the fire that rises), which are read, never edited.

C is the BOOK: the storyteller's hearth, heavy old pages, a quill, paper that burns through; then the ink world
(wind, fire, beacons), the council's torches and a hooded murmur that falls silent when she sets the Ring down, the
sea at the Havens, and the hearth again.  Levels are matched to COMPOSER-C's designs cue by cue.
"""
import json
import os
import sound_flint_v3 as FL

# SOUND-C: every page, pen and burn (and the council's dips) sits on the frame where the PICTURE does it, measured
# from the frames by sync_audit_v3.py (the table, with what was measured and why: sound/picture_sync_C.json)
_PS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sound", "picture_sync_C.json")
PICTURE = {k: v["f"] for k, v in json.load(open(_PS))["t"].items()}

HEARTH = [("fs:681366", 1.0, 82.5, 1.0)]                    # a small wood fire, close, on a very quiet night
FIRE = [("fs:595483", 12.0, 90.0, 1.0)]                      # a wood fire, established
STORM = [("fs:754256", 2.0, 182.0, 1.0)]                     # a real snowstorm's gusts
WIND = [("fs:725630", 0.5, 150.0, 1.0)]                      # moderate mountain wind (Schoeps ORTF)
SEA = [("sn:soft_waves_cliffs", 0.5, 119.5, 1.0)]            # JSE: soft waves at the cliffs of a northern coast
LAVA = [("fs:172630", 1.0, 57.5, 1.0)]                       # molten rock seething (the melt, the blaze's weight)
PAPER = "fs:528662"                                          # a sheet of paper burning, 65 s
QUILL = "fs:194905"                                          # a quill on hard paper, various speeds
PAGES = "sn:slow_page_turns"                                 # 344 Audio: an antique book's pages, turned slowly
FLIP = "sn:flicking_pages"                                   # 344 Audio: the same books, flicked through


def page(src, hit, pre=0.5, post=1.0, **kw):
    """a leaf turned: the lift's rustle rises under the page as it lifts (-14 dB at the clip's start), the loudest
    moment is the leaf crossing over (the hit, on the measured crossing frame), then it lands and settles"""
    # after the crossing the leaf lands within 0.1-0.2 s and lies still: its later rustles are held down, so no
    # second "page" is heard over a still page (the blank take has a pop +0.42 s nearly as loud as its crossing)
    kw.setdefault("env", [(-pre, -14.0), (-0.12, -4.0), (0.0, 0.0), (0.2, -2.0), (0.45, -9.0), (post, -15.0)])
    return dict(src=(src, hit), pre=pre, post=post, hp=120, fi=0.08, fo=0.3, crest=13.0, **kw)


def pen(start, dur):
    return dict(src=(QUILL, start), pre=0.0, post=dur, hp=250, fi=0.05, fo=min(0.3, dur / 3), width=0.7, crest=12.0)


def burn(start, dur, swell=0.5):
    """paper catching at a burn-through: the crackle swells in over `swell` s to the hit, then burns on"""
    return dict(src=(PAPER, start), pre=swell, post=dur, hp=200, fi=swell, fo=min(1.2, dur / 2), crest=12.0,
                env=[(-swell, -10.0), (0.0, 0.0), (dur, -4.0)])


def flare(src_t, dist, stretch=1.0):
    """a far beacon catching: the same real whoomp as the roar, at distance (src_t = the burst's measured arrival:
    8 dB under its peak, so the flame's burst lands on the frame)"""
    return dict(layers=[dict(src=("fs:595483", src_t), pre=0.5, post=2.6, hp=90, fi=0.05, fo=1.2, stretch=stretch,
                             g=0.0, dt=0.0)], dist=dist)


RECIPES = {
    # ---------------------------------------------------------------- the storyteller's hearth, the book
    "C.hearth.open": dict(src=HEARTH, seg=(10, 18), xf=2.5, hp=80, lp=9000, width=0.7),
    "C.page_turn_0": page(PAGES, 7.028, pre=0.6, post=0.9),
    # the riffle: the flicking take's bursts span 29 frames, exactly the picture's flutter (320 -> lands 349)
    "C.riffle": page(FLIP, 0.208, pre=0.1, post=1.6, env=[(-0.1, -6.0), (0.0, 0.0), (1.25, 0.0), (1.6, -12.0)]),
    "C.page_turn": page(PAGES, 0.705, pre=0.45, post=1.0),
    # the pen draws the mountain 355-535 (crater, slopes, hatching, cloud 355-454; the dense sky hatching 455-491;
    # hills and rocks to 535): one quill take whose level follows the drawing
    "C+.pen.mountain": dict(pen(63.5, 7.8), fo=0.25, env=[(0.0, -5.0), (1.2, -4.0), (4.1, -3.0), (4.2, 0.0),
                                                        (5.65, 0.0), (5.75, -4.0), (7.1, -6.0), (7.5, -14.0)]),
    "C+.pen.T1": pen(9.15, 1.0),
    "C+.pen.deep": pen(17.3, 8.6),                       # first stroke 1705, the last ~1912
    "C+.pen.T7": pen(28.75, 1.0),
    "C+.pen.T9": pen(31.55, 1.0),
    "C+.pen.T14": pen(40.6, 1.0),
    "C+.burn.letters": burn(8.0, 3.0, 0.6),
    # the ember edge 1686 -> 1700: rises with it, loudest mid-sweep, and dies with it (clean parchment from 1700,
    # the pen from 1705)
    "C+.burn.deep": dict(burn(22.0, 1.0, 0.42), env=[(-0.42, -10.0), (0.0, 0.0), (0.25, -3.0), (0.6, -14.0),
                                                   (1.0, -30.0)]),        # pages_book: the sweep 1689-1705
    "C+.burn.eye": burn(33.5, 2.2, 0.3),                 # the glow 1921, the hole 1924
    # the X1s (PAGES-C's x1burn mattes): the burn opens over ~28-35 frames; the crackle rises from its first frame,
    # peaks as half the sheet is open and dies with it
    "C+.burn.map": dict(burn(45.0, 1.8, 0.62), env=[(-0.62, -12.0), (0.0, 0.0), (0.25, -2.0), (0.55, -8.0), (1.8, -26.0)]),
    "C+.burn.remains": dict(burn(52.0, 1.8, 0.79), env=[(-0.79, -12.0), (0.0, 0.0), (0.3, -2.0), (0.65, -8.0), (1.8, -26.0)]),
    "C+.burn.title": burn(58.5, 2.6, 0.54),              # first spark 6982, the letters burn on to 7016
    # ---------------------------------------------------------------- the fire born from the page, the forge
    "C.fire.born": dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=0.0),
                                dict(src=HEARTH, seg=(8, 14), xf=2.0, g=-3.0)], hp=70),
    "C.fire.forge": dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=0.0),
                                 dict(src=LAVA, seg=(8, 14), xf=2.0, g=-8.0)], hp=50),
    "C.storm.eye": dict(src=STORM, seg=(6, 10), xf=1.5, hp=60),
    "C+.drop": dict(crest=14.0, peak_room=2.0, layers=[dict(src=("fs:197900", 0.40), pre=0.02, post=0.45, hp=300, fo=0.1, g=0.0, dt=0.0),
                            dict(src=("fs:197900", 0.10), pre=0.02, post=0.25, hp=300, fo=0.1, g=-8.0, dt=0.44)]),
                            # the strike on f2200 exactly; the bead falling back ~f2210.6 (MIRROR, 21:10Z)
    "C+.cock": dict(src=("fs:482119", 0.13), pre=0.1, post=2.2, hp=400, lp=3500, fo=0.5, dist=0.85, width=0.3),
    "C.wind.fall": dict(src=WIND, seg=(10, 18), xf=3.0, hp=50),
    "C.snow_hiss": dict(layers=[
        dict(src=("fs:541035", 0.06), pre=0.02, post=1.2, hp=500, fo=0.4, g=0.0, dt=0.0),        # the quench
        dict(src=("fs:870180", 0.3), pre=0.0, post=2.3, hp=800, fi=0.1, fo=1.2, g=-8.0, dt=0.1),  # the steam
    ]),
    # ---------------------------------------------------------------- the find, the flint take (shared), the test
    **FL.take("C"),
    "C+.coldtick": dict(layers=[dict(src=("fs:185608" if k % 2 else "fs:185609", 0.005), pre=0.004, post=0.09,
                                     hp=1500, fo=0.03, stretch=1.0 + 0.03 * ((k * 7) % 5 - 2), g=-2.0 * (k % 3),
                                     dt=d)
                                for k, d in enumerate((0.0, 0.62, 1.55, 2.1, 3.3, 4.05, 5.2, 6.1))]),
    # ---------------------------------------------------------------- the run: beacons catching at distance
    "C.beacon1": flare(9.944, 0.23), "C.beacon2": flare(26.558, 0.31, 0.97), "C.beacon3": flare(9.944, 0.39, 1.03),
    "C.beacon4": flare(20.070, 0.47, 0.96), "C.beacon5": flare(26.558, 0.55, 1.04), "C.beacon6": flare(9.944, 0.63, 0.98),
    "C.beacon7": flare(20.070, 0.71, 1.02),
    # ---------------------------------------------------------------- the council: torches, murmur, the unmaking
    "C.hearth.council": dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=0.0),
                                     dict(src=HEARTH, seg=(8, 14), xf=2.0, g=-2.0)], hp=90, width=1.0),
    "C+.fire.rises": dict(layers=[dict(src=FIRE, seg=(8, 12), xf=1.5, g=0.0),
                                  dict(src=LAVA, seg=(6, 10), xf=1.5, g=-6.0)], hp=45, crest=12.0,
                          env=[(0.0, -12.0), (8.0, 0.0)]),
    "C+.seethe": dict(crest=14.0, layers=[dict(src=("fs:172630", 20.0), pre=0.0, post=7.0, hp=120, fi=0.4, fo=0.3, g=0.0, dt=0.0,
                                   env=[(0.0, -8.0), (1.5, -2.0), (1.9, 0.0), (3.4, -3.0), (3.5, -40.0)]),
                              dict(src=("fs:194635", 4.0), pre=0.0, post=3.4, hp=900, fi=0.3, fo=0.25, g=-6.0, dt=0.0,
                                   env=[(0.0, -12.0), (1.8, 0.0), (3.35, -30.0)])]),
    "C+.dip.0": dict(src=("fs:426208", 0.5), pre=0.35, post=0.9, hp=150, lp=6000, fo=0.4, pan=-0.35),
    "C+.dip.1": dict(src=("fs:426208", 0.5), pre=0.35, post=0.9, hp=150, lp=6000, fo=0.4, pan=0.3, stretch=1.06),
    "C+.dip.2": dict(src=("fs:426208", 0.5), pre=0.35, post=0.9, hp=150, lp=6000, fo=0.4, pan=-0.1, stretch=0.95),
    "C+.dip.3": dict(src=("fs:426208", 0.5), pre=0.35, post=0.9, hp=150, lp=6000, fo=0.4, pan=0.45, stretch=1.1),
    "C+.dip.4": dict(src=("fs:426208", 0.5), pre=0.35, post=0.9, hp=150, lp=6000, fo=0.4, pan=-0.5, stretch=0.92),
    "C+.fire.remains": dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=0.0),
                                    dict(src=HEARTH, seg=(8, 14), xf=2.0, g=-3.0)], hp=70),
    "C+.roads": dict(layers=[dict(src=("fs:681366", 20.0 + 7 * k), pre=0.0, post=0.9, hp=300, fi=0.15, fo=0.5,
                                  g=-3.0 * k, dt=0.45 * k) for k in range(6)], dist=0.5),
    "C.air.dawn": dict(src=[("fs:725630", 108.0, 150.0, 1.0)], seg=(12, 20), xf=4.0, hp=60, lp=2500, trim=-3.0),
    # ---------------------------------------------------------------- plenty, the Havens, the last pages
    "C.hearth.end": dict(src=HEARTH, seg=(10, 18), xf=2.5, hp=80, lp=9000, width=0.7),
    "C.page.plenty": dict(skip=True, why="the picture cuts to the book on 6160 and no page turns (measured): a page "
                                         "sound there is the wrong event (PAGES-C: a turn into C25 gets it back)"),
    "C.sea": dict(src=SEA, seg=(14, 24), xf=3.5, hp=40, lp=10000),
    "C.page.havens": page(PAGES, 9.016, pre=0.6, post=0.9),
    "C.page.blank": page(PAGES, 12.443, pre=0.6, post=0.9),
    "C.page.blank_2": page(PAGES, 4.887, pre=0.45, post=0.8),
}

EXTRA_BEDS = [
    # the council's murmur: hooded, low, no words; it falls away as she sets the Ring on the stone
    dict(id="C.x.murmur", t0={"at": "council"}, t1={"at": "ring_set"}, fade_in=3.0, fade_out=2.5),
]
EXTRA_EVENTS = [
    dict(id="C.x.catch", t={"at": "catch"}),                  # the flint take's catch (+196), on every cut
]
# the murmur must be HEARD under the council's clarinet and chorale (score ~-22 LUFS short-term there) for its falling
# silent at ring_set to register: ~12 dB under the score, low-passed so no word is intelligible
RECIPES["C.x.murmur"] = dict(src=[("fs:766658", 4.0, 150.0, 1.0)], seg=(12, 20), xf=3.0, hp=150, lp=2200,
                             width=0.8, level=-50.0)
RECIPES["C.x.catch"] = dict(FL.CATCH, level_from="C.blow", trim=1.0)
RECIPES.pop("C.catch", None)                                  # C's sheet has no catch cue: C.x.catch is it
RECIPES["C+.sea.near"] = dict(src=SEA, seg=(14, 24), xf=3.5, hp=40, width=1.0)   # the Havens, nearer the water
# C5's forge hammers (the Trap's ostinato accents, the lone hammer under the map's pause, the stroke cut on the
# shutdown): VSCO-2-CE's anvil, the recording the score's forging already plays (score_v3_C "anvil"), so the Trap's
# strikes are the forges' own. Hard hit v3: peak at 0.002 s, 40 dB down by 0.67 s; medium hit v2 for the faint one.
ANVIL, ANVIL_SOFT = "vsco:Percussion/Anvil_Hit1_v3_Sum.wav", "vsco:Percussion/Anvil_Hit1_v2_Sum.wav"
RECIPES["C5.hammer"] = dict(src=(ANVIL, 0.002), pre=0.002, post=1.4, hp=90, fo=0.4, width=0.4, dist=0.5)
RECIPES["C5.hammer.faint"] = dict(src=(ANVIL_SOFT, 0.002), pre=0.002, post=1.2, hp=90, fo=0.4, width=0.3, dist=0.8)

SPACE = dict(distance="forest20", outdoor="forest20", event_send=0.08, bed_send=0.0, wet_hp=150, wet_lp=9000,
             stem_hp=25, bed_crest=14.0)


def extra_cues(bm):
    """COMPOSER-C's own effects (in memory, as score_v3_C.build adds them), level-matched to its designs"""
    import score_v3_C
    return score_v3_C.extra_effects(bm)
