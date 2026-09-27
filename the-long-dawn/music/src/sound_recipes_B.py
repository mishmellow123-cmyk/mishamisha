"""THE LONG DAWN v3 - SOUND lane: B (THE KEEPER / THE VIGIL), every effect from a real recording.

Keys are the cue ids of music/v3/cues_B.json (locked; read, never edited) plus this lane's EXTRA events.
Sources ('fs:<id>' Freesound, 'el:<tag_nn>' ElevenLabs) and their licences are in
music/SFX_CREDITS.md.  Levels: each cue is matched to COMPOSER's synthesized design for the same cue, then `trim` dB.

Picture notes that shaped the choices (vigil.py, BIBLE_V3 B beat sheet, H5 calls):
  * B is wordless, so the sound carries it, and restraint carries it too.  There are no birds until bar 65 (the wind
    sources were checked for chirps), no insects, bells, engines or water anywhere, and no voices.
  * The summit is open air above a cloud sea. The only space is a short measured outdoor response (forest20, small
    send), with nothing synthesized.
  * THE VIGIL's locked frame is ~44 m from her fire, so the vigil's fire and feeds are air-absorbed (lp, dist).
  * The far answer is only pinpricks of light kilometres away, so no sound carries from them: B.his_fire is
    silent. The bar-40 flare of HER fire is added instead (F_FLARE in vigil.py: bar 40 b2).
  * The first birdsong (bar 65) is a robin at dawn, answered by a far robin 9.5 beats later: two birds counter-singing,
    the film's call and answer found in nature.  The source is a solo robin recorded very close with no other bird
    over it, so it is distanced here.
"""

import sound_flint_v3 as FL

# ---------------------------------------------------------------- the sources' usable ranges
WIND = [("fs:725630", 0.5, 150.0, 1.0)]                    # La Rhune: moderate mountain wind, Schoeps ORTF
GUSTS = [("fs:754256", 50.0, 72.0, 1.0), ("fs:754256", 150.0, 182.0, 1.0)]   # a real snowstorm's gusts
STORMBED = [("fs:754256", 2.0, 182.0, 1.0)]
FIRE_CLOSE = [("fs:681366", 1.0, 82.5, 1.0), ("fs:681367", 0.5, 21.5, 0.4)]  # a campfire, very quiet night
FIRE_WOOD = [("fs:595483", 12.0, 90.0, 1.0)]                # a wood fire, established

# ---------------------------------------------------------------- the feeds: a wood knock, then the fire answers
KNOCKS = [0.416, 3.162, 5.627, 2.136]                       # fs:734628 (dry logs), knock times in the source
BURSTS = [26.5, 20.0, 25.75, 10.0]                          # fs:595483 crackle swells (dB over its bed: 10, 5, 7, 13)


def feed(k, big=False):
    kn, bu = KNOCKS[k % len(KNOCKS)], BURSTS[(k * 3 + 1) % len(BURSTS)]
    if big:
        bu = 10.0
    return dict(layers=[
        dict(src=("fs:734628", kn), pre=0.02, post=0.45, hp=160, lp=6500, fo=0.2, g=0.0, dt=0.0),
        dict(src=("fs:595483", bu), pre=0.35, post=2.4, hp=120, lp=7000, fi=0.25, fo=1.2,
             g=-3.0 if not big else 0.0, dt=0.3),
    ], dist=0.35, width=0.6)


RECIPES = {
    # ---------------------------------------------------------------- B1-B3: dusk, the climb, the dead ember
    "B.wind.dusk": dict(src=WIND, seg=(10, 20), xf=3.0, hp=40),
    "B.wind.climb": dict(src=WIND + GUSTS, seg=(6, 12), xf=2.0, hp=80),
    "B.spindrift": dict(src=STORMBED, seg=(5, 10), xf=1.5, hp=2600, lp=12000, trim=-2.0),
    "B.wind.ember": dict(src=WIND, seg=(10, 18), xf=3.0, hp=50, trim=-1.0),
    "B.lid": dict(src=("fs:453789", 12.022), pre=0.03, post=0.55, hp=110, fo=0.12),
    "B.blow_ember": dict(layers=[                            # two breathy blows (real breath, the mic pops cut)
        dict(src=("fs:848421", 2.62), pre=0.04, post=1.55, hp=400, fo=0.35, g=0.0, dt=0.0),
        dict(src=("fs:273979", 3.12), pre=0.04, post=1.3, hp=400, fo=0.4, g=-2.0, dt=1.75),
        dict(src=("fs:681366", 30.0), pre=0.0, post=3.2, hp=1500, fi=0.4, fo=1.2, g=-16.0, dt=0.0),
    ]),
    "B.ember_hiss": dict(src=("fs:660297", 28.6), pre=0.3, post=1.7, hp=1400, lp=9000, fi=0.3, fo=0.9, trim=-3.0),
    # ---------------------------------------------------------------- B4-B5: the first fire (the shared flint take)
    **FL.take("B", blow_id="B.blow_tinder"),               # THE FLINT TAKE: the same recordings in A, B and C
    "B.fire.reveal": dict(layers=[dict(src=FIRE_WOOD, seg=(10, 18), xf=2.5, g=0.0),      # the fire's body ...
                                  dict(src=FIRE_CLOSE, seg=(8, 16), xf=2.0, g=-2.0)],  # ... and its crackle
                          hp=70, trim=-1.5),       # under her CALL (bar 18 b3): B's first half stays 6 LU down
    "B.wind.night1": dict(src=WIND, seg=(10, 20), xf=3.0, hp=45),
    # ---------------------------------------------------------------- B6-B12: THE VIGIL (one night, the locked frame)
    "B.fire.vigil": dict(layers=[dict(src=FIRE_WOOD, seg=(12, 22), xf=3.0, g=0.0),
                                 dict(src=FIRE_CLOSE, seg=(10, 20), xf=3.0, g=-3.0)],
                         hp=90, lp=6500, width=0.5, trim=-1.0),                       # ~44 m away: air-absorbed
    "B.wind.life": dict(src=WIND, seg=(10, 20), xf=3.0, hp=45),
    "B.storm.year_2": dict(src=GUSTS, seg=(5, 9), xf=1.5, hp=90),
    "B.snow.year_2": dict(src=STORMBED, seg=(4, 8), xf=1.2, hp=2600, lp=12000),
    "B.wind.far_peak_1": dict(src=WIND, seg=(6, 10), xf=2.0, hp=50, trim=-1.0),
    "B.wind.far_peak_2": dict(src=WIND, seg=(6, 10), xf=2.0, hp=50, trim=-1.0),
    "B.feed.stone_1": feed(0), "B.feed.stone_2": feed(1), "B.feed.stone_3": feed(2), "B.feed.stone_5": feed(3),
    "B.feed.stone_8": feed(4), "B.feed.stone_12": feed(5), "B.feed.stone_20": feed(6), "B.feed.stone_30": feed(7),
    "B.feed.stone_45": feed(8), "B.feed.y60_strikes": feed(9),
    "B.traveller": dict(layers=[
        dict(src=("fs:595483", 26.5), pre=0.3, post=2.0, hp=120, lp=7000, fi=0.2, fo=0.9, g=0.0, dt=0.0),
        dict(src=("el:B_torch_02", 0.05), pre=0.05, post=3.3, hp=150, fi=0.15, fo=1.2, g=-4.0, dt=0.25),
    ], dist=0.35, width=0.6),
    "B.his_fire": dict(skip=True, why="THE VIGIL: the far answer is a pinprick kilometres off; no sound carries (her own "
                                      "flare on bar 40 b2 is B.x.flare)"),
    # ---------------------------------------------------------------- B13-B14: the hand-back, the first birdsong
    "B.wind.high": dict(layers=[dict(src=WIND, seg=(8, 14), xf=2.5, g=0.0),            # altitude: the same wind,
                                dict(src=STORMBED, seg=(6, 10), xf=2.0, g=-9.0)],      # a little of the storm's
                        hp=120, shelves=[("highshelf", 3000, 3.0)]),                   # air, brighter
    "B.air.dawn": dict(src=[("fs:725630", 108.0, 150.0, 1.0)], seg=(12, 20), xf=4.0, hp=60, lp=2500, trim=-3.0),
    "B.birds": dict(src=("fs:725219", 15.94), pre=0.4, post=12.8, hp=1800, lp=11000, fo=1.0, dist=0.4),
    "B.birds2": dict(src=("fs:725219", 79.52), pre=0.3, post=4.0, hp=1800, lp=11000, fo=0.8, dist=0.7),
}

EXTRA_BEDS = []
EXTRA_EVENTS = [
    dict(id="B.x.flare", t={"at": "far_peak_2", "beats": 1.0}),   # bar 40 b2: her fire flares (vigil.py F_FLARE)
]
RECIPES["B.x.flare"] = dict(feed(10, big=True), level_from="B.feed.stone_20", trim=2.0)

RECIPES["B.roar"] = dict(FL.ROAR, trim=-1.5)                 # under the reveal: B's first half stays 6 LU down

SPACE = dict(distance="forest20", outdoor="forest20", event_send=0.10, bed_send=0.0, wet_hp=150, wet_lp=9000,
             stem_hp=25)
