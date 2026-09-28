"""THE LONG DAWN v3 - SOUND lane: A (EVERY STEP CLOSER), real recordings for A's cue sheet (music/v3/cues_A.json).

A's fire act: the crater of fire at THE EDGE, the vortex of THE BRINK (H5: fire physics, embers dragged up into one
roaring updraft), the IMPACT into white, the grey ash wind of the dead valley, then the night, the shared flint take,
the beacons catching ridge after ridge (KARST, DESERT, the run), the roped lantern line passing the watch-fires, and
the blue hour.  Levels are matched cue by cue to COMPOSER-A's synthesized designs.
"""
import json
import os
import sound_flint_v3 as FL

# SOUND-C: the first fire laid on the frames where the PICTURE does it (sound/picture_sync_A.json, measured)
_PS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sound", "picture_sync_A.json")
PICTURE = {k: v["f"] for k, v in json.load(open(_PS))["t"].items()}

WIND = [("fs:725630", 0.5, 150.0, 1.0)]                      # moderate mountain wind (Schoeps ORTF)
STORM = [("fs:754256", 2.0, 182.0, 1.0)]                     # a real snowstorm's gusts
UPDRAFT = [("sn:eye_of_the_storm", 0.5, 59.5, 1.0)]          # 344 Audio: a storm's full, wide roar
FIRE = [("fs:595483", 12.0, 90.0, 1.0)]                      # a wood fire, established
CRACKLE = [("fs:681366", 1.0, 82.5, 1.0)]                    # a small wood fire on a very quiet night
OIL = [("sn:oil_flare", 5.0, 19.5, 1.0), ("sn:oil_flare", 22.5, 59.5, 1.0)]   # a campfire fed with oil: pops
LAVA = [("fs:172630", 1.0, 57.5, 1.0)]                       # molten rock seething: the crater's weight
PINE = [("sn:pine_branches", 2.0, 85.0, 1.0)]                # fresh pine branches in a fire: sizzle and spit

# the real flare-ups available (source, onset): oil igniting (Sonniss) and kindling taking (Freesound)
FLARES = [("sn:oil_flare", 2.974), ("fs:595483", 9.944), ("sn:oil_flare", 20.288), ("fs:595483", 26.558)]


def flare(k, dist, stretch=1.0):
    ref, t = FLARES[k % len(FLARES)]
    return dict(src=(ref, t), pre=0.3, post=2.8, hp=90, fi=0.05, fo=1.4, stretch=stretch, dist=dist)


def watchfire():
    """the lantern line passes a watch-fire: five seconds of a close fire, in and out"""
    return dict(src=("fs:681366", 50.0), pre=2.5, post=2.5, hp=90, fi=1.6, fo=1.8, width=0.8)


RECIPES = {
    # ---------------------------------------------------------------- the ridge at midnight, false dawn
    "A.wind.open": dict(src=WIND, seg=(10, 20), xf=3.0, hp=40),
    # ---------------------------------------------------------------- the edge, the brink, the impact
    "A.fire.edge": dict(layers=[dict(src=FIRE, seg=(8, 14), xf=2.0, g=0.0),
                                dict(src=PINE, seg=(8, 14), xf=2.0, g=-4.0),
                                dict(src=LAVA, seg=(8, 14), xf=2.0, g=-7.0)], hp=40, width=1.0),
    "A.storm.brink": dict(layers=[dict(src=UPDRAFT, seg=(6, 10), xf=1.5, g=0.0),
                                  dict(src=FIRE, seg=(6, 10), xf=1.5, g=-3.0),
                                  dict(src=LAVA, seg=(6, 10), xf=1.5, g=-6.0)], hp=35),
    "A.impact": dict(layers=[
        dict(src=("sn:oil_flare", 20.288), pre=0.17, post=3.05, hp=40, fi=0.01, fo=1.8, g=0.0, dt=0.0),
        dict(src=("fs:595483", 9.944), pre=0.09, post=4.46, hp=35, fi=0.01, fo=2.2, g=-2.0, dt=0.0),
        dict(src=("fs:172630", 44.0), pre=0.0, post=5.0, hp=30, fi=0.02, fo=3.0, g=-6.0, dt=0.0),
    ]),
    # ---------------------------------------------------------------- the dead valley, the night
    "A.wind.ash": dict(layers=[dict(src=WIND, seg=(8, 14), xf=2.0, g=0.0),
                               dict(src=STORM, seg=(6, 10), xf=1.5, g=-10.0)], hp=60, lp=7000),
    # the flint take is close on her hands: the night wind drops 6 dB from 3340 to the roar so her three blows, the
    # in-breaths, the puff and the catch are heard over it (they stay at their designs' levels); t from the bed's 2848
    "A.wind.night": dict(src=WIND, seg=(10, 20), xf=3.0, hp=45,
                         env=[(0.0, 0.0), (20.5, 0.0), (21.0, -6.0), (31.2, -6.0), (31.75, 0.0)]),
    # ---------------------------------------------------------------- the first fire: the shared flint take
    **FL.take("A"),
    # ---------------------------------------------------------------- the beacons: ridge after ridge
    "A.karst_flare": flare(0, 0.55), "A.desert_fire": flare(1, 0.5, 0.97),
    "A.run_1": flare(2, 0.2), "A.run_2": flare(3, 0.3, 1.03), "A.run_3": flare(0, 0.35, 0.96),
    "A.run_4": flare(1, 0.45, 1.04), "A.run_5": flare(2, 0.5, 0.98), "A.run_6": flare(3, 0.6, 1.02),
    "A.run_7": flare(0, 0.65, 1.05),
    "A.wind.watch": dict(src=WIND, seg=(10, 20), xf=3.0, hp=45),
    # ---------------------------------------------------------------- the crossing, the watch-fires, the blue hour
    "A.wind.crossing": dict(layers=[dict(src=WIND, seg=(8, 14), xf=2.5, g=0.0),
                                    dict(src=STORM, seg=(6, 10), xf=2.0, g=-12.0)], hp=50),
    # measured scale (A18): watch-fire 1 passes big in the foreground (4990-5040); 2 is small mid-frame (5230); 3 and 4
    # are pinpricks in the wide frames: the same fire heard from farther off (level still matched to the design)
    "A.watchfire1": watchfire(), "A.watchfire2": dict(watchfire(), dist=0.4),
    "A.watchfire3": dict(watchfire(), dist=0.6), "A.watchfire4": dict(watchfire(), dist=0.7),
    "A.air.blue": dict(src=[("fs:725630", 108.0, 150.0, 1.0)], seg=(12, 20), xf=4.0, hp=60, lp=2500, trim=-2.0),
    "A.fire.blue": dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=-2.0),
                                dict(src=CRACKLE, seg=(10, 18), xf=2.5, g=0.0)], hp=90, lp=8000, width=0.8),
}
RECIPES.pop("A.catch", None)                                  # A's sheet has no catch cue: A.x.catch is it

EXTRA_BEDS = []
EXTRA_EVENTS = [
    dict(id="A.x.catch", t={"at": "catch"}),                  # the flint take's catch (+196), on every cut
]
# the kindling takes on 3563 (measured): the catch must be HEARD (final_A had none: the flames grew in silence), then
# the fire it lit grows under the picture's flames into the roar (3600)
RECIPES["A.x.catch"] = dict(FL.CATCH, level_from="A.blow", trim=6.0)
EXTRA_BEDS.append(dict(id="A.x.take", t0={"at": "catch", "frames": 5}, t1={"at": "roar", "frames": 2},
                       fade_in=0.25, fade_out=0.12))
RECIPES["A.x.take"] = dict(src=FIRE, seg=(3, 4), xf=0.5, hp=120, level=-36.0, env=[(0.0, -12.0), (1.55, 0.0)])
# the blow, laid on the picture's breaths: the ember brightens on three blows (3447-3469, 3479-3503, 3513-3539) and
# dims on each in-breath; a last puff 3556-3561 (dt from the first blow, 3447). The same breath takes as B's.
RECIPES["A.blow"] = dict(layers=[
    dict(src=("fs:848421", 0.33), pre=0.03, post=0.95, hp=400, fo=0.2, g=8.0, dt=0.0),                # blow 1
    dict(src=("fs:273979", 1.55), pre=0.02, post=0.32, hp=400, fi=0.1, fo=0.1, g=-10.0, dt=0.97),     # in-breath
    dict(src=("fs:273979", 3.17), pre=0.04, post=1.0, hp=400, fo=0.25, g=0.0, dt=1.333),              # blow 2
    dict(src=("fs:273979", 1.85), pre=0.02, post=0.32, hp=400, fi=0.1, fo=0.1, g=-10.0, dt=2.38),     # in-breath
    dict(src=("fs:848421", 2.55), pre=0.04, post=1.07, hp=400, fo=0.3, g=-1.0, dt=2.75),              # blow 3
    dict(src=("fs:406648", 2.25), pre=0.03, post=0.3, hp=400, fo=0.12, g=-4.0, dt=4.54),              # the puff
    dict(src=("fs:595483", 7.3), pre=0.0, post=4.9, hp=700, fi=0.8, fo=0.3, g=-12.0, dt=0.0,          # the ember
         env=[(0.0, -14.0), (0.96, -11.0), (1.05, -18.0), (1.33, -10.0), (2.37, -9.0), (2.45, -16.0),
              (2.75, -8.0), (3.85, -6.0), (3.95, -14.0), (4.54, -5.0), (4.9, -3.0)]),              # glowing up
])

SPACE = dict(distance="forest20", outdoor="forest20", event_send=0.08, bed_send=0.0, wet_hp=150, wet_lp=9000,
             stem_hp=25)
