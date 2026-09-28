"""THE LONG DAWN v3 - SOUND lane: A (EVERY STEP CLOSER), real recordings for A's cue sheet (music/v3/cues_A.json).

A's fire act: the crater of fire at THE EDGE, the vortex of THE BRINK (H5: fire physics, embers dragged up into one
roaring updraft), the IMPACT into white, the grey ash wind of the dead valley, then the night, the shared flint take,
the beacons catching ridge after ridge (KARST, DESERT, the run), the roped lantern line passing the watch-fires, and
the blue hour.  Levels are matched cue by cue to COMPOSER-A's synthesized designs.
"""
import sound_flint_v3 as FL

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
    "A.wind.night": dict(src=WIND, seg=(10, 20), xf=3.0, hp=45),
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
    "A.watchfire1": watchfire(), "A.watchfire2": watchfire(), "A.watchfire3": watchfire(),
    "A.watchfire4": watchfire(),
    "A.air.blue": dict(src=[("fs:725630", 108.0, 150.0, 1.0)], seg=(12, 20), xf=4.0, hp=60, lp=2500, trim=-2.0),
    "A.fire.blue": dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=-2.0),
                                dict(src=CRACKLE, seg=(10, 18), xf=2.5, g=0.0)], hp=90, lp=8000, width=0.8),
}
RECIPES.pop("A.catch", None)                                  # A's sheet has no catch cue: A.x.catch is it

EXTRA_BEDS = []
EXTRA_EVENTS = [
    dict(id="A.x.catch", t={"at": "catch"}),                  # the flint take's catch (+196), on every cut
]
RECIPES["A.x.catch"] = dict(FL.CATCH, level_from="A.blow", trim=1.0)

SPACE = dict(distance="forest20", outdoor="forest20", event_send=0.08, bed_send=0.0, wet_hp=150, wet_lp=9000,
             stem_hp=25)
