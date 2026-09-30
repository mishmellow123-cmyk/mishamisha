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
    # Historical ash envelope, t from 2640. syncA's finished-frame scan now finds full white at 2638-2639,
    # then the valley aperture on 2640 (mean Y 255 -> 151.65): the old white-hold claim is stale.
    # Do not move this bed independently of the impact, score and their shared 2637-2640 breath.
    "A.wind.ash": dict(layers=[dict(src=WIND, seg=(8, 14), xf=2.0, g=0.0),
                               dict(src=STORM, seg=(6, 10), xf=1.5, g=-10.0)], hp=60, lp=7000,
                       env=[(0.0, -20.0), (0.667, -20.0), (2.0, 0.0), (5.083, 0.0), (6.5, -24.0), (6.667, -40.0)]),
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

# syncA (29 Sep): reconciled sound_AP2's strike1 rises at A3358.752 (4-12 kHz, causal order-4
# bandpass, trailing 5 ms RMS / 1 ms hop, peak -20 dB; adjacent 3348-3356 solo control is silent), while
# the hands and first spark enter on A3360. Keep the click's hit and the approved loudness target;
# suppress its pre-cut scrape until the last half-frame, with a quarter-frame fade to the click.
# Strikes 2/3 arrive at 3387.912/3417.312 against first sparks 3388/3417: leave those alone.
# env_after is applied before match_gain by sound_v3.build; the owner must remeasure the new master.
RECIPES["A.strike1"] = dict(RECIPES["A.strike1"],
                            env_after=[(-0.09, -120.0), (-0.5 / 24.0, -120.0),
                                       (-0.25 / 24.0, 0.0), (0.55, 0.0)])

EXTRA_BEDS = []
EXTRA_EVENTS = [
    dict(id="A.x.catch", t={"at": "catch"}),                  # the flint take's catch (+196), on every cut
]
# syncA: the finished catch ROI first has a pixel above R180/Y140 at 3555 (none on 3540-3554),
# then seven at 3557. The reconciled AP2 catch's mix proxy starts at 3554.856 (-0.144 f, within the
# hard tolerance): retain its 3556 hit. The growing-fire bed follows that authored hit by five frames.
RECIPES["A.x.catch"] = dict(FL.CATCH, level_from="A.blow", trim=6.0)
EXTRA_BEDS.append(dict(id="A.x.take", t0={"at": "catch", "frames": 5}, t1={"at": "roar", "frames": 2},
                       fade_in=0.25, fade_out=0.12))
RECIPES["A.x.take"] = dict(src=FIRE, seg=(3, 4), xf=0.5, hp=120, level=-36.0, env=[(0.0, -12.0), (1.55, 0.0)])
# the blow, laid on the picture's breaths (h1_A, measured): the ember brightens on three blows 3448-3470, 3481-3505,
# 3515-3541 and dims on each in-breath; no puff (the catch flares on 3556). dt from the first blow (3448). B's breath takes.
RECIPES["A.blow"] = dict(layers=[
    dict(src=("fs:848421", 0.33), pre=0.03, post=0.95, hp=400, fo=0.2, g=8.0, dt=0.0),                # blow 1
    dict(src=("fs:273979", 1.55), pre=0.02, post=0.36, hp=400, fi=0.1, fo=0.1, g=-10.0, dt=0.958),    # in-breath
    dict(src=("fs:848421", 2.55), pre=0.04, post=1.02, hp=400, fo=0.25, g=-1.0, dt=1.375),            # blow 2
    dict(src=("fs:273979", 1.85), pre=0.02, post=0.32, hp=400, fi=0.1, fo=0.1, g=-10.0, dt=2.417),    # in-breath
    dict(src=("fs:848421", 3.40), pre=0.04, post=1.0, hp=400, fo=0.3, stretch=1.1, g=0.0, dt=2.792),  # blow 3
    dict(src=("fs:273979", 3.17), pre=0.04, post=0.6, hp=400, fo=0.3, g=-4.0, dt=3.56),               # its last
    # push: the ember flares brightest at the end of blow 3 (3536-3539)
    dict(src=("fs:273979", 1.6), pre=0.02, post=0.4, hp=400, fi=0.12, fo=0.15, g=-12.0, dt=3.917),    # in-breath
    dict(src=("fs:595483", 7.3), pre=0.0, post=4.4, hp=700, fi=0.8, fo=0.3, g=-12.0, dt=0.0,          # the ember
         env=[(0.0, -14.0), (0.95, -11.0), (1.0, -18.0), (1.375, -10.0), (2.40, -9.0), (2.46, -16.0),
              (2.79, -8.0), (3.88, -6.0), (3.95, -14.0), (4.4, -10.0)]),                             # glowing up
])
# the roar blooms on the flames' leap (3598-3608) and then HER FIRE (3612-3660, B's reveal) pulls back: the fire's
# footprint shrinks ~2.7x, so the roar settles and recedes under it (as B's did), -9 dB by 3660
RECIPES["A.roar"] = dict(layers=[dict(ly, post=3.0, fo=1.6, env=[(0.0, 0.0), (0.5, -1.0), (1.0, -4.0), (2.5, -9.0),
                                                                (3.0, -14.0)]) for ly in FL.ROAR["layers"]])

SPACE = dict(distance="forest20", outdoor="forest20", event_send=0.08, bed_send=0.0, wet_hp=150, wet_lp=9000,
             stem_hp=25)

# A4/A5: keep the old note/performance cache and reshape its local premaster only.
# AP2 forwards this hook at lookup time, as it does every other A effects attribute.
from polish_ignition_A import prepare_master, bound_master, refresh_breath_probes  # noqa: E402

# A 2400-3360 (polishdoom): the updraft rises into the white and carries through the old breath, the ash wind is up
# on the white, recorded air holds the black and the ember arrives on its first light. AP2 forwards these hooks as it
# does every other A recipe attribute; their frames lie inside polish_ignition_A.MASTER_WINDOWS.
from sound_polishdoom import BED_TIMING, install as _install_doom, polish_score  # noqa: E402
_install_doom(RECIPES, EXTRA_BEDS, EXTRA_EVENTS, WIND)

# ---------------------------------------------------------------- A 3600-6479 on the delivered picture (29 Sep night)
# The second half's effects were laid on the bar grid while its shots were slates. sound_a_table.py builds them from the
# MEASURED picture (music/v3/events_A_measured.json; snapshot sound/a_sound_events.json): a far flare on each of the 45
# ridge catches (3686-3738), the beacon run's links far to near on their measured catches, the watchers' hearth from its
# catch to the cut, watch-fire 1 from the crossing's first frame on its measured footprint, the far watch-fire until the
# great lantern covers it, and the blue hour's fire bed paling with EDIT's watch-fires. The bar-grid cues they replace
# keep their recipes above and are retired here with the reason (sound_v3 prints them as SKIP); every other cue is
# untouched. AP2 (score pass 2) forwards to this module, so it plays these rows too.
import sound_a_table as _T  # noqa: E402

A_TABLE = _T.build()
_BAD = _T.problems(A_TABLE)
if _BAD:
    raise SystemExit("REFUSED: A's second-half sound table breaks its contract:\n  " + "\n  ".join(_BAD))


def _measured_recipe(r):
    """a table row -> a sound_v3 recipe: flares are SOUND's real flare-ups (flare() above), with no pre-roll before the
    measured first light; fire beds are A.fire.blue's recordings (an established wood fire and its crackle)"""
    if r["recipe"] == "flare":
        rc = flare(r["flare_index"], r["dist"], r.get("stretch", 1.0))
        rc.update(pre=r["pre"], fi=r["fi"], post=r["post"], fo=r["fo"])
    else:
        rc = dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=-2.0), dict(src=CRACKLE, seg=(10, 18), xf=2.5, g=0.0)],
                  hp=90, lp=r["lp"], width=r["width"])
    rc["level"] = r["level"]                  # an absolute target (sound_v3.match_gain), from an approved level
    if r.get("pan") is not None:
        rc["pan"] = r["pan"]
    return rc


for _rid, _why in A_TABLE["replaced"].items():
    RECIPES[_rid] = dict(skip=True, why=_why)
for _r in A_TABLE["events"]:
    RECIPES[_r["id"]] = _measured_recipe(_r)
    if _r["kind"] == "event":
        EXTRA_EVENTS.append(dict(id=_r["id"], t=_r["hit_f"] / 24.0))
    else:
        _b = dict(id=_r["id"], t0=_r["f0"] / 24.0, t1=_r["f1"] / 24.0, fade_in=_r["fade_in"], fade_out=_r["fade_out"])
        if _r.get("env_f"):
            _b["env"] = [[(f - _r["f0"]) / 24.0, db] for f, db in _r["env_f"]]
        EXTRA_BEDS.append(_b)
