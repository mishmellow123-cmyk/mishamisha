"""THE LONG DAWN v3 - SOUND lane: C (THE LAST PAGES), real recordings for the cue sheet AND for COMPOSER-C's own
effects (score_v3_C.extra_effects: the pen, the burn-throughs, the drop, the cock, the cold metal's tick, the seethe,
the torches' dip, the roads, the fire that remains, the fire that rises), which are read, never edited.

C is the BOOK: the storyteller's hearth, heavy old pages, a quill, paper that burns through; then the ink world
(wind, fire, beacons), the council's torches and a hooded murmur that falls silent when she sets the Ring down, the
sea at the Havens, and the hearth again.  Levels are matched to COMPOSER-C's designs cue by cue.
"""
import sound_flint_v3 as FL

HEARTH = [("fs:681366", 1.0, 82.5, 1.0)]                    # a small wood fire, close, on a very quiet night
FIRE = [("fs:595483", 12.0, 90.0, 1.0)]                      # a wood fire, established
STORM = [("fs:754256", 2.0, 182.0, 1.0)]                     # a real snowstorm's gusts
WIND = [("fs:725630", 0.5, 150.0, 1.0)]                      # moderate mountain wind (Schoeps ORTF)
SEA = [("fs:525029", 2.0, 166.0, 1.0)]                       # 1 m waves on rocks under a 10 m cliff (ORTF)
WASH = [("fs:648860", 2.0, 88.0, 1.0)]                       # a quieter sea against rocks
LAVA = [("fs:172630", 1.0, 57.5, 1.0)]                       # molten rock seething (the melt, the blaze's weight)
PAPER = "fs:528662"                                          # a sheet of paper burning, 65 s
QUILL = "fs:194905"                                          # a quill on hard paper, various speeds
PAGES = "fs:388947"                                          # a very old book's pages turning (CC-BY 4.0)
FLIP = "fs:481077"                                           # a book handled, pages flipped (CC0)


def page(src, hit, pre=0.5, post=1.0, **kw):
    return dict(src=(src, hit), pre=pre, post=post, hp=120, fi=0.08, fo=0.3, **kw)


def pen(start, dur):
    return dict(src=(QUILL, start), pre=0.0, post=dur, hp=250, fi=0.05, fo=min(0.3, dur / 3), width=0.7)


def burn(start, dur, swell=0.5):
    """paper catching at a burn-through: the crackle swells in over `swell` s to the hit, then burns on"""
    return dict(src=(PAPER, start), pre=swell, post=dur, hp=200, fi=swell, fo=min(1.2, dur / 2),
                env=[(-swell, -10.0), (0.0, 0.0), (dur, -4.0)])


def flare(src_t, dist, stretch=1.0):
    """a far beacon catching: the same real whoomp as the roar, at distance"""
    return dict(layers=[dict(src=("fs:595483", src_t), pre=0.5, post=2.6, hp=90, fi=0.05, fo=1.2, stretch=stretch,
                             g=0.0, dt=0.0)], dist=dist)


RECIPES = {
    # ---------------------------------------------------------------- the storyteller's hearth, the book
    "C.hearth.open": dict(src=HEARTH, seg=(10, 18), xf=2.5, hp=80, lp=9000, width=0.7),
    "C.page_turn_0": page(PAGES, 3.179, pre=0.85, post=0.7),
    "C.riffle": page(FLIP, 35.812, pre=1.1, post=1.9),
    "C.page_turn": page(PAGES, 25.367, pre=0.2, post=0.5),
    "C+.pen.mountain": pen(14.0, 2.4),
    "C+.pen.T1": pen(9.15, 1.0),
    "C+.pen.deep": pen(17.3, 9.0),
    "C+.pen.T7": pen(28.75, 1.0),
    "C+.pen.T9": pen(31.55, 1.0),
    "C+.pen.T14": pen(40.6, 1.0),
    "C+.burn.letters": burn(8.0, 3.0, 0.6),
    "C+.burn.deep": burn(22.0, 2.2),
    "C+.burn.eye": burn(33.5, 2.2),
    "C+.burn.map": burn(45.0, 2.6),
    "C+.burn.remains": burn(52.0, 2.0),
    "C+.burn.title": burn(58.5, 2.6, 0.8),
    # ---------------------------------------------------------------- the fire born from the page, the forge
    "C.fire.born": dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=0.0),
                                dict(src=HEARTH, seg=(8, 14), xf=2.0, g=-3.0)], hp=70),
    "C.fire.forge": dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=0.0),
                                 dict(src=LAVA, seg=(8, 14), xf=2.0, g=-8.0)], hp=50),
    "C.storm.eye": dict(src=STORM, seg=(6, 10), xf=1.5, hp=60),
    "C+.drop": dict(src=("fs:197900", 0.40), pre=0.02, post=0.45, hp=300, fo=0.1),
    "C+.cock": dict(src=("fs:482119", 0.12), pre=0.1, post=2.2, hp=400, lp=3500, fo=0.5, dist=0.85, width=0.3),
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
    "C.beacon1": flare(9.90, 0.23), "C.beacon2": flare(26.40, 0.31, 0.97), "C.beacon3": flare(9.90, 0.39, 1.03),
    "C.beacon4": flare(20.00, 0.47, 0.96), "C.beacon5": flare(26.40, 0.55, 1.04), "C.beacon6": flare(9.90, 0.63, 0.98),
    "C.beacon7": flare(20.00, 0.71, 1.02),
    # ---------------------------------------------------------------- the council: torches, murmur, the unmaking
    "C.hearth.council": dict(layers=[dict(src=FIRE, seg=(10, 18), xf=2.5, g=0.0),
                                     dict(src=HEARTH, seg=(8, 14), xf=2.0, g=-2.0)], hp=90, width=1.0),
    "C+.fire.rises": dict(layers=[dict(src=FIRE, seg=(8, 12), xf=1.5, g=0.0),
                                  dict(src=LAVA, seg=(6, 10), xf=1.5, g=-6.0)], hp=45,
                          env=[(0.0, -12.0), (8.0, 0.0)]),
    "C+.seethe": dict(layers=[dict(src=("fs:172630", 20.0), pre=0.0, post=7.0, hp=120, fi=0.4, fo=0.3, g=0.0, dt=0.0,
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
    "C.page.plenty": page(PAGES, 4.791, pre=0.35, post=0.8),
    "C.sea": dict(layers=[dict(src=SEA, seg=(12, 20), xf=3.0, g=0.0),
                          dict(src=WASH, seg=(12, 20), xf=3.0, g=-8.0)], hp=40, lp=9000),
    "C.page.havens": page(PAGES, 8.499, pre=0.3, post=0.6),
    "C.page.blank": page(PAGES, 21.151, pre=0.3, post=0.6),
    "C.page.blank_2": page(FLIP, 38.940, pre=0.2, post=0.6),
}

EXTRA_BEDS = [
    # the council's murmur: hooded, low, no words; it falls away as she sets the Ring on the stone
    dict(id="C.x.murmur", t0={"at": "council"}, t1={"at": "ring_set"}, fade_in=3.0, fade_out=2.5),
]
EXTRA_EVENTS = [
    dict(id="C.x.catch", t={"at": "catch"}),                  # the flint take's catch (+196), on every cut
]
RECIPES["C.x.murmur"] = dict(src=[("fs:766658", 4.0, 150.0, 1.0)], seg=(12, 20), xf=3.0, hp=150, lp=2200,
                             width=0.8, level_from="C.hearth.council", trim=-6.0)
RECIPES["C.x.catch"] = dict(FL.CATCH, level_from="C.blow", trim=1.0)

SPACE = dict(distance="forest20", outdoor="forest20", event_send=0.08, bed_send=0.0, wet_hp=150, wet_lp=9000,
             stem_hp=25)


def extra_cues(bm):
    """COMPOSER-C's own effects (in memory, as score_v3_C.build adds them), level-matched to its designs"""
    import score_v3_C
    return score_v3_C.extra_effects(bm)
