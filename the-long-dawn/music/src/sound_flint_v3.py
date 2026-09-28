"""THE LONG DAWN v3 - SOUND lane: THE FLINT TAKE (H1), one set of recordings shared by all three cuts.

The picture has one master timing for every cut, counted from strike 1: strike 2 at +29 f, strike 3 at +58 (the
tinder takes), the long blow +84..+180, the catch at +196, the roar at +240 on a downbeat.  A and B enter on strike 1;
C lays the find between strikes 2 and 3 and keeps strike 3 to the roar exactly.  So the SAME real strikes, the same
breath, the same catch and the same roar play in A, B and C; each cut's cue level comes from its own cue sheet.

Sources (see music/SFX_CREDITS.md):
  strikes  fs:499027  "05_Flint and steel" (CC0), 48 kHz / 24-bit: a real flint struck on a steel. Each strike is a
                      ~50 ms glancing scrape ending in a sharp click with a short spark sizzle.  The three picked rise
                      in brightness (HF share 64% -> 77% -> 94%) as the tinder takes on the third.
  breath   fs:848421  "Inhale and blow (Soprar)" (CC-BY 4.0) + fs:273979 "inhale/exhale" (CC0) + fs:406648 (CC0):
                      unvoiced, breathy blows (no lip whistle: spectral peakiness 17-24 dB vs 30-40 dB for the
                      whistling takes, which were rejected), mouth pops removed by a 400 Hz high-pass.
  catch    fs:595483  "Fire being started and crackling" (CC0): the moment the kindling takes (7.4 s).
  roar     fs:595483  the same fire at 9.9 s: a real low 'whoomp' as the flames climb (no gas, no designed whoosh),
                      with fs:681366 (a campfire on a quiet night, CC0) for the crackle that stays.
"""

STRIKES = {1: 16.860, 2: 6.902, 3: 3.999}        # click times in fs:499027 (the steepest HF onset = the spark)


def strike(k):
    return dict(src=("fs:499027", STRIKES[k]), pre=0.09, post=0.55 if k == 3 else 0.45, hp=300,
                fo=0.2 if k == 3 else 0.15)


BLOW = dict(layers=[                                  # blow, a soft in-breath, blow, a last puff (+84..+180)
    dict(src=("fs:848421", 2.62), pre=0.04, post=1.55, hp=400, fo=0.3, stretch=1.2, g=0.0, dt=0.0),
    dict(src=("fs:273979", 1.30), pre=0.02, post=1.0, hp=400, fi=0.15, fo=0.3, g=-10.0, dt=1.95),
    dict(src=("fs:273979", 3.12), pre=0.04, post=1.3, hp=400, fo=0.35, stretch=1.2, g=0.0, dt=2.5),
    dict(src=("fs:406648", 2.25), pre=0.03, post=0.45, hp=400, fo=0.15, g=-4.0, dt=4.05),
    dict(src=("fs:595483", 7.3), pre=0.0, post=4.6, hp=700, fi=1.2, fo=0.3, g=-12.0, dt=0.0,
         env=[(0.0, -14.0), (2.5, -7.0), (4.6, 0.0)]),      # the tinder's glow crackling up under the breath
])

CATCH = dict(src=("fs:595483", 7.47), pre=0.67, post=0.56, hp=90, fi=0.4, fo=0.06)   # hit = the burst's arrival

ROAR = dict(layers=[
    dict(src=("fs:595483", 9.944), pre=0.684, post=5.16, hp=70, fi=0.06, fo=1.8, g=0.0, dt=0.0),  # arrival
    dict(src=("fs:681366", 40.0), pre=0.1, post=5.0, hp=70, fi=0.25, fo=1.8, g=-5.0, dt=0.05),
])


def take(prefix, blow_id=None):
    """the flint take's recipes under one cut's cue ids"""
    return {f"{prefix}.strike1": strike(1), f"{prefix}.strike2": strike(2), f"{prefix}.strike3": strike(3),
            blow_id or f"{prefix}.blow": BLOW, f"{prefix}.catch": CATCH, f"{prefix}.roar": ROAR}
