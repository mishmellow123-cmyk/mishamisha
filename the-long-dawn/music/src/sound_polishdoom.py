"""A 2400-3360: the fall rises into the white, the white is the impact, the valley's air arrives with the valley, the
black keeps a floor, and the ember arrives small and exact.

Fader, timing and layering choices only: no score note, pitch or harmony changes, and the cached score premaster is
never rewritten (polish_score rides an in-memory copy). sound_recipes_A installs these recipes; AP2 forwards them.
Every changed effect lies inside [2400, 3360): the updraft bed ends by 2652, the ash wind by 2800, the black's air by
2912, the ember by 2892; the score ride spans 2858-2922.
"""
import numpy as np

FPS = 24
SR = 48000
FR = SR // FPS
START, END = 2400, 3360
WHITE = 2640            # the impact: A.impact's hit, the full-white frame, A9's first frame
EMBER = 2832            # the ember's first light (finished frames: max luma 14 -> 51 on 2832)
PIANO = 2860            # the silence piano's first D (36 b4), its measured onset 2860.0
PIANO_2 = 2920          # its A (37 b3), onset 2920.1

# The master outside this lane's frames keeps the delivered AP2's envelope and guarded effects premaster through
# the ignition lane's bounded master: [START, END) lies inside polish_ignition_A.MASTER_WINDOWS (a music test
# fails if it does not, since bound_master would restore this lane's effects to the delivered reference).

# Timings apply after the cue's reference level is looked up (sound_v3.build), so a longer release does not move the
# design level. The updraft releases into the impact; the ash wind is at its bed level on the white.
BED_TIMING = {
    "A.storm.brink": dict(t1=2652 / FPS, fade_out=12 / FPS),
    "A.wind.ash": dict(t0=2634 / FPS, fade_in=6 / FPS, fade_out=8 / FPS),
}

# the small fire recorded on a very quiet night (sound_recipes_A.CRACKLE): one isolated pop, onset 72.53715 s
# (0.2 ms rise, down 30 dB in 10 ms, spectral centroid 3.6 kHz), and its calmest 2.5 s (from 48.0 s: 26 dB crest)
_CRACKLE = "fs:681366"


# The updraft's top over 2636-2637, dB over its bed level. The lane's render 2 used +10: the impact then rose only
# +10.75 dB in 50 ms, a storm transient at 2636.48 outranked the impact's spectral onset (578 vs 561), and A9's loudest
# 3 s went to -6.93 LU under the film's loudest (band edge -8.0, pass line -7.5; delivered -8.05). Owner integration:
# lowered, and measured on the owner render (see the integration commit).
UPDRAFT_TOP = 4.0
INHALE_TO = -5.0       # the inhale's floor into the impact (and the release under the aftermath): 9 dB below the top
# The ash wind on the white, dB under its bed level; it reaches the bed level by 2688, the frame the delivered bed did
# (from -20 dB, a hole under the impact's tail). The lane had it full on the white; with the updraft's +5 dB top that
# put A9's first measured window (2628-2700) at -7.3 LU, over the level map's -7.5 pass line (owner render doom2);
# with the top at +4, the inhale at -5, the ash from 2634 and -4.5 here: -7.511 LU, PASS (owner render doom6).
ASH_ON_WHITE = -4.5


def install(recipes, extra_beds, extra_events, wind):
    """Change the two doom beds; add the black's air and the ember's arrival."""
    # the updraft rises with the light: 0 dB at 2520, then 40% / 70% / all of UPDRAFT_TOP at 2600 / 2620 / 2636 (t from
    # the bed's 2400); the 125 ms before the impact is an inhale with a floor instead of the old gate (the updraft falls
    # to INHALE_TO over 2637-2639.5, while the score decays to -86 dBFS at 2638), the impact lands at 2640, and the
    # updraft releases under the aftermath at INHALE_TO (its own 12-frame fade to 2652). Owner render (doom6: +4 / -5,
    # ash -4.5 on the white): the impact rises +17.7 dB in 50 ms and is the strongest spectral onset of 2600-2660; the
    # lane's +10 dB top gave +10.8 dB and let a bed transient at 2636.5 outrank it.
    recipes["A.storm.brink"] = dict(
        recipes["A.storm.brink"], no_breath=True,
        env=[(0, 0), (120 / FPS, 0), (200 / FPS, UPDRAFT_TOP * .4), (220 / FPS, UPDRAFT_TOP * .7),
             (236 / FPS, UPDRAFT_TOP), (237 / FPS, UPDRAFT_TOP), (239.5 / FPS, INHALE_TO), (252 / FPS, INHALE_TO)],
    )
    # the ash wind is present on the white at ASH_ON_WHITE (t from its 2634 start), reaches its bed level by 2688, holds
    # while the valley is open, and follows the window closing (2762-2796) down to the black's air (t: 2640 = 0.25 s,
    # 2688 = 2.25 s, 2762 = 5.333 s, 2796 = 6.75 s, 2800 = 6.917 s); the shared gate at 2637-2640 does not cut it
    recipes["A.wind.ash"] = dict(
        recipes["A.wind.ash"], no_breath=True,
        env=[(0, -8), (6 / FPS, ASH_ON_WHITE), (54 / FPS, 0), (128 / FPS, 0),
             (162 / FPS, -24), (166 / FPS, -34)],
    )
    # the black's floor: the existing night-wind recording at -74 LUFS before the master (about -66 dBFS 5 ms RMS
    # after it), from under the ash wind's release through the black; it lifts 6 dB with the ember's light
    # (2832-2836) and hands over to the night bed, which still starts at 2848 (t from the bed's 2772)
    recipes["A.doom.black_air"] = dict(
        src=wind, seg=(4, 6), xf=1.0, hp=45, lp=1800, level=-74.0,
        no_breath=True, send=0.0,
        env=[(0, 0), (60 / FPS, 0), (64 / FPS, 6),
             (76 / FPS, 6), (108 / FPS, 0), (140 / FPS, -24)],
    )
    extra_beds.append(dict(id="A.doom.black_air", t0=2772 / FPS, t1=2912 / FPS,
                           fade_in=24 / FPS, fade_out=24 / FPS))
    # the ember's arrival: one small pop ON its first light (the hit is the pop's onset, at 2832), and under it the
    # same fire's quietest breath, rising with the picture's glow (2836-2850) and dying into the night wind by 2892.
    # Estimated offline on the clip alone, plus the approved master envelope's +10.8 dB at 2832: the pop's 5 ms RMS
    # -46 dBFS and the breath about -59 dBFS over 2846-2858 (the render measures the mix; see the lane report).
    recipes["A.doom.ember"] = dict(layers=[
        dict(src=(_CRACKLE, 72.53715), pre=0.004, post=0.30, hp=250, lp=7000, fi=0.002, fo=0.20, g=0.0, dt=0.0),
        dict(src=(_CRACKLE, 48.0), pre=0.0, post=2.5, hp=150, lp=3500, fi=0.75, fo=1.4, g=30.0, dt=0.0),
    ], level=-64.0, no_breath=True)
    extra_events.append(dict(id="A.doom.ember", t=EMBER / FPS))


def eased_gain(frames, points):
    """Cosine-interpolated amplitude; exact endpoint values and unity outside."""
    frames = np.asarray(frames, dtype=np.float64)
    out = np.ones(frames.shape, dtype=np.float32)
    for (a, ga), (b, gb) in zip(points, points[1:]):
        at = (frames >= a) & (frames < b)
        u = (frames[at] - a) / (b - a)
        out[at] = ga + (gb - ga) * (0.5 - 0.5 * np.cos(np.pi * u))
    return out


def score_gain(frames):
    """The silence piano's first D: its hammer enters 14 dB down and opens to -6 dB over 1.5 frames, rings at -6 dB,
    and the score is back at its own level under the next note's attack (the A at 2920), so no ringing tail swells.
    Before 2860 the score is silent (the black), so the ride's first shoulder changes nothing audible."""
    g14, g6 = 10 ** (-14 / 20), 10 ** (-6 / 20)
    return eased_gain(frames, [(PIANO - 2, 1), (PIANO - 1, g14), (PIANO, g14), (PIANO + 1.5, g6),
                               (PIANO_2, g6), (PIANO_2 + 1, 1)])


def polish_score(score, sr=SR):
    """In-place, bounded amplitude ride; never rewrite the saved premaster."""
    a, b = round((PIANO - 2) * sr / FPS), min(len(score), round((PIANO_2 + 1) * sr / FPS))
    if b > a:
        frames = np.arange(a, b, dtype=np.float64) * FPS / sr
        score[a:b] *= score_gain(frames)[:, None]
    return score
