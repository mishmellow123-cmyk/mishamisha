"""Master: stem-linked glue compression + true-peak limiting, loudness to
-16 LUFS integrated, TP <= -1 dBTP, exact length 117.000 s, 24-bit TPDF.

The same gain envelopes are applied to score and sfx, so
score.wav + sfx.wav == mix.wav (up to dither)."""
import json
import os

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

import mix as MX
from dsl import SR, TOTAL_N

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
OUT = os.path.join(MUSIC, "out")

TARGET_LUFS = -16.0
CEIL_DB = -1.25
SFX_BALANCE_DB = 0.0     # sfx gains are already set per event


def lufs(x):
    meter = pyln.Meter(SR)
    return meter.integrated_loudness(x.astype(np.float64))


def finish(score, sfx, events):
    score = score[:TOTAL_N].astype(np.float32)
    sfx = (sfx[:TOTAL_N] * 10 ** (SFX_BALANCE_DB / 20)).astype(np.float32)
    # DC removal (both stems) - gentle 8 Hz high-pass
    score = MX.highpass(score, 8.0, order=1)
    sfx = MX.highpass(sfx, 8.0, order=1)
    pre = score + sfx
    G = 10 ** ((TARGET_LUFS - lufs(pre)) / 20)
    for it in range(4):
        x = pre * G
        comp = MX.glue_comp_gain(x, thresh_db=-16.0, ratio=1.5, attack=0.04, release=0.4)
        y = x * comp[:, None]
        lim = MX.tp_limiter_gain(y, ceiling_db=CEIL_DB, lookahead=0.006, release=0.15)
        mixd = y * lim[:, None]
        L = lufs(mixd)
        err = TARGET_LUFS - L
        print(f"  master iter {it}: {L:.2f} LUFS (gain {20 * np.log10(G):+.2f} dB)")
        if abs(err) < 0.1:
            break
        G *= 10 ** (err / 20)
    env = (G * comp * lim).astype(np.float32)[:, None]
    s_out = score * env
    x_out = sfx * env
    # final safety fades: from silence at 0, to silence at 117.000 s
    fi = int(0.003 * SR)
    fo = int(0.35 * SR)
    for y in (s_out, x_out):
        y[:fi] *= np.linspace(0, 1, fi)[:, None]
        y[-fo:] *= (np.cos(np.linspace(0, np.pi / 2, fo)) ** 2)[:, None]
    m_out = s_out + x_out
    tp = MX.true_peak(m_out)
    if 20 * np.log10(tp) > -1.05:
        k = 10 ** ((-1.1 - 20 * np.log10(tp)) / 20)
        s_out *= k
        x_out *= k
        m_out = s_out + x_out
    rng = np.random.default_rng(0)
    for name, y in (("score", s_out), ("sfx", x_out), ("mix", m_out)):
        y = MX.tpdf_dither_24(y, rng)
        assert len(y) == TOTAL_N
        sf.write(os.path.join(OUT, f"{name}.wav"), y, SR, subtype="PCM_24")
    # sfx event gains in the final mix domain (master gain, before limiting)
    gdb = 20 * np.log10(G)
    evp = os.path.join(OUT, "sfx_events.json")
    if os.path.exists(evp):
        ev = json.load(open(evp))
        for e in ev:
            e["gain_db"] = round(e["gain_db"] + SFX_BALANCE_DB + gdb, 2)
        json.dump(ev, open(evp, "w"), indent=1)
    print(f"  wrote score/sfx/mix: {lufs(m_out):.2f} LUFS, TP {20 * np.log10(MX.true_peak(m_out)):.2f} dBTP, "
          f"master gain {gdb:+.2f} dB, max comp GR {20 * np.log10(comp.min()):.1f} dB, "
          f"max lim GR {20 * np.log10(lim.min()):.1f} dB")
    np.save(os.path.join(MUSIC, "cache", "master_env.npy"), env[:, 0])
