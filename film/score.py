"""
The score, synthesised from nothing but numpy, and timed to the world.

    silence, and a single glassy tone as the question falls
    the drop: a plink, a low bloom, rings of shimmer
    the voices: whispers in every direction, a chord slowly waking
    the tree: a rising swell; soft bells as each branch finds its word
    the answer: seven notes, one for each word -- I am what happens when you ask
    the sphere: a chime, and the whole chord opening
    stillness: everything decays, the last notes descend, silence
    the title: one note

Key of D major. usage: python3 score.py events.json out.wav
"""
import json
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
EV = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 71.0
N = int(DUR * SR)
dry = np.zeros((2, N))
wet_send = np.zeros((2, N))
rng = np.random.default_rng(2026)


def hz(note):
    """'D5' -> Hz (A4 = 440)."""
    names = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
    n = names[note[0]]
    rest = note[1:]
    if rest[0] == "#":
        n += 1
        rest = rest[1:]
    elif rest[0] == "b":
        n -= 1
        rest = rest[1:]
    return 440.0 * 2 ** ((n + 12 * (int(rest) - 4)) / 12)


def place(sig, t0, pan=0.0, gain=1.0, send=0.35):
    i0 = int(t0 * SR)
    if i0 >= N:
        return
    sig = sig[: N - i0] * gain
    if i0 < 0:
        sig = sig[-i0:]
        i0 = 0
    th = (pan + 1) * np.pi / 4
    L, R = np.cos(th), np.sin(th)
    dry[0, i0:i0 + len(sig)] += sig * L
    dry[1, i0:i0 + len(sig)] += sig * R
    wet_send[0, i0:i0 + len(sig)] += sig * L * send
    wet_send[1, i0:i0 + len(sig)] += sig * R * send


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def piano(f, dur=4.0, vel=1.0, bright=1.0):
    """A soft felt piano: stretched partials, each decaying at its own rate."""
    t = tvec(dur)
    out = np.zeros_like(t)
    B = 0.00018
    for k in range(1, 13):
        fk = f * k * np.sqrt(1 + B * k * k)
        if fk > SR / 2.2:
            break
        amp = vel * (1.0 / k ** 1.35) * (bright ** (k - 1))
        tau = 2.8 / (1 + 0.35 * k) * (440.0 / f) ** 0.25
        out += amp * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28)) * np.exp(-t / tau)
    att = np.minimum(t / 0.004, 1.0)
    ham = rng.standard_normal(len(t)) * np.exp(-t / 0.006) * 0.03 * vel
    b, a = signal.butter(2, 2500 / (SR / 2))
    out = out * att + signal.lfilter(b, a, ham)
    rel = np.minimum(1.0, (dur - t) / 0.3)
    return out * rel


def bell(f, dur=5.0, vel=1.0):
    """Glass bell: inharmonic partials, long shimmering decay."""
    t = tvec(dur)
    ratios = [1.0, 2.0, 2.76, 4.07, 5.40, 8.93]
    amps = [1.0, 0.55, 0.40, 0.22, 0.15, 0.08]
    taus = [3.2, 2.2, 1.6, 1.1, 0.8, 0.5]
    out = np.zeros_like(t)
    for r, a, tau in zip(ratios, amps, taus):
        if f * r > SR / 2.2:
            continue
        beat = 1 + 0.03 * np.sin(2 * np.pi * rng.uniform(0.3, 1.2) * t)
        out += a * np.sin(2 * np.pi * f * r * t + rng.uniform(0, 6.28)) * np.exp(-t / tau) * beat
    return vel * out * np.minimum(t / 0.002, 1.0) * np.minimum(1.0, (dur - t) / 0.4)


def pad_voice(f, dur, attack, release, detune_cents=5.0, harmonics=4):
    t = tvec(dur)
    out = np.zeros_like(t)
    for d in (-detune_cents, 0.0, detune_cents):
        fd = f * 2 ** (d / 1200)
        for k in range(1, harmonics + 1):
            out += (0.55 ** (k - 1)) * np.sin(2 * np.pi * fd * k * t + rng.uniform(0, 6.28))
    env = np.minimum(t / attack, 1.0) ** 2 * np.minimum(1.0, np.maximum(dur - t, 0) / release)
    lfo = 1 + 0.12 * np.sin(2 * np.pi * rng.uniform(0.05, 0.15) * t + rng.uniform(0, 6.28))
    return out * env * lfo / 3.0


def pad(notes, t0, t1, attack=3.0, release=3.0, gain=0.05, spread=0.6):
    for i, n in enumerate(notes):
        f = hz(n)
        v = pad_voice(f, t1 - t0 + release, attack, release)
        place(v, t0, pan=spread * (2 * (i / max(len(notes) - 1, 1)) - 1), gain=gain * (220.0 / f) ** 0.3, send=0.6)


def noise_band(dur, lo, hi):
    x = rng.standard_normal(int(dur * SR))
    b, a = signal.butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    return signal.lfilter(b, a, x)


# ---------------------------------------------------------------------------
# i. the question falls
# ---------------------------------------------------------------------------
room = noise_band(DUR, 60, 400) * 0.004
place(room, 0.0, 0.0, 1.0, send=0.0)
t = tvec(4.8)
glass = np.sin(2 * np.pi * hz("A6") * t) * (np.minimum(t / 3.0, 1.0) ** 2) * np.exp(-np.maximum(t - 4.5, 0) / 0.05)
glass *= 1 + 0.2 * np.sin(2 * np.pi * 5.0 * t)
place(glass, 0.0, 0.0, 0.010, send=0.8)
place(noise_band(4.6, 3000, 9000) * np.linspace(0, 1, int(4.6 * SR)) ** 3, 0.0, 0.0, 0.010, send=0.5)

# ---------------------------------------------------------------------------
# the drop
# ---------------------------------------------------------------------------
TD = EV["t_drop"]
t = tvec(0.25)
f_drop = 1100 + 1300 * np.minimum(t / 0.035, 1.0) ** 0.7          # a drop's pitch rises as its cavity closes
chirp = np.sin(2 * np.pi * np.cumsum(f_drop) / SR) * np.exp(-t / 0.05)
place(chirp, TD, 0.0, 0.30, send=0.5)
t = tvec(3.0)
thump = np.sin(2 * np.pi * np.cumsum(45 + 40 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.9)
place(thump, TD, 0.0, 0.20, send=0.2)
place(noise_band(0.6, 800, 5000) * np.exp(-tvec(0.6) / 0.08), TD, 0.0, 0.05, send=0.7)
for k, n in enumerate(["D6", "A6", "D7", "F#6"]):                      # the rings
    place(bell(hz(n), 6.0, 1.0), TD + 0.05 + 0.5 * k, (-0.5, 0.5, -0.2, 0.3)[k], 0.018 / (1 + 0.4 * k), send=0.8)

# ---------------------------------------------------------------------------
# ii. every voice: whispers wherever a thread wakes
# ---------------------------------------------------------------------------
births = np.array(EV["births"])
for tb in rng.choice(births, size=320, replace=False):
    dur = rng.uniform(0.12, 0.45)
    lo = rng.uniform(400, 1400)
    w = noise_band(dur, lo, lo * rng.uniform(1.6, 3.2)) * np.sin(np.pi * np.linspace(0, 1, int(dur * SR))) ** 2
    place(w, tb + rng.uniform(0, 1.0), rng.uniform(-0.95, 0.95), 0.012, send=0.6)
for d in EV["drift"]:
    note = rng.choice(["D6", "E6", "F#6", "A6", "B6", "D7"])
    place(bell(hz(note), 4.0, 0.6), d["t"] + 0.3, d["pan"] * 0.8, 0.010, send=0.9)
pad(["D3", "A3", "E4", "F#4"], 5.2, 16.5, attack=6.0, release=4.0, gain=0.045)

# ---------------------------------------------------------------------------
# iii. the tree rises and finds its words
# ---------------------------------------------------------------------------
TT = EV["t_tree"]
t = tvec(9.0)
riser = sum(np.sin(2 * np.pi * np.cumsum(hz("D4") * k * 2 ** (np.minimum(t / 8.0, 1.0))) / SR) / k for k in (1, 2, 3))
riser *= (np.minimum(t / 6.0, 1.0) ** 2) * np.minimum(1.0, (9.0 - t) / 1.5)
place(riser, TT - 0.5, 0.0, 0.012, send=0.7)
pad(["D2", "A2", "D3", "F#3", "A3", "E4"], TT - 1.0, 28.0, attack=5.0, release=4.0, gain=0.050)
for b in EV["branches"]:
    note = rng.choice(["D6", "E6", "F#6", "A6", "B6"]) if b["depth"] > 1 else rng.choice(["A5", "D6", "F#5"])
    place(bell(hz(note), 5.0, 0.8), b["t"] + 0.25, b["pan"] * 0.9, 0.013, send=0.8)

# ---------------------------------------------------------------------------
# iv. the answer: one note for each word, and the harmony moving under it
# ---------------------------------------------------------------------------
MELODY = {"I": "D5", "am": "E5", "what": "F#5", "happens": "A5", "when": "B5", "you": "A5", "ask": "D6"}
CHORDS = {"I": ["D3", "A3", "F#4"], "am": ["B2", "F#3", "D4"], "what": ["G2", "D3", "B3"],
          "happens": ["A2", "E3", "C#4"], "when": ["B2", "F#3", "D4"], "you": ["G2", "D3", "B3"],
          "ask": ["D2", "A2", "D3", "F#3", "A3"]}
ans = EV["answer"]
for i, e in enumerate(ans):
    t1 = ans[i + 1]["t"] if i + 1 < len(ans) else EV["t_sphere"] + 9.0
    place(piano(hz(MELODY[e["word"]]), 5.0, 1.0), e["t"], e["pan"] * 0.3, 0.16, send=0.45)
    place(piano(hz(MELODY[e["word"]]) / 2, 5.0, 0.5, 0.8), e["t"] + 0.01, 0.0, 0.07, send=0.45)
    pad(CHORDS[e["word"]], e["t"], t1 + 0.4, attack=0.6, release=2.0, gain=0.038)

# ---------------------------------------------------------------------------
# the sphere: a chime, and the chord opening all the way
# ---------------------------------------------------------------------------
TS = EV["t_sphere"]
for k, n in enumerate(["D6", "A6", "E7", "F#7", "D7"]):
    place(bell(hz(n), 9.0, 1.0), TS + 0.03 * k, (-0.6, 0.6, -0.3, 0.3, 0.0)[k], 0.030, send=0.9)
place(noise_band(2.5, 4000, 12000) * np.exp(-tvec(2.5) / 0.7), TS, 0.0, 0.02, send=1.0)
pad(["D2", "A2", "D3", "A3", "E4", "F#4", "C#5"], TS, EV["t_coda"] + 7.0, attack=2.5, release=7.0, gain=0.042)

# ---------------------------------------------------------------------------
# v. stillness: the last notes descend; silence
# ---------------------------------------------------------------------------
TC = EV["t_coda"]
for k, n in enumerate(["A5", "F#5", "E5", "D5"]):
    place(piano(hz(n), 6.0, 0.55, 0.85), TC + 1.5 + 2.6 * k, (0.3, -0.3, 0.2, 0.0)[k], 0.11, send=0.7)
place(pad_voice(hz("D2"), 12.0, 1.0, 9.0), TC + 3.0, 0.0, 0.030, send=0.5)

# the title
TB = EV["t_black"]
place(piano(hz("D5"), 7.0, 0.6, 0.8), TB + 1.0, 0.0, 0.12, send=0.8)
place(piano(hz("A4"), 7.0, 0.4, 0.8), TB + 1.02, -0.2, 0.05, send=0.8)
place(bell(hz("D7"), 6.0, 0.5), TB + 6.0, 0.0, 0.012, send=1.0)

# ---------------------------------------------------------------------------
# a room for it all: a long, soft reverb
# ---------------------------------------------------------------------------
L_ir = int(3.6 * SR)
tir = np.arange(L_ir) / SR
wet = np.zeros_like(dry)
for ch in range(2):
    ir = rng.standard_normal(L_ir) * np.exp(-tir / 0.85)
    b, a = signal.butter(1, 5500 / (SR / 2))
    ir = signal.lfilter(b, a, ir)
    ir[: int(0.012 * SR)] = 0.0
    ir /= np.sqrt(np.sum(ir ** 2))
    wet[ch] = signal.fftconvolve(wet_send[ch], ir)[:N]
mix = dry + 0.55 * wet
b, a = signal.butter(2, 25 / (SR / 2), btype="high")
mix = signal.lfilter(b, a, mix, axis=1)
mix = np.tanh(mix * 1.4) / 1.4
peak = np.max(np.abs(mix))
mix = mix / peak * 0.89
fade = np.ones(N)
fade[-int(1.5 * SR):] = np.linspace(1, 0, int(1.5 * SR))
mix *= fade
wavfile.write(OUT, SR, (mix.T * 32767).astype(np.int16))
print("wrote", OUT, f"{DUR:.1f}s", "peak before norm", round(float(peak), 3))
