"""Final re-mix: score stem + individually placed SFX events -> mastered mix.

Lets the edit nudge any sound effect to where the picture actually landed
without asking the composer to re-render. Event list format (from the music
department): [{"name", "file", "frame", "gain_db"}]; `OVERRIDES` below moves
or re-levels events by name.

    python3 edit/mix.py            # writes music/out/final_mix.wav
"""
import json
import os
import sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import Compressor, Limiter, Pedalboard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'music', 'out')
SR = 48000
FPS = 24
LENGTH = int(117.0 * SR)

# name -> dict(frame=..., gain_db=...) to override the composer's placement
OVERRIDES = {}
SCORE_GAIN_DB = 0.0
SFX_GAIN_DB = 0.0
TARGET_LUFS = -16.0


def load(path):
    x, sr = sf.read(path, always_2d=True, dtype='float32')
    assert sr == SR, (path, sr)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x


def place(bus, clip, start):
    if start < 0:
        clip, start = clip[-start:], 0
    end = min(len(bus), start + len(clip))
    if end > start:
        bus[start:end] += clip[:end - start]


def main():
    score = load(os.path.join(OUT, 'score.wav'))
    bus = np.zeros((LENGTH, 2), np.float32)
    place(bus, score * 10 ** (SCORE_GAIN_DB / 20), 0)
    ev_path = os.path.join(OUT, 'sfx_events.json')
    if os.path.exists(ev_path):
        events = json.load(open(ev_path))
        sfx = np.zeros_like(bus)
        for ev in events:
            o = OVERRIDES.get(ev['name'], {})
            frame = o.get('frame', ev['frame'])
            gain = o.get('gain_db', ev.get('gain_db', 0.0))
            f = ev['file'] if os.path.isabs(ev['file']) else os.path.join(OUT, ev['file'])
            if not os.path.exists(f):
                f = os.path.join(OUT, 'sfx_events', os.path.basename(ev['file']))
            place(sfx, load(f) * 10 ** (gain / 20), int(round(frame / FPS * SR)))
        bus += sfx * 10 ** (SFX_GAIN_DB / 20)
    elif os.path.exists(os.path.join(OUT, 'sfx.wav')):
        place(bus, load(os.path.join(OUT, 'sfx.wav')) * 10 ** (SFX_GAIN_DB / 20), 0)

    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(bus.astype(np.float64))
    bus *= 10 ** ((TARGET_LUFS - lufs) / 20)
    board = Pedalboard([Compressor(threshold_db=-14, ratio=1.6, attack_ms=30, release_ms=250),
                        Limiter(threshold_db=-1.5, release_ms=120)])
    master = board(bus.T, SR).T
    # final fades so the file starts/ends in true silence
    n = int(0.02 * SR)
    master[:n] *= np.linspace(0, 1, n)[:, None]
    master[-int(1.5 * SR):] *= np.linspace(1, 0, int(1.5 * SR))[:, None] ** 2
    out = os.path.join(OUT, 'final_mix.wav')
    sf.write(out, master, SR, subtype='PCM_24')
    print(f'in {lufs:.1f} LUFS -> out {meter.integrated_loudness(master.astype(np.float64)):.1f} LUFS,'
          f' peak {20 * np.log10(np.abs(master).max() + 1e-9):.2f} dBFS -> {out}')


if __name__ == '__main__':
    sys.exit(main())
