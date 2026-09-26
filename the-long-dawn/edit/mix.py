"""Final re-mix: v1 stems -> restored breaths + downbeat reinforcement -> mastered final_mix.wav.

    python edit/mix.py               # writes music/out/final_mix.wav
                                     # (renders music/out/reinforce_*.wav first if missing or stale)
    python edit/mix.py --rerender    # force a re-render of the reinforcement stems
    python edit/mix.py --baseline    # no breaths / reinforcement / push: reproduces mix.wav (equivalence check)
    python edit/mix.py --out X.wav   # write somewhere else (A/B renders)

Baseline.  The first version of this script rebuilt the SFX from the dry clips in
sfx_events/ and re-mastered through a pedalboard Compressor + Limiter.  That was NOT
equivalent to mix.wav: +4.7 LU louder (-11.4 LUFS), true peak +0.34 dBTP (hard clip at
0 dBFS), section balance squeezed by up to ~2 LU, and the SFX lost the outdoor space /
hall the music department gave them (the clips are dry).  So the mix is now built from
score.wav + sfx.wav (they sum to mix.wav, stem-linked) and re-mastered transparently:
one static gain to -16 LUFS + the department's own 4x-oversampled look-ahead true-peak
limiter (music/src/mix.py), 24-bit TPDF.  `--baseline` checks it reproduces mix.wav.

What changes (only around frames 640 and 2240):
  * BREATHS: the v1 stems do not contain the breaths described in music/NOTES.md and
    score.py BREATHS (the music plays straight through into both downbeats).  They are
    restored exactly as render.py specifies them: the score is ducked to -45 dB (30 ms
    cos^2 fade down, 5 ms ramp up ending on the downbeat) for 200 ms before 640 and
    275 ms before 2240, and only the reversed cymbal rushes in (re-rendered with the
    department's synth.revcym - the v1 score contains this exact cymbal, sample-aligned).
    SFX are not ducked (the hearth flare whooshes into 2240, as in the department's mix).
    Set RESTORE_BREATHS = [] to keep v1's run-in.  (The 1037-1040 "suck" before the IMPACT
    is missing from v1 too; it is left as delivered - add (1040, 0.15) to restore it.)
  * REINFORCE: extra hits rendered with the department's own instruments (VSCO sampler
    kits/timpani/brass, synthesized taiko, hybrid impact, sub, choir), seated/EQ'd/sent
    to the same synthesized hall as the score, tuned to D (race: D pedal; climax: D major),
    onset- and phase-aligned to the existing downbeat (shift/polarity maximising the <160 Hz
    sum), each stem through its own true-peak bus limiter so its transients sit under the
    score's peaks.  The local VSCO sparse clone has no F horn / tenor trombone samples, so the
    climax's low-brass sforzando is voiced on the tuba patch.  Cached as
    music/out/reinforce_<name>.wav (+ reinforce_events.json with a spec hash).
  * PUSH: a short gain lift on the score just after each downbeat, split at 150 Hz (more lift
    above than below, so it adds loudness and brightness rather than limiter-feeding low end).

SFX nudges: OVERRIDES = {name: dict(frame=..., gain_db=...)} moves/re-levels an event by
subtracting its clip at the delivered place and adding it at the new one (the event's
outdoor/hall tail stays behind - a small approximation, fine for nudges of a few frames).
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'music', 'out')
SRC = os.path.join(ROOT, 'music', 'src')
SR = 48000
FPS = 24
LENGTH = int(117.0 * SR)

# name -> dict(frame=..., gain_db=...) to override the composer's placement
OVERRIDES = {   # ignitions the music was masking (1600 ice fully, 1730 sea marginal; the sea flare is now a hero light)
    'montage_ignition_3': dict(gain_db=-7.2),    # was -15.2 (+8: parity with the audible ignitions)
    'montage_ignition_6': dict(gain_db=-9.2),    # was -15.2 (+6)
}
SCORE_GAIN_DB = 0.0
SFX_GAIN_DB = 0.0
TARGET_LUFS = -16.0
TP_CEILING_DB = -1.2          # limiter ceiling (spec: true peak <= -1.0 dBTP)

# (downbeat frame, breath length in beats) - score.py BREATHS entries for the two arrivals
RESTORE_BREATHS = [(640, 0.24), (1040, 0.15), (2240, 0.33)]   # + the composer's 1037->1040 'suck' before the IMPACT
BREATH_FLOOR_DB = -45.0

D1, D2 = 36.708, 73.416       # Hz

# Reinforcement: per downbeat, a list of layers (kind, seat, gain_db, params).
# kind: 'kit' (VSCO drum kit patch), 'note' (VSCO pitched patch; params pitch/dur/vel/dyn),
#       'taiko' (synth._drum_hit ensemble), 'impact' (synth.impact recipe tuned to f_end),
#       'sub' (held sub sine, synth.subdrop recipe without the drop), 'choir' (synth.choir).
# seat: score.py seating + fader of the matching part (see SEAT).
REINFORCE = {
    'race': dict(frame=640, tail_s=4.5, peak_db=-4.0, layers=[
        ('taiko', 'taiko', -5.0, dict(f=D1, players=3, vel=1.0, decay=0.5, skin=0.7)),
        ('taiko', 'taiko', -5.0, dict(f=D2, players=3, vel=1.0, decay=0.3, skin=1.4, bp=(250, 2800), drop=1.6)),
        ('kit', 'giant', +0.0, dict(patch='giant_mallet', vel=1.0)),
        ('kit', 'bdrum', +0.0, dict(patch='bdrum', vel=1.0)),
        ('note', 'timp', -3.0, dict(patch='timp', pitch='D2', dur=2, vel=1.0)),
        ('impact', 'impact', -3.0, dict(f=D1, size=0.6, crack=5.0, boom=0.4)),
        ('sub', 'subdrop', -14.0, dict(f=D1, dur=1.0)),
        ('kit', 'tenor', +5.0, dict(patch='tenor_lo', vel=1.0)),
        ('kit', 'snare', +6.0, dict(patch='snare', vel=1.0)),
        ('kit', 'cym', +8.0, dict(patch='clash', vel=1.0)),
        ('kit', 'crash', +2.0, dict(patch='crash', vel=1.0)),
    ]),
    'dawn': dict(frame=2240, tail_s=7.0, peak_db=-4.0, layers=[
        ('kit', 'cym', +4.0, dict(patch='clash', vel=1.0, rr=0, pan=-0.45)),
        ('kit', 'cym', +4.0, dict(patch='clash', vel=1.0, rr=1, pan=0.45)),
        ('kit', 'crash', +1.0, dict(patch='crash', vel=1.0)),
        ('kit', 'gong', +2.0, dict(patch='gong', vel=1.0)),
        ('note', 'timp', +0.0, dict(patch='timp', pitch='D2', dur=3, vel=1.0)),
        ('note', 'timp', -3.0, dict(patch='timp', pitch='A2', dur=3, vel=0.85, delay=0.02)),
        ('kit', 'bdrum', -2.0, dict(patch='bdrum', vel=1.0)),
        ('kit', 'giant', -4.0, dict(patch='giant_mallet', vel=1.0)),
        ('impact', 'impact', -3.0, dict(f=D1, size=1.2, crack=1.0, boom=0.4)),
        ('sub', 'subdrop', -15.0, dict(f=D1, dur=3.0)),
        # low brass: a D major sforzando that melts into the score's own chords.  (The local
        # VSCO sparse clone has no F horn / tenor trombone samples, so it is voiced on the tuba
        # patch and spread over the rear brass seats.)
        ('note', 'tuba', -2.0, dict(patch='tuba', pitch='D2', dur=2.5, vel=1.0, dyn='sfz')),
        ('note', 'tuba', -2.0, dict(patch='tuba', pitch='A2', dur=2.5, vel=1.0, dyn='sfz')),
        ('note', 'tbn', +2.0, dict(patch='tuba', pitch='D3', dur=2.5, vel=1.0, dyn='sfz')),
        ('note', 'tbn', +2.0, dict(patch='tuba', pitch='F#3', dur=2.5, vel=1.0, dyn='sfz')),
        ('note', 'hns', +2.0, dict(patch='tuba', pitch='A3', dur=2.5, vel=1.0, dyn='sfz')),
        # full choir "AAH" accent (D major, wide)
        ('choir', 'choir_w', +11.0, dict(pitches=['D3', 'A3', 'D4', 'F#4', 'A4', 'D5', 'F#5', 'A5'], dur=2.5)),
    ]),
}
# score gain lift right after each downbeat, split at 150 Hz (more lift above, less below so
# the low end does not feed the limiter): frame -> (dB above, dB below, hold s, release s)
PUSH = {640: (1.5, 0.0, 0.4, 0.8), 2240: (4.5, 1.0, 1.2, 2.5)}
PUSH_SPLIT_HZ = 150.0

SFZ_DYN = [(0.0, 1.0), (0.35, 0.78), (1.2, 0.5), (2.5, 0.22)]   # brass sforzando (beats)

# score.py setup() seating (pan, width, depth, send) + FADERS gain for the matching part
SEAT = dict(taiko=(0.0, 1.0, 0.55, 0.30, -13.1), giant=(-0.1, 0.9, 0.6, 0.35, 7.7),
            bdrum=(0.1, 0.8, 0.7, 0.4, 7.5), timp=(-0.05, 0.6, 0.7, 0.42, 6.7),
            cym=(0.25, 1.0, 0.6, 0.45, 6.8), crash=(-0.25, 1.0, 0.6, 0.45, 9.9),
            gong=(0.0, 1.0, 0.75, 0.5, 4.3), impact=(0.0, 1.0, 0.3, 0.35, -2.5),
            subdrop=(0.0, 0.0, 0.0, 0.0, -6.1), tbn=(0.42, 0.6, 0.65, 0.42, -2.8),
            tuba=(0.55, 0.4, 0.65, 0.4, -0.4), hns=(-0.35, 0.7, 0.6, 0.45, -0.8),
            choir_w=(0.0, 1.0, 0.6, 0.45, -23.5), revcym=(0.0, 1.0, 0.4, 0.3, 8.2),
            tenor=(0.2, 0.8, 0.6, 0.3, 0.5), snare=(0.15, 0.6, 0.65, 0.35, 4.5))
V1_MASTER_GAIN_DB = -2.25     # the v1 master's static gain (sfx_events.json gain_db - clip gain)


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------
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


def lufs(x):
    return pyln.Meter(SR).integrated_loudness(np.asarray(x, np.float64))


def tp_peaks(x, chunk=10 * SR, pad=128):
    """Per-sample max |x| over 4x oversampling (chunked, low memory)."""
    n = len(x)
    out = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        a0, b0 = max(0, a - pad), min(n, b + pad)
        y = signal.resample_poly(x[a0:b0].astype(np.float64), 4, 1, axis=0)
        pk = np.abs(y).max(1).reshape(-1, 4).max(1)
        out[a:b] = pk[a - a0:a - a0 + (b - a)]
    return out


def true_peak_db(x):
    return 20 * np.log10(float(tp_peaks(x).max()) + 1e-12)


# ---------------------------------------------------------------------------
# the music department's code (instruments, hall, limiter)
# ---------------------------------------------------------------------------
_DEPT = {}


def dept():
    if not _DEPT:
        if SRC not in sys.path:
            sys.path.insert(0, SRC)
        import dsl
        import sampler
        import synth
        spec = importlib.util.spec_from_file_location('dept_mix', os.path.join(SRC, 'mix.py'))
        dmx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dmx)
        # per-sample fine tuning (check_samples.py -> cache/pitch_fix.json); rebuild it from the
        # audit if the cache is absent so sustained brass is tuned exactly like the score's
        if not os.path.exists(os.path.join(SRC, '..', 'cache', 'pitch_fix.json')):
            sampler._FIX = pitch_fix_from_audit()
        _DEPT.update(dsl=dsl, sampler=sampler, synth=synth, dmx=dmx)
    return _DEPT


def pitch_fix_from_audit():
    fixes = {}
    rx = re.compile(r'^(\S+)\s+(.+?)\s+key\s+\d+\s+exp\s+[\d.]+ Hz\s+meas\s+[\d.]+ Hz\s+([+-]?[\d.]+) c')
    with open(os.path.join(ROOT, 'music', 'analysis', 'sample_pitch.txt')) as fh:
        for ln in fh:
            mm = rx.match(ln)
            if not mm:
                continue
            pk, rel, cents = mm.group(1), mm.group(2).strip(), float(mm.group(3))
            octe = int(np.round(cents / 1200))
            fine = cents - 1200 * octe
            if octe == 0 and 12 < abs(fine) < 60 and pk not in ('timp', 'timp_roll', 'tubular', 'glock'):
                fixes[rel] = -round(float(fine), 1)
    return fixes


def seat_mix(items, pre, n):
    """items: (stereo y, offset rel. to the downbeat in samples, seat, gain_db, pan or None).  Same
    chain as render.mix_score: fader -> pan/width -> distance EQ -> dry*(1-0.35*depth) + send -> hall."""
    d = dept()
    dmx = d['dmx']
    dry = np.zeros((n, 2), np.float32)
    send = np.zeros((n, 2), np.float32)
    for y, off, seat, gdb, pan_o in items:
        pan, width, depth, snd, fader = SEAT[seat]
        pan = pan if pan_o is None else pan_o
        y = dmx.pan_width(np.asarray(y, np.float32) * np.float32(10 ** ((fader + gdb) / 20)), pan, width)
        y = dmx.depth_eq(y, depth)
        place(dry, y * np.float32(1.0 - 0.35 * depth), pre + int(off))
        place(send, y * np.float32(snd), pre + int(off))
    wet = dmx.convolve_stereo(send, hall())
    out = dmx.highpass(dry + wet, 22.0)
    return out * np.float32(10 ** (V1_MASTER_GAIN_DB / 20))


def hall():
    if 'hall' not in _DEPT:
        _DEPT['hall'] = dept()['dmx'].make_ir(rt_mid=3.1)    # render.py: MX.get_ir("hall", rt_mid=3.1)
    return _DEPT['hall']


# ---------------------------------------------------------------------------
# reinforcement rendering
# ---------------------------------------------------------------------------
def render_layer(kind, prm, rng):
    d = dept()
    dsl, sampler, synth = d['dsl'], d['sampler'], d['synth']
    if kind in ('kit', 'note'):
        pitch = dsl.m(prm.get('pitch', 60))
        dyn = SFZ_DYN if prm.get('dyn') == 'sfz' else []
        rr = sampler.RR()
        for c, regs in sampler.pick(prm['patch'], pitch):      # choose the round-robin sample
            for _ in range(prm.get('rr', 0)):
                rr.next((prm['patch'], regs[0]['lokey'], c), max(r['seq_length'] for r in regs))
        y, off = sampler.render_note(prm['patch'], pitch, 0.0, prm.get('dur', 4), prm.get('vel', 1.0), dyn, rr,
                                     rng, kw=dict(vary=False))
        return y, off
    if kind == 'taiko':
        dec = prm.get('decay', 0.55)
        out = np.zeros((int((dec * 1.06 * 5 + 0.05) * SR), 2), np.float32)
        for pl in range(prm.get('players', 3)):
            fe = prm['f'] * 2 ** (rng.normal(0, 0.1) / 12)
            y = synth._drum_hit(rng, fe, dec * rng.uniform(0.95, 1.05),
                                prm.get('vel', 1.0) * rng.uniform(0.93, 1.0), skin=prm.get('skin', 0.35),
                                pitch_drop=prm.get('drop', 1.8), noise_bp=prm.get('bp', (150, 1500)))
            st = synth._pan(y, float(rng.uniform(-0.5, 0.5))).astype(np.float32) * 0.5
            dt = 0 if pl == 0 else int(abs(rng.normal(0, 0.003)) * SR)
            k = min(len(st), len(out) - dt)
            out[dt:dt + k] += st[:k]
        return out, 0
    if kind == 'impact':
        # synth.impact (sub thump + boom + crack), the thump settling on f instead of 38 Hz
        n = int(6.0 * SR)
        tt = np.arange(n) / SR
        size = prm.get('size', 1.0)
        f = prm['f'] * (1 + (55 / 38) * np.exp(-tt / 0.06))
        sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / (1.1 * size))
        sub = np.tanh(sub * 1.6) / np.tanh(1.6)
        bm = rng.normal(0, 1, n) * np.exp(-tt / (0.18 * size))
        b, a = signal.butter(2, 180 / (SR / 2))
        bm = signal.lfilter(b, a, bm) * 6
        ck = rng.normal(0, 1, n) * np.exp(-tt / 0.012)
        b, a = signal.butter(2, [1500 / (SR / 2), 9000 / (SR / 2)], btype='band')
        ck = signal.lfilter(b, a, ck) * 0.6
        y = sub * 0.9 + bm * prm.get('boom', 0.5) + ck * prm.get('crack', 1.0)
        a0 = int(0.001 * SR)
        y[:a0] *= np.linspace(0, 1, a0)
        side = rng.normal(0, 1, n) * np.exp(-tt / 0.3)
        b, a = signal.butter(2, [300 / (SR / 2), 3000 / (SR / 2)], btype='band')
        side = signal.lfilter(b, a, side) * 0.15
        return (np.stack([y + side, y - side], 1) * prm.get('vel', 1.0) * 0.5).astype(np.float32), 0
    if kind == 'sub':
        # synth.subdrop recipe held on one pitch: sine + soft saturation, fast attack, exp decay
        n = int(prm['dur'] * SR)
        tt = np.arange(n) / SR
        y = np.tanh(1.3 * np.sin(2 * np.pi * prm['f'] * tt)) / np.tanh(1.3)
        env = (1 - np.exp(-tt / 0.004)) * np.exp(-tt / (prm['dur'] / 2.2))
        k = int(0.05 * SR)
        env[-k:] *= np.linspace(1, 0, k)
        y = (y * env * prm.get('vel', 1.0) * 0.6).astype(np.float32)
        return np.stack([y, y], 1), 0
    if kind == 'choir':
        notes = [dict(pitch=dsl.m(p), start=0.0, dur=prm['dur'], vel=None, art=None, legato=False, sync=True,
                      gain_db=0.0, pan=None, kw=dict(atk=0.09, rel=0.9)) for p in prm['pitches']]
        part = dict(name='choir_rf', inst='choir', kind='synth', seed=2240, pan=0.0, humanize_ms=0,
                    params=dict(vowel='a', voices=8, spread=1.0, breath=0.04),
                    dyn=[(0.0, 1.0), (0.4, 0.82), (1.5, 0.5), (2.5, 0.25)], notes=notes)
        # choir notes speak atk*0.35 early: render with 1 beat of lead-in, then shift back
        lead = 40000
        for nt in notes:
            nt['start'] = 1.0
        part['dyn'] = [(b + 1.0, v) for b, v in part['dyn']]
        y = synth.render_part(part, int((prm['dur'] * 0.8333 + 3.0) * SR) + lead)
        return y, -lead
    raise ValueError(kind)


def spec_hash(name):
    src = ''.join(inspect.getsource(f) for f in (render_layer, seat_mix, bus_limiter, render_reinforcement)) + json.dumps(
        [REINFORCE[name], SEAT, SFZ_DYN, V1_MASTER_GAIN_DB], sort_keys=True, default=str)
    return hashlib.sha1(src.encode()).hexdigest()[:16]


def render_reinforcement(name, force=False):
    """Render one reinforcement stem -> music/out/reinforce_<name>.wav (cached by spec hash).
    Returns (clip, index of the downbeat inside the clip)."""
    spec = REINFORCE[name]
    path = os.path.join(OUT, f'reinforce_{name}.wav')
    meta_path = os.path.join(OUT, 'reinforce_events.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    h = spec_hash(name)
    if not force and os.path.exists(path) and meta.get(name, {}).get('hash') == h:
        return load(path), meta[name]['downbeat_index']
    print(f'  rendering reinforcement "{name}" ({len(spec["layers"])} layers) ...', flush=True)
    rng = np.random.default_rng(zlib_crc(name))
    pre = int(0.25 * SR)
    n = pre + int((spec['tail_s'] + 3.5) * SR)
    items = []
    for kind, seat, gdb, prm in spec['layers']:
        y, off = render_layer(kind, prm, rng)
        off += int(round(prm.get('delay', 0.0) * SR))
        items.append((y, off, seat, gdb, prm.get('pan')))
    clip = seat_mix(items, pre, n)
    # bus limiter on the reinforcement alone (4x true-peak, 6 ms look-ahead, 80 ms release) so its
    # transients sit under the score's peaks instead of making the master limiter pump
    clip = clip * bus_limiter(clip, spec.get('peak_db', -4.0), 0.08)[:, None]
    # sensible tails: everything of the reinforcement is gone by tail_s (+ hall)
    t = (np.arange(n) - pre) / SR
    f0, f1 = spec['tail_s'] - 2.5, spec['tail_s']
    fade = np.clip((f1 - t) / (f1 - f0), 0, 1)
    fade = np.where(t < f0, 1.0, np.sin(fade * np.pi / 2) ** 2)
    clip = (clip * fade[:, None]).astype(np.float32)
    last = np.where(np.abs(clip).max(1) > 1e-6)[0]
    clip = clip[:last[-1] + 1] if len(last) else clip
    sf.write(path, clip, SR, subtype='FLOAT')
    meta[name] = dict(file=os.path.relpath(path, OUT), frame=spec['frame'], downbeat_index=pre, hash=h)
    json.dump(meta, open(meta_path, 'w'), indent=1)
    return clip, pre


def bus_limiter(x, ceiling_db, release):
    dmx = dept()['dmx']
    la = int(0.006 * SR)
    need = np.minimum(1.0, 10 ** (ceiling_db / 20) / np.maximum(tp_peaks(x), 1e-9))
    gdb = 20 * np.log10(minimum_filter1d(need, size=2 * la + 1))
    return (10 ** (np.minimum(dmx._smooth_gain(gdb, 0.0005, release), gdb) / 20)).astype(np.float32)


def zlib_crc(s):
    import zlib
    return zlib.crc32(s.encode()) % 100000


# ---------------------------------------------------------------------------
# breaths (render.breath_env, applied to the score stem) and the reversed cymbal
# ---------------------------------------------------------------------------
def breath_env(n, windows):
    D = np.ones(n, np.float32)
    floor = 10 ** (BREATH_FLOOR_DB / 20)
    for frame, beats in windows:
        i1 = int(round(frame / FPS * SR))
        i0 = i1 - int(round(beats * 40000))
        fd, fu = int(0.03 * SR), int(0.005 * SR)
        seg = np.full(i1 - i0, floor, np.float32)
        seg[:fd] = floor + (1 - floor) * np.cos(np.linspace(0, np.pi / 2, fd)) ** 2
        seg[-fu:] = floor + (1 - floor) * np.sin(np.linspace(0, np.pi / 2, fu)) ** 2
        D[i0:i1] = np.minimum(D[i0:i1], seg)
    return D


def revcym_dry(score):
    """The score's reversed cymbal (synth.revcym, same notes/seed as score.py), dry path as
    render.mix_score seats it, gain-matched to the v1 score (least squares on >3 kHz)."""
    d = dept()
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import score as sc
    parts = sc.build()
    y = d['synth'].render_part(parts['revcym'].to_dict(), LENGTH + SR)[:LENGTH]
    pan, width, depth, snd, fader = SEAT['revcym']
    z = d['dmx'].pan_width(y * np.float32(10 ** (fader / 20)), pan, width)
    z = d['dmx'].depth_eq(z, depth) * np.float32(1 - 0.35 * depth)
    z = d['dmx'].highpass(z, 22.0) * np.float32(10 ** (V1_MASTER_GAIN_DB / 20))
    sos = signal.butter(4, 3000 / (SR / 2), 'high', output='sos')
    gains = {}
    for frame, beats in RESTORE_BREATHS:
        T = frame / FPS
        a, b = int((T - 0.8) * SR), int((T - 0.004) * SR)
        s = signal.sosfiltfilt(sos, score[a:b].mean(1).astype(np.float64))
        r = signal.sosfiltfilt(sos, z[a:b].mean(1).astype(np.float64))
        g = float(np.clip((s * r).sum() / ((r * r).sum() + 1e-20), 10 ** (-6 / 20), 10 ** (1 / 20)))
        i0 = int(round(frame / FPS * SR)) - int(round(beats * 40000)) - int(0.1 * SR)
        i1 = int(round(frame / FPS * SR)) + int(0.01 * SR)
        gains[frame] = (i0, i1, g)
    return z, gains


def push_curve(n, frame, db, hold, rel):
    g = np.zeros(n, np.float32)       # dB
    i = int(round(frame / FPS * SR)) - int(0.005 * SR)      # starts inside the breath's ramp-up
    h, r = int(hold * SR) + int(0.005 * SR), int(rel * SR)
    seg = np.concatenate([np.full(h, db), db * np.cos(np.linspace(0, np.pi / 2, r)) ** 2])
    k = max(0, min(len(seg), n - i))
    g[i:i + k] = seg[:k]
    return g


def apply_push(score):
    """Band-split gain lift after each downbeat: zero-phase 150 Hz split (low = filtfilt
    Butterworth, high = x - low, so the bands sum back exactly), separate gain curves."""
    sos = signal.butter(2, PUSH_SPLIT_HZ / (SR / 2), 'low', output='sos')
    for frame, (db_hi, db_lo, hold, rel) in PUSH.items():
        T = int(round(frame / FPS * SR))
        a, b = T - int(0.5 * SR), min(len(score), T + int((hold + rel + 1.0) * SR))
        seg = score[a:b].astype(np.float64)
        lo = signal.sosfiltfilt(sos, seg, axis=0)
        hi = seg - lo
        n = b - a
        g_hi = 10 ** (push_curve(n, frame - a / SR * FPS, db_hi, hold, rel) / 20)
        g_lo = 10 ** (push_curve(n, frame - a / SR * FPS, db_lo, hold, rel) / 20)
        score[a:b] = (hi * g_hi[:, None] + lo * g_lo[:, None]).astype(np.float32)
    return score


def align(bus, clip, pre, T_idx, search_ms=3.0):
    """Onset/phase alignment: shift (+-search_ms) and polarity maximising the low-band (<160 Hz)
    energy of bus + clip over the first 120 ms after the downbeat (constructive summation)."""
    sos = signal.butter(4, 160 / (SR / 2), 'low', output='sos')
    w0, w1 = T_idx - int(0.01 * SR), T_idx + int(0.12 * SR)
    m = int(search_ms / 1000 * SR)
    B = signal.sosfiltfilt(sos, bus[w0 - m:w1 + m].mean(1).astype(np.float64))
    C = signal.sosfiltfilt(sos, clip[pre - (T_idx - w0) - m:pre + (w1 - T_idx) + m].mean(1).astype(np.float64))
    best = (-1, 0, 1)
    for s in range(-m, m + 1, 6):
        c = C[m - s:m - s + (w1 - w0)] if 0 <= m - s and m - s + (w1 - w0) <= len(C) else None
        if c is None:
            continue
        b = B[m:m + (w1 - w0)]
        for pol in (1, -1):
            e = float(((b + pol * c) ** 2).sum())
            if e > best[0]:
                best = (e, s, pol)
    return best[1], best[2]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT, 'final_mix.wav'))
    ap.add_argument('--rerender', action='store_true')
    ap.add_argument('--baseline', action='store_true', help='v1 stems only (no breaths/reinforcement/push)')
    ap.add_argument('--no-breaths', action='store_true')
    ap.add_argument('--no-reinforce', action='store_true')
    ap.add_argument('--no-push', action='store_true')
    args = ap.parse_args()
    if args.baseline:
        args.no_breaths = args.no_reinforce = args.no_push = True

    score = load(os.path.join(OUT, 'score.wav'))[:LENGTH] * np.float32(10 ** (SCORE_GAIN_DB / 20))
    sfx = load(os.path.join(OUT, 'sfx.wav'))[:LENGTH]
    # SFX nudges as deltas on the delivered stem
    if OVERRIDES:
        events = {e['name']: e for e in json.load(open(os.path.join(OUT, 'sfx_events.json')))}
        for name, o in OVERRIDES.items():
            ev = events[name]
            clip = load(os.path.join(OUT, 'sfx_events', os.path.basename(ev['file'])))
            place(sfx, -clip * np.float32(10 ** (ev['gain_db'] / 20)), int(round(ev['frame'] / FPS * SR)))
            place(sfx, clip * np.float32(10 ** (o.get('gain_db', ev['gain_db']) / 20)),
                  int(round(o.get('frame', ev['frame']) / FPS * SR)))
    sfx *= np.float32(10 ** (SFX_GAIN_DB / 20))

    if RESTORE_BREATHS and not args.no_breaths:
        D = breath_env(LENGTH, RESTORE_BREATHS)
        rc, gains = revcym_dry(score)
        score *= D[:, None]
        for frame, (i0, i1, g) in gains.items():
            score[i0:i1] += rc[i0:i1] * (np.float32(g) * (1 - D[i0:i1]))[:, None]
            print(f'  breath restored before {frame} (revcym matched at {20 * np.log10(g):+.2f} dB)')
        del rc
    if PUSH and not args.no_push:
        score = apply_push(score)

    bus = score + sfx
    if not args.no_reinforce:
        D = breath_env(LENGTH, RESTORE_BREATHS) if (RESTORE_BREATHS and not args.no_breaths) else None
        for name, spec in REINFORCE.items():
            clip, pre = render_reinforcement(name, force=args.rerender)
            T_idx = int(round(spec['frame'] / FPS * SR))
            s, pol = align(bus, clip, pre, T_idx)
            start = T_idx - pre + s
            clip = clip * np.float32(pol)
            if D is not None:              # the reinforcement's pre-roll also breathes
                clip = clip * D[start:start + len(clip)][:, None] if start + len(clip) <= LENGTH else clip
            place(bus, clip, start)
            print(f'  reinforcement "{name}" at frame {spec["frame"]}: shift {s / SR * 1000:+.2f} ms, '
                  f'polarity {"+" if pol > 0 else "-"}')
    del score, sfx

    # master: static gain to target, true-peak look-ahead limiter (music/src/mix.py recipe), TPDF
    dmx = dept()['dmx']
    G = 10 ** ((TARGET_LUFS - lufs(bus)) / 20)
    ceil = 10 ** (TP_CEILING_DB / 20)
    la = int(0.006 * SR)
    for it in range(3):
        y = bus * np.float32(G)
        need = np.minimum(1.0, ceil / np.maximum(tp_peaks(y), 1e-9))
        need = minimum_filter1d(need, size=2 * la + 1)
        gdb = 20 * np.log10(need)
        g = np.minimum(dmx._smooth_gain(gdb, 0.0005, 0.15), gdb)
        lim = (10 ** (g / 20)).astype(np.float32)
        y *= lim[:, None]
        L = lufs(y)
        if abs(TARGET_LUFS - L) < 0.05:
            break
        G *= 10 ** ((TARGET_LUFS - L) / 20)
    for name, spec in REINFORCE.items():
        i = int(round(spec['frame'] / FPS * SR))
        print(f'  limiter GR at {spec["frame"]}: max {20 * np.log10(float(lim[i - 2400:i + 2 * SR].min())):.2f} dB')
    tp = true_peak_db(y)
    if tp > TP_CEILING_DB + 0.1:
        y *= np.float32(10 ** ((TP_CEILING_DB - tp) / 20))
    y = dmx.tpdf_dither_24(y, np.random.default_rng(0))
    assert len(y) == LENGTH
    sf.write(args.out, y, SR, subtype='PCM_24')
    print(f'master gain {20 * np.log10(G):+.2f} dB, max limiter GR {20 * np.log10(float(lim.min())):.2f} dB '
          f'-> {lufs(y):.2f} LUFS, TP {true_peak_db(y):.2f} dBTP -> {args.out}')


if __name__ == '__main__':
    sys.exit(main())
