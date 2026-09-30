"""A4/A5's release and inhale, using the existing notes and their cached performances.

This is a bounded premaster treatment, applied by sound_v3.write for A's adopted AP2. The source
score cache stays intact: it still serves the score provenance and AP2 walk checks.
Only frames 960..1120 are replaced, with zero-slope joins. No notes are synthesized,
retimed, or re-sampled. The hall tail continues through the old hard breath gate.
"""
import importlib
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal

from timeline_v3 import FR, SR, BarMap

START, END = 960, 1120
MASTER_WINDOWS = ((960, 1440),)  # Owner integration must use the UNION of every adopted polish lane.
GAIN_POINTS = ((960, 0), (984, -1.5), (1000, -5), (1020, -18),
               (1026, -18), (1034, -10), (1040, -3), (1052, 0), (1120, 0))


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u*u*u*(u*(u*6.0 - 15.0) + 10.0)


def curve(frames, points, outside=0.0):
    """Quintic interpolation in dB; explicit exterior values make the edit bounded."""
    out = np.full(np.shape(frames), outside, dtype=np.float64)
    for (f0, g0), (f1, g1) in zip(points, points[1:]):
        take = (frames >= f0) & (frames < f1)
        out[take] = g0 + (g1-g0)*smooth((frames[take]-f0)/(f1-f0))
    return out


def local_score(cut, first=START, last=END):
    """Mix the cached performances locally, with the hall alive across the ignition breath.

    The preroll covers the complete hall IR plus one second of filter settling. Parts
    still use the renderer's seats, EQ, gain and sends. Fail if an active group or an
    additional score processing stage makes this local reconstruction inapplicable.
    """
    import mix as MX
    import render_v3 as RV

    bm = BarMap(cut)
    score = importlib.import_module(f'score_v3_{cut}').build(bm)
    # Building the note sheet reads sample attacks; the mix below uses only mmap stems.
    import sampler
    sampler.clear_caches()
    if score.push or getattr(score, 'fader', None):
        raise ValueError('ignition local mix requires the unmodified A score processing chain')
    irs = RV.hall_ir()
    a = max(0, first*FR - max(map(len, irs)) - SR)
    b = last*FR
    dry = np.zeros((b-a, 2), np.float32)
    send = np.zeros_like(dry)
    manifest = json.loads((Path(RV.CACHE)/f'manifest_final_{cut}.json').read_text())
    for name, part in score.used().items():
        key = manifest[name]
        if RV.part_key(part.to_dict(), bm.render_n) != key:
            raise ValueError(f'ignition local mix: stale cached part {name}; render current score first')
        off, raw = RV.load_stem(name, key)
        lo, hi = max(a, off), min(b, off+len(raw))
        if hi <= lo:
            continue
        y = np.array(raw[lo-off:hi-off], dtype=np.float32)
        if not np.any(y):
            continue
        if any(name.startswith(group) for group in score.groups):
            raise ValueError(f'ignition local mix: active grouped part {name} needs its group chain')
        y = MX.pan_width(y*np.float32(10**(part.gain_db/20)), part.pan, part.width)
        y = MX.depth_eq(y, part.depth)
        spec = score.eq.get(name, (None, None))
        if spec[0]:
            y = MX.highpass(y, spec[0])
        if spec[1]:
            bb, aa = signal.butter(2, spec[1]/(SR/2))
            y = signal.lfilter(bb, aa, y, axis=0).astype(np.float32)
        if len(spec) > 2 and spec[2]:
            y += MX.highpass(y, spec[3] if len(spec)>3 else 6000, order=2)*np.float32(10**(spec[2]/20)-1)
        dry[lo-a:hi-a] += y*(1.0-0.35*part.depth)
        send[lo-a:hi-a] += y*part.send
    mixed = MX.highpass(dry + MX.convolve_stereo(send, irs), 22.0)
    return mixed[first*FR-a:]


def blend_score(score, replacement):
    """In-place, exact identity outside [960,1120); no gain change on later music."""
    if len(replacement) != (END-START)*FR:
        raise ValueError('ignition replacement has the wrong length')
    f = START + np.arange(len(replacement))/FR
    weight = smooth((f-START)/16)*smooth((END-f)/24)
    shaped = replacement * np.power(10.0, curve(f, GAIN_POINTS)/20)[:, None]
    region = score[START*FR:END*FR]
    region += ((shaped-region)*weight[:, None]).astype(np.float32)
    return score


def inhale(sfx):
    """Recorded night air supplies a quiet floor, rises with the disc, then releases.

    This source already belongs to A's sound world. Its peak target (-48 dBFS RMS in
    premaster) is an authored level; the delivered measurement is in the lane audit.
    The quintic envelope has no attack transient or percussion stroke.
    """
    import sound_recipes_A as R
    import sound_v3 as SV

    first, last = 1004, 1088
    count = (last-first)*FR
    # Fixed recorded excerpt, rather than a loop, preserves the take's air movement.
    ref, t0, t1, _ = R.WIND[0]
    y = SV.load(ref, t0, t0+count/SR).copy()
    if len(y) != count:
        raise ValueError('ignition inhale source is shorter than its bounded window')
    y = SV.proc(y, hp=180, lp=3200)
    rms = float(np.sqrt(np.mean(y.astype(np.float64)**2)))
    if rms <= 1e-10:
        raise ValueError('ignition inhale source is silent')
    y *= np.float32(10**(-48/20)/rms)
    f = first + np.arange(count)/FR
    floor = 0.24
    envelope = smooth((f-first)/16)*(floor+(1-floor)*smooth((f-1026)/14))*smooth((last-f)/48)
    sfx[first*FR:last*FR] += y*envelope[:, None].astype(np.float32)
    return sfx


def prepare_master(cut, score, sfx):
    """Called once after loading the untouched score cache, before stem-linked mastering."""
    if cut != 'AP2':
        return score, sfx
    blend_score(score, local_score(cut))
    inhale(sfx)
    return score, sfx


def bound_master(cut, sfx):
    """Preserve the delivered exterior before limiting and through the master chain.

    The two cache references are delivery artifacts, named and SHA-256 pinned in
    polish_ignition_reference.json. They must be transferred with this edit; a missing
    or different reference is a hard error. Ordinary renders never update them.
    """
    if cut != 'AP2':
        return sfx, {}
    import soundfile as sf
    import render_v3 as RV

    manifest = json.loads(Path(__file__).with_name('polish_ignition_reference.json').read_text())
    files = {}
    for kind, row in manifest['files'].items():
        path = Path(RV.CACHE)/row['filename']
        if not path.is_file():
            raise ValueError(f'missing ignition reference {path.name}; copy the lane delivery artifact into music/cache/v3')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for data in iter(lambda: handle.read(1<<20), b''):
                digest.update(data)
        if digest.hexdigest() != row['sha256']:
            raise ValueError(f'ignition reference hash mismatch: {path.name}')
        files[kind] = path
    original = sf.SoundFile(files['sfx'])
    if original.frames != len(sfx) or original.samplerate != SR or original.channels != 2:
        raise ValueError('ignition reference effects disagree with render format')
    # The peak guard's global loudness estimate changes when the breath changes.
    # Retain its original result outside this lane, rather than re-limiting those cues.
    with original:
        # Preserve the complement of the windows; other adopted lanes must be added
        # here before integration, or their effects would be restored to the reference.
        cursor = 0
        exterior = []
        for first,last in sorted(MASTER_WINDOWS):
            a,b = first*FR,last*FR
            if a>cursor:
                exterior.append((cursor,a))
            cursor=max(cursor,b)
        if cursor<len(sfx):
            exterior.append((cursor,len(sfx)))
        for a,b in exterior:
            original.seek(a)
            sfx[a:b] = original.read(b-a,dtype='float32')
    return sfx, dict(reference_env=np.load(files['envelope'],mmap_mode='r'), edit_windows=MASTER_WINDOWS)


def refresh_breath_probes(cut, score, probes):
    """The sound premaster differs from the cached score; do not report the old gate."""
    if cut != 'AP2':
        return probes
    result = [dict(p) for p in probes]
    rows = [p for p in result if abs(p['start_s']-1020/24)<1e-6]
    if len(rows) != 1:
        raise ValueError('ignition breath probe anchor is missing or ambiguous')
    row = rows[0]
    a,b = 1020*FR,1040*FR
    def rms(lo,hi):
        return float(10*np.log10(np.mean(np.asarray(score[lo:hi],np.float64)**2)+1e-20))
    row.update(before_db=rms(a-int(.3*SR),a),inside_db=rms(a+int(.036*SR),b-int(.006*SR)),
               after_db=rms(b,b+int(.2*SR)),measurement='post-polish score premaster, dry plus hall')
    return result
