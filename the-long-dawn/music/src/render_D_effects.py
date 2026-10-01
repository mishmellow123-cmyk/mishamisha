"""Private, opt-in D effects and draft mix. Run this CLI through onepy.

All processing uses short clips/blocks and private temporary memmaps. No shared
cache is written. The surviving score input is a mastered WAV, not a premaster;
donor effects are reconstructed with their saved master envelopes to match that
gain stage. Only the protected AP2 minute promises packed-PCM identity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d

ROOT = Path(__file__).resolve().parents[2]
SR, FPS, FR, FRAMES = 48000, 24, 2000, 9200
N = FRAMES * FR
BLOCK = 8 * SR


def blocks(n, size=BLOCK):
    for a in range(0, n, size):
        yield a, min(n, a + size)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rss():
    # This production runs on macOS, where ru_maxrss is bytes.
    import sys
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)


def mmap_audio(path, n=N):
    return np.memmap(path, dtype='float32', mode='w+', shape=(n, 2))


def allocate_audio(path, n=N, *, resident=False):
    if type(resident) is not bool:
        raise ValueError('resident storage must be explicitly boolean')
    return np.zeros((n, 2), np.float32) if resident else mmap_audio(path, n)


def read_into(path, target, start=0, stop=None):
    stop = len(target) if stop is None else stop
    with sf.SoundFile(path) as f:
        if f.samplerate != SR or f.channels != 2 or len(f) < stop:
            raise ValueError(f'Invalid donor format or length: {Path(path).name}')
        f.seek(start)
        for a, b in blocks(stop - start):
            target[start + a:start + b] = f.read(b-a, dtype='float32', always_2d=True)


def put(stem, y, offset, *, stop=None):
    a, b = max(0, offset), min(len(stem), offset + len(y))
    if stop is not None:
        b = min(b, stop)
    if a < b:
        stem[a:b] += y[a-offset:b-offset]


def effective_stop(row, controls):
    """A request cutoff also stops every expanded event's processed wet tail."""
    stops = [row['stop_f']] if row.get('stop_f') is not None else []
    stops += [c['hit_f'] for c in controls
              if c['action'] == 'cut_existing_layers' and row['request_id'] in c['targets']]
    return min(stops) if stops else None


def log_stage(name, **values):
    print(json.dumps(dict(stage=name, peak_rss_bytes=rss(), **values)), flush=True)


def hard_zero(x, a, b, release=480):
    """Release ends before the first required silent sample."""
    a, b = int(a), int(b)
    k = min(release, a)
    if k:
        x[a-k:a] *= np.linspace(1, 0, k, dtype=np.float32)[:, None]
    x[a:b] = 0


def voice_ceiling(sfx, ceiling_db=-45):
    a, b = 8640*FR, 8880*FR
    pk = float(np.max(np.abs(sfx[a:b])))
    g = min(1.0, 10**(ceiling_db/20) / max(pk, 1e-12))
    k = 10*FR
    sfx[a-k:a] *= np.linspace(1, g, k, dtype=np.float32)[:, None]
    sfx[a:b] *= np.float32(g)
    sfx[b:b+k] *= np.linspace(g, 1, k, dtype=np.float32)[:, None]
    return float(20*np.log10(max(g, 1e-12)))


def constraints_inplace(sfx, score=None):
    hard_zero(sfx, 3400*FR, 3440*FR)
    # Keep the effects faint throughout the entire voice beat, then leave
    # an exact 80-frame digital opening shared with the score.
    voice_gain_db = voice_ceiling(sfx)
    hard_zero(sfx, 8660*FR, 8740*FR)
    sfx[-int(.35*SR):] *= np.linspace(1, 0, int(.35*SR), dtype=np.float32)[:, None]**2
    if score is not None:
        hard_zero(score, 3400*FR, 3440*FR)
        hard_zero(score, 8660*FR, 8740*FR)
        # The isolated effect owns this single stroke. The score's bar44 is
        # another anvil, so retain one audible hammer rather than two attacks.
        score[3440*FR:3520*FR] = 0
        hard_zero(score, 9120*FR, N)
    return dict(voice_effect_gain_db=voice_gain_db,
                score_gap_anvil='muted3440:3520; effects own the single stroke')


def readonly_space(name):
    p = ROOT / 'music/cache/sound/ir' / (name + '.npy')
    if not p.is_file():
        raise FileNotFoundError(f'Measured IR must already exist: {name}')
    return list(np.load(p))


def clear_audio_cache(S):
    S._LOADED.clear()


def picture_provenance(row):
    """Copy the evidence scope, not merely a measured boolean, to delivery."""
    fields = ('picture_frame', 'measured_ref', 'measurement_scope', 'measurement',
              'timing_note', 'supersedes', 'editorial_reference',
              'inherited_measurement_ref', 'inherited_measurement_file', 'inherited_measurement_scope',
              'envelope_provenance', 'distance_basis', 'level_trim_basis', 'pan_basis', 'design_reference',
              'level_adjustment_provenance', 'phase3_level_adjustment_provenance',
              'race_work_clock', 'race_role', 'design_revision', 'texture_provenance',
              'end_provenance', 'continuity_provenance', 'picture_revision')
    fields += ('bridge_provenance', 'post_gain_f', 'source_span', 'source_seed')
    result = {key: deepcopy(row[key]) for key in fields if key in row}
    result['measured_picture'] = row.get('source') == 'measured'
    return result


def spatial_and_tail(y, rc, space, S, *, breaths=(), source_offset=0):
    send = rc.get('send', space.get('event_send', .08))
    if send:
        ir = readonly_space(space.get('outdoor', 'forest20'))
        y = np.pad(y, ((0, max(len(x) for x in ir)), (0, 0)))
        local_wins = [(max(0, a-source_offset), min(len(y), b-source_offset), ex)
                      for a, b, ex in breaths if b > source_offset and a < source_offset+len(y)]
        wet = (S.R2.convolve_with_breaths(y, ir, local_wins, len(y))
               if local_wins else S.convolve(y, ir))
        y += np.float32(send) * S.proc(wet, hp=space.get('wet_hp', 150), lp=space.get('wet_lp', 9000))
    return S.proc(y, hp=space.get('stem_hp', 25))


def event_clip(rc, seed, space, S, *, donor_cut=None, donor_hit_f=None):
    y, hit = S.event(rc, S.rng_for(seed))
    if rc.get('env_after'):
        pts = np.asarray(rc['env_after'], float)
        tt = (np.arange(len(y))-hit)/SR
        y *= S.db(np.interp(tt, pts[:, 0], pts[:, 1])).astype(np.float32)[:, None]
    if rc.get('pan') is not None:
        y = S.pan(y, rc['pan'])
        if rc['pan'] in (-1, 1) and not rc.get('dist'):
            y[:, 1 if rc['pan'] == -1 else 0] = 0
    if rc.get('dist'):
        y = S.distance(y, rc['dist'], readonly_space(space.get('distance', 'forest20')))
    y = S.limit_crest(y, rc.get('crest', space.get('event_crest')))
    g = S.match_gain(y, rc['level'], 'event', rc.get('trim', 0))
    y *= np.float32(S.db(g))
    breaths, source_offset = (), 0
    if donor_cut:
        from timeline_v3 import BarMap
        breaths = S.R2.breath_windows(BarMap(donor_cut).breath_beats())
        source_offset = int(round(donor_hit_f*FR))-hit
        if not rc.get('no_breath'):
            y *= S.R2.breath_env(len(y), breaths, offset=source_offset)[:, None]
    return spatial_and_tail(y, rc, space, S, breaths=breaths, source_offset=source_offset), hit


def bed_clip(rc, row, seed, space, S):
    f0, f1 = row['f0'], row['f1']
    dur = (f1-f0)/FPS
    span = row.get('source_span')
    if span is not None:
        if len(span) != 2 or not span[0] <= f0 < f1 <= span[1] or not row.get('source_seed'):
            raise ValueError('continuous bed crop needs a containing source span and seed')
        source_dur = (span[1]-span[0])/FPS
    else:
        source_dur = dur
    y = S.limit_crest(S.bed(rc, source_dur, S.rng_for(row.get('source_seed', seed))), rc.get('crest', space.get('bed_crest', 18)))
    if 'level' in rc:
        ref = rc['level']
    else:
        # A ambience keeps its original synthesized reference and trim.
        from timeline_v3 import BarMap
        bm = BarMap('AP2')
        ref = S.ref_levels(bm, [(row, 'bed')], 'A')[row['id']]
    y *= np.float32(S.db(S.match_gain(y, ref, 'bed', rc.get('trim', 0))))
    if span is not None:
        a, b = [int(round((f-span[0])*FR)) for f in (f0, f1)]
        y = y[a:b].copy()
    tt = np.arange(len(y))/SR
    fi, fo = row.get('fade_in_s', row.get('fade_in', 1.5)), row.get('fade_out_s', row.get('fade_out', 1.5))
    env = np.sin(np.clip(tt/max(fi, 1e-3), 0, 1)*np.clip((dur-tt)/max(fo, 1e-3), 0, 1)*np.pi/2)**2
    if row.get('env_f'):
        pts = np.asarray(row['env_f'], float)
        env *= S.db(np.interp(f0+tt*FPS, pts[:, 0], pts[:, 1]))
    if rc.get('env'):
        pts = np.asarray(rc['env'], float)
        env *= S.db(np.interp(tt, pts[:, 0], pts[:, 1]))
    y *= env.astype(np.float32)[:, None]
    if rc.get('pan') is not None:
        y = S.pan(y, rc['pan'])
    if rc.get('width') is not None:
        y = S.width(y, rc['width'])
    return spatial_and_tail(y, dict(rc, send=rc.get('send', space.get('bed_send', 0))), space, S)


def post_gain(y, row, first_sample):
    """Authored absolute-D gain after wet processing and donor mastering."""
    if 'post_gain_f' not in row:
        return y
    pts = np.asarray(row['post_gain_f'], float)
    if (pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or not np.isfinite(pts).all()
            or np.any(np.diff(pts[:, 0]) <= 0)):
        raise ValueError('bridge gain needs ordered finite frame/dB pairs')
    for a, b in blocks(len(y)):
        frames = (first_sample+np.arange(a, b))/FR
        y[a:b] *= (10**(np.interp(frames, pts[:, 0], pts[:, 1])/20)).astype(np.float32)[:, None]
    return y


def donor_gain(y, donor_start, cut):
    # render_v3.master applies this causal filter before its saved envelope.
    sos = signal.butter(1, 8, 'high', fs=SR, output='sos')
    y[:] = signal.sosfilt(sos, y, axis=0).astype(np.float32)
    env = np.load(ROOT / 'music/cache/v3' / f'master_env_sound_{cut}.npy', mmap_mode='r')
    for a, b in blocks(len(y)):
        indices = np.clip(np.arange(a, b) + int(round(donor_start)), 0, len(env)-1)
        y[a:b] *= np.asarray(env[indices], np.float32).reshape(-1, 1)
    return y


def render_effects(stem, binding, receipt):
    import sound_v3 as S
    import sound_d_designs as D
    import sound_d_foley as F
    import sound_recipes_C as C
    S._CACHE_READ_ONLY = True
    p = binding['reuse_plan']
    # The phase1 recipe export has donor_cut implicit in reference/seed IDs.
    for r in p['EXTRA_EVENTS']:
        rc = deepcopy(p['RECIPES'][r['id']])
        cut = 'AP2' if r['seed_id'].startswith('A.') else 'C5P2'
        y, hit = event_clip(rc, r['seed_id'], r['space'], S, donor_cut=cut, donor_hit_f=r['donor_hit_f'])
        donor_gain(y, r['donor_hit_f']*FR-hit, cut)
        offset = int(round(r['hit_f']*FR))-hit
        post_gain(y, r, offset)
        put(stem, y, offset)
        receipt['events'].append(dict(id=r['id'], hit_f=r['hit_f'], source=r['source'],
                                      source_sync=r['source_sync'], sample_hit=offset+hit,
                                      **({'hook': r['hook']} if 'hook' in r else {}),
                                      **picture_provenance(r),
                                      method='donor recipe + saved donor master envelope'))
        clear_audio_cache(S)
    log_stage('reused events complete', count=len(p['EXTRA_EVENTS']))
    for r in p['BED_CROPS']:
        rc = deepcopy(p['RECIPES'][r['id']])
        original = r['donor_row']
        y = bed_clip(rc, original, r['seed_id'], r['space'], S)
        cut = 'AP2' if r['seed_id'].startswith('A.') else 'C5P2'
        donor_gain(y, original['f0']*FR, cut)
        a, b = [int(round((f-original['f0'])*FR)) for f in r['donor_crop']]
        cropped = y[a:b]
        post_gain(cropped, r, int(round(r['f0']*FR)))
        put(stem, cropped, int(round(r['f0']*FR)))
        receipt['beds'].append(dict(id=r['id'], f0=r['f0'], f1=r['f1'], source='derived',
                                    donor_crop=r['donor_crop'], seed=r['seed_id'], **picture_provenance(r)))
        clear_audio_cache(S)
    log_stage('reused beds complete', count=len(p['BED_CROPS']))
    env = np.load(ROOT/'music/cache/v3/master_env_sound_C5P2.npy', mmap_mode='r')
    # A measured gain-stage reference, not a D loudness result. The actual mix
    # below measures and adjusts the resulting programme.
    reference_gain = float(np.median(env[::4800]))
    receipt['new_effect_reference_gain_db'] = float(20*np.log10(reference_gain))
    crowns = {}
    for r in binding['new_events']:
        rc = F.build(r['design']) if r['design'].startswith('crossing_') else D.build(r['design'])
        rc.update(r.get('recipe_overrides', {}))
        for k in ('pan', 'dist'):
            if r.get(k) is not None:
                rc[k] = r[k]
        rc['level'] += r.get('level_trim_db', 0)
        y, hit = event_clip(rc, r['id'], C.SPACE, S)
        y *= np.float32(reference_gain)
        offset = int(round(r['hit_f']*FR))-hit
        post_gain(y, r, offset)
        stop_f = effective_stop(r, binding['controls'])
        stop = None if stop_f is None else int(stop_f*FR)
        if stop is not None and 0 < stop-offset < len(y):
            k = min(FR, stop-offset)
            y[stop-offset-k:stop-offset] *= np.linspace(1, 0, k, dtype=np.float32)[:, None]
        put(stem, y, offset, stop=stop)
        receipt['events'].append(dict(id=r['id'], request_id=r['request_id'], hit_f=r['hit_f'],
                                      source=r['source'], hook=r['hook'], sample_hit=offset+hit,
                                      source_recipe=rc, stop_f=stop_f, **picture_provenance(r),
                                      isolated_channel_peaks=np.abs(y).max(axis=0).tolist()))
        if r['request_id'] in ('D.new.crowns.left', 'D.new.crowns.right'):
            crowns[r['pan']] = (y[:, 0 if r['pan'] == -1 else 1].copy(), hit, offset+hit)
        clear_audio_cache(S)
    if set(crowns) != {-1, 1}:
        raise ValueError('Both crown channels must be rendered')
    left, lh, left_sample = crowns[-1]
    right, rh, right_sample = crowns[1]
    n = min(len(left), len(right))
    receipt['crown_pair'] = dict(left_hit_sample=left_sample, right_hit_sample=right_sample,
        source_hit_offsets=[lh, rh], compared_samples=n,
        max_channel_difference=float(np.max(np.abs(left[:n]-right[:n]))),
        correlation=float(np.corrcoef(left[:n], right[:n])[0, 1]))
    if receipt['crown_pair']['max_channel_difference'] == 0:
        raise ValueError('Identical simultaneous crowns collapse the authored stereo placement')
    del crowns, left, right
    log_stage('new events complete', count=len(binding['new_events']))
    for r in binding['new_beds']:
        rc = D.build(r['design'])
        rc.update(r.get('recipe_overrides', {}))
        for k in ('pan', 'dist'):
            if r.get(k) is not None:
                rc[k] = r[k]
        rc['level'] += r.get('level_trim_db', 0)
        y = bed_clip(rc, r, r['id'], C.SPACE, S) * np.float32(reference_gain)
        stop_f = effective_stop(r, binding['controls'])
        stop_f = min(r['f1'], stop_f) if stop_f is not None else r['f1']
        stop = int(stop_f*FR)
        offset = int(r['f0']*FR)
        post_gain(y, r, offset)
        if 0 < stop-offset < len(y):
            k = min(FR, stop-offset)
            y[stop-offset-k:stop-offset] *= np.linspace(1, 0, k, dtype=np.float32)[:, None]
        put(stem, y, offset, stop=stop)
        receipt['beds'].append(dict(id=r['id'], request_id=r['request_id'], f0=r['f0'], f1=r['f1'],
                                    source=r['source'], hook=r['hook'], **picture_provenance(r)))
        clear_audio_cache(S)
    log_stage('new beds complete', count=len(binding['new_beds']))
    return stem


def sum_into(mix, score, sfx):
    for a, b in blocks(N):
        mix[a:b] = score[a:b] + sfx[a:b]
    read_into(ROOT/'music/out/v3/sound_AP2.wav', mix, stop=1440*FR)


def gain_suffix(score, sfx, db):
    gain = np.float32(10**(db/20))
    for a, b in blocks(N-1440*FR):
        lo, hi = a+1440*FR, b+1440*FR
        score[lo:hi] *= gain
        sfx[lo:hi] *= gain


def linked_limit(score, sfx, mix, ceiling_db=-1.35):
    """Small linked limiter; 4x peaks, 6ms lookahead, 150ms release.

    Each processing block has 0.5s left/right context; only its central region
    is written. A full per-sample envelope is never allocated.
    """
    from scipy.signal import lfilter
    threshold = 10**(ceiling_db/20)
    worst = 1.0
    for a, b in blocks(N-1440*FR):
        a, b = a+1440*FR, b+1440*FR
        lo, hi = max(1440*FR, a-int(.5*SR)), min(N, b+int(.5*SR))
        y = np.asarray(mix[lo:hi], np.float64)
        up = signal.resample_poly(y, 4, 1, axis=0)
        peak = np.abs(up).reshape(len(y), 4, 2).max(axis=(1, 2))
        # Cancellation in the mix must not hide an over-peak delivery stem.
        for source in (score, sfx):
            separate = signal.resample_poly(np.asarray(source[lo:hi], np.float64), 4, 1, axis=0)
            peak = np.maximum(peak, np.abs(separate).reshape(len(y), 4, 2).max(axis=(1, 2)))
        need = np.minimum(1., threshold/(peak+1e-15))
        need = minimum_filter1d(need, size=2*int(.006*SR)+1)
        coeff = np.exp(-1/(.15*SR))
        smooth = lfilter([1-coeff], [1, -coeff], need, zi=[coeff])[0]
        g = np.minimum(need, smooth)[a-lo:b-lo].astype(np.float32)
        worst = min(worst, float(g.min()))
        score[a:b] *= g[:, None]
        sfx[a:b] *= g[:, None]
    return float(20*np.log10(max(worst, 1e-12)))


def load_contract(path, score_path):
    doc = json.loads(Path(path).read_text())
    if doc.get('ok') is not True or not all(r.get('ok') is True for r in doc.get('checks', [])):
        raise ValueError('Scored verification must pass all checks')
    if doc.get('wav_sha256') != sha(score_path):
        raise ValueError('Scored verification does not fingerprint this score WAV')
    bands = [(r['section'], *r['band_lu']) for r in doc['checks'] if r['check'] == 'level band']
    if len(bands) != 34:
        raise ValueError('Need all 34 current score level bands')
    return bands, doc['wav_sha256']


def write_final(path, x, donor, zero_ranges):
    rng = np.random.default_rng(3030)
    with sf.SoundFile(path, mode='w', samplerate=SR, channels=2, subtype='PCM_24') as f:
        for a, b in blocks(N):
            y = np.asarray(x[a:b], np.float32).copy()
            # Explicit TPDF dither only outside protected/zero windows.
            y += ((rng.random(y.shape)-rng.random(y.shape))/2**24).astype(np.float32)
            if a < 1440*FR:
                with sf.SoundFile(donor) as src:
                    src.seek(a)
                    end = min(b, 1440*FR)
                    y[:end-a] = src.read(end-a, dtype='float32', always_2d=True)
            for z0, z1 in zero_ranges:
                lo, hi = max(a, z0*FR), min(b, z1*FR)
                if lo < hi:
                    y[lo-a:hi-a] = 0
            if b == N:
                y[-1] = 0
            f.write(y)


def run(out, score_path=None, score_contract=None, temp_parent=None, measurements=None, race_revision=None, picture_revision=None,
        transition_pass=False):
    from sound_d_binding import build
    import verify_D_effects as V
    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    binding = build(measurements=measurements, race_revision=race_revision, picture_revision=picture_revision,
                    transition_pass=transition_pass)
    phase = binding.get('phase', 2)
    resident = picture_revision == 'locked'
    binding_path = out/f'effects_binding_phase{phase}.json'
    binding_path.write_text(json.dumps(binding, indent=2)+'\n')
    source_counts = {}
    for row in binding['new_events'] + binding['new_beds']:
        source_counts[row['source']] = source_counts.get(row['source'], 0) + 1
    receipt = dict(phase=phase, status='scoped-evidence draft', frames=FRAMES, sample_frames=N,
                   new_timing_source_counts=source_counts,
                   picture_scope='Per-row native plate, cover-grid, inherited or estimated; no finished-film sync claim.',
                   binding=dict(file=binding_path.name, sha256=sha(binding_path)),
                   measurement_overlay_sha256=binding.get('measurement_overlay', {}).get('sha256'),
                   score_input=None, events=[], beds=[], stages=[],
                   audio_storage=dict(programme='resident float32' if resident else 'private float32 memmap',
                                      processing_block_samples=BLOCK, resident_verification=resident))
    from sound_d_table import BEATS
    sections = [dict(id=f'D{i+1:02d}', name=n, f0=a, f1=b) for i, (n, a, b) in enumerate(BEATS)]
    if score_path:
        score_bands, score_sha = load_contract(score_contract, score_path)
        bands = [(sid, lo, -35 if sid == 'D34' else hi) for sid, lo, hi in score_bands]
        receipt['level_contract'] = dict(source='current scored verification, matched WAV SHA256',
            score_bands=score_bands, mix_bands=bands,
            authored_mix_override=dict(section='D34', band_lu=[-120, -35],
                reason='Final hearth release belongs to the effects mix. The score-only -60 LU ceiling and contained-silence policy do not describe this audible release. All 33 other bands and exact analyze_v3 windows remain unchanged.'))
        receipt['score_input'] = dict(file=Path(score_path).name, sha256=score_sha, gain_stage='already-mastered score WAV')
    else:
        # Explicit fallback can render effects and mix against silence. A level
        # map needs the score's published contract; do not invent one here.
        bands, score_bands = [], []
        receipt['score_input'] = dict(file=None, gain_stage='silence; no score available')
    with tempfile.TemporaryDirectory(prefix=f'soundd-phase{phase}-', dir=temp_parent) as tmp:
        sfx = allocate_audio(Path(tmp)/'sfx.f32', resident=resident)
        score = allocate_audio(Path(tmp)/'score.f32', resident=resident)
        mix = allocate_audio(Path(tmp)/'mix.f32', resident=resident)
        sfx[:] = score[:] = mix[:] = 0
        t = time.monotonic()
        render_effects(sfx, binding, receipt)
        receipt['stages'].append(dict(name='effects', seconds=time.monotonic()-t, peak_rss_bytes=rss()))
        if score_path:
            read_into(score_path, score)
        else:
            read_into(ROOT/'music/out/v3/sound_AP2_score.wav', score, stop=1440*FR)
        receipt['constraints'] = constraints_inplace(sfx, score)
        read_into(ROOT/'music/out/v3/sound_AP2_sfx.wav', sfx, stop=1440*FR)
        read_into(ROOT/'music/out/v3/sound_AP2_score.wav', score, stop=1440*FR)
        sum_into(mix, score, sfx)
        initial_reduction = linked_limit(score, sfx, mix)
        sum_into(mix, score, sfx)
        receipt['initial_limiter_reduction_db'] = initial_reduction
        # Solve a suffix-only loudness correction with AP2's minute fixed.
        adjustments = []
        for iteration in range(8):
            loud = V.lufs(mix)
            log_stage('master loudness', iteration=iteration, lufs=loud)
            correction = float(np.clip(-16-loud, -6, 6))
            if abs(correction) < .025:
                break
            gain_suffix(score, sfx, correction)
            sum_into(mix, score, sfx)
            reduction = linked_limit(score, sfx, mix)
            sum_into(mix, score, sfx)
            adjustments.append(dict(iteration=iteration, measured_lufs=loud, suffix_gain_db=correction,
                                    max_limiter_reduction_db=reduction))
        receipt['master_adjustments'] = adjustments
        receipt['final_voice_gain_db'] = voice_ceiling(sfx, -40.5)
        sum_into(mix, score, sfx)
        # Validate section bands before writing; report actual failures. A
        # draft never silently drops a band to obtain a green report.
        track = V.st_loudness(mix)
        receipt['level_map'] = V.level_rows(track, sections, bands) if bands else []
        receipt['prewrite_lufs'] = V.lufs(mix)
        receipt['prewrite_true_peak_dbtp'] = V.true_peak_db(mix)
        log_stage('prewrite checks', lufs=receipt['prewrite_lufs'],
                  true_peak_dbtp=receipt['prewrite_true_peak_dbtp'],
                  failed_bands=[r for r in receipt['level_map'] if not r['ok']])
        paths = dict(sfx=out/'sfx_D_draft.wav', score=out/'score_D_mix_draft.wav', mix=out/'sound_D_draft.wav')
        for kind, x in (('sfx', sfx), ('score', score), ('mix', mix)):
            donor = ROOT/'music/out/v3'/('sound_AP2.wav' if kind == 'mix' else f'sound_AP2_{kind}.wav')
            zeros = [(3400,3440), (8660,8740)] + ([(9120,9200)] if kind == 'score' else [])
            write_final(paths[kind], x, donor, zeros)
            info = sf.info(paths[kind])
            if (info.frames, info.samplerate, info.channels, info.subtype) != (N, SR, 2, 'PCM_24'):
                raise ValueError('Final WAV format/length verification failed: ' + kind)
            # The encoded stage is complete. Release its private backing file
            # before writing the next stem, keeping retained drafts plus peak
            # scratch use inside the lane disk budget. Full PCM checks follow.
            if phase >= 4:
                lane = out.parent if out.name.startswith('phase') else out
                retained = sum(p.stat().st_size for p in lane.rglob('*') if p.is_file())
                scratch = sum(p.stat().st_size for p in Path(tmp).rglob('*') if p.is_file())
                receipt.setdefault('disk_usage_samples', []).append(dict(
                    stage='encoded ' + kind + ' before scratch release', lane_bytes=retained,
                    private_audio_bytes=scratch, combined_bytes=retained + scratch))
                if retained + scratch >= 1_500_000_000:
                    raise ValueError('sound lane disk use exceeds 1.5 GB')
            if isinstance(x, np.memmap):
                backing = Path(x.filename)
                x.flush()
                x._mmap.close()
                backing.unlink()
        del sfx, score, mix, x
    receipt['seconds'] = time.monotonic()-started
    receipt['peak_rss_bytes'] = rss()
    receipt['temporary_intermediates_deleted'] = True
    receipt['output'] = {k: dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p)) for k, p in paths.items()}
    receipt['nominal_binding_audit'] = V.verify_event_bindings(receipt, binding)
    (out/'render_effects_D_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    verification = V.verify_paths(paths['sfx'], paths['mix'], paths['score'], bands=bands, sections=sections,
                                  score_bands=score_bands, ap2_root=ROOT/'music/out/v3',
                                  out_json=out/'verify_effects_D.json', resident=resident)
    receipt['whole_job_peak_rss_bytes'] = rss()
    receipt['render_and_verification_seconds'] = time.monotonic()-started
    receipt['verification_ok'] = verification['ok'] and receipt['nominal_binding_audit']['ok']
    (out/'render_effects_D_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(outputs=receipt['output'], seconds=receipt['seconds'], peak_rss_bytes=rss(),
                          verification_ok=receipt['verification_ok'])), flush=True)
    return receipt, verification


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--score', type=Path)
    p.add_argument('--score-contract', type=Path)
    p.add_argument('--temp-parent', type=Path)
    p.add_argument('--measurements', type=Path, help='Validated native picture evidence overlay')
    p.add_argument('--race-revision', choices=['round5'], help='Opt in to the remeasured Round5 work rhythm')
    p.add_argument('--picture-revision', choices=['locked'], help='Final locked-picture fire, caption and dissolve treatment')
    p.add_argument('--transition-pass', action='store_true', help='Apply the adopted R8 effects bridges')
    args = p.parse_args()
    if args.score and not args.score_contract:
        p.error('--score requires current --score-contract verification JSON')
    receipt, verification = run(args.out, args.score, args.score_contract, args.temp_parent, args.measurements, args.race_revision, args.picture_revision, args.transition_pass)
    if not receipt['verification_ok']:
        raise SystemExit(2)
