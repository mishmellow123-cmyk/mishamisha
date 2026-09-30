"""THE LONG DAWN v3: the delivery chain. Full-resolution masters of A, A's ALT master, B and C, with QC.

Each film is built from one encoded segment per EDL shot. A segment is keyed by everything that makes its pictures
(the shot's EDL entry and take, every source frame's path, size and mtime, the text over it, the frame-pipeline
code), so a segment is re-encoded only when one of those changes. When the last render lands, only its shot is
encoded; the film is re-joined (a byte-exact join of raw H.264 segments with no B-frames), muxed and QC'd in
minutes. The same engine makes the half-res animatics (--profile animatic).

    bash edit/deliver.sh                                   # A, A ALT, B, C masters + QC (through renderq)
    CUTS="B" bash edit/deliver.sh                          # one film
    python3 edit/deliver.py --cut A --variant codedtowers  # A's ALT master
    python3 edit/deliver.py --cut C --qc-only              # QC the existing master again
    python3 edit/deliver.py --cut A --profile animatic     # the half-res animatic, incrementally

Outputs in $LD_DELIVERY, else ~/mishamisha/_local_logs/delivery/ (animatics in _local_logs/animatic/ as before):
    <cut>_master[_codedtowers].mov     H.264 High (x264 CRF 14, slow, BT.709 limited, 1920x804, 24 fps) + the
                                       cut's sound master as 24-bit PCM, 48 kHz stereo
    <cut>_master[_codedtowers].mp4     the same picture with AAC 320k: the screener for players
    <cut>_master[...]_QC.txt / .json   exact lengths, format, black frames, flashes, audio peaks and loudness, slates
    cache/<profile>/<key>.h264         the segments (content-addressed; unreferenced ones are pruned)
"""
import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'edit'))
import assemble as AS  # noqa: E402

EDL = AS.EDL
titles = AS.titles
FPS, SR = 24, 48000
DELIVERY = os.environ.get('LD_DELIVERY') or os.path.expanduser('~/mishamisha/_local_logs/delivery')
ENGINE = 'x5.1'                                        # bump when the encode itself changes
PROFILES = {
    'master': dict(scale=1.0, clean=True, crf=14, preset='slow', out=DELIVERY, finish=True),   # FINISH
    'animatic': dict(scale=0.5, clean=False, crf=23, preset='medium', out=AS.ANIMATIC_DIR),
    'animatic_clean': dict(scale=0.5, clean=True, crf=23, preset='medium', out=AS.ANIMATIC_DIR),
}
# the segments' SPS carries only the matrix; this stamps full BT.709 (primaries, transfer, matrix, limited range)
VUI = 'h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0'
FILM = {'A': 'EVERY STEP CLOSER', 'B': 'THE VIGIL', 'C': 'THE LAST PAGES'}


def _code_hash():
    """(frame code, text code). The frame pipeline's own code re-keys every segment; the text-drawing code only
    re-keys segments with text over them (each segment also keys its own lines' words, frames and setting, so a
    table change re-encodes only the shots it touches)."""
    ctx = {n: inspect.getsource(o) for n, o in inspect.getmembers(AS.Ctx, inspect.isfunction)}
    frame = [ctx[n] for n in sorted(ctx) if n not in ('x2', 'slate', 'ember')] + [
        inspect.getsource(o) for o in (AS.grade, AS.burn_in, AS.smooth, AS.bar_beat, AS.locate, AS.chain,
                                       AS.plan_shot)]
    slate = [ctx['slate'], inspect.getsource(AS.make_slate), inspect.getsource(AS.draw_slate_clock)]
    title = [ctx['ember'], open(os.path.join(ROOT, 'edit', 'title_scene.py')).read()]
    text = [inspect.getsource(o) for o in (titles.TextV3, titles.render_line, titles.render_block, titles.lines_v3,
                                           titles.composite_v3, titles._noise, titles._heat_rgb, titles.smooth,
                                           titles._font)]
    text.append(repr([titles.Y_LOWER, titles.Y_TOP, titles.Y_BOTTOM, titles.Y_MID, titles.PARCH.tolist(),
                      titles.IRON.tolist(), titles.FIRE_RAMP.tolist(), titles.INK.tolist(), titles.GLOW.tolist(),
                      titles.W, titles.ITALIC, titles.EBG_ITALIC, titles.CINZEL, titles.LUMA.tolist(),
                      titles.RIM_FLOOR, titles.RIM_GAIN, titles.RIM_MAX, titles.GLOW_BACKOFF, titles.HEAT_LIFT]))
    h = lambda xs: hashlib.sha1('\n'.join(xs).encode()).hexdigest()[:12]
    return dict(frame=h(frame), text=h(text), slate=h(slate), x2=h([ctx['x2']]), title=h(title))


def name_of(cut, variant, profile):
    if profile.startswith('animatic'):
        return f"{cut}_animatic{'_' + variant if variant else ''}{'_clean' if profile == 'animatic_clean' else ''}"
    return f"{cut}_master{'_' + variant if variant else ''}"


# ------------------------------------------------------------------------------------------------ segments
def _stat(p):
    try:
        st = os.stat(p)
        return f'{p}|{st.st_size}|{st.st_mtime_ns}'
    except OSError:
        return f'{p}|missing'


def frame_sources(cut, variant, plan, f):
    """Every file frame f of this plan reads (mirrors Ctx.take_frame and Ctx.under)."""
    if plan['kind'] != 'take':
        return []
    take = plan['take']
    p, _ = AS.locate(take, cut, variant, f)
    if p is None:
        return ['none']
    if isinstance(p, tuple):                              # a video take: the file and the index in it
        return [f'{_stat(p[1])}#{p[2]}']
    out = [_stat(p)]
    if take.get('add'):
        out.append(_stat(AS.index(os.path.join(AS.RENDERS, take['add']))[f + take['off']]))
    if take.get('matte'):
        mp = AS.index(os.path.join(AS.RENDERS, take['matte'])).get(f + take['off'])
        out.append(_stat(mp) if mp else 'matte:none')
        spec = take.get('under')
        if spec:
            up, _ = AS.locate_under(spec, f)
            out.append(_stat(up) if up else 'under:none')
    return out


def _finish_id():
    """FINISH: the finish's identity (the look, its code, its baked LUTs) for the master segments' keys."""
    sys.path.insert(0, os.path.join(ROOT, 'finish'))
    import stage
    return f'{stage.LOOK}:{stage.code_id()}'



_TRANS_CODE = []


def transition_sources(cut, variant, t, f):
    """What a transition frame is made of: the window (spec + the comp's code), its layer frames and both sides'
    source frames (the outgoing held after the boundary, the incoming held before it)."""
    if not _TRANS_CODE:                                       # the core, once; then each kind's own code
        _TRANS_CODE.append({'': hashlib.sha1(inspect.getsource(AS._transitions).encode()).hexdigest()[:12]})
    kc = _TRANS_CODE[0]
    if t['kind'] not in kc:
        kc[t['kind']] = hashlib.sha1(AS.transition_code(t['kind']).encode()).hexdigest()[:12]
    code = kc[''] + ':' + kc[t['kind']]
    out = [json.dumps([{k: t[k] for k in sorted(t) if k != 'note'}, code], default=str)]
    lay = AS.transition_layers(t, f)
    if lay is None:
        return out + ['no layer: plain cut']
    out += [_stat(p) for p in lay.values()]
    for g in ((min(f, t['cut'] - 1), max(f, t['cut'])) if 'cut' in t else ()):     # finish_ramp: one shot
        s = next(s for s in AS.EDL.EDL[cut] if s['f0'] <= g < s['f1'])
        out += frame_sources(cut, variant, AS.plan_shot(s, cut, variant), g)
    return out

def segment_key(cut, variant, prof, i, shot, plan, code, table, fin=None):
    h = hashlib.sha1()
    take = plan['take']
    tdesc = None if take is None else {k: take.get(k) for k in ('stem', 'off', 'mode', 'crop', 'grade', 'matte',
                                                                  'under', 'video', 'note', 'add')}
    rows = [(r['id'], r['line'], r['f_in'], r['f_out'], r['set'])
            + tuple((k, r[k]) for k in ('x', 'y', 'lines', 'parts', 'backing') if k in r)
            for r in table if r['f_in'] < shot['f1'] and r['f_out'] > shot['f0']]   # C5 placement and local backing
    n = shot['f1'] - shot['f0']
    slated = plan['kind'] == 'slate' or (plan['kind'] == 'take' and plan['have'] < n)
    head = [ENGINE, code['frame'], code['text'] if rows else None, code['slate'] if slated else None,
            code['x2'] if plan['kind'] == 'x2' else None, code['title'] if shot['kind'] == 'title' else None,
            prof['scale'], prof['clean'], prof['crf'], prof['preset'],
            cut, i,
            {k: shot[k] for k in ('sec', 'f0', 'f1', 'code', 'name', 'owner', 'desc', 'kind')},
            plan['kind'], tdesc, plan['have'], bool(plan['alt'])]
    if fin:                                                   # FINISH: finished segments carry the finish's identity
        head.append(['finish', fin])
    h.update(json.dumps(head, sort_keys=True, default=str).encode())
    title = shot['kind'] == 'title'
    if title and cut == 'B':                                  # B's stand-in sky is dusk_B's first frame
        h.update(_stat(os.path.join(AS.RENDERS, 'dusk_B', 'f_00000.jpg')).encode())
    for f in range(shot['f0'], shot['f1']):
        src = frame_sources(cut, variant, plan, f)
        if title:                                             # the ember title layer over the plate
            src.append(_stat(os.path.join(AS.RENDERS, f'title_{cut}', f'f_{f:05d}.exr')))
        tw = AS.transition_at(cut, f)
        if tw:                                                # an EDIT transition window (EDL.TRANS) over this frame
            src += transition_sources(cut, variant, tw, f)
        h.update('\n'.join(src).encode())
    h.update(json.dumps(rows).encode())
    return h.hexdigest()[:24]


def encode_cmd(W, H, prof, out):
    return ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
            '-framerate', str(FPS), '-i', '-',
            '-vf', 'scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,'
                   'format=yuv420p',
            '-c:v', 'libx264', '-preset', prof['preset'], '-crf', str(prof['crf']), '-profile:v', 'high',
            '-bf', '0', '-g', '48', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
            '-color_range', 'tv', '-f', 'h264', out]


def build_video(cut, variant, profile, workers, log):
    """Encode the stale segments of one film; return (ordered segment paths, stats)."""
    prof = PROFILES[profile]
    cache = os.path.join(DELIVERY, 'cache', profile)
    os.makedirs(cache, exist_ok=True)
    W, H = int(round(AS.FULL_W * prof['scale'])), int(round(AS.FULL_H * prof['scale']))
    shots = EDL.EDL[cut]
    plans = [AS.plan_shot(s, cut, variant) for s in shots]
    code, table = _code_hash(), titles.text_table(cut)
    fin = _finish_id() if prof.get('finish') else None       # FINISH (see finish/wire_edit.py)
    budget = None if os.environ.get('FINISH_ALL') == '1' else int(os.environ.get('FINISH_BUDGET', '1200'))
    segs, todo, backlog = [], [], []
    for i, (s, pl) in enumerate(zip(shots, plans)):
        sfin = fin if (pl['kind'] == 'take' and pl['have'] > 0) else None
        p = os.path.join(cache, segment_key(cut, variant, prof, i, s, pl, code, table, sfin) + '.h264')
        if sfin and not os.path.exists(p):
            q = os.path.join(cache, segment_key(cut, variant, prof, i, s, pl, code, table) + '.h264')
            n = s['f1'] - s['f0']
            if os.path.exists(q):                             # only the finish is missing: the backlog, on a budget
                if budget is not None and budget <= 0:          # (overshoots by at most one segment: always progress)
                    backlog.append(n)
                    segs.append(q)
                    continue
                if budget is not None:
                    budget -= n
        segs.append(p)
        if not os.path.exists(p):
            todo.append((i, s, p))
    t0 = time.time()
    nf = sum(s['f1'] - s['f0'] for _, s, _ in todo)
    log(f'{name_of(cut, variant, profile)}: {len(shots)} segments, {len(todo)} to encode ({nf} f)'
        + (f'; finish {fin}, backlog {len(backlog)} segments ({sum(backlog)} f) left' if fin else ''))
    if todo:
        with Pool(workers, initializer=AS._init,
                  initargs=(cut, variant, prof['scale'], prof['clean'], bool(fin))) as pool:
            for i, s, p in todo:
                tmp = p + '.part'
                proc = subprocess.Popen(encode_cmd(W, H, prof, tmp), stdin=subprocess.PIPE)
                for buf in pool.imap(AS._job, range(s['f0'], s['f1']), chunksize=8):
                    proc.stdin.write(buf)
                proc.stdin.close()
                if proc.wait() != 0 or not os.path.getsize(tmp):
                    raise SystemExit(f'encode failed: {cut} {s["sec"]} {s["code"]}')
                os.replace(tmp, p)
                log(f'  {s["sec"]:>4} {s["code"]:<9} {s["f0"]:>5}-{s["f1"]:<5} {time.time() - t0:5.0f}s')
    return segs, dict(segments=len(shots), encoded=len(todo), frames_encoded=nf, seconds=round(time.time() - t0, 1),
                      finish=fin, finish_backlog_segments=len(backlog), finish_backlog_frames=sum(backlog))


def join_and_mux(segs, audio, out_path, profile, cut):
    """Byte-join the segments (Annex B, no B-frames) into one stream and mux it with the sound, frame-exact."""
    dur = f'{EDL.TOTAL[cut] / FPS:.6f}'
    acodec = ['-c:a', 'pcm_s24le'] if out_path.endswith('.mov') else ['-c:a', 'aac', '-b:a', '160k']
    tmp = out_path + '.part' + os.path.splitext(out_path)[1]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'h264', '-framerate', str(FPS), '-i', '-', '-i', audio,
           '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-bsf:v', VUI] + acodec + [
           '-af', 'apad', '-t', dur, '-movflags', '+faststart+write_colr', tmp]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for p in segs:
        with open(p, 'rb') as fh:
            shutil.copyfileobj(fh, proc.stdin, 1 << 20)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f'mux failed: {out_path}')
    os.replace(tmp, out_path)


def screener(mov, mp4):
    """The same picture with AAC 320k (AudioToolbox, else ffmpeg's aac)."""
    for enc in (['-c:a', 'aac_at', '-b:a', '320k'], ['-c:a', 'aac', '-b:a', '320k']):
        tmp = mp4 + '.part.mp4'
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mov, '-map', '0:v', '-map', '0:a',
                            '-c:v', 'copy'] + enc + ['-movflags', '+faststart+write_colr', tmp])
        if r.returncode == 0:
            os.replace(tmp, mp4)
            return enc[1]
    raise SystemExit(f'screener failed: {mp4}')


def prune(profile, keep):
    """Record this film's segments; delete segments no film uses any more (never ones younger than 2 h, which
    a concurrent build may be about to use)."""
    import fcntl
    cache = os.path.join(DELIVERY, 'cache', profile)
    man = os.path.join(cache, 'manifests.json')
    with open(os.path.join(cache, '.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        m = json.load(open(man)) if os.path.exists(man) else {}
        m.update(keep)
        live = {os.path.basename(p) for v in m.values() for p in v}
        gone, now = 0, time.time()
        for n in os.listdir(cache):
            p = os.path.join(cache, n)
            if n.endswith('.h264') and n not in live and now - os.path.getmtime(p) > 7200:
                os.remove(p)
                gone += 1
        json.dump(m, open(man + '.part', 'w'), indent=0)
        os.replace(man + '.part', man)
    return gone


# ------------------------------------------------------------------------------------------------------ QC
def ff_info(path):
    err = subprocess.run(['ffmpeg', '-hide_banner', '-i', path], capture_output=True, text=True).stderr
    v = re.search(r'Video: (.+)', err)
    a = re.search(r'Audio: (.+)', err)
    return (v.group(1).strip() if v else ''), (a.group(1).strip() if a else '')


def count_video_frames(path):
    err = subprocess.run(['ffmpeg', '-hide_banner', '-i', path, '-map', '0:v', '-c', 'copy', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    m = re.findall(r'frame=\s*(\d+)', err)
    return int(m[-1]) if m else -1


def decode_audio(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:a', '-f', 'f32le', '-ac', '2', '-ar', str(SR),
                          '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def true_peak_db(x):
    from scipy.signal import resample_poly
    peak, n, hop = 0.0, len(x), SR * 10
    for a in range(0, n, hop):
        seg = x[max(0, a - 64):min(n, a + hop + 64)]
        peak = max(peak, float(np.abs(resample_poly(seg, 4, 1, axis=0)).max()))
    return 20 * np.log10(max(peak, 1e-9))


def picture_stats(path, n):
    """Per frame, at 240x100: sRGB luma p99.5 (black) and linear relative luminance on a 24x10 grid (flashes)."""
    w, h = 240, 100
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-map', '0:v',
           '-vf', f'scale={w}:{h}:flags=area+accurate_rnd+full_chroma_int:in_color_matrix=bt709:in_range=tv,'
                  'format=rgb24',                   # default flags read limited range ~2/255 dark (measured)
           '-f', 'rawvideo', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    lut = (((np.arange(256) / 255.0) + 0.055) / 1.055) ** 2.4
    lut[:11] = np.arange(11) / 255.0 / 12.92
    p995, grid, k = np.zeros(n, np.float32), np.zeros((n, 240), np.float32), 0
    while True:
        buf = proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        if k < n:
            luma = (img.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255.0
            p995[k] = np.percentile(luma, 99.5)
            lin = lut[img] @ np.array([0.2126, 0.7152, 0.0722])
            grid[k] = lin.reshape(10, 10, 24, 10).mean(axis=(1, 3)).reshape(-1)
        k += 1
    proc.wait()
    return p995[:k], grid[:k], k


def flashes(grid):
    """ITU-R BT.1702-style: a transition is a monotonic run of relative luminance of >= 0.1 with the darker
    state below 0.8, over >= 25% of the screen (cells concluding same-sign runs within one frame of each
    other); a flash is a pair of opposing transitions. Returns (max flashes in any 1 s, first worst frame)."""
    n = len(grid)
    if n < 3:
        return 0, None
    up = np.zeros(n)
    dn = np.zeros(n)
    for c in range(grid.shape[1]):
        L = grid[:, c].astype(float).tolist()
        d = np.diff(grid[:, c])
        s = np.sign(np.where(np.abs(d) < 0.002, 0, d)).tolist()
        start, cur = 0, 0
        for t in range(1, n):
            st = s[t - 1]
            if st != 0 and st != cur:
                if cur != 0:
                    amp = L[t - 1] - L[start]
                    if abs(amp) >= 0.1 and min(L[t - 1], L[start]) < 0.8:
                        (up if amp > 0 else dn)[t - 1] += 1
                start, cur = t - 1, st
        amp = L[-1] - L[start]
        if cur != 0 and abs(amp) >= 0.1 and min(L[-1], L[start]) < 0.8:
            (up if amp > 0 else dn)[n - 1] += 1
    cells = grid.shape[1]
    k = np.ones(3)
    U = np.convolve(up, k, 'same') / cells >= 0.25
    D = np.convolve(dn, k, 'same') / cells >= 0.25
    events, last = [], 0
    for t in range(n):
        sg = 1 if U[t] and not D[t] else -1 if D[t] and not U[t] else 0
        if sg and sg != last and (not events or t - events[-1][0] > 1):
            events.append((t, sg))
            last = sg
    ts = np.array([t for t, _ in events])
    worst, at = 0, None
    for i, t in enumerate(ts):
        cnt = int(np.sum((ts >= t) & (ts < t + FPS)))
        if cnt // 2 > worst:
            worst, at = cnt // 2, int(t)
    return worst, at


def status_frames(cut, variant):
    shots = EDL.EDL[cut]
    plans = [AS.plan_shot(s, cut, variant) for s in shots]
    slate, proxy, planned_black = 0, 0, []
    missing, proxies = [], []
    for s, pl in zip(shots, plans):
        n = s['f1'] - s['f0']
        if pl['kind'] in ('black', 'x2') or 'BLACK' in s['name']:
            planned_black.append((s['f0'], s['f1']))
        gap = n - pl['have'] if pl['kind'] == 'take' else 0
        if pl['kind'] in ('x2', 'titlesky') or (s['kind'] == 'title' and gap):
            k = n if pl['kind'] != 'take' else gap
            proxy += k
            proxies.append(f"{s['sec']} {s['code']}" + (f' ({k} f)' if k < n else ''))
        elif pl['kind'] == 'slate':
            slate += n
            missing.append(s['sec'] + ' ' + s['code'])
        elif gap:
            slate += gap
            missing.append(f"{s['sec']} {s['code']} ({gap} f)")
    if cut == 'A':
        planned_black.append((6456, 6480))                    # the fade to black from bar 81 b3.8
    # REVIEW (29 Sep): a floor window eases the film base in after a true black (or out at C's end), so the black runs
    # into it; count its frames as planned and merge touching intervals, so one run across both reads as planned
    planned_black += [(t['f0'], t['f1']) for t in EDL.TRANS.get(cut, ()) if t.get('kind') == 'floor']
    # joinsA (29 Sep) made A's 80-127 a dissolve from A1's true black instead of a floor: a dissolve that starts where
    # a planned black ends is a fade-up, and its first frames are black by design (A 80-85 measured under the threshold)
    black_ends = {b for _, b in planned_black}
    planned_black += [(t['f0'], t['f1']) for t in EDL.TRANS.get(cut, ())
                      if t.get('kind') == 'dissolve' and t.get('cut', t['f0']) in black_ends]
    merged = []
    for a, b in sorted(planned_black):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return slate, merged, missing, proxy, proxies


def provisional_frames(cut, variant):
    """Count present frames using declared provisional sources, once per frame; selection is unchanged."""
    count, sources = 0, []
    for shot in EDL.EDL[cut]:
        plan = AS.plan_shot(shot, cut, variant)
        if plan['kind'] != 'take':
            continue
        n, used = 0, set()
        for f in range(shot['f0'], shot['f1']):
            provisional = AS.provisional_sources(plan['take'], cut, variant, f)
            if provisional:
                n += 1
                used.update(provisional)
        if n:
            count += n
            sources.append(f"{shot['sec']} {shot['code']} ({n} f, {', '.join(sorted(used))})")
    return count, sources


def qc(cut, variant, mov, mp4, audio_label, build):
    total = EDL.TOTAL[cut]
    lines, res = [], {'file': mov, 'checks': []}
    worst = 'PASS'

    def check(level, what, msg):
        nonlocal worst
        lines.append(f'[{level}] {what}: {msg}')
        res['checks'].append(dict(level=level, what=what, msg=msg))
        if level == 'FAIL' or (level == 'WARN' and worst == 'PASS'):
            worst = level if worst != 'FAIL' else worst

    vinfo, ainfo = ff_info(mov)
    nv = count_video_frames(mov)
    check('PASS' if nv == total else 'FAIL', 'length, picture',
          f'{nv:,} frames = {nv / FPS:.3f} s (the bar map: {total:,} f = {total / FPS:.3f} s)')
    x = decode_audio(mov)
    check('PASS' if len(x) == total * SR // FPS else 'FAIL', 'length, sound',
          f'{len(x):,} samples at 48 kHz = {len(x) / SR:.3f} s (expected {total * SR // FPS:,})')
    fmt_ok = all(k in vinfo for k in ('h264', 'yuv420p', '1920x804', '24 fps')) and 'bt709' in vinfo
    check('PASS' if fmt_ok else 'FAIL', 'format, picture', vinfo[:110])
    afmt_ok = ('48000 Hz' in ainfo and 'stereo' in ainfo) or 'mono' in ainfo
    check('PASS' if afmt_ok else 'FAIL', 'format, sound', ainfo[:90])

    p995, grid, nd = picture_stats(mov, total)
    black = p995 < 0.03
    slate, planned, missing, proxy, proxies = status_frames(cut, variant)
    provisional, provisional_sources = provisional_frames(cut, variant)
    runs, t = [], 0
    while t < nd:
        if black[t]:
            u = t
            while u < nd and black[u]:
                u += 1
            runs.append((t, u))
            t = u
        else:
            t += 1
    unexpected = [(a, b) for a, b in runs if not any(pa <= a and b <= pb for pa, pb in planned)]
    shot_at = lambda f: next(s for s in EDL.EDL[cut] if s['f0'] <= f < s['f1'])
    if unexpected:
        check('WARN', 'black frames', f'{len(runs)} black runs; not planned: ' + ', '.join(
            f"{a}-{b - 1} ({shot_at(a)['sec']} {shot_at(a)['name']})" for a, b in unexpected[:8]))
    else:
        check('PASS', 'black frames', (f'{len(runs)} black runs, all planned: ' + ', '.join(
            f"{a}-{b - 1} ({shot_at(a)['sec']})" for a, b in runs[:8])) if runs else 'none')
    fl, at = flashes(grid)
    if fl > 3:
        check('FAIL', 'flashes', f'{fl} flashes in one second from f {at} ({shot_at(at)["sec"]}); limit 3')
    elif fl == 3:
        check('WARN', 'flashes', f'3 flashes in one second from f {at} ({shot_at(at)["sec"]}); limit 3')
    else:
        check('PASS', 'flashes', f'at most {fl} per second (limit 3; >= 25% of the screen, >= 0.1 relative '
                                 'luminance, darker state < 0.8)')

    pk = float(np.abs(x).max()) if len(x) else 0.0
    clips = int(np.sum(np.abs(x) >= 0.99997))
    tp = true_peak_db(x) if len(x) else -120.0
    check('PASS' if tp <= -1.0 and clips == 0 else 'FAIL', 'sound peaks',
          f'sample {20 * np.log10(max(pk, 1e-9)):.2f} dBFS, true peak {tp:.2f} dBTP (limit -1.0), '
          f'{clips} clipped samples')
    click = 'CLICK' in audio_label.upper()
    try:
        import pyloudnorm
        lufs = pyloudnorm.Meter(SR).integrated_loudness(x.astype(np.float64))
    except Exception:
        lufs = float('nan')
    if click:
        check('WARN', 'loudness', f'{lufs:.1f} LUFS: the sync click track, not a sound master')
    else:
        check('PASS' if abs(lufs + 16) <= 1.0 else 'WARN', 'loudness', f'{lufs:.1f} LUFS integrated (target -16 ± 1)')
    if mp4 and os.path.exists(mp4):
        y = decode_audio(mp4)
        tq = true_peak_db(y) if len(y) else -120.0
        check('PASS' if tq <= -0.5 else 'WARN', 'screener sound (AAC)', f'true peak {tq:.2f} dBTP after encoding')
    check('INFO', 'sound source', audio_label)
    check('INFO' if slate == 0 else 'WARN', 'coverage',
          'every frame rendered' if slate == 0 else f'{slate:,} of {total:,} frames are slates: ' + ', '.join(missing))
    if proxy:
        check('WARN', 'EDIT proxies', f'{proxy:,} frames are EDIT stand-ins until their plates land: ' +
              ', '.join(proxies))
    if provisional:
        check('WARN', 'provisional picture sources', f'{provisional:,} frames use provisional sources: ' +
              ', '.join(provisional_sources))
    check('INFO', 'build', f"{build['encoded']} of {build['segments']} segments encoded "
                           f"({build['frames_encoded']:,} f) in {build['seconds']:.0f} s")
    if build.get('finish'):                                   # FINISH: the film finish and what is still unfinished
        nb = build.get('finish_backlog_segments', 0)
        check('WARN' if nb else 'INFO', 'finish',
              f"{build['finish']}" + (f"; {nb} segments ({build.get('finish_backlog_frames', 0):,} f) still "
                                      f"unfinished (the budgeted backlog; FINISH_ALL=1 clears it)" if nb else
                                      '; every rendered frame finished'))
    # Picture-source completeness only: audio/finish warnings and creative approval are separate decisions.
    complete = slate == 0 and proxy == 0 and provisional == 0
    head = (f"THE LONG DAWN v3 · {cut} · {FILM[cut]}{' · ALT (coded towers)' if variant else ''} · master QC · "
            f"{time.strftime('%d %b %H:%MZ', time.gmtime())}\n{mov} ({os.path.getsize(mov) / 1e6:.1f} MB)"
            f"{'  +  ' + os.path.basename(mp4) if mp4 else ''}\nRESULT: {worst}"
            f"{'' if complete else '  (picture sources incomplete: slates, EDIT proxies or provisional sources remain)'}\n")
    res.update(result=worst, complete=complete, slate_frames=slate, proxy_frames=proxy,
               provisional_frames=provisional, provisional_sources=provisional_sources)
    return head + '\n'.join(lines) + '\n', res


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cut', required=True, type=str.upper, choices=['A', 'B', 'C'])
    ap.add_argument('--variant', default=None, choices=[None, 'codedtowers'])
    ap.add_argument('--profile', default='master', choices=list(PROFILES))
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--qc-only', action='store_true')
    ap.add_argument('--no-screener', action='store_true')
    a = ap.parse_args()
    EDL.check(os.path.join(ROOT, 'music', 'v3'))
    prof = PROFILES[a.profile]
    os.makedirs(prof['out'], exist_ok=True)
    name = name_of(a.cut, a.variant, a.profile)
    log = lambda m: print(m, flush=True)
    ext = '.mov' if a.profile == 'master' else '.mp4'
    out = os.path.join(prof['out'], name + ext)
    audio, label = AS.resolve_audio(a.cut)
    build = dict(segments=0, encoded=0, frames_encoded=0, seconds=0)
    if not a.qc_only:
        segs, build = build_video(a.cut, a.variant, a.profile, a.workers, log)
        snap = AS.snapshot_audio(audio, a.cut) if not os.path.basename(audio).startswith('click_') else audio
        try:
            join_and_mux(segs, snap, out, a.profile, a.cut)
        finally:
            if snap != audio and os.path.exists(snap):
                os.remove(snap)
        gone = prune(a.profile, {name: segs})
        log(f'wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) [{label}]; pruned {gone} old segments')
    if a.profile != 'master':
        return
    mp4 = None if a.no_screener else os.path.join(prof['out'], name + '.mp4')
    if mp4 and not a.qc_only:
        enc = screener(out, mp4)
        log(f'wrote {mp4} (AAC 320k, {enc})')
    txt, res = qc(a.cut, a.variant, out, mp4, label, build)
    open(os.path.join(prof['out'], name + '_QC.txt'), 'w').write(txt)
    json.dump(res, open(os.path.join(prof['out'], name + '_QC.json'), 'w'), indent=1)
    log(txt)


if __name__ == '__main__':
    main()
