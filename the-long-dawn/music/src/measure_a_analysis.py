"""A 3600-6479: the second half's picture events, measured -> music/v3/events_A_measured.json (29 Sep 2026, night).

    cd the-long-dawn
    python3 music/src/measure_a_analysis.py            # write v3/events_A_measured.json
    python3 music/src/measure_a_analysis.py --check    # exit 1 if the committed file is not what the evidence gives

Input: the batches measure_a_points.py wrote (one finished frame at a time through EDIT, 960x402, before captions;
<report dir>/asound-evidence/claude/points_<label>_<a>_<b>.json) and the source-projection receipts of the Codex lane
(ridge_geometry.json, crossing_geometry.json: WHERE a fire is, never WHEN; see their own limitations). This module
opens no frame and no recording: it decides every frame from the recorded pixel statistics, and every frame it decides
was also looked at on stills (asound-evidence/claude/crops_*.png, sheet_*.jpg), which is the OBSERVED basis.

The schema is events_C5_measured.json's (id, shot, stem, event, frames, method, region, evidence, confidence, basis,
negative_controls, observed), with the fields C5 defined for a catch: first = the first frame its flame is visible,
fastest = its largest one-frame growth; a sound's burst arrives on first..fastest.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUSIC = HERE.parent
ROOT = MUSIC.parent
OUT = MUSIC / 'v3' / 'events_A_measured.json'
EVID = ROOT.parents[2] / 'outputs' / 'claude-owner-night' / 'codex' / 'asound-evidence'
POINTS = EVID / 'claude'
FPS = 24
HALF = '960x402 finished picture before captions (AS._CTX.picture at half size)'

# ---------------------------------------------------------------------------------------------- thresholds
# EVERY RIDGE. A far fire is a point: its luma top-hat (9x9) at the projected spot. Before any ignition the spots
# read a median 11, 90th percentile 32 (2,844 on-screen ROI-frames, source ignition >= 3 frames away); a fire that
# catches jumps by 20-140 in one or two frames and stays. JUMP is the catch; HOLD keeps it; WARM (the chroma top-hat
# R-(G+B)/2 at the peak) rejects snow glints, which are bluer than the night around them.
EDGE = 6                   # px: an ROI this close to the frame edge is entering, not catching
JUMP, HOLD, HOLD_N, WARM = 15, 10, 5, 8
IGN_WINDOW = (-2, 4)       # frames around the source's ignition in which the pixels are searched for the catch
PRESENT = (20, 8)          # luma top-hat and chroma top-hat of a point that is a lit fire


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def batches(label):
    """{frame: row} from every points_<label>_*.json; frames must not repeat; receipts for the file"""
    rows, receipts = {}, []
    paths = sorted(POINTS.glob(f'points_{label}_[0-9]*.json'))
    if not paths:
        raise SystemExit(f'no evidence batches points_{label}_*.json in {POINTS}')
    for p in paths:
        d = json.loads(p.read_text())
        for r in d['rows']:
            if r['f'] in rows:
                raise SystemExit(f'{p.name}: frame {r["f"]} measured twice')
            rows[r['f']] = r
        receipts.append(dict(file=f'asound-evidence/claude/{p.name}', sha256=sha(p), first=d['first'], last=d['last'],
                             frames_examined=d['frames_examined'], frame_set_sha256=d['frame_set_sha256'],
                             take_override=d.get('take_override'), edit=portable(d.get('edit')),
                             frames_skipped_not_landed=d.get('frames_skipped_not_landed', [])))
    return rows, receipts


def portable(edit):
    """the batch's edit receipt with its worktree root recorded as '.' and paths under it made relative: the output
    is committed to a public repo, and ending_events() resolves '.' against this checkout's ROOT"""
    if not edit or not edit.get('root') or not Path(edit['root']).is_absolute():
        return edit
    root = edit['root'].rstrip('/')
    e = {k: (v[len(root) + 1:] if isinstance(v, str) and v.startswith(root + '/') else v) for k, v in edit.items()}
    e['root'] = '.'
    return e


def contiguous(rows, a, b, what):
    miss = [f for f in range(a, b + 1) if f not in rows]
    if miss:
        raise SystemExit(f'{what}: {len(miss)} frames of {a}-{b} not measured (first {miss[:5]})')


def shot_receipt(rows, receipts, a, b, stem):
    return dict(stem=stem, first=a, last=b, frames_examined=sum(1 for f in rows if a <= f <= b),
                sources=sorted({rows[f]['source'].rsplit('/', 1)[0] for f in rows if a <= f <= b}), receipts=receipts)


def pan_of(x):
    return round(0.8 * (2 * x / 960 - 1), 3)


# ---------------------------------------------------------------------------------------------- EVERY RIDGE
def ridge_series(rows, k):
    s = {}
    for f in sorted(rows):
        x = next((x for x in rows[f]['rois'] if x['id'] == f'ridge.{k}'), None)
        if x is None or x.get('off_frame'):
            continue
        if EDGE <= x['x'] < 960 - EDGE and EDGE <= x['y'] < 402 - EDGE:
            s[f] = x
    return s


def catch_in(s, frames, key='pk', ck='ck_at'):
    """the first f in frames with a catch: a jump >= JUMP over the median of f-8..f-2, held >= HOLD for HOLD_N more
    frames, warm at f+2. -> (f, base) or None"""
    for f in frames:
        pre = [s[g][key] for g in range(f - 8, f - 1) if g in s]
        if len(pre) < 5 or any(g not in s for g in range(f, f + HOLD_N + 1)):
            continue
        base = st.median(pre)
        if s[f][key] - base < JUMP or any(s[g][key] - base < HOLD for g in range(f, f + HOLD_N + 1)):
            continue
        if ck and s[f + 2].get(ck, 0) < WARM:
            continue
        return f, base
    return None


def ridge_events():
    rows, rec = batches('ridges')
    contiguous(rows, 3660, 3799, 'ridges')
    geo_p = EVID / 'ridge_geometry.json'
    geo = json.loads(geo_p.read_text())
    ign = {f['source_index']: f['source_ignition'] for f in geo['frames'][0]['fires']}
    events, catches, visible, hidden = [], [], [], []
    ctl_temporal = dict(windows=0, frames=0, triggers=[])
    ctl_spatial = dict(windows=0, frames=0, triggers=[])
    for k in sorted(ign, key=lambda k: (ign[k], k)):
        s = ridge_series(rows, k)
        on = sorted(s)
        if not on:
            hidden.append(dict(source_index=k, reason='never inside the frame (6 px margin)'))
            continue
        i0 = math.floor(ign[k])
        window = [f for f in range(i0 + IGN_WINDOW[0], i0 + IGN_WINDOW[1] + 1) if f in s]
        c = catch_in(s, window)
        # negative controls, same detector: 20-26 frames before the ignition at the same spot (no fire drawn there
        # yet: the ignition envelope starts 1.5 frames before it), and 12 px either side inside the ignition window
        tw = [f for f in range(i0 - 26, i0 - 19) if f in s]
        if tw:
            ctl_temporal['windows'] += 1
            ctl_temporal['frames'] += len(tw)
            hit = catch_in(s, tw)
            if hit:
                ctl_temporal['triggers'].append([k, hit[0]])
        for side in ('ctl_l', 'ctl_r'):
            cs = {f: {'pk': x[side], 'ck_at': x.get(side + '_ck') or 0} for f, x in s.items()
                  if x.get(side) is not None and EDGE + 12 <= x['x'] < 960 - EDGE - 12}
            cw = [f for f in window if f in cs]
            if cw:
                ctl_spatial['windows'] += 1
                ctl_spatial['frames'] += len(cw)
                hit = catch_in(cs, cw, ck=None)
                if hit:
                    ctl_spatial['triggers'].append([k, side, hit[0]])
        # sustained presence (any fire, caught or not): the first frame from which it is present in >= 90% of its
        # remaining on-screen frames and in 5 of the next 6
        pres = lambda f: s[f]['pk'] >= PRESENT[0] and s[f]['ck_at'] >= PRESENT[1]
        first_vis = None
        for f in on:
            if not pres(f):
                continue
            rest = [g for g in on if g >= f]
            if sum(pres(g) for g in rest) >= 0.9 * len(rest) and sum(pres(g) for g in rest[:6]) >= 5:
                first_vis = f
                break
        if c is None and first_vis is None:
            hidden.append(dict(source_index=k, reason='no sustained point at its projected spot (behind terrain, '
                               'under her fire\'s glow, or below the top-hat floor)', on_screen=[on[0], on[-1]]))
            continue
        if c is not None:
            f1, base = c
            fast = max(range(f1, f1 + 5), key=lambda g: s[g]['pk'] - s[g - 1]['pk'] if g - 1 in s else -999)
            ref = min(f1 + 6, on[-1])
            kind = 'catch'
        else:
            f1, base, fast = first_vis, None, None
            ref = min(f1 + 6, on[-1])
            kind = 'enters_lit' if f1 - on[0] <= 2 and on[0] > 3660 else 'uncovered_lit'
        x = s[ref]
        after = [s[g]['pk'] for g in on if g >= f1]
        ev = dict(id=f'ridges.fire_{k}', shot='ridges', stem='reveal_A',
                  event=('a far fire catches' if kind == 'catch' else
                         'a far fire, already burning, comes into view' if kind == 'enters_lit' else
                         'a far fire, already burning, is uncovered as the crane rises'),
                  kind=kind,
                  frames=dict(first=f1, fastest=fast, catch=f1 if kind == 'catch' else None,
                              lit_through=on[-1], on_screen=[on[0], on[-1]]),
                  method=('luma top-hat (9x9) at the source-projected hot point, r 3 px, 960x402: catch = a jump >= '
                          f'{JUMP} over the median of the 7 frames 8..2 before, held >= {HOLD} for {HOLD_N} frames, '
                          f'chroma top-hat >= {WARM} two frames on, searched {IGN_WINDOW[0]}..+{IGN_WINDOW[1]} frames '
                          'around the source ignition (the source says WHERE and roughly WHEN to look; the pixels '
                          'decide the frame); fastest = its largest one-frame rise in the next 4 frames' if kind == 'catch'
                          else 'first frame from which the point (luma top-hat >= 20, chroma top-hat >= 8 at the '
                          'projected spot) is present in >= 90% of its remaining on-screen frames; no catch-like jump '
                          'in its ignition window'),
                  region=dict(x=x['px'], y=x['py'], space=HALF + f', the peak pixel {ref - f1} frames after first'),
                  size=dict(tophat=x['pk'], area_tophat8_px=x.get('area8'), rgb=x['rgb'],
                            tophat_median_after=st.median(after), frame=ref),
                  evidence=dict(base=base, source_ignition=round(ign[k], 3),
                                source_ignition_minus_first=round(ign[k] - f1, 2),
                                tophat={str(g): s[g]['pk'] for g in range(f1 - 3, f1 + 7) if g in s},
                                chroma_tophat={str(g): s[g]['ck_at'] for g in range(f1 - 3, f1 + 7) if g in s}),
                  confidence='high' if kind == 'catch' and abs(ign[k] - f1) <= 1.5 else 'medium',
                  basis=['MEASURED'], pan=pan_of(x['px']))
        events.append(ev)
        (catches if kind == 'catch' else visible).append(ev['id'])
    # her fire, burning through the shot: the largest warm component with a bright core, every frame
    her = []
    for f in range(3660, 3800):
        comps = [c for c in rows[f]['warm'] if 380 <= c[5] <= 620]
        big = max(comps, key=lambda c: c[0]) if comps else None
        her.append((f, big))
    miss = [f for f, b in her if b is None or b[0] < 60]
    events.append(dict(id='ridges.her_fire', shot='ridges', stem='reveal_A', kind='already_burning',
                       event="her fire, already burning, pulled away from as the crane rises",
                       frames=dict(first=3660, last=3799, catch=None),
                       method='largest warm component (R>=100, R-G>=15, R-B>=35; 960x402) with centroid x 380-620',
                       region=dict(x=her[0][1][5], y=her[0][1][6], space=HALF + ' at 3660'),
                       size=dict(area_px={str(f): b[0] for f, b in her[::20]}),
                       evidence=dict(frames_under_60px=miss), confidence='high', basis=['MEASURED', 'OBSERVED'],
                       note='the roar (3600) is SOUND-C\'s; this fire has no catch in 3660-3799'))
    summary = dict(id='ridges.catches', shot='ridges', stem='reveal_A', kind='summary',
                   event='fires catch on the ridges: every measured catch, in order',
                   frames=dict(first=min(e['frames']['first'] for e in events if e.get('kind') == 'catch'),
                               last=max(e['frames']['first'] for e in events if e.get('kind') == 'catch'),
                               catches=sorted(e['frames']['first'] for e in events if e.get('kind') == 'catch')),
                   method='the catches above', evidence=dict(catch_ids=catches, lit_but_no_catch=visible,
                                                             not_visible=hidden),
                   negative_controls=[
                       dict(detector='catch', kind='temporal', selection_reason='the same spot 20-26 frames before '
                            'its source ignition (nothing drawn there yet)', windows=ctl_temporal['windows'],
                            frames_examined=ctl_temporal['frames'], triggers=len(ctl_temporal['triggers']),
                            trigger_frames=ctl_temporal['triggers']),
                       dict(detector='catch (without the chroma test)', kind='spatial',
                            selection_reason='12 px left and right of each spot, inside its ignition window',
                            windows=ctl_spatial['windows'], frames_examined=ctl_spatial['frames'],
                            triggers=len(ctl_spatial['triggers']), trigger_frames=ctl_spatial['triggers'])],
                   basis=['MEASURED', 'OBSERVED'], confidence='high',
                   observed=dict(sheet='asound-evidence/claude/crops_ridge_catches_1.png',
                                 saw='ridge.34: faint glow 3686, full flash 3687; ridge.7 3695/3696; ridge.13 '
                                     '3689/3690; ridge.0 3711/3712; ridge.21 3718-3719 faint, 3720 full; ridge.12 and '
                                     'ridge.10 rise from behind terrain at 3717-3719 and 3712-3715 (no catch); '
                                     'ridge.47 faint at 3726; ridge.29 flashes at 3729 inside the red under-glow band'))
    return [summary] + events, shot_receipt(rows, rec, 3660, 3799, 'reveal_A') | dict(
        geometry=dict(file='asound-evidence/ridge_geometry.json', sha256=sha(geo_p), note=geo['kind']))


# ---------------------------------------------------------------------------------------------- KARST, DESERT
def insert_events():
    rows, rec = batches('inserts')
    contiguous(rows, 3800, 3919, 'inserts')
    karst = {f: sum(c[0] for c in rows[f]['warm']) for f in range(3800, 3860)}
    k1 = next(f for f in range(3808, 3860) if karst[f] >= 100 and karst[f] >= 10 * max(1, st.median(
        [karst[g] for g in range(f - 8, f - 1)])))
    kfast = max(range(k1, k1 + 10), key=lambda g: karst[g] - karst[g - 1])
    box = (540, 150, 720, 260)
    dim = {f: sum(c[0] for c in rows[f]['dim'] if box[0] <= c[5] <= box[2] and box[1] <= c[6] <= box[3])
           for f in range(3860, 3920)}
    d1 = next(f for f in range(3868, 3920) if dim[f] >= 100 and dim[f] >= 5 * max(1, st.median(
        [dim[g] for g in range(f - 8, f - 1)])))
    dfast = max(range(d1, d1 + 10), key=lambda g: dim[g] - dim[g - 1])
    ev = [dict(id='karst.flare', shot='karst', stem='montage3d_v3/karst_slow', kind='catch',
               event='the karst beacon flares and floods the rock with light',
               frames=dict(first=k1, fastest=kfast, catch=k1),
               method='whole-frame warm area (R>=100, R-G>=15, R-B>=35; 960x402): first frame >= 100 px and >= 10x '
                      'the median of the 7 frames 8..2 before; fastest = largest one-frame rise in the next 10',
               region=dict(x=round(next(c for c in sorted(rows[k1 + 6]['warm'], key=lambda c: -c[0]))[5]),
                           y=round(next(c for c in sorted(rows[k1 + 6]['warm'], key=lambda c: -c[0]))[6]),
                           space=HALF + ', largest warm component 6 frames after first'),
               evidence=dict(warm_area={str(f): karst[f] for f in range(k1 - 3, k1 + 8)}),
               confidence='high', basis=['MEASURED', 'OBSERVED'],
               observed=dict(sheet='asound-evidence/claude/crops_inserts.png',
                             saw='3814 a small ember-lit figure; 3815 the whole rock floods warm'),
               sync='A.karst_flare (barmap karst_flare 3815)'),
          dict(id='desert.catch', shot='desert', stem='montage3d_v3/desert', kind='catch',
               event="the robed figure's stone beacon catches; its light runs out over the sand",
               frames=dict(first=d1, fastest=dfast, catch=d1),
               method='dim-warm area (R>=60, R-G>=10, R-B>=20; 960x402) of components centred in x 540-720, '
                      'y 150-260: first frame >= 100 px and >= 5x the median of the 7 frames 8..2 before; fastest = '
                      'largest one-frame rise in the next 10',
               region=dict(x=625, y=198, space=HALF + ', the lit sand strip (dim component centroid 3881-3886)'),
               evidence=dict(dim_area={str(f): dim[f] for f in range(d1 - 3, d1 + 8)}),
               confidence='high', basis=['MEASURED', 'OBSERVED'],
               observed=dict(sheet='asound-evidence/claude/crops_inserts.png',
                             saw='3879 grey sand; 3880 a faint warm strip on the sand; brighter each frame to 3888'),
               sync='A.desert_fire (barmap desert_fire 3880)')]
    return ev, dict(karst=shot_receipt(rows, rec, 3800, 3859, 'montage3d_v3/karst_slow'),
                    desert=shot_receipt(rows, rec, 3860, 3919, 'montage3d_v3/desert'))


# ---------------------------------------------------------------------------------------------- CROSSING
def wf1_component(warm, w):
    """the largest warm component that holds watch-fire 1's projected flame axis (base..top, +-3 px)"""
    cx = w['center'][0]
    lo, hi = min(w['top'][1], w['base'][1]), max(w['top'][1], w['base'][1])
    best = None
    for comp in warm:
        a, x0, y0, x1, y1, _, _ = comp
        if x0 - 3 <= cx <= x1 + 3 and y0 <= hi + 3 and y1 >= lo - 3 and (best is None or a > best[0]):
            best = comp
    return best


def far_component(warm, fires):
    """watch-fires 2-4 (they project within 11 px of each other at 4880): the largest warm component of <= 400 px
    centred within 8 px of any of their projections"""
    pts = [w['center'][:2] for w in fires[1:]]
    best = None
    for comp in warm:
        a, x0, y0, x1, y1, x, y = comp
        if a <= 400 and any(math.hypot(x - p[0], y - p[1]) <= 8 for p in pts) and (best is None or a > best[0]):
            best = comp
    return best


def cap_vs_decal(rows, geo, dv):
    """A18's staged first take, cand_crossing_both_decal_cap (owner bea534d; plays once all 960 frames land), measured
    through the same EDIT (points_crossing-cap_*: measure_a_points.py --take), against the measured both_decal frame
    by frame: the two sizes every A18 row of the table reads. Once the cap is complete every frame the rows are built
    from is derived from it the same way (crossing_derived) and compared; before that, frames the cap has not landed
    decide nothing, so the derived frames that need them are listed, not assumed."""
    far0, far_last = dv['far0'], dv['far_last']
    if not sorted(POINTS.glob('points_crossing-cap_[0-9]*.json')):
        return dict(measured=False, note='no cap frames measured')
    cap, rec = batches('crossing-cap')
    common = sorted(f for f in cap if f in rows)
    complete = all(f in cap for f in range(4880, 5840))

    def cmp(fn, frames):
        d = {f: ((fn(rows[f]) or [0])[0], (fn(cap[f]) or [0])[0]) for f in frames}
        diff = {f: abs(b - a) for f, (a, b) in d.items()}
        db = {f: abs(10 * math.log10(b / a)) for f, (a, b) in d.items() if a > 0 and b > 0}
        worst = max(diff, key=diff.get) if diff else None
        return dict(frames=len(d), identical=sum(1 for v in diff.values() if v == 0),
                    max_abs_diff_px=diff[worst] if worst else None, at=worst,
                    area_px_there=list(d[worst]) if worst else None,
                    max_abs_db=round(max(db.values()), 2) if db else None,
                    presence_differs=[f for f, (a, b) in d.items() if (a > 0) != (b > 0)]), d
    w1, _ = cmp(lambda r: wf1_component(r['warm'], geo[r['f']][0]), common)
    fr = [f for f in common if f <= far_last + 5]
    fa, fd = cmp(lambda r: far_component(r['warm'], geo[r['f']]), fr)
    # the far fire's last frame is where its area first stays under half its opening median for 3 frames running
    fa['half_threshold_side_differs'] = [f for f, (a, b) in fd.items() if (a < 0.5 * far0) != (b < 0.5 * far0)]
    out = dict(measured=True, receipts=rec, landed_measured=len(cap), compared=len(common), complete=complete,
               watchfire_1=w1, watchfire_far=fa)
    if not complete:
        need = [f for f in range(4880, far_last + 4) if f not in cap]
        out['undecided'] = dict(far_last_needs=f'{len(need)} cap frames of 4880-{far_last + 3} not landed'
                                               f'{" (first " + str(need[:4]) + ")" if need else ""}',
                                note='the recede frames need 9-frame windows and each feeding flare the 20 frames '
                                     'before it: re-measure the crossing on the cap once all 960 frames land')
        return out
    dc = crossing_derived(cap, geo)
    frames = dict(recede_6db=[dv['recede_6db'], dc['recede_6db']], recede_20db=[dv['recede_20db'], dc['recede_20db']],
                  feeds=[[x['first'] for x in dv['flares']], [x['first'] for x in dc['flares']]],
                  far_last=[dv['far_last'], dc['far_last']])
    env = max(abs(dv['rel'][f] - dc['rel'][f]) for f in range(4880, 5840, 10))
    out['derived'] = dict(both_decal_vs_cap=frames, every_frame_equal=all(a == b for a, b in frames.values()),
                          envelope_max_abs_db=round(env, 2),
                          far_to_wf1_db=[round(10 * math.log10(max(dv['far0'], 1) / dv['ref']), 1),
                                         round(10 * math.log10(max(dc['far0'], 1) / dc['ref']), 1)])
    return out


def crossing_derived(rows, geo):
    """every frame A18's rows are built from, from one take's measured rows: watch-fire 1's footprint and its recede
    and feeding frames, the far fire's opening size and last frame (one function, so a second take is read the same)"""
    area = {f: (wf1_component(rows[f]['warm'], geo[f][0]) or [0])[0] for f in range(4880, 5840)}
    ref = st.median([area[f] for f in range(4880, 4920)])       # the close shot's first 40 frames
    # the footprint relative to the close shot, smoothed over 9 frames (the flame's flicker is +-1 dB frame to frame)
    rel = {}
    for f in range(4880, 5840):
        win = [area[g] for g in range(max(4880, f - 4), min(5839, f + 4) + 1)]
        rel[f] = round(10 * math.log10(max(st.median(win), 1) / ref), 2)
    # feeding flares in the wides (crossing.wf_burn: fed every ~13 degrees of sky once it wheels): a rise of the
    # footprint by >= 4 dB over the median of the 20 frames before, where the footprint had settled (< -12 dB)
    flares = []
    for f in range(5300, 5840):
        prev = [area[g] for g in range(f - 20, f) if area[g] > 0]
        if not prev or area[f] <= 0:
            continue
        b = st.median(prev)
        if 10 * math.log10(b / ref) < -12 and 10 * math.log10(area[f] / b) >= 4 and (
                not flares or f - flares[-1]['first'] > 40):
            peak = max(range(f, min(5839, f + 20) + 1), key=lambda g: area[g])
            flares.append(dict(first=f, peak=peak, base_px=b, peak_px=area[peak],
                               rise_db=round(10 * math.log10(area[peak] / b), 2)))
    fa = {f: far_component(rows[f]['warm'], geo[f]) for f in range(4880, 5840)}
    far0 = st.median((fa[f] or [0])[0] for f in range(4880, 4920))
    # visible until the great lantern starts to cover it: the frame before its area first stays under half of its
    # 4880-4919 median for 3 frames running (crops_far_watchfire.png: under the lantern 5148-5160, gone 5166-5178)
    lf = next(f - 1 for f in range(4881, 5838) if all((fa[g] or [0])[0] < 0.5 * far0 for g in range(f, f + 3)))
    return dict(area=area, ref=ref, rel=rel, flares=flares, fa=fa, far0=far0, far_last=lf,
                recede_6db=next(f for f in range(4880, 5840) if rel[f] <= -6.0),
                recede_20db=next(f for f in range(4880, 5840) if rel[f] <= -20.0))


def crossing_events():
    rows, rec = batches('crossing')
    contiguous(rows, 4880, 5839, 'crossing')
    geo_p = EVID / 'crossing_geometry.json'
    geo = {fr['frame']: fr['watchfires'] for fr in json.loads(geo_p.read_text())['frames']}

    def wf1(f):
        return wf1_component(rows[f]['warm'], geo[f][0])
    dv = crossing_derived(rows, geo)
    area, ref, rel, flares = dv['area'], dv['ref'], dv['rel'], dv['flares']
    absent = [f for f in area if area[f] == 0]
    samples = {str(f): [area[f], rel[f]] for f in range(4880, 5840, 10)}
    samples['5839'] = [area[5839], rel[5839]]
    first_quiet = dv['recede_6db']
    ev = [dict(id='crossing.watchfire_1', shot='crossing', stem='cand_crossing_both_decal', kind='burning',
               event='watch-fire 1 burns large beside the standing line from the cut, then recedes as the camera '
                     'draws back; it never leaves the frame',
               frames=dict(first=4880, last=5839, catch=None, in_frame=[4880, 5839], leaves_frame=None,
                           recede_6db=first_quiet,
                           recede_20db=dv['recede_20db']),
               method='largest warm component (R>=100, R-G>=15, R-B>=35; 960x402) holding the source-projected '
                      'flame axis (crossing_geometry.json watch-fire 1, base..top +-3 px); relative size = 10 log10 '
                      'of its 9-frame median area over the median of 4880-4919 (area goes as 1/distance^2 on screen, '
                      'as sound power does in the air: the dB are comparable, not identical)',
               region=dict(x=round(wf1(4880)[5]), y=round(wf1(4880)[6]), space=HALF + ' at 4880'),
               size=dict(reference_area_px=ref, area_px_and_db=samples),
               evidence=dict(frames_without_component=absent, feeding_flares=flares),
               confidence='high', basis=['MEASURED', 'OBSERVED'],
               observed=dict(sheet='asound-evidence/claude/crops_watchfire_1.png', frames='4880, 5040, 5200, 5300, 5520',
                             saw='4880/5040 a 1.3 m flame fills the right third behind the hooded bearers; 5200 a '
                                 'small fire beside the great lantern; 5300/5520 a warm point right of the lantern '
                                 'group, the size of a hand lantern',
                             not_seen='the feeds (second row, 5420-5729): too small at this scale to tell by eye')),
          ]
    for fl in flares:
        ev.append(dict(id=f'crossing.watchfire_1.fed_{fl["first"]}', shot='crossing', stem='cand_crossing_both_decal',
                       kind='flare', event='watch-fire 1 is fed and flares (seen small, in the wide)',
                       frames=dict(first=fl['first'], fastest=None, full=fl['peak']),
                       method='watch-fire 1 footprint (above): a rise >= 4 dB over the median of the 20 frames before, '
                              'from a settled footprint (< -12 dB re the close shot)',
                       region=dict(x=round(wf1(fl['peak'])[5]), y=round(wf1(fl['peak'])[6]), space=HALF + ' at full'),
                       evidence=fl, confidence='medium', basis=['MEASURED'],
                       note='crossing.wf_burn feeds every watch-fire every ~13 degrees of sky; a hand lantern '
                            'passing in front would read the same way: the owner checks these by eye'))
    # watch-fires 2-4: one small fire beside a hooded keeper at the far end of the ridge, where all three project
    # within 11 px of each other at 4880; the great lantern then passes in front of it
    fa, far0, lf = dv['fa'], dv['far0'], dv['far_last']
    ev.append(dict(id='crossing.watchfire_far', shot='crossing', stem='cand_crossing_both_decal', kind='burning',
                   event='a far watch-fire (2, 3 or 4) burns small beside its keeper at the end of the ridge until '
                         'the great lantern passes in front of it',
                   frames=dict(first=4880, last=lf, catch=None, in_frame=[4880, lf]),
                   method='largest warm component (<= 400 px) centred within 8 px of any of watch-fires 2-4\'s '
                          'projections (crossing_geometry.json); the run from 4880 with gaps <= 3 frames',
                   region=dict(x=round(fa[4880][5]), y=round(fa[4880][6]), space=HALF + ' at 4880'),
                   size=dict(area_px={str(f): (fa[f] or [0])[0] for f in range(4880, lf + 1, 20)},
                             ratio_to_watchfire_1_db=round(10 * math.log10(
                                 st.median([(fa[f] or [1])[0] for f in range(4880, 4920)]) / ref), 1)),
                   evidence=dict(identity='watch-fires 2/3/4 project to (430,295), (437,303), (431,306) at 4880; the '
                                          'component sits at ({:.0f},{:.0f})'.format(fa[4880][5], fa[4880][6]),
                                 median_area_4880_4919_px=far0,
                                 area_px_around_last={str(f): (fa[f] or [0])[0] for f in range(lf - 3, lf + 6)}),
                   confidence='medium', basis=['MEASURED', 'OBSERVED'],
                   observed=dict(sheet='asound-evidence/claude/crops_watchfires_2to4.png',
                                 saw='4880-5040 one small fire on a rock beside a standing hooded keeper; 5120 the '
                                     'great lantern enters the crop above it; 5148-5160 the fire under the lantern; '
                                     '5166-5178 not seen; 5184-5196 glimpsed beneath the lantern; 5208 on the lantern '
                                     'and then a bearer cover the spot; from 5280 it holds the lantern, its bearers and '
                                     'their hand lanterns, and no fire can be told apart',
                                 sheet2='asound-evidence/claude/crops_far_watchfire.png'),
                   note='After this no warm pixel near watch-fires 2-4 can be separated from the great lantern, the '
                        'bearers and their hand lanterns (5200-5839): they have no measured interval of their own.'))
    raw = POINTS / 'crossing_cap_vs_decal.txt'
    summary = next((ln.split(' ', 1)[1] for ln in raw.read_text().splitlines() if ln.startswith('SUMMARY ')),
                   None) if raw.exists() else None
    return ev, shot_receipt(rows, rec, 4880, 5839, 'cand_crossing_both_decal') | dict(
        geometry=dict(file='asound-evidence/crossing_geometry.json', sha256=sha(geo_p)),
        both_decal_cap=dict(
            doc='owner bea534d stages cand_crossing_both_decal_cap first for A18 (plays once all 960 frames land): '
                'both_decal with ridge row 3 given a round cap. The measurements above are of both_decal; this is how '
                'far they carry to the cap on the frames that had landed',
            raw_source_frames=dict(file='asound-evidence/claude/crossing_cap_vs_decal.txt',
                                   sha256=sha(raw) if raw.exists() else None,
                                   method='native source JPEGs, |diff| per pixel (max over RGB) inside each '
                                          'watch-fire box projected by crossing_geometry.json',
                                   summary=summary),
            through_edit=cap_vs_decal(rows, geo, dv)))


# ---------------------------------------------------------------------------------------------- A19-A20
# what draws A19-A20's watch-fires in afix_comp: EDIT dispatches the 'watchfires' kind through KINDS/apply to
# watchfires(), which reads WATCHFIRES and calls REF_W, _ell, _grid, _lin, _srgb and _ss (an ast walk of the module,
# 22:12); the helpers it shares with bloom/vision/iceheart/ember are in the path, so an edit to one of them refuses
WF_PATH = ('REF_W', '_lin', '_srgb', '_ss', '_grid', '_ell', 'watchfires', 'KINDS', 'apply', 'WATCHFIRES')


# what the stills of the owner's comp show (crops_ending_owner_comp.png: five stills per fire around its pale, the
# lantern at 5840, 5876, 6200, 6456, 6470; the title's rising sparks cross fires 1-3's crops from 6240, not fires)
SAW_END = ('full 6210-6220, dimmer 6230, pale from 6240', 'full to 6280, pale 6290', 'full to 6330, pale 6340',
           'full to 6380, pale 6390', 'bright 5840-6456, going with the fade by 6470')


def watchfire_path(text):
    """SHA-256 of the watch-fire path's source (the module's imports and every WF_PATH definition) -> (sha, nodes)"""
    import ast
    tree = ast.parse(text)
    parts = [ast.get_source_segment(text, n) for n in tree.body
             if isinstance(n, (ast.Import, ast.ImportFrom))
             or (isinstance(n, ast.FunctionDef) and n.name in WF_PATH)
             or (isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) in WF_PATH)]
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest(), len(parts)


def ending_events():
    rows, rec = batches('ending')
    contiguous(rows, 5840, 6479, 'ending')
    comp = None
    for r in rec:
        e = r['edit'] or {}
        comp = comp or e.get('afix_comp_sha256')
        if e.get('afix_comp_sha256') != comp:
            raise SystemExit('ending batches measured through different watch-fire comps')
    fires = None
    root = Path(rec[0]['edit']['root'])                # recorded repo-relative ('.'): the public repo carries no host paths
    root = root if root.is_absolute() else ROOT / root
    src = root / 'edit' / 'afix_comp.py'
    now = src.read_bytes()
    measured_src = now
    if hashlib.sha256(now).hexdigest() != comp:
        # the owner's afix_comp moved on (22:10: a staged edit to A2's bloom, 536-560). Recover the source that was
        # measured from the commit the batches name, prove it by its SHA-256, and refuse only if the watch-fire path
        # itself changed: nothing else in the module draws 5840-6479
        import subprocess
        try:
            measured_src = subprocess.run(['git', '-C', str(root), 'show',
                                           f'{rec[0]["edit"]["commit"]}:./edit/afix_comp.py'],
                                          capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            raise SystemExit(f'{src} changed since the ending was measured and the measured source is not '
                             f'recoverable ({e}): re-measure')
        if hashlib.sha256(measured_src).hexdigest() != comp:
            raise SystemExit(f'{src} changed since the ending was measured (re-measure)')
    path_measured, path_nodes = watchfire_path(measured_src.decode())
    if watchfire_path(now.decode())[0] != path_measured:
        raise SystemExit(f'{src}: the watch-fire path changed since the ending was measured (re-measure)')
    import ast
    tree = ast.parse(measured_src.decode())
    for node in tree.body:                   # WATCHFIRES' literal fire list, read without importing EDIT
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'WATCHFIRES':
            for kw in node.value.keywords:
                if kw.arg == 'fires':
                    fires = ast.literal_eval(kw.value)
    names = ['near', 'far', 'farther', 'farthest', 'lantern']
    # the owner's branch moved during the collection (6160-6239 was read at d322b61, which changes A2's takes only);
    # the one afix_comp SHA-256 checked above is what pins these pixels, and the text names every commit read
    commits = {}
    for r in rec:
        commits[r['edit']['commit'][:7]] = commits.get(r['edit']['commit'][:7], 0) + 1
    read_at = ', '.join(f'{c} ({n} of {len(rec)} batches)' for c, n in commits.items())
    ev = []
    for k, (x, y, sz, pale) in enumerate(fires):
        rid = f'edit.{k + 1}'
        s = {f: next(v for v in rows[f]['rois'] if v['id'] == rid) for f in range(5840, 6480)}
        pk = {f: s[f]['pk'] for f in s}
        # the comp dims each fire to 15% over [pale-20, pale]: measure where the 9-frame median of the peak's
        # luma top-hat crosses 10% and 90% of its fall (pre = median 30..6 frames before pale-20, post = 6..30 after)
        med = {f: st.median(pk[g] for g in range(max(5840, f - 4), min(6479, f + 4) + 1)) for f in pk}
        e = dict(id=f'ending.{"lantern" if k == 4 else "fire_" + str(k + 1)}', shot='ending', stem='dawnrev_A + EDIT',
                 kind='burning', event=(f'watch-fire {k + 1} ({names[k]}) burns and pales' if k < 4 else
                                        'the set-down lantern burns among them to the fade'),
                 frames=dict(first=5840, last=6479, catch=None),
                 method='EDIT draws it (afix_comp.WATCHFIRES, measured through the owner\'s EDIT at '
                        f'{read_at}; afix_comp {comp[:12]}): luma top-hat (9x9) at its draw point, r 3 px, 960x402',
                 region=dict(x=x / 2, y=y / 2, space=HALF + ' (EDIT\'s draw point / 2)'),
                 size=dict(tophat={str(f): pk[f] for f in range(5840, 6480, 40)}),
                 confidence='high', basis=['MEASURED', 'OBSERVED'],
                 observed=dict(sheet='asound-evidence/claude/crops_ending_owner_comp.png', saw=SAW_END[k]))
        if pale < 90000:
            pre = st.median(med[g] for g in range(pale - 50, pale - 26))
            post = st.median(med[g] for g in range(pale + 6, min(6455, pale + 30)))
            drop = pre - post
            f10 = next(f for f in range(pale - 40, pale + 30) if med[f] <= pre - 0.1 * drop)
            f90 = next(f for f in range(pale - 40, pale + 30) if med[f] <= pre - 0.9 * drop)
            e['frames'].update(pale_start=f10, pale_end=f90)
            e['evidence'] = dict(tophat_pre=pre, tophat_post=post, pale_fraction=round(post / pre, 3),
                                 comp_pale_frame=pale)
        ev.append(e)
    fade = [f for f in range(6440, 6480)]
    lum = {f: st.mean(v['pk'] for v in rows[f]['rois']) for f in fade}
    return ev, shot_receipt(rows, rec, 5840, 6479, 'dawnrev_A') | dict(
        comp=dict(file='edit/afix_comp.py', sha256=comp, commits=commits, watchfire_path_sha256=path_measured,
                  watchfire_path=f'{path_nodes} top-level nodes of afix_comp: its imports, {", ".join(WF_PATH)}'),
        final_fade=dict(first=6456, last=6479, method='assemble.Ctx.picture fades the plate from 6456 (A: bar 81 '
                        'b3.8); the comp fades its own light with it (afix_comp arrival fade 6456-6479)',
                        mean_point_tophat={str(f): round(lum[f], 1) for f in fade[::4]}))


# ---------------------------------------------------------------------------------------------- A14 / A15
def _births(rows, a, b, min_area=150, hold_n=8, merge=60):
    """new warm fires: components (R>=100, R-G>=15, R-B>=35; >= min_area px) whose centroid lies in no box (grown by
    20 px) of a component of the frame before that is at least a quarter its size (a catch's one-frame spark before
    the burst does not make the burst a continuation), holding for hold_n frames (a component >= min_area/2 centred
    inside its box grown by 20 px + 3 px per frame: a near fire's glow spreads as it catches); births in one frame
    within `merge` px are one fire (its tongues). -> [(frame, comp)]"""
    out = []
    for f in range(a + 1, b + 1):
        prev = rows[f - 1]['warm']
        new = []
        for comp in sorted(rows[f]['warm'], key=lambda c: -c[0]):
            area, x0, y0, x1, y1, x, y = comp
            if area < min_area:
                continue
            if any(p[1] - 20 <= x <= p[3] + 20 and p[2] - 20 <= y <= p[4] + 20 for p in prev
                   if p[0] >= max(12, area / 4)):
                continue
            if any(math.hypot(n[5] - x, n[6] - y) <= merge for n in new):
                continue
            if all(g in rows and any(x0 - 20 - 3 * (g - f) <= q[5] <= x1 + 20 + 3 * (g - f) and
                                     y0 - 20 - 3 * (g - f) <= q[6] <= y1 + 20 + 3 * (g - f) and q[0] >= min_area / 2
                                     for q in rows[g]['warm']) for g in range(f + 1, min(b, f + hold_n) + 1)):
                new.append(comp)
        out += [(f, c) for c in new]
    return out


def _track(rows, f, comp, last, gate=20):
    """the component continuing `comp` from frame f to `last` (the largest within gate + its half width)"""
    x, y = comp[5], comp[6]
    seq = {f: comp}
    for g in range(f + 1, last + 1):
        cands = [q for q in rows[g]['warm'] if math.hypot(q[5] - x, q[6] - y) <= gate + 0.5 * (q[3] - q[1])]
        if not cands:
            break
        q = max(cands, key=lambda q: q[0])
        seq[g] = q
        x, y = q[5], q[6]
    return seq


def run_events(label, stem, alternative):
    """A14's links. The linked-fires take (adopted 29 Sep night) draws every link as a resolved flame: a link is a new
    warm fire (_births). The accepted take (now ALTERNATIVES['A14']) drew its far links as 1-2 px points no warm
    predicate sees: there a link is read like a ridge catch, at its source-projected spot (beacon_geometry.json)."""
    rows, rec = batches(label)
    contiguous(rows, 3920, 4239, label)
    ev = []
    if not alternative:
        # the score's run beats (bar map run_1..7): a link that catches before its beat is checked for a flare ON it
        beats = {e['id']: e['f'] for e in json.loads((MUSIC / 'v3' / 'barmap_A.json').read_text())['sync']}
        births = _births(rows, 3920, 4239)
        for k, (f, comp) in enumerate(births, 1):
            seq = _track(rows, f, comp, 4239)
            ref = min(f + 6, max(seq))
            c = seq[ref]
            area = {g: seq[g][0] for g in seq}
            fast = max(range(f, min(f + 6, max(seq)) + 1), key=lambda g: area[g] - area.get(g - 1, 0))
            flare = None                 # a later flare of the same fire: its area doubles the median of the 8 before
            for g in range(f + 12, max(seq) + 1):
                prev = [area[h] for h in range(g - 8, g) if h in area]
                if len(prev) == 8 and area[g] >= 2 * st.median(prev):
                    flare = g
                    break
            spark = [g for g in (f - 2, f - 1) if any(math.hypot(q[5] - comp[5], q[6] - comp[6]) <= 30 and q[0] <= 50
                                                     for q in rows[g]['warm'])]
            e = dict(id=f'run.link_{k}', shot='run', stem=stem, kind='catch', event=f'beacon-run link {k} catches',
                     frames=dict(first=f, fastest=fast, catch=f, lit_through=max(seq),
                                 spark=spark[0] if spark else None),
                     method='a new warm fire: a component (R>=100, R-G>=15, R-B>=35; 960x402, >= 150 px) whose '
                            'centroid lies in no box (+20 px) of a component of the frame before at least a quarter its '
                            'size, holding 8 frames (tongues within 60 px are one fire); fastest = its largest one-frame '
                            'growth in 6 frames; spark = a warm component of <= 50 px within 30 px of it 1-2 frames before '
                            '(the ignition envelope\'s first quarter-size frame); flare = the first frame >= 12 after '
                            'the catch whose area doubles the median of the 8 before',
                     region=dict(x=round(c[5], 1), y=round(c[6], 1), space=HALF + f', {ref - f} frames after first'),
                     size=dict(area_px=c[0], box=c[1:5], area_series={str(g): area[g] for g in sorted(area)[:16]}),
                     confidence='high', basis=['MEASURED', 'OBSERVED'],
                     observed=dict(sheet='asound-evidence/claude/crops_run_links.png',
                                   saw='links 1-6 each: dark two frames before, a warm spark on the frame before, the '
                                       'whole flame on the catch frame; link 7: the rock below the watcher floods '
                                       'warm on 4200 (its hearth)',
                                   sheet2='asound-evidence/claude/sheet_linked_farmtest.jpg (whole frames 3920-4399)'))
            if flare and any(abs(flare - g) <= 3 for g, _ in births):
                flare = None                # the tracker walked into the next link's catch: not this fire's flare
            if flare:
                e['frames']['flare'] = flare
                e['evidence'] = dict(area_around_flare={str(g): area.get(g) for g in range(flare - 3, flare + 4)})
            beat = beats.get(f'run_{k}')
            if beat is not None and beat != f and beat - 8 in area and beat in area:
                ratio = area[beat] / st.median(area[h] for h in range(beat - 8, beat))
                e.setdefault('evidence', {})['score_beat'] = dict(
                    id=f'run_{k}', f=beat, area_px={str(g): area.get(g) for g in range(beat - 4, beat + 5)},
                    ratio_to_median_of_8_before=round(ratio, 2),
                    reading=('it flares on the beat' if ratio >= 2 else 'it burns on through the beat: no flare '
                             '(a flare doubles its area)'))
                if k == 1:                   # the one beat looked at on stills: link 1 at run_1 (3960)
                    e['observed']['sheet3'] = 'asound-evidence/claude/crops_link1.png: 3956-3964, a steady flame'
            ev.append(e)
        return ev, shot_receipt(rows, rec, 3920, 4239, stem)
    geo_p = EVID / 'beacon_geometry.json'
    geo = json.loads(geo_p.read_text())
    ign = {c['id']: c['source_ignition'] for c in geo['frames'][0]['chain']}
    for k in sorted(ign):
        s = {}
        for f in sorted(rows):
            x = next((x for x in rows[f]['rois'] if x['id'] == f'link.{k}'), None)
            if x is not None and not x.get('off_frame') and EDGE <= x['x'] < 960 - EDGE and EDGE <= x['y'] < 402 - EDGE:
                s[f] = x
        i0 = math.floor(ign[k])
        window = [f for f in range(i0 + IGN_WINDOW[0], i0 + IGN_WINDOW[1] + 1) if f in s]
        c = catch_in(s, window)
        if c is None and all('warm' in s[f] for f in window):
            # a resolved flame (links 5 and 7 are 3-23 px wide) is wider than the top-hat: its warm pixels in the box
            ws = {f: {'warm': s[f]['warm']} for f in s if 'warm' in s[f]}
            c = catch_in(ws, window, key='warm', ck=None)
        e = dict(id=f'run_accepted.link_{k}', shot='run_accepted', stem=stem,
                 event=f'beacon-run link {k} catches (the accepted take, now the alternative)',
                 method='as ridges.*: luma top-hat jump at the source-projected spot (beacon_geometry.json) searched '
                        f'{IGN_WINDOW[0]}..+{IGN_WINDOW[1]} frames around the source ignition',
                 confidence='medium', basis=['MEASURED'])
        if c is None:
            e.update(kind='not_measured', frames=dict(first=None, catch=None),
                     note='no catch-like jump at the projected spot (the spot leaves the frame or is hidden)')
        else:
            f1 = c[0]
            ref = min(f1 + 6, max(s))
            e.update(kind='catch', frames=dict(first=f1, catch=f1),
                     region=dict(x=s[ref]['px'], y=s[ref]['py'], space=HALF),
                     size=dict(tophat=s[ref]['pk'], rgb=s[ref]['rgb']),
                     evidence=dict(source_ignition=ign[k], tophat={str(g): s[g]['pk'] for g in range(f1 - 3, f1 + 7)
                                                                   if g in s}))
        ev.append(e)
    return ev, shot_receipt(rows, rec, 3920, 4239, stem) | dict(
        geometry=dict(file='asound-evidence/beacon_geometry.json', sha256=sha(geo_p)))


def watcher_events(label, stem, alternative):
    rows, rec = batches(label)
    contiguous(rows, 4240, 4399, label)
    big = {f: max(rows[f]['warm'], key=lambda c: c[0]) for f in range(4240, 4400)}
    areas = [big[f][0] for f in range(4240, 4400)]
    e = dict(id='watchers_accepted.near_fire' if alternative else 'watchers.near_fire', shot='watchers', stem=stem,
             kind='burning', event="the watcher's near fire burns through the shot (link 7's hearth, lit at 4200 in "
                                   "the run)", frames=dict(first=4240, last=4399, catch=None),
             method='the largest warm component (R>=100, R-G>=15, R-B>=35; 960x402) in every frame',
             region=dict(x=round(st.median(big[f][5] for f in big), 1), y=round(st.median(big[f][6] for f in big), 1),
                         space=HALF + ', median centroid'),
             size=dict(area_px_median=st.median(areas), area_px_min=min(areas), area_px_max=max(areas),
                       area_px={str(f): big[f][0] for f in range(4240, 4400, 20)}),
             confidence='high', basis=['MEASURED'])
    if not alternative:                      # only the adopted take was looked at on stills tonight
        e['basis'].append('OBSERVED')
        e['observed'] = dict(sheet='asound-evidence/claude/sheet_linked_farmtest.jpg',
                             saw='4240, 4241, 4320, 4399: the watcher a silhouette before the near fire at the lower '
                                 'left, the links burning along the ridge behind')
    return [e], shot_receipt(rows, rec, 4240, 4399, stem)


def build():
    ridge, ridge_shot = ridge_events()
    inserts, insert_shots = insert_events()
    run, run_shot = run_events('run_linked', 'cand_beaconrun_linked-fires', False)
    run_acc, run_acc_shot = run_events('run-accepted', 'beaconrun_A_catches3', True)
    wat, wat_shot = watcher_events('watchers_linked', 'cand_watchers_linked-fires', False)
    wat_acc, wat_acc_shot = watcher_events('watchers-accepted', 'watchers_A_catches3', True)
    cross, cross_shot = crossing_events()
    ending, ending_shot = ending_events()
    fade = ending_shot.pop('final_fade')
    ending.append(dict(id='ending.final_fade', shot='ending', stem='dawnrev_A + EDIT', kind='fade',
                       event='the last fade to black', frames=dict(first=fade['first'], last=fade['last']),
                       method=fade['method'], evidence=dict(mean_point_tophat=fade['mean_point_tophat']),
                       confidence='high', basis=['MEASURED']))
    events = ridge + inserts + run + wat + cross + ending + run_acc + wat_acc
    ids = [e['id'] for e in events]
    if len(ids) != len(set(ids)):
        raise SystemExit('duplicate event ids')
    here = Path(__file__)
    return dict(schema='long-dawn/a-events-measured/1', cut='A', fps=FPS,
                frames='absolute A cut frame numbers; first/last inclusive; positions in 960x402 finished pixels',
                generated_by='music/src/measure_a_analysis.py', generator_sha256=sha(here),
                collector='music/src/measure_a_points.py', collector_sha256=sha(HERE / 'measure_a_points.py'),
                basis_labels=dict(MEASURED='a pixel statistic decided the frame(s) (see method)',
                                  OBSERVED='checked by eye on half-size stills of the same finished frames (never in '
                                           'motion): asound-evidence/claude/crops_*.png and sheet_*.jpg',
                                  SOURCE='a renderer parameter, kept as evidence beside the measurement; never the '
                                         'measured frame'),
                sync_definitions=dict(
                    catch='first = the first frame its flame is visible (a sound burst arrives on first..fastest); '
                          'fastest = its largest one-frame growth',
                    burning='already alight when the shot opens: first/last are the frames it is in view, catch null',
                    recede='a burning fire the camera draws away from: its footprint in dB re its close-shot size',
                    pale='EDIT dims a watch-fire to 15% of its light over [pale-20, pale]: pale_start/pale_end = the '
                         'frames its 9-frame median top-hat crosses 10% / 90% of its measured fall'),
                shots=dict(ridges=ridge_shot, **insert_shots, run=run_shot, watchers=wat_shot, crossing=cross_shot,
                           ending=ending_shot, run_accepted=run_acc_shot, watchers_accepted=wat_acc_shot),
                not_measured=dict(
                    first='3600-3659 (the roar, HER FIRE): SOUND-C measured them (picture_sync_A: roar 3600); '
                          'unchanged here',
                    towers_heart='4400-4879 (A16-A17, EMBERS): no fire effect of the table plays there',
                    setoff='A18\'s set-off is the AP2 lane\'s measurement (codex/a18_setoff.json reports 5180, first '
                           'displaced 5181: theirs, not re-measured here); no effect in this table depends on it (the '
                           'walking is the score\'s), so it is not read'),
                alternatives='run_accepted.* and watchers_accepted.* measure the takes the cut played before tonight\'s '
                             'adoption (beaconrun_A_catches3, watchers_A_catches3): EDL ALTERNATIVES[A14/A15]',
                events=events)


def dumps(d):
    return json.dumps(d, indent=1) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    out = dumps(build())
    if a.check:
        ok = OUT.exists() and OUT.read_text() == out
        print('events_A_measured.json is current' if ok else 'STALE: events_A_measured.json differs from the evidence')
        sys.exit(0 if ok else 1)
    tmp = OUT.with_suffix('.json.part')
    tmp.write_text(out)
    tmp.replace(OUT)
    print(f'wrote {OUT}')
