"""A18 THE CROSSING, measured for A's score pass 2 (AP2): when the line sets off, how it gets up to pace, and when
the great lantern passes each watch-fire. Read-only; renders nothing.

    cd the-long-dawn
    .../claude-owner-night/tools/onepy ~/ldfarm/venv/bin/python music/src/measure_ap2_crossing.py --out a18_setoff.json

Two readings, independent where they can be:

  PROJECTION  the renderer's own path, timing and camera (shots/run/crossing.py, checked against the sha256 that
              crossing_candidates.py pins as the baseline of the delivered candidate cand_crossing_both_decal; the
              candidate's rock and rope options change geometry and rasterising only, never a clock). Positions are
              projected through camera(frame) with RCam.project: no image is rendered (the scene is never built).
  FRAMES      the delivered frames renders/cand_crossing_both_decal/f_<cut>.jpg (EDL A18 plays them 'exact': cut
              frame = file frame), one frame at a time at half size (960x402). The camera FOLLOWS the lantern
              (crossing.heart_pos tracks lantern_s), so the walkers barely move on screen when they set off: the
              terrain slides past them. Frame differencing is therefore confounded by the draw-back; instead the
              lantern (the bearers' pole lantern, the line's front) is located in every frame and terrain patches
              near it are tracked frame to frame by phase correlation. Their motion relative to the lantern is
              compared with the projection of the same terrain points under two hypotheses: the line as rendered,
              and the line never setting off (crossing.T_GO patched to infinity for the camera and the bearers only).
              The departure from the standing hypothesis is fitted for its onset frame.

Frames are absolute A cut frames (the crossing's shot frame 0 = cut 4880). Screen coordinates are full size
(1920x804) unless a key says half.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, 'shots', 'run')
FRAMES_DIR = os.path.join(ROOT, 'renders', 'cand_crossing_both_decal')
CUT0, CUT_END = 4880, 5839
FPS = 24.0
HALF = (960, 402)
# terrain patches tracked on the delivered frames (half-size pixel centre at TRACK_F0, half-width, half-height),
# chosen by eye on f_05120/f_05180 away from every figure: the stones on the first watch-fire's rock (right of its
# flames), the snow crest just below the bearers, and two stretches of the arete's lit south flank
PATCHES = {'fire_rock': (790, 238, 14, 10), 'crest': (655, 262, 22, 10), 'flank_mid': (700, 335, 32, 24),
           'flank_left': (560, 330, 32, 24), 'flank_a': (620, 300, 24, 16), 'flank_b': (750, 300, 24, 16),
           'flank_c': (650, 365, 24, 16), 'flank_d': (760, 370, 24, 16), 'flank_e': (520, 360, 24, 16)}
TRACK_F0, TRACK_F1 = 5100, 5260
SEARCH = (5150, 5215)                         # onset candidates for the fit (cut frames)


def _rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def load_crossing():
    """The pinned renderer module (import only: its scene is never built here)."""
    sys.path.insert(0, RUN)
    import crossing_candidates as CC
    src = os.path.join(RUN, 'crossing.py')
    sha = hashlib.sha256(open(src, 'rb').read()).hexdigest()
    if sha != CC.BASELINE_SHA256:
        raise RuntimeError(f'crossing.py {sha} is not the delivered candidate baseline {CC.BASELINE_SHA256}')
    import crossing as C
    return C, sha


@contextlib.contextmanager
def standing(C):
    """The counterfactual: the line never sets off (lantern_s, walk_k and the camera that follows them read T_GO)."""
    old = C.T_GO
    C.T_GO = 1e9
    try:
        yield
    finally:
        C.T_GO = old


def _proj(cam, P):
    x, y, z = cam.project(np.asarray(P, np.float64))
    return float(x), float(y), float(z)


# ---------------------------------------------------------------------------------------------------- projection
def projection(C):
    cfg = C.variant_cfg('main')
    fires = C.wfs()        # FIRST: it finishes the set (CR gains the watch-fire rocks, which the feet stand near)
    t_go = C.T_GO
    rest = C.line_s(t_go, cfg)                     # every figure's arc at the instant the front sets off (all stand)
    B0 = C.bearers(t_go, cfg)
    p_rest = C.at(rest[0])[0]
    feet_rest = (B0['front'][1], B0['front'][2])
    n_fig = len(rest)
    rows = []
    prev = None
    first_move = np.full(n_fig, -1)
    first_pace = np.full(n_fig, -1)
    for f in range(CUT0, CUT_END + 1):
        sf = f - CUT0
        t = sf / FPS
        cam = C.camera(sf, cfg=cfg)
        S = C.line_s(t, cfg)
        B = C.bearers(t, cfg)
        ds = S - rest
        v = (S - prev) * FPS if prev is not None else np.zeros(n_fig)
        prev = S
        for i in range(n_fig):
            if first_move[i] < 0 and ds[i] > 1e-3:
                first_move[i] = f
            if first_pace[i] < 0 and v[i] >= 0.9 * C.SPEED:
                first_pace[i] = f
        xa, ya, _ = _proj(cam, C.at(S[0])[0])
        xr, yr, _ = _proj(cam, p_rest)
        foot = []
        for k in (1, 2):
            xf, yf, _ = _proj(cam, B['front'][k])
            xf0, yf0, _ = _proj(cam, feet_rest[k - 1])
            foot.append(math.hypot(xf - xf0, yf - yf0))
        hx, hy, hz = _proj(cam, B['heart'])
        px, py, _ = _proj(cam, B['front'][0])
        fx = []
        for F in fires:
            x, y, z = _proj(cam, F)
            fx.append((x, y, z))
        rows.append(dict(f=f, ds_front_m=float(ds[0]), v_front_ms=float(v[0]), walk_k=float(C.walk_k(t)),
                         screen_path_px=math.hypot(xa - xr, ya - yr), screen_feet_px=[round(a, 4) for a in foot],
                         heart=(hx, hy, hz), pelvis_front=(px, py), fires=fx, f_px=cam.f,
                         moving_frac=float(np.mean(ds > 1e-3)),
                         line_speed_frac=float(np.mean(np.clip(v / C.SPEED, 0.0, None))) if f > CUT0 else 0.0))
    by = {r['f']: r for r in rows}

    def first(key, thr, f_from=CUT0):
        for r in rows:
            if r['f'] >= f_from and (max(r[key]) if isinstance(r[key], list) else r[key]) >= thr:
                return r['f']
        return None
    t_go_frame = CUT0 + t_go * FPS
    # the watch-fire passes: the frame on which the lantern heart's screen x crosses the fire's (interpolated), and the
    # front bearer's; a fire behind the camera or out of frame is skipped
    passes = []
    for k in range(len(fires)):
        out = dict(fire=k + 1, world=[round(float(c), 3) for c in fires[k]])
        for who, key in (('lantern', 'heart'), ('front_bearer', 'pelvis_front')):
            cross = None
            for a, b in zip(rows[:-1], rows[1:]):
                if a['fires'][k][2] <= 0 or b['fires'][k][2] <= 0:
                    continue
                da = a[key][0] - a['fires'][k][0]
                db = b[key][0] - b['fires'][k][0]
                if da < 0 <= db or da > 0 >= db:
                    cross = a['f'] + da / (da - db)
                    break
            out[f'{who}_crosses_fire_x'] = None if cross is None else round(cross, 2)
        best = min((r for r in rows if r['fires'][k][2] > 0),
                   key=lambda r: math.hypot(r['heart'][0] - r['fires'][k][0], r['heart'][1] - r['fires'][k][1]))
        out['closest_lantern_frame'] = best['f']
        out['closest_lantern_px'] = round(math.hypot(best['heart'][0] - best['fires'][k][0],
                                                     best['heart'][1] - best['fires'][k][1]), 2)
        near = [r['f'] for r in rows if r['fires'][k][2] > 0 and math.hypot(
            r['heart'][0] - r['fires'][k][0], r['heart'][1] - r['fires'][k][1]) < 15.0]
        out['lantern_within_15px'] = [near[0], near[-1]] if near else None
        vis = [r['f'] for r in rows if r['fires'][k][2] > 0 and 0 <= r['fires'][k][0] < 1920
               and 0 <= r['fires'][k][1] < 804]
        out['in_frame'] = [vis[0], vis[-1]] if vis else None
        passes.append(out)
    for p, pf in zip(passes, C.PASS_FRAMES):
        p['renderer_pass_frame'] = CUT0 + pf
        p['heart_to_fire_px_at_renderer_pass'] = round(math.hypot(
            by[CUT0 + pf]['heart'][0] - by[CUT0 + pf]['fires'][p['fire'] - 1][0],
            by[CUT0 + pf]['heart'][1] - by[CUT0 + pf]['fires'][p['fire'] - 1][1]), 2)
    onsets = sorted(int(x) for x in first_move if x >= 0)
    paces = sorted(int(x) for x in first_pace if x >= 0)
    table = [dict(f=r['f'], ds_front_m=round(r['ds_front_m'], 6), v_front_ms=round(r['v_front_ms'], 5),
                  speed_frac_front=round(r['walk_k'], 4), screen_path_px=round(r['screen_path_px'], 4),
                  screen_feet_px=r['screen_feet_px'], heart_px=[round(r['heart'][0], 3), round(r['heart'][1], 3)],
                  line_moving_frac=round(r['moving_frac'], 4), line_speed_frac=round(r['line_speed_frac'], 4))
             for r in rows if 5100 <= r['f'] <= 5460]
    return dict(
        renderer_constants=dict(T_GO_s=t_go, TAU_GO_s=C.TAU_GO, SPEED_ms=C.SPEED,
                                set_off_instant_cut_frame=t_go_frame,
                                speed_model='lantern_s: v = SPEED * (t - T_GO) / TAU_GO for T_GO < t < T_GO + TAU_GO, '
                                            'then SPEED; the front bearer = lantern_s + 1.3 m'),
        front=dict(first_frame_displaced=first('ds_front_m', 1e-9),
                   first_frame_path_ge_0p5px=first('screen_path_px', 0.5),
                   first_frame_path_ge_1px=first('screen_path_px', 1.0),
                   first_frame_path_ge_2px=first('screen_path_px', 2.0),
                   first_frame_foot_ge_0p5px=first('screen_feet_px', 0.5),
                   first_frame_foot_ge_1px=first('screen_feet_px', 1.0),
                   speed_frac_ge_0p5=first('walk_k', 0.5), speed_frac_ge_0p9=first('walk_k', 0.9),
                   full_pace=first('walk_k', 1.0)),
        line=dict(figures=n_fig, first_figure_moves=onsets[0] if onsets else None,
                  half_the_line_moving=onsets[(len(onsets) - 1) // 2] if onsets else None,
                  last_figure_moves=onsets[-1] if onsets else None,
                  figures_reaching_0p9_pace=len(paces), first_at_0p9_pace=paces[0] if paces else None,
                  half_at_0p9_pace=paces[(len(paces) - 1) // 2] if paces else None,
                  last_at_0p9_pace=paces[-1] if paces else None,
                  note='a walker = arc displacement > 1 mm; at pace = backward-difference speed >= 0.9 SPEED (the '
                       'groups breathe +-0.3 m, so a walker at pace is not steady)'),
        watch_fires=passes, table_5100_5460=table)


# ---------------------------------------------------------------------------------------------------- frames
def _read_half(f):
    import cv2
    p = os.path.join(FRAMES_DIR, f'f_{f:05d}.jpg')
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (804, 1920):
        raise FileNotFoundError(p)
    y = (im[..., 2] * 0.2126 + im[..., 1] * 0.7152 + im[..., 0] * 0.0722).astype(np.float32)
    del im
    return cv2.resize(y, HALF, interpolation=cv2.INTER_AREA)


def _backproject(C, cam, u_half, v_half):
    """The terrain point seen at a half-size pixel: march the camera ray over the renderer's own height field."""
    u, v = 2.0 * u_half + 0.5, 2.0 * v_half + 0.5
    d = cam.ray(np.array([u]), np.array([v]))[0]
    d = d / np.linalg.norm(d)
    ts = np.arange(0.5, 600.0, 0.02)
    P = cam.pos[None, :] + d[None, :] * ts[:, None]
    g = C.ground_many(P)
    below = np.flatnonzero(P[:, 1] <= g)
    if not len(below):
        raise ValueError('the ray meets no terrain')
    a, b = ts[max(below[0] - 1, 0)], ts[below[0]]
    for _ in range(30):
        m = 0.5 * (a + b)
        q = cam.pos + d * m
        if q[1] <= C.ground_many(q)[0]:
            b = m
        else:
            a = m
    return cam.pos + d * b, float(b)


def _lantern(y, x0, y0, r=12):
    """Intensity-weighted centroid of the lantern's glass (pixels above 60 % of the window's peak)."""
    xi, yi = int(round(x0)), int(round(y0))
    w = y[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(np.float64)
    thr = 0.6 * w.max()
    m = np.clip(w - thr, 0.0, None)
    yy, xx = np.mgrid[yi - r:yi + r + 1, xi - r:xi + r + 1]
    s = m.sum()
    return float((xx * m).sum() / s), float((yy * m).sum() / s), float(w.max())


def frames(C):
    import cv2
    # the sign convention of cv2.phaseCorrelate, proved on a known shift before it is trusted
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.random((80, 80)).astype(np.float32), (0, 0), 1.5)
    moved = np.roll(base, 3, axis=1)
    (sx, _), _ = cv2.phaseCorrelate(base[10:70, 10:70], moved[10:70, 10:70])
    if not 2.5 < sx < 3.5:
        raise RuntimeError(f'phaseCorrelate sign/scale check failed: {sx}')
    cfg = C.variant_cfg('main')
    cam0 = C.camera(TRACK_F0 - CUT0, cfg=cfg)
    pts = {}
    for k, (u, v, hw, hh) in PATCHES.items():
        P, depth = _backproject(C, cam0, u, v)
        pts[k] = dict(P=P, depth_m=depth, hw=hw, hh=hh, pos=[float(u), float(v)])

    def predict(f, stand):
        sf, t = f - CUT0, (f - CUT0) / FPS
        ctx = standing(C) if stand else contextlib.nullcontext()
        with ctx:
            cam = C.camera(sf, cfg=cfg)
            hx, hy, _ = _proj(cam, C.bearers(t, cfg)['heart'])
            out = {'heart': ((hx - 0.5) / 2, (hy - 0.5) / 2)}
            for k, d in pts.items():
                x, y, _ = _proj(cam, d['P'])
                out[k] = ((x - 0.5) / 2, (y - 0.5) / 2)
        return out
    win = {k: cv2.createHanningWindow((2 * d['hw'], 2 * d['hh']), cv2.CV_32F) for k, d in pts.items()}
    rows = []
    prev = None
    for f in range(TRACK_F0, TRACK_F1 + 1):
        y = _read_half(f)
        pw, ps = predict(f, False), predict(f, True)
        lx, ly, lpk = _lantern(y, *pw['heart'])
        row = dict(f=f, lantern=[lx, ly], lantern_peak=lpk, pred_walk=pw, pred_stand=ps, patch={})
        for k, d in pts.items():
            if prev is not None:
                cx, cy = int(round(d['pos'][0])), int(round(d['pos'][1]))
                hw, hh = d['hw'], d['hh']
                a = prev[cy - hh:cy + hh, cx - hw:cx + hw]
                b = y[cy - hh:cy + hh, cx - hw:cx + hw]
                (dx, dy), resp = cv2.phaseCorrelate(a, b, win[k])
                d['pos'] = [d['pos'][0] + dx, d['pos'][1] + dy]
                row['patch'][k] = dict(pos=list(d['pos']), step=[dx, dy], response=resp)
            else:
                row['patch'][k] = dict(pos=list(d['pos']), step=[0.0, 0.0], response=1.0)
        rows.append(row)
        prev = y
    # relative to the lantern, from TRACK_F0: measured, walking (as rendered) and standing (counterfactual)
    out = dict(track=[TRACK_F0, TRACK_F1], patches={}, sign_check_px=round(float(sx), 4))
    r0 = rows[0]
    fr = np.array([r['f'] for r in rows], float)
    pre = fr < 5176
    grid = np.arange(SEARCH[0] - 5180.0, SEARCH[1] - 5180.0 + 1e-9, 0.25)
    marks = (5170, 5180, 5185, 5190, 5195, 5200, 5210, 5220, 5240, 5260)

    def profile(res, sig):
        """SSE of res = c0 + c1 (f - 5180) + A sig(f - delta) over the delta grid (the linear term absorbs a terrain
        point's small depth error and the tracker's slow drift, both of which the standing line also has)"""
        out = []
        for dl in grid:
            s_ = np.interp(fr - dl, fr, sig, left=0.0)
            A = np.stack([s_, np.ones_like(s_), fr - 5180.0], 1)
            coef, *_ = np.linalg.lstsq(A, res, rcond=None)
            out.append((float(np.sum((A @ coef - res) ** 2)), coef))
        return out
    total = np.zeros(len(grid))
    n_tot, n_par, used = 0, 1, []
    for k, d in pts.items():
        meas = np.array([r['patch'][k]['pos'][0] - r['lantern'][0] for r in rows])
        walk = np.array([r['pred_walk'][k][0] - r['pred_walk']['heart'][0] for r in rows])
        stand = np.array([r['pred_stand'][k][0] - r['pred_stand']['heart'][0] for r in rows])
        meas, walk, stand = meas - meas[0], walk - walk[0], stand - stand[0]
        res = meas - stand                       # what the standing line cannot explain
        sig = walk - stand                       # what the set-off predicts
        L = np.polyfit(fr[pre], res[pre], 1)     # the pre-set-off trend, extrapolated
        det = res - np.polyval(L, fr)
        noise = float(np.std(det[pre]))
        prof = profile(res, sig)
        sse = np.array([x[0] for x in prof])
        j = int(np.argmin(sse))
        n = len(res)
        ok = grid[sse <= sse[j] * (1.0 + 4.0 / (n - 4))]       # ~95 % profile interval (F-test, one parameter)
        resp = min(r['patch'][k]['response'] for r in rows[1:])
        # a TRACKING gate, blind to the onset: the tracker held its terrain (phase-correlation peak >= 0.5 in every
        # frame) and the pre-set-off residual is smooth (<= 0.5 half-px about its own trend)
        good = resp >= 0.5 and noise <= 0.5
        if good:
            total += sse
            n_tot += n
            n_par += 3
            used.append(k)
        exceed = [int(f) for f, x in zip(fr, det) if f >= 5176 and abs(x) > 3.0 * noise]
        out['patches'][k] = dict(
            start_half_px=[round(PATCHES[k][0], 2), round(PATCHES[k][1], 2)], depth_m=round(d['depth_m'], 3),
            world=[round(float(c), 3) for c in d['P']], used_in_joint_fit=good,
            fit_onset_frame=round(5180.0 + float(grid[j]), 2),
            fit_interval=[round(5180.0 + float(ok[0]), 2), round(5180.0 + float(ok[-1]), 2)],
            fit_amplitude=round(float(prof[j][1][0]), 4), pre_noise_sd_half_px=round(noise, 4),
            first_frame_beyond_3sd_of_pre_trend=exceed[0] if exceed else None,
            detrended_residual_at={str(int(f)): round(float(x), 3) for f, x in zip(fr, det) if int(f) in marks},
            predicted_signal_at={str(int(f)): round(float(x), 3) for f, x in zip(fr, sig) if int(f) in marks},
            min_response=round(resp, 3))
    if used:
        j = int(np.argmin(total))
        ok = grid[total <= total[j] * (1.0 + 4.0 / max(n_tot - n_par, 1))]
        out['joint_fit'] = dict(patches=used, onset_frame=round(5180.0 + float(grid[j]), 2),
                                interval=[round(5180.0 + float(ok[0]), 2), round(5180.0 + float(ok[-1]), 2)],
                                sse_at_5180=round(float(total[int(np.argmin(np.abs(grid)))]), 3),
                                sse_min=round(float(total[j]), 3))
    # the same test in VELOCITY (per-frame steps), where a tracker's error does not accumulate: each patch's measured
    # step against the steps of its terrain point projected under the standing line; the model is
    #   v(f) = c + B * stand_step(f) + A * set_off_step(f - delta)
    # (B absorbs a small depth error of the back-projected point, which scales its parallax; c the tracker's bias)
    vt = np.zeros(len(grid))
    vn, vp, vused = 0, 1, []
    out['velocity'] = {}
    frv = fr[1:]
    for k, d in pts.items():
        meas = np.array([r['patch'][k]['step'][0] for r in rows[1:]])
        pw_ = np.diff([r['pred_walk'][k][0] for r in rows])
        ps_ = np.diff([r['pred_stand'][k][0] for r in rows])
        v = meas - ps_
        u = pw_ - ps_
        prev_ = frv < 5176
        Ap = np.stack([np.ones(prev_.sum()), ps_[prev_]], 1)
        cp, *_ = np.linalg.lstsq(Ap, v[prev_], rcond=None)
        sd = float(np.std(v[prev_] - Ap @ cp))
        prof = []
        for dl in grid:
            s_ = np.interp(frv - dl, frv, u, left=0.0)
            A = np.stack([s_, np.ones_like(s_), ps_], 1)
            coef, *_ = np.linalg.lstsq(A, v, rcond=None)
            prof.append((float(np.sum((A @ coef - v) ** 2)), coef))
        sse = np.array([x[0] for x in prof])
        j = int(np.argmin(sse))
        n = len(v)
        ok = grid[sse <= sse[j] * (1.0 + 4.0 / (n - 4))]
        resp = min(r['patch'][k]['response'] for r in rows[1:])
        good = resp >= 0.3 and sd <= 0.2           # the tracker's per-frame noise, before any set-off (blind to it)
        if good:
            vt += sse / sd ** 2                     # each patch weighted by its own pre-set-off noise
            vn += n
            vp += 3
            vused.append(k)
        out['velocity'][k] = dict(pre_sd_half_px_per_frame=round(sd, 4), used=good,
                                  fit_onset_frame=round(5180.0 + float(grid[j]), 2),
                                  fit_interval=[round(5180.0 + float(ok[0]), 2), round(5180.0 + float(ok[-1]), 2)],
                                  fit_amplitude=round(float(prof[j][1][0]), 4))
    if vused:
        j = int(np.argmin(vt))
        # chi-square profile (each patch's residuals scaled by its own pre-set-off sd): delta-chi2 <= 4 ~ 95 %
        scale = max(vt[j] / max(vn - vp, 1), 1.0)
        ok = grid[vt <= vt[j] + 4.0 * scale]
        out['velocity_joint'] = dict(patches=vused, onset_frame=round(5180.0 + float(grid[j]), 2),
                                     interval=[round(5180.0 + float(ok[0]), 2), round(5180.0 + float(ok[-1]), 2)],
                                     chi2_min=round(float(vt[j]), 2), dof=int(vn - vp), scale=round(float(scale), 3),
                                     chi2_at_5180=round(float(vt[int(np.argmin(np.abs(grid)))]), 2))
    lant = np.array([r['lantern'][0] for r in rows]) - rows[0]['lantern'][0]
    lpred = np.array([r['pred_walk']['heart'][0] for r in rows]) - r0['pred_walk']['heart'][0]
    out['lantern_vs_projection_half_px'] = dict(max_abs_dev=round(float(np.max(np.abs(lant - lpred))), 3),
                                                rms_dev=round(float(np.sqrt(np.mean((lant - lpred) ** 2))), 3))
    out['series'] = [dict(f=r['f'], lantern_half=[round(r['lantern'][0], 3), round(r['lantern'][1], 3)],
                          heart_walk_half=[round(r['pred_walk']['heart'][0], 3), round(r['pred_walk']['heart'][1], 3)],
                          patches={k: dict(x=round(r['patch'][k]['pos'][0], 3), step=round(r['patch'][k]['step'][0], 4),
                                           resp=round(r['patch'][k]['response'], 3),
                                           walk=round(r['pred_walk'][k][0], 3), stand=round(r['pred_stand'][k][0], 3))
                                   for k in pts}) for r in rows]
    return out


def evidence(C, res, out_dir, prefix):
    """two images: the front bearer against its rest point on the terrain (full-size crops of the delivered frames,
    projected markers), and the curves (projection; frames against both hypotheses)"""
    import cv2
    cfg = C.variant_cfg('main')
    rest = C.line_s(C.T_GO, cfg)
    p_rest = C.at(rest[0])[0]
    tiles = []
    for f in (5170, 5180, 5190, 5200, 5210, 5220, 5240, 5260):
        sf, t = f - CUT0, (f - CUT0) / FPS
        cam = C.camera(sf, cfg=cfg)
        im = cv2.imread(os.path.join(FRAMES_DIR, f'f_{f:05d}.jpg'), cv2.IMREAD_COLOR)
        xr, yr, _ = _proj(cam, p_rest)
        xa, ya, _ = _proj(cam, C.at(C.line_s(t, cfg)[0])[0])
        hx, hy, _ = _proj(cam, C.bearers(t, cfg)['heart'])
        x0, y0 = int(hx) - 260, int(hy) - 110
        crop = im[max(y0, 0):y0 + 260, max(x0, 0):x0 + 520].copy()
        del im
        for (x, y), col in (((xr, yr), (255, 0, 255)), ((xa, ya), (0, 255, 255))):
            cv2.drawMarker(crop, (int(round(x - max(x0, 0))), int(round(y - max(y0, 0)))), col, cv2.MARKER_CROSS, 14, 2)
        d = math.hypot(xa - xr, ya - yr)
        cv2.putText(crop, f'{f}  path {d:5.1f} px', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(crop, (520, 260)))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in (0, 4)])
    cv2.putText(sheet, 'magenta: the front bearer\'s ground point at rest (fixed terrain); yellow: where it is now '
                '(crossing.line_s through camera(frame))', (8, sheet.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    p1 = os.path.join(out_dir, f'{prefix}_front_vs_rest.jpg')
    cv2.imwrite(p1, sheet)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    tb = res['projection']['table_5100_5460']
    f_ = [r['f'] for r in tb]
    ax[0].plot(f_, [r['speed_frac_front'] for r in tb], label='front bearers: speed / SPEED')
    ax[0].plot(f_, [r['line_speed_frac'] for r in tb], label='whole line (42): mean speed / SPEED')
    ax[0].plot(f_, [r['line_moving_frac'] for r in tb], label='whole line: fraction moving (> 1 mm)')
    ax0b = ax[0].twinx()
    ax0b.plot(f_, [r['screen_path_px'] for r in tb], 'k:', label='front bearer vs its rest point (full-size px)')
    ax0b.set_ylabel('px (full size)')
    for x in (5180, 5276):
        ax[0].axvline(x, color='grey', lw=0.6)
    ax[0].set_title('projection (crossing.py + camera): set-off 5180.0, full pace 5276')
    ax[0].set_xlabel('A cut frame')
    ax[0].legend(loc='upper left', fontsize=8)
    ax0b.legend(loc='lower right', fontsize=8)
    fr = res['frames']
    ser = fr['series']
    ff = np.array([r['f'] for r in ser])
    i0 = int(np.flatnonzero(ff == 5180)[0])
    for k in fr['patches']:
        x = np.array([r['patches'][k]['x'] for r in ser])
        w = np.array([r['patches'][k]['walk'] for r in ser])
        st = np.array([r['patches'][k]['stand'] for r in ser])
        l_, = ax[1].plot(ff, (x - st) - (x - st)[i0], lw=1.0, label=f'{k}: measured - standing')
        ax[1].plot(ff, (w - st) - (w - st)[i0], '--', lw=0.8, color=l_.get_color())
    ax[1].axvline(5180, color='grey', lw=0.6)
    ax[1].set_title('delivered frames: terrain patches, measured minus the STANDING line (solid)\n'
                    'against the set-off as rendered (dashed); half-size px, zeroed at 5180')
    ax[1].set_xlabel('A cut frame')
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    p2 = os.path.join(out_dir, f'{prefix}_curves.png')
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    return [p1, p2]


def summary(res):
    """the numbers a reader needs first (every one is repeated, with its method, in the sections below)"""
    pr = res['projection']
    fr_ = pr['front']
    c = pr['renderer_constants']
    tb = {r['f']: r for r in pr['table_5100_5460']}
    out = dict(
        set_off_frame=int(round(c['set_off_instant_cut_frame'])),
        set_off_definition=(
            "the front bearer (the line's front figure: lantern_s + 1.3 m) keeps a constant arc length along its path "
            "through the image sampled at this frame and moves from this frame's instant on (crossing.T_GO = 12.5 s "
            "after 4880, speed 0 there and rising linearly); the first image showing it displaced is "
            "first_displaced_frame, by 0.1 mm"),
        first_displaced_frame=fr_['first_frame_displaced'],
        first_visible_full_size_px=dict(swinging_foot_ge_1px=fr_['first_frame_foot_ge_1px'],
                                        body_path_ge_1px=fr_['first_frame_path_ge_1px'],
                                        body_path_ge_2px=fr_['first_frame_path_ge_2px']),
        acceleration=dict(
            front=dict(model='speed rises linearly from 0 at set-off to SPEED (0.45 m/s) over TAU_GO = 4.0 s '
                             '(constant 0.1125 m/s2), then holds (crossing.lantern_s)',
                       half_pace=fr_['speed_frac_ge_0p5'], pace_0p9=fr_['speed_frac_ge_0p9'],
                       full_pace=fr_['full_pace'],
                       frames_to_full_pace=fr_['full_pace'] - int(round(c['set_off_instant_cut_frame'])),
                       speed_frac_at={str(f): tb[f]['speed_frac_front'] for f in (5180, 5200, 5220, 5240, 5260, 5276)}),
            whole_line=dict(pr['line'], mean_speed_frac_at={str(f): tb[f]['line_speed_frac'] for f in
                                                            (5200, 5240, 5280, 5320, 5360, 5400, 5440)},
                            moving_frac_at={str(f): tb[f]['line_moving_frac'] for f in
                                            (5200, 5240, 5280, 5320, 5360, 5400)})),
        watch_fire_passes=[{k: w[k] for k in ('fire', 'renderer_pass_frame', 'heart_to_fire_px_at_renderer_pass',
                                              'closest_lantern_frame', 'closest_lantern_px', 'lantern_within_15px',
                                              'lantern_crosses_fire_x', 'front_bearer_crosses_fire_x')}
                           for w in pr['watch_fires']])
    fr = res.get('frames')
    if fr:
        ser = fr['series']
        ff = [r['f'] for r in ser]
        i0, i1 = ff.index(5180), ff.index(5260)
        miss = {}
        for k, v in fr['patches'].items():
            x0 = ser[i0]['patches'][k]
            x1 = ser[i1]['patches'][k]
            miss[k] = dict(
                standing=round((x1['x'] - x1['stand']) - (x0['x'] - x0['stand']), 2),
                set_off=round((x1['x'] - x1['walk']) - (x0['x'] - x0['walk']), 2),
                tracker_ok=v['min_response'] >= 0.5)
        out['frames_confirmation'] = dict(
            lantern_vs_projection_half_px=fr['lantern_vs_projection_half_px'],
            miss_at_5260_half_px_zeroed_at_5180=miss,
            onset_fits_per_patch=sorted(v['fit_onset_frame'] for v in fr['patches'].values()),
            velocity_joint_fit=fr.get('velocity_joint'),
            reading=('the delivered frames reproduce the projection (the lantern within max_abs_dev half-px of the '
                     'projected heart over 5100-5260) and, after 5180, the terrain slides past the line as the set-off '
                     'predicts and as a standing line cannot; they do not resolve the onset to a frame by themselves: '
                     'the terrain near the line moves under 1 half-size px in the first 13 frames after the set-off, '
                     'inside the trackers\' drift, and the per-patch onset fits scatter (onset_fits_per_patch)'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-frames', action='store_true')
    ap.add_argument('--evidence-dir', default='')
    ap.add_argument('--prefix', default='a18_setoff')
    a = ap.parse_args()
    t0 = time.time()
    C, sha = load_crossing()
    res = dict(generated_by='music/src/measure_ap2_crossing.py', crossing_py_sha256=sha,
               frames_dir='renders/cand_crossing_both_decal', projection=projection(C))
    print(f'projection done ({time.time() - t0:.0f} s, max RSS {_rss_mib():.0f} MiB)', flush=True)
    if not a.no_frames:
        res['frames'] = frames(C)
        print(f'frames done ({time.time() - t0:.0f} s, max RSS {_rss_mib():.0f} MiB)', flush=True)
    res = dict(summary=summary(res), **res)
    if a.evidence_dir and not a.no_frames:
        res['evidence_images'] = evidence(C, res, a.evidence_dir, a.prefix)
    res['max_rss_mib'] = round(_rss_mib(), 1)
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res['projection'][k] for k in ('front', 'line')}, indent=1))
    for p in res['projection']['watch_fires']:
        print(p)
    if 'frames' in res:
        for k, v in res['frames']['patches'].items():
            print(k, {x: v[x] for x in ('depth_m', 'used_in_joint_fit', 'fit_onset_frame', 'fit_interval',
                                          'fit_amplitude', 'pre_noise_sd_half_px', 'first_frame_beyond_3sd_of_pre_trend',
                                          'min_response')})
        print('joint', res['frames'].get('joint_fit'))
        for k, v in res['frames']['velocity'].items():
            print('v', k, v)
        print('velocity joint', res['frames'].get('velocity_joint'))
        print('lantern', res['frames']['lantern_vs_projection_half_px'])


if __name__ == '__main__':
    main()
