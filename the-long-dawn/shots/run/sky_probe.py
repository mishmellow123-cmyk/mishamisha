"""A2 cloud-deck isolation, native sky crops and measured edge statistics.

Run from the-long-dawn with one process. --mode native renders only the sky
at native sampling (no terrain marcher); --mode half renders one actual A212
terrain plate at 960x402, then reuses that identical terrain for comparisons.
Outputs never go to renders. Native crops are 700x400 pixels, not enlargements.
The before/after sky crops include the master finish but no terrain or captions
outside the crop. They are a layer study, not farm-delivered replacement frames.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest import mock

for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE_REF = 'ea7cdf385a4751b8be4ae596b11cc05af9d907b1'
sys.path.insert(0, str(HERE))
import falsedawn as FD

cv2.setNumThreads(1)


def pixels(a):
    return np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)


def save(path, a):
    if not cv2.imwrite(str(path), a[..., ::-1]):
        raise OSError(path)


def stats(rgb):
    """Fixed ROI x[0,700), y[40,400); RGB code-value luminance, not physical light.

    Absolute first differences detect the thin horizontal marks as large
    vertical derivatives. Second differences emphasize one-row disturbances.
    No claim that these alone distinguish natural texture from aliasing.
    """
    luma = rgb[40:400, :700].astype(np.float64) @ np.array([.2126, .7152, .0722])
    dx, dy = np.diff(luma, axis=1), np.diff(luma, axis=0)
    return dict(mean_abs_dx=float(np.abs(dx).mean()), mean_abs_dy=float(np.abs(dy).mean()),
                p99_abs_dy=float(np.percentile(np.abs(dy), 99)),
                rms_dyy=float(np.sqrt(np.mean(np.diff(luma, n=2, axis=0)**2))),
                vertical_edges_gt_2=int(np.count_nonzero(np.abs(dy) > 2)),
                vertical_edge_count=int(dy.size))


def sky(frame, candidate='accepted', ss=1.5, scale=1., no_mw=False):
    local = frame - FD.CUT0
    fr = FD.PI.Frame(FD.camera(local, round(1920*scale), round(804*scale)), ss)
    shape = (fr.src.H, fr.src.W)
    fr.dist = np.full(shape, 1.e9, np.float32)
    fr.zb = fr.dist
    fr.img = np.zeros((*shape, 3), np.float32)
    trans = np.zeros(shape, np.float32)
    _, gp = FD.light(local, 'arc')
    gp = FD.sky_parameters(gp, candidate)
    if no_mw:
        gp = gp.copy()
        gp[26] = 0.
    skl0, skld, skl = FD.skyline(fr.src.pos)
    FD.sky_pass(fr.img, fr.dist, trans, fr.src.params(), gp, FD.milky_way(),
                skl0, skld, skl, fr.src.pos[1], local/24., np.zeros((0, 6)))
    # Opacity statistics use transmittance before splatting; RGB uses the
    # actual star catalogue, whose visibility changes with cloud coverage.
    if FD._STARS is None:
        FD._STARS = FD.SK.make_stars(16000, 101, lum_scale=6.)
    FD.SK.splat_stars(fr.img, fr.src, FD._STARS, trans, t=local/24.,
                      gain=ss*ss, scale=FD.PI.src_scale(fr))
    # RC.warp_maps builds float64 ray arrays over the entire supersampled
    # target. Evaluate the identical homography in 32-output-row strips after
    # the owner-night memory failure; no native 3x full-frame ray mesh.
    hdr = np.empty((fr.t_out.H, fr.t_out.W, 3), np.float32)
    alpha = np.empty((fr.t_out.H, fr.t_out.W), np.float32)
    for y0 in range(0, fr.t_out.H, 32):
        y1 = min(y0+32, fr.t_out.H)
        v, u = np.mgrid[round(y0*ss):round(y1*ss), :fr.t_ss.W].astype(np.float64)
        d = fr.t_ss.ray(u+.5, v+.5)
        c = d @ fr.src.fwd
        mx = (fr.src.cx + fr.src.f*(d @ fr.src.right)/c-.5).astype(np.float32)
        my = (fr.src.cyy-fr.src.f*d[...,1]/c-.5).astype(np.float32)
        hdr[y0:y1] = cv2.resize(FD.RC.warp(fr.img, mx, my), (fr.t_out.W,y1-y0),
                               interpolation=cv2.INTER_AREA)
        alpha[y0:y1] = cv2.resize(FD.RC.warp(trans, mx, my), (fr.t_out.W,y1-y0),
                                 interpolation=cv2.INTER_AREA)
    return hdr, alpha


def master_context(scale):
    sys.path.insert(0, str(ROOT/'edit'))
    import assemble as AS
    AS._init('A', None, scale, True, True)
    return AS, AS._CTX.take_frame


def master_from(ctx, take_frame, rgb, frame):
    # Only the selected A2 plate is substituted in memory. The EDL, finish,
    # floor ramp and titles still run through the actual master pipeline.
    ctx.take_frame = lambda take, f: rgb.copy() if f == frame else take_frame(take, f)
    try:
        return ctx.frame(frame)
    finally:
        ctx.take_frame = take_frame


def native(out):
    AS, take_frame = master_context(1.)
    rows = []
    for frame in (212, 260):
        delivered = AS._CTX.frame(frame)
        save(out/f'sky_delivered_{frame}_crop.png', delivered[:400, :700])
        alphas = {}
        for candidate in ('accepted', 'clear_high_deck'):
            hdr, alpha = sky(frame, candidate)
            source = pixels(FD.PI.look.finish(hdr, **FD.FINISH))
            alphas[candidate] = alpha
            finished = master_from(AS._CTX, take_frame, source.astype(np.float32)/255, frame)
            save(out/f'sky_{candidate}_{frame}_crop.png', finished[:400, :700])
            row = dict(frame=frame, candidate=candidate, source_stats=stats(source), master_stats=stats(finished))
            rows.append(row)
            print(json.dumps(row), flush=True)
            del hdr, finished
        # No-deck sky is the control for alpha: glow kills stars equally.
        opacity = 1-alphas['accepted']/np.maximum(alphas['clear_high_deck'], 1.e-12)
        roi = opacity[40:400, :700]
        rows.append(dict(frame=frame, layer='isolated_deck_opacity', pixels=int(roi.size),
                         fraction_opacity_gt_half=float(np.mean(roi > .5)),
                         max_opacity=float(roi.max()),
                         mean_abs_dx=float(np.abs(np.diff(roi, axis=1)).mean()),
                         mean_abs_dy=float(np.abs(np.diff(roi, axis=0)).mean())))
        save(out/f'sky_deck_opacity_{frame}_crop.png', np.repeat(pixels(opacity[:400, :700])[...,None],3,axis=2))
        if frame == 212:
            # Double source sampling; true broad cloudlets should persist.
            hdr_hi, _ = sky(frame, ss=3.)
            source_hi = pixels(FD.PI.look.finish(hdr_hi, **FD.FINISH))
            save(out/'sky_accepted_212_ss3_crop.png', source_hi[:400,:700])
            rows.append(dict(frame=frame, candidate='accepted_ss3', source_stats=stats(source_hi)))
            del hdr_hi, source_hi
        del alphas
        gc.collect()
    return rows


def half(out):
    frame = 212
    local = frame-FD.CUT0
    AS, take_frame = master_context(.5)
    # Capture the one actual terrain render. Reuse identical buffers in every
    # route so the default-equality test does not pay for terrain repeatedly.
    render_terrain = FD.PI.render_terrain
    captured = {}
    def capture(fr, *args, **kwargs):
        result = render_terrain(fr, *args, **kwargs)
        captured.update(img=fr.img.copy(), zb=fr.zb.copy(), dist=fr.dist.copy())
        return result
    def reuse(fr, *args, **kwargs):
        fr.img, fr.zb, fr.dist = (captured[k].copy() for k in ('img','zb','dist'))
        return fr
    # Baseline is the literal pre-change render function at this lane's base,
    # executed with the same unchanged dependencies; no git metadata writes.
    source = subprocess.check_output(['git','show',BASELINE_REF+':the-long-dawn/shots/run/falsedawn.py'], text=True)
    tree = ast.parse(source)
    old_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'render')
    namespace = dict(FD.__dict__)
    exec(compile(ast.Module(body=[old_fn], type_ignores=[]), '<baseline falsedawn.render>', 'exec'), namespace)
    with mock.patch.object(FD.PI, 'render_terrain', capture):
        original = namespace['render'](local, scale=.5, ss=1.5)
    rows = []
    with mock.patch.object(FD.PI, 'render_terrain', reuse):
        default = FD.render(local, scale=.5, ss=1.5)
        if not np.array_equal(original, default):
            raise AssertionError('Omitted renderer default changed the original HDR')
        rows.append(dict(frame=frame, original_default_exact_equal=True,
                         baseline_ref=BASELINE_REF,
                         hdr_sha256=hashlib.sha256(default.tobytes()).hexdigest()))
        for candidate in ('accepted', 'clear_high_deck'):
            hdr = default if candidate == 'accepted' else FD.render(local, scale=.5, ss=1.5, sky_candidate=candidate)
            rgb = pixels(FD.PI.look.finish(hdr, **FD.FINISH)).astype(np.float32)/255
            final = master_from(AS._CTX, take_frame, rgb, frame)
            save(out/f'sky_{candidate}_{frame}_half.png', final)
            if candidate != 'accepted':
                changed = np.any(default != hdr, axis=2)
                # Pure terrain: require all warped neighbours to be non-sky;
                # exclude a 2-pixel guard at the skyline before comparing HDR.
                fr = FD.PI.Frame(FD.camera(local, 960, 402), 1.5)
                fr.img, fr.zb, fr.dist = (captured[k] for k in ('img','zb','dist'))
                _, _, distance = FD.PI.to_target(fr)
                terrain = cv2.erode((distance < 1.e8).astype(np.uint8), np.ones((5,5), np.uint8)) > 0
                if changed[terrain].any():
                    raise AssertionError('Candidate changed terrain HDR away from the skyline')
                rows.append(dict(frame=frame, protected_terrain_pixels=int(terrain.sum()),
                                 changed_protected_terrain_pixels=int(changed[terrain].sum()),
                                 changed_hdr_pixels=int(changed.sum())))
    return rows


def sampling(out):
    # Direct scalar probe of the shared noise's fractional-octave boundary.
    # This is diagnosis only: a shared-noise fix would change other shots.
    values = []
    for octaves in (.499999, .500001, .999999, 1., 1.000001, 1.999999, 2., 2.000001):
        values.append(dict(octaves=octaves, fbm=float(FD.fbm2(.317, .619, octaves, 171))))
    rows = [dict(noise_probe_coordinates=[.317,.619], seed=171, values=values)]
    # Read geometric limits across the actual native ROI. The cloud projection
    # compresses world-space detail vertically towards the horizon.
    cam = FD.camera(132)
    y, x = np.mgrid[40:400, :700]
    d = cam.ray(x+.5, y+.5)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    dy = d[...,1]
    rows.append(dict(frame=212, roi=[0,40,700,400],
                     min_elevation_deg=float(np.degrees(np.arcsin(dy.min()))),
                     max_elevation_deg=float(np.degrees(np.arcsin(dy.max())))))
    for frame in (80, 127, 212, 260, 500, 559):
        pair = []
        for candidate in ('accepted','clear_high_deck'):
            hdr, _ = sky(frame, candidate, scale=.5)
            rgb = pixels(FD.PI.look.finish(hdr,**FD.FINISH))
            pair.append(rgb[:200])
        save(out/f'sky_timeline_{frame}_half.png', np.concatenate(pair,axis=1))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('native','half','sampling'), required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    forbidden = (ROOT/'renders').resolve()
    if out == forbidden or forbidden in out.parents:
        parser.error('Evidence must not be written inside renders')
    out.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    rows = globals()[args.mode](out)
    receipt = dict(mode=args.mode, rows=rows, seconds=time.perf_counter()-start,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/f'sky_measurements_{args.mode}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
