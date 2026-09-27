"""EMBERS-C2 look-dev: a contact sheet of one of c2's shots WITHOUT the 230 MB tower geometry (post layers, own
particles, the claw, the Ring), fast enough for the Mac's render queue at 0.4 scale (~1-3 s a frame after the numba
warm-up). Full tests with the towers go to the farm (cloud/jobs/embers_C3_{eye,grasp,fall}.json --test).
    python3 ~/mishamisha/_local_logs/renderq.py -- python c2lab.py SHOT OUT.jpg SCALE f1,f2,...  [--towers]
SHOT: eye | grasp | fall"""
import sys, os, time
import numpy as np, cv2
E = os.path.expanduser('~/mishamisha/the-long-dawn/shots/embers')
sys.path.insert(0, os.path.join(E, '..', '..', 'lib')); sys.path.insert(0, E)
import look, variant
variant.set_cut('C3')
from core import Frame
import render as RD
shot, out, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
frames = [int(x) for x in sys.argv[4].split(',')]
towers = '--towers' in sys.argv
import c2
c2.ENABLED = {shot}
if shot == 'eye':
    import c_eye as M
    sc = M.EyeShot()
    if not towers:
        class NoTw:
            k_all = 0
            def prepare(self, ctx): pass
            def emit(self, *a): pass
            def height(self, i, t): return 0.0
        sc._cache['towers'] = NoTw()
        sc._cache['tsmoke'] = NoTw(); sc._cache['tembers'] = NoTw()
elif shot == 'grasp':
    import c_grasp as M
    sc = M.GraspShot(lab=not towers)
else:
    import c_fall as M
    sc = M.FallShot()
tiles = []
for f in frames:
    t0 = time.time()
    s0, s1 = [s for s in RD.SHOTS_V3['C3'] if s[0] <= f < s[1]][0]
    fr = Frame(scale)
    ctx = RD.Ctx()
    ctx.f, ctx.t, ctx.fr, ctx.scale = f, float(f), fr, scale
    ctx.t0, ctx.t1 = max(f - 0.25, s0), min(f + 0.25, s1 - 0.02)
    ctx.cam0, ctx.cam1, ctx.cam = sc.camera(ctx.t0), sc.camera(ctx.t1), sc.camera(float(f))
    fr.set(focus=ctx.cam.focus, aperture=ctx.cam.aperture, **sc.render_opts(f))
    sc.emit(ctx)
    hdr = fr.resolve()
    hdr = sc.post(ctx, hdr)
    img = look.finish(np.nan_to_num(hdr), **sc.finish_opts(f))
    im = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    cv2.putText(im, '%d  %.1fs' % (f, time.time() - t0), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    tiles.append(im)
    print(f, '%.1fs' % (time.time() - t0), flush=True)
cols = 2
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(out)
