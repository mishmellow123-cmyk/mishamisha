"""B14 . THE HAND-BACK (R11), key still: the sixtieth dawn, from behind her.

Her summit, the beacon, the sixty-stone cairn and two small figures seen from behind (the keeper, old, standing; the
child asleep against her in the red scarf), looking ESE past the nearer shoulder (the 347 m peak, 2.3 km out) toward
the sunrise. The sun is breaking the far skyline beside the shoulder: the far ranges are already in gold light and
each beacon there has paled to a shimmer (a fire pales by its own sun clearance, the same march as DUSK); the
shoulder's shadow still lies across the cloud sea toward her, so the nearer beacons and hers still burn. Same world
and set as the lifetime and DUSK (keeper.CR_B); the sun and sky are dawn.py's (a clean limb-darkened disc, a tight
gold aureole, blue air away from the sun).

  python shots/run/handback.py --scale 0.25          # look-dev
  python shots/run/handback.py --scale 0.5 --ss 1.5  # the key still
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import world as WD          # noqa: E402
import rcam as RC           # noqa: E402
import pipe as PI           # noqa: E402
import keeper as KP         # noqa: E402
import dawn as DW           # noqa: E402
import dusk as DK           # noqa: E402
import fire2 as F2          # noqa: E402
from mt import fire as F, figure as FG, props as PR   # noqa: E402
from mt.noise import smoothstep   # noqa: E402

CM = PI.CM
look = PI.look
CR = KP.CR_B


def _find_shoulder():
    """The nearer shoulder: the 347 m summit ~2.3 km ESE of her top (hill-climb from the coarse catalogue hit)."""
    x, z = KP.S_SUM[0] + 1920.0, KP.S_SUM[2] - 1260.0
    h = WD.h_rock(x, z, 2.0, CR)
    step = 20.0
    for _ in range(60):
        imp = False
        for dx, dz in ((step, 0), (-step, 0), (0, step), (0, -step)):
            hh = WD.h_rock(x + dx, z + dz, 2.0, CR)
            if hh > h:
                h, x, z, imp = hh, x + dx, z + dz, True
        if not imp:
            step *= 0.5
            if step < 0.5:
                break
    return np.array([x, h, z])


SHOULDER = _find_shoulder()
_d = SHOULDER - KP.STAND
VIEW_AZ = math.degrees(math.atan2(_d[0], _d[2]))          # ~123: her line of sight to the shoulder
CAM_BACK = 46.0                                           # metres behind her
CAM_SIDE = -4.5                                           # metres to her right (+) / left (-) of that line
CAM_UP = 1.3
HFOV = 44.0
PITCH = -0.6
YAW_OFF = 3.5


def camera(W, H):
    f = KP._dirxz(VIEW_AZ)
    r = np.array([f[2], 0.0, -f[0]])
    pos = KP.STAND - f * CAM_BACK + r * CAM_SIDE + np.array([0.0, CAM_UP, 0.0])
    to = KP.STAND + f * 30.0 - pos
    yaw = math.degrees(math.atan2(to[0], to[2])) + YAW_OFF
    return RC.RCam(pos, yaw, PITCH, 0.0, HFOV, W, H)


def skyline_el(az_deg, eye):
    """Elevation (deg) of the terrain + cloud skyline from `eye` toward azimuth az (incl. curvature), B's set."""
    a = math.radians(az_deg)
    d = np.array([math.sin(a), math.cos(a)])
    best = -10.0
    for r in np.geomspace(60.0, 90000.0, 900):
        x, z = eye[0] + d[0] * r, eye[2] + d[1] * r
        h = max(WD.ground(x, z, CR, r / 2000.0), WD.h_cloud(x, z, r / 2000.0, 0.0)) - r * r / (2 * WD.R_EARTH)
        best = max(best, math.degrees(math.atan2(h - eye[1], r)))
    return best


def sun_setup():
    """Place the sun at the shoulder's right edge on the skyline, as seen from her: the disc half behind rock."""
    best = None
    for da in np.linspace(0.5, 7.0, 40):
        az = VIEW_AZ + da
        el = skyline_el(az, KP.STAND + np.array([0, 1.6, 0]))
        if best is None:
            best = (az, el)
        if el < best[1] - 0.05:
            best = (az, el)
            break
    return best


def far_beacons(scam, dist):
    """Summits in this frame (1.2-45 km) visible from the camera, each a beacon for the sixtieth night."""
    cand = []
    for r in np.geomspace(1200.0, 45000.0, 64):
        for a in np.linspace(-24, 24, 40):
            d = KP._dirxz(scam_yaw(scam) + a)
            cand.append((scam.pos[0] + d[0] * r, scam.pos[2] + d[2] * r, r))
    peaks = []
    for x, z, r in cand:
        fp = r / 600.0
        step = max(r * 0.012, 40.0)
        bx, bz = x, z
        bh = WD.h_rock(bx, bz, fp, CR)
        for _it in range(12):
            imp = False
            for ddx, ddz in ((step, 0), (-step, 0), (0, step), (0, -step)):
                h = WD.h_rock(bx + ddx, bz + ddz, fp, CR)
                if h > bh:
                    bh, bx, bz, imp = h, bx + ddx, bz + ddz, True
            if not imp:
                step *= 0.5
        if bh > WD.CLOUD_Y + 120:
            peaks.append((bx, bh, bz))
    keep = []
    for p in peaks:
        dd = math.hypot(p[0] - scam.pos[0], p[2] - scam.pos[2])
        if all(math.hypot(p[0] - q[0], p[2] - q[2]) > max(0.03 * dd, 120) for q in keep):
            keep.append(p)
    P = np.array(keep)
    dd = np.linalg.norm(P - scam.pos, axis=1)
    curv = ((P[:, 0] - scam.pos[0]) ** 2 + (P[:, 2] - scam.pos[2]) ** 2) / (2 * WD.R_EARTH)
    Pc = P.copy()
    Pc[:, 1] = P[:, 1] - curv + 3.0
    sx, sy, z = scam.project(Pc)
    ok = []
    for i in range(len(P)):
        ix, iy = int(sx[i]), int(sy[i])
        if 2 <= ix < scam.W - 2 and 2 <= iy < scam.H - 2 and z[i] > 0:
            if dist[iy - 1:iy + 2, ix - 1:ix + 2].max() > dd[i] * 0.97:
                ok.append(i)
    return Pc[ok], P[ok], dd[ok]


def scam_yaw(scam):
    return math.degrees(math.atan2(scam.fwd[0], scam.fwd[2]))


def render(scale=0.25, ss=1.5):
    W, H = int(round(1920 * scale)), int(round(804 * scale))
    tc = camera(W, H)
    fr = PI.Frame(tc, ss)
    scam = fr.src
    C = scam.params()
    P = np.array([scam.pos[0], scam.pos[2], 0.0, 0.0])
    D = np.zeros((scam.H, scam.W))
    WD.march(P, CR, C, 0.2, 90000.0, 0.0035, 0.35, 900.0, 9, D)
    # the sun: at the shoulder's right edge, its lower half still behind the rock
    saz, sel = sun_setup()
    sel = sel + 0.06
    e, a = math.radians(sel), math.radians(saz)
    sd = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
    lin = CM.lin
    sc = np.array([1.0, 0.50, 0.22])
    Lk = np.r_[sd, sc]
    amb = lin('#5577BB') * 0.15
    SD = np.zeros(24)
    SD[0:3] = sd
    SD[3:6] = sc
    SD[6] = 900.0
    SD[7] = math.radians(0.27)
    SD[8:11] = lin('#2F5AA8') * 0.55
    SD[11:14] = lin('#FFBE78') * 0.95
    SD[14:17] = lin('#A9A6CE') * 0.36
    SD[17] = 1.3
    SD[18] = 0.045
    SD[19] = 0.034
    SD[20] = 0.55
    fogc = lin('#6F83B8') * 0.30
    fogp = np.array([5.0e-5, 1 / 1500.0, 2.0e-4, 1 / 150.0, 1.2, fogc[0], fogc[1], fogc[2]])
    QD = np.zeros(20)
    QD[0] = 2.3
    QD[1] = 2.4
    QD[2] = 8.0
    QD[3] = 0.28
    QD[4] = 0.95
    QD[5] = 1.5
    QD[6] = 90.0
    QD[7] = 45000.0
    QD[8] = 0.35
    QD[9] = 0.10
    QD[10] = 0.45
    QD[11:14] = lin('#FFC58A') * 0.8
    QD[14] = 30.0
    QD[15] = 40.0
    QD[16] = 22.0
    out = np.zeros((scam.H, scam.W, 3), np.float32)
    zb = np.zeros((scam.H, scam.W), np.float32)
    di = np.zeros((scam.H, scam.W), np.float32)
    DW.dawn_shade(C, D, P, CR, SD, Lk, QD, amb, fogp, out, zb, di)
    img = out
    t = 60.0
    # --- the beacons of the sixtieth night, each pale where the sun already reaches it
    Pc, Pw, dd = far_beacons(scam, di)
    sx, sy, z = scam.project(Pc)
    nlit = 0
    for k in range(len(Pc)):
        curv = ((Pw[k, 0] - P[0]) ** 2 + (Pw[k, 2] - P[1]) ** 2) / (2 * WD.R_EARTH)
        c = DK.clearance(P, CR, Pw[k, 0], Pw[k, 1] + 4.0 - curv, Pw[k, 2], sd[0], sd[1], sd[2], 2.0, 48, 90000.0)
        sun = smoothstep(-0.003, 0.004, c)
        nlit += sun > 0.5
        fl = F.flicker(t + k, k)
        en = 9.0 * fl * (0.45 + 0.55 * min(2500.0 / dd[k], 1.0)) * ss * ss * (1.0 - 0.95 * sun)
        F2.glow(img, zb, sx[k], sy[k], 0.8 * ss, en, z=z[k], zbias=z[k] * 0.02, col=np.array([1.0, 0.52, 0.18]))
        F2.halo(img, zb, sx[k], sy[k], 3.0 * ss, 0.03 * (1.0 - 0.95 * sun), z=z[k], zbias=z[k] * 0.02,
                col=np.array([1.0, 0.45, 0.12]))
    # --- her fire (still in the shoulder's shadow), the cairn, the two figures
    fire_I = 1.8
    lights = [dict(dir=sd, col=sc * 0.0, I=0.0), dict(pos=KP.BEACON + np.array([0, 1.3, 0]), col=F.FIRE_LIGHT,
                                                        I=fire_I, r0=0.5)]
    amb_f = amb * 2.2
    FG.render(img, zb, scam, KP.stone_cairn(60), KP.CAIRN, lights, amb=amb_f, mats=KP.M, t=t, write_depth=True,
              zbias=0.3)
    back, front, fb, rb = PR.cairn(seed=9, height=0.85, base_w=1.2, top_w=0.95, basket=True, basket_h=0.5,
                                   basket_w=1.1)
    FG.render(img, zb, scam, back, KP.BEACON, lights, amb=amb_f, t=t, emissive_gain=0.9, write_depth=True,
              zbias=0.3)
    base = KP.BEACON + np.array([0.0, fb, 0.0])
    F2.flame(img, zb, scam, base, 1.0, 0.42, t, seed=4, I=12.0, lean=0.2, zbias=0.5, tongues=5, warp=1.2)
    bx, by, bz = scam.project(base + np.array([0, 0.6, 0]))
    F2.halo(img, zb, bx, by, 5.0 * scam.f / bz, 0.004, z=bz, zbias=3.0)
    FG.render(img, zb, scam, front, KP.BEACON, lights, amb=amb_f, t=t, write_depth=False, zbias=0.3)
    f = KP._dirxz(VIEW_AZ)
    r = np.array([f[2], 0.0, -f[0]])
    herp = KP.on_ground(KP.BEACON - r * 1.35 - f * 0.5)          # at the dawn she stands left of her fire, facing it
    kid, _ = KP.person('look', 0.0, child=True, scarf=True, face=-0.3, wind=0.6)
    FG.render(img, zb, scam, kid, KP.on_ground(herp + r * 0.55 - f * 0.1), lights, amb=amb_f, mats=KP.M, t=t,
              write_depth=False, zbias=0.3)
    her, _ = KP.person('look', 1.0, stick=True, face=0.0, scarf=False)
    FG.render(img, zb, scam, her, herp, lights, amb=amb_f, mats=KP.M, t=t, write_depth=False, zbias=0.3)
    fr.img, fr.zb, fr.dist = img, zb, di
    out, _, _ = PI.to_target(fr)
    print(f'sun az {saz:.2f} el {sel:.2f}; beacons {len(Pc)}, in sun {nlit}', flush=True)
    return out


FINISH = dict(exposure=0.72, bloom_strength=0.08, bloom_threshold=1.4, streak_strength=0.003, vignette_amount=0.22)


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--ss', type=float, default=1.5)
    ap.add_argument('--out', default='handback')
    a = ap.parse_args()
    out = os.path.join(KP.OUTDIR, a.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    img = look.finish(render(a.scale, a.ss), **FINISH)
    p = os.path.join(out, f'handback_{a.scale:.2f}.png')
    look.save_png(p, img)
    print(p, f'{time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
