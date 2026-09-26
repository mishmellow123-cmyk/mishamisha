"""Camera paths with weight: C2 spline positions, heading from velocity (blended toward look
targets), bank from lateral acceleration (a coordinated turn), gentle turbulence and kicks.

Exported per frame as JSON for Blender: location, quaternion (w, x, y, z) in Blender camera
convention (camera looks down its local -Z, local +Y is up), horizontal FOV (deg).
"""
import json
import math

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

FPS = 24.0
G = 9.81


def _smooth_noise(t, seed, freq):
    """Sum of a few sines with random phases: smooth 1D noise ~[-1, 1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(t)
    amp = 0.0
    for k in range(4):
        f = freq * (1.0 + 0.9 * k) * rng.uniform(0.8, 1.25)
        a = 1.0 / (1.0 + k)
        out += a * np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi))
        amp += a
    return out / amp


def basis_from(fwd, roll):
    """Right/up vectors for a forward vector and roll angle (radians, + = bank right)."""
    fwd = fwd / np.linalg.norm(fwd)
    up0 = np.array([0.0, 0.0, 1.0])
    r = np.cross(fwd, up0)
    r /= np.linalg.norm(r)
    u = np.cross(r, fwd)
    c, s = math.cos(roll), math.sin(roll)
    r2 = r * c - u * s
    u2 = u * c + r * s
    return r2, u2


def mat_to_quat(m):
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def explicit_track(horiz, speed_keys, z_keys, f_start, s_start=0.0, fmin=None, fmax=None):
    """Positions from (1) a horizontal curve through `horiz` points (smoothed, arc-length
    parametrised), (2) speed keys [(frame, m/s)] (PCHIP) integrated from f_start where the
    arc position is s_start, (3) altitude keys [(frame, z)] (PCHIP).
    Returns waypoints [(frame, x, y, z)] every 1/4 frame for Path()."""
    H = np.asarray(horiz, float)
    # dense Catmull-Rom through the horizontal points
    pts = [H[0]]
    Hp = np.vstack([2 * H[0] - H[1], H, 2 * H[-1] - H[-2]])
    for i in range(1, len(Hp) - 2):
        p0, p1, p2, p3 = Hp[i - 1], Hp[i], Hp[i + 1], Hp[i + 2]
        for t in np.linspace(0, 1, 200)[1:]:
            t2, t3 = t * t, t * t * t
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    sk = np.asarray(speed_keys, float)
    zk = np.asarray(z_keys, float)
    vf = PchipInterpolator(sk[:, 0], sk[:, 1], extrapolate=True)
    zf = PchipInterpolator(zk[:, 0], zk[:, 1], extrapolate=True)
    fmin = sk[0, 0] if fmin is None else fmin
    fmax = sk[-1, 0] if fmax is None else fmax
    fr = np.arange(fmin, fmax + 1e-9, 0.25)
    # integrate speed (m/s -> m per frame) from f_start
    v = vf(fr) / FPS
    s = np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * 0.25)])
    s = s - np.interp(f_start, fr, s) + s_start
    x = np.interp(s, arc, pts[:, 0])
    y = np.interp(s, arc, pts[:, 1])
    # extrapolate linearly beyond the curve ends
    t_end = (pts[-1] - pts[-2]) / max(seg[-1], 1e-9)
    t_beg = (pts[1] - pts[0]) / max(seg[0], 1e-9)
    over = s > arc[-1]
    x[over] = pts[-1, 0] + t_end[0] * (s[over] - arc[-1])
    y[over] = pts[-1, 1] + t_end[1] * (s[over] - arc[-1])
    under = s < 0
    x[under] = pts[0, 0] + t_beg[0] * s[under]
    y[under] = pts[0, 1] + t_beg[1] * s[under]
    return [(f, xx, yy, zz) for f, xx, yy, zz in zip(fr, x, y, zf(fr))]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Path:
    """Weighted camera.

    waypoints : [(frame, x, y, z), ...] -> C2 cubic spline (positions are exact at the keys)
    look      : f -> (target_xyz or None, weight)   yaw/pitch pulled toward the target
    pitch_k   : fraction of the flight-path angle the camera follows (1 = nose on velocity)
    pitch_off : f -> degrees added (+ = look up)
    bank_gain : roll = bank_gain * atan(a_lat / g)
    turb      : f -> (roll, pitch, yaw) degrees of buffeting
    hfov      : f -> degrees
    """

    def __init__(self, waypoints, frames, look=None, pitch_k=0.6, pitch_off=None,
                 bank_gain=0.8, turb=None, hfov=None, yaw_off=None, roll_off=None,
                 bank_max=14.0):
        wp = np.array(waypoints, float)
        self.spl = CubicSpline(wp[:, 0], wp[:, 1:4], bc_type='natural')
        self.frames = np.asarray(frames, float)
        self.look = look
        self.pitch_k = pitch_k
        self.pitch_off = pitch_off
        self.bank_gain = bank_gain
        self.turb = turb
        self.hfov = hfov
        self.yaw_off = yaw_off
        self.roll_off = roll_off
        self.bank_max = bank_max

    def pos(self, f):
        return self.spl(f)

    def vel(self, f):          # m / s
        return self.spl(f, 1) * FPS

    def acc(self, f):          # m / s^2
        return self.spl(f, 2) * FPS * FPS

    def angles(self, f):
        p = self.pos(f)
        v = self.vel(f)
        a = self.acc(f)
        vh = math.hypot(v[0], v[1])
        yaw = math.atan2(v[0], v[1])                  # bearing (0 = north, + = east)
        gamma = math.atan2(v[2], max(vh, 1e-6))       # flight-path angle
        pitch = self.pitch_k * gamma
        if self.pitch_off is not None:
            pitch += math.radians(self.pitch_off(f))
        if self.look is not None:
            lk = self.look(f)
            tgt, w = lk[0], lk[1]
            wp = lk[2] if len(lk) > 2 else w
            if tgt is not None and (w > 0 or wp > 0):
                d = np.asarray(tgt, float) - p
                ty = math.atan2(d[0], d[1])
                tp = math.atan2(d[2], math.hypot(d[0], d[1]))
                yaw = yaw + w * _wrap(ty - yaw)
                pitch = pitch + wp * (tp - pitch)
        # bank from lateral acceleration (positive = turning right -> bank right), tasteful:
        # soft-clamped to +-bank_max degrees
        vh_ = np.array([v[0], v[1], 0.0])
        hdg = vh_ / max(np.linalg.norm(vh_), 1e-6)
        lat = np.array([hdg[1], -hdg[0], 0.0])
        roll = math.atan2(float(a @ lat), G) * self.bank_gain
        roll = math.radians(self.bank_max) * math.tanh(roll / math.radians(self.bank_max))
        if self.yaw_off is not None:
            yaw += math.radians(self.yaw_off(f))
        if self.roll_off is not None:
            roll += math.radians(self.roll_off(f))
        if self.turb is not None:
            tr, tp_, ty = self.turb(f)
            roll += math.radians(tr)
            pitch += math.radians(tp_)
            yaw += math.radians(ty)
        return p, v, yaw, pitch, roll

    def evaluate(self, sig_roll=5.0, sig_yp=1.2):
        from scipy.ndimage import gaussian_filter1d
        rows = [self.angles(f) for f in self.frames]
        yaw = np.unwrap(np.array([r[2] for r in rows]))
        pitch = np.array([r[3] for r in rows])
        roll = np.array([r[4] for r in rows])
        yaw = gaussian_filter1d(yaw, sig_yp, mode='nearest')
        pitch = gaussian_filter1d(pitch, sig_yp, mode='nearest')
        roll = gaussian_filter1d(roll, sig_roll, mode='nearest')
        out = []
        for i, f in enumerate(self.frames):
            p, v = rows[i][0], rows[i][1]
            yaw_, pitch_, roll_ = yaw[i], pitch[i], roll[i]
            fwd = np.array([math.sin(yaw_) * math.cos(pitch_), math.cos(yaw_) * math.cos(pitch_),
                            math.sin(pitch_)])
            r, u = basis_from(fwd, roll_)
            M = np.stack([r, u, -fwd], 1)
            q = mat_to_quat(M)
            hf = self.hfov(f) if self.hfov is not None else 60.0
            out.append(dict(frame=float(f), loc=p.tolist(), quat=q.tolist(), hfov=hf,
                            fwd=fwd.tolist(), right=r.tolist(), up=u.tolist(),
                            speed=float(np.linalg.norm(v)), yaw=math.degrees(yaw_),
                            pitch=math.degrees(pitch_), roll=math.degrees(roll_)))
        return out


def save(path, frames_data, meta=None):
    json.dump(dict(meta=meta or {}, frames=frames_data), open(path, 'w'), indent=0)


def load(path):
    return json.load(open(path))
