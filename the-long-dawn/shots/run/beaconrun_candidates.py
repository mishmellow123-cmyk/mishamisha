"""Default-off A14/A15 look development: a leftward relay of resolved flames.

The accepted shot modules stay intact. ``render`` without an explicit candidate
delegates directly to them; the scoped candidate restores every shared binding,
including on failure. One process renders one frame at a time (the farm adapter
already owns that contract). This is art direction, not a fire-size simulation.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import math
import os

import numpy as np

SYNC = (3960, 4000, 4040, 4080, 4120, 4160, 4200)
OPTION = 'linked-fires'
YAW_OFFSET = 18.5
# Ground intersections measured with world.march + world.ground in this
# worktree, on the candidate cameras at run_1..6. The prescribed half-size
# ground pixels were (590,140), (510,175), (430,210), (350,240), (270,275),
# (195,300). The seventh hearth remains the accepted world-space hearth.
ANCHORS = np.array([
    [-1175.138066660323, -252.90742546280285, 3805.151643363935],
    [-1395.0647276780207, -332.5611960737974, 3969.7055863528126],
    [-1009.1594536198359, -181.7898155024842, -312.70104483344693],
    [-1031.3803450517462, -192.47587485490038, -299.6447878678196],
    [-1068.7386711864353, -209.2532885910669, -223.39096560464935],
    [-1090.503385021237, -218.43892044550867, -216.6579398963458],
], dtype=np.float64)


def modules():
    import beaconrun_a as B
    import watchers_a as W
    return B, W


def candidate_chain(accepted):
    chain = np.asarray(accepted, np.float64).copy()
    if chain.shape != (7, 6):
        raise ValueError('The relay needs exactly seven fire rows')
    chain[:6, :3] = ANCHORS
    chain[:, 3] = SYNC
    # Early tinder answers the desert cut at +16f; its larger flare still
    # lands on run_1 at 3960. All later ignitions retain the score anchors.
    chain[0, 3] = 3936.
    chain[:6, 4] = 1.
    chain[6, 4] = 1.25
    return chain


def turned(camera, R):
    return R.RCam(camera.pos, camera.yaw_d + YAW_OFFSET, camera.pitch_d,
                  camera.roll_d, camera.hfov_d, camera.W, camera.H)


def pool_weight(distance):
    """Continuous irradiance rolloff in world metres; no clipped screen box."""
    return np.exp(-np.square(np.asarray(distance) / 2.2))


@contextmanager
def linked_scene():
    B, W = modules()
    import fire2 as F2
    from mt import fire as F
    # Look-dev environment overrides would invalidate the measured anchor
    # positions. Reject them rather than silently publish a different staging.
    keys = ('A14_CAM', 'A14_LABELS', 'A14_NOUG', 'NIGHT_CLOUD',
            'W15_P7', 'W15_CAM', 'W15_OWN_CAM')
    if any(os.environ.get(k) for k in keys):
        raise ValueError('Clear A14/W15 look-dev overrides for linked-fires')
    original_plate = B._PLATE
    plate = B.end_cam()
    B._PLATE = original_plate
    chain = candidate_chain(B.CHAIN)
    # In the turned composition the near watcher is on the left. Face the
    # first relay fire across the frame; the accepted glow heading would look
    # out of frame. The cached plate below still fixes both sides of the join.
    look_az = math.degrees(math.atan2(chain[0, 0] - chain[6, 0], chain[0, 2] - chain[6, 2]))
    old_camera, old_watch_camera = B.camera, W.camera
    old_fires, old_shade = B.NA.fires_layer, B.WD.shade
    old_lighting = W.hearth_lighting
    cam = lambda f, width=1920, height=804: turned(old_camera(f, width, height), B.RC)
    watch_cam = lambda f, width=1920, height=804: turned(old_watch_camera(f, width, height), B.RC)

    # Fix each distant emitter's dimensions in WORLD space from its catch
    # camera. A moving camera then changes apparent size naturally; no
    # screen-locked sprite resizing. Local lighting retains watch-fire energy.
    dimensions = []
    with patch.object(B, '_PLATE', plate):
        for row, frame in zip(chain[:6], SYNC):
            c = cam(frame)
            depth = float(c.project(row[:3])[2])
            dimensions.append((150. * depth / c.f, 40. * depth / c.f))

    def spot(P7):
        u = np.asarray(P7[:3]) - plate[0]
        u[1] = 0.
        u /= np.linalg.norm(u)
        right = np.cross(W.UP, u)
        p = np.asarray(P7[:3]).copy() - u * .95 + right * .06
        p[1] = W.ground(p[0], p[2])
        return p, W._dir(W.LOOK_AZ)

    def lighting(LT, PL, P7):
        LT, PL = old_lighting(LT, PL, P7)
        # The new hearth's own stones expose its fuel; omit the cleared-snow
        # material patch, whose height cutoff can outline a lit shelf.
        return LT, PL[np.hypot(PL[:, 0] - P7[0], PL[:, 2] - P7[2]) >= .01]

    def shade(C, D, P, CR, S, LT, Lk, Q, amb, fogp, out, zb, dist, PL):
        old_shade(C, D, P, CR, S, LT, Lk, Q, amb, fogp, out, zb, dist, PL)
        near = np.hypot(LT[:, 0] - chain[6, 0], LT[:, 2] - chain[6, 2]) < .01
        if not near.any():
            return
        unlit = np.empty_like(out)
        old_shade(C, D, P, CR, S, LT[~near], Lk, Q, amb, fogp, unlit, zb, dist, PL)
        # Recover the actual source-space terrain hit from its forward depth;
        # taper only this hearth's illumination, preserving moon and far fires.
        j, i = np.ogrid[:out.shape[0], :out.shape[1]]
        u, v = (i + .5 - C[8]) / C[7], (C[9] - j - .5) / C[7]
        dx = C[0] + zb * (C[3] + C[5] * u) - chain[6, 0]
        dy = C[1] + zb * v - chain[6, 1]
        dz = C[2] + zb * (C[4] + C[6] * u) - chain[6, 2]
        weight = pool_weight(np.sqrt(dx * dx + dy * dy + dz * dz))
        out[:] = unlit + (out - unlit) * weight[..., None]

    def fires(img, zb, scam, rows, f, pxs=1., **kwargs):
        ids = {int(row[5]): k for k, row in enumerate(chain[:6])}
        rows = np.asarray(rows)
        # Flames absorb RGB but do not write opaque depth. Sort the complete
        # table so a distant tongue cannot absorb an already drawn near hearth.
        for index in np.argsort(-np.linalg.norm(rows[:, :3] - scam.pos, axis=1)):
            row = rows[index]
            if int(row[5]) not in ids:
                old_fires(img, zb, scam, rows[index:index + 1], f, pxs, **kwargs)
                continue
            k = ids[int(row[5])]
            x, y, z, fi, size, seed = row
            if f < fi - 1.5:
                continue
            sz, intensity, light = F.ignite_env(f / 24., fi / 24.)
            # At each score marker the body has caught, rather than spending
            # that exact frame at the accepted envelope's half-size start.
            sz = max(sz, .92) if f >= fi else sz
            height, radius = dimensions[k]
            base = np.array([x, y + .15, z])
            sx, sy, depth = scam.project(base)
            if depth <= 1.:
                continue
            tr = B.NA.fire_trans(scam.pos, base, kwargs['fogp'], B.WD)
            bias = B.NA.fire_depth_bias(depth, True)
            F2.flame(img, zb, scam, base, height * sz, radius * sz, f / 24.,
                     seed=int(seed) * 7 + 3, I=30. * intensity * tr,
                     lean=.09 * height, zbias=bias, tongues=7)
            F2.halo(img, zb, sx, sy - .3 * height * scam.f / depth,
                    24. * pxs, .045 * light * tr, z=depth, zbias=bias, col=F2.AURA_COL)

    def flares(img, zb, scam, f, pxs, fogp):
        for k, fi in enumerate(SYNC):
            env = B.flare_env(f - fi + 2.)
            if env <= 0.:
                continue
            row = chain[k]
            height = dimensions[k][0] if k < 6 else B.NA.fire_dims(row[4])[0]
            sx, sy, depth = scam.project(row[:3] + np.array([0., .3 * height, 0.]))
            if depth > 1.:
                tr = B.NA.fire_trans(scam.pos, row[:3], fogp, B.WD)
                F2.halo(img, zb, sx, sy, 26. * pxs, .16 * env * tr,
                        z=depth, zbias=B.NA.fire_depth_bias(depth, True), col=F2.AURA_COL)

    with ExitStack() as stack:
        for obj, key, value in ((B, '_PLATE', plate), (B, 'CHAIN', chain), (B, 'FIRES', chain),
                                (W, 'LOOK_AZ', look_az),
                                (B, 'camera', cam), (W, 'camera', watch_cam),
                                (W, 'seventh_spot', spot), (W, 'hearth_lighting', lighting),
                                (B.WD, 'shade', shade), (B.NA, 'fires_layer', fires),
                                (B.NA, '_SMOKE', {}), (B.NA, '_SPARKS', {}),
                                (B, 'draw_flares', flares)):
            stack.enter_context(patch.object(obj, key, value))
        yield chain


def render(f, kind='beaconrun', candidate='accepted', scale=1., ss=1.5):
    if kind not in ('beaconrun', 'watchers') or candidate not in ('accepted', OPTION):
        raise ValueError('Unknown shot or candidate')
    low, high = (3920, 4239) if kind == 'beaconrun' else (4240, 4399)
    if not low <= f <= high:
        raise ValueError('Frame outside the requested shot')
    if not math.isfinite(scale) or not 0. < scale <= 1.:
        raise ValueError('Scale must be finite, positive and at most native')
    B, W = modules()

    def draw():
        if kind == 'beaconrun':
            return B.PI.look.finish(B.render(f, scale, ss)[0], **B.FINISH)
        return W.finish(W.render(f, scale, ss))

    if candidate == 'accepted':
        return draw()
    with linked_scene():
        return draw()
