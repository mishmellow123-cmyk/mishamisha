"""Cut D optical flare confined to the opening, shared with adjacent D shots.

The cap silhouette supplies the aperture size. No whole-band bounds enter this
calculation. Coordinates are world space; the result is additive linear HDR.
"""
import numpy as np


def render(camera, width, height, ends, cap_outline, score, pulse=0., visibility=None):
    out = np.zeros((height, width, 3), np.float32)
    if score.gain <= 0.:
        return out
    x, y, _ = camera.project(np.asarray(ends), width, height)
    cx, cy = x.mean(), y.mean()
    bx, by, _ = camera.project(np.asarray(cap_outline), width, height)
    chord = np.array([x[1] - x[0], y[1] - y[0]])
    chord /= max(np.linalg.norm(chord), 1e-6)
    across = np.array([-chord[1], chord[0]])
    boundary = np.column_stack([bx - cx, by - cy])
    fringe = score.radius_px * width / 1920.
    # The opening is oblique. Its long axis joins the ends; its short axis is
    # the thickness of the cut faces. A circle also covered the rear band.
    radius = np.abs(boundary @ chord).max() + fringe
    minor = np.abs(boundary @ across).max() + .5 * fringe
    yy, xx = np.ogrid[:height, :width]
    dx, dy = xx - cx, yy - cy
    distance = np.hypot(dx, dy)
    along = dx * chord[0] + dy * chord[1]
    cross = dx * across[0] + dy * across[1]
    rho = np.hypot(along / max(radius, 1e-6), cross / max(minor, 1e-6))
    pulse = float(np.clip(pulse, 0., 1.))
    bloom = np.exp(-1.25 * rho ** 6)
    core = np.exp(-.5 * (distance / max(minor * .24, .7)) ** 2)
    # Finite optical support prevents the old fog from veiling the far arc.
    bloom *= np.clip((1.30 - rho) / .18, 0., 1.)
    streak = np.exp(-np.abs(dx) / max(radius * 1.1, 1.))
    streak = streak * np.exp(-.5 * (dy / max(.8, height / 804.)) ** 2)
    energy = score.gain * ((.32 + 1.18 * pulse) * bloom + (2.4 + 5.6 * pulse) * core
                           + (.008 + .012 * pulse) * streak)
    if visibility is not None:
        energy *= visibility
    out[:] = energy[..., None] * np.array([1., .965, .86])
    return out
