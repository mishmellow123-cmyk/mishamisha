"""Beacon props (stone cairn + iron fire-basket, oil drum) as layered SDF drawings."""
import math

import numpy as np

from .figure import Drawing


def cairn(seed=1, height=1.05, base_w=1.05, top_w=0.70, basket=True, basket_h=0.40, basket_w=0.86, logs=True):
    """Dry-stone cairn: irregular flattened angular slabs in uneven, tapering courses, over a dark
    core whose chinks glow (emissive) once the fire is lit, topped by an iron fire-basket.
    Returns (back_drawing, front_drawing, fire_base_y, fire_half_width).
    back = core + stones + logs (drawn before the flame), front = basket bars (after)."""
    rng = np.random.default_rng(seed)
    d = Drawing()
    # core behind the stones: shows through the chinks (mat 12 = chink glow)
    d.new_group()
    d.trap((0, 0.02), (0, height - 0.02), base_w * 0.47, top_w * 0.47, rnd=0.01, mat=12)
    y = 0.0
    while y < height - 0.05:
        u = y / height
        w = base_w + (top_w - base_w) * u
        ch = 0.075 + 0.075 * rng.random()          # course height (uneven)
        ch = min(ch, height - y)
        x = -w / 2 + (rng.random() - 0.5) * 0.04
        while x < w / 2 - 0.04:
            sw = 0.14 + 0.30 * rng.random() ** 1.5
            sw = min(sw, w / 2 - x + 0.02)
            cx = x + sw / 2
            hb = ch * (0.85 + 0.25 * rng.random())
            yy = y + (rng.random() - 0.5) * 0.02
            tilt = (rng.random() - 0.5) * 0.14
            wb = sw * 0.5 * (0.96 + 0.08 * rng.random())
            wt = wb * (0.78 + 0.25 * rng.random())
            d.new_group()
            p0 = (cx - math.sin(tilt) * hb * 0.5, yy + 0.004)
            p1 = (cx + math.sin(tilt) * hb * 0.5, yy + hb - 0.004)
            d.trap(p0, p1, wb - 0.008, wt - 0.008, rnd=0.012, mat=7, fuzz=0.006, ff=26.0)
            x += sw + 0.008 + 0.012 * rng.random()
        y += ch
    top = y
    fb = top + 0.07
    if logs:
        # crossed firewood stacked in the basket, a few ends poking above the rim
        for i in range(6):
            d.new_group()
            a = (rng.random() - 0.5) * 1.4 + (0.9 if i % 2 else -0.9) * 0.6
            L = basket_w * (0.55 + 0.25 * rng.random())
            cx = (rng.random() - 0.5) * 0.28
            cy = top + 0.10 + 0.07 * i
            dx = math.cos(a) * L / 2
            dy = math.sin(a) * L / 2
            d.capsule((cx - dx, cy - dy), (cx + dx, cy + dy), 0.045, 0.038, mat=8)
    front = Drawing()
    if basket:
        r0 = basket_w * 0.24
        r1 = basket_w * 0.52
        yb0 = top - 0.03
        yb1 = top + basket_h
        nb = 7
        for i in range(nb):
            u = -1 + 2 * i / (nb - 1)
            front.new_group()
            pa = (u * r0, yb0)
            pm = (u * (r0 + (r1 - r0) * 0.72), yb0 + basket_h * 0.55)
            pb = (u * r1, yb1)
            front.capsule(pa, pm, 0.012, 0.011, mat=5)
            front.capsule(pm, pb, 0.011, 0.010, mat=5)
        front.new_group()
        front.capsule((-r1 - 0.01, yb1), (r1 + 0.01, yb1), 0.014, 0.014, mat=5)
        front.new_group()
        front.capsule((-r0 - 0.01, yb0 + 0.01), (r0 + 0.01, yb0 + 0.01), 0.014, 0.014, mat=5)
    return d, front, fb, basket_w * 0.42


def drum(seed=2, h=0.88, r=0.29):
    """Steel oil drum with a grate on top (city rooftop beacon)."""
    d = Drawing()
    d.new_group()
    d.trap((0, 0.0), (0, h), r, r, rnd=0.01, mat=11)
    for y in (0.3 * h, 0.66 * h):
        d.new_group()
        d.capsule((-r - 0.005, y), (r + 0.005, y), 0.012, 0.012, mat=11)
    front = Drawing()
    front.new_group()
    front.capsule((-r - 0.01, h + 0.01), (r + 0.01, h + 0.01), 0.016, 0.016, mat=5)
    for i in range(5):
        u = -0.8 + 1.6 * i / 4
        front.new_group()
        front.capsule((u * r, h - 0.02), (u * r * 1.05, h + 0.06), 0.008, 0.008, mat=5)
    return d, front, h + 0.02, r * 0.9


def local_to_world(cam, feet_world, X, Y, Z=0.0):
    """Figure-local (x right on screen, y up, z toward camera) -> world."""
    P = np.asarray(feet_world, np.float64)
    return P + cam.right * X + np.array([0.0, Y, 0.0]) - cam.fwd * Z
