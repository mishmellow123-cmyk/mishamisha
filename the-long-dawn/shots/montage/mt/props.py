"""Beacon props (stone cairn + iron fire-basket, oil drum) as layered SDF drawings."""
import math

import numpy as np

from .figure import Drawing


def cairn(seed=1, height=1.05, base_w=1.0, top_w=0.72, basket=True, basket_h=0.42, basket_w=0.86, lit=0.0,
          logs=True):
    """Returns (back_drawing, front_drawing, fire_base_y, fire_half_width).
    back = stones + logs (drawn before the flame), front = iron basket bars (drawn after)."""
    rng = np.random.default_rng(seed)
    d = Drawing()
    courses = 4
    ch = height / courses
    y = 0.0
    for c in range(courses):
        w = base_w + (top_w - base_w) * c / (courses - 1)
        n = 4 if c < 2 else 3
        xs = np.linspace(-w / 2, w / 2, n + 1)
        off = (rng.random() - 0.5) * 0.08
        for i in range(n):
            cx = 0.5 * (xs[i] + xs[i + 1]) + off + (rng.random() - 0.5) * 0.03
            rx = 0.5 * (xs[i + 1] - xs[i]) * (0.98 + rng.random() * 0.12)
            ry = ch * (0.52 + rng.random() * 0.12)
            d.new_group()
            d.ellipse((cx, y + ry * 0.95), rx, ry, ang=(rng.random() - 0.5) * 0.25, mat=7)
        y += ch * 0.98
    top = y
    fb = top + 0.08
    if logs:
        for i in range(5):
            d.new_group()
            a = (rng.random() - 0.5) * 0.9
            L = basket_w * (0.45 + 0.2 * rng.random())
            cx = (rng.random() - 0.5) * 0.25
            cy = top + 0.12 + 0.08 * i / 5
            dx = math.cos(a) * L / 2
            dy = math.sin(a) * L / 2 * 0.4
            d.capsule((cx - dx, cy - dy), (cx + dx, cy + dy), 0.055, 0.05, mat=8)
    front = Drawing()
    if basket:
        bw0 = basket_w * 0.45
        bw1 = basket_w * 0.5
        yb0 = top - 0.02
        yb1 = top + basket_h
        nb = 7
        for i in range(nb):
            u = -1 + 2 * i / (nb - 1)
            front.new_group()
            p0 = (u * bw0 * 0.8, yb0)
            p1 = (u * bw1 * 1.02, yb1 + 0.03 * (1 - abs(u)))
            front.capsule(p0, p1, 0.013, 0.011, mat=5)
        front.new_group()
        front.capsule((-bw1 * 1.02, yb1), (bw1 * 1.02, yb1), 0.015, 0.015, mat=5)
        front.new_group()
        front.capsule((-bw0 * 0.82, yb0 + 0.02), (bw0 * 0.82, yb0 + 0.02), 0.014, 0.014, mat=5)
        front.new_group()
        front.capsule((-bw0 * 0.9, yb0 + basket_h * 0.5), (bw0 * 0.9, yb0 + basket_h * 0.5), 0.010, 0.010, mat=5)
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
