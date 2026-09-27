"""B's shared props and the child (RUN-B-3), drawn with bfig.render, so the vigil and the hand-back show the SAME
cairn and the SAME child.

  cairn3(seed)                    the counting cairn: irregular faceted stones (bfig STONE, no internal seams), each
                                  its own shaded body with snow on its upper faces, dark gaps between them, a
                                  ragged top of stones on end, a drift of snow at its foot. Same footprint as
                                  bset.rubble_cairn (H 1.62, R 1.05), base centre at the origin.
  child3(wake, reach, hold, ...)  the traveller's child sitting against her (she is screen-LEFT of the child),
                                  wrapped head and shoulders in HER red shawl: a distinct head (large, a child's)
                                  on a small body, the shawl over the head and falling as a draped triangle down the
                                  back, the child's own dark coat below, legs out in front. Asleep the head is bowed
                                  and rests toward her; waking it lifts and turns to the sun.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_MT = os.path.join(os.path.dirname(HERE), 'montage')          # mt (the shared puppet kit)
if _MT not in sys.path:
    sys.path.insert(0, _MT)

import bfig as BF                      # noqa: E402
from mt import figure as FG            # noqa: E402

_CAIRN = {}


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def cairn3(seed=3, H=1.62, R=1.05):
    key = (seed, H, R)
    if key in _CAIRN:
        return _CAIRN[key]
    rng = np.random.default_rng(seed)
    d = FG.Drawing()
    # the dark body of the heap: seen only through the gaps between the stones
    d.new_group()
    d.trap((0.0, 0.02), (0.0, H * 0.74), R * 0.62, R * 0.10, rnd=0.04, k=0.0, mat=BF.GAP)
    mats = (16, 17, 23)
    n = 0
    y = 0.0
    # courses from the bottom up: big flat stones low, smaller and more tilted higher; the profile a worn cone
    while y < H * 0.80:
        uu = y / H
        half = R * (1.0 - uu) ** 0.9 * (0.93 + 0.12 * rng.random())
        r = 0.19 * (1.0 - 0.40 * uu) * (0.85 + 0.30 * rng.random())
        x = -half + r * 1.1
        while x < half - r * 0.8:
            rr = r * (0.75 + 0.5 * rng.random())
            el = 1.25 + 0.6 * rng.random()
            tilt = (rng.random() - 0.5) * (0.35 + 0.5 * uu)
            cy = y + rr * (0.95 + 0.2 * rng.random())
            d.new_group()
            BF.add_stone(d, (x + rng.normal(0, 0.02), cy), rr, elong=el, ang=tilt, facet=0.65 + 0.3 * rng.random(),
                         seed=seed * 1000 + n, mat=int(mats[rng.integers(0, 3)]))
            n += 1
            x += rr * el * (1.80 + 0.22 * rng.random())                   # a narrow gap between neighbours
        y += r * (1.45 + 0.20 * rng.random())
    # the ragged top: two or three stones on end, off-centre, leaning; one small flat stone wedged beside them
    top = y
    for j, (dx, lean, rr, el) in enumerate(((-0.10, 0.28, 0.11, 0.62), (0.08, -0.22, 0.09, 0.58),
                                           (0.20, -0.45, 0.075, 0.9))):
        if j == 2 and rng.random() < 0.35:
            continue
        d.new_group()
        BF.add_stone(d, (dx + rng.normal(0, 0.015), top + rr * 0.9), rr, elong=el, ang=math.pi / 2 + lean,
                     facet=0.85, seed=seed * 1000 + 900 + j, mat=int(mats[rng.integers(0, 3)]))
    # a drift of old snow at its foot, over the lowest stones' bases
    d.new_group()
    d.ellipse((0.05, 0.02), R * 1.05, 0.07, k=0.0, mat=BF.SNOW, fuzz=0.012, ff=10.0)
    d.ellipse((-R * 0.55, 0.05), R * 0.40, 0.08, ang=0.12, k=0.05, mat=BF.SNOW, fuzz=0.01, ff=12.0)
    _CAIRN[key] = d
    return d


def child3(wake=0.0, reach=0.0, hold=0.0, lean_to=-1.0):
    """Returns (Drawing, pts): pts['hand'] (palm centre, local metres, or None), pts['head'], pts['steel']
    (where the steel lies in the small hand once it is given)."""
    d = FG.Drawing()
    w = _ease(wake)
    lean = 0.34 * (1.0 - w)                              # asleep: the whole upper body rests toward her
    s = lean_to                                          # -1: she is screen-left
    fz, ff = 0.0, 44.0                                   # wool: a clean soft edge (fuzz made scratch lines)

    def rot(p, a, piv=(0.0, 0.16)):
        c, sn = math.cos(a), math.sin(a)
        x, y = p[0] - piv[0], p[1] - piv[1]
        return np.array([piv[0] + c * x - sn * y, piv[1] + sn * x + c * y])
    a = lean * (-s)                                      # leans toward her side
    # the child's own dark coat: the seat and the lower back (legs out in front, hidden)
    d.new_group()
    d.ellipse((0.0, 0.07), 0.19, 0.08, k=0.06, mat=14)                                          # the seat
    d.trap((0.0, 0.05), rot((0.0, 0.27), a * 0.5), 0.17, 0.12, rnd=0.05, k=0.06, mat=14)        # the lower back
    # the head: a child's (large for the body), bowed when asleep, lifting toward the sun when awake
    bow = 0.035 * (1.0 - w)
    head = rot((0.0 - 0.015 * w * (-s), 0.525 - bow + 0.015 * w), a * 1.1)
    # the red shawl: over the head, round the neck, over sloping shoulders and down the back to a point
    d.new_group()
    d.ellipse(head, 0.086, 0.096, ang=a, k=0.0, mat=13, fuzz=fz, ff=ff)                         # the covered head
    d.trap(rot((0.0, 0.38), a), head + np.array([0.0, -0.03]), 0.12, 0.075, rnd=0.02, k=0.05, mat=13)  # the wrap
    d.ellipse(rot((0.0, 0.370), a), 0.150, 0.055, ang=a, k=0.05, mat=13, fuzz=fz, ff=ff)          # the shoulders
    d.tri(rot((-0.14, 0.365), a), rot((0.14, 0.365), a), rot((0.02 * s, 0.12), a), rnd=0.025, k=0.05, mat=13,
          fuzz=fz, ff=ff)                                                                      # the point down the back
    # the weave: one darker band across the back; a short ragged fringe at the point
    d.new_group()
    d.capsule(rot((-0.10, 0.29), a), rot((0.10, 0.29), a), 0.009, 0.009, mat=20)
    for q in (-0.04, 0.0, 0.04):
        p0 = rot((0.02 * s + q * 0.8, 0.135), a)
        d.capsule(p0, p0 + np.array([0.003, -0.04]), 0.005, 0.0035, mat=13)
    # a small gloved hand from the wrap toward her, palm up (reach); then closed on the steel at the chest (hold)
    hand = None
    steel = None
    r_ = _ease(reach)
    h_ = _ease(hold)
    if r_ > 0.0 or h_ > 0.0:
        d.new_group()
        shp = rot((0.10 * s, 0.35), a)
        out = shp + np.array([0.12 * s + 0.06 * s * r_, -0.12 + 0.03 * r_])
        back = rot((0.05 * s, 0.29), a)
        hand = out + (back - out) * h_
        d.capsule(shp, hand, 0.042, 0.030, k=0.04, mat=13)                                      # the shawl-wrapped arm
        d.ellipse(hand + np.array([0.022 * s, 0.004]), 0.028, 0.018, ang=0.2 * s, k=0.02, mat=19)   # mitten, palm up
        steel = hand + np.array([0.024 * s, 0.020])
    return d, dict(head=head, hand=hand, steel=steel)
