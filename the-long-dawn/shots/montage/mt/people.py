"""Character builders (2D SDF puppets). Units: metres, origin between the feet, x right on screen.

Each builder returns a Drawing plus named points (hands, torch head...) in figure-local space.
Poses are plain parameters so shots can keyframe them with easing.
"""
import math

import numpy as np

from .figure import Drawing


def lerp(a, b, t):
    return np.asarray(a, np.float64) * (1 - t) + np.asarray(b, np.float64) * t


def rot(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def limb(root, a1, l1, a2, l2):
    """Two-segment limb hanging from root. Angles from straight down, +ve swings toward +x."""
    root = np.asarray(root, np.float64)
    j1 = root + np.array([math.sin(a1), -math.cos(a1)]) * l1
    j2 = j1 + np.array([math.sin(a1 + a2), -math.cos(a1 + a2)]) * l2
    return j1, j2


# ------------------------------------------------------------------ shepherd ---

def shepherd(turn=0.0, arm=0.0, lean=0.0, head_up=0.0, breath=0.0, coat_pts=None, shawl_pts=None,
             step=0.0, recoil=0.0):
    """Older man, long wool coat, blanket-shawl, knitted cap, torch in right hand.
    turn: 0 = back to camera, 1 = right profile.  arm: 0 low hold (hidden in front of body),
    0.5 raised, 1 thrust forward-right into the basket.  Returns (drawing, pts)."""
    d = Drawing()
    d.new_group()
    T = turn
    H = 1.78
    # --- legs + boots (profile: stride; back: side by side)
    stride = 0.05 + 0.13 * T + 0.08 * step
    hipL = np.array(lerp((-0.10, 0.90), (-0.02, 0.90), T))
    hipR = np.array(lerp((0.10, 0.90), (0.03, 0.90), T))
    ankL = np.array(lerp((-0.13, 0.09), (-stride, 0.09), T))
    ankR = np.array(lerp((0.13, 0.09), (stride + 0.02, 0.09), T))
    for hp, ak in ((hipL, ankL), (hipR, ankR)):
        kn = 0.5 * (hp + ak) + np.array([0.02 * T, 0.0])
        d.capsule(hp, kn, 0.095, 0.078, k=0.04)
        d.capsule(kn, ak, 0.078, 0.066, k=0.03)
    bl = 0.085 + 0.06 * T
    d.ellipse(ankL + np.array([0.03 * T, -0.03]), bl, 0.065, k=0.03)
    d.ellipse(ankR + np.array([0.05 * T, -0.03]), bl, 0.065, k=0.03)
    # --- torso frame (lean rotates about the hips)
    hip = np.array([0.0 + 0.02 * T, 0.92])
    up = rot(np.array([0.0, 1.0]), -lean)
    side = rot(np.array([1.0, 0.0]), -lean)
    chest = hip + up * (0.40 + 0.005 * breath)
    neck = hip + up * 0.58
    sw = 0.235 * (1 - T) + 0.085 * T            # shoulder half-width seen by camera
    shL = chest + side * (-sw) + up * 0.06
    shR = chest + side * (sw + 0.03 * T) + up * 0.06
    # coat: long, slightly flared; profile is deeper at the back (shawl) than the front
    hem_y = 0.33
    cx_off = -0.02 * T
    hem_w = 0.33 * (1 - T) + 0.25 * T
    d.trap(np.array([cx_off, hem_y]) + (hip - np.array([0, 0.92])) * 0.2, chest + up * 0.02 + side * (cx_off - 0.02 * T),
           hem_w, sw + 0.01, rnd=0.035, k=0.05, fuzz=0.010, ff=18.0)
    # coat hem flap (verlet points supplied by the shot), trailing on the windward side
    if coat_pts is not None and len(coat_pts) > 1:
        d.chain(coat_pts, 0.05, 0.02, k=0.06)
    # belly / chest volume in profile
    if T > 0.05:
        d.ellipse(hip + up * 0.28 + side * (0.05 * T), 0.10 * T + 0.06, 0.24, ang=-lean, k=0.06)
    # shawl / blanket over the shoulders (fuzzy fringe)
    d.trap(chest + up * (-0.30) + side * (-0.03 * T), chest + up * 0.08, sw + 0.075, sw + 0.02, rnd=0.04, k=0.05,
           mat=1, fuzz=0.012, ff=55.0)
    if shawl_pts is not None and len(shawl_pts) > 1:
        d.chain(shawl_pts, 0.06, 0.025, k=0.05, mat=1, fuzz=0.01, ff=50.0)
    # shoulders
    d.ellipse(chest + up * 0.05, sw + 0.05, 0.12, ang=-lean, k=0.06, mat=1, fuzz=0.008, ff=50.0)
    # --- arms
    # left arm hangs (in profile it is mostly hidden behind the body)
    aL1 = -0.12 * (1 - T) + 0.25 * T
    eL, wL = limb(shL, aL1, 0.29, -0.08 + 0.3 * T, 0.27)
    d.capsule(shL, eL, 0.072, 0.06, k=0.04)
    d.capsule(eL, wL, 0.06, 0.05, k=0.03)
    d.ellipse(wL + np.array([0.0, -0.04]), 0.045, 0.055, k=0.02, mat=2)
    # right (torch) arm: low hold -> raised -> thrust forward-right into the basket.
    # angles: (upper arm from straight down, elbow bend relative); +ve swings toward +x.
    a_low = (0.12, -0.75)       # forearm across the front of the body (hidden from behind)
    a_up = (1.05, 0.70)         # raised, torch up
    a_thr = (1.30, 0.28)        # reaching forward-down, elbow slightly bent
    if arm < 0.5:
        u = arm / 0.5
        aR1 = a_low[0] + (a_up[0] - a_low[0]) * u
        aR2 = a_low[1] + (a_up[1] - a_low[1]) * u
    else:
        u = (arm - 0.5) / 0.5
        aR1 = a_up[0] + (a_thr[0] - a_up[0]) * u
        aR2 = a_up[1] + (a_thr[1] - a_up[1]) * u
    aR1 -= 0.30 * recoil
    aR2 += 0.35 * recoil
    # engaged shoulder: it comes forward and up with the reach
    eng = min(max(arm - 0.3, 0.0) / 0.7, 1.0)
    shR = shR + side * (0.05 * eng) + up * (0.03 * eng)
    upper = 0.29 * (0.72 + 0.28 * max(T, min(arm * 1.4, 1.0)))   # foreshortened while reaching away
    eR, wR = limb(shR, aR1, upper, aR2, 0.27 * (0.75 + 0.25 * max(T, min(arm * 1.4, 1.0))))
    d.capsule(shR, eR, 0.074, 0.062, k=0.05)
    d.capsule(eR, wR, 0.062, 0.05, k=0.04)
    fdir = (wR - eR) / (np.linalg.norm(wR - eR) + 1e-9)
    hand = wR + fdir * 0.05
    d.ellipse(hand, 0.05, 0.045, k=0.02, mat=2)
    # torch direction: upright in the low hold, up when raised, forward-down into the basket
    v_low = np.array([0.12, 1.0])
    v_up = np.array([0.45, 1.0])
    v_thr = rot(fdir, -0.42)
    if arm < 0.5:
        tdir = lerp(v_low, v_up, arm / 0.5)
    else:
        tdir = lerp(v_up / np.linalg.norm(v_up), v_thr, (arm - 0.5) / 0.5)
    tdir = tdir / (np.linalg.norm(tdir) + 1e-9)
    fs = 0.42 + 0.58 * max(T, min(arm / 0.5, 1.0))           # foreshortening while pointing away
    t0 = hand - tdir * 0.10 * fs
    t1 = hand + tdir * 0.56 * fs
    d.capsule(t0, t1, 0.018, 0.022, mat=4)
    d.ellipse(t1, 0.034, 0.05, ang=math.atan2(tdir[1], tdir[0]) - math.pi / 2, mat=8)
    # --- neck, head, cap
    d.capsule(neck - up * 0.04, neck + up * 0.06, 0.06, 0.055, k=0.03, mat=2)
    hc = neck + up * 0.13 + rot(np.array([0.025 * T, 0.0]), -lean) + np.array([0.0, 0.01 * head_up])
    d.ellipse(hc, 0.096 - 0.004 * T, 0.118, ang=-lean + 0.15 * head_up * (1 - T), k=0.02, mat=2)
    # coat collar
    d.capsule(neck + side * (-0.09) - up * 0.02, neck + side * (0.09) - up * 0.02, 0.055, 0.055, k=0.04,
              mat=1, fuzz=0.006, ff=60)
    # knitted cap
    capc = hc + up * 0.065 + side * (-0.012 * T)
    d.ellipse(capc, 0.104, 0.074, ang=-lean + 0.2 * T, k=0.01, mat=1, fuzz=0.004, ff=70)
    d.ellipse(capc - up * 0.045, 0.108, 0.03, ang=-lean + 0.1 * T, k=0.005, mat=1)
    # profile features: nose, beard
    if T > 0.3:
        s = (T - 0.3) / 0.7
        nose0 = hc + side * (0.085) + up * 0.005
        d.tri(nose0 + up * 0.03, nose0 + side * (0.035 * s) - up * 0.01, nose0 - up * 0.03, rnd=0.004, k=0.01, mat=2)
        d.ellipse(hc + side * (0.05 * s) - up * 0.085, 0.07 * s + 0.01, 0.06 * s + 0.01, k=0.03, mat=6,
                  fuzz=0.008, ff=80)
    # hair tufts below the cap (back view)
    d.ellipse(hc - up * 0.04 + side * (-0.02 * T), 0.1, 0.05, k=0.02, mat=6, fuzz=0.01, ff=90)
    pts = dict(hand=hand, torch_head=t1, torch_dir=tdir, head=hc, chest=chest, shR=shR, wR=wR)
    return d, pts


# ------------------------------------------------------------ generic figure ---

def body_basic(d, hip, lean, sw, legs, arms, head_r=(0.095, 0.115), neck_len=0.14, torso_len=0.52,
               leg_r=(0.08, 0.062, 0.05), arm_r=(0.062, 0.05, 0.042), mat=0, k=0.04):
    """Minimal body. legs: [(hip_off, knee, ankle)], arms: [(shoulder, elbow, wrist)] in local coords."""
    up = rot(np.array([0.0, 1.0]), -lean)
    for hp, kn, ak in legs:
        d.capsule(hp, kn, leg_r[0], leg_r[1], k=k, mat=mat)
        d.capsule(kn, ak, leg_r[1], leg_r[2], k=k, mat=mat)
    for sh, el, wr in arms:
        d.capsule(sh, el, arm_r[0], arm_r[1], k=k, mat=mat)
        d.capsule(el, wr, arm_r[1], arm_r[2], k=k, mat=mat)
    return up


# ------------------------------------------------------------------ desert robe ---

def robed(arm=0.0, lean=0.0, tail_pts=None, hem_pts=None, sleeve_pts=None, turn=0.35):
    """Tall figure in a long robe and head-wrap with a trailing cloth tail. Three-quarter view
    facing right (+x). arm: 0 torch raised high, 0.6 lowering, 1 torch in the basket (right).
    Returns (drawing, pts)."""
    d = Drawing()
    d.new_group()
    up = rot(np.array([0.0, 1.0]), -lean)
    side = rot(np.array([1.0, 0.0]), -lean)
    hip = np.array([0.0, 0.95])
    chest = hip + up * 0.42
    neck = hip + up * 0.60
    # robe: long, falls to the ankles, flares; the hem trails downwind (-x)
    d.trap((-0.03, 0.03), chest + up * 0.02, 0.30, 0.19, rnd=0.03, k=0.06, fuzz=0.01, ff=16.0)
    if hem_pts is not None and len(hem_pts) > 1:
        d.chain(hem_pts, 0.07, 0.03, k=0.08)
    # feet peeking out
    d.ellipse((0.12, 0.03), 0.09, 0.035, k=0.02)
    d.ellipse((-0.10, 0.03), 0.08, 0.035, k=0.02)
    # shoulders + mantle
    d.ellipse(chest + up * 0.07, 0.21, 0.11, ang=-lean, k=0.06)
    # left arm (rear) hangs, partly hidden
    shL = chest + up * 0.06 - side * 0.16
    eL, wL = limb(shL, -0.15, 0.29, 0.25, 0.26)
    d.capsule(shL, eL, 0.075, 0.065, k=0.05)
    d.capsule(eL, wL, 0.065, 0.055, k=0.04)
    # right arm: holds the torch high, then lowers it into the basket
    shR = chest + up * 0.07 + side * 0.15
    a_hi = (2.75, 0.15)     # nearly straight up
    a_lo = (1.25, 0.25)     # forward-down into the basket
    aR1 = a_hi[0] + (a_lo[0] - a_hi[0]) * arm
    aR2 = a_hi[1] + (a_lo[1] - a_hi[1]) * arm
    eR, wR = limb(shR, aR1, 0.29, aR2, 0.27)
    d.capsule(shR, eR, 0.07, 0.06, k=0.05)
    d.capsule(eR, wR, 0.06, 0.05, k=0.04)
    if sleeve_pts is not None and len(sleeve_pts) > 1:
        d.chain(sleeve_pts, 0.05, 0.02, k=0.06)
    fdir = (wR - eR) / (np.linalg.norm(wR - eR) + 1e-9)
    hand = wR + fdir * 0.05
    d.ellipse(hand, 0.045, 0.045, k=0.02, mat=2)
    tdir = lerp(np.array([0.1, 1.0]), rot(fdir, -0.35), min(max((arm - 0.3) / 0.7, 0.0), 1.0))
    tdir = tdir / (np.linalg.norm(tdir) + 1e-9)
    t1 = hand + tdir * 0.5
    d.capsule(hand - tdir * 0.08, t1, 0.017, 0.02, mat=4)
    d.ellipse(t1, 0.03, 0.045, ang=math.atan2(tdir[1], tdir[0]) - math.pi / 2, mat=8)
    # head + wrap (a bulky turban with the veil across the face) and the trailing tail
    d.capsule(neck - up * 0.03, neck + up * 0.07, 0.07, 0.065, k=0.03)
    hc = neck + up * 0.15 + side * 0.02
    d.ellipse(hc, 0.11, 0.125, ang=-lean, k=0.02)
    d.ellipse(hc + up * 0.06 - side * 0.01, 0.125, 0.085, ang=-lean + 0.1, k=0.03, mat=1, fuzz=0.004, ff=60)
    if tail_pts is not None and len(tail_pts) > 1:
        d.chain(tail_pts, 0.045, 0.02, k=0.04, mat=1)
    return d, dict(hand=hand, torch_head=t1, head=hc, neck=neck, tail_anchor=hc - side * 0.08 + up * 0.02,
                   hem_anchor=np.array([-0.30, 0.10]))
