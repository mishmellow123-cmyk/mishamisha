"""A15 cairn geometry comparisons; accepted shot, camera, light and material code stay unchanged.

Run from the-long-dawn (one process/output directory per variant):
  python -B shots/run/studies/cairn_shapes.py --variant fractured-fieldstone --out /absolute/study/fieldstone
  python -B shots/run/studies/cairn_shapes.py --variant packing-only --frames 4280,4360 --out /absolute/study/packing

Requires the normal RUN renderer: NumPy, Numba, OpenCV, Pillow and its checked-in scene arrays.
The variants are proposals, not approved replacements. They do not change material 5's fixed shader.

Visual result at A15 frame 4360, half resolution: none of the three proposals is adopted. Packing-only
exposes floating gaps; fractured-fieldstone reads as stepped masonry; uneven-fieldstone removes the
courses but still reads as rounded rectangular chunks. These are diagnostic controls for a future
stone-shape study, not a completed cairn improvement. Further placement tweaks alone have not been
shown to solve the shape problem. The accepted baseline remains unchanged.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
import watchers_a as WA
import sdfppl as SP

BASELINE = WA.hearth_scene


def primitive_bounds(rows):
    """Conservative bounds for the cone/rounded-box primitives used by these studies."""
    lower, upper = [], []
    for row in rows:
        if row[0] == 0:
            lower.append(np.minimum(row[1:4] - row[7], row[4:7] - row[8]))
            upper.append(np.maximum(row[1:4] + row[7], row[4:7] + row[8]))
        elif row[0] == 1:
            cy, sy, cp, sp = math.cos(row[4]), math.sin(row[4]), math.cos(row[5]), math.sin(row[5])
            rotation = np.array([[cy, -sy * sp, sy * cp], [0, cp, sp], [-sy, -cy * sp, cy * cp]])
            extent = np.abs(rotation) @ row[11:14]
            lower.append(row[1:4] - extent)
            upper.append(row[1:4] + extent)
        else:
            raise ValueError(f'unexpected study primitive: {row[0]}')
    return np.min(lower, axis=0), np.max(upper, axis=0)


def baseline_bounds(seed, scale=1.0):
    scene = SP.Scene()
    BASELINE(scene, np.zeros(6), seed, scale)
    return primitive_bounds(scene.arrays()[0])


def packing_only(sc, P, seed, scale=1.0):
    """Keep all 21 capsule shapes; stagger their placement and expose hard seams between stones.

    This is a packing control: no new planar faces or material detail are introduced.
    """
    original = SP.Scene()
    BASELINE(original, np.zeros(6), seed, scale)
    rows, _ = original.arrays()
    rng = np.random.default_rng(int(seed) + 4101)
    for row in rows:
        centre = 0.5 * (row[1:4] + row[4:7])
        az = math.atan2(centre[2], centre[0]) + rng.uniform(-0.45, 0.45)
        radius = np.linalg.norm(centre[[0, 2]]) * rng.uniform(0.80, 1.08)
        delta = np.array([radius * math.cos(az) - centre[0], rng.uniform(-0.045, 0.045) * scale,
                          radius * math.sin(az) - centre[2]])
        row[1:4] += delta
        row[4:7] += delta
        row[10] = 0.0
    # Keep the fire seated at the same height; the lower stones remain partly embedded in the ground.
    rows[:, [2, 5]] += baseline_bounds(seed, scale)[1][1] - primitive_bounds(rows)[1][1]
    base = np.asarray(P[:3], np.float64)
    sc.begin(rgb=(0.050, 0.047, 0.045))
    for row in rows:
        sc.cone(base + row[1:4], base + row[4:7], row[7], row[8], mat=5, k=0.0)
    sc.end()


def fractured_fieldstone(sc, P, seed, scale=1.0):
    """Unequal flat stones with broken outlines, assembled from pairs of overlapping rounded boxes.

    Broad faces replace the capsule lobes. Each pair forms one stone; all unions are hard. The unequal
    thicknesses and tier offsets did not conceal the courses in the rendered comparison: this remains
    a rejected masonry-like control, despite the nonuniform primitive dimensions.
    """
    rng = np.random.default_rng(int(seed) + 6203)
    local = SP.Scene()
    local.begin(rgb=(0.050, 0.047, 0.045))
    phase = rng.uniform(0, 2 * math.pi)
    # Height, footprint radius and count. These unequal levels still read as masonry in the render.
    for tier, (height, radius, count) in enumerate(((0.045, 0.365, 8), (0.185, 0.290, 6),
                                                  (0.335, 0.215, 5), (0.505, 0.105, 3))):
        widths = rng.uniform(0.7, 1.3, count)
        angles = phase + np.cumsum(widths) / widths.sum() * (2 * math.pi)
        phase += 1.43
        for az in angles:
            r = radius * rng.uniform(0.80, 1.06)
            centre = np.array([r * math.cos(az), height + rng.uniform(-0.024, 0.024), r * math.sin(az)])
            length = rng.uniform(0.125, 0.185) * (1.0 - 0.08 * tier)
            half = np.array([length, rng.uniform(0.058, 0.081), rng.uniform(0.066, 0.108)])
            yaw = -(az + math.pi / 2) + rng.uniform(-0.35, 0.35)
            pitch = rng.uniform(-0.20, 0.20)
            local.box(centre, half, yaw=yaw, pitch=pitch, rnd=0.015, mat=5, k=0.0)
            axis = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
            side = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            fragment = centre + axis * length * rng.uniform(0.22, 0.40) + side * half[2] * rng.uniform(-0.25, 0.25)
            fragment[1] += rng.uniform(-0.012, 0.012)
            local.box(fragment, half * rng.uniform([0.56, 0.82, 0.67], [0.76, 1.04, 0.94]),
                      yaw=yaw + rng.uniform(0.23, 0.48), pitch=pitch + rng.uniform(-0.13, 0.13),
                      rnd=0.012, mat=5, k=0.0)
    local.end()
    rows, _ = local.arrays()
    ref_lower, ref_upper = baseline_bounds(seed)
    lower, upper = primitive_bounds(rows)
    # Match the original heap's horizontal span; a smaller hearth would confound the shape comparison.
    footprint_scale = np.max((ref_upper - ref_lower)[[0, 2]]) / np.max((upper - lower)[[0, 2]])
    rows[:, [1, 3, 11, 13]] *= footprint_scale
    rows[:, 2] += ref_upper[1] - primitive_bounds(rows)[1][1]
    base = np.asarray(P[:3], np.float64)
    sc.begin(rgb=(0.050, 0.047, 0.045))
    for row in rows:
        sc.box(base + row[1:4] * scale, row[11:14] * scale, yaw=row[4], pitch=row[5],
               rnd=row[7] * scale, mat=5, k=0.0)
    sc.end()


def uneven_fieldstone(sc, P, seed, scale=1.0):
    """A compact heap with continuously staggered heights and tilted, unequal field stones.

    The golden-angle packing has no horizontal courses. Stout embedded core stones fill the centre;
    longer axes run through the pitch rotation so they need not all appear horizontal. Two hard-unioned
    fragments break each stone's outline. This remains a proposal using the existing box primitives,
    whose planar faces and limited rotation may still read as cut stone in the actual shot.
    """
    rng = np.random.default_rng(int(seed) + 8309)
    local = SP.Scene()
    local.begin(rgb=(0.050, 0.047, 0.045))

    def stone(centre, half, yaw, pitch):
        local.box(centre, half, yaw=yaw, pitch=pitch, rnd=0.045, mat=5, k=0.0)
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        axis = np.array([sy * cp, sp, cy * cp])
        fragment = centre + axis * half[2] * rng.uniform(0.12, 0.22)
        local.box(fragment, half * rng.uniform([0.76, 0.80, 0.73], [0.96, 1.04, 0.94]),
                  yaw=yaw + rng.uniform(-0.28, 0.28), pitch=pitch + rng.uniform(-0.14, 0.14),
                  rnd=0.038, mat=5, k=0.0)

    # Overlapping core stones sit partly below the ground, not on a visible horizontal plinth.
    for centre, half, yaw, pitch in (
            ((-0.12, 0.045, -0.05), (0.18, 0.12, 0.22), 0.7, 0.38),
            ((0.12, 0.10, 0.06), (0.16, 0.15, 0.19), -0.8, -0.46),
            ((-0.04, 0.265, 0.015), (0.15, 0.15, 0.17), 1.8, 0.51)):
        stone(np.array(centre), np.array(half), yaw, pitch)
    phase = rng.uniform(0, 2 * math.pi)
    for i in range(19):
        u = i / 18.0
        az = phase + i * math.pi * (3.0 - math.sqrt(5.0)) + rng.uniform(-0.17, 0.17)
        radius = 0.365 * (1.0 - u) ** 0.65 + rng.uniform(-0.02, 0.025)
        centre = np.array([radius * math.cos(az), 0.025 + 0.365 * u + rng.uniform(-0.03, 0.03),
                           radius * math.sin(az)])
        half = np.array([rng.uniform(0.09, 0.14), rng.uniform(0.07, 0.105), rng.uniform(0.10, 0.145)])
        yaw = -az + rng.uniform(-0.65, 0.65)
        pitch = rng.uniform(0.30, 0.74) * (-1 if i % 2 else 1)
        stone(centre, half, yaw, pitch)
    local.end()
    rows, _ = local.arrays()
    ref_lower, ref_upper = baseline_bounds(seed)
    lower, upper = primitive_bounds(rows)
    # Uniformly scale the primitive geometry to retain the original heap's maximum horizontal span.
    factor = np.max((ref_upper - ref_lower)[[0, 2]]) / np.max((upper - lower)[[0, 2]])
    rows[:, 1:4] *= factor
    rows[:, 11:14] *= factor
    rows[:, 7] *= factor
    rows[:, 2] += ref_upper[1] - primitive_bounds(rows)[1][1]
    base = np.asarray(P[:3], np.float64)
    sc.begin(rgb=(0.050, 0.047, 0.045))
    for row in rows:
        sc.box(base + row[1:4] * scale, row[11:14] * scale, yaw=row[4], pitch=row[5],
               rnd=row[7] * scale, mat=5, k=0.0)
    sc.end()


BUILDERS = {'baseline': BASELINE, 'packing-only': packing_only, 'fractured-fieldstone': fractured_fieldstone,
            'uneven-fieldstone': uneven_fieldstone}


def geometry_report(variant, seed=7070, scale=1.0):
    """Build geometry only; fail on nonfinite data, changed material or an unexpected footprint/height."""
    scene = SP.Scene()
    BUILDERS[variant](scene, np.zeros(6), seed, scale)
    rows, objects = scene.arrays()
    if not np.isfinite(rows).all() or not np.isfinite(objects).all():
        raise ValueError('nonfinite cairn geometry')
    if len(objects) != 1 or not np.all(rows[:, 9] == 5):
        raise ValueError('study must remain one stone-material object')
    lower, upper = primitive_bounds(rows)
    if np.max(np.abs(np.concatenate([lower[[0, 2]], upper[[0, 2]]]))) > 0.65 * scale:
        raise ValueError('cairn exceeded the baseline footprint allowance')
    ref_lower, ref_upper = baseline_bounds(seed, scale)
    if not (-0.16 * scale <= lower[1] <= 0.04 * scale):
        raise ValueError('cairn bottom bound left the allowed partly embedded range')
    if not np.isclose(upper[1], ref_upper[1], rtol=0.0, atol=1e-12 * scale):
        raise ValueError('cairn top bound differs from the baseline for this seed and scale')
    return {'variant': variant, 'seed': seed, 'scale': scale, 'primitives': len(rows),
            'bounds_min': lower.tolist(), 'bounds_max': upper.tolist(),
            'baseline_bounds_min': ref_lower.tolist(), 'baseline_bounds_max': ref_upper.tolist(),
            'height_check': 'conservative primitive top bound; does not establish contact or visual quality'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=tuple(BUILDERS), required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--frames', default='4240,4399')
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--ss', type=float, default=1.5)
    args = parser.parse_args()
    if not args.out.is_absolute():
        parser.error('--out must be an explicit absolute path to a study directory')
    if not all(math.isfinite(x) and x > 0 for x in (args.scale, args.ss)):
        parser.error('--scale and --ss must be finite and positive')
    try:
        frames = list(dict.fromkeys(int(x.strip()) for x in args.frames.split(',')))
    except ValueError:
        parser.error('--frames must be comma-separated A15 frame numbers')
    if not frames or any(not WA.FR0 <= f < WA.FR0 + WA.NFR for f in frames):
        parser.error(f'--frames must lie in {WA.FR0}..{WA.FR0 + WA.NFR - 1}')
    # Camera overrides would change the comparison's question; require the accepted camera explicitly.
    if any(os.environ.get(k) for k in ('W15_CAM', 'W15_P7', 'W15_OWN_CAM', 'A14_CAM', 'NIGHT_CLOUD')):
        parser.error('clear camera/cloud look-dev overrides for this geometry-only comparison')
    report = geometry_report(args.variant)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = dict(report, frames=frames, image_scale=args.scale, supersampling=args.ss,
                    source='watchers_a.render; only hearth_scene replaced in this process',
                    study_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / 'cairn_study.json').write_text(json.dumps(metadata, indent=2) + '\n')
    previous = WA.hearth_scene
    WA.hearth_scene = BUILDERS[args.variant]
    try:
        for frame in frames:
            started = time.monotonic()
            picture = WA.finish(WA.render(frame, scale=args.scale, ss=args.ss))
            path = WA.PI.look.frame_path(str(args.out), frame)
            WA.PI.look.save_png(path, picture)
            print(f'{args.variant} A15 {frame}: {time.monotonic() - started:.1f}s -> {path}', flush=True)
    finally:
        WA.hearth_scene = previous


if __name__ == '__main__':
    main()
