"""Finite-grid torch-field/own-figure visibility probe. Does not render images."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = repo / 'the-long-dawn/shots/accord'
    # This probe tests only the inner torch's own geometry, not a complete scene.
    os.environ['CROWD'] = '0'
    sys.path.insert(0, str(source))
    import accord3 as A
    S, G, F = A.SC, A.G, A.FL3
    t = 5160.0
    scale = 0.5
    figures = S.figures(t)
    heads = S.torch_heads(t, figures)
    flames, _ = A.flames_and_lights(t, figures)
    camera = S.camera(t, scale)
    noise = A.resources()['noise3']
    pixel = 1.0 / camera[12]
    width, height = A.frame_size(scale)
    axes = [np.arange(-0.13, 0.26, 0.012),
            np.arange(-0.21, 0.14, 0.012),
            np.arange(-0.10, 0.57, 0.012)]
    result = {
        'frame': t, 'scale': scale, 'image_size': [width, height],
        'repo': str(repo),
        'git_head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ['accord3.py', 'scene3.py', 'geom3.py', 'flame3.py', 'shade3.py']},
        'python': sys.version, 'numpy': np.__version__,
        'grid': {'coordinates': 'world-axis offsets from current flame base, metres',
                 'x': [-0.13, 0.26], 'y': [-0.21, 0.14], 'z': [-0.10, 0.57],
                 'step': 0.012, 'stop_exclusive': True,
                 'shape': [len(a) for a in axes],
                 'points_per_torch': int(np.prod([len(a) for a in axes]))},
        'weight': 'density * (0.05 + temperature**2.6)',
        'columns': ['all positive-density samples', 'temperature > 0.6', 'temperature > 0.8'],
        'variants': ['current', 'hypothetical upper-fuel anchor'],
        'torches': [],
    }
    for i in [0, 5, 6, 7]:
        axis = heads[i, 3:6] / np.linalg.norm(heads[i, 3:6])
        lowered = max(-axis[2], 0.0)
        blend = lowered * lowered * (3.0 - 2.0 * lowered)
        # Test hypothesis only: move toward the proximal fuel cap + its upper radius.
        shift = blend * (-0.08 * axis + np.array([0.0, 0.0, 0.034]))
        total = np.zeros(3)
        inside = np.zeros(3)
        occluded = np.zeros((2, 3))
        onscreen = np.zeros((2, 3))
        visible = np.zeros((2, 3))
        centroid_sum = np.zeros((2, 2))
        positive = 0
        for qz in axes[2]:
            for qx in axes[0]:
                for qy in axes[1]:
                    density, temperature = F.torch_density(qx, qy, qz, flames, i, t, noise)
                    if density <= 0.0:
                        continue
                    positive += 1
                    weight = density * (0.05 + temperature ** 2.6)
                    weights = np.array([weight, weight if temperature > 0.6 else 0.0,
                                        weight if temperature > 0.8 else 0.0])
                    total += weights
                    point = flames[i, :3] + [qx, qy, qz]
                    distance, _ = G.sd_fig(*point, figures, i)
                    if distance < 0.0:
                        inside += weights
                    for variant, delta in enumerate([np.zeros(3), shift]):
                        position = point + delta
                        xy, _ = S.project(camera, position)
                        in_frame = 0 <= xy[0] < width and 0 <= xy[1] < height
                        if in_frame:
                            onscreen[variant] += weights
                        direction = position - camera[:3]
                        ray_length = np.linalg.norm(direction)
                        direction /= ray_length
                        inverse = np.where(abs(direction) > 1e-9, 1.0 / direction, 1e9)
                        near, far = G.ray_aabb(*camera[:3], *inverse,
                                              *figures[i, G.F_BB:G.F_BB + 6])
                        hit = -1.0
                        if far >= max(near, 0.0) and near < ray_length:
                            hit, _ = G.trace_fig(*camera[:3], *direction, near, min(far, ray_length),
                                                  figures, i, pixel)
                        blocked = hit >= 0.0 and hit < ray_length - 0.001
                        if blocked:
                            occluded[variant] += weights
                        if in_frame and not blocked:
                            visible[variant] += weights
                            centroid_sum[variant] += weight * xy
        position = figures[i, G.F_X:G.F_Y + 1]
        angle = figures[i, G.F_ANG]
        hand = figures[i, G.F_HX:G.F_HZ + 1]
        local_axis = figures[i, G.F_TX:G.F_TZ + 1]
        ends = S.to_world(i, np.array([hand + 0.33 * local_axis, hand + 0.45 * local_axis]),
                          pos=position, ang=angle)
        flame_height = flames[i, 3] * F.torch_flicker(flames[i, 8], t, noise)
        core = flames[i, :3] + [0.0, 0.0, 0.22 * flame_height]
        core_distance, _ = G.sd_fig(*core, figures, i)
        item = {
            'torch': i, 'lit': float(flames[i, 9]), 'axis_z': float(axis[2]),
            'base_z': float(flames[i, 2]),
            'nominal_fuel_max_z': float(max(ends[0, 2] + 0.034, ends[1, 2] + 0.029)),
            'nominal_core_z': float(core[2]), 'nominal_core_sdf': float(core_distance),
            'hypothetical_shift_xyz': shift.tolist(),
            'base_screen_xy': [S.project(camera, flames[i, :3])[0].tolist(),
                               S.project(camera, flames[i, :3] + shift)[0].tolist()],
            'positive_density_sample_count': positive,
            'weight_sum': total.tolist(),
            'inside_current_geometry_fraction': (inside / total).tolist(),
            'own_figure_occluded_fraction': (occluded / total).tolist(),
            'onscreen_fraction': (onscreen / total).tolist(),
            'onscreen_unoccluded_fraction': (visible / total).tolist(),
            'onscreen_unoccluded_centroid': (centroid_sum / visible[:, 0, None]).tolist(),
        }
        result['torches'].append(item)
        print(f"torch {i}: current/candidate onscreen-unoccluded hot weight fractions "
              f"{visible[0, 1] / total[1]:.6f} / {visible[1, 1] / total[1]:.6f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
