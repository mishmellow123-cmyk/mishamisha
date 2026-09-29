"""Default-off A crossing step-rock study; no crossing import or JIT at import.

The accepted three-box step is recognized from its complete source recipe.
low_shoulders keeps a level .42m tread and puts the asymmetric flanks below it,
so they cannot form a raised perimeter. STEP_H and step_h are never changed.
Actual terrain/foot agreement still requires probe_support and rendered review.
Finals remain held; this module neither renders nor writes files.
"""
import math

import numpy as np

CANDIDATES = ('accepted', 'low_shoulders')
UP = np.array([0., 1., 0.])


def _anchor(crossing, cfg):
    if crossing.STEP_H != .42:
        raise ValueError('This study requires the existing .42m analytic step')
    event = crossing.step_event(cfg)
    if event is None:
        return None
    position, forward = crossing.at(event['s_edge']+.65)
    ground = float(crossing.ground_many(position)[0])
    yaw = math.atan2(forward[0], forward[2])
    return event, np.asarray(position), np.asarray(forward), ground, yaw


def _rock_scene(scene_type, crossing, anchor, candidate):
    _, p, forward, ground, yaw = anchor
    lateral = np.cross(UP, forward)
    step = crossing.STEP_H
    rock = scene_type()
    rock.begin(rgb=(.07, .07, .07))
    if candidate == 'accepted':
        # Exact build_scene recipe at RUN-A4 restage 76aec6c. Matching these
        # rows prevents an unrelated material-6 hearth stone being replaced.
        rock.box(np.array([p[0], ground+step*.5-.14, p[2]]), (.58, step*.5+.12, .66),
                 yaw=yaw+.18, pitch=.07, rnd=.10, mat=6, k=0., rough=.055, rough_f=5.5)
        rock.box(np.array([p[0], ground+step*.72, p[2]])-lateral*.22+forward*.30, (.30, .16, .30),
                 yaw=yaw-.6, pitch=-.12, rnd=.07, mat=6, k=.10, rough=.04, rough_f=7.)
        rock.box(np.array([p[0], ground+step*.30, p[2]])+lateral*.50-forward*.35, (.34, step*.40, .36),
                 yaw=yaw-.9, pitch=.10, rnd=.08, mat=6, k=.12, rough=.045, rough_f=6.)
    elif candidate == 'low_shoulders':
        # Central support: no pitch, displacement or smooth-union uplift. Its
        # flat top is exactly ground_at_anchor + STEP_H. Do not infer that the
        # terrain at every foot is equal to that anchor datum.
        rock.box(np.array([p[0], ground+step-.28, p[2]]), (.27, .28, .60),
                 yaw=yaw, rnd=.035, mat=6, k=0.)
        # Unequal, skewed shoulders slope into the ridge and stay below the
        # tread, including a conservative 2.85*rough noise displacement bound.
        rock.box(np.array([p[0], ground+.08, p[2]])-lateral*.40+forward*.10, (.36, .16, .53),
                 yaw=yaw+.47, pitch=.16, rnd=.055, mat=6, k=0., rough=.018, rough_f=5.5)
        rock.box(np.array([p[0], ground+.08, p[2]])+lateral*.42-forward*.20, (.30, .17, .37),
                 yaw=yaw-.65, pitch=-.20, rnd=.055, mat=6, k=0., rough=.018, rough_f=6.5)
    else:
        raise ValueError('Unknown rock candidate')
    rock.end()
    rock.O[-1][14] = -1.
    return rock


def _find_owned(scene, expected):
    wanted = np.asarray(expected.P, dtype=np.float64)
    matches = []
    for index, obj in enumerate(scene.O):
        first, count = int(obj[0]), int(obj[1])
        if (first != obj[0] or count != obj[1] or first < 0 or count < 0 or first+count > len(scene.P)):
            raise ValueError('Invalid scene primitive ownership range')
        if count != len(wanted) or obj[14] != -1.:
            continue
        rows = np.asarray(scene.P[first:first+count])
        if (rows.shape == wanted.shape and np.allclose(rows, wanted, rtol=0, atol=1e-10)
                and np.allclose(np.asarray(obj)[1:], np.asarray(expected.O[0])[1:], rtol=0, atol=1e-10)):
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one source-matching crossing step rock; found {len(matches)}')
    return matches[0]


def support_bounds(crossing, anchor, rock):
    """Conservative geometry bounds, not measured foot contact.

gnoise3 interpolates gradients each bounded by 2, then multiplies by .95;
the two displacement octaves therefore have absolute sum <= 2.85. Rounded
boxes lie inside their oriented half extents. All candidate blend k are zero.
"""
    event, position, forward, ground, yaw = anchor
    upper = []
    for row in rock.P:
        height = abs(math.cos(row[5]))*row[12]+abs(math.sin(row[5]))*row[13]
        upper.append(float(row[2]+height+2.85*row[14]))
    return dict(anchor_world=position.tolist(), anchor_ground_y=ground, forward=forward.tolist(), yaw=yaw,
        analytic_step_height_m=crossing.STEP_H, core_top_y=ground+crossing.STEP_H,
        flat_core_local_bounds_m=dict(lateral=[-.235, .235], forward=[-.565, .565]),
        flat_core_nominal_arc_bounds=[event['s_edge']+.085, event['s_edge']+1.215],
        primitive_upper_y_bounds=upper, shoulder_upper_below_tread_m=[ground+crossing.STEP_H-v for v in upper[1:]],
        limitation='The core datum uses ground at the anchor. Path curvature, terrain differences, rounded edges and real boot trajectories still need probing.')


def apply_to_scene(scene, crossing, cfg, candidate='accepted', diagnostics=None):
    """Return accepted identity or a separate scene with only the owned rock replaced."""
    if candidate not in CANDIDATES:
        raise ValueError('Unknown crossing rock candidate')
    if candidate == 'accepted':
        return scene
    anchor = _anchor(crossing, cfg)
    if anchor is None:
        if diagnostics is not None:
            diagnostics.update(candidate=candidate, applied=False, reason='No step event in this crossing configuration')
        return scene
    original = _rock_scene(type(scene), crossing, anchor, 'accepted')
    index = _find_owned(scene, original)
    replacement = _rock_scene(type(scene), crossing, anchor, candidate)
    first, count = map(int, scene.O[index][:2])
    shift = len(replacement.P)-count
    result = type(scene)()
    result.P = list(scene.P[:first])+list(replacement.P)+list(scene.P[first+count:])
    result.O = [list(row) for row in scene.O]
    result.O[index] = list(replacement.O[0])
    result.O[index][0] = first
    for other, row in enumerate(result.O):
        if other != index and row[0] >= first+count:
            row[0] += shift
    metadata = dict(candidate=candidate, applied=True, object_index=index,
                    replaced_primitive_range=[first, first+count], source_recipe='crossing.build_scene at restage 76aec6c',
                    support=support_bounds(crossing, anchor, replacement), boot_contact_proved=False)
    result.crossing_rock_study = metadata
    if diagnostics is not None:
        diagnostics.update(metadata)
    return result


def _surface_y(map_function, primitives, x, z, low, high):
    """Highest sampled solid interval, refined by bisection (not a ray render)."""
    levels = np.linspace(low, high, 129)
    inside = [map_function(x, float(y), z, primitives, 0, len(primitives))[0] <= 0 for y in levels]
    occupied = np.flatnonzero(inside)
    if not len(occupied):
        return None
    last = int(occupied[-1])
    if last == len(levels)-1:
        raise RuntimeError('Rock surface probe upper bound intersects geometry')
    a, b = float(levels[last]), float(levels[last+1])
    for _ in range(18):
        mid = .5*(a+b)
        if map_function(x, mid, z, primitives, 0, len(primitives))[0] <= 0:
            a = mid
        else:
            b = mid
    return .5*(a+b)


def probe_support(crossing, cfg, scene, frames, tolerance_m=.04):
    """Explicit optional geometry probe for a gated, already-loaded runner.

This calls the supplied renderer's actual SDF and foot/path functions; they
may JIT on first use. It must not be called casually at module import. Samples
are boot-centre undersides, not the complete sole or rendered occlusion.
"""
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('Tolerance must be positive and finite')
    metadata = getattr(scene, 'crossing_rock_study', None)
    if metadata is None:
        raise ValueError('Support probe requires the explicit candidate scene')
    frame_list = list(frames)
    if not frame_list or any(type(f) is not int or not 4880 <= f < 5840 for f in frame_list):
        raise ValueError('Use explicit absolute A4880..5839 frames')
    obj = scene.O[metadata['object_index']]
    start, count = map(int, obj[:2])
    primitives = np.asarray(scene.P[start:start+count], dtype=np.float64)
    event = crossing.step_event(cfg)
    samples = []
    datum = metadata['support']['anchor_ground_y']
    for frame in frame_list:
        t = (frame-4880)/24.
        arcs = crossing.line_s(t, cfg)
        old_arcs = crossing.line_s(t-.25, cfg)
        go = np.clip((arcs-old_arcs)/(.25*crossing.SPEED), 0., 1.)
        for role, k in (('helper', event['ka']), ('climber', event['kb'])):
            i = k+2
            height = crossing.walker_cut(k)['h']
            planted = crossing.plant([(arcs[i], i, height, 0.)], cfg=cfg, go=[go[i]], t=t)[0]
            _, left, right, direction = planted
            facing = direction
            if role == 'helper':
                turn = crossing.smoothstep(event['t0']-.3, event['t0']+.5, t)*(1.-crossing.smoothstep(event['t_go']-.5, event['t_go'], t))
                facing = crossing._facing(direction, math.pi*turn)
            stride, phase = crossing.stride_of(i), crossing._hh(i, 6)
            for name, ankle, delta in (('left', left, 0.), ('right', right, .5)):
                foot_arc, lift = crossing.foot_cycle(arcs[i], stride, phase, delta)
                lift *= go[i]*.85
                # Original traveller boot: centre ank+w*.045h-up*.035h,
                # half-height .048h. The pair uses leg_k=1.
                bottom = ankle+facing*.045*height-UP*.083*height
                terrain = float(crossing.ground_many(bottom)[0])
                top = _surface_y(crossing.SP._map, primitives, float(bottom[0]), float(bottom[2]), datum-1., datum+1.)
                surface = max(terrain, top) if top is not None else terrain
                plateau = event['s_edge']+.06 <= foot_arc <= event['s_edge']+1.25
                assess = plateau and lift <= 1e-6
                gap = float(bottom[1]-surface)
                samples.append(dict(frame=frame, role=role, foot=name, foot_arc=float(foot_arc), swing_lift_m=float(lift),
                    ankle_world=ankle.tolist(), boot_bottom_centre_world=bottom.tolist(), terrain_y=terrain, rock_surface_y=top,
                    support_gap_m=gap, plateau_stance_sample=bool(assess), within_tolerance=bool(abs(gap) <= tolerance_m) if assess else None))
    checked = [s for s in samples if s['plateau_stance_sample']]
    return dict(candidate=metadata['candidate'], samples=samples, tolerance_m=tolerance_m,
        plateau_stance_samples=len(checked), plateau_stance_failures=sum(not s['within_tolerance'] for s in checked),
        max_abs_plateau_stance_gap_m=max((abs(s['support_gap_m']) for s in checked), default=None),
        status='no_plateau_stance_samples' if not checked else 'failed' if any(not s['within_tolerance'] for s in checked) else 'sampled_geometry_pass',
        boot_contact_proved=False, limitations=['Checks sampled boot-centre undersides, not complete soles or image visibility.',
            'Surface search uses 128 intervals plus bisection; thinner disconnected details could be missed.',
            'A geometry pass does not establish perceived contact or approve the held A finals.'])
