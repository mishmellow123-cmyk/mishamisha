"""Scalar ridge contracts using actual noise source, without renderer imports/JIT."""
import ast
import copy
import math
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    picked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            picked.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names for target in node.targets):
            picked.append(node)
    assert len(picked) == len(names), 'Accepted scalar definitions moved'
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), 'exec'), namespace)


NS = dict(np=np, math=math, NCR=22, CLOUD_Y=0.0)
definitions(HERE.parents[0] / 'montage/mt/noise.py',
            ('_G2X', '_G2Y', 'ihash', 'hash2i', 'fade', 'gnoise2', 'fbm2', 'ridged2', 'smoothstep'), NS)
definitions(HERE / 'world.py', ('_lod', 'ridge_row', 'ridge', 'smax', 'h_rock'), NS)
RIDGE = NS['ridge']

# Restore the original bound rejection and remove the marked cap-width block
# to execute the legacy path. Noise, strata and ridge arithmetic stay actual.
WORLD_SOURCE = (HERE / 'world.py').read_text()
BASELINE = ast.parse(WORLD_SOURCE)
BASE_RIDGE = next(node for node in BASELINE.body if isinstance(node, ast.FunctionDef) and node.name == 'ridge')
BASE_RIDGE = copy.deepcopy(BASE_RIDGE)
CAP_BLOCKS = [node for node in BASE_RIDGE.body if isinstance(node, ast.If)
              and ast.unparse(node.test).startswith('CR[k, 16]')]
BOUND_BLOCKS = [node for node in BASE_RIDGE.body if isinstance(node, ast.If)
               and ast.unparse(node.test) == 'ub < hcur - CR[k, 9]']
assert len(CAP_BLOCKS) == len(BOUND_BLOCKS) == 1, 'Review opt-in blocks before changing this baseline'
BOUND_BLOCKS[0].body = ast.parse('return -1e5').body
BASE_RIDGE.body = [node for node in BASE_RIDGE.body if node not in CAP_BLOCKS]
BASE_RIDGE.decorator_list = []
BASE_NS = dict(NS)
exec(compile(ast.Module(body=[BASE_RIDGE], type_ignores=[]), 'ridge_without_opt_in', 'exec'), BASE_NS)
LEGACY = BASE_NS['ridge']


def row(wl=15.0, wr=13.0):
    return NS['ridge_row']((0., 150., 0.), (100., 140., 0.), wl=wl, wr=wr,
                           seed=50, detail=.14, slope=1.9)[None, :]


def warped(x, z, fp, seed=50):
    ow = NS['_lod'](60., fp, 1., 4.)
    return (x + 9. * NS['fbm2'](x / 70. + 1.3, z / 70., ow, seed),
            z + 9. * NS['fbm2'](x / 70. - 4.1, z / 70. + 2.2, ow, seed + 1))


def query_at_warped(wx, wz, fp=3.0):
    # Invert this gentle domain warp to probe the actual endpoint geometry.
    x, z = wx, wz
    for _ in range(60):
        actual_x, actual_z = warped(x, z, fp)
        x += wx - actual_x
        z += wz - actual_z
    np.testing.assert_allclose(warped(x, z, fp), (wx, wz), atol=1e-12, rtol=0)
    return x, z


def height(kernel, cr, point, fp=3.0, hcur=-1e6):
    return kernel(*point, fp, cr, 0, hcur)


def combined_height(kernel, cr, point, fp, hcur):
    # Execute the real union/early-out caller with one ridge and a controlled
    # synthetic background height; this is not a full production-ground probe.
    namespace = dict(NS, ridge=kernel, S1=SimpleNamespace(h_far=lambda x, z, fp: hcur))
    h_rock = FunctionType(NS['h_rock'].__code__, namespace)
    return h_rock(*point, fp, cr)


def test_disabled_is_exact_legacy_on_integer_and_measured_queries():
    cr = row()
    assert cr[0, 16] == 0.0
    points = [(x, z) for x in (-20, 0, 25, 75, 100, 120) for z in (-15, 0, 15)]
    points += [(-5196.914065968755, 14390.889570360014)]
    for flag in (0.0, -1.0):
        cr[0, 16] = flag
        for fp in (.003, 3.):
            for hcur in (-1e6, 130., 160.):
                for point in points:
                    assert height(RIDGE, cr, point, fp, hcur) == height(LEGACY, cr, point, fp, hcur)


def test_interior_uses_exact_legacy_arithmetic():
    cr = row()
    cr[0, 16] = 1.
    for wx in (10., 50., 90.):
        for wz in (-30., -1., 1., 30.):
            for fp in (.003, 3.):
                point = query_at_warped(wx, wz, fp)
                original = height(LEGACY, cr, point, fp)
                assert height(RIDGE, cr, point, fp) == original
                for offset in (-2., -1., -.5, 0., .5, 1., 2., 4.):
                    hcur = original + offset * cr[0, 9]
                    assert height(RIDGE, cr, point, fp, hcur) == height(LEGACY, cr, point, fp, hcur)
                    assert combined_height(RIDGE, cr, point, fp, hcur) == combined_height(LEGACY, cr, point, fp, hcur)


def test_interior_preserves_legacy_rejection_between_the_two_bounds():
    # Reviewer-reproduced query: old upper bound 112.2727957944436 m, wider
    # bound 114.22132466221797 m. This hcur lies between those bounds plus k.
    cr = row()
    cr[0, 16] = 1.
    point = (53.036088720635604, -27.540175001287096)
    np.testing.assert_allclose(warped(*point, 3.), (50., -30.), atol=1e-12, rtol=0)
    assert height(LEGACY, cr, point) == pytest.approx(74.86144397184113, abs=1e-9, rel=0)
    hcur = 121.24706022833078
    assert height(LEGACY, cr, point, hcur=hcur) == -1e5
    assert height(RIDGE, cr, point, hcur=hcur) == -1e5
    assert combined_height(RIDGE, cr, point, 3., hcur) == combined_height(LEGACY, cr, point, 3., hcur)


def test_equal_width_caps_use_exact_legacy_arithmetic():
    cr = row(13., 13.)
    cr[0, 16] = 1.
    for wx in (-40., 0., 50., 100., 140.):
        for wz in (-20., 0., 20.):
            point = query_at_warped(wx, wz)
            assert height(RIDGE, cr, point) == height(LEGACY, cr, point)


def test_zero_radius_keeps_legacy_value_without_division_by_zero():
    cr = row()
    cr[0, 16] = 1.
    point = (7., -3.)
    wx, wz = warped(*point, 3.)
    for endpoint in (0, 1):
        cr[0, :6] = [wx - endpoint * 100., wz, 150., wx + (1 - endpoint) * 100., wz, 140.]
        assert height(RIDGE, cr, point) == height(LEGACY, cr, point)


def test_both_caps_are_continuous_across_axis_and_match_flanks():
    cr = row()
    cr[0, 16] = 1.
    for end in (0., 100.):
        wx = end + (-20. if end == 0. else 20.)
        a = query_at_warped(wx, -1e-7)
        b = query_at_warped(wx, 1e-7)
        assert abs(height(LEGACY, cr, a) - height(LEGACY, cr, b)) > 1.
        assert abs(height(RIDGE, cr, a) - height(RIDGE, cr, b)) < 1e-5
        # Join each cap to both flanks: the angle reaches +/- pi/2.
        for side in (-8., 8.):
            a = query_at_warped(end - 1e-7, side)
            b = query_at_warped(end + 1e-7, side)
            assert abs(height(RIDGE, cr, a) - height(RIDGE, cr, b)) < 1e-5


def test_bound_cannot_cull_an_interpolated_cap_above_current_height():
    cr = row(40., 2.)
    cr[0, 10] = 0.0
    cr[0, 16] = 1.
    point = query_at_warped(160., -5.)
    assert point[1] < 0.0  # The pre-warp query must lie on the narrow side too.
    actual = height(RIDGE, cr, point)
    assert actual > -1e5
    # A current surface just below the evaluated cap must not cull that cap.
    assert height(RIDGE, cr, point, hcur=actual - .01) == actual


def test_measured_a5043_seam_collapses_with_real_noise():
    # Frozen coordinates/row from dash_terrain_seam.json, controls[0:2]. The
    # measured locations project near (100.518,784.924) and (200.523,755.560).
    cr = np.array([[-5161.535898384862, 14341.378221735089, 131.,
                    -5196.732050807569, 14390.339745962156, 126., 15., 13.,
                    50., 6., .14, 1.9, -1., 436.8993442427131, .95, 0., 1., 0., 0., 0., 0., 0.]])
    cases = [(-5196.914065968755, -5196.914085968754, 14390.889570360014,
              .0026079180930227877, 121.1192862057514, 121.2755582838497),
             (-5197.068970172673, -5197.068990172673, 14391.209845711965,
              .002599132596014548, 120.71777025389999, 120.86948089217441)]
    for xa, xb, z, fp, legacy_a, legacy_b in cases:
        assert height(LEGACY, cr, (xa, z), fp) == pytest.approx(legacy_a, abs=1e-9, rel=0)
        assert height(LEGACY, cr, (xb, z), fp) == pytest.approx(legacy_b, abs=1e-9, rel=0)
        assert abs(height(RIDGE, cr, (xa, z), fp) - height(RIDGE, cr, (xb, z), fp)) < 2e-5


def test_interior_shaded_pixels_are_bitwise_identical():
    # A synthetic 1x3 depth patch on a 1000 m ridge. Run the actual shader,
    # shadow, terrain-union and fog code; the other surfaces are flat and low.
    # This checks three numerical pixels, not any delivered-frame region.
    namespace = dict(NS, prange=range,
                     S1=SimpleNamespace(h_far=lambda x, z, fp: -1e5),
                     h_near=lambda x, z, fp: -1e5,
                     h_cloud_cr=lambda x, z, fp, t, cr: -1e5)
    definitions(HERE / 'world.py',
                ('R_EARTH', 'ridge', 'h_rock', 'hfun', 'soft_shadow', 'height_fog_tau', 'shade'), namespace)
    actual_ridge = namespace['ridge']
    queried_t = []

    def checked_ridge(x, z, fp, cr, k, hcur):
        wx, wz = warped(x, z, fp)
        dx, dz = cr[k, 3] - cr[k, 0], cr[k, 4] - cr[k, 1]
        queried_t.append(((wx - cr[k, 0]) * dx + (wz - cr[k, 1]) * dz) / (dx * dx + dz * dz))
        return actual_ridge(x, z, fp, cr, k, hcur)

    namespace['ridge'] = checked_ridge
    cr = NS['ridge_row']((0., 150., 0.), (600., 150., 800.),
                         wl=15., wr=13., seed=50, detail=.14, slope=1.9)[None, :]
    camera = np.array([320., 160., 365., 0., 1., 1., 0., 600., 1.5, .5])
    depths = np.full((1, 3), 20.)
    origin = np.array([320., 365., 0.])
    key = np.array([0., 1., 0., .9, .95, 1.])  # Vertical: shadow rays stay inside the segment.
    knobs = np.zeros(20)
    knobs[[0, 3, 4, 7, 10, 11]] = [1., .4, .6, .1, 1., 6.]
    fog = np.array([0., .001, 0., .001, 0., 0., 0., 0.])
    outputs = []
    for flag in (0., 1.):
        cr[0, 16] = flag
        rgb, zbuf, distance = np.zeros((1, 3, 3)), np.zeros((1, 3)), np.zeros((1, 3))
        namespace['shade'](camera, depths, origin, cr, np.zeros(1), np.empty((0, 8)), key,
                           knobs, np.array([.1, .12, .15]), fog, rgb, zbuf, distance, np.empty((0, 4)))
        for buffer in (rgb, zbuf, distance):
            assert np.isfinite(buffer).all() and np.any(buffer > 0.)
        outputs.append((rgb, zbuf, distance))
    assert queried_t and all(0. < t < 1. for t in queried_t)
    for original, enabled in zip(*outputs):
        assert original.tobytes() == enabled.tobytes()
