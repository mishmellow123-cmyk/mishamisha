"""Real shader checks for opt-in smooth gilt and ink laid over its surface."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import book as B


def shade_patch(*, burnished=None, ink=0., gilt=1., light=(-55., 50., 30.), reflection=None,
                view=None, position_x=None):
    """Fixed geometry/light over varying page UVs isolates material texture."""
    h = w = 48
    G = np.zeros((h, w, B.NG), np.float64)
    G[..., 0] = B.M_PAGE_L
    G[..., 1] = 20.
    G[..., 7] = 1.
    if position_x is not None:
        G[..., 2] = np.linspace(*position_x, w)
    u, v = np.meshgrid(np.linspace(5., 10., w), np.linspace(5., 10., h))
    G[..., 8], G[..., 9] = u, v
    G[..., 10] = .015
    G[..., 11:13] = 1.
    G[..., 14] = 1. / np.sqrt(2.)
    G[..., 15] = -1. / np.sqrt(2.)
    if view is not None:
        direction = np.asarray(view, np.float64)
        G[..., 13:16] = -direction / np.linalg.norm(direction)
    tx = B.PageTex(20., 29., 6)
    tx.chan[..., 0], tx.chan[..., 2] = ink, gilt
    tx.build()
    out = np.zeros((h, w, 3), np.float64)
    alpha = np.zeros((h, w), np.float64)
    vec = lambda a: np.array(a, np.float64)
    args = (out, alpha, G, vec([20., 29.]), vec([0., -20., 20.]), vec(light),
            vec([1.9, .988, .456]), 10., vec([.35, -.5, .8]), vec([.010, .012, .016]),
            vec([.012, .009, .007]), vec([.9, .5, .22]),
            tx.data, tx.meta, float(tx.ppc), tx.data, tx.meta, float(tx.ppc),
            13., 30., 1., np.zeros(20), np.zeros(20), np.zeros((0, 6)), 0.,
            tx.data, tx.meta, float(tx.ppc), tx.data, tx.meta, float(tx.ppc),
            tx.data, tx.meta, float(tx.ppc), 0., 0., .03)
    if burnished is None and reflection is None:
        B.shade_kernel(*args)
    else:
        source = None if reflection is None else np.asarray(reflection, np.float64)
        B.shade_kernel(*args, bool(burnished), source)
    return out, alpha


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


# Measured from book.py before adding burnished_gilt; these exercise the real
# compiled shader, page mip sampling, noise and lighting, not a material mock.
LEGACY_GOLD = 'c67f95fa10da45ca2d5c0538f003858c84f4001cfa452c07e73f991334191b3e'
LEGACY_PAPER = '1181750221cc1f4d50ba48c5e507956a59fca85172593c3415f15c04cad39b02'
LEGACY_ALPHA = '1c3d4f4b0df2fec9f03eb1313aafbe6f84ef9d6082d6a51c2837d3c186e2d3b7'


def test_omitted_and_disabled_material_are_byte_exact_legacy():
    for gilt, expected in ((1., LEGACY_GOLD), (0., LEGACY_PAPER)):
        original, alpha = shade_patch(gilt=gilt)
        off, off_alpha = shade_patch(gilt=gilt, burnished=False)
        assert digest(original) == expected
        assert digest(alpha) == LEGACY_ALPHA
        np.testing.assert_array_equal(original, off)
        np.testing.assert_array_equal(alpha, off_alpha)
    changed, _ = shade_patch(burnished=True)
    with pytest.raises(AssertionError):
        assert digest(changed) == LEGACY_GOLD


def assert_even_leaf(rgb):
    # A flat patch under fixed illumination must not become a field of facets.
    assert np.max(rgb.std(axis=(0, 1)) / rgb.mean(axis=(0, 1))) < .02


def test_burnished_leaf_is_even_and_legacy_facets_fail_the_same_gate():
    smooth, _ = shade_patch(burnished=True)
    assert_even_leaf(smooth)
    assert np.isfinite(smooth).all() and np.min(smooth) >= 0.
    legacy, _ = shade_patch()
    with pytest.raises(AssertionError):
        assert_even_leaf(legacy)


def assert_ink_contrast(ink, gold):
    assert np.max(ink.mean(axis=(0, 1)) / gold.mean(axis=(0, 1))) < .15


def test_opaque_ink_lies_above_gilt_and_legacy_layering_fails_the_gate():
    gold, _ = shade_patch(burnished=True)
    ink, _ = shade_patch(burnished=True, ink=1.)
    assert_ink_contrast(ink, gold)
    legacy_gold, _ = shade_patch()
    legacy_ink, _ = shade_patch(ink=1.)
    with pytest.raises(AssertionError):
        assert_ink_contrast(legacy_ink, legacy_gold)


def test_material_keeps_paper_exact_and_responds_to_light_direction():
    paper, alpha = shade_patch(gilt=0.)
    enabled_paper, enabled_alpha = shade_patch(gilt=0., burnished=True)
    np.testing.assert_array_equal(paper, enabled_paper)
    np.testing.assert_array_equal(alpha, enabled_alpha)
    left, _ = shade_patch(burnished=True)
    opposite, _ = shade_patch(burnished=True, light=(55., -50., 30.))
    assert not np.allclose(left.mean(axis=(0, 1)), opposite.mean(axis=(0, 1)), rtol=.1)
    with pytest.raises(AssertionError):
        assert not np.allclose(left, left)


def test_render_forwards_opt_in_without_changing_default_geometry():
    book = B.Book(TL=2.8, TR=1.2, seed=13)
    cam = B.Cam([.1, -38.8, 41.], [.6, .2, 2.], 38., 32, 14)
    light = B.Light([-55., 50., 30.], power=1.9)
    page = B.PageTex(20., 29., 4)
    page.chan[..., 2] = 1.
    page.build()
    omitted = B.render(book, cam, light, page, page, 0.)
    disabled = B.render(book, cam, light, page, page, 0., burnished_gilt=False)
    enabled = B.render(book, cam, light, page, page, 0., burnished_gilt=True)
    for a, b in zip(omitted, disabled):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(omitted[1], enabled[1])
    np.testing.assert_array_equal(omitted[2], enabled[2])
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(omitted[0], enabled[0])


REFLECTION = (0., 40., 40., .008, .0068, .0068, 3.)


def test_zero_reflection_is_exact_and_the_source_does_not_enable_legacy_gilt():
    current, alpha = shade_patch(burnished=True)
    zero, zero_alpha = shade_patch(burnished=True, reflection=(*REFLECTION[:3], 0., 0., 0., 3.))
    np.testing.assert_array_equal(current, zero)
    np.testing.assert_array_equal(alpha, zero_alpha)
    legacy, _ = shade_patch(burnished=False, reflection=REFLECTION)
    assert digest(legacy) == LEGACY_GOLD


def test_reflection_changes_only_uncovered_metal():
    for kwargs in ({'gilt': 0.}, {'ink': 1.}):
        baseline, alpha = shade_patch(burnished=True, **kwargs)
        reflected, reflected_alpha = shade_patch(burnished=True, reflection=REFLECTION, **kwargs)
        np.testing.assert_array_equal(baseline, reflected)
        np.testing.assert_array_equal(alpha, reflected_alpha)


def assert_directional_gain(aligned, misaligned):
    assert np.min(aligned.mean(axis=(0, 1))) > 0.
    assert np.min(aligned.mean(axis=(0, 1)) / misaligned.mean(axis=(0, 1))) > 20.


def test_reflection_follows_both_source_and_camera_angles():
    base, _ = shade_patch(burnished=True)
    aligned, _ = shade_patch(burnished=True, reflection=REFLECTION)
    same_side, _ = shade_patch(burnished=True, reflection=(0., -40., 40., *REFLECTION[3:]))
    turned_base, _ = shade_patch(burnished=True, view=(0., 1., 1.))
    turned, _ = shade_patch(burnished=True, reflection=REFLECTION, view=(0., 1., 1.))
    hit = aligned-base
    for miss in (same_side-base, turned-turned_base):
        assert_directional_gain(hit, miss)
        with pytest.raises(AssertionError):
            assert_directional_gain(miss, hit)


def assert_localized_gain(gain):
    profile = gain.mean(axis=(0, 2))
    assert profile[20:28].max() > 3. * max(profile[0], profile[-1])


def test_reflection_is_a_localized_sheen_not_uniform_gilt_brightening():
    base, _ = shade_patch(burnished=True, position_x=(-5., 5.))
    reflected, _ = shade_patch(burnished=True, position_x=(-5., 5.), reflection=REFLECTION)
    gain = reflected-base
    assert_localized_gain(gain)
    with pytest.raises(AssertionError):
        assert_localized_gain(np.full_like(gain, gain.mean()))


@pytest.mark.parametrize('source', [(0.,)*6, (0.,)*8, (0., 1., 2., -.1, 0., 0., 1.),
                                   (0., 1., 2., 1., 1., 1., -1.),
                                   (0., 1., np.nan, 1., 1., 1., 1.)])
def test_invalid_reflection_source_is_rejected_before_tracing(source):
    with pytest.raises(ValueError):
        B.render(None, None, None, None, None, 0., burnished_gilt=True,
                 burnished_reflection=source)
