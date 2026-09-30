"""D v2 artwork interruption and exact C22 pen identity, without scene renders."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lastleaf_v2_assets as A


@pytest.fixture(scope='module')
def book_mesh():
    book = A.B.Book(TL=2.8, TR=1.2, seed=13)
    return book, A.geometry(book)


@pytest.fixture(scope='module')
def textures():
    ppc = 40
    old = A.RP.RingPlate(ppc=ppc).texture().chan.copy()
    base = A.RP.RingPlate(ppc=ppc, gap_degrees=50.).texture().chan.copy()
    new = A.leaf(ppc=ppc).chan.copy()
    return ppc, old, base, new


def test_exact_plate_and_first_two_lines_are_unchanged(textures):
    ppc, _, base, new = textures
    np.testing.assert_array_equal(new[int(4.5*ppc):], base[int(4.5*ppc):])
    np.testing.assert_array_equal(new[:int(3.85*ppc)], base[:int(3.85*ppc)])
    assert not np.array_equal(new, base)
    # New assets never mutate the shared RingPlate class or cached strips.
    np.testing.assert_array_equal(A.RP.RingPlate(ppc=ppc, gap_degrees=50.).texture().chan, base)


def test_r2_default_texture_is_unchanged_after_v2_asset_build(textures):
    ppc, old, _, _ = textures
    np.testing.assert_array_equal(A.RP.RingPlate(ppc=ppc).texture().chan, old)


def test_broken_glyph_is_existing_hand_lozenge_stopped_during_its_path():
    strokes, evidence = A.broken_line()
    full, partial = evidence['full_glyph'], evidence['partial_glyph']
    assert evidence['glyph'] == 'lo'
    assert evidence['final_word'] == ('cw', 'df', 'lo')
    np.testing.assert_allclose(full[0], full[-1], atol=1e-12)
    np.testing.assert_allclose(partial, strokes.P[-1], atol=1e-12)
    np.testing.assert_allclose(partial[-1], A.BREAK_UV, atol=1e-12)
    fraction = A.pen.arclen(partial)[-1] / A.pen.arclen(full)[-1]
    assert fraction == pytest.approx(.52, abs=1e-12)
    assert len(evidence['remaining_glyph']) > 10
    # The line nearly fills the normal text block; its shortened last glyph
    # is within a word, rather than the normal short paragraph treatment.
    assert A.BREAK_UV[0]-3.3 > .94*(16.7-3.3)


def test_break_guard_rejects_complete_or_short_paragraph_endings():
    _, evidence = A.broken_line()
    complete = dict(evidence, partial_glyph=evidence['full_glyph'])
    with pytest.raises(ValueError, match='within its pen path'):
        A.validate_break(complete)
    short = dict(evidence, partial_glyph=evidence['partial_glyph']-np.array([4., 0.]))
    with pytest.raises(ValueError, match='meet the nib'):
        A.validate_break(short)


def test_wet_ink_is_small_and_located_at_the_break(textures):
    ppc, _, _, new = textures
    y, x = np.nonzero(new[..., 1] > .01)
    assert len(x) > 0
    assert new[..., 1].max() > .2
    distance = np.hypot((x+.5)/ppc-A.BREAK_UV[0], (y+.5)/ppc-A.BREAK_UV[1])
    assert distance.max() < .07
    assert np.all(new[y, x, 0] > 0.)
    np.testing.assert_array_equal(new[int(13.5*ppc):], 0.)
    assert not new[..., [3, 4, 5, 6]].any()


def assert_c22_identity(book, mesh):
    original = A.PEN.geometry(book)
    for index in (2, 3, 4):
        np.testing.assert_array_equal(mesh[index], original[index])
    # A rigid pose preserves every pairwise vertex distance and normal dot
    # product; a shortened shaft or bent R2 nib fails these assertions.
    for index in (0, 1):
        a = mesh[index][:, None, :]-mesh[index][None, :, :]
        b = original[index][:, None, :]-original[index][None, :, :]
        np.testing.assert_allclose(np.sum(a*a, axis=-1), np.sum(b*b, axis=-1), atol=2e-12)


def test_actual_c22_mesh_is_only_rigidly_reposed(book_mesh):
    book, mesh = book_mesh
    assert A.C.PEN is A.PEN
    assert_c22_identity(book, mesh)
    assert set(mesh[4]) == {0, 1, 2}
    np.testing.assert_allclose(np.linalg.norm(mesh[1], axis=1), 1., atol=1e-12)


def test_c22_identity_guard_rejects_a_bent_nib_and_r2_mesh(book_mesh):
    book, mesh = book_mesh
    bent = tuple(x.copy() for x in mesh)
    bent[0][bent[2][:, 0] > .855, 2] -= .20
    with pytest.raises(AssertionError):
        assert_c22_identity(book, bent)
    with pytest.raises(AssertionError):
        assert_c22_identity(book, A.SUPPORT.geometry(book))


def test_nib_matches_ink_and_both_supports_touch_actual_page(book_mesh):
    book, mesh = book_mesh
    report = A.validate_pose(book, mesh)
    assert report['min_gap_cm'] >= -.003
    assert report['nib_gap_cm'] < .005
    assert report['barrel_gap_cm'] < .005
    assert report['barrel_u'] < .5 < report['nib_u']
    point = book.page_to_world('L', np.array([A.BREAK_UV[0]]), np.array([A.BREAK_UV[1]]))[0]
    np.testing.assert_allclose(A.pen_tip(book, mesh)[:2], point[:2], atol=1e-12)
    # The whole pen occupies the upper page; it cannot become the R2 bottom
    # diagonal that conflicts with the two-row caption area.
    assert (book.PH/2-mesh[0][:, 1]).max() < 10.1
    assert (book.PH/2-mesh[0][:, 1]).min() > 4.


def test_pose_guard_rejects_wrong_stop_floating_and_intersection(book_mesh):
    book, mesh = book_mesh
    for offset, message in (([.1, 0., 0.], 'broken glyph'),
                            ([0., 0., .2], 'contact'),
                            ([0., 0., -.05], 'intersects')):
        changed = tuple(x.copy() for x in mesh)
        changed[0][:] += offset
        with pytest.raises(ValueError, match=message):
            A.validate_pose(book, changed)


def test_compositor_uses_accepted_steel_nib_response(monkeypatch, book_mesh):
    book, mesh = book_mesh
    camera = A.B.Cam([.1, -38.8, 41.], [.6, .2, 2.], 38., 48, 20)
    light = A.B.Light([-55., 50., 30.])
    hdr = np.zeros((20, 48, 3), np.float32)
    G = np.zeros((20, 48, 16))
    depth = np.full((20, 48), 100.)
    calls = []
    def supported(hdr, G, bk, cam, light, actual_mesh):
        assert actual_mesh is mesh
        return hdr, depth
    def steel(hdr, G, actual_depth, actual_mesh, cam, light):
        assert actual_depth is depth and actual_mesh is mesh
        calls.append('accepted steel')
        return hdr+1
    monkeypatch.setattr(A.SUPPORT, 'composite', supported)
    monkeypatch.setattr(A.C, 'replace_visible_nib', steel)
    result, actual_depth = A.composite(hdr, G, book, camera, light, mesh)
    assert calls == ['accepted steel']
    assert actual_depth is depth
    np.testing.assert_array_equal(result, 1.)


def test_shadow_disk_is_deterministic_centered_and_matches_light(book_mesh):
    _, mesh = book_mesh
    light = A.B.Light([-55., 50., 30.], radius=10.)
    points = A.shadow_sources(light, mesh)
    assert points.shape == (65, 3)
    np.testing.assert_array_equal(points, A.shadow_sources(light, mesh))
    np.testing.assert_allclose(points.mean(0), light.pos, atol=1e-12)
    offsets = points-light.pos
    toward = mesh[0].mean(0)-light.pos
    toward /= np.linalg.norm(toward)
    np.testing.assert_allclose(offsets@toward, 0., atol=1e-12)
    assert np.linalg.norm(offsets, axis=1).max() < light.radius
    assert np.linalg.norm(offsets, axis=1).max() > .98*light.radius
    translated = tuple(array.copy() for array in mesh)
    shift = np.array([3., -2., 1.])
    translated[0][:] += shift
    moved = A.B.Light(light.pos+shift, radius=light.radius)
    np.testing.assert_allclose(A.shadow_sources(moved, translated), points+shift, atol=1e-12)
    point_light = A.B.Light(light.pos, radius=0.)
    np.testing.assert_allclose(A.shadow_sources(point_light, mesh), np.tile(light.pos, (65, 1)))


def test_long_pen_shadow_rejects_nine_sample_banding(book_mesh):
    book, mesh = book_mesh
    light = A.B.Light([-55., 50., 30.], radius=10.)
    # A fine physical receiver cross-section under the suspended shaft.
    # This is a real curved page and actual C22 triangles, without tracing
    # a whole image. The inherited nine-source mask is the negative control.
    G = np.zeros((1, 801, 16))
    G[..., 0] = A.B.M_PAGE_R
    G[0, :, 2] = 8.
    G[0, :, 3] = np.linspace(1., 11., 801)
    G[0, :, 4] = A.SUPPORT._floor(G[0, :, 2:5], book.params, book.ck)
    old = A.SUPPORT.shadow_mask(G, light, mesh)[0]
    new = A.shadow_mask(G, light, mesh)[0]
    reference = A.shadow_mask(G, light, mesh, samples=257)[0]
    def assert_smooth_profile(profile):
        assert len(np.unique(profile)) > 25
        assert np.abs(np.diff(profile)).max() < .08
    assert_smooth_profile(new)
    with pytest.raises(AssertionError):
        assert_smooth_profile(old)
    assert np.abs(new-reference).mean() < .5*np.abs(old-reference).mean()
    assert np.abs(new-reference).max() < .06


def test_dense_shadow_replaces_old_factor_without_double_darkening(monkeypatch, book_mesh):
    book, mesh = book_mesh
    old = np.array([[0., .5, 1., 1.]], np.float32)
    new = np.array([[.4, .2, 1., 0.]], np.float32)
    contact = np.array([[1., .5, .6, 1.]], np.float32)
    depth = np.ones_like(old)
    monkeypatch.setattr(A.SUPPORT, 'shadow_mask', lambda *args: old)
    monkeypatch.setattr(A, 'shadow_mask', lambda *args: new)
    def inherited(hdr, *args):
        return hdr*(1.-.88*old[..., None])*contact[..., None], depth
    monkeypatch.setattr(A.SUPPORT, 'composite', inherited)
    monkeypatch.setattr(A.C, 'replace_visible_nib', lambda hdr, *args: hdr)
    result, _ = A.composite(np.ones((1, 4, 3), np.float32), None, book, None, None, mesh)
    expected = np.repeat(((1.-.88*new)*contact)[..., None], 3, axis=2)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    assert np.isfinite(result).all()
