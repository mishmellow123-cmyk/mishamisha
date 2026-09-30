"""Physical support and local shadow checks for the opt-in last-leaf pen."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import lastleaf_pen as P


@pytest.fixture(scope='module')
def supported():
    book = P.B.Book(TL=2.8, TR=1.2, seed=13)
    return book, P.geometry(book)


def test_actual_triangle_surfaces_touch_at_barrel_and_nib(supported):
    book, mesh = supported
    report = P.contact_report(book, mesh, divisions=9)
    assert report['min_gap_cm'] >= -.003
    assert report['barrel_gap_cm'] < .015
    assert report['nib_gap_cm'] < .015
    assert report['barrel_u'] < .5 < report['nib_u']
    # The Ring plate ends near v=15; the pen's actual surface stays below it.
    assert (book.PH / 2 - mesh[0][:, 1]).min() > 15.5
    assert mesh[0][:, 0].max() < -4.
    assert set(mesh[4]) == {0, 1, 2}
    np.testing.assert_allclose(np.linalg.norm(mesh[1], axis=1), 1., atol=1e-12)


def test_whole_pen_stays_in_the_native_lastleaf_camera(supported):
    import lastleaf
    _, mesh = supported
    scene = lastleaf.LastLeaf(1920, 804, plate=True)
    for frame in (0, 120, 239):
        _, camera = scene.layout(frame / 24.)
        pixels, depth = camera.project(mesh[0])
        assert (depth > 0).all()
        assert pixels[:, 0].min() > 20 and pixels[:, 0].max() < camera.W - 20
        assert pixels[:, 1].min() > 20 and pixels[:, 1].max() < camera.H - 20


def test_support_guard_rejects_a_lifted_or_sunken_pen(supported):
    book, mesh = supported
    for shift, message in ((.20, 'contact'), (-.05, 'intersects')):
        broken = tuple(array.copy() for array in mesh)
        broken[0][:, 2] += shift
        with pytest.raises(ValueError, match=message):
            P.validate_contacts(book, broken)


def test_split_nib_is_an_open_mesh_silhouette(supported):
    _, mesh = supported
    _, _, uv, faces, materials = mesh
    for face, material in zip(faces, materials):
        points = uv[face]
        if points[:, 0].min() >= .91 - 1e-10:
            angular = points[:, 1] * 20 / (2 * np.pi)
            assert not (angular.min() >= 4 - 1e-10 and angular.max() <= 6 + 1e-10)
            assert material == 2


def test_receiver_depth_and_actual_triangle_control_shadow():
    # Identical x/y need opposite answers when a folded receiver rises above
    # the caster. Filling a projected shadow polygon cannot establish this.
    triangles = np.array([[[-1., -1., 1.], [1., -1., 1.], [-1., 1., 1.]]])
    bounds = np.stack((triangles.min((0, 1)), triangles.max((0, 1))))
    G = np.zeros((1, 5, 16))
    G[..., 0] = P.B.M_PAGE_L
    G[0, :, 2:5] = [[-.3, -.3, 0.], [-.3, -.3, 2.], [.8, .8, 0.],
                    [-.3, -.3, 0.], [-.3, -.3, 5.]]
    G[0, 3, 0] = P.B.M_LEATHER
    sources = np.array([[0., 0., 4.]])
    np.testing.assert_array_equal(P._receiver_mask(G, sources, triangles, bounds), [[1., 0., 0., 0., 0.]])
    # The third point crosses the pen AABB but misses the actual triangle;
    # the leather cannot receive a page-only shadow. Two-source visibility
    # gives half coverage, rather than an arbitrary image-space blur.
    sources = np.array([[0., 0., 4.], [8., 8., 4.]])
    np.testing.assert_array_equal(P._receiver_mask(G, sources, triangles, bounds), [[.5, 0., 0., 0., 0.]])


def receivers(book):
    yy, xx = np.meshgrid(np.linspace(-9., 2., 80), np.linspace(-14., 14., 160), indexing='ij')
    G = np.zeros((*xx.shape, 16))
    G[..., 0] = np.where(xx < 0, P.B.M_PAGE_L, P.B.M_PAGE_R)
    G[..., 2], G[..., 3] = xx, yy
    points = G[..., 2:5].reshape(-1, 3)
    G[..., 4] = P._floor(points, book.params, book.ck).reshape(xx.shape)
    return G


@pytest.mark.parametrize('position,radius', [([-55., 50., 30.], 10.), ([-48., -80., 27.], 2.4)])
def test_area_shadow_uses_actual_curved_receiver(supported, position, radius):
    book, mesh = supported
    G = receivers(book)
    light = P.B.Light(position, radius=radius)
    mask = P.shadow_mask(G, light, mesh)
    assert mask.shape == (80, 160)
    assert np.isfinite(mask).all()
    assert mask.min() >= 0 and mask.max() <= 1.000001
    assert mask.max() > .9
    assert (mask == 0).any()
    assert ((mask > 0) & (mask < 1)).any()


def test_reduced_shadow_mesh_preserves_dense_mesh_visibility(supported):
    book, mesh = supported
    G = receivers(book)
    vertices, faces = P.shadow_geometry(mesh)
    bounds = np.stack((vertices.min(0), vertices.max(0)))
    sources = np.array([[-55., 50., 30.], [-48., -80., 27.], [-90., 38., 13.]])
    reduced = P._receiver_mask(G, sources, vertices[faces], bounds)
    dense = P._receiver_mask(G, sources, mesh[0][mesh[3]], bounds)
    assert (dense > 0).any()
    np.testing.assert_array_equal(reduced, dense)


def test_contact_occlusion_disappears_when_pen_is_lifted(supported):
    book, mesh = supported
    camera = P.B.Cam([.1, -38.8, 41.], [.6, .2, 2.], 38., 480, 201)
    mask = P.contact_mask(book, camera, mesh)
    report = P.contact_report(book, mesh)
    points, _ = camera.project(np.array([report['barrel_point'], report['nib_point']]))
    for x, y in points:
        ix, iy = int(x), int(y)
        assert mask[iy - 2:iy + 3, ix - 2:ix + 3].max() > .2
    floating = tuple(array.copy() for array in mesh)
    floating[0][:, 2] += .2
    np.testing.assert_array_equal(P.contact_mask(book, camera, floating), 0.)
