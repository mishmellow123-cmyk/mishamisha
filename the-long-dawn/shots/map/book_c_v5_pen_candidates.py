"""Default-off Pen studies; accepted renderer, geometry and shadows stay intact.

soft_spine adds local bounced paper light inside the existing curved gutter.
soft_spine_metal adds the same light and changes only the visible nib material.
Both retain the original camera, book surface, solved mesh, shadow and DOF.
No renderer globals are replaced. Candidate appearance needs rendered review.
"""
from pathlib import Path
import re

import numpy as np
from numba import njit

import book_c_v5 as V
import book as B
import v5_pen as PEN


CANDIDATES = ('accepted', 'soft_spine', 'soft_spine_metal')


@njit(cache=True)
def _gutter_bounce(out, G, PW, PH, seed, light_col):
    """One added indirect-light term on shaded page surfaces, in world cm.

The original page normals, direct-light shadows and occlusion are untouched.
The seam retains a lower light level than its curved shoulders. The compact
support reaches zero with zero slope at 3.2 cm from the spine; lit paper and
every non-page material receive no term. Existing procedural paper supplies
the albedo, so this does not paint a smooth stripe over its texture.
"""
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            material = int(G[y, x, 0])
            if material != B.M_PAGE_L and material != B.M_PAGE_R:
                continue
            distance = abs(G[y, x, 2])
            if distance >= 3.2:
                continue
            shadow = min(max(G[y, x, 11], 0.), 1.)
            if shadow >= 1.:
                continue
            shoulder = min(distance/.8, 1.)
            shoulder = shoulder*shoulder*(3.-2.*shoulder)
            taper = (1.-(distance/3.2)**2)**2
            amount = .11*(.42+.58*shoulder)*taper*(1.-shadow)**2
            right = material == B.M_PAGE_R
            paper = B.paper(G[y, x, 8], G[y, x, 9], G[y, x, 10], PW, PH,
                            seed+17 if right else seed, 1., right)
            for channel in range(3):
                out[y, x, channel] += amount*paper[channel]*light_col[channel]


def soften_spine(hdr, G, bk, light):
    result = hdr.copy()
    _gutter_bounce(result, G, float(bk.PW), float(bk.PH), float(bk.seed), light.col)
    return result


def metal_nib_rgb(position, normal, uv, cam, light):
    """Dark steel, directional reflections and a small dark breather mark.

The existing open split and the iron-gall-stained tip remain in the mesh and
material respectively. The breather is a surface mark, not a claimed new hole.
"""
    N = normal/np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    eye = cam.pos-position
    eye /= np.maximum(np.linalg.norm(eye, axis=-1, keepdims=True), 1e-12)
    N = np.where((N*eye).sum(-1, keepdims=True) < 0, -N, N)
    to_light = light.pos-position
    to_light /= np.maximum(np.linalg.norm(to_light, axis=-1, keepdims=True), 1e-12)
    half = to_light+eye
    half /= np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1e-12)
    diffuse = np.maximum((N*to_light).sum(-1), 0)
    glint = np.maximum((N*half).sum(-1), 0)**100
    reflected = 2*(N*eye).sum(-1, keepdims=True)*N-eye
    room_direction = np.array([.3, .5, .81])
    room_direction /= np.linalg.norm(room_direction)
    room_glint = np.maximum(reflected@room_direction, 0)**48
    rgb = (np.array([.065, .070, .075])*(.18+.36*diffuse[..., None])*light.col
           + .38*glint[..., None]*light.col
           + np.array([.018, .022, .025])
           + room_glint[..., None]*np.array([.20, .23, .25]))
    # A restrained dark oval ahead of the ferrule; no silhouette/pose edits.
    oval = ((uv[..., 0]-.900)/.006)**2+(np.cos(uv[..., 1])/.22)**2
    mark = np.clip((1.-oval)/.3, 0, 1)*(np.sin(uv[..., 1]) > 0)
    rgb = rgb*(1-.92*mark[..., None])
    # Exactly the accepted tip-stain function.
    rgb *= 1-.75*np.clip((uv[..., 0]-.968)/.032, 0, 1)[..., None]
    return rgb


def replace_visible_nib(hdr, G, depth, mesh, cam, light):
    """Repaint only metal faces that won the original compositor's z test.

The original compositor has already solved shadow and depth. Reprojection is
identical, so exact depth equality identifies its visible triangles without a
screen-space ROI. Page, shaft, ferrule, slit and shadow pixels stay unchanged.
"""
    verts, normals, uv, faces, materials = mesh
    result = hdr.copy()
    for triangle, material in zip(faces, materials):
        if material != 2:
            continue
        projected = PEN._pixels(cam, verts, triangle)
        if projected is None:
            continue
        sl, weights, z, inside = projected
        hit = inside & (z == depth[sl]) & (z < G[sl][..., 1])
        if not hit.any():
            continue
        rgb = metal_nib_rgb(weights@verts[triangle], weights@normals[triangle],
                            weights@uv[triangle], cam, light)
        result[sl][hit] = rgb[hit]
    return result


class CandidatePagesV5(V.PagesV5):
    def __init__(self, candidate, W=960, H=402, ppc=110):
        if candidate not in CANDIDATES[1:]:
            raise ValueError('CandidatePagesV5 requires an explicit Pen study')
        self.candidate = candidate
        super().__init__('pen', W, H, ppc)

    def shot_pen(self, t, f):
        # These setup values are copied exactly from the accepted shot. The
        # only inserted operations are paper bounce and optional nib material.
        bk = self.book(2.8, 1.2, seed=13)
        target = np.array([.6, .2, 2.0])
        cam = B.Cam(target+np.array([-.5, -39., 39.])*(1-.025*V.BC.smooth(t/10)),
                    target, 38., self.W, self.H)
        cam.dof_k = 2.
        light = self.light((-55, 50, 30), 1.9, t, seed=5, amt=.14, col=(1., .52, .24), radius=10)
        hdr, alpha, G = B.render(bk, cam, light, self.blank, self.blank, t,
                                 fill=((.35, -.5, .8), (.010, .012, .016)))
        hdr = soften_spine(hdr, G, bk, light)
        mesh = self.once('physical_pen', lambda: PEN.geometry(bk))
        hdr, depth = PEN.composite(hdr, G, bk, cam, light, mesh)
        if self.candidate == 'soft_spine_metal':
            hdr = replace_visible_nib(hdr, G, depth, mesh, cam, light)
        return B.dof(hdr, depth.astype(np.float32), cam, cam.dof_k*self.W/1920), alpha


def make_renderer(candidate='accepted', scale=.5, ppc=110):
    if candidate not in CANDIDATES:
        raise ValueError('Unknown Pen candidate')
    if not np.isfinite(scale) or scale <= 0 or int(1920*scale) < 1 or int(804*scale) < 1:
        raise ValueError('Scale must produce positive dimensions')
    if not isinstance(ppc, int) or isinstance(ppc, bool) or ppc <= 0:
        raise ValueError('Page resolution must be a positive integer')
    width, height = int(1920*scale), int(804*scale)
    if candidate == 'accepted':
        return V.PagesV5('pen', width, height, ppc)
    return CandidatePagesV5(candidate, width, height, ppc)


def validate_output_dir(path):
    path = Path(path)
    root = Path.home()/'ldfarm/out'
    if (not path.is_absolute() or path.is_symlink() or path.resolve().parent != root.resolve()
            or not re.fullmatch(r'cand_pen_[a-z0-9_-]+', path.name)):
        raise ValueError('Pen studies require a new cand_pen_* directory under ldfarm/out')
    return path.resolve()
