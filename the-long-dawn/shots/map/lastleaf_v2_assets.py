"""Opt-in D last-leaf assets: the accepted plate, interrupted hand and C22 pen.

Page coordinates are centimetres on the verso. The drawing and its double
frame are copied from RingPlate without alteration; only its final header
line changes. The pen is a rigid pose of v5_pen's actual mesh, with its
original wood, ferrule, steel-nib material and split geometry.
"""
import math

import numpy as np
from scipy.optimize import brentq

import book as B
import book_c_v5_pen_candidates as C
import lastleaf_pen as SUPPORT
import pen
import ringpage as RP
import v5_pen as PEN


BREAK_UV = (16.05, 4.10)
LINE_BASE = 4.14
CLEARANCE = .0015
# From shaft to nib: the shaft rests across the upper recto, away from the plate.
PEN_AZIMUTH = math.radians(162.)
SHADOW_SAMPLES = 65


def _append_stroke(dst, src, index, stop=None):
    end = len(src.P[index]) if stop is None else stop
    dst.add(src.P[index][:end], src.R[index][:end], src.D[index][:end],
            layer=src.L[index], gain=src.G[index], nib=src.N[index])


def broken_line():
    """Return the real Hand strokes and evidence of the interrupted final glyph.

    The last word contains three of the existing book-hand glyphs. Its last
    lozenge is stopped halfway through the pen path; its missing upper return
    is not a shortened paragraph or a complete small glyph.
    """
    hand = pen.Hand(seed=121, xh=.20, nib=.20, thin=.03, dens=.88)
    word = pen.Strokes()
    x = 0.
    for glyph in ('cw', 'df'):
        x = hand.write_word(word, [glyph], x, 0.) + hand.gap * hand.xh
    glyph_start = len(word)
    hand.write_word(word, ['lo'], x, 0.)
    full = word.P[glyph_start].copy()
    lengths = pen.arclen(full)
    cut_length = .52 * lengths[-1]
    segment = int(np.searchsorted(lengths, cut_length))
    q = (cut_length - lengths[segment-1]) / (lengths[segment] - lengths[segment-1])
    endpoint = full[segment-1] * (1-q) + full[segment] * q
    shift = np.asarray(BREAK_UV) - endpoint
    for points in word.P:
        points += shift
    full += shift
    partial = np.vstack((full[:segment], BREAK_UV))
    radii = np.r_[word.R[glyph_start][:segment],
                  word.R[glyph_start][segment-1] * (1-q) + word.R[glyph_start][segment] * q]
    density = np.r_[word.D[glyph_start][:segment],
                    word.D[glyph_start][segment-1] * (1-q) + word.D[glyph_start][segment] * q]
    strokes = pen.Strokes()
    # Full-width preceding words establish that the terminal is interrupted.
    prefix = pen.Hand(seed=120, xh=.20, nib=.20, thin=.03, dens=.88)
    prefix.write_block(strokes, 3.3, LINE_BASE, shift[0]-3.3-.10,
                       1, .62, last_frac=None)
    for index in range(glyph_start):
        _append_stroke(strokes, word, index)
    strokes.add(partial, radii, density, nib=word.N[glyph_start])
    evidence = dict(glyph='lo', final_word=('cw', 'df', 'lo'),
                    full_glyph=full, partial_glyph=partial,
                    remaining_glyph=np.vstack((BREAK_UV, full[segment:])),
                    fraction=.52, break_uv=BREAK_UV)
    validate_break(evidence)
    return strokes, evidence


def validate_break(evidence):
    full, partial = evidence['full_glyph'], evidence['partial_glyph']
    fraction = pen.arclen(partial)[-1] / pen.arclen(full)[-1]
    if not .35 < fraction < .65 or len(evidence['remaining_glyph']) < 3:
        raise ValueError('The last glyph must stop within its pen path')
    if not np.allclose(partial[-1], BREAK_UV, atol=1e-12):
        raise ValueError('The last ink point must meet the nib')
    if np.linalg.norm(full[-1]-partial[-1]) < .04:
        raise ValueError('A completed glyph is not an interrupted line')
    return evidence


def leaf(ppc=90, gap_degrees=50.):
    """A built PageTex. Existing RingPlate drawing/frame pixels are preserved."""
    texture = RP.RingPlate(ppc=ppc, gap_degrees=gap_degrees).texture()
    # Only the short third header line is removed; the first two lines remain.
    first, last = int(math.floor(3.85*ppc)), int(math.ceil(4.5*ppc))
    texture.chan[first:last, :, :] = 0.
    strokes, _ = broken_line()
    ink, _ = pen.raster(strokes.pack(), 1e9, ppc, last-first, texture.W,
                        pen.INK, oy=first/ppc)
    texture.chan[first:last, :, 0] = ink
    # A small wet bead at the point where the nib stopped, on existing ink.
    bead = pen.Strokes()
    bead.add([np.asarray(BREAK_UV)-[.006, 0.], BREAK_UV], .016, .96)
    wet_ink, _ = pen.raster(bead.pack(), 1e9, ppc, last-first, texture.W,
                            pen.INK, oy=first/ppc)
    np.maximum(texture.chan[first:last, :, 0], wet_ink,
               out=texture.chan[first:last, :, 0])
    texture.chan[first:last, :, 1] = .78 * wet_ink
    return texture.build()


def _basis(direction):
    direction = direction / np.linalg.norm(direction)
    right = np.cross(direction, [0., 0., 1.])
    right /= np.linalg.norm(right)
    return np.column_stack((direction, right, np.cross(right, direction)))


def tip_index(mesh):
    uv = mesh[2]
    candidates = np.flatnonzero(np.isclose(uv[:, 0], 1.))
    return int(candidates[np.argmin(np.abs(uv[candidates, 1]-1.5*np.pi))])


def pen_tip(bk, mesh=None):
    mesh = geometry(bk) if mesh is None else mesh
    return mesh[0][tip_index(mesh)].copy()


def geometry(bk):
    """Rigidly pose the exact accepted C22 mesh with writing tip at BREAK_UV.

    Pitch is solved against actual triangle interiors. The fixed nib and a
    second barrel contact support the pen without the R2 nib-bending change.
    """
    source = PEN.geometry(bk)
    points, normals, uv, faces, materials = source
    a = points[np.isclose(uv[:, 0], 0.)].mean(0)
    b = points[np.isclose(uv[:, 0], 1.)].mean(0)
    local = (points-a) @ _basis(b-a)
    local_normals = normals @ _basis(b-a)
    tip = tip_index(source)
    target = bk.page_to_world('L', np.array([BREAK_UV[0]]), np.array([BREAK_UV[1]]))[0]
    target[2] = B.height(target[0], target[1], bk.params, bk.ck)[0] + CLEARANCE

    def posed(pitch):
        direction = np.array([math.cos(PEN_AZIMUTH)*math.cos(pitch),
                              math.sin(PEN_AZIMUTH)*math.cos(pitch), math.sin(pitch)])
        basis = _basis(direction)
        positioned = (local-local[tip]) @ basis.T + target
        return positioned, local_normals @ basis.T, uv.copy(), faces.copy(), materials.copy()

    def clearance(pitch):
        mesh = posed(pitch)
        samples, along = SUPPORT.surface_samples(mesh, divisions=5)
        samples = samples[along < 1.-1e-8]
        return float(np.min(samples[:, 2]-SUPPORT._floor(samples, bk.params, bk.ck))-CLEARANCE)

    pitch = brentq(clearance, -.25, .35, xtol=1e-12)
    mesh = posed(pitch)
    validate_pose(bk, mesh)
    return mesh


def validate_pose(bk, mesh):
    report = SUPPORT.contact_report(bk, mesh, divisions=9)
    target = bk.page_to_world('L', np.array([BREAK_UV[0]]), np.array([BREAK_UV[1]]))[0]
    tip = pen_tip(bk, mesh)
    if np.linalg.norm(tip[:2]-target[:2]) > .003:
        raise ValueError('The nib does not meet the broken glyph')
    if report['min_gap_cm'] < -.003:
        raise ValueError('The C22 pen intersects the page')
    if report['barrel_gap_cm'] > .015 or report['nib_gap_cm'] > .015:
        raise ValueError('The C22 pen needs barrel and nib contact')
    if report['barrel_u'] >= .65:
        raise ValueError('The C22 pen lacks support behind its centre of mass')
    report['tip'] = tip.tolist()
    report['break_uv'] = BREAK_UV
    return report


def shadow_sources(light, mesh, samples=SHADOW_SAMPLES):
    """Fixed sunflower quadrature over the actual area-light disk.

    Antipodal pairs keep the source centroid exactly at the light position.
    The disk faces the pen, as in the shared physical shadow implementation.
    No camera or frame value participates, so the pattern cannot crawl.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3 or samples % 2 != 1:
        raise ValueError('Area shadow sampling needs an odd count of at least three')
    vertices, _ = SUPPORT.shadow_geometry(mesh)
    toward = vertices.mean(0)-light.pos
    toward /= np.linalg.norm(toward)
    right = np.cross(toward, [0., 0., 1.])
    norm = np.linalg.norm(right)
    right = right/norm if norm > 1e-12 else np.array([1., 0., 0.])
    up = np.cross(right, toward)
    count = (samples-1)//2
    index = np.arange(count)
    radius = np.sqrt((index+.5)/count)
    angle = index * (np.pi*(3.-np.sqrt(5.)))
    offsets = radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    offsets = np.vstack((np.zeros((1, 2)), offsets, -offsets))
    return light.pos + light.radius * (offsets[:, :1]*right + offsets[:, 1:]*up)


def shadow_mask(G, light, mesh, samples=SHADOW_SAMPLES):
    vertices, faces = SUPPORT.shadow_geometry(mesh)
    return SUPPORT._receiver_mask(G, shadow_sources(light, mesh, samples),
                                   vertices[faces], np.stack((vertices.min(0), vertices.max(0))))


def composite(hdr, G, bk, cam, light, mesh=None):
    """C22 materials with the contact-aware shadow path, before shared DOF.

    The steel nib's fixed room reflection is included, exactly as in C22.
    A renderer with two light passes must subtract that term from its second
    pass, as the existing last-leaf renderer does.
    """
    mesh = geometry(bk) if mesh is None else mesh
    # Replace only the inherited nine-point cast shadow. Its factor remains
    # in SUPPORT.composite, so pre-divide it and insert the denser visibility;
    # contact occlusion, depth and accepted material equations remain shared.
    old_shadow = SUPPORT.shadow_mask(G, light, mesh)
    new_shadow = shadow_mask(G, light, mesh)
    hdr *= ((1.-.88*new_shadow)/np.maximum(1.-.88*old_shadow, .12))[..., None]
    hdr, depth = SUPPORT.composite(hdr, G, bk, cam, light, mesh)
    return C.replace_visible_nib(hdr, G, depth, mesh, cam, light), depth
