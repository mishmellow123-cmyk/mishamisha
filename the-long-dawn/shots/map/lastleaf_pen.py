"""Opt-in last-leaf pen: supported lower-page pose and spatial area shadows.

The accepted v5 pen renderer is unchanged. This mesh retains its section
profile, materials and open nib slit, but rests below the illustration.
Shadow softness follows source size and actual separation at each point.
"""
import math

import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar

import book as B
import v5_pen as P


START = np.array([-17.8, -1.3, 0.])
END = np.array([-4.5, -7.8, 0.])
NIB_DROP = .24  # cm of downward curvature from ferrule to writing tip.
SECTIONS = np.array([
    (0., .055, .055, 0), (.025, .14, .14, 0), (.10, .225, .225, 0),
    (.63, .22, .22, 0), (.78, .17, .17, 0), (.80, .19, .19, 1),
    (.855, .19, .19, 1), (.865, .24, .055, 2), (.91, .28, .05, 2),
    (.98, .11, .025, 2), (1., .012, .012, 2),
])
CLEARANCE = .0015  # cm; small separation avoids depth fighting at support.


@njit(cache=True)
def _floor(points, params, cockle):
    height = np.empty(len(points), np.float64)
    for i in range(len(points)):
        height[i] = B.height(points[i, 0], points[i, 1], params, cockle)[0]
    return height


def _shape(slope, lift=0.):
    # Extra axial samples locate physical support against the cockled page.
    # Shadow visibility uses the original planar faces, never their receiver
    # projections: a projected triangle can bridge a curved gutter incorrectly.
    sections = []
    for left, right in zip(SECTIONS[:-1], SECTIONS[1:]):
        steps = max(1, int(math.ceil((right[0] - left[0]) / .035)))
        for k in range(steps):
            u = k / steps
            value = left * (1 - u) + right * u
            value[3] = right[3]
            sections.append(value)
    sections.append(SECTIONS[-1].copy())
    a, b = START.copy(), END.copy()
    a[2], b[2] = lift, lift + slope
    axis = b - a
    direction = axis / np.linalg.norm(axis)
    right = np.cross(direction, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    vertices, normals, uv, faces, materials = [], [], [], [], []
    sides = 20
    for u, rx, rz, _ in sections:
        curve = max((u - .855) / .145, 0.)
        bend = NIB_DROP * curve * curve
        gradient = direction * (2 * NIB_DROP * curve / .145 / np.linalg.norm(axis))
        for j in range(sides):
            angle = 2 * np.pi * j / sides
            position = a + axis * u + right * rx * np.cos(angle) + up * rz * np.sin(angle)
            position[2] -= bend
            vertices.append(position)
            normal = right * np.cos(angle) / rx + up * np.sin(angle) / rz
            normal = normal + gradient * normal[2] / (1 - gradient[2])
            normals.append(normal / np.linalg.norm(normal))
            uv.append((u, angle))
    for k, section in enumerate(sections[:-1]):
        for j in range(sides):
            if section[0] >= .91 - 1e-10 and j in (4, 5):
                continue
            nxt = (j + 1) % sides
            x, y, z, w = k * sides + j, k * sides + nxt, (k + 1) * sides + j, (k + 1) * sides + nxt
            faces.extend(((x, z, y), (y, z, w)))
            materials.extend((int(section[3]), int(section[3])))
    return (np.array(vertices), np.array(normals), np.array(uv),
            np.array(faces), np.array(materials))


def surface_samples(mesh, divisions=3):
    """Sample actual triangle interiors; the supports need not be vertices."""
    vertices, _, uv, faces, _ = mesh
    weights = np.array([(a / divisions, b / divisions, 1 - (a + b) / divisions)
                        for a in range(divisions + 1) for b in range(divisions + 1 - a)])
    points = np.einsum('ij,fjk->fik', weights, vertices[faces]).reshape(-1, 3)
    along = np.einsum('ij,fj->fi', weights, uv[faces, 0]).ravel()
    return points, along


def contact_report(bk, mesh, divisions=3):
    points, along = surface_samples(mesh, divisions)
    gaps = points[:, 2] - _floor(points, bk.params, bk.ck)
    report = {'min_gap_cm': float(gaps.min())}
    for name, selected in (('barrel', along < .78), ('nib', along >= .98)):
        candidates = np.flatnonzero(selected)
        index = candidates[np.argmin(gaps[candidates])]
        report[name + '_gap_cm'] = float(gaps[index])
        report[name + '_u'] = float(along[index])
        report[name + '_point'] = points[index].tolist()
    return report


def validate_contacts(bk, mesh):
    report = contact_report(bk, mesh)
    if report['min_gap_cm'] < -.003:
        raise ValueError('The pen intersects the page')
    if report['barrel_gap_cm'] > .015 or report['nib_gap_cm'] > .015:
        raise ValueError('The pen needs both barrel and nib contact')
    if report['nib_u'] - report['barrel_u'] < .4:
        raise ValueError('The pen support points do not span its centre of mass')
    return report


def geometry(bk):
    q = np.linspace(0, 1, 401)
    mass = np.interp(q, SECTIONS[:, 0], SECTIONS[:, 1]) * np.interp(q, SECTIONS[:, 0], SECTIONS[:, 2])
    centre = float(np.sum(q * mass) / mass.sum())

    def pose(slope):
        mesh = _shape(slope)
        points, _ = surface_samples(mesh, 2)
        lift = float(np.max(_floor(points, bk.params, bk.ck) - points[:, 2]) + CLEARANCE)
        return lift, mesh

    fit = minimize_scalar(lambda slope: pose(slope)[0] + slope * centre,
                          bounds=(-8., 8.), method='bounded', options={'xatol': 1e-10})
    if not fit.success:
        raise RuntimeError('The last-leaf pen support solve failed')
    lift, mesh = pose(fit.x)
    mesh[0][:, 2] += lift
    # A denser final surface check accounts for points between solve samples.
    points, _ = surface_samples(mesh, 5)
    residue = float(np.min(points[:, 2] - _floor(points, bk.params, bk.ck)))
    mesh[0][:, 2] += max(0., CLEARANCE - residue)
    validate_contacts(bk, mesh)
    return mesh


def shadow_geometry(mesh):
    """Remove only linear subdivisions; retain the actual shape and nib slit."""
    vertices, _, uv, _, _ = mesh
    selected = []
    retained = []
    for along in np.unique(uv[:, 0]):
        if np.any(np.isclose(SECTIONS[:, 0], along, rtol=0., atol=1e-10)):
            retained.append(along)
            continue
        k = int(np.searchsorted(SECTIONS[:, 0], along)) - 1
        first, second = SECTIONS[k:k + 2]
        # Circular/homothetic cross-sections form planar ruled faces. Keep
        # intermediate stations when the nib's aspect ratio changes: those
        # faces can be warped and their triangulation matters to visibility.
        if abs(first[1] * second[2] - first[2] * second[1]) > 1e-12:
            retained.append(along)
    for along in retained:
        section = np.flatnonzero(np.isclose(uv[:, 0], along, rtol=0., atol=1e-10))
        if len(section) != 20:
            raise ValueError('Shadow reduction requires the last-leaf section topology')
        selected.extend(section)
    faces = []
    for k, along in enumerate(retained[:-1]):
        for j in range(20):
            if along >= .91 - 1e-10 and j in (4, 5):
                continue
            nxt = (j + 1) % 20
            x, y, z, w = k * 20 + j, k * 20 + nxt, (k + 1) * 20 + j, (k + 1) * 20 + nxt
            faces.extend(((x, z, y), (y, z, w)))
    return vertices[selected], np.array(faces, np.int64)


@njit(cache=True)
def _occluded(origin, source, triangles, bounds):
    """AABB rejection, then two-sided Möller–Trumbore on the light segment."""
    direction = source - origin
    near, far = 1e-8, 1. - 1e-8
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < bounds[0, axis] or origin[axis] > bounds[1, axis]:
                return False
        else:
            first = (bounds[0, axis] - origin[axis]) / direction[axis]
            second = (bounds[1, axis] - origin[axis]) / direction[axis]
            near = max(near, min(first, second))
            far = min(far, max(first, second))
            if near > far:
                return False
    dx, dy, dz = direction
    for triangle in triangles:
        ax, ay, az = triangle[0]
        e1x, e1y, e1z = triangle[1, 0] - ax, triangle[1, 1] - ay, triangle[1, 2] - az
        e2x, e2y, e2z = triangle[2, 0] - ax, triangle[2, 1] - ay, triangle[2, 2] - az
        hx, hy, hz = dy * e2z - dz * e2y, dz * e2x - dx * e2z, dx * e2y - dy * e2x
        determinant = e1x * hx + e1y * hy + e1z * hz
        if abs(determinant) < 1e-12:
            continue
        inverse = 1. / determinant
        sx, sy, sz = origin[0] - ax, origin[1] - ay, origin[2] - az
        u = (sx * hx + sy * hy + sz * hz) * inverse
        if u < -1e-10 or u > 1 + 1e-10:
            continue
        qx, qy, qz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
        v = (dx * qx + dy * qy + dz * qz) * inverse
        if v < -1e-10 or u + v > 1 + 1e-10:
            continue
        along = (e2x * qx + e2y * qy + e2z * qz) * inverse
        if 1e-8 < along < 1. - 1e-8:
            return True
    return False


@njit(cache=True)
def _receiver_mask(G, sources, triangles, bounds):
    result = np.zeros(G.shape[:2], np.float32)
    for y in range(G.shape[0]):
        for x in range(G.shape[1]):
            material = int(G[y, x, 0])
            if material != B.M_PAGE_L and material != B.M_PAGE_R:
                continue
            count = 0
            for source in sources:
                if _occluded(G[y, x, 2:5], source, triangles, bounds):
                    count += 1
            result[y, x] = count / len(sources)
    return result


def shadow_mask(G, light, mesh):
    """Area-light visibility at each actual visible receiver position.

    No shadow vertex is projected onto the page, and no triangle is filled
    between unrelated receiver intersections across the gutter.
    """
    vertices, faces = shadow_geometry(mesh)
    toward = np.mean(vertices, axis=0) - light.pos
    toward /= np.linalg.norm(toward)
    right = np.cross(toward, [0., 0., 1.])
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, toward)
    # Nine deterministic disk samples. At contact their shadows coincide;
    # separation broadens them naturally away from the contact point.
    offsets = [(0., 0.)] + [(.8 * np.cos(i * np.pi / 4), .8 * np.sin(i * np.pi / 4)) for i in range(8)]
    sources = np.array([light.pos + light.radius * (right * x + up * y) for x, y in offsets])
    return _receiver_mask(G, sources, vertices[faces], np.stack((vertices.min(0), vertices.max(0))))


@njit(cache=True)
def _contact_splats(mask, pixels, depth, gaps, focal):
    for i in range(len(pixels)):
        sigma = max(.55, .035 * focal / depth[i])
        radius = int(math.ceil(3 * sigma))
        cx, cy = pixels[i, 0], pixels[i, 1]
        amount = math.exp(-(max(gaps[i], 0.) / .035) ** 2)
        for y in range(max(0, int(cy) - radius), min(mask.shape[0], int(cy) + radius + 1)):
            for x in range(max(0, int(cx) - radius), min(mask.shape[1], int(cx) + radius + 1)):
                distance = (x + .5 - cx) ** 2 + (y + .5 - cy) ** 2
                mask[y, x] = max(mask[y, x], amount * math.exp(-.5 * distance / sigma ** 2))


def contact_mask(bk, cam, mesh):
    """Short-range ambient occlusion from the actual supported underside.

    A low light can cast past a downward page shoulder even at support. This
    local term depicts the paper/pen contact independently of that cast shadow;
    no point farther than .06 cm above the paper contributes.
    """
    points, _ = surface_samples(mesh, 3)
    ground = _floor(points, bk.params, bk.ck)
    gaps = points[:, 2] - ground
    near = (gaps >= -.003) & (gaps < .06)
    result = np.zeros((cam.H, cam.W), np.float32)
    if not near.any():
        return result
    points = points[near].copy()
    points[:, 2] = ground[near]
    pixels, depth = cam.project(points)
    _contact_splats(result, pixels, depth, gaps[near], cam.F)
    return result


def composite(hdr, G, bk, cam, light, mesh=None):
    mesh = geometry(bk) if mesh is None else mesh
    vertices, normals, uv, faces, materials = mesh
    pages = (G[..., 0] == B.M_PAGE_L) | (G[..., 0] == B.M_PAGE_R)
    hdr *= 1 - .88 * shadow_mask(G, light, mesh)[..., None]
    hdr *= 1 - .55 * (contact_mask(bk, cam, mesh) * pages)[..., None]
    depth = G[..., 1].copy()
    # Accepted material equations and perspective-correct triangle projection.
    for triangle, material in zip(faces, materials):
        projected = P._pixels(cam, vertices, triangle)
        if projected is None:
            continue
        sl, weights, z, inside = projected
        hit = inside & (z < depth[sl])
        if not hit.any():
            continue
        position = weights @ vertices[triangle]
        normal = weights @ normals[triangle]
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        eye = cam.pos - position
        eye /= np.linalg.norm(eye, axis=-1, keepdims=True)
        normal = np.where((normal * eye).sum(-1, keepdims=True) < 0, -normal, normal)
        incoming = light.pos - position
        incoming /= np.linalg.norm(incoming, axis=-1, keepdims=True)
        half = incoming + eye
        half /= np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1e-12)
        diffuse = np.maximum((normal * incoming).sum(-1), 0)
        specular = np.maximum((normal * half).sum(-1), 0) ** (32 if material == 0 else 65)
        coordinates = weights @ uv[triangle]
        if material == 0:
            grain = .88 + .08 * np.sin(coordinates[..., 0] * 110 + np.sin(coordinates[..., 1] * 7)) + .04 * np.sin(coordinates[..., 1] * 21)
            albedo, shine = grain[..., None] * np.array([.13, .065, .028]), .17
        else:
            albedo = np.array([.36, .25, .105]) if material == 1 else np.array([.32, .30, .25])
            shine = .55
        rgb = albedo * (.13 + .84 * diffuse[..., None]) * light.col + shine * specular[..., None] * light.col
        if material == 2:
            rgb *= 1 - .75 * np.clip((coordinates[..., 0] - .968) / .032, 0, 1)[..., None]
        hdr[sl][hit], depth[sl][hit] = rgb[hit], z[hit]
    return hdr, depth
