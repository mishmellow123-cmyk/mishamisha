"""Default-off camera study for D's last40 Deep frames (source C4440–4479).

The adopted leaned_ladders drawing, book, light and texture path stay intact. This study translates and dollies
its real pinhole camera, settling by source4476 before D's eight-frame dissolve. An explicit measured target
body size is required: the drawn lamp and crossing lantern have different aspect ratios, so the camera matches
projected bounding-box area without stretching the picture. It cannot make their silhouettes identical.
"""
from dataclasses import dataclass

import numpy as np

import book as B
import book_c_v5_deep_candidates as C


@dataclass(frozen=True)
class LampMatch:
    target_center: tuple
    target_body_size: tuple
    first_frame: int = 4440
    settle_frame: int = 4476

    def __post_init__(self):
        center = np.asarray(self.target_center, np.float64)
        size = np.asarray(self.target_body_size, np.float64)
        if (center.shape != (2,) or size.shape != (2,) or not np.isfinite(center).all()
                or not np.isfinite(size).all() or min(size) <= 0):
            raise ValueError('lamp match requires a finite centre and positive measured body width/height')
        if (type(self.first_frame) is not int or type(self.settle_frame) is not int
                or not 4240 <= self.first_frame < self.settle_frame <= 4479):
            raise ValueError('lamp-match interval must be ordered within the Deep source shot')


def base_camera(book, width=1920, height=804):
    """The existing shot_deep_abandoned camera, with its exact arithmetic preserved."""
    target = book.page_to_world('R', np.array([9.6]), np.array([14.2]))[0]
    cam = B.Cam(target + np.array([-.4, -50., 36.]), target, 38., width, height)
    cam.dof_k = 1.2
    return cam


def lamp_body(book):
    """World-space drawn cap/cage/reservoir paths, excluding the carrying bail (same R3 quantity)."""
    points = np.concatenate(C.CandidateDeep('leaned_ladders').lamp_paths()[1:])
    return book.page_to_world('R', points[:, 0], points[:, 1])


def body_bounds(cam, points):
    screen, _ = cam.project(points)
    depth = (points - cam.pos) @ cam.f  # Cam.project clamps its returned depth; inspect the physical depth here.
    if not np.isfinite(screen).all() or min(depth) <= 0:
        raise ValueError('lamp body must remain in front of the camera')
    return screen.min(axis=0), screen.max(axis=0)


def translated_camera(base, movement):
    shift = base.r * movement[0] + base.u * movement[1] + base.f * movement[2]
    cam = B.Cam(base.pos + shift, base.target + shift, base.hfov, base.W, base.H)
    cam.dof_k = base.dof_k
    return cam


def solve_camera_movement(book, match):
    """Solve three physical camera translations for screen centre and equal projected bounding-box area.

    Uniform perspective magnification preserves the drawn form; matching area balances width/height mismatch
    multiplicatively when the two lamp aspect ratios differ. No pixels are warped or interpolated here.
    """
    base, points = base_camera(book), lamp_body(book)
    center = np.asarray(match.target_center, np.float64)
    area = float(np.prod(match.target_body_size))

    def error(movement):
        low, high = body_bounds(translated_camera(base, movement), points)
        return np.r_[.5 * (low + high) - center, 100. * np.log(np.prod(high - low) / area)]

    movement = np.zeros(3, np.float64)
    for _ in range(30):
        residual = error(movement)
        if np.max(np.abs(residual)) < 1e-7:
            return movement
        eps = 1e-3
        jac = np.column_stack([(error(movement + np.eye(3)[j] * eps) - residual) / eps for j in range(3)])
        step = np.linalg.solve(jac, residual)
        # A bounded Newton step avoids crossing the page plane on the first large framing correction.
        movement -= step * min(1., 10. / max(float(np.linalg.norm(step)), 1e-12))
    raise ValueError('lamp-match camera did not converge')


def camera_at(book, frame, match, width=1920, height=804, movement=None):
    base = base_camera(book, width, height)
    if frame <= match.first_frame:
        return base
    if movement is None:
        movement = solve_camera_movement(book, match)
    u = float(np.clip((frame - match.first_frame) / (match.settle_frame - match.first_frame), 0., 1.))
    eased = u ** 3 * (u * (u * 6. - 15.) + 10.)
    return translated_camera(base, np.asarray(movement) * eased)


class LampMatchPages(C.CandidatePagesV5):
    def __init__(self, match, W=1920, H=804, ppc=110):
        self.match = match
        super().__init__('leaned_ladders', W, H, ppc)

    def shot_deep_abandoned(self, t, f):
        if f <= self.match.first_frame:
            return super().shot_deep_abandoned(t, f)
        bk = self.book(1.6, 2.2, seed=9)
        _, texture = self.once('abandoned', self.abandoned)
        movement = self.once('lamp_match_movement', lambda: solve_camera_movement(bk, self.match))
        cam = camera_at(bk, f, self.match, self.W, self.H, movement)
        light = self.light((-50, 48, 36), 2.1, t, amt=.09)
        return self.finish_layer(bk, cam, light, self.tex_text(31, 40), texture, t)


def make_renderer(match=None, scale=.5, ppc=110):
    """Omitted match returns the unchanged adopted candidate; explicit LampMatch opts into the camera study."""
    if match is None:
        return C.make_renderer('leaned_ladders', scale=scale, ppc=ppc)
    if not isinstance(match, LampMatch):
        raise TypeError('match must be a LampMatch or None')
    if not np.isfinite(scale) or scale <= 0 or min(int(1920 * scale), int(804 * scale)) < 1:
        raise ValueError('scale must produce a finite positive frame')
    return LampMatchPages(match, int(1920 * scale), int(804 * scale), ppc)
