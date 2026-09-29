"""Geometry-only C3 inscription audit. No page textures, scene renders or farm API.

Run with --csv PATH to record every C400..559 frame. Bounds include the source
glyphs' wet-bleed support, even before write-on and after dissolve. Active ink
columns are null when the source callback writes no ink. These are projected
source bounds, not predictions of a particular graded-JPEG darkness threshold.
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

import book_c_t1_candidates as C


WIDTH, HEIGHT, PPC = 1920, 804, 110
# Owner's delivered current-words JPEG measurements, supplied in PROMPT_r02.md.
# Clipped row 803 from ~C515 is censored evidence, not a full glyph measurement.
REFERENCE_ROWS = {470: 788, 500: 799, 510: 802}


def surface_points(book, u, v):
    """Use the same arc map and cockled height field as book.trace_kernel.

    page_to_world alone omits cockle (book.py); the actual ray hits height().
    This only allocates point arrays and Book's small arc/cockle tables.
    """
    points = book.page_to_world('R', u, v)
    points[:, 2] = [C.BC.B.height(float(x), float(y), book.params, book.ck)[0]
                    for x, y in points[:, :2]]
    return points


class InscriptionGeometry:
    def __init__(self):
        # Riffle is finished by C400: shot_mountain's thicknesses are .6/2.8.
        self.book = C.BC.B.Book(TL=.6, TR=2.8, seed=3)
        self.line = C.make_line('current-words', PPC)
        line = self.line
        h, w = line.a.shape
        # InkLine.apply's only expansion is this local Gaussian wet bleed.
        # Both write-on and erosion only remove support. Include every nonzero
        # source texel's four cell corners, not a rectangle with empty corners.
        support = C.cv2.GaussianBlur(line.a, (0, 0), .9) > 0
        cells = np.zeros((h+1, w+1), bool)
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            cells[dy:dy+h, dx:dx+w] |= support
        rows, cols = np.nonzero(cells)
        self.bounds = surface_points(self.book, (line.c0+cols)/PPC,
                                     (line.r0+rows)/PPC)
        rows, cols = np.indices(line.a.shape)
        self.centres = surface_points(self.book, (line.c0+cols.ravel()+.5)/PPC,
                                      (line.r0+rows.ravel()+.5)/PPC)
        self.glyphs = self.centres[line.a.ravel() > .01]

    @staticmethod
    def screen(camera, points):
        # trace_kernel samples (j+.5, i+.5); report pixel-index coordinates.
        return camera.project(points)[0] - .5

    def bottom(self, camera):
        return float(self.screen(camera, self.bounds)[:, 1].max())

    def active_points(self, frame):
        # Only the small glyph crop, never the 29 x 20 cm page channels.
        local = copy.copy(self.line)
        local.r0 = local.c0 = 0
        channels = np.zeros((*local.a.shape, C.BC.B.NCH), np.float32)
        local.apply(channels, frame)
        return self.centres[channels[..., 0].ravel() > 0]


def speed_profile(renderer, book, t, dt=1e-4):
    """Central-difference speed: cm/s for position/target, deg/s for sightline."""
    a = renderer.cam_mountain(book, t-dt)
    b = renderer.cam_mountain(book, t+dt)
    angle = np.arctan2(np.linalg.norm(np.cross(a.f, b.f)), np.dot(a.f, b.f))
    return dict(position_speed=float(np.linalg.norm(b.pos-a.pos)/(2*dt)),
                target_speed=float(np.linalg.norm(b.target-a.target)/(2*dt)),
                angular_speed=float(np.degrees(angle)/(2*dt)),
                hfov_speed=float((b.hfov-a.hfov)/(2*dt)))


def audit():
    geometry = InscriptionGeometry()
    original = C.make_renderer('current-words', scale=1)
    held = C.make_renderer('current-words-held', scale=1)
    rows = []
    for frame in range(400, 560):
        t = (frame-320)/24.0
        row = dict(frame=frame)
        active = geometry.active_points(frame)
        for name, renderer in (('original', original), ('held', held)):
            camera = renderer.cam_mountain(geometry.book, t)
            points = geometry.screen(camera, geometry.bounds)
            row[name+'_bound_bottom'] = float(points[:, 1].max())
            row[name+'_glyph_bottom'] = float(geometry.screen(camera, geometry.glyphs)[:, 1].max())
            row[name+'_active_ink_bottom'] = (float(geometry.screen(camera, active)[:, 1].max())
                                             if len(active) else None)
            row[name+'_left'], row[name+'_top'] = map(float, points.min(0))
            row[name+'_right'] = float(points[:, 0].max())
            row[name+'_margin'] = HEIGHT-1-row[name+'_bound_bottom']
            row.update({name+'_'+key: value for key, value in speed_profile(renderer, geometry.book, t).items()})
        rows.append(row)
    by_frame = {r['frame']: r for r in rows}
    residuals = {f: by_frame[f]['original_glyph_bottom']-observed for f, observed in REFERENCE_ROWS.items()}
    summary = dict(
        glyph_shape=list(geometry.line.a.shape),
        source_origin=[geometry.line.c0, geometry.line.r0],
        bound_points=len(geometry.bounds),
        original_residual_px=residuals,
        max_absolute_residual_px=max(map(abs, residuals.values())),
        minimum_margin_through_527=min(r['held_margin'] for r in rows if r['frame'] <= 527),
        minimum_margin_through_539=min(r['held_margin'] for r in rows if r['frame'] <= 539),
        peak_tilt_degrees=float(np.degrees(C.HELD_MAX_TILT)),
        speed_difference_max=max(abs(r['original_position_speed']-r['held_position_speed']) for r in rows),
    )
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    args = parser.parse_args()
    # Reports belong outside the delivered render tree, including its symlink.
    renders = Path(__file__).resolve().parents[2] / 'renders'
    if args.csv.resolve().is_relative_to(renders.resolve()):
        parser.error('projection reports must not be written under renders/')
    C.cv2.setNumThreads(0)
    rows, summary = audit()
    with args.csv.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
