"""Cut D's study lamps: local surface irradiance from the visible lamp positions.

Powers are linear-light intensity at one world unit. A finite source radius
softens the inverse-square singularity; the surface normal rejects light from
behind the metal. This module changes no accepted C emitter or environment.
"""
import numpy as np


GOLD_WHITE = np.array([1., .91, .67])
SOURCE_RADIUS = .38


class ReadingLights:
    """Callable RingState.reading_light; colors, powers and radii may vary per lamp."""

    def __init__(self, positions, powers=1., color=GOLD_WHITE, radius=SOURCE_RADIUS):
        self.positions = np.asarray(positions, np.float64).reshape(-1, 3).copy()
        n = len(self.positions)
        self.powers = np.broadcast_to(np.asarray(powers, np.float64), (n,)).copy()
        self.colors = np.broadcast_to(np.asarray(color, np.float64), (n, 3)).copy()
        self.radii = np.broadcast_to(np.asarray(radius, np.float64), (n,)).copy()
        values = (self.positions, self.powers, self.colors, self.radii)
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError('Reading lights require finite inputs')
        if np.any(self.powers < 0) or np.any(self.colors < 0) or np.any(self.radii <= 0):
            raise ValueError('Reading lights require nonnegative power/color and positive radius')

    def __call__(self, points, normals):
        points = np.asarray(points, np.float64)
        normals = np.asarray(normals, np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
            raise ValueError('Reading light samples require matching (n, 3) positions and normals')
        out = np.zeros_like(points)
        # One lamp at a time keeps the renderer's 150k-sample shading chunks bounded.
        for position, power, color, radius in zip(self.positions, self.powers, self.colors, self.radii):
            if power == 0.:
                continue
            delta = position[None, :] - points
            distance2 = np.sum(delta * delta, axis=1)
            direction = delta / np.maximum(np.sqrt(distance2), 1e-12)[:, None]
            incidence = np.maximum(np.sum(normals * direction, axis=1), 0.)
            weight = power * incidence / (distance2 + radius * radius)
            out += weight[:, None] * color[None, :]
        return out
