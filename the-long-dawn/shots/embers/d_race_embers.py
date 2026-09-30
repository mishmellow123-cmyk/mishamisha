"""C's continuous forge-surface embers, kept clear of D's moving cut faces."""
from copy import copy

import numpy as np

import scene_b as B
import c_eye as EYE


class ForgeEmbers(B.TowerEmbers):
    def __init__(self, towers):
        super().__init__(EYE._AllTowers(towers), per=2000)

    def emit_bounded(self, ctx, ends0, ends1, gain):
        if gain <= 0.:
            return
        target = ctx.fr

        class Sink:
            def splat(self, p0, p1, radius, energy, *args, **kwargs):
                # Surface embers supply the city's moving texture. Only the
                # scored giant streams may reach the band: these grains stay
                # outside a sphere enclosing each physical cut face.
                relative = p0[:, None] - ends0[None]
                velocity = (p1 - p0)[:, None] - (ends1 - ends0)[None]
                closest = np.clip(-(relative * velocity).sum(2) /
                                  np.maximum((velocity * velocity).sum(2), 1e-12), 0., 1.)
                distance = np.linalg.norm(relative + closest[..., None] * velocity, axis=2)
                keep = distance.min(1) > 3.
                target.splat(p0, p1, radius, np.asarray(energy) * gain * keep,
                             *args, **kwargs)

        proxy = copy(ctx)
        proxy.fr = Sink()
        super().emit(proxy)
