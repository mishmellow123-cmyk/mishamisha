"""One optional close camera for the forming inscription and solid cut section.

Use with C3's existing scene and absolute frame clock. The camera follows the
leading end in material coordinates; it does not alter the ring's placement.
"""
import numpy as np

import c3
from core import Camera

EXPOSURE = .58  # retain inscription detail beside the white-hot edge in this close study


def camera(t):
    if not c3.OR.enabled():
        raise ValueError('The end study requires LD_OPEN_RING=1')
    rot, centre, size = c3.ring_frame(t)
    theta = c3.OR.theta_range(t)[1]
    radial = np.array([np.cos(theta), 0., np.sin(theta)])
    tangent = np.array([-np.sin(theta), 0., np.cos(theta)])
    axis = np.array([0., 1., 0.])
    end = c3.RS.R_MID * radial
    target = centre + size * (end - .42 * tangent) @ rot.T
    pos = centre + size * (end + 1.5 * tangent + 3.5 * radial + .65 * axis) @ rot.T
    return Camera(pos, target, hfov=38., focus=np.linalg.norm(target - pos), aperture=.01)
