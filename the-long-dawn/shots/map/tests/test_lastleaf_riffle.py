"""Opening study geometry and write-boundary checks; no image rendering."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import lastleaf_riffle as R


def test_default_turn_keeps_ring_head_inside_the_existing_c2_camera():
    _, book, camera, _, time, _ = R.setup()
    assert time == (150 - 80) / 24.
    x, z = R.B.leaf_curve(book, R.PHASE)
    # The back's u runs oppositely to the front (book.trace_kernel). At this
    # phase the Ring's region has landed while the outer page edge is raised.
    coordinates = []
    for u, v in ((6.9, 5.), (13.5, 5.), (6.9, 10.), (13.5, 10.)):
        s = (book.PW - u) / book.PW * (len(x) - 1)
        coordinates.append([np.interp(s, np.arange(len(x)), x),
                            book.PH / 2 - v, np.interp(s, np.arange(len(z)), z)])
    pixels, depth = camera.project(np.array(coordinates))
    assert (depth > 0).all()
    assert (pixels[:, 0] > 0).all() and (pixels[:, 0] < camera.W).all()
    assert (pixels[:, 1] > 0).all() and (pixels[:, 1] < camera.H).all()
    landed_x, landed_z = R.B.leaf_curve(book, 1.)
    assert z[-1] > landed_z[-1] + .1
    assert x[-1] > landed_x[-1]


@pytest.mark.parametrize('frame,phase,scale', [
    (79, .8, .5), (320, .8, .5), (150, 0., .5), (150, 1., .5),
    (150, float('nan'), .5), (150, .8, 0.),
])
def test_invalid_and_non_turn_controls_are_rejected(frame, phase, scale):
    with pytest.raises(ValueError):
        R.validate(frame, phase, scale)


def test_shared_render_directory_is_rejected_even_through_an_alias(tmp_path):
    shared = R.ROOT / 'renders'
    with pytest.raises(ValueError):
        R.output_path(shared / 'opening.png')
    alias = tmp_path / 'alias'
    alias.symlink_to(shared)
    with pytest.raises(ValueError):
        R.output_path(alias / 'opening.png')
    assert R.output_path(tmp_path / 'opening.png') == tmp_path / 'opening.png'
