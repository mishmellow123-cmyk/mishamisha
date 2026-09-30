"""Camera geometry/default routing; visual acceptance needs the rendered40-frame study."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import book_c_v5_deep_lamp_match as M
import book_c_v5_deep_candidates as C
import book_c as BC

# A5180 lantern_v3 body bounds, cap/base/vent included and bail excluded, projected from the pinned renderer.
MATCH=M.LampMatch((1213.6865913838851,458.9564169630003),(39.01222460814097,86.81091314954449))


def book():
    return BC.Book3().book(1.6,2.2,seed=9)


def test_default_routes_to_unchanged_adopted_renderer():
    with patch.object(C,'make_renderer',return_value=object()) as original:
        result=M.make_renderer()
        assert result is original.return_value
        original.assert_called_once_with('leaned_ladders',scale=.5,ppc=110)
    with patch.object(C.CandidatePagesV5,'shot_deep_abandoned',return_value=('hdr','alpha')) as parent:
        renderer=object.__new__(M.LampMatchPages);renderer.match=MATCH
        for frame in (4240,4439,4440):
            assert renderer.shot_deep_abandoned((frame-4240)/24,frame)==('hdr','alpha')
        assert parent.call_count==3
        # A later frame cannot quietly take the original-camera branch.
        with patch.object(renderer,'book',side_effect=RuntimeError('camera path reached')):
            with pytest.raises(RuntimeError,match='camera path reached'):
                renderer.shot_deep_abandoned(201/24,4441)


def assert_target(cam,points):
    low,high=M.body_bounds(cam,points)
    np.testing.assert_allclose((low+high)/2,MATCH.target_center,atol=1e-6,rtol=0)
    np.testing.assert_allclose(np.prod(high-low),np.prod(MATCH.target_body_size),atol=1e-5,rtol=0)


def test_real_camera_hits_centre_and_area_without_a_picture_transform():
    bk=book();movement=M.solve_camera_movement(bk,MATCH);base=M.base_camera(bk);points=M.lamp_body(bk)
    end=M.camera_at(bk,4476,MATCH,movement=movement)
    assert_target(end,points)
    with pytest.raises(AssertionError):
        assert_target(base,points)
    assert not np.array_equal(end.pos,base.pos)
    np.testing.assert_allclose(end.pos-base.pos,end.target-base.target,atol=1e-13)
    for attr in ('r','u','f'):
        np.testing.assert_allclose(getattr(end,attr),getattr(base,attr),atol=1e-13)
    assert end.hfov==base.hfov==38.
    # Matching area does not pretend the different source silhouettes have equal width and height.
    low,high=M.body_bounds(end,points)
    assert not np.allclose(high-low,MATCH.target_body_size,rtol=.1)


def test_start_is_exact_and_camera_settles_before_the_dissolve():
    bk=book();points=M.lamp_body(bk);movement=M.solve_camera_movement(bk,MATCH)
    base=M.base_camera(bk);first=M.camera_at(bk,4440,MATCH,movement=movement)
    np.testing.assert_array_equal(base.pos,first.pos)
    np.testing.assert_array_equal(base.target,first.target)
    screen0=first.project(points)[0]
    screen1=M.camera_at(bk,4441,MATCH,movement=movement).project(points)[0]
    assert float(np.max(np.linalg.norm(screen1-screen0,axis=1)))<1.
    def assert_settled(frame):
        np.testing.assert_array_equal(M.camera_at(bk,frame,MATCH,movement=movement).pos,
                                      M.camera_at(bk,4476,MATCH,movement=movement).pos)
    for frame in range(4476,4480):assert_settled(frame)
    with pytest.raises(AssertionError):assert_settled(4470)


def test_body_bounds_rejects_behind_camera_despite_project_depth_clamp():
    cam=M.base_camera(book())
    visible=np.array([cam.pos+cam.f*3+cam.r,cam.pos+cam.f*4+cam.u])
    low,high=M.body_bounds(cam,visible)
    assert np.isfinite([low,high]).all()
    behind=np.array([cam.pos-cam.f*3+cam.r,cam.pos-cam.f*4+cam.u])
    # The projection helper itself reports clamped positive depths for this invalid input.
    assert (cam.project(behind)[1]>0).all()
    with pytest.raises(ValueError,match='in front of the camera'):
        M.body_bounds(cam,behind)


@pytest.mark.parametrize('center,size,start,end',[
    ((1.,2.),(0.,5.),4440,4476),((np.nan,2.),(4.,5.),4440,4476),
    ((1.,2.,3.),(4.,),4440,4476),((1.,),(2.,3.,4.),4440,4476),
    ((1.,2.),(4.,5.),4476,4440),((1.,2.),(4.,5.),4440,4480)])
def test_invalid_match_rejected(center,size,start,end):
    with pytest.raises(ValueError):M.LampMatch(center,size,start,end)
