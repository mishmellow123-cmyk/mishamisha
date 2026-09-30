"""D crowns timing, fixed projection and shared-world isolation; no heavy render."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'lib')]
import d_crowns as C
import d_forges as F


def bare_towers(phase='trap'):
    tw = F.Forges.__new__(F.Forges)
    tw.k_all = 18
    tw.schedule = F.Schedule(phase, 4079. if phase == 'trap' else 2959.)
    tw._frozen_height = np.full(18, 36.)
    tw._frozen_height[list(F.LEADERS)] = 55.
    tw.ang = np.linspace(0., 2*np.pi, 18, endpoint=False)
    tw.rad = np.full(18, 22.)
    tw.rot = np.linspace(0.,2*np.pi,18)
    return tw


def assert_catch(levels):
    assert levels[0] == 0.
    assert 0. < levels[1] < .03
    assert levels[1] < levels[2] < levels[3] < levels[4] == 1.


def test_both_baskets_spark_on_4320_then_take_over_sixteen_frames():
    assert_catch([C.beacon_level(f) for f in (4319,4320,4326,4332,4336)])
    with pytest.raises(AssertionError):
        assert_catch([.1,.1,.3,.6,1.])  # early light
    with pytest.raises(AssertionError):
        assert_catch([0.,1.,1.,1.,1.])  # the rejected pop
    tw=bare_towers(); C.place_crowns(tw)
    ctx=SimpleNamespace(cam=C.camera(),fr=SimpleNamespace(H=80,W=192),scale=.1)
    with patch.object(C,'_basket_mask',return_value=np.zeros((80,192))), patch.object(C.c3,'occ_vis',return_value=np.ones((80,192))):
        for f in (4319,4320):
            ctx.t=f
            hdr=np.zeros((80,192,3),np.float32)
            C.draw_beacons(hdr,ctx,tw,C.beacon_level(f))
            for x,y in C.RIDGE_ROOTS:
                region=hdr[int(y*.1)-3:int(y*.1)+4,int(x*.1)-3:int(x*.1)+4]
                # The dim air halo must not satisfy the ignition gate.
                assert bool(region.max()>.5) == (f==4320)


def test_camera_roots_and_ring_centre_match_fixed_ridge_for_every_frame():
    tw=bare_towers(); C.place_crowns(tw)
    reference=C.camera(C.START).params(1920,804)
    roots=C.beacon_points(tw,C.START)
    for f in range(C.START,C.END):
        np.testing.assert_array_equal(C.camera(f).params(1920,804),reference)
        np.testing.assert_array_equal(C.beacon_points(tw,f),roots)
    np.testing.assert_allclose(C.beacon_screen_positions(tw),C.RIDGE_ROOTS,atol=1e-9)
    u,_,_=C.camera().project(F.RING_C[None],1920,804)
    assert u[0] == pytest.approx(1060.)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(C.beacon_screen_positions(tw)+[7.,0.],C.RIDGE_ROOTS,atol=6.)


def test_shared_industrial_mesh_and_normal_transform_are_used_directly():
    assert F.Forges.world is F.CD.Forges.world
    assert F.Forges.nworld is F.CD.Forges.nworld
    assert F.Forges.ZF == F.CD.Forges.ZF
    assert F.LEADERS == F.CD.LEADERS == (4,5)
    tw=bare_towers()
    rotation=tw.rot.copy(); height=tw._frozen_height.copy()
    others=[i for i in range(18) if i not in F.LEADERS]
    base=np.array([tw.base(i) for i in others])
    C.place_crowns(tw)
    np.testing.assert_array_equal(tw.rot,rotation)
    np.testing.assert_array_equal(tw._frozen_height,height)
    np.testing.assert_array_equal(np.array([tw.base(i) for i in others]),base)


def test_broad_basket_proportions_and_independent_right_leaning_tongues():
    def broad(width,height):
        assert 1. <= height/width <= 1.5
    broad(C.BASKET_WIDTH,C.FLAME_HEIGHT)
    with pytest.raises(AssertionError):
        broad(C.BASKET_WIDTH,C.BASKET_WIDTH*3.)
    a=C.tongue_geometry(np.array([500.,450.]),100.,125.,4340.,7.3)
    b=C.tongue_geometry(np.array([500.,450.]),100.,125.,4341.,7.3)
    assert len(a)>=4
    roots=np.array([r for r,t in a]); tips=np.array([t for r,t in a])
    assert np.ptp(roots[:,0])>75.
    assert np.all(tips[:,0]>roots[:,0])
    delta=np.array([t for r,t in b])-tips
    assert np.linalg.matrix_rank(delta)>1
    with pytest.raises(AssertionError):
        assert np.linalg.matrix_rank(np.repeat(delta[:1],5,axis=0))>1


def test_shared_gap_stays_55_degrees_and_entire_band_fits():
    a,b=C.theta_range()
    assert 360.-np.degrees(b-a) == pytest.approx(55.)
    with pytest.raises(AssertionError):
        assert 360.-np.degrees(b-a) == pytest.approx(30.)
    theta,psi=np.meshgrid(np.linspace(a,b,121),np.linspace(0,2*np.pi,25))
    local,_=C.RS.local_points(theta.ravel(),psi.ravel())
    rot,centre,size=C.ring_pose()
    u,v,z=C.camera().project(centre+size*local@rot.T,1920,804)
    assert z.min()>0.
    assert u.min()>30. and u.max()<1890.
    assert v.min()>30. and v.max()<643.



def test_world_restores_legacy_state_after_success_nested_scope_and_failure():
    scene = F.Scene.__new__(F.Scene)
    scene.schedule = F.Schedule()
    before = F.variant.CUT, F.B.SCHED, F.B.IGN, F.B.BEATS, F.B._tower_cache
    with scene.world(lean={4:.2}, red=.7):
        assert F.B.SCHED is scene.schedule
        assert scene.schedule.amount(4) == .2
        with scene.world(lean=.1, red=.4):
            assert scene.schedule.amount(4) == .1
        assert scene.schedule.amount(4) == .2
    with pytest.raises(RuntimeError):
        with scene.world():
            raise RuntimeError('forced render failure')
    after = F.variant.CUT, F.B.SCHED, F.B.IGN, F.B.BEATS, F.B._tower_cache
    assert all(x is y if index in (1,3,4) else x == y for index,(x,y) in enumerate(zip(before,after)))
    broken = (*after[:1], object(), *after[2:])
    with pytest.raises(AssertionError):
        assert all(x is y if index in (1,3,4) else x == y for index,(x,y) in enumerate(zip(before,broken)))


def test_individual_lean_moves_leader_top_and_leaves_other_towers_upright():
    tw = bare_towers('race')
    upright = np.array([tw.top(i,2400) for i in range(18)])
    tw.schedule.lean_amount = {F.LEADERS[0]:.2}
    bent = np.array([tw.top(i,2400) for i in range(18)])
    changed = np.any(upright != bent, axis=1)
    assert np.flatnonzero(changed).tolist() == [F.LEADERS[0]]
    assert np.linalg.norm(bent[F.LEADERS[0],[0,2]]) < np.linalg.norm(upright[F.LEADERS[0],[0,2]])
    with pytest.raises(AssertionError):
        assert np.array_equal(upright,bent)


def test_smoke_source_tracks_bent_crown_once_without_moving_geometry_base():
    tw=bare_towers()
    tw.schedule.lean_amount={F.LEADERS[0]:.325}
    sources=F.SmokeSources(tw)
    sources.t=2490.
    for i in range(tw.k_all):
        actual=sources.base(i)+[0.,tw.height(i,sources.t),0.]
        np.testing.assert_allclose(actual,tw.top(i,sources.t),atol=1e-12)
    i=F.LEADERS[0]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(tw.base(i)+[0.,tw.height(i,sources.t),0.],tw.top(i,sources.t))


def test_own_crown_light_and_reciprocal_light_have_real_surface_response():
    tw=bare_towers(); C.place_crowns(tw)
    i,other=F.LEADERS
    # Below the own-crown falloff, so only the other beacon can pass this gate.
    point=tw.top(i,4340)-[0.,20.,0.]
    source=tw.top(other,4340)+[0.,2.,0.]
    toward=(source-point)/np.linalg.norm(source-point)
    stone={'P':np.stack([point,point]),'N':np.stack([toward,-toward])}
    tw.cur=[None]*18; tw.cur[i]={'parts':{0:stone}}
    tw.schedule.beacon_power=1.
    ctx=SimpleNamespace(t=4340,cam=SimpleNamespace(pos=source))
    captured=[]
    with patch.object(F.CD.Forges,'_splat',lambda *a:captured.append(a[4])):
        tw._splat(ctx,i,stone,np.zeros((2,3)),None,None,None,None,None)
    assert captured[0][0,0]>0.
    assert np.all(captured[0][1]==0.)
    np.testing.assert_allclose(captured[0][0]/captured[0][0,0],F.ORANGE)
    frames=np.arange(4340.,4350.)
    flux=np.array([[tw.schedule.beacon_flux(j,t) for j in F.LEADERS] for t in frames])
    assert np.all(np.ptp(flux,axis=0)>.01)
    assert not np.allclose(flux[:,0],flux[:,1])
    with pytest.raises(AssertionError):
        assert np.all(np.ptp(np.ones_like(flux),axis=0)>.01)
    # A horizontal crenellation must receive its own light from above, even
    # when the other beacon is placed far beyond useful reciprocal range.
    realtop=tw.top
    def isolated_top(j,t):
        return realtop(j,t)+(np.array([100000.,0.,0.]) if j==other else 0.)
    crown=realtop(i,4340)+[.5,-.2,0.]
    stone={'P':np.stack([crown,crown]),'N':np.array([[0.,1.,0.],[0.,-1.,0.]])}
    tw.cur[i]['parts'][0]=stone; captured.clear()
    ctx.cam=SimpleNamespace(pos=crown+[0.,10.,1.])
    with patch.object(tw,'top',isolated_top), patch.object(F.CD.Forges,'_splat',lambda *a:captured.append(a[4])):
        tw._splat(ctx,i,stone,np.zeros((2,3)),None,None,None,None,None)
    assert captured[0][0,0]>.01
    with pytest.raises(AssertionError):
        assert captured[0][1,0]>.01



def test_sparks_remain_active_before_and_after_kindle_at_both_ends():
    scene = F.Scene.__new__(F.Scene)
    scene.towers = bare_towers()
    scene.spark_velocity = np.zeros((90,3))
    scene.spark_energy = np.ones(90)
    calls=[]
    ctx = SimpleNamespace(t=4300,cam=C.camera(),cam0=C.camera(),cam1=C.camera(),
                          fr=SimpleNamespace(splat=lambda *a,**k:calls.append(a)))
    ends = np.array([[-3.,40.,0.],[3.,40.,0.]])
    for f,k in ((4300,0),(4310,1),(4320,0),(4330,1),(4540,0),(4550,1)):
        calls.clear()
        ctx.t=f
        scene.hammer_sparks(ctx,ends)
        assert len(calls) == 3
        to_camera=ctx.cam.pos-ends[k]
        expected=ends[k]+1.25*to_camera/np.linalg.norm(to_camera)
        np.testing.assert_allclose(calls[0][0][0],expected,atol=1e-12)
        assert np.all(calls[0][3]>0.)
        # Reintroducing the cap-buried centreline fails the visibility guard.
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(calls[0][0][0],ends[k])


def test_geometry_cache_fallback_never_invokes_legacy_writer(tmp_path):
    source=tmp_path/'geometry.py'
    source.write_text('# deterministic fixture')
    module=SimpleNamespace(__file__=str(source),build_all=lambda *a,**k:['forges'],build_skyline=lambda:['skyline'])
    assert F._geometry_readonly(module,(),False) == (['forges'],['skyline'])
    assert sorted(p.name for p in tmp_path.iterdir()) == ['geometry.py']


def test_gold_receiver_is_an_existing_vertical_inward_window():
    tw=bare_towers('race')
    tw.TW2=SimpleNamespace(HMAX=80.)
    tw.rot=np.zeros(18)
    tw.lean=0.
    i=F.LEADERS[0]
    inward=-tw.base(i)*[1.,0.,1.]
    inward/=np.linalg.norm(inward)
    points=np.array([[-1.,76.,0.],[1.,76.,0.],[0.,79.,0.]],np.float32)
    normals=np.stack([inward,-inward,[0.,1.,0.]])
    tw.G=[None]*18
    tw.G[i]={3:{'p':points,'n':normals}}
    target=tw.receiving_window(i,4320)
    expected=tw.world(i,points[:1],tw.height(i,4320))[0]
    np.testing.assert_allclose(target,expected)
    tw.G[i][3]['n']=np.tile([0.,1.,0.],(3,1))
    with pytest.raises(ValueError,match='no inward-facing'):
        tw.receiving_window(i,4320)


def test_invalid_crown_frame_fails_before_renderer_work():
    scene=C.Scene.__new__(C.Scene)
    for frame in (4239,4560):
        with pytest.raises(ValueError): scene.frame(frame)
