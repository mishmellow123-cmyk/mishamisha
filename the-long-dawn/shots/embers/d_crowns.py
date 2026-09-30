"""D4240–4559: industrial crowns catch two broad basket fires at D4320.

Opt in by calling render_frame; accepted render entrypoints are unchanged.
"""
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'lib')]

import cv2
import numpy as np
import c3
import cflame
import d_forges as F
import look
import openring_d as D
import ringsolid as RS
from core import Camera

START, KINDLE, END = 4240, 4320, 4560
RING_SIZE = 2.2
BASKET_WIDTH = 3.4
FLAME_HEIGHT = 1.25*BASKET_WIDTH
# Ridge summits are fixed; the crown staging follows these measured roots.
RIDGE_ROOTS = np.array([[733.4482365330008, 463.0645838537898],
                        [1386.8711906600877, 455.13424081989217]])
_SCENE = None


def beacon_level(f):
    if f < KINDLE:
        return 0.
    # The first frame is a spark, followed by a lick and a sixteen-frame catch.
    return .012 + .988*float(F.V5.ease(KINDLE, KINDLE+16, f))


def camera(f=None):
    az = F.AZIMUTH
    forward = np.array([math.cos(az), 0., math.sin(az)])
    right = np.array([math.sin(az), 0., -math.cos(az)])
    pos = 105.*forward + [0., 44., 0.]
    focal = 960./math.tan(math.radians(49.)/2.)
    target = -(1060.-959.5)*105./focal*right + [0., 44., 0.]
    return Camera(pos, target, hfov=49., focus=np.linalg.norm(F.RING_C-pos), aperture=.012)


def ring_pose():
    rot, _, _ = c3._closed_ring_frame(1580.)
    rot = D.placement(rot, camera().pos, F.RING_C, 4079.)
    return rot, F.RING_C.copy(), RING_SIZE


def theta_range():
    return D.theta_range(4079.)


def place_crowns(towers):
    """Translate the two unchanged industrial meshes onto the ridge roots.

    The shared trap state has a 0.14 inward crown bend. Invert that displacement
    to choose the base position; shape, height, facade, and rotation are kept.
    """
    cam = camera()
    for i, pixel in zip(F.LEADERS, RIDGE_ROOTS):
        local = np.array([(pixel[0]-959.5)/cam.f_px(1920),
                          -(pixel[1]-401.5)/cam.f_px(1920), 1.])
        ray = local @ cam.R
        y = F.B.GROUND + towers.height(i, KINDLE) + .65
        point = cam.pos + ray*((y-cam.pos[1])/ray[1])
        radius = np.linalg.norm(point[[0,2]])
        towers.rad[i] = radius + .14*towers.height(i, KINDLE)
        towers.ang[i] = math.atan2(point[2], point[0])


def beacon_points(towers, f=KINDLE):
    return np.array([towers.top(i, f) + np.array([0., .65, 0.]) for i in F.LEADERS])


def beacon_screen_positions(towers):
    """Native pixel anchors are flame roots, at the centres of the basket rims."""
    u, v, _ = camera().project(beacon_points(towers), 1920, 804)
    return np.stack([u, v], axis=1)


def _basket_mask(shape, root, width, scale):
    mask = np.zeros(shape, np.float32)
    x,y = np.rint(root*scale).astype(int)
    half = max(3,int(round(width*scale*.5)))
    depth = max(3,int(round(width*scale*.22)))
    stroke = max(1,int(round(2.*scale)))
    cv2.ellipse(mask,(x,y),(half,max(2,int(half*.24))),0.,0.,360.,1.,stroke,cv2.LINE_AA)
    for side in (-1.,-.66,-.33,0.,.33,.66,1.):
        cv2.line(mask,(int(x+side*half),y),(int(x+side*half*.68),y+depth),1.,stroke,cv2.LINE_AA)
    cv2.line(mask,(int(x-half*.68),y+depth),(int(x+half*.68),y+depth),1.,stroke,cv2.LINE_AA)
    for side in (-1.,1.):
        cv2.line(mask,(int(x+side*half*.4),y+depth),(int(x+side*half*.6),y+int(depth*1.6)),1.,stroke,cv2.LINE_AA)
    return mask


def _air(hdr, root, width, height, t, level, scale, seed):
    """Orange haze, right-leaning lit smoke, and embers born after the catch."""
    if level <= 0.:
        return
    H,W=hdr.shape[:2]
    x0,y0=root*scale
    width,height=width*scale,height*scale
    yy,xx=np.mgrid[:H,:W].astype(np.float32)
    halo=np.exp(-(((xx-x0-.12*width)/(.8*width))**2+((yy-y0+.4*height)/(.85*height))**2)*2.)
    hdr += (halo*.04*level)[...,None]*F.ORANGE
    age=t-KINDLE
    for k in range(22):
        life=46.+(k*17%31)
        a=(t+seed*11+k*13)%life
        if a>age:
            continue
        u=a/life
        x=x0+width*(.15+.75*u**1.4+.13*math.sin(k*3.1+t*.021))
        y=y0-height*(.7+1.5*u)
        radius=width*(.11+.20*u)
        puff=np.exp(-(((xx-x)/radius)**2+((yy-y)/(radius*.68))**2)*1.8)
        hdr += (puff*.028*level*(1.-u))[...,None]*np.array([1.,.32,.09])
    for k in range(19):
        life=22.+(k*13%29)
        a=(t+seed*7+k*19)%life
        if a>age:
            continue
        u=a/life
        x=x0+width*(-.40+(k*.618%1.)*.8+.7*u*u)
        y=y0-height*(.28+1.8*u)
        value=.25*level*(1.-u)**.7
        cv2.line(hdr,(int(x),int(y)),(int(x-width*.02),int(y+height*.035)),
                 (value,value*.27,value*.022),max(1,int(round(scale))),cv2.LINE_AA)


def tongue_geometry(root, width, height, f, seed):
    """Five separate flame axes spread across the entire basket."""
    growth = beacon_level(f)
    spread = .15+.85*growth
    out=[]
    for k,offset in enumerate((-.42,-.23,.015,.27,.41)):
        motion=math.sin((.043+.012*k)*f+1.9*k+seed)**2
        reach=(.91+.08*motion) if k==2 else (.54+.45*motion)
        h=height*reach*(.04+.96*growth)
        lateral=.025*math.sin((.035+.008*k)*f+2.7*k+seed)
        r=root+np.array([(offset+lateral)*width*spread,-width*.025])
        tip=r+np.array([width*(.12+.035*math.sin(.11*f+2.3*k+seed))*growth,-h])
        out.append((r,tip))
    return out


def draw_beacons(hdr, ctx, towers, level):
    roots=beacon_points(towers,ctx.t)
    u,v,z=ctx.cam.project(roots,1920,804)
    pixels=np.stack([u,v],1)
    for j,root in enumerate(pixels):
        width=BASKET_WIDTH*ctx.cam.f_px(1920)/z[j]
        height=FLAME_HEIGHT*ctx.cam.f_px(1920)/z[j]
        _air(hdr,root,width,height,ctx.t,level,ctx.scale,7.3+9.*j)
        if level>0.:
            flame=np.zeros_like(hdr)
            visible=c3.occ_vis(ctx.fr,float(z[j]-.9),ctx.fr.H,ctx.fr.W)
            if ctx.t==KINDLE:
                x,y=np.rint(root*ctx.scale).astype(int)
                cv2.circle(flame,(x,y),max(1,int(round(2*ctx.scale))),(2.,.3,.01),-1,cv2.LINE_AA)
            else:
                for k,(r,tip) in enumerate(tongue_geometry(root,width,height,ctx.t,7.3+9.*j)):
                    cflame.draw(flame,r,tip,ctx.t+31.*j+13.*k,
                                bright=.72,calm=.22,vis=visible,glow=.005,
                                seed=7.3+9.*j+3.1*k,scale=ctx.scale)
            hdr += np.minimum(flame[...,:1],3.5)*F.ORANGE
        mask=_basket_mask(hdr.shape[:2],root,width,ctx.scale)
        iron=np.array([.020,.017,.014])+level*np.array([.28,.050,.006])
        hdr[:] = hdr*(1.-mask[...,None]) + iron*mask[...,None]


class Scene:
    def __init__(self):
        self.world = F.Scene(phase='trap')
        place_crowns(self.world.towers)

    def frame(self, f, scale=1.):
        if not START <= f < END:
            raise ValueError(f'D crown frame {f} outside [{START}, {END})')
        if not 0. < scale <= 1.:
            raise ValueError('scale must be in (0, 1]')
        ctx = F.make_context(f, scale, camera)
        level = beacon_level(f)
        pose = ring_pose()
        arc = theta_range()
        with self.world.world(beacon_power=level):
            self.world.prepare(ctx)
            state = RS.RingState()
            from d_vision import inscription
            state.end_caps = state.worked_caps = True
            state.inscription = inscription()
            state.letters, state.glow, state.hammer = .58, .065, .18
            state.write = D.write(4079.)
            def heat(theta):
                theta = np.asarray(theta)
                distance = np.minimum(np.abs((theta-arc[0]+np.pi)%(2*np.pi)-np.pi),
                                      np.abs((theta-arc[1]+np.pi)%(2*np.pi)-np.pi))
                return (.73+.20*math.exp(-(f%10.)/3.))*np.exp(-(distance/math.radians(7.))**2)
            state.heat = heat
            env = RS.Env(above=(.09,.075,.046), horizon=(.29,.16,.07), below=(.06,.018,.005))
            env.lobe([-.7,.6,.4], [.58,.47,.25], 5.)
            for i in F.LEADERS:
                env.point(self.world.towers.top(i, f), [5., 2.7, .65], 2.)
            rgb, alpha, depth = RS.render(ctx.cam, ctx.fr.W, ctx.fr.H, *pose, state, env, th_range=arc)
            before = RS.merge_occluder(ctx.fr, alpha, depth)
            ring = rgb * RS.visibility(before, depth, ctx.fr.H, ctx.fr.W)[..., None]
            self.world.emit(ctx)
            p, _ = RS.local_points(np.array(arc), np.zeros(2))
            self.world.hammer_sparks(ctx, pose[1]+pose[2]*p@pose[0].T)
            self.world.gold(ctx, pose, arc)
            hdr = ctx.fr.resolve() + ring
            draw_beacons(hdr, ctx, self.world.towers, level)
        if not np.isfinite(hdr).all():
            raise ValueError(f'Non-finite crown HDR at D{f}')
        return look.finish(hdr, exposure=1., bloom_strength=.10, bloom_threshold=1.1,
                           streak_strength=0., vignette_amount=.24).astype(np.float32)


def render_frame(f, scale=1., outdir=None, save=False, verbose=False):
    global _SCENE
    if not START <= f < END:
        raise ValueError(f'D crown frame {f} outside [{START}, {END})')
    if save and outdir is None:
        raise ValueError('Saving a D crown frame requires an explicit output directory')
    if save:
        from render_d import output_path
        outdir = output_path('crowns', outdir)
    path = Path(outdir) / f'f_{int(f):05d}.png' if save else None
    if path is not None and path.exists():
        raise FileExistsError(path)
    if _SCENE is None:
        _SCENE = Scene()
    rgb = _SCENE.frame(f, scale)
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        look.save_png(str(path), rgb)
    if verbose:
        print(f'D crowns {int(f)}: {rgb.shape[1]}x{rgb.shape[0]}', flush=True)
    return rgb
