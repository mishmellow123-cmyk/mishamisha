"""Opt-in filmed Deep composites. Build independent coefficients with tools/rebake_cutd_deep.py.

The sweep retains its original footage clock and eight-frame linear-light soft entrance. Its coefficients
allow the held race to change without subtracting or reconstructing the baked closed-Ring C picture.
"""
from pathlib import Path
import cv2
import numpy as np


def source():
    return Path(__file__).read_text()


def to_lin(x):
    return np.where(x <= .04045,x/12.92,((x+.055)/1.055)**2.4).astype(np.float32)


def to_srgb(x):
    x=np.maximum(x,0.)
    return np.clip(np.where(x <= .0031308,x*12.92,1.055*x**(1/2.4)-.055),0.,1.).astype(np.float32)


def coeff_path(directory,frame):
    return str(Path(directory)/f'f_{frame:05d}.npz')


def sweep(race,path,frame):
    with np.load(path,allow_pickle=False) as data:
        gain=data['gain'].astype(np.float32)
        base=data['base'].astype(np.float32)
    if gain.shape[:2] != race.shape[:2]:
        size=(race.shape[1],race.shape[0])
        gain=cv2.resize(gain,size,interpolation=cv2.INTER_AREA)
        base=cv2.resize(base,size,interpolation=cv2.INTER_AREA)
    u=float(np.clip((frame-2720)/8.,0.,1.))
    weight=u*u*(3.-2.*u)
    if weight == 0.:
        return race.copy()
    mixed=to_lin(race)*gain+base
    if weight < 1.:
        # ftburn writes a clipped display image before the accepted soft-entry helper blends it.
        mixed=np.clip(mixed,0.,1.)
        mixed=to_lin(race)*(1.-weight)+mixed*weight
    return to_srgb(mixed)


def read_layer(path,shape,gray=False):
    image=cv2.imread(path,cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode Deep layer: {path}')
    if image.shape[:2] != shape:
        image=cv2.resize(image,(shape[1],shape[0]),interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)/255. if gray else image[...,::-1].astype(np.float32)/255.


def tail_weight(frame,tail_clear=None):
    if tail_clear is None:
        return 1.
    start,end=tail_clear
    if frame is None or end <= start:
        raise ValueError('Deep tail_clear requires an increasing frame range and the current frame')
    u=float(np.clip((frame-start)/(end-start),0.,1.))
    return 1.-u*u*(3.-2.*u)


def reveal(incoming,rgb_path,cover_path,*,frame=None,tail_clear=None):
    """Original ftburn display premultiplied page+fire, then incoming through its scalar cover."""
    weight=tail_weight(frame,tail_clear)
    if weight == 0.:
        return incoming.copy()
    rgb=read_layer(rgb_path,incoming.shape[:2])
    cover=read_layer(cover_path,incoming.shape[:2],gray=True)
    if weight != 1.:
        rgb=rgb*weight
        cover=cover*weight
    return np.clip(rgb+(1.-cover[...,None])*incoming,0.,1.)
