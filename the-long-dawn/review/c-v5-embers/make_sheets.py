"""Deterministic review sheets from existing PNGs; never invokes a renderer."""
import argparse
from pathlib import Path
import cv2
import numpy as np

WINDOWS = {'withdrawal': (2418,2441), 'others-surge': (2478,2501),
           'return': (2538,2561), 'rivals': (2592,2615),
           'shutdown': (3840,3863), 'cooling': (4100,4123)}
# Source-pixel crops; no enlargement or sharpening. Each 24-frame crop strip
# preserves native detail in the moving subject while the other sheet shows composition.
CROPS = {'withdrawal': (280,150,130,240), 'others-surge': (690,130,160,260),
         'return': (270,100,150,290), 'rivals': (400,65,170,275),
         'shutdown': (400,70,170,320), 'cooling': (400,75,170,220)}


def sheet(frames, source, size, crop=None):
    w,h=size
    out=np.full((4*(h+24),6*w,3),18,np.uint8)
    for j,f in enumerate(frames):
        im=cv2.imread(str(source/f'f_{f:05d}.png'))
        if im is None: raise FileNotFoundError(f)
        if crop:
            x,y,cw,ch=crop;im=im[y:y+ch,x:x+cw]
            assert im.shape[:2] == (h,w)
        else: im=cv2.resize(im,size,interpolation=cv2.INTER_AREA)
        row,col=divmod(j,6);y=row*(h+24);x=col*w
        out[y+24:y+24+h,x:x+w]=im
        cv2.putText(out,f'C{f}',(x+5,y+17),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('out',type=Path)
    a=ap.parse_args();cv2.setNumThreads(0);a.out.mkdir(parents=True,exist_ok=True)
    for name,(start,end) in WINDOWS.items():
        fs=range(start,end+1)
        cv2.imwrite(str(a.out/f'{name}-24.png'),sheet(fs,a.source,(240,100)))
        crop=CROPS[name]
        cv2.imwrite(str(a.out/f'{name}-native-24.png'),sheet(fs,a.source,crop[2:],crop))

if __name__=='__main__':main()
