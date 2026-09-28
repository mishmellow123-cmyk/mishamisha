"""A resting dip pen, in the book's centimetre world; no changes to book.py.

Triangle projection uses camera-forward depth, as book.trace_kernel does. The
pen is composited into HDR before the shared depth of field and film finish.
"""
import cv2
import numpy as np
from scipy.optimize import minimize_scalar
import book as B


def geometry(bk):
    a=np.array([-8.0,3.2,0.]); b=np.array([10.1,-2.3,0.])
    # Wood handle, narrow brass ferrule, then a flattened split nib. A slight
    # shoulder catches the broad hearth light; the shaft is never a flat decal.
    sections=[(0,.055,.055,0),(.025,.14,.14,0),(.10,.225,.225,0),
              (.63,.22,.22,0),(.78,.17,.17,0),(.80,.19,.19,1),
              (.855,.19,.19,1),(.865,.24,.055,2),(.91,.28,.05,2),
              (.98,.11,.025,2),(1.,.012,.012,2)]
    # Minimise the gravitational height of the shaft's centre of mass over all
    # non-penetrating straight poses. A simple endpoint lift left only one
    # support and suspended the nib: this supporting-line solve gives contacts
    # on either side of the centre of mass, including the lower right page.
    q=np.linspace(0,1,401); axis=a[None,:]+q[:,None]*(b-a)
    radii=np.interp(q,[s[0] for s in sections],[s[2] for s in sections])
    floor=np.array([B.height(p[0],p[1],bk.params,bk.ck)[0] for p in axis])
    required=floor+radii+.008
    mass=radii*np.interp(q,[s[0] for s in sections],[s[1] for s in sections])
    centre=float(np.sum(q*mass)/np.sum(mass))
    objective=lambda slope:float(np.max(required-slope*q)+slope*centre)
    fit=minimize_scalar(objective,bounds=(-8.,8.),method='bounded',options={'xatol':1e-11})
    if not fit.success:raise RuntimeError('resting pen pose did not converge')
    a[2]=np.max(required-fit.x*q);b[2]=a[2]+fit.x
    d=b-a; length=np.linalg.norm(d); d/=length
    right=np.cross(d,[0,0,1.]); right/=np.linalg.norm(right)
    up=np.cross(right,d)
    vertices=[]; normals=[]; uv=[]
    sides=20
    for u,rx,rz,mat in sections:
        for j in range(sides):
            t=2*np.pi*j/sides
            vertices.append(a+d*length*u+right*rx*np.cos(t)+up*rz*np.sin(t))
            n=right*np.cos(t)/rx+up*np.sin(t)/rz
            normals.append(n/np.linalg.norm(n)); uv.append([u,t])
    triangles=[]; materials=[]
    for k in range(len(sections)-1):
        for j in range(sides):
            nj=(j+1)%sides
            # Actual narrow slit in the upper nib, not a black line on paper.
            if k>=8 and j in (4,5):continue
            x=k*sides+j; y=k*sides+nj; z=(k+1)*sides+j; w=(k+1)*sides+nj
            triangles.extend([(x,z,y),(y,z,w)]); materials.extend([sections[k+1][3]]*2)
    return np.array(vertices),np.array(normals),np.array(uv),np.array(triangles),np.array(materials)


def _pixels(cam, points, triangle):
    xy,z=cam.project(points[triangle])
    x0=max(0,int(np.floor(xy[:,0].min()))-1);x1=min(cam.W,int(np.ceil(xy[:,0].max()))+1)
    y0=max(0,int(np.floor(xy[:,1].min()))-1);y1=min(cam.H,int(np.ceil(xy[:,1].max()))+1)
    if x1<=x0 or y1<=y0:return None
    yy,xx=np.mgrid[y0:y1,x0:x1];xx=xx+.5;yy=yy+.5
    a,b,c=xy
    den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
    if abs(den)<1e-9:return None
    wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
    wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
    weights=np.stack([wa,wb,1-wa-wb],-1)
    mask=(weights.min(-1)>=0)
    weights=weights/z
    inv=weights.sum(-1)
    depth=1/np.maximum(inv,1e-12)
    weights*=depth[...,None]
    return (slice(y0,y1),slice(x0,x1)),weights,depth,mask


def composite(hdr,G,bk,cam,light,mesh=None):
    verts,normals,uv,faces,mats=geometry(bk) if mesh is None else mesh
    # Project the silhouette along the real light-to-pen direction onto the
    # cockled page. Iteration accounts for page height beneath each shadow point.
    shadow=verts.copy()
    for _ in range(4):
        heights=np.array([B.height(p[0],p[1],bk.params,bk.ck)[0] for p in shadow])
        k=(heights-verts[:,2])/(verts[:,2]-light.pos[2])
        shadow=verts+(verts-light.pos)*k[:,None]
    mask=np.zeros((cam.H,cam.W),np.float32)
    for tri in faces:
        p=_pixels(cam,shadow,tri)
        if p is not None:
            sl,_,_,inside=p;mask[sl]=np.maximum(mask[sl],inside)
    # Broad area light: a pen on the page has a soft shadow, darkest at contact.
    gaps=verts[:,2]-np.array([B.height(p[0],p[1],bk.params,bk.ck)[0] for p in verts])
    sigma=max(.6,float(gaps.mean())*light.radius/max(light.pos[2],1)*cam.F/cam.dist*.8)
    mask=cv2.GaussianBlur(mask,(0,0),sigma)
    hdr*=1-.58*mask[...,None]
    depth=G[...,1].copy()
    for tri,material in zip(faces,mats):
        p=_pixels(cam,verts,tri)
        if p is None:continue
        sl,w,z,inside=p
        hit=inside&(z<depth[sl])
        if not hit.any():continue
        pos=w@verts[tri];N=w@normals[tri];N/=np.maximum(np.linalg.norm(N,axis=-1,keepdims=True),1e-12)
        V=cam.pos-pos;V/=np.linalg.norm(V,axis=-1,keepdims=True)
        N=np.where((N*V).sum(-1,keepdims=True)<0,-N,N)
        L=light.pos-pos;L/=np.linalg.norm(L,axis=-1,keepdims=True)
        half=L+V;half/=np.maximum(np.linalg.norm(half,axis=-1,keepdims=True),1e-12)
        diffuse=np.maximum((N*L).sum(-1),0)
        spec=np.maximum((N*half).sum(-1),0)**(32 if material==0 else 65)
        st=w@uv[tri]
        if material==0:
            grain=.88+.08*np.sin(st[...,0]*110+np.sin(st[...,1]*7))+.04*np.sin(st[...,1]*21)
            albedo=grain[...,None]*np.array([.13,.065,.028]);shine=.17
        else:
            albedo=np.array([.36,.25,.105]) if material==1 else np.array([.32,.30,.25]);shine=.55
        rgb=albedo*(.13+.84*diffuse[...,None])*light.col + shine*spec[...,None]*light.col
        # Dried iron-gall stain at the writing tip, adhering to the metal.
        if material==2:rgb*=1-.75*np.clip((st[...,0]-.968)/.032,0,1)[...,None]
        hdr[sl][hit]=rgb[hit];depth[sl][hit]=z[hit]
    return hdr,depth
