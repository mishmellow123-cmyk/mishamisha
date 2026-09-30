"""Opt-in D-v2 Old Fire: close unmaking, then the Mountain page and its caption.

Absolute D5840..6079. The accepted Mountain strokes, paper coordinates and
material shader are reused. A high-density cropped texture strip supplies the
close shot without allocating a high-density texture for the whole page.
"""
import math
import os

for _pool in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_pool, '1')

import cv2
import numpy as np

import oldfire as R3

B, BC, PG, RP, pen, T1 = R3.B, R3.BC, R3.PG, R3.RP, R3.pen, R3.T1
START, END, FPS = 5840, 6080, 24
CAPTION = R3.CAPTION
CLOSE_END = 140
PULL_END = 184
CAPTION_START = 176
# Round 5 is explicit so the adopted Round 4 take remains reproducible.
HOLD_CLOSE_END, HOLD_PULL_END, HOLD_CAPTION_START = 99, 123, 119
HOLD_BEAD_FRAME = 91
HOLD_FIRST_CHANGED = 5922


def local_time(frame):
    if (isinstance(frame, bool) or not isinstance(frame, (int, np.integer))
            or not START <= frame < END):
        raise ValueError('OldFireV2 accepts absolute integer D frames 5840..6079')
    return (frame - START) / FPS


def smooth01(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def unmaking_time(t, caption_hold=False):
    """Canonical fall; flare at local 60, full bead by the close's last frame."""
    if caption_hold and t > (HOLD_FIRST_CHANGED-START-1)/FPS:
        boundary = (HOLD_FIRST_CHANGED-START-1)/FPS
        return float(np.interp(t, [boundary, HOLD_BEAD_FRAME/FPS, 239/FPS],
                               [unmaking_time(boundary), 5.4, 6.3]))
    return float(np.interp(t, [0., .55, .90, 2.5, 3.6, 140/24., 239/24.],
                          [.2, .2, .55, 2.65, 3.5, 5.4, 6.3]))


def ember_shapes(embers):
    """Separate rounded coals, with small irregular hot centers, in source cm."""
    for k, (x, y, radius) in enumerate(embers):
        angle = np.linspace(0., 2.*np.pi, 33)[:-1] + .3*k
        radius = .55*radius*(1. + .12*np.sin(3.*angle+k) + .06*np.sin(5.*angle-k))
        offset = np.column_stack((.95*radius*np.cos(angle), .70*radius*np.sin(angle)))
        center = np.array([x, y])
        yield center+offset, center+.45*offset, .76+.18*math.sin(k*2.3)


def flame_base(bx, by, width, t, key):
    """A shallow curved continuation from a tongue's right foot to its left."""
    angle = np.linspace(0., np.pi, 19)
    depth = .32*width*(1.+.12*math.sin(t*1.9+key*1.7))
    return np.column_stack((bx+.75*width*np.cos(angle), by+depth*np.sin(angle)))


class MountainRingV2(R3.MountainRing):
    """Canonical unmaking with a coal bed suited to the much closer camera.

    The inherited wide, uniformly bright eight-sided embers merge into a bar
    at this scale. Tongue feet gain curved continuations into those coals;
    their existing upper bodies, Ring state/geometry and late bead contour
    remain inherited.
    """

    def _embers(self, win, ox, oy, t, flare):
        height, width = win.shape[:2]
        coal = np.zeros((height, width), np.float32)
        heart = np.zeros_like(coal)
        strokes = pen.Strokes()
        for k, (outer, inner, power) in enumerate(ember_shapes(self.embers)):
            for target, points, value in ((coal, outer, 1.), (heart, inner, power)):
                poly = np.rint((points-[ox, oy])*self.ppc*16).astype(np.int32)
                cv2.fillPoly(target, [poly], value, lineType=cv2.LINE_AA, shift=4)
            pen.line(strokes, outer, .014, 300+k, smooth=0, closed=True)
        ink, _ = pen.raster(strokes.pack(), 1e9, self.ppc, height, width,
                             pen.INK, ox=ox, oy=oy)
        occupied = coal > 0.
        flicker = 1. + .15*math.sin(t*7.) + .1*math.sin(t*17.)
        win[..., 0] = np.where(occupied, np.maximum(.55*coal, ink),
                               np.maximum(win[..., 0], ink))
        win[..., 2] = np.where(occupied, .10*heart*(1.-.85*ink), win[..., 2])
        win[..., 4] = np.where(occupied, (.018*coal+.30*heart)*flicker*flare,
                               win[..., 4])

    def _fire(self, win, ox, oy, t, tongues, flare, seed, grow=1.):
        super()._fire(win, ox, oy, t, tongues, flare, seed, grow)
        actual_grow = grow*(R3.foreground_growth(t) if tongues is self.front else 1.)
        height, width = win.shape[:2]
        original = np.zeros((height, width), np.float32)
        gilt, heart, core = (np.zeros_like(original) for _ in range(3))
        strokes = pen.Strokes()
        for k, (bx, by, h, w, lean) in enumerate(tongues):
            key = k+seed
            body = RP.tongue(bx, by, h*actual_grow, w, lean, t, key)
            poly = np.rint((body-[ox, oy])*self.ppc*16).astype(np.int32)
            cv2.fillPoly(original, [poly], 1., lineType=cv2.LINE_AA, shift=4)
            for target, dy, hscale, wscale, dt in (
                    (gilt, 0., 1., 1., 0.), (heart, .03, .62, .72, .03),
                    (core, .05, .45, .42, .05)):
                upper = RP.tongue(bx, by+dy, h*actual_grow*hscale,
                                   w*wscale, lean, t+dt, key)
                base = flame_base(bx, by+dy, w*wscale, t+dt, key)
                rounded = np.vstack((upper, base[1:-1]))
                poly = np.rint((rounded-[ox, oy])*self.ppc*16).astype(np.int32)
                cv2.fillPoly(target, [poly], 1., lineType=cv2.LINE_AA, shift=4)
            pen.line(strokes, flame_base(bx, by, w, t, key), .016,
                     seed*7+k, smooth=0, lift=(2., 4.))
        ink, _ = pen.raster(strokes.pack(), 1e9, self.ppc, height, width,
                             pen.INK, ox=ox, oy=oy)
        added = (gilt > 0.) & (original == 0.)
        flicker = 1.+.1*math.sin(t*13.)+.06*math.sin(t*29.+1.)
        # Paint only the new feet; no original flame-body pixels are erased.
        win[..., 0] = np.where(added, np.maximum(ink*(1.-.45*core), .04*gilt), win[..., 0])
        win[..., 2] = np.where(added, .6*heart*(1.-.85*ink), win[..., 2])
        win[..., 4] = np.where(added, (.07*gilt+.18*heart+.5*core*flicker)*flare, win[..., 4])
        win[..., 3] = np.where(added, 0., win[..., 3])


class DetailStrip:
    """A cropped mip pyramid addressed with unchanged absolute page UVs.

    Native rows 1..7 cm are stored; previews include additional padding.
    Metadata keeps the full page's row/column coordinates and subtracts the
    omitted prefix from each level's offset. Padding covers recursive mip
    filtering plus bilinear reach. Shader callers must exclude pixels outside
    SAFE, never sample missing rows.
    """
    Y0, Y1 = 1., 7.
    SAFE = (1.75, 6.25)
    LEVELS = 7

    def __init__(self, ppc):
        if not isinstance(ppc, int) or ppc <= 0 or ppc % 64:
            raise ValueError('detail density must be a positive multiple of 64')
        self.ppc = ppc
        # Seven pyrDown levels reach 126 base pixels; bilinear sampling can
        # reach one further 64-pixel interval. Low-density previews therefore
        # retain the true top boundary and enough additional bottom rows.
        if ppc == 64:
            self.Y0, self.Y1 = 0., 10.
        elif ppc < 256:
            self.Y0, self.Y1 = 0., 8.
        self.W, self.H = int(PG.PW * ppc), int((self.Y1-self.Y0) * ppc)
        self.row0 = int(self.Y0 * ppc)
        self.chan = np.zeros((self.H, self.W, B.NCH), np.float32)
        self.data = None

    def build(self):
        self.data = None  # release the preceding packed allocation first
        levels = [self.chan]
        for _ in range(self.LEVELS - 1):
            levels.append(cv2.pyrDown(levels[-1]))
        self.meta = np.zeros((self.LEVELS, 3), np.int64)
        count = sum(level.size for level in levels)
        packed = np.empty(count, np.float32)
        offset = 0
        for level, array in enumerate(levels):
            h, w = int(PG.PH*self.ppc) >> level, self.W >> level
            row0 = self.row0 >> level
            self.meta[level] = (offset-row0*w*B.NCH, h, w)
            packed[offset:offset+array.size] = array.ravel()
            offset += array.size
        self.data = packed
        return self

    def sample(self, u, v, footprint):
        """Guarded diagnostic sampler; rendering applies this guard per pixel."""
        if not (0 <= u <= PG.PW and self.SAFE[0] <= v <= self.SAFE[1]):
            raise ValueError('sample outside the detail strip safe interval')
        value = np.zeros(B.NCH, np.float64)
        B.tex(self.data, self.meta, float(self.ppc), u, v, footprint, value)
        return value


class OldFireV2(R3.OldFire):
    def __init__(self, W=1920, H=804, ppc=110, caption_hold=False):
        super().__init__(W, H, ppc)
        self.caption_hold = bool(caption_hold)
        self.close_end, self.pull_end, self.caption_start = (
            (HOLD_CLOSE_END, HOLD_PULL_END, HOLD_CAPTION_START) if self.caption_hold
            else (CLOSE_END, PULL_END, CAPTION_START))
        # Multiples of 64 align all seven mip levels with the full page.
        self.detail_ppc = 64 * math.ceil(max(ppc, 320*W/1920.) / 64)

    def layout(self, t):
        book = self.book(.6, 2.8, seed=3)
        q = float(smooth01((t*FPS-self.close_end)/(self.pull_end-self.close_end)))
        close_v = 3.8 + .4 * float(smooth01(t/.95))
        # Aim down the page slightly ahead of the pull, so the caption enters
        # before its ink reveal. Both motions arrive with zero velocity.
        aim = float(smooth01((t*FPS-self.close_end)/(self.pull_end-self.close_end)*1.18))
        uv = np.array([10.62, close_v]) * (1-aim) + np.array([9.8, 8.8]) * aim
        target = book.page_to_world('R', uv[:1], uv[1:])[0]
        close = np.array([-.1, -4.9, 5.8])
        wide = np.array([-.6, -25.5, 25.]) * .975
        camera = B.Cam(target + close*(1-q) + wide*q, target, 38., self.W, self.H)
        camera.dof_k = 2.
        return book, camera

    def caption(self):
        def make():
            line = T1.TextInkLine(CAPTION, self.ppc)
            line.f_in, line.f_out = START+self.caption_start, END+12
            return line
        return self.once('oldfire_v2_caption', make)

    def ring_for(self, density):
        return self.once(('oldfire_v2_ring', density), lambda: MountainRingV2(
            ppc=density*R3.RING_SCALE, open_band=False))

    def patch_at(self, t, density):
        ring = self.ring_for(density)
        if ring.gap_degrees != 0.:
            raise ValueError('The old story requires the canonical closed Ring')
        x0, y0, x1, y1 = ring.WIN
        i0, i1 = int(y0*ring.ppc), int(y1*ring.ppc)
        j0, j1 = int(x0*ring.ppc), int(x1*ring.ppc)
        offsets = ((np.arange(9)+.5)/9./48.-1./96. if .55 < t < .97 else (0.,))
        patch = None
        for dt in offsets:
            sample = ring.texture(unmaking_time(t+dt, self.caption_hold)).chan[i0:i1, j0:j1]
            if patch is None:
                patch = sample.copy()
            else:
                patch += sample
        patch /= len(offsets)
        origin = R3.mapped_point((j0/ring.ppc, i0/ring.ppc))
        col, row = np.rint(origin*density).astype(int)
        return row, col, patch

    def texture(self, frame):
        t = local_time(frame)
        texture, base = self.mountain_page()
        texture.chan[:] = base
        row, col, patch = self.patch_at(t, self.ppc)
        height, width = patch.shape[:2]
        R3.overlay_patch(texture.chan[row:row+height, col:col+width], patch)
        self.caption().apply(texture.chan, frame)
        return texture.build()

    def detail(self, frame):
        t = local_time(frame)
        row, col, patch = self.patch_at(t, self.detail_ppc)
        height, width = patch.shape[:2]
        def make():
            strip = DetailStrip(self.detail_ppc)
            _, strokes = R3.mountain_strokes()
            packed = strokes.pack()
            for layer, channel in ((pen.INK, 0), (pen.GILT, 2), (pen.PENCIL, 3)):
                coverage, wet = pen.raster(packed, 1e9, strip.ppc, strip.H, strip.W,
                                            layer, dry=.9, oy=strip.Y0)
                strip.chan[..., channel] = coverage
                if layer == pen.INK:
                    strip.chan[..., 1] = wet
            base = strip.chan[row-strip.row0:row-strip.row0+height, col:col+width].copy()
            return strip, base
        strip, base = self.once('oldfire_v2_detail', make)
        receiver = strip.chan[row-strip.row0:row-strip.row0+height, col:col+width]
        if receiver.shape != patch.shape:
            raise ValueError('unmaking patch outside the high-detail strip')
        receiver[:] = base
        R3.overlay_patch(receiver, patch)
        return strip.build()

    def refine(self, hdr, alpha, G, book, camera, light, left, strip, fire_light, t):
        v = G[..., 9]
        weight = (smooth01((v-strip.SAFE[0])/.3) * smooth01((strip.SAFE[1]-v)/.3)
                  * (G[..., 0] == B.M_PAGE_R))
        valid = weight > 0.
        if not valid.any():
            return hdr, alpha
        # Keep world positions, normals, absolute paper UVs and all lighting.
        # Excluded receivers become M_NONE for this temporary shading pass.
        materials = G[..., 0].copy()
        G[..., 0][~valid] = B.M_NONE
        refined = np.zeros_like(hdr, dtype=np.float64)
        coverage = np.ones_like(alpha, dtype=np.float64)
        fill = np.array([.35, -.5, .8]); fill /= np.linalg.norm(fill)
        zero = np.zeros(20)
        try:
            B.shade_kernel(refined, coverage, G, book.params, camera.pos,
                light.pos, light.col, light.radius, fill, np.array([.010, .012, .016]),
                np.array([.012, .009, .007]), np.array([.9, .5, .22]),
                left.data, left.meta, float(left.ppc), strip.data, strip.meta, float(strip.ppc),
                float(book.seed), float(book.seed+17), 1., zero, zero, fire_light, float(t),
                strip.data, strip.meta, float(strip.ppc), strip.data, strip.meta, float(strip.ppc),
                left.data, left.meta, float(left.ppc), .05, 1., .03, False, None)
        finally:
            G[..., 0] = materials
        hdr = (hdr*(1-weight[..., None]) + refined*weight[..., None]).astype(np.float32)
        return hdr, alpha

    def frame(self, frame):
        t = local_time(frame)
        book, camera = self.layout(t)
        right = self.texture(frame)
        strip = self.detail(frame)
        left = self.tex_text(41)
        light = self.light((-50., 48., 36.), 2.2, t)
        fire_uv = R3.mapped_point(RP.RingPage.FIRE)
        fire_world = book.page_to_world('R', fire_uv[:1], fire_uv[1:])[0]
        strength = B.flicker(t*1.6, 5, .14)
        fire_light = np.array([[*fire_world[:2], fire_world[2]+.7,
                                .10*strength, .045*strength, .012*strength]])
        hdr, alpha, G = B.render(book, camera, light, left, right, t,
                                 xlights=fire_light, texS=left, st_amt=.05, fire_k=1.)
        hdr, alpha = self.refine(hdr, alpha, G, book, camera, light, left, strip, fire_light, t)
        hdr = B.dof(hdr, G[..., 1].astype(np.float32), camera, camera.dof_k*self.W/1920.)
        return hdr, alpha
