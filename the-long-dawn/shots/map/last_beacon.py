"""C v5.2 shot 15, LAST BEACON: C3440..3839 (400 frames).

Opt-in continuation of THE MAP ANSWERS. The invented terra v5 world, parchment,
camera projection, pen flames and living fire all come from the accepted map.
Eight fictional kingdoms answer in an irregular near/far order. The seventh
flame is fully drawn at C3724; one kingdom stays dark through C3783. The last catches over
C3784..3791, leaving C3792..3839 for EDIT's provisional caption.

    python shots/map/last_beacon.py bake
    python shots/map/last_beacon.py frames --frames 3440,3724,3783,3792,3839 --scale .5

No titles, grain, burn, standing stones, road or changes to the old map entry
points. A cropped mip atlas calls the original baker at 48 pixels/map-unit;
its manifest pins every backing source file. It is not a replacement map.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'lib'))
import geo
import road
import render as MAP
import pen
import look

cv2.setNumThreads(0)
F0, F1 = 3440, 3840                 # exclusive end
CATCH = 8
ONE_DARK = (3724, 3784)              # exactly 60 frames, exclusive end
CAPTION_HOLD = (3792, 3840)          # exactly 48 frames, exclusive end
# First is the existing range's beacon (terra.BEACON). The ordering deliberately
# crosses the map: neither distance from that fire nor left-to-right progression.
BEACONS = np.array([(13, 8), (-20, 3), (-11, 25), (4, 27),
                    (-6, 9), (26, 27), (31, 8), (7, 0)], np.float64)
IGNITION = F0 + np.array([-8, 205, 26, 238, 344, 142, 276, 81])
LAST = 4
PPD = 48.0
CACHE = ROOT / 'renders' / 'map_last_beacon_C' / 'atlas'


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def catch_at(frame):
    return smooth((frame - IGNITION) / CATCH)


def camera_at(frame, W=1920, H=804):
    # The same westward crane / raking sheet perspective as road.CAM_KEYS.
    # It settles before the long silence; no push towards the former stone ring.
    u = float(smooth((frame - F0) / 276.0))
    return MAP.Cam(7.0 - 2.0 * u, 12.0, 76.0 + 5.0 * u,
                   37.0 - 5.0 * u, -3.5 + .5 * u, W, H)


def kingdom_at(X, Y):
    """Eight exclusive territories, with gently wandering shared ink borders.

    Domain warping makes the boundaries follow an irregular hand, avoiding a
    diagram of straight Voronoi edges. This is invented political geography on
    terra's invented land; no real geography, heraldry or names are introduced.
    """
    def warp(x, y):
        return (x + 1.8 * np.sin(y * .17) + .7 * np.sin(x * .29 + y * .11),
                y + 1.6 * np.sin(x * .13) + .65 * np.sin(y * .31 - x * .19))
    x, y = warp(np.asarray(X), np.asarray(Y))
    bx, by = warp(BEACONS[:, 0], BEACONS[:, 1])
    best = np.full(np.broadcast_shapes(x.shape, y.shape), np.inf)
    label = np.zeros(best.shape, np.uint8)
    for i, (cx, cy) in enumerate(zip(bx, by)):
        d = (x - cx) ** 2 + 1.12 * (y - cy) ** 2
        take = d < best
        label[take], best[take] = i, d[take]
    return label


def atlas_spec():
    corners = []
    for f in (F0, F0 + 138, F0 + 276, F1 - 1):
        cam = camera_at(f)
        x, y = cam.screen_to_map(np.array([0, 1920, 1920, 0]), np.array([0, 0, 804, 804]))
        corners.extend(zip(x, y))
    q = np.array(corners)
    x0, y0 = np.floor(q.min(axis=0) - 5)
    x1, y1 = np.ceil(q.max(axis=0) + 5)
    return dict(x0=float(x0), y1=float(y1), width=int((x1-x0)*PPD),
                height=int((y1-y0)*PPD), ppd=PPD)


def source_manifest():
    # Source content, not a branch name or mtime, identifies a reusable atlas.
    names = ['bake.py', 'geo.py', 'terra.py', 'features.py', 'sheet.py', 'ink.py', 'pen.py', 'noise.py']
    sources = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names}
    for n in ('look.py',):
        sources['../../lib/' + n] = hashlib.sha256((ROOT / 'lib' / n).read_bytes()).hexdigest()
    return dict(format=1, world=geo.WORLD_VER, atlas=atlas_spec(), sources=sources)


def atlas_dir():
    key = hashlib.sha256(json.dumps(source_manifest(), sort_keys=True).encode()).hexdigest()[:16]
    return CACHE / key


def prepare_world_cache():
    """Source-pin generated geography too, without changing any shared module.

    geo's default versioned cache is suitable for its old shot contract; this
    new atlas must not attribute a pre-existing cache to the source it hashes.
    The override is private to this opt-in Python process.
    """
    manifest = source_manifest()
    identity = dict(world=manifest['world'], sources=manifest['sources'])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = CACHE.parent / 'world' / key
    directory.mkdir(parents=True, exist_ok=True)
    geo.CACHE = str(directory)


def validate_atlas(path, manifest):
    """A completion marker alone cannot attest to a missing/truncated atlas."""
    if json.loads((path / 'manifest.json').read_text()) != manifest:
        raise RuntimeError(f'Atlas source manifest mismatch: {path}')
    s = manifest['atlas']
    h, w = s['height'], s['width']
    try:
        for L in range(6):
            a = np.load(path / f'L{L}.npy', mmap_mode='r', allow_pickle=False)
            if a.shape != (h, w, 3) or a.dtype != np.uint8:
                raise ValueError(f'L{L} has unexpected shape or dtype')
            h, w = h//2, w//2
        a = np.load(path / 'relief.npy', mmap_mode='r', allow_pickle=False)
        if a.shape != (s['height']//12, s['width']//12) or a.dtype != np.float32:
            raise ValueError('relief has unexpected shape or dtype')
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Incomplete or corrupt atlas at {path}: {exc}') from exc


def validate_beacon_ground():
    land = geo.sample('land', BEACONS[:, 0], BEACONS[:, 1])
    lake = geo.sample('lake', BEACONS[:, 0], BEACONS[:, 1])
    if not np.all(np.isfinite(land)) or not np.all(np.isfinite(lake)) or np.any(land < .95) or np.any(lake > .05):
        raise RuntimeError(f'Beacon must stand on dry terra land; land={land}, lake={lake}')


def build_atlas():
    """Serial tiles bound temporary RAM; all terrain is the original bake_region.

    No multiprocessing or farm access. World/glyph caches are built by the
    original deterministic generators in this shot's source-pinned cache.
    """
    import bake
    import sheet
    cv2.setNumThreads(0)
    prepare_world_cache()
    path = atlas_dir()
    manifest = source_manifest()
    done = path / 'manifest.json'
    if done.exists() and json.loads(done.read_text()) == manifest:
        validate_atlas(path, manifest)
        print('atlas ready', path, flush=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    bake.feats()
    bake.glyph_bank()
    validate_beacon_ground()
    print('original world/glyphs ready; all eight beacons on dry land', flush=True)
    s = manifest['atlas']
    W, H = s['width'], s['height']
    mm = np.lib.format.open_memmap(path / 'L0.npy', mode='w+', dtype=np.uint8, shape=(H, W, 3))
    tile = 768
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            t0 = time.monotonic()
            w, h = min(tile, W-x), min(tile, H-y)
            r = bake.bake_region(s['x0']+x/PPD, s['y1']-y/PPD, w, h, PPD)
            rgb = bake.l2s(r['rgb'])
            rng = np.random.default_rng(y * W + x)
            rgb += (rng.random(rgb.shape, np.float32)-rng.random(rgb.shape, np.float32))/255
            mm[y:y+h, x:x+w] = np.clip(np.round(rgb*255), 0, 255).astype(np.uint8)
            mm.flush()
            print(f'atlas tile {x},{y} {w}x{h}: {time.monotonic()-t0:.1f}s', flush=True)
    del mm
    lut = look.srgb_to_linear(np.arange(256, dtype=np.float32)/255)
    for L in range(5):
        src = np.load(path / f'L{L}.npy', mmap_mode='r')
        h, w = src.shape[:2]
        dst = np.lib.format.open_memmap(path / f'L{L+1}.npy', mode='w+', dtype=np.uint8,
                                       shape=(h//2, w//2, 3))
        for y in range(0, h//2*2, 256):
            block = lut[np.asarray(src[y:min(y+256, h//2*2), :w//2*2])]
            small = cv2.resize(block, (w//2, block.shape[0]//2), interpolation=cv2.INTER_AREA)
            dst[y//2:y//2+len(small)] = np.clip(np.round(bake.l2s(small)*255), 0, 255).astype(np.uint8)
        dst.flush()
        del src, dst
    # Same paper-relief normal construction and grid density as MAP.Sheet.
    _, height = sheet.parchment(s['x0'], s['y1'], int(H/12), int(W/12), 4.0,
                                sheet.stain_list()[:0], sheet.spot_list()[:0])
    np.save(path / 'relief.npy', height.astype(np.float32))
    done.write_text(json.dumps(manifest, indent=2) + '\n')
    validate_atlas(path, manifest)
    print('atlas complete', path, flush=True)
    return path


class AtlasSheet(MAP.Sheet):
    """Original trilinear sheet sampler with a cropped texture origin."""
    def __init__(self):
        path = atlas_dir()
        if not (path / 'manifest.json').exists():
            raise RuntimeError('Missing source-pinned atlas; run last_beacon.py bake first')
        validate_atlas(path, source_manifest())
        self.spec = atlas_spec()
        self.ppd0 = PPD
        self.levels = [np.load(path / f'L{L}.npy', mmap_mode='r') for L in range(6)]
        self.lut = look.srgb_to_linear(np.arange(256, dtype=np.float32)/255)
        rel = cv2.GaussianBlur(np.load(path / 'relief.npy'), (0, 0), 1.0)
        gy, gx = np.gradient(rel, .25)
        self.nrm = np.dstack([-gx, gy]).astype(np.float32)
        self.nrm_ppd = 4.0

    def _A(self, ppd):
        return np.array([[ppd, 0, -self.spec['x0']*ppd-.5],
                         [0, -ppd, self.spec['y1']*ppd-.5], [0, 0, 1.]])


class LastBeacon(road.RoadShot):
    def __init__(self):
        # Deliberately bypass RoadShot/Shot constructors: neither the old relay
        # nor its ring/road belongs in this opt-in shot.
        self.sheet = AtlasSheet()
        prepare_world_cache()
        validate_beacon_ground()
        self.P = BEACONS.copy()
        self.t_ign = self.clock(IGNITION)
        self.org = np.zeros(8, bool)
        self.size = np.array([2.0, 2.0, 2.1, 2.0, 2.2, 2.0, 2.0, 2.0])
        self.gain = np.array([1.0, .92, 1.05, .98, 1.12, 1.02, .95, 1.0])
        self.rest = np.full(8, .55)
        self.phase = np.random.default_rng(1505).uniform(0, 1000, 8)
        self.f = F0
        self.borders = self.make_borders()

    @staticmethod
    def clock(f):
        return 1985.0 + (f-F0)*.20

    def camera(self, t, W, H):
        return camera_at(self.f, W, H)

    def make_borders(self):
        # Contours are derived from ONE territorial partition so neighbours
        # share a boundary. Clip only the ink to land; never invent a new coast.
        s = atlas_spec()
        xs = np.arange(s['x0'], s['x0']+s['width']/PPD+.1, .1)
        ys = np.arange(s['y1'], s['y1']-s['height']/PPD-.1, -.1)
        label = kingdom_at(xs[None, :], ys[:, None])
        lines = []
        for i in range(8):
            contours, _ = cv2.findContours((label == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                p = c[:, 0].astype(np.float64)
                p = np.column_stack([xs[0]+p[:, 0]*.1, ys[0]-p[:, 1]*.1])
                if len(p) > 2:
                    p = pen.resample(np.vstack([p, p[:1]]), .12)
                    land = geo.sample('land', p[:, 0], p[:, 1])
                    arc = pen.arclen(p)
                    keep = (np.minimum(land[:-1], land[1:]) >= .95) & (arc[:-1] % .64 <= .43)
                    lines.append((p, arc, np.flatnonzero(keep)))
        return lines

    def ink_layer(self, cam, f):
        C = np.zeros((cam.H, cam.W), np.float32)
        dry = np.zeros_like(C)
        for line, arc, visible in self.borders:
            uv, z = cam.project(np.column_stack([line, np.zeros(len(line))]))
            # A fine, broken sepia rule belongs to the same pen as the terrain;
            # it remains distinguishable from a continuous river or coastline.
            radius = (.029 * cam.F/z) * (1+.16*np.sin(arc*1.9))
            for j in visible:
                a, b = uv[j:j+2]
                pen._seg(C, dry, 0, cam.H, *a, *b, radius[j], radius[j+1], .78, .78, 0., 0.)
        # The unlit beacon's low ink stack is visible before it catches.
        uv, z = cam.project(np.column_stack([self.P, np.zeros(8)]))
        up, _ = cam.up2d(np.column_stack([self.P, np.zeros(8)]))
        catches = catch_at(f)
        for i, ((x, y), (ux, uy)) in enumerate(zip(uv, up)):
            sc = cam.F/z[i]
            # The existing two-stroke stack must survive a 480px-wide review.
            # Its heavier unlit ink yields smoothly to the flame's own stack.
            radius = (.055*(1-catches[i]) + .021*catches[i])*sc
            for off, half in ((0, .43), (-.15, .27)):
                a = np.array([x, y])+sc*(off*np.array([ux, uy])-half*np.array([-uy, ux]))
                b = np.array([x, y])+sc*(off*np.array([ux, uy])+half*np.array([-uy, ux]))
                pen._seg(C, dry, 0, cam.H, *a, *b, radius, radius, .95, .95, 0., 0.)
        return C, dry

    def glyph_frame(self, cam, t):
        catches = catch_at(self.f)
        points = np.column_stack([self.P, np.zeros(8)])
        uv, z = cam.project(points)
        up, _ = cam.up2d(points)
        return [(i, *uv[i], *up[i], .72*self.size[i]*cam.F/z[i], catches[i], cam.F/z[i], False)
                for i in range(8) if catches[i] > 0]

    def draw_sparks(self, *args):
        pass                         # no old relay source fountain

    def hush_fn(self, t, H):
        return None                  # EDIT owns provisional caption timing

    def flare(self, t):
        return 0.0

    def room_gain(self, t):
        return 1.35                  # retain border readability during silence

    def render_c(self, f, scale=1.0):
        if not F0 <= f < F1:
            raise ValueError(f'C frame {f} outside {F0}..{F1-1}')
        return super().render_c(f, scale)


def parse_frames(spec):
    frames = []
    for item in spec.split(','):
        if '-' in item:
            a, b = map(int, item.split('-'))
            if b < a:
                raise ValueError('descending frame range')
            frames.extend(range(a, b+1))
        else:
            frames.append(int(item))
    if not frames or any(f < F0 or f >= F1 for f in frames):
        raise ValueError(f'frames must be within {F0}..{F1-1}')
    return sorted(set(frames))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['bake', 'frames'])
    parser.add_argument('--frames', default=f'{F0}-{F1-1}')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--format', choices=['jpg', 'png'], default='jpg',
                        help='farm runner consumes PNG and converts to JPEG for shipping')
    parser.add_argument('--out', type=Path, default=ROOT/'renders'/'map_last_beacon_C')
    args = parser.parse_args()
    if args.action == 'bake':
        build_atlas()
        return
    if not 0 < args.scale <= 1:
        parser.error('--scale must be in (0, 1]')
    frames = parse_frames(args.frames)
    args.out.mkdir(parents=True, exist_ok=True)
    shot = LastBeacon()
    from PIL import Image
    for f in frames:
        t0 = time.monotonic()
        rgb = shot.render_c(f, args.scale)
        img = np.clip(np.round(rgb*255), 0, 255).astype(np.uint8)
        target = args.out/f'f_{f:05d}.{args.format}'
        temporary = target.with_name(target.stem + '.part' + target.suffix)
        if args.format == 'jpg':
            Image.fromarray(img).save(temporary, quality=95, subsampling=0)
        else:
            Image.fromarray(img).save(temporary)
        temporary.replace(target)       # farm never reads a partly written frame
        print(f'C{f} {img.shape[1]}x{img.shape[0]} {time.monotonic()-t0:.2f}s', flush=True)


if __name__ == '__main__':
    main()
