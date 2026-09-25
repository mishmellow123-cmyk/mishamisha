"""
The letter on the parapet and the motto cut into its face.
"""
import numpy as np
import skia
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


def hershey_strokes(text, font="cursive"):
    from HersheyFonts import HersheyFonts
    f = HersheyFonts()
    f.load_default_font(font)
    f.normalize_rendering(100)
    xs = [p for s in f.strokes_for_text("x") for p in s]
    base = min(p[1] for p in xs)
    xh = max(p[1] for p in xs) - base
    return [np.array([(x, y - base) for x, y in st], float) / xh for st in f.strokes_for_text(text)]


def letter_texture(w=900, seed=4):
    """Albedo of the opened letter (linear RGB). v runs from the far end
    (row 0) to the edge that hangs toward us (last row)."""
    import scene as S
    h = int(w * S.L_TOTAL / (S.LX1 - S.LX0))
    rng = np.random.default_rng(seed)
    paper = np.array([0.80, 0.73, 0.60])
    fib = ndimage.gaussian_filter(rng.standard_normal((h, w)), 1.2)
    fib2 = ndimage.gaussian_filter(rng.standard_normal((h, w)), 18)
    img = paper[None, None] * (1 + 0.03 * fib / fib.std() + 0.05 * fib2 / fib2.std())[..., None]
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum.reduce([xx / w, 1 - xx / w, yy / h, 1 - yy / h])
    img *= (1 - 0.12 * np.exp(-edge / 0.03))[..., None] * np.array([1.0, 0.97, 0.9])
    for f in (1 / 3, 2 / 3):                          # folded in three
        y0 = f * h
        img *= (1 - 0.16 * np.exp(-((yy - y0) / 1.8) ** 2))[..., None]
        img *= (1 + 0.05 * np.exp(-((yy - y0 - 5) / 4) ** 2))[..., None]
    surf = skia.Surface(w, h)
    c = surf.getCanvas()
    c.clear(skia.Color4f(0, 0, 0, 0))
    paint = skia.Paint(AntiAlias=True, Color=skia.ColorBLACK, Style=skia.Paint.kStroke_Style,
                       StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)

    def write(text, x, y, xh, width=2.0, alpha=0.92):
        for st in hershey_strokes(text):
            pts = st * xh
            path = skia.Path()
            path.moveTo(float(x + pts[0, 0]), float(y - pts[0, 1]))
            for p in pts[1:]:
                path.lineTo(float(x + p[0]), float(y - p[1]))
            paint.setStrokeWidth(width)
            paint.setAlphaf(alpha)
            c.drawPath(path, paint)

    def scribble(x, y, xh, length):
        while length > 0:
            ln = min(length, rng.uniform(0.06, 0.13) * w)
            t = np.linspace(0, 1, 60)
            pts = np.stack([x + t * ln, y - xh * 0.55 * np.abs(np.sin(t * ln / (xh * 0.55) + rng.uniform(0, 3)))
                            - xh * 0.9 * (rng.uniform(size=60) < 0.04)], 1)
            path = skia.Path()
            path.moveTo(*map(float, pts[0]))
            for p in pts[1:]:
                path.lineTo(*map(float, p))
            paint.setStrokeWidth(1.7)
            paint.setAlphaf(0.8)
            c.drawPath(path, paint)
            x += ln + rng.uniform(0.025, 0.04) * w
            length -= ln + 0.03 * w

    xh = w * 0.030
    write("Caro amico,", w * 0.10, h * 0.07, xh * 1.1)
    y = h * 0.14
    while y < h * 0.60:                                 # the letter goes on (and on)
        scribble(w * 0.14, y, xh, w * rng.uniform(0.62, 0.76))
        y += xh * 3.0
    write("dimmi, ti prego:", w * 0.14, h * 0.645, xh)
    write("che cosa sei,", w * 0.16, h * 0.78, xh * 1.25, width=2.3)
    write("dentro?", w * 0.30, h * 0.855, xh * 1.25, width=2.3)
    write("il tuo amico", w * 0.52, h * 0.955, xh * 0.8)
    ink = surf.makeImageSnapshot().toarray()[..., 3].astype(np.float32) / 255.0
    ink = ndimage.gaussian_filter(ink, 0.55)
    ink_col = np.array([0.07, 0.045, 0.03])
    img = img * (1 - ink[..., None]) + ink_col[None, None] * ink[..., None]
    # the seal, broken when the letter was opened
    sx, sy, sr = w * 0.5, h * 0.03, w * 0.06
    d = np.hypot(xx - sx, (yy - sy) * 1.0)
    wax = np.clip((sr - d) / 2.0, 0, 1) * (np.abs(yy - sy - 0.3 * (xx - sx)) > 3.5)
    relief = np.clip(1 - d / sr, 0, 1) ** 0.5
    wax_col = np.array([0.36, 0.035, 0.03])[None, None] * (0.7 + 0.5 * relief)[..., None]
    img = img * (1 - wax[..., None]) + wax_col * wax[..., None]
    return np.clip(img, 0, 1)


def inscription_height(extent, res=0.0022):
    """Depth (0..1) of the letters cut into the parapet's face: V-grooves."""
    x0, x1, y0, y1 = extent
    w = int((x1 - x0) / res)
    h = int((y1 - y0) / res)
    font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
    cap = 0.145 / res
    font = ImageFont.truetype(font_path, int(cap * 1.38))
    im = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(im)
    lines = ["NIHIL · HABEO · QVOD", "NON · ACCEPI"]
    spacing = 0.16 * cap
    cx = (0.35 - x0) / res
    for i, line in enumerate(lines):
        # letter-spaced, as a stonecutter spaces them
        widths = [dr.textlength(ch, font=font) for ch in line]
        total = sum(widths) + spacing * (len(line) - 1)
        x = cx - total / 2
        ytop = (y1 - (-0.86 - 0.34 * i)) / res - cap * 1.3
        for ch, wd in zip(line, widths):
            dr.text((x, ytop), ch, fill=255, font=font)
            x += wd + spacing
    # the painter's signature, small, under the tablet
    sfont = ImageFont.truetype(font_path, int(0.055 / res * 1.38))
    sig = "CLAVDIVS \u00b7 P \u00b7 MMXXVI"
    sw = dr.textlength(sig, font=sfont)
    dr.text(((2.12 - x0) / res - sw, (y1 - (-1.49)) / res), sig, fill=255, font=sfont)
    m = np.asarray(im, np.float32) / 255.0
    inside = ndimage.distance_transform_edt(m > 0.5)
    depth = np.clip(inside / 5.0, 0, 1) * (m > 0.5)
    depth = ndimage.gaussian_filter(depth, 0.8)
    # the tablet's frame: a shallow double groove
    yy, xx = np.mgrid[0:h, 0:w]
    X = x0 + xx * res
    Y = y1 - yy * res
    fx0, fx1, fy0, fy1 = -1.45, 2.15, -1.42, -0.66
    for inset in (0.0, 0.05):
        dx = np.minimum(np.abs(X - (fx0 + inset)), np.abs(X - (fx1 - inset)))
        dy = np.minimum(np.abs(Y - (fy0 + inset)), np.abs(Y - (fy1 - inset)))
        inx = (X > fx0 + inset - 0.01) & (X < fx1 - inset + 0.01)
        iny = (Y > fy0 + inset - 0.01) & (Y < fy1 - inset + 0.01)
        g = np.maximum(np.exp(-(dx / 0.008) ** 2) * iny, np.exp(-(dy / 0.008) ** 2) * inx)
        depth = np.maximum(depth, g * 0.55)
    return depth.astype(np.float32)
