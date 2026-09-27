"""FINISH banding check on the darkest night skies (python3 finish/banding.py, through renderq): frames from the real
edit (Ctx.picture), with and without the finish, through deliver's own x264 master encode (CRF 14, BT.709 limited),
decoded back; the sky's contrast stretched x8, and the largest flat patch of one code value (banding = big flat
patches with hard contours). Outputs: _local_logs/review/finish/qc/banding_darkest_skies.{jpg,txt}."""
import os, sys, subprocess
import numpy as np, cv2
ROOT = os.path.expanduser('~/mishamisha/the-long-dawn')
sys.path.insert(0, os.path.join(ROOT, 'edit')); sys.path.insert(0, os.path.join(ROOT, 'finish'))
import assemble as AS, deliver as DL, stage, titles
OUT = os.path.expanduser('~/mishamisha/_local_logs/review/finish/qc')
os.makedirs(OUT, exist_ok=True)
fin = stage.Finisher()
prof = DL.PROFILES['master']
TESTS = [('A', 3860, 3872, 'DESERT night sky'), ('A', 3800, 3812, 'KARST night'), ('A', 1500, 1512, 'embers towers on black'),
         ('B', 1180, 1192, 'B H1 crop, deep night blue grade')]
ctxs = {}
rows, report = [], []
for cut, f0, f1, what in TESTS:
    ctx = ctxs.setdefault(cut, AS.Ctx(cut, None, 1.0, True))
    dec = {}
    for tag, use in (('before', False), ('after', True)):
        h264 = os.path.join(OUT, f'_{cut}{f0}_{tag}.h264')
        p = subprocess.Popen(DL.encode_cmd(ctx.W, ctx.H, prof, h264), stdin=subprocess.PIPE)
        for f in range(f0, f1):
            img, shot, status, src = ctx.picture(f)
            if use and src is not None and not status.startswith('SLATE'):
                img = fin(img, src, cut, f)
            img = np.ascontiguousarray(img, np.float32)
            titles.composite_v3(img, ctx.lines, f)
            p.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
        p.stdin.close(); p.wait()
        k = (f1 - f0) // 2
        raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', h264, '-vf',
                              f'select=eq(n\\,{k}),scale=in_color_matrix=bt709:in_range=tv:flags=accurate_rnd+full_chroma_int,format=rgb24',
                              '-frames:v', '1', '-f', 'rawvideo', '-'], capture_output=True).stdout
        dec[tag] = np.frombuffer(raw, np.uint8).reshape(ctx.H, ctx.W, 3)
        os.remove(h264)
    # the darkest smooth band: the 25% of rows with the lowest mean luma in the top half (the sky), full width
    lum = dec['before'].astype(np.float32).mean(axis=2)
    top = lum[:ctx.H // 2]
    rs = np.argsort(cv2.GaussianBlur(top, (0, 0), 8).mean(axis=1))[: ctx.H // 8]
    y0, y1 = int(rs.min()), int(rs.max()) + 1
    y1 = max(y1, y0 + 120)
    x0, x1 = 480, 1440
    line = [f'{cut} {f0 + (f1 - f0) // 2} {what}: sky rows {y0}-{y1}']
    tiles = []
    for tag in ('before', 'after'):
        reg = dec[tag][y0:y1, x0:x1].astype(np.float32)
        Y = reg @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        Yi = np.round(Y).astype(np.int32)
        big = 0
        for v in np.unique(Yi):
            m = (Yi == v).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
            if n > 1:
                big = max(big, int(stats[1:, cv2.CC_STAT_AREA].max()))
        levels = len(np.unique(Yi))
        smooth = cv2.GaussianBlur(Y, (0, 0), 6)
        steps = np.abs(np.diff(np.round(smooth), axis=0)).sum() / smooth.shape[1]
        line.append(f'{tag}: mean {Y.mean():.1f}, {levels} luma codes, largest flat patch {big} px '
                    f'({100 * big / Y.size:.2f}% of the region)')
        m = reg.mean()
        st = np.clip((reg - m) * 8 + 128, 0, 255).astype(np.uint8)
        cv2.putText(st, f'{cut} f{f0 + (f1 - f0) // 2} {what}: {tag.upper()} (decoded master, contrast x8)', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(st[..., ::-1])
    rows.append(np.concatenate([tiles[0], np.full((tiles[0].shape[0], 6, 3), 30, np.uint8), tiles[1]], 1))
    report.append('\n  '.join(line))
    print(report[-1], flush=True)
w = max(r.shape[1] for r in rows)
sheet = np.concatenate([np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0))) for r in rows], 0)
cv2.imwrite(os.path.join(OUT, 'banding_darkest_skies.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
open(os.path.join(OUT, 'banding_darkest_skies.txt'), 'w').write('\n'.join(report) + '\n')
