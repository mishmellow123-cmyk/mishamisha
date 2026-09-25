"""Glyph atlas: point clouds sampled from real font glyph masks (Noto + OFL fonts).

build() renders every item with PIL (libraqm shaping, so Arabic joins and
Devanagari conjuncts form), samples points on the inked area, and caches:
    pts     (M,2) float32  glyph-local coords in em units, centred, y up
    off     (G,)  int      start of each glyph's points in pts
    cnt     (G,)  int      number of points available
    size    (G,2) float32  ink width/height in em
    cat     (G,)  int      category index (see CATS)
    hero    (G,)  bool     good legible hero candidate
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont, TTCollection

N = '/usr/share/fonts/truetype/noto/'
NO = '/usr/share/fonts/opentype/noto/'
A = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'fonts') + '/'

CJK = (NO + 'NotoSerifCJK-Regular.ttc', 2)      # SC
CJK_JP = (NO + 'NotoSerifCJK-Regular.ttc', 0)
CJK_KR = (NO + 'NotoSerifCJK-Regular.ttc', 1)

# (category, font, index, items, heroes)
SETS = [
    ('latin', A + 'EBGaramond.ttf', 0,
     list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') + list('&ÆßñçøłéŒ?') +
     ['Word', 'fire', 'light', 'dream', 'mind', 'We'],
     ['A', 'R', 'g', 'Q', '&', 'Word', 'fire', 'light']),
    ('greek', N + 'NotoSerif-Regular.ttf', 0,
     list('ΑΒΓΔΘΛΞΠΣΦΨΩαβγδεζηθλμξπσφψω') + ['λόγος', 'πῦρ', 'φῶς'],
     ['Ω', 'λόγος', 'Σ', 'φῶς', 'ψ']),
    ('cyrillic', N + 'NotoSerif-Regular.ttf', 0,
     list('БГДЖЗИЛПФЦЧШЩЫЭЮЯжфыя') + ['слово', 'огонь', 'свет'],
     ['Ж', 'слово', 'Я']),
    ('arabic', N + 'NotoNaskhArabic-Regular.ttf', 0,
     list('ابتجحدرسشصطعغفقكلمنهوي') + ['نار', 'نور', 'كلمة', 'فكر', 'آتش'],
     ['نور', 'كلمة', 'ع']),
    ('hebrew', N + 'NotoSerifHebrew-Regular.ttf', 0,
     list('אבגדהוזחטיכלמנסעפצקרשת') + ['אש', 'אור', 'דבר'],
     ['אור', 'א', 'ש']),
    ('devanagari', N + 'NotoSerifDevanagari-Regular.ttf', 0,
     list('अआइकखगघचजटडतदनपबमयरलवशसहॐ') + ['अग्नि', 'ज्ञान', 'शब्द'],
     ['ज्ञान', 'ॐ', 'क']),
    ('cjk', CJK[0], CJK[1],
     list('火光字文言智心人天地水木金土日月山道知学書思夢永愛和力生明語念識星風雲龍') + ['言葉', '思考'],
     ['火', '光', '字', '道', '言葉', '永']),
    ('kana', CJK_JP[0], CJK_JP[1], list('あかさたなはまやらわアカサタナ'), ['あ']),
    ('hangul', CJK_KR[0], CJK_KR[1], list('한글불빛말꿈사람마음생각하늘') + ['생각', '마음'],
     ['한', '생각', '빛']),
    ('geez', N + 'NotoSerifEthiopic-Regular.ttf', 0,
     list('ሀለሐመሠረሰሸቀበተቸኀነኘአከወዐዘዠየደጀገጠጨጰጸፀፈፐ') + ['እሳት', 'ብርሃን'],
     ['ሀ', 'እሳት', 'ጸ']),
    ('tamil', N + 'NotoSerifTamil-Regular.ttf', 0,
     list('அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவழளறன') + ['தீ', 'அறிவு', 'ஒளி'],
     ['அ', 'அறிவு', 'ஒளி']),
    ('math', N + 'NotoSansMath-Regular.ttf', 0,
     list('∑∫∂∇∞√π∀∃∈≈≠≤≥⊕⊗∮ℵ∆∏∝∴⊂∪∩') + ['F=ma', '∫f(x)dx', 'e^iπ+1=0', '∇×B'],
     ['∑', '∫', '∞', '∇', 'π']),
    ('formula', N + 'NotoSerif-Italic.ttf', 0,
     ['E=mc²', 'a²+b²=c²', 'ℏ', 'x²', 'πr²', 'ψ'],
     ['E=mc²', 'a²+b²=c²']),
    ('music', N + 'NotoMusic-Regular.ttf', 0,
     ['\U0001D11E', '\U0001D122', '\U0001D121', '\U0001D15D', '\U0001D15E', '\U0001D15F',
      '\U0001D160', '\U0001D161', '\U0001D13D', '\U0001D12A', '\U0001D110'],
     ['\U0001D11E', '\U0001D160', '\U0001D122']),
    ('music2', N + 'NotoMusic-Regular.ttf', 0, list('♩♪♫♬♭♮♯'), ['♫', '♪']),
    ('dna', N + 'NotoSansMono-Bold.ttf', 0,
     list('ATGC') + ['ATCG', 'GATTACA', 'ACGT', 'TTAGGG', 'AUG', 'CGCG'],
     ['GATTACA', 'ATCG']),
    ('code', N + 'NotoSansMono-Regular.ttf', 0,
     ['{ }', '</>', 'if', 'for', '=>', '0x2A', 'while(1)', 'return', 'λx.x', '01101', '#include',
      'print()', 'def', 'fn()', ';', '!=', '&&', '[ ]', '//', '::', '1', '0'],
     ['{ }', '</>', 'while(1)', '01101']),
    ('cuneiform', N + 'NotoSansCuneiform-Regular.ttf', 0,
     ['\U000120AD', '\U000121A0', '\U00012097', '\U0001230B', '\U00012079', '\U00012038', '\U00012000'],
     ['\U000120AD']),
    ('hiero', N + 'NotoSansEgyptianHieroglyphs-Regular.ttf', 0,
     ['\U00013080', '\U000131A3', '\U000132F9', '\U000131F3', '\U000132B9', '\U00013153', '\U00013079',
      '\U000130ED', '\U00013193', '\U000133CF'],
     ['\U00013080', '\U000132F9', '\U00013153']),
    ('runic', N + 'NotoSansRunic-Regular.ttf', 0, list('ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟ'), ['ᚠ', 'ᛟ']),
    ('phoenician', N + 'NotoSansPhoenician-Regular.ttf', 0,
     ['\U00010900', '\U00010901', '\U00010902', '\U00010903', '\U00010904', '\U00010905', '\U0001090A'], []),
    ('linearb', N + 'NotoSansLinearB-Regular.ttf', 0,
     ['\U00010000', '\U00010001', '\U00010002', '\U00010003', '\U00010080', '\U00010081'], []),
    ('georgian', N + 'NotoSerifGeorgian-Regular.ttf', 0, list('აბგდევზთიკლმ'), ['ა']),
    ('armenian', N + 'NotoSerifArmenian-Regular.ttf', 0, list('ԱԲԳԴԵԶԷԸԹԺ'), ['Ա']),
    ('thai', N + 'NotoSerifThai-Regular.ttf', 0, list('กขคงจฉช'), []),
    ('bengali', N + 'NotoSerifBengali-Regular.ttf', 0, list('অআইকখগ'), ['অ']),
    ('tibetan', N + 'NotoSerifTibetan-Regular.ttf', 0, list('ཀཁགང'), []),
    ('cherokee', N + 'NotoSansCherokee-Regular.ttf', 0, list('ᎠᎡᎢᎣᎤ'), []),
    ('syllabics', N + 'NotoSansCanadianAboriginal-Regular.ttf', 0, list('ᐁᐃᐅᐊᐸᑎ'), []),
    ('tifinagh', N + 'NotoSansTifinagh-Regular.ttf', 0, list('ⵣⴰⴱⴳⴷ'), ['ⵣ']),
    ('khmer', N + 'NotoSerifKhmer-Regular.ttf', 0, list('កខគឃ'), []),
    ('sinhala', N + 'NotoSerifSinhala-Regular.ttf', 0, list('අආඇක'), []),
    ('gujarati', N + 'NotoSerifGujarati-Regular.ttf', 0, list('અકખગ'), []),
    ('telugu', N + 'NotoSerifTelugu-Regular.ttf', 0, list('అకఖగ'), []),
    ('myanmar', N + 'NotoSerifMyanmar-Regular.ttf', 0, list('ကခဂ'), []),
    ('yi', N + 'NotoSansYi-Regular.ttf', 0, list('ꀀꀁꀂꀃ'), []),
    ('syriac', N + 'NotoSansSyriac-Regular.ttf', 0, list('ܐܒܓܕ'), []),
]
CATS = [s[0] for s in SETS]
# relative frequency of each category among glyph instances
CAT_W = {'latin': 12, 'greek': 6, 'cyrillic': 6, 'arabic': 9, 'hebrew': 5, 'devanagari': 7,
         'cjk': 11, 'kana': 3, 'hangul': 5, 'geez': 4, 'tamil': 4, 'math': 6, 'formula': 2, 'music': 3,
         'music2': 3, 'dna': 4, 'code': 6, 'cuneiform': 1.5, 'hiero': 2, 'runic': 2,
         'phoenician': 1, 'linearb': 1, 'georgian': 1.5, 'armenian': 1.5, 'thai': 1.5,
         'bengali': 1.5, 'tibetan': 1, 'cherokee': 1, 'syllabics': 1, 'tifinagh': 1,
         'khmer': 1, 'sinhala': 1, 'gujarati': 1, 'telugu': 1, 'myanmar': 1, 'yi': 1, 'syriac': 1}

PX = 128          # render size (px per em)
KMAX_CHAR = 1000
KMAX_WORD = 2200


def _cmap(path, idx):
    if path.endswith('.ttc'):
        f = TTCollection(path).fonts[idx]
    else:
        f = TTFont(path, fontNumber=idx)
    return f.getBestCmap() or {}


def _covered(cmap, text):
    for ch in text:
        if ch in ' ':
            continue
        o = ord(ch)
        if 0x300 <= o <= 0x36F or 0x900 <= o <= 0x97F and o in (0x94D,):
            continue
        if o not in cmap:
            return False
    return True


def render_mask(text, path, idx):
    font = ImageFont.truetype(path, PX, index=idx, layout_engine=ImageFont.Layout.RAQM)
    l, t, r, b = font.getbbox(text)
    w, h = r - l + 8, b - t + 8
    img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img).text((4 - l, 4 - t), text, fill=255, font=font)
    return np.asarray(img, np.float32) / 255.0


def sample_mask(m, k, rs):
    ys, xs = np.nonzero(m > 0.3)
    if len(xs) == 0:
        return None
    w = m[ys, xs]
    k = min(k, len(xs) * 3)
    sel = rs.choice(len(xs), size=k, replace=len(xs) < k, p=w / w.sum())
    x = xs[sel] + rs.random(k) - 0.5
    y = ys[sel] + rs.random(k) - 0.5
    # centre on ink bbox
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    pts = np.stack([(x - cx) / PX, -(y - cy) / PX], 1).astype(np.float32)
    return pts, np.array([(x1 - x0 + 1) / PX, (y1 - y0 + 1) / PX], np.float32)


def build(cache):
    rs = np.random.default_rng(11)
    pts, off, cnt, size, cat, hero, text = [], [], [], [], [], [], []
    o = 0
    dropped = []
    for ci, (cname, path, idx, items, heroes) in enumerate(SETS):
        cmap = _cmap(path, idx)
        for it in items:
            if not _covered(cmap, it):
                dropped.append((cname, it))
                continue
            m = render_mask(it, path, idx)
            kmax = KMAX_WORD if len(it) > 1 else KMAX_CHAR
            res = sample_mask(m, kmax, rs)
            if res is None:
                dropped.append((cname, it))
                continue
            p, sz = res
            pts.append(p)
            off.append(o)
            cnt.append(len(p))
            size.append(sz)
            cat.append(ci)
            hero.append(it in heroes)
            text.append(it)
            o += len(p)
    np.savez_compressed(cache, pts=np.concatenate(pts), off=np.array(off), cnt=np.array(cnt),
                        size=np.array(size), cat=np.array(cat), hero=np.array(hero),
                        text=np.array(text))
    return dropped


def load(cache):
    if not os.path.exists(cache):
        d = build(cache)
        if d:
            print('glyphs: dropped (no coverage):', d)
    z = np.load(cache)
    return {k: z[k] for k in z.files}


if __name__ == '__main__':
    import sys
    c = sys.argv[1]
    d = build(c)
    z = np.load(c)
    print('glyphs', len(z['off']), 'points', len(z['pts']), 'dropped', d)
