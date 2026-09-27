"""THE LONG DAWN v3: the three EDLs on the bar grid (edit X4).

Source of truth: BIBLE_V3.md "REVISION 1 · LOCKED BEAT SHEETS" as amended by the DIRECTOR'S H5 CALLS, and
music/v3/barmap_{A,B,C}.json (section boundaries are checked against the bar map on import).

Each shot is a half-open frame range [f0, f1) on its cut's own timeline, and it lists TAKES in priority order.
A take names a render folder STEM and how frames are found:

    src = cut_frame + off
    mode 'v3'      : renders/<stem>_<CUT>/ -> renders/<stem>_v3/            (new v3 shots; the cut's own numbering)
    mode 'layered' : renders/<stem>_<CUT>/ -> _v3 -> _v2 -> renders/<stem>/  (REUSE of v2 frames in src numbering)
    mode 'exact'   : renders/<stem>/ only                                    (a department's own v3 folder name)

With --variant codedtowers every folder is first tried with the suffix _alt_codedtowers (the ALT folders hold
only the frames that differ). A take is used for a whole shot when it covers every frame; otherwise the take
covering the most frames wins and the missing frames become the SLATE. Nothing found at all -> SLATE.

To point a shot at a department's new folder: edit one T(...) line below and re-run edit/animatic.sh.
"""

BAR, BEAT = 80, 20
TOTAL = {'A': 6480, 'B': 5440, 'C': 7200}


def bf(bar, beat=1.0):
    """First frame of bar n, beat b (1-based, fractional beats allowed)."""
    return int(round((bar - 1) * BAR + (beat - 1) * BEAT))


def T(stem, off=0, mode='v3', note='', crop=None, grade=None, matte=None, under=None, video=None, need=None,
      add=None):
    """need=(a, b): the take is used only once its folder holds every src frame a..b (a shared take such as H1
    switches over in one piece, never frame by frame while a render is still landing).
    add='<folder>': an additive layer in the same src numbering (renders/<folder>/), added after the matte comp;
    a frame counts as rendered only when the add layer has it too."""
    return dict(stem=stem, off=off, mode=mode, note=note, crop=crop, grade=grade, matte=matte, under=under,
                video=video, need=need, add=add)


def S(sec, f0, f1, code, name, owner, desc, takes=(), kind='takes'):
    return dict(sec=sec, f0=f0, f1=f1, code=code, name=name, owner=owner, desc=desc, takes=list(takes), kind=kind)


# ------------------------------------------------------------------------------------------------ shared takes
# THE FIRST FIRE (H1), one master timing for every cut: strike 1 at src 1236, strike 2 1265, strike 3 1294,
# the catch 1432, the ROAR 1476; the take runs 1200-1555 (the pull-back starts about 1500).
H1_S1, H1_S3, H1_ROAR = 1236, 1294, 1476
H1_PRE5 = 'H1 v3 master timing, pre-H5 look (beanie, dish-rack basket, white sparks)'
H1_POST5 = 'H1 v3 master timing with the H5 calls (hood, shawl, thin gloves, orange sparks)'


def h1(off, **kw):
    return [T('h1_v3h5', off, 'exact', H1_POST5, need=(1200, 1555), **kw), T('h1_v3', off, 'exact', H1_PRE5, **kw)]


# B: "H1 in B cropped to hands, tinder and sparks" (B DECISION 06:40Z): x, y, w, h as fractions of the frame
# re-framed 27 Sep 15:50Z for the H5 re-key (h1_v3h5): its lifted sky turned the old box's empty basket bars into flat
# blue triangles. B4 now centres the gloved hands, steel and tinder and stays left of the scarf (x <= 0.43 while
# she leans in to blow); B5 (the roar, 12 f while the camera pulls back) sits higher so the flame stays in frame
# and the hood stays out (x <= 0.50). Square fractions keep 2.39:1.
B_H1_CROP = (0.17, 0.22, 0.26, 0.26)
B_H1_ROAR_CROP = (0.20, 0.08, 0.30, 0.30)

EMB_A = [T('embers_A3', 0, 'exact', 'EMBERS v3, A timeline')]
EMB_C = [T('embers_C3', 0, 'exact', 'EMBERS v3, C timeline'),
         T('embers_C3_half', 0, 'exact', 'EMBERS v3 half-res preview, pre-H5')]
BOOK = dict(matte='book_C_matte')


def book(note='MAP-v3 book engine', **kw):
    return T('book_C', 0, 'exact', note, **dict(BOOK, **kw))


def book_e15():
    return book('MAP-L page (page-only) + EMBERS-C E15 fire, additive', add='embers_C3_e15')


X1_TEST = T('x1_letters_C_test', -560, 'video', 'MAP-v3 X1 motion test (half-res mp4, C 560-906)',
            video='edit/cache/x1_letters_C_test.mp4')


def ring(shot):
    return [T('ring', 0, 'v3', 'MONTAGE-3D-2 Blender Ring'),
            T(f'montage3d_v3/{shot}', 0, 'exact', f'MONTAGE-3D-2 Blender Ring ({shot})')]


# ------------------------------------------------------------------------------------------------------- A
A = [
    S('A1', 0, 80, 'A1', 'BLACK', 'EDIT', 'Black. Wind rises from silence.', kind='black'),
    S('A2', 80, 560, 'R1', 'FALSE DAWN', 'RUN-A',
      'Midnight ridge: a cold white glow swells under the far horizon, silvers the cloud, puts out the nearest stars.',
      [T('falsedawn'), T('run')]),
    S('A3', 560, 960, 'E1', 'INTO THE LIGHT · GLYPHS', 'EMBERS',
      'The push goes into the glow to white; letters of every script drift in, then spiral and compress.', EMB_A),
    S('A4', 960, 1040, 'E1', 'THE POINT', 'EMBERS', 'The spiral collapses to a blinding point; a held breath.', EMB_A),
    S('A5', 1040, 1440, 'E2 · E3', 'IGNITION · THE PROMISE', 'EMBERS',
      'The thinking fire, ice-white with a gold edge, breathing too evenly; its light shows a valley of embers, then not.',
      EMB_A),
    S('A6', 1440, 1840, 'E5-A', 'THE TOWERS · TWO GIANTS', 'EMBERS',
      'Forge-towers rise, lit only on the face to the fire; from bar 22 b3.5 two giants outgrow the rest, black to each other.',
      EMB_A),
    S('A7', 1840, 2400, 'E6', 'THE EDGE', 'EMBERS',
      'On every beat the towers surge; the ground falls into a crater of fire; the gilded lean in and the rim crumbles.',
      EMB_A),
    S('A8', 2400, 2640, 'E7-A', 'THE BRINK · OVER THE RIM', 'EMBERS',
      'One roaring updraft strips the tower tops; the rim gives way; a gilded crown falls; the camera follows it into white.',
      EMB_A),
    S('A9', 2640, 2800, 'E4', 'THE DEAD VALLEY', 'EMBERS',
      "Out of the white: the promise's valley again, grey and unlit, cinder under falling ash.", EMB_A),
    S('A10', 2800, 3120, 'E9', 'BLACK · THE EMBER', 'EMBERS', 'Black. One ember drifts down and hangs, flickering. It does not die.',
      EMB_A),
    S('A11', 3120, 3360, 'X2', 'DARK ADAPTATION', 'EDIT (+ RUN-A star plate)',
      'Out of the black the stars come back: a few, then many, then the Milky Way; the ember one light among them.',
      [T('stars'), T('x2')], kind='x2'),
    S('A12', 3360, 3600, 'H1-A', 'THE FIRST FIRE', 'HEROINE',
      'Tinder, gloves, sparks, breath: three strikes, a long blow, the catch. The face never lit.', h1(H1_S1 - 3360)),
    S('A13', 3600, 3680, 'H1-A', 'THE ROAR', 'HEROINE', 'The roar on the downbeat; the pull-back from her summit begins.',
      h1(H1_ROAR - 3600)),
    S('A13', 3680, 3800, 'R2-A', 'EVERY RIDGE', 'RUN-A',
      'On every ridge to the horizon fires catch in the same breath; the cold glow pulses beyond; red under the cloud.',
      [T('reveal'), T('run')]),
    S('A13', 3800, 3860, 'M5', 'KARST', 'MONTAGE-3D-2',
      'Weathered rock towers in a mist sea; a beacon on a crown flares on bar 48 b3.75.',
      [T('montage3d_v3/karst_slow', -3800, 'exact', 'Blender KARST v3 final, 2/3 speed (H5: rock towers in mist)')]),
    S('A13', 3860, 3920, 'M5', 'DESERT', 'MONTAGE-3D-2',
      'A dune crest under the Milky Way; a robed figure; her stone beacon catches on bar 49 b3; she looks out.',
      [T('montage3d_v3/desert', -3860, 'exact', 'Blender DESERT v3 final (H5: wide, plain cloak, irregular prints)'),
       T('montage', 1520 - 3860, 'layered', 'Blender DESERT montage_v2 1520-1579, pre-H5 (framing, footprints)')]),
    S('A14', 3920, 4240, 'R3', 'THE BEACON RUN', 'RUN-A',
      "Following her look, fires link across the ranges every two beats, seven of them; the red under-glow pulses.",
      [T('beaconrun'), T('run')]),
    S('A15', 4240, 4400, 'R16', 'THE WATCHERS', 'RUN-A + HILLS',
      'Behind a backlit watcher at the seventh fire, looking to the cold glow; small figures on far ridges, eyelines only.',
      [T('watchers'), T('run')]),
    S('A16', 4400, 4720, 'E10', 'TOWERS IN THE LIGHT', 'EMBERS',
      "Far ridge fires light the towers' backs; the surges stop; the two giants open their shutters to each other first.",
      EMB_A),
    S('A17', 4720, 4880, 'E10', 'THE FIRE, SEEN', 'EMBERS',
      'The camera walks down to the calm fire; small lights come to the rim; it gathers into one small heart.', EMB_A),
    S('A18', 4880, 5840, 'R6', 'THE CROSSING', 'RUN-A',
      'One take: the great lantern on poles; forty roped bearers on a knife-edge above the cloud; the sky wheels.',
      [T('crossing'), T('run')]),
    S('A19', 5840, 6240, 'R7', 'THE BLUE HOUR', 'RUN-A',
      'The lantern set down among the watch-fires; they sit and unrope; the east pales to rose; hearth smoke below.',
      [T('bluehour'), T('run')]),
    S('A20', 6240, 6480, 'X3', 'TITLE', 'EDIT over RUN-A',
      'THE LONG DAWN kindles in the rose sky, holds, crumbles into rising sparks; fade to black from bar 81 b3.8.',
      [T('bluehour'), T('run')]),
]

# ------------------------------------------------------------------------------------------------------- B
# THE VIGIL (B DECISION 06:40Z): one night in RUN-B's summit set, same 68 bars; M2/M3 cutaways CUT.
VIGIL = [T('vigil', 0, 'v3', 'RUN-B THE VIGIL')]
B = [
    S('B1', 0, 640, 'R8', 'DUSK', 'RUN-B',
      'The range at sunset above the cloud sea; the light leaves the peaks lowest first; one last red point goes out.',
      [T('dusk'), T('run_b_tests/dusk_motion', 0, 'exact', 'quarter-res motion test, pre-H5 (triangle peak)')]),
    S('B2', 640, 880, 'H4', 'THE CLIMB', 'HEROINE + RUN-B',
      'Blue night: a tiny figure on a snow arete carries a clay fire-pot, the only warm point; then close behind her.',
      [T('climb'), T('heroine')]),
    S('B3', 880, 1120, 'H5', 'THE DEAD EMBER', 'HEROINE',
      'The summit cairn: she lifts the lid on one red eye of ember; it greys to ash under her breath. Hands only.',
      # director 27 Sep: HEROINE's job h5_deadember_b renders 960-1199 into heroine_B = B3 880-1119 (off +80);
      # later HEROINE deliveries use locked B numbering (deadember_B).
      [T('deadember'), T('heroine', 80, 'v3', 'HEROINE job numbering 960-1199 = B3 880-1119')]),
    S('B4', 1120, 1360, 'H1-B', 'THE FIRST FIRE', 'HEROINE (EDIT crop)',
      'Match cut on the last red point to the first spark: three strikes, a long blow, the catch. Hands, tinder, sparks.',
      h1(H1_S1 - 1120, crop=B_H1_CROP, grade='B')),
    S('B5', 1360, 1372, 'H1-B', 'THE ROAR', 'HEROINE (EDIT crop)', 'The roar on the downbeat, in the hands-and-tinder crop.',
      h1(H1_ROAR - 1360, crop=B_H1_ROAR_CROP, grade='B')),  # the take pulls back to A/C's wide from src ~1480
    S('B5', 1372, 1520, 'R2-B', 'THE REVEAL', 'RUN-B',
      'Moonlit silver, no red but the scarf: she is tiny on her summit above a silver cloud sea under the Milky Way.',
      [T('reveal'), T('vigil')]),
    S('B6', 1520, 2000, 'R9', 'THE VIGIL · FIRST HOURS', 'RUN-B + HILLS',
      'The locked frame: her shelf, the cairn, her small figure by the fire. Stars wheel; she feeds it; nothing answers.',
      VIGIL),
    S('B7', 2000, 2480, 'R9', 'THE VIGIL · NOTHING ANSWERS', 'RUN-B + HILLS',
      'Hours pass in the same frame: snow, then clear stars, then fog; she shields the flame. Nothing answers.', VIGIL),
    S('B8', 2480, 2640, 'R9', 'THE VIGIL · SOMEONE HAS SEEN', 'RUN-B',
      'In her frame: the night holds its breath; far off on the horizon, a pinprick of light.', VIGIL),
    S('B9', 2640, 3120, 'R9', 'THE VIGIL · THE FIRST ANSWERS', 'RUN-B + HILLS',
      'A pinprick answers from the far horizon; a traveller climbs in, takes flame, carries it down; village lights.',
      VIGIL),
    S('B10', 3120, 3280, 'R9', 'THE VIGIL · FIRE ANSWERS FIRE', 'RUN-B',
      'In her frame: her fire flares; a nearer far fire flares in answer to hers.', VIGIL),
    S('B11', 3280, 3600, 'R9', 'THE VIGIL · THE WORLD COMES', 'RUN-B + HILLS',
      'Torches thread down the NE arete; more far peaks burn; the valleys brighten under the cloud.', VIGIL),
    S('B12', 3600, 3840, 'R9', 'THE VIGIL · THE CHILD', 'RUN-B + HILLS',
      "A traveller's child stays; she feeds the beacon; the child sits against her and falls asleep.", VIGIL),
    S('B13', 3840, 5200, 'R11', 'THE HAND-BACK', 'RUN-B + HILLS',
      'One take: the crane to the range alight; the sun walks in, beacons pale, hers last; her fire-steel into the child\'s palm.',
      [T('handback')]),
    S('B14', 5200, 5440, 'X3', 'TITLE', 'EDIT over RUN-B',
      'THE LONG DAWN kindles in the dawn sky and fades into the light.', [T('handback'), T('dawntitle')]),
]

# ------------------------------------------------------------------------------------------------------- C
# H5 CALL 4: C23 THE EYE FALLS (E14) is gone; bars 70-71 are THE FIRE REMAINS (bar 70 ACCORD, bar 71 MAP).
C = [
    S('C1', 0, 80, 'C1', 'BLACK · THE HEARTH', 'EDIT', 'Black; a hearth crackles; a heavy page turns.', kind='black'),
    S('C2', 80, 320, 'P1', 'THE RED BOOK', 'MAP',
      'An old red-bound book open by a hearth: flowing unknown script, small drawings; then a sheaf of blank leaves.',
      [book()]),
    S('C3', 320, 560, 'P2', 'INK PAGE · THE MOUNTAIN', 'MAP',
      'The leaves riffle back; a pen draws a mountain with a fire in its throat and in it a small gold ring.', [book()]),
    S('C4', 560, 700, 'E15 · X1', 'LETTERS TO FIRE', 'MAP',
      'A dense leaf darkens; its letters glow, lift as sparks and pour into one point.',
      [book(), X1_TEST]),
    # director 27 Sep ~19:00Z: C 700-1039 = EMBERS-C's E15 fire added over MAP-L's page (book_C page-only there):
    # out = book_rgb + (1 - matte) * black + e15
    S('C4', 700, 880, 'E15 · X1', 'LETTERS TO FIRE · THE FIRE CATCHES', 'MAP + EMBERS',
      'At that point a fire catches and burns the page open; the burnt rim glows.', [book_e15(), X1_TEST]),
    S('C5', 880, 1040, 'E15', 'THE FIRE, ALONE', 'MAP + EMBERS',
      'The fire burns alone in the black, gold and calm, a few letters still legible in it; the burnt rim glows.',
      [book_e15(), X1_TEST]),
    S('C6', 1040, 1440, 'E5-C', 'THE FORGING', 'EMBERS',
      'Forge-towers of every realm rise round the fire, none tallest; its light is beaten into a band: the Ring.', EMB_C),
    S('C7', 1440, 1680, 'E11', 'THE RACE UNDER THE RING', 'EMBERS',
      'A hush; gold falls from the Ring into the nearest windows; then the towers surge and walls of red rise.', EMB_C),
    S('C8', 1680, 1920, 'P2', 'INK PAGE · THE DEEP', 'MAP',
      'An ember edge sweeps the race away to parchment; the pen draws pillared halls down a gilt vein; a red glow wakes.',
      [book(under=('hold', 'embers_C3', 1679))]),
    S('C9', 1920, 1992, 'E12', 'THE EYE (burn-through)', 'MAP + EMBERS',
      'The glow burns through the page into the storm, which resolves into a lidless Eye over all the towers.',
      [book(under=('same', 'embers_C3')), T('embers_C3', 0, 'exact', 'EMBERS v3, C timeline'),
       T('embers_C3_half', 0, 'exact', 'EMBERS v3 half-res preview, pre-H5')]),
    S('C9', 1992, 2080, 'E12', 'THE EYE ONTO NOTHING', 'EMBERS',
      'On bar 26 b1 its slit opens for the first time, onto empty black: no one is behind it.', EMB_C),
    S('C10', 2080, 2320, 'M4', 'THE MIRROR (optional)', 'MONTAGE',
      'A stone basin of dark water with stars in it; ripples carry the lands burning; a drop: for one breath, a golden dawn.',
      [T('mirror')]),
    S('C11', 2320, 2480, 'E8-C', 'THE GRASP THAT CANNOT HOLD', 'EMBERS',
      'A claw of embers descends and closes on the Ring; the crust glows and cracks; the band slips through and falls.',
      EMB_C),
    S('C12', 2480, 2720, 'C12', 'BLACK · THE OLD LAW', 'EDIT', 'Black. Far away, a cock crows once (bar 34 b4).',
      kind='black'),
    S('C13', 2720, 2840, 'E13', 'THE RING FALLS', 'EMBERS',
      'Out of the black a gold glint tumbles slowly, its letters faintly awake, down through cloud.', EMB_C),
    S('C13', 2840, 2960, 'R13', 'THE RING FALLS · THE STAR', 'RUN-C',
      'Over a moonlit range drawn in ink it streaks like a falling star and strikes the snow by an old cairn: steam.',
      [T('ringfall'), T('run')]),
    S('C14', 2960, 3000, 'H1-C', 'FLINT', 'HEROINE',
      'Dark, her breath, the scarf. Strike 1 on bar 38 b2: a spark in the dark.', h1(H1_S1 - 2980, grade='C')),
    S('C14', 3000, 3080, 'H2', 'THE FIND', 'MONTAGE-3D-2 (Blender)',
      "Strike 2's spark shows a gold band in a melted hollow by her knee; she stops; her gloved hand closes on it.",
      ring('find_a')),
    S('C14', 3080, 3150, 'H2', 'THE FIND · THE VISION', 'MONTAGE-3D-2 (Blender)',
      'She opens her hand: for two seconds forge-towers bow toward her on the polished band; she closes her fist.',
      ring('find_b')),
    S('C14', 3150, 3360, 'H1-C', 'FLINT · THE CATCH', 'HEROINE',
      'She strikes again (bar 40 b3.9); a long blow; the kindling catches (bar 42 b2.8). Hands only.',
      h1(H1_S3 - 3178, grade='C')),
    S('C15', 3360, 3600, 'H2', 'THE FIRE TEST', 'MONTAGE-3D-2 (Blender)',
      'The roar. The Ring on the tip of her C-shaped fire-steel in the flames, unmarked; it will not fall; she draws it out.',
      ring('fire')),
    S('C16', 3600, 3840, 'R2-C', 'THE REVEAL', 'RUN-C', 'Her fire, small on the vast range drawn in ink.',
      [T('runC_reveal', -3600, 'exact', 'RUN-C ink final')]),
    S('C17', 3840, 4160, 'R12', 'THE LIVING INK RUN', 'RUN-C',
      'A lateral track at beacon height like a scroll unrolling; seven beacons bloom like gold leaf, every two beats.',
      [T('runC_scroll', -3840, 'exact', 'RUN-C ink final')]),
    S('C18', 4160, 4480, 'P4', 'THE MAP ANSWERS · THE ROAD', 'MAP',
      'The bloom burns through onto the map: fire runs hill by hill off the sheet; a slow route reaches a ring of stones.',
      [T('map_C', 0, 'exact', 'MAP-v3 map (C timeline)')]),
    S('C19', 4480, 4800, 'AC1', 'THE COUNCIL', 'ACCORD',
      'From above, slowly orbiting: rivers of torches converge on a ring of stones; one small figure in a red scarf.',
      [T('accord')]),
    S('C20', 4800, 5120, 'AC4', 'BRING OUT THE RING', 'ACCORD',
      'She sets the Ring on the stone; the orbit; every hand holds a torch and none reaches; one gilded, scarred hand.',
      [T('accord')]),
    S('C21', 5120, 5360, 'AC2', 'THE BEARER', 'ACCORD + HEROINE',
      'The torches come down; top-down and close, her gloved hand goes back to the Ring; the fire rises round her fist.',
      [T('accord')]),
    S('C22', 5360, 5520, 'AC3 · H3', 'THE UNMAKING', 'MONTAGE-3D-2 / ACCORD',
      'In the white heart her fingers are forced open; the Ring drops, slumps, runs to a bead; its letters flare and go out.',
      ring('melt') + [T('accord')]),
    S('C23', 5520, 5600, 'AC', 'THE FIRE REMAINS · THE STONE', 'ACCORD',
      'Out of the white the fire everyone lit settles warm and steady on the stone where the Ring was; torches dip.',
      [T('accord'), T('fireremains')]),
    S('C23', 5600, 5680, 'P4 · X1', 'THE FIRE REMAINS · ROADS OF FIRE', 'MAP',
      'Burn-through to the map: from the ring of stones many roads run outward as small flames; hearth glyphs kindle.',
      [T('map_C', 0, 'exact', 'MAP-v3 map (C timeline)'), book(), T('fireremains')]),
    S('C24', 5680, 6160, 'R15', 'THE ILLUMINATION', 'RUN-C',
      "The sun breaks over the drawn world's eastern ranges; wherever its light touches, the ink fills with colour.",
      [T('runC_illum', 2398 - 5680, 'exact', 'RUN-C ink final')]),
    S('C25', 6160, 6400, 'P2', 'THE YEAR OF PLENTY', 'MAP',
      'The red book again, the hearth low: a silver-barked tree in golden flower alone in a field; hearth smoke beyond.',
      [book()]),
    S('C26', 6400, 6720, 'P2', 'THE HAVENS', 'MAP',
      'A harbour at dusk: a grey swan-prowed ship slips out; coast fires catch in farewell; her bound hand raises a light.',
      [book()]),
    S('C27', 6720, 6960, 'P1', 'THE LAST PAGES', 'MAP', 'The page turns: blank; and the next. The scorched edges have healed.',
      [book()]),
    S('C28', 6960, 7200, 'X3', 'TITLE', 'EDIT over MAP',
      'THE LONG DAWN burns onto the blank page in fire-letters and cools to ink.', [book()]),
]

EDL = {'A': A, 'B': B, 'C': C}


def check(barmap_dir):
    """Shots must tile each cut exactly, and every section boundary must match the locked bar map."""
    import json
    import os
    for cut, shots in EDL.items():
        bm = json.load(open(os.path.join(barmap_dir, f'barmap_{cut}.json')))
        assert bm['frames'] == TOTAL[cut], (cut, bm['frames'])
        f = 0
        for s in shots:
            assert s['f0'] == f and s['f1'] > s['f0'], (cut, s['code'], s['f0'], f)
            f = s['f1']
        assert f == TOTAL[cut], (cut, f)
        for sec in bm['sections']:
            mine = [s for s in shots if s['sec'] == sec['id']]
            assert mine, (cut, sec['id'])
            assert mine[0]['f0'] == sec['f0'] and mine[-1]['f1'] == sec['f1'], (cut, sec['id'])
    return True
