"""THE LONG DAWN v3: the three EDLs on the bar grid (edit X4).

Source of truth: BIBLE_V3.md "REVISION 1 · LOCKED BEAT SHEETS" as amended by the DIRECTOR'S H5 CALLS, and
music/v3/barmap_{A,B}.json and barmap_C5.json (BARMAP; section boundaries are checked by check()). C is C5: the
HANDOVER.md v5.2 timeline, readiness-checked by edit/c5_readiness.py (full and --partial).

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
# C is C5 (29 Sep, EDIT-C5): THE LAST PAGES on script v5.2, 5,920 f = 74 bars. The 7,200-frame C is retired; it
# lives in git history (edl_C.json before this change), never beside C5.
TOTAL = {'A': 6480, 'B': 5440, 'C': 5920}
BARMAP = {'A': 'barmap_A.json', 'B': 'barmap_B.json', 'C': 'barmap_C5.json'}   # C5: sections and grid only; its
# text block and several sync notes carry retired single-leader wording (titles.C5_TEXT holds C's words)


def bf(bar, beat=1.0):
    """First frame of bar n, beat b (1-based, fractional beats allowed)."""
    return int(round((bar - 1) * BAR + (beat - 1) * BEAT))


def T(stem, off=0, mode='v3', note='', crop=None, grade=None, matte=None, under=None, video=None, need=None,
      add=None, final_eligible=True):
    """need=(a, b): the take is used only once its folder holds every src frame a..b (a shared take such as H1
    switches over in one piece, never frame by frame while a render is still landing).
    add='<folder>': an additive layer in the same src numbering (renders/<folder>/), added after the matte comp;
    a frame counts as rendered only when the add layer has it too.
    final_eligible=False: an explicitly provisional take may still play in partial masters and previews,
    but cannot establish picture-source completeness. Approved reuse remains eligible."""
    return dict(stem=stem, off=off, mode=mode, note=note, crop=crop, grade=grade, matte=matte, under=under,
                video=video, need=need, add=add, final_eligible=final_eligible)


def is_final_take(take):
    """Declared source eligibility, independent of filenames, resolution, age and descriptive wording.

    Takes from older callers without the field retain their existing eligibility. This is not a creative
    approval or a check of the encoded media; absence and EDIT proxies are handled by their own status.
    """
    return take is not None and take.get('final_eligible', True)


# Explicit folder eligibility when a source is composited beneath a take. This does not infer eligibility
# from the '_half' suffix: approved reuse remains eligible unless declared here.
UNDER_FINAL_ELIGIBILITY = {'embers_C3_half': False}


def S(sec, f0, f1, code, name, owner, desc, takes=(), kind='takes'):
    return dict(sec=sec, f0=f0, f1=f1, code=code, name=name, owner=owner, desc=desc, takes=list(takes), kind=kind)


# ------------------------------------------------------------------------------------------------ shared takes
# THE FIRST FIRE (H1), one master timing for every cut: strike 1 at src 1236, strike 2 1265, strike 3 1294,
# the catch 1432, the ROAR 1476; the take runs 1200-1555 (the pull-back starts about 1500).
H1_S1, H1_S3, H1_ROAR = 1236, 1294, 1476
H1_PRE5 = 'H1 v3 master timing, pre-H5 look (beanie, dish-rack basket, white sparks)'
H1_POST5 = 'H1 v3 master timing with the H5 calls (hood, shawl, thin gloves, orange sparks)'


def h1(off, **kw):
    return [T('h1_v3h5', off, 'exact', H1_POST5, need=(1200, 1555), **kw),
            T('h1_v3', off, 'exact', H1_PRE5, final_eligible=False, **kw)]


# B: "H1 in B cropped to hands, tinder and sparks" (B DECISION 06:40Z): x, y, w, h as fractions of the frame
# re-framed 27 Sep 15:50Z for the H5 re-key (h1_v3h5): its lifted sky turned the old box's empty basket bars into flat
# blue triangles. B4 now centres the gloved hands, steel and tinder and stays left of the scarf (x <= 0.43 while
# she leans in to blow); B5 (the roar, 12 f while the camera pulls back) sits higher so the flame stays in frame
# and the hood stays out (x <= 0.50). Square fractions keep 2.39:1.
# H9 CRITIC B M4 (28 Sep): the 3.85x box read as girders and a spinner of dots and kept her hands off the edge; now
# 2.5x: glove, steel and sparks at ~0.75-0.9 of the width on the strikes (src 1236/1265/1294), the basket's rim
# reads as a basket; on the blow (src 1320-1416) her profile enters at the right edge as a dim silhouette.
B_H1_CROP = (0.12, 0.08, 0.40, 0.40)
B_H1_ROAR_CROP = (0.20, 0.08, 0.30, 0.30)

EMB_A = [T('embers_A3', 0, 'exact', 'EMBERS v3, A timeline')]
EMB_C = [T('embers_C3', 0, 'exact', 'EMBERS v3, C timeline'),
         T('embers_C3_half', 0, 'exact', 'EMBERS v3 half-res preview, pre-H5', final_eligible=False)]
BOOK = dict(matte='book_C_matte')


def book(note='MAP-v3 book engine', off=0, **kw):
    return T('book_C', off, 'exact', note, **dict(BOOK, **kw))


# BURN-C's filmed burns (review/BURN_NOTES.md; rollout ef90134, pass (a) adopted): renders/book_C_ft +
# renders/book_C_ft_matte on the farm's clean plates, for C 841-1039, 1680-1717 and 1905-1991 ONLY. Each has its own
# comp convention, and a row never reaches past its range: plan_shot takes ONE take per row, so a row wider than the
# filmed frames would silently fall back to the old burn (book_C) for the whole row. The old book_C burn frames are
# superseded there, so no ft row falls back to them.
FT = dict(matte='book_C_ft_matte')
FT_RANGES = ((841, 1040), (1680, 1718), (1905, 1992))                  # [a, b), C frames (first half: C5 = C)


def ft_letters():
    """841-1039: the page (premultiplied) + the fire; matte = cover; the under is black (none); e15 still added."""
    return T('book_C_ft', 0, 'exact', 'BURN-C filmed burn (a): letters page + fire; e15 added', add='embers_C3_e15',
             **FT)


def ft_sweep():
    """1680-1717: the held race (embers_C3 f1679) burning away over the page is BAKED and the matte is 1, so there is
    NO under: compositing the race again through it is the double-comp defect BURN_NOTES warns of."""
    return T('book_C_ft', 0, 'exact', 'BURN-C filmed sweep: the held race baked in (matte 1, no under)', **FT)


def ft_eye(under=('same', 'embers_C3'), note='BURN-C filmed Eye burn-through: page + fire, live storm under'):
    """1905-1991: the page + the fire; matte = cover; EDIT adds the live storm through it (under = same frame).
    THE GLOW (1905-1919) holds the storm's first live frame instead: c5_readiness.FT_HELD says why."""
    return T('book_C_ft', 0, 'exact', note, under=under, **FT)


def book_e15():
    return book('MAP-L page (page-only) + EMBERS-C E15 fire, additive', add='embers_C3_e15')


X1_TEST = T('x1_letters_C_test', -560, 'video', 'MAP-v3 X1 motion test (half-res mp4, C 560-906)',
            video='edit/cache/x1_letters_C_test.mp4', final_eligible=False)


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
    # A-FIX (28 Sep, approved): A's own H1 (renders/h1_A) goes live when it lands (its final waits on one more face fix)
    S('A12', 3360, 3600, 'H1-A', 'THE FIRST FIRE', 'HEROINE',
      'Tinder, gloves, sparks, breath: three strikes, a long blow, the catch. The face never lit.',
      [T('h1_A', H1_S1 - 3360, 'exact', 'A-FIX H1', need=(1236, 1487))] + h1(H1_S1 - 3360)),
    S('A13', 3600, 3612, 'H1-A', 'THE ROAR', 'HEROINE', 'The roar on the downbeat; the pull-back from her summit begins.',
      [T('h1_A', H1_ROAR - 3600, 'exact', 'A-FIX H1', need=(1236, 1487))] + h1(H1_ROAR - 3600)),
    # A-FIX hinge (28 Sep): HER FIRE 48 f of B's reveal (1372-1419) in a static 1.56x crop that keeps B's cairn out,
    # then EVERY RIDGE from 3660 (RUN-A/A-FIX re-render reveal_a_1 3660-3799: one continuous move into the crane)
    S('A13', 3612, 3660, 'R2-B', 'HER FIRE', 'RUN-B harvest',
      "Moonlit silver: she is tiny by her new fire on her summit above the cloud sea (film B's reveal, harvested).",
      [T('reveal_B', 1372 - 3612, 'exact', 'B5 reveal harvested', crop=(0.344, 0.20, 0.64, 0.64))]),
    S('A13', 3660, 3800, 'R2-A', 'EVERY RIDGE', 'RUN-A',
      'On every ridge to the horizon fires catch in the same breath; the cold glow pulses beyond; red under the cloud.',
      [T('reveal'), T('run')]),
    S('A13', 3800, 3860, 'M5', 'KARST', 'MONTAGE-3D-2',
      'Weathered rock towers in a mist sea; a beacon on a crown flares on bar 48 b3.75.',
      [T('montage3d_v3/karst_slow', -3800, 'exact', 'Blender KARST v3 final, 2/3 speed (H5: rock towers in mist)')]),
    S('A13', 3860, 3920, 'M5', 'DESERT', 'MONTAGE-3D-2',
      'A dune crest under the Milky Way; a robed figure; her stone beacon catches on bar 49 b3; she looks out.',
      [T('montage3d_v3/desert', -3860, 'exact', 'Blender DESERT v3 final (H5: wide, plain cloak, irregular prints)'),
       T('montage', 1520 - 3860, 'layered', 'Blender DESERT montage_v2 1520-1579, pre-H5 (framing, footprints)',
         final_eligible=False)]),
    S('A14', 3920, 4240, 'R3', 'THE BEACON RUN', 'RUN-A',
      "Following her look, fires link across the ranges every two beats, seven of them; the red under-glow pulses.",
      # Codex PR #8 (merged d6ef556): job beaconrun_a_catches3 -> renders/beaconrun_A_catches3. The farm landed
      # 4027-4239 only; 3920-4026 is rendered on the M4 from the same shots/run code (unchanged d6ef556 -> 6059baf ->
      # this branch). need: it plays only once all 320 frames are there (the delivered A master had A14 as a SLATE).
      [T('beaconrun_A_catches3', 0, 'exact', 'PR #8 catches (farm 4027-4239, M4 3920-4026)', need=(3920, 4239)),
       T('beaconrun'), T('run')]),
    S('A15', 4240, 4400, 'R16', 'THE WATCHERS', 'RUN-A + HILLS',
      'Behind a backlit watcher at the seventh fire, looking to the cold glow; small figures on far ridges, eyelines only.',
      # PR #8's companion job watchers_a_catches3 (160/160) carries A14 catches3's fire sizes across the cut: the
      # right-hand ridge fire measures 117 px at A14 4239, 128 px at catches3 4240, but 52 px at hearth3 4240 (the
      # contraction PR #8 warned of; warm-pixel blobs, 29 Sep). Codex PR #7's restaged hearth (hearth3) stays next.
      [T('watchers_A_catches3', 0, 'exact', 'PR #8 companion: A14 catches3 fire sizes', need=(4240, 4399)),
       T('watchers_A_hearth3', 0, 'exact', 'Codex restaged hearth (PR #7)'), T('watchers'), T('run')]),
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
      [T('dawnrev'), T('bluehour'), T('run')]),       # A-FIX ENDING (approved 28 Sep): B's dusk reversed into a dawn
    S('A20', 6240, 6480, 'X3', 'TITLE', 'EDIT over RUN-A',
      'THE LONG DAWN kindles in the rose sky over the valley, holds, crumbles into rising sparks as the sky pales.',
      [T('dawnrev'), T('bluehour'), T('run')], kind='title'),                  # ember title: edit/ember_title_v3.py
]

# ------------------------------------------------------------------------------------------------------- B
# THE VIGIL (B DECISION 06:40Z): one night in RUN-B's summit set, same 68 bars; M2/M3 cutaways CUT.
VIGIL = [T('vigil', 0, 'v3', 'RUN-B THE VIGIL')]
B = [
    S('B1', 0, 640, 'R8', 'DUSK', 'RUN-B',
      'The range at sunset above the cloud sea; the light leaves the peaks lowest first; one last red point goes out.',
      [T('dusk'), T('run_b_tests/dusk_motion', 0, 'exact', 'quarter-res motion test, pre-H5 (triangle peak)',
                    final_eligible=False)]),
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
      'THE LONG DAWN kindles in the dawn sky and fades into the light.', [T('handback'), T('dawntitle')],
      kind='title'),
]

# ------------------------------------------------------------------------------------------------------- C
# C5 (script v5.2; HANDOVER.md "C v5 timeline"). Frames 0-2079 are the first half as it was, split only where BURN-C's
# filmed burns take over (FT_RANGES). From 2080 every shot is new or re-timed; the nine Codex deliveries (PRs 11-14)
# are read in ABSOLUTE C5 frames (off 0), by their exact folder names (mode 'exact': no other name is ever tried).
# Reused sources keep their own numbering: src = C5 frame + off (runC_scroll -3120, runC_illum 2398-4720, book_C
# +960 for Plenty and +1280 for the title), the same source frames the 7,200-frame cut played there.
# Pages' mattes (book_C5_*_matte) are delivered opaque: 255 in every pixel of all 720 frames (240 each, decoded by
# EDIT-C5 on 29 Sep; REVIEW-C's count agrees). They are declared so the pair travels together, and with no under-layer
# they composite nothing. They carry no burn or page-turn hole either, so no transition may take its hole from them
# (edit/c5_readiness.py FAILS a window that names a shot's matte): the page turn into the Pen is EDIT's own comp, and
# the Refusal's burn needs its own layers.
PR11, PR12, PR13, PR14 = ('Codex PR11 (embers, 9fa0822)', 'Codex PR12 (ink pages, ebcdd0a)',
                          'Codex PR13 (last beacon, 2bdfb09)', 'Codex PR14 (run, 7412f2a)')


def c5(stem, note, **kw):
    """A delivered C5 shot: its own folder, absolute C5 numbering, no fallback."""
    return T(stem, 0, 'exact', note, **kw)


# ---------------------------------------------------------------------------------- C12 FLINT: a DECISION
# The adopted hands-only flint (HANDOVER "existing `ring` stem (C14 hands-only frames)") is MONTAGE-3D-5's, delivered
# into renders/ring_C in the numbering of the OLD cut: flint_a 2960-2999 (strike 1 at 2980) and flint_b 3150-3359
# (strike 3 at 3178, the ember takes 3181, the long blow 3206-3312, the catch 3316, the kindling 3337;
# shots/montage3d/ringc.py FLINT; jobs ringC_flint_a_1, ringC_flint_b_1..7; 3300-3359 re-rendered with the polished
# glove, seam 3299|3300 checked by MONTAGE-3D-5). Between them ring_C holds 3000-3149, the RETIRED find and vision (the
# Ring in her palm): no selection may read it, so no uniform offset can work (+320 or +480 both land in it).
# 250 frames for a 240-frame slot, and nobody on the EDIT-C5 Mac has seen them (no still of either take exists
# locally: the mailbox files, the repo's tracked review images and the handoff clips were searched on 29 Sep). The
# slot therefore stays a DECISION: FLINT_CHOICE is None until someone picks a candidate (or writes a better one) from
# the frames; until then C12 is a slate and edit/c5_readiness.py FAILS naming it.
FLINT = dict(slot=(2640, 2880), stem='ring_C', takes=dict(flint_a=(2960, 3000), flint_b=(3150, 3360)),
             retired=(3000, 3150),
             events=dict(strike1=2980, strike3=3178, ember=3181, blow_start=3206, blow_end=3312, catch=3316,
                         kindle=3337, glove_polish_from=3300))
# pieces: (C5 f0, C5 f1, source f0 in ring_C | None = EDIT black), half-open, tiling the slot
FLINT_CANDIDATES = {
    'A': dict(pieces=((2640, 2670, 2970), (2670, 2880, 3150)),
              note="flint_a from 2970 (10 f of the opening exhale dropped) + ALL of flint_b: both strikes, every "
                   "blow, the catch 44 f before the Reveal's ignition (the old H1 catch-to-roar spacing). Look at: "
                   "the a|b join 2669|2670 (src 2999|3150: a hard cut inside one framing; breath phase, smoke and hand "
                   "drift jump), and whether 2960-2969 held the shot's own fade-up."),
    'B': dict(pieces=((2640, 2680, 2960), (2680, 2880, 3160)),
              note="ALL of flint_a + flint_b from 3160 (10 f of the tension breath 3149-3167 dropped). Look at: the "
                   "a|b join 2679|2680 (src 2999|3160, mid-breath), and whether strike 3 still has its wind-up."),
    'C': dict(pieces=((2640, 2680, 2960), (2680, 2880, 3150)),
              note="ALL of flint_a + flint_b to 3349 (its last 10 f dropped, as the camera eases back and the flame "
                   "rises). The catch lands 34 f before the ignition cut. Look at: the a|b join 2679|2680 and the "
                   "cut at 2879 (src 3349) into the Reveal."),
    'D': dict(pieces=((2640, 2670, None), (2670, 2880, 3150)),
              note="30 f of EDIT black (the bar map's 'black and silence; the click of a flint') + ALL of flint_b; no "
                   "strike 1, no a|b join. Look at: whether one strike reads as a first attempt, and whether the Trap's "
                   "roar cut to black is wanted."),
}
FLINT_CHOICE = 'A'       # chosen 29 Sep from the frames (renders/ring_C): both strikes, no fade to lose at 2960-2969
                         # (luma flat 27.7-27.9), and the a|b join 2999|3150 changes less (0.0044) than the take's own
                         # wind-up onset 2969|2970 (0.0067); B, C and D stay above as one-constant alternatives.
FLINT_DESC = ('Hands only in the dark: the flint struck, sparks into the tinder, a long blow, the kindling catches. '
              'SOURCE SELECTION UNDECIDED (edl_v3.FLINT_CANDIDATES).')


def flint_events(key):
    """{event: C5 frame} for a candidate (None where the event's source frame is not selected)."""
    out = {}
    for ev, sf in FLINT['events'].items():
        if ev.startswith('blow') or ev == 'glove_polish_from':
            continue
        out[ev] = next((a + sf - s0 for a, b, s0 in FLINT_CANDIDATES[key]['pieces']
                        if s0 is not None and s0 <= sf < s0 + (b - a)), None)
    return out


# ------------------------------------------------------------------------ C15 | C16: the one movable picture cut
# The picture cut from the last beacon to the forges is ONE number. As briefed it sits on the bar map's section line
# (3840). A queued picture candidate (Codex's, 29 Sep, not adopted) would move it EARLIER, into the map's all-lit hold,
# and give the forges a lit lead-in; the shutdown stays at 3848 because Cold is read at its own C5 frame numbers.
# Measured 29 Sep from the delivered frames: the last kingdom catches 3785-3791 (the region's luminance 78 -> 118), so
# 3792-3839 is all lit; embers_C5_cold goes dark between 3847 and 3848 (mean luminance 24.0 -> 18.1). Setting
# COLD_CUT anywhere in 3792-3840 is the whole edit: the map ends there and Cold plays from there as a second C15 row
# (the bar map's section line does not move). The gates then name what must exist: the lead-in frames (embers_C5_cold
# COLD_CUT-3839, outside PR11's 3840-3999; edit/c5_readiness.py GAPs them, tools/c5_assets.py reports them OWED and
# records the new delivery's range in its DELIVERED once it lands) and any caption left running across the new cut
# (R15 3740-3835 would be: the gate WARNs by name). A lead-in delivered under another folder name changes the stem
# below as well.
COLD_CUT = 3840


def last_beacon_rows(cold_cut):
    """Section C15 (3440-3840 in the bar map): the map up to cold_cut, then, if earlier than 3840, Cold's lit lead-in."""
    rows = [S('C15', 3440, cold_cut, '#15', 'THE LAST BEACON', 'MAP',
              'The map of the kingdoms: beacons flare in no order; one kingdom stays dark (3724-3783), then catches.',
              [c5('map_last_beacon_C', PR13)])]
    if cold_cut < 3840:
        rows.append(S('C15', cold_cut, 3840, '#16', 'THE FORGES GO COLD · LIT LEAD-IN', 'EMBERS',
                      f'Every forge still burning, {3848 - cold_cut} frames before 3848 puts them all out.',
                      [c5('embers_C5_cold', f'{PR11}; the lead-in {cold_cut}-3839 is not in that delivery')]))
    return rows


def flint_rows(choice):
    if choice is None:
        return [S('C12', 2640, 2880, '#12', 'FLINT', 'MONTAGE-3D-5 (renders/ring_C)', FLINT_DESC, kind='decision')]
    rows = []
    for a, b, s0 in FLINT_CANDIDATES[choice]['pieces']:
        if s0 is None:
            rows.append(S('C12', a, b, '#12', 'FLINT · BLACK', 'EDIT', 'Black and silence; the click of a flint.',
                          kind='black'))
        else:
            rows.append(S('C12', a, b, '#12', 'FLINT', 'MONTAGE-3D-5 (renders/ring_C)',
                          'Hands only in the dark: the flint struck, sparks, the long blow, the catch.',
                          [T(FLINT['stem'], s0 - a, 'exact', f'MONTAGE-3D-5 hands-only flint, candidate {choice}')]))
    return rows


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
    # out = book_rgb + (1 - matte) * black + e15; from 841 the filmed burn replaces the page (BURN-C, ft_letters)
    S('C4', 700, 841, 'E15 · X1', 'LETTERS TO FIRE · THE FIRE CATCHES', 'MAP + EMBERS',
      'At that point a fire catches and burns the page open; the burnt rim glows.', [book_e15(), X1_TEST]),
    S('C4', 841, 880, 'E15 · X1', 'LETTERS TO FIRE · THE BURN', 'BURN-C + EMBERS',
      'A slit tears open in the page (842-870) and the flames flare along it; the page lingers.',
      [ft_letters(), X1_TEST]),
    S('C5', 880, 1040, 'E15', 'THE FIRE, ALONE', 'BURN-C + EMBERS',
      'The fire burns alone in the black, gold and calm, a few letters still legible in it; the burnt rim glows.',
      [ft_letters(), X1_TEST]),
    S('C6', 1040, 1440, 'E5-C', 'THE FORGING', 'EMBERS',
      'Forge-towers of every realm rise round the fire, none tallest; its light is beaten into a band: the Ring.', EMB_C),
    S('C7', 1440, 1680, 'E11', 'THE RACE UNDER THE RING', 'EMBERS',
      'A hush; gold falls from the Ring into the nearest windows; then the towers surge and walls of red rise.', EMB_C),
    S('C8', 1680, 1718, 'P2', 'INK PAGE · THE DEEP · THE SWEEP', 'BURN-C',
      'A filmed ember edge sweeps the held race away to parchment (the race is baked into the burn).', [ft_sweep()]),
    S('C8', 1718, 1905, 'P2', 'INK PAGE · THE DEEP', 'MAP',
      'The pen draws pillared halls down a gilt vein; a red glow wakes.',
      [book(under=('hold', 'embers_C3', 1679))]),
    S('C8', 1905, 1920, 'P2', 'INK PAGE · THE DEEP · THE GLOW', 'BURN-C',
      'The red glow at the bottom of the Deep catches: the page begins to burn.',
      # the matte is opaque in all 15 frames and embers_C3 1680-1919 was never rendered: hold 1920 (FT_HELD)
      [ft_eye(('hold', 'embers_C3', 1920), 'BURN-C filmed Eye burn-through: page + fire; the storm held at 1920 under '
              'an opaque matte')]),
    S('C9', 1920, 1992, 'E12', 'THE EYE (burn-through)', 'BURN-C + EMBERS',
      'The glow burns through the page into the storm, which resolves into a lidless Eye over all the towers.',
      [ft_eye()]),
    S('C9', 1992, 2080, 'E12', 'THE EYE ONTO NOTHING', 'EMBERS',
      'On bar 26 b1 its slit opens for the first time, onto empty black: no one is behind it.', EMB_C),
    S('C10', 2080, 2320, '#10', 'THE REFUSAL', 'PAGES',
      'Back at the book: a gloved palm offers the ring; a hooded figure turns away, one hand raised.',
      [c5('book_C5_refusal', PR12, matte='book_C5_refusal_matte')]),
    S('C11', 2320, 2640, '#11', 'THE TRAP', 'EMBERS',
      'One forge sinks, the others surge, it flares back and races; two matching towers pull ahead, neck and neck.',
      [c5('embers_C5_trap', PR11)]),
] + flint_rows(FLINT_CHOICE) + [
    S('C13', 2880, 3120, '#13', 'THE REVEAL · A PROMISE', 'RUN',
      'The vast ink range: her small fire and, at the same moment, a second fire on a far peak.',
      [c5('runC_reveal_pair_v5', PR14)]),
    # the 7,200-frame cut's C17 (3840-4159 = runC_scroll 0-319), the same 320 source frames
    S('C14', 3120, 3440, '#14', 'THE BEACON RUN', 'RUN-C',
      'A lateral track at beacon height like a scroll unrolling; beacons bloom along the peaks, every two beats.',
      [T('runC_scroll', -3120, 'exact', 'RUN-C ink final')]),
] + last_beacon_rows(COLD_CUT) + [
    S('C16', 3840, 4000, '#16', 'THE FORGES GO COLD', 'EMBERS',
      'Every forge goes dark at the same instant (3848); smoke over cold masonry; the Ring still gold.',
      [c5('embers_C5_cold', PR11)]),
    S('C17', 4000, 4240, '#17', 'THE RING, UNFINISHED', 'EMBERS',
      'The Ring hangs over the dark towers; its glow drains to grey; the storm thins; a held final frame (4217-4239).',
      [c5('embers_C5_unfinished', PR11)]),
    S('C18', 4240, 4480, '#18', 'THE DEEP, ABANDONED', 'PAGES',
      'The mine page again, still: empty ladders, a lantern set down, the gold vein still glinting.',
      [c5('book_C5_deep_abandoned', PR12, matte='book_C5_deep_abandoned_matte')]),
    S('C19', 4480, 4720, '#19', 'THE WATCH', 'RUN',
      'The ink range with a beacon burning on every peak as the camera drifts (the run\'s poses 80-319, all lit).',
      [c5('runC_watch_v5', PR14)]),
    # the 7,200-frame cut's C24 (5680-6159 = runC_illum 2398-2877), the same 480 source frames
    S('C20', 4720, 5200, '#20', 'THE ILLUMINATION', 'RUN-C',
      "The sun breaks over the drawn world's eastern ranges; wherever its light touches, the ink fills with colour.",
      [T('runC_illum', 2398 - 4720, 'exact', 'RUN-C ink final')]),
    # the 7,200-frame cut's C25 (6160-6399): it opens pixel for pixel on runC_illum 2877, so the join holds
    S('C21', 5200, 5440, '#21', 'THE YEAR OF PLENTY', 'MAP',
      'The red book again, the hearth low: a silver-barked tree in golden flower alone in a field; hearth smoke beyond.',
      [book(off=960)]),
    S('C22', 5440, 5680, '#22', 'THE LAST PAGES · THE PEN', 'PAGES',
      'The blank spread with a wooden dip pen resting across it; the camera and the hearth light move slowly.',
      [c5('book_C5_pen', PR12, matte='book_C5_pen_matte')]),
    # the 7,200-frame cut's C28 (6960-7199): THE LONG DAWN burns on in book space (book_C 6980-7160 = C5 5700-5880)
    S('C23', 5680, 5920, 'X3', 'TITLE', 'MAP',
      'The blank recto: THE LONG DAWN burns on in fire-letters and cools to ink; the book goes back into the dark.',
      [book(off=1280)]),
]

EDL = {'A': A, 'B': B, 'C': C}

# EDIT transitions (assemble._transitions; the comp, not a cut). A's are A-FIX's; C5's are EDIT-C5's (29 Sep), designed
# around the adopted shots (the 7,200-frame cut's C windows are retired with it).
# Across an EDL boundary 'cut' the outgoing shot plays its own frames up to it, then holds its last; the incoming holds
# its first frame until it, then plays. A window whose layer frames are missing, or with a slate on either side, plays
# as the plain cut. ready=False holds a window back (and edit/c5_readiness.py FAILS a full build on it).
#   burn         PAGES' burn-through (x1burn.py v2): out = O * keep + I * (1 - cover) + glow; keep is COLOUR (the scorch
#                tint and the char band), cover = 1 - hole, glow is display sRGB
#   x1           MAP-L2's first formula (fallback): out = O * keep + I * (1 - keep) + glow (shows the map through the char)
#   dissolve     linear light, smoothstep over the window
#   finish_ramp  one shot, no cut: the finish goes from the ink look to the film look, lerp(ink, film, smoothstep)
#   page_turn    EDIT-C5: the outgoing page curls over right to left onto the incoming (assemble.page_turn)
# Kept as hard cuts on purpose: 2080 (the Eye -> the Refusal: the hearth flare that brings us back to the book is in
# the render), 2640 (the Trap's roar -> the dark of the flint), 2880 (the catch -> the Reveal's simultaneous ignition,
# which must land on the cut), COLD_CUT (3840 as briefed: the last beacon -> the forges, lit for 8 frames before 3848
# puts them all out; movable, see COLD_CUT), 4000 (Cold -> Unfinished: one renderer, one camera, continuous).
TRANS = {'A': [], 'B': [], 'C': [
    # the ember README (PR11) leaves "the page burn" to EDIT: the book's grammar for page -> ember world is the
    # burn-through (C4-C5, C8-C9). DESIGNED, NOT BUILT: it needs burn layers for the Refusal page (ftburn/x1burn,
    # BURN/PAGES), because the page's delivered matte is opaque on all 240 frames and holds no hole; the three folder
    # names are PROPOSED, nothing has been rendered. Until then: a hard cut.
    dict(f0=2310, f1=2346, cut=2320, kind='burn', glow='x1_refusal_C5', keep='x1_refusal_C5_matte',
         cover='x1_refusal_C5_cover', ready=False,
         note='C10 the Refusal page burns through onto C11 the Trap (DESIGNED; layers PROPOSED, not rendered)'),
    # the same source pair as the 7,200-frame cut's #16 (runC_reveal pose 239 -> runC_scroll 0): two parchment views
    # of the range read as a jump cut; runC_scroll is not on the EDIT-C5 Mac, so this is unseen here
    dict(f0=3117, f1=3123, cut=3120, kind='dissolve',
         note='C13 the Reveal -> C14 the beacon run: two parchment views of the range (old #16, same sources)'),
    # the 7,200-frame cut's #17 (the seventh beacon burns through onto the map) on the NEW map: its x1_map_C layers
    # were aligned to the retired map_C, so it needs new ones. DESIGNED, NOT BUILT; folder names PROPOSED.
    dict(f0=3430, f1=3466, cut=3440, kind='burn', glow='x1_map_C5', keep='x1_map_C5_matte', cover='x1_map_C5_cover',
         ready=False, note='C14 the seventh beacon burns through onto C15 the map (DESIGNED; layers PROPOSED)'),
    # PR11 leaves "the page dissolve" to EDIT: the grey Ring over the dark towers into the abandoned mine; centred on
    # the downbeat, it keeps 4217-4227 of the intended final hold clean and fades through the rest
    dict(f0=4228, f1=4252, cut=4240, kind='dissolve',
         note='C17 the Ring, unfinished -> C18 the Deep, abandoned: the page dissolve PR11 leaves to EDIT'),
    dict(f0=4468, f1=4492, cut=4480, kind='dissolve',
         note='C18 the mine page -> C19 the watch: ink drawing into ink range, through the paper'),
    # runC_watch_v5 ends on the run's pose 319; runC_illum 2398 is another view: unseen here (illum not on this Mac)
    dict(f0=4712, f1=4728, cut=4720, kind='dissolve',
         note='C19 the watch -> C20 the illumination: two ink views of the range (UNSEEN: runC_illum not here)'),
    dict(f0=5200, f1=5264, kind='finish_ramp',
         note="C21 opens pixel for pixel on C20's last frame (runC_illum 2877, ink look) and takes the film look by "
              "5264 (the 7,200-frame cut's #21, 6160-6224, same sources)"),
    # PR12: the Pen "starts on the blank spread, so EDIT must join it to the preceding page turn"; centred on bar 69 b1,
    # done before the score's voice-line window opens at 5460. The curl is EDIT's own comp (the Pen's delivered matte
    # is opaque, with no turn in it). UNRENDERED: Plenty (book_C) is not on the EDIT-C5 Mac.
    dict(f0=5430, f1=5452, cut=5440, kind='page_turn', tilt=8.0, radius=0.11,
         note='C21 Plenty -> C22 the Pen: the page turns onto the blank spread (EDIT 2D curl; UNRENDERED here)'),
    dict(f0=5668, f1=5692, cut=5680, kind='dissolve',
         note='C22 the pen insert -> C23 the blank recto the title burns onto (UNSEEN: book_C not here)'),
]}

# A (28 Sep, approved): A-FIX's six windows (edit/afix_comp.py: bloom, dissolve, vision x2, iceheart, ember) + EDIT's
# grade-match of the harvested B reveal (B's sky is a touch darker and cooler than reveal_A's at the 3680 cut)
import afix_comp  # noqa: E402
TRANS['A'] = list(afix_comp.A_TRANS) + [afix_comp.WATCHFIRES] + [      # + A-FIX's ENDING (approved 28 Sep)
    dict(f0=1026, f1=1048, cut=1040, kind='swell', p0=(966, 238), c1=(964, 238), disc=(962, 235), flame=(959, 245),
         s0=0.2, s1=0.4, blend=(1038, 1042), R=330,
         note='A4 -> A5 (user 28 Sep: the point jumped to the ignition disc in one frame): the point swells, '
              'catches and becomes the flame'),
    dict(f0=3612, f1=3660, kind='grade', gain=(1.18, 0.99, 0.94),
         note="HER FIRE (reveal_B, cropped) matched to reveal_A's sky: top-third means (.101 .142 .263) -> (.119 .140 "
              ".246) (reveal_A 3680; re-check on 3660 when reveal_a_1 lands)"),
]


def barmap_path(cut, barmap_dir):
    import os
    return os.path.join(barmap_dir, BARMAP[cut])


def check(barmap_dir):
    """Shots must tile each cut exactly, and every section boundary must match its bar map (C: barmap_C5.json)."""
    import json
    for cut, shots in EDL.items():
        with open(barmap_path(cut, barmap_dir)) as fh:
            bm = json.load(fh)
        assert bm['frames'] == TOTAL[cut], (cut, bm['frames'])
        if 'bars' in bm:
            assert bm['bars'] * BAR == TOTAL[cut], (cut, bm['bars'])
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


def c5_export_extra():
    """What edl_C.json carries beyond the shared EDL format: the C5 contract a department or a reviewer needs."""
    import titles
    return dict(version='C5', script='HANDOVER.md "Script v5.2" (+ paired-rivals amendment); captions PROVISIONAL',
                barmap='music/v3/' + BARMAP['C'], bars=TOTAL['C'] // BAR,
                filmed_burns=[list(r) for r in FT_RANGES],
                decisions=[dict(id='C12_FLINT', frames=list(FLINT['slot']), status='decision_required'
                                if FLINT_CHOICE is None else f'chosen:{FLINT_CHOICE}', choice=FLINT_CHOICE,
                                source=FLINT['stem'], takes={k: list(v) for k, v in FLINT['takes'].items()},
                                retired=list(FLINT['retired']), events=FLINT['events'],
                                candidates={k: dict(pieces=[list(p) for p in v['pieces']], note=v['note'],
                                                    events=flint_events(k))
                                            for k, v in FLINT_CANDIDATES.items()})],
                movable_cuts=[dict(id='COLD_CUT', frame=COLD_CUT, section_line=3840, shutdown=3848,
                                   note="the last beacon -> the forges; may move earlier into the map's all-lit hold "
                                        '(3792-3839) for a lit lead-in while the shutdown stays at 3848. The edit '
                                        'changes edl_v3.COLD_CUT alone; embers_C5_cold must then hold the lead-in '
                                        'frames')],
                baked_text=[dict(b, src=list(b['src'])) for b in titles.BAKED_TEXT])
