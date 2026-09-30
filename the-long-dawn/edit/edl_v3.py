"""THE LONG DAWN v3: the EDLs on the bar grid (edit X4), including the consolidated cut D.

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
from copy import deepcopy

BAR, BEAT = 80, 20
# C is C5 (29 Sep, EDIT-C5): THE LAST PAGES on script v5.2, 5,920 f = 74 bars. The 7,200-frame C is retired; it
# lives in git history (edl_C.json before this change), never beside C5.
TOTAL = {'A': 6480, 'B': 5440, 'C': 5920, 'D': 9200}
BARMAP = {'A': 'barmap_A.json', 'B': 'barmap_B.json', 'C': 'barmap_C5.json', 'D': 'barmap_D.json'}
# C5: sections and grid only; its
# text block and several sync notes carry retired single-leader wording (titles.C5_TEXT holds C's words)


def bf(bar, beat=1.0):
    """First frame of bar n, beat b (1-based, fractional beats allowed)."""
    return int(round((bar - 1) * BAR + (beat - 1) * BEAT))


def T(stem, off=0, mode='v3', note='', crop=None, grade=None, matte=None, under=None, video=None, need=None,
      add=None, final_eligible=True, hold=None, clamp=None, screen_transform=None, baked_text=(), linear_mix=None):
    """need=(a, b): the take is used only once its folder holds every src frame a..b (a shared take such as H1
    switches over in one piece, never frame by frame while a render is still landing).
    add='<folder>': an additive layer in the same src numbering (renders/<folder>/), added after the matte comp;
    a frame counts as rendered only when the add layer has it too.
    under=('same', stem) reads the cut frame, independent of the take's off; ('hold', stem, frame) holds one frame.
    For a harvested composite, under=('offset', stem, offset) reads cut frame + offset (set explicitly for that
    under-layer; matching the foreground's off is appropriate only when both sources share frame numbering).
    hold=<integer source frame>: repeat that frame for RGB, matte and add; omit it for the legacy offset clock.
    clamp=(first,last): clamp the offset source clock to this inclusive range; mutually exclusive with hold.
    screen_transform=<dict>: explicit screen camera applied after the take's composition and grade.
    under=('take', take, frame, cut): hold an explicit direct plate take, including its source clock/transform.
    linear_mix=<stem>: independent filmed-sweep coefficients, with this required explicit under take.
    baked_text=('D14',): explicitly declares captions written into this adopted plate. D's overlay changes
    to in_picture only when the complete take is selected; a missing or partial delivery keeps the overlay.
    final_eligible=False: an explicitly provisional take may still play in partial masters and previews,
    but cannot establish picture-source completeness. Approved reuse remains eligible."""
    take = dict(stem=stem, off=off, mode=mode, note=note, crop=crop, grade=grade, matte=matte, under=under,
                video=video, need=need, add=add, final_eligible=final_eligible)
    if hold is not None:
        if type(hold) is not int or hold < 0:
            raise ValueError('hold must be a nonnegative integer source frame')
        take['hold'] = hold
    if clamp is not None:
        if hold is not None or len(clamp) != 2 or any(type(n) is not int or n < 0 for n in clamp) or clamp[0] > clamp[1]:
            raise ValueError('clamp needs an ordered nonnegative integer pair and no hold')
        take['clamp'] = tuple(clamp)
    if screen_transform is not None:
        take['screen_transform'] = deepcopy(screen_transform)
    if baked_text:
        if isinstance(baked_text, str) or any(not isinstance(i, str) or not i for i in baked_text):
            raise ValueError('baked_text must be a sequence of caption ids')
        take['baked_text'] = tuple(baked_text)
    if linear_mix is not None:
        if not under or under[0] != 'take' or any(v is not None for v in (crop,grade,matte,add,screen_transform)):
            raise ValueError('linear_mix requires an explicit under take and no second crop/grade/layer/transform')
        take['linear_mix'] = linear_mix
    return take


def source_frame(take, frame):
    """Source numbering: explicit hold, or offset clock with an optional declared end-frame clamp."""
    source = take.get('hold', frame + take['off'])
    if 'clamp' in take:
        source = min(take['clamp'][1], max(take['clamp'][0], source))
    return source


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


def ft_sweep_entry():
    """1680-1687: the same filmed sweep eased in over the held race (Codex lane sweepC, soft-entry): the delivered
    burn image blended from embers_C3 f1679 in linear light, a cubic opacity 0 -> 1 over 1680-1688, rendered 29 Sep
    from the delivered frames (renders/cand_sweep_soft-entry; inputs pinned by assets/burn_footage/sweep_entry/
    source.json and kept out of git). Same convention as ft_sweep(): race baked in, matte 1, no under. The 1679|1680
    step drops from RGB MAD 13.33 to 0.38; no step over 1680-1688 exceeds 3.04 (the embers' own is ~3.05)."""
    return T('cand_sweep_soft-entry', 0, 'exact', 'sweepC soft entry: the filmed sweep eased in over the held race '
             '(matte 1, no under)', matte='cand_sweep_soft-entry_matte')


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
      # clear_high_deck (Codex lane sky; plays once all 480 frames land): the dark horizontal dashes across the Milky
      # Way in A's first image (the naive viewer read them as a digital smear) are falsedawn.py's high 'mackerel' deck
      # compressing into streaks towards the horizon (3x supersampling leaves them: A212 99th-percentile vertical step
      # 12.36 at 1.5x, 13.43 at 3x). The candidate drops that deck only: A212 source 99th-percentile step 12.36 -> 1.00,
      # Milky Way, stars, glow, camera and terrain unchanged (0 of 151,138 protected terrain pixels changed); the deck's
      # late silvering is lost. ALTERNATIVES['A2'] is the accepted sky
      [T('cand_falsedawn_clear_high_deck', 0, 'exact', 'clear_high_deck (Codex lane sky, farm 29 Sep)', need=(80, 559)),
       T('falsedawn'), T('run')]),
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
      # owner (30 Sep; "jumpy and unpolished"): the crater view 1920-2399 from the eased-tower re-render (LD_SURGE_EASE
      # + LD_TOWER_PULSE_EASE, farm 0930-102643, renders/embers_A3_pulse): per-beat peak one-frame MAD 12.38 -> 9.28,
      # max 15.67 -> 11.20, better on 23/23 beats. THE EDGE and the gilding insert 1840-1919 stay the accepted
      # embers_A3 frames (APFS clones): in that wide view the eased flashes line up (peak 10.22 -> 12.46 at 1844).
      # ALTERNATIVES['A7 plates'] lists the other folders.
      [T('embers_A3_crater_pulse', 0, 'exact', 'EMBERS v3 A7: accepted 1840-1919, eased towers 1920-2399',
         need=(1840, 2399))] + EMB_A),
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
      # ADOPTED 29 Sep (plays once all 320 frames land): linked-fires (Codex lane beaconrun; farm from fb685a1). The
      # first link catches at 3936 where A13's desert beacon sat (upper centre-right), the rest on the score's run_1..7
      # (4000-4200), every link drawn as a fire: on the farm test frames three fires burn at 4040 and four along the
      # ridge at 4200, where catches3 showed 1-2 px specks until 4148. ALTERNATIVES['A14'] is catches3
      [T('cand_beaconrun_linked-fires', 0, 'exact', 'linked-fires (Codex lane beaconrun, farm 29 Sep)', need=(3920, 4239)),
       T('beaconrun_A_catches3', 0, 'exact', 'PR #8 catches (farm: 4027-4239, then 3920-4026 on 29 Sep; the joins 3955|3956, 3990|3991 and 4026|4027 differ from their neighbours by 0.93/0.92/1.00 vs 0.92/0.92/1.00 mean |diff|)', need=(3920, 4239)),
       T('beaconrun'), T('run')]),
    S('A15', 4240, 4400, 'R16', 'THE WATCHERS', 'RUN-A + HILLS',
      'Behind a backlit watcher at the seventh fire, looking to the cold glow; small figures on far ridges, eyelines only.',
      # PR #8's companion job watchers_a_catches3 (160/160) carries A14 catches3's fire sizes across the cut: the
      # right-hand ridge fire measures 117 px at A14 4239, 128 px at catches3 4240, but 52 px at hearth3 4240 (the
      # contraction PR #8 warned of; warm-pixel blobs, 29 Sep). Codex PR #7's restaged hearth (hearth3) stays next.
      # ADOPTED 29 Sep with A14 (same lane, same turned camera and fire table across 4239|4240): the watcher reads as
      # a silhouette in front of the near fire, whose light falls off on the rock; ALTERNATIVES['A15'] is catches3
      [T('cand_watchers_linked-fires', 0, 'exact', 'linked-fires companion (Codex lane beaconrun, farm 29 Sep)',
         need=(4240, 4399)),
       T('watchers_A_catches3', 0, 'exact', 'PR #8 companion: A14 catches3 fire sizes', need=(4240, 4399)),
       T('watchers_A_hearth3', 0, 'exact', 'Codex restaged hearth (PR #7)'), T('watchers'), T('run')]),
    S('A16', 4400, 4720, 'E10', 'TOWERS IN THE LIGHT', 'EMBERS',
      "Far ridge fires light the towers' backs; the surges stop; the two giants open their shutters to each other first.",
      EMB_A),
    S('A17', 4720, 4880, 'E10', 'THE FIRE, SEEN', 'EMBERS',
      'The camera walks down to the calm fire; small lights come to the rim; it gathers into one small heart.', EMB_A),
    # OWNER NIGHT (29 Sep): the RUN-A4 restage, held since 27 Sep for two fixes, rendered on the farm with both: the
    # line stands at the first watch-fire through the close and medium shots and sets off from 5180 (the user: "the
    # walking at ~5050 was glitchy"), the rock step at 5584 lowered (low_shoulders: it read as a tub) and the slack rope
    # drawn as a decal on the snow (it read as a dotted line, 5011-5113+). need= keeps the accepted crossing until all
    # 960 frames have landed; ALTERNATIVES['A18'] is the accepted crossing.
    S('A18', 4880, 5840, 'R6', 'THE CROSSING', 'RUN-A',
      'One take: the great lantern on poles; forty roped bearers on a knife-edge above the cloud; the sky wheels.',
      # both_decal_cap (Codex lane dash; plays once all 960 frames land) is both_decal with crossing ridge CR0[3]'s
      # endpoint cap made continuous: its 15 m / 13 m flanks switched by side beyond the segment end, a 0.156 m height
      # step that shaded as a dashed line on the snow (lower left, ~5011-5113 and later); the farm test frames 5043,
      # 5071 and 5113 show no line (changed pixels confined to the terrain below y 537)
      [T('cand_crossing_both_decal_cap', 0, 'exact', 'both_decal + ridge row 3 round cap (Codex lane dash, farm 29 Sep)',
         need=(4880, 5839)),
       T('cand_crossing_both_decal', 0, 'exact', 'RUN-A4 restage + low_shoulders rock + rope decal (farm 29 Sep)',
         need=(4880, 5839)), T('crossing'), T('run')]),
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


# Codex PR #15's candidates (ADOPTION.md), rendered natively on the farm 29 Sep from claude/owner-night-20260929 at
# 5bc5fe9, each unit's setup first proving the candidate's default path pixel-equal to the accepted renderer. An
# ADOPTED candidate plays first, guarded by need= (it switches over whole, never frame by frame), and the accepted take
# stays second as the alternative: deleting the first take restores the accepted shot.
CAND = 'Codex PR #15 candidate, farm 29 Sep (5bc5fe9)'
# ADOPTED candidates play as their shot's ONE take (a C5 shot has exactly one source, no fallback: c5_readiness). The
# accepted takes they replaced stay delivered and untouched; restoring one is swapping its take back into the row (and
# its stem back into c5_readiness.SHOT_MAP), or COLD_CUT = 3840 for the Cold lead-in. Section -> the accepted take.
ALTERNATIVES = {
    'C3': ("c5('cand_t1_current-words', f'current-words: {CAND}', matte='cand_t1_current-words_matte')",
           'the unheld current-words camera (R02 cut by the frame bottom ~515-538); further back, book(): book_C '
           '320-559 with the old wording "...forges the Ring in secret." baked at 400-539'),
    'C8 sweep': ("one row S('C8', 1680, 1718, ...) with [ft_sweep()]", 'the hard entry: the filmed sweep pops on at '
                 '1680 (RGB MAD 13.33 against ~3 either side)'),
    'C16/C17': ("C15 lead-in c5('cand_cold_lead24', ...), C16 c5('embers_C5_cold', PR11), C17 "
                "c5('embers_C5_unfinished', PR11); restore the three together",
                'the forges go cold: every forge dark at 3848, smoke over cold masonry, the Ring draining to grey '
                'over dead towers (the owner, 29 Sep: it read as AI stopping)'),
    'C11': ("c5('embers_C5_trap', PR11)", 'no smoke: the sunk forge simply goes dark, then relights'),
    'C13': ("c5('runC_reveal_pair_v5', PR14)", 'the parchment Reveal (day page, gold beacon marks); restore together '
            "with C14's parchment scroll, or the 3112-3127 dissolve goes night into day"),
    'C14': ("T('runC_scroll', -3120, 'exact', 'RUN-C ink final')", 'the parchment scroll (RUN-C ink final)'),
    'C15': ("c5('map_last_beacon_C', PR13) with COLD_CUT = 3840", 'beacons as points on a greying map; the Cold cut at '
            '3840 (8 lit frames before 3848)'),
    'C18': ("c5('book_C5_deep_abandoned', PR12, matte='book_C5_deep_abandoned_matte')", 'upright ladders, box lamp'),
    'C19': ("c5('runC_watch_v5', PR14)", 'the parchment Watch'),
    'C22': ("c5('book_C5_pen', PR12, matte='book_C5_pen_matte')", 'pale nib, hard black gutter'),
    'A7 plates': ("EMB_A (T('embers_A3', ...)) for the whole row; or T('embers_A3_pulse', ...) for eased towers over all of "
                  "1840-2399; renders/embers_A3_eased is the height-only ease (LD_SURGE_EASE)",
                  'the accepted towers: each flashes to (1 + 0.9 pulse) x (1 + 2.5 heat band) in one frame on its beat '
                  '(per-beat peak one-frame MAD 12.38); the all-pulse folder also eases THE EDGE, where the flashes '
                  'line up (1844 MAD 10.22 -> 12.46); the height-only ease measured no better (12.38 -> 11.97)'),
    'A2': ("T('falsedawn')", 'the accepted sky: the high mackerel deck, silvering late, drawn as dark dashes over the Milky Way'),
    'A14': ("T('beaconrun_A_catches3', 0, 'exact', need=(3920, 4239))", 'PR #8 catches3: the run as 1-2 px specks '
            'until 4148, the camera on the left-hand ranges'),
    'A15': ("T('watchers_A_catches3', 0, 'exact', need=(4240, 4399))", "catches3's watcher: the near fire on a hard-edged rock"),
    'A18': ("T('crossing')", 'the accepted crossing (renders/crossing_A, 27 Sep): the walking line the user found '
            'glitchy at ~5050'),
}


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
COLD_CUT = 3816     # ADOPTED 29 Sep (Codex's lead24): 32 lit frames before 3848 instead of 8; the map keeps 24 all-lit
                    # frames (3792-3815). The alternative is 3840 (the accepted cut): set it back to restore it.


def last_beacon_rows(cold_cut):
    """Section C15 (3440-3840 in the bar map): the map up to cold_cut, then, if earlier than 3840, Cold's lit lead-in."""
    rows = [S('C15', 3440, cold_cut, '#15', 'THE LAST BEACON', 'MAP',
              'The map of the kingdoms: beacons flare in no order; one kingdom stays dark (3724-3783), then catches.',
              # ADOPTED 29 Sep: beacon-falloff. Each kingdom lights as a territory when its beacon catches, so the
              # holdout (3724-3783) is one dark kingdom among lit ones and its catch (3785-3791, measured on the
              # candidate) lights a region, not a point; line art aligned at zero shift with the delivered map
              [c5('cand_map_beacon-falloff', f'beacon-falloff: {CAND}')])]
    if cold_cut < 3840:
        rows.append(S('C15', cold_cut, 3840, '#16', 'THE FORGES FALL INTO STEP · LIT LEAD-IN', 'EMBERS',
                      f'Every forge still racing, {3848 - cold_cut} frames before 3848 brings them into step.',
                      [c5('cand_cold_in-step', f'in-step: {CAND}; before 3848 it equals the accepted lead24 '
                           '(proved at 3816, 3840, 3847)')]))
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
      'The leaves riffle back; a pen draws a mountain with a fire in its throat and in it a small gold ring.',
      # ADOPTED 29 Sep: current-words-held (Codex lane r02; farm 29 Sep 19:43 from ea7cdf3). The current-words page
      # (R02's current line baked into the handwriting 400-539) with the Mountain camera tilted down 1.43 deg while the
      # line is read: eased in from the arrival (t 2.1 s, 370) to 450, held to 512, eased out by 557, so the whole
      # inscription stays in frame (bottom ink row 709-736 in the hold, measured on the landed frames; the unheld page
      # ran 780-803 and the frame edge cut it ~515-538). Identical to current-words at 320-370 and 557-559 (MAD 0.00),
      # so C4's opening is untouched; the return peaks at a frame-to-frame MAD of 6.7 (half size) at 530, eased both
      # ways. ALTERNATIVES['C3'] is the unheld current-words
      [c5('cand_t1_current-words-held', 'current-words-held: Codex lane r02, farm 29 Sep (ea7cdf3)',
          matte='cand_t1_current-words-held_matte')]),
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
    S('C8', 1680, 1688, 'P2', 'INK PAGE · THE DEEP · THE SWEEP', 'BURN-C',
      'The filmed ember edge eases in over the held race (sweepC soft entry).', [ft_sweep_entry()]),
    S('C8', 1688, 1718, 'P2', 'INK PAGE · THE DEEP · THE SWEEP', 'BURN-C',
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
      # ADOPTED 29 Sep: front_smoke_near. The sunk forge smokes (a wisp at 2436, a column by 2490) and flares back
      # through its smoke at 2546, so its going out reads as going out; every other pixel and event frame as accepted
      [c5('cand_trap_front_smoke_near', f'front_smoke_near: {CAND}')]),
] + flint_rows(FLINT_CHOICE) + [
    S('C13', 2880, 3120, '#13', 'THE REVEAL · A PROMISE', 'RUN',
      'The vast ink range: her small fire and, at the same moment, a second fire on a far peak.',
      # ADOPTED 29 Sep with C14 as a night pair (director #1, editor #1: the parchment Reveal and Scroll showed the
      # first beacons as gold marks at or below the paper, glyph peaks 210-216 on a page of 194-200, the night as bright
      # as C20's dawn, and the cut at 2880 jumped from FLINT's mean luma 31 to 186). The night-fire grade of the Reveal
      # (rendered and kept since this afternoon) now pairs with C14's night-fire scroll, so the 3112-3127 dissolve is
      # night into night; ALTERNATIVES['C13'] is the parchment Reveal
      [c5('cand_reveal_night-fire', CAND)]),
    # the 7,200-frame cut's C17 (3840-4159 = runC_scroll 0-319), the same 320 source frames
    S('C14', 3120, 3440, '#14', 'THE BEACON RUN', 'RUN-C',
      'A lateral track at beacon height like a scroll unrolling; beacons bloom along the peaks, every two beats.',
      # ADOPTED 29 Sep: the night-fire scroll (Codex lane beacons; ink_final's frames through the shared night-fire
      # compositor, default route pixel-equal to runC_scroll at source 199; farm 29 Sep, 320/320): the beacons are
      # fires on the grey night page, as in C13 and C19. Source numbering 0-319 as runC_scroll's; ALTERNATIVES['C14']
      [T('cand_scroll_night-fire', -3120, 'exact', 'night-fire scroll (Codex lane beacons, farm 29 Sep)')]),
] + last_beacon_rows(COLD_CUT) + [
    # ADOPTED 30 Sep (owner's pace-not-prohibition note): no forge goes dark. ALTERNATIVES['C16/C17'] keeps the cold.
    S('C16', 3840, 4000, '#16', 'THE FORGES FALL INTO STEP', 'EMBERS',
      'At the last beacon no forge goes dark: from 3848 every forge eases to one shared working glow (by 3872) and '
      'breathes on the beat; faint warm chimney smoke; the Ring still gold.',
      [c5('cand_cold_in-step', f'in-step: {CAND}')]),
    S('C17', 4000, 4240, '#17', 'THE RING, UNFINISHED', 'EMBERS',
      'The Ring hangs, unfinished and gold, over working towers; its gold falls into every tower\'s windows; the storm '
      'thins; a held final frame (4217-4239).',
      [c5('cand_unfinished_in-step', f'in-step: {CAND}')]),
    S('C18', 4240, 4480, '#18', 'THE DEEP, ABANDONED', 'PAGES',
      'The mine page again, still: empty ladders, a lantern set down, the gold vein still glinting.',
      # ADOPTED 29 Sep: leaned_ladders. The ladders lean askew in a firmer line (left where they stood, not fixtures)
      # and the lamp is a globe lamp on its base where the accepted one reads as a small house at film size
      [c5('cand_deep_leaned_ladders', f'leaned_ladders: {CAND}', matte='cand_deep_leaned_ladders_matte')]),
    S('C19', 4480, 4720, '#19', 'THE WATCH', 'RUN',
      'The ink range with a beacon burning on every peak as the camera drifts (the run\'s poses 80-319, all lit).',
      # ADOPTED 29 Sep: night-fire. The watch through the night: grey ink, every beacon a drawn flame, so the dissolve
      # into the Illumination (4712-4728) is the night lightening into the dawn that fills the ink with colour. The
      # Reveal's night-fire (cand_reveal_night-fire, rendered) is NOT adopted: it dissolves (3117-3123) into the
      # Beacon Run, which is parchment and has no night version, over the same range on the same night.
      [c5('cand_watch_night-fire', f'night-fire: {CAND}')]),
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
      # ADOPTED 29 Sep: soft_spine_metal. A steel nib with its vent and slit (the accepted tip is a pale cone: a pencil,
      # a spill) and the gutter a graded fold into the spine instead of a flat black band
      [c5('cand_pen_soft_spine_metal', f'soft_spine_metal: {CAND}', matte='cand_pen_soft_spine_metal_matte')]),
    # the 7,200-frame cut's C28 (6960-7199): THE LONG DAWN burns on in book space (book_C 6980-7160 = C5 5700-5880)
    S('C23', 5680, 5920, 'X3', 'TITLE', 'MAP',
      'The blank recto: THE LONG DAWN burns on in fire-letters and cools to ink; the book goes back into the dark.',
      [book(off=1280)]),
]

# ------------------------------------------------------------------------------------------------------- D
# THE LONG DAWN, treatment v2 (30 Sep): 34 beats, 115 bars, 9,200 frames. A and C remain alternatives.
# NEW picture has NO takes until explicitly adopted; expected stems, off=0 and complete source needs below
# document D numbering without making folder arrival an adoption. D07's separately approved burn is integrated
# by the burn lane. Reused source clocks and holds are explicit, and A's entire opening is copied independently.
D_CROSSING_SOURCE_START = 5180  # Reviewed A5180-5579: walk set-off; excludes the A4960 silhouette openings.
# Explicit adoption switches. Uncomment ONE line after reviewing its complete delivery; folder arrival alone
# changes nothing. D08's shared 400-frame plate feeds both D07's pre-roll and D08's body. D23's same one-line
# adoption declares caption14 baked; titles keeps its overlay until this complete take actually wins the plan.
D_NEW_TAKES = {
    'D08': [T('embers_D_inscription', 0, 'exact', need=(1680, 2079))],
    'D09': [T('embers_D_forging', 0, 'exact', need=(2080, 2399))],
    'D10': [T('embers_D_race', 0, 'exact', need=(2400, 2719))],
    'D12': [T('embers_D_brink', 0, 'exact', need=(2960, 3199))],
    'D13': [T('embers_D_vision', 0, 'exact', need=(3200, 3439))],
    'D14': [T('embers_D_gap', 0, 'exact', need=(3440, 3519))],
    'D16': [T('embers_D_trap', 0, 'exact', need=(3760, 4079))],
    'D17': [T('falsedawn_brink', 0, 'exact', need=(4080, 4239))],
    'D18': [T('embers_D_crowns', 0, 'exact', need=(4240, 4559))],
    'D19': [T('falsedawn_twofires', 0, 'exact', need=(4560, 5039))],
    'D21b': [T('embers_D_holdout', 0, 'exact', need=(5480, 5519))],
    'D22': [T('embers_D_instep', 0, 'exact', need=(5680, 5839))],
    'D23': [T('book_D_oldfire', 0, 'exact', need=(5840, 6079), baked_text=('D14',))],
    'D24': [T('embers_D_unfinished', 0, 'exact', need=(6080, 6399))],
    'D27': [T('falsedawn_watch', 0, 'exact', need=(7040, 7359))],
    'D28': [T('falsedawn_truedawn', 0, 'exact', need=(7360, 7839))],
    'D31': [T('book_lastleaf_open', 0, 'exact', need=(8320, 8639))],
    # 'D33': [T('book_D_title', 0, 'exact', need=(8880, 9119))],  # optional replacement
}
# D10/D12 also supply Deep's held under-plates; a fallback list could select a different picture in the row.
for _d_code, _d_takes in D_NEW_TAKES.items():
    if not isinstance(_d_takes, (list, tuple)) or len(_d_takes) != 1:
        raise ValueError(f'D_NEW_TAKES[{_d_code!r}] requires exactly one explicit take')
# D07: explicit stand-in adoption. C559's gilt Ring component is 59x29 px, centred at approximately (998,92).
# The measured white-core centroids of A1400 and A1439 anchor the flame, never the whole valley's bounding box.
# ONE-LINE future adoption: uncomment D08 in D_NEW_TAKES above (camera aligned from D1680;
# need=(1680, 2079) gates the whole 400-frame incoming plate and forging front together).
D_BURN_PLATE = T('embers_A3', -280, 'exact', 'STAND-IN A1400-1439 thinking fire; source1439 held thereafter',
                 need=(1400, 1439), clamp=(1400, 1439), final_eligible=False,
                 screen_transform=dict(f0=1680, f1=1760, source_f0=1680, source_f1=1719,
                                       anchor0=(960.293, 460.242), anchor1=(959.523, 515.664),
                                       target0=(998., 92.), target1=(960., 402.), scale0=.10, scale1=1.,
                                       border=(14/255., 14/255., 14/255.)))
# Explicit local stand-ins; one-line future adoption is D10/D12 in D_NEW_TAKES. The same selected plate feeds
# each held under-layer. These local takes stay provisional even when every local file is present.
D_DEEP_RACE_PLATE = T('cutd_deep_race',0,'exact','STAND-IN native open-Ring race held at D2719',
                      hold=2719,need=(2719,2719),final_eligible=False)
D_DEEP_BRINK_PLATE = T('cutd_deep_brink',0,'exact','STAND-IN native brink held at D2960',
                       hold=2960,need=(2960,2960),final_eligible=False)
D_DEEP_RACE_SOURCE = D_NEW_TAKES.get('D10',[D_DEEP_RACE_PLATE])[0]
D_DEEP_BRINK_SOURCE = D_NEW_TAKES.get('D12',[D_DEEP_BRINK_PLATE])[0]
D_DEEP_RACE_UNDER = ('take',D_DEEP_RACE_SOURCE,2719,'D')
D_DEEP_BRINK_UNDER = ('take',D_DEEP_BRINK_SOURCE,2960,'D')
D_DEEP_ENTRY = T('cutd_deep_clean',0,'exact','Filmed re-bake over embers_D_race D2719; provisional local sources',
                 under=D_DEEP_RACE_UNDER,linear_mix='cutd_deep_sweep_coeff',need=(2720,2757),final_eligible=False)
D_DEEP_EXIT = T('cutd_deep_exit',0,'exact','Filmed re-bake onto embers_D_brink D2960; 36-frame exit',
                matte='cutd_deep_exit_matte',under=D_DEEP_BRINK_UNDER,need=(2945,2980),final_eligible=False)
D = [dict(deepcopy(row), sec='D0', code=f'D{number:02d}') for number, row in enumerate(A[:5], 1)] + [
    S('D1', 1440, 1680, 'D06', 'THE OLD STORY', 'PAGES-C reuse',
      'The red book: a pen draws a mountain with fire in its throat and a small closed gold ring.',
      [T('cand_t1_current-words-held', -1120, 'exact', 'C3 source320-559; old-story words baked on the page',
         matte='cand_t1_current-words-held_matte')]),
    # x1burn_D_drawnring: generated x1burn v2 transition at D1680-1759; no shared layer-folder writes.
    S('D1', 1680, 1760, 'D07', 'THE BURN', 'EDIT',
      'A burn born at the drawn gold Ring opens the page onto ours; incoming fire is an explicit A stand-in.',
      D_NEW_TAKES.get('D08', [D_BURN_PLATE])),
    # NEW: D_NEW_TAKES['D08']; its need=(1680, 2079) includes the preceding burn pre-roll.
    S('D1', 1760, 2080, 'D08', 'THE FORGING FRONT', 'VISION',
      'NEW embers_D_inscription: A5\'s ice-white thinking fire, edged in gold, draws out into a band; the camera '
      'rides the front as it writes every script, stops mid-letter, and leaves the band hanging open.',
      D_NEW_TAKES.get('D08', [])),
    # NEW: expected stem embers_D_forging, off=0, need=(2080, 2399); no take until explicitly adopted.
    S('D1', 2080, 2400, 'D09', 'THE FORGING', 'OPENRING',
      'NEW embers_D_forging: towers rise under the open band; only the two tallest towers send sparks to its ends; '
      'each hammer stroke pays gold into the striker\'s windows.',
      D_NEW_TAKES.get('D09', [])),
    # NEW: expected stem embers_D_race, off=0, need=(2400, 2719); D_NEW_TAKES replaces the provisional still.
    S('D2', 2400, 2720, 'D10', 'THE RACE', 'OPENRING',
      'NEW embers_D_race: surges on every beat; glare hides the narrowing gap, gold reaches the nearest windows, '
      'and two giants pull ahead.',D_NEW_TAKES.get('D10',[D_DEEP_RACE_PLATE])),
    S('D2', 2720, 2728, 'D11a', 'THE DEEP · THE SWEEP ENTRY', 'BURN-C reuse',
      'Filmed soft-entry re-bake over embers_D_race D2719; explicit local assets use a provisional open-Ring plate. '
      'C\'s closed-Ring baked picture remains the unconfigured fallback.',
      [deepcopy(D_DEEP_ENTRY), dict(ft_sweep_entry(), off=-1040, final_eligible=False,
            note='C1680-1687 reused; needs re-bake over embers_D_race D2719 (baked C closed-Ring race)')]),
    S('D2', 2728, 2758, 'D11b', 'THE DEEP · THE SWEEP', 'BURN-C reuse',
      'The original filmed ember edge sweeps to parchment in a re-bake over embers_D_race D2719; '
      'the local plate is provisional and C\'s closed-Ring bake remains the unconfigured fallback.',
      [deepcopy(D_DEEP_ENTRY), dict(ft_sweep(), off=-1040, final_eligible=False,
            note='C1688-1717 reused; needs re-bake over embers_D_race D2719 (baked C closed-Ring race)')]),
    S('D2', 2758, 2945, 'D11c', 'THE DEEP', 'MAP-C reuse',
      'The pen follows a gilt vein down pillared halls; the re-bake under-layer holds embers_D_race D2719 '
      'through the Deep. The unconfigured fallback holds C\'s closed-Ring race at source1679.',
      [book('Filmed re-bake under embers_D_race D2719 from the explicitly adopted D10 plate; provisional local sources',
            off=-1040,under=D_DEEP_RACE_UNDER,final_eligible=False),
       book('C1718-1904 reused; holds embers_C3 source1679; needs re-bake over embers_D_race D2719',
            off=-1040, under=('hold', 'embers_C3', 1679), final_eligible=False)]),
    S('D2', 2945, 2960, 'D11d', 'THE DEEP · THE GLOW', 'BURN-C reuse',
      'The red glow opens onto embers_D_brink D2960 in a 36-frame filmed re-bake through D2980; '
      'the local Brink plate is provisional. The unconfigured fallback holds C\'s storm source1920.',
      [deepcopy(D_DEEP_EXIT), dict(ft_eye(('hold', 'embers_C3', 1920)), off=-1040, final_eligible=False,
            note='C1905-1919 reused; holds source1920; needs re-bake over embers_D_race D2719 / embers_D_brink D2960')]),
    # NEW: expected stem embers_D_brink, off=0, need=(2960, 3199); D_NEW_TAKES replaces the provisional still.
    S('D2', 2960, 3200, 'D12', 'THE BRINK', 'VISION',
      'NEW embers_D_brink: the red glow burns through onto the Ring, its gap at the narrowest yet; towers lean in.',
      D_NEW_TAKES.get('D12',[D_DEEP_BRINK_PLATE])),
    # NEW: expected stem embers_D_vision, off=0, need=(3200, 3439); no take until explicitly adopted.
    S('D2', 3200, 3440, 'D13', 'IF IT CLOSED', 'VISION',
      'NEW embers_D_vision: the ends touch; a white seam and missing letters kindle; the circle becomes an '
      'ice-white Eye; the two giants bow first, then every tower; push into the slit.',
      D_NEW_TAKES.get('D13', [])),
    # NEW: expected stem embers_D_gap, off=0, need=(3440, 3519); no take until explicitly adopted.
    S('D2', 3440, 3520, 'D14', 'THE GAP', 'VISION',
      'NEW embers_D_gap: hard cut on the downbeat from slit to open gap as the next hammer stroke lands.',
      D_NEW_TAKES.get('D14', [])),
    S('D3', 3520, 3760, 'D15', 'THE REFUSAL', 'PAGES-C reuse',
      'A gloved palm offers a closed ring; a hooded figure turns away.',
      [T('book_C5_refusal', -1440, 'exact', 'C10 whole source2080-2319', matte='book_C5_refusal_matte')]),
    # NEW: expected stem embers_D_trap, off=0, need=(3760, 4079); no take until explicitly adopted.
    S('D3', 3760, 4080, 'D16', 'THE TRAP', 'OPENRING',
      'NEW embers_D_trap: the page burns onto ours; one forge slows, loses gold to its neighbours and is passed; '
      'it flares back into the race; two giants run neck and neck.',
      D_NEW_TAKES.get('D16', [])),
    # NEW: expected stem falsedawn_brink, off=0, need=(4080, 4239); no take until explicitly adopted.
    S('D4', 4080, 4240, 'D17', 'THE GLOW AT THE BRINK', 'GLOWVARS',
      'NEW falsedawn_brink: the opening ridge again; the glow pulses on the race\'s beat over red cloud.',
      D_NEW_TAKES.get('D17', [])),
    # NEW: expected stem embers_D_crowns, off=0, need=(4240, 4559); no take until explicitly adopted.
    S('D4', 4240, 4560, 'D18', 'THE TWO CROWNS', 'VISION',
      'NEW embers_D_crowns: a locked frame; two orange beacons kindle together at D4320 on the giants\' crowns, '
      'each lighting the other\'s face while both forges keep hammering.',
      D_NEW_TAKES.get('D18', [])),
    # NEW: expected stem falsedawn_twofires, off=0, need=(4560, 5039); no take until explicitly adopted.
    S('D4', 4560, 5040, 'D19', 'THE TWO FIRES', 'GLOWVARS',
      'NEW falsedawn_twofires: match the crowns to two beacons either side of the glow; fires answer toward us.',
      D_NEW_TAKES.get('D19', [])),
    S('D4', 5040, 5180, 'D20', 'EVERY RIDGE', 'RUN-A reuse',
      'Fires to the horizon over a cloud sea lit warm from beneath; the nearest fire burns the picture open.',
      [T('reveal_A', -1380, 'exact', 'A13 source3660-3799')]),
    # D5180-5219: the same x1burn v2 grammar as D07, born at A3799's measured nearest ridge fire.
    S('D4', 5180, 5480, 'D21a', 'THE LAST KINGDOM', 'MAP-C reuse',
      'The nearest ridge fire burns open the held A3799 image onto C3440 and the advancing map; '
      'kingdoms light in no order until one stays dark.',
      [T('cand_map_beacon-falloff', -1740, 'exact', 'C15 source3440-3739; pause in the dark holdout stretch')]),
    # NEW: expected stem embers_D_holdout, off=0, need=(5480, 5519); no take until explicitly adopted.
    S('D4', 5480, 5520, 'D21b', 'THE HOLDOUT', 'OPENRING',
      'NEW embers_D_holdout: the dark kingdom\'s hammer rings; planned first contact D5496; its sparks reach '
      'the band and narrow the gap a notch.',
      D_NEW_TAKES.get('D21b', [])),
    S('D4', 5520, 5554, 'D21c', 'THE LAST KINGDOM · WAIT', 'MAP-C reuse',
      'Return to the map and hold source3739, the frame just before the holdout insert.',
      [T('cand_map_beacon-falloff', 0, 'exact', 'Hold C15 source3739 for34 frames', hold=3739)]),
    S('D4', 5554, 5630, 'D21d', 'THE LAST KINGDOM · THE CATCH', 'MAP-C reuse',
      'The holdout\'s beacon catches on the downbeat at D5600 (measured C3786) and joins the others.',
      [T('cand_map_beacon-falloff', -1814, 'exact', 'C15 source3740-3815; source3786 maps exactly to D5600')]),
    S('D4', 5630, 5680, 'D21e', 'THE LAST KINGDOM · ALL LIT', 'MAP-C reuse',
      'Hold the last delivered all-lit map frame to the in-step downbeat.',
      [T('cand_map_beacon-falloff', 0, 'exact', 'Hold C15 source3815; never request source3816+', hold=3815)]),
    # NEW: expected stem embers_D_instep, off=0, need=(5680, 5839); no take until explicitly adopted.
    S('D4', 5680, 5840, 'D22', 'IN STEP', 'OPENRING',
      'NEW embers_D_instep: every forge eases into a shared working glow; hammers land together, the ends cool '
      'from white to gold, and the glare clears.',
      D_NEW_TAKES.get('D22', [])),
    # NEW: expected stem book_D_oldfire, off=0, need=(5840, 6079); no take until explicitly adopted.
    S('D5', 5840, 6080, 'D23', 'THE OLD FIRE', 'LASTPAGE',
      'NEW book_D_oldfire: in the book, the drawn Ring drops into the Mountain\'s drawn fire.',
      D_NEW_TAKES.get('D23', [])),
    # NEW: expected stem embers_D_unfinished, off=0, need=(6080, 6399); no take until explicitly adopted.
    S('D5', 6080, 6400, 'D24', 'THE RING, UNFINISHED', 'OPENRING',
      'NEW embers_D_unfinished: the gap stands sharp over working towers; lamps rise around the band to light its '
      'letters, and gold falls into every window.',
      D_NEW_TAKES.get('D24', [])),
    S('D5', 6400, 6640, 'D25', 'THE DEEP, ABANDONED', 'PAGES-C reuse',
      'Falling gold becomes the vein; empty ladders and a lantern set down where its light ends.',
      [T('cand_deep_leaned_ladders', -2160, 'exact', 'C18 source4240-4479',
         matte='cand_deep_leaned_ladders_matte')]),
    S('D5', 6640, 7040, 'D26', 'THE CROSSING', 'RUN-A reuse',
      'Forty bearers carry the great lantern forward, no faster than its light shows the path; original framing '
      'is retained, so the preceding Deep lantern does not align with it on screen.',
      [T('cand_crossing_both_decal_cap', D_CROSSING_SOURCE_START - 6640, 'exact',
         'A5180-5579: reviewed walk set-off; original framing, without a screen-position match to the Deep lantern',
         need=(D_CROSSING_SOURCE_START, D_CROSSING_SOURCE_START + 399))]),
    # NEW: expected stem falsedawn_watch, off=0, need=(7040, 7359); no take until explicitly adopted.
    S('D6', 7040, 7360, 'D27', 'THE WATCH', 'GLOWVARS',
      'NEW falsedawn_watch: hours later, fires surround the warm steady glow, which breathes once a bar.',
      D_NEW_TAKES.get('D27', [])),
    # NEW: expected stem falsedawn_truedawn, off=0, need=(7360, 7839); no take until explicitly adopted.
    S('D7', 7360, 7840, 'D28', 'THE LONG DAWN', 'GLOWVARS',
      'NEW falsedawn_truedawn: the same ridge; rose light rises where the false dawn stood; the beacons still burn.',
      D_NEW_TAKES.get('D28', [])),
    S('D7', 7840, 8080, 'D29', 'THE TERRACES', 'RUN-A reuse',
      'The first light reaches the terraced summits across the cloud sea.',
      [T('dawnrev_A', -1840, 'exact', 'A19 source6000-6239')]),
    S('D7', 8080, 8320, 'D30', 'THE YEAR OF PLENTY', 'MAP-C reuse',
      'The book\'s plenty tree in the light of dawn; the preceding terraces already use the film finish.',
      [book('C21 source6160-6399; no inherited ink-to-film finish ramp', off=-1920)]),
    # NEW: expected stem book_lastleaf_open, off=0, need=(8320, 8639); no take until explicitly adopted.
    S('D7', 8320, 8640, 'D31', 'THE LAST WRITTEN LEAF', 'LASTPAGE',
      'NEW book_lastleaf_open: the storyteller\'s last written line breaks off mid-word beside the drawn open Ring; '
      'the facing page is blank and the pen rests.',
      D_NEW_TAKES.get('D31', [])),
    S('D7', 8640, 8880, 'D32', 'THE LAST PAGES', 'PAGES-C reuse',
      'The blank spread with the pen resting across it.',
      [T('cand_pen_soft_spine_metal', -3200, 'exact', 'C22 source5440-5679',
         matte='cand_pen_soft_spine_metal_matte')]),
    # Optional NEW book_D_title, off=0, need=(8880, 9119), is not adopted; retain the accepted C title.
    S('D7', 8880, 9120, 'D33', 'TITLE', 'MAP-C reuse',
      'THE LONG DAWN burns onto the blank recto and cools to ink; C\'s title remains until a replacement is adopted.',
      D_NEW_TAKES.get('D33', [book('C title source6960-7199; caption source6980-7159', off=-1920)])),
    S('D7', 9120, 9200, 'D34', 'THE HEARTH GOES OUT', 'EDIT', 'Black; the hearth goes out.', kind='black'),
]
EDL = {'A': A, 'B': B, 'C': C, 'D': D}

# EDIT transitions (assemble._transitions; the comp, not a cut). A's are A-FIX's; C5's are EDIT-C5's (29 Sep), designed
# around the adopted shots (the 7,200-frame cut's C windows are retired with it).
# Across an EDL boundary 'cut' the outgoing shot plays its own frames up to it, then holds its last; the incoming holds
# its first frame until it, then plays. A window whose layer frames are missing, or with a slate on either side, plays
# as the plain cut. ready=False holds a window back (and edit/c5_readiness.py FAILS a full build on it).
#   burn         PAGES' burn-through (x1burn.py v2): out = O * keep + I * (1 - cover) + glow; keep is COLOUR (the scorch
#                tint and the char band), cover = 1 - hole, glow is display sRGB
#   x1           MAP-L2's first formula (fallback): out = O * keep + I * (1 - keep) + glow (shows the map through the char)
#   dissolve     linear light, smoothstep over the window
#   dawn_dissolve  C19 -> C20 dissolve with the incoming plate at the sweep's neutral starting grade
#   dawn_sweep   one shot, no cut: a broad left-to-right front restores delivered colour and luminance
#   finish_ramp  one shot, no cut: the finish goes from the ink look to the film look, lerp(ink, film, smoothstep)
#   page_turn    EDIT-C5: the outgoing page curls over right to left onto the incoming (assemble.page_turn)
# Kept as hard cuts on purpose: 2080 (the Eye -> the Refusal: the hearth flare that brings us back to the book is in
# the render), 2640 (the Trap's roar -> the dark of the flint), 2880 (the catch -> the Reveal's simultaneous ignition,
# which must land on the cut), COLD_CUT (3840 as briefed: the last beacon -> the forges, lit for 8 frames before 3848
# puts them all out; movable, see COLD_CUT), 4000 (Cold -> Unfinished: one renderer, one camera, continuous).
# REVIEW (29 Sep): finished 960x402 C4711 mean RGB (.385976,.390486,.386703), luma .389254; C4728 luma
# .725026 and saturation .331380. Divide C4711's RGB by C4728's luma for the neutral incoming grade: the old
# dissolve spent the dawn's colour immediately. C5P2 sunrise 4720 / home 5040 / plenty 5200: hold grey through
# the dissolve, cross the centre at 4944 (inside R20, 4860-5019), ease across the home cadence, finish at 5160
# (bar 65 beat 3), leaving 40 unchanged frames before Plenty. Same absolute grade on both adjacent windows.
# A neutral prelight rises ahead of the colour: without it R20's full-size worst slice was 2.907:1 across
# 122 steady frames (4886-5007). It eases from zero, so the measured night match at the dissolve stays intact.
C_DAWN = dict(start=4728, done=5160, width=0.85, prelight=0.20,
              night_rgb=(0.532361384, 0.538582617, 0.533364320))

# Native projected caption-band geometry, exported without a book render (captionsC, 29 Sep). Bind these
# windows to those exact bytes and source offsets: a different camera/plate needs a new audit and export.
C_CAPTION_BAND_SHA = '9089b1f9e46ef507c80c1fc91b7e3f5173889691b6c0a5f1d9bf9bbe0f021e0c'

TRANS = {'A': [], 'B': [], 'C': [
    # Baked ink cannot be moved independently of the page. The current dry holds measure 2.422:1 at C528
    # and 2.782:1 at C5808 with projected glyph masks. Broad feathering avoids a bright caption-shaped strip;
    # both grades return to the original picture at their endpoints and leave the words/cameras untouched.
    dict(f0=400, f1=540, kind='caption_grade', id='R02', full0=430, full1=528,
         source_stem='cand_t1_current-words-held', source_off=0, band_sha256=C_CAPTION_BAND_SHA,
         curve=dict(kind='shadow_shoulder', low=0.04, high=0.14, gain=1.30),
         note='C3: deepen the dry ink and lift its paper gently, using the held-camera caption band'),
    dict(f0=5748, f1=5880, kind='caption_grade', id='title', full0=5808, full1=5847,
         source_stem='book_C', source_off=1280, band_sha256=C_CAPTION_BAND_SHA,
         curve=dict(kind='blackpoint_gain', blackpoint=0.0, gain=2.50),
         note='C23: a broad paper exposure lift as the fire cools to ink; fade with the sinking title'),
    # the ember README (PR11) leaves "the page burn" to EDIT: the book's grammar for page -> ember world is the
    # burn-through (C4-C5, C8-C9). DESIGNED, NOT BUILT: it needs burn layers for the Refusal page (ftburn/x1burn,
    # BURN/PAGES), because the page's delivered matte is opaque on all 240 frames and holds no hole; the three folder
    # names are PROPOSED, nothing has been rendered. Until then: a hard cut.
    # BUILT 29 Sep (x1burn v2: --center 870,234 --t-open 2312 --speed 9.4 --frames 2310-2345): the burn is born at the
    # drawn ring the hand offers, 100 px from where the Trap's Ring hangs (972,224), so the hole opens onto the Ring.
    # Hole 1% 2316, 13% 2322, 62% 2331 (fastest), 100% by 2340 (measured from _cover).
    dict(f0=2310, f1=2346, cut=2320, kind='burn', glow='x1_refusal_C5', keep='x1_refusal_C5_matte',
         cover='x1_refusal_C5_cover',
         note='C10 the Refusal page burns through onto C11 the Trap, born at the drawn ring (x1burn v2)'),
    # the same source pair as the 7,200-frame cut's #16 (runC_reveal pose 239 -> runC_scroll 0): two parchment views
    # of the range read as a jump cut; runC_scroll is not on the EDIT-C5 Mac, so this is unseen here
    dict(f0=3112, f1=3128, cut=3120, kind='dissolve',
         note='C13 the Reveal -> C14 the beacon run: two parchment views of the range (old #16, same sources); '
              'REVIEW (29 Sep): 6 frames read as a stutter between two sets of peaks, now 16'),
    # the 7,200-frame cut's #17 (the seventh beacon burns through onto the map) on the NEW map: its x1_map_C layers
    # were aligned to the retired map_C, so it needs new ones. DESIGNED, NOT BUILT; folder names PROPOSED.
    # BUILT 29 Sep: the 7,200-frame cut's #17 parameters shifted -720 (x1burn v2 --center 1130,485 --t-open 3432
    # --speed 9.4 --frames 3430-3465). The seventh beacon's flame sits at (1129-1135, 481) over 3432-3439 (measured);
    # the new map's lit beacon is 20 px away at (1111,468). Hole 1% 3436, 12% 3442, fastest 3450, 100% by 3460.
    dict(f0=3430, f1=3466, cut=3440, kind='burn', glow='x1_map_C5', keep='x1_map_C5_matte', cover='x1_map_C5_cover',
         note='C14 the seventh beacon burns through onto C15 the map (x1burn v2, the old #17 at -720)'),
    # PR11 leaves "the page dissolve" to EDIT: the grey Ring over the dark towers into the abandoned mine; centred on
    # the downbeat, it keeps 4217-4227 of the intended final hold clean and fades through the rest
    dict(f0=4228, f1=4252, cut=4240, kind='dissolve',
         note='C17 the Ring, unfinished -> C18 the Deep, abandoned: the page dissolve PR11 leaves to EDIT'),
    dict(f0=4468, f1=4492, cut=4480, kind='dissolve',
         note='C18 the mine page -> C19 the watch: ink drawing into ink range, through the paper'),
    dict(f0=4712, f1=4728, cut=4720, kind='dawn_dissolve', **C_DAWN,
         note='C19 the watch -> C20 the illumination: grey into grey; preserve the outgoing beacon fade'),
    dict(f0=4728, f1=5161, kind='dawn_sweep', **C_DAWN,
         note='C20: daylight fills the ink from frame left, centre at 4944 under R20, unchanged from 5160; '
              'C5P2 sunrise 4720, home 5040, plenty 5200'),
    dict(f0=5200, f1=5264, kind='finish_ramp',
         note="C21 opens pixel for pixel on C20's last frame (runC_illum 2877, ink look) and takes the film look by "
              "5264 (the 7,200-frame cut's #21, 6160-6224, same sources)"),
    # PR12: the Pen "starts on the blank spread, so EDIT must join it to the preceding page turn"; centred on bar 69 b1,
    # done before the score's voice-line window opens at 5460. The curl is EDIT's own comp (the Pen's delivered matte
    # is opaque, with no turn in it). REVIEW (29 Sep): C5440's straight bottom was the camera crop treated as a leaf
    # edge. The comp now carries book_C 6399's measured fore-edge and paper through the same cylinder and timing.
    dict(f0=5430, f1=5452, cut=5440, kind='page_turn', tilt=8.0, radius=0.11,
         note='C21 Plenty -> C22 the Pen: EDIT curl with the outgoing leaf\'s sampled edge, fibres and light'),
    dict(f0=5668, f1=5692, cut=5680, kind='dissolve',
         note='C22 the pen insert -> C23 the blank recto the title burns onto (UNSEEN: book_C not here)'),
    # REVIEW (29 Sep, all four picture reviewers; measured on the finished frames): a rendered black carries the film
    # base (14/255), EDIT's black is 0, and at these joins the whole screen stepped between them. The film base now
    # eases in over the incoming shot's first two seconds (floor: assemble._tk_floor), and C ends on a true black.
    dict(f0=80, f1=128, kind='floor', k0=1.0, k1=0.0,
         note='C1 black (0) -> C2 the red book, whose dark opens at the film base: the base eases in with the hearth'),
    dict(f0=1040, f1=1088, kind='floor', k0=1.0, k1=0.0,
         note='C5 the fire alone (true black) -> C6 the forging (film base) under the same flame'),
    dict(f0=5884, f1=5920, kind='floor', k0=0.0, k1=1.0,
         note="C's last 36 frames take the film base out, so C ends on a true black like A"),
]}

# A (28 Sep, approved): A-FIX's six windows (edit/afix_comp.py: bloom, dissolve, vision x2, iceheart, ember) + EDIT's
# grade-match of the harvested B reveal (B's sky is a touch darker and cooler than reveal_A's at the 3680 cut)
import afix_comp  # noqa: E402
import edge_polish  # noqa: E402
TRANS['A'] = list(afix_comp.A_TRANS) + [afix_comp.WATCHFIRES, edge_polish.WINDOW] + [   # + A-FIX's ENDING (28 Sep)
    dict(f0=960, f1=1026, cut=1012, kind='collapse', source_end=1012, land=1016, shutter=2.0,
         note='A4: a five-tap shutter and eased arrival settle the glyphs for the breath at1020'),
    dict(f0=1026, f1=1073, cut=1040, kind='swell', breathing=True,
         p0=(966, 238), c1=(964, 238), disc=(962, 235), flame=(959, 245),
         s0=0.2, s1=0.4, blend=(1038, 1042), R=330,
         note='A4 -> A5: inhale1026, full disc1040, moving light hands over to the flame through1072'),
    dict(f0=3612, f1=3660, kind='grade', gain=(1.18, 0.99, 0.94),
         note="HER FIRE (reveal_B, cropped) matched to reveal_A's sky: top-third means (.101 .142 .263) -> (.119 .140 "
              ".246) (reveal_A 3680; re-check on 3660 when reveal_a_1 lands)"),
    # joinsA (29 Sep): removing only the film base left A79|80 at mean Y 0 -> 10.808/255 with clear_high_deck
    # (finished 960x402 frames). Dissolve from A79's true black over the same 48 frames so the sky itself enters.
    dict(f0=80, f1=128, cut=80, kind='dissolve',
         note='A black -> FALSE DAWN: the finished sky and its film base ease in together over two seconds'),
    # The second floor window ends where A-FIX's ember window begins; no two windows share a frame.
    dict(f0=2800, f1=2836, kind='floor', k0=1.0, k1=0.0,
         note="A9's true black -> A10's void (film base): the base eases in before the ember's window"),
]


# Restore A's delivered opening windows only after TRANS['A'] is fully initialized; the nested tracks are D-owned.
# New pair windows remain plain cuts while either side is a slate.
TRANS['D'] = [deepcopy(t) for t in TRANS['A'] if 0 <= t['f0'] and t['f1'] <= 1440] + [
    dict(f0=1520, f1=1660, kind='caption_grade', id='R02', full0=1550, full1=1648,
         source_stem='cand_t1_current-words-held', source_off=-1120, band_frame_off=-1120,
         band_sha256=C_CAPTION_BAND_SHA, curve=dict(kind='shadow_shoulder', low=0.04, high=0.14, gain=1.30),
         note='D06 old-story inscription: C caption geometry400-539 on the same source pixels'),
    dict(f0=1680, f1=1760, cut=1680, kind='ring_burn', center=(998., 92.), t_open=1682., speed=2.8, seed=12,
         note='x1burn v2 at the measured gilt Ring on held C559; real burn with explicit stand-in incoming fire'),
    dict(f0=2945,f1=2981,kind='deep_reveal',under_start=2960,glow='cutd_deep_exit',cover='cutd_deep_exit_matte',
         tail_clear=(2976,2980),
         note='Original filmed C1905-1991 remapped across36 D frames; reveal the selected brink throughD2980'),
    dict(f0=3750, f1=3786, cut=3760, kind='burn', glow='x1_refusal_C5', keep='x1_refusal_C5_matte',
         cover='x1_refusal_C5_cover', layer_off=-1440,
         note='D15 refusal -> D16 trap: C burn source2310-2345; requires NEW trap under-plate D3760'),
    dict(f0=5180, f1=5220, cut=5180, kind='ring_burn', center=(1136., 623.),
         t_open=5182., speed=9.4, seed=12,
         note='Nearest ridge flame on held reveal_A A3799 (D5179); x1burn v2 opens onto map C3440 at D5180'),
    dict(f0=6388, f1=6412, cut=6400, kind='dissolve',
         note='D24 unfinished Ring -> D25 abandoned Deep; suppressed while unfinished is a slate'),
    dict(f0=6636, f1=6644, cut=6640, kind='dissolve',
         note='D25 abandoned Deep -> D26 crossing: eight-frame dissolve, original framing; lantern positions differ'),
    dict(f0=8630, f1=8652, cut=8640, kind='page_turn', tilt=8.0, radius=0.11,
         note='D31 last written leaf -> D32 blank spread and pen; suppressed while the last leaf is a slate'),
    dict(f0=8868, f1=8892, cut=8880, kind='dissolve',
         note='D32 pen insert -> D33 title on the blank recto'),
    dict(f0=8948, f1=9080, kind='caption_grade', id='title', full0=9008, full1=9047,
         source_stem='book_C', source_off=-1920, band_frame_off=-3200, band_sha256=C_CAPTION_BAND_SHA,
         curve=dict(kind='blackpoint_gain', blackpoint=0.0, gain=2.50),
         note='D33 title: C caption geometry5748-5879 on book_C source7028-7159'),
    dict(f0=9084, f1=9120, kind='floor', k0=0.0, k1=1.0,
         note='D33 title takes out the film base before D34 ends in black'),
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


def d_export_extra():
    """D's treatment timeline is real; unadopted new picture remains visible as slates."""
    return dict(version='D', status='treatment; new picture pending',
                script='THE LONG DAWN cut D treatment v2 (30 Sep 2026): 34 beats, 115 bars',
                barmap='music/v3/' + BARMAP['D'], bars=TOTAL['D'] // BAR, beats=34,
                pending_picture='NEW plates require an explicit D_NEW_TAKES entry; approved stand-ins stay '
                                'provisional. Folder arrival alone never adopts a plate.',
                adoption='Uncomment one literal T(...) line in D_NEW_TAKES after review. D08 gates all '
                         '1680-2079 for D07/D08 together. D23 declares baked_text=(D14,) and its complete '
                         'selection changes that caption to in_picture; missing/partial delivery retains the overlay.',
                rebake='D11 local entry coefficients composite over the selected D10 plate held at D2719; '
                       'the C page body uses the same input. Filmed C1905-1991 exits across D2945-2980 over '
                       'the selected D12 brink. A D-only cubic tail clear scales both premultiplied RGB and '
                       'cover across D2976-2980, reaching the incoming plate exactly at D2980 before finish. '
                       'Held direct takes honor their complete need gates. Original '
                       'C fallback pictures, local stand-ins and current Deep composites remain provisional.',
                map_burn='Adopted x1burn v2 at D5180-5219, born at nearest ridge flame (1136,623) on held '
                         'reveal_A A3799; opens onto the advancing map from C3440. t_open=5182 and speed=9.4 '
                         'are renderer parameters; cover-grid timing measurements are recorded separately.',
                score_status='stand-in required until D score exists')
