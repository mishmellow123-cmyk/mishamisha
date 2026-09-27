# EDIT v3 (lane EDIT): the three EDLs, text, animatics

## STATE AT HANDOFF (EDIT-v3, 27 Sep ~11:00Z; paused by the director for usage pacing, RESUME 15:00Z)

- **Done and pushed** (`claude/long-dawn-v2`): `edit/edl_v3.py` (the three EDLs as data, checked against the bar maps
  on every run), `edit/assemble.py` (v3; the v2 one is `edit/assemble_v2.py`), `edit/titles.py` (`lines_v3`,
  `composite_v3`; the v2 API is untouched), `edit/animatic.sh`, `edit/edl/edl_{A,B,C}.json`.
- **Animatics** (verified: frame counts, durations, streams, contact sheets of every cut):
  `~/mishamisha/_local_logs/animatic/{A,B,C}_animatic.mp4`, 960x402 H.264 CRF 23 + AAC; A 4:30.000 (click),
  B 3:46.667 (COMPOSER `final_B.wav`, the 10:17Z VIGIL render), C 5:00.000 (click). About 5 min for all three.
  The director re-runs `edit/animatic.sh` as renders land (he will when `renders/h1_v3h5` completes).
- **The director's calls since (applied):** B3 takes HEROINE's job numbering (`heroine_B` 960-1199 = B3 880-1119,
  off +80) after `deadember_B` in locked numbering. The folder table below is now every department's delivery
  convention: finals in these folder names, numbered in cut frames.
- **H5 CALLS in the edit:** the six lines (A-T2, A-T7, A-T9, A-T10a; C-T6, C-T7; frames unchanged), C23 = THE FIRE
  REMAINS (bar 70 ACCORD, bar 71 MAP), E14 gone, B = THE VIGIL (M2/M3 cutaways cut; H1 in B cropped to hands, tinder
  and sparks with a cool grade; its roar crop is 12 f because the take dissolves to A/C's wide from src ~1488).
- **Verified:** EMBERS' text-band dimming (`shots/embers/render.py` TEXT['A3'], TEXT['C3']) matches the bar maps to
  the frame and is enabled for A3 and C3 (`variant.text_band()`); the band (y 560-700) covers the lower third. The
  `--variant codedtowers` path passed a synthetic test (ALT frames replace MAIN only where they exist; coverage
  reports VARIANT). A-T6a/T6b share one row; A-T10a/b and C-T8a/b stack on black; C-T2 is legible at 1:1 on the
  brighter page (f596).
- **NEXT (at RESUME, the director's order):** (1) the ALT master `A_animatic_codedtowers.mp4` (it builds only once
  `renders/embers_A3_alt_codedtowers` has A-timeline frames; the variant is otherwise identical to MAIN); (2) the
  fallback-master audio for A and C (the resolver already takes `music/out/v3/fallback_<cut>.wav` or the MASTERS
  table in `music/NOTES_v3.md`; re-run the animatics when they land); (3) H9 gate prep (clean versions with
  `CLEAN=1`, contact sheets, the lines).
- **Open, lower priority:** the X1 glow/matte comps other than the book matte (`x1_map_C` onto the map at C17 to C18);
  the MOTION-LAB E2 challenger is not in the cut; C ink-line positions are the lower third until MAP gives page
  coordinates; `edit/mix.py` is the v1 re-mix and is not used in v3 (the masters come from `music/src/render_v3.py`).

## How the EDLs work

- Every shot is `[f0, f1)` on its cut's own timeline, with TAKES in priority order (`edl_v3.py`). A take is a render
  folder STEM plus `off` (`src = cut frame + off`) and a mode: `v3` = `renders/<stem>_<CUT>/` then `<stem>_v3/`;
  `layered` = those, then `<stem>_v2/`, then `renders/<stem>/` (REUSE of v2 frames in src numbering); `exact` =
  `renders/<stem>/` only (a department's own v3 folder, e.g. `embers_A3`, `runC_scroll`).
- `--variant codedtowers` tries `<folder>_alt_codedtowers` before every folder (the ALT folders hold only the
  frames that differ). A's ALT master is built by `animatic.sh` only once `renders/embers_A3_alt_codedtowers` has
  frames; until then it would equal MAIN.
- A shot uses the first take that covers every frame; otherwise the take covering the most frames, with the gaps as
  slates. A shared take can require a complete src range (`need=`): the H5 re-key of H1 (`renders/h1_v3h5`, landing
  now) replaces `h1_v3` in all three cuts at once when all of 1200-1555 is in, so no cut mixes the two looks.
- Book shots (MAP-v3) composite `rgb + (1 - matte) * under` from `book_C` + `book_C_matte`, per MAP's spec.
- Slates: the shot code, owner, name, one line, bars and frames in the top third (the lower third and centre stay
  free for the text), a progress line with beat ticks, and a beat light (brighter on downbeats) for sync checks.
- Burn-ins (off with `--clean` or `CLEAN=1`): shot and source frame top left; bar, beat, frame and seconds top right.

## Where each department's frames are expected (one line in `edl_v3.py` to change)

| lane | shots | folder (numbering) |
|---|---|---|
| EMBERS | A3-A10, A16-A17; C6-C7, C9, C11, C13a | `embers_A3` (+`_alt_codedtowers`), `embers_C3` (cut frames); `embers_C3_half` as a stand-in |
| RUN-A | A2, A13 R2-A, A14, A15, A18, A19-A20 | `falsedawn_A`, `reveal_A`, `beaconrun_A`, `watchers_A`, `crossing_A`, `bluehour_A`, or `run_A` (A frames) |
| RUN-B | B1, B5 R2-B, B6-B12, B13-B14 | `dusk_B`, `reveal_B`, `vigil_B`, `handback_B` (B frames; the cloud jobs already use these) |
| RUN-C | C13b, C16, C17, C24 | `ringfall_C`/`run_C` (C frames); `runC_reveal` (-3600), `runC_scroll` (-3840), `runC_illum` (+2398-5680) |
| HEROINE | H1 all cuts; B2, B3 | `h1_v3h5` then `h1_v3` (src 1200-1555; strike 1 = 1236, roar = 1476); `climb_B`/`deadember_B`/`heroine_B` (B frames) |
| MONTAGE-3D-2 | A13 KARST, DESERT; C14 find, C15 fire test, C22 melt | `montage3d_v3/karst_slow` (0-59); `montage3d_v3/desert` or `montage_v2` 1520-1579; `ring_C` or `montage3d_v3/{find_a,find_b,fire,melt}` (C frames) |
| MAP | C2-C5, C8, C9 head, C18, C23 bar 71, C25-C28 | `book_C` + `book_C_matte`; `map_C` 4160-4479 and 5600-5679 (C frames) |
| ACCORD | C19-C23 bar 70 | `accord_C` (C frames 4480-5599) |

Heroine note: the RENDER_SPEC on `claude/v3-heroine` puts THE DEAD EMBER at "B 960-1199" (pre-lock numbering); the
locked sheet has B3 at 880-1119. Frames must arrive in the locked numbering or the EDL line needs an `off`.

## Text (titles.py v3)

- Words from `music/v3/barmap_<cut>.json`, with `H5_WORDS` applied on top; the table prints with
  `python3 -c "import titles; print(titles.text_table('A'))"` from `edit/`.
- A: Cormorant Garamond italic 56, lower third (y 648), letters staggered inside a 12-frame fade in and a 12-frame
  fade out, all inside [f_in, f_out); T6a and T6b on one row (T6b joins at 1720); T10a/T10b centred on black.
- C: ink lines (T1, T7, T9, T14) in EB Garamond italic, a slanted pen-nib wipe over 24 frames with a little bleed at
  the nib, a grainy dissolve out over 12; iron-gall ink on a light page, parchment white on a dark ground. Fire lines
  (T2, T5a, T5b, T6, T10, T11, T12; T8a/T8b stacked on black) kindle over 12 frames (staggered heat, glow) and crumble
  over 12 into rising sparks.
- Titles (X3, Cinzel): A kindles in the sky, holds, crumbles into rising sparks (last 46 f) before the fade to black
  from bar 81 b3.8 (f 6456); B kindles and fades into the light (last 60 f); C burns on in fire-letters and cools to ink.
- **EMBERS text dimming (checked):** `shots/embers/render.py` TEXT['C3'] = (1056, 1190) T5b and (1446, 1550) T6, which
  are exactly the bar map's windows for the lines over EMBERS' C shots (T8a/b sit on black; T7 is on MAP's page).
  TEXT['A3'] = T4, T5, T6a/b, T7, T8, T9 also match the bar map to the frame. The H5 calls change words only, so no
  dimming window moves. If EMBERS ever renders C5 (E15's tail) it needs (920, 1030) for T5a.

## Coverage
<!-- COVERAGE:BEGIN -->
_Generated by `edit/assemble.py --coverage --notes` at 27 Sep 10:40Z._

**A** (6480 f): 760 f picture (12%), audio: click track (no score or fallback yet)

| sec | shot | frames | bars | status | source / expected folder |
|---|---|---|---|---|---|
| A1 | A1 BLACK | 0-80 | 1-1 | EDIT (black) | edit |
| A2 | R1 FALSE DAWN | 80-560 | 2-7 | SLATE | expects renders/falsedawn_A, renders/run_A |
| A3 | E1 INTO THE LIGHT · GLYPHS | 560-960 | 8-12 | SLATE | expects renders/embers_A3 |
| A4 | E1 THE POINT | 960-1040 | 13-13 | SLATE | expects renders/embers_A3 |
| A5 | E2 · E3 IGNITION · THE PROMISE | 1040-1440 | 14-18 | SLATE | expects renders/embers_A3 |
| A6 | E5-A THE TOWERS · TWO GIANTS | 1440-1840 | 19-23 | SLATE | expects renders/embers_A3 |
| A7 | E6 THE EDGE | 1840-2400 | 24-30 | SLATE | expects renders/embers_A3 |
| A8 | E7-A THE BRINK · OVER THE RIM | 2400-2640 | 31-33 | SLATE | expects renders/embers_A3 |
| A9 | E4 THE DEAD VALLEY | 2640-2800 | 34-35 | SLATE | expects renders/embers_A3 |
| A10 | E9 BLACK · THE EMBER | 2800-3120 | 36-39 | SLATE | expects renders/embers_A3 |
| A11 | X2 DARK ADAPTATION | 3120-3360 | 40-42 | EDIT proxy | edit star field (RUN-A plate pending) |
| A12 | H1-A THE FIRST FIRE | 3360-3600 | 43-45 | RENDERED | h1_v3 1236-1475 (H1 v3 master timing, pre-H5 look (beanie, dish-rack basket, white sparks)) |
| A13 | H1-A THE ROAR | 3600-3680 | 46-46 | RENDERED | h1_v3 1476-1555 (H1 v3 master timing, pre-H5 look (beanie, dish-rack basket, white sparks)) |
| A13 | R2-A EVERY RIDGE | 3680-3800 | 47-48 | SLATE | expects renders/reveal_A, renders/run_A |
| A13 | M5 KARST | 3800-3860 | 48-49 | RENDERED | montage3d_v3/karst_slow 0-59 (Blender KARST at 2/3 speed, pre-H5 (pines, pillar profile)) |
| A13 | M5 DESERT | 3860-3920 | 49-49 | RENDERED | montage_v2, montage 1520-1579 (Blender DESERT montage_v2 1520-1579, pre-H5 (framing, footprints)) |
| A14 | R3 THE BEACON RUN | 3920-4240 | 50-53 | SLATE | expects renders/beaconrun_A, renders/run_A |
| A15 | R16 THE WATCHERS | 4240-4400 | 54-55 | SLATE | expects renders/watchers_A, renders/run_A |
| A16 | E10 TOWERS IN THE LIGHT | 4400-4720 | 56-59 | SLATE | expects renders/embers_A3 |
| A17 | E10 THE FIRE, SEEN | 4720-4880 | 60-61 | SLATE | expects renders/embers_A3 |
| A18 | R6 THE CROSSING | 4880-5840 | 62-73 | SLATE | expects renders/crossing_A, renders/run_A |
| A19 | R7 THE BLUE HOUR | 5840-6240 | 74-78 | SLATE | expects renders/bluehour_A, renders/run_A |
| A20 | X3 TITLE | 6240-6480 | 79-81 | SLATE | expects renders/bluehour_A, renders/run_A |

A with `--variant codedtowers`: no `_alt_codedtowers` frames on the A timeline yet (embers_A3_alt_codedtowers is empty), so the ALT master is identical to MAIN.

**B** (5440 f): 892 f picture (16%), audio: COMPOSER master `music/out/v3/final_B.wav`

| sec | shot | frames | bars | status | source / expected folder |
|---|---|---|---|---|---|
| B1 | R8 DUSK | 0-640 | 1-8 | RENDERED | run_b_tests/dusk_motion 0-639 (quarter-res motion test, pre-H5 (triangle peak)) |
| B2 | H4 THE CLIMB | 640-880 | 9-11 | SLATE | expects renders/climb_B, renders/heroine_B |
| B3 | H5 THE DEAD EMBER | 880-1120 | 12-14 | SLATE | expects renders/deadember_B, renders/heroine_B |
| B4 | H1-B THE FIRST FIRE | 1120-1360 | 15-17 | RENDERED | h1_v3 1236-1475 (H1 v3 master timing, pre-H5 look (beanie, dish-rack basket, white sparks)) |
| B5 | H1-B THE ROAR | 1360-1372 | 18-18 | RENDERED | h1_v3 1476-1487 (H1 v3 master timing, pre-H5 look (beanie, dish-rack basket, white sparks)) |
| B5 | R2-B THE REVEAL | 1372-1520 | 18-19 | SLATE | expects renders/reveal_B, renders/vigil_B |
| B6 | R9 THE VIGIL · FIRST HOURS | 1520-2000 | 20-25 | SLATE | expects renders/vigil_B |
| B7 | R9 THE VIGIL · NOTHING ANSWERS | 2000-2480 | 26-31 | SLATE | expects renders/vigil_B |
| B8 | R9 THE VIGIL · SOMEONE HAS SEEN | 2480-2640 | 32-33 | SLATE | expects renders/vigil_B |
| B9 | R9 THE VIGIL · THE FIRST ANSWERS | 2640-3120 | 34-39 | SLATE | expects renders/vigil_B |
| B10 | R9 THE VIGIL · FIRE ANSWERS FIRE | 3120-3280 | 40-41 | SLATE | expects renders/vigil_B |
| B11 | R9 THE VIGIL · THE WORLD COMES | 3280-3600 | 42-45 | SLATE | expects renders/vigil_B |
| B12 | R9 THE VIGIL · THE CHILD | 3600-3840 | 46-48 | SLATE | expects renders/vigil_B |
| B13 | R11 THE HAND-BACK | 3840-5200 | 49-65 | SLATE | expects renders/handback_B |
| B14 | X3 TITLE | 5200-5440 | 66-68 | SLATE | expects renders/handback_B, renders/dawntitle_B |

**C** (7200 f): 1850 f picture (26%), audio: click track (no score or fallback yet)

| sec | shot | frames | bars | status | source / expected folder |
|---|---|---|---|---|---|
| C1 | C1 BLACK · THE HEARTH | 0-80 | 1-1 | EDIT (black) | edit |
| C2 | P1 THE RED BOOK | 80-320 | 2-4 | SLATE | expects renders/book_C |
| C3 | P2 INK PAGE · THE MOUNTAIN | 320-560 | 5-7 | SLATE | expects renders/book_C |
| C4 | E15 · X1 LETTERS TO FIRE | 560-880 | 8-11 | RENDERED | edit/cache/x1_letters_C_test.mp4 0-319 (MAP-v3 X1 motion test (half-res mp4, C 560-906)) |
| C5 | E15 THE FIRE, ALONE | 880-1040 | 12-13 | PARTIAL 27/160 | edit/cache/x1_letters_C_test.mp4 320-479 (MAP-v3 X1 motion test (half-res mp4, C 560-906)) |
| C6 | E5-C THE FORGING | 1040-1440 | 14-18 | RENDERED | embers_C3_half 1040-1439 (EMBERS v3 half-res preview, pre-H5) |
| C7 | E11 THE RACE UNDER THE RING | 1440-1680 | 19-21 | RENDERED | embers_C3_half 1440-1679 (EMBERS v3 half-res preview, pre-H5) |
| C8 | P2 INK PAGE · THE DEEP | 1680-1920 | 22-24 | SLATE | expects renders/book_C |
| C9 | E12 THE EYE (burn-through) | 1920-1992 | 25-25 | RENDERED | embers_C3_half 1920-1991 (EMBERS v3 half-res preview, pre-H5) |
| C9 | E12 THE EYE ONTO NOTHING | 1992-2080 | 25-26 | RENDERED | embers_C3_half 1992-2079 (EMBERS v3 half-res preview, pre-H5) |
| C10 | M4 THE MIRROR (optional) | 2080-2320 | 27-29 | SLATE | expects renders/mirror_C |
| C11 | E8-C THE GRASP THAT CANNOT HOLD | 2320-2480 | 30-31 | RENDERED | embers_C3_half 2320-2479 (EMBERS v3 half-res preview, pre-H5) |
| C12 | C12 BLACK · THE OLD LAW | 2480-2720 | 32-34 | EDIT (black) | edit |
| C13 | E13 THE RING FALLS | 2720-2840 | 35-36 | SLATE | expects renders/embers_C3, renders/embers_C3_half |
| C13 | R13 THE RING FALLS · THE STAR | 2840-2960 | 36-37 | SLATE | expects renders/ringfall_C, renders/run_C |
| C14 | H1-C FLINT | 2960-3000 | 38-38 | RENDERED | h1_v3 1216-1255 (H1 v3 master timing, pre-H5 look (beanie, dish-rack basket, white sparks)) |
| C14 | H2 THE FIND | 3000-3080 | 38-39 | SLATE | expects renders/ring_C, renders/montage3d_v3/find_a |
| C14 | H2 THE FIND · THE VISION | 3080-3150 | 39-40 | SLATE | expects renders/ring_C, renders/montage3d_v3/find_b |
| C14 | H1-C FLINT · THE CATCH | 3150-3360 | 40-42 | RENDERED | h1_v3 1266-1475 (H1 v3 master timing, pre-H5 look (beanie, dish-rack basket, white sparks)) |
| C15 | H2 THE FIRE TEST | 3360-3600 | 43-45 | SLATE | expects renders/ring_C, renders/montage3d_v3/fire |
| C16 | R2-C THE REVEAL | 3600-3840 | 46-48 | SLATE | expects renders/runC_reveal |
| C17 | R12 THE LIVING INK RUN | 3840-4160 | 49-52 | SLATE | expects renders/runC_scroll |
| C18 | P4 THE MAP ANSWERS · THE ROAD | 4160-4480 | 53-56 | SLATE | expects renders/map_C |
| C19 | AC1 THE COUNCIL | 4480-4800 | 57-60 | SLATE | expects renders/accord_C |
| C20 | AC4 BRING OUT THE RING | 4800-5120 | 61-64 | SLATE | expects renders/accord_C |
| C21 | AC2 THE BEARER | 5120-5360 | 65-67 | SLATE | expects renders/accord_C |
| C22 | AC3 · H3 THE UNMAKING | 5360-5520 | 68-69 | SLATE | expects renders/ring_C, renders/montage3d_v3/melt |
| C23 | AC THE FIRE REMAINS · THE STONE | 5520-5600 | 70-70 | SLATE | expects renders/accord_C, renders/fireremains_C |
| C23 | P4 · X1 THE FIRE REMAINS · ROADS OF FIRE | 5600-5680 | 71-71 | SLATE | expects renders/map_C, renders/book_C |
| C24 | R15 THE ILLUMINATION | 5680-6160 | 72-77 | SLATE | expects renders/runC_illum |
| C25 | P2 THE YEAR OF PLENTY | 6160-6400 | 78-80 | SLATE | expects renders/book_C |
| C26 | P2 THE HAVENS | 6400-6720 | 81-84 | SLATE | expects renders/book_C |
| C27 | P1 THE LAST PAGES | 6720-6960 | 85-87 | SLATE | expects renders/book_C |
| C28 | X3 TITLE | 6960-7200 | 88-90 | SLATE | expects renders/book_C |

<!-- COVERAGE:END -->
