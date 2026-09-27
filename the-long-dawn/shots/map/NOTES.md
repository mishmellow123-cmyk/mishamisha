# >>> MAP-L PAUSE STATE (27 Sep ~16:40 local / ~20:40Z; usage gap to ~23:30Z) -- READ FIRST ON RESUME <<<
* JOB READY sent for `cloud/jobs/map_v3_book.json` (2,472 frames -> renders/book_C + book_C_matte, C numbering; C2,
  C3, C4-C5 page only, C8+C9 burn, C22 ink-ring fallback, C25-C28 with the C28 title). Command once approved:
  `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/map_v3_book.json --nodes 3` (~30-40 min).
  If it was approved and launched before the gap, check on resume: `farm.py status`; then `--missing` for any holes.
* Check stills (all passed): renders/_farmtest/map_v3_book/book_C (request 61819, commit 3461166).
* ON RESUME: (1) verify the delivery: count frames 80-319, 320-559, 560-1039, 1680-1991, 5360-5519, 6160-7199 in
  renders/book_C and _matte; spot-check C3's riffle (335), C4's leaf turn (574), the C22 fall (5366-5374) and C26's
  leaf turn (6400-6434) for motion-blur artefacts; (2) contact sheet of every 40th frame -> review; (3) nitpicks for
  a cheap re-render: the initial's ink diaper too faint on the red (redbook.initial: lattice width 0.0085 -> ~0.012,
  dens 1.0); the Plenty plate dim in the low hearth (as scripted; could lift the C25 light a touch in
  book_c.last_pages_at); (4) EMBERS-C's e15 over the C4-C5 page layer: check the comp in EDIT's animatic.
* Owned files: book.py, book_c.py, burn.py, pages.py (Mountain, Deep, Plenty, Havens), pen.py, redbook.py,
  ringpage.py, titleburn.py; job map_v3_book. The MAP (road.py etc.) is MAP-L2's.

# >>> MAP-L STATE (27 Sep ~16:25 local): check stills reviewed, fixes pushed, JOB READY next <<<
* Pushed: book pass 1f5d629; C22 fallback flames 3439334; C3 Mountain 7b60e0b; check-still fixes 93ba9ac.
* Farm check stills (12, request 0927-154521, commit 3083493) in renders/_farmtest/map_v3_book/book_C, reviewed at
  reduced size and in full-res crops. PASS: C2 drift/ship/sheaf; C3 the great cone on a huge base; C4 glow
  (page only); C5 the hole onto black (page only, EMBERS-C's fire goes in it); C8 the Deep's thick gilt seam;
  C26 the bound-hand roundel (reads as a linen-bound fist raising an oil lamp) and the watch-tower; C27 blank;
  C28 the title burning (7000) and in ink (7060). FIXED after review (93ba9ac): the initial's letter read as a
  gold tick in a checkbox ('wc' -> 'lp'); Plenty's limbs read as umbrella spokes from one point (now staggered up a
  central leader, curving out then up) and its gold was lost in the low light (bigger, more racemes).
* Farm request 0927-161846-mapv3book-61819 (FARM re-queued at the tip) re-renders the 12 stills with the fixes.
  After a look at 175 and 6380: JOB READY map_v3_book (2,472 frames).
* Contract notes: C4-C5 page only (EMBERS-C: renders/embers_C3_e15, EDIT adds); C28 title in book_C (EDIT drops
  its C title overlay; director informed EDIT).

# >>> STATE AT HANDOFF (MAP-L takes over, 27 Sep ~18:45Z local; MAP-v3's cloud session is unreachable) <<<
* Imported from `claude/v3-map` (bfe5d4e) by explicit path onto `claude/long-dawn-v2` (f3aa4ee): all of shots/map's
  book/page/X1/road/ring-page engines, both held jobs and EMBERS' `review/v3/x1_letters.json`. Review sheets are in
  `~/mishamisha/_local_logs/review/map_v3/` (not in the repo). Nothing of MAP-v3's was ever rendered: renders/book_C and
  renders/map_C (v3) do not exist yet; both jobs are still HELD.
* BLOCKER FOUND ON IMPORT: C18 and bar 71's map was our REAL EARTH (her beacon on Everest, the council ring at the Pamir
  knot; India, Arabia, the Gulf, the Caspian legible at 4300-4440). Breaks H5 #2 and the north star. **SPLIT APPROVED by
  the director:** MAP-L2 rebuilds the MAP on an invented continent (C18 THE ROAD, bar 71 THE ROADS OUT; road.py incl.
  its two map X1s, relay.py, render.py, geo/features/rivers/webmap/answer/bake/sheet; job map_v3_road). fire.py and
  ink.py are shared: small, noted commits only.
* **MAP-L owns the BOOK:** C2 P1 THE RED BOOK; the P2 pages C3 MOUNTAIN, C8 DEEP, C25 PLENTY, C26 HAVENS, C27 LAST
  PAGES; the PAGE side of C4 LETTERS TO FIRE and the book X1s (C4's burn onto black, C8's sweep, C9's burn-through,
  C27's healing edges); EMBERS-C owns C4's particles and fire; C22's ink-ring fallback; C28 the title burn-on, in book
  space (EDIT's flat title overlay for C to be dropped). Files: book.py, book_c.py, burn.py, pages.py, pen.py,
  redbook.py, ringpage.py; job map_v3_book.
* H5 calls already applied by MAP-v3 (verified on the sheets): no "Red Book" on screen; one inscription (assets/ring);
  the new book hand; X1 notes; the ink-ring fallback; the Deep's seam; the swan ship; a gilt initial; edge striation;
  no dotted route on the Mountain; the Road as a pen line; hand-inked beacons.
* MAP-L's own review of the inherited sheets (still to do): the gilt initial reads as a framed gold diamond, not an
  illumination; the Plenty crown reads as cotton balls; the Havens roundel reads as a gesture, not her bound hand; the
  Mountain is a near-perfect cone (H5 "perfect triangle peak"); the C22 ink-ring fallback's flames read as paper
  cut-outs. C4 needs a page-only layer for EMBERS-C.

# MAP-v3 (lane `map`): THE RED BOOK, the four ink pages, the burn-throughs, THE MAP ANSWERS

## STATE (paused 27 Sep 16:35Z for the usage window; resume ~20:00Z)
**Complete and pushed** (d74bd7b on `claude/v3-map` and `claude/map-red-book-deep-yurk7k`, rebased on
`claude/long-dawn-v2`): every H5 call in the lane, both script notes and the X1 notes (log below); review sheets in
`review/v3/` (`map_X1_letters_to_fire.jpg/.mp4`, `map_C22_ring_page.jpg`, `map_C23_fire_remains.jpg`,
`map_C18_road.jpg`, `map_P1_redbook_set.jpg`, `map_P2_pages_pencil_ink.jpg`, `map_P2_pages_scans.jpg`);
`review/v3/x1_letters.json` re-exported. **No renders are running**, locally or on the cloud: both jobs wait for the
director (`cloud/jobs/map_v3_book.json` 2472 frames incl. the C22 fallback 5360-5519; `cloud/jobs/map_v3_road.json`
400 frames incl. bar 71 5600-5679, plus the two X1 layers).
**Next (on resume, in order):**
1. `git pull --rebase origin claude/long-dawn-v2`; read any new director notes at the top of `BIBLE_V3.md`.
2. If the jobs have run: check `cloud_logs/map_v3_book_status.txt` / `map_v3_road_status.txt`, spot-check shipped jpgs
   (C4 streams 740-790, the C22 fall 5365-5374, bar 71's hearths, C18's end on the ring).
3. Polish, if time allows: a rounder bead (C22), the fall visible a little longer, bar 71's roads kept off the small
   hill glyphs; a half-res playback check of C22 and bar 71 (only stills so far).
4. Re-run `python3 the-long-dawn/cloud/verify_session.py`; push to both branches; 5-line summary.

## H5 LOG (27 Sep ~15:50Z) · the director's H5 CALLS, the two script notes and the X1 notes: all applied
- [x] **C is the book:** the words "Red Book" appear nowhere on screen (no titling is drawn in any page or plate; checked).
- [x] **One inscription on both rings:** the Ring drawn in the book carries the canonical script of fire from `assets/ring`
  (MONTAGE-3D, rev e7083c6), outer face and inner face (`ringpage.strip`).
- [x] **THE BOOK HAND** (script note): a human calligraphic hand of its own, and deliberately not the Ring's. `pen.BOOK`:
  16 unjoined letters built from the lying wave, the open wedge, the low heavy fall, the feather's fork, the small coil,
  the lozenge and the nib's own lozenge dot; no stems, bowls, arches, cups, hooks or accent marks; no two waves or two
  wedges side by side (they read as m/n). Nothing reads as l, u, y or any Latin letter, nor Tengwar, runes or the Ring's
  teardrops (specimens checked at glyph size and at page size). Every page, the C4 leaf and the gilt initial use it.
- [x] **X1 letters to fire** (director's notes on f560-906): (1) the script glows AS LETTERS first, in the Ring's fire
  colour (deep orange-red core, never white; `book.py` fire channel), legible; a peel then runs in from the rim, word by
  word and stroke by stroke (each stroke flares and goes out as its sparks leave it), and the sparks are drawn to the
  page's heart by a draught: they descend a gently uneven slope (distance plus a smooth noise, turned only sideways), a
  flow with no turning in it (so never a spiral), whose streamlines gather into curving streams that merge like rivulets
  and quicken near the heart; trails 0.035 s, capped at 0.28 cm (`Kindling._fly`). No dot field at any frame. (2) The
  catch is ONE flame, gold and calm (`fire.flame` gains an additive `fray` kwarg; cool -0.22). (3) The burn-through onto
  black with the fire in the hole is untouched. Motion test re-rendered (half res, 560-906).
- [x] **THE RING CLOSE-UP FALLBACK = the ink ring on the page** (`ringpage.py`; frames via `book_c.py`, shot
  `ring_melt`, C22 5360-5519 -> `renders/book_C`): a band of gold leaf drawn in pen in three-quarter view, the inscription
  cut in it, hatching that follows the form, on a page by a drawn hearth fire (living ink tongues with gold laid in them,
  an ember bed). 5366 the Ring drops in from above the frame, crisp (eight frames, a nine-sample shutter); it slumps in
  the heat, darker; 5420 its letters flare once, still letters; 5440 they go out; the gold runs forward (the hole slides
  back to a slit and closes, never a round hole: no donut) into a bead; 5490 the hearth flares to white by 5519. Stills
  for the find and the fire test (`python3 ringpage.py stills`; letters awake and unmarked in the fire).
- [x] **THE DEEP:** the gold seam thicker, tapered, gilt and branching. **THE HAVENS:** the ship redrawn swan-prowed at the
  plates' line weight. **One gilt initial** (the last spread, a letter of the new hand in gold leaf, penwork tendrils).
  **Page-edge striation.** **Mount Doom:** the dotted route dropped (the door stays).
- [x] **THE ROAD on the map:** a fine continuous pen line that grows as the pen goes (a swell and thinning of pressure, the
  last half-degree wet, a bead of ink at the nib); the ring of stones drawn as stones. **C's beacons hand-inked** at the
  terrain's line weight with shell gold laid in and a live flame the glyph's own size inside; pools cut to 0.4, no halo.
- [x] **C23 THE FIRE REMAINS, bar 71** (5600-5679 -> `renders/map_C`): out of the council fire on the ring's stone the
  roads run outward (a shortest-path tree over the land from the ring to 22 lowland hearths: 5 trunks forking round the
  ranges by the passes), a small moving flame at every road's head; a hearth glyph kindles where each arrives, gold laid
  in it; the war-beacons settle low and the room warms; a breath dim at 5668-5676 before the sunrise. The X1 from
  ACCORD's plate: `road.py x1 --center 960,402 --frames 5594-5640` (opens 5600, centred on the council fire).

## REPORT (MAP-v3, updated 27 Sep ~15:50Z) · on the LOCKED bar map (`music/v3/barmap_C.json`)
**What it is.** C's book as one 2.5-D engine, all shots numbered on C's own timeline (src frame = C frame):
| shot | C frames | what happens | code |
|---|---|---|---|
| C2 THE RED BOOK | 80-319 | the drift over the last written leaves (the book hand, a mountain, a ring and a ship drawn small, the one gilt initial); from bar 4 b1 the blank recto and the striated sheaf of leaf-edges | `book_c.shot_red_book` |
| C3 THE MOUNTAIN | 320-559 | the leaves riffle back (12 leaves, motion-blurred); the pen draws the fire, the gilt ring on bar 5 b3, then the mountain round it (no route); caption band left clear for T1 | `shot_mountain`, `pages.Mountain(SCHED_C3)` |
| C4+C5 LETTERS TO FIRE (X1a/E15) | 560-1039 | the leaf turns to a leaf dense with the book hand; the hearth sinks; 9b4 the letters glow as letters in the Ring's fire colour; 10b1 the peel runs in from the rim and the sparks stream down to the page's heart in curving, merging streams; 11b1 one calm gold flame catches; 11b3 it burns the page open; C5 the fire alone | `Kindling`, `burn.hold_params` |
| C8 THE DEEP (+C9 burn) | 1680-1991 | an ember edge sweeps the race away (X1 reversed); the pen draws the halls level by level on the dividing tick down the thick, branching gilt seam; 1920 the glow burns through (into the Eye) | `shot_deep`, `burn.sweep_params` |
| C18 THE MAP ANSWERS · THE ROAD (P4) | 4160-4479 | map_C rev 3 retimed to four bars, the beacons now hand-inked flame glyphs with gold laid in; THE ROAD a fine pen line drawn from her beacon's glyph west to the ring of stones, arriving on bar 56 b3; ends centred on the ring for the match to C19; `road.py x1` burns through onto the map | `road.py` |
| C22 (FALLBACK) THE UNMAKING as the ink ring | 5360-5519 | the Ring drawn on the page drops into a drawn fire, slumps, its letters flare (5420) and go out (5440), it runs into a bead, white by 5519 | `ringpage.py`, `shot_ring_melt` |
| C23 THE FIRE REMAINS, bar 71 | 5600-5679 | the burn-through from the council plate onto the map; the roads run out from the ring, a small flame at each head, and hearths kindle where they arrive | `road.FireRemains` |
| C25-C28 THE LAST PAGES | 6160-7199 | THE YEAR OF PLENTY draws itself (the tree alone, the smoke of many hearths); 81b1 the leaf turns: THE HAVENS (coast fires, the swan-ship, the roundel of her bound hand raising the lamp; 84b1 the ship slides west); blank leaves; the edges heal by T14; the blank recto for the title | `shot_last_pages` |

**The risk test: PASSED against its fallbacks** (unchanged). This pass: the X1 motion test re-rendered and checked frame to
frame (no pops; the only large differences are the motion-blurred leaf turn at 574-579 and the densest streams at 761-766,
smooth); the ring fallback checked on its beats; bar 71 and C18 checked as stills.
Sheets (`review/v3/`): `map_P1_redbook_set.jpg`, `map_P2_pages_pencil_ink.jpg`, `map_P2_pages_scans.jpg` (the four pages
flat, and the new hand), `map_X1_letters_to_fire.jpg` + `map_X1_letters_to_fire.mp4`, `map_C22_ring_page.jpg`,
`map_C18_road.jpg`, `map_C23_fire_remains.jpg`, `map_C_book_timeline.jpg` (older timeline sheet).

**RENDER_SPEC** (director launches; one machine type per job, 4 procs x 1 thread, `"ship": "jpg"`):
* `cloud/jobs/map_v3_book.json`: 2472 frames -> `renders/book_C/` + `renders/book_C_matte/` (80-319, 320-559, 560-1039,
  1680-1991, 5360-5519, 6160-7199), ~3 s/frame on 4 threads (~12 s per 1-thread proc): ~2 h 15 on one 4-core box. The
  C22 fallback builds a 130 px/cm page texture per frame (+~1 GB RSS while that shot renders; the fall's eight frames
  take nine shutter samples each). EMBERS' handoff `review/v3/x1_letters.json` regenerated for the new sparks
  (`python3 book_c.py export --out DIR`).
* `cloud/jobs/map_v3_road.json`: 400 frames -> `renders/map_C/` (4160-4479 and 5600-5679) + two X1 layers
  (`renders/x1_map_C/` 4150-4185 at 960,300; `renders/x1_map_C71/` 5594-5640 at 960,402, each with its `_matte` keep).
  Setup bakes the sheet if the cache is missing (~16 min on 4 cores); frames ~2-4 s.
* Comp (EDIT): book layer `out = rgb + (1 - matte) * next shot`. The map X1s: `out = plate * keep + map * (1 - keep) +
  glow` (C18: the Run's seventh beacon over the map; C23: ACCORD's council plate over the map). The C22 fallback is a
  full-frame book shot ending in white (bar 70 opens in white).

**Weaknesses (next pass).** The bead is a drawn dome and could be rounder; the ring's fall is fast (eight frames by
design) and only its last four are fully in frame; the Plenty crown still reads a little like cotton balls; the Havens
roundel reads as a gesture more than a hand; the book hand has no ascenders (a band texture; accepted, it is not Latin);
bar 71's roads are pen lines over the terrain glyphs (they go round the ranges by the passes, but cross small hills);
motion is judged on the X1 mp4 and on stills for the rest.

# MAP (cut C) — THE WORLD ANSWERS, told on a map

## Revision 2 (2026-09-26): fire, not fibre — per critic_tone.md §B2 / critic_framing.md m8, m9

The ending no longer becomes a network diagram:
* **No new edges after 2010** (`answer.T_CUT`), and no loop-closing cross-links at all
  (`MapWeb.cut`): the burn up to 2010 is a branching tree out of the Himalaya (411 threads,
  412 beacons), not a Delaunay lattice.
* **Burnt threads fade to scorch:** the hot front cools to embers that die away (τ≈38 frames);
  no sustained glow, no travelling pulses. What is left is brown scorch trails in the paper.
* **The peoples answer as fire** (`answer.py`): 631 beacons catch in chains along the
  inked ranges (a flame on the summit of a drawn peak), down the great rivers, on headlands
  and on a few hills, following a minimum spanning tree that prefers same-kind neighbours, so fire
  runs along a ridge or a shore like a line of signal fires. No lines are drawn. The answer
  starts on every continent at once from 2011 (spontaneous seeds everywhere, plus catches from
  the burnt web where it touches), so no region is visibly last (m8). All lit by ~2057.
* **End state:** flames of varying size (peaks biggest, sized by the drawn peak; river and
  shore fires smaller) with a screen-size floor so they still read as flames at the wide
  framing. Relay beacons of the first half that don't stand on a height, shore or river burn
  down to dull embers. The scorch trails remain between the older fires. The compass rose is
  kept.
* **Caption window** is now 1960–2040 ("And all the peoples answered."): the fire and fire-light
  in the lower-third band are hushed from 1950 to ~2050 (no stripe; soft edges).
* The previous delivery is kept in `renders/map_C_prev_v1/`.


v2 frames **1920–2087**, 1920×804 → `renders/map_C/f_%05d.png` (v2 numbering; the edit dissolves
to ACCORD over 2072–2087). Cut C only.

Our real Earth drawn as a hand-inked map in the Tolkien tradition, lying on a table in a dark room
lit by a hearth. At 1920 a small flame on the high Himalaya flares, lighting the inked peaks around
it; threads of fire leave it slowly, then faster and faster, burning across the parchment. Each
thread is a white-gold front with a licking flame that cools to granular embers and leaves a
scorched brown line. Every beacon it reaches catches as a small flame with its own pool of light.
Sparks leap the oceans above the paper and leave dotted sea-routes of embers. The camera starts
low and close over the Himalaya with a shallow, table-top depth of field. It cranes up and back,
slides west and tilts down until the whole sheet is in frame (border, compass rose, the table's
edge). As the world is laced with light, the hearth burns down, so at the end the continents glow
by their own fire and the light gathers to the centre of frame for the dissolve.

## Re-render

```
cd shots/map; source ~/.venvs/longdawn/env.sh; export NUMBA_NUM_THREADS=2
python geo.py; python features.py; python webmap.py        # caches (idempotent, ~1 min)
python bake.py pyramid full 48                               # the sheet: ~20-25 min, 2 procs
python render.py 1920 2003 --tag full &  python render.py 2004 2087 --tag full
python render.py --frames 1920,1960,2000,2040,2080 --scale 0.5 --test --tag full   # stills
```

Caches live in `renders/map_C/cache/` (level 0 of the sheet is a 18624×9275 uint8 memmap,
~520 MB; the renderer reads only the crop it needs). Every frame is a pure function of its frame
number.

## Files

* `geo.py` — the projection: central meridian 11°E, so the seam runs down the Bering Strait and
  no continent is cut. y = φ(1 + 0.08 φ²) sits between plate carrée and Miller, and the map spans
  lat −60…85 (aspect ≈ 2.18). Also the NE 50m land rings (with the Caspian hole) and the coarse
  fields. **Elevation** is a Frankot–Chellappa integration of `assets/textures/earth_normal_2048.jpg`,
  referenced to sea level. **Forest, desert and lakes** are classified from NASA Blue Marble (the
  GLOBE cache copy, public domain); no imagery is shown, it is only used to decide where to draw.
* `features.py` — where every mark goes and how it is drawn:
  * **Peaks** sit on range crests (Hessian ridge strength of the elevation at 1.2°) and range
    fronts (coarse slope), so the Himalaya, Karakoram, Kunlun, Tian Shan, Andes, Rockies, Alps,
    Zagros, Caucasus, Atlas, the East African and Ethiopian highlands and so on appear as rows of
    peaks, and plateaus like Tibet stay open paper.
  * Each **peak** glyph is a sharp, twin or blunt profile with a ridge line, fall-line hatching
    on the east face and a soft sepia shadow wash. **Hills** are sparse hatched arcs. Forests are
    **round-crowned trees** in the tropics and temperate zones and **conifers** in the boreal belt.
    Deserts get **stipple**.
  * Glyphs are drawn in painter's order (north first), each one erasing the ink under its
    silhouette.
  * **Coasts** are hand-wobbled with pressure-varied ink. Lakes (Great Lakes, Victoria, Baikal…)
    and about 80 major rivers are traced by hand (`rivers.py`).
* `sheet.py` — the parchment, all functions of map position so any tile at any resolution agrees:
  warped tone, blotches, fibres and tooth, tide-mark stains, foxing, three fold creases, and an
  aged, irregular, bitten edge with the table beyond it. Also the frame furniture: a double outer
  rule, a zebra degree band (every 5° of latitude and longitude), corner blocks, and a
  sixteen-point compass rose in the South Pacific. **There is no lettering anywhere.**
* `bake.py` — the coastal ink wash and water-lining (four offshore ripple lines from the distance
  field, the outer ones broken); ink composition (ragged edges from the paper tooth, pen
  pressure, wear on the folds); and the tiled bake of the mip pyramid plus a low-res relief map.
* `ink.py`, `noise.py` — numba anti-aliased variable-width capsule strokes; gradient-noise fBm.
* `webmap.py` — the web, adapted from GLOBE's `web.py`:
  * 797 beacons: Everest, GLOBE's curated heights, Natural Earth cities (≥200k, thinned) and
    relief-weighted hill fillers. 1027 threads. The graph is a planar Delaunay graph that never
    crosses the seam, and short straits are allowed.
  * 29 designated ocean **leaps** (Atlantic narrows, Iceland–Greenland, Ireland–Newfoundland,
    Canaries–Caribbean, Java–Darwin, Sydney–Auckland, SF–Hawaii, Santiago–Easter–Tahiti, the
    Indian Ocean islands…).
  * Same seeded spread as GLOBE (2–3 children per beacon, angular diversity, late answers,
    cross-links), but run in travel time and remapped to frames by frame = 1920 + k·s^p (p≈0.39).
    So the first generation is slow and heavy and the rest accelerates with no seams:
    * threads leave the first beacon at 1926.5, 1937 and 1943 and land at 1946, 1951 and 1954;
    * half the world is lit by ~2011;
    * the Atlantic is crossed ~2015–2030;
    * 99% is lit by 2052 and the last beacon at ~2055.
* `fire.py` — numba screen-space fire: Gaussian-profile lines (max within a polyline, additive
  between polylines), points, and a procedural **living flame**. The flame is a warped teardrop
  whose top frays into licking tongues, with a yellow heart, orange flanks and a red tip; it
  animates per frame.
* `render.py` — the shot:
  * **Camera:** a perspective camera over the table plane, sampled through its homography from
    the mip pyramid (per-pixel trilinear LOD). Keys run from 42 map-degrees wide at 46° tilt over
    Everest to 368 wide at 13° over the sheet's centre.
  * **Light:** a flickering hearth pool composed in frame (from the upper left early, gathering
    to centre at the end), raking over the paper relief, plus a cool moonlit fill. Fire light
    from every flame and fresh thread is splatted in a map-space grid and spread by a
    sum-of-Gaussians "light pool" kernel. The first beacon has its own flare light.
  * **Threads:** each has a fire front with a small flame and sparks; embers that are uneven
    along the line and breathe; now and then a pulse of light runs along a settled thread;
    scorch plus a singed halo are multiplied into the paper.
  * **Leaps:** comet sparks arc above the paper, light the sea beneath them, and leave dotted
    ember routes that scorch.
  * **Other:** ignition flashes and spark bursts; a steady spark stream and the flare's
    fountain from the first beacon; tilt-shift DoF while the camera is low (to ~1990).
  * **Finish:** `look.finish` (exposure 1.5, bloom 0.075, vignette 0.3 rising to 0.55 for the
    dissolve).

## Beats (v2 frames)

| frame | event |
|---|---|
| 1920 | cut in: close over the Himalaya, a small flame burning on the high peaks |
| 1920–1923 | the flare: the flame leaps up, its light floods the peaks, a fountain of sparks |
| 1926–1954 | the first three threads creep out (launch 1926.5 / 1937 / 1943), heavy and slow |
| 1960–2040 | text window: the lower-third band is hushed (fire and fire-light ×0.3, soft edges) |
| 1960–2010 | the web explodes across Asia, into Europe, Arabia, Africa and Indonesia; the camera cranes up and slides west |
| 2010 | the last threads are launched; after this no new lines — burns cool to scorch |
| 2011–2057 | the peoples answer: chains of beacons along the ranges, rivers and coasts of every continent |
| 2040–2055 | the last beacons; the whole sheet in frame; the hearth burns down |
| 2057–2087 | fire on every range and shore, scorch trails between the older fires; light gathers to centre, vignette deepens (dissolve to ACCORD 2072–2087) |

## Deviations / choices

* **No text in frame** (none on the map either). The text band is hushed only during 1935–2040.
* The web is on a flat sheet, so it is not GLOBE's spherical graph. Seam-crossing Pacific leaps
  were replaced by leaps reached from the Americas and Australia (SF→Hawaii, Santiago→Easter→
  Tahiti, Brisbane→New Caledonia→Fiji, Manila→Guam).
* The hearth light is composed in frame rather than fixed in the room, so every framing has a
  pool and falloff. It dims through the second half so that at the end the map is lit by its
  own fires.
* Antarctica is outside the map (south of 60°S); the zebra border marks 5° steps of the map's
  own latitude scale.

## Delivery (2026-09-26)

* 168 frames `renders/map_C/f_01920.png` … `f_02087.png` (1920×804, `look.finish` + `save_png`).
* Render: mean **3.9 s/frame** (rev 2) at full res (2 processes × 1 numba thread on the shared machine);
  sheet bake 963 s (91 tiles, 2 procs). Peak RSS ~0.6 GB per render process.
* Review sheet: `~/mishamisha/_local_logs/review/map_C.jpg` (16 frames). Test stills from
  development: `~/mishamisha/_local_logs/review/map_tests/`.
* The dissolve was previewed in linear light against ACCORD src 1912–1927 (v1 frames as a
  stand-in for `accord_C`): the lace of beacons melts into the torch rivers converging on the ring.

## Known weaknesses / next passes

* At the whole-sheet framing (2045→) the ink reads mostly as tone and the web as points joined by
  lines — handsome, but closer to a "network" than the close-ups. Larger, flame-shaped beacons
  there, or a gentle push-in during 2060–2087, would keep it more hand-made.
* The flare peak (~1922–1926) briefly blooms the first flame toward white.
* The parchment is essentially one warm hue; firelight vs moonlit fill gives only mild colour
  separation.
* Geography is honest but approximate: peaks come from a relief proxy (normal-map integration),
  forests and lakes from a Blue Marble classification, and rivers are hand-traced waypoints (~1°).
* Motion was checked on stills and short frame runs, not on playback; watch the preview for
  flicker in the finest hatching during the fast part of the crane (1985–2030).
