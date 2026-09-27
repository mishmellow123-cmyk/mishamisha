# MAP-v3 (lane `map`): THE RED BOOK, the four ink pages, the burn-throughs, THE MAP ANSWERS

## REPORT (MAP-v3, 27 Sep 05:25Z) · on the LOCKED bar map (`music/v3/barmap_C.json`)
**What it is.** C's book as one 2.5-D engine, all shots numbered on C's own timeline (src frame = C frame):
| shot | C frames | what happens | code |
|---|---|---|---|
| C2 THE RED BOOK | 80-319 | the drift over the last written leaves (the flowing invented hand, a mountain, a ring and a ship drawn small); from bar 4 b1 the blank recto and the sheaf of leaf-edges | `book_c.shot_red_book` |
| C3 THE MOUNTAIN | 320-559 | the leaves riffle back (12 leaves, motion-blurred); the pen draws the fire, the gilt ring on bar 5 b3 (1.62-1.66 s), its letters kindle faintly in the drawn fire, then the mountain grows round it; caption band left clear for T1 | `shot_mountain`, `pages.Mountain(SCHED_C3)` |
| C4+C5 LETTERS TO FIRE (X1a/E15) | 560-1039 | the leaf turns to a leaf dense with script; the hearth sinks until the page is almost dark; 9b4 the letters glow like the Ring's; 10b1 they lift as sparks drawn down together to the page's heart (no spiral); 11b1 the fire catches; 11b3 it burns the page open; C5 the fire alone, the burnt rim settling at the frame's edges | `Kindling`, `burn.hold_params` |
| C8 THE DEEP (+C9 burn) | 1680-1991 | an ember edge sweeps the race away down the frame leaving parchment (X1 reversed); the pen draws the halls level by level on the dividing tick (quarters, eighths from 1760, sixteenths from 1840) and the camera follows it down the gilt vein; the red glow grows, the paper browns; 1920 the glow burns through (into the Eye) | `shot_deep`, `burn.sweep_params` |
| C18 THE MAP ANSWERS · THE ROAD (P4) | 4160-4479 | map_C rev 3 (the relay of fires, from `claude/fix-map_fix`) retimed to four bars; THE ROAD: a slow dashed route from her beacon's glyph west along the ranges to a ring of stones at the Pamir knot, arriving on bar 56 b3; the camera ends centred on the ring for the match to C19; `road.py x1` gives the burn-through onto the map as glow + keep-matte | `road.py` |
| C25-C28 THE LAST PAGES | 6160-7199 | the hearth low, the edges scorched; THE YEAR OF PLENTY draws itself (no doors, no houses: the tree alone in the field, the smoke of many hearths beyond); 81b1 the leaf turns: THE HAVENS (coast fires 81b3, 82b3, 83b3; the stern light answers; the roundel: her bound hand raising the lamp; 84b1 the ship slides west off the page); 85b1 blank, 85b3 the next; the edges heal by T14; the blank recto held for the title | `shot_last_pages` |

**The risk test: PASSED against its fallbacks** (P2's "clip-art → pencil first, judged against map_C's close-ups"; X1's
"the letters glow and X1 burns through from their glow"). The Mountain and the Deep read as engraved pen drawings on a real
page (tonal sky and flank hatching, puffed smoke, carved dark halls down a gilt vein), not clip-art; pencil → ink sheets are
in `review/v3/`. The book set reads as an old red book by a hearth, and the letters-to-fire burn-through reads in 8 frames.
Sheets: `review/v3/map_P1_redbook_set.jpg`, `map_P2_pages_pencil_ink.jpg`, `map_P2_pages_scans.jpg`,
`map_X1_letters_to_fire.jpg`, `map_C_book_timeline.jpg`, `map_C18_road.jpg`; motion test `map_X1_letters_to_fire.mp4`.

**RENDER_SPEC** (director launches; one machine type per job, 4 procs x 1 thread, `"ship": "jpg"`):
* `cloud/jobs/map_v3_book.json`: 2312 frames `renders/book_C/` + `renders/book_C_matte/` (80-319, 320-559, 560-1039,
  1680-1991, 6160-7199), ~3 s/frame on 4 threads (~12 s per 1-thread proc): ~2 h on one 4-core box, ~1.2 GB RSS per proc.
  Page turns and the riffle are motion-blurred (5 and 3 shutter samples), which adds ~30 min. EMBERS' handoff is committed:
  `review/v3/x1_letters.json` (glow 700, lift 720, fire 800, burn 840; 1961 spark seeds in screen px with their lift and
  arrival frames; the heart's screen track and flame height per frame to 1039, where the fire stands at 961,426, 412 px
  tall: E5-C's first frame should match it). Regenerate with `python3 book_c.py export --out DIR`.
* `cloud/jobs/map_v3_road.json`: 320 frames `renders/map_C/f_04160..04479` + the X1 onto the map
  (`renders/x1_map_C/` glow and `_matte` keep, 4150-4185, centred at 960,300: change `--center` to the seventh beacon's
  bloom). Setup bakes the sheet if the cache is missing (~16 min on 4 cores); frames ~4 s.
* Comp (EDIT): book layer `out = rgb + (1 - matte) * next shot` (both display-referred). The next shots: C4/C5 → E5-C,
  C8's head ← E11 (the race shows ahead of the sweeping edge), C8's tail → E12. Lines are EDIT's: T1 in the Mountain's
  caption band (page y 12.8-14.2 cm, under the plate), T2 over the darkening dense leaf, T7 on the Deep, T14 on the first
  healed blank page, the title on the last blank recto.

**Weaknesses (next pass).** The Plenty tree's crown still reads a little like cotton balls; the Havens roundel's hand is
legible as a gesture more than as a hand; the flame is three tongues of `fire.flame` (EMBERS' fire may replace it); the
riffle is dark and fast (reads as motion, not as pages); the Deep's halls repeat one design per level; the invented hand
has a few Latin-looking letters (l, u, y) at small sizes; motion is judged on the X1 mp4 only (the drift, riffle and
descent were checked as frame sequences, not as playback).

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
