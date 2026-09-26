# MAP (cut C) — THE WORLD ANSWERS, told on a map

## Revision 3 (2026-09-26): hill by hill — per review/picture_redteam.md §12 and TOP 8 #1

The spread read as neurons (forked dendrite tips), missiles (comets over the sea) and a dot-density
world map. Now the unit of the spread is a flame, and the shot ends regional:
* **A relay of single flames** (`relay.py`, replaces `webmap.py` + `answer.py`): every fire lights ONE
  fire further along its line, on the summit of a drawn peak or the crown of a drawn hill, at an
  irregular distance (hops of 1.6–5.4 map degrees, so peaks are skipped and gaps vary). Nothing is
  drawn between fires: no threads, no fronts, no forked tips, no dots.
* **Lines along the real ranges**, each leaving the frame: west (Himalaya, Karakoram, Hindu Kush, Elburz,
  Anatolia), east (Hengduan, Qinling, Taihang, Korea, Japan), south (across the plain to the Eastern
  Ghats and Sri Lanka), the Western Ghats and Aravalli, Myanmar, Indochina, south China, the Tian Shan
  and Altai, Mongolia, Pakistan–Makran–Oman–Yemen–the Horn, the Zagros, the Caucasus, the Levant. Each
  line keeps to a corridor (±1.8°) around its hand-placed range polyline, so it reads as a line.
* **Branches are never forks:** a second line leaves an older fire only after that fire's own line has
  passed on two hops, and no fire except the first beacon passes the fire on more than twice
  (checked: 0 forks at a tip). The first beacon's three answers keep the old first landings
  (1946 west, 1951 east, 1954 south).
* **No flight over water** (`LEAPS` removed with `webmap.py`): a strait (Hormuz, Bab-el-Mandeb, Palk,
  Tsushima) is crossed by a pause and then a fire on the far shore.
* **The newest fires flare, the older ones settle:** a fire catches over ~2.5 frames, flares (taller,
  brighter, its light thrown wide on the paper) for ~6 frames, then burns down over ~24 frames to its
  own resting level, lower and redder (`fire.flame(..., cool)`). Across the settled fires brightness
  spans ~2.5:1 (p90/p10; ~4:1 extremes); the flaring ones stand above that.
* **Pace:** a line hops every ~6 frames once under way (slow and heavy first); eastern lines run a
  little slower and western a little faster, so the fire reaches every edge of the last framing at
  about the same time (catches in the left/right thirds of the frame: 7/8 over 2020–40, 9/7 over
  2040–60, 6/3 after 2060). No region is visibly last.
* **Regional end:** the camera cranes only from 42 to ~79 map degrees wide (was 42 → 372) and ends
  over the Himalaya, Iran, Arabia's south coast, India, Myanmar and western China, with the lines of fire
  running off every edge. At ~24 px per map degree the peak glyphs, their hatching and the flames stay
  legible. It keeps drifting back very slowly to the end (no hold, no push-in).
* **Grade:** the hearth burns down further (to 17% by ~2060) and the moonlit fill rises (×4.4), so at
  the end the unlit paper is a neutral moonlit grey and the only warm light is fire.
* **Light fix:** `LightGrid.irradiance` divided by an extra `cell²/(2πσ²)`, which made every fire's
  light pool ~1000× too weak and camera-dependent. It is now power per cell / cell area. The first
  beacon's pool is still its analytic flare light (`big`, +10% for its old grid share), so 1920–1945 is
  unchanged (mean difference 0.6/255 at 1920).
* **Hand-traced hill country** (`features.HILL_RANGES`): the relief proxy drew almost nothing in India,
  which would have left a whole people dark. 52 hill marks are set along the Western and Eastern Ghats,
  Vindhya, Satpura, Aravalli, Chota Nagpur, Sri Lanka's highlands and the Sulaiman/Kirthar, placed last
  with their own seed (every other mark is exactly where it was; 15 trees under them are dropped).
  **The sheet must be re-baked** (`bake.py pyramid`) after this change.
* The rev-2 delivery (whole-world web + answer) is superseded; keep it as `renders/map_C_prev_v2/` if
  wanted.


v2 frames **1920–2087**, 1920×804 → `renders/map_C/f_%05d.png` (v2 numbering; the edit dissolves
to ACCORD over 2072–2087). Cut C only.

Our real Earth drawn as a hand-inked map in the Tolkien tradition, lying on a table in a dark room
lit by a hearth. At 1920 a small flame on the high Himalaya flares, lighting the inked peaks around
it. After a held breath a summit to the west answers (1946), then one to the east, then a hill across
the plain to the south; from each, the fire passes on hill by hill along the drawn ranges, one flame at
a time, each catching with a flare and then settling to a steady, redder burn. The camera starts low
and close over the Himalaya with a shallow, table-top depth of field and cranes gently up and back
over the region, never further: the lines of fire run off the frame on every side. As the land fills
with fires the hearth burns down, so at the end the moonlit sheet is warm only where fire stands, and
the light gathers to the centre of frame for the dissolve.

## Re-render

```
cd shots/map; source ~/.venvs/longdawn/env.sh; export NUMBA_NUM_THREADS=2
python geo.py; python features.py                           # caches (idempotent, ~1 min)
python bake.py pyramid full 48                               # the sheet: ~20-25 min, 2 procs (6 min with 4)
python relay.py                                              # the relay's timing summary (cached on first render)
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
* `relay.py` — the relay (rev 3): sites are the summits of the drawn peaks and the crowns of the drawn
  hills; `ROUTES` are the great lines (parent line, branch point, waypoints along the ranges); an event
  simulation passes the fire on one hop at a time (`hop_delay`, `PACE`, straits, long looks across
  plateaus and deserts), with a few short side lines up to nearby peaks. Cached as `relay_5.npz`
  (delete it after changing the sites or the routes).
* `fire.py` — numba screen-space fire: Gaussian-profile lines (max within a polyline, additive
  between polylines; now only the spark streaks), points, and a procedural **living flame**. The flame
  is a warped teardrop whose top frays into licking tongues, with a yellow heart, orange flanks and a
  red tip; it animates per frame; `cool` lowers its temperature for a fire that has burned down.
* `render.py` — the shot:
  * **Camera:** a perspective camera over the table plane, sampled through its homography from
    the mip pyramid (per-pixel trilinear LOD). Keys run from 42 map-degrees wide at 46° tilt over
    Everest to ~79 wide at 30° over (lon 75°E, lat 26°N), heading −7° → −3°.
  * **Light:** a flickering hearth pool composed in frame (from the upper left early, gathering
    to centre at the end, burning down to 17%), raking over the paper relief, plus a cool moonlit fill
    that rises as the hearth dies. Each fire's light is splatted in a map-space grid and spread by a
    sum-of-Gaussians "light pool" kernel (wide while it flares, small once settled; the pools grow a
    little as the camera pulls back). The first beacon has its own flare light and soft glow.
  * **Fires:** every lit fire is a living flame on its summit (physical height, with a screen-size
    floor so it still reads as a flame in the wide), a light pool and a singe under it; a burst of
    spark streaks as it catches; the first beacon's steady spark stream and the flare's fountain.
  * **Other:** tilt-shift DoF while the camera is low (to ~1990).
  * **Finish:** `look.finish` (exposure 1.5, bloom 0.075, vignette 0.3 rising to 0.55 for the
    dissolve).

## Beats (v2 frames)

| frame | event |
|---|---|
| 1920 | cut in: close over the Himalaya, a small flame burning on the high peaks |
| 1920–1923 | the flare: the flame leaps up, its light floods the peaks, a fountain of sparks |
| 1946 / 1951 / 1954 | the first answers: a summit to the west, one to the east, then a hill across the plain to the south |
| 1960–2040 | text window ("And hill by hill, the peoples answered."): the lower-third band is hushed (fire and fire-light ×0.3, soft edges) |
| 1960–2020 | lines of fire run along the Himalaya, Karakoram, Hindu Kush, Tian Shan, Hengduan, the Ghats and Myanmar; the camera cranes gently up and back |
| 2020–2087 | the frontier reaches Iran, Oman, Yemen, the Caucasus, Mongolia, China and Indochina and runs off every edge; the fires behind it settle; the hearth burns down and the paper turns moonlit |
| 2072–2087 | dissolve to ACCORD: the light gathers to the centre, the vignette deepens |

## Deviations / choices

* **No text in frame** (none on the map either). The text band is hushed only during 1950–2050.
* **Nothing between the fires.** The critic suggested a brown scorch line behind each hop; it was left
  out on purpose: visible trails would show the relay's branching topology and bring the dendrite
  reading back. The chain is carried by the order of the catches along each range.
* The hearth light is composed in frame rather than fixed in the room, so every framing has a
  pool and falloff. It dims through the second half so that at the end the map is lit by its
  own fires.
* Antarctica is outside the map (south of 60°S); the zebra border marks 5° steps of the map's
  own latitude scale.

## Delivery (rev 3, 2026-09-26)

* 168 frames `renders/map_C/f_01920.png` … `f_02087.png` (1920×804, `look.finish` + `save_png`), rendered
  in the cloud by `cloud/jobs/map_fix.json` and pushed to `claude/render-map_fix`.
* Render: ~1 s/frame at full res here (4 processes × 1 numba thread); sheet bake ~6 min with 4 workers.
* Review: `review/map_fix_before_after.jpg`, report `review/map_fix_report.md`.

## Known weaknesses / next passes

* The settled fires are small in the last framing (about 9–15 px tall, flaring to ~25); on a phone they
  read as warm points, though never as round dots, and each still has a flame shape and flicker.
* The fold crease at 71°E runs vertically through the left-centre of the last framing.
* Geography is honest but approximate: peaks come from a relief proxy (normal-map integration),
  forests and lakes from a Blue Marble classification, rivers and India's hills are hand-traced
  waypoints (~1°).
* Motion was checked on stills and short frame runs, not on playback; watch the preview for the
  cadence of the catches during the line (1960–2040) and for flicker in the finest hatching.
