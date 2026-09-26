# >>> HEROINE v2 (FIRST BEACON, Sep 2026) <<<

The Young Woman in FIRST BEACON is now a sculpted 3-D figure lit by the fire. `beacon.py`
`HEROINE_V2 = True` gates everything: the figure, the choreography, the camera, the dark-adaptation
floor and the s1-world params. `peaks.py` `S1_WORLD = True` (v8 cache) gates the new far ranges.
Output: `renders/hills_v2/f_01200..01439.png` (src numbering; the edit reads hills_v2 first).
INTRO/CODA import neither beacon.py nor heroine*. Do not edit characters.py/coda.py/intro.py
(the CODA dept owns them).

**Re-render**: `python shots/hills/render.py --shot beacon --frames 1200-1439 --workers 2 --out renders/hills_v2`
(add `--skip-existing` to resume). Half-res tests go to renders/hills/tests/hv2_*.
Lookdev and debug tools: `~/mishamisha/_local_logs/heroine_lookdev/` (t_shot.py = heroine-only at a
shot frame, shot/neutral/noshadow; t_debug.py = profile vs target + prim-ID map; t_hand.py; t_zoom.py).

**What it is**
* `heroine_sdf.py`: numba SDF sphere tracer.
  * Primitives: ellipsoid, round cone (+ Lipschitz-safe folds), round box, tapered/tilted torus arc,
    bent half-space, eyelid shell with an angular fissure (+ re-carve mode).
  * Culling: per 16 px tile, then per ray; 2x2 (half res) / 3x3 (full) supersampling on edges.
  * Lighting: point lights E = I/(d^2+r^2) with SDF soft shadows. A negative soft-k marks a source held
    in her hands (strike, ember, tinder flame), so her hands, tools and sleeves cast no shadow for it.
  * Materials: wrap/SSS skin with cold-reddened nose, cheeks, knuckles and fingertips; GGX; wool sheen
    and bump maps; Kajiya-Kay hair; a thin cold moon-rim from behind; AO; iris/pupil.
  * Also: strand renderer (hair, flyaways, lashes); fog-puff breath.
* `heroine.py`:
  * Head fitted to a female anthropometric profile (within ~3 mm).
  * Knitted beanie with folded brim; long dark hair escaping at the nape, streaming as locks plus fine strands.
  * Anatomical hands (knuckles, pads, nails); flint in the near LEFT hand (the camera sees her left
    side), C-steel in the far RIGHT hand. Heavy wool coat with folds; trousers, boots.
  * Scarf: `scarf_red` #9E1B1B knit, two loops, twisting/rippling 1 m tail, 9 fringe tassels.
  * She kneels on a low summit rock so her eyes clear the basket rim.
* `beacon.py` choreography (`V2_KEYS` + overlays):
  * strikes 1236/1262/1290: the steel bar scrapes the flint's edge on the half-frame the sparks are
    born, then follows through;
  * blowing 1297-1318 (face 13 cm from the ember, lit by it, breath jet to the tinder);
  * catch 1318 (ember flare, face lit from below, breath);
  * she watches the flame 1330-1360;
  * ROAR 1360: flinch with an open-hand guard (head-relative), eyes shut, turning away;
  * rise and step back 1364-1393, arms down;
  * 1393-1439: turns three-quarters away to watch the far range, scarf whipping. No poster poses.
* Camera (v2): bigger, higher pull-back (46 m back, 5 m up, hfov 40->66) - she ends ~50 px tall.
* Dark adaptation (v2): before the catch the moonlit summit and her rim sit at a 17% floor (cut B
  is never an empty screen); the fire drops it to 4%; the reveal brings the world up.
* The reveal's world = the shepherd's world:
  * v8 grids are baked from `shots/montage/s1_peak.py` `h_far`/`h_cloud` around the top of s1's
    first-beacon massif (-5450, 282, 31997 in s1 coords), with s1's moon;
  * radial resolution is doubled (the old smear);
  * s1's snow-hold threshold, rock albedo and moon level are passed from beacon.py.

**Done**
* Baseline check: local render matches delivered frames — f1420 51.1 dB (dither only). f1230/1300 are
  41.5 dB with a ~1.5-level background offset, because the delivered 1200-1359 predate TRUE_PEAKS.
  => Plan: re-render the WHOLE take 1200-1439 into hills_v2 (no join inside the take).
* `heroine_sdf.py` (new, numba): 3-D SDF sphere tracer for her only. Primitives: ellipsoid, round cone
  (+ folds), round box, tapered/tilted torus arc, bent half-space, eyelid shell with an angular fissure
  (+ fissure re-carve mode). Tile + per-ray culling, 3x3 SS on edges, soft shadows, AO, wrap/SSS skin,
  GGX, wool sheen, Kajiya-Kay hair, cold rim, head-space tints (lips/brows/flush), iris/pupil.
  `sdf_points` = debug evaluator.
* `heroine.py` (new): Builder (ordered CSG per group), materials, head sculpt fitted to a female
  anthropometric profile (within ~3 mm; see lookdev), eyes/lids that read in profile + 3/4, hair cap.
  Lesson: keep smooth-union k small (<=0.014) — smin inflation accumulates across overlapping masses.
* Just appended (UNTESTED): ik3/skeleton/hand()/build_figure()/scarf_tail() — the body, coat
  (folds), sleeves + cuffs, fingered hands, steel + flint, legs/boots, rock, scarf loops + tail +
  fringe, and hair locks.
* Lookdev tools + latest images: `~/mishamisha/_local_logs/heroine_lookdev/` (lookdev.py is the
  helper; t_debug.py = profile-vs-target + prim-ID map; t_zoom.py = eye zoom; t_prof.py = clay +
  firelit profile). Run them from that folder with the venv.

**Next steps**
1. Smoke-test build_figure in lookdev with a kneeling pose. Fix IK and hand orientation, then iterate the
   hands (flint and steel grips) and the coat.
2. beacon.py: add a `HEROINE_V2 = True` flag that gates everything. Kneeling on a rock beside the cairn,
   eyes ~1.14 m, pelvis ~(0.66, 0.56). Add yw2_pose(f) choreography:
   * strikes 1236/1262/1290 (raise, snap down; sparks come from the new flint anchor — same RNG
     call order, so the embers are unchanged);
   * cupped hands and blowing 1296-1318 (purse expression, breath aimed at the ember);
   * catch 1318: face lit from below, breath fog;
   * 1318-1360: near hand shields the flame;
   * 1360: flinch (head away, eyes shut, arm up), step back and rise, then stand firm 1385+.
   Scarf and hair chain anchors come from the new rig.
3. Compositing: render her with heroine_sdf (lights = strike flash / ember / flame / roar, plus rim and
   ambient). Mask the flame by her near hand, then add breath fog, hair strands and lashes.
4. Half-res tests (1240, 1262, 1290, 1320, 1340, 1362, 1372, 1390) → iterate → full-res 1200-1439
   into renders/hills_v2 (2 workers) → before/after sheet at `_local_logs/review/heroine_v2.jpg`.

# CODA v2 (CODA dept, 2026-09-26): INTRO + CODA figures, cairn, answering fires -> renders/hills_v2

Fixes critic_tone B1 + M6 (HILLS parts). `V2 = True` in `intro.py` and `coda.py` (False = v1; v1 copies of
both files in `shots/hills/v1_backup/`). New modules (opt-in, nothing shared was changed; beacon.py untouched):
* `silhouette.py` - **merged-silhouette renderer**: every group's SDF is unioned BEFORE lighting; depth below
  the outline = exact distance transform of the union (no seams between primitives); rims (sky + fire) only on
  the union's OUTER contour, the sky rim taken from the blurred background behind each edge; soft 2-D shadow
  march for the fire (thin things - torch, fingers, wisps, fringe - cast none). Scarf = the only lit cloth.
* `figures2.py` - `elder2` / `child2` (same pose dicts + anchors as characters.elder/child): spline-cut
  outlines (tailored long coat, flared back hem with fold scallops that move in the wind, heavy sleeves with
  cuffs, gripping hands, ankle boots; child's quilted puffer, hood, puffy sleeves, mittens, snow trousers,
  beanie with fuzzy pom-pom), profile faces left UNLIT (1-2 px fire edge only), sheen <= 0.4 on cloth/skin.
  **Scarf** = heroine.py's wool (albedo 0.29/0.022/0.019): two rolls + crossing + short front end, and a
  1.05 m tail (verlet chain 20 x 0.053 m) that tapers, twists (face normals turn in/out of the firelight),
  ripples, and ends in 9 yarn tassels; a knitted rib runs along it (in the ribbon's own frame), like the
  young woman's knit. Grey flyaways at the bun are fine curled strands, not wires.
* `cairn2.py` - `DryStoneCairn`: flat-bedded irregular stones (broken ends, chipped corners, staggered joints,
  pinning stones, capstone), hard chamfered arrises, per-face tilt, grain/lichen, dark hearting in the gaps.
  Basket and every dimension identical to characters.Cairn (fire base unchanged for the ember title); the
  woodpile is rebuilt as split logs with flat-cut ends (teepee + cross logs through the bars + kindling).
  Basket + wood drawn as a merged silhouette; the iron bars are redrawn over the beacon's flame (dark cage).
  *For the HEROINE lane (optional, their call):* beacon.py's cairn is still the v1 pillow stones. Drop-in:
  `CAIRN = cairn2.DryStoneCairn(seed=11)` (same top/bk_bot/bk_top), then draw the stones with
  `CAIRN.render(cam, [0, 0, 0], lights, amb_top, bg=img)` (returns y0, x0, rgb, a for over_region) and the
  basket/wood with `silhouette.Silhouette([0, 0, 0], CAIRN.basket_groups(x=0.0)).render(cam, lights, amb_top,
  np.zeros(3), bg=img)`; lights in the same (n,8) world format beacon.py already builds.
* `fires2.py` - distinct distant fires: supersampled bonfire-shaped flames with overshoot on ignition, a burning
  pile glow, tight air-scatter glow (no bokeh halo), hillside light, a lit smoke wisp occluded by nearer
  ridges; placed on the BARE terrain (v1 ridge peaks included treetops); `crest_texture` breaks up the
  firelit hilltop. Also a corrected plume (`smoke_wisp`): **fire.render_smoke streaks horizontally late in the
  film** (it divides Y - t*rise by a height-dependent width; at t ~ 110 s the vertical frequency explodes) -
  other lanes using it after t ~ 60 s should check their smoke.

Timing kept (all keyframes, camera moves, sky). Changed per director: the child's beacon **catches at 2639 and
whooshes with overshoot ON 2640** (music hit v2 2800). Answering fires = `fires2.PLAN`: 14 fires on successive
ridgelines, outward and alternating sides; the 12 voiced ones ignite ON the SFX whumps (src 2645 2651 2657 2663
2669 2675 2682 2689 2696 2703 2710 2716 = v2 2805..2876), 2 far unvoiced ones between; plus the 7 INTRO
festival fires. `renders/hills_v2/coda_fires.json` has frames (src + v2), world positions, flame heights,
screen x/y and pan at ignition. TITLE: re-extract `edit/ember_title_fires.py` (its `_extract()` works
unchanged: coda.answering_fires / place_beacons API kept); the SFX pans could be matched to the JSON `pan`.

Re-render: `python shots/hills/render.py --shot coda --frames 2460-2807 --workers 2 --out renders/hills_v2`
(and `--shot intro --frames 0-359`). Lookdev: `_local_logs`-style scripts are in the session scratchpad only.

# HILLS: notes

Deliverables (global frame numbers, 1920×804, `look.save_png`) go to `renders/hills/f_%05d.png`:

* **INTRO** 0–359
* **FIRST BEACON** 1200–1439
* **CODA** 2460–2807

Also in `renders/hills/`: `preview.mp4` (the three pieces back to back), `contact.png`, and
`coda_fires.json`. The JSON holds the ignition frame of every answering fire in the CODA
wave. The first 18 are marked `voiced`, for the bell cascade. Test renders are in
`renders/hills/tests/`.

## How to re-render

```
cd the-long-dawn
python3 shots/hills/render.py --shot intro  --frames 0-359     --workers 2
python3 shots/hills/render.py --shot beacon --frames 1200-1439 --workers 2
python3 shots/hills/render.py --shot coda   --frames 2460-2807 --workers 2
#   --scale 0.5 for tests, --out <dir>, --step N, --skip-existing
```

Each worker is one process with `NUMBA_NUM_THREADS=1`. Every frame is a pure function of
its frame number: the simulations (scarf, hair and wisp verlet chains, particle systems)
are deterministic and re-run from the start of each piece in every worker. After editing
a shared `@njit` helper in `core.py`, `sky.py` or `land.py`, run `shots/hills/clear_cache.sh`,
because numba's disk cache does not track cross-module inlining.

## What is built (all procedural)

* `core.py`: the pinhole camera (metres; x right, y up, z forward), easing and keyframe
  tracks, numba Perlin noise and fBm, energy-normalised gaussian, disc and streak
  splats, and compositing.
* `sky.py`: the twilight gradient with an asymmetric afterglow, a 14–26k-star catalogue
  (magnitude power law, colour, twinkle, horizon extinction, visibility against sky
  brightness), and the Milky Way. The Moon is a 2× crescent with earthshine and clustered
  warm city lights on its night side. The **orbital ring** is a geometric equatorial ring
  seen from 57°N. It is sunlit toward the sunset and cut by Earth's shadow on the far side,
  with faint station nodes.
* `land.py` + `hillworld.py`: 8 ridge "cards" at real distances (480 m–56 km), each
  defined by the apparent angle of its crest. They have trees, analytic anti-aliased
  edges, aerial perspective, noise-varied valley mist, beacon glow on the hills and a
  depth buffer. (Their moonlit-snow pyramid mode is no longer used; see `peaks.py`.)
* `peaks.py` (FIRST BEACON only, imported by `beacon.py` alone; `TRUE_PEAKS` flag): a real 3-D
  moonlit Himalaya. Ridged-multifractal range on a polar grid (azimuth x log-range, so each
  cell is about a pixel at any distance, octaves prefiltered by footprint), Earth curvature,
  a cloud-sea top surface that thins where peaks pierce it, precomputed soft moon shadows,
  snow/rock by slope, exponential height haze toward the sky's horizon colour, and a 10 cm
  summit grid (flat top for the cairn, serrated rocky arete, calm snow face under T9).
  Column-coherent per-pixel ray march, 1.3x supersampled at full res. The grids are cached in
  `shots/hills/cache/beacon_peaks_v7.npz`; delete it after changing a height function.
* `puppet.py` + `characters.py`: 2-D SDF puppets on camera-facing cards: smooth-union
  capsules, ellipses and polygon profiles, two-bone IK, and verlet scarf, hair and wisps.
  **Backlit-silhouette shading** gives near-black interiors, a thin cool rim from the sky
  behind, and a thin warm rim where the fire grazes an edge. The scarf is the only colour:
  dark oxblood that glows ember-red where the fire reaches it, with a fringed tail. The
  Elder, the Child (pom-pom hat) and the Young Woman share one rig style. The same scarf
  function is used for the Elder and the Young Woman.
* `fire.py`: the HDR flame (multi-scale domain warp, torch and bonfire envelopes, splitting
  and tearing tongues, white-yellow core low, broken red tips), ignition with
  whoomp-and-overshoot, flicker, distant beacon points with halo, spark and ember particles
  (tapered hot-head streaks, cooling colour, curl-noise meander, bokeh when defocused),
  smoke lit only near the fire, and a veiling glow.
* `intro.py`, `beacon.py`, `coda.py`: the three pieces. `render.py` is the CLI.

## The pieces

**INTRO**
* 0–159: wide blue-hour establishing shot. Beacons ignite one by one at 22, 80, 112, 138
  (then 196, 238 and 262 behind the two-shot). The Child points at the fire at 80, turns
  (about 104) and looks up at the Elder, under T1.
* 160–359: medium two-shot (cut on the bar-3 downbeat). The Elder looks down at the Child,
  then out to the fires (T2). From about 228 the camera pushes into the torch, the wind
  drops, and the frame tilts up a column of rising embers over the flame's glow. By 300
  this matches the first EMBERS frame: a warm glow at the bottom centre and embers rising
  on near-black.

**FIRST BEACON** is one take.
* 1200–1235: black, with a breath of snow.
* Strikes at 1236, 1262 and 1290 each give a flash and a spark shower into the basket, and
  light the hands, the steel, her profile and the scarf wrap. They grow in strength.
* A spark lives in the tinder and pulses as she blows. At 1318 the kindling catches,
  lighting her face from below, and her breath shows.
* At 1360 the beacon roars. She rises with an arm up against the heat. The camera pulls
  back and up, with an explosive ease-out, to the snowy summit, a sea of moonlit peaks and
  cloud, spindrift, the stars and the Milky Way. There is no ring or lit Moon, because this
  is sixty years earlier. The moonlit world fades up as it is revealed.

**CODA**
* 2460–2627: medium shot. The Child looks up (T15). The Elder meets her eyes, looks away
  and says nothing. A gust lifts the scarf at 2548–2618.
* 2580–2618: the Elder lowers the torch into the Child's hands and moves her hand to the
  Child's shoulder.
* 2628–2807: wide. The Child lifts the torch into the basket, and it **catches at 2640**.
  About 50 answering fires ignite in a rolling wave outward and across the valley between
  2645 and 2720 (see the JSON). The INTRO fires are still burning.
* From about 2662 the camera cranes up into the sky: the ring, the lit Moon, stars and four
  slow ship lights. The centre of the frame is dark sky for the title.

## Deviations from the bible

* **The Moon is drawn at about 2× real size** so that its city lights read.
* **The orbital ring is placed for the picture, not exact astronomy.** Its sky path is
  physical: an equatorial ring at 2.6 Earth radii seen from 57°N. The Earth-shadow cut is
  placed where it reads in frame.
* **FIRST BEACON's reveal is now a true heightfield** (`peaks.py`, Sep 2026), replacing the
  pyramid ridge cards, which read as stage flats. Only 1360-1439 were re-rendered; before 1360
  the range sits at 3% behind the close-up's defocus, so the join is invisible.
* **Cairn scale.** The cairn is 0.76 m, with the basket rim at about 1.12 m, so the Child
  can reach it.
* **Frame edges.** INTRO starts at full exposure; the edit handles the fade-in. The CODA
  ends on the sky; the edit fades it to black.
