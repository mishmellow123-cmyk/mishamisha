# MONTAGE-3D — notes

## >>> CANONICAL RING INSCRIPTION (published 10:40Z, letterforms revised 10:58Z per the director; APPROVED; RING ONLY, not the book): HEROINE, EMBERS, ACCORD-v3 match THIS <<<
* **Texture:** `the-long-dawn/assets/ring/inscription_outer.png` (9040x640) and `inscription_inner.png` (7264x640),
  16-bit grey, white = letter coverage. Mapping + proportions: `assets/ring/inscription.json`. Specimen:
  `assets/ring/inscription_specimen.png`. Generator at any resolution: `shots/montage3d/ring_script.py`
  (`strip(W, H, 'outer'|'inner')`, W/H = circumference/band width; `python ring_script.py publish`).
* **The script of fire** (invented; no real alphabet, no real text): each letter is the outline of a small flame in
  one fine line, heavy at the base, a hairline at the tip (the book: "fine lines ... lines of fire that seemed to
  form the letters of a flowing script", outside AND inside). No broad nib, no marks above the line, no dots, no
  line inside a flame. Letters are joined base to base within a word.
* **Mapping:** u once round (outer: counter-clockwise seen from +axis; inner: clockwise, so both read left to right
  and never mirrored); row 0 = the +axis edge; x-height 0.30 of the band width, baseline 0.66 from the top edge.
  One fixed sequence: never re-randomise. The metal is plain when cold; the letters exist only in fire (emission
  through this mask, deep orange-red, never white).

## >>> STATE AT HANDOFF (MONTAGE-3D-2 fresh lane, 27 Sep 10:15Z) <<<
**What exists (checked 10:12Z; no MONTAGE-3D process running):**
* KARST v2 (pines, Zhangjiajie profile): `renders/montage_v2/f_01640-01679` (40 fr) and the 2/3-speed re-time
  `renders/montage3d_v3/karst_slow/f_00000-00059` (60 fr, complete). BOTH SUPERSEDED by H5 (de-specify: no pines,
  no Zhangjiajie/Huangshan pillar profile, irregular weathered towers in mist). Kept only until the new karst lands.
  Re-time code is in `karst.py` (`KARST_SLOW=1`: source time 1640 + i*2/3, flare at i=15; `--outdir`).
* DESERT v2: `renders/montage_v2/f_01520-01579` (60 fr). Needs the H5 fixes (wide only; plain cloak with wind in
  its folds, no hooded robe, no figurine; ripples randomised in frequency/direction and fading at the crest;
  footprints smaller, irregular, half-filled, in a real gait).
* Ring bake-off stills (the PROTECTED find + fire test): `tests/ring_final_{find,fire,melt}/f_00100.png`, sheet
  `_local_logs/review/ring_bakeoff.jpg`. `ring3d.py` also holds UNTESTED animated inserts (find_a 3000-3079,
  find_b 3080-3149, fire 3360-3599, melt 5360-5519). Their prep died writing the vision EXR (OpenCV EXR codec off
  outside render.py); the 158 knitted-glove meshes it made (`cache/ring3d/data/glove_find_*`, ~700 MB) are the
  descending "bear paw" the H5 critic rejected -> deleted.
**H5 calls to apply before any new render:** KARST de-specified + 2/3 speed (60 fr, flare 15 in); DESERT wide,
cloak, ripples, prints; Ring: canonical inscription; find = her glove enters from the SIDE, thin leather, lit only
by the spark flash (no jewellery-ad fill); fire test = the Ring on the tip of her C-shaped fire-steel in the
beacon's flames, unmarked, letters awake, no glow fringe; melt = darker, crisp drop, letters stay letters until
they go out, no doubled rim.
**Heavy renders -> cloud jobs** (`cloud/jobs/<name>.json`, `"ship": "jpg"`): Blender in the cloud = `pip install
bpy==4.5.14` (the same 4.5.14 LTS as this Mac; cp311 manylinux wheel, 373 MB). No earlier job installed Blender.


## >>> HANDOFF (MONTAGE-3D-2, 2026-09-27) <<<
**Right now:** KARST ACCEPTED (renders/montage_v2/f_01640-01679). **DESERT FINAL DONE** (f_01520-01579, 60 frames,
2374 s = ~40 s/frame, 96 spp full res, peak 368 MB, no post failures, contact-checked: no pops), review sheet
`~/mishamisha/_local_logs/review/montage3d_desert.jpg`, awaiting the director. Calls made: moon az 100; blown sand
dropped (did not read at full res; `opts.streamers`); ripple fade gone before < ~5 px/ripple (no shimmer);
megaripples (1.7 m) give the far dune flanks wind texture; footprint rims softened. Remaining weaknesses: the
prints still read a little like stamped ovals at 1:1; the robe silhouette is simple (no hands at this size).
RING BAKE-OFF (v3 cut C) DONE, stills only: `ring3d.py` (RING_SHOT=find|fire|melt; `python render.py ring3d --frames 100
--scale 1.0 --samples 128 --out ring_final_<shot>`), sheet `~/mishamisha/_local_logs/review/ring_bakeoff.jpg`. Analytic band
(superellipse profile, 512x64, smooth normals), ring.py's invented inscription as emission, full-res screen tracing + a sphere
probe for reflections. MELT uses an SDF slump (`_melt_mesh`) with UVs by angle. Verdicts: FIND Blender, FIRE Blender (clearly),
MELT Blender but weakest (hazy; crisper drop + darker exposure needed). ~20-110 s per still, peak 258 MB.

### 1. State
* **CITY (src 1680-1719): FINAL, accepted by the director, in the cut.** 40 frames in
  `renders/montage_v2/f_01680..f_01719.png` (51.6 s/frame, 96 samples, DOF f/1.8, motion blur). Don't touch.
* **KARST (src 1640-1679, IGN 1650): FINAL RENDERED, awaiting the director's review.** 40 frames in
  `renders/montage_v2/f_01640..f_01679.png`; review sheet `~/mishamisha/_local_logs/review/montage3d_karst.jpg`.
  Kept from MONTAGE-3D-1: camera (z=30, drift x +1.6 -> -2.2, +0.8 m fwd, kick), pillar layout, ATMOS mist
  sea, 4 mist banks, figure action. Fixed:
  (a) pines: new `_pine` = crooked trunk leaning out over the drop + 3-6 tiers of flat RAGGED needle
      clouds (10-30 small noisy blobs each, not one smooth pad), dark needles with AO in the gaps + a
      fine tuft bump (sheen made them read as popcorn); ledge pines lean out of cracks (yaw = outward
      from the pillar axis); 12 variants (crown 0-4, ledge 5-7, low-poly far 8-11) + 2 scrub variants.
      Hero crown composed by hand (`HERO_PINES`, all behind/beside the beacon) + dart-thrown scrub
      clear of the beacon, the figure and the sight line; crown flattened (cap 0.1 R0) and roughened.
  (b) rim: moon az -58 el 16, strength 5.5 (volume factor 0.4 keeps the mist as it was); rock albedo
      up a touch. Moon-facing clefts now catch light (the long vertical joint on mid0 is real geometry).
  (c) glow: fire lights volume_factor 2.2 (halo in the air without over-lighting rock), `beacon_halo`
      glow_haze r 17 m d 0.013, and a wispy `veil` mist bank wrapping the upper shaft 4.5 m under the
      crown, lit from above by the fire. Sparks toned down (burst 120, rate 30) -- no firework.
  (d) memory: pillars are meshed only in the camera's visible z window with pixel-matched spacing
      (`_pillar_mesh` in karst.py; same shapes/seeds as karstgen) -> 1.30 M unique verts (was 5.2 M);
      new foreach_set mesh loader (`_mesh_fast`). Blender peak 616 MB half-res / 640 MB full-res tile 4
      (was 2.2 GB). Final: 96 samples, full res, vol tile 4, motion blur; ~90 s/frame on this Mac.
* **DESERT (src 1520-1579, IGN 1540): BUILT (`desert.py`), look nearly settled in half-res stills;
  NOT rendered final (the director reviews KARST first).** Latest stills: `tests/t_des6/` (moon az 100),
  `tests/t_des5/` (5 key frames, az 118), full-density summit crop `tests/t_des5c/f_01545.png`.
  Design (full Blender, no numba plate): every dune is its own crest-conforming mesh (rows along the
  crest, one column exactly on it, split windward/slip objects) -> true knife edges, never lumpy; radial
  ground grid with gentle mid-scale rolling; 0.61 M verts; Blender peak ~340 MB half-res, ~5 s/frame at
  half-res 24 spp. The camera stands on a high point of the hero crest (eye 1.62 m); the crest dips
  through a saddle (we look down on it: its moon-shadowed slip face runs as a dark band) and climbs to a
  small summit plateau (`_plateau`, cairn seated flat) where the robed, hooded SDF figure (robe + hood
  + neck wrap; hem billows downwind) thrusts the torch into the cairn beacon on 1539.5, lets go, steps
  back 0.42 m with the forearm up (elbow forward), watches. Flame/smoke/sparks stream downwind (-x).
  Shader ripples (wind-aligned, asymmetric, faded by pixel footprint), grainflow on slip faces, a line
  of footprints up the crest to her (UV procedural, `_footprints`), trampled sand round the cairn, faint
  fire-lit sand streamers off the summit crest, Milky Way core low left, horizon haze u=1/7000.
  Fixed on the way: rings in the stacked-ellipsoid robe (-> two tapered cones + hem billow); scarf end
  read as an outstretched arm from behind (tube, then band) -> removed; PCHIP extrapolation wall; hero
  mesh open end + summit objects threw odd moon shadows into the lower-left -> dune tapered at both ends,
  moon az 118 -> 100 (the summit's long shadow now falls out of frame).
  **Open before DESERT final:** (1) regenerate with the committed moon (az 100, el 17, I 2.3):
  `MT3D_REDUNE=1 MT3D_REFIG=1 python render.py desert --frames 1520,1539,1541,1545,1560,1579 --scale 0.5 --samples 24 --out t_des7`
  and check all key frames (the az-100 test only covered 1520/1545); (2) full-density crops of the summit
  (`--scale 0.33334 --opts '{"crop": [1150, 300, 640]}'`) and of the foreground footprints/ripples
  (`{"crop": [700, 700, 640]}`) -- ripples may alias near 30-90 m; (3) the left mid-ground dune face is
  plain -- acceptable, maybe a touch darker/softer; (4) streamers are subtle (dens 1.2): keep or drop;
  (5) 2 full-res frames at 64 spp (tile 4) for memory/time before the final; (6) review sheet + report.

### 2. Next steps (commands from `the-long-dawn/shots/montage3d/`, after `source ~/.venvs/longdawn/env.sh`)
1. `pgrep -fl Blender` -- the idle stray Blender app (pid 77512, ~9 MB, not ours) is always listed; any
   OTHER Blender means someone is rendering: wait. Because of that stray, `render.py --final` would wait
   forever -- pass `--force` after checking by hand.
2. DESERT stills: `python render.py desert --frames 1520,1539,1541,1545,1560,1579 --scale 0.5 --samples 24 --out t_desN`
   (MT3D_REDUNE=1 rebuilds the dune meshes, MT3D_REFIG=1 the robe meshes -- ~10 min for 64 frames).
   Full-density crops: `--scale 0.33334 --opts '{"crop": [cx, cy, 640]}'` (cx, cy in full-res pixels).
   Composition preview without Blender: numpy raymarch of `desert.ground_at` (see gotchas).
3. DESERT final (after the director's OK): `nohup python render.py desert --final --force > cache/desert/final_run.log 2>&1 &`
   (60 frames -> renders/montage_v2/f_01520..f_01579.png). Review sheet:
   `python tests/review_sheet.py desert ~/mishamisha/_local_logs/review/montage3d_desert.jpg 1520,1540,1545,1579 1545 <x0,y0,w,h>`

### 3. Memory-safe Blender on this Mac (8 GB, shared)
* ONE Blender process at a time (render.py writes `cache/blender.lock`; also check `pgrep -fl Blender.app`
  — the driver only auto-waits for `--final` jobs > 3 frames). Never run a test while a final renders.
* Volumetrics: `volumetric_tile_size` 8 for tests, 4 for finals (tile 2 at full res = 2.2 GB peak —
  never), custom range as tight as possible, 48-96 slices.
* `shadow_pool_size` '256' (core default is '512'; pass shadow_pool='256' to setup_render for karst).
* Geometry budget < ~2 M verts. Mesh only what the camera can see: karst pillars are meshed on the
  visible z window (the mist sea hides everything under ~-26 m, the near-left pillar only shows z 12-52)
  at ~2 px spacing for their distance (`karst._res_for`). Load big meshes with foreach_set
  (`karst._mesh_fast`, `desert._mesh_ch`), never Python lists of tuples (doubles the peak RAM).
* Instancing (GN `C.instancer`) for trees/tanks/far pillars; source collections view-layer-excluded.
* EXRs are deleted after post (keep with `--keep-exr` only for tests); 32-bit multilayer ~20 MB/frame.
* No textures at all (everything procedural), so texture limits don't apply.

### 4. Gotchas learned
* Blender's bundled numpy can't load on macOS 12 — keep Blender-side code numpy-free; do heavy work in
  the venv and pass binary meshes (`figures.write_mesh` format; 4th index -1 = triangle) / JSON.
* numba: every kernel needs `cache=True` (NUMBA_CACHE_DIR=cache/numba) — `_pillar_r` takes ~5-10 MIN
  to compile the first time (heavy inlined mt.noise fbm); after that ~1 s per pillar.
* EEVEE blended (additive) flame cards don't appear in screen-space reflections.
* Separate rigid capsule bodies always read as mannequins — use the SDF body (`figures.py`:
  smooth-min round cones/ellipsoids + surface nets, per-frame meshes swapped by bl_main's per_frame).
  FK abduction sign was wrong in the old kit version (fixed in figures.fk).
* Voronoi crease bumps on cloth look like cracked stone — don't.
* ATMOS shader fog needs geometry under every downward ray (the karst 'valley' plane), otherwise rays
  fall through to the world and pillars show through the mist. `enters` test: Incoming.z > 0 = looking down.
* Tree placement must be spacing-based (dart throwing), not per-vertex probability (mesh-resolution
  dependent: 12,891 trees once).
* numba invalidates a file's whole cache when the SOURCE FILE changes (mtime/size): editing karstgen.py
  costs a 5-10 min `_pillar_r` recompile -- put new venv helpers in the shot module (karst.py) instead.
* PCHIP crest tables extrapolate wildly past their last control point: clip/taper outside the range
  (`desert.hero_height`) -- a 'wall' at the horizon in a preview came from that.
* Quick composition check with Blender busy: raymarch the venv height function in numpy at ~288x121
  (desert: `ground_at`, moon-only shading, figure/beacon points projected) -- ~2 min, no shadows.
* Crop test at full-res pixel density: `--scale w/1920 --opts '{"crop": [cx, cy, w]}'` (karst & desert):
  narrower lens + shift, same pixels as the final; post (sparks) still projects right.
* Fire-lit foliage: white-ish sheen + smooth convex blobs = popcorn/cotton. Dark albedo, AO node in the
  material, a fine bump, many small blobs; keep scrub > 5 m from the fire.
* `~/mishamisha/_local_logs/sheet.py` resolves paths under renders/ — for tests use
  `render.make_sheet(dir, frames, out, cols, tw)`.
* zsh doesn't word-split unquoted vars; no `timeout` command on macOS; Bash tool timeout max 10 min —
  use nohup/background for long jobs.

### 5. Taste bar
* One clear idea per shot, ruthlessly composed; the fire is the brightest, warmest thing — everything
  else lives in blue darkness with real atmospheric depth (layers that fade), never a flat backdrop.
* Figures are silhouettes carved by firelight with working gestures — never a doll/mannequin read,
  never a torch held aloft; judge figures in neutral grey turnarounds (`tests/t_figview.py`) first.
* Only replace a v1 shot if it's clearly better side by side at full res; if in doubt, keep v1.
* Real-world specificity sells it (water tanks, office-floor bands, clinging pines, ripples) — but
  nothing that reads as a tech demo (no firework sparks, no snow-cap crowns, no mushroom trees).

## Pipeline (built)
* `render.py` (venv driver): flame sprites + timing tables (montage toolkit `ignite_env`/`flicker`, so
  envelopes match every other beacon) → one Blender process (`bl_main.py`, lock file, waits for other
  departments' Blender before long full-res jobs) → per-frame post in a thread (EXR read, sparks, heat
  shimmer, distant fire glows) → `look.finish()` → `look.save_png()`. Tests → `tests/<name>/`,
  finals → `renders/montage_v2/f_%05d.png` (src numbering).
* `exr.py`: numpy multilayer-EXR reader (Blender's ZIP/FLOAT). `fireparts.py`: sprite generator (montage
  `_bonfire`), exact Blender-camera projection (`BCam` from per-frame `cam_%05d.json`), Z-up sparks adapter.
* `figures.py` (venv): SDF people (fk/ik, hoodie_body, mesh_sdf surface nets, write_mesh).
  `karstgen.py` (venv): pillar meshes (lobes + striations + fractured blocks + crown), tree_points.
  `city.py`, `karst.py`: shot modules (venv spec + prep + post; Blender build + per_frame).
  `tests/t_figview.py` (figure turnarounds), `tests/t_pillar.py` (view any .bin mesh).
* `kit/` (inside Blender, no numpy — Blender's bundled numpy can't load on macOS 12):
  `nodes.py` node DSL; `core.py` render setup (EEVEE Next, Standard view, 32-bit multilayer EXR +
  Depth), camera keys (yaw/pitch/roll), GN instancer, ATMOS shader-fog group (exact exponential height
  fog + street-glow term + moon forward scatter), procedural night sky (gradient, moon, Voronoi stars,
  Milky Way); `fire.py` cairn/basket/wood/drum, flame card (sprite sequence, camera-facing, additive
  BLENDED), fire lights (keyed per frame), smoke plume + glow haze volumes (fire-lit only: moon
  volume_factor 0); `figure.py` rigid capsule body + keyed transforms, 2-bone IK, parametric cloth
  (robe, hood, sleeves, scarf ribbon) as per-frame shape keys, torch.

## Shot designs (planned)
* CITY 1680–1719 (IGN 1690, answers 1694–1716): eye-level rooftop, wet tar deck, parapet at ~12 m,
  hooded young person right of centre + drum (right third); grid city rotated 28°, near neighbours
  around our height with water tanks, downtown towers right of centre; camera = original move.
* KARST 1640–1679 (IGN 1650): 3-D fluted/cracked sandstone-limestone pillars (Python + mathutils
  noise meshes, instanced variants) with GN-instanced clinging pines on ledges/crowns; mist sea via
  ATMOS (noise-modulated top) + mist sheets; fire-lit local mist volume around the hero pinnacle so the
  glow blooms inside it; lateral drift camera as the original.
* DESERT 1520–1579 (IGN 1540): heightfield dunes (venv-generated, loaded via raw floats), knife-edge
  S crest, shader wind ripples, scalloped slip faces, blown-sand streamers off the crest, robed
  hooded figure (robe/hood/sleeve/scarf cloth) backlit, Milky Way; slow truck right as the original.
