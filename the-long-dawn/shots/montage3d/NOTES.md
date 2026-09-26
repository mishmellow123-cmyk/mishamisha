# MONTAGE-3D — notes

## >>> PAUSED STATE #3 (director asked for a pause; resume from here) <<<

**RUNNING in background:** CITY final render — `python render.py city --final` (pid was 52884),
log `shots/montage3d/cache/city/final_run.log` (+ Blender log `cache/city/blender_final.log`),
writing `renders/montage_v2/f_01680..f_01719.png` (SAMPLES=96, full res, motion blur, DOF f/1.8).
Expect ~30-40 s/frame -> ~25 min. On resume: check the log finished (`blender ... exit 0`), count
40 PNGs, look at a contact sheet of all 40 + full-res 1680/1689/1690/1693/1700/1716/1719.
If a background render wakes me while paused: just note it here.

**CITY — DONE in design (pending final QC):** 40-deg lens from 15 m back, two-pass grid city with
downtown cluster left third, water tanks, answer chain that climbs toward the horizon (near answers on
small bulkheads just over the parapet line, far ones on roofs chosen so each lands higher; hot glow
points + haze halos in post), a small silhouette answerer beside the first answer fire. The young
person is now an SDF body (figures.py: smooth-min round cones/ellipsoids + surface nets, per-frame
meshes swapped in by bl_main's per_frame hook) — reads as a real hooded person, not a doll; torch IK
thrust into the drum mouth on the hit, torch released into the drum (RELEASE 1690.5), left hand in
the hoodie pocket (IK), head turns to the answers. Tests: `tests/t_city11f/` (full-res key frames).

**NEXT after CITY QC:** KARST, then DESERT. Director note: MOUNTAIN's Blender terrain came out weak on
this Mac; only replace a v1 terrain shot if mine is clearly better; for terrain I may use the montage
numba toolkit (`shots/montage/mt/`: per-pixel heightfield marcher, fog, fire) instead of Blender.
Plan: karst pillars are not heightfields (overhangs, vertical faces) -> keep them Blender meshes but
test early at low res vs v1 `renders/montage/f_01660.png`; desert dunes are a heightfield -> consider
the numba marcher (v1 s2_desert.py) for the sand + SDF robed figure/beacon composited, or Blender if
it wins in a side-by-side. Before/after sheet `~/mishamisha/_local_logs/review/montage3d.jpg`.

## Pipeline (built)
* `render.py` (venv driver): flame sprites + timing tables (montage toolkit `ignite_env`/`flicker`, so
  envelopes match every other beacon) → one Blender process (`bl_main.py`, lock file, waits for other
  departments' Blender before long full-res jobs) → per-frame post in a thread (EXR read, sparks, heat
  shimmer, distant fire glows) → `look.finish()` → `look.save_png()`. Tests → `tests/<name>/`,
  finals → `renders/montage_v2/f_%05d.png` (src numbering).
* `exr.py`: numpy multilayer-EXR reader (Blender's ZIP/FLOAT). `fireparts.py`: sprite generator (montage
  `_bonfire`), exact Blender-camera projection (`BCam` from per-frame `cam_%05d.json`), Z-up sparks adapter.
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
