# MONTAGE-3D — notes

## >>> HANDOFF (Blender crashed on the Mac — memory; a fresh agent continues from here) <<<

### 1. State
* **CITY (src 1680-1719): FINAL, accepted by the director, in the cut.** 40 frames in
  `renders/montage_v2/f_01680..f_01719.png` (51.6 s/frame, 96 samples, DOF f/1.8, motion blur). Don't touch.
* **KARST (src 1640-1679, IGN 1650): look in progress, NOT final, nothing in renders/montage_v2.**
  Latest test `shots/montage3d/tests/t_karst6/f_01660.png` (0.5 scale). My last Blender run (t_karst6)
  finished normally ("DONE", "Blender quit") before the crash — nothing of mine was running then; a
  foreign Blender (pid 77512, not mine) was up and swap was 8 GB used.
  Decided (keep): level camera z=30 looking across layered pillars (ink-painting planes), moon az -36
  el 13 out of frame upper-left (halo in sky), hero pillar c=(28,165) top 26 on the right third with the
  cairn beacon + SDF figure on its crown, near-left pillar framing the left edge, 14 mid pillars in
  layers 250-1250 m + 150 instanced far pillars, ATMOS 'karst mode' mist sea (MIST_TOP -8: textured
  moonlit billows at the ray's entry point + beacon glow term) + 4 volumetric mist banks between the
  layers + a local fire-glow volume round the crown, sparks/shimmer in post, v1 camera move (drift x
  +1.6 -> -2.2, +0.8 m fwd, kick on the hit). Already clearly better than v1 in depth/atmosphere.
  **Open problems (fix before final):** (a) the hero-crown pines are big flat pads that read as
  mushroom caps AND now hide the beacon fire — make foliage clumpier/smaller (more, smaller, noisier
  pads; fewer giant ones), keep crown trees BEHIND/beside the beacon, widen the sightline corridor
  (`in_corridor` in karst.prep) or move the beacon to the camera-facing crown edge; (b) pillars still
  read mostly as dark cutouts — moon rim is weak: try the moon a bit more to the side (az -50..-60) or
  stronger (sun 2.6 -> 4) and check the fractured rock reads at full res; (c) the fire's glow in the
  mist should bloom more obviously warm around the crown (volume `beacon_mist` density/radius); (d)
  memory: 5.2 M pillar verts -> Blender 1.75 GB (peak 2.2 GB) + GPU — cut to < 2 M (see 3).
  Figure (working gestures, per the critic): crouched at the cairn -> IK thrust of the torch into the
  kindling on 1649.5 -> releases it on the hit -> steps back 0.55 m -> left forearm up to shield the
  face (IK) -> lowers it. No torch held aloft.
* **DESERT (src 1520-1579, IGN 1540): NOT STARTED** (no code). Design notes below ("Shot designs").

### 2. Next steps (commands from `the-long-dawn/shots/montage3d/`, after `source ~/.venvs/longdawn/env.sh`)
1. `pgrep -fl Blender.app` — proceed only if no other Blender is running (one Blender on the Mac, ever).
2. KARST fixes (above) in `karst.py` / `karstgen.py`; then low-res key frames:
   `MT3D_REKARST=1 python render.py karst --frames 1640,1649,1650,1653,1660,1679 --scale 0.5 --samples 24 --out t_karst7`
   (MT3D_REKARST=1 regenerates pillars/trees; omit when geometry is unchanged. MT3D_REFIG=1 remakes the
   SDF figure meshes when poses change.) Compare with v1 `renders/montage/f_01645.png`, `f_01660.png`
   and neighbours ice `f_01600`, sea `f_01745`, city `renders/montage_v2/f_01700.png`.
3. Full-res check of 1-2 frames: `python render.py karst --frames 1650,1660 --scale 1.0 --samples 64 --out t_karst7f`
4. Send the director a RENDER_SPEC before any long render (they asked). Final:
   `nohup python render.py karst --final > cache/karst/final_run.log 2>&1 &` (40 frames -> renders/montage_v2/).
5. DESERT: new `desert.py` in the same shot-module pattern (START/END/IGN/SAMPLES/FINISH, flame_specs,
   timing, prep, post, build, per_frame). Decide Blender vs numba by a side-by-side against v1
   `renders/montage/f_01540.png`: v1's numba marcher (`shots/montage/s2_desert.py`: PCHIP-designed hero
   crest `h_dune`, ripple normals, soft moon shadows, Milky Way) is crisp; the director warns Blender
   terrain tends to lumpy mounds on this Mac. A good hybrid: numba heightfield terrain + sky (cloud-OK,
   CPU) and Blender/SDF for the robed figure + beacon + fire composited with depth — or reuse v1's
   terrain as the base and upgrade what's weak (tiny figure, no blown sand, bland dune faces).
6. Before/after sheet `~/mishamisha/_local_logs/review/montage3d.jpg` (v1 vs v2 key frames, all 3 shots),
   finish these NOTES (engine/settings, s/frame, frames per shot, weaknesses).

### 3. Memory-safe Blender on this Mac (8 GB, shared)
* ONE Blender process at a time (render.py writes `cache/blender.lock`; also check `pgrep -fl Blender.app`
  — the driver only auto-waits for `--final` jobs > 3 frames). Never run a test while a final renders.
* Volumetrics: `volumetric_tile_size` 8 for tests, 4 for finals (tile 2 at full res = 2.2 GB peak —
  never), custom range as tight as possible, 48-96 slices.
* `shadow_pool_size` '256' (core default is '512'; pass shadow_pool='256' to setup_render for karst).
* Geometry budget < ~2 M verts: karst `pillar(nt=...)` sets verts = nt x nz with nz auto from height —
  cap nz (e.g. nz <= 700 hero, <= 300 mids) and use nt 384 hero / 192-256 mids / 128 far variants.
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
