# MOUNTAIN — notes

## >>> STATE (2026-09-26 ~08:05) — read this first <<<

**THE BEACON RUN: v1 delivered** — `renders/run/f_01520..f_01679.png` (160 frames, full res).
Review sheet: `~/mishamisha/_local_logs/review/run.jpg`.
**Refinement pass RUNNING in the background** (launched ~08:00): frames 1520-1600 re-rendered with
the summits truncated into rocky plateaus (S0 and the pyre knoll no longer read as snow domes /
a draped sheet), less summit snow override, and a 2.6x bigger spark burst at the pyre. Camera
path and every other beacon are unchanged (pads pinned in `world.py`), so frames 1601-1679 stay
as they are. v1 PNGs of 1520-1600 are backed up in `cache/run_v1_png/`. Log:
`cache/render_run.log` (ALLDONE at the end). ~20 s/frame.

### Next steps
1. When the render is done: check `renders/run/` has 160 frames; make the sheet
   `~/mishamisha/_local_logs/review/run.jpg` (e.g. every 8th frame) and look at it.
2. Refinements (re-render only the affected frames with FRAMES=a-b ./render_run.sh after deleting
   their EXRs): the S0 summit (1520-1531) and the pyre knoll (1570-1590) read as smooth snow domes
   — give them rock/boulder relief (`mtn/land.py: pads()/zone_height()`); optional heat shimmer in
   post around S0/B3; optional spindrift ribbons off ridges.
3. DAWN_C (v2 2400-2655 -> `renders/dawn_C/`) — CODE WRITTEN, NOT YET TESTED (Blender was busy):
   - `shot_dawn.py` (DONE, ran OK): camera high over the cloud sea looking east (SUN_AZ 95), slow
     crane up; sun elevation solved from the camera's own horizon so the limb clears the ridge
     exactly at 2400, then +0.33 deg/s (-> cache/dawn_sun.json); 6 eagles (dawn_eagles.json);
     6 smoking beacons D0-D5 on visible summits 16-31 km (dawn_beacons.json).
   - `bl/dawn.py` (dawn sky, sun-lit land/cloud via baked `sunhor`, dawn fog, sunlit smoke,
     eagle mesh + flap) and `bl/render_dawn.py` (the driver) — untested.
   - `post_run.py dawn` clamps the sun disc and adds a glare (halo + 8-ray starburst + streak)
     scaled by the sun's visible energy, then `look.finish` -> `renders/dawn_C/`.
   - TODO when the run render is done (memory!): `python build_mesh.py dawn --k 0.0048 --smin 0.9
     --cloudk 3.0` (bakes `sunhor`), then test `blender ... bl/render_dawn.py -- --frames
     2400,2420,2480,2560,2640 --scale 0.5 --samples 16 --out $SCRATCH/d1` and iterate; then a
     `render_dawn.sh` like `render_run.sh`.
4. Sheets + finish this NOTES file.

## How the run is built (all procedural, in `shots/mountain/`)

Venv side (`source ~/.venvs/longdawn/env.sh`; numba, 2 threads):
- `mtn/land.py` — analytic land `height(x, y, fp, P, SP, Z, ZC, LP)`: slope-damped fBm massifs
  + smooth-max field of polygonal "horn" peaks (aretes where faces meet, cols between) +
  slope-aligned gully erosion noise; designed horns for B1/B2 and the chain; the story zone
  (S0 summit, the B3 pyre ridge = elongated horns with the natural relief grafted on); flat
  cairn pads + boulders. `fp` = footprint -> alias-free LOD.
- `mtn/cloud.py` — cloud-deck heights (swells + multi-scale Worley puffs).
- `mtn/mesher.py` — camera-aware adaptive meshes: nested grids by distance to the camera path,
  exact vertical-segment/frustum culling, Delaunay; long-edge triangle filter.
- `mtn/bake.py` — horizon-march bakes: moon visibility (long shadows, exact), AO, slope and
  concavity (snow), sightline visibility tests.
- `mtn/campath.py` — weighted camera: arc-length track + speed keys + altitude keys, bank from
  lateral acceleration (clamped), look-at yaw/pitch weights, smoothed rotation, turbulence.
- `world.py` / `world_bl.py` — geography, land params (`LAND_PARAMS`), zone, moon (az -118,
  el 21), `LAND_VERSION` (grid cache key: bump when the land changes).
- `shot_run.py` — the run: camera path (dive past S0, skim, eye-level pass 18 m from the great
  pyre B3 at ~1586, zoom-climb to ~935 m); B1/B2 horns placed by inverse projection from their
  ignition frames; the chain (B4..B7 + F1..F8) = designed great peaks on a perspective line
  converging to a vanishing point on the final frame's horizon, with automatic sightline
  clearings so every fire is seen when it ignites and at the end; side branches S1..S8 on
  natural summits. `python shot_run.py check` prints clearance, rotation rates (<1.8 deg/frame),
  beacon screen positions.
- `build_mesh.py run --k 0.0048 --smin 0.9 --cloudk 3.0` — land (1.15 M verts) + cloud meshes
  with baked attributes (moon, snow, slope, cz, ao / clear, moon, hollow).
- `sparks.py run` — deterministic spark sim for S0 and B3 -> per-frame streak quads
  (camera-facing, energy-conserving width, head/tail over the shutter).

Blender side (`bl/`; Blender's numpy can't load on macOS 12, so IO is `array`-based):
- `scene.py` — sky (gradient, 3-layer procedural stars, faint Milky Way, moon disc only for camera
  rays), land material (snow/rock from baked structure + fall-line couloirs + ledges; moonlight
  emission-coded from the baked visibility; sky fill), cloud material (wrap + forward scatter,
  baked self-shadow, wisps near land), exact two-layer aerial fog group (cloud-top mist + haze).
- `beacon.py` — dry-stone cairn, flared iron basket, stacked wood (heat -> char + embers),
  camera-facing procedural flame cards (tongues, blackbody ramp), glow, fire light (shadowed for
  the heroes), smoke plume card, distance-compensated fire sprite for far fires; ignition envelope
  identical to the montage's (catch at -1 frame, whoomp + 1.4x overshoot).
- `render_shot.py` — builds everything from the cache and renders EXRs.
- `post_run.py` — EXR -> `look.finish(exposure 1.0, bloom 0.08/0.8, vignette 0.2)` -> PNG.
