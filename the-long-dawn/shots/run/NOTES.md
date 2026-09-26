# RENDER_SPECs (cloud) — run everything from `the-long-dawn/`

Fresh Linux x86 box (tested here: Python 3.12, numpy 2.5.3, numba 0.67.0, opencv 4.10, scipy 1.18.1):
```
python3 -m venv ~/ld && . ~/ld/bin/activate && pip install numpy numba scipy opencv-python-headless
```
No env vars needed (the drivers set NUMBA_NUM_THREADS=1 per worker; `--procs N` = N workers). First frame of a
fresh box compiles numba kernels (~1-2 min, cached in `__pycache__`). Committed data that MUST be present (do not
regenerate): `shots/run/summits.npy`, `shots/run/beacons.npy`. All drivers take `--skip` (resume a block).

## DAWN_C — v2 frames 2400–2655 → `renders/dawn_C/f_%05d.png` (cut C only)  **READY**
```
python shots/run/dawn.py --range 2400-2655 --procs 4 --skip
```
Split suggestion (4 machines): 2400–2463 / 2464–2527 / 2528–2591 / 2592–2655. Depends on shots/run/{dawn,world,run,
rcam,pipe}.py + beacons.npy (smoke sources). Deterministic. Cost ≈ the Run's far-terrain frames (~2–4 min/frame
single-threaded on this Mac; no near terrain).

## s1 re-render with the Run's fire — v2/src 1440–1519 → `renders/montage_v2/f_%05d.png`  **READY**
```
python shots/run/s1_v2.py --range 1440-1519 --procs 4 --skip
```
Runs shots/montage/s1_peak.py unchanged with fire2's flame (orange→yellow→white, soot absorption), spark ramp
and glow colour patched in. ~72 s/frame on this Mac with 2 threads at full res (s1's own cost).

## THE BEACON RUN — v2 frames 1520–1679 → `renders/run_v2/` (run_fix look; see the RUN spec further down)
Rendered in the cloud by `cloud/jobs/run_fix.json` (branch `claude/render-run_fix`).

## World settings for HEROINE (to match the FIRST BEACON reveal's far ranges)
The ranges ARE s1's (`shots/montage/s1_peak.py`); the Run and Dawn only add foothills near the shepherd.
* Coordinates: metres, y up, +z = s1's view axis, the shepherd's summit at the origin (y≈0). The heroine's beacon
  massif summit = `S1.summit()` ≈ (−5450, 286, 31997): 32 km from the shepherd, bearing −10° from s1's axis.
  From her summit the shepherd is at bearing ≈ +170° (looking back down −z).
* Ranges `S1.h_far`: ridged multifractal (seed 21) at 3400 m, octaves by pixel footprint (≤11), domain-warped by
  fbm at 7000 m (seeds 61/62, ×0.55); massif envelope fbm 11000 m (seed 63): prom = 0.55 + 0.45·smoothstep(−0.35,
  0.35, m); h = −1800 + 2100·prom·r^1.45; + her massif (Gaussian, σ 2600 m, +HB+450 at XB, ZB); a corridor term
  thins peaks along s1's sight line. Earth curvature d²/2R everywhere.
* Cloud sea `S1.h_cloud`: −650 m + 150·fbm(2400 m, seed 33) + 55·(1−|fbm(520 m, seed 34)|) + 10·fbm(120 m, seed 35).
* Snow (far): snow = smoothstep(0.38, 0.58, n_y(45 m-smoothed normal) + 0.18·noise(380 m, 73) + 0.06·noise(95 m, 74));
  rock albedo (0.055, 0.056, 0.062), snow (0.80, 0.86, 0.98). Cloud: wrap light, forward scatter 1+1.6·cos^4,
  peak shadows (floor 0.35), trough darkening 0.55.
* Moon: dir normalize(−0.80, 0.36, 0.48) (az −59°, el 21°), colour #9DB4D9, intensity 0.55; ambient #27335E×0.35.
  Sky: zenith #070B1C, horizon #2A3866, horizon glow 0.25, moon halo 0.025/0.22 + 0.012/0.7; stars
  `SK.make_stars(14000, 101, lum_scale=7)`. Fog: 5e-5 (scale height 1500 m) + cloud-top mist 2.2e-4 (140 m above
  −650), colour #2E3D66×0.95, forward boost 1.5 toward the moon. Finish: exposure 1.0, bloom 0.07 @ 0.8, vignette 0.25.
* Easiest route: import `shots/run/world.py` (`night_light()`, `march()`, `shade()`) with an empty crag table
  (`np.zeros((0, world.NCR))`) and a camera at her summit — that is s1's world (checked visually against s1 frame
  1500 with `shots/run/test_s1match.py`; since then shade() adds a small bounce fill (Q[16]=0.10) and close-range
  snow detail, both invisible at the reveal's distances).
* To continue the RUN's look instead (run_fix): `world.run_light()` in place of `night_light()` and
  `pipe.render_terrain(..., relief=world.CLOUD_RELIEF)` (P[3] of march/shade). It switches on Q[17..23]: haze at
  the far horizon instead of "sky below the horizon", a deeper haze layer over the cloud sea (7e-5/m, 450 m),
  billowed cloud tops (60 m) shaded with a tighter wrap (0.35) and a wide normal, wisps at the mountains' feet,
  firelight on the cloud tops, and the form-driven snow rule. With the defaults (Q[17]=0, relief 0) every
  function returns exactly what it did before, so DAWN_C and s1 are untouched.

# RUN department (v2) — THE BEACON RUN + DAWN_C

## RENDER_SPEC — THE BEACON RUN (v2 frames 1520–1679 → `renders/run_v2/f_%05d.png`)

Look: run_fix (the picture critic's #5, see `review/run_fix_report.md`). Everything the render needs is in
`shots/run/` (+ `shots/montage/` toolkit, `lib/look.py`). Cached data that MUST be present (committed, not
regenerated): `shots/run/summits.npy`, `shots/run/beacons.npy` (the 1550 filler the catalogue missed is added in
code, `beacons.EXTRA`).

Fresh Linux x86 box:
```
python3 -m venv ~/ld && . ~/ld/bin/activate
pip install numpy numba scipy opencv-python-headless    # tested: py3.12, numpy 2.4, numba 0.67.0, cv2 4.10, scipy 1.17
cd the-long-dawn
# the whole shot (N workers x 1 numba thread each; ~400 MB RAM per worker):
python shots/run/render_run.py --range 1520-1679 --procs 4 --skip
# or as a cloud job (renders, checks and pushes frames to claude/render-run_fix):
python3 cloud/run_job.py cloud/jobs/run_fix.json
```
Output: `renders/run_v2/f_01520.png` … (v2 numbering, 1920×804, final look incl. motion blur). `--skip`
skips frames already on disk, so a block can be resumed. Deterministic apart from the PNG dither.

Cost: ~25–55 s/frame with 4 numba threads on a 4-core cloud box (≈2–3.5 min/frame/core); the near-terrain
frames (1520–1600) are the heaviest.

---

## What was built
One world, flown through. `world.py` extends s1's heightfield world (its `h_far` ranges, `h_cloud`
cloud sea and `h_near` summit are imported unchanged, same seeds, same moon, sky, fog, snow rule) with:
* **the shepherd's foothills** (`near_range`): s1's ridged-multifractal recipe at 420 m scale, envelope
  held ≥12 m under s1's summit-lip sight line inside s1's frustum (verified: max −12 m) so s1 never saw
  them; a spine crest along the flight line;
* a **faceted-horn primitive** (`crag`) for the pyre tower (3 irregular facets, ribs/gullies from ridged
  noise, weak strata, broken summit slabs), smooth-maxed into the range;
* close-range detail: multi-scale snow (s1's rule far away; near: ledges hold snow, ribs shed it), rock
  albedo variation, cleared rock platforms under beacons, bounce fill from the snowfields/cloud.

`rcam.py`: the column-coherent marcher needs no roll and lens-shift tilt, so each frame renders a lens-shift
SOURCE camera covering the target frustum and warps it (an exact rotation homography) to the TARGET glider
camera with true pitch and roll. Stars and the pyre are drawn in source space, so they roll with the horizon for
free. Then depth-based vector motion blur (McGuire 2012 tile/neighbour-max reconstruction, 126° shutter, 27
taps) in target space. Sparks are streaked through the moving camera: each shutter sample is projected with the
camera at that instant; the beacons' flames, glows and catch flares are laid down the same way after the blur
(over half the shutter path: the eye tracks the fires, and a small flame must keep its shape).

`run.py` (run_fix): flight dynamics. Speed/heading keys → ground track (PCHIP, no stops at keys); altitude =
terrain-following upper envelope with limited vertical acceleration + 24 m clearance and a cruise FLOOR: off the
lip the glider rides the cliff's updraft at 80–100 m over the valley (no dive), 55–68 m/s instead of 92, then
settles to eye level with the pyre; after the pass a steady climb to ~330 m. Bank = coordinated-turn bank × 0.13,
fading to × 0.05 (a stabilised head) once the long lens is on. The look: a world-anchored swing right before the
pass (half the old pan rate) with a lean toward the pyre; after it, a world-anchored pan with the signal (≤26 px
per frame) and the pitch holding the cloud-sea horizon at 23.5 % of the frame through the zoom. The lens:
54° until the pyre has gone, eased in to 42° (1590–1614: the great peak, the 1610/1620 fires and the start of
1620's chain share a frame), then on in to 29° (1632–1666) to stack the ranges for the last beats.

`beacons.py`: the chain. Beats 1540 (foothill summit, 1 km), 1560 (spine summit, 430 m), 1580 THE PYRE,
1600 (s1's great pointed peak, 7.4 km), 1620/1640/1660 (islands 4–9 km, alternating); fillers between (1550–1670,
SFX `run_far_*`); from 1620/1660 the signal races on as chains of links toward the horizon (9→38 km, each link
sooner than the last). The heroine's beacon (lit 1360) burns on the horizon where s1 showed it, drawn with s1's
own far-light recipe so it is the same light across the cut. Every fire (run_fix):
* a real bonfire at every size (fire2's tongue model, pyre-style tables; tiny far ones are rendered supersampled
  and area-filtered, so they keep a hot base and a ragged tip instead of becoming dots), sized from its distance
  and the lens at its beat so a beat's VISIBLE flame is ~26 px (big fires breathe slower: tongue periods ~Hf^0.35);
* fillers vary ~2.8:1 in size and in brightness (seeded per fire) and flicker each at its own rate;
* a 2–3 frame gold catch flare shooting up the flame, and the pool of light on the summit flaring with it;
* a smoke column (puffs thrown up by the catch, then a slower buoyant column leaning downwind) lit orange from
  below, moonlit grey-blue on its moon side, fogged by its distance;
* a warm pool of light on the snow out to ~2 fire-heights (point light, also on the cloud tops);
* seated on its summit as the renderer draws it: positions snapped to the true local summit, then to the top of
  the terrain eroded by half a march step (thin needles are skipped by the marcher), and dropped by the Earth's
  curvature as the terrain is — the catalogue's far fires otherwise floated 10–25 px above their peaks.

`fire2.py`: montage's bonfire with the critic's fix (orange → yellow → white ramp, no crimson end; the flame body
partly opaque, soot). run_fix adds the rotated emission+opacity sprite (`flame_sprite`), the shutter-sampled
compositor (`composite_blur`), `tongues_for` and the `Plume` smoke.

`world.py` (run_fix, opt-in via `run_light()` / `CLOUD_RELIEF`): billowed cloud tops (|noise| fBm from 640 m, 60 m)
shaded by the normal of their larger forms with a tighter wrap; wisps where the mountains meet the cloud;
firelight on the cloud tops; a haze layer over the cloud sea (each range paler than the last); the far horizon
ends in haze; snow by FORM: slope from a smooth central difference at the snow scale + convexity (ridges and
ledges hold snow, gullies go bare), the noise terms cut to a third and sheared by height, and nothing on walls
steeper than ~65° (a 12 m-scale slope, so faces do not streak). The pyre tower's walls are bare rock (a
negative-radius row in the platform table).

The pyre (`render_run.py`): a big cairn with an iron basket on the tower 14 m left of the track. It ignites at 1580
at 42 m, with the catch one frame before and the whoosh overshoot peaking at 1585. Six tongues are separated by
dark gaps and lean in a 6 m/s summit gale. The spark fountain streaks through the moving camera, the smoke is lit
from below, and a 48-unit point light washes the tower. We pass it at 14 m at ~1594 (the camera at eye level from
~1576), then climb away.

## DAWN_C (`dawn.py`)
Same world, the morning after, from where the Run ended (camera ≈ (60, 150→170, 480), yaw −19°, hfov 46°, a slow
20 m crane rise easing to stillness). The sun (az −30°, x≈0.27 of frame) touches the skyline at 2399.5, clears
it by ~2432 and climbs slowly (+1.5° over the shot). "The flood": the terrain is lit from a direction lifted
+2.3° over 2400–2470, with sunlight ramping 0.3→1 over 2397–2425, so the eastern ranges' long shadows sweep
back across the cloud sea while the disc itself moves gently (no time-lapse look). Dawn sky: a tight gold aureole,
a gold band on the skyline toward the sun, clear blue within a few degrees; the sun is a clean limb-darkened disc
+ bloom (no starburst), streak 0.004. Warm haze only near the sun, blue-violet air elsewhere; blue shadows.
Thin smoke rises from every summit that held a beacon (beacons.npy + the pyre), lit by the low sun. Six eagles,
one loose irregular pass, gliding (two give a couple of slow beats), from 2476: they enter at the right edge at
76–94 m (wingspan 55–69 px ≤ 4% of width), fly away toward a point right of and above the sun (never across the
disc), shrinking to ~27 px, always in the upper band (never in the text band 2512–2600).

## Re-render / tests
```
python shots/run/render_run.py --frames 1520,1580,1679 --scale 0.4 --out tests/t --procs 3
python shots/run/render_run.py --range 1520-1679 --procs 4 --skip        # final
python shots/run/plan.py                                                  # top-down plan + camera table
python shots/run/review_sheet.py --before <old frames> --after renders/run_v2 --out review/run_fix_before_after.jpg
python shots/run/catalog.py && rm shots/run/beacons.npy                   # ONLY to rebuild the chain (never:
                                                                          # DAWN_C's smoke and the SFX use it)
```
