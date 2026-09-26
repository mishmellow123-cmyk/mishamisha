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

## THE BEACON RUN — v2 frames 1520–1679 → `renders/run_v2/` (look FINAL; see the RUN spec further down)
Local fallback 1660–1679 already rendered here.

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

# RUN department (v2) — THE BEACON RUN + DAWN_C

## RENDER_SPEC — THE BEACON RUN (v2 frames 1520–1679 → `renders/run_v2/f_%05d.png`)

Look is FINAL. Everything the render needs is in `shots/run/` (+ `shots/montage/` toolkit, `lib/look.py`).
Cached data that MUST be present (committed, not regenerated): `shots/run/summits.npy`, `shots/run/beacons.npy`.

Fresh Linux x86 box:
```
python3 -m venv ~/ld && . ~/ld/bin/activate
pip install numpy numba scipy opencv-python-headless    # tested: py3.12, numpy 2.5.3, numba 0.67.0, cv2 4.10, scipy 1.18.1
cd the-long-dawn
# optional warm-up (compiles numba kernels into __pycache__, ~2 min):
NUMBA_NUM_THREADS=4 python shots/run/render_run.py --frames 1600 --scale 0.25 --out tests/warm --threads 4
# render a block (N workers x 1 numba thread each; ~400 MB RAM per worker):
python shots/run/render_run.py --range 1520-1554 --procs 4 --skip
```
Output: `renders/run_v2/f_01520.png` … (v2 numbering, 1920×804, final look incl. motion blur). `--skip`
skips frames already on disk, so a block can be resumed. Deterministic apart from the PNG dither.

Split (claimed):
| frames | where |
|---|---|
| 1520–1554 | cloud A |
| 1555–1589 | cloud B |
| 1590–1624 | cloud C |
| 1625–1659 | cloud D |
| 1660–1679 | LOCAL (this Mac, 3 workers) |

Cost: ~6–7 min/frame single-threaded on this (loaded) Mac at full res; near-terrain frames (1520–1600) are
the heaviest.

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
camera with true pitch and roll. Fire, smoke, cairns and stars are drawn in source space, so they roll with
the horizon for free. Then depth-based vector motion blur (McGuire 2012 tile/neighbour-max reconstruction,
126° shutter) in target space. Sparks are streaked through the moving camera: each shutter sample is
projected with the camera at that instant.

`run.py`: flight dynamics. Speed/heading keys → ground track (PCHIP, no stops at keys); altitude =
terrain-following upper envelope with limited vertical acceleration + 24 m clearance; then a zoom climb.
Bank = coordinated-turn bank from the turn rate, rolled in 5 frames ahead through a critically damped
filter. Operator's look = critically damped follow of heading + a clamped lean toward the pyre.

`beacons.py`: the chain. Beats 1540 (foothill summit, 1 km), 1560 (spine summit, 390 m), 1580 THE PYRE,
1600 (s1's great pointed peak, 7.4 km, the peak the audience saw on the right in s1), 1620/1640/1660
(islands 4–9 km, alternating); small far ones between (1570–1670, wide; SFX `run_far_*`); from 1620/1640/1660
the signal races on as chains of links toward the horizon (25→37 km, each link sooner than the last).
The heroine's beacon (lit 1360) burns on the horizon where s1 showed it.

`fire2.py`: montage's bonfire with the critic's fix. The ramp is orange → yellow → white with no crimson end,
because a crimson tip over the blue sky reads magenta. The flame body is also partly opaque (soot), so no sky
shows through the tips. Air glow is yellow-orange and tight. Distant fires are a warm hot point plus a soft
orange aura.

The pyre (`render_run.py`): the pyre is a big cairn with an iron basket on the tower 14 m left of the track.
It ignites at 1580 at 44 m, with the catch one frame before and the whoosh overshoot peaking at 1585. Six tongues
are separated by dark gaps and lean in a 6 m/s summit gale. The spark fountain streaks through the moving camera,
the smoke is lit from below, and a 48-unit point light washes the tower. We pass it at 13 m at ~1592, then
bank left into a climbing turn toward the moon side of the sky.

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
python shots/run/render_run.py --range 1520-1679 --procs 3 --skip        # final
python shots/run/plan.py                                                  # top-down plan + camera table
python shots/run/catalog.py && rm shots/run/beacons.npy                   # ONLY to rebuild the chain
```
