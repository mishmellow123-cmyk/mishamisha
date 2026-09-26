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

## Re-render / tests
```
python shots/run/render_run.py --frames 1520,1580,1679 --scale 0.4 --out tests/t --procs 3
python shots/run/render_run.py --range 1520-1679 --procs 3 --skip        # final
python shots/run/plan.py                                                  # top-down plan + camera table
python shots/run/catalog.py && rm shots/run/beacons.npy                   # ONLY to rebuild the chain
```
