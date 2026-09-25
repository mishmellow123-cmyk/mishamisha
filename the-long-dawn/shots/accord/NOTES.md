# ACCORD — notes

Global frames **1912–2247** → `renders/accord/f_%05d.png` (1920×804, 8-bit, `look.finish` + `look.save_png`).
The cut is 1920–2239; the rest are dissolve handles. Also `renders/accord/preview.mp4` and `renders/accord/contact.png`.

## What it is
A custom CPU renderer (numpy + numba) for a top-down perspective camera that descends and rolls.
* **Descent (1912–1995).** A night plain (moonlit fbm terrain, worn paths) with ~1,900 torch-bearers
  on 26 converging paths. Time-lapse speed at altitude slows to real time near the ground, and each torch
  draws a trail, so the plain reads as rivers of light (a rhyme with GLOBE's web). Their torchlight
  is splatted into world-space irradiance grids that light the ground. Three moonlit mist layers slide
  past for parallax. A crowd gathers outside a ring of 19 standing stones.
* **Table.** Dark weathered granite (R 1.97 m, top 0.62 m) with a hearth basin and 12 carved sun-rays. Twelve
  hooded emissaries (SDF cloaks: five hood/head-wrap/collar types, muted cloth colours, firelit sheen)
  raise their torches (1977–1990) and reach in. Their flames lean and stream inward as ribbons (one
  holdout joins last). The ribbons **fuse at 2000** into a golden, ray-marched fire volume: swirling
  tongues fanning outward, a basin-sized white core, heat haze, and sparse fine embers with a few slow
  out-of-focus ones near the lens.
* **Light.** The hearth is modelled as two on-axis soft lights. A low one rakes the carvings; a high one
  throws each emissary's long radial shadow across the floor (the sundial mandala). Moonlight is a few
  percent of the fire. The standing stones catch firelight.
* **Oaths.** Cinzel Bold is rasterised, turned into a signed distance field and warped onto arcs. It is
  carved as a **V-cut chisel groove** (depth ∝ distance inside the stroke), so the fire lights one wall and
  shadows the other. At 2000/2040/2080/2120 gold light runs along the letters (a white-hot writing
  front, groove centres first, spill onto the stone) and settles to a steady HDR glow. An eased
  **camera roll** about the hearth (20 frames, landing on each hit) brings each oath upright to the top.
* **"Together".** All 27 words from bible §8 are carved in two staggered rows on a stone band in the floor
  around the emissaries, set in Noto Serif / Serif CJK / Naskh / Nastaliq / Hebrew / Devanagari /
  Bengali / Gurmukhi / Tamil / Thai / Georgian / Armenian with RAQM shaping. Arabic, Persian and Urdu
  join and run RTL; there is no tofu (checked in `renders/accord/tests/scripts_zoom.png`). They ignite in a
  clockwise sweep from the top of frame, 2160–2220, while the oath band's border rings close (2138–2158).
* **Flare (2224→).** Hearth intensity ×55 and the fire volume goes white. The rays and inscriptions blaze.
  The camera pushes in until the centre is white at 2240 (for the match-cut to the sun).

## Re-render
```
export NUMBA_NUM_THREADS=1
python3 shots/accord/accord.py range 1912 2247 --worker 0/2 &   # two workers = 2 cores
python3 shots/accord/accord.py range 1912 2247 --worker 1/2 &
python3 shots/accord/accord.py finish                            # preview.mp4 + contact.png
python3 shots/accord/accord.py still 2030 --scale 0.5            # a test still -> renders/accord/tests/
```
`range` skips frames that already exist (use `--force` to redo). Caches (text maps, path masks)
rebuild automatically into `renders/accord/cache/` and can be deleted. The numba cache is wiped
automatically whenever any source file here changes, since numba does not track cross-module edits.
Files: `accord.py` (pipeline/CLI), `scene.py` (timeline, camera, emissaries, lights), `shade.py`
(surfaces), `geom.py` (SDFs/shadows), `fire.py` (fire volume, splats, mist, motion blur),
`particles.py` (embers/flames/ribbons), `plain.py` (paths/walkers), `textmaps.py` (inscriptions;
`python3 textmaps.py test` rebuilds the script sheet), `firetest.py`/`keycheck.py` (look-dev helpers).

## Deviations from the bible (and why)
1. **The flames merge on the 2000 hit**, together with Oath I (NO SINGLE HAND SHALL HOLD IT), instead
   of during 2010–2060. The ribbons stream in over 1986–1999, and the golden fire keeps blooming and
   sparking through 2000–2060. This puts the strongest visual moment on the music accent and ties it to
   the oath's meaning.
2. **Each oath is carved as two concentric lines.** A single ring of four long oaths can't be read at
   1920×804 (about 12 px caps). This follows the director's note. The caps are about 36–40 px during the
   oaths, with the hearth just below the inscription.
3. **The "together" ring is on a floor band around the emissaries**, not on the table, so their radial
   shadows sweep across it. For the sweep the camera rises (2136–2162) to show the whole mandala, which
   makes the scripts about 18–22 px tall.
4. **Lighting cheats for readability.** The hearth light is two on-axis lights (low for raking, high for the
   long shadows), and the ground gets a softer falloff (`P_GNDK`) so the shadow spokes read against
   the moonlit floor. Walkers move in time-lapse at altitude.
5. **T12 window (1922–1998).** Torch splats in the band y 560–700 are dimmed by 55% and the surface by about
   18%, feathered. Some torches still pass behind the text at 1988–1998, while the ring of emissaries
   is in frame.
6. **Exposure rides** from 5.0 on the moonlit plain down to 1.0 once the golden fire is lit, like an iris
   closing as the fire takes over.
