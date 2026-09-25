# GLOBE — notes

Two planetary shots rendered by one CPU ray tracer (numba, 2 threads).

| shot | frames delivered | cut | file |
|------|------------------|-----|------|
| THE WORLD ANSWERS | 1752–1935 | 1760–1919 | `renders/globe/f_01752.png` … `f_01935.png` |
| DAWN | 2232–2495 | 2240–2479 | `renders/globe/f_02232.png` … `f_02495.png` |

Also `renders/globe/preview.mp4` (both shots back to back, 960 px) and `renders/globe/contact.png`.

## Re-render

```
python3 shots/globe/prep.py                                   # caches (idempotent, ~30 s)
NUMBA_NUM_THREADS=2 python3 shots/globe/render.py answers 1752 1935
NUMBA_NUM_THREADS=2 python3 shots/globe/render.py dawn 2232 2495
python3 shots/globe/render.py --finalize                      # preview.mp4 + contact.png
# stills for review:  render.py dawn --frames 2240,2300 --scale 0.5 --test
```

Every frame is a pure function of its global frame number (no state between frames), so any
range can be re-rendered in isolation or split across processes.

## What is in the files

* `prep.py`: builds caches in `renders/globe/cache/`: an anti-aliased 8192×4096 land mask
  rasterised from Natural Earth 50m land; an albedo map (land colour from NASA Blue Marble pushed
  into the crisp mask, so coastlines stay sharp; a procedural deep-blue ocean); a 2.5 M-point
  night-light cloud importance-sampled from NASA Black Marble (lights extracted from its blue
  moonlit base) plus the assets DMSP lights, plus population-weighted sprawling city cores from
  `ne_10m_populated_places_simple`; clouds and normals from the assets.
* `globe.py`: the renderer. It does per-pixel ray/sphere tests with 4×4 supersampling only on
  the limb and the sun disk. The atmosphere is single-scattering Rayleigh + Mie + ozone from a
  precomputed transmittance LUT, with a non-uniform march and the Earth's shadow; the shell is
  thickened (X = 2.0 / 2.6) with optical depth kept, so the band reads at 1920 px. There is a
  moonlit rim gain on grazing rays and a hairline airglow integrated over its exact shell
  crossings. The surface has a land-relief normal map, GGX ocean glint, a Fresnel sky
  reflection, moonlight with a faint moon glint on water, and a twilight skylight near the
  terminator. Clouds use the map as low-frequency coverage only; the shapes come from two-scale
  domain-warped fbm with a soft opacity ramp, plus streaky cirrus. Cloud tops are lit at their
  own height (so they stay lit past the ground terminator), get relief shading from the density
  gradient in raking light, and cast shadows. Night lights, stars and all fire are splatted in
  screen space (resolution independent). The file also holds the sun lens layer (core, halo,
  16 diffraction spikes with chromatic tips, a burst flash) and `finish_frame()`.
* `web.py`: the golden web. Nodes are the Himalayan origin, about 430 curated heights, deserts,
  ice, karst, islands and coasts on every continent (`places.py`), plus distance-thinned
  cities and relief-weighted hill beacons. A seeded spread simulation runs over a spherical
  Delaunay graph: each new light throws 2–3 arcs outward with angular diversity, designated
  ocean leaps sail over the limb, "late answers" fill gaps, and sparse cross-links close loops.
  Arcs are great circles lifted by a sin profile (7% of length for hops, 15% for leaps).
  Threads are thin and cool from white-gold to amber after the fire passes; they are brighter
  where lifted and fainter edge-on at the limb. Now and then a soft pulse of light runs along a
  settled thread. Comet heads, shed sparks, beacons (the brightest elements) with a small
  anamorphic glint, and an ignition flare complete the look. `Hearths` holds the DAWN village
  lights.
* `shots.py`: the two shots: camera rigs, lighting, timing, exposure. `render.py`: the CLI.

## Beats (global frames)

THE WORLD ANSWERS: the first beacon already burns in the Himalaya at 1752; it flares at **1760**
(choir) and throws its first three arcs; the web grows exponentially (arc landings per 10
frames: 1, 5, 8, 19, 28 … 60+) and by ~1890 the visible night side of Asia is laced, with
leaps over the limb toward the Americas, Alaska and the Indian Ocean islands. The camera is a
slow, eased pull-back and rise from 2,300 to 8,200 km, drifting west.

DAWN: the camera is over central Africa looking east. 2232–2239: the limb above the sunrise
point glows. **2240**: the sun's upper edge breaks the limb near frame centre (x ≈ 990,
y ≈ 330), in a white-gold burst (flash decays in ~15 frames), diffraction spikes and an
anamorphic streak from `look.streak`. The terminator then races toward camera: gold light on
cloud tops near the horizon by ~2300, and about a third of the disc in dawn light by
~2400–2480. The web and the hearths fade as daylight takes each region. From ~2280, thousands
of village hearths bloom across dark Africa and Arabia, spreading out from the beacons. The
camera rises from 2,400 to 4,300 km. During the text windows the band y≈545–715 is calmed:
web and hearths at 50%, and the sun's spikes masked out of it.

## Deviations / choices (and why)

* **Lighting cheat (DAWN):** the sunlight on the surface leads the visible sun disk by up to
  26°. Physically, from orbit the lit sliver stays tiny until the sun is far above the limb.
  The cheat lets the light flood the world while the sun stays near the limb and frame centre.
  The sun's apparent elevation above the limb is scripted relative to the camera (it crests
  exactly at 2240).
* **Weather (DAWN):** the cloud field is shifted 16° west / 2° south and cleared near the sunrise,
  so the African cloud mass sits where the terminator sweeps and land and sea read by the
  sunrise. It is artistic weather, not a specific date.
* **Atmosphere thickness** is exaggerated (see above), and the airglow is placed a little higher
  than real (135 km) so it separates from the thicker haze as a hairline.
* **Finish:** `finish_frame()` uses look's bloom / streak / vignette / tonemap in the same order
  as `look.finish()`, with two differences. The sun's lens layer is added after the bloom, so the
  glare is not bloomed a second time into a grey veil. `look.streak` is driven by the sun layer
  only, so the hundreds of HDR beacons don't each smear sideways.
* **Imagery provenance:** `earth-blue-marble.jpg` (NASA Blue Marble Next Generation, Reto
  Stöckli / NASA Earth Observatory) and `earth-night.jpg` (NASA Earth Observatory / NOAA NGDC
  "Earth at Night" Black Marble 2012), both public domain. They come from the npm package
  `three-globe@2.45.2` (`example/img/`), which `prep.py` fetches with `npm pack`. Both are
  4096×2048. The package's `clouds.png` was **not** used because its provenance is unclear.
  Everything else is the assets folder plus procedural generation.
