# GLOBE — plan

Two planetary shots, one renderer. Both share one Earth, one golden web, one sky.

## Shots (global frames)

### THE WORLD ANSWERS — 1752–1935 (cut 1760–1919, handles for dissolves)
* **Frame idea:** the night side of the Earth seen from high orbit *south of India, looking
  north* — the dense speckle of India's lights in the foreground, the black wall of the
  Himalaya / Tibet in the middle, Central Asia and Siberia beyond, the limb arcing across the
  top of frame with a thin moonlit/airglow rim and stars above. Europe/Arabia/Africa on the
  left, China/Japan on the right.
* **Beat:** at 1752 only the first beacon burns — one gold point high in the Himalaya.
  1760 (choir): it pulses and the spread begins. Each beacon, a few frames after it catches,
  throws 2–3 arcs of fire to the next heights and cities; short hops hug the ground, long
  great-circle arcs lift off to leap seas and, near the limb, sail over the horizon toward
  the Americas / Africa / Australia against the stars. Growth is exponential then saturating:
  by ~1900 the visible night side is laced with gold.
* **Camera:** slow, heavy pull-back and rise (≈3.5k → 9k km altitude, eased), with a slight
  drift west so the web's reach (Europe, Africa) comes into view. No robot moves.

### DAWN — 2232–2495 (cut 2240–2479) — the climax
* **Frame idea:** in orbit above the curved limb, looking **east** into the sunrise. Night
  below (Africa / Arabia / Central Asia) with the complete golden web still glowing; the
  sunrise point on the limb near frame centre (≈ x 960, y 330–380) for the match-cut from the
  hearth's white flare.
* **2232–2239:** the limb above the sunrise point already glowing (pre-dawn forward scatter),
  brightening fast.
* **2240 (downbeat):** the sun's upper edge breaks the limb — a "diamond ring" bead that
  blooms into a white-gold starburst (diffraction spikes), anamorphic gold streak
  (`look.streak` on the sun layer only), veiling glare; the atmosphere ignites in a thin
  layered band (deep orange at the surface → rose → white-gold → blue → black), brightest at
  the sun and spreading along the limb.
* **2240–2480:** the sun clears the limb; sunlit cloud tops near the horizon catch
  rose/gold, then the terminator races toward camera across the clouds (surface-lighting sun
  rotated faster than the visible disk — a deliberate cheat; physically the lit sliver would
  stay tiny), ocean glint path under the sun. On the remaining night side, thousands of
  small warm "hearth" lights bloom into places that were dark (rural Africa, the Sahel,
  Central Asia), spreading outward from the web's nodes. Camera rises slowly (≈2.5k → 5k km),
  curvature grows. Exposure adapts after the flash.
* **Text windows 2270–2345, 2352–2440:** band y≈560–700 kept calm (night side, web dimmed
  there, no glare/streak crossing it too hot).

## Technique
* **Renderer** (`globe.py`): numba, 2 threads. Per-pixel ray vs. sphere (Earth radius = 1)
  and atmosphere shell; analytic limb anti-aliasing; supersampling only where needed.
* **Atmosphere:** single-scattering Rayleigh + Mie + ozone absorption with a precomputed
  transmittance LUT T(r, μ); non-uniform ray-march concentrated at the tangent point;
  thickness exaggerated (≈2–3×, optical depth preserved) so the band reads at 1920 px; Earth
  shadow on the atmosphere; moonlit variant for the night rim; faint airglow line.
* **Surface:** land/sea mask rasterised at 8192×4096 (anti-aliased) from Natural Earth
  50m land; land colour from NASA Blue Marble 4k (public domain, via the `three-globe` npm
  package) pushed into the crisp mask; procedural dark ocean with GGX sun-glint + Fresnel;
  normal map (assets) for relief at grazing sun; clouds = assets cloud map as coverage +
  procedural fbm/warped detail, cloud-top lighting that stays lit past the ground terminator,
  cloud shadows; moonlight on the night side; twilight sky-light across the terminator.
* **Night lights:** a point cloud (~1–2 M) importance-sampled from NASA Black Marble 2012
  (4k, lights extracted from the blue base) + the assets DMSP lights, plus
  population-weighted city cores from `ne_10m_populated_places_simple`; splatted in screen
  space (resolution-independent, crisp), energy ∝ 1/d², fade across the terminator,
  dimmed by clouds and atmosphere.
* **The web** (`web.py`): nodes = curated high places on every continent (Himalaya origin,
  peaks, deserts, ice, karst, islands) + spaced cities; a seeded branching process (each
  new light spawns 2–3; some long ocean leaps) gives ignition times; arcs = great circles
  lifted by a sin profile ∝ length, drawn as AA HDR line splats with white-hot comet heads,
  ember sparks shed along the path, steady golden threads behind, flickering beacon nodes
  with ignition flashes. Occlusion vs. the Earth per vertex. HDR gold (`accord_*`, `dawn_*`).
* **Sky:** procedural star field (magnitude-distributed, coloured) + faint Milky Way; stars
  hidden by the Earth, dimmed through the atmosphere, faded by exposure after sunrise.
* **Sun:** analytic limb-darkened disk seen through the atmosphere (reddened, partially
  occluded by the limb) + post starburst/glare/streak/ghosts.
* **Finish:** `look.finish()` (bloom, vignette, ACES), exposure animated per shot.

## Process
1. `prep.py` builds caches in `renders/globe/cache/` (land mask, albedo, lights points, web).
2. Low-res stills at key frames → look → critique → iterate (WIP in `renders/globe/tests/`).
3. Contact sheets of consecutive frames to check motion.
4. Full render early (half-res if needed), then improve and re-render at full res.
5. `preview.mp4`, `contact.png`, `NOTES.md`.
