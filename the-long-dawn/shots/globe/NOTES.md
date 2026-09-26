# GLOBE — notes

## v2 (2026-09-26): fire, not fibre

The critics (review/critic_tone.md B2, critic_framing.md M5 + m9) found the v1 globe's answering
world ending as a uniform lattice of nodes and great-circle arcs (the "connected planet" of
telecom/"global AI" films), the same mesh printed over the sunlit Earth at the climax under an
8-ray starburst, and village "hearths" blooming across dark Africa under the resolution line.
v2 replaces the web with fires and cleans up the dawn.

| shot | frames (src) | cut A | cut B |
|------|--------------|-------|-------|
| THE WORLD ANSWERS | 1752–1935 (edit plays 1760–1927) | `renders/globe_v2/` | same frames (no text there; B falls through to globe_v2) |
| DAWN | 2232–2495 (edit plays 2240–2495) | `renders/globe_v2/` (text calm 2352–2440) | `renders/globe_B/` (no calm) |

The v1 frames stay in `renders/globe/`; the stale partial v1 cut-B dawn was moved to
`renders/globe_B_stale_backup/`. v1's `shots.py`, `render.py` and NOTES are in `v1_backup/`.

### THE WORLD ANSWERS v2 (`fires.py`)
* **Where the fires are.** Beacon sites sit where people have always lit signal fires:
  * **range crests** — a Hessian ridge measure of an elevation proxy (Frankot–Chellappa
    integration of the relief normal map, the MAP department's method), non-maximum suppressed to
    thin crest lines and kept by hysteresis (Himalaya, Karakoram, Hindu Kush, Kunlun, Tian Shan,
    Zagros, Elburz, Caucasus, Ethiopian highlands, Andes, Rockies, Scandinavian range…), ~70–150 km
    apart;
  * **the great rivers** (79 hand-traced courses, copied from MAP into `rivers.py`), ~150–320 km;
  * **coasts** (Natural Earth 50m, no Antarctica / high Arctic), ~210–470 km;
  * **lonely fires** over open land (desert, steppe, forest; uniform over land, not weighted by
    night lights), ~340–500 km; islands and the ocean-leap landfalls.
  Spacing is noise-modulated (clusters and gaps), crest/river fires are nudged off the exact line,
  and every beacon gets 0–3 lesser fires 6–26 km around it, lit 3–16 frames later (a hill-top and
  its village) — so nothing reads as a string of beads or a dot map. 1414 beacons + 1470 lesser
  fires worldwide; ~450 beacons in view at the end.
* **How the fire spreads.** A chain graph: crest links (only where the connecting line runs along
  the crest), consecutive river and coast links, at most two line-of-sight links per site (short
  sea crossings allowed), the designated ocean leaps (`places.LEAPS`). Each watcher throws to at
  most 2–3 unlit neighbours (cheapest first: crest 0.72×, river 0.9×, coast 0.95×, line of sight
  1.3–1.6× per km), one after another; a site nobody threw to is answered late by a lit neighbour.
  So the fire runs as long chains along ranges, rivers and coasts, branching now and then.
  **The first beacon throws far**: three heavy embers 450–550 km west (Annapurna massif), east
  (Bhutan Himalaya) and south (the Ganges), and the chains grow back and onward from where they
  land. Travel time s (km-like) maps to frames as f = 1760 + K·s^0.35 (K puts the throws down 34
  frames after 1760): slow and heavy at first, then faster and faster, no seams.
* **Beats:** 1752 the first beacon burns alone · **1760 it flares** (a burst of light, a warm wash
  over the range, a slow stream of sparks drifting up; a few fires answer around it 1764–1782) ·
  1766–1772 the three throws lift off, land 1791–1798 · 1800–1840 fuses burn along the Himalaya ·
  1840–1900 the explosion (40–90 visible ignitions per 10 frames), leaps sail over the limb toward
  the Americas, Alaska and the Indian Ocean islands (1838–1890) · ~1912 the visible night side is
  studded with fire; nothing new after ~1915 (the dissolve starts at 1912).
* **Embers.** Every link is a travelling ember: a small gold-white head and a short tail that
  fades both behind the head (≤ 26% of the hop, ≤ 50 km; leaps 420 km) and in time (τ 2.8–6
  frames); gone within ~1 s of landing, nothing persists. Chain hops are low arcs (3.5% of length),
  sea crossings 8%, the throws 7.5%, leaps 15% (they sail against the stars). A few sparks shed
  from each head.
* **Fires.** Small flickering flames of uneven size (lognormal, crest bonfires biggest; 6% big
  bonfires), each with its own colour temperature; the flicker (three incommensurate 1–4 Hz
  breaths + a lick) also moves the colour (brighter = yellower). A catching fire flares (a
  2.6-frame flash) and builds to its steady size over ~10 frames; the bigger ones throw a few
  sparks. Near the limb they dim and redden through the long air.
* **Firelight on the land.** Each fire lights ~16–35 km around it in a screen-space light buffer
  (1/4 res) multiplied into the moonlit surface render (capped), so snow, rock and cloud catch the
  light and the dark sea does not; the first beacon's flare washes the range.
* The city lights are a little lower (0.3e-7, was 0.5e-7) so the fires are the protagonists.

### DAWN v2 (`shots.Dawn`, `lens.py`)
* **No arcs, no hearths.** Every answering fire burns on the night side (the whole net, lit) and
  **pales as the terminator reaches it**: from sun elevation −1.5° to +5.5° it dims to nothing and
  its colour goes pale gold; its light on the land goes first. The sun takes over from the beacons.
  (`LONGDAWN_HEARTHS=1` would bring v1's hearths back; `LONGDAWN_NO_HEARTHS` is now moot.)
* **Lens.** The 16-spike diffraction starburst is gone: a clean disc glare (hot core, soft halo,
  wide veil), a round flash as the sun breaks the limb (decays in ~5 frames; the edit adds its own
  white flash over 2240–2254), and a thin, restrained anamorphic line (σ 1.25 px, ±~240 px
  falloff, fading after 2262). `look.streak` is no longer used.
* **Framing ('aden').** The camera moved north and a little higher (over the Sahel, 2,700 →
  5,200 km, heading 72° → 70°): the sun now breaks the limb over the Arabian Sea **in the notch
  between Arabia and the Horn**, so the first light falls on Asia and Africa at once (the Gulf of
  Aden catches the glint, the Red Sea runs down from it), with Iran, the Caspian and the Black Sea
  on the far-left limb and East Africa on the right: several continents share the light, no single
  region receives it. The sun's screen position at 2240 is (981, 352) (v1: 993, 352), so the
  match-cut from the hearth's flare still lands. `LONGDAWN_DAWN_CAM=v1` restores the v1 camera.
* Kept: the sun crest at 2240, the terminator sweep (lighting sun leads the disc by up to 26°),
  the camera rise, the atmosphere band, the weather.
* **Text calm (cut A only):** "In time, the race was over." (v2 2512–2600 = src 2352–2440): the
  band y 545–715 keeps its fires at half strength (ramps 2343–2353 / 2439–2449). The v1 window
  2270–2345 is gone (no text there any more). Cut B (`globe_B`) has no calm; outside the window
  A and B are identical.

### Re-render (v2)
```
source ~/.venvs/longdawn/env.sh; export NUMBA_NUM_THREADS=1
python3 shots/globe/prep.py                                    # v1 caches (idempotent)
# fires_geo_v1.npz and fires_net_v8_3.npz build themselves in renders/globe/cache (~10 s)
LONGDAWN_GLOBE_OUT=$PWD/renders/globe_v2 python shots/globe/render.py answers 1752 1935
python shots/globe/render.py dawn 2232 2495 --both            # A -> globe_v2, B -> globe_B, one planet pass
# stills: LONGDAWN_GLOBE_TESTS=<dir> python shots/globe/render.py dawn --frames 2240,2400 --scale 0.5 --test
```
Every frame is a pure function of its frame number; split ranges across processes freely
(`--frames a,b,c --skip-existing`). Changing any constant in `FireNet._build/_spread` needs a
`VERSION` bump (the net is cached).

RENDER_STATS_PLACEHOLDER

### Known weaknesses (v2)
* The fires are points of light at orbital scale; what makes them fire is flicker, colour,
  unevenness and the light they throw — watch the playback for whether the flicker reads.
* In the middle of the spread (1840–1890) a fire that throws two embers still makes a brief "V"
  of two short comets; siblings leave one after another to soften it.
* The crest detector finds big ranges well but misses escarpments (Western Ghats, Great Dividing
  Range, the Urals are weak); coasts and rivers carry those regions.
* DAWN's night side (Sahel, Sudan) is sparser than Asia's end state: it is desert and savanna with
  crest, river and lonely fires only (by design not weighted by night lights).
* The sunrise over the curve of the Earth is still the classic orbital sunrise (m9); v2 removes the
  mesh and the starburst that made it a keynote slide, and shares the light across continents.

---

## v1 (delivered 2026-09-25; frames in renders/globe/)

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
range can be re-rendered in isolation or split across processes. Render time at 1920×804 with
2 numba threads on the shared, loaded machine: about 5.5 s/frame for THE WORLD ANSWERS and
about 8.5 s/frame for DAWN, plus about 20 s of JIT/cache warm-up per process.

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
  crossings. The surface has a land-relief normal map, GGX ocean glint, a Fresnel blue-sky
  reflection on the sea, moonlight with a faint moon glint on water, and a twilight skylight near the
  terminator. Clouds use the map as low-frequency coverage only; the shapes come from two-scale
  domain-warped fbm with a soft opacity ramp. A fine-scale erosion feathers thin cloud into
  wisps while thick cores stay whole, and there are streaky cirrus veils. Cloud tops are lit at their
  own height (so they stay lit past the ground terminator), get relief shading from the density
  gradient in raking light, and cast shadows. Night lights, stars and all fire are splatted in
  screen space (resolution independent). The file also holds the sun lens layer (core, halo,
  16 diffraction spikes with chromatic tips, a burst flash) and `finish_frame()`.
* `web.py`: the golden web. Nodes are the Himalayan origin, about 420 curated heights, deserts,
  ice, karst, islands and coasts on every continent (`places.py`), plus distance-thinned
  cities and relief-weighted hill beacons. A seeded spread simulation runs over a spherical
  Delaunay graph: each new light throws 2–3 arcs outward with angular diversity (the first
  three generations travel 2.2×, 1.5× and 1.2× slower, for weight while big on screen); designated
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
(choir) and throws its first three arcs, which land heavily around 1778–1790. The web then
grows exponentially (arc landings per 10 frames from 1760: 0, 1, 3, 4, 9, 15, 17, 26, 36 …
40–58), and by ~1890 the visible night side of Asia is laced, with
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
* **Weather (DAWN):** the cloud field is shifted 16° west / 2° south and thinned within 17° of the sunrise,
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
