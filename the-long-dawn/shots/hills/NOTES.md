# HILLS: notes

Deliverables (global frame numbers, 1920×804, `look.save_png`) go to `renders/hills/f_%05d.png`:

* **INTRO** 0–359
* **FIRST BEACON** 1200–1439
* **CODA** 2460–2807

Also in `renders/hills/`: `preview.mp4` (the three pieces back to back), `contact.png`, and
`coda_fires.json`. The JSON holds the ignition frame of every answering fire in the CODA
wave. The first 18 are marked `voiced`, for the bell cascade. Test renders are in
`renders/hills/tests/`.

## How to re-render

```
cd the-long-dawn
python3 shots/hills/render.py --shot intro  --frames 0-359     --workers 2
python3 shots/hills/render.py --shot beacon --frames 1200-1439 --workers 2
python3 shots/hills/render.py --shot coda   --frames 2460-2807 --workers 2
#   --scale 0.5 for tests, --out <dir>, --step N, --skip-existing
```

Each worker is one process with `NUMBA_NUM_THREADS=1`. Every frame is a pure function of
its frame number: the simulations (scarf, hair and wisp verlet chains, particle systems)
are deterministic and re-run from the start of each piece in every worker. After editing
a shared `@njit` helper in `core.py`, `sky.py` or `land.py`, run `shots/hills/clear_cache.sh`,
because numba's disk cache does not track cross-module inlining.

## What is built (all procedural)

* `core.py`: the pinhole camera (metres; x right, y up, z forward), easing and keyframe
  tracks, numba Perlin noise and fBm, energy-normalised gaussian, disc and streak
  splats, and compositing.
* `sky.py`: the twilight gradient with an asymmetric afterglow, a 14–26k-star catalogue
  (magnitude power law, colour, twinkle, horizon extinction, visibility against sky
  brightness), and the Milky Way. The Moon is a 2× crescent with earthshine and clustered
  warm city lights on its night side. The **orbital ring** is a geometric equatorial ring
  seen from 57°N. It is sunlit toward the sunset and cut by Earth's shadow on the far side,
  with faint station nodes.
* `land.py` + `hillworld.py`: 8 ridge "cards" at real distances (480 m–56 km), each
  defined by the apparent angle of its crest. They have trees, analytic anti-aliased
  edges, aerial perspective, noise-varied valley mist, beacon glow on the hills and a
  depth buffer. A moonlit-snow shading mode (pyramid peaks, lit and shadow faces, a
  crest-hugging cloud sea) is used for the Himalaya.
* `puppet.py` + `characters.py`: 2-D SDF puppets on camera-facing cards: smooth-union
  capsules, ellipses and polygon profiles, two-bone IK, and verlet scarf, hair and wisps.
  **Backlit-silhouette shading** gives near-black interiors, a thin cool rim from the sky
  behind, and a thin warm rim where the fire grazes an edge. The scarf is the only colour:
  dark oxblood that glows ember-red where the fire reaches it, with a fringed tail. The
  Elder, the Child (pom-pom hat) and the Young Woman share one rig style. The same scarf
  function is used for the Elder and the Young Woman.
* `fire.py`: the HDR flame (multi-scale domain warp, torch and bonfire envelopes, splitting
  and tearing tongues, white-yellow core low, broken red tips), ignition with
  whoomp-and-overshoot, flicker, distant beacon points with halo, spark and ember particles
  (tapered hot-head streaks, cooling colour, curl-noise meander, bokeh when defocused),
  smoke lit only near the fire, and a veiling glow.
* `intro.py`, `beacon.py`, `coda.py`: the three pieces. `render.py` is the CLI.

## The pieces

**INTRO**
* 0–159: wide blue-hour establishing shot. Beacons ignite one by one at 22, 80, 112, 138
  (then 196, 238 and 262 behind the two-shot). The Child points at the fire at 80, turns
  (about 104) and looks up at the Elder, under T1.
* 160–359: medium two-shot (cut on the bar-3 downbeat). The Elder looks down at the Child,
  then out to the fires (T2). From about 228 the camera pushes into the torch, the wind
  drops, and the frame tilts up a column of rising embers over the flame's glow. By 300
  this matches the first EMBERS frame: a warm glow at the bottom centre and embers rising
  on near-black.

**FIRST BEACON** is one take.
* 1200–1235: black, with a breath of snow.
* Strikes at 1236, 1262 and 1290 each give a flash and a spark shower into the basket, and
  light the hands, the steel, her profile and the scarf wrap. They grow in strength.
* A spark lives in the tinder and pulses as she blows. At 1318 the kindling catches,
  lighting her face from below, and her breath shows.
* At 1360 the beacon roars. She rises with an arm up against the heat. The camera pulls
  back and up, with an explosive ease-out, to the snowy summit, a sea of moonlit peaks and
  cloud, spindrift, the stars and the Milky Way. There is no ring or lit Moon, because this
  is sixty years earlier. The moonlit world fades up as it is revealed.

**CODA**
* 2460–2627: medium shot. The Child looks up (T15). The Elder meets her eyes, looks away
  and says nothing. A gust lifts the scarf at 2548–2618.
* 2580–2618: the Elder lowers the torch into the Child's hands and moves her hand to the
  Child's shoulder.
* 2628–2807: wide. The Child lifts the torch into the basket, and it **catches at 2640**.
  About 50 answering fires ignite in a rolling wave outward and across the valley between
  2645 and 2720 (see the JSON). The INTRO fires are still burning.
* From about 2662 the camera cranes up into the sky: the ring, the lit Moon, stars and four
  slow ship lights. The centre of the frame is dark sky for the title.

## Deviations from the bible

* **The Moon is drawn at about 2× real size** so that its city lights read.
* **The orbital ring is placed for the picture, not exact astronomy.** Its sky path is
  physical: an equatorial ring at 2.6 Earth radii seen from 57°N. The Earth-shadow cut is
  placed where it reads in frame.
* **FIRST BEACON uses 2.5-D cards, as the director allowed.** The mountains are pyramid
  ridge cards with moonlit and shadowed faces, not a true heightfield. The look is stylised
  and graphic, like a woodcut.
* **Cairn scale.** The cairn is 0.76 m, with the basket rim at about 1.12 m, so the Child
  can reach it.
* **Frame edges.** INTRO starts at full exposure; the edit handles the fade-in. The CODA
  ends on the sky; the edit fades it to black.
