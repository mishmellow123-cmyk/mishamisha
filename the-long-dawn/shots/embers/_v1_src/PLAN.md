# EMBERS: plan (frames 300–1199)

The legend told in the fire: one continuous vision made of glowing particles in
real 3D space, seen through a moving perspective camera (thin-lens depth of
field, 180° shutter motion blur), with bloom and faint lit smoke on deep
darkness. Everything is a pure function of time, so any frame can be
re-rendered on its own and frame ranges can go to separate processes.

## World layout (world units, Y up)

* Origin `O = (0,0,0)`: the Kindling point, where the glyphs converge and the fire ignites.
* Ground (dark, never drawn as a surface) at `y = -14`. Towers come up out of it.
* Ring of **7 towers** at radius `R = 18`: needle spire, stepped ziggurat, lattice
  mast, ringed cylinder, twisted prism, tiered pagoda with flared eaves, and a
  pod tower (a sphere on a shaft). Seven gives an asymmetric ring.
* Crown floats at `y ≈ +5` by 640 and climbs during the race.
* **Radial** walls (between neighbouring towers, from the centre outward) *divide*
  the kingdoms into sectors.
* The globe (880–950) and the hand (960–1040) share the same renderer. The globe
  sits in its own scene space and is reached by a cut.

## Beat sheet: composition and camera

| frames | beat | what we see | camera |
|---|---|---|---|
| 300–340 | embers | Bright orange embers rise from a warm glow below frame centre (the Elder's torch, continuing the INTRO crossfade), some drifting toward the lens as big soft bokeh. They slow down, and from about 322 each ember **opens into a tiny letter**. | Low, tilted up. Rises with the embers and slows as they slow. |
| 320–400 | fireflies | Thousands of glyphs from every script and discipline fade in all around: near ones legible and gently defocused, far ones pinpricks. They drift and meander like fireflies and twinkle. They are warm ember-gold. | Slow push forward *through* the cloud. Glyphs slide past the lens (parallax), with a rack focus onto hero glyphs. |
| 400–470 | the spiral | The cloud flattens into a tilted disc and winds inward, a spiral galaxy of writing. Inner glyphs go first ("down the drain"). They accelerate, streak, shrink and heat from gold to white, and a single point builds at the centre. | Pulls back and up (about 35° elevation) so the spiral reads, then eases in. The point sits at about (960, 330). |
| 470–480 | the point | One blinding point with a faint anamorphic streak. It trembles and draws in a breath. | Near still, micro drift. |
| **480** | **IGNITION** | Flash, then a spherical shockwave of streaking sparks. The point blooms into the **thinking fire**: a white core, ice-blue circulating inner light, gold flame edges, and branching filaments inside along which bright pulses travel. Its toroidal flow (up the middle, out over the top, down the sides) is structured, *not* ordinary fire. It breathes once per bar. | Medium-close with a slow orbit. |
| 520–600 | towers | Seven towers rise out of the dark ground around it, with a bright emergence line at the base and ember sparks. Each has its own silhouette. Their inner faces are lit ice-gold by the fire, their outer faces are dim ember red, and tiny window lights show. | Long pull-back and crane-up from inside the ring, **passing between two towers** (strong parallax), to outside the ring. |
| 560–640 | crown | The fire lifts to y≈5, and its toroidal flow opens into a **ring**: a floating crown of fire with flame tines. It turns slowly. | Settles on a wide framing: the crown in the upper-centre third, framed by the nearest towers. T5 over it, with a calm lower band. |
| 640–800 | RACE | On every beat (640, 660, …) all towers **surge** upward in hard eased jolts, spraying streaking sparks. The leader changes from beat to beat (leap-frogging). From 680 radial **red walls** of rising embers divide the kingdoms. The palette bleeds orange to red to crimson. Smoke is lit red from below. | Flies **up between the towers**: the camera is in a sector, with a wall receding to the crown as a one-point-perspective divider. It keeps climbing and gets a shake on each beat. |
| 800–880 | storm | The crown **balloons into a vast unstable vortex**: a spiral disc of fire with a white-hot eye, red and crimson arms, flickering, and lightning-like arcs. It dwarfs the towers. | Keeps rising, pulls back, and tilts to take in the vortex. |
| 880–950 | the world | Cut on the beat to **the world as an ember globe** (Natural Earth land as ember points, ocean nearly dark). **Cracks of fire** spread from the ring's location, the plates part, and fire bursts through. | Slow push-in and drift. The globe sits in the upper two-thirds. |
| 950–960 | flare | The cracks flare and the globe blows out. | |
| 960–1040 | GRASP | Cut on the downbeat. The crown is compact again, blazing at the eye of the dimmed storm, above the towers. A **colossal hand of embers** (a rigged palm, 4 fingers × 3 phalanges, and a thumb) rises from below, its rim lit by the crown. The fingers spread, it reaches, then it clenches, with the **fingers closed at 1036** and light leaking between them. 1036–1039: a white-red flash. | Low, looking up at the crown. A slow push-in. |
| **1040** | cut | Pure black. | |
| 1040–1199 | SILENCE | One small ember drifts down and sways from the upper-middle. It dims and flickers, and dies by about 1150 with a tiny wisp. The lower third stays clear for T8. Black after that. | Locked off. |

Text windows (the band y≈560–700 is kept calm): T3 340–440, T4 490–550,
T5 565–635, T6 660–730, T7 820–900, T8 1055–1195. I get there through composition
(the action sits in the upper two-thirds) plus a soft screen-space attenuation of
particles inside the band during those windows.

## Rendering technique

* **Point splatting in numba** (`core.py`). Each particle is evaluated at shutter-open
  and shutter-close (180°), projected through a pinhole camera at both times (so
  camera motion blurs too), and splatted as a line of sub-splats (motion
  streaks). The footprint radius comes from projected size, thin-lens circle of
  confusion (`A·f·|1/z − 1/z_focus|`) and a minimum anti-alias footprint, and is
  energy-normalised exactly. Defocused particles get a controlled boost (bokeh).
  Large footprints go to 1/2, 1/4 or 1/8 resolution buffers, so cost per particle stays
  bounded. Each thread has its own accumulation buffer (2 threads).
* Colour: `look.blackbody()` temperature per particle, flicker, and the
  mind/race palettes. Depth haze attenuates distant particles. Smoke is big, faint,
  low-resolution puffs lit by the fire (inverse-square).
* Motion: easing curves, analytic ballistic sparks with drag, spring-like
  jolts (`easeOutBack`), coherent 3D gradient-noise displacement for turbulence.
* Shapes are precomputed once and cached: glyph point sets (fonts through PIL/raqm,
  mask, sampled points), tower surfaces with normals and windows, fire filament
  trees, globe land points (point-in-polygon on ne_110m_land), and hand-rig
  capsules.
* `look.finish()` with per-section exposure, bloom and streak.

## Order of work

1. Renderer and camera, then a speed test.
2. Glyph atlas, then a legibility test.
3. Build every section roughly, then **full-length low-res pass**, then a full-res pass so the edit has something.
4. Iterate on the weakest beats with key-frame stills and contact sheets, then re-render.
