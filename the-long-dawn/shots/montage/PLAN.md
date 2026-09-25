# MONTAGE (BEACONS) — plan

Frames 1440–1767 (global), 1920×804, six shots, hard cuts at 1520/1580/1640/1680/1720,
tail handle 1760–1767. Ignitions on the music hits: 1480, 1540, 1600, 1650, 1690, 1730.

## One idea per shot
| # | frames | world | the one idea | ignition | camera |
|---|--------|-------|--------------|----------|--------|
| 1 | 1440–1519 | far peak | *a watcher sees the first light on the horizon — and answers* (Friedrich's Wanderer, back to us; a held torch haloes him; far pin-prick flares ~1446; he turns, thrusts the torch into the cairn) | 1480 | slow push toward the horizon light; kick + tilt up with the flames |
| 2 | 1520–1579 | desert | *a lone robe on a knife-edge crest under the whole galaxy* (raking moon, sharp lit/shadow crests, Milky Way arch) | 1540 | slow truck right (dune parallax) |
| 3 | 1580–1639 | ice | *warm fire under a green sky* (fur-ruffed parka, glacier front across a black fjord, aurora curtains rippling, reflections) | 1600 | slow push + rise, tilt toward aurora after ignition |
| 4 | 1640–1679 | karst | *one spark in a sea of mist* (layers of limestone towers dissolving into moonlit mist; beacon on the nearest summit, its glow blooming in the mist) | 1650 | lateral drift — the strongest parallax shot |
| 5 | 1680–1719 | city | *this is our world* (rooftop, water tank, antennas, window grids; a kid in a hoodie lights a drum-fire; rooftops across the city answer) | 1690 | slow push, slight pan toward the answering fires |
| 6 | 1720–1767 | sea | *a boat in the moon-path raises fire, and the whole coast answers* (sailor in oilskins, flare at the bow, chain of beacons igniting along the far shore) | 1730 | gentle swell bob + slow push |

Text windows (edit): 1530–1600 and 1620–1700 → keep y≈560–700 calm in shots 2, 3, 4, 5:
fires and their reflections sit above y≈540 or far to the side; lower thirds are smooth sand /
snow / mist / roof deck.

## Technique (one shared toolkit `mt/`, six presets)
* **Camera**: pinhole, yaw + lens-shift tilt (verticals stay vertical, no keystone), real focal
  lengths, eased keyframes, ignition "kick" (damped shake/push).
* **Terrain**: numba *column-coherent heightfield ray marcher* (each screen column is a vertical
  plane → hit distance is monotonic up the column → ~1 march per column instead of per pixel).
  Rendered at 1.5× and area-downsampled (AA). Normals by finite differences with a pixel-footprint
  epsilon (auto-LOD), soft moon shadows by short shadow rays, fire point lights with falloff,
  exponential distance + height fog, analytic fire in-scatter glow in the fog.
  Height functions per world: ridged-multifractal alps + valley cloud; asymmetric sharp-crested
  dunes; glacier front + fjord + floes; Voronoi limestone towers with fluting + mist sea; swells.
* **City**: per-pixel 2D-grid DDA through lots → building boxes with filtered window grids,
  analytic rooftop props (tank, parapet, antennas) lit by the drum fire.
* **Water**: plane / swell heightfield, animated procedural normals with distance-dependent
  roughness (moon glitter → smooth moon path), Fresnel, sky/aurora reflection, fire reflections.
* **Sky**: palette gradient, moon with halo, splatted star catalogue (no shimmer under camera
  moves), Milky Way band with dust lanes, ray-marched aurora slab (curtains with vertical rays,
  green → violet).
* **Fire**: 2D-projected HDR flame (domain-warped rising noise in a teardrop envelope,
  look.blackbody ramp, cores 5–30× ambient); ignition envelope (whoosh + overshoot + light flash);
  sparks = deterministic 3D particles integrated from spawn time (any frame renders independently)
  drawn as motion-blurred energy-conserving streaks; smoke = billowing noise puffs lit from below.
* **Figures**: anti-aliased 2D SDF puppets (tapered capsules/ellipses/trapezoids, smooth union),
  keyframed joints with easing, verlet cloth tails in wind, pseudo-3D normals from the SDF for
  warm rim/fill from their fire and faint cool moon rim. Silhouette colour #05060B.
* **Finish**: look.finish() per shot (exposure/bloom/streak tuned), save_png. No grain/letterbox.

## Process
1. Toolkit + low-res stills of shot 1 → critique → iterate.
2. Rough versions of all six → full v1 render (so the edit has something).
3. Iterate each shot (stills at key frames + consecutive-frame contact sheets for motion).
4. Final render (2 procs × 1 numba thread), preview.mp4, contact.png, NOTES.md.
