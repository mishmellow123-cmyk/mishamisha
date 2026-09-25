# MONTAGE (BEACONS) — notes

Global frames **1440–1767**, 1920×804, `renders/montage/f_%05d.png` (1760–1767 is the tail handle).
Hard cuts at 1520 / 1580 / 1640 / 1680 / 1720. Ignitions land on the hits: **1480, 1540, 1600,
1650, 1690, 1730** (the flame "catches" one frame before and whooshes up with overshoot on the hit
frame; light, sparks and a small camera kick all start on the hit).

| shot | frames | world | idea | technique |
|------|--------|-------|------|-----------|
| s1 `s1_peak.py` | 1440–1519 | far peak | a shepherd, back to us, sees the first light flare on the horizon (≈1447), turns, thrusts his torch into the cairn | column-coherent heightfield ray-march (summit, ridged ranges, moonlit cloud sea, Earth curvature, soft moon shadows, peaks shadowing the clouds) |
| s2 `s2_desert.py` | 1520–1579 | desert | a robe on a knife-edge crest under the Milky Way | heightfield: an art-directed hero crest (PCHIP control points designed in image space) over a transverse dune sea; raking moon; Milky Way + star catalogue |
| s3 `s3_ice.py` | 1580–1639 | ice | warm fire under a green sky | heightfield shore/promontory/glacier/mountains + fjord water; aurora ray-marched at half res, mirrored into the water via each pixel's reflection vector |
| s4 `s4_karst.py` | 1640–1679 | karst | one spark in a sea of mist | true 2.5D: 8 silhouette layers of limestone towers at real depths (parallax from the lateral drift), aerial haze, mist sea and drifting mist sheets; fire glow blooms in the mist |
| s5 `s5_city.py` | 1680–1719 | city | this is our world | 2.5D skyline (6 layers, supersampled window grids, rooftop tanks/antennas, street-glow haze), ray-cast roof deck + parapet lit by the drum fire, SDF water tank/mast; rooftop fires answer across the city 1694–1716 |
| s6 `s6_sea.py` | 1720–1767 | sea | a boat in the moon-path raises fire; the coast answers | numba water: 9-wave spectrum with footprint-filtered normals (glitter → smooth moon path with distance), Fresnel, fire reflections as specular streaks; coast beacons ignite left→right 1735–1766 |

Shared toolkit `mt/`: `noise` (numba gradient/value noise, fbm, ridged), `cam` (pinhole, yaw +
lens-shift tilt so verticals stay vertical, easing/keyframes/kick), `terrain` (column-coherent
marcher, soft shadows, height fog, point in-scatter), `sky` (palette gradient, moon, Milky Way,
splatted star catalogue, aurora), `fire` (tongue-based HDR bonfire, ignition envelope, sparks +
curl-noise embers with hot head / tapered tail, billowing smoke lit from below, heat shimmer,
glows), `figure` (grouped smooth-union SDF puppets with pseudo-3D normals, rim/halo light from fires,
verlet cloth), `props` (dry-stone cairn with glowing chinks + iron fire-basket, oil drum),
`people` (shepherd, robed figure, parka / hoodie / sailor torch-bearers).

## Re-render
```
cd shots/montage
python3 render.py all --scale 1.0 --procs 2                  # everything, 2 workers x 1 thread
python3 render.py s4 --range 1640-1679 --scale 1.0 --threads 2   # one shot
python3 render.py s1 --frames 1440,1480 --scale 0.5 --out tests/s1 --sheet   # test stills
```
Every frame is a pure function of its frame number (particles/cloth are integrated from their
birth each frame), so any frame can be re-rendered alone. First run compiles numba kernels
(cached in `__pycache__`).

## Deviations from the bible
* City beacon is an oil drum with a grate (grounded rooftop fire) rather than a stone cairn; the
  sea beacon is a hand-held burning flare at the bow (per the brief). All other beacons are the
  bible's cairn + iron fire-basket.
* Night skies follow the palette; the moon is enlarged (0.55–1.1°) in s4/s6 for composition.
* s4/s5 are 2.5D layer renders (not ray-marched) for speed; parallax is still physically correct
  for the camera drift because every layer sits at its real depth.
