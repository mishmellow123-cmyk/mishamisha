# EMBERS: notes (global frames 300–1199)

The legend told in the fire. Everything is a particle in true 3D space, seen
through a moving pinhole camera with thin-lens depth of field and 180° shutter
motion blur. It is splatted into linear HDR and finished with `look.finish()`
(bloom, plus a touch of anamorphic streak at the point and the flash). There is no
grain, letterbox or text.

## Re-rendering

```
cd shots/embers
./render_all.sh                          # full res, two single-threaded workers (300-759, 760-1199)
python3 render.py 300-1199               # or any range / list: "480,520,640" or "640-960:20"
python3 render.py 960-1039 --scale 0.5 --out ../../renders/embers/tests/x   # half-res test
```

* Frames are **independent**: every element is a pure function of time, so any
  frame renders on its own, in any order and in any process. There is no simulation state.
* The only cache is `renders/embers/cache/glyphs.npz` (the glyph atlas). It is rebuilt
  automatically if missing, in about 3 s.
* Cost per full-res frame on one thread: about 3 s for glyphs, fire, towers and race, about 1 s for the
  globe, about 7.5 s for the grasp (hand plus occlusion layer) and about 1 s for the silence. The whole
  900 frames take about 25 min on two workers. Peak RAM is under 0.8 GB per worker.

## Files

| file | what |
|---|---|
| `core.py` | numba point-splat renderer: projection at shutter open and close (camera motion blurs too), CoC `A·f·|1/z−1/zf|`, exact energy-normalised soft discs, sub-splat motion streaks, 1/2, 1/4 and 1/8 resolution buffers for big footprints, per-thread buffers, inverse-square term with a per-element reference distance, text-band attenuation; gradient noise; easing; camera. |
| `glyphs.py` | glyph atlas: 560 items in about 36 scripts and disciplines (Latin, Greek, Cyrillic, Arabic, Hebrew, Devanagari, CJK, kana, Hangul, Ge'ez, Tamil, math, music, DNA, code, cuneiform, hieroglyphs, runes, Phoenician, Linear B, Georgian, Armenian, Thai and more). They are rendered with PIL and libraqm (Arabic joins, Devanagari conjuncts), then point-sampled. |
| `scene_a.py` | 300–480: torch embers slowing into letters, the glyph cloud with 8 choreographed hero passes, the spiral "galaxy of writing", the point. |
| `towers.py` | 8 procedural towers (needle spire, stepped ziggurat, lattice mast, ringed cylinder, twisted prism, pagoda with flared eaves, pod tower, curved blade). |
| `scene_b.py` | 480–880 and the backdrop for 960–1040. It holds the thinking fire (toroidal flow in a teardrop, gold tongues, spark column, white core, 18-root neural filament tree with travelling pulses), the ignition shockwave, towers with fire lighting and beat surges, sparks, radial red walls, smoke, dust, crown ring with tines, and the vortex (the fire's filaments stretched across the storm). |
| `scene_c.py` | 880–960 ember globe (Natural Earth 110m land, coastlines, Voronoi plate cracks, fire spill, plates parting, molten seams). 960–1040 hand rig: palm with thenar/hypothenar pads, knuckle ridge and tendons, 4 fingers × 3 tapered phalanges with spherical joints, a 3-segment opposing thumb and a forearm, posed by FK. 1040+ the last ember. |
| `timeline.py` | the element schedule, cameras, render options and post (torch glow, hand occlusion composite, flash). |
| `render.py` / `render_all.sh` | driver. |

## Beats as built

* 300–340: embers stream up from the torch glow at the bottom (continuity with the
  INTRO push-in), then slow on a time warp. From about 320 each ember opens into a letter, with a rack
  focus to the ember column.
* 340–395: a deep field of about 5,000 glyphs drifting in like fireflies. Eight hero glyphs pass
  the lens in focus: 火, 𝄞, كلمة, π, ज्ञान, A, ACGT, ∞. Readable glyphs always face the camera
  within about 34°, with no back faces, and the mirror-looking Cyrillic letters were removed.
* 395–470: the cloud flattens into a three-armed logarithmic spiral that winds in, accelerates,
  shrinks and heats to white. 470–480: a single blinding point.
* 480: ignition flash and streaking shockwave. The thinking fire is an ice-blue/white core with
  visible pulses racing along the filaments, gold tongues burning upward and a spark column. It breathes once per bar.
* 520–640: the towers rise out of the dark as the camera pulls back between two of them.
  The fire lifts and opens into a floating ring-crown (ring radius 5, 9 flame tines), tilted
  toward the lens. The camera looks at the crown through a gap, so the opposite gap sits behind it.
* 640–800: every beat (640 + 20k) the towers jolt upward (easeOutBack) with leap-frogging leaders.
  Sparks fly from the tops and bases, the brightness pulses and the camera kicks. Radial red ember walls rise
  between the kingdoms from about 660. The palette is fully crimson by about 780, and the crown ring stays cold white-blue.
* 800–880: the crown balloons into a vortex, with the camera dropping below it to look up into the eye.
* 880–960: a hard cut on the beat to the ember globe. Cracks of fire spread from five scattered origins
  (deliberately not pointing at any one region), the plates part, and it flares out at 950–959.
* 960–1040: a hard cut on the downbeat. The colossal ember hand rises from below the storm's eye,
  back of the hand to camera and backlit, with cold rim light from the crown and sparks shedding. The fingers spread to reach,
  close slowly from 1015, then decisively from 1025, shut at 1036. A white-red flash ramps over 1036–1039.
* 1040: black. One ember drifts down in the upper middle, dims, gives a last flicker and dies at about 1150,
  leaving a faint wisp. Pure black to 1199. The lower third is never touched.

## Deviations from the bible, and why

* **8 towers** (within the 6–8 range). With an odd count, the tower opposite any gap sits
  directly behind the crown from the camera, so an even count keeps the crown framed clean.
* **Walls are radial** (spokes between neighbouring towers) so they *divide* the kingdoms into
  sectors instead of forming a palisade. They are offset slightly so none is seen exactly edge-on.
* **Cuts** at 880 (to the globe glimpse) and 960 (back for the grasp), both on beats. The shutter never
  straddles a cut.
* **Text band:** besides composition, particles in y≈560–700 are softly attenuated (up to 62%)
  during each text window (T3–T8), a screen-space cheat.
* **Hand occlusion:** additive embers can't hide what's behind them, so the hand is rendered into its own
  layer and its point coverage builds an alpha that occludes the world behind it. This is what makes it read
  as a solid backlit silhouette.
* **Inverse-square falloff** on point energy (reference distance per element): distant particles
  dim physically, while extended objects keep their surface brightness.
