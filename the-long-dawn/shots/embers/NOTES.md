# EMBERS v2 — second pass (framing & tone reviews, 2026-09-26)

Re-rendered (src numbering): embers_v2 316-830, 866-959 · embers_B 331-449, 481-747, 791-959 · embers_C 562-959.
(embers_v2 831-865 and every folder's 960-1039 are unchanged; B/C fall back to v2 wherever they would be identical.)

* **Globe (880-959, all cuts; framing review / tone M5).** v1 framed East Asia and parted along a seam past Japan,
  Taiwan and the Philippines. Now: seen from ~82 N, every continent on the rim together, turning eastward ~27 deg;
  the fire starts in the high Arctic and reaches every continent at about the same time; the plates are laid out by
  `globe_plates.py` (Voronoi seeds optimised against real geography: no seam within 4-8 deg of the first island
  chain, the Taiwan Strait, Korea, Kashmir/the Himalaya, Ukraine/the Baltic, the Levant/Suez/Gulf/Hormuz, the
  Bering Strait; soft margins from capitals/AI hubs; seams kept off land borders and dense population), then merged
  into 13 irregular plates (`GLOBE_GROUPS`; 28 equal cells read as a football), a grouping in which the fracture
  crosses every continent. Viewers see coastlines, not borders, so the test was: no seam along a coast or strait and
  none cutting off a peninsula or island. Timing unchanged (cut in 880, flare-out 950-959).
* **The thinking fire (480-628, all cuts; M2).** No longer a hanging bulb: a compact flame (base -0.8R, tip +2.3R)
  whose white core sits up in the body, a cool dim base, three asymmetric licking tongues (`FLAME_TONGUES`),
  filaments rising into the tongues (none coiled at the base), glow drawn up the flame, sparks from the tongue
  tips. Camera ~15% further back 490-548 and a wider lens (46->52 deg, v1 44) so the tip is always in frame and
  the flame sits above the caption. All weighted by (1 - crown_morph): the crown / Ring and race are unchanged.
* **The crown (A and B, 564-830; m7).** Thirteen irregular licking tongues (uneven spacing, heights 0.9-3.9,
  own flicker, sway, a lick wave running up each, bent tips) over a low fringe of flame, not nine equal triangles.
* **Glyphs (316-480, all cuts; m3).** No whole English words ('Word', 'fire', 'light', 'dream', 'mind', 'We'),
  no GATTACA/ATCG/TTAGGG/CGCG, and 3/4 of the random ACGT/AUG strings gone (157 instances swapped for letters of
  their own script). The v1 field is drawn exactly as before (same stream, same choreography, same hero passes)
  and only those instances change; the v2 atlas (renders/embers_v2/cache/glyphs_v2.npz) is the v1 atlas minus
  the removed items, with identical point sets.
* **Text band (render.py TEXT, per cut).** A: 340-440, 490-565, 580-648, 668-738, 800-866 · C: 340-440, 490-565,
  628-695, 705-770, 780-834 · B: none. (Lines on black 1060-1186 sit mid-frame; silence frames unaffected.)

# EMBERS v2 (BIBLE_V2 §4) — what changed and how to re-render

Outputs (src numbering; the edit falls back embers_<cut> -> embers_v2 -> embers):
* `renders/embers_v2/`  cut A (and the base for C): 520-879 (towers of embers + the vortex fix) and 960-1039 (the
  grasp re-rendered because the solid towers change its backdrop; the hand's code is untouched). Text band ON.
* `renders/embers_B/`   cut B (wordless): every frame whose look depended on the text-band attenuation, re-rendered
  without it: 331-449, 481-644, 651-739, 811-909 (towers/vortex frames in there are v2). The silence (1046-1199)
  is unaffected by the band (the last ember stays above it) and falls back.
* `renders/embers_C/`   cut C (Tolkien): 562-879 and 960-1039 (the Ring, the Eye, the grasp on the Ring). Band ON.

Render: `python render.py <frames> --cut A|B|C [--scale 0.5 --out DIR]` (default outputs above; it refuses to write
renders/embers, the delivered v1). v1 source is kept in `_v1_src/`. Review sheets: `python review_sheets.py`.

## 1. Towers made of embers (all cuts) — towers2.py, scene_b.Towers
* v1 drew points along box edges (CAD wireframes, transparent). v2 builds each of the eight designs as a solid
  ember surface: a glowing-coal CRUST (slow ash/heat patches streaked upward), fire in the JOINTS (importance-
  sampled masonry courses for the needle spire, ziggurat and drum tower; a mullion grid for the twisted prism,
  pagoda, pod tower and blade; plus a few large fissures), BURNING EDGES (tier lips, eaves, ribs), WINDOWS of fire
  (a third lit, most smouldering, a few roaring; unlit ones are dark openings), rims, heat at the base and a
  base-to-top gradient so the tops recede into the dark, SMOKE rolling off the tops (TowerSmoke) and EMBERS shed
  off the edges (TowerEmbers). Every surge (v1 motion, unchanged) sends a heat wave up the tower; palette bleeds
  to crimson as before.
* Occlusion (core.py): the crust feeds an occluder pass (depth / coverage / id pyramid at half res, linear alpha
  from defocused coverage); every splat is depth-tested per pixel against it, a tower's own points with a larger
  same-id bias. The towers are opaque masses: they hide the walls, smoke, crown and storm behind them.
* Level of detail by distance; gaussian splats for the towers (soft defocus, no bokeh "glitter").
* Bodies extend to local y=-26 (towers surge above HMAX from ~840; in v1 their bases lifted off the ground).

## 2. The vortex (all cuts) — scene_b.vortex_tilt / crown_tilt / STORM_LIFT
* The storm's disc turns its underside to the lens as it grows (held at ~38 deg from the first frame of growth),
  so it reads as a maelstrom at every frame (v1: nearly edge-on 805-825, a flat bright smear).
* The crown ring keeps turning its face toward the lens (798-842) instead of flattening through edge-on; its
  tines, tongues and spark column are drawn into the storm (802-830).
* Because the towers are solid now, the storm climbs 16 units clear of their crowns (804-852, t<880 only) and the
  camera tilts up to follow (CAM_B 840/880 targets); the funnel's throat sits on the ring.

## 3. Cut B — `--cut B` sets the text-band attenuation to zero (variant.py).

## 4. Cut C — tolkien.py, inscription.py
* The Ring: the thinking fire is forged into a plain heavy gold band (a white-hot front runs round the circle
  596-622; the metal cools through orange to gold); fine lines of fire burn up out of it (616-642) as an
  inscription outside and in, in an ORIGINAL invented script (inscription.py: crozier stems, flame loops, spirals,
  moon crescents, looped crosses; broad-nib calligraphy; no real text, and deliberately unlike Tengwar, Latin,
  Arabic or any living script). Gold ember-light with a polished highlight; turns slowly; stays gold while the
  world goes crimson. No crown tines in C.
* The Eye (836-879): the Ring faces the lens and burns from gold to a ring of fire (flames lick off it); inside,
  an iris of flame fibres drawn slowly inward to a vertical slit of real darkness (post mask: light behind the
  slit is removed, the storm's glare behind the iris dimmed), rimmed with white heat; the slit narrows a little
  and holds. The storm's white core turns to fire around it.
* The grasp (960-1039): the Ring, small and bright, hangs in the storm's eye (C only: the funnel throat is moved
  onto crown_centre(960), where the fist closes); the hand's own occlusion hides it as the fingers close, its gold
  light leaking round them.
* Grasp backdrop (all cuts): the ring of towers is rotated rigidly by 0.64 rad for the grasp shot (after the hard
  cut, invisible) so a gap between two slender towers lies in front of the storm's eye; towers dimmed x0.6.

## Render log (this Mac, M2, one thread per process, two processes, alongside other departments)
| folder | frames | median / frame | total CPU |
|---|---|---|---|
| embers_v2 | 520-879 (360) | 3.6 s | 26.5 min |
| embers_v2 | 960-1039 (80) | 8.6 s | 12.8 min |
| embers_B | 331-449, 481-644, 651-739, 811-909 (471) | 3.4 s | 40.2 min |
| embers_C | 562-879 (318) | 4.3 s | 25.9 min |
| embers_C | 960-1039 (80) | 7.8 s | 12.1 min |
First frame of a process: +20-100 s (numba compile, tower build ~8 s, hand build). Peak RSS ~0.8 GB.
Checks: B's silence frames equal A's to the dither (max 2/255); local renders match the cloud-rendered v1 at the
seams to noise level (mean |diff| ~2/255, well under frame-to-frame motion).

## Known weaknesses (v2)
* Tower crusts are point-splatted: at full res their ember grain is fine texture, but at a distance (and in the
  LOD tail) they read softer and slightly noisy; the occluder is half-res, so tower silhouettes against the
  bright storm have a soft, faintly stepped edge.
* In the grasp (all cuts) the storm's eye is framed by two tower silhouettes (hard dark verticals at its sides,
  960-1000) — the price of solid towers; the rotation was chosen to keep the eye's centre clear.
* Cut C: the inscription reads as writing while the Ring is near and turned (620-720); later it is fine flecks
  of fire. The Ring slides ~5 units into the hollow of the grip 1004-1018 (hidden by the fingers for most of it).

----------------------------------------------------------------------------------------------------------------

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
  globe, about 8–15 s for the grasp (hand surface projection, crust, cracks, rim and occlusion layers; the first grasp
  frame in a process also builds the hand, about 40–60 s) and about 1 s for the silence. The whole
  900 frames take about 25 min on two workers. Peak RAM is under 0.8 GB per worker.

## Files

| file | what |
|---|---|
| `core.py` | numba point-splat renderer: projection at shutter open and close (camera motion blurs too), CoC `A·f·|1/z−1/zf|`, exact energy-normalised soft discs, sub-splat motion streaks, 1/2, 1/4 and 1/8 resolution buffers for big footprints, per-thread buffers, inverse-square term with a per-element reference distance, text-band attenuation; gradient noise; easing; camera. |
| `glyphs.py` | glyph atlas: 560 items in about 36 scripts and disciplines (Latin, Greek, Cyrillic, Arabic, Hebrew, Devanagari, CJK, kana, Hangul, Ge'ez, Tamil, math, music, DNA, code, cuneiform, hieroglyphs, runes, Phoenician, Linear B, Georgian, Armenian, Thai and more). They are rendered with PIL and libraqm (Arabic joins, Devanagari conjuncts), then point-sampled. |
| `scene_a.py` | 300–480: torch embers slowing into letters, the glyph cloud with 8 choreographed hero passes, the spiral "galaxy of writing", the point. |
| `towers.py` | 8 procedural towers (needle spire, stepped ziggurat, lattice mast, ringed cylinder, twisted prism, pagoda with flared eaves, pod tower, curved blade). |
| `scene_b.py` | 480–880 and the backdrop for 960–1040. It holds the thinking fire (toroidal flow in a teardrop, gold tongues, spark column, white core, 18-root neural filament tree with travelling pulses), the ignition shockwave, towers with fire lighting and beat surges, sparks, radial red walls, smoke, dust, crown ring with tines, and the vortex (the fire's filaments stretched across the storm). |
| `scene_c.py` | 880–960 ember globe (Natural Earth 110m land, coastlines, Voronoi plate cracks, fire spill, plates parting, molten seams). 960–1040 the hand: ONE closed surface, a signed distance field made of 40 tapered bones in smooth union (carpus, 4 metacarpals fanning to the knuckle line, fingers with knuckle bulges, a thumb whose metacarpal melts into the palm as the thenar mass, radius/ulna/muscle belly), posed by FK. Points ride their bones and are Newton-projected onto the union each frame (numba); a softmin ownership weight (partition of unity over bones) removes doubled layers at joints. Shading: camera-facing points only, cosine-weighted (no limb brightening), near-black crust, a static Worley crack network on the rest-pose skin (coal bed, hottest at the knuckles, heat pulses up the arm, surges as it clenches), burning rim + cold crown rim on their own layer, embers shed off the edges. 1040+ the last ember. |
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
  back of the hand to camera, a dark coal-bed mass with fire in its cracks and burning edges, backlit by the eye, embers
  streaming up off its edges. The fingers spread to reach, close slowly from 1015, then decisively from 1025, shut at 1036;
  while it closes the hand turns toward its thumb side so the fingers' curl and the thumb wrapping across them read in
  3/4 profile (a fist, not a mitten). A white-red flash ramps over 1036–1039.
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
  layer and its point coverage (area-weighted, front faces, no fog) builds an alpha that occludes 98.5% of the
  world behind it. This is what makes it read as a solid backlit silhouette.
* **Hand rim light:** rims are splatted into a separate layer and, in post, kept only near the OUTER silhouette
  (blurred coverage < 1), because a backlight cannot reach the edge of a finger lying in front of the palm. This
  keeps a curled finger seen end-on from drawing a lit ring.
* **Inverse-square falloff** on point energy (reference distance per element): distant particles
  dim physically, while extended objects keep their surface brightness.
