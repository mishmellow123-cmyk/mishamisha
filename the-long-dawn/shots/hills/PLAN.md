# HILLS — plan

Three pieces, two worlds. INTRO and CODA share one place (the festival hill,
far future, blue hour → night). FIRST BEACON is the same woman sixty years
earlier on a Himalayan summit. The pieces rhyme on purpose: same screen
direction (figures face LEFT, wind blows from the left, the red scarf streams to
the RIGHT), same scarf (deep red, one long tail, fringed end), same beacon
design (stone cairn + iron basket).

## World / technique

* **Camera**: real pinhole camera in metres (x right, y up, z forward =
  toward the afterglow / toward the range). Every element lives at a real depth,
  so dolly/crane/push moves give true parallax.
* **Sky** (numba, per-pixel direction): twilight gradient (zenith `#070B1C` →
  violet → rose → amber band near the set sun), Milky Way band (dark sky only),
  star catalogue (12k stars, magnitude power law, colour temperature, twinkle,
  horizon extinction), Moon (3× real size, thin waxing crescent low in the west,
  earthshine, *city lights* speckled on the night side), **orbital ring**
  (geometric equatorial ring seen from a high-latitude observer: rises out of
  the western haze and arches south; sunlit segments bright, the part inside
  Earth's shadow cylinder fades out; faint station nodes). No ring / no lit Moon
  in FIRST BEACON (60 years earlier).
* **Hills** (INTRO/CODA): 9 ridge "cards" at real distances (14 m … 30 km),
  each a 1-D fBm heightfield (+ sparse tree silhouettes) intersected per pixel
  → analytic anti-aliased edge; aerial perspective + height fog (valley mist)
  using the sky colour as the fog colour, so layers dissolve into violet haze.
* **Peaks** (FIRST BEACON): true 3-D heightfield ray-marcher (numba) with a
  LOD mip pyramid (no shimmer), precomputed moon shadow map, snow/rock shading,
  distance haze, and a moonlit cloud sea filling the valleys. A fine summit
  heightfield (8 cm cells) is nested inside the 80 km range.
* **Figures**: 2-D SDF puppets on cards (round cones, ellipses, polygons with
  smooth union), rasterised per primitive bbox. Keyframed joint angles with
  easing + procedural breathing/idle. Lit in 2.5-D from the SDF normal:
  warm rim/fill from the fire on the side facing it, cool sky rim from above;
  the red scarf and skin have their own albedo; everything else reads black.
  Scarf tail, hair, pom-pom sway: verlet chains driven by gusting wind with
  travelling-wave flutter.
* **Fire**: procedural HDR flame (domain-warped 3-D noise in a teardrop
  envelope, blackbody colour, leaning with wind), coal bed, smoke lit from
  beneath, sparks/embers as motion-blurred splats (streaks), DOF bokeh for
  near embers, light cast onto figures/ground/snow with flicker. Distant
  beacons = small HDR flames + haze halo + glow on their hill.
* Finish: `look.finish()` (bloom + vignette + ACES). No grain/letterbox/text.

## INTRO 0–359 (deliver 0–359; edit crossfades to EMBERS 300–340)

**Shot A — WIDE, 0–159** (cut at 160 = bar 3 downbeat).
Low camera (≈0.6 m) on the hilltop, 50° lens, looking west over a valley at
layered hills; horizon ≈ 60 % down the frame. Sky: amber band low left-centre,
rose → violet → deep blue; first stars; the ring rises from the right-hand haze
and arches up-left out of frame; thin crescent Moon upper-left with speckled
night side. Figures on the crest, right third: Child (left, pom-pom hat),
Elder (right, stooped, coat, torch raised forward, scarf streaming right), the
unlit cairn a metre left of the Child. Heads against the luminous lower sky.
Grass along the crest bending in the wind, catching the torch rim-light.
Camera: slow truck left + slight push (≈1.2 m) → parallax.
Distant beacons ignite one by one: 22, 80 (flute entry), 112, 138 (+ more in
shot B). Child points at the 80 fire, then (≈110) turns her face up to the Elder
→ T1 120–200. Lower third = dark ground (calm).

**Shot B — TWO-SHOT + PUSH, 160–359**.
Camera ≈3 m away, hip height, 40° lens, slight low angle: both heads and the
torch against the sky; the torch flame between their faces lights both (the
Child's upturned face lit from above). 160–215: Elder looks down at the Child,
then (≈215, T2) lifts her gaze to the far fires. Beacons keep igniting far
behind (196, 238). From ≈230 the camera pushes toward the torch, accelerating
(ease-in), focus racks to the flame, background melts into bokeh; ember rate
rises; embers stream past the lens. 290–359: frame filling with rising embers
over the huge soft flame at the bottom edge, dark above (matches EMBERS' start:
rising embers on darkness).

## FIRST BEACON 1200–1439 (one continuous take)

1200–1235 black (almost: a hint of wind-blown snow). Camera: medium close-up,
profile of the Young Woman kneeling at the cairn, facing left; basket rim +
kindling lower-left, her hands centre, face upper-right, hair and scarf
streaming right. Slow handheld drift.
* 1236 strike 1 (small shower: only the hands + steel lit),
* 1262 strike 2 (bigger: hands + face profile),
* 1290 strike 3 (biggest: hands, face, the red scarf; a spark stays alive in
  the tinder, pulsing as she blows),
* 1318 kindling catches: small flame lights her face from below, breath vapour
  drifts right, scarf glows red, snow grains glint in the light.
* 1360 ROAR: fire leaps out of the basket, column of sparks; she rises and
  steps back; camera pulls back and up (strong ease-out, 1360–1439) to the
  reveal: tiny figure + roaring beacon on a snowy summit, spindrift streaming
  off the ridge, a sea of moonlit peaks and a cloud sea below, a vast starfield
  and Milky Way above (no ring/lit Moon — it's the past). The moonlit world
  fades up from black as it is revealed (eye adaptation). T9 1370–1435 lower
  third = the summit's shadowed flank (calm).

## CODA 2460–2807

**Shot C1 — MEDIUM, 2460–2627.** Same hill, deeper blue, many more stars,
ring and Moon (lights on its dark side) clearer. Like shot B but closer,
slightly lower. Child looks up (T15 2500–2570); the Elder meets her eyes, then
looks away to the distance — says nothing. Gust 2555–2600: the scarf lifts high
(rhymes with the summit). 2580–2620 she lowers the torch into the Child's
raised hands; her hand leaves the handle and settles on the Child's shoulder.
The flame now lights the Child's face.

**Shot C2 — WIDE + CRANE, 2628–2807.** Shot A's set-up, later in the night
(the INTRO beacons still burn). The Child lifts the torch into the basket;
**2640 it catches**, grows to a roar with a spark column by ≈2660.
2645–2720 answering fires ignite in a rolling wave from near to far, left
across the valley and back to the horizon (~20 "voiced" fires at distinct
frames + a shimmer of tiny far ones following the wave front; ignition frames
exported to `renders/hills/coda_fires.json` for the bells). Camera cranes up and
tilts into the sky 2650–2807 (ease in-out): the fire-strewn land sinks to the
bottom edge; stars, the ring arch, the lit Moon, and a few slow ship lights.
Title 2668–2780: centre of frame is dark sky.

## Process

1. core/sky/hills → still of shot A background, 0.5× → iterate.
2. Figure sheet (Elder, Child, Young Woman in key poses, large) → iterate.
3. Fire tests (torch, beacon, sparks, distant).
4. Peaks heightfield tests.
5. Assemble, 0.5× sequences + contact sheets → full-res v1 of everything →
   improve → re-render.
