# THE LONG DAWN v2: picture red-team (fresh eyes)

Independent red-team pass on the picture (cinematography, VFX, design), 26 Sep 2026.
Session gate `cloud/verify_session.py`: **GATE PASS** before the review started.

**What I looked at**
* All 16 contact sheets in `review_sheets/`: `embers_v2`, `embers_C`, `grasp_before_after`,
  `beacon_before_after`, `run`, `ice_before_after`, `sea_before_after`, `globe_v2`, `cloud_globe_answers`,
  `cloud_globe_dawnA`, `cloud_globe_dawnB`, `map_C`, `accord_v2`, `coda_v2`, `title`, `critic_tone_evidence`.
  I viewed each sheet whole, then cropped every thumbnail I cite at 2–4×. Where a sheet is before/after, only
  the right-hand (v2) side is judged.
* `BIBLE_V2.md` (all of it), `BIBLE.md` §4–§6, and the three earlier critiques (picture, tone, framing).
* Department notes and code, read only to aim the fixes: `shots/embers/scene_b.py`, `scene_c.py`, `tolkien.py`,
  `shots/globe/fires.py`, `shots/globe/shots.py`, `shots/map/webmap.py`, `shots/run/NOTES.md`,
  `shots/hills/sky.py`, `shots/hills/cairn2.py`, `edit/ember_title.py`.

**Limits**
* The v2 masters and renders aren't in this repo; only v1's `out/the_long_dawn.mp4` is. This is a **stills
  review**. I can't judge motion, flicker, cadence, sync or sound, and I couldn't re-run the QC scan. Where a note
  depends on motion, I say so.
* Thumbnails are 1/6 to 1/2 of 1920 px wide, and JPEG. Everything below is visible at that size, so it will be
  visible to everyone.
* No sheet covers these shots, so I have **not** reviewed them: the glyph field 300–470, SILENCE 1040–1199, the
  heroine close-ups 1200–1375, FAR PEAK 1440–1519, DESERT 1680–1739, KARST 1800–1839, CITY 1840–1879 and DAWN_C
  2400–2655. Nobody should sign off the picture until each has a sheet.
* Frame numbers are **v2** unless marked *src*. The globe, ICE and SEA sheets use src numbering
  (v2 = src + 160). The coda sheet prints both.

---------------------------------------------------------------------------------------------------------

## Verdict

v2 fixed what it was asked to fix. The bulb, the node-and-arc mesh, the East-Asia globe, the pastel emissaries,
the jointed dolls, the 8-ray starburst, the "dark Africa" hearths and most of the coda glitter are gone. The new
pieces show real ambition: the Ring, the Eye, the ink map's close-ups and the Run's single world. But a
red-team pass finds that several fixes over-corrected, or swapped one stock image for another. It also finds
misreads that the earlier critics didn't flag:

1. **The grasp ends as a raised fist.** A low-angle vertical forearm with a fist on top, in a red flash: the
   salute of protest and revolution. The film's image of one hand seizing the world reads as its opposite.
2. **The world answers with a missile, neurons and a dot map.** The globe traded its arcs for a comet that
   reads as a launch over the Arabian Sea. C's map spreads as forked dendrite trees (a neural net) and ends on a
   dot-density world map (the corporate "global presence" slide).
3. **The first act's three icons read as household objects.** The thinking fire is now a **candle**: orange
   tips over a blue-white base, filaments gone. A/B's crown is a **gas-hob burner ring**. C's Ring is a
   **glitter bangle**.
4. **A and B still climax on the keynote sunrise from orbit.** Its hit frame is a grey veil with stars showing
   through the sun.
5. **The new 3-D showpiece reads as a flight-sim terrain demo.** The Run's beacons are LED dots on dalmatian
   molehills. The story's turn (the FIRST BEACON reveal) still stands on a 2000s heightfield with a mannequin.
6. **The payoff frame is crossed out.** The orbital-ring hairline runs through the beacon flame and, at 2880,
   straight through "DAWN" in the title. The cairn reads as a brick chimney, and the title is a particle logo
   reveal.

None of this needs a new department. Most of it is parameters or staging in code that already exists.

---------------------------------------------------------------------------------------------------------

## Per-shot notes (film order)

### 1. INTRO: the hill at dusk (0–339; all cuts). Sheet: `coda_v2` rows src 60, 180, 276
**Works.**
* The v2 figures are true merged silhouettes. The joint outlines are gone, and the child's pom-pom and the
  elder's stoop read instantly.
* The scarf is now cloth, with a fringe and flutter.
* The future is signalled and never explained: violet layered hills, an amber band, a crescent Moon with its tiny
  city lights.
* The push into the torch (276) is clean.

**Fails.**
* **A vector-wallpaper register.** Flat black cut-outs on a smooth purple gradient, a clip-art crescent and a
  hairline across the sky make the stock "silhouette at dusk" phone wallpaper. This is the one register in the
  film that looks templated rather than lit.
* **Nothing is lit by the torch except its own flame.** The scarf is the same flat red on the side next to the
  flame and on the far side (180, and again at 2770). The elder's sleeve and face get little or no warm edge, and
  the grass gets no pool of light.
* **The orbital ring is a straight hairline.** It is a 0.75-px line (`sky.py draw_ring`, `width=0.75`) that
  reads as a contrail, a satellite streak or a scratch on the print. It also cuts through the most important
  parts of the frame (see 15 and 16).

**Fix** (HILLS `sky.py`, `intro.py`, `coda.py`).
* Light the silhouettes from the torch, following the bible's own silhouette rule:
  * a 1–2 px warm rim on the torch side (sleeve, cheek, scarf) and a cool sky rim on the other;
  * the scarf graded by its distance to the flame: bright scarlet near the torch, near-black crimson at the tail;
  * a small warm pool on the grass under the torch.
* Draw the ring as a 2–3 px soft band with a lower peak, and drop its gain wherever it would cross the figures,
  the fire or the type.

### 2. KINDLING: the thinking fire (480–600; all cuts). Sheet: `embers_v2` 500/520/546/570
**Works.** The bulb is gone. The tip is in frame, the tongues are asymmetric, and the ignition reads as a birth.

**Fails.** It has over-corrected into an **ordinary flame**, and in fact into a candle: blue-white at the base,
orange tips, one small flame alone in the dark.
* The tongues are coloured with `look.blackbody()` (`scene_b.py:404`), the same ramp as every beacon in the film.
* v2 weights dim the filaments except where a pulse passes (`scene_b.py:420–421`), so at sheet size there is no
  visible *thought* left in it.
* After the pull-back (`CAM_B` at 500: radius 17.4, wider lens) it is only about 1/5 of the frame height.

The bible says the thinking fire must be "unmistakably *not* ordinary fire". As it stands, the colour arc
(mind white/ice, then race red, then accord gold) starts from the wrong place. A's "And it began to think." now
sits under a candle in the dark, the oldest poster image of all.

**Fix** (EMBERS `scene_b.py`, all variants, src 480–600).
* **Colour.** Tongues in the mind palette, not blackbody: core → `C_ICE` along the tongue, with `C_GOLD` only on
  the outer ~15% edge. No orange until `redness()` starts.
* **Filaments.** Restore their base visibility inside the body: drop the `0.3*fw` cut and keep the "no coil at
  the base" mask. We should see slow pulses travelling through a calm body.
* **Motion** (can't be checked on a sheet, so this is a spec). It should move *wrongly* for a flame: tongue `om`
  about ×0.5, and a regular breathing rather than a flicker. It should be the one fire in the film that doesn't
  behave like fire.
* **Size.** Keep the tip in frame, but bring the fire up to 35–40% of frame height. Move the camera back in and
  tilt up rather than pulling back.

### 3. The TOWERS and the CROWN (520–800; the crown is A/B only). Sheet: `embers_v2` 546–760
**Works.**
* The towers now read as masses made of embers, with courses, seams and windows, and the scale is clear.
* The distinct silhouettes (obelisk, pagoda, stepped tower, office block) make "the companies and the countries"
  legible without naming anyone.

**Fails.**
* **The crown reads as a gas-hob burner.** It is a continuous ice-white torus (the thinking fire's ring) with
  short orange licks on its upper edge. It's seen nearly edge-on as a perfect ellipse (640, 700, 760), which is
  a burner ring or a halo, not a crown. The v2 "licking tongues" (`Crown`, `scene_b.py:521–531`) made the tines
  uneven, but they stay short (`tH` 0.9–3.1 on a ring of radius 5–5.6) and blackbody orange (`scene_b.py:573`),
  so the ring still dominates.
* **Mud.** v2 added a brown haze and large, soft orange bokeh blobs (640–760). The true black that the whole
  embers act is built on is lifted to brown, and the frame reads as if through a dirty lens. At 570, a brown
  smear crosses the top third.
* **Signifiers.** The pagoda and the obelisk are the strongest national signifiers in the film. Balanced around
  the ring they're fine, but never let the camera, the light or the Eye single one out (see 4).

**Fix.**
* **Crown.**
  * Make the tines 2–3× taller (`tH` 3–7, tallest at the front), in white-gold from the mind palette.
  * Fade the ice torus to a thin circlet (≤30% of its current energy) as the tines rise, so the *flames* are the
    crown.
  * Tilt it toward the lens (`crown_tilt`) so we see it from 25–35° above, not edge-on.
* **Grade.** Pull back the tower-smoke and haze density so the gaps between the towers go truly black again, and
  cap the bokeh blobs' size and count.

### 4. RACE and VORTEX (640–880; A/B), and the EYE (832–880; C). Sheets: `embers_v2`, `embers_C`
**Works.**
* The vortex now opens toward the lens, which fixes v1's edge-on smear. The beat surges and the white-hot centre
  are powerful.
* C's Eye is instantly the Eye: it resolves out of the storm at 840–852 and holds.

**Fails.**
* **A/B, 850–870: a stock vortex.** A symmetric spiral with a perfect white ring at its centre is the stock
  galaxy, hurricane or portal image. Nothing in it is being *consumed*: the towers below stand untouched.
* **C, 852–879: the Eye's iris is a sticker.** It's a smooth orange disc with a clean black lozenge for a pupil:
  a cat's-eye marble laid over a storm of particles. It's the one thing in the embers act not made of embers.
* **C: the Eye sits on one tower.** The stepped tower's apex points straight at the Eye, in the Barad-dûr
  position, which makes that one kingdom the Enemy's.

**Fix.**
* **Vortex (all cuts).**
  * Feed it from the towers: strip embers off the nearest tower tops and spiral them in.
  * Break the symmetry: an off-centre eye, with one arm heavier than the others.
  * The central ring should flicker and tear, not glow like a ring light.
* **Eye (C, `tolkien.py` EYE).**
  * Draw the iris from the vortex's own particles streaming inward to the slit. The notes promise "fibres drawn
    slowly inward", but at sheet size it reads smooth, so raise the fibres' contrast.
  * Give the slit mask ragged, shimmering edges, and make the rim ragged fire rather than a clean white ring.
  * Centre the Eye over the ring of towers, with every tower leaning toward it, not over one.

### 5. The EMBER GLOBE (880–960; all cuts). Sheet: `embers_v2`
**Works.** The polar view removes the "first island chain" reading. The fire starts in the empty Arctic and
reaches every continent together.

**Fails.**
* **A football.** At 916–946 the plates are still nearly equal cells bounded by uniform glowing seams, and at a
  glance they read as a football or a turtle shell. The code comment at `scene_c.py:104` names this problem; the
  merge into 1–4-cell plates hasn't beaten it at this size.
* **The Earth is slow to read.** At 890, seen from straight above the pole, the continents take several frames
  to read as Earth.

**Fix.**
* Use fewer, longer primary fractures that cross continents, with branching secondaries and a few tiny plates.
* Make the plates visibly *part*, with depth, parallax and molten light from inside, instead of lines drawn on a
  sphere.
* Tilt the pole 25–30° off the lens axis, so that a recognisable coastline sits near the centre while the seams
  still run between everyone.

### 6. The GRASP (960–1040; all cuts). Sheets: `grasp_before_after` (right), `embers_C`
**Works.** The cracked-crust hand is a big upgrade on v1's sausage fingers. The anatomy and the close are
convincing, and the flash at 1035 lands.

**Fails.**
* **It ends as a raised fist.** In `scene_c.py`:
  * the forearm rises vertically from the bottom of frame (`hand_transform`: `W0 = Wf + (0, -85, 0)`), seen from
    a low camera;
  * at the close, the thumb side turns to the lens (`hand_yaw` → −0.74 rad), with the vortex rings as a halo
    behind (1024–1034);
  * then the frame floods red (1035).

  That is the raised-fist salute of protest posters and revolutionary banners, in the colour of revolution. The
  film's image for "one hand seizes the world" reads as "power to the people". It's worst in B, where no line of
  text redirects it.
* **A procedural crust.** The cracks form a uniform Voronoi net (1000, 1016), with cells the same size on palm,
  fingers and wrist: the stock generative-art texture. The cells are also see-through, so a "colossal" hand is a
  glowing line drawing with no mass (990–1016).
* **A/B: no visible target.** At 1000–1016 the crown doesn't read in frame, only a soft gold haze at top right.
  We don't see what the hand wants until it has it.

**Fix** (EMBERS `scene_c.py`).
* **Restage the close as a seizure.**
  * The hand descends on a diagonal from upper frame-left or right (`W0` above and to the side, not 85 units
    below) and closes palm-down over the prize, like a claw with its knuckles to the lens.
  * Over 1030–1036 it pulls the prize *down and toward itself*: hoarding, not raising.
  * Keep the camera level with the hand or above it, never below.
  * No frame may show a vertical forearm with a fist on top.
* **Crust.**
  * Cracks follow the anatomy: long creases along the palm lines and knuckles, small cells at the joints, large
    plates on the back of the hand.
  * The crust between them is dark and opaque, so it hides the storm, with heat glowing *through* where it is
    thin.
* **A/B.** Keep the crown bright and sharp in the upper third over 990–1015, so the reach has a target.

### 7. FIRST BEACON: the reveal (1376–1439; all cuts). Sheet: `beacon_before_after` (right)
**Works.** The fire at 1375 is the best big flame in the film, and the pull-back move is right.

**Fails.** The story's turn is its least convincing 3-D shot.
* **No altitude.** The heightfield range is a *wall* of vertically streaked spikes at the heroine's own height
  (1390–1439). Nothing says she is on top of anything, there's no aerial perspective, and a lumpy snow ridge
  fills the foreground.
* **The mannequin.** She is still the placeholder, with a fist raised by her face (1375) and a rigid pointing arm
  (1439).
* **The wrong scarf.** Hers is a dark crimson tube, not the elder's bright, fringed scarlet.
* **Two different ranges.** The story says it is one range, but it doesn't match the Run's world 80 frames later.

**Fix.** The Run has already built this world and knows her summit: `shots/run/NOTES.md` gives `S1.summit()` ≈
(−5450, 286, 31997), and the route "import `world.py` with an empty crag table and a camera at her summit".
* Render the reveal's plate there: the cloud sea ~900 m below, the far ranges in haze, and the shepherd's peak on
  the horizon at bearing +170°.
* Composite the rebuilt heroine, cairn and fire from HILLS on top.
* Her pose: she stands and watches. No fist, no pointing.
* Her scarf: the elder's `scarf_red`, the same length and fringe, streaming screen-right as it does on the hill.
  It must read in this wide.

### 8. THE BEACON RUN (1520–1679; all cuts). Sheet: `run`
**Works.**
* The idea is the right 3-D shot for this film: one continuous flight chasing the signal from the shepherd's peak
  to the horizon.
* The pyre at 1584 is a real event.
* The vector motion blur and the banked turn (star streaks at 1576) are technically sophisticated.

**Fails.** On the sheet it reads as a flight-sim terrain demo, not the beacons of Gondor.
* **Snow.** It gives "dalmatian" blotches (1520, 1592–1608) because noise, not slope, decides where the snow
  lies. In the snow rule (`shots/run/NOTES.md`), the noise terms `0.18·noise(380 m) + 0.06·noise(95 m)` together
  swing about as far as the whole `smoothstep(0.38, 0.58)` window.
* **Artefact.** The near peak at 1560–1580 has vertical curtain streaks down its face (a stretched heightfield
  or normal artefact), and it fills the centre of frame for about a second.
* **Scale reads small.** Rounded islands in a flat grey plane look like rock piles on a floor. The cloud sea has
  no relief, no moonlit tops and no holes where the peaks pierce it, and the far ranges are as contrasty as the
  near ones.
* **The beacons are fairy lights.** From 1600 on they are identical round glowing dots, evenly placed on summit
  tips, with no flame, no smoke column and no warm pool on the snow (1640–1679). They are the "warm hot point
  plus a soft orange aura" of `fire2.py`'s distant fires. The result is a string of LEDs, and next to the globe
  and the map it's one more node graph.
* **Energy dips.** There is no fire in frame at 1528–1576, and the beat ignitions are single dots.

**Fix** (RUN `render_run.py`, `beacons.py`, `fire2.py`, `world.py`).
* **Beacons.**
  * Every beat beacon (1540, 1560, 1580, 1600, 1620, 1640, 1660) gets a flame shape at least 25 px tall at 1920,
    a 2–3 frame flash, a smoke column lit from below, and a visible warm pool on the snow (the pyre's point-light
    idea, scaled down).
  * The far fillers vary about 3:1 in size and brightness, flicker, and sit at irregular spacing: no two alike.
* **Snow.**
  * Cut the noise terms to about a third (0.18 → 0.06, 0.06 → 0.02), and add a curvature term so ridgelines and
    ledges hold snow while steep faces and gullies go bare. The mountains then read by their form, not a pattern.
  * Fix the 1560–1580 streaks: derive normals from the smoothed heightfield, and clamp texture stretch on slopes
    steeper than 60°.
* **Scale.**
  * Keep the thrilling close pass on the pyre (1580–1592), but before and after it fly higher and slower. The
    notes' 24 m terrain clearance makes the massifs look like hills rushing past.
  * Use a longer lens for 1600–1679 (about 25–30° horizontal field of view) to stack the ranges.
  * Add fog layers between the ranges.
  * Give the cloud sea relief: `h_cloud`'s 150 m fbm is too flat at this camera height. Add moonlit tops and
    wisps around the peaks.
* **Continuity.** Keep at least one fire in frame at all times, including 1528–1576.

### 9. ICE (1740–1799; src 1580–1639). Sheet: `ice_before_after` (right)
**Works.** The new aurora curtains are elegant, the fjord mirror is lovely, and the ignition lights the snow
believably (1760).

**Fails.**
* **The aurora wins.** It's the most saturated thing in frame and out-draws the fire (the earlier picture minor
  7 still stands).
* **A sky beam.** One bright ray right of centre (src 1585–1635, at x ≈ 1260 of 1920) runs straight down *below
  its curtain's hem, to the ridge*. That's a searchlight or sky beam, not an aurora ray.
* **A pudding.** The snow mound is a perfectly smooth dome.
* **A triumph pose.** The torch is held aloft after the lighting (v2 1772 and 1795).

**Fix** (`shots/montage/s3_ice.py`).
* Aurora: saturation −35%, gain −25%.
* Clip every ray at its curtain's lower border.
* Wind-scour the dome, with sastrugi.
* After the ignition the figure lowers the torch and steps back.

### 10. SEA (1880–1919; src 1720–1759). Sheet: `sea_before_after` (right)
**Works.** This is the best-composed shot of the montage. The flare's warm reflection beside the moon's cold
glitter is exactly "fire as the only warm light". Keep it.

**Fails.** The coast beacons (v2 1905–1919) are evenly spaced, identical dots with no reflections in the water:
the string of pearls again.

**Fix** (`s6_sea.py`).
* Irregular spacing and brightness.
* A thin vertical glint in the water under each one.
* A visible delay between ignitions: they answer, they don't switch on.

### 11. THE WORLD ANSWERS: the globe (A/B, 1920–2087; src 1760–1927). Sheets: `globe_v2`, `cloud_globe_answers`
**Works.**
* The node-and-arc mesh is gone. Fire runs along the real ranges: Himalaya, Karakoram, Tien Shan, Zagros.
* The first beacon's flare (1920–1926) is a clear start.
* The city lights are dimmer.

**Fails.**
* **A missile.** At src 1860 (v2 2020) an orange comet with a long tail flies off the subcontinent and out over
  the Arabian Sea (`fires.py`: `F_LEAP`, "a short comet", `ltail` 170). Seen from orbit, a glowing streak
  arcing over an ocean is a launch. In a film whose race "nearly consumed the world" it's the one image that
  reads as war, and it lands on the hope beat. Because it flies off the subcontinent's coast, it also points at a
  country.
* **Dotted route lines.** From src 1880 (v2 2040) the chains are evenly beaded dotted lines along the ranges:
  the travel-map line of adventure films. By the dissolve (v2 2072–2087) there's also an even sprinkle of orange
  points over everything.
* **Fire and cities merge.** The orange fires and the warm-white city lights are both round glowing dots, and at
  phone size they blend.

**Fix** (GLOBE `fires.py`, `places.py`).
* **No free flight over water.** Drop the `F_LEAP` comets entirely. An ocean crossing is a *pause* followed by a
  fire on the far coast (the SEA shot has already shown boats carrying it), or a hop along island chains.
* **Break the beading.**
  * Space the fires from actual summits and passes, with gaps.
  * Vary size and brightness about 3:1.
  * Each chain's newest fire flares hot while the older ones settle to embers, so the chain carries time instead
    of a finished diagram.
* **Cities.** Dim the city lights by another stop and cool them toward white-blue, so warm means fire only.

### 12. THE WORLD ANSWERS: the map (C, 1920–2087). Sheet: `map_C`
**Works.** 1920–1948 is the most Tolkien image in cut C: hatched mountain glyphs, rivers, a double-lined coast,
paper folds, and one small flame standing on the Himalaya. Keep all of it.

**Fails.**
* **Neurons.** From 1960 to 2008 each lit beacon throws 2–3 glowing threads that end in forked tips with bright
  dots. `webmap.py:3–4` specifies "2-3 children per beacon with angular diversity". At 1972 and 1984 it's a
  dendrite tree, the stock picture of a neural network, spreading across the world in a film about AI. At 2008,
  columns of forked marks cross Africa.
* **A dot-density world map.** 2020–2087 ends on a whole-world equirectangular map sprinkled evenly with
  thousands of glowing dots: the corporate "global presence" map.
  * At this scale the hatching, the ink texture and the flame glyphs are gone, so C loses its Tolkien texture in
    its biggest wide.
  * The Andes and Tibet become bright blobs; Tibet's is an "@"-shaped ring.
* **Launch arcs.** Sea crossings are glowing arcs over the ocean (1996–2008, lower right): the same
  launch-trajectory problem as the globe.
* The line at 1935–2040 ("…the free peoples answered.", or its replacement) sits over the neuron phase.

**Fix** (MAP `webmap.py`, `answer.py`).
* **The unit of spread is a flame.**
  * One child per hop, along ranges and coasts: a line of fires, not a tree.
  * No forked ends, and no dots at the tips.
  * Each new fire is a flame glyph like the one at 1920, leaving a brown scorch line in the paper behind it.
* **Stay regional.** End on about a third of the world: the Himalaya out to the Horn of Africa, the
  Mediterranean and Asia's Pacific coast.
  * Keep the camera close enough that the hatching and the flame glyphs stay legible.
  * Let the fire run *off the edge of the map* in every direction, instead of flattening the whole globe into
    frame.
* **Sea crossings** are dotted ink ship-lines drawn on the paper (the cartographer's convention), not glowing
  arcs.

### 13. THE ACCORD (2072–2399; all cuts). Sheet: `accord_v2`
**Works.**
* The pastel toys are gone.
* The oaths are crisp, upright on the beat, and legible.
* The Ring lying small in the hearth in C (2154) is a perfect quiet image.
* B's mandala sweep is elegant.

**Fails.**
* **From dead overhead, it's a coin.** A pale disc, a ring of serif caps around a radiant centre, then a toothed
  ring of flames around that (2350–2392): a seal, a medal, the face of a coin. The tone critic's seal/laurel note
  (M10) still holds. The perfect radial symmetry of the whole top-down, with shadows like spokes, turns it into a
  logo.
* **Sepia.** The whole sequence is brown monochrome with lifted blacks; the floor sits at mid-brown. The bible's
  rule is fire as the only warm light, in *blue* darkness. Here a gold hearth sits in a gold room, so the
  climax colour has nothing to be warmer than.
* **White pills.** At 2154 the dozen torches are clipped white capsules evenly spaced around the table: lamps on
  a clock face, not flames.
* **Pawns.** From above, the emissaries are still round hood-tops (2398). They're charcoal now, with orange
  V-shaped torch flames like paper cut-outs.
* **The whiteout** (2392–2398) still bleaches the floor to paper-white, with grey figures on it (earlier picture
  M1).

**Fix** (ACCORD `scene.py`, `shade.py`, `figlook.py`, `fire.py`).
* **Cool the world.** Everything the hearth doesn't reach becomes moonlit blue-black stone: ambient toward
  `#27335E`, not umber, with deep blacks at the frame edge. Then the hearth's gold visibly falls off across the
  table.
* **Break the symmetry.** Set the camera 10–15° off vertical and orbit it slowly, so the table has perspective,
  the figures have volume, and the oath ring reads as carving on a surface rather than as a graphic.
* **Torches.** Real flame shapes on the orange ramp. Nothing clips to neutral white before 2385.
* **Whiteout.** An additive radial burst of light from the hearth, not an exposure push (earlier picture M1).

### 14. DAWN (A/B, 2400–2655; src 2240–2495). Sheets: `cloud_globe_dawnA`, `cloud_globe_dawnB`, `globe_v2`
**Works.** The web, the starburst and the hearths are gone. The sun is a clean disc with a restrained streak,
and the limb glow is well made.

**Fails.**
* **Still the keynote slide.** The sun rising over Earth's limb, seen from orbit, is the most-used closing image
  of tech and telecom brand films, and here it is *the climax* of two cuts.
* **The hit frame is grey.** At 2400 (src 2240) a grey-beige veil covers half the frame, with stars clearly
  visible *inside* the sun's glare. Physically impossible, and the climax lands neither white nor gold.
* **Static, and never gold.** From 2420 to 2655 the sun sits as a white star with a hairline streak, and the
  image barely changes (earlier picture minor 10). The bible's "ACCORD → DAWN: pure warm gold" never happens.

**Fix.**
* **Best: give A and B the mountain dawn** from the Run's world: DAWN_C *without the eagles*. The sun rises on
  the range where the first beacon was lit, with the smoke of the night's beacons still rising from the summits.
  The loop closes on the ground, and all three cuts share one climax. (DAWN_C isn't on any sheet; I haven't seen
  it.)
* **If the orbit stays:**
  * Fade the stars to zero under the glare from 2398 (stars × exp(−k·glare)).
  * Push the glare *halo* toward the `dawn_*` gold, not just its core. `sun_glare_clean`'s tint (1.0, 0.88, 0.70)
    reads grey once it's spread over black space.
  * Let the terminator visibly sweep, and the night-side fires pale one by one, over 2440–2620.

### 15. CODA (2624–2880; all cuts). Sheets: `coda_v2`, `title` 2810–2850
**Works.**
* The dolls are fixed.
* The handover (2720–2770) is tender and clear.
* The catch now happens in the wide at 2800, and the whoomp reads (2804).
* The answering fires are distinct small fires with smoke (2850), not bokeh.

**Fails.**
* **The ring hairline cuts through the payoff.** At 2810 it passes through the top of the beacon flame, just above
  the elder's head, and at 2850 through the smoke column. A straight white line across the protagonist fire, in
  the last story beat, reads as a wire or a scratch.
* **A brick chimney.** The v2 "dry stone" cairn (`cairn2.py`) renders as regular rectangular courses with
  vertical sides (2810–2850): a brick pillar with a garden brazier on top.
* **A flat scarf.** It's still one flat red with no light gradient, even a metre from a roaring fire (2810).
* **Smoke smears.** The beacon's smoke is a red-brown diagonal across the sky (2850, and the right edge at 2880)
  that reads as a smudge on the lens.

**Fix** (HILLS `cairn2.py`, `sky.py`, `coda.py`).
* **Ring.** In the coda wide, drop its gain to near zero, or re-place it (the ring's latitude and radius
  parameters) so it runs high above the fire column and never through the title band (y 250–350).
* **Cairn.**
  * Batter the sides (a taper of about 8–10°).
  * Vary the course heights about 2:1, and mix large base stones with small pinning stones.
  * Give it a rounded shoulder at the top.
  * Ideally, make it a replica of the FIRST BEACON's cairn and basket. That's a wordless rhyme that helps cut B's
    reveal: the child relights *her* beacon.
* **Scarf.** Shade it from the beacon: scarlet on the fire side, near-black on the far side.
* **Smoke.** Dense at the base, lit orange only low down, cooling to blue-grey as it rises, and gone before it
  reaches the upper third.

### 16. TITLE (2828–2967; all cuts). Sheet: `title`
**Works.**
* The Cinzel caps echo the carved oaths.
* The crumble back into embers (2935–2950) is beautiful.
* The timing to the final chord works.

**Fails.**
* **The title is struck through.** At 2880 the orbital-ring line enters over the N of LONG, crosses the G, and
  cuts diagonally through the D and A of DAWN: a strikethrough on the film's name. At 2865 the Moon crowds the T.
* **A logo reveal.** At 2830, embers leave the answering fires on curling S-paths and fly off to assemble the
  letters. It's the particle-to-logo template of every trailer pack, and still "cute sparkle".
  * The rising embers are round discs, not streaks, so they will strobe (the bible asks for streaks, not strobing
    dots).
  * Embers visible a hundred metres above fires that are kilometres away wreck the valley's scale: the fires
    suddenly read as small and close.
* **An awkward crop.** At 2865 the crane leaves the beacon flame and a fragment of red scarf cut off at the
  bottom edge.

**Fix** (`edit/ember_title.py`).
* Let the embers behave as embers: they rise almost vertically from the nearest fires, drift, cool and die.
* Let the title *kindle in place* in empty sky, the way the oaths light in the stone, then crumble as now.
* Draw every moving ember as a motion-blurred streak.
* Keep the ring line and the Moon at least 60 px clear of the title's bounding box.
* Finish the crane before the letters form, or let the beacon leave frame completely by 2860.

### Earlier notes I could not re-check from stills
These depend on motion or on the masters, so the stills can't confirm or clear them. They stand until someone
checks them on the masters:
* picture B1 (no placeholder frames in the masters; one Run in all three cuts);
* picture M2 (the ACCORD pop at 2171, the cloud-layer strobes, the 10 Hz hearth flicker);
* picture M5 (the payoff's accent on 2800);
* picture M6 (cut B's 13 s of near-black).

---------------------------------------------------------------------------------------------------------

## Cross-shot consistency

**Fire, the protagonist.** Six different renderings of fire are in use:
* blackbody orange-yellow: the hills, the montage, the coda, the Run, and now also the *thinking fire* and the
  *crown*;
* clipped neutral white: the ACCORD torches (2154) and the dawn sun;
* glitter gold: the Ring;
* pale yellow dots: the map;
* orange dots with round halos: the globe, the Run's fillers, the sea-coast beacons;
* white-hot tubes: the ember globe's seams.

Rules to adopt across the film:
* **One `fire_*` ramp** for every *human* fire: beacons, torches, flares, the answering fires, and the map's flame
  glyphs. Its core never clips to neutral white except at the designed flashes (1035, 2385–2400, 2800).
* **One `mind_*` ramp**, for the thinking fire and the crown only. `race_*` bleeds in through `redness()`;
  `accord_*` and `dawn_*` gold appear only from 2160.
* **Distant fires are never perfect round dots.** Each gets a two-pixel vertical flame, a flicker, and a 3:1 spread
  of size and brightness. This one rule fixes the Run, the sea coast, the globe and the map together, and removes
  the film's recurring "node graph" read.
* **Sparks and embers are always motion-blurred streaks.** EMBERS and HILLS already do this; the title embers
  don't.

**The beacon itself.** There are four designs: the first beacon (loaf-shaped stones, square bar basket), ICE (a
round basket on a cylindrical cairn), the Run's pyre, and the coda (a brick pillar with a square basket). Keep
the regional variation, but make every cairn dry stone, and make the hill's festival beacon a copy of the first.

**Sky, moon and stars.**
* The legend's night has a full moon (SEA) and moonlight from the left (the Run). The future's night has a
  crescent with city lights. Good: keep that distinction deliberate, and never show a crescent in the legend.
* Stars must die under glare (the dawn at 2400).
* The orbital ring must never cross fire, faces or type.

**Grade and contrast.** Section by section:

| section | grade now |
|---|---|
| EMBERS | true black, now lifted by brown haze (640–760) |
| FIRST BEACON | steel blue |
| Run | grey-blue, low contrast (the milky cloud plane) |
| ICE | teal |
| globe | neutral black space |
| MAP and ACCORD | sepia brown with lifted blacks |
| hills | violet |

The two sepia sections play back to back in C (map, then accord), so the legend's climax looks like an old
photograph. Put the ACCORD back into blue darkness, and keep the map warm only where fire has touched the paper.

**Scale.** Scale breaks in three places:
* the Run: islands read as rock piles, the camera is fast and low, and the cloud texture is flat;
* the FIRST BEACON reveal: no aerial perspective, no altitude;
* the title: giant embers above distant fires.

The rule to apply: the farther away something is, the hazier, bluer and lower in contrast it gets, and the slower
it moves across frame.

**People.**
* The hill now uses one figure language: true silhouettes.
* From above, the ACCORD emissaries are still pawns.
* The heroine is still a mannequin (a placeholder).
* ICE lifts the torch in triumph.

The rule for every department: silhouettes shaped by firelight and moonlight, and working gestures only.

**The red scarf.**
* The elder's is a bright, flat scarlet with a fringe; the heroine's (1375–1439) is a darker crimson tube. They
  must be the same object: the same hue under neutral light, the same length and fringe, and the same screen
  direction. Both already stream right; keep that.
* It must read in the reveal wide.
* In B the scarf is the only link between the girl and the old woman. Double it with the cairn rhyme (15).

**Typography.** Two faces, used coherently: Cormorant italic for the storyteller, Cinzel caps for the carved
oaths and the title. The title's glow and tracking suit C. In A and B, the kindle-in-place title (16) will read
less like a trailer.

**Grain and motion blur.** The sheets are mostly raw renders, so I can't judge the unified grain. Motion blur is
inconsistent: the Run and EMBERS use real vector or shutter blur, while the title embers are unblurred dots.

**What reads as generated, screensaver, keynote or kitsch** (highest risk first):
1. The orbital sunrise (A/B 2400–2655): a keynote slide.
2. The spreading dendrite tree and the dot-density world map (C 1960–2087): AI stock imagery and a corporate map.
3. The comet over the Arabian Sea (A/B 2020): a missile.
4. The raised fist in a red flash (all cuts, 1024–1035): a poster.
5. Particles flying in to assemble the title (all cuts, 2830–2880): a trailer-pack logo reveal.
6. The Ring as a glitter bangle (C 604–760), the crown as a burner (A/B 640–800), the thinking fire as a candle
   (all cuts, 480–600).
7. The Run's LED-dot beacons and dalmatian snow (1600–1679).
8. The ACCORD as a coin or seal from above (2350–2398).
9. The purple-gradient silhouette wallpaper (INTRO/CODA) and the aurora-over-a-figure wallpaper (ICE).
10. The Voronoi hand and the football globe (960–1035, 916–946): generative-art textures.

---------------------------------------------------------------------------------------------------------

## TOP 8 (priority order)

1. **The world answers with a missile, neurons and a dot map.**
   * **Shot and frames:** globe (A/B) 2020 (src 1860) and 2040–2087; map (C) 1960–2087.
   * **Change:**
     * Remove the comets that fly over water (`fires.py` `F_LEAP`; `webmap.py` `LEAPS`).
     * Make the unit of spread a flame that hops from summit to summit: one child per hop, no forked tips, no dots
       at the tips.
     * Irregular spacing and a 3:1 spread of brightness; the newest fires flare while older ones settle.
     * End C's map regional, and close enough that the ink hatching stays legible.
   * **Why:** this is the hope turn of all three cuts. Today it can be read as war (a launch), as the AI itself
     spreading (a neural net), or as a corporate slide (the dotted world), and B has no words to steer it.

2. **The grasp ends as a raised fist.**
   * **Shot and frames:** GRASP, all cuts, 990–1035.
   * **Change:**
     * Restage `hand_transform` and `hand_yaw` in `scene_c.py` as a descending, palm-down claw that closes over
       the prize and pulls it down and in.
     * Camera level with the hand or above it.
     * No frame with a vertical forearm and a fist on top.
     * An opaque crust with cracks that follow the anatomy.
   * **Why:** the film's image of concentration of power currently reads as the salute of popular resistance, in
     red, five frames before the hardest cut in the film.

3. **A/B's climax is the keynote sunrise, and its hit frame is grey with stars in the glare.**
   * **Shot and frames:** DAWN, A/B, 2400–2655.
   * **Change:**
     * Give A and B the mountain dawn from the Run's world, without the eagles.
     * If the orbit stays: kill the stars under the glare from 2398, push the flash to gold, and let the
       terminator sweep and the fires pale through 2440–2620.
   * **Why:** it's the climax of two cuts, and the single most stock image in the film.

4. **Act 1's three icons read as household objects.**
   * **Shot and frames:** the thinking fire, all cuts, 480–600 (a candle); the crown, A/B, 600–800 (a gas
     burner); the Ring, C, 596–760 (a glitter bangle).
   * **Change:**
     * **Thinking fire:** the mind palette instead of `look.blackbody` (`scene_b.py:404`); filaments restored;
       35–40% of frame height; a too-calm, breathing motion.
     * **Crown:** tines 2–3× taller in white-gold; the ice torus faded to a circlet; a higher tilt.
     * **Ring:** a smooth polished band:
       * a continuous surface, with one moving highlight that reflects the tower fires;
       * about half the current band height (`HB`), with a rounded section (`PEXP` ≈ 2);
       * forged by cooling from white through yellow to gold, throwing sparks, rather than by a sharp
         left-to-right wipe (600);
       * the inscription is the only thing that burns.
   * **Why:** these are the protagonist and the prize, and each now invites a laugh or a shrug.

5. **The Beacon Run is a terrain demo, not the beacons.**
   * **Shot and frames:** THE BEACON RUN, all cuts, 1520–1679.
   * **Change:**
     * Every beat beacon at least 25 px tall, with a flame, a flash, lit smoke and a pool of light on the snow;
       the far fillers varied and flickering.
     * Slope-aware snow on the near terrain; fix the curtain streaks at 1560–1580.
     * Fly higher and slower, with a longer lens after 1600; give the cloud sea relief and put haze between the
       ranges.
     * Always keep one fire in frame.
   * **Why:** it's the new 3-D showpiece, the makers' favourite kind of shot, and in C the centre of the Tolkien
     homage. Today its beacons are LEDs on molehills.

6. **The turn stands on a heightfield wall, with a mannequin.**
   * **Shot and frames:** FIRST BEACON reveal, all cuts, 1376–1439.
   * **Change:**
     * Render the reveal's plate in the Run's world, from her summit (`S1.summit()`).
     * Composite the rebuilt heroine: standing and watching, her scarf identical to the elder's, streaming right,
       and readable in the wide.
   * **Why:**
     * "Someone else lit a beacon" is the story's hinge, so the shot must show altitude and solitude.
     * One mountain world must run from here through the Far Peak and the Run.
     * In B, this shot is the only proof that the girl is the old woman.

7. **The payoff frame is crossed out.**
   * **Shot and frames:** CODA and TITLE, all cuts, 2788–2880.
   * **Change:**
     * Keep the orbital-ring line off the fire, the figures and the title; it strikes through "DAWN" at 2880.
     * Rebuild the cairn as a battered dry-stone cairn, ideally the first beacon's twin.
     * Shade the scarf from the fire.
     * Let the title kindle in place in clear sky, while the embers rise and die at the fires.
   * **Why:** this is the last image and the title card of every cut. A hairline through the film's name and a
     particle logo reveal are exactly what turns reverence into template.

8. **The ACCORD looks like a coin in an old photograph.**
   * **Shot and frames:** ACCORD, all cuts, 2120–2398.
   * **Change:**
     * Move the ambient into blue darkness, so the hearth is the only warm light.
     * Tilt the camera 10–15° off vertical, so the table has perspective.
     * Give the torches real flame shapes (no white capsules at 2154).
     * Do the whiteout with light (earlier picture M1).
   * **Why:** the moral climax mustn't read as an institutional seal, and the accord's gold has to be warmer than
     its surroundings to mean anything.
