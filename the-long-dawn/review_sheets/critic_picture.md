# THE LONG DAWN v2: picture critique (cinematography / VFX / editorial)

Reviewed 2026-09-26 ~08:35–09:15 against the draft masters `out/the_long_dawn_A.mp4` (built 06:27–06:38),
`_B.mp4` (06:38–06:50) and `_C.mp4` (08:36–08:48). All three have 2968 frames at 24 fps, and the letterbox is
clean (bar max 0.1/255). I read BIBLE_V2 (all of it), BIBLE §4/§6, `edit/assemble.py`, `edit/titles.py` and the two
music sync reports. I looked at full-film sheets of every cut (every 24th frame) and 1-frame sheets around
every cut, dissolve and key hit. I ran a whole-film per-frame luma / frame-difference scan of all three masters
(`review/critic_picture/qc_scan.py`), plus phone-size (390 px wide) views. Evidence sheets are in
`review/critic_picture/` (numbered to match the findings). All frame numbers are **v2**.
Known placeholders (the Run, heroine close-ups, C's DAWN, desert/karst/city) are judged only where they break
what surrounds them, or where the rebuild needs a spec.

**One-paragraph verdict.** The first half is close to trailer-grade. The torch-to-embers dissolve, the glyph
spiral, the 480 ignition, the RACE surges on every beat, the vortex (the edge-on fix works), the grasp and the 1040
IMPACT to black are all excellent and frame-exact. The Eye in cut C is instantly readable. The picture/music
sync at the hits I could measure is right to the frame almost everywhere. The film loses altitude in the second
half, for four reasons. (1) The masters are not deliverable as they stand: cut A has a gray `MISSING run_A 1645`
debug frame, and the three cuts carry three different versions of the Run. (2) The climax transition is a
global exposure push that bleaches the ACCORD like an overexposed photo. The ACCORD also has a visible
pop at 2171 and some strobing. (3) The "web of fire" (globe A/B, map C) reads as a network diagram, which is
a semantic risk in a film about AI and is worst in wordless cut B. (4) The figures still carry the producer's
"toy/mannequin" problem in the INTRO/CODA puppets and the ACCORD emissaries, and five departments draw people
five different ways. Cut B has **no** text-band dimming anywhere; I checked the code and diffed against A.

---------------------------------------------------------------------------------------------------------

## BLOCKER

### B1. Cut A master contains a debug placeholder frame, and the three masters carry three different Runs
* **Cuts:** A (the placeholder); A, B and C (version skew). **Frames:** A **1645**; Run 1520–1600.
* **What:** A 1645 is the edit's gray fallback frame with the text **"MISSING run_A 1645"** (evidence 01). It
  plays as a one-frame gray flash in the middle of the Beacon Run. The cause: `renders/run/` PNGs were being
  rewritten (06:24–07:04) while A and B were assembled, so `auto_assemble.log` shows `libpng error: Read Error`
  and `load()` silently substituted the placeholder. As a result the masters disagree with one another. A's
  1520–1528 come from an older Run than B/C (mean abs diff ~3 vs ~1.5 codec noise), and C's 1580 comes from a
  newer one than A/B (diff 7.9). A and B were also built before the 08:32 edit of `assemble.py`. The
  MOUNTAIN refinement pass has finished its EXRs (`render_run.log`: ALLDONE), but the PNG post step that will
  overwrite `renders/run/` in place **has not run yet**, so the same race will happen again.
* **Why it matters:** anyone watching A sees a debug card. Any A/B/C comparison is also comparing different
  Runs.
* **Fix (director / edit / lib):**
  1. `edit/assemble.py: load()`: when not `--draft`, retry an unreadable or missing frame 3 times (1 s apart),
     then **raise**. Never substitute a placeholder in a master.
  2. `lib/look.py: save_png()` (used by every department): write to `f_XXXXX.png.tmp`, then `os.replace()`.
     That makes every write atomic and removes the race for good.
  3. Don't assemble while any department is posting. Wait for the Run PNG post, then rebuild **all three**
     masters.
  4. Run a QC scan on every master: flag a frame whose luma departs from both neighbours by more than 2.5/255,
     a frame-diff above 3.5× the local median that isn't on the EDL cut list, and the placeholder signature
     (uniform ~38 gray). `review/critic_picture/qc_scan.py` does the first half of this in about 1 min per cut.
     It is how I found 1645, 2171 and the ACCORD flicker below.

---------------------------------------------------------------------------------------------------------

## MAJOR (ranked)

### M1. The climax flare (hearth to white to sun) is a global exposure push that bleaches the frame
* **Cuts:** A, B, C. **Frames:** 2388–2414 (evidence 02).
* **What:** `assemble.py` multiplies the whole ACCORD frame by up to +2.2 stops (2388–2399). The table and floor
  bleach uniformly while the emissaries and the radial shadows stay as **gray cut-outs on white**
  (2396–2399), which reads as an overexposed photograph, not a hearth erupting into light. The frame mean then
  goes 0.15 → **0.96 at 2399** → **0.63 at 2400**, so the brightest frame is the one *before* the hit, and the
  2400 climax lands on a frame 35% darker. (Whiteout in the music's breath is a valid idea, but the hit
  should still land on the white.)
* **Why:** this is the eucatastrophe transition, the single most important cut in the film.
* **Fix:** ACCORD already flares the hearth ×55 in-render (`scene.py: flare_mult`, src 2224–2240). Let *light*
  do the whiteout.
  1. `assemble.py`: replace the multiplicative push with an **additive radial burst** centred on the hearth
     (~960,402). Work in linear light: `lin += k(t)·warm_white·exp(-(r/R(t))²)`, with R growing from about 80 px
     to about 2500 px over 2385–2399 and k rising exponentially. Keep at most +0.5 stop of global lift, so the
     figures are *swallowed* by light rather than silhouetted.
  2. Hold peak white through 2400–2401. Start the dawn-side decay at 2401
     (`exposure(img, 3.0*(1-smooth((f-2401)/14)))`), so the hit lands on white and the sun condenses out of it.
  3. Optionally, ACCORD widens its bloom radius in src 2230–2240 so the in-render glare engulfs the frame.

### M2. ACCORD temporal artifacts: a pop at 2171, cloud-layer strobes, 10 Hz hearth flicker
* **Cuts:** A, B, C.
* **(a) Pop at 2170→2171** (evidence 03). The merge glare vanishes in one frame (frame mean 67.5 → 46.9, −30%),
  exactly as Oath I appears. Root cause: `shots/accord/particles.py: ignition_flash()` returns `[]` for
  `t > IGNITE+10` (src 2010), while its pale blob still carries `9·exp(-10/2.5) ≈ 0.16` over a ~1.5-unit radius.
  **Fix:** end at `IGNITE+40`, or taper
  (`k = exp(-a/2.5) * (1 - smoothstep(a, 8, 16))`), then re-render src 2000–2040 for accord_A/B/C.
* **(b) Cloud-layer punch-throughs, 2100–2117** (src 1940–1957; evidence 04). A warm haze ring swells for about 6
  frames and then snaps off in 1–2 frames, twice (luma 77.6→56.3→49.0 at 2106–08; 82.2→55.0→45.4 at 2114–17).
  At speed that is a ~3 Hz strobe. **Fix:** fade each layer's density by camera distance
  (`alpha *= smoothstep(0, ~2·thickness, |h_cam − h_layer|)`) so the camera dissolves through it.
* **(c) Hearth flicker.** `scene.py: hearth_intensity()` includes `0.05·sin(t·2.71)`, about **10.4 Hz at 24 fps**
  and close to Nyquist. With the other terms it most likely produces the single-frame, whole-frame dips of 8–10%
  that the scan found. At 2227 the table goes 99.7 → 89.7 → 95.2; there are 9 such dips in 2184–2278 in A/C, and
  4 at the same frames in B, which points to the shared hearth code. It reads as a frame flicker,
  not firelight. **Fix:** keep the flicker that lights the *table and room* at or below 4 Hz (≤ 1.0 rad/frame)
  and ≤ 3% amplitude. The flames themselves can dance faster.

### M3. The "web of fire" reads as a network diagram (a semantic risk in a film about AI)
* **Cuts:** A/B globe 1920–2087 and DAWN 2400–2655; C map 1920–2087 (evidence 05, 06).
* **What:** the web is drawn as smooth, uniform glowing strokes between nodes: an airline-route map or the stock
  "connected world" tech graphic. In C's wide map (~2030–2087) it becomes an even triangulated mesh over every
  continent (the MAP dept already suspected "network diagram"; I confirm it). The film's subject is AI, and a
  glowing global network is exactly how AI and the internet are usually pictured. So the image can read as the
  *thing spreading*, the opposite of "people lighting fires". Cut B has no words to steer the reading.
* **Fix (GLOBE `shots/globe/web.py`; MAP `shots/map/webmap.py`):** show propagation as **travelling flares** (a
  hot head with a short ember tail that cools and fades within 12–20 frames) from beacon to beacon. What
  *persists* is the **fires**: small flickering points with halos, varied size and brightness, not the lines.
  If links must persist, draw them as broken chains of tiny fires with irregular brightness, never uniform
  strokes. In the map, keep a branching tree (cap node degree; no Delaunay), and let the paper stay dark
  where no fire has reached. At DAWN, let the day side swallow the links.

### M4. People: puppet seams, clay emissaries, five figure styles, and the scarf link
* **Cuts:** A, B, C. (Evidence 07, 08.)
* **INTRO/CODA** (hills; the two-shot 160–300 and the medium 2656–2788, where the audience studies them): every
  limb capsule carries its own rim outline, so the shoulder and elbow joints show and the Elder and Child read
  as jointed mannequins. The scarf is a rigid tapered tube with finger-like fringe; it doesn't fly, it points.
  **Fix (`shots/hills/puppet.py` / `characters.py`):** smooth-union the body parts *before* the rim/outline pass
  and drop all internal outlines. Make the scarf a ribbon: width taper, a travelling wave along its length,
  twist, secondary flutter, and finer fringe.
* **ACCORD emissaries** (2150–2400): from above they are smooth terracotta and tan lumps with round hood-domes and
  pale hand-nubs, which read as clay pawns. That is the producer's "toy-like emissaries" note in a new colour;
  v2 asked for *dark cloaks*. **Fix (`shots/accord/figlook.py`):** near-black cloth in the silhouette palette,
  lit only by a warm hearth rim on the table side; draped folds (displacement); deep peaked hoods; varied
  builds; drop the pale nubs.
* **One figure language:** the montage uses flat silhouettes, the hills puppets with rim lines, the accord clay,
  the city a lit realistic figure, and the incoming heroine is 3-D. **Rule for all departments:** people are
  silhouettes, shaped only by firelight (warm rim/fill on the fire side) and moonlight (cool rim), in the
  shadow-play spirit of the bible's references.
* **Heroine rebuild (in progress):** her scarf must be *the Elder's scarf*: same `scarf_red`, same length and
  fringe, streaming to the same side. It must read at least once in the wide reveal (1380–1439), where today she
  is a tiny scarecrow. In cut B the scarf is the **only** thing that tells us the girl on the mountain is the old
  woman on the hill.

### M5. The payoff (the child lights the beacon) has no accent on the hit, and the cut pre-empts it
* **Cuts:** A, B, C. **Frames:** 2780–2830 (evidence 10).
* **What:** the medium-to-wide cut is at **2788**, 8 frames off the 2780 beat and only 12 frames before the catch,
  so the eye is still re-reading the new framing when the fire takes. In the wide the child is ~230 px tall
  (≈45 px on a phone), but the torch-in-basket contact is ~60 px (≈12 px on a phone). The first flame shows on
  **2801**, one frame after the timpani (audio-leading 42 ms, the more noticeable direction). The visible "whoomp"
  only builds over 2807–2812 (frame mean 0.115 → 0.156), so the timpani at 2800 hits an almost invisible event.
  This is the film's last story beat.
* **Fix (`shots/hills/coda.py`, `CUT=2628`, `CATCH=2640` src):**
  * **Preferred:** stage the catch *in the medium*; the cairn is already in frame left. Torch into the
    basket, fire catches on 2800 and lights the child's face and the Elder's scarf. Then cut wide on the next
    strong beat (2820) as the fire roars and the answering wave rolls out.
  * **Cheap:** `CUT=2620` (on the 2780 beat and the piano's "torch in the child's hands") and `CATCH=2639`, and
    start the wide ~15% tighter before the crane.
  * **Either way:** put a contact accent on 2800 itself: a 2–3 frame spark burst and flare from the basket, then
    the whoomp, rather than a flame that only grows 7–12 frames later.

### M6. Cut B: about 13 s of screen that is effectively black (1040–1360)
* **Cut:** B (A/C are carried by text over 1060–1190).
* **What:** 1040–1199 is black with one ember whose core is about 6×6 px (≈1 px on a phone) and gone by about 1150.
  Then 1200–1235 is near-black (mean 0.046), and the flint strikes light the scene for only 3–4 frames each.
  That is 11% of the film in which a phone viewer sees nothing, in the one cut with no words.
* **Fix:** give B's silence a subject. EMBERS renders `embers_B` for 1040–1199: a closer, larger ember (hot core
  20–30 px, soft bokeh 60–80 px, a thread of smoke) drifting down and dying on the piano's last note, with 20–30
  frames of true black before 1200. In the heroine rebuild (all cuts), keep a faint moon rim on her and the cairn
  between strikes, and let each strike's light decay over 6–8 frames.

### M7. The FIRST BEACON reveal lands flat, and there are three mountain worlds in 12 seconds
* **Cuts:** A, B, C. **Frames:** 1376–1439 (hills), then 1440 (montage), then 1520 (Run) (evidence 09).
* **What:** the "sea of moonlit peaks below" is a wall of vertically streaked, extruded spikes at summit height.
  There's no aerial perspective (far ridges are as contrasty as near ones) and no sense of altitude, so the moment
  written as "she is tiny and the fire is the only warm thing in a cold world" doesn't land. Then 1440 (dark
  faceted rock pyramids on a flat plain) and 1520 (Blender massifs over a cloud sea) are two more *different*
  ranges, although the story says it is one range. The close-ups also have a teal aurora sky that the wide
  doesn't have.
* **Fix:** once the Run's look is settled, MOUNTAIN renders the reveal's background plate from the Blender world at
  the summit camera, with the heroine, cairn and fire from HILLS composited on top. The minimum fix, in
  `shots/hills/peaks.py`: drop the far range 10–15° below the horizon line, add 3–4 haze layers and a valley cloud
  sea, and kill the vertical-streak texture (slope-aware snow). Pick aurora or no aurora for the whole FIRST
  BEACON.

### M8. The Beacon Run (placeholder): what breaks now, as a spec for the rebuild
* **Cuts:** A, B, C. **Frames:** 1520–1679.
* The **1520 cut** (on a timpani hit) goes from the shepherd's roaring beacon to a small candle-like flame on a
  white dome, so fire energy *drops* on the hit. The Run is also brighter and greyer (mean 0.157 → 0.207) than
  the deep-blue far peak.
* **No visible fire in frame** (nothing warm-bright) in 1528–1539 and 1573–1576.
* **Beats 1540/1560:** the new beacons are 2–14 px of fire. The timpani plus "whoomp" on every beat hits nothing
  visible, which is the Mickey-mousing the tone rules warn about, only with the picture missing. 1600–1660
  register only as +50–100 px of fire.
* **1584→1585:** the great pyre leaves frame and its light spill on the snow vanishes in one frame (luma 56→46).
* **Spec:**
  * Open on a big fire so the 1520 cut carries energy.
  * Always keep at least one fire in frame.
  * Make every beat ignition a readable event: a flame at least ~25 px tall at 1920, a 2–3 frame flash, and
    light spill on its peak. Place them on thirds, not only on the horizon line.
  * Fade light spill out over 6–8 frames when a fire exits frame.
  * Match the far peak's palette and exposure.

### M9. The fourth oath gets the shortest read, and the most important one
* **Cuts:** A, C. **Frames:** 2282–2320 (evidence 15).
* **What:** "WHAT IT GIVES, IT GIVES TO ALL" lights at about 2282–2288. The pull-back starts at about 2300 and
  shrinks the table ~2.3× by 2318, so the vow that resolves the story is upright and full-size for about
  25–30 frames, less than the other three.
* **Fix (`shots/accord/scene.py _H_KEYS`, variants A/C):** shift the rise keys (src 2136/2149/2162) about +14
  frames. The ring sweep (2320–2380) can start while the camera is still rising.

### M10. Leftover text-band "calm" in cut A's DAWN, and in cut C's fallback DAWN
* **Cuts:** A, C. **Frames:** A 2422–2510; C 2422–2606 (evidence 12).
* **What:** both use the v1 `renders/globe` frames. There, `shots/globe/shots.py: calm()` halves the web and the
  **blooming hearth lights** in y≈545–715 for v1's T13/T14 windows, and a sun-spike mask sits in y 470–560 from
  2415. A has no text until 2512, and C has no text over the dawn at all. So the "new lights bloom" beat is dimmed
  in a horizontal stripe for nothing (the A-vs-B diff shows it clearly).
* **Fix (`edit/assemble.py: dawn()`):** use `globe_B` (rendered with `LONGDAWN_NOCALM=1`) for every cut until
  DAWN_C lands. A's one line (2512–2600) sits over dark night side, and the title halo handles legibility.

### M11. The CODA's answering fires drown in the title's ember glitter; the title forms over the fire
* **Cuts:** A, B, C. **Frames:** 2807–2865 (evidence 11).
* **What:** the ember title's gathering embers make a dense cloud of out-of-focus gold sparkles over the whole
  valley. It reads as pixie dust ("cute sparkle" is on the blacklist) and can't be told apart from the answering
  fires, which are the story beat (the world answers again). From 2840 to 2860 the letters form across the
  beacon flame and the figures.
* **Fix:**
  * `edit/ember_title.py`: source the embers *from* the answering fires as thin, motion-blurred rising streaks.
    Use about a third of the count and smaller bokeh.
  * `shots/hills/coda.py`: make the answering fires distinct beacon points (about 2× size and brightness, small
    halos) igniting in a visible rolling wave.
  * Delay the gather ~12 frames, or start the crane-up earlier, so the letters form in clean sky.

### M12. The desert reads as snow, so two white mounds play back to back
* **Cuts:** A, B, C. **Frames:** 1680–1739, then 1740–1799 (evidence 09, 14).
* **What:** a symmetric pale grey-white cone, a black diagonal band at lower left, and a figure that is invisible
  on a phone. The next shot (ICE) is another white mound with a figure and a beacon on top. The montage's point,
  *every kind of place*, is lost at its first step. (Listed as "may be upgraded"; this is why it must be.)
* **Fix (MONTAGE-3D / `shots/montage/s2_desert.py`):** knife-edge crests with the slip face in deep shadow, several
  dune ridges receding into haze, a faint warm-khaki bias in otherwise desaturated sand, and a clear robed
  silhouette on the crest. Remove the black band.

---------------------------------------------------------------------------------------------------------

## MINOR

1. **Lines bleed across hard cuts** (A, C). The 1370 line ("So someone else lit a beacon." / "And on a cold
   mountain…") runs 7 frames into the FAR PEAK (1440–1447), carrying the heavy snow halo (0.95/0.78) onto a
   dark plate (evidence 13). The 1060 lines run 2 frames into 1200. **Fix (`edit/titles.py`):** `t_out=1426`
   and `t_out=1186`.
2. **Phone legibility.** Cut A's opening card is 46 px at 0.9 opacity (≈9 px on a 390 px-wide phone, marginal).
   Raise it to 52–54 px at opacity 1.0. The 56 px Cormorant italic story lines are at the lower limit (evidence
   14); consider 60–62 px or weight 600.
3. **The torch catch is 1 frame late** (2801 vs the 2800 timpani): `CATCH=2639` (see M5).
4. **DAWN→CODA dissolve (2624–2655).** The sun-to-torch alignment is lovely; keep it. But the Earth's limb arc
   ghosts through the Elder and Child for about 1 s. Use a luma-keyed dissolve (the bright sun/torch leads and
   the dark disc clears first) or shorten it to ~20 frames.
5. **ACCORD hearth.** From above it is a flat, pale, marbled disc with no flame volume and no sparks rising toward
   the lens. The protagonist fire at the council reads as a plate, and in C the Ring (~120 px) lies on a plate.
   **Fix (`shots/accord/fire.py`):** top-down flame tongues, sparks rising toward camera with parallax, more gold
   saturation.
6. **MAP (C) is too warm and bright.** Tan parchment fills the frame, so the fire lines have low contrast and fire
   is no longer "the only saturated warm light". Darken and cool the paper (candle-dark with a vignette), and let
   the fire lines light the paper around them.
7. **ICE aurora** (1740–1799) is the most saturated thing in frame and out-draws the fire. Take it down 30–40%.
8. **FAR PEAK motivation** (1440–1479). The first beacon on the far horizon, the reason the shepherd acts, is a
   ~6 px dot (≈1 px on a phone). Give it a flare and a 25–30 px glow column, flaring on the 1460 beat.
9. **KARST and CITY.** The karst pinnacles read as flat-topped cylinders. CITY (1840–1879) is the only locked-off
   shot I measured (zero camera motion; everything else breathes); add a slow push.
10. **DAWN second half** (~2520–2624) is nearly static, and the sun stays a white star. The climax never reaches
    the bible's "ACCORD → DAWN: pure warm gold". Warm the sun and limb glow, and let the terminator sweep or new
    lights bloom through 2520–2620.
11. **C: "One ring to rule them all." (565)** lands about 30 frames before the Ring exists (~595–608), and the
    Ring's invented inscription is never legible. Shift the line to ~588, or forge the ring earlier
    (`embers/scene_b.py forged(t)`), and give the inscription one readable flare as the ring turns (~610–640).
12. **A/B crown** reads as a halo or ring-light (an ice-white ring with 4 small prongs). More and taller tines
    would read as "crown".
13. **INTRO 284–296.** The "So we remember…" halo darkens the torch flame under the word "how". Lift the line
    ~30 px or lower that line's halo.
14. **The orbital ring** in the INTRO/CODA sky is a 1-px straight line; on a phone it reads as a scratch. Give it
    a soft glow and a faint arc.

---------------------------------------------------------------------------------------------------------

## Checked and clean
* **Cut B text-band dimming: none.** `embers_B` (331–909) and `globe_B` render without the band. Where B falls
  back to other layers (embers_v2 910–1039, v1 embers 1040–1199), the band weight is 0 or sits on black away
  from the ember. hills and montage never dimmed; they used compositional calm.
* **Banding:** none. Contrast-stretched sun glare into space, dusk sky, silence black and title sky are all
  dithered by the grain.
* **Aliasing / crawl:** none on the orbital-ring line, the rotating carved oaths or the karst silhouettes.
* **Missing / duplicated frames:** only A 1645 (B1). B and C have none.

## What works (keep it)
* **Transitions:** the torch-to-rising-embers dissolve (300–339) is the best transition in the film. The IMPACT
  (fist closes 1028–1034, red-white flash 1035–1039, hard black with the ember at 1040) is superb. The
  globe/map → ACCORD dissolve (2072–2087) is a clean graphic match of radial lines. The sun lines up with the
  Elder's torch in the dawn→coda dissolve.
* **Images:** the glyph field (real scripts, math, DNA, notes) and its spiral; the thinking fire's ignition;
  tower surges on every beat; the vortex facing the lens; the Eye (C) is instantly iconic; the grasp.
  The SEA (the flare's warm reflection beside the moon's cold glitter) and the ICE ignition lighting the snow are
  lovely. The oaths are crisp, light upright at the top on the beat, and read at phone size. B's ornament
  mandala and its sweep are elegant. The ember title is gorgeous and finishes exactly on the final chord (2880).
* **Cut B reads without words:** a fire born from writing, a hunger, a grasp, a world cracking, one ember dying; a
  woman on a summit; the chain of fires; the council; the sun; the child. The weak spots are M3 (the web could
  be misread), M4 (the scarf link), M6 (the long black) and M5 (the payoff's size).

## Picture ↔ music sync (picture side, measured on the masters)
| v2 | event | picture |
|---|---|---|
| 480 | IGNITION | onset exactly on 480 (luma 12.7→31.5), peak 483. OK |
| 640 | RACE | surge onset on every beat (640, 660, 680…), peak +6. OK |
| 1036–1040 | fingers close, flash, IMPACT | flash 1035–1039, hard black 1040. OK |
| 1236 / 1262 / 1290 | flints | full light exactly on the beat, with a 1-frame spark lead (1235/1261/1289). OK |
| 1318 / 1360 | kindling / ROAR | 1319 soft catch / 1360 (luma 16→61). OK |
| 1480 | shepherd's beacon | 1480 (tiny anticipation at 1479). OK |
| 1520–1660 | Run cut and beats | cut 1520 OK; beat ignitions sub-pixel at 1540/1560, small after (M8) |
| 1700 / 1760 / 1810 / 1850 / 1890 | montage ignitions | all on the frame. OK |
| 1920 | globe/map | hard cut, first beacon visible. OK |
| 2160 | merge | 2160 exactly. OK |
| 2200 / 2240 / 2280 | oaths (A/C) | glow begins +2 frames, upright. OK |
| 2320–2380 | ring sweep | OK |
| 2385–2400 | flare → sun | timing OK; look is wrong (M1) |
| 2800 | child's beacon | first flame **2801** (one frame late), whoomp only at 2807–2812, 12 frames after an off-beat cut (M5) |
| 2880 | final chord | title completes on it. OK |

---------------------------------------------------------------------------------------------------------

## TOP 5 (in order)
1. **B1: rebuild all three masters, and make it impossible to ship a placeholder.** A plays a gray
   `MISSING run_A 1645` frame, and the three cuts carry three different Runs. Fixes: `load()` fails hard in
   masters, `save_png()` writes atomically (tmp + `os.replace`), no assembly while departments post (the Run PNG
   post is still pending), and a QC scan on every master.
2. **M1: redo the climax flare (2388–2401).** Replace the global exposure push, which bleaches the ACCORD with
   figures as cut-outs, with an additive radial light burst from the hearth. Put the peak white on 2400, the hit.
3. **M2: fix the ACCORD's temporal artifacts** (all cuts; cheap):
   * the 2171 pop (`particles.py ignition_flash` hard-stops at src 2010);
   * the cloud-layer strobes at 2106/2115;
   * the ~10 Hz `sin(t·2.71)` hearth flicker.
4. **M3: make the web of fire read as fire, not as a network** (globe A/B, map C). Use travelling flares and
   persistent *fires*, not persistent uniform lines or meshes. In a film about AI, and wordless in B, this is the
   biggest misreading risk.
5. **M4: people.** Remove the puppet joint outlines and the rigid tube scarf in the INTRO/CODA. Turn the ACCORD's
   clay pawns into dark cloaks. Adopt one silhouette-lit figure language across departments. Make the rebuilt
   heroine's scarf unmistakably the Elder's, since it is the only link in cut B.

(Next in line: M5 the payoff staging at 2788–2801, M6 cut B's 13 s of black, M7 the flat FIRST BEACON reveal
and the three mountain worlds.)
