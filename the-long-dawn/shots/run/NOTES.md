# RUN-A (lane runA) REPORT: THE CROSSING greybox (risk test #3) + FALSE DAWN stills
*27 Sep, 01:00Z. Revision 1 absorbed (A 4:30; R4 cut; A ends in the blue hour, no sun disc; FALSE DAWN and the red
under-glow are must-haves).*

## STATE AT THE BLOCK (01:20Z to 05:00Z): renders running on this box
* `scratchpad/bg_chain.sh` (launched 00:52Z): `renders/crossing_A/grey_q` (0.25 scale, all 960 frames) ->
  `grey_q_few` (the fallback: 12 larger walkers, every 2nd frame) -> `grey_h` (0.5 scale, 960 frames). Each stage
  writes an mp4 to `review/v3/crossing_greybox_{q,q_fewer_larger,h}.mp4`. Log: `scratchpad/bg_chain.log`.
* `scratchpad/bg_fd.sh` (00:58Z): FALSE DAWN, three designs at 0.5 scale (frames 160, 300, 420) ->
  `renders/falsedawn_A/half/{arc,cone,veil}`, then the arc at full res -> `renders/falsedawn_A/full_arc`.
* **Do not edit `crossing.py` while the chain runs**: later stages start new processes that import it.
* 01:03Z changes (picked up by `grey_q_few` and `grey_h`, NOT by `grey_q`): about half the walkers carry the lantern
  on the far side (their bodies silhouette against their own light: less "string of lights"); the rope much subtler
  (it had been stringing the lights together like a festoon); the red under-glow +50% and deeper red (it now reads
  as a dull smoulder along the horizon band). FALSE DAWN arc: a thicker high deck silvered from beneath (the bible's
  "silvers the undersides of high cloud"); `scratchpad/bg_fd2.sh` renders it to `renders/falsedawn_A/full_arc2`
  (full res, frames 300, 420) and `half_arc2` (8 frames across the swell).
* NEXT on resume: contact sheets from grey_q / grey_h / half (-> review/v3/*.jpg); judge main vs fewer-larger at 0.5
  scale; the fixes below; commit + push `claude/v3-runA`; hand the director the RENDER_SPEC.

## THE CROSSING (R6): what is built (`shots/run/crossing.py`, `shots/run/sdfppl.py`, additive `world.py`)
* **Set.** A knife-edge arete (8 `ridge_row` segments, a wandering crest, flanks falling ~60 deg to the cloud sea
  800 m below) in a clear basin of the cloud sea at world (-5200, 14400), between two faceted summits (`crag_row`);
  two small rock shoulders beside the crest carry the watch-fires of this stretch. The line walks EAST (yaw -30 deg,
  the world's sunrise azimuth); the camera stays on the south side and never gets ahead of the lantern.
* **People.** 2 hooded bearers carry the great lantern (hexagonal, iron and glass, pierced roof, finial ring) hung
  from a crossbar between two poles on their shoulders; 40 roped walkers (packs, a third with staffs, varied heights
  and gait), each with a small orange hand-lantern that swings with the step. Walk cycles follow the distance
  actually walked (no foot sliding); the spacing breathes like an accordion. All 3-D SDF (`sdfppl.py`), depth-tested
  into the terrain, lit by every lantern (inverse square) plus the moon; no faces anywhere (hoods, seen from behind
  or side-on at 20-45 px).
* **Pace.** 0.45 m/s: one 0.375 m step per beat at 72 BPM. The great lantern is a 2.6-unit point light: its pool on
  the snow is a few paces across. The camera draws back (5.5 m -> 85 m, exponential, eased), swings from behind the
  rear bearer to side-on, and settles at crest level.
* **Two clocks.** The people keep real time. The sky keeps a faster clock that eases in once the draw-back is under
  way (46 deg of sky, about three hours): stars become arcs about a low pole (latitude 8 deg, so the pole sits inside
  the wide), the moon sets behind the arete (backlight at first, darkness by the end), the watch-fires burn low and
  are fed (flare) on that clock, 34 far watch-fires on the islands do the same, the red under the cloud goes out
  patch by patch, the lantern's fire warms from ice-white to gold, and a faint trail of light opens on the path
  behind the line.
* **The red under-glow (must-have)** is a reusable ADDITIVE post-pass, `world.cloud_glow(C, D, P, CR, UG, fogp, out)`:
  emission on cloud-sea pixels from a patch table (x, z, radius, rgb, noise scale, trough bias), brightest in the
  troughs, lighting the cloud-top mist in front of it. Existing callers are untouched (also added: `world.heights`).
  In the crossing the only visible cloud sea is the band under the horizon beyond ~15 km (the arete hides the rest),
  so the patches sit 14-48 km out and read as a dull red smoulder along the horizon, going out one by one.

## Risk test result (quarter-scale stills; motion greybox rendering)
* **Composition: PASS.** Close-up match cut on the heart from behind the rear bearer; the draw-back; the wide at crest
  level with the line silhouetted against the islands, the cloud band and the sky; the star trails arcing about
  the pole read as "the sky wheels" (first version read as warp-speed streaks; fixed with a low pole in frame,
  shorter arcs, compressed star brightness).
* **The line: NOT YET PASSED at 0.25 scale.** From above, the first wide read as LED dots, as the bible predicted.
  With the camera lowered to crest level (figures ~43 px at full res, backlit by the moon and the far cloud) it
  reads as people walking in the half-wide, but the final wide still leans "string of lights" at 0.25 scale.
  Judge on `grey_h` (0.5 scale) against `grey_q_few` (12 larger walkers). If main still reads as dots: ship the
  fewer-larger variant (`--variant few`: 12 walkers, camera at 52 m).
* **Lantern's pool: PASS** (visible warm pool on the crest in the wides). **Rope: weak** (sub-pixel; reads only
  near lanterns). **Trails: too faint to judge**; keep subtle. **Red under-glow: faint**, needs +50% and one patch
  nearer the frame centre.
* **Weaknesses to fix before any full render:** the bearers in the first 5 s are mannequin-grey where the lantern
  lights them (cloth needs fold normals and noise albedo, a darker grade); the heart needs its filament detail tuned
  at full res; the far islands read a little flat.

## FALSE DAWN (R1): three still designs (`shots/run/falsedawn.py --design {arc,cone,veil}`)
Moonless midnight from the Dawn-C vantage over the foothills, looking at the world's sunrise azimuth; the Milky
Way rising from the left horizon; a high altocumulus deck; stars and the Milky Way attenuated by the glow's local
brightness (they go out near it); terrain lit from just over the horizon so the far peaks rim and cast long shadow.
* **arc**: a wide low arch, cold white with a breath of cyan, soft shadow rays fanning up from the far peaks.
* **cone**: a leaning pyramid of pale light (the zodiacal light, the astronomer's own "false dawn").
* **veil**: no visible source, only the deck lit from beneath and a thin line on the skyline.
* **WINNER: arc.** It is the only one that reads as morning come too early: the rays come from a point below the
  horizon, which no city makes and a moon rarely does, and it is too broad to be a moonrise. The cone reads as a
  searchlight or sci-fi beacon. The veil reads as cloud lit by a city (light pollution). Keep the rays soft and
  few (a "studio logo" read appears as soon as they are crisp).

## RENDER_SPEC (director launches; do NOT launch before the look is approved at 0.5 scale)
* THE CROSSING: `cloud/jobs/crossing_a_{1..8}.json`, 120 frames each, `python3 shots/run/crossing.py --range A-B
  --procs 4 --skip` -> `renders/crossing_A/f_%05d` (shot-local 0-959; 1920x804; ship jpg). One machine type.
  Estimated cost: 3-10 min/frame single-thread at full res (close-up frames 0-300 heaviest), ~80 box-hours
  single-thread; about 2-3 h wall on 8 four-core boxes. Fallback variant: add `--variant few`.
* FALSE DAWN: `cloud/jobs/falsedawn_a_{1..4}.json`, `python3 shots/run/falsedawn.py --design arc --range A-B --procs
  4 --skip` -> `renders/falsedawn_A/f_%05d` (shot-local 0-479). Far terrain only (Dawn-C cost, 2-4 min/frame).
* Data that must be committed: `shots/run/crossing_fires.npy` (the island fires), plus the Run's summits/beacons.

# >>> RUN-C (C . THE LIVING INK: C16 REVEAL, C17 INK RUN, C24 ILLUMINATION) - report 27 Sep <<<
## PAUSED 27 Sep ~10:40Z (director's usage pacing). Nothing is rendering. RESUME HERE
Done since the handoff: read the H5 calls, the brief, the log and the critic, and looked at the latest test frames.
The pre-H5 code is committed as b791569 (a restore point). No H5 fix has been coded yet.

**Next steps, in order.** Iterate with `inkpass.py` on the EXISTING half-res AOV caches (`renders/run_c_tests/aov/*_050_*`:
scroll 12/168/232-255/300, reveal 0/108-131/200, illum 2410-2800), so no AOV re-render is needed while the look is
worked out.
1. **Cross-contour hatching** (replaces the all-vertical fall-line dashes). Family F1 is the iso-lines of the world
   height h (A[...,5]). Space them from the one-sided screen gradient (take the side whose depth stays continuous),
   with nested intervals dh = 0.5*2^L m. Give each line a rank from its birth level relative to the continuous lam
   (c = M - lam), which gives continuous LOD with no pops: coarse lines are drawn at light tones and in-between
   lines fill in as the tone darkens. Break the lines into dashes with world-anchored octave noise, band-passed by
   screen size, and taper them at the gaps. Family F2 is the shadow crosshatch: iso-lines of the tilted planes
   h*cos(b) + psi*sin(b), with psi = the plan coordinate along the shot's mean camera right vector and b = 45 deg.
   They run diagonally on cliffs, only where T > 0.45. Plan-only fields go vertical on cliffs, so don't use them.
2. **Outlines at 2x.** Vary the weight by light: thin and broken on lit edges, heavy in shadow. Add world-anchored
   pressure noise and pen lifts, and taper into the fire's light. This fixes the reveal's near-ridge "stock
   chart". Blur the line coverage ~0.5 px at 2x before the downsample.
3. **Inked beacons** replace fire2's gouache flame and the gold-leaf aureole disc. Draw a glyph of 3-5 swaying
   teardrop tongues, outlined at the terrain's pen weight and filled with a gold wash (vermilion at the root),
   depth-tested and anti-aliased from analytic distance. Two inked smoke strands curl up and drift. The catch blooms
   as a watercolour gold wash on the firelit rock (world-anchored, from the firelight AOV), never a halo in the
   sky. Remove the horizontal gold striping on the reveal's near slope.
4. **The sun in ink.** A slightly irregular pen circle with paper and a pale gold wash inside (pooled at the rim),
   plus many fine ruled rays of broken, varied length. Try it with and without the rays. The skyline cuts it.
5. **R15 fill follows the light.** On the land, use the sunlit term (n.l x shadow), so the edge is the terminator.
   In the sky, spread by the plate's luminance threshold falling over time. Feather it wide with grain. There
   must be no hard edge anywhere; check 2410-2800 (dawn_C plates exist only for 2400-2655; the cloud driver renders
   its own).
6. **2x production.** `ink_final.py` defaults to scale 1 and ss 2 (drawn at 3840, INTER_AREA to 1920),
   ~2 GB per worker. Tests use scale 0.5 and ss 2, viewed at 960 (a 2x downsample) plus 1920 crops.
7. Re-run the boil gate (`ink_check.py`) on scroll 232-255 and reveal 108-131. Rewrite the 14 cloud jobs, make a
   review sheet, and send JOB READY. Commit keeper.py first: ask RUN-B or the director, since C16 needs it on the
   cloud box.

## STATE AT HANDOFF (new RUN-C agent, 27 Sep 10:10Z)
* **Code** (`ink_aov.py`, `render_ink.py`, `inkpass.py`, `ink_final.py`, `ink_check.py`) was never committed; all
  five were untracked at handoff. Also untracked: `keeper.py` (RUN-B's), which C16 imports for B's summit dressing.
  The cloud box needs it in the repo.
* **Renders on disk** (tests only, half res, all PRE-H5; `renders/run_c_tests/`, 2.2 GB): scroll 232-255 plus the
  re-dealt control and stills 12/168/300; reveal 108-131 plus control and stills 0/200; illum 2410/2440/2480/2560/2640.
  The illum chain died at 2720 because `renders/dawn_C` only holds 2400-2655 (the cloud driver renders its own plate).
  No production frames exist.
* **Boil gate PASS** (anchored/re-dealt residual): reveal land 0.246, scroll land 0.257. The scroll cloud reads 0.890
  because the contours ride the drifting sea. It is movement, not boil.
* **Cloud jobs** `cloud/jobs/runC_{reveal_a-d,scroll_a-d,illum_a-f}.json` were written but NEVER launched. They are
  PRE-H5 and must not be launched as they stand. They will be rewritten after the fixes (2x, new driver flags).
* **H5 calls in my area, none applied at handoff:** beacons hand-inked at the terrain's line weight with a gold wash
  (no emoji flame, no scalloped halo, no pixel edge); cross-contour hatching that follows the form; render at 2x
  and downsample (no jaggies); feather R15's colour so it follows the light (no hard edge in the sky); break or vary
  the heavy near-ridge line in the reveal (it read as a stock chart); redraw the flat-coin sun as ink.

**What it is.** C's mountain world drawn as a moving illustration in the old book's manner, from the Run world's own
AOVs: page-fixed parchment; iron-gall hatching down the fall line; outlines at depth breaks; the cloud sea in
iso-height contour lines with stipple; a sepia wash in the shadows; the fires the only colour, painted like gouache,
each catch blooming as gold leaf laid on the page, the firelit ground gilded; at dawn the drawing is coloured like a
hand-coloured engraving wherever the sun's light touches, with the sun a tooled gold-leaf disc. REVISION 1 shots
(locked bar map `music/v3/barmap_C.json`): **C16 THE REVEAL** (bars 46-48, 240 f), **C17 THE LIVING INK RUN** (bars
49-52, 320 f: a lateral truck at beacon height, like a scroll unrolling; catches at local 40, 80 ... 280 = bar 49 b3
to bar 52 b3; her own fire burns on the horizon from frame 0), **C24 THE ILLUMINATION** (bars 72-77, 480 f, no eagles).

**The gate: PASS (strokes do not boil).** Hatching is anchored to the terrain by construction (every mark is a function
of world position: seeds on a nested jittered world grid, the fall line of the height field at the level's own scale,
a fixed rank per seed for tone, tone band-limited to the stroke scale) and it was measured, not just eyeballed:
`ink_check.py` reprojects every inked world point of frame N+1 into frame N and compares the ink, against a control
where the same frames have every stroke re-dealt (a boiling pass). RESULTS_PLACEHOLDER
By eye (drift-compensated crops, the half-res mp4s) the same strokes ride the same rock from frame to frame; new
strokes only fade in between old ones as the camera nears. The named fallback (ink on the far ranges only, the near
terrain in a parchment grade) is NOT needed.

**Remaining weaknesses (honest).**
* The strokes are single straight-in-plan dashes bent by the local fall line; on the biggest near faces the hatching is
  dense and even ("grain" rather than an engraver's varied line). More variation (stroke-length rhythm, occasional
  double lines, looser far work) would make it more hand-made.
* Outlines come from the depth buffer at supersampled resolution: fractal skylines are drawn faithfully, which is
  busy on the far horizon, and a 1-px silhouette crawl remains (it reads as the pen line redrawn).
* The cloud sea's contour lines are iso-heights of a cloud that drifts (7 m/s at its largest scale), so they flow
  slowly; that is the sea moving, not boiling, but the metric counts it (cloud numbers above).
* C24's wash takes its colour from the dawn plate (`dawn.py`, no eagles), so the palette is the dawn's: lilac and
  gold, with the sun and the sun's path on the cloud gilded. The peaks stay ink (they face away from the sun); if the
  producers want colour on the land too, the rule is one line (a cool wash on the shadow side late in the shot).
* C16 uses an illustrator's key light from the page's left (the moon is behind this camera and would leave every face
  front-lit). C only; A and B keep the moon for R2. C16 uses RUN-B's summit dressing (keeper.CR_B, keeper.BEACON), so
  the shared reveal shows one summit in every cut; HEROINE composites her figure at keeper.BEACON as for B.
* Near fires are true size and gilded in the flame; far ones are drawn 5x (an illustrator's licence) with a gold
  aureole so a fire 3-8 km off still reads as a fire, not a point.

**RENDER_SPECs (cloud; the director launches; job files written, NOT launched).** One driver renders a final frame
from scratch (AOVs -> ink -> 1920x804; no caches): `shots/run/ink_final.py`. Fresh box: the Run's usual setup
(numpy, numba, scipy, opencv-python-headless); files needed: `shots/run/{ink_final,render_ink,ink_aov,inkpass,keeper,
world,run,rcam,pipe,beacons,fire2,dawn}.py`, `shots/run/{summits,beacons}.npy`, `shots/montage/` (s1_peak, mt, common),
`lib/look.py`. First frame per worker compiles numba (~2-3 min; each job warms the cache first). ~1.2 GB RAM per
worker at full res (use --procs 3 on boxes under 12 GB). Deterministic apart from the PNG dither.

| shot | command (from `the-long-dawn/`) | frames -> folder | cut C frames | jobs | est. cost |
|---|---|---|---|---|---|
| C16 THE REVEAL | `python3 shots/run/ink_final.py --shot reveal --range 0-239 --procs 4 --skip` | 0-239 -> `renders/runC_reveal/` | 3600-3839 | `cloud/jobs/runC_reveal_{a..d}.json` | COST_REVEAL |
| C17 THE LIVING INK RUN | `python3 shots/run/ink_final.py --shot scroll --range 0-319 --procs 4 --skip` | 0-319 -> `renders/runC_scroll/` | 3840-4159 | `cloud/jobs/runC_scroll_{a..d}.json` | COST_SCROLL |
| C24 THE ILLUMINATION | `python3 shots/run/ink_final.py --shot illum --range 2398-2877 --procs 4 --skip` | 2398-2877 -> `renders/runC_illum/` (local = frame - 2398) | 5680-6159 | `cloud/jobs/runC_illum_{a..f}.json` | COST_ILLUM |

C24 renders dawn.py's colour plate in the same worker (half scale: it only feeds a blurred wash), with its smoke and
WITHOUT the eagles; the sun breaks at local 1.5 (dawn 2399.5); the wash completes as the line settles home (local 320).
C17's seven beacons are fixed by `render_ink.scroll_beacons()` (summits.npy; deterministic); retime by editing
`SC_CATCH` only. Tests and caches: `renders/run_c_tests/` (half-res mp4s `scroll_232-255_half.mp4`,
`reveal_108-131_half.mp4`; `boil_check_*.txt`; `aov/*.npz` caches are disposable). Review sheet:
`_local_logs/review/runC_ink.jpg`.

**Code (all new files; world.py, render_run.py, beacons.py, dawn.py and keeper.py untouched).** `ink_aov.py` (AOV
kernel = world.shade's surface logic writing 16 channels; `grad_at`), `render_ink.py` (cameras, the scroll's beacons,
AOV + fire/smoke layers, caches), `inkpass.py` (the drawing), `ink_final.py` (production driver), `ink_check.py` (the
boiling metric).

---

# >>> RUN-B STATE AT PAUSE (27 Sep ~11:25Z, director's usage pause; RESUME ~15:00Z) <<<
**Nothing is rendering. No cloud job is ready** (the three pre-H5 jobs were deleted as stale; none written for the new look).
**Done since handoff.** keeper.py + the pre-H5 baseline committed (f37cf60) for RUN-C's C16 (keeper.CR_B/BEACON stay in
world.py's row format). NEW `bworld.py` = B's own landform + kernels (world.py untouched), committed as WIP:
* terrain `h_rock`: `field()` = soft-ridged multifractal (rounded crests, low persistence, S=7.6 km, warped) with a
  convex top map (u = hr-0.63 -> CLOUD_Y + 3291(1-e^-1.5u)/1.5): ~25% of land above the cloud, broad rounded tops,
  cloud-filled basins between ranges (layering), a cloud bay NNE of her; rows (CR_B): her massif = dome (R 2.7 km)
  whose flanks carry the same soft-ridged spurs (none within 420 m of the top), the NE ridge (two round-crested
  `bridge` rows), the sunrise shoulder (dome, az 124, 3 km, TOP_Y+18), `knoll()` = her stony rounded summit boss
  (TOP_Y - 0.03 d^1.62, boulders at the rim). Row types: 0 dome, 1 ridge, 2 cirque carve (unused now), 3 uplift
  (unused: uplifting a flat floor made smooth blobs).
* `march`, `gbuffer`/`build` (sun clearance channel + K moon soft-visibility channels for the vigil's moving moon),
  snow from the form's own slope with broad patches (no fall-line couloirs = no vertical streaks; strata ledges were
  tried and REMOVED: they read as zebra dashes), stony knoll (snow only in hollows within 22-40 m of the top).
* skies: `dusk_sky` with a SOFT Earth-shadow edge (SK[18] half-width, ~3 deg) + broad Belt of Venus (SK[16] ~6 deg);
  `dawn_sky` (smaller aureole, disc tint) for a tamed sun; `night_sky` (moon halo/disc, east-greying term).
* `shade(G, LP, SKY, amb, fogp, cam_y, out)`: LP layout in the file (NLP 48): sun via clearance, moon via vis
  channels (LP 32-34 mix two), fire point light (LP 36-43), fogp[12:15] = in-scatter direction.
**Lessons (don't repeat).** Pure domes + bowl cirques read as meringues / craters / cow-spots; uplifts over a flat
floor read as bubbles; the field (soft-ridged) is what reads as real mountains. At a low sun, round cirques read as
holes. The cloud sea's fine billows at a grazing low sun read as wrinkled paper.
**NEXT (in order).** (1) Re-run the greybox views: `python ~/mishamisha/_local_logs/runB/bviews_greybox.py all 0.5` ( cameras DUSK = az 212 from her top, 12 km, y 650, hfov 38, look_off -5, pitch -1.2; VIGIL = 44 m
SSW, +1.5 m, yaw 31, pitch -1.2, hfov 40; sun 3.0/0.5 deg az 205). Check her massif + shoulder now read as ridged
massifs, not blobs; tune `detail` (spur relief) and `keep`. (2) Port `dusk.py` motion (DuskShot) to bworld: SK[18]
soft seam, rose-on-snow LP, clamp_to_her + six-peak schedule on the new G-buffer; test 8 frames at quarter scale;
write `cloud/jobs/dusk_b.json` -> renders/dusk_B (frames 0-639). (3) Port `handback_motion.py` to bworld (crane =
per-frame build+shade with night LP; settled = relight), compose the hand-off (bar 63: she presses the fire-steel
into the child's palm; HEROINE cloud lane builds the hands), tamed starburst (no streak, disc I low, bloom thr high),
no poster framing -> jobs -> renders/handback_B (3840-5199). (4) THE VIGIL (bars 18/20-48) on bworld: locked frame,
continuous moon path via vis channels, weather as continuous layers, figures 60-120 px (hood + red woven shawl,
never front-lit), stone cairn -> renders/vigil_B; the REVEAL -> renders/reveal_B. Delivery naming per the director:
renders/dusk_B, reveal_B, vigil_B, handback_B in B frame numbering.

# >>> RUN-B STATE AT HANDOFF (new RUN-B agent, 27 Sep ~10:10Z) <<<
**Inherited (all pre-H5; nothing launched).** Code is UNTRACKED in git until this agent commits it: `keeper.py` (lifetime
greybox + B set), `dusk.py` (R8 still + `DuskShot` motion via relight), `relight.py` (G-buffer relight: per-pixel sun
clearance), `handback.py` (R11 still), `handback_motion.py` (R11 crane + settled, figures = mt puppets). Cloud jobs
`cloud/jobs/dusk_b.json`, `handback_b_crane.json`, `handback_b_settled.json` were WRITTEN BUT NEVER LAUNCHED and are
superseded (they render the pre-H5 look: s1 needle ranges, Toblerone peak, pancake cairn, hard shadow seam).
Test renders on disk: `renders/run_b_tests/{dusk_motion (640 f, quarter), hbm_t1..t4, dusk_motion_t1..t4, lifetime*}`.
**What H5 requires of RUN-B (BIBLE_V3 top block §1, §2, §5, §6 + B DECISION):** B = THE VIGIL (one night, same 68 bars,
no M2/M3 cutaways, far answers only as pinpricks in her frame). B's OWN landform: old, broad, glaciated massifs, a
rounded summit above the cloud sea, far ranges as layered silhouettes (no needle ranges shared with A). DUSK: no
perfect triangle, rose on snow, soft Belt of Venus, no vertical streaks, no hard Earth-shadow seam. VIGIL: one
continuously moving sky (no blinking plates; 6-10 f dissolves if ever needed; grey dawn graded down), her figure
60-120 px in hood/cowl + red woven shawl, never front-lit; cairn of visible irregular stones, ragged top. HAND-BACK:
real hand-off on bar 63 (she presses her fire-steel into the child's palm; the cloud HEROINE lane builds the hands,
RUN-B composes), tamed sun starburst, no motivational-poster composition. PROTECT the far beacons in the cloud sea.
**Plan (this agent):** (1) `bworld.py` = B's own terrain + kernels (world.py untouched); (2) DUSK + HAND-BACK re-built
on it -> cloud jobs; (3) THE VIGIL locked frame with a continuous sky -> cloud job; (4) CLIMB wide + REVEAL (R2-B).
**Live status:** see the RUN-B block below; this line is updated as work lands.
* 10:25Z commit f37cf60: keeper.py (+ dusk/relight/handback/handback_motion) committed as the pre-H5 baseline, so RUN-C's
  C16 (`render_ink.py` imports keeper.CR_B / keeper.BEACON, world.py row format) can run on the cloud. keeper.CR_B keeps
  that format; B's new landform is `bworld.py` (its own CR_B). The three stale pre-H5 jobs (dusk_b, handback_b_*) are deleted.

# >>> RUN-B (B . THE KEEPER: R8 DUSK, R9 LIFETIME, R11 HAND-BACK) - report 27 Sep ~02:40Z <<<
## Report
**What it is.** Risk test #1 (BIBLE_V3 s10): THE LIFETIME (R9) as a 45 s quarter-scale greybox in one locked frame on
her summit (plates as stills, puppets for figures, no audio), cut one bar a night to the chaconne; plus a DUSK (R8)
still and a HAND-BACK (R11, now showpiece #1) still, all in the Run world on B's own set. Review sheets
(`_local_logs/review/`): `runB_lifetime.jpg` (v2), `runB_lifetime_v1.jpg`, `runB_lifetime_v2_strip.jpg` (3 fps),
`runB_dusk_handback.jpg`. Movies: `renders/run_b_tests/runB_lifetime_greybox.mp4` (v2), `..._v1.mp4`. Stills (half
scale): `renders/run_b_tests/dusk/dusk_-0.72_0.50.png` (+ four quarter-scale sun states), `renders/run_b_tests/
handback/handback_0.50.png`.
**VERDICT: FAIL. It plays as a slideshow, so B falls back to THE VIGIL (one night in the same set, Q2).** Two
builds were each read cold by a fresh critic that saw only the sheet and a filmstrip. Both said "slideshow"; neither
answered "why does she keep going back?" with "she promised" or anything like it.
* v1 (10 nights, a bar each): nights passed but years did not. Every night was the same ~3 s loop re-lit by a
  whole-world preset, each bright dawn landed like a slide advancing, and "she seems to live there".
* v2 (the critic's own fix applied): the cairn as the clock, her walk up the arete each dusk, years quickening from
  80 to 32 frames a night, dimmer dawns, and M2 cut before the torch-thrust. Still a slideshow: the look presets
  swap all at once (the rock turns snow-streaked between nights), the fast stretch strobes, figures ghost in, and
  the keeper never visibly ages. For whom she keeps the fire stays unclear.
* The critic's structural note for the showrunner is one line of sight: the child points at her spark, then the
  grown man stands on the same ledge looking at the same spark.
My own read agrees. The single frame works: a tiny fire on a summit in a vast night, and the far answers, the thread
of torches and the village glow filling in are the strongest pictures. A night pressed into 1-3 s cannot carry a
vow, though. What reads is the setup and the payoff, which suits one real-time night.
**Laugh / doll risks found (fix wherever these assets are reused).**
* The v2 cairn, a smooth dark rounded column, read as "a giant thumb (or worse)". It is fixed in `keeper.
  stone_cairn`: a broad heap whose outline is the stones, grown self-similarly with a steady yearly increase.
* M2 (REUSE of the s1 shepherd) reads as a game loading-screen pose, and its torch-thrust reads as LotR's "the
  beacons are lit!".
* M3's greybox parent and child read as clip-art.
* The moonlit wides read as a terrain-generator demo (evenly spaced ridged spikes, snow that follows the slope).
**What carries into THE VIGIL (all in `keeper.py`).**
* The set: her shelf, the NE arete, the cloud bay.
* The moon, fog and grey-dawn plates, with a real-time wheel of the sky.
* The far pinpricks answering one by one after hers catches, which already happens within one night.
* Travellers coming up for flame, then the thread of torches going home down the arete at dawn.
* The village glow under the cloud, and one stone on an old cairn at the grey dawn.
**DUSK (showpiece #5), still: works.** The sun sets behind the camera (SSW). Each point is lit through its own
grazing path, so the shadow line climbs every face and only the highest tips keep the last red points (see the
four-state strip). The cloud sea sits in the Earth's shadow under the Belt of Venus. Her summit is the tallest lit
pyramid, right of centre, and keeps the last light.
Weaknesses:
* The lit faces go orange (bare rock) rather than rose on snow.
* The near peaks' ridged-noise facets still read slightly CG at half scale, despite the added fall-line gullies.
* The far skyline is a dark serrated line.
**THE HAND-BACK (showpiece #1), still: the strongest of the three.** The sixtieth dawn, seen from behind her. The
sun breaks at the right edge of the 357 m shoulder, 2.3 km ESE. The far beacons already in sun have paled to specks
(each by its own sun clearance, the DUSK march). Those still in the shoulder's shadow burn, and hers burns last.
Her old figure and the child stand by the fire, backlit, with the sixty-stone cairn beside them.
Weaknesses:
* Her stoop and stick don't read yet.
* The scarf isn't lit (the sun has not reached it in this moment: correct, but it must be in the motion).
* The paling order is shown only as one moment.
* The opening crane (R10 folded in) is not built.
**Continuity (flag for HEROINE/RUN-A).** B's summit is a `crag` shelf 15 m above `S1.summit()`, with a new NE
arete. H1/R2 show HILLS' snow dome, so either R2's B grade and H5 adopt this set, or B's shots adopt the dome.

## RENDER_SPEC
* **THE LIFETIME: do not render** (failed its test). THE VIGIL needs a timeline first, a new `NIGHTS` of one night in
  `keeper.py`. Its render then goes on one box, the frames' box rendering its own plates (about 300 MB of npz; never
  ship them through git):
  `python3 shots/run/keeper.py --plates --scale 1.0 --ss 1.5` then
  `python3 shots/run/keeper.py --range 0-<N> --scale 1.0 --out renders/vigil_B --procs 4 --skip`
  The job json is as in `cloud/jobs/dawn_c_a.json` with those two lines, "out_dir": "renders/vigil_B" and
  "ship": "jpg". Cost: plates ~5 min each on 4 threads, frames ~2-5 s each (layers only), so about an hour on one box.
* **DUSK and THE HAND-BACK, full-scale stills** (local, one at a time, ~4-8 min each; no cloud needed):
  `python3 shots/run/dusk.py --still -0.72 --scale 1.0 --ss 1.5 --out dusk` and
  `python3 shots/run/handback.py --scale 1.0 --ss 1.5 --out handback`.
  Their motion is the next build. Both are locked frames whose light moves, so march each pixel's sun clearance to
  the fixed sun azimuth once, then relight every frame cheaply. THE HAND-BACK's opening crane (R10) is the only
  per-frame march, at about dawn_C's cost.

## State
**Code (world.py untouched).** `keeper.py`: B's set = a `crag` shelf on her horn (top 301.5 m, 10 m W of
`S1.summit()`, burying the rock fin; s_lo 2.2 so it only shapes the top ~50 m) + a long NE arete (two `ridge` rows,
slope ~0.13, to the cloud at 2.7 km) that carries the thread of torches. Lifetime camera (v2): 44 m SSW of the top,
1.5 m over the shelf, yaw 31, pitch -2.6, hfov 44. Plates cached in `renders/run_b_tests/cache/` (moon = s1's,
stars, storm, fog, grey dawn; each + a unit fire-light difference plate). Per-frame layers: fire2 flame + firelight,
the counting cairn (`stone_cairn(n)`: a stony, round-shouldered pile, 64 stones), her figure at every age
(mt/figure puppets, ghosted over three moments), the child (year 60), travellers + the thread of torches down the
arete, far pinpricks (124 visible summits; the first answer year 8, all by 60), village glow under the cloud,
stars, driving snow, dissolves; cutaways M3 (greybox) and M2 (REUSE montage_v2 1440-1475). Timeline = `NIGHTS`.
`dusk.py`: R8 shader (grazing-path sun colour per point from a long clearance march with curvature; art-directed
small sun radius so the shadow line reads as it climbs; Earth's shadow + Belt of Venus sky); camera SSW 5.2 km,
430 m, looking NNE. `handback.py`: R11 key still (dawn.py's dawn_shade + sky, sun at the 357 m shoulder's edge,
beacons paled by their own clearance, figures from behind).
**Re-render.** Lifetime: `_local_logs/runB/make_lifetime.sh` (plates first if `CR_B` or the camera changed:
`python shots/run/keeper.py --plates`). Stills: `dusk.py --stills` / `--still E --scale 0.5`, `handback.py --scale 0.5`.
**Next.** Per the verdict: THE VIGIL (one night in this set) or the lifetime with H6's rigs; DUSK and HAND-BACK motion
drivers (cached clearance relight); more air in B's plates (the terrain-demo read).

# RENDER_SPECs (cloud) — run everything from `the-long-dawn/`

Fresh Linux x86 box (tested here: Python 3.12, numpy 2.5.3, numba 0.67.0, opencv 4.10, scipy 1.18.1):
```
python3 -m venv ~/ld && . ~/ld/bin/activate && pip install numpy numba scipy opencv-python-headless
```
No env vars needed (the drivers set NUMBA_NUM_THREADS=1 per worker; `--procs N` = N workers). First frame of a
fresh box compiles numba kernels (~1-2 min, cached in `__pycache__`). Committed data that MUST be present (do not
regenerate): `shots/run/summits.npy`, `shots/run/beacons.npy`. All drivers take `--skip` (resume a block).

## DAWN_C — v2 frames 2400–2655 → `renders/dawn_C/f_%05d.png` (cut C only)  **READY**
```
python shots/run/dawn.py --range 2400-2655 --procs 4 --skip
```
Split suggestion (4 machines): 2400–2463 / 2464–2527 / 2528–2591 / 2592–2655. Depends on shots/run/{dawn,world,run,
rcam,pipe}.py + beacons.npy (smoke sources). Deterministic. Cost ≈ the Run's far-terrain frames (~2–4 min/frame
single-threaded on this Mac; no near terrain).

## s1 re-render with the Run's fire — v2/src 1440–1519 → `renders/montage_v2/f_%05d.png`  **READY**
```
python shots/run/s1_v2.py --range 1440-1519 --procs 4 --skip
```
Runs shots/montage/s1_peak.py unchanged with fire2's flame (orange→yellow→white, soot absorption), spark ramp
and glow colour patched in. ~72 s/frame on this Mac with 2 threads at full res (s1's own cost).

## THE BEACON RUN — v2 frames 1520–1679 → `renders/run_v2/` (look FINAL; see the RUN spec further down)
Local fallback 1660–1679 already rendered here.

## World settings for HEROINE (to match the FIRST BEACON reveal's far ranges)
The ranges ARE s1's (`shots/montage/s1_peak.py`); the Run and Dawn only add foothills near the shepherd.
* Coordinates: metres, y up, +z = s1's view axis, the shepherd's summit at the origin (y≈0). The heroine's beacon
  massif summit = `S1.summit()` ≈ (−5450, 286, 31997): 32 km from the shepherd, bearing −10° from s1's axis.
  From her summit the shepherd is at bearing ≈ +170° (looking back down −z).
* Ranges `S1.h_far`: ridged multifractal (seed 21) at 3400 m, octaves by pixel footprint (≤11), domain-warped by
  fbm at 7000 m (seeds 61/62, ×0.55); massif envelope fbm 11000 m (seed 63): prom = 0.55 + 0.45·smoothstep(−0.35,
  0.35, m); h = −1800 + 2100·prom·r^1.45; + her massif (Gaussian, σ 2600 m, +HB+450 at XB, ZB); a corridor term
  thins peaks along s1's sight line. Earth curvature d²/2R everywhere.
* Cloud sea `S1.h_cloud`: −650 m + 150·fbm(2400 m, seed 33) + 55·(1−|fbm(520 m, seed 34)|) + 10·fbm(120 m, seed 35).
* Snow (far): snow = smoothstep(0.38, 0.58, n_y(45 m-smoothed normal) + 0.18·noise(380 m, 73) + 0.06·noise(95 m, 74));
  rock albedo (0.055, 0.056, 0.062), snow (0.80, 0.86, 0.98). Cloud: wrap light, forward scatter 1+1.6·cos^4,
  peak shadows (floor 0.35), trough darkening 0.55.
* Moon: dir normalize(−0.80, 0.36, 0.48) (az −59°, el 21°), colour #9DB4D9, intensity 0.55; ambient #27335E×0.35.
  Sky: zenith #070B1C, horizon #2A3866, horizon glow 0.25, moon halo 0.025/0.22 + 0.012/0.7; stars
  `SK.make_stars(14000, 101, lum_scale=7)`. Fog: 5e-5 (scale height 1500 m) + cloud-top mist 2.2e-4 (140 m above
  −650), colour #2E3D66×0.95, forward boost 1.5 toward the moon. Finish: exposure 1.0, bloom 0.07 @ 0.8, vignette 0.25.
* Easiest route: import `shots/run/world.py` (`night_light()`, `march()`, `shade()`) with an empty crag table
  (`np.zeros((0, world.NCR))`) and a camera at her summit — that is s1's world (checked visually against s1 frame
  1500 with `shots/run/test_s1match.py`; since then shade() adds a small bounce fill (Q[16]=0.10) and close-range
  snow detail, both invisible at the reveal's distances).

# RUN department (v2) — THE BEACON RUN + DAWN_C

## RENDER_SPEC — THE BEACON RUN (v2 frames 1520–1679 → `renders/run_v2/f_%05d.png`)

Look is FINAL. Everything the render needs is in `shots/run/` (+ `shots/montage/` toolkit, `lib/look.py`).
Cached data that MUST be present (committed, not regenerated): `shots/run/summits.npy`, `shots/run/beacons.npy`.

Fresh Linux x86 box:
```
python3 -m venv ~/ld && . ~/ld/bin/activate
pip install numpy numba scipy opencv-python-headless    # tested: py3.12, numpy 2.5.3, numba 0.67.0, cv2 4.10, scipy 1.18.1
cd the-long-dawn
# optional warm-up (compiles numba kernels into __pycache__, ~2 min):
NUMBA_NUM_THREADS=4 python shots/run/render_run.py --frames 1600 --scale 0.25 --out tests/warm --threads 4
# render a block (N workers x 1 numba thread each; ~400 MB RAM per worker):
python shots/run/render_run.py --range 1520-1554 --procs 4 --skip
```
Output: `renders/run_v2/f_01520.png` … (v2 numbering, 1920×804, final look incl. motion blur). `--skip`
skips frames already on disk, so a block can be resumed. Deterministic apart from the PNG dither.

Split (claimed):
| frames | where |
|---|---|
| 1520–1554 | cloud A |
| 1555–1589 | cloud B |
| 1590–1624 | cloud C |
| 1625–1659 | cloud D |
| 1660–1679 | LOCAL (this Mac, 3 workers) |

Cost: ~6–7 min/frame single-threaded on this (loaded) Mac at full res; near-terrain frames (1520–1600) are
the heaviest.

---

## What was built
One world, flown through. `world.py` extends s1's heightfield world (its `h_far` ranges, `h_cloud`
cloud sea and `h_near` summit are imported unchanged, same seeds, same moon, sky, fog, snow rule) with:
* **the shepherd's foothills** (`near_range`): s1's ridged-multifractal recipe at 420 m scale, envelope
  held ≥12 m under s1's summit-lip sight line inside s1's frustum (verified: max −12 m) so s1 never saw
  them; a spine crest along the flight line;
* a **faceted-horn primitive** (`crag`) for the pyre tower (3 irregular facets, ribs/gullies from ridged
  noise, weak strata, broken summit slabs), smooth-maxed into the range;
* close-range detail: multi-scale snow (s1's rule far away; near: ledges hold snow, ribs shed it), rock
  albedo variation, cleared rock platforms under beacons, bounce fill from the snowfields/cloud.

`rcam.py`: the column-coherent marcher needs no roll and lens-shift tilt, so each frame renders a lens-shift
SOURCE camera covering the target frustum and warps it (an exact rotation homography) to the TARGET glider
camera with true pitch and roll. Fire, smoke, cairns and stars are drawn in source space, so they roll with
the horizon for free. Then depth-based vector motion blur (McGuire 2012 tile/neighbour-max reconstruction,
126° shutter) in target space. Sparks are streaked through the moving camera: each shutter sample is
projected with the camera at that instant.

`run.py`: flight dynamics. Speed/heading keys → ground track (PCHIP, no stops at keys); altitude =
terrain-following upper envelope with limited vertical acceleration + 24 m clearance; then a zoom climb.
Bank = coordinated-turn bank from the turn rate, rolled in 5 frames ahead through a critically damped
filter. Operator's look = critically damped follow of heading + a clamped lean toward the pyre.

`beacons.py`: the chain. Beats 1540 (foothill summit, 1 km), 1560 (spine summit, 390 m), 1580 THE PYRE,
1600 (s1's great pointed peak, 7.4 km, the peak the audience saw on the right in s1), 1620/1640/1660
(islands 4–9 km, alternating); small far ones between (1570–1670, wide; SFX `run_far_*`); from 1620/1640/1660
the signal races on as chains of links toward the horizon (25→37 km, each link sooner than the last).
The heroine's beacon (lit 1360) burns on the horizon where s1 showed it.

`fire2.py`: montage's bonfire with the critic's fix. The ramp is orange → yellow → white with no crimson end,
because a crimson tip over the blue sky reads magenta. The flame body is also partly opaque (soot), so no sky
shows through the tips. Air glow is yellow-orange and tight. Distant fires are a warm hot point plus a soft
orange aura.

The pyre (`render_run.py`): the pyre is a big cairn with an iron basket on the tower 14 m left of the track.
It ignites at 1580 at 44 m, with the catch one frame before and the whoosh overshoot peaking at 1585. Six tongues
are separated by dark gaps and lean in a 6 m/s summit gale. The spark fountain streaks through the moving camera,
the smoke is lit from below, and a 48-unit point light washes the tower. We pass it at 13 m at ~1592, then
bank left into a climbing turn toward the moon side of the sky.

## DAWN_C (`dawn.py`)
Same world, the morning after, from where the Run ended (camera ≈ (60, 150→170, 480), yaw −19°, hfov 46°, a slow
20 m crane rise easing to stillness). The sun (az −30°, x≈0.27 of frame) touches the skyline at 2399.5, clears
it by ~2432 and climbs slowly (+1.5° over the shot). "The flood": the terrain is lit from a direction lifted
+2.3° over 2400–2470, with sunlight ramping 0.3→1 over 2397–2425, so the eastern ranges' long shadows sweep
back across the cloud sea while the disc itself moves gently (no time-lapse look). Dawn sky: a tight gold aureole,
a gold band on the skyline toward the sun, clear blue within a few degrees; the sun is a clean limb-darkened disc
+ bloom (no starburst), streak 0.004. Warm haze only near the sun, blue-violet air elsewhere; blue shadows.
Thin smoke rises from every summit that held a beacon (beacons.npy + the pyre), lit by the low sun. Six eagles,
one loose irregular pass, gliding (two give a couple of slow beats), from 2476: they enter at the right edge at
76–94 m (wingspan 55–69 px ≤ 4% of width), fly away toward a point right of and above the sun (never across the
disc), shrinking to ~27 px, always in the upper band (never in the text band 2512–2600).

## Re-render / tests
```
python shots/run/render_run.py --frames 1520,1580,1679 --scale 0.4 --out tests/t --procs 3
python shots/run/render_run.py --range 1520-1679 --procs 3 --skip        # final
python shots/run/plan.py                                                  # top-down plan + camera table
python shots/run/catalog.py && rm shots/run/beacons.npy                   # ONLY to rebuild the chain
```
