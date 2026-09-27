# >>> PAUSED 27 Sep ~11:55Z (director: usage at the window's end). RESUME 15:00Z: the EDGE/BRINK pass <<<

## DIRECTOR'S NOTES (resume_notes/embers.txt, verbatim) -- apply these first
Director's review of the EDGE/BRINK check stills (review/embers_A3_check/edge/, f8a7a02). HOLD both jobs for one more pass:
(1) GEOGRAPHY FIRST. The audience must see: (a) the ground falling away into a CRATER OF FIRE around the fire, a deep glowing bowl of molten light with heat shimmer and embers, never a flat region with a thin glowing outline (it reads as a puddle or a map outline); (b) the towers standing on its LIP and leaning in, silhouetted against the pit's glare; (c) the two giants on opposite sides. Pull the orbit out to a 3/4 view from just above rim height, looking across the bowl. Keep the near rim dark in the lower third for T7/T8, with the pit's glow in the middle band.
(2) THE GILDING must read: on each surge, bright liquid gold runs down the nearest faces while the farthest go dark. The incentive has to be visible.
(3) From bar 26, the rim crumbles under the gilded ones: crust chunks break off into the glow. Make the lip a physical edge (breaking crust, glowing seams), not a graphic line.
(4) The crown fall (about 2480-2520) must be a HERO moment: bigger in frame, a readable crenellated crown breaking off and tumbling into the glare, visible for about 1 s. Then the camera follows it over the rim into white on 2640.
(5) The brink's updraft (fire physics, no vortex) is the right direction. Keep it.
f2440 (looking up at the flaring fire, sparks falling) is the strongest of the set; build the rest to that intensity.

## EXACT STATE AT PAUSE
* Cloud, APPROVED and running (launched by the director): embers_A3_a5 (A5 1040-1439), embers_A3_a6 (A6 1440-1839
  MAIN), embers_A3_alt_a (A6 1440-1839 ALT). Their code is frozen at 65985e3 in the jobs' clones; later commits touch
  only frames >= 1840 (edge.py, the EDGE/BRINK camera, the crater) EXCEPT the Smoke/crust edits, which were already in
  65985e3. Frames arrive via the importer into renders/embers_A3 and renders/embers_A3_alt_codedtowers.
* HELD by the director: cloud/jobs/embers_A3_edge.json and embers_A3_alt_edge.json (1840-2639). Do not launch; they
  need the pass below first.
* Code at HEAD (f8a7a02 + this commit): THE EDGE crater solid (occluder hook), torn rim (`edge.rim_r`), hot walls,
  lit plain; orbit r46 y30 ty-12 hf62; BRINK: Updraft pillar + StripEmbers (2400-2440), rim lurch + breaking lip at
  2440, crowns raining sparks, FALL_TOWER=7 breaks at 2480 with the camera following (`FallingCrown.centre`,
  `CrownTrail`), over the rim to white at 2640.
* Check stills: the-long-dawn/review/embers_A3_check/{v3 (A6, approved), edge (EDGE/BRINK, held)}.
* Contact sheet of A9/A10/A16/A17 as they stand: FINISHED at scratchpad e3/v1.jpg (0.3 scale), not yet reviewed.

## NEXT STEPS (the EDGE/BRINK pass, in the director's order)
1. GEOGRAPHY: the crater must read as a deep glowing BOWL of molten light (heat shimmer, embers rising out of it),
   not a flat area with an outline. Ideas: widen/deepen the visible wall (WALL 5 -> 9, a sloped bowl, not a shaft),
   brighter wall with a strong gradient to a white-gold molten floor, embers + heat-haze particles rising out of the
   pit, the plain's lip region breaking into glowing crust. Orbit: 3/4 view from just above rim height looking ACROSS
   the bowl (e.g. r ~40-45, y ~ -4..0, ty ~ -12, hf ~60), towers on the far lip silhouetted against the glare, near
   rim dark in the lower third, the two giants on opposite sides (az ~2.7 rel. ALPHA_C puts them left/right).
2. GILDING: make `_gold_runs` visibly bright liquid gold on each surge on the nearest faces (check GILD_W and the
   surge pulse: SCHED.gild returns g; scene_b._gold_runs), the farthest going dark.
3. RIM CRUMBLE from bar 26 (2000): crust chunks (bigger than the debris sparks) breaking off the lip into the glow;
   the lip as a physical edge (crust + glowing seams), not a line (Crater.l_p is a thin line now).
4. HERO CROWN 2480-2520: bigger in frame (tighter lens / closer camera on FALL_TOWER=7), a readable crenellated crown
   tumbling ~1 s into the glare; then follow it over the rim into white on 2640.
5. Keep the updraft (director: right direction). f2440 is the intensity target.
Then new check stills -> director -> relaunch the two held jobs. After that: A9/A10/A16/A17 review (e3/v1.jpg), E1
glyphs retime, then C (canonical Ring inscription in assets/ring/, forging, race, Eye, grasp).

# >>> EMBERS-v3 #2 -- STATE 27 Sep ~11:35Z (read this first) <<<

**Cloud (approved + launched by the director):** `embers_A3_a5` (A5 1040-1439), `embers_A3_a6` (A6 1440-1839 MAIN),
`embers_A3_alt_a` (A6 1440-1839 ALT -> renders/embers_A3_alt_codedtowers). All at 65985e3. Frames arrive through the
importer into renders/embers_A3 (+ _alt_codedtowers) in A's cut numbering. Check stills: the-long-dawn/review/
embers_A3_check/ (v1 first pass, v3 = the approved A6 look). Director's verdicts: A5 calm fire + promise right; A6
charcoal crust + banded forge-stacks right; the low wide shot up at the black giants is "the scale and dread".

**Done (all committed):** H5 fire (one tongue, inner embers, no starfield/streak, ~36% frame height, x2 through A6),
H5 towers (squared/crenellated forge family, no lips/bowls/bulbs/galleries/flares; iron banding, soot, glowing
throats; charcoal crust with the fire in seams/joints/edges), A6 cameras (fly-in; outside, low, wide up at the giants
growing out of the top; cut on bar 23 to behind giant 2 with glare), lit forge-smoke canopy (Smoke broad term on A3),
giants black above the ring with shutters shut. BRINK: the vortex is REMOVED from A (a3 emit); `edge.Updraft` (a
pillar of flame + torn-off embers, 2400-2470 roar, subsides by the tip) + StripEmbers dragged in and up (no spiral).

**11:40Z: THE EDGE now reads** (scratchpad e3/b4.jpg): the crater joins the occluder (`edge._extra_occluders`
via `SCHED.extra_occluders` in Towers.prepare; crater splats carry myid=CRATER_ID), an irregular torn rim
(`edge.rim_r`), hot walls lit from the molten floor, a dark plain whose edge glows; the orbit r 46 y 30 ty -12 looks
down into the pit with the fire in the upper frame.
**11:43Z: THE BRINK updraft reads** (e3/b6.jpg): embers torn off the tower crowns in bursts (StripEmbers x10) are
dragged into a golden column rising from the swollen fire (Updraft: 60k soft blobs rw 0.35-0.8, core-dense rr^1.3,
billow noise with dark gaps, hot at the root; x16). Still open: the rim collapse (2440) and the crown fall (2480,
FALL_TOWER=4) need a legibility check at 0.5 scale; the tip over the rim to white (2520-2640) reads (b4 2530-2610).
Then a full-res check still pair (2000 THE EDGE, 2450 THE BRINK) -> director -> cloud job 1840-2639 (+ ALT: the
giants appear in THE EDGE too, so the ALT job must cover 1840-2639 as well).
**11:51Z BRINK staging** (e3/b10.jpg): updraft 2400-2440 (camera up), down to the rim for its collapse at 2446 (the
gilded towers lurch, `edge._tower_lean`; the lip under them breaks and drops, `Crater._lip`), up to the crowns raining
stripped sparks (2470-2488), FALL_TOWER=7 (gilded, not a giant, on the far rim) breaks at 2480 and the camera
follows it down (`FallingCrown.centre`, CrownTrail embers), then over the rim to white. First pass: the crown is
small and quick in frame. Jobs written: cloud/jobs/embers_A3_edge.json (MAIN 1840-2639) and
embers_A3_alt_edge.json (ALT 1840-2639).

**NEXT (in order):**
1. (DONE 11:40Z, see above) THE EDGE crater must read as SOLID rock (now a translucent "glass cylinder": splats are additive and only tower
   crust feeds the occluder). Add a SCHED hook in Towers.prepare (e.g. `SCHED.extra_occluders(ctx)` -> P, N, A, ids)
   and feed it edge.Crater's wall (w_p, w_n) and plain (p_p, up normals) so the near wall hides the far wall and the
   plain hides what is under it. Then re-light: plain edge glow, wall gradient to the molten floor (already coded).
2. THE EDGE camera: one steady orbit near rim height across the pit (lower third = dark near ground). Lab frames:
   e3/ce1-3.jpg (best candidates r50 y14 ty-12 hf56 and r62 y24 ty-14 at az 2.7 rel. ALPHA_C: both giants flank).
   Current edge.camera: r 60 y 36 ty -16 (too high/close: near towers fill the frame).
3. BRINK look-dev: the Updraft pillar (e3/b3.jpg was the first pass: a thin dotty column) needs density/heat; the
   camera tilts up with it 2400-2466, back down to the rim for the crown fall (2480, FALL_TOWER=4), then over the rim
   into white on 2640. Check the rim collapse (Crater debris at 2440) reads.
4. A9 dead valley / A10 ember / A16-A17 (edge.py) review; E1 GLYPHS retime (A 560-1040); then C (canonical Ring
   inscription assets/ring/, Ring proportions outer R 2.25 x width; forging must not read as a halo; Eye and grasp
   without the galaxy spiral).
Tools (scratchpad e3/): camlab.py (one scene build, many cameras), hz.py, lineup2.py, rt.py, sheet.py.

# STATE AT HANDOFF (EMBERS-v3 #2 takes over, 27 Sep ~10:10Z)

The previous agent was cut off at ~07:27Z. Its render processes are dead. What exists:
* **Code (uncommitted until 10:15Z):** `a3.py` (A's own timeline + schedule, THE PROMISE valley, A5/A6 cameras),
  `edge.py` (THE EDGE crater, gilding, lean, the falling crown, BRINK vortex shape + ember stripping, dead-valley ash,
  the living ember, A16 ridge fires / shutters, A17 small lights), `c3.py` (C's FORGING + RACE schedule, gold rain;
  EYE and GRASP by warped v2 src time); hooks in scene_b / scene_c / tolkien / towers2 / variant / render / timeline.
* **Renders:** `renders/embers_C3_half/` = C3 at 0.5 scale, 1040-1679 + 1920-2079 + 2320-2479 (960 frames, made
  BEFORE the H5 calls: stale look, review only). `renders/embers_A3/` = cache only, no frames. `embers_v2`,
  `embers_v2_alt_codedtowers`, `embers_C` = the v2-numbered fixes (superseded by A3 / C3).
* **Look at handoff (0.3-scale tests in the scratchpad):** the fire still forks into two tongues (A3 1140, C3 1060);
  A6's cameras sit among the towers (frames full of close slabs, the ring and the giants don't read); THE EDGE is
  dim and illegible; THE BRINK and C's EYE / GRASP still have the blue-white galaxy spiral (the Winamp read); the
  promise reads (orchards, river, roofs) but the river is a clean ribbon; C's Ring reads as a halo over the towers.
* **H5 items to apply before any new render:** (1) the fire: one tapering tongue, no fork, inner embers + filaments,
  35-40% of frame height, ice-white / gold edge, no starfield or lens streak at ignition, the fire from below is a
  flame (not a slab); (2) towers: no platform ring / flared base, no cooling-tower flare, no bottle or beaker lips
  (square / crenellate), the round-capped stack off the glowing axis, no red pole, ember life on every fire side;
  (3) THE BRINK: an updraft column of stripped embers (1-2 s), not a vortex; rim gives way 2440, a gilded crown
  (never a giant's) falls 2480, over the rim into white by 2640; (4) C's Ring carries `shots/accord/ring.py`'s
  inscription (the canonical Blender ring's), and never reads as a halo or a ring-toss trophy.

# EMBERS v3 -- PART 2 (27 Sep, after the director's review): the locked beat sheets (IN PROGRESS)

Director (review of fire + towers): towers' fire-facing sides need real ember life (not flat black slabs); the
fire must own the frame (35-40% of frame height); and build A's v3 embers act to the LOCKED beat sheet (BIBLE_V3
"REVISION 1 . LOCKED BEAT SHEETS"; no crown in A, THE EDGE crater, the rim collapse, the dead valley in grey
promise-points) with the coded-pair ALT for tower shots, and C's pieces (Ring forged, gold rain, Eye onto nothing,
the claw that cannot hold) to C's sheet. Renders are in each cut's OWN frame numbers:
`render.py FRAMES --cut A3` -> renders/embers_A3 (ALT: --alt-towers -> renders/embers_A3_alt_codedtowers);
`--cut C3` -> renders/embers_C3. The v2-numbered folders (embers_v2, embers_C) are superseded.

* Code: `a3.py` (A's schedule A3Sched installed as scene_b.SCHED; the promise valley E3/E4; A5/A6 cameras;
  TimelineA3), `edge.py` (THE EDGE crater, gilding, lean, the falling crown, the BRINK vortex shape and ember
  stripping, the dead valley's ash, the living ember, A16 ridge fires / shutters / warm backs, A17 small lights and
  the walk down), `c3.py` (C's schedule for THE FORGING + THE RACE incl. gold rain; THE EYE and THE GRASP render
  the v2 src scene at warped times). scene_b/tolkien functions consult `SCHED` only when a v3 cut sets it; the v2
  src timeline is unchanged when it is None.
* Frames (A): A5 1040-1440 ignition + promise, A6 1440-1840 towers + two giants, A7 1840-2400 THE EDGE, A8
  2400-2640 THE BRINK (white on 2640), A9 2640-2800 dead valley, A10 2800-3120 the ember, A16 4400-4720 towers in
  the light, A17 4720-4880 the fire seen. (C): C6 1040-1440 forging, C7 1440-1680 race + gold rain, C9 1920-2080
  the Eye onto nothing, C11 2320-2480 the grasp that cannot hold.
* A uses the eight forges only (no skyline: the ring and the giants must read); C keeps the skyline, no giants.
* Tower geometry is cached (renders/embers_A3/cache/towers_<hash>.pkl, keyed by towers2.py + switches).
* This Mac is heavily loaded by other lanes (load ~150-200): a 0.3-scale test frame takes 5-15 s, a process's
  first frame several minutes.

# EMBERS v3 (EMBERS-v3 lane, BIBLE_V3 rev. 1), 27 Sep 2026 -- report

Director's order: (1) towers "TWO GIANTS, UNCODED" + the ALT switch, (3) the thinking fire (A only), then (2) the
grasp (C only, the claw that cannot hold), (4) C's Eye onto empty black, (5) C's Ring polish. All five are in code;
renders run in the background from `_local_logs/embers_v3_logs/queue_*.sh` (each writes `<name>.done`).
Review sheets: `_local_logs/review/embers_v3_{fire,towers,grasp,ring,eye}.jpg` (left the v2 frame, right v3; the
v2 frames are kept as JPEG q92 in `_local_logs/embers_v3_before/{embers_v2,embers_C}/` 480-1039).

**Renders.** A: 480-879 MAIN -> renders/embers_v2 (480-628 re-rendered with the final fire camera); ALT 540-879 ->
renders/embers_v2_alt_codedtowers, then `dedupe_alt.py` deletes every ALT frame equal to MAIN to the dither (only
the frames that differ stay; its log lists them). C: 960-1039 (grasp) then 562-879 (race: new towers, Ring, Eye) ->
renders/embers_C. Not re-rendered: A 960-1039 (A has no grasp now; the old frames are stale), C 480-561 (C no longer
uses the ignition), B (no embers act).

## (1) Towers: one family of forge-stacks, two giants (towers2.py, scene_b.Towers, variant.py, render.py)
* The eight nearest the fire are one invented family (`towers2.FORGES`): tall chimney, bottle kiln, bellows house
  (walls folded like a bellows, a burning rib on every fold), twin flues, blast furnace (gallery, bosh, a roaring
  tuyere ring), buttressed stack, crucible tower (a brimming iron bowl), telescoping flue. Brick courses, burning
  iron hoops, slot vents, corbelled lips, a roaring throat. No pagoda, obelisk, dome or flag (the v2 designs stay in
  the file, unused). Behind them a far skyline of ten more of the same family (`SKYLINE`: other seeds and girths,
  33-46 units out, 0.4 density, kept out of the camera's sector and out of the gap behind the fire).
* Giants (A only, `variant.giants()` = towers 2 and 6, opposite each other across the fire, left and right of it
  in the race camera): +2.0 per beat from 660 and a 1.8 lead on alternate beats (they leap-frog); heights at 800:
  82 / 73 vs 49-61; at 880: 102 / 96 vs 66-72 (built down to y=-52 so no base ever shows).
* A only (`variant.fire_side_only()`): every tower lit only on the face turned to the fire, black toward the others
  (crust, joints, windows, seams, edges x a soft terminator), a faint glare fringe on the backs.
* ALT (`render.py --alt-towers`, A only): the two giants dressed as `pagoda` and `obelisk`, everything else
  identical. C: the same forge family, no giants, no switch, self-lit as in v2, and (v3) every tower leans toward
  the Eye 826-862.
* Motion is v2's (same RNG stream); the other departments' walls/sparks/embers/smoke stay with the eight forges.

## (3) The thinking fire, A (scene_b: MIND_* constants, breath(), FLAME_CLOCK, _rim(), CAM_B 500-548)
* Mind palette, not blackbody: an ice-WHITE body (C_MIND_ICE = 45% mind_core + 55% mind_ice, never gas-blue), a
  white core, tongues white at the root and ice along their length, gold only on the silhouette (a view-dependent
  rim: the outer ~15% of the body's and tongues' left/right edges, a deeper gold than mind_gold, never orange).
  No blue base, no orange tips.
* The thought is visible: the body is dimmed (x0.18) and translucent, the filament tree 5x brighter with thicker
  threads, pulses on a slower clock, and on every breath one wave of light climbs from the core through the whole
  tree; no filament tangle at the base.
* Motion that is wrong for a flame: one even breath per bar (80 frames: radius +/-12%, light +/-28%, all three
  tongues rising and sinking together), tongue flicker rates x0.5, particles up the tongues at x0.55 speed,
  laminar tongues, no per-particle sparkle (frame-to-frame luminance change median 1.2%).
* Size and camera: back in (16-17 units, v2 17.4-19.6 but the v2 flame was dim), below it and tilted up; the whole
  flame ~45% of the frame height, the tip in frame at the fullest breath, above the caption band.
* Everything v3 is weighted by flame_w(), so from 628 on the crown / Ring are as before.

## (2) The grasp, C only: the claw that cannot hold (scene_c v3 block, tolkien.Ring.frame, timeline)
* Camera ABOVE the hand (27 -> 20 deg), looking down into the storm's eye where the Ring hangs bright; the claw
  descends on a diagonal from the upper left, palm down, knuckles to the lens, spread and hooked (962-1014); it
  closes on the Ring (1012-1029, slow then decisive); the crust glows from inside and cracks (1017+: a fine network
  opens from the grip outward, the creases burn hotter, gold where the hand touches the band, the Ring's glare
  spills between the fingers); at 1029.5 the band slips out between the middle and ring fingers, the claw jerks
  back and its grip gives, and the Ring falls away tumbling into the eye. No flash (GRASP_FLASH = False).
* Crust: opaque (coverage x1.8) and near-black; cracks follow the anatomy (`anat_cracks`): large plates on the back
  of the hand and forearm, the extensor tendons to each knuckle, knuckle wrinkles and flexion creases at every
  joint, the palm's three lines, the wrist, small cells at the joints, bark along the fingers.

## (4) The Eye, C (tolkien.Eye, scene_b CAM_B_C, Towers.lean)
* 260 thin fibres with dark gaps (most smoulder, a few burn), streaming inward faster, thinner splats; the slit's
  edge is ragged and shimmers (Eye.slit_w), its rim a ragged band of fire; the slit is masked to true black
  (alpha 0.997): its first opening shows nothing behind it. The band burns as ragged fire (angular noise), not a
  clean ring.
* Centred over the whole ring: C's camera swings back to the gap opposite (azimuth 0) and rises (840: 60 away, 36
  up; 880: 68, 34) so the Eye stands above every tower top; the towers lean toward it.

## (5) The Ring, C (tolkien: HB 0.5, TB 0.38, PEXP 2, inscription_v3 cache)
* Half the band height, a rounded section; one continuous surface (no per-point sparkle, overlapping splats);
  polished metal: a dark gold body mirroring the burning towers below, ONE sharp highlight (the fires' reflection)
  that travels round the band as it turns. Forged by cooling: the whole band white-hot at once (596-600), through
  pale yellow to gold by ~640, throwing 2,600 sparks; only the inscription burns. In the grasp it is x11 bright.

## Verified (half-res look-dev unless noted)
* Fire: 480-628 full res (sheet embers_v3_fire), motion test 484-580 (m_fire1: calm, breath visible).
* Towers: A MAIN 540-879 full res (sheet embers_v3_towers); ALT spot-checked at 740/800.
* Grasp: 966-1039 at 0.35-0.5 scale; Ring: 600-760 at 0.5; Eye: 820-878 at 0.5. Full-res C renders queued.

## Remaining weaknesses
* The two giants barely read in the v2 race camera (it sits inside the ring near the towers: the giants are at the
  frame edges or seen from the back). They will read in a wider or orbiting camera (E6's); until then ALT and MAIN
  differ in few frames.
* A 640-880 still has the v2 crown (the bible's A has none: E6/E7 are another lane).
* Grasp: from above, the fingers read a little thick (glove-like); the Ring is mostly hidden 1000-1014 (its glare
  shows between the fingers); the failure glow at 1026-1034 is strong. Motion not yet reviewed at full res.
* Eye: at sheet size it is still an orange disc with a slit (now fibrous and ragged); menace depends on motion.
* Ring: 606-612 the white-hot band blooms hard; the inscription reads as flecks except in close-up.

----------------------------------------------------------------------------------------------------------------

# EMBERS v2 — third pass (director: the globe read as a football), 2026-09-26

Re-rendered: embers_v2 486-627, 880-959 · embers_B 486-627, 880-959 · embers_C 562-627, 880-959.

* **Globe plates (880-959, all cuts): irregular, not a football** (`globe_layout.py`, replaces the 28 near-equal
  Voronoi cells of the second pass). A few big continental shields (unions of weighted cells: Europe-to-the-Gulf,
  East Asia and the western Pacific rim, South Asia and the Himalaya, North America, Siberia and the Arctic, Bering)
  each hold a flashpoint region WHOLE; many small shards toward the rim (oceans, Sahara, Arabian Sea, Pacific):
  equivalent radii 3-31 deg (~10x). Additively weighted Voronoi (curved seams between unequal plates) on a
  3-octave domain-warped sphere (seams wander), plus a finer fissure network that branches off the main seams and
  dies out before any listed site. Every flashpoint / capital / hub adds a smooth pull to the plate holding it, so
  no seam comes within its margin: `python globe_layout.py` reports 0 hard violations (all margins 4-8 deg as
  before); soft margins met except Chengdu, Karachi, Bangkok (<= 1 deg short, at the rim). Border check: no seam
  traces a land border except where two big shields meet in Central Asia and at the rim in West Africa / Indochina.
* **Staggered opening:** every seam has its own width, heat and start (-4..+18 frames on top of the Arctic-outward
  front), fissures grow out of their seam after it, and each plate parts on its own clock (916-934 start, shards
  further than shields); the seam's fire stays mid-rift (it no longer outlines both plates like ball panels).
  Timing kept: cut in 880, fire reaches every continent at about the same time, flare-out 950-959.
* Render log (third pass): 670 frames, ~1.5-2.5 s/frame (globe ~2-5 s), 0 errors; B/C globe frames equal v2's to
  the dither (mean |diff| 0.45/255). Sheet: _local_logs/review/embers_globe.jpg (v1 | 2nd pass | now at 916/932/946).
* **Flame sparks (486-627, all cuts; optional note):** while it is a flame only 4% of the spark column shows,
  lifting ~1-2 units off the tongue tips and dying (the column used to climb out of the frame above the tip).

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
* Render log (second pass; 2 procs, other departments rendering alongside, so times ran long):

| frames | n | median | total |
|---|---|---|---|
| A 316-830,866-959 | 609 | 4.3 s | 69 min |
| B 331-449,481-747 | 386 | 4.3 s | 47 min |
| B 791-959 | 169 | 2.5 s | 11 min |
| C 562-596 | 35 | 4.4 s | 3 min |
| C 597-959 | 363 | 5.0 s | 30 min |

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
