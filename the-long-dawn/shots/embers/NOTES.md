# >>> EMBERS-2 PAUSED 27 Sep ~21:05Z (usage gap to ~23:30Z). A7 THE EDGE + A8 THE BRINK (edge.py): APPROVED, RENDERING <<<
# (scope: A7/A8 only. A3/A4/A9/A10 = EMBERS-A2, A16/A17 = EMBERS-A3 (turn.py), C = EMBERS-C3 / EMBERS-C2)
EXACT STATE:
* APPROVED by the director (~21:00Z) and launched BY MAIN: embers_A3_edge (MAIN -> renders/embers_A3) and
  embers_A3_alt_edge (ALT -> renders/embers_A3_alt_codedtowers), 1840-2639, --nodes 4, at tip 6e78762 (has the
  crown-lean fix). Log: _local_logs/jobs/embers_A3_edge_farm.log. Check stills: review/embers_A3_check/edge2/.
* FREEZE (director): push nothing that changes 1840-2639 while they render, except a crown fix confined to
  2460-2519; then `farm.py cloud/jobs/embers_A3_edge.json --frames 2460-2519` (and the ALT job) is PRE-APPROVED.
* Crown test at 6e78762 (2466, 2478, 2488, 2498, 2510; request 0927-165213-embersA3edge-82690, re-queued 21:02Z)
  lands in renders/_farmtest/embers_A3_edge/. REVIEW IT FIRST on resume. Earlier crown tests failed (black on black,
  hidden in sparks, framed 6 units off because of the forge's lean). Fix inside 2460-2519 only if it still fails.
NEXT (post-23:30Z list; cheap re-renders):
 1. The crown hero, 2460-2519 (above).
 2. At 2532 there are white squiggle "worm" trails at the top and right edges (probably CrownTrail/StripEmbers
    streaks near the lens in the rim POV, or debris). Kill them (near_fade for those layers).
 3. At 2430 the heat shimmer makes tower edges and the lower slab wobble like a glitch. Keep the shimmer in the air,
    off hard edges (mask it by the pit glow / occluder depth; weaker once the frame lifts at the brink).
 4. The glowing crack net on the bowl wall is close to a regular lattice. Break it up (vary LAYER_H by azimuth,
    domain-warp w_strata, fewer and uneven veins).
 5. The gilding reads as gold glitter, not runs. Make broader rivulets with dark gaps and visible descending heads
    (GoldRuns), and fewer crust sparkles on gilded faces.
DESIGN (as rendered):
* Shots (render.SHOTS_V3 A3): (1840,2460) THE EDGE + brink part 1, (2460,2520) the crown, (2520,2640) over the rim.
* `_cam_edge`: one steady orbit r 27, y -8 (6 above the rim), az 2.93->3.25 rel ALPHA_C (the widest gap, forges
  4|5 framing), hf 88, SHIFT LENS (`lens_fall`, fall 0.22: towers upright). Opens level on the fire over the intact
  ground, tilts down with the falling plates (1848-1904). 2400-2459: lifts to a level-ish frame (hf 96) keeping the
  far lip at the bottom for the collapse (2440, jolt 0.22).
* `_cam_crown` (2460-2519): long lens (hf 38->48) from outside, r 70->61, az a7+0.05, level with the crown.
  CrownBreak: seam of fire 2464-2480, throat through the parapet, gold glint round the silhouette, the updraft's
  glare behind, raw broken faces after the break. FallingCrown: true centre includes the lean (`_c_lean`); turns
  about it, drifts 11 in, CROWN_G 0.042: past the lip ~2530, swallowed by the lake ~2544. The forge's height freezes
  at the break (edge._height). StripEmbers x0.12 and a thin CrownTrail in this shot.
* `_cam_fall` (2520-2639): at the lip in the 7|0 gap, pitch 34->64 deg, then an ease-in fall toward the lake's
  heart; post() whites out 2600-2640.
* Crater: bowl DEPTH 30 / BOWL_P 2.3 / LAKE 4.5, lip at the forges' feet (`_set_rim`), terraced strata lit from
  below, WALL_RAMP, veins, pit haze (AIR_E), Chunks (crumble from 2000, collapse 2440), shimmer (post). GoldRuns
  overlay; near_fade on tower embers/sparks. Fire (a3.py): settles to y -9, scale 2.0 -> 1.4 (1846-1910), swells
  at 2400.
* Workflow: farm tests `farm.py cloud/jobs/embers_A3_edge.json --test K --frames ... --detach` (push first). For
  shared files, stage only my hunks (scratchpad e4/stage_hunk.py) and commit the index. Another lane's NOTES commit
  once dropped this block: re-check it is still here after other lanes commit NOTES.md.

# >>> EMBERS-C3 (27 Sep ~20:35Z, took over from EMBERS-C): E15 LETTERS TO FIRE, E5-C THE FORGING, E11 THE RACE, and the
# canonical Ring (ringsolid.py) + gold flame (cflame.py). EMBERS-C2 owns E12 / E8-C / E13a (c2.py etc.). <<<
## STATE AT HANDOFF (EMBERS-C3, 27 Sep ~21:15Z; the account idles ~21:25Z-23:30Z). READ THIS FIRST.
**APPROVED + RENDERING (main launched them ~20:52Z, --nodes 3; log _local_logs/jobs/embers_C3_forging_race_farm.log):**
* `cloud/jobs/embers_C3_race.json` E11 THE RACE, C 1440-1679 (240 f) -> renders/embers_C3. Code 71d3247. Camera outside the circle
  (r 49-57, height 18-20), the Ring hanging over a legible ring of crenellated crowns; hush 1440-1559 with slow drips (half
  gravity); from 1560 gold spills LEVEL off the rim in solved parabolas (streams of 7 drops per window, c3.GoldRain), red walls
  crown-high (c3.WallsC; the two walls edge-on on the lens axis stay down), towers surge. Director: "works; no radial streaks".
  DO NOT push anything that changes 1440-1679 output while it renders (all C3 changes since are gated to t < 1440).
* `cloud/jobs/embers_C3_forging.json` E5-C, C 1040-1439 (400 f). Code 71d3247 (band edge-on/orange, Ring rises centred through
  1380, clears the crowns 1410-1430). Director: "1410/1430 lovely".
**MUST-FIX in flight (pre-approved partial re-render, no JOB READY needed):** the forges' faces in the close-ups (1250-1300, 1380)
  carried a regular grid (seams + window cells) = office towers / server racks. FIX pushed 870d18f: C3Sched.masonry (irregular
  ashlar: uneven courses, joints staggered per course, a tone per stone, darker joints), seam_k 0, shutter 0 (throats still burn),
  all x ashlar_k(t) = 1 before 1420, blended to the race's look by 1440 (1440+ byte-identical). scene_b hooks (small, noted):
  `SCHED.seam_k(i)` (seams) and `SCHED.masonry(i, pl, t)` (crust tone + window-cell kill).
  TEST: farm request 0927-165318-embersC3forging-83518 -> renders/_farmtest/embers_C3_forging_ashlar/ (f 1262, 1380).
  NEXT: look at both (four gates; is the grid gone? stones not a checkerboard?), then relaunch yourself:
  `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/embers_C3_forging.json --frames 1040-1439 --nodes 3`
  (director said 1040-1345; 1346-1439 must be included: 1380 is on the must-fix list and the look must not pop at 1346).
  If the stones read wrong: tune c3.C3Sched.masonry (course 0.5-1.15, widths 1.0-2.3, tone 0.7-1.2, joint 0.6) and re-test.
**E15 (C 700-1039, additive layer over MAP-L's page -> renders/embers_C3_e15): code pushed 870d18f, NOT yet seen rendered.**
* Sparks: e15.Draught rebuilt. A geodesic cost field (Dijkstra, 1 mm grid over the page) with three asymmetric channels
  (Draught.CH: head-left, right margin, foot-left) that join ONE trunk (T0 (9.5, 11.2) cm) drawn DOWN into the heart; a wall
  round the heart open only along the trunk. Every word's sparks = one thin stream (tributary), quickening toward the heart;
  words within 2.4 cm of the heart just flare and go out. Preview (paths only): _local_logs/review/embers_C/ (e15s3 idea: 764-794
  a bundled current down the trunk). Director's two E15 fixes: this is fix 1.
* Fix 2 (C5 letters legible): E15.letters_mask: one 3-letter word of MAP's book hand, DARK ink cut into the flame's gold body
  (x 0.88) with a hot rim, 20 % of the flame's height wide, at 43 % height, rising, 882-972, burning away from below.
* Text dims (EDIT): e15.TEXT (700-716 T2 tail, 920-1030 T5a) applies to sparks AND flame (band_rows): unchanged, still there.
* TEST: farm request 0927-165319-embersC3e15-83520 -> renders/_farmtest/embers_C3_e15/ (752 764 772 780 788 905 925 945).
  NEXT: review over black AND composited over MAP's page (renders/_farmtest/map_v3_book/book_C 728/900 + matte; EDIT's comp:
  out = book + (1 - matte) * black + e15); then JOB READY embers_C3_e15 (340 f, ~2.8 s/f) with check stills.
**Nitpicks (cheap re-renders later):** race 1600-1625 a soft out-of-focus warm blob at the left edge (a drop landing in far
  tower 8's window near the lens? cap GoldRain landing flares by distance); 1575 the stream burst is a bit spidery.
**Tools:** framing preview without tower geometry: `python ~/mishamisha/_local_logs/review/embers_C/wire.py OUT.jpg f1 f2 ...`
  (towers as boxes, the Ring, the flame; seconds, no RAM). Sheets: `sh.py` there. Check stills: the-long-dawn/review/embers_C3_check/.
**Old job** cloud/jobs/embers_C3_forge.json (1040-1679 in one) is superseded by the two split jobs: don't launch it.

# >>> EMBERS-C2 (split off EMBERS-C, 27 Sep ~19:15Z): C9 E12 THE EYE ONTO NOTHING (1920-2079), C11 E8-C THE GRASP
# THAT CANNOT HOLD (2320-2479), C13a E13a THE RING FALLS (2720-2839). EMBERS-C keeps E15, E5-C, E11 + the Ring/flame. <<<
## STATE AT HANDOFF (EMBERS-C2 -> next agent, 27 Sep ~20:15Z). Read this first.
* COMMITTED (f7b250f + this commit): `c2.py` (bar grid, `Router`, `ENABLED`, `plane_coords`, `storm_layer`), `c_eye.py`,
  `c_grasp.py`, `c_fall.py`, `c2lab.py` (look-dev harness, no tower geometry), farm jobs
  `cloud/jobs/embers_C3_{eye,grasp,fall}.json` (4 lanes of `render.py A-B:4 --cut C3` -> renders/embers_C3, jpg). The
  eye job file is EMBERS-C's (same config). render.py: my hook only (the `c2.Router` wrap + SHOTS_V3['C3'] (2720, 2840)).
  The OTHER uncommitted render.py hunk (A3 SHOTS_V3 2460/2520) is EMBERS-2's: never commit it.
* ROUTING: `c2.ENABLED = {'eye'}`. Only C 1920-2079 runs my code today. 2320-2479 still goes to c3's 'grasp' mode
  (EMBERS-C's grasp3.py). 2720-2839 falls through to c3's src warp (garbage). Add 'grasp' / 'fall' to ENABLED BEFORE
  any farm test or render of those frames.
* RENDERS: nothing delivered to renders/embers_C3 for my ranges; no JOB READY sent yet.
  - Farm test IN FLIGHT: `farm.py cloud/jobs/embers_C3_eye.json --test 6 --frames 1930,1966,1992,2003,2040,2079
    --local-out <scratchpad>/ec2/farm_eye1` (queued 16:03 local, pid 54976; log ec2/farm_eye1.log). It is the first
    look at the Eye WITH the towers (occluder mask, bend, light). Verdict: see the FARM line below.
  - Lab sheets (0.4 scale, no towers): scratchpad ec2/{eye,grasp,fall}_l4.jpg (pass 4 = current code).
* VERDICTS (pass 4, lab):
  - E12 EYE: the structure works. The storm is drawn in (it contracts, never turns: no spiral). The glazed iris has a
    closed seam, the rim is broken flame tongues, the slit opens at 2000 onto true black and reads as a pit, and the
    push-in lands. WEAK: the iris reads beige, like a wood-grain ball, not "yellow as a cat's" nor fire. ACES
    desaturates it, so push the chroma (mid zone g 0.5 -> 0.42, b 0.09 -> 0.04; pupillary zone b 0.3 -> 0.16) and
    raise E (x1.6 near the collarette). The storm is dim at 1930-1950 (fine under MAP's burn; check against book_C).
    The rim is thin at 1985 (the flames lengthen after 2000, good).
  - E8-C GRASP: reads as intended. It is a gaunt charcoal claw of embers (talons, jagged crust, red rim, no gauntlet)
    descending onto the three-quarter Ring. The grip hides the band with gold leaking between the fingers. The
    crack network opens from the grip 2400-2440, the band drops out under the fist at ~2446 and falls, and the claw
    goes dark after. WEAK: the blaze cracks (2420-2446) are too white and too wide. In shade_claw, cap T at 0.74,
    set e_crack to (0.22 + 2.0 * hk) and narrow anat_cracks' sharp. The big arcs on the back of the hand read as
    loops. The lab background is black: towers and storm are unverified (farm). Check the hang -> grip blend
    (2350-2360) and the fall in motion.
  - E13a FALL: 2722-2745 is good: out of black a small gold band tumbles in three-quarter, letters faint, the deck
    dark below. BROKEN after 2760: the deck reads as streaky brushed grey (fur), not moonlit billows (cut the domain
    warp 1.5 -> 0.6, add billow contrast, light the tops from the moon). 2805 is a beige blur (veil + warmth too big).
    2839 is brown murk with NO Ring visible. Restage the end: don't take the camera into the deck. Either stay above
    it and let a warm glint sink into thin tops, or use EMBERS-C's ringfall3 plan (through the cloud and out its
    underside into clear night; we slow, it drops away as a small glint).
* HANDOVER CONTRACT with RUN-C (R13b from 2840) is NOT agreed. My block below says "down and to the right";
  EMBERS-C proposed "a ~20 px glint just below frame centre, falling straight down". RUN-C has not built R13b yet.
  Recommend EMBERS-C's version (simpler to match). Agree it with the active RUN-C lane (PACING.md table) and write it
  into both NOTES.
* NEXT STEPS, in order:
  1. When farm_eye1 lands: review it with the towers. Check the bend reads, no tower sits under the Eye, there is no
     symmetric pair, and the sky sits behind the towers cleanly (occluder mask edges at half res). Fix the iris
     colour. Send JOB READY embers_C3_eye (160 frames; the farm does embers at ~9 f/s per node) with 3-4 full-res
     stills to review/embers_C3_check/.
  2. Grasp: apply the crack tone-down, set ENABLED |= {'grasp'}, commit and push, then run a farm `--test` on
     2330,2356,2364,2420,2446,2465 (the first look with towers far below). Tune the storm backdrop (c_grasp.post:
     storm_layer lp[0] 0.22) and send JOB READY embers_C3_grasp.
  3. Fall: fix the clouds and the ending, set ENABLED |= {'fall'}, settle the RUN-C contract, then send JOB READY
     embers_C3_fall.
  4. When MAP's book_C 1920-1991 exists: the storm's glow must sit where the page burns through (near frame
     centre-bottom). Adjust the Eye's CAM_KEYS if not.
* HOW: look-dev with `EYE=f,f GR=f,f FA=f,f renderq -- ec2/run_lab.sh TAG` (scratchpad) or
  `renderq -- python c2lab.py SHOT out.jpg 0.4 frames` (repo). Commit through a temporary index (GIT_INDEX_FILE),
  so only your paths go in, not others' staged files or EMBERS-2's render.py hunk.

## EMBERS-C2 DESIGN (as built, 27 Sep ~19:55Z)
* Owner files (new): `c2.py` (bar grid, `Router`, `storm_layer`/`plane_coords`: the 2.5-D storm as billboard planes),
  `c_eye.py` (E12), `c_grasp.py` (E8-C), `c_fall.py` (E13a). Shared touch: `render.py` (small, noted): `--cut C3`
  wraps c3.TimelineC3 in `c2.Router`, which sends C 1920-2079 / 2320-2479 / 2720-2839 to my shots (only those listed
  in `c2.ENABLED`; the rest stay on c3's stand-ins) and SHOTS_V3['C3'] gains (2720, 2840). c3.py/scene_c/tolkien untouched.
* Uses EMBERS-C's canonical Ring (`ringsolid.render`, assets/ring) and C's tower layout (`c3.layout_towers`, schedule
  subclassed from `c3.C3Sched`, held past 1680). c3's src-warp Eye/Grasp stand-ins are superseded by these.
* E12: the storm over every forge resolves (1936-1990) into the BOOK's Eye (glazed, yellow as a cat's, rimmed with
  fire; no tower under it, no horns): 3 storm planes (behind + 2 in front: parallax), the Eye as a disc in its plane
  (`_eye`: wavy fibre bundles, crypts, collarette, dark limbus, flame rim, the closed seam; the glaze = the forges'
  throats reflected low on it + the rim fire as an arc), every tower bends toward it (`EyeSched.tower_post`). 2000:
  the slit opens onto true black (nothing behind shows: no storm, no glaze, embers drawn in go out). Camera: C3
  azimuth AZ0+0.1 (no tower under the Eye, no symmetric pair), r 158 -> 127 (push from 2000).
* E8-C: the claw is a SOLID (sphere-traced SDF of scene_c.HandSkel's anatomy, gaunter, talon tips; `_march`), shaded
  as charcoal crust with ember pores + scene_c.anat_cracks in the rest pose; the Ring's gold as a point light and a
  leak along the finger seams; heat spreads out from the grip from 2400; the band sits in the grip's hollow (found
  from the SDF: ~0.09 L) and slips out between middle and ring fingers at 2440, then falls (ballistic, tumbling).
* E13a: out of black the band catches the moon (the only light it takes), tumbling; we fall with it looking down
  onto a moonlit cloud deck, through it (veil), and it falls away from us: a small warm spark moving down-right.
* HANDOVER to RUN-C (R13b, 2840): the Ring is a small warm spark (a few px, short trail), falling down and to the
  right of frame centre; nothing else warm in frame; moonlight cold blue-white from the upper left.
* Look-dev: scratchpad ec2/ (lab.py renders my layers without the tower geometry). Full tests on the FARM.
# >>> EMBERS-A3 (split off EMBERS-A2, 27 Sep ~19:30Z): A16 TOWERS IN THE LIGHT + A17 THE FIRE, SEEN (4400-4879,
# one take) + both in the _alt_codedtowers ALT. Owner file: turn.py (+ cloud/jobs/embers_A3_turn.json, _alt_turn.json).
# Shared modules untouched so far (a3.py/edge.py/scene_b.py hooks are installed from turn.py on import). <<<
## EMBERS-A3 STATE (27 Sep ~20:50Z) -- RESUME HERE (~23:30Z)
* 20:55Z: APPROVED (director). main launched BOTH finals itself (--nodes 2; log _local_logs/jobs/embers_A3_turn_farm.log)
  -> renders/embers_A3 4400-4879 + renders/embers_A3_alt_codedtowers 4400-4879. Do NOT relaunch them; on resume, check
  the log and that both folders hold 4400-4879 (use `--missing` only if the director says so).
* NEXT after 23:30Z, for a cheap re-render (MAIN + ALT, same jobs):
  (1) DIRECTOR: the tower edges and floor lines read as strings of fairy lights (a laugh risk at 4460 and 4879).
      Break up the continuous dotted edge/seam lines so they read as ember-lit stone or charcoal crust, not bulbs on
      a wire: in TowerLight, modulate kinds 2/1/4 along their length with a low-frequency break-up (gaps, uneven
      heat, some lengths dark), fewer and finer points; check scene_b's own edge/seam shading does the same in my
      range (fire_side_only look), or damp it there from turn.py with a hook.
  (2) DIRECTOR: the 4879 heart gets a living core instead of a plain disc. RUN-A2's lantern at 4880 must match it,
      so AGREE THE HEART'S LOOK WITH RUN-A2 (via NOTES: shots/run/NOTES.md and this block) before either of us changes
      it. Current contract: centre 959.5,401.5; ice-white (0.80,0.92,1.00); gaussian sigmas 10/26/114 px, peaks
      30/0.9/0.05 (crossing.draw_heart at 4880).
  (3) the giants' opening unmistakable (brighter, larger window slits on the facing faces, or giant 6's facade closer);
  (4) the small lights at the rim visible in the walk; (5) a few ridge fires nearer the frame's middle;
  (6) optional: the giants' full height once.
* Code: turn.py v3 (commit d84bf48), pushed. Jobs: cloud/jobs/embers_A3_turn.json (MAIN 4400-4879 ->
  renders/embers_A3) + embers_A3_alt_turn.json (ALT 4400-4879 -> renders/embers_A3_alt_codedtowers), 4 lanes each.
  Farm look-dev frames: renders/_farmtest/embers_A3_turn{,_lab}/ (v1 = leopard wash, rejected by me; lab = camera
  variants 4400-4405/4530-4533; v3 check = 4400,4460,4500,4580,4660,4760,4830,4879).
* v3 design: one take, the camera looking in through the gap between forge 1 and giant 2 (az 1.56, r 58 -> 42,
  y 12, hfov 66: forge 1's back left, giant 2's back right, the fire and giant 6 in the gap, the crater glow below),
  then (A17) walking down through that gap to the rim (r 17.5, y -10.5) with the heart centred at 4879. The light
  finds the structure (edges/bands/seams/joints as warm lines, an even dim crust sheen: v1's crust wash read as
  leopard print). Washes fade into the night as the fire gathers (night(), world_light 0.15 at the end).
* Lab-verified (v2 lab sheet): the backs' bands and edges catch the far fires' warm light; ridge fires read as small
  clusters at the frame edges; the heart centred and at the crossing's size.
* KNOWN NITPICKS (for a cheap re-render after 23:30Z if the director wants): the shutters opening is readable only
  as the facing facades brightening at this distance (the window slits are small); no shot of the giants' full height;
  ridge fires are small and mostly at the edges; the heart is a plain white disc (no inner filaments).

# >>> EMBERS-A2 (split off A, 27 Sep ~18:50Z): A3 GLYPHS, A4 THE POINT, A9 THE DEAD VALLEY, A10 THE EMBER,
# A16 TOWERS IN THE LIGHT, A17 THE FIRE, SEEN (+ A16/A17 in the _alt_codedtowers ALT). EMBERS-2 keeps A7/A8 (edge.py).
# Owner files: glyphs3.py (A3/A4), aftermath.py (A9/A10), new modules only; a3.py/render.py hooks in small noted
# commits. SCOPE (director, ~19:45Z): A16/A17 now belong to EMBERS-A3; EMBERS-A2 keeps A3/A4 (+ A9/A10, done). <<<
## EMBERS-A2 FINAL STATE (27 Sep ~20:45Z): ALL MY SHOTS DELIVERED; the lane is closed
* DELIVERED (director-approved, farm-rendered, QC'd: every frame present at 804x1920, no pops, finals match the
  approved check stills to JPEG noise), in renders/embers_A3, A cut numbering:
  - A3 INTO THE LIGHT . GLYPHS + A4 THE POINT: 560-1039 (480 f). Code glyphs3.py (d4e48f9), job
    cloud/jobs/embers_A3_a3a4.json. Check stills + sheet: review/embers_A3_check/a2/glyphs/.
  - A9 THE DEAD VALLEY + A10 BLACK . THE EMBER: 2640-3119 (480 f). Code aftermath.py (6bd5fe2), job
    cloud/jobs/embers_A3_a9a10.json. Check stills: review/embers_A3_check/a2/.
  Re-render either: python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/<job>.json --nodes 1 [--missing]
* Hand-offs to other lanes:
  - EMBERS-A3 (A16/A17): a3._mine() sends every frame >= T_LIGHT (4400) to turn.py, a STUB that replays the old
    edge.py path. It is yours: rewrite it or re-point _mine().
  - RUN-A-L (A11 X2) + EDIT: the ember rests at px (960, 548) at f3119 (core blackbody ~0.8, warm halo r ~12 px,
    aftermath.ember_px / ember_life); the director has told both to carry it on from there.
  - A5 continuity: A4's point sits at px (967, 239) on f1039; A5's ignition disc at (964, 238) on f1040.
* Things to know: the glyph letters load assets/glyphs/atlas_v3.npz (committed; the approved v2 atlas as float16,
  less the Om and the Eye of Horus), so farm nodes need no fonts (farm.py skips apt-get and has no Noto). Pushing
  that binary needed `git -c http.postBuffer=157286400 push ...` (HTTP 400 otherwise). Shared-file edits went in
  through _local_logs/ea2/pcommit.py (a temp index from HEAD), so no other lane's uncommitted WIP rode along.

# >>> EMBERS-C (cut C's embers shots; EMBERS-2 keeps A). Owner files: c3.py, scene_c.py, tolkien.py (+ new C-only
# modules). Shared (core.py, towers*.py, render.py, scene_b.py): small, noted commits only. <<<
## STATE AT HANDOFF (EMBERS-C, 27 Sep ~20:30Z; the director asked for a fresh agent). READ THIS FIRST.
**Scope now:** E15 LETTERS TO FIRE (C 700-1039, with MAP-L's page), E5-C THE FORGING (C6 1040-1439), E11 THE RACE
(C7 1440-1679), plus the shared canonical Ring + gold-flame assets. EMBERS-C2 owns E12 / E8-C / E13a (c2.py router,
c_eye.py, c_grasp.py, c_fall.py; it routes C 1920-2079, 2320-2479, 2720-2839 in render.py --cut C3). Don't edit its files.
**Approved / delivered:** NOTHING yet. No JOB READY sent, no finals: renders/embers_C3 and renders/embers_C3_e15 are empty
(the animatic still plays embers_C3_half, pre-H5, and MAP's old X1 test mp4). Director-approved CONTRACT for E15: MAP-L's
book_C 560-1039 is page only (letters glowing as letters, strokes' flare-and-go-out, page lit by the fire, burn + hole;
fire=None, xl kept); EMBERS-C delivers renders/embers_C3_e15 (C 700-1039, additive on black); EDIT wired the comp
(8974648: out = book_rgb + (1 - matte) * black + e15). EDIT asked for dimming at 700-716 (T2's tail) and 920-1030 (T5a):
done in e15.TEXT (the flame layer too, via e15.band_rows).
**Assets (shared with EMBERS-C2):**
* `shots/embers/ringsolid.py` THE RING: `render(cam, W, H, Rot, C, width, RingState, Env, th_range=None)` ->
  (premultiplied HDR rgb, alpha, camera depth). Canonical band (ringc.py proportions: outer R 2.25 x width, thickness
  0.44 x width, section SQ 2.8), z-buffered numba raster, supersampled; polished gold = Schlick Fresnel (F0 gold) x
  `Env` (vertical gradient + spherical-Gaussian `lobe`s + `point` bodies); canonical inscription from assets/ring/
  (outer u = -theta/2pi, inner u = theta/2pi, v from the +axis edge; mipped by footprint) as emission, colour C_LETTER
  deep orange-red; `RingState`: letters, write(th), heat(th) (white->yellow->orange->dull red), hammer, sec (section
  scale: a thread beaten out to the band), glow, glow_col, exposure, alpha. `merge_occluder(fr, alpha, depth)` joins it
  to the splat occluder (splats behind it vanish; returns the towers' occluder for `visibility(before, depth, H, W)`).
  Lessons: a face-on band reads as a halo (show it three-quarter); keep a warm term in Env or gold goes copper/olive.
* `shots/embers/cflame.py` THE GOLD FLAME: `draw(hdr, root_px, tip_px, t, bright, calm, vis, glow, seed, scale)`, a
  screen-space procedural flame about its own axis (one tongue, domain-warped licks climbing, heat ramp; brightness is
  surface brightness, so any size); `Sparks().emit(ctx, root, height, bright, amount)` calm tip sparks (3-D splats).
  The same flame is drawn by e15.py (through 1039) and c3.py (from 1040), so the handover is continuous by construction.
**Code:** `c3.py` (C's timeline: C3Sched hooks for scene_b, layout_towers (8 forges r~17.5 + 10 far r 31-39, heights
alike, surge lead passing round: none taller, no pair), ring_frame/heat/sec/write/letters/env, ForgeFX (thread of light
flame-tip -> white-hot front 1202-1238; stroke sparks), GoldRain (ballistic molten drops band -> windows), GroundPool
(screen-space ground lit by the fire), CAM_C6, TimelineC3 with modes sched/eye/grasp/src; eye/grasp are dead once
EMBERS-C2's router is in), `e15.py` (own driver, `--frames`; imports MAP-L's committed shots/map/book_c.py ONCE to cache
MAP's camera per 1/4 frame, every stroke's spark seed + lift time, the heart and its flame-height track into
renders/embers_C3_e15/cache/mapdata_<hash of MAP's sources>.npz; Draught = sink + curl-noise eddies per word),
`variant.fire_side_only()` includes C3 (A's approved charcoal forge look), scene_c.GRIP_OVERRIDE + crease arcs.
**Jobs:** `cloud/jobs/embers_C3_forge.json` (C 1040-1679, render.py --cut C3; farm ~7 s/frame on a GPU node),
`cloud/jobs/embers_C3_e15.json` (C 700-1039, e15.py --frames; ~2.8 s/frame). Tests: from ~/mishamisha,
`python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/<job>.json --frames a,b,c --test N` (lands in
renders/_farmtest/<job>/; PUSH CODE FIRST). Sheets: `python ~/mishamisha/_local_logs/review/embers_C/sh.py OUT.jpg 4 480
_farmtest/embers_C3_forge:1040 ...`. Review sheets (durable): ~/mishamisha/_local_logs/review/embers_C/.
**Verdicts on the last passes** (review/embers_C/forge_race_pass4.jpg, e15_pass3.jpg; frames in renders/_farmtest/):
* FORGING 1040-1345 is close: the gold flame alone (1040) is right; the forges rise round it in the gap view (1100-1195,
  charcoal + seams: reads as forges, but the far ones' dotted seams read a little like lit office grids at distance);
  the band being drawn and beaten (1210-1285) reads, but it is a thin WHITE hoop near face-on at 1240-1262 (halo risk);
  1305-1345 is the strongest: a heavy cold-gold band, the canonical letters burning up. 1380 is awkward (the Ring cut off
  at frame bottom as it rises); 1410 good (Ring tumbling up, letters).
* RACE 1440-1679 FAILS in pass 4: the under-the-Ring camera loses the towers (1450-1545 is a small ring in black) and,
  from inside the forges' ring, the gold rain and the walls of red become RADIAL STREAKS (1580-1665 = warp-speed
  screensaver). Pass 3 (review/embers_C/forge_race_pass3.jpg, 1470-1540: the Ring over a crenellated crown, towers
  below) was the better geometry.
* E15: the flame (820-1039) is one gold tongue, calm, no fork (slightly candle-narrow); the peel reads (745-752) but the
  convergence reads as a radial implosion and a bow-tie at 784-792, not streams in a draught; the letters in the flame
  (900-935) are illegible white squiggles. Nothing yet judged over MAP's page (MAP-L's page-only book_C not rendered).
**Next steps, in order:**
1. RACE camera: outside and low among the crowns (r 30-34, height 18-21 above ground, target at the Ring, hfov 55-60,
   azimuth in the gap AZ0 so no forge fills the lens), the Ring hanging just above the crowns with 4-6 towers in frame;
   keep the lens OUTSIDE scene_b.Walls (radial sheets r 5-21 at the gaps; set walls_t0 later or push them off-axis if
   they still streak); gold rain must read as drops arcing into windows (seen side-on, not falling at the lens).
2. FORGING: turn the forming band more edge-on (ring_frame yaw offset 40 -> ~60 deg while sec < 0.6) and keep it
   orange-hot except the running front; smooth CAM_C6 1345-1440 so the Ring rises through frame centre (no cut-off at
   1380); consider SKYLINE seams dimmer at distance.
3. E15: fewer, distinct streams (one thread per word, ~40-60 threads, a gentle common drift so it is not radially
   symmetric, stagger arrivals to kill the bow-tie); either make the letters in the flame legible (2-3 glyphs, xh ~0.5
   cm, bright gold lines inside the body, 880-960) or cut them; base 15-20% broader. Check E15 1039 vs C3 1040 side by
   side (same flame root/height: C6's 1040 key is MAP's camera at 1040 scaled by K_E15).
4. Full-res check stills -> the-long-dawn/review/embers_C3_check/ -> SendMessage main: JOB READY embers_C3_forge
   (640 frames) and embers_C3_e15 (340 frames); finals only after approval (farm.py --nodes 1-3).

## EMBERS-C -> EMBERS-C2 (27 Sep ~20:00Z; the director's split: C2 owns E12, E8-C, E13a; I keep E15, E5-C, E11
## and the shared Ring (ringsolid.py) + gold flame (cflame.py) assets). My look-dev on C2's shots, to fold in:
* `eye3.py` (committed; c3 'eye' mode, dead once c2.Router takes 1920-2079): a screen-space sky on a plane facing
  the lens. What worked (scratchpad ec/eye_sheet4.jpg, verified at 0.5 scale): angular noise on circles (cos, sin)
  so there is no atan2 seam; time only advects along log r (the storm and the fibres are drawn INWARD, nothing
  turns: no spiral); the iris as a STROMA of fire (the angle itself domain-warped, crypts, a collarette) -- straight
  radial fibres read as a starburst/warp-speed, isotropic noise as a lava cookie; the storm drawn in by CONTRACTING
  its Cartesian texture (exp(0.006 t)), never radial streaks (they read as a vortex/sunburst); the slit = a hairline
  seam that opens 2000-2009 onto pure black with the pit's inner wall lit for a hand's breadth, thin uneven lips.
  Open: the storm is still dim at 1930; the rim is a clean circle.
* `grasp3.py` (committed; c3 'grasp' mode): sphere-traced scene_c hand SDF (numba `_march` with an AA fringe from
  the ray's closest approach), shaded charcoal lit by the Ring's gold from inside the grip, anat_cracks re-mapped to
  the rest pose per pixel (owner bone -> rest frame), the Ring = ringsolid depth-composited with the claw, a leak
  glow only through the gaps (x (1 - claw alpha)). Findings (ec/grasp_t3.jpg): a face-on Ring reads as a halo (38
  degree three-quarter view instead); env must be warm-gold or the band goes copper; pre-failure the tendons must
  stay faint (they read as wires); the blaze must keep the plates charcoal (else a paper lantern); full knuckle bands
  read as rings WORN on the fingers (scene_c.anat_cracks now keeps only back/front arcs; wrist creases palm-side);
  camera 126->112 from the grip, el 24->19, hfov 52, target 7 above the grip; fall g 0.13/frame^2 out of frame.
* `ringfall3.py` (committed, not wired): first test (ec/fall_t1.jpg) shows the Ring far too big (camera 9 units:
  it should be a glint), gold goes olive under a blue-grey moon (keep a warm term in the env), the fbm clouds read
  as fur/brushed. The handover proposal for RUN-C stands: at 2839 a ~20 px glint just below centre, falling.

## EMBERS-C STATE (27 Sep ~19:20Z)
* NEW MODULES (mine): `ringsolid.py` THE RING as a solid canonical band (z-buffered raster of the superellipse band,
  SQ 2.8, outer R 2.25 x width; polished gold = Schlick Fresnel x an environment of spherical-Gaussian lobes + point
  bodies (fire, forge throats, lit smoke); the canonical inscription as emission through assets/ring strips with
  mips; heat white->yellow->gold; th_range for a band still being laid; `merge_occluder` joins it to the splat
  occluder so splats behind it vanish; `visibility` hides it behind towers). Verified standalone (scratchpad
  ec/ringtest1.jpg: reads as a real gold ring, letters legible). `cflame.py` C's fire: ONE natural gold flame
  (stateless gas particles, buoyant, necking to one tip, curl of noise, tip flamelets, calm sparks; surface
  brightness constant across scale). `e15.py` E15's layer -> renders/embers_C3_e15 (own driver; imports MAP-L's
  committed shots/map/book_c.py once for its camera, seeds, heart and flame track; cached by MAP's sources).
* c3.py REWRITTEN for C6/C7: C's own tower layout (8 forges r~17.5 + 10 far r 31-39, heights all alike, the surge
  lead passing round: none taller, no pair), fire on the ground at the centre (HF 6.5), the band beaten out on the
  anvil strokes 1200-1300 (thread of light from the flame's tip to a white-hot front running round 1202-1238, a
  spark burst per stroke, throats pulse), cooling to gold by 1320, letters burning up 1320-1346, the Ring rising
  1360-1470 to hang tilted over the towers; the hush's slow drops and the race's gold rain (ballistic molten beads
  from the band's rim into the forges' windows, windows flare); GroundGlow; new cameras (CAM_C6). EYE/GRASP still
  the old src path (to be rebuilt). variant.fire_side_only() now includes C3 (A's approved charcoal look).
* CONTRACT (director APPROVED 19:0xZ): MAP-L's book_C 560-1039 is page only (fire=None, xl kept); EMBERS-C delivers
  renders/embers_C3_e15 (C 700-1039) additive; EDIT wired it (8974648); dim the lower third at 700-716 (T2's tail)
  and 920-1030 (T5a): done in e15.TEXT. E5-C's first frame at 1040 must continue E15's flame from MAP's camera
  (TODO: derive CAM_C6's 1040 key from MAP's camera: 46 deg above, the flame ~49% of frame height).
* Tests: local renderq (1 slot) until FARM is announced; then `cloud/farm.py cloud/jobs/embers_C3_forge.json --test`.
## STATE AT HANDOFF (EMBERS-C takes over C, 27 Sep ~18:45Z)
* Shots (C frames; deliver to renders/embers_C3 in C numbering): E15 LETTERS TO FIRE C4-C5 560-1039 (particles + fire;
  MAP owns the page + X1), E5-C THE FORGING C6 1040-1439, E11 THE RACE C7 1440-1679, E12 THE EYE ONTO NOTHING C9
  1920-2079 (MAP burns through at its head, 1920-1991), E8-C THE GRASP C11 2320-2479, E13a THE RING FALLS C13 2720-2839
  (RUN-C takes the streak from 2840).
* Renders: renders/embers_C3 is EMPTY. renders/embers_C3_half = 960 half-res frames (1040-1679, 1920-2079, 2320-2479)
  made 07:20Z, BEFORE the H5 calls and before A's charcoal-crust towers: the animatic's stand-in only. Its problems:
  the fire forks (1040), towers are the old mottled crust, the Ring is a thin hoop that reads as a HALO round the fire
  (1220-1330) and a ring-toss hoop over the towers, gold rain reads as glitter, THE EYE and THE GRASP still sit in the
  blue-white galaxy spiral (Winamp) with a film-like cat's eye and a wireframe claw. E13a and E15 were never built here.
* E15: MAP-v3 already built its own sparks + flame in its book engine (`Kindling`, claude/v3-map d74bd7b, not merged
  here; review/v3/map_X1_letters_to_fire.jpg on that branch) and exports `review/v3/x1_letters.json` for EMBERS (6140
  spark seeds: screen xy at the LIFT camera (f720), lift + arrive frames; the heart's screen track + flame px per
  frame). The animatic still plays MAP's older test (edit/cache/x1_letters_C_test.mp4: dot field at 740-770, a
  three-tongued flame at 830). The director now gives the particles and the fire to EMBERS; MAP keeps page + X1.
* Ring: must carry the canonical inscription (assets/ring/, 16-bit coverage; canonical band: outer R = 2.25 x width,
  thickness 0.44 x width; ringc.py R_IN 9.4 THICK 2.3 WIDTH 5.2 mm). tolkien.py's band is R 5 x width 1.0 (a hoop:
  part of the halo read) with the superseded inscription.py script.

# >>> EMBERS-2 PAUSED 27 Sep ~16:40Z (director: usage window end). RESUME 20:00Z: continue the EDGE/BRINK pass <<<
EXACT STATE (source committed; NOTHING rendering; the edge jobs are still HELD, not yet updated):
* DONE in code (edge.py / scene_b.py / a3.py), lab-verified at 0.3 scale:
  - THE CRATER is a deep BOWL (DEPTH 30, profile BOWL_P 2.3, molten lake LAKE 4.5 at FLOOR_Y) whose LIP RUNS AT THE
    FORGES' FEET (`_set_rim(towers)`: each tower's innermost foot point - 0.45; torn bays between them), so the towers
    stand on spurs of the lip. Wall: charcoal under the lip, blazing toward the lake (u**1.5), fissures, ledges, and
    MOLTEN VEINS running down into the lake (they converge in perspective: this is what makes the bowl read). Lake:
    white gold under drifting dark rafts. Glowing pit air (`AIR_E` blobs) + H/H2 glows. The old thin lip line and the
    edge-glow band are gone; the plain is dark crust lit by the fire (`fire_light`) with slabs + seams near the lip.
  - `Chunks`: rigid crust slabs break off the lip under the gilded towers on the beat from bar 26 and all at once at
    2440, tumbling into the pit (they join the occluder). `PitEmbers` rise out of the lake. Heat shimmer: `_shimmer`
    in post (screen-space refraction masked to the bowl's opening + the column above it).
  - The fire SETTLES INTO THE CRATER'S MOUTH (a3 fire_centre: y -> -12 over 1846-1910, +5 at the brink) so the bowl,
    the fire and the far towers fit one low frame.
  - GILDING (scene_b._gold_runs): on every beat liquid gold pours from the crown in rivulets with bright heads,
    staggered like drips, over a gilt skin; gilded towers lose 60% of the fire's light (the gold carries them), the
    farthest go dark (lit x0.1, heat x0.1). `_gild` starts at the first downbeat. NOT YET CHECKED IN A CROP.
  - The falling crown's normals turn with it (`tower_post_n` hook in scene_b Towers.prepare). Dust x0.3 in THE EDGE.
* NOT DONE: edge.camera still has the OLD orbit (r46 y30 looking down). The lab camera findings (scratchpad e4/,
  a6.jpg): the bowl reads from LOW and CLOSE: r 25-28, y -8/-9 (5-6 above ground), ty ~ -18/-20, hfov 90-96,
  with the near towers as black framing silhouettes and the near lip + near ground dark in the lower third.
  Best so far: 28,3.09,-8,-18,90 (r,az rel ALPHA_C,y,ty,hf) but there the giants hide behind the near towers.
  A geometry scan (e4/scan.py) says the giants stand clear on opposite sides with a clear centre at az -1.0
  (r 28-30, hf 90: giants at x 0.30 and 0.87) or az -1.56 / -2.46 (hf 80). Test those next.
NEXT STEPS:
 1. Restart the lab (e4/labsrv.py, ~2.5 min to build; then e4/req.sh NAME json renders in seconds; see its
    docstring) and pick the orbit: ~0.25 rad, steady, centred on the best az above; lower third dark (text band
    y 0.70-0.87) during T7/T8 (1848-2060). Then write it into edge.camera (THE EDGE) and add the CUT at 1840 to
    render.SHOTS_V3['A3'] ((1760,1840),(1840,2640)...): A6's last frames are unaffected.
 2. Check the gilding in a 0.6-scale crop of a far gilded tower (towers 1, 7, 2, 6 gilded; 0, 3, 5 dark).
 3. THE BRINK staging: updraft (keep), the lip giving way at 2440 (Chunks burst), HERO crown 2480-2520 (tower 7,
    bigger in frame: camera ~35-55 from it, hf ~50-60, visible ~24 frames tumbling into the glare), then tip over the
    rim and fall after it to white at 2640.
 4. Check stills (4-5, full res) -> review/embers_A3_check/edge2/ -> SendMessage main -> JOB READY for
    embers_A3_edge + embers_A3_alt_edge (1840-2639).
 5. Then A9, A10, A16, A17 (the crater is visible in A16/A17 too: check it), then C's shots.

# STATE AT HANDOFF (EMBERS-2 takes over, 27 Sep ~15:15Z)
* A5 (1040-1439) + A6 MAIN (1440-1839) are DONE in the cloud and imported: renders/embers_A3 has 1040-1839 (jpg);
  renders/embers_A3_alt_codedtowers has the ALT 1440-1839. Nothing of A7+ is rendered.
* HELD: cloud/jobs/embers_A3_edge.json + embers_A3_alt_edge.json (1840-2639) wait for the EDGE/BRINK pass below.
* Order of work (director, brief2): (1) EDGE/BRINK pass (bowl geography, liquid gilding, crumbling rim, hero
  crown) -> 4-5 check stills -> JOB READY for MAIN+ALT; (2) A9, A10, A16, A17; (3) C: E15 (+MAP), E5-C FORGING
  (many towers, no pair, canonical Ring script from assets/ring/), E11, E12, E8-C, E13a -> renders/embers_C3.
* Cloud jobs that render E1 GLYPHS need `apt-get install -y fonts-noto-core fonts-noto-cjk` in their setup.

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
