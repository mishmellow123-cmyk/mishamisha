# >>> RUN-A (A . FALSE DAWN R1 · X2 DARK ADAPTATION · R2-A REVEAL · R3 BEACON RUN · R16 WATCHERS · R6 THE CROSSING · R7 THE BLUE HOUR · A20 title sky) <<<

## RUN-A2 STATE (split off 27 Sep ~19:00Z: A18 THE CROSSING, A19 THE BLUE HOUR, A20 title sky; owns crossing.py, bluehour.py, sdfppl.py, crossing_fires.npy)
* **Doing now:** the gate's fixes before any render. CROSSING: star trails open with the wheel (bar 66 b1) as long concentric
  arcs about a pole held inside the frame; one big knife-edge range under the pole plus stronger aerial depth; cloaks rebuilt
  as cloth (elliptical, sharp-creased folds, wind billow and hem lift, flutter, SDF occlusion, wool rim). BLUE HOUR: about 40
  seated, unroped bearers round the set-down lantern, all turned to the east (no lit faces); a valley revealed through the
  thinning cloud sea with thin grey-blue hearth columns; the rose only from bar 75 b1. A20: the same take continues 6240-6479.
* **Nothing rendered yet, no job ready.** `farm.py` is not in the repo yet (jobs will be cut to <= 45 min when it lands).

## SPLIT (director approved ~19:05Z): two agents in this department
* **RUN-A-L:** A2 FALSE DAWN (`falsedawn.py`), A11 X2 (`stars_a.py`, new), A13 R2-A REVEAL, A14 BEACON RUN and A15
  WATCHERS. Folders: `falsedawn_A`, `stars_A`, `reveal_A`, `beaconrun_A` and `watchers_A`.
* **RUN-A2:** A18 THE CROSSING, A19 THE BLUE HOUR and A20's title sky (`bluehour_A` 6240-6479). Owns `crossing.py`,
  `bluehour.py`, `sdfppl.py` and `crossing_fires.npy`. RUN-A-L uses sdfppl read-only for the watchers and sends
  figure requests through RUN-A2.
* **Both:**
  * `world.py` stays additive-only.
  * Commit through `_local_logs/runA/commit_runA.py`.
  * Keep your own sub-block (`# >>> RUN-A2 ...`) directly under this one. The helper carries every `# >>> RUN-A*`
    block together.
  * No old job launches. Heavy renders go to the FARM in jobs of 45 minutes or less, with JOB READY and check stills.

## STATE AT HANDOFF (RUN-A-L on the local Mac, 27 Sep ~18:50Z)
* **Who.** The cloud RUN-A session (account now unreachable) paused at ~16:35Z. RUN-A-L took the lane over locally.
  Its branch `claude/v3-runA` (fbe4ada) was imported onto `claude/long-dawn-v2` in **6587a6e** with explicit paths, not
  merged: `crossing.py`, `sdfppl.py`, `crossing_fires.npy`, `falsedawn.py`, `bluehour.py`, `world.py` and 17 job files.
  The tracking ref was deleted afterwards. The review sheets are NOT committed, because long-dawn-v2 carries no review
  images; they are in `_local_logs/review/runA/`, with the branch's own NOTES block as `NOTES_runA_branch_block.md`.
* **world.py is safe for every lane.** Its additive flags were checked against every caller: crag and ridge rows read
  cols 14 and 15, which `crag_row` and `ridge_row` leave at 0, and `range_row` never reaches crag() or ridge();
  `shade` reads Q[18], and every Q passed in has at least 20 entries, all zero by default. `bworld.py` has its own row
  layout and never calls `world.crag` or `world.ridge`. long-dawn-v2 had not touched world.py since the merge base.
* **No RUN-A render exists anywhere.** No job was ever launched: there is no `claude/render-*-a-*` branch on origin,
  and there is no `renders/{falsedawn,reveal,beaconrun,watchers,crossing,bluehour}_A`. The jobs `crossing_a_1..8`,
  `falsedawn_a_1..4` and `bluehour_a_1..5` target run_job.py on four-core boxes. They get re-cut for the FARM
  (`cloud/farm.py`, pending its announcement).
* **Built (H5 applied by the cloud lane):** A2 FALSE DAWN (arc), A18 THE CROSSING, and A19 THE BLUE HOUR (greybox).
  **Not built:**
  * A11 X2 star plate (EDIT's proxy plays; the folder is `stars_A` or `x2_A`).
  * A13 R2-A reveal pull-back (cut 3680-3799; H1's roar holds 3600-3679).
  * A14 BEACON RUN (3920-4239; v2's `run.py` is a different shot).
  * A15 WATCHERS (4240-4399).
  * A20 title sky (6240-6479 in `bluehour_A`).
* **My gate read of the branch's latest sheets.** These are the fixes to make before any cloud render.
  * **CROSSING, wides from bar 70.** The star trails still read as short random dashes ("rain" or "warp"). Every
    background is a field of uniform needles, a terrain tell. In the close-ups the cloaks read as smooth clay bells.
  * **FALSE DAWN.** The far horizon is a uniform sawtooth of tiny needles. The cloud sea reads as a dark lake with
    islands. The mackerel deck looks like flat blobs pasted on.
  * **BLUE HOUR.**
    * The figures read as plush or clay, with a Jawa risk.
    * T14 shows only 2+3 people, and it must show everyone together.
    * The hearth smoke rises straight out of the cloud sea as evenly spaced vents, so it reads as geysers, and there is
      no valley.
    * The rose is already full on bar 74; it should pale from bar 75.
* **Next.**
  1. Split proposal to main: RUN-A-L keeps A2, A11, A13, A14 and A15, the night of the cold glow. A second agent takes
     A18, A19 and A20, the crossing set and the dawn, and owns `crossing.py`, `bluehour.py` and `sdfppl.py`.
  2. Fix per shot, testing locally at low resolution through renderq.
  3. FARM jobs.
* **Commits.** Use `python3 _local_logs/runA/commit_runA.py -m "msg" <paths>`. It commits explicit paths only, refuses an
  empty list, and splices only this RUN-A block of this shared NOTES, so other lanes' uncommitted lines never ride along.

## Previous RUN-A report (cloud lane, kept verbatim below; its "RESUME HERE" is superseded by the block above)

### Cloud lane report (27 Sep ~16:05Z, paused ~16:35Z)

#### PAUSED 27 Sep ~16:35Z (director: usage window end). Nothing is rendering. RESUME HERE
**State.** Everything in this lane is committed and pushed (7c7338c on `claude/v3-runA` and
`claude/run-a-crossing-greybox-ub8wbz`). THE CROSSING, FALSE DAWN (arc) and THE BLUE HOUR are ready to render on the
cloud; the jobs are written and not launched (the director launches): `crossing_a_1..8`, `falsedawn_a_1..4`,
`bluehour_a_1..5` (RENDER_SPEC below). H5 items 1-8 are DONE; item 9 (R16 THE WATCHERS) is not built.
**Next steps, in order.**
1. First `git pull --rebase origin claude/long-dawn-v2`; if the rebase rewrites this lane's commits, push with
   `--force-with-lease=<branch>:<old sha>` (both branches hold only this lane's commits on top of upstream).
2. If jobs were launched: read `cloud_logs/<job>_status.txt` for any silent job, then review the shipped jpgs in
   `renders/crossing_A`, `falsedawn_A`, `bluehour_A` (a contact sheet per shot) and fix anything that reads wrong.
3. R16 THE WATCHERS (backlit silhouettes only): build it in `shots/run/` from the same kit (`sdfppl.traveller`,
   `bluehour.sitter` and `cowl`, the crossing's keeper staging), write its job files and name its fallback.
4. Blue-hour polish if time allows: cloaks less like smooth clay (stronger fold normals and mottle), less rose on
   bar 74, the lantern ~0.1 m further right (it overlaps the right bearer's edge at the end), and the fine wavy lines
   in the snow of the lantern's pool.
**THE CROSSING** (`shots/run/crossing.py`, figures `sdfppl.py`), **FALSE DAWN** (`falsedawn.py`) and **THE BLUE HOUR**
(`bluehour.py`, new) carry every H5 call in RUN-A's area (checklist below) and are ready to render on the cloud:
finals numbered in A's cut frames, per EDIT-v3. Nothing is rendering; no job has been launched by this lane.

#### H5 checklist (director's H5 calls, 07:10Z)
| # | item | status |
|---|---|---|
| 1 | CROSSING: the uphill bearer's feet planted in frame; nothing hangs from the pole | **DONE.** Feet planted by a foot cycle on the real terrain (a planted foot never moves; two-bone IK legs; the pelvis drops where a leg could not reach). Opening re-staged side-on from just below the crest: both bearers in profile on the crest's skyline, boots on the snow; both hands steady the poles (a two-pole litter), so no arm hangs free below a cloak. |
| 2 | CROSSING: bearers as cloaked silhouettes with cloth and weight; no capsules in finals | **DONE.** `sdfppl.traveller`: hooded wool cloak (a "bell" primitive: folds deepening to the hem and swinging with the gait, wavy hem), sloping cape, loose sleeves, gloved hands, boots; bearers lean into the load on bent knees. Walkers in knee-length hooded coats (one in five a long cloak), packs under the cloth, staffs. Cloth keeps little light-wrap, so lights behind a figure rim it. |
| 3 | CROSSING: star trails concentric from their first frame, or held back until the wide | **DONE.** The sky turns from bar 66 b1; the trails open only on bar 69 b1, once the camera has settled with the pole in frame; arcs capped at 6 deg. |
| 4 | CROSSING: the locked bar map (A18, cut 4880-5840) | **DONE.** Draw-back from bar 63 b1, settled on bar 69 b1; sky from bar 66 b1 (T13); red steady until bar 70 b1, then out patch by patch; four watch-fires passed on bars 64, 67, 70, 73: the first on a shoulder just beyond the crest (the bearers pass in front of it as backlit silhouettes while its keeper kneels to feed it), the other three on pinnacles beyond the arete placed on the camera's line of sight through the lantern at their bar, so the lantern passes in front of each. No cold glow. |
| 5 | FALSE DAWN = ARC only; clouds with structure; one or two big landforms | **DONE.** Arc only, locked to A2 (the glow shows on bar 3 b1, the nearest stars go out from bar 5 b1). One great knife-edge sierra 9.5 km out (A's landform), left of the glow: an irregular crest, one dominant asymmetric horn, and the glow rises behind its right shoulder so its notches cast the rays. The deck is a structured mackerel sky in banks with clear gaps, lit softly through each cloudlet with brighter edges. Lenticulars were tried and CUT (at night, lit from beneath, a stack reads as a fleet of saucers). |
| 6 | Terrain tells out | **DONE** in this lane's shots. The vertical-stripe tell had two sources: (a) combed snow on steep flanks: gully detail halved, snow held on steeper ground, and an ADDITIVE `world.shade` flag `Q[18]` samples the fine snow noise in a height-skewed domain; (b) walls: `crag_row`'s cutoff radius is too short for elongated crags (the flank along the long axis is still ~200 m above the cloud where it is cut; the column marcher draws that wall as stripes). Fixed in this lane's rows (radius x2.5); worth checking in any other lane that uses `crag_row` with aniso > 1. Also additive: a safe early-out bound for crag/ridge rows (`CR[k,14] > 0`; 0 = v2). |
| 7 | Humans: silhouettes or gloved hands; never a lit face; hoods/cowls; >= 40 px | **DONE.** Every head is a closed cowl (no face exists); keepers kneel or stand in profile or on the camera side of their fires; walkers ~57 px in the wide at full res. |
| 8 | THE BLUE HOUR (R7): T14 over people together; the seated, unroped bearers and the set-down lantern in the foreground as the camera drifts to the hearth smoke; sun -6 to -2 deg, no disc | **DONE** (greybox verified at full res on T14's first frame; jobs written). A19 = cut 5840-6239 (bars 74-78). On a broad snow shoulder past the arete (the crossing's own set) the two bearers sit close with their backs to us, unroped: one's arm round the other, whose head leans on its shoulder; the great lantern set on the snow at their side lights their backs and a warm pool; the rope coiled, the poles laid down; the others of the line sit and stand in small groups, two watch-fires fed. The sun climbs -6.2 to -2.8 deg (no disc): the east pales to rose from bar 75 b1 under a deep blue sky, the stars fade. From bar 76 b1 the camera (a standing bearer's eye, a few paces behind them) drifts right and turns toward the hearth smoke rising from the valley through the cloud sea; the pair and the lantern slide into the lower-left third and never leave the frame through T14 (cut 6080-6200). |
| 9 | R16 THE WATCHERS: backlit silhouettes only | logged, not yet built. The kit: `sdfppl.traveller` + the crossing's keeper staging. |

#### Review files (`review/v3/`)
* `runA_crossing_h_keyframes.jpg`: six cut frames at half scale with the final code (bars 62, 64, 67, 70, 72, 73).
* `runA_crossing_greybox_q_12fps.mp4` + `runA_crossing_greybox_q_contact.jpg`: the whole take at quarter scale,
  12 fps (rendered 10:30Z, before the crag-radius fix: identical to the eye, no walls in frame).
* `runA_falsedawn_arc_swell.jpg` (cut 240, 380, 500 at half scale) and `runA_falsedawn_arc_full_f500.jpg` (the peak,
  full res). `runA_falsedawn_three_designs.jpg` is the original three-way comparison (pre-H5).
* THE BLUE HOUR: `runA_bluehour_T14_full_f6080.jpg` (full res, T14's first frame), `runA_bluehour_design_h.jpg`
  (cut 5840, 6080, 6239 at half scale) and `runA_bluehour_greybox_q_12fps.mp4` (the whole take, quarter scale, 12 fps).
* Earlier: `runA_crossing_wide_*` (main vs the fewer-larger fallback), `runA_underglow_run_demo.jpg` (the red
  under-glow in the Run's own world, for R3).

#### RENDER_SPEC (director launches; finals in A's cut frames, EDIT-v3 picks them up)
* THE CROSSING: `cloud/jobs/crossing_a_{1..8}.json`, 120 frames each, `python3 shots/run/crossing.py --range A-B
  --procs 4 --skip` -> `renders/crossing_A/f_04880`..`f_05839` (1920x804, ss 1.5, ship jpg), one machine type.
  Measured at half scale (4 threads): close-ups ~11 s, wides ~5 s, so full res ~6 min (close-ups) to ~3 min (wides)
  single-thread: ~64 worker-hours, about 2 h on 8 four-core boxes. Fallback: `--variant few` (12 larger walkers).
* FALSE DAWN: `cloud/jobs/falsedawn_a_{1..4}.json`, 120 frames each, `python3 shots/run/falsedawn.py --design arc
  --range A-B --procs 4 --skip` -> `renders/falsedawn_A/f_00080`..`f_00559`; ~1.5 min/frame single-thread at full res.
* THE BLUE HOUR: `cloud/jobs/bluehour_a_{1..5}.json`, one bar (80 frames) each, `python3 shots/run/bluehour.py
  --range A-B --procs 4 --skip` -> `renders/bluehour_A/f_05840`..`f_06239` (1920x804, ss 1.5, ship jpg), one machine
  type. Measured: 112 s/frame at full res on 4 threads, so ~7 min single-thread: ~47 worker-hours, ~2.3 h per job on
  a four-core box. Fallback: the design still (cut 6080) held as a slow push-in, or the crossing's final wide.
* Tests use `--numbering shot` (0-959 / 0-479) and an `--out` subfolder; the EDL reads only the folder's own files.
  (bluehour.py takes cut frames only; for more than one thread in one process set `NUMBA_NUM_THREADS` too.)
* Data that must be committed: `shots/run/crossing_fires.npy` (the island fires), the Run's `summits.npy`/`beacons.npy`.

#### How the pieces work (for whoever picks this up)
* Set: a knife-edge arete (8 `ridge_row`s) in a clear basin at world (-5200, 14400) between two faceted summits; the
  line walks east (yaw -30 = the world's sunrise azimuth); the camera stays on the south side, never ahead of the
  lantern. 40 roped walkers + 2 bearers; 0.45 m/s (a step per beat at 72 BPM). Two clocks: people in real time, the
  sky on a faster clock (46 deg, ~3 h): stars, the moon setting behind the arete, the watch-fires fed, 34 island
  fires, the red under-glow going out (`world.cloud_glow`, additive, orange-red: crimson over blue reads magenta),
  the lantern warming from ice-white toward gold.
* The blue hour reuses the crossing's set (`crossing.wfs()`), lantern, fires and `sdfppl`; its own parts are a sky
  kernel (deep blue zenith, rose held low in the east, no disc), a soft high rose key with blue skylight on matte
  snow, 56 hearth-smoke plumes (`dawn` puffs), and `sitter()` (seated, knees drawn up, a cowl that drapes to the
  shoulders: a hood on a neck reads as a pawn). ADDITIVE world flag: crag row col 15 in (0, 1] buries the summit
  boulders of a shelf crag under wind-packed snow (0 = v2); the shoulder uses 0.85.
* Lessons: background renders on the session box do not survive a usage pause (the box is frozen, then rebooted);
  use cloud jobs. In rebases, this shared NOTES file conflicts: keep each lane's block intact. In this world
  `cross(UP, w)` is a figure's RIGHT (sdfppl's comment says left; symmetric figures never showed it).

# >>> RUN-C (C . THE LIVING INK: C16 REVEAL, C17 INK RUN, C24 ILLUMINATION) - report 27 Sep <<<
## PAUSED 27 Sep ~16:45Z (director: usage window end). RESUME HERE
**State.** All H5 fixes are coded and pushed (ce69649 look; 0e404d8 driver fixes plus the 14 job files rewritten for
2x). JOB READY has NOT been sent yet. Two things run in the background (scratchpad `rc/`):
1. `gate_h5b.sh` (log `gate_h5b.log`): the boil gate re-run with a TRUE boiling control. The salt now shifts every
   hatch line's position (`_lines(..., shift)`), where before it only re-dealt ranks and dashes. Results go to
   `renders/run_c_tests/boil_check_{scroll,reveal}_h5.txt`. First pass (the old, too-weak control): anchored
   residual scroll 0.0178, reveal 0.0188. Pre-H5 it was 0.021-0.025, which passed, so the new hatching is at
   least as anchored. After that it re-checks the production driver on reveal 0, illum 2640 and scroll 168-169
   (the `--procs 2 --threads 2` spawn path). The first check crashed on NUMBA_NUM_THREADS; fixed in 0e404d8, the
   env is now set in main() before numba loads.
2. `crops.sh` (log `crops.log`; waits for (1)): three production-scale check crops (`--scale 1 --ss 2 --crop 0.5`,
   i.e. 1:1 pixels of the final frame): scroll 168, reveal 0, illum 2640 -> `renders/runC_{shot}/chk_crop/`.
**Next, on resume, in order.**
(a) Read `gate_h5b.log` (the anchored/re-dealt ratio should be well under 0.5; PASS if so) and `crops.log`. Look
    at the three crops at 1:1 (beacon glyph, her fire, the inked sun, the hatching, no jaggies). Use the aov/ink
    times in the logs (the scale 0.25 frame is ~1/16 of the full cost) for the cost estimate.
(b) Build the review sheet `_local_logs/review/runC_ink.jpg` (the pre-H5 one is kept in `review/h5/`): 3 frames
    per shot from the half-res tests (`inkpass.py --scale 0.5 --tag _x4`, AOV caches exist: scroll 168/232-239/300,
    reveal 0/108-115/200, illum 2410/2480/2560/2640) plus the three 1:1 crops. No boil heat map on it.
(c) SendMessage main: `JOB READY runC_reveal_{a-d}, runC_scroll_{a-d}, runC_illum_{a-f}: C16/C17/C24 ink at 2x,
    1040 frames, est. N min per job`, with the sheet path and the gate numbers. Then commit, clean up the tests,
    and move the NOTES report to final.
* Local renders are tests only; the Mac is at swap limit (0.8 GB free), so no full frames here.

## STATUS 27 Sep ~16:30Z: every H5 item in the ink pass is coded (commit ce69649); the gate re-run and job rewrite are next
The H5 calls as coded (`inkpass.py`), checked by eye on half-res stills of all three shots:
* **Hatching follows the form.** F1 is cross-contour: iso-lines of planes tilted 20 deg toward the camera
  (h - 0.36 chi), so every ring wraps its peak even from beacon height. F2 is a diagonal crosshatch (tilted
  planes along the shot's fixed right axis) and carries the shadows. The intervals are nested, and a line's rank
  comes from its birth level against the continuous scale, so lines fade and never pop. Tone comes from the MACRO
  form (normals smoothed at 40 page px, world footprint), so there are no vertical gully streaks. The contour field
  is smoothed at 2.5 px. The old fall-line dashes are deleted.
* **Outlines follow the light** (`line_mod`): heavy in shadow, thin with pen lifts where lit, fading into a fire's
  light, with world-anchored pressure and a 0.3 px soften before the 2x downsample. The reveal's near ridge no
  longer reads as a stock chart.
* **Beacons hand-inked** (`ink_flames`): a glyph of 3/5/7 fat tapering tongues leaning in a common wind, with crisp
  notches, curling tips and detached fragments. The outline is at the terrain's pen weight with lifts; nested
  inner lines appear only where the tongues part. The wash is gold-only, paler in a soft core and deeper at the
  root (no emoji red). Thin inked smoke threads rise from each fire. There is no halo. The catch blooms as a gold
  wash on the rock the fire lights (`fire_soft`: a smooth, world-anchored falloff with distance; the AOV's
  firelight banded on stratified rock). Far flames are drawn 7x life (FLAME_SCALE, the illustrator's licence).
* **C24 colour follows the light** (`illum_fill`). A feathered (~9 deg) wash floods out from the sun along the
  horizon first; brighter sky is reached sooner, and page-fixed irregularity is added to the edge. Cool hues are
  laid as lilac, because blue on this yellow paper turned green. Only sunlit land and lit cloud take colour, and a
  faint cool wash reaches the shadow faces late. There is no mask edge anywhere. **The sun is ink** (`sun_ink`):
  an uneven pen circle with a lift, a pale gold wash pooled at the rim, and 44 fine broken rays.
  **Smoke rises from every beacon summit** as inked strands (`ink_plumes`, following dawn.py's plume columns).
* **2x:** `ink_final.py` now defaults to `--ss 2.0` (drawn at 3840x1608, INTER_AREA to 1920x804).
  `render_ink.render_aov` no longer renders the gouache fire or the volumetric smoke layers.
* Disk: `renders/run_c_tests` was trimmed to 24 AOV caches plus the current test stills (director's disk emergency).
* **Cloud jobs rewritten for 2x** (NOT yet announced): `cloud/jobs/runC_reveal_{a-d}` (60 f each),
  `runC_scroll_{a-d}` (80 f each) and `runC_illum_{a-f}` (80 f each, dawn frames 2398-2877 = C24 local 0-479).
  Each job is one 4-vCPU box running 2 workers x 2 numba threads (`--procs 2 --threads 2`, ~2.5 GB per worker), with a
  numba warm-up in setup. `ink_final` prints an aov/plate/ink time breakdown per frame (it shows in the status file).
Next: the boil gate on scroll 232-239 and reveal 108-115, plus a small-scale `ink_final` end-to-end check (chain
`scratchpad/rc/gate_h5.sh`, log `gate_h5.log`). Then the review sheet and JOB READY.

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

# >>> RUN-B STATE AT PAUSE 2 (27 Sep ~16:40Z; RESUME 20:00Z) <<<
**Nothing is rendering locally. No cloud job running for RUN-B.**
**DONE:** `renders/dusk_B` = DUSK v1, 640/640 frames (cloud job dusk_b, relaunched after my --skip fix 68e0f72);
smooth, no flicker. The director approved the landform and asked for a DUSK v2 (below).
**Code state (all committed in this pause's commit; bworld VERSION 'b11'; caches keyed by it):**
* `bworld.py` b11: soft-ridged field + layered range rings (9.5-63 km) + a sunrise basin ESE + rough worn crests
  (outcrops/tors on the upper ground, never needles) + her massif/knoll/NE ridge + the sunrise shoulder (13 km,
  az 118, TOP_Y-100). Snow: slope-based, stony summit boss within ~12 m of the top, wind-packed snow below with
  sparse stones. Cloud: broad-billow normals, wide wrap. `add_band` = the Milky Way (wheels with the stars).
* `bset.py`: set points (BEACON, CAIRN 5.5 m NNE-ish of her seat, KNEEL, STAND, SEAT, LIP), the vigil camera,
  the NE path, puppets: `person()` (hood/cowl, red woven shawl, long cloak, staff, reach), `child_asleep()` (a
  hooded lump wrapped in the shawl; the old two-circle child read as a SNOWMAN), `rubble_cairn()`,
  `beacon_base()` (rough stone plinth: the old coursed base read as a BRICK BBQ), `fire_steel()`, `night_params()`.
* `dusk.py` (v2 WIP, untested since dusk_B): stronger aerial perspective (fog 1.15e-4 / 3600 m), `wisps()` a thin
  streaky cloud layer just above the cloud sea (rose while the sun reaches it), knoll-top glint (several px).
* `handback_b.py`: crane 3840-4079 (per-frame build; +40 m out/+20 m up/+18 deg hfov at mid) + settled 4080-5199
  (relight; push-in 4800-5000 to hfov 30; hands open 4840; hand-off 4960-5040 with a warm glint on the steel;
  child wakes 5040). Tests OK (renders/run_b_tests/hbb_t9 = latest): story reads, sun tamed, fire dies to embers
  in the sun. OPEN ISSUE: a smooth mid-distance dome (the field's island at az ~122, ~7.5 km, top ~-200) reads as
  a sand pile under the shoulder -> deepen/shift the sunrise basin toward (az 122, 7 km), keep a few islands.
* `vigil.py` (1520-3839): tested at quarter/half (vig_t3 look approved by me): moonlit snow dome, fire + her on the
  skyline, storm squall (fine streaks, grey sky), fog hours, first answer bar 35 (she stands at the lip and sees
  it), travellers + thread of torches, villages, the child 46-48. Her moves are a walked track (no teleports).
* `reveal.py` (1360-1519): close behind her at the roar -> pull back 520 m WNW, level with the top. OPEN: the fire
  must read as a warm point at the end (add a distance-sized glow), Milky Way stronger (0.12), the summit
  silhouette is a smooth dark double dome (acceptable; could move the end camera to side-light it).
**Jobs written, NOT yet READY (send JOB READY + 2-3 check stills each after the fixes):**
`cloud/jobs/handback_b_settled.json` (4080-5199), `handback_b_crane_a/b.json` (3840-3959/3960-4079),
`vigil_b_a/b.json` (1520-2679/2680-3839), `reveal_b_a/b.json` (1360-1439/1440-1519). All render into
renders/{handback_B, vigil_B, reveal_B} in B frame numbers with ship jpg.
**NEXT on RESUME (in order):** (1) fix the mid dome (basin), half-res check stills 4300/4880/4976 -> JOB READY
handback_b_settled + crane_a/b. (2) vigil: half-res check stills 2060/2740/3700 -> JOB READY vigil_b_a/b. (3) reveal
fire glow + band -> stills 1360/1440/1519 -> JOB READY reveal_b_a/b. (4) DUSK v2 test (quarter, then full-res
f0/f320/f590) -> write dusk_b2.json (same as dusk_b but out renders/dusk_B2? ask the director whether to overwrite
dusk_B) -> JOB READY. (5) THE CLIMB wide (bars 9-11, HEROINE + RUN-B fallback: her figure on the NE ridge carrying
the pot's glow) if time allows. Status of cloud jobs: `gh api 'repos/mishmellow123-cmyk/mishamisha/contents/
the-long-dawn/cloud_logs/<job>_status.txt?ref=claude/render-<branch>' --jq .content | base64 -d` (never fetch
render branches: a fetch left a 1.4 GB temp pack).

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
