# >>> RUN-A (A . FALSE DAWN R1 · X2 DARK ADAPTATION · R2-A REVEAL · R3 BEACON RUN · R16 WATCHERS · R6 THE CROSSING · R7 THE BLUE HOUR · A20 title sky) <<<

## RUN-A2 STATE AT PAUSE (27 Sep ~20:55Z, the account's usage window; RESUME ~23:30Z). A18 THE CROSSING, A19 THE BLUE HOUR, A20 title sky; owns crossing.py, bluehour.py, sdfppl.py, crossing_fires.npy
* **RESUME HERE.**
  1. `python3 the-long-dawn/cloud/farm.py status`: did the director launch `crossing_a2_01..18` (sent JOB READY
     ~20:52Z)? If approved and not launched: `farm.py the-long-dawn/cloud/jobs/crossing_a2_{01..18}.json --nodes 3`.
     If frames have landed in renders/crossing_A: contact-sheet them (bars 62, 64, 66, 69, 70, 73) and check the trails
     (concentric, never dashes), the red going out from bar 70, the keeper's kneel, no flame spikes.
  2. Look at `renders/_farmtest/crossing_a_lookdev3/` (pass-3 code: kneeling keeper, less flame soot, lens blur on
     the far background in the close-up, one-piece cowl, stronger far red) and `renders/_farmtest/bluehour_a_lookdev2/`
     (the FIRST look at the rebuilt blue hour; 5840, 5960, 6080, 6200, 6400 at half res). Fix, re-test, then JOB READY
     `bluehour_a2_01..20` (cut 5840-6479 incl. A20; ~33 min each at an estimated 500 s/frame single-thread).
  3. Re-cut the jobs if measured costs differ: `python3 the-long-dawn/cloud/jobs/make_runA2_jobs.py --x-spf S
     --b-spf S` (it deletes and rewrites crossing_a2_* / bluehour_a2_*). Note: the first farm timings included numba
     compiles (world.py changed), so they overstate; a warm node's `s/frame` line is the number to use.
* **Committed + pushed** (46153c1, b8964e7, 297446f, dffa1bf, a92fde6, 5cde879, 3461166, 1fd0716):
  * `sdfppl.py`: v3 CLOTH (default; `cloth=False` = the v2 figure): elliptical cloak and cape bells, creased uneven
    folds, wind billow + lee hem lift + flutter (`wind` = world vector toward the lee, |v| = hem billow in m), SDF crease
    occlusion, a back-lit wool rim, fibre-streak albedo, one deep cowl (no brow bump: it read as a face). `seated()`:
    six poses (knees, cross, back, side, kneel, lie), lean adult proportions: FOR ANY LANE's seated figures (RUN-A3).
  * `crossing.py`: headwind down the arete with travelling gusts; trails open with the wheel (bar 66 b1) and grow to
    36 deg concentric arcs about the pole (LAT 6, the pole ~1/4 down the wide); the great knife-edge range 7 km north
    under the pole (GR_KNOTS; a far range was tried and cut: a pyramid, and it hid the sea); haze 1e-4; far red patches
    x2.6 so the red is there to go out; opening camera H0 = -0.45 (the uphill bearer's boots on the snow, full-res
    check PASSED); near keeper = `seated` kneel; fires lean -0.15, soot 0.22; `lens_blur` (focus on the lantern).
  * `world.py` ADDITIVE (one comparison per call for every other table): -5 `hole_row` and -6 `valley_row` rows (read
    only while they LEAD the table), Q[19] snow line (fields and woods below), fogp[8] low-mist cap.
  * `bluehour.py` (A19 + A20, 640 f): ~42 seated people round the set-down lantern, all facing the dawn; a glacial
    valley carved toward the rose (VAL_KNOTS 5.6-34 km) seen through the thinning cloud (a cloudless sub-window pass
    composited under torn cloud), a river, ~54 hearth columns and glows; the gathering placed from the shoulder's
    measured edge (`edge_f` = 19 m); CAM_KEYS (the camera rises 1.6 -> 2.6 m as it drifts from bar 76, so the valley
    opens past the edge; A20 lifts to the rose for the title at y 360). The first farm test crashed (fixed: per-disc
    depth bias in `_splat_over`); no picture of it exists yet.
  * Workers print a frame's full traceback (farm logs showed only the pool's re-raise).
* **Risks to check first on resume:** the blue hour's valley may not show (the carve is under a sea of peaks; the
  sub-window composite is untested in a picture); the people may still read plush at full res; A20's sky composition.

## RUN-A3 STATE AT PAUSE (27 Sep ~20:40Z; the account's usage ends ~21:25Z, RESUME ~23:30Z)
* **A13 R2-A THE REVEAL: APPROVED and RENDERED** (`reveal_a_1`: 120/120 in `renders/reveal_A` at 20:46Z, QC'd: no gaps,
  brightness smooth, max frame-to-frame jump 0.36/255). Continues H1's take in H1's own renderer (continuity 3680 vs
  h1_v3h5 1555 = 0.64/255).
  **Pending (director's note, cheap re-render, ~26 min on one node):** the cloud sea reads as a flat snowfield. RUN-A-L's
  knob grid was inconclusive (its read: geometry, not shading); the fix after 23:30Z is RUN-A-L's additive cumuliform
  relief for the A night (a leading CR row adding 25-40 m cauliflower tops at a 90-150 m scale to h_cloud_cr). When it and
  any `nighta.CLOUD` values land: A13's world layer takes CR from H1's s1 world (`Wd['CR']`, empty), so ADD the relief row
  there (stack it onto `Wd['CR']` in `reveal_a.world_layer`, faded in with the night ramp or from the first frame if it is
  invisible at H1's distance), check 3680 against 3679 again, then delete `renders/reveal_A` and re-run
  `farm.py the-long-dawn/cloud/jobs/reveal_a_1.json --nodes 1`. The re-render also carries nighta 2913ad7 (the far haze
  without the glow's shadow rays, which fell as faint vertical slabs under the horizon; the landed final predates it).
* **A15 R16 THE WATCHERS: look-dev v2 on the farm** (`watchers_a_look`, 3 frames -> `renders/_farmtest/watchers_a_look`);
  the final job `watchers_a_1` (160 frames) is written and pushed, NOT approved. v1 failed at 1:1 (the lit ledge read as
  clay, the flame floated over a clip-art wood teepee, the lighter stood beside the fire). v2: the seventh fire on a
  dry-stone beacon cairn (5 courses, ~1.02 m); the lighter almost in front of it (a backlit silhouette, the flame above the
  hood); `nighta.POOL_I` 14 -> 6; eye 1.9 m (fires 1 and 3 in sight); a 3.5 m push easing in from A14's hold; A14's catch
  flares drawn after the fires. Plate = A14's END_CAM: (-999.58, -192.88, -523.22), yaw -23, pitch -1.8, hfov 40.
  **Next:** review the v2 look-dev (sheet + 1:1 crop of the lighter and the cairn). If good: JOB READY watchers_a_1 with
  check stills; if the near ground still reads as clay, the fallback is to drop the plate's lower third (pitch up, the
  ground to the frame's edge) or to let the push carry the lighter past the ground faster.
* **Shared kit `nighta.py`:** night_light() (world.night_light + TERRAIN + CLOUD, `NIGHT_CLOUD` env override), the cold glow
  (I0 0.14, breathing once a bar), the red under the cloud (150 world-fixed patches, beat pulse), fires (`POOL_I` 6).
  A14 (`beaconrun_a.py`, RUN-A-L) imports it, plus `watchers_a.plate/draw_figures/extra_fires`.
* **Check stills:** `_local_logs/review/runA/runA3_reveal_*.jpg`.

## RUN-A-L STATE AT THE USAGE GAP (27 Sep ~20:45Z; resume ~23:30Z). RESUME HERE
**Mine:** A2 FALSE DAWN (`falsedawn.py`), A11 X2 (`stars_a.py`) and A14 BEACON RUN (`beaconrun_a.py` +
`beaconrun_a_chain.npy`). The final farm jobs are one per shot, `falsedawn_a`, `stars_a` and `beaconrun_a`; launch
with `farm.py <job> --nodes 3`, ONLY after the director approves JOB READY. The farm requests are in
`~/.cache/ldfarm/req/`; check `farm.py status`.
* **A11 X2** (cut 3120-3359 -> `stars_A`):
  * v1 look-dev: the sequence works (ember, a few stars, more, then the Milky Way). But the band read as smoke over a
    sparse field.
  * v2 (7d5c4d9) adds 90k faint stars on the galactic plane and lowers the smooth glow to 0.06. Its look-dev is
    request `...starsalook-71354`.
  * The ember starts at EMBERS' measured centroid (958.6, 547.5) at 3120 and eases to (960, 548).
  * NEXT: check the v2 sheet, then JOB READY `stars_a` (1 node, cheap).
* **APPROVED and RENDERING:** `falsedawn_a` (main launched it with --nodes 6; log `_local_logs/jobs/falsedawn_a_farm.log`).
  `stars_a` is JOB READY (sent 20:48Z).
* **Post-23:30 list (director + mine), cheap re-renders:**
  * A2: skew and notch the left sierra's highest spire (x ~430 at full res, a near-symmetric "perfect triangle").
  * A2: vary the widths and lean of the foreground needles (lower centre), and add a broken crest or two.
  * A2: the mackerel deck reads as flat streaks.
  * A2: the cloud-sea strip at the bottom is dark.
  * THE CLOUD SEA for all A nights (see below).
* **A2 FALSE DAWN** (cut 80-559 -> `falsedawn_A`):
  * In: FD_WALL v2 (a fractal far crest, 46 km, az -15). A thin moon key (FD_MOON=1: az -59 el 21, I 0.28) for
    moonlit snow and aerial depth. glow_haze (0.25 share, no rays: the rays made vertical slabs, fixed in 049265c).
    Anti-streak snow.
  * The A/B test showed the moon version has far better depth than the glow key; the slab bug was then fixed.
  * Re-test: request `...falsedawnalook-72930` (380, 500).
  * NEXT: if clean, JOB READY `falsedawn_a` (--nodes 3).
  * Still weak: the mackerel deck reads as flat streaks, and the strip of cloud sea at the bottom is dark and flat.
* **A14 BEACON RUN** (cut 3920-4239 -> `beaconrun_A`):
  * The chain is fixed (`beaconrun_a_chain.npy`: the seventh on RUN-A3's brink). Near fires 1/3/5 are size 1, at
    1.07 km, 530 m and 190 m. Far fires 2/4/6 are great beacons (2.7), at 18.5, 9.3 and 5.2 km. Fire 7 is a hearth
    (0.6) on RUN-A3's cairn.
  * The camera glides from 170 m back / 50 m up (hfov 34, pitch -4, yaw -26) into A15's plate
    (watchers_a.plate(CHAIN[6])) by 4230, then holds.
  * Each catch flares (draw_flares). The lighters come from watchers_a.draw_figures (RUN-A3).
  * The v1 look-dev failed at full frame: 2-5 px fires, and a flame towering over the seventh watcher. v2 (7d5c4d9)
    fixes both; its look-dev is request `...starsalook-71354`.
  * NEXT: check the v2 sheet and a low-res motion test, then JOB READY.
* **THE ONE A NIGHT CLOUD SEA** (director): the knob grid was INCONCLUSIVE, because its camera looked into a crag. The
  knob sets barely change the look. I told RUN-A3 to keep CLOUD at the world values and not to re-render A13 yet.
  * The likely fix is geometry: an additive cumuliform relief for A's cloud sea. That is a leading CR row, like RUN-A2's
    hole rows, which h_cloud_cr applies: 25-40 m cauliflower tops at a 90-150 m scale. Add a softer-wrap override if
    needed.
  * Test it from an A13-like high view over the sea. Then set nighta.CLOUD and re-render A13, A14 and A2 once.
  * `nighta.night_light()` (RUN-A3) holds the CLOUD knobs. RUN-A-L owes
  RUN-A3 the final values from the knob grid (`beaconrun_a_cloud`, request `...beaconrunacloud-64823`: 6 static
  views 3920-3925 with NIGHT_CLOUD sets). The sets are listed in the job file. A13 will be re-rendered with them.
* **Helpers:** `_local_logs/runA/pick_chain.py` (the chain picker), `_local_logs/runA/commit_runA.py` (commits).

## SPLIT (director approved ~19:05Z; RUN-A3 split off ~19:15Z for A13 + A15)
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

# >>> RUN-C (C . THE LIVING INK: C16 REVEAL, C17 INK RUN, C24 ILLUMINATION, R13b RING FALLS) - report 27 Sep <<<
## >>> STATE FOR RESUME (RUN-C-3, 27 Sep ~20:50Z; usage window ends 21:25Z, resume ~23:30Z) <<<
* **C16/C17/C24 FINALS: APPROVED + RUNNING on the farm** (director launched all 45 runC jobs, `--nodes 15`; log
  `~/mishamisha/_local_logs/jobs/runC_finals_farm.log`; frames land in `renders/runC_{reveal,scroll,illum}/`).
  At 20:32Z 286/1040 landed, no failures. All three final-code checks reviewed CLEAN (reveal 114, scroll 236,
  illum 2640; also scroll 240 catch + illum 2490; scroll 256-295 frame-to-frame diffs smooth, no pops).
  ON RESUME: `grep -c` the log for GAVE UP / failed; count landed frames (reveal 240, scroll 320, illum 480);
  any missing -> `python3 the-long-dawn/cloud/farm.py <job>.json --missing --nodes 1-3` (after asking main).
* **R13b THE RING FALLS part b (C 2840-2959): BUILT, look-dev on the farm at 20:48Z.** New driver
  `shots/run/ringfall_ink.py` (eb22781) + jobs `cloud/jobs/runC_ringfall_{a,b,c}.json` (40 f each ->
  `renders/runC_ringfall/`). Test frames 2850/2885/2899/2903/2935 land in `renders/_farmtest/runC_ringfall_*`
  (log `scratchpad/rc4/farm_rf.log`). If JOB READY was not sent before the window closed: look at those, fix,
  push, send JOB READY runC_ringfall_{a,b,c}. Design: keeper's summit at night (cool multiplied wash, moon out of
  frame upper left per E13a), the cairn drawn in ink at keeper.CAIRN, the Ring a drawn falling star (gold head,
  tapering broken tail between two fine lines) on an even screen path (FALL z=200 ease 0.8), strike on 2900 beside
  the cairn: gold bloom (0.5 s), 7 short sparks (2 f), steam plume (pale strands) leaning downwind to 2959.
  Hand-offs: IN from EMBERS-C2 E13a (spark falling down-right, cold moon upper left); OUT to MONTAGE-3D-4 C14
  find_a (a Blender macro of the band in a clean snow hollow; only the read must match).
* FREEZE until the C16/C17/C24 finals have all landed: no pushes to inkpass/render_ink/ink_aov/ink_final.
* Polish list for after the finals (cheap re-renders): the mid-size fire's slight crown silhouette (reveal
  ~100-200), the leftmost short tongue a lobe; C17 wash patches; triangle summits accepted by the director.
## (older) NOW (RUN-C-3, 27 Sep 20:20Z): JOB READY SENT for the 45 ink jobs (runC_reveal_a..o, runC_scroll_a..j,
## runC_illum_a..t; 1040 frames; code = branch tip, last RUN-C push 60dffb0). Launched ~20:25Z.
* Sheet: `_local_logs/review/runC_ink.jpg` (built by `scratchpad/rc3/sheet4.py` from `renders/_farmtest/`).
* Gate PASS on the farm: 0.336 scroll / 0.341 reveal. Farm cost 34/36/42-44 s per frame per process (illum/scroll/reveal).
* FREEZE: no RUN-C render-code pushes while the finals run (nodes fetch the tip before every unit).
* Pending: final-code check frames reveal 114 / scroll 236 / illum 2640 (`rc3/farm4.log`) -> look, flag to main.
* **NEXT ITEM (director 20:30Z): R13b THE RING FALLS, part b - NOT BUILT.** C13 bar 36 b3 -> end of bar 37 = cut-C
  2840-2959 (120 f): the streak 2840-2899, the strike on 37 b2 = 2900 "in the snow beside an old cairn", flash,
  hiss, steam plume to 2959. IN (EMBERS-C2 E13a, `shots/embers/c_fall.py`, NOTES l.76): at 2840 a small warm spark,
  a few px, short trail, falling down-right of centre, nothing else warm, moonlight cold blue-white from the upper
  left. OUT (MONTAGE-3D-4 C14 find_a, `shots/montage3d/ringc.py` `_find_a`): a Blender macro close-up of the band
  in a clean snow hollow, low cold moon (15 deg, behind right), strike-2's flash just off frame upper left; not in
  the keeper world, so only the read must match (snow on HER summit, a cairn by it). Plan: a NEW driver file
  (never touching the running finals' files) on the same ink pass at night (cool wash, moon upper left): keeper's
  summit (C16's) with keeper.CAIRN; the Ring a drawn gold head + tapering broken tail smeared per shutter sample;
  a brief gold bloom on the snow (no starburst), an inked steam plume (ink_plumes' strands). Farm look-dev ->
  JOB READY (3-4 jobs). MONTAGE-3D-4 not reachable by name: asked the director to relay.
* Next polish (after the finals land, re-render only what changes): C17's sepia wash patches still read as soft
  blobs at full-frame size; triangle summits (shared terrain); the leftmost short tongue on near fires is a lobe.

## STATE AT HANDOFF (RUN-C-3, 27 Sep 18:40Z) - live status below it
Inherited: H5 fixes pushed (ce69649, 0e404d8); flicker gate PASSED (anchored/re-dealt 0.380 reveal, 0.395 scroll);
three 1:1 check crops rendered (`renders/runC_{scroll,reveal,illum}/chk_crop/`), not yet reviewed; 14 job files
written for 2x, JOB READY not sent. Orders: review the crops and fix what they show; re-split the jobs to <= 45 min
each (more jobs are fine); JOB READY with crops + job list; then keep polishing.
**What the 1:1 crops showed (18:45Z):** (1) the flames still read as glossy vector icons: a modelled highlight blob,
a clean heavy outline, a fan of straight spikes (a crown), detached gold droplets; her fire's inner lines zigzag;
smoke threads run ~3 flame-heights up as long straight scratches. (2) A BUG in the reveal: on grazing far crests
the AOV ray march hits in some columns and misses in the next (land 4.3 km / cloud 11.4 km alternating in 1-4 px
runs), and the outliner draws the comb as organ-pipe bars ~50 px tall (reveal 0 and 112, left of centre).
(3) Hatch lines carry a pixel-scale sawtooth tremor at 1:1 (iso-lines of a fractal height smoothed only 2.5 px).
(4) C24: the plumes are dark staple/crack shapes (two strands either side of a thin column); the sun's heavy dark
ring reads as an icon.
**Fixes (inkpass.py, 18:50Z):** `deteeth` (a (2w+1)-tap running median of depth per row, w = 2.5 kpx; outliers take
all channels from the nearest median-depth pixel; nothing nearer than 300 m touched) at the top of compose; the
woodcut flame (full-bodied tongues with long hooked tips, one leading, flat gold wash pooling at the rim with the
paper's grain, no highlight, no droplets, per-tongue flow lines, pen pressure varies, far flames lighter/browner by
the fog); smoke curls ~1 flame-height; HSMOOTH 2.5 -> 5; thin plumes one pale strand; the sun's ring finer/lighter.
**Round 2 (19:05Z, after crops chk_crop2):** the bars were gone, but the median had filled the densest comb solid
into a flat-topped CHIMNEY on the ridge (a building read) -> `_deteeth` now ERODES first (a sliver <= w px nearer
than both neighbours takes the far side's data), then the median fills thin far slits. The hatch tremor survived
HSMOOTH=5: its source is the hit point's plan position (x, z) jumping pixel to pixel on rough rock (the
cross-contour field inherits it) -> `_surf_blur`: the line-placing fields (height, x, z) smoothed along the surface
(depth-aware, never across a silhouette) at HFIELD_SIG=1.5 page px before `xhatch`. Flames: S-curve and sway
stronger (the leading tongue was a straight horn), wash warmer. Plume strands give out before the plume bends over
(the two strands joined in a staple). Wash ramp is now a smoothstep with module constants WASH_LO/HI/BLUR (A/B on
scroll 232: 0.30/0.85/2.5 vs 0.40/0.62/1.2, the soft sepia blobs read as smudges at full-frame size).
~~Test chain `scratchpad/rc3/t3.sh`~~ (killed unrun: director FARM READY 19:20Z; the local queue is for light work).
**19:28Z: pushed f8f1fd8 (code + 45 jobs + runC_gate + runC_washB); farm look-dev running** (log `scratchpad/rc3/
farm1.log`): full production frames reveal 0/114, scroll 168/236, illum 2480/2640 -> `renders/_farmtest/runC_<job>/`;
the flicker gate as one unit (`shots/run/runc_gate.py`: AOV caches -> ink _g + re-dealt control -> ink_check -> a
report-card FRAME `renders/_farmtest/runC_gate/gate_report/f_00000.jpg`, because the farm returns frames, not logs;
plus its composites 108-115 / 232-239); wash B (RUNC_WASH=0.40,0.62,1.2) on scroll 236 vs the default.
Farm usage: `python3 the-long-dawn/cloud/farm.py <jobs...> --test 1 --frames <list> --nodes 3` from ~/mishamisha.
**FLICKER GATE on the farm (19:31Z, code 56d16cc = f8f1fd8 + others): PASS, better than before.** scroll 232-239
anchored 0.0149 / re-dealt 0.0442 = **0.336** (was 0.395); reveal 108-115 anchored 0.0169 / re-dealt 0.0495 =
**0.341** (was 0.380). The surface-smoothed line fields steadied the strokes. (12 s/frame on the node at half scale.)
**Round-1 production frames (in `renders/_farmtest/runC_r1/`): reviewed 19:40Z.** Comb/chimney gone, crests clean;
hatch lines smooth pen curves (no sawtooth). Farm cost: reveal 42-44 s, scroll 36 s, illum 34-36 s per frame (1
process, 2 threads, cpu-8). **FAIL: her fire at reveal f0 (production size, ~500 px at 2x) read as a DUCK'S HEAD**:
short fat side tongues + the base ellipse merged into one square body, the leading tongue a horn, a hooked tip
closed a loop that read as an eye; flow lines crossed in an X. -> **bonfire v3 (8ee58fc)**: `flame_tongues` (n
slender tongues spread +-0.26 H, side tongues 0.40-0.88 H tall, gentle bends, no hooks), `flame_sdf` returns outer
+ inner flame (same tongues at 58% x 62%: the illustrator's flame within a flame; chunked for near fires),
`_fill_holes` (enclosed pockets filled, no rings), `flame_glyph` draws every fire (ink_flames + flame_sheet.py).
Wash A/B (scroll 236): B (0.40/0.62/1.2) firmer and less smudged -> adopted as default. Known, not fixed: some
scroll summits are near-perfect triangles (the Run world's terrain; shared with A, not mine to re-terrain tonight).
**Round 2 (bonfire v3, 19:57Z, `_farmtest/runC_r2/`):** her fire at reveal f0 reads as a bonfire (slender tongues,
inner flame, flat warm wash, pen line); glyph sheet good from 11 px to ~380 px final; the near ridge in C16 is thin
and broken (no stock chart). At mid size (reveal 114, ~55 px) the outer tongues splayed like petals (a crown) ->
**60dffb0**: one common wind (lean 0.09-0.15 H), splay 0.45 -> 0.22, downwind tongues up to 30% taller; wash B
default. Round 3 check (sheet, reveal 114, scroll 236, illum 2640) = `rc3/farm4.log`.
**FREEZE RULE once the finals launch:** farm nodes fetch the branch tip before EVERY unit, so any push to inkpass.py /
render_ink.py / ink_aov.py / ink_final.py while runC_* finals run would mix two looks inside one shot. Polish goes
into a local branch or waits until the finals have landed (then re-render only what changed).

## PAUSED 27 Sep ~16:45Z (director: usage window end). (superseded by the block above)
* **Background finished (noted, not reviewed).** Gate: MEAN anchored 0.0188 | re-dealt 0.0494 | screen-fixed 0.0397 | anchored/re-dealt 0.380;MEAN anchored 0.0178 | re-dealt 0.0450 | screen-fixed 0.0645 | anchored/re-dealt 0.395; Driver checks: GATE_DONE;reveal 0 2.8s (aov 2 plate 0 ink 1);illum 2640 2.0s (aov 1 plate 0 ink 1);scroll 169 3.0s (aov 2 plate 0 ink 1);scroll 168 3.0s (aov 2 plate 0 ink 1);FINAL_CHK_DONE; Crops: scroll 168 7.5s (aov 5 plate 0 ink 3);reveal 0 14.1s (aov 6 plate 0 ink 8);illum 2640 9.5s (aov 4 plate 0 ink 5);CROPS_DONE;
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

**RENDER_SPECs (cloud; the director launches).** One driver renders a final frame from scratch (AOVs at 2x -> ink
drawn at 3840x1608 -> INTER_AREA to 1920x804; no caches): `shots/run/ink_final.py`. Fresh box: numpy, numba 0.67,
scipy, opencv-python-headless; files: `shots/run/{ink_final,render_ink,ink_aov,inkpass,keeper,world,run,rcam,pipe,
beacons,fire2,dawn}.py`, `shots/run/{summits,beacons}.npy`, `shots/montage/` (s1_peak, mt, common), `lib/look.py`.
Each job = one 4-vCPU box, 2 workers x 2 numba threads (~2.5-3 GB per worker at 2x), numba warmed in setup (~3 min).
**Re-split 27 Sep 19:10Z (RUN-C-3) to <= 45 min per job: 45 jobs** (the old 14 are overwritten; names a..o/j/t).
Cost basis: 1:1 quarter-frame crops at 2 threads on this (loaded) M2: scroll 13.1 s, illum 16.1 s, reveal ~25 s
-> full frame per 2-worker box ~26/32/50 s on M2 cores; assumed 2x slower on a cloud 4-vCPU box -> ~52/64/100 s.

| shot | frames -> folder | cut C frames | jobs | frames/job | est. per job |
|---|---|---|---|---|---|
| C16 THE REVEAL | 0-239 -> `renders/runC_reveal/` | 3600-3839 | `runC_reveal_{a..o}` (15) | 16 | ~27 min + ~5 setup |
| C17 THE LIVING INK RUN | 0-319 -> `renders/runC_scroll/` | 3840-4159 | `runC_scroll_{a..j}` (10) | 32 | ~28 min + ~5 setup |
| C24 THE ILLUMINATION | 2398-2877 -> `renders/runC_illum/` (local = frame - 2398) | 5680-6159 | `runC_illum_{a..t}` (20) | 24 | ~26 min + ~5 setup |

Job command (per block): `python3 shots/run/ink_final.py --shot <shot> --range <a>-<b> --procs 2 --threads 2 --skip`
(branch `claude/render-runC-<shot>-<letter>`, ship jpg). Regenerate the split with any block size (even blocks,
replaces every runC_*.json): `python3 shots/run/runc_jobs.py <reveal> <scroll> <illum>` (today: 16 32 24).

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

# >>> RUN-B-3 STATE AT PAUSE (27 Sep ~21:15Z, account usage ends; RESUME ~23:30Z) <<<
**Lane:** RUN-B-3 = THE HAND-BACK (3840-5199) + DUSK v2. RUN-B2 (agent adab49d5d24d90ffb) owns THE VIGIL + THE REVEAL.
**Code (all pushed; last edfa370):** bworld b15 (sand-pile fix); handback_b.py (person2 + sleeve_arm arms,
child3, cairn3, all via bfig.render; vigil join: fire light, vigil BAND/BAND_GAIN, STAR_GAIN, horizon_match,
finish_at eases vigil.FINISH -> ours by 4160; child_pos()); NEW bfig.py (B's shared figure light: rim only on
sun-facing edges = the director's "kill the uniform halo"; red wool translucency; cloak folds; STONE primitive;
snow caps) and bprops.py (cairn3, child3), shared with RUN-B2 (it adopts them in the vigil/reveal).
**Jobs (cloud/jobs):** handback_b_dawn_a (4240-4719), handback_b_dawn_b (4720-5199) = JOB READY sent 19:5xZ, then
the director HELD them for the figure pass (done in edfa370); handback_b_night (4080-4239), handback_b_crane_a
(3840-3959), handback_b_crane_b (3960-4079) = HELD until the 3839/3840 join matches RUN-B2's vigil still.
Each ~18-30 min on 1 cpu-8 node (farm: `python3 the-long-dawn/cloud/farm.py <job.json> --nodes N`).
**Look-dev:** farm test farm_hbb_t3 (3840/4300/4880/4976/5100 -> renders/_farmtest/handback_b_*) was queued behind
~64 units at 20:35Z; local half-res hbb_t15 (same frames) running. See the live status line below for the verdict.
**NEXT on RESUME:** (1) review hbb_t15 / _farmtest stills (figures: human silhouettes, child reads red, no halo,
cairn stones with snow caps) -> JOB READY / launch dawn_a+b if approved; (2) match the join with RUN-B2's 3839
still -> JOB READY night + crane_a/b (re-check the crane's first bar: RUN-B2 moved the vigil camera to WSW
looking ENE, CAM_BEAR 62 / YAW 70; the set points are frozen on the old axes); (3) DUSK v2 into renders/dusk_B2
(director's three notes: scale via haze + 3-4 layered far ridgelines + rock bands on her massif; the flat blue
valleys must read as a textured cloud sea with wisps; the last red point several px, clearly rose). Gate any
terrain change for DUSK behind DUSK-only switches (P[3] cloud relief; a DUSK-only CR row) so approved hand-back
jobs never change under the director. dusk.py already has v2 WIP (fog 1.15e-4, wisps(), knoll-top glint).
**Live status:** (updated below as things land)
* 20:50Z: local half-res stills renders/run_b_tests/hbb_t15/{3840,4300,4880,4976,5100} look right (no halo, soft
  silhouettes with folds, the hand-off reads, the child is a child in red, the cairn = snow-capped stones).
  JOB READY sent for ALL FIVE hand-back jobs (dawn_a, dawn_b, night, crane_a, crane_b), join A/B flagged as
  pending RUN-B2's 3839 (renders/run_b_tests/b2_check_local/f_03839). If the director approved: check
  `python3 the-long-dawn/cloud/farm.py status` and renders/handback_B (frames stream back as they land).
  Nitpicks queued for after 23:30Z: the fire-basket plinth (bset.beacon_base) still uses triangle-fan stones (faint
  seams) -> rebuild with bfig.add_stone as bprops.beacon_base3 (tell RUN-B2: shared with the vigil); RUN-B2's
  bset.summit_snow(G) (old snow + sparse stones on the boss) to switch on together in vigil + crane + settled.
* 20:52Z: full-res farm stills landed (renders/_farmtest/handback_b_{dawn_a,dawn_b,crane_a}); they confirm the
  look. Farm speeds: settled ~18 s/frame, crane ~96 s/frame per node. Waiting on the director's approval to launch
  the five hand-back jobs (`python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/handback_b_{dawn_a,dawn_b,night,crane_a,crane_b}.json --nodes 3`).
  DUSK v2: job cloud/jobs/dusk_b2.json (out renders/dusk_B2) written; baseline test (v2 WIP on b15, frames
  0/320/575/600) queued on the farm -> renders/_farmtest/dusk_b2 (log _local_logs/runB/farm_dusk2_t1.log).

# >>> RUN-B STATE AT HANDOFF 3 (new agent RUN-B-3, 27 Sep ~19:10Z) <<<
**Inherited:** PAUSE 2 state below (DUSK v1 done in renders/dusk_B; hand-back/vigil/reveal coded and tested; jobs
written, none READY). **Director's order for this agent:** (1) hand-back: fix the smooth middle-distance mountain
that reads as a sand pile (rock bands, snow texture, haze) -> JOB READY + 2-3 check stills; (2) vigil -> JOB READY;
(3) reveal: the fire reads as a warm point in the last wide frame, Milky Way stronger -> JOB READY; (4) DUSK v2 test
into a NEW folder renders/dusk_B2 (v1 stays in renders/dusk_B as the fallback; EDIT repoints after approval).
Heavy renders as cloud jobs of <= 45 min each; the director launches them when cloud access is back.
**Live status (updated as work lands):**
* bworld **b15** (the sand-pile fix; the dome was 100% snow at ~34 deg = sand's angle of repose, backlit, smooth):
  `field()` beyond 1.7-2.8 km of her, above the cloud: (a) soft-ridged spur/gully relief (S 1150 m, 150 m) = what
  makes the shoulder read; (b) strata cliff bands (P 64 m, irregular spacing, some missing, 4 deg dip, broad ~2 km
  mask so bands run long; a 560 m mask made isolated "zebra dashes" = b12, rejected). `gbuffer`: wind-scoured
  convex crests shed snow (dtop > 1.5 km). handback dawn fog: deeper valley haze (2.6e-4, H 220 m); crust sheen
  0.25 -> 0.12 (read as a lake under the sun); flame absorb scales with the fire (dark "ears" in full sun).
  Tests hbb_t12/t13: the mid mountain reads as a rock-banded snow massif rising out of haze.
* **LANE SPLIT (19:3xZ):** RUN-B2 (agent adab49d5d24d90ffb) owns THE VIGIL + THE REVEAL (vigil.py, reveal.py; it is
  adding a style=2 puppet to bset.person and may move bset CAM_*). RUN-B-3 (this block) = the hand-back + DUSK v2. My
  one reveal commit (d36b553: end cam 650 m out/70 m up, far-fire warm point, band 0.14) is handed to RUN-B2.
  THE JOIN 3839/3840 (RUN-B2's asks, done in handback_b 0b?): fire point light LP[36..43] in night()+dawn, the
  vigil's Milky Way (add_band with vigil.BAND / BAND_GAIN), FINISH eases from vigil.FINISH to ours by 4160
  (finish_at). RUN-B2 imports handback_b.beacon_catalogue() / villages_world() (keep names + cache keys).
  Don't send the crane (3840-4079) or settled night (4080-~4160) as final until RUN-B2 and I have matched stills.
* **Puppets (handback_b.py, not bset):** `sleeve_arm` (tapered arm, hanging sleeve, gloved mitten + thumb),
  `keeper_hb(hands, give)` (opening: forearms out at chest height, palms open; the V read as "hooray" and hands-up
  read as surrender), `child_hb(wake, reach, hold)` (one shawl drape: covered head, drape-filled neck, sloped
  shoulders; the old two-ellipse child read as a snowman/pawn, a pure bell read as a pyramid).
* **Farm (19:23Z on):** heavy tests via `cloud/farm.py <job> --test K --frames ...` (push first). First farm test OK
  (renders/_farmtest/handback_b_*: settled setup 162 s; crane 64-80 s/frame single-thread). 19:41Z: CPU nodes
  failing to boot farm-wide (all lanes' CPU tests "incomplete"); fell back to a light local half-res check (hbb_t14).

## RUN-B2 (split off RUN-B ~19:25Z: B5 THE REVEAL + B6-B12 THE VIGIL; owns vigil.py, reveal.py; RUN-B-3 keeps
## handback_b.py + dusk; bworld.py/bset.py shared, additive only)
**>>> RUN-B2 STATE AT PAUSE (27 Sep ~20:50Z; usage gap to ~23:30Z) <<<**
* CODE (all pushed, tip 6d7714f): vigil.py (the WSW locked frame, the hand-back's sky/beacons/villages, the join at
  3839 = handback_b's crane frame 3840 incl. HandBack.figures_at from 3810, barmap_B sync, the hail, squall, fog,
  travellers on the ENE way, the child), reveal.py (held roar behind her left shoulder -> log pull to 130 m WSW on
  the vigil's axis, the Milky Way arching over her summit, bset.summit_snow), bset.py (person2 incl. 'hail',
  summit_snow, match_horizon, night_params(horizon_match), CAM_* WSW with SET axes frozen at 31).
* JOBS (farm): cloud/jobs/vigil_b.json (1520-3839 -> renders/vigil_B, suggest --nodes 3, ~30-45 min) and
  reveal_b.json (1360-1519 -> renders/reveal_B, --nodes 2-3, ~30 min). Check stills: renders/run_b_tests/
  b2_check_local (local, half res) and renders/_farmtest/b2_check (farm, 11 f). JOB READY sent? see live status.
* NEXT on resume: (1) if the director approved: launch `python3 the-long-dawn/cloud/farm.py
  the-long-dawn/cloud/jobs/vigil_b.json --nodes 3` (and reveal_b.json --nodes 2) from ~/mishamisha, then review the
  landed frames (contact sheets every ~200 f; the join 3839 vs renders/handback_B 3840); (2) nitpicks list below.
* NITPICKS (cheap re-renders later): summit_snow is reveal-only (the vigil/crane join must change together: offer
  to RUN-B-3); the far answers at full res may read as an even skyline string (a distance falloff, shared with
  handback_b.layers); the fog hours are a flat veil (drifting wisps would help); stars sparse at half res.

**STATE AT HANDOFF (RUN-B2, 27 Sep ~19:30Z):** vigil.py + reveal.py as committed in 118eafd (tested at quarter/half
on b11; no renders in renders/vigil_B or reveal_B). Jobs vigil_b_a/b + reveal_b_a/b written, NOT READY. Open: reveal
end frame (the summit read as a smooth dark sand dune; the fire not a warm point; band weak; a hard ragged seam at the
cloud-sea horizon); the vigil->crane join at 3839/3840 (stars, band, far beacons, villages must be identical to
handback_b's first crane frame; coordinating with RUN-B-3). **Live status:**
* 19:30Z look-dev on the farm (vigil_b_look, 16 f, half res) of the committed vigil: the dead foreground dome,
  a hard ragged cloud-sea horizon, stars as dashes (read as snow), no Milky Way in frame, travellers a row of
  identical pointed black cones (Nazgul), her shawl a red box, the answers an even string of lights.
* JOIN with RUN-B-3 agreed (their messages 19:40-20:15Z): at 3839/3840 the vigil adopts the hand-back's world
  lists (handback_b.beacon_catalogue / villages_world / HandBack.chosen, glow formulas copied from layers()), the
  hand-back's star catalogue (SK.make_stars 14000/101/7) with the wheel reaching identity at 3840 (theta(f); eases
  to a stop over bar 48), the fire/light/flame settling into the hand-back's values over bar 48, and from 3810 the
  pair drawn by HandBack.figures_at(frame=3840). RUN-B-3 adds: the fire point light in night(), add_band with
  vigil.BAND/BAND_GAIN, star gain x vigil.STAR_GAIN, night_params(horizon_match=True), crane exposure from
  vigil.FINISH easing to 0.80 by 4160. FIGURES split: person2 (bset) = MINE, B's one figure; bfig.render (no
  uniform halo), bprops.cairn3 / child3 = RUN-B-3's (new files), both lanes switch when pushed.
* CAMERA: 4 variants tested (vigil_b_cams): WSW looking ENE wins (cairn, her, fire separate on the crest; cloud sea
  + rock islands; more sky). bset CAM_BEAR 62 / CAM_DIST 55 / CAM_UP 6 / CAM_YAW 70 / PITCH -3 / HFOV 36; the SET
  axes YAW/FWD/RIGHT stay 31 (shelf_pt builds the set from them; RUN-B-3's catch). Pushed cec7290.
* SEAM fixed: bset.match_horizon (the far haze = the sky's horizon colour); vigil/reveal use it.
* Travellers: vigil-only way down the ENE slope (vpath), seven looks, their own torchlight, fewer at once.
* REVEAL: held beat on the roar (20 m), log-distance pull from the cello's CALL (1398) to 700 m WNW +45 m (1506),
  hfov 44->50, the summit low-left, the Milky Way (BAND posed for this wide, gain 0.16) from the right horizon.
* 20:25Z look-dev of both (vigil_b_look 18 f + reveal_b_look 6 f) submitted to the farm (queue backed up ~25 min).
* 20:30Z local quarter-res check (vg2_q, rev2_q): the WSW frame works (cairn / her / fire separate on the crest,
  the cloud sea + rock islands, the far skyline answers, torch-bearers, fog veils, villages); open: the far answers
  along the skyline may read as an even string of lights at full res (judge at half res), the Milky Way is not in
  the vigil's frame (it is posed for the reveal; fine), the cairn read as a pagoda (-> cairn3). Reveal: the band now
  strong across the last wide, the fire a warm point; but from 700 m level the plateau still read as a dune -> the
  end moved to 240 m / +24 m (like f1450, which read as a snowy summit crest); the roar held through the cut (1372).
* Vigil retimed to barmap_B sync (c0e9f69): 'traveller' 2880 = the first torch takes; three together bar 38 b3;
  'year_60' 3600 = the child tops the path with its parent; 'child_points' 2560 = a faint unsure far light before
  the answer takes on 2720; feeds on every stone_N (1960 ... 3560) and y60_strikes (3680); she sits on y60_sits.
* 957a2d8: vigil + reveal use RUN-B-3's bfig.render (no uniform halo) + bprops.cairn3; the child = bprops.child3
  once wrapped in her shawl, placed by handback_b.child_pos(). Finals written: cloud/jobs/vigil_b.json (1520-3839,
  --nodes 3 -> 3 units of ~775 f) and reveal_b.json (1360-1519, --nodes 2); old vigil_b_a/b + reveal_b_a/b deleted.

## FROM HEROINE-B (B2 THE CLIMB, B 640-879 -> renders/climb_B; `shots/hills/climb_b.py`), 27 Sep ~20:05Z
* Per the director, THE CLIMB uses B's shared keeper: `bset.person2('stand', arms=False, staff=False, age=0.85)` plus
  my own carrying arms (`handback_b.sleeve_arm`, imported read-only) under her shawl (I re-layer person2's shawl rows
  on top), the clay pot at her right hip, and heel lifts for a rear-view trudge. I call `bfig.render(...)` with
  FG.render's signature if `bfig` has one, else FG.render. **RUN-B-3: please tell me here bfig's call and your 3-D
  cairn's function** (I draw `bset.rubble_cairn()` at `bset.CAIRN` and `beacon_base()` cold until then).
* Read-only imports: bworld (G-buffer per locked camera), bset (path_at, CAIRN, BEACON, night_params,
  match_horizon if present, person2), vigil (moon_at(F0) = az 80 el 14; `draw_sky` if present, frozen at f 1360).
* **bworld h_rock STEP (for RUN-B-3):** a 0.35-0.37 m VERTICAL step in `h_rock` on the NE crest ~24-26 m from the top
  (dtop 23.9-26.1), e.g. at x -5435.86, z 32012.46 the height jumps +0.349 m between x and x+0.05 (same at fp 0.067 and
  0.01; `knoll()` there is 1.5 m LOWER than h_rock, so it comes from another row: the NE ridge `bridge` / dome?). Seen
  edge-on (any low or side-on view of the crest) it renders as a row of dark vertical posts, a FENCE along the crest
  (normals ny 0.14, lit by ambient only). My wide avoids it (behind her, from above); the vigil may see it too.
* FYI: `bset.path_at(s)` for s < ~30 is the straight chord from LIP to the first crest point and floats up to ~1.3 m
  above the rounded crest (s 12: path y 299.90, ground 298.65). Anything walking the last 30 m (the vigil's
  travellers and child) floats unless put `on_ground()`.

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
