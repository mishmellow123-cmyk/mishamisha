# >>> HEROINE-v3 REPORT 2 (2026-09-27 ~05:30Z; after the LOCKED beat sheets, director commit 99e9bfb) <<<

The locked sheets moved every heroine shot; all are re-keyed to them. REPORT 1 (01:15Z) follows below for history.

**H1 on the MASTER TIMING, all cuts (must-have).** `beacon.use_master_timing()` rebinds the take's timing constants:
strike 2 +29, strike 3 +58, the long blow +84..+180 (three breaths), the catch +196, the roar +240 (on a downbeat); the
accepted post-catch poses are stretched 42 -> 44 frames to the roar and shifted +116 after it; four frame literals
became constants with identical v2b values, so `--shot beacon` is unchanged. `render.py --shot beacon_v3` = this timing
+ the B7 re-key; src 1200-1555, strike 1 = src 1236 -> `renders/h1_v3`, JPEGs to branch **`claude/render-h1-v3-master`**
(`cloud/jobs/h1_v3_master.json`; RUNNING on this box since 05:09Z, ~1-1.5 h; log
`/home/user/h1m/the-long-dawn/cloud_logs/h1_v3_master_runner.log`). Verified at half res: 1265 (strike 2), 1334/1400
(the blow), 1436/1456 (catch, small flame), 1478 (roar), 1500/1555 (pull-back): face dark throughout.
Edit mapping (src -> cut): A src 1236 = A 3360 (+2124); B src 1236 = B 1120 (-116); C src 1236 = C 2980 (+1744) through
strike 2 (C 3009), the find inserts, then src 1294 (strike 3) = C 3178 (+1884) through the roar (src 1476 = C 3360).
The 00:42Z re-key on the v2b timing (`renders/hills_v3`, branch `claude/render-h1-v3-rekey`, `--shot beacon_v3_rekey`)
is now only the fallback.

**B THE DEAD EMBER (H5), B 880-1119.** Re-keyed and restaged: her FAR gloved hand lifts the lid away from the lens and
sets it down beyond the pot on the knock (bar 12 b3 = 920), then rests on the pot's belly clear of the mouth; the red
eye 920-980; from bar 13 b2 (980) she blows, a breath of hope, then the eye greys (1000-1100); the last red point on bar
14 b4 (1100), out by 1116; the camera drifts 1040-1100 so that point sits exactly where H1's first spark is born
(full-res px 757, 256): the match cut on bar 15 b1. Quarter-res motion test `review/v3/heroine_v3_deadember_mt2.jpg`:
the lid lift is clean now. Fixed since: the ash poking through the wall, the soot (M_COAL greyed to ash) now dark.
Verdict: PASS (beats its fallback). Stills: `renders/heroine_B/stills/deadember_{936,1100}.png`; sheet `review/v3/heroine_v3_sheet4_locked.jpg`.

**C THE FIRE TEST (H2, Bag End), C 3392-3599.** Locked frame on the fire; her fist brings the steel in (3392-3404), the
Ring lies on its tip in the flames from bar 43 b3 (3400); on bar 44 b3 (3480) the steel dips and rolls, the Ring slides
toward the edge and does not fall; on bar 45 b3 (3560) she draws it out of frame right. PASS. Still `firetest_3484.png`.

**C THE FIND (H2), C 3009-3059.** Strike 2 (C 3009) finds the band in the melted hollow; the insert is the fallback
(the band alone, `Find.HAND = False`); the edit cuts to her closed fist for bar 39 b2. The vision on the band (towers
bowing, bar 39 b3.5) is a Ring close-up: the Blender bake-off lane's. Still `find_3012.png`.

**B THE CLIMB (H4), B 640-879.** The close from behind (bar 11 b1) FAILED in five framings (the pot is hidden by her
body from behind; from the side the frame is a dark mass; the stand-in arete reads as a roof) -> the sheet's FALLBACK:
the wide alone, her figure <= 60 px with the pot's glow the only warm point, on RUN-B's plate. This lane supplies
`heroine_v3.climb_layer(cam, f, world_pos, heading_deg)`: the trudging, gloved figure with the glowing clay pot as an
RGBA + depth layer for any camera (B's moon; the pot's light; the scarf and hair in the wind); `climb_wide_test` is a
crude stand-in wide for scale only.

**For ACCORD (AC2, "with HEROINE's hand"):** `heroine_v3.glove_hand_layer(cam, wrist, fdir, palm, curls, thumb, side,
lights)` renders her gloved hand and sleeve cuff alone as an RGBA + depth layer in any camera (e.g. the top-down fist
over the Ring on the stone); smoke-tested.

**RENDER_SPEC (one 4-core box each, 2 processes x 2 workers, `--step 2`, ship jpg; `render.py --shot <name>`):**
* `cloud/jobs/h1_v3_master.json` (beacon_v3): RUNNING here, see above.
* `cloud/jobs/h5_deadember_b.json` (deadember): B 880-1119 -> `renders/heroine_B`, ~1 h. Ready.
* `cloud/jobs/h2_firetest_c.json` (firetest): C 3392-3599 -> `renders/heroine_C`, ~50 min. Ready.
* `cloud/jobs/h2_find_c.json` (find): C 3009-3059 -> `renders/heroine_C`, ~15 min. Ready.

**Remaining weaknesses.** The glove fingers are a touch thick at 1:1; the dead ember's snow wedge between her knee and
body is a hard shape; the Ring is ring.py's band (the bake-off may replace it); H3 not attempted (optional); the
climb's close is the fallback; H6 belongs to HILLS on the locked sheet.

# >>> HEROINE-v3 REPORT (2026-09-27 ~01:15Z, lane `heroine`, code on branch claude/v3-heroine) <<<

**What it is.** (1) **H1 RE-KEYED** (BIBLE_V3 REVISION 1, red-team B7) and rendered in full: the accepted v2b flint take
with her hands in leather gloves and her head a rim-lit silhouette at every phase (no lit face anywhere), src
1200-1439 -> `renders/hills_v3`, 240 JPEGs on branch `claude/render-h1-v3-rekey` (`render.py --shot beacon_v3`).
(2) **Risk test #5 (people)** stills: B THE DEAD EMBER (H5), C THE FIRE TEST as Bag End (H2), C THE FIND (H2); code in
`shots/hills/heroine_v3.py` on a fork of the heroine tracer (`hsdf3.py`) so nothing accepted can change.
Sheets: `review/v3/heroine_v3_sheet1.jpg`, `heroine_v3_sheet2.jpg`, `heroine_v3_sheet3.jpg` (final stills),
`heroine_v3_h1_contact.jpg` (the re-keyed take, every 12th frame), `heroine_v3_deadember_motion.jpg` + `_mt1.mp4`.

**Verdicts against the fallbacks.**
* H1 re-key: PASS (a re-key, not a crop: the composition, timing and every other element are the accepted take).
* FIRE TEST (Bag End): PASS. The Ring lies on the tip of her steel in the flames, balanced, gold and unmarked, its
  letters burning; her gloved fist holds it there (tremble, one dip toward the coals, drawn back). Beats its fallback.
* DEAD EMBER: PASS with notes (stills). Seen high on her left, looking into the clay fire-pot (her head above frame):
  the lid off, one red eye in a char crust, her breath, the eye greying to ash. Beats its fallback on legibility; in
  motion the lid lift still needs restaging (see RENDER_SPEC).
* FIND: SPLIT. The band in the melted hollow, found by the strike's flash and faintly awake in the dark: PASS (and
  it is the bible's fallback image). Her gloved hand closing on it: FAIL (at the reach the glove silhouette breaks
  into a claw) -> FALLBACK taken: the band in the hollow, then cut to her closed fist.
* H3 (hand forced open): not attempted (REVISION 1: optional, last). H4 climb, H6 silhouettes: not started.

**Remaining weaknesses.** Glove fingers a touch thick and smooth at 1:1 (inflate 0.9 mm; a seam would help); the
pot was CG-clean (now a hand-thrown tilt, heavier shoulder, chipped lip: not yet seen at full res); the melt pool is
frosted slate ice (it was a black void); the Ring here is ring.py's
band (the Blender bake-off may replace it); H1 keeps the v2b timing (the v3 retime waits on the bar maps).

**RENDER_SPEC (for the director; one 4-core box per job, 2 processes x 2 workers, ship jpg):**
* `cloud/jobs/h1_v3_rekey.json`: DONE on this box (see above).
* `cloud/jobs/h2_firetest_c.json`: C 4040-4183 (144 fr) -> `renders/heroine_C` (`--shot firetest`), ~60-90 s/frame/
  core, ~45 min. Ready.
* `cloud/jobs/h5_deadember_b.json`: B 960-1199 (240 fr) -> `renders/heroine_B` (`--shot deadember`), ~1 h. NOT YET:
  the quarter-res motion test (`review/v3/heroine_v3_deadember_motion.jpg`, `..._mt1.mp4`, 976-1152 every 4th) holds
  from 1008 on (the eye, her glove on the rim, the scarf end, the greying) but the lid lift 976-1004 is a dark smear:
  her near arm sweeps across the lens. Restage (lift the lid away to frame left, or start with the lid already off)
  and re-test before launching; the snow wedge between her knee and body also wants softening.
* `cloud/jobs/h2_find_c.json`: C 3557-3620 (64 fr) -> `renders/heroine_C` (`--shot find`), the fallback version
  (`Find.HAND = False`: the flash reveal, the sparks dying, the letters' glow; no hand), ~25 min. Ready.

**Review sheet:** `review/v3/heroine_v3_sheet1.jpg` (H1 v2b vs re-key; H1 re-key strike + roar; B dead ember lid-off
and greyed).

**DONE (00:42-01:13Z, local 4-core box, 7.6 s/frame wall): H1 RE-KEY full render, 240/240 JPEGs pushed.** `cloud/jobs/h1_v3_rekey.json` run by
`cloud/run_job.py` from the detached worktree `/home/user/h1job` (commit 216d9f1): src 1200-1439 -> `renders/hills_v3`,
JPEGs pushed every 5 min to branch **`claude/render-h1-v3-rekey`** (created by hand at 00:57: a detached-HEAD worktree
cannot push to a branch that does not exist yet; the runner's first rounds failed on that). Runner log: `/home/user/h1job/the-long-dawn/cloud_logs/h1_v3_rekey_runner.log`. If the box died: pull that branch,
copy its jpgs' frame numbers out, and relaunch the same job on any box (`--skip-existing` only sees local PNGs).
* What it is: `render.py --shot beacon_v3` = the accepted v2b take (timing, poses, camera, scarf, breath, sparks,
  fire, world) with `beacon.V3_REKEY = True`: her hands in dark leather gloves (`hsdf3.gloves`: M_GLOVE, +0.9 mm,
  no nails, no cold flush) and the head groups (skin, eyes, cap, brim, hair) RIM-ONLY under the warm sources
  (`hsdf3` G col 9: strike/ember/tinder/roar light reaches them only at grazing angles; the moon unchanged); hair
  strands and lashes get 12% of the warm light. Rendered through `hsdf3` (fork), so v2b (`--shot beacon`) is
  bit-for-bit untouched.
* Verified at half res (1262 strike, 1305 blow, 1322 catch, 1345 watch, 1366 flinch, 1390 rise): the face is a dark
  silhouette with a thin warm profile rim at every phase; the gloves read as leather (tan in the strike flash);
  nothing else changed. Test frames: `renders/hills/tests/h1v3b/` (+ v2b 1305 in `tests/h1v2b/`).
* Not done: the v3 RETIME (strikes ~1.2 s apart, longer blow, roar on a downbeat) waits on SHOWRUNNER-REV's bar maps
  (`music/v3/barmap_*.json` not on the branch yet); this render is the v2b timing, re-keyed.

**B H5 THE DEAD EMBER: first full-res stills** `renders/heroine_B/stills/deadember_{1030,1120}.png` (sheet 1).
Staging: she kneels at the cairn, the round clay fire-pot on its lowest course; camera high on her left looking ~50
deg down into the pot (her head and shoulders above frame: no face possible); B's moon high on her left; the lid
lifted off by her near hand and set down, her far gloved hand steadying the pot's rim; the ember's light shadowed by
the pot walls; the breath (thin) lit from below; `life` greys the ember (1044-1130). Verdict: READS (clay pot in
moonlight, the ember, the grey) = beats its fallback on legibility; NOT YET PERFECT: the ember too big/yellow with a
busy "coral" crack pattern (want one small deep-red eye in a char rim; emission since made redder, untested at full
res), the pot CG-smooth (want a hand-thrown wobble, soot, chips), the glove fingertips read as dark balls.
Alternative coded: `deadember_pov` (her POV into the pot; breaks B's observational grammar). Fallback (bible): from
behind her, the pot's glow the subject.

**C H2 THE FIRE TEST (Bag End):** `renders/heroine_C/stills/firetest_4090.png` (sheet 2). Her gloved right fist holds
her steel like a key, flat, its end in the flames at the basket rim; the Ring LIES flat on the steel's tip, balanced
(a tilt of her wrist would drop it into the coals; she cannot); gold, unmarked, its letters burning; the fire behind in
soft focus. Motion keys: tremble, one dip toward the coals (4100-4140), drawn back 4160+. Verdict: the strongest of
the set; hands/pose/light read; glove reads as dark suede/leather. Weak: glove fingers a little thick and smooth at
1:1; the burning logs (below frame now) looked like sausages when visible.

**C H2 THE FIND:** `renders/heroine_C/stills/find_{3560,3645}.png` (sheet 2). Low on the snow by her knee: a melted
hollow with a slumped lip, a frozen melt pool, the Ring lying tilted on it with its letters faintly awake; the strike
flash (3557) and a few sparks dying on the snow; her gloved near hand comes down and closes on it (3600-3660). Verdict:
the band-in-the-hollow reads (= the bible's fallback image, and it is good); the hand closing is NOT there yet: four
blunt dark fingers descend like a claw. The full-res stills in sheet 2 predate these fixes (01:06Z, tested at 0.4):
* the brown/green stains on the snow were heroine_sdf's skin wrap (red wraps furthest) applied to snow: new
  materials now wrap neutrally, snow wraps blue deepest (H1's materials 0-10 unchanged);
* camera raised to ~45 deg down (`Find.camera`): her gloved hand now reads as a SILHOUETTE against the moonlit snow
  (no detail needed = no doll), and the Ring's letters warm as her fingers come near (`near`, 3612-3668) and light
  the glove and the hollow from below: the temptation is the key light;
* VERDICT 01:09Z: the hand closing FAILS (at 3630-3645 with the shoulders lowered so she can reach, the glove
  silhouette breaks into floating finger segments: a claw read) -> FALLBACK NOW per the rule: the Find is the band in
  the hollow (flash reveal 3557+ and the letters' glow; `find_3560.png` re-rendered full res with the fixes), then cut
  to her closed fist (H1 close-up / the fire test). Next attempt, if the hand is wanted back: a pinch posed in the
  hand lab (`hero.hand` curls ~(0.40, 0.70, 0.85, 0.92), thumb 0.6, fdir (-0.6,-0.75,0.1), palm (-0.55,0.45,0.7) read
  as a clean C-shaped pinch against snow) with the wrist placed so the hand is not foreshortened, and the arm
  entering from frame right; the pool is pure black (give it the lip's reflection and some frost).

**RENDER_SPECs (not launched; each waits on its look being approved):** `cloud/jobs/h5_deadember_b.json` (B 960-1199 ->
`renders/heroine_B`), `h2_find_c.json` (C 3557-3683 -> `renders/heroine_C`), `h2_firetest_c.json` (C 4040-4183);
`render.py --shot deadember|find|firetest`; one 4-core box each (2 processes x 2 workers, --step 2), ~60-90 s/frame/core
at full res, so ~1 h per job; ship jpg.

**Code (new; nothing accepted changes behaviour unless a v3 flag/shot is used):**
* `shots/hills/hsdf3.py`: fork of heroine_sdf's tracer. T_BAND ring primitive (ring.py's rounded section or
  tolkien's superellipse); materials GOLD (env-map reflection + ring.py's inscription engraved/burning), CLAY,
  EMBER/COAL (emissive cracks under spreading ash, `life`), SNOW (glints), ICE, IRON, ASH, GLOVE; rim-only groups;
  returns a group-id buffer.
* `shots/hills/beacon.py`: `V3_REKEY` (default False), `V3_SIL`, `V3_STRAND_WARM`; `render.py`: shot `beacon_v3`.
* `shots/hills/heroine_v3.py` (WIP): DeadEmber (B H5) being laid out: kneeling bent over a round clay fire-pot on the
  cairn's foot-stone, camera high on her left looking steeply into the pot (head and shoulders above frame),
  moon behind her, ember light shadowed by the pot walls; `V3_CAM=px,py,pz,tx,ty,tz,hfov` / `V3_EXPO` debug hooks.
  `python3 shots/hills/heroine_v3.py still deadember 1030 out.png --scale 0.3`.

**Next steps (in order):**
1. Find: the pinch (thumb + index, oblique), re-render with the snow fixes; DeadEmber pass (ember eye, pot wobble,
   glove tips); glove fingers slimmer (inflate 0.9 -> 0.5 mm) + a seam; then launch the three RENDER_SPECs.
2. H4 THE CLIMB (close from behind: the scarf and the pot) as a still; H3 last, if at all.
3. When the H1 job ends: check `claude/render-h1-v3-rekey` has 240 jpgs; contact sheet; this section -> the report.

# >>> HEROINE-2 (2026-09-26): FIRST BEACON v2b (s1 world, dry-stone courses, ignition) + RING fix - both LANDED <<<

**LANDED 2026-09-26 18:4x (director accepted v2b).** Full-res FIRST BEACON src 1200-1439 rendered locally (240 frames,
27.5 min, 2 workers, log `~/mishamisha/_local_logs/heroine_lookdev/stage_hv2_render.log`), verified (all 240 decode at
1920x804; every-20th sheet; the catch 1316-1354 at 1:1) and swapped into `renders/hills_v2` in one batch. The old
accepted take is backed up in `renders/hills/tests/hv2_accepted/`. Review sheet rebuilt from full res:
`before_after_v2b.py hills_v2 <out> --before hills/tests/hv2_accepted --ring hills/tests/ring_old`. RENDER_SPEC 1 and
`cloud/jobs/hills_v2b_{a,b}.json` are no longer needed (keep as a recipe). Nothing is running.
* **P1 ignition (producer note): DONE in code**, verified at half res frame by frame (1312-1331) and full res
  (1314, 1346); before/after in the review sheet (`hv2_final2` half-res after-set). Details under "What changed".
* **P2 CODA/INTRO ring hairline: DONE and LANDED** in `renders/hills_v2` (218 frames; old ones in
  `renders/hills/tests/ring_old/`). See "ORBITAL RING FIX" below.
* **P3 "the person looks a bit weird": checked, nothing changed** (no anatomy/silhouette glitch in a 1204-1390
  sheet; see "Remaining weaknesses").
* Files touched (uncommitted; the director commits): `shots/hills/{beacon,peaks,heroine,sky,hillworld}.py`, this NOTES.md,
  `cloud/jobs/hills_v2b_{a,b}.json`; tools/sheets in `~/mishamisha/_local_logs/` (heroine_lookdev, review).

**State.** Code final; v2b is in `renders/hills_v2/f_01200..01439` (old take: `renders/hills/tests/hv2_accepted/`). Review sheet: `~/mishamisha/_local_logs/review/heroine_v2b.jpg`
(tool `_local_logs/heroine_lookdev/before_after_v2b.py hills/tests/hv2_final2 <out> --ring hills/tests/ring_old`).
Final after-set: `renders/hills/tests/hv2_final2` (half res; 1372 + 1439 full res). Earlier tests: `hv2_h`, `hv2_i`
(half res), `hv2_j_full` (full res; before the close-up grade and the ignition). Pre-review code (the hv2_g state):
`_local_logs/heroine_lookdev/{beacon,peaks}_hv2g_backup.py`, before the ignition `beacon_before_ignition.py`;
old notes `NOTES_before_heroine2.md`.

**What hv2_g got wrong (found in review)**
* s1's summit crest is ~2 m above her camera in the close-up and a 15 m rock fin stands 10 m from the beacon:
  the camera sat INSIDE s1's rock (black close-up: 1230 was an empty screen) and the fin rose as a black
  spire behind the fire in the reveal (1372-1439).
* s1's lens-shift camera moved the frame edges by up to 5 px (close-up) and 15.5 px (pull-back corners).
* CODA's merged-silhouette basket broke into lit fragments at close-up size; the dry-stone courses read as
  pale boards under the strike and the roar.

**What changed vs the accepted take** (all under `HEROINE_V2` in beacon.py)
* **Camera = the accepted pitched camera.** `world_layer` renders the shepherd's world on a lens-shift canvas
  (32 px margin) from the same eye point and resamples it ray-exactly into our camera (a rotation about the eye
  is a homography); stars are splatted directly in our camera (`_splat_stars`, s1's field + splat). Framing
  change: 0 (phase correlation vs accepted < 0.15 px on her and the dome = content only).
* **Background = the shepherd's world** (`shots/run/world.py`: sky, ranges, cloud sea, fog, moon) with the accepted
  v8 grid's keep-clear rule (`KEEP`: s1 rock -60 m within 70 m of her summit, easing out by 160 m). world.py is
  not edited: `_world_keep_module()` writes a patched copy to `shots/hills/cache/world_keep.py` (gitignored,
  regenerated on any box; raises if world.py's two patched lines change).
* **Close-up grade** (`CLOSE_SKY`, eased out over the reveal, which ends exactly at s1's exposure): s1's sky is
  brighter than the accepted close-up's; x0.45, rising x(1 + 5 cos^4) toward the moon (behind the basket).
  Matches the accepted background within ~1-4 levels: the basket reads faintly before the catch and her
  streaming hair stays quiet (at s1's level the locks read as bare twigs against the sky).
* **Cairn:** CODA's dry-stone courses (`CAIRN2.render`); iron basket + wood exactly as accepted (puppet `Figure`
  with the old 'wood' + 'basket' groups). `STONE_SHADE` = 0.3 of the strike/ember/flame/fire light reaches the
  courses (the basket floor and the wood shade them).
* **Ignition (producer note: "the way the fire initially lights seems wrong").** The accepted catch was a lit
  breath-fog ball floating at the top rail, then a 5 cm torch flame popping in mid-air ~10 cm clear of any wood,
  drawn over the bars. Now: a twisted bundle of dry grass rests on the top split log at TINDER (`tinder_groups`,
  drawn with the basket from 1200 until the roar at 1360) under a lean-to of kindling against the teepee (`KINDLING`).
  The ember is a cluster of glowing fibres in the bundle (`IG_EMBER`) with a thread of smoke (`_tinder_smoke`);
  her breath jet stays lit but its fog evaporates within ~5 cm of the ember (no lit ball). The flame
  (`ig_flames`, PCHIP keys `IG_MAIN`) fades in from nothing in the bundle at 1318-1320, wavers (1324-1326),
  grows between the bars, rises above the rim ~1338 and climbs the kindling (two flamelets, `IG_FLAMELETS`,
  from 1330/1337) toward the teepee; the bars and rim occlude the pre-roar flame and ember (they burn inside
  the basket). Sparks leave from the flame's tip. `flame_level()` (her light, the spark RNG stream), the timing,
  the roar at 1360, her pose and the camera are unchanged. Tests: `renders/hills/tests/ig_a..ig_g` (half res),
  `ig_full2` (full res 1314/1346); final after-set `hv2_final2`.
* **Smoke:** `t - 58.3` phase: the accepted frames' horizontal streaks above the fire (1370+) are gone.
* **Scarf:** the Elder's wool (0.29, 0.022, 0.019) in heroine.py; fringe, tail and animation unchanged.
* `peaks.render(summit_only=True)` only needs her summit grid: `peaks.summit_grids()` (bakes in 1.4 s on a fresh
  box, bit-identical to the v8 cache's S/M), so the 114 MB far-grid cache is not needed anywhere.

**RENDER_SPEC 1 - FIRST BEACON v2b: src 1200-1439 -> `renders/hills_v2/f_%05d.png` (240 frames)**
* Commit first: `shots/hills/{beacon,peaks,heroine,cairn2}.py` (cairn2.py = CODA's working copy, as rendered
  since 14:52 and as the stones were checked) + `cloud/jobs/hills_v2b_{a,b}.json`. No data files: the caches
  regenerate. Needs the committed `shots/run/world.py`, `shots/montage/{s1_peak,common}.py` + `mt/*`, `lib/look.py`.
* Box: `python3 -m pip install numba numpy scipy opencv-python-headless` (the jobs' setup line does it; beacon.py
  now needs scipy for the ignition keys).
* `python3 cloud/run_job.py cloud/jobs/hills_v2b_a.json` (1200-1319) and `.../hills_v2b_b.json` (1320-1439), one box
  each; each job runs two render.py processes (even/odd frames, 2 workers each = 4 cores). By hand, from
  the-long-dawn/: `python3 shots/hills/render.py --shot beacon --frames 1200-1439 --workers 2 --out renders/hills_v2
  --skip-existing` (render.py caps at 2 workers per process; use `--step 2` on 1200-/1201- to fill 4 cores).
* Cost: ~45-90 s/frame/core (3x3-supersampled heroine; the reveal's full-res world adds ~20 s); the first frame of
  each process compiles numba (~1-2 min). About 30-45 min per box.
* Landing: back up the accepted 240 (e.g. `renders/hills/tests/hv2_accepted/`), then replace all 240 at once (the
  edit must never mix takes). Never touch hills_v2's INTRO/CODA frames.

**ORBITAL RING FIX - DONE, landed in `renders/hills_v2` (INTRO src 257-294 + CODA src 2628-2807, 218 frames)**
* Red-team: the ring hairline ran through the CODA beacon flame (2636-2700) and, in the crane-up, through the
  title band (2696-2740; "DAWN" at v2 2880). In the INTRO it crossed the sparks rising from the torch during the
  push into the flame (276-290). Fix in the sky code only: `sky.draw_ring` honours an optional
  `ring['fade'] = (t0, t1)` (s); `hillworld.sky_coda` removes the ring at the cut to the WIDE (2627.5/24 s) and
  `sky_intro` fades it out over 256-272 as the torch fills the frame. C1 (2460-2627), INTRO 0-256 and 295+
  are unchanged (the ring is not drawable after 294); FIRST BEACON has no ring.
* Rendered locally (full res, 1 worker, ~20 min), verified (every frame decodes; the only differences from
  the old frames are the ring band: ~6k px per CODA frame, a 9-20 px blurred band in the INTRO push; C1 and
  INTRO 257 identical to +-2 dither), then swapped in as one batch. Old frames: `renders/hills/tests/ring_old/`.
* Commit with it: `shots/hills/{sky,hillworld}.py` (+ CODA's working copies `{coda,figures2,fires2,cairn2}.py`,
  which rendered the delivered INTRO/CODA frames at 14:52 and these). Re-render if ever needed:
  `python3 shots/hills/render.py --shot intro --frames 257-294 --workers 2 --out renders/hills_v2` and
  `--shot coda --frames 2628-2807`.

**Remaining weaknesses (not blocking)**
* Close-up background: s1's navy is bluer than the accepted grey-blue (brightness matched, hue is s1's).
* Reveal: the accepted's bright snow slope in the lower-right corner (1400-1439) is now dark moonlit rock ridges
  (s1's snow rule); no milky-way band (s1 has none).
* Her streaming hair locks read as branching twigs against a bright sky: keep the close-up sky dark.
* "The person looks a bit weird" (producer, low priority): no anatomy or silhouette glitch found in a sheet
  of 1204-1390. The likely causes are the dark hair mass beside her cheek (reads as a shadow patch in 1262/1345),
  the long scarf tail as a dark band in the pre-strike silhouette (1204-1280) and the twig-like locks; all
  are protected look elements (scarf, hair), so nothing was changed.
* The dry-stone courses are a flat 2-D card with per-face shading (tuned by CODA for distance); fine at 0.3 light
  from 1.5 m and beyond, but they would not survive a closer camera.

**(3) Files / functions**
* Ours: `beacon.py` (`HEROINE_V2`, `camera`, `KEEP`, `CLOSE_SKY`, `_world_keep_module`, `s1_world`, `world_layer`,
  `_splat_stars`, `STONE_SHADE`, `yw2_pose` + `V2_KEYS`, `_heroine` light rig, `_breath2`, compositing, `CAIRN2`),
  `peaks.py` (`render(summit_only)`, `summit_grids`, `S1_WORLD`), `heroine.py`, `heroine_sdf.py`; ring: `sky.py`
  (`draw_ring` fade), `hillworld.py` (`sky_intro`/`sky_coda` ring dicts).
* Read-only (other departments): `shots/run/world.py`, `shots/montage/s1_peak.py` + `mt/*`, `cairn2.py` and
  `silhouette.py` (CODA), `characters.py`/`coda.py`/`intro.py` (CODA: do NOT edit).
* Tools: `~/mishamisha/_local_logs/heroine_lookdev/` (t_shot.py heroine-only at a frame; t_debug.py; sheets).

**(4) Must not move**
* Timing: strikes 1236/1262/1290 (sparks born on the half frame before; steel scrapes the flint edge then),
  catch 1318, ROAR 1360, pull-back 1360-1439 (same c_pos/c_tgt/w_pos/w_tgt/hfov 40->66 path).
* Her look and choreography (`V2_KEYS`, overlays, lights) exactly as accepted, including no poster poses and
  the turn to the far range at the end.
* The scarf: the Elder's wool, the same fringe (9 tassels) and the long tapered tail.
* Composition of the close-up (basket left, hands centre, face right) and her size in the reveal (~50 px).
* The camera: the accepted pitched camera (never s1's lens-shift model for our layers).
* Max 2 worker processes, no git writes; hills_v2's INTRO/CODA frames are CODA's (the ring fix is landed).

**(5) Taste bar**
* Controlled darkness: fire is the only saturated warm light. She is carved out of the dark by it; the moon
  is a thin cold rim.
* Nothing may read as a clay doll or a stage flat: check every change at native resolution, not just thumbnails.
* One world: the reveal's ranges, cloud sea and sky must be recognisably the ones in s1 and the Run.
* Working gestures only, restraint over spectacle. When in doubt, darker and quieter.

# >>> HEROINE v2 (FIRST BEACON, Sep 2026) <<<

The Young Woman in FIRST BEACON is now a sculpted 3-D figure lit by the fire. `beacon.py`
`HEROINE_V2 = True` gates everything: the figure, the choreography, the camera, the dark-adaptation
floor and the s1-world params. `peaks.py` `S1_WORLD = True` (v8 cache) gates the new far ranges.
Output: `renders/hills_v2/f_01200..01439.png` (src numbering; the edit reads hills_v2 first).
INTRO/CODA import neither beacon.py nor heroine*. Do not edit characters.py/coda.py/intro.py
(the CODA dept owns them).

**Re-render**: `python shots/hills/render.py --shot beacon --frames 1200-1439 --workers 2 --out renders/hills_v2`
(add `--skip-existing` to resume). Half-res tests go to renders/hills/tests/hv2_*.
Lookdev and debug tools: `~/mishamisha/_local_logs/heroine_lookdev/` (t_shot.py = heroine-only at a
shot frame, shot/neutral/noshadow; t_debug.py = profile vs target + prim-ID map; t_hand.py; t_zoom.py).

**What it is**
* `heroine_sdf.py`: numba SDF sphere tracer.
  * Primitives: ellipsoid, round cone (+ Lipschitz-safe folds), round box, tapered/tilted torus arc,
    bent half-space, eyelid shell with an angular fissure (+ re-carve mode).
  * Culling: per 16 px tile, then per ray; 2x2 (half res) / 3x3 (full) supersampling on edges.
  * Lighting: point lights E = I/(d^2+r^2) with SDF soft shadows. A negative soft-k marks a source held
    in her hands (strike, ember, tinder flame), so her hands, tools and sleeves cast no shadow for it.
  * Materials: wrap/SSS skin with cold-reddened nose, cheeks, knuckles and fingertips; GGX; wool sheen
    and bump maps; Kajiya-Kay hair; a thin cold moon-rim from behind; AO; iris/pupil.
  * Also: strand renderer (hair, flyaways, lashes); fog-puff breath.
* `heroine.py`:
  * Head fitted to a female anthropometric profile (within ~3 mm).
  * Knitted beanie with folded brim; long dark hair escaping at the nape, streaming as locks plus fine strands.
  * Anatomical hands (knuckles, pads, nails); flint in the near LEFT hand (the camera sees her left
    side), C-steel in the far RIGHT hand. Heavy wool coat with folds; trousers, boots.
  * Scarf: `scarf_red` #9E1B1B knit, two loops, twisting/rippling 1 m tail, 9 fringe tassels.
  * She kneels on a low summit rock so her eyes clear the basket rim.
* `beacon.py` choreography (`V2_KEYS` + overlays):
  * strikes 1236/1262/1290: the steel bar scrapes the flint's edge on the half-frame the sparks are
    born, then follows through;
  * blowing 1297-1318 (face 13 cm from the ember, lit by it, breath jet to the tinder);
  * catch 1318 (ember flare, face lit from below, breath);
  * she watches the flame 1330-1360;
  * ROAR 1360: flinch with an open-hand guard (head-relative), eyes shut, turning away;
  * rise and step back 1364-1393, arms down;
  * 1393-1439: turns three-quarters away to watch the far range, scarf whipping. No poster poses.
* Camera (v2): bigger, higher pull-back (46 m back, 5 m up, hfov 40->66) - she ends ~50 px tall.
* Dark adaptation (v2): before the catch the moonlit summit and her rim sit at a 17% floor (cut B
  is never an empty screen); the fire drops it to 4%; the reveal brings the world up.
* The reveal's world = the shepherd's world:
  * v8 grids are baked from `shots/montage/s1_peak.py` `h_far`/`h_cloud` around the top of s1's
    first-beacon massif (-5450, 282, 31997 in s1 coords), with s1's moon;
  * radial resolution is doubled (the old smear);
  * s1's snow-hold threshold, rock albedo and moon level are passed from beacon.py.

**Done**
* Baseline check: local render matches delivered frames — f1420 51.1 dB (dither only). f1230/1300 are
  41.5 dB with a ~1.5-level background offset, because the delivered 1200-1359 predate TRUE_PEAKS.
  => Plan: re-render the WHOLE take 1200-1439 into hills_v2 (no join inside the take).
* `heroine_sdf.py` (new, numba): 3-D SDF sphere tracer for her only. Primitives: ellipsoid, round cone
  (+ folds), round box, tapered/tilted torus arc, bent half-space, eyelid shell with an angular fissure
  (+ fissure re-carve mode). Tile + per-ray culling, 3x3 SS on edges, soft shadows, AO, wrap/SSS skin,
  GGX, wool sheen, Kajiya-Kay hair, cold rim, head-space tints (lips/brows/flush), iris/pupil.
  `sdf_points` = debug evaluator.
* `heroine.py` (new): Builder (ordered CSG per group), materials, head sculpt fitted to a female
  anthropometric profile (within ~3 mm; see lookdev), eyes/lids that read in profile + 3/4, hair cap.
  Lesson: keep smooth-union k small (<=0.014) — smin inflation accumulates across overlapping masses.
* Just appended (UNTESTED): ik3/skeleton/hand()/build_figure()/scarf_tail() — the body, coat
  (folds), sleeves + cuffs, fingered hands, steel + flint, legs/boots, rock, scarf loops + tail +
  fringe, and hair locks.
* Lookdev tools + latest images: `~/mishamisha/_local_logs/heroine_lookdev/` (lookdev.py is the
  helper; t_debug.py = profile-vs-target + prim-ID map; t_zoom.py = eye zoom; t_prof.py = clay +
  firelit profile). Run them from that folder with the venv.

**Next steps**
1. Smoke-test build_figure in lookdev with a kneeling pose. Fix IK and hand orientation, then iterate the
   hands (flint and steel grips) and the coat.
2. beacon.py: add a `HEROINE_V2 = True` flag that gates everything. Kneeling on a rock beside the cairn,
   eyes ~1.14 m, pelvis ~(0.66, 0.56). Add yw2_pose(f) choreography:
   * strikes 1236/1262/1290 (raise, snap down; sparks come from the new flint anchor — same RNG
     call order, so the embers are unchanged);
   * cupped hands and blowing 1296-1318 (purse expression, breath aimed at the ember);
   * catch 1318: face lit from below, breath fog;
   * 1318-1360: near hand shields the flame;
   * 1360: flinch (head away, eyes shut, arm up), step back and rise, then stand firm 1385+.
   Scarf and hair chain anchors come from the new rig.
3. Compositing: render her with heroine_sdf (lights = strike flash / ember / flame / roar, plus rim and
   ambient). Mask the flame by her near hand, then add breath fog, hair strands and lashes.
4. Half-res tests (1240, 1262, 1290, 1320, 1340, 1362, 1372, 1390) → iterate → full-res 1200-1439
   into renders/hills_v2 (2 workers) → before/after sheet at `_local_logs/review/heroine_v2.jpg`.

# CODA v2 (CODA dept, 2026-09-26): INTRO + CODA figures, cairn, answering fires -> renders/hills_v2

Fixes critic_tone B1 + M6 (HILLS parts). `V2 = True` in `intro.py` and `coda.py` (False = v1; v1 copies of
both files in `shots/hills/v1_backup/`). New modules (opt-in, nothing shared was changed; beacon.py untouched):
* `silhouette.py` - **merged-silhouette renderer**: every group's SDF is unioned BEFORE lighting; depth below
  the outline = exact distance transform of the union (no seams between primitives); rims (sky + fire) only on
  the union's OUTER contour, the sky rim taken from the blurred background behind each edge; soft 2-D shadow
  march for the fire (thin things - torch, fingers, wisps, fringe - cast none). Scarf = the only lit cloth.
* `figures2.py` - `elder2` / `child2` (same pose dicts + anchors as characters.elder/child): spline-cut
  outlines (tailored long coat, flared back hem with fold scallops that move in the wind, heavy sleeves with
  cuffs, gripping hands, ankle boots; child's quilted puffer, hood, puffy sleeves, mittens, snow trousers,
  beanie with fuzzy pom-pom), profile faces left UNLIT (1-2 px fire edge only), sheen <= 0.4 on cloth/skin.
  **Scarf** = heroine.py's wool (albedo 0.29/0.022/0.019): two rolls + crossing + short front end, and a
  1.05 m tail (verlet chain 20 x 0.053 m) that tapers, twists (face normals turn in/out of the firelight),
  ripples, and ends in 9 yarn tassels; a knitted rib runs along it (in the ribbon's own frame), like the
  young woman's knit. Grey flyaways at the bun are fine curled strands, not wires.
* `cairn2.py` - `DryStoneCairn`: flat-bedded irregular stones (broken ends, chipped corners, staggered joints,
  pinning stones, capstone), hard chamfered arrises, per-face tilt, grain/lichen, dark hearting in the gaps.
  Basket and every dimension identical to characters.Cairn (fire base unchanged for the ember title); the
  woodpile is rebuilt as split logs with flat-cut ends (teepee + cross logs through the bars + kindling).
  Basket + wood drawn as a merged silhouette; the iron bars are redrawn over the beacon's flame (dark cage).
  *For the HEROINE lane (optional, their call):* beacon.py's cairn is still the v1 pillow stones. Drop-in:
  `CAIRN = cairn2.DryStoneCairn(seed=11)` (same top/bk_bot/bk_top), then draw the stones with
  `CAIRN.render(cam, [0, 0, 0], lights, amb_top, bg=img)` (returns y0, x0, rgb, a for over_region) and the
  basket/wood with `silhouette.Silhouette([0, 0, 0], CAIRN.basket_groups(x=0.0)).render(cam, lights, amb_top,
  np.zeros(3), bg=img)`; lights in the same (n,8) world format beacon.py already builds.
* `fires2.py` - distinct distant fires: supersampled bonfire-shaped flames with overshoot on ignition, a burning
  pile glow, tight air-scatter glow (no bokeh halo), hillside light, a lit smoke wisp occluded by nearer
  ridges; placed on the BARE terrain (v1 ridge peaks included treetops); `crest_texture` breaks up the
  firelit hilltop. Also a corrected plume (`smoke_wisp`): **fire.render_smoke streaks horizontally late in the
  film** (it divides Y - t*rise by a height-dependent width; at t ~ 110 s the vertical frequency explodes) -
  other lanes using it after t ~ 60 s should check their smoke.

Timing kept (all keyframes, camera moves, sky). Changed per director: the child's beacon **catches at 2639 and
whooshes with overshoot ON 2640** (music hit v2 2800). Answering fires = `fires2.PLAN`: 14 fires on successive
ridgelines, outward and alternating sides; the 12 voiced ones ignite ON the SFX whumps (src 2645 2651 2657 2663
2669 2675 2682 2689 2696 2703 2710 2716 = v2 2805..2876), 2 far unvoiced ones between; plus the 7 INTRO
festival fires. `renders/hills_v2/coda_fires.json` has frames (src + v2), world positions, flame heights,
screen x/y and pan at ignition. TITLE: re-extract `edit/ember_title_fires.py` (its `_extract()` works
unchanged: coda.answering_fires / place_beacons API kept); the SFX pans could be matched to the JSON `pan`.

**Delivered (2026-09-26 15:0x):** `renders/hills_v2/` src 0-359 (INTRO) + 2460-2807 (CODA), full res, 708 frames,
26 min with 2 workers on a quiet machine (coda 740 s, intro 832 s; ~20 s/frame under heavy load). Before/after
sheet: `~/mishamisha/_local_logs/review/coda_v2.jpg`. Launcher: `~/mishamisha/_local_logs/r_coda_v2.sh` (renders to
`renders/hills/tests/stage_v2`, then moves the complete set into hills_v2 so the edit never mixes versions).

Re-render: `python shots/hills/render.py --shot coda --frames 2460-2807 --workers 2 --out renders/hills_v2`
(and `--shot intro --frames 0-359`). Lookdev: `_local_logs`-style scripts are in the session scratchpad only.

# HILLS: notes

Deliverables (global frame numbers, 1920×804, `look.save_png`) go to `renders/hills/f_%05d.png`:

* **INTRO** 0–359
* **FIRST BEACON** 1200–1439
* **CODA** 2460–2807

Also in `renders/hills/`: `preview.mp4` (the three pieces back to back), `contact.png`, and
`coda_fires.json`. The JSON holds the ignition frame of every answering fire in the CODA
wave. The first 18 are marked `voiced`, for the bell cascade. Test renders are in
`renders/hills/tests/`.

## How to re-render

```
cd the-long-dawn
python3 shots/hills/render.py --shot intro  --frames 0-359     --workers 2
python3 shots/hills/render.py --shot beacon --frames 1200-1439 --workers 2
python3 shots/hills/render.py --shot coda   --frames 2460-2807 --workers 2
#   --scale 0.5 for tests, --out <dir>, --step N, --skip-existing
```

Each worker is one process with `NUMBA_NUM_THREADS=1`. Every frame is a pure function of
its frame number: the simulations (scarf, hair and wisp verlet chains, particle systems)
are deterministic and re-run from the start of each piece in every worker. After editing
a shared `@njit` helper in `core.py`, `sky.py` or `land.py`, run `shots/hills/clear_cache.sh`,
because numba's disk cache does not track cross-module inlining.

## What is built (all procedural)

* `core.py`: the pinhole camera (metres; x right, y up, z forward), easing and keyframe
  tracks, numba Perlin noise and fBm, energy-normalised gaussian, disc and streak
  splats, and compositing.
* `sky.py`: the twilight gradient with an asymmetric afterglow, a 14–26k-star catalogue
  (magnitude power law, colour, twinkle, horizon extinction, visibility against sky
  brightness), and the Milky Way. The Moon is a 2× crescent with earthshine and clustered
  warm city lights on its night side. The **orbital ring** is a geometric equatorial ring
  seen from 57°N. It is sunlit toward the sunset and cut by Earth's shadow on the far side,
  with faint station nodes.
* `land.py` + `hillworld.py`: 8 ridge "cards" at real distances (480 m–56 km), each
  defined by the apparent angle of its crest. They have trees, analytic anti-aliased
  edges, aerial perspective, noise-varied valley mist, beacon glow on the hills and a
  depth buffer. (Their moonlit-snow pyramid mode is no longer used; see `peaks.py`.)
* `peaks.py` (FIRST BEACON only, imported by `beacon.py` alone; `TRUE_PEAKS` flag): a real 3-D
  moonlit Himalaya. Ridged-multifractal range on a polar grid (azimuth x log-range, so each
  cell is about a pixel at any distance, octaves prefiltered by footprint), Earth curvature,
  a cloud-sea top surface that thins where peaks pierce it, precomputed soft moon shadows,
  snow/rock by slope, exponential height haze toward the sky's horizon colour, and a 10 cm
  summit grid (flat top for the cairn, serrated rocky arete, calm snow face under T9).
  Column-coherent per-pixel ray march, 1.3x supersampled at full res. The grids are cached in
  `shots/hills/cache/beacon_peaks_v7.npz`; delete it after changing a height function.
* `puppet.py` + `characters.py`: 2-D SDF puppets on camera-facing cards: smooth-union
  capsules, ellipses and polygon profiles, two-bone IK, and verlet scarf, hair and wisps.
  **Backlit-silhouette shading** gives near-black interiors, a thin cool rim from the sky
  behind, and a thin warm rim where the fire grazes an edge. The scarf is the only colour:
  dark oxblood that glows ember-red where the fire reaches it, with a fringed tail. The
  Elder, the Child (pom-pom hat) and the Young Woman share one rig style. The same scarf
  function is used for the Elder and the Young Woman.
* `fire.py`: the HDR flame (multi-scale domain warp, torch and bonfire envelopes, splitting
  and tearing tongues, white-yellow core low, broken red tips), ignition with
  whoomp-and-overshoot, flicker, distant beacon points with halo, spark and ember particles
  (tapered hot-head streaks, cooling colour, curl-noise meander, bokeh when defocused),
  smoke lit only near the fire, and a veiling glow.
* `intro.py`, `beacon.py`, `coda.py`: the three pieces. `render.py` is the CLI.

## The pieces

**INTRO**
* 0–159: wide blue-hour establishing shot. Beacons ignite one by one at 22, 80, 112, 138
  (then 196, 238 and 262 behind the two-shot). The Child points at the fire at 80, turns
  (about 104) and looks up at the Elder, under T1.
* 160–359: medium two-shot (cut on the bar-3 downbeat). The Elder looks down at the Child,
  then out to the fires (T2). From about 228 the camera pushes into the torch, the wind
  drops, and the frame tilts up a column of rising embers over the flame's glow. By 300
  this matches the first EMBERS frame: a warm glow at the bottom centre and embers rising
  on near-black.

**FIRST BEACON** is one take.
* 1200–1235: black, with a breath of snow.
* Strikes at 1236, 1262 and 1290 each give a flash and a spark shower into the basket, and
  light the hands, the steel, her profile and the scarf wrap. They grow in strength.
* A spark lives in the tinder and pulses as she blows. At 1318 the kindling catches,
  lighting her face from below, and her breath shows.
* At 1360 the beacon roars. She rises with an arm up against the heat. The camera pulls
  back and up, with an explosive ease-out, to the snowy summit, a sea of moonlit peaks and
  cloud, spindrift, the stars and the Milky Way. There is no ring or lit Moon, because this
  is sixty years earlier. The moonlit world fades up as it is revealed.

**CODA**
* 2460–2627: medium shot. The Child looks up (T15). The Elder meets her eyes, looks away
  and says nothing. A gust lifts the scarf at 2548–2618.
* 2580–2618: the Elder lowers the torch into the Child's hands and moves her hand to the
  Child's shoulder.
* 2628–2807: wide. The Child lifts the torch into the basket, and it **catches at 2640**.
  About 50 answering fires ignite in a rolling wave outward and across the valley between
  2645 and 2720 (see the JSON). The INTRO fires are still burning.
* From about 2662 the camera cranes up into the sky: the ring, the lit Moon, stars and four
  slow ship lights. The centre of the frame is dark sky for the title.

## Deviations from the bible

* **The Moon is drawn at about 2× real size** so that its city lights read.
* **The orbital ring is placed for the picture, not exact astronomy.** Its sky path is
  physical: an equatorial ring at 2.6 Earth radii seen from 57°N. The Earth-shadow cut is
  placed where it reads in frame.
* **FIRST BEACON's reveal is now a true heightfield** (`peaks.py`, Sep 2026), replacing the
  pyramid ridge cards, which read as stage flats. Only 1360-1439 were re-rendered; before 1360
  the range sits at 3% behind the close-up's defocus, so the join is invisible.
* **Cairn scale.** The cairn is 0.76 m, with the basket rim at about 1.12 m, so the Child
  can reach it.
* **Frame edges.** INTRO starts at full exposure; the edit handles the fade-in. The CODA
  ends on the sky; the edit fades it to black.
