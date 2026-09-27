# >>> HEROINE-B STATE (27 Sep ~20:57Z, before the usage gap; B2 THE CLIMB only, B 640-879 -> renders/climb_B) <<<
**JOB READY climb_b sent to main ~20:50Z** (240 f, full res, ss 1.5, ship jpg; ~25 min on 2 cpu-8 nodes). On approval:
`python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/climb_b.json --nodes 2` (from ~/mishamisha). If it ran
while I was idle: check `renders/climb_B` has 640-879 (`farm.py ... --missing` fills gaps), then contact-sheet it.
**Code: `shots/hills/climb_b.py` at cd9fbb1 (pushed)**; job `cloud/jobs/climb_b.json`. Deps (read-only): RUN-B's bworld,
bset (person2, CAIRN, BEACON, path_at, night_params, match_horizon), vigil (moon_at(F0) az 80 el 14; draw_sky frozen
at 1360), pipe, rcam, keeper, fire2, bfig + bprops (committed edfa370; cairn3), handback_b.sleeve_arm; HEROINE-L's
heroine / hsdf3 / heroine_v3 only for `CLIMB_FIG=h` (the old 3-D figure, superseded by the director's call).
* WIDE 640-799: locked, 115 m behind her right shoulder, 14 m above her feet, hfov 30. She is 50-52 px on the crest
  (s 34 -> 32 m below the lip), the pot at her right hip with a warm core + halo; the summit (cold beacon + cairn3) on
  the skyline; the far ranges beyond. No moon shadow here (it aliases on the grazing crest).
* CLOSE 800-879: locked, 20 m behind her right shoulder (25 deg off her back), 2.2 m up, hfov 32. She is 270-290 px,
  walking up toward the lip; the beacon + cairn3 on the skyline right; her long moon shadow; ground drift grains;
  the pot's warm pool on the snow (only her body shadows it).
* Her = `keeper_carry()`: bset.person2('stand', arms=False, age 0.9) + sleeved arms + the pot (B3's vessel in profile,
  a leak of light under the lid, the chip) + heel lifts; drawn by bfig.render (FG.render fallback).
* bworld's crest RISER (0.36 m vertical step ~25 m below the top, reported to RUN-B-3 in shots/run/NOTES.md) is patched
  in OUR G-buffer copy only (`Shot.patch_risers`): near-vertical ground pixels near the summit get the up-normal.
**Check stills:** renders/_farmtest/climb_b/ (700/760/840, before the glow fix), climb_b_v2/ (720/850, glow fix),
climb_b_v3/ (760/850, + pool + riser patch; landing ~21:00Z); half res: renders/heroine_tests/climb_t4/.
**Nitpicks for after 23:30Z (cheap re-render):** the pot reads weakly in the close (a bigger warm halo, or hold it a
little further out); spindrift in the wide is faint (the view faces away from the moon: back-scatter); the foreground
snow's dark stone specks are RUN-B's texture; if RUN-B-3 fixes the riser, `patch_risers` becomes a no-op.
**Rejected wides (don't retry):** side-on or low views of the crest show bworld's 0.36 m h_rock STEP ~25 m below the
top as a fence of dark posts (told RUN-B-3 in shots/run/NOTES.md); the 3-D hsdf figure (the director wants B's
shared keeper); a low NW-flank view hides the summit behind the dome.
**Status:** farm --test 700/760/840 requested ~20:45Z (renders/_farmtest/climb_b/); local half-res check (climb_t4)
queued. NEXT: look at the stills -> JOB READY climb_b (240 f; est. ~25 min on 2 cpu-8 nodes) -> on approval
`farm.py cloud/jobs/climb_b.json --nodes 2`. Nitpicks for after 23:30Z: spindrift density/brightness, the pot's
readability in the close, bfig's look on the carry arms.

# >>> STATE AT HANDOFF (HEROINE-L, 27 Sep ~20:40Z; session ended by the director for context size) <<<
**DONE: H1 THE ROAR, final for all three cuts.** The director approved job `h1_v3_roar2` (shot `beacon_v3_roar2`,
`beacon.apply_roar2()`). 80/80 frames (src 1476-1555) landed IN PLACE in `renders/h1_v3h5` at ~20:23Z and match the
approved check (mean |diff| 0.5/255); src 1200-1475 is unchanged. Her forearm is across her face by ROAR+1.5, her head
turned away and her weight back. She is a dark shape against the flare (fully flagged from the fire's key and bounce,
the exposure stopped down 34% and eased back by ROAR+40), and her far fist is empty. Sheet:
`review/v3/heroine_L_roar2_check.jpg`.

**B3 THE DEAD EMBER: APPROVED by the director (~20:35Z) and LAUNCHED on the farm** (`h5_deadember_b_{a,b,c}.json`,
`--nodes 3`, into `renders/deadember_B` in locked B numbering 880-1119; request log
`renders/heroine_tests/farm_deadember_final.log`).
* NEXT for whoever picks this up: when the request is DONE, check that all 240 frames decode (1920x804), look at a
  contact sheet (especially 920-944, where the hand comes back from the lid, and 1100-1119), then tell main. EDIT's B3
  row reads `deadember_B` first. If frames are missing: `farm.py <the three jobs> --missing --nodes 3`.
* Check stills: `renders/_farmtest/heroine_chk2/deadember`; sheet `review/v3/heroine_L_deadember_check.jpg`.
* What changed (`heroine_v3.DeadEmber`, `CRAFTED = True`):
  * A crafted clay pot: a carinated shoulder, a flat foot, a rolled lip, lugs and a knotted thong.
  * The far glove wraps the pot's lens-side belly after the knock.
  * The ground is graded down (M_SNOW x0.4), and the glove has a moon rim of 1.6.
  * The match cut: the last red point lands at 1100 on (1654, 303), which is strike 1's spark in B4 under EDIT's
    `B_H1_CROP` (0.17, 0.22, 0.26, 0.26). If EDIT re-frames that crop, move `DeadEmber.SPARK_PX` and re-render
    1040-1119.

**H1-C: RETIRED from this lane** (director, ~20:35Z). C14's hands (strikes, the find, the fist, the catch) become one
hands-only BLENDER sequence built by the Blender department's successor to match C15; EDIT keeps its fallback
(`h1_v3h5` with grade 'C').
* My h1c code stays in `beacon.py` / `render.py` behind flags (`H1C`, `H1C_POV`), but it FAILED the gate in both
  looks:
  * Telephoto (`review/v3/heroine_L_h1c_tele_FAIL.jpg`): the take's sleeves read as giant mottled gourds.
  * POV (`review/v3/heroine_L_h1c_pov_FAIL.jpg`): her own shoulders fill the lens as domes.
* Do NOT launch `h1c_a/b`, and do not add EDIT's `T('h1', 0, 'v3')` line.
* **Facts for the Blender successor:**
  * The master timing in C frames: strike 1 at 2980, strike 2 at 3009 (in the find's flash), the fist at 3140,
    strike 3 at 3178, the blow ~3204-3300, the catch at 3316, the roar at 3360 (= src +1744 before the find,
    +1884 after).
  * The flint is in her LEFT hand, which also takes the Ring. The C-shaped fire-steel is in her RIGHT hand.
  * The steel: flat stock 7 x 2.4 mm, a C with scroll ends (`heroine.py` tools `both_c`, `heroine_v3.FireTest.steel`).
  * Thin dark leather gloves (albedo ~0.03, 0.021, 0.016). Short orange sparks that fall and curve. The head never in
    frame.

**Handed on:** B2 THE CLIMB goes to HEROINE-B (their block above); the bar-63 hand-off to RUN-B-3; every Ring close-up
to MONTAGE-3D-4.
**Files:**
* Code: `shots/hills/{beacon,heroine,heroine_v3,render,hsdf3}.py`.
* Jobs: `cloud/jobs/{h1_v3_roar2,h5_deadember_b_a,h5_deadember_b_b,h5_deadember_b_c,h1c_a,h1c_b,heroine_chk,heroine_chk2}.json`.
* Checks: `renders/_farmtest/heroine_chk{,2}/`. Farm logs: `renders/heroine_tests/farm_*.log`.

# >>> HEROINE-L STATE (27 Sep ~19:40Z; SUPERSEDED by the handoff block above) <<<
**Done (committed cd96111, pushed).** All new work is behind flags; `beacon_v3` (the landed take) is unchanged.
* **THE ROAR.** The cloud lane's roar fix IS in `renders/h1_v3h5` 1474-1555. My read at 1:1: she is darker and the
  recoil reads from 1478 on, but 1477 is still an arm thrust at the fire, she is still lit a smooth brown (doll-ish at
  1:1), and the far hand shows a slab of steel (a phone read). Re-polish = shot `beacon_v3_roar2` (`apply_roar2()`):
  fully flagged from the fire's key (x0.2) and bounce (x0.03), the shawl at 0.95, the exposure stopped down 34% for the
  flare and eased back by ROAR+40, the guard up by ROAR+1.5, both tools gone at ROAR+1. Frames <= 1475 are identical,
  so job `h1_v3_roar2` re-renders 1476-1555 IN PLACE in `renders/h1_v3h5` (80 fr, ~10 min on one box). NOT YET SEEN.
* **H1-C** = shot `h1c` (`apply_h1c()` + `render.H1CShot`), C numbering -> `renders/h1_C`: C 2960-2999 = src 1216-1255
  (strike 1 = C 2980), C 3150-3359 = src 1266-1475 (strike 3 = C 3178, catch C 3316). A telephoto window (`camera_c`,
  hfov 12.8, from beside the take's lens) on her gloved hands, the flint and the tinder in the basket. Checked by
  projection: her head never enters it (nose >= x 2144 px of 1920). In the take her nose is 11 cm from the tinder
  while she blows, so in H1-C she blows from where she kneels (BLOW -> KNEEL) and the breath jets in from off-frame
  right. Strike 2 is not struck (no sparks, flash or hand move): THE FIND's, at C 3009. Glove = dark thin leather
  (`H1C_GLOVE`, the take's brown read as skin in the strike light). Steel = the C-shaped fire-steel (heroine.py tools
  `both_c`: its straight back is where the old bar's scraping edge was, so every strike still lands; the ends curl
  round the fingers to two scrolls). Jobs `h1c_a` / `h1c_b` (125 fr each). NOT YET SEEN.
  **EDIT needs one line:** in both C14 H1-C rows of `edit/edl_v3.py`, put `T('h1', 0, 'v3', 'H1-C hands and flint')`
  before `h1(...)`, so the EDL reads `renders/h1_C/` in C frames; grade it like MONTAGE's `ring_C` (both or neither).
* **B3 THE DEAD EMBER**: the pot is a crafted vessel (`DeadEmber.CRAFTED`: a carinated shoulder, a flat foot, a rolled
  lip, two pierced lugs, a leather thong knotted round the neck; fired clay: displacement 1.1 -> 0.35 mm, bump 0.10 ->
  0.03). Jobs `h5_deadember_b_a/b/c` (880-959 / 960-1039 / 1040-1119 -> `renders/deadember_B`). The 16:00Z stills at
  890/904 (`renders/heroine_tests/deadember_chk`, the OLD ball pot) read as a lid off a glowing vessel, but the pot was
  a lumpy brown ball (potato / coconut) and the holding glove was dark on dark.
* **Check stills: NONE yet.** My local stills were frozen by the director at 18:54Z (pids 10385-10387 in
  `_local_logs/renderq/frozen_pids.txt`). I asked main (19:40Z) to launch `cloud/jobs/heroine_chk.json` (13 key frames:
  roar2 1477/1479/1484, h1c 2980/3178/3250/3316/3345, dead ember 890/960/990/1030/1100 -> `renders/heroine_chk/*`), or
  to give me one local slot after the thaw. Then JOB READY per full job.
* **Superseded, removed:** `h2_find_c`, `h2_firetest_c` (MONTAGE-3D-4 owns every Ring close-up), `h5_deadember_b` (split).
**SCOPE (director, ~19:50Z):** HEROINE-L keeps H1 (roar2, H1-C) and B3 THE DEAD EMBER. **B2 THE CLIMB belongs to the new
agent HEROINE-B**; HEROINE-L does not start it. The handoff notes are below.

## FOR HEROINE-B: THE CLIMB (B2, B 640-879 -> `renders/climb_B` in B frames; EDIT's B2 row reads `climb_B` first)
Measured on bworld `b15` at 19:45Z (RUN-B-3 is still reshaping the far field, so read `BW.VERSION`, never bake a plate):
* **Set** (`shots/run/bset.py`, read-only): TOP (-5457.7, 301.5, 31998.3); LIP (-5452.9, 300.9, 31998.9), the NE lip
  where the path leaves the summit; BEACON (-5456.6, 301.2, 31998.3); CAIRN (-5455.2, 300.8, 32003.3).
  `BS.path_at(s)` = the NE path, s m down from the lip along the NE ridge crest (`BW.RIDGE_AZ` 40; the local ridge
  direction at s=60 is az 44, its normal az 134/314). Heights below the lip: s=20 -1.7, 40 -3.1, 60 -4.5, 80 -6.4,
  100 -8.7, 140 -14.7, 200 -24.9, 300 -40.8. It is a broad, ROUNDED ridge (B's landform, not a knife-edge). The
  cross-section at s=60 drops 1.5 m at +-10 m, about 5 m at +-20 m, 14-17 m at +-40 m and about 30 m at +-80 m; the
  slopes steepen beyond +-20 m.
* **Moon** = the vigil's (`VG.moon_at(VG.F0)`): az 80, el 14, low in the east. She climbs SW (az ~224) toward the
  summit, so the moon is behind her, on her left.
* **My intended cameras** (untested ideas only):
  * WIDE 640-799, locked, from the NW side at about her height. For example, 100 m out along az 314 from path_at(60),
    3-5 m below her, looking SE with hfov ~32. That puts her at ~50-60 px on the crest, walking screen-right toward
    the summit. She stands against the moonlit cloud sea (`BW.CLOUD_Y` -650) and the layered far ranges. The moon is
    ~50 deg left of frame, so the crest rims and backlit spindrift glow on the lee side. The pot is in front of her,
    the only warm point.
  * CLOSE 800-879 (bar 11 b1), locked, behind her and below on the path, looking up the ridge (SW) at the summit
    shelf on the skyline, with her at ~120-150 px. The shawl is on her back; the hidden pot's glow pools warm on the
    snow ahead of her feet and rims her edges. Time can jump between the cuts (the wide near s~60, the close near
    s~12), since she must reach the cairn by B3 at 880.
* **Pipeline pattern** = `shots/run/reveal.py` (per-frame `BW.build`) or `vigil.py` (one cached G-buffer for a locked
  camera, then per-frame `BW.shade`). Build it once per camera and re-shade per frame. The pot's light on the snow
  goes in the terrain shader's one fire light: LP[36] intensity, LP[37:40] position, LP[40:43] colour, LP[43]
  radius (the beacon uses 9-17 and 40; the pot needs far less). The figure: `bset.person()` is a 2-D SDF billboard
  that holds at 60-120 px (`age=0.85` as in the vigil: stooped, white hair at the hood's edge; `staff=False`, since
  both hands are on the pot). It has no carry pose, and 'walk' swings the legs in screen-x, which is right for the
  side view and a waddle from behind. So write your own carrier variant in your own file rather than editing
  RUN-B-3's `bset.py`. Add the pot's glow with `fire2.halo`, and the stars and band as reveal.py does
  (`KP.draw_stars`, `BW.add_band`).
* **Old material:** `heroine_v3.Climb` / `climb_layer()` / `ClimbWideTest` (the 3-D hsdf figure). The close from
  behind FAILED in five framings with it (REPORT 2 below). The sheet's fallback is the wide alone.
* **The pot:** keep it the same vessel as B3's crafted pot (`heroine_v3.DeadEmber.CRAFTED`): squat, a carinated
  shoulder, a rolled lip, lugs and a knotted thong. At 60 px only its glow reads.

## FOR MONTAGE-3D-4 (the cut points around C14/C15; from their NOTES of 18:45Z and EDIT's C table)
* C 2960-2999 H1-C (HEROINE) | 3000-3079 find_a (MONTAGE: strike 2's flash and sparks at 3009, her LEFT glove) |
  3080-3149 find_b (MONTAGE: the fist at 3140) | 3150-3359 H1-C (HEROINE: strike 3 at 3178, the catch at 3316) |
  3360-3599 fire (MONTAGE: the roar at 3360). No overlap and no gap. H1-C never shows strike 2.
* Hands: the flint is in her LEFT (near) hand, which also takes the Ring in the find, so the Ring rides in that fist
  through strike 3. The C-steel is in her RIGHT hand, as in your fire test. My H1-C steel is flat stock, 7 x 2.4 mm,
  bent into a C round her fingers (a 5.4 cm straight back, 1.25 cm corners, short arms, 4.5 mm scrolls). If your fire
  test's steel differs, tell me and I'll match it. My glove is dark brown-black leather, albedo (0.030, 0.021, 0.016),
  roughness 0.34.

# >>> STATE AT HANDOFF (HEROINE-L takes over the HEROINE lane, 27 Sep ~18:40Z) <<<
**Import.** The cloud lane (account unreachable) left its state on `claude/v3-heroine` (head b86c2a0). Its own files were
brought onto `claude/long-dawn-v2` by explicit path (commit 750222a; no merge): `heroine_v3.py`, `hsdf3.py`, `beacon.py`
(all v3 work gated behind `V3_REKEY` / `V3_H5`, default off, so RUN/mountain callers are unchanged), `render.py` shots,
this NOTES, and 7 job files in `cloud/jobs/` (`h1_v3_{rekey,master,h5,h5_roar}`, `h2_find_c`, `h2_firetest_c`,
`h5_deadember_b`). Its review sheets are on disk (untracked) in `review/v3/heroine_v3_*`. Tracking ref deleted.
**On disk at takeover.** `renders/h1_v3h5` = 356/356 JPEGs (src 1200-1555); **1474-1555 are the roar-fix re-render**
(landed 15:39-15:45Z via the importer), 1200-1473 the 06:21-06:58Z H5 render. `renders/deadember_B`, `climb_B`,
`heroine_C`: none. No HEROINE process running. Local python: `source ~/.venvs/longdawn/env.sh` (numba 0.67).
**Scope now (director, ~18:30Z):** (1) THE ROAR on H1 (src ~1476-1510): a DARK SHAPE against the flare; the recoil =
forearm across the face, head turned away, weight back. (2) H1-C: C14's flint take, hands and flint only; the find
(C 3000-3149) and the fire test (C 3360-3599) are MONTAGE-3D-4's (`renders/ring_C`), so H1-C = C 2960-2999 + 3150-3359
(EDIT's C14 rows). (3) H4 THE CLIMB (B2, B 640-879). (4) H5 THE DEAD EMBER (B3, B 880-1119 locked numbering). NOT
ours: the bar-63 hand-off (RUN-B-3); `h2_find_c` / `h2_firetest_c` are SUPERSEDED by MONTAGE-3D-4's Ring close-ups
(do not launch). Heavy renders go through the FARM (`cloud/farm.py`, being built) with JOB READY + check stills.

# >>> HEROINE-v3 STATE (2026-09-27 ~16:05Z; PAUSED for the usage window, RESUME ~20:00Z) <<<
**Exact state.** Nothing of this lane is rendering (the roar job finished 15:46Z; no process left running). Everything is
committed and pushed to `claude/v3-heroine` (= `claude/heroine-character-work-0o4wv2`). H1 is DONE for all three cuts:
356/356 JPEGs on `claude/render-h1-v3-h5` (head c55be10, the roar fix included). The C close-ups (the find with the side
glove and the meltwater sheen, the fire test on the C-steel) and B's dead ember (restaged, B 880-1119) are verified as
stills; their jobs are written and NOT launched (director launches): `cloud/jobs/h2_firetest_c.json`,
`cloud/jobs/h2_find_c.json`, `cloud/jobs/h5_deadember_b.json` (-> `renders/deadember_B`). The bar-63 hand-off is RUN-B's.
**Next steps on RESUME:** (1) `git pull --rebase origin claude/long-dawn-v2`; (2) for each close-up job the director has
launched, read its `cloud_logs/<job>_status.txt` on its branch, then check the landed frames (decode, luma, a contact
sheet in `review/v3/`) against the stills in REPORT 4; (3) polish if asked: H1's far-hand steel to the C-steel at the
roar (~6 frames), the find's hard dark foreground (her knee), the dead ember's near glove (dark on dark), the fire
test's sleeve-cuff weave.

# >>> HEROINE-v3 REPORT 4 (2026-09-27 ~16:00Z; the director's H1 review + scope change) <<<

**H1 roar fix (director 15:15Z): DONE and RENDERED.** At the roar (src 1476-1510) the flare no longer lights her into a
smooth brown doll: from ROAR-1 she is FLAGGED from the fire's key (its light on her x0.55, the warm bounce x0.15, and her
hood, coat, sleeves, gloves, legs and shawl go rim-only, `V3_ROAR_SIL` 0.92): a dark shape against the flare with a thin
warm edge, easing to half by ROAR+75. The recoil is re-posed (`FLINCH_V3` + the V3 guard in `yw2_pose`): the near
FOREARM lies level ACROSS HER FACE between the flare and her head (the elbow thrown out toward the fire at eye height,
the wrist at the brow, the hand carried round behind the hood, the thumb tucked: no hand held up in the air), her head
turned right away (head_yaw -85, chin down 36), her weight back (lean -18, pelvis +3 cm). Job
`cloud/jobs/h1_v3_h5_roar.json` (src 1474-1555, 82 frames) ran on this box 15:35-15:46Z: **82/82 JPEGs pushed onto
`claude/render-h1-v3-h5`** (head c55be10), replacing 1474-1555 in the 356-frame set (1200-1473 unchanged). All 356 decode
at 1920x804, none at <= 20/255 (darkest mean 31.9). Sheet `review/v3/heroine_v3_h1_roar_fix.jpg` (1474-1510). **The take
is done for all three cuts.**

**Scope change (director):** the bar-63 hand-off is RUN-B's own shot (silhouette figures at hand-back scale): no hands
from this lane. Priorities: the roar fix (above), then C's hands (the find, the fire test) with the canonical Ring.

**C with the canonical Ring.** Every Ring here reads `assets/ring` (MONTAGE-3D's script of fire, the revised letterforms
of e7083c6: lean, flickering tips, uneven bellies, a spur) at the canonical band proportions (inner radius 8.8 mm);
checked close on the fire test: the letters upright on the outer face and readable on the inner, deep orange-red.
* THE FIRE TEST (C 3392-3599, fallback to the Blender fire test): the Ring on the up-turned tip of her C-shaped
  fire-steel in the flames; no glow fringe round the hand. PASS against its fallback (the band in the coals, her fist).
* THE FIND (C 3009-3059, fallback to the Blender find): the clean pit, a **meltwater sheen** on the frozen disc (the
  moonlit far lip mirrored along the grazing edge, a slow ripple), no stain; **her glove comes in from the SIDE** (frame
  right, low, palm down, fingers loosely curled, the moon rimming the leather) over 3036-3058, and the edit cuts to her
  closed fist. PASS (the hand reads as a reaching glove, not a claw); `V3_FIND_HAND=none` = the band alone.
* B THE DEAD EMBER (B 880-1119, locked numbering) restaged: ~45 deg down, from her front-left; **the pot is HELD**, cradled
  on her near glove at her chest (her dark coat and red shawl behind it: **no snow wedge**, the far ground graded down);
  her far glove lifts the lid off the mouth (888-904) and sets it down below (the knock, 920); the breath is **visible**
  (982 blow, 1030 hanging); the last red point on 1100 still drifts onto H1's spark (the match cut). PASS.
Sheet `review/v3/heroine_v3_h5_closeups.jpg` (find 3030/3056, fire test 3440/3486, dead ember 900/990); stills in
`renders/heroine_C/stills`, `renders/deadember_B/stills`.

**RENDER_SPEC (director launches; one 4-core CPU box each, 2 processes x 2 workers, `--step 2`, ship jpg):**
* `cloud/jobs/h2_firetest_c.json` (firetest, C 3392-3599, 208 fr) -> `renders/heroine_C`: ~40 s/frame/process at full
  res, ~1.2 h. Fallback: the band in the coals and her closed fist (the Blender fire test is the A-side).
* `cloud/jobs/h2_find_c.json` (find, C 3009-3059, 51 fr) -> `renders/heroine_C`: ~20 min. Fallback: `V3_FIND_HAND=none`.
* `cloud/jobs/h5_deadember_b.json` (deadember, B 880-1119, 240 fr) -> `renders/deadember_B`: ~1.3 h. Fallback: the
  2-D ember insert (the eye greying) on RUN-B's plate.

**Remaining weaknesses.** The H1 steel in her far hand is still the v2b bar-and-loop (visible for ~6 frames at the roar);
the find's big dark foreground shape (her knee) is a hard edge; the dead ember's near glove under the pot is dark on
dark; the fire test's sleeve cuff carries a zigzag weave; the Ring close-ups are the fallbacks to the Blender versions.

# >>> HEROINE-v3 REPORT 3 (2026-09-27 ~10:50Z; the DIRECTOR'S H5 CALLS; superseded where REPORT 4 says so) <<<

**State.** H1 with every H5 call is **RENDERED**: `cloud/jobs/h1_v3_h5.json` (`render.py --shot beacon_v3`, src 1200-1555
master numbering, strike 1 = 1236, roar = 1476) -> `renders/h1_v3h5`, **356/356 JPEGs on branch `claude/render-h1-v3-h5`**
(JOB COMPLETE 11:04Z, 48 min on this box). Every frame decodes at 1920x804; per-frame mean luma 31.9-80/255 (darkest
src 1443), **no frame at <= 20/255**: her silhouette never vanishes. Contact sheet `review/v3/heroine_v3_h1_h5_contact.jpg`.
It supersedes `claude/render-h1-v3-master` (pre-H5) and the 00:42Z re-key. The
close-ups below are code + stills only; no close-up job is launched. B renders use LOCKED numbering and the B folders
(`deadember_B`, `climb_B`, `heroine_B`).

**H5 calls in my area (BIBLE_V3 top block), done / not done:**
* [x] Humans silhouette or gloved hands only; H1 v2b un-accepted: `--shot beacon_v3` = the v3 silhouette re-key + the
  master timing + `beacon.apply_h5_calls()`; the close-ups keep every head out of frame or a dark silhouette.
* [x] Thin leather gloves (no felt mitts): `hsdf3.gloves` (0.35 mm over the anatomical hand, M_GLOVE dark leather).
* [x] Flint sparks short, orange, falling and curving (gamma speeds, 0.10-0.32 s lives, gravity + drag + curl).
* [x] The f1366 flinch re-posed (`FLINCH_V3`): head turned away, forearm across the face, weight back.
* [x] A hood / wool cowl, no knit beanie; the red scarf as a woven wool shawl; no hair wires (`hsdf3.wardrobe_v3`).
* [x] B: H1's crushed frames lifted (`V3_FLOOR`, `V3_CLOSE_SKY`, pre-roar exposure lift, black lift 0.011): confirmed on
  the full render, every frame mean >= 31.9/255.
* [x] B: the fire-basket aged (`hsdf3.basket_v3`: rust, soot line, uneven bars, bent finials), 3-D, in H1 and the fire test.
* [ ] B: the dead ember at ~45 deg, a lidded vessel held by her hand, no blue wedge, a visible breath: NOT DONE (next).
* [ ] B: the hands for the bar-63 hand-off (her fire-steel pressed into the child's palm): NOT DONE (next).
* [x] C fire test: the Ring on the up-turned tip of her **C-shaped fire-steel** (flat stock forged into a C: the back in
  her fist, the upper end rounding into the long arm and hook, the lower into a scroll; broad face to the lens), in the
  flames; **no glow fringe round the hand** (bloom held off her glove and sleeve, the glove's rim gain down, the front
  tongues masked by her figure, the basket's far half composited behind the flames). The whole C reads in frame; the
  gold is gold, not white-hot. Tested 3440 (full res) and 3486 (the tip: the Ring edges over and holds).
* [~] C find: the clean pit (fallback, `Find.HAND = False`) kept; no rust-yellow stain in 3012 or 3050. NOT DONE: the
  meltwater / ice sheen on the disc (it still reads as a dark hole) and a side-entry glove test (else no glove at all).
* [x] C inscription: **THE SCRIPT OF FIRE** (MONTAGE-3D-2, `assets/ring`) on every Ring here: outer and inner strips
  mapped as `inscription.json` says (u CCW from the +axis end outside, 1 - that inside, row 0 = the +axis edge), the band
  at the canonical proportions (inner circumference / width 11.358, outer 14.137, inner radius 8.8 mm); the letters
  glow deep orange-red, never white (`hsdf3.inscription_canon`, `ring_xp_canon`; ring.py's strip only if missing).

**Next steps (RESUME ~15:00Z), in order:** (1) H1 H5 is done (above); (2) Find: the ice sheen + one side-entry glove test;
(3) DeadEmber restage (45 deg, held vessel, breath, no blue wedge) in B 880-1119 -> `renders/deadember_B`
(`cloud/jobs/h5_deadember_b.json` now points there; HOLD it until the restage is verified); (4) the bar-63 hand-off hands
as a layer for RUN-B -> `renders/heroine_B`; (5) review sheet, jobs, push, verify_session, the 5-line summary.

# >>> HEROINE-v3 REPORT 2 (2026-09-27 ~05:30Z; after the LOCKED beat sheets, director commit 99e9bfb) <<<

The locked sheets moved every heroine shot; all are re-keyed to them. REPORT 1 (01:15Z) follows below for history.

**H1 on the MASTER TIMING, all cuts (must-have).** `beacon.use_master_timing()` rebinds the take's timing constants:
strike 2 +29, strike 3 +58, the long blow +84..+180 (three breaths), the catch +196, the roar +240 (on a downbeat); the
accepted post-catch poses are stretched 42 -> 44 frames to the roar and shifted +116 after it; four frame literals
became constants with identical v2b values, so `--shot beacon` is unchanged. `render.py --shot beacon_v3` = this timing
+ the B7 re-key; src 1200-1555, strike 1 = src 1236 -> `renders/h1_v3`, JPEGs to branch **`claude/render-h1-v3-master`**
(`cloud/jobs/h1_v3_master.json`): **DONE 05:09-05:50Z on this box, 356/356 JPEGs pushed, every frame decodes**; contact
sheet `review/v3/heroine_v3_h1_master_contact.jpg` (strikes, the long blow, the catch, the roar, the pull-back). Verified at half res: 1265 (strike 2), 1334/1400
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
* `cloud/jobs/h1_v3_master.json` (beacon_v3): DONE (branch `claude/render-h1-v3-master`).
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
