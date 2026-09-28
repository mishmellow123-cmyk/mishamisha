# A-FIX: making film A's story clear (lane A-FIX, 28 Sep ~00:55Z-02:30Z) - CLOSED

## >>> STATE AT CLOSE (A-FIX, 28 Sep ~02:30Z). The lane is finished; the open items below have owners. <<<
Brief: the director's A-FIX message (scope 1-6) + `_local_logs/review/A_CLARITY.md`. Captions were EDIT-3's.
Stills and sheets: `~/mishamisha/_local_logs/review/a_fix/` (`sheet.py` is the contact-sheet helper).

**Done and rendered, or wired by EDIT-3:**
1. **FIRE-LIGHTING (A12).** `renders/h1_A` = 260 f, src 1236-1495, checked. It comes from beacon.py's A-only AFIX
   flag (shot `beacon_v3_afix`, job `h1_a_afix`; defaults unchanged):
   - pinpoint strikes, dark leather gloves and wool, a jumbled log pile (no "A");
   - a 20 f lean-in, the face a pure silhouette, the scarf dark until the flame, no breath after the catch;
   - the roar ramped over 10 f and a 12 f lean-back, with no pull-back or whip;
   - strike 1's spark lands on A11's ember at (958, 547): the match cut.
   Why not the 2x crop: it showed her lit face in profile and gourd sleeves (`h1_crops.jpg`).
2. **THE HINGE (A13).**
   - ROAR: h1_A, 3600-3612.
   - HER FIRE: reveal_B 1372-1419 at 3612-3660, EDIT crop (0.344, 0.20, 0.64, 0.64) past B's cairn, grade-matched.
   - EVERY RIDGE: `renders/reveal_A` 3660-3799, re-rendered and checked (continuity: max frame diff ~2/255). The
     far fires are 3x and answer across the ranges at 3684-3740; there are no searchlight rays; her smoke is a pale
     veil; 3660-3679 is H1's own pull-back tail.
3. **TRANSITIONS + VISIONS + E4.** `edit/afix_comp.py` `A_TRANS`, wired by EDIT-3 (16f0376):
   - bloom 536-560;
   - dissolve 596-672;
   - the PROMISE vision 1280-1440: it opens out of the flame, the flame keyed in front, a thin flickering rim with
     flecks;
   - iceheart 2520-2640;
   - the DEAD VALLEY vision 2640-2800: opens from a flame-shaped ghost at 2656, sits above T9, closes to a point by
     2796;
   - ember 2836-3359.
4. **SOUND SYNC.** edge.py: the white lands ON 2640 (flash 2637-2639). SOUND-C has the frames.
5. **ENDING (A19-A20).** `renders/dawnrev_A` = 640 symlinks, dusk_B2 reversed. `afix_comp.WATCHFIRES` (4bed23a):
   - four watch-fires pale ON COMPOSER-A2's horn frames 6240/6290/6340/6390;
   - the lantern and four tiny seated figures at (1745, 640), clear of T14.
6. **THE RACE + THE EDGE.** Finals launched by main (log `_local_logs/jobs/race_edge_afix_farm.log`): a5 1420-1439,
   a6 + alt_a 1440-1839, edge + alt_edge 1840-2639, at 66eb629. What they carry:
   - `race_afix.ThroatFire`: every forge's throat roars and throws sparks on the beat (t % 20), the giants hardest,
     none on the ALT's coded giants;
   - the a3 CAM_A5 race camera re-keyed, so the crowns and the giants' growth are in frame;
   - edge.py: a creeping push and zoom, `SCHED.edge_joint_k` (forges at 35% at the edge), GILD_W[1] 0.3, GoldRuns
     x2.4 down the grazing sides.

**OPEN ITEMS (owners set by the director):**
- **GILDING INSERT (A7, under T7): EMBERS-C4**, after its C shots. The spec is below.
- **DESERT foot slide + KARST/DESERT soft catch: MONTAGE-3D-5**, after its C shots. The spec is below.
- **Verify the race/edge finals when they land** (whoever reviews A next): see HOW TO VERIFY.

**HOW TO VERIFY (on the A master, or on stills via `a_fix/sheet.py`):**
- 536-560: the push into the glow flooding to white (no one-frame slam). 596-672: an even fall to the letters (no iris).
- 1280-1440: the promise inside the fire's window. The flame stands in front, the rim flickers, it closes into the flame.
- 1440-1760: every forge's crown roars and throws sparks ON each beat. The two giants' crowns roar highest and reach
  the top edge by 1759. Also check 1439 -> 1440 for a pop, now that a5's 1420-1439 are re-rendered.
- 1840-2400: THE EDGE visibly creeps in (compare 1950, 2100, 2300); the forges are dark charcoal; four gilded.
- 2600-2640: the plunge accelerates; the flash is 2637-2639; full white exactly on 2640 (the IMPACT).
- 2656-2800: the dead valley opens from a flame shape, clear of T9 (2700-2740), and closes to a point by 2796.
- 2836-3360: the ember is a readable coal; at 3359/3360 it lands on strike 1's spark.
- 3360-3612: no halo, no "A", no lurch, no lit face, no floating breath; the roar grows over 10 f; she leans back.
- 3612-3660: her still silhouette by her fire, no cairn. 3660-3800: the ranges answer with fires; no searchlight.
- 5840-6480: the reversed dawn (no sun disc); the fires pale on 6240/6290/6340/6390; the lantern group is clear of
  T14 (6080-6200); the title sits in the rose sky.
- People-movement gate: h1_A 1300-1324 and 1476-1495 (filmstrips `afix_lean.jpg`, `afix_t4_her.jpg`) and reveal_B
  (`revealB_strip.jpg`) all passed.

## SPEC for EMBERS-C4: THE GILDING INSERT (A7 THE EDGE, under T7; director approved ~02:25Z)
- **Why.** T7 says *"Every step closer made them richer."* The picture must show gold RUNNING down a forge on each
  surge while the others stay dark. CLARITY-A's V4 asks for molten gold visibly running down the nearest two on the
  surges under T7 (1848-1944).
- **What I tried, and why THE EDGE's camera can't show it.**
  - `_cam_edge` sits outside the ring (r 27, then 22 with the A-FIX push) and looks inward across the crater through
    the widest gap. The gilded forges' FIRE-FACING faces point inward, at the fire, so this camera sees their backs
    (black, by the fire-side-only rule) or their sides edge-on.
  - `GoldRuns.emit` draws gold only on points that face the fire (`fs > GOLD_FS`) AND face the camera (`ndv > 0.03`).
    From outside the ring, only a thin grazing strip qualifies.
  - I darkened every forge to 35% at the edge (`SCHED.edge_joint_k`), ungilded tower 1 (GILD_W[1] 0.3), raised the
    runs x2.4 and lowered GOLD_FS 0.2 -> 0.08. The contrast is better, but the runs still read as speckle, not
    pours: `_local_logs/review/a_fix/edge_t4_gold.jpg` (1886, 1906, 1926 vs OLD 1906).
- **The insert.**
  - Slot: a hard cut IN on the beat at 1860 and OUT on the beat at 1920 (60 f, bar 24 b2 -> bar 25 b1). The
    director's 1864-1924 also works, but cutting on the beats sits better in the taiko.
  - Two surges land inside it (1880, 1900), so gold pours twice.
  - It is part of the same render: a camera branch in `edge.camera` for 1860 <= t < 1920, e.g. `_cam_gild(tl, t)`,
    in both MAIN and ALT.
  - Subject: **tower 7 (FALL_TOWER)**, a gilded forge that is never a giant. Its crown breaks at 2460-2519, so the
    insert sets that up: this is the one whose gold we watched.
  - Camera: INSIDE the ring, between the fire and tower 7's fire-facing face, the fire behind the lens.
    - pos ~ `tw.base(7)` + (the unit vector from the tower to the fire) x 9-11 units, at y ~ 4-8;
    - a medium lens (hf ~32-38) framing ~14-18 units of the face, from its roaring throat (ThroatFire, top of frame)
      down to the upper wall;
    - look slightly up; a slow push-in (~1 unit over 60 f);
    - so fs ~ 1 and ndv ~ 1 on the face: GoldRuns' rivulets, trails and white-hot descending heads resolve.
  - Keep: the lower third darker for T7's caption (the face's lower part in soot); the forge's charcoal crust
    between the runs (dark gaps, no glitter); the gold ring sound on each gilding (the music already has it).
  - The ALT: identical (tower 7 is not a coded giant).
- **Verify.**
  - `farm.py cloud/jobs/embers_A3_edge.json --test 5 --frames 1862,1878,1884,1898,1904`: the gold should read as
    runs pouring down, with bright heads, on the surges at 1880 and 1900.
  - Check that no giant is in frame and the cut points sit on the beats.
  - Then JOB READY `--frames 1860-1919` for MAIN and ALT.

## SPEC for MONTAGE-3D-5: DESERT + KARST re-renders (director assigned, 28 Sep ~02:00Z)
- **DESERT foot slide** (A 3881-3885 = desert src 1541-1545): `shots/montage3d/desert.py` `_figure_frames()` sets
  `back = 0.42 * FG.ease((f - 1540.5) / 5.0)` and `step[str(f)] = -back`, which translates the whole figure 0.42 m in
  5 frames with no stepping legs. That's a slide, and it reads as a jump. Evidence: `_local_logs/review/a_fix/desert_feet2.jpg`.
  Fix:
  - `back = 0.0`. She stays planted after she lets the torch go (it stays in the basket, since the grip term uses `back`).
  - The flinch arm `sh = FG.ease((f - 1540.0) / 3.0) ...` raises her forearm in 3 frames. Make it `(f - 1540.0) / 10.0`,
    so it rises over about 10 frames, and lower it slowly (`(f - 1552.0) / 14.0` becomes `(f - 1556.0) / 24.0`).
  - Check FIG_KEYS 1539.5 -> 1544 (lean 0.3 -> -0.12 in 4.5 f): stretch the 1544 key to about 1550 so the body eases back.
  - People-movement gate: a filmstrip of every frame 1536-1560 at 1:1 on the figure.
- **Soft catch, DESERT and KARST**: `shots/montage/mt/fire.py ignite_env()` goes from nothing to size 0.5 and light
  2.2x in 1.5 frames (the one-frame switch-on at A 3815 and A 3880). Add an `ignite_env_soft(t, t0)` and use it in
  `desert.py timing()` and karst.py's equivalent:
  - size: 0.08 -> 1 over about 8 frames with a small overshoot (1.15 at +10 f);
  - light: 0.15 -> 1.2 over about 6 frames, then settling to 1.0;
  - a spark or two at 0.
  Other users of ignite_env stay unchanged.
- Frames: desert src 1520-1579 -> `renders/montage3d_v3/desert` (A 3860-3919); karst_slow 0-59 (A 3800-3859).
  Farm GPU (Cycles), about 60 f each.

## Log
- 02:30Z CLOSED (director). Gilding insert -> EMBERS-C4; desert/karst -> MONTAGE-3D-5; race/edge finals rendering.
- 00:55Z read the kit. 01:05Z crop test (fails). 01:15Z AFIX code (7598e6b), t1. 01:25Z t2 (e768eb7). 01:30Z comps.
  01:32Z t3 (e436c0a: match cut, slimmer roar). 01:38Z t4 (2c01912: the lean-back).
