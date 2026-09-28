# A-FIX: making film A's story clear (lane A-FIX, from 28 Sep ~00:55Z)

## >>> STATE (A-FIX, 28 Sep ~02:20Z) <<<
- Brief: the director's A-FIX message (scope 1-6) + `_local_logs/review/A_CLARITY.md`. Captions are EDIT-3's.
  Stills and sheets: `~/mishamisha/_local_logs/review/a_fix/` (`sheet.py` is the contact-sheet helper).
- **1 FIRE-LIGHTING: DONE.** `h1_A` (260 f, src 1236-1495) is rendered and checked (main launched it). beacon.py AFIX gives:
  - pinpoint strikes, the dark leather and wool, the jumbled logs, the 20 f lean-in;
  - the face a pure silhouette and the scarf dark until the flame;
  - no breath after the catch;
  - the roar ramped over 10 f and a 12 f lean-back;
  - no pull-back, and the match cut: strike 1's spark on A11's ember at (958, 547).
  EDIT-3 has wired it (16f0376): A12 and A13 ROAR 3600-3612 play h1_A.
- **HINGE: DONE / rendering.**
  - HER FIRE = reveal_B 1372-1419 at A 3612-3660, EDIT crop (0.344, 0.20, 0.64, 0.64) to lose B's cairn, grade-matched.
  - EVERY RIDGE = reveal_A 3660-3799, re-rendering (main launched reveal_a_1 v2): far fires 3x, no searchlight
    rays, pale thin smoke, starting at 3660.
- **2 TRANSITIONS + 3 VISIONS + E4: DONE, wired by EDIT-3.** `edit/afix_comp.py` `A_TRANS`:
  - bloom 536-560;
  - dissolve 596-672;
  - the promise vision 1280-1440;
  - iceheart 2520-2640;
  - the dead-valley vision 2640-2800 (it opens from a flame-shaped ghost and sits above T9);
  - ember 2836-3359.
  The rim is thin, uneven and flickering, with flecks.
- **SOUND sync: DONE.** edge.py: the white lands ON 2640 (flash 2637-2639). Main launched edge 2520-2639 MAIN + ALT.
- **5 ENDING: DONE, EDIT-3 wiring.**
  - renders/dawnrev_A: 640 symlinks to dusk_B2, reversed.
  - `afix_comp.WATCHFIRES`: four watch-fires pale ON the composer's horn frames 6240/6290/6340/6390, and the
    lantern with four tiny seated figures sits at (1745, 640), clear of T14.
- **6 DESERT/KARST:** spec below. The director assigned it to MONTAGE-3D-5.
- **4 RACE: JOB READY sent (~02:20Z, at 66eb629).** The jobs:
  - embers_A3_a5 --frames 1420-1439 (the spline's tangent);
  - embers_A3_a6 1440-1839 + embers_A3_alt_a;
  - embers_A3_edge 1840-2639 + embers_A3_alt_edge.
  What changed:
  - `race_afix.ThroatFire` (a3.TimelineA3.emit, 1470-2639): every forge's throat roars forge-orange and throws
    sparks on the beat grid (t % 20); the giants hardest; none on the ALT's coded giants.
  - a3 CAM_A5 1540-1759 re-keyed: a medium-low frame, the crowns in view, the giants reaching the top edge by 1759.
  - edge.py: a creeping push (EDGE_PUSH 5) and zoom (EDGE_ZOOM 10) over THE EDGE; `SCHED.edge_joint_k` darkens
    every forge's joint fire and crust sheen to 35% at the edge; GILD_W[1] 0.85 -> 0.3; GoldRuns x2.4 down the
    grazing sides (GOLD_FS 0.08).
  - Sheets: `race_t3.jpg`, `edge_t3.jpg`, `edge_t4_gold.jpg`.
  - OPEN (the director's call): the gilding still doesn't read as runs, because the edge camera sees the gilded faces
    edge-on. Proposed fix: a 60 f insert under T7 (~1864-1924), close on a gilded forge's fire-facing face as the gold
    pours.

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
- 00:55Z read the kit. 01:05Z crop test (fails). 01:15Z AFIX code (7598e6b), t1. 01:25Z t2 (e768eb7). 01:30Z comps.
  01:32Z t3 (e436c0a: match cut, slimmer roar). 01:38Z t4 (2c01912: the lean-back).
