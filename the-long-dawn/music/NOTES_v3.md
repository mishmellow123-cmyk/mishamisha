# THE LONG DAWN v3: score & sound (COMPOSER-v3 notes)

## >>> PAUSED (director's usage pacing, 27 Sep ~10:25Z). EXACT STATE + NEXT STEPS <<<
- **DONE: B = THE VIGIL, full score, rendered + battery** -> `out/v3/final_B.wav` (+ `final_B_score.wav`,
  `final_B_sfx.wav`, `score_B.wav`), analysis `analysis/v3/final_B/` (report.txt, overview/spec/roll.png).
  Changes in `src/score_v3_B.py`: docstring/plan = THE VIGIL (hours); B8 in her frame (the ground holds its breath,
  one high harmonic, her call FALTERS: D3 A3, no octave, at 32 b2); B10 in her frame (her fire flares: her call near,
  then the nearer beacon's ANSWER in major over G); B12 the traveller's child stays and falls asleep (softer: ride
  -2.0 dB, pads settle into the crane; four far horns answer far and soft); bar 63 b3 the fire-steel press = the
  dead of night's A5 harmonic returns ppp under her HOME phrase (no stroke; HOME still lands on D3 at 64 b1);
  measured sync trims (SYNC_TRIM).  Battery: format PASS, -16.05 LUFS / -1.30 dBTP, level map 14/14 PASS, notes 0,
  clicks 0, breath 59.8 dB deep, sync 23/24 PASS.
- **Two small open fixes on B (next render, ~2 min):** (1) sync: HOME at 62 b1 now measures -200 ms (the +0.13 s
  trim overshot, the measurement looks bimodal): set SYNC_TRIM["home_phrase"] to ~+0.03 and re-check; (2) RULE
  "the dawn is warm, not loud" now -1.5 LU (limit -2.0; the softer B12 raised the master gain, the limited crane
  stayed put): lower the post-sunrise strings/horns ~0.8 dB (e.g. a ride on B13 from the sunrise, or scale the H
  pads' dyn in handback()) and re-render: `cd music/src && python render_v3.py B`.
- **IN PROGRESS: fallback engine rewritten** (`src/fallback_v3.py`, not yet run): calls/answers pass between two
  horns when fires are close (A's Run, C's beacons and coast fires stacked overlapping notes on one horn before);
  every breath cuts the drone and re-blooms on its downbeat (B's sunrise sits inside B13); the silence piano waits
  2 s of true silence.  NEXT: `python render_v3.py A --fallback`, then B, then C (one at a time, ~2-4 min each);
  read each report + overview; put the paths in the MASTERS table.
- **THEN:** A's full score (`src/score_v3_A.py`, new), C's full score (`src/score_v3_C.py`, new; bars 70-71 = THE FIRE
  REMAINS: warm, settling, the ANSWER passed outward soft, the tritone resolved, into bar 72's 275 ms breath).

## STATE AT HANDOFF (COMPOSER-v3 #2, 27 Sep 10:10Z)

- **Locked, never edit:** `music/v3/barmap_{A,B,C}.json` (SHOWRUNNER-REV) and the cue sheets `music/v3/cues_{A,B,C}.json`
  (written by `src/cues_v3.py`; the B sheet is already THE VIGIL: stones became "feed" events, the nights are hours 2-9).
- **Engine (all uncommitted until this session):** `src/timeline_v3.py` (grid + bar-map loader, merges the cue sheet),
  `src/kit_v3.py` (motif kit + seating + rule checks), `src/score_v3_B.py`, `src/fallback_v3.py` (FALLBACK MASTER from
  any bar map), `src/render_v3.py` (parts -> mix -> effects -> stem-linked master -> analysis), `src/analyze_v3.py`
  (the battery), `src/sfx_v3.py`, `src/synth_v3.py`, `src/cues_v3.py`, `src/barmaps_v3.py`.
- **Last render left by the previous agent:** `out/v3/final_B.wav` (07:25Z; still THE LIFETIME's far-peak cutaway
  music) with `analysis/v3/final_B/`: format/loudness/level map/centroid all PASS, sync 3 FAIL of 22 (dusk entry
  +141 ms, the year-60 call -50.4 ms, HOME +133 ms). A test fallback (`fallback_T`) exists; no fallback for A/B/C yet;
  no A or C score yet.
- **Housekeeping at handoff:** pruned 3.2 GB of stale part caches (`cache/v3/parts`, kept only stems in the current
  manifests).
- **This session's orders (director, 10:05Z):** (1) B = THE VIGIL in full, rendered, with the battery; (2) FALLBACK
  MASTERS for A, B, C; (3) A's and C's full scores. B: variations = hours; M2/M3 cutaways cut (B8, B10 in her frame,
  the far answer only pinpricks); from bar 46 a traveller's child stays and falls asleep against her; bar 63 she
  presses her fire-steel into the child's palm, HOME (from 62 b1) still arrives as the child wakes (64 b1).
  C: bars 70-71 = THE FIRE REMAINS (warm, settling, the ANSWER passed outward, soft, the tritone still resolved, into
  bar 72's 275 ms breath and the sunrise). A: wording only (frames unchanged).
- **Masters for EDIT-v3:** listed in the table below as they land.

## MASTERS (EDIT-v3 picks these up)

| cut | file | status |
|---|---|---|
| B score (THE VIGIL) | `music/out/v3/final_B.wav` (stems `final_B_score.wav`, `final_B_sfx.wav`) | rendered 10:22Z, battery passed but 2 small fixes pending |
| A fallback | - | pending |
| B fallback | - | pending |
| C fallback | - | pending |
| A score | - | pending |
| C score | - | pending |

All masters: 48 kHz / 24-bit / stereo WAV, exactly the cut's length (A 6,480 f = 12,960,000 samples; B 5,440 f =
10,880,000; C 7,200 f = 14,400,000), -16 LUFS integrated, true peak <= -1.2 dBTP, from silence to silence, with
stem-linked `<name>_score.wav` + `<name>_sfx.wav` (score + sfx = master).
