# THE LONG DAWN v3: score & sound (COMPOSER-v3 notes)

## MASTERS: THE ONE TABLE (EDIT wires these into the animatics; both composers update only their own rows)

| cut | file | status | owner |
|---|---|---|---|
| B score (THE VIGIL) | `music/out/v3/final_B.wav` (stems `final_B_score.wav`, `final_B_sfx.wav`) | **FINAL 10:50Z**: battery all PASS (sync 25/25, level map 14/14, dawn -2.3 LU under the crane, notes 0, clicks 0) | - |
| A fallback | `music/out/v3/fallback_A.wav` (stems `fallback_A_score.wav`, `fallback_A_sfx.wav`) | **DONE 19:10Z** (re-rendered on the current engine so the file matches the committed source): battery all PASS (level map 20/20, sync 8/8, notes 0, clicks 0) | COMPOSER-A |
| B fallback | `music/out/v3/fallback_B.wav` (stems `fallback_B_score.wav`, `fallback_B_sfx.wav`) | **DONE 11:45Z**: battery all PASS (level map 14/14, sync 5/5, notes 0, clicks 0) | - |
| C fallback | `music/out/v3/fallback_C.wav` (stems `fallback_C_score.wav`, `fallback_C_sfx.wav`) | **DONE 19:10Z**: battery all PASS (level map 28/28, sync 7/7, notes 0, clicks 0). Fix: the ride (C2 bar-2 horn -6.5 dB, C12 piano -4.5 dB) + C11's race drone now falls away after the slip into C12's black (its tail was C12's peak, -6.4 LU) | COMPOSER-A |
| A score | `music/out/v3/final_A.wav` (stems `final_A_score.wav`, `final_A_sfx.wav`) | pending: `src/score_v3_A.py` being written | COMPOSER-A |
| C score | `music/out/v3/final_C.wav` (stems `final_C_score.wav`, `final_C_sfx.wav`) | pending: `src/score_v3_C.py` being written | COMPOSER-C |

All masters: 48 kHz / 24-bit / stereo WAV, exactly the cut's length (A 6,480 f = 12,960,000 samples; B 5,440 f =
10,880,000; C 7,200 f = 14,400,000), -16 LUFS integrated, true peak <= -1.2 dBTP, from silence to silence, with
stem-linked `<name>_score.wav` + `<name>_sfx.wav` (score + sfx = master).

## TWO COMPOSERS FROM 18:40Z: who owns what
- **Shared, additive only:** `src/kit_v3.py` (the motif kit), `src/timeline_v3.py`, `src/render_v3.py`,
  `src/analyze_v3.py`, `src/sfx_v3.py`, `src/synth_v3.py`. Add new functions; never change the behaviour of an
  existing one without telling the other composer (B's FINAL score and the three fallbacks depend on them).
  Cut-specific material lives in each cut's own score file.
- **Locked, never edit:** `music/v3/barmap_{A,B,C}.json`, `music/v3/cues_{A,B,C}.json`.
- **COMPOSER-A owns:** `src/score_v3_A.py`, `src/fallback_v3.py` (the fallback engine), the A/C fallback rows.
- **COMPOSER-C owns:** `src/score_v3_C.py`, the C score row.
- **Renders:** one each at a time (the Mac is at ~4.4 of 5 GB swap); outputs never collide (`final_A` vs `final_C`).
- **Git:** commit with explicit paths only (`git commit -m ... -- <your paths>`).

## COMPOSER-A (cut A) · STATE 19:15Z
- Took over from COMPOSER-v3 at 18:35Z. B FINAL and the B fallback are untouched.
- **1. C fallback: DONE** (table). The ride alone fixed C2 but not C12: C12's loudest 3 s was C11's race drone
  swelling into the black and its 3 s hall tail. `fallback_v3.py` now lets a race section that is cut to silence
  or black (not the brink) fall away over its last two beats, as C's sheet asks at the slip ("one glass tone falls
  away into silence"). Only C11 matches (checked: A's and B's part keys unchanged by it). A's fallback was also
  re-rendered, because the previous agent's last engine change (the horn ride on note gain) had not been rendered
  for A: it still passes, and the file now matches the committed engine.
- **2. A's full score:** `src/score_v3_A.py` written (the plan is in its docstring, bar by bar). Next: render
  (`python render_v3.py A`), battery, ride, iterate.

## COMPOSER-C (cut C) · STATE
- **18:55Z: writing** `src/score_v3_C.py` (the score) + `src/voices_v3_C.py` (C's own synth voices and effect
  designs: the pen, paper burn, cold-metal tick, the melt's seethe, the cock crow, the drop, the torches' dip, the
  harp harmonics, the anvil, the tam-tam swell, the falling glass tone). Both owned by COMPOSER-C. They REGISTER
  into `synth_v3.VOICES` / `sfx_v3.DESIGNS` at import time, so no shared file changes (no one's caches bust).
- The cue sheet stays locked: the score adds C's extra effects (the beat sheet's C column: pen, burn-throughs,
  cold-metal tick, seethe, cock, drop, the fire that remains) **in memory** in `build()`, on top of `cues_C.json`.
- H5: bars 70-71 = THE FIRE REMAINS (warm, settling, the ANSWER passed outward, the tritone resolved, into the
  275 ms breath and the sunrise); C10 THE MIRROR scored (one high held tone, harp harmonics, a drop).
- Renders: one at a time, never while COMPOSER-A's render runs (checked with `ps`).

---
## HISTORY (COMPOSER-v3, before 18:35Z)

## >>> PAUSED (director: usage window end, 27 Sep ~12:05Z; RESUME 20:00Z). EXACT STATE + NEXT STEPS <<<
- **1. B score: FINAL.** `out/v3/final_B.wav` (+ stems). Battery all PASS: sync 25/25 (HOME at 62 b1 is measured per
  layer: the doubled B3 beats in the narrow band; each layer's anticipation is set in ANTIC_SET), level map 14/14,
  the dawn -2.3 LU under the crane, first half -6.4 LU, notes 0, clicks 0.  Dawn fix = the first paling horn and
  her late CALL a little softer, the bloom's swell smaller.
- **2. Fallback masters** (`src/fallback_v3.py`, generic from bar map + cue sheet; per-cut conductor's ride
  FALLBACK_RIDE from each battery; horns take the ride on note gain so their sample layer/attack stays measured):
  * **A: DONE** `out/v3/fallback_A.wav`: battery all PASS (20/20 sections vs the cue sheet's bands, sync 8/8).
    Rendered before the last engine change (horn ride on gain, anticipation after the ride): an optional
    re-render for consistency, `python render_v3.py A --fallback`, then re-check (should still pass).
  * **B: DONE** `out/v3/fallback_B.wav`: battery all PASS (14/14, sync 5/5; the crane is the loudest, the night
    calls one bar each with the bar of silence after, the dawn blooms after the breath).
  * **C: rendered once** `out/v3/fallback_C.wav`: sync 7/7, notes 0, but 2 sections out of band (C2 the book's
    horn CALL -2.9 LU, limit -8; C12 the silence piano -6.9, limit -10). A ride for C is now in FALLBACK_RIDE
    (untested). **NEXT: `cd music/src && python render_v3.py C --fallback`** (~10 min: the effects stem is now
    cached, so ~4 min), check the table, adjust the ride once more if needed, update MASTERS.
- **3. THEN A's and C's full scores** (`src/score_v3_A.py`, `src/score_v3_C.py`, both new). Notes for A: the v1/v2
  race and kindling (score.py) are NOT reusable as is (strokes on the ignition, 32nd rolls, cymbals); write fresh
  in the v3 rules on the kit. C: bars 70-71 = THE FIRE REMAINS (warm, settling, the ANSWER passed outward, soft,
  the tritone resolved, into bar 72's 275 ms breath).
- Memory is very tight (swap 12 of 13 GB used by the whole machine at 12:00Z): one render at a time.

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

