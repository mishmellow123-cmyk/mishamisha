# THE LONG DAWN v3: score & sound (COMPOSER-v3 notes)

## MASTERS: THE ONE TABLE (EDIT wires these into the animatics; both composers update only their own rows)

| cut | file | status | owner |
|---|---|---|---|
| B score (THE VIGIL) | `music/out/v3/final_B.wav` (stems `final_B_score.wav`, `final_B_sfx.wav`) | **FINAL 10:50Z**: battery all PASS (sync 25/25, level map 14/14, dawn -2.3 LU under the crane, notes 0, clicks 0) | - |
| A fallback | `music/out/v3/fallback_A.wav` (stems `fallback_A_score.wav`, `fallback_A_sfx.wav`) | **DONE 19:10Z** (re-rendered on the current engine so the file matches the committed source): battery all PASS (level map 20/20, sync 8/8, notes 0, clicks 0) | COMPOSER-A |
| B fallback | `music/out/v3/fallback_B.wav` (stems `fallback_B_score.wav`, `fallback_B_sfx.wav`) | **DONE 11:45Z**: battery all PASS (level map 14/14, sync 5/5, notes 0, clicks 0) | - |
| C fallback | `music/out/v3/fallback_C.wav` (stems `fallback_C_score.wav`, `fallback_C_sfx.wav`) | **DONE 19:10Z**: battery all PASS (level map 28/28, sync 7/7, notes 0, clicks 0). Fix: the ride (C2 bar-2 horn -6.5 dB, C12 piano -4.5 dB) + C11's race drone now falls away after the slip into C12's black (its tail was C12's peak, -6.4 LU) | COMPOSER-A |
| A score | `music/out/v3/final_A.wav` (stems `final_A_score.wav`, `final_A_sfx.wav`) | render 4 (20:40Z) battery ALL PASS (level map 20/20, rules 3/3, centroid 5/5, sync 55/55); a balance pass (render 5) is rendering: use the file once this row says FINAL | COMPOSER-A |
| C score | `music/out/v3/final_C.wav` (stems `final_C_score.wav`, `final_C_sfx.wav`) | in progress (the current file is usable temp): render 3 20:21Z = level map 27/28, rules 3/3, sync 47/50; render 4 queued 20:30Z | COMPOSER-C |

All masters: 48 kHz / 24-bit / stereo WAV, exactly the cut's length (A 6,480 f = 12,960,000 samples; B 5,440 f =
10,880,000; C 7,200 f = 14,400,000), -16 LUFS integrated, true peak <= -1.2 dBTP, from silence to silence, with
stem-linked `<name>_score.wav` + `<name>_sfx.wav` (score + sfx = master).

## SOUND (real effects lane, from 19:00Z): THE INTERFACE (COMPOSER-A, COMPOSER-C, EDIT: please read)
- **What:** SOUND is replacing the synthesized effects with REAL recordings: Freesound (CC0 / CC-BY), the Sonniss GDC
  2026 bundle, and ElevenLabs only for the gaps. Every cue of the locked cue sheets keeps its id, time, fades and envelope,
  and **its level is matched to your synthesized design of the same cue** (beds by integrated K-loudness, events by their
  loudest 400 ms), then trimmed by ear-proxies (analysis). Credits: `music/SFX_CREDITS.md`. Source: `src/sound_v3.py`
  (engine), `src/sound_recipes_<cut>.py` (what plays), `src/ir_v3.py` (MEASURED impulse responses).
- **Files per cut (48 kHz, exact length):**
  - `out/v3/sfxpre_<cut>.wav` is the real effects in the PRE-master domain, as float32 (what `render_v3.master()` takes).
  - `out/v3/sound_<cut>.wav` is the master, `sound_<cut>_score.wav` + `sound_<cut>_sfx.wav`. It is made by YOUR `render_v3.master()`,
    run on the saved `cache/v3/premaster_score_final_<cut>.npy` plus the real effects, so it has the same -16 LUFS / <= -1.2
    dBTP chain, and the battery (`analysis/v3/sound_<cut>/`) is run on it.
  - `out/v3/sfx_<cut>.wav` is a copy of `sound_<cut>_sfx.wav`, the deliverable effects stem.
- **For COMPOSER-A / COMPOSER-C:** nothing changes in your renders unless you opt in. When your final is locked, tell me
  (or just re-run `python sound_v3.py <cut>`, about 2 minutes): it re-masters your newest pre-master score with the real
  effects and runs your battery. The cue sheets stay locked. Where the picture changed (B is THE VIGIL), the
  recipe says why, e.g. `B.his_fire` is silent (the far answer is a pinprick) and `B.x.flare` is added (her flare, 40 b2).
- **Reverb (director, 20:00Z):** the synthesized hall in `mix.make_ir` is a machine tell. A MEASURED stone church is ready:
  `ir_v3.irs("church")` (Freesound CC0, balloon IR, T30 about 2.2 s mid, highs shorter), in the same 4-IR, energy-normalised form
  as `mix.make_ir`. Voxengo's free set is NOT measured (it was made with Impulse Modeler), so it is not used. The proposed
  opt-in is `LONGDAWN_HALL=church` in `render_v3.hall_ir()`. It re-mixes the score only (the parts cache is dry, so it is not busted),
  and then the battery must be re-run. **Not wired yet: SOUND asks COMPOSER-A/C to agree here first.**
- **Status:** B first (all cues, real), then the flint take shared by A/B/C, C's book and council, then A's fire act.

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
- **2. A's full score:** `src/score_v3_A.py` (the plan is in its docstring, bar by bar).
  * Render 1 (19:36Z): format/breaths/notes/clicks PASS; level map 6 FAIL, sync 17/54 FAIL, rule "brink loudest"
    FAIL. Causes: the master crushed the brink by ~13 dB (transient peaks: taiko, the crumble piano, the drum roll),
    so the edge tied it; the glass too hot in bars 12-13; the score's own impact rang into bar 36's true silence.
    Sync: bowed D3 fundamentals arrive 0.15-0.4 s after the kit's broadband attack estimate; legato octave drops
    ~0.55 s; entries that crescendo from nothing through a velocity-layer boundary ~0.25 s.
  * Render 2 (19:55Z): level map 18/20 (the edge -0.4 LU under the brink; A10's piano -7.1 vs -10), sync 50/54
    (the watch tone +60 ms; the low pizz measured in its band caught the previous note's leakage). The master
    still took 6-7 dB off every taiko hit in the edge (pumping) and 8-11 dB off the brink.
  * Render 3 (20:15Z): level map 20/20, sync 55/55, centroid 5/5; one rule FAIL: bars 28-29 (four spiccato
    sections in sixteenths + five brass) came within 0.8 LU of the brink. Checked: bar 36's two seconds are true
    silence (score -111 dBFS; only the effects' ash bed's 0.3 s fade tail).
  * Render 4 (20:40Z): **battery ALL PASS** (level map 20/20, rules 3/3, centroid 5/5, sync 55/55, notes 0, clicks 0,
    -16.04 LUFS, TP -1.29 dBTP). Fix: sparse beats dense in the sixteenths (only violas + second violins double;
    cellos and first violins keep eighths), softer brass call. Kept as `out/v3/_keep_A_r4/` (+ its analysis in
    `analysis/v3/_keep_A_r4/`).
  * Render 5 (20:55Z): the balance worked (per-part loudness: every story line now leads its window: the
    promise's line level with the bloom's top, the violas' CALL 1.8 dB over its pad, the first ANSWER leading the
    Run, the watchers' corrupted call over the hold, the dawn's ANSWER and HOME on top). But the master gain rose
    0.7 dB with the brink pinned at the limiter, so A2, A3 and A10 crept 0.2-0.5 LU over their bands, and run fire
    1's CALL arrived +56 ms. Lesson: the level map is relative to a limited anchor, so every quieter pass lifts all
    the untouched sections; keep ~1 LU of margin.
  * Render 6 (21:10Z, running): `RIDE` A2 -1.5, A3 -1.5, A4 -2.8, A7 -2.4, A10 -5, A16 -1, A17 -0.8; trim run fire 1.
  * Tool: `scratchpad/who.py <cut> <name> t0 t1 ...` ranks the parts by loudness in a window (from the part cache).

## COMPOSER-C (cut C) · STATE
- **20:25Z: render 3 battery**: level map 27/28 (C13 0.2 LU over: the glass streak meets the hiss in the snow),
  rules 3/3, sync 47/50 (three misses of 50-90 ms, per-layer trims set), notes 0, clicks 0; master GR 11 dB at
  the slit (down from 13), 6 at a race surge. Effects now own their moments (the roar +2 LU over the music, the
  dips, the seethe, the burns within 5 LU); still weak: the blaze (the design was sub-bass: redesigned mid-range,
  calibrated to ~4 LU under C21's orchestra at its peak), the Havens' sea (a nearer layer added), the cold ticks.
  Lines over the foundation (per-part probe): basses lighter, solo violin +6 dB, the dawn's line forward.
  **Render 4 queued 20:30Z** behind A: those + no low tremolo at the slit (the arco carries it), the tam-tam's
  bloom later in its wash, the Run's timpani/basses under the horn calls, C26 -3.
- **20:05Z: render 2 battery**: level map **28/28 PASS**, all 3 rules PASS (everything after the Eye >= 1.8 LU
  under the slit; the dawn -3.1 LU; the prologue -9.4 LU), sfx -25.7 LUFS vs score -16.0 (B-like), notes 0, clicks
  0; sync 45/50 (the misses: two crescendo blooms probed as arrivals, and entries the attack table over-reads:
  per-layer the quiet sections arrive ~0.52 s after starting, the violins' D5 0.16 s, the solo violin 0.44 s).
  The master pulled 13 dB at the slit (the cello tremolo's 15 dB crest) and 7.5 dB at a race surge (taiko +
  timpani). **Render 3 (queued 20:10Z behind A)**: the slit's weight on sustained arco cellos and basses (tremolo
  lower), the race's drums lower; the blaze, the seethe, the torches' dips and the cold ticks up where they are
  the moment (still 8-33 LU under in render 2), the music ridden down around them (C21 -2.5, the white heart's loop,
  bar 70, the horn under the roar); anticipation from the measured attacks; the blooms' onsets on the beat.
- **19:45Z: render 1 battery** (`analysis/v3/final_C/`): format/loudness/TP PASS, notes 0, clicks 0, breaths 47-55 dB
  deep; level map 23/28 (C4, C6, C10, C12, C13 too loud: the harp-harmonics voice ~14 dB hot, the glass and the
  falling glass a few dB hot); sync 41/50 (soft string entries under-anticipated: the quiet sections' bow needs
  ~0.55 s; two probes measured the wrong thing). **The effects were buried** (16 LU under the score; B 8, the
  fallback 11): the roar 9 LU under its horn, the beacons 13-15 LU under the Run, the seethe 28 LU under.
  **Render 2 (19:45Z)**: `SCORE_TRIM_DB` -4.5 (gain only), voices rebalanced, the music leaves room at each fire
  (the horn under the roar, the Run ridden -3, the Eye and the map bloom after their burn-throughs), the white heart
  thinned to the healing Ring loop + the seethe, a growing blaze for the fire everyone lit (`C+.fire.rises`), sync
  trims from per-layer measurements (`ANTIC_HI`, `ANTIC_DT`, `ANTIC_SET`).
- **19:15Z: score written** (`src/score_v3_C.py`, plan bar by bar in its docstring; 63 parts, ~1,400 notes, 50
  sync points; page checks 0: notes, rates, no stroke on any ignition, no ANSWER before beacon 2, no HOME before
  the dawn). **Render 1 queued** behind COMPOSER-A's (`wait_and_render.sh` waits for any render_v3 to finish;
  log `C1.log` in the scratchpad). Next: the battery, a ride (`RIDE` in score_v3_C.py), iterate.
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

