# THE LONG DAWN v3: score & sound (COMPOSER-v3 notes)

## MASTERS: THE ONE TABLE (EDIT wires these into the animatics; both composers update only their own rows)

| cut | file | status | owner |
|---|---|---|---|
| B score (THE VIGIL) | `music/out/v3/final_B.wav` (stems `final_B_score.wav`, `final_B_sfx.wav`) | **FINAL 10:50Z**: battery all PASS (sync 25/25, level map 14/14, dawn -2.3 LU under the crane, notes 0, clicks 0) | - |
| A fallback | `music/out/v3/fallback_A.wav` (stems `fallback_A_score.wav`, `fallback_A_sfx.wav`) | **DONE 18:58Z** (re-rendered on the current engine so the file matches the committed source): battery all PASS (level map 20/20, sync 8/8, notes 0, clicks 0) | COMPOSER-A |
| B fallback | `music/out/v3/fallback_B.wav` (stems `fallback_B_score.wav`, `fallback_B_sfx.wav`) | **DONE 11:45Z**: battery all PASS (level map 14/14, sync 5/5, notes 0, clicks 0) | - |
| C fallback | `music/out/v3/fallback_C.wav` (stems `fallback_C_score.wav`, `fallback_C_sfx.wav`) | **DONE 18:54Z**: battery all PASS (level map 28/28, sync 7/7, notes 0, clicks 0). Fix: the ride (C2 bar-2 horn -6.5 dB, C12 piano -4.5 dB) + C11's race drone now falls away after the slip into C12's black (its tail was C12's peak, -6.4 LU) | COMPOSER-A |
| A score | `music/out/v3/final_A.wav` (stems `final_A_score.wav`, `final_A_sfx.wav`; review copy `score_A.wav`) | **FINAL v2 (28 Sep, COMPOSER-A2): the race re-voiced as dread** (see COMPOSER-A2 below). Battery ALL PASS: level map 20/20, rules 3/3 (the edge -1.6 LU under the brink; the blue hour -4.9; the first fire -14.6), centroid arc 5/5 (A7 275 Hz), sync 55/55, notes 0, clicks 0, -16.03 LUFS, TP -1.29 dBTP. **v1 kept for A/B:** `final_A_v1.wav` (+ `final_A_v1_score.wav`, report `analysis/v3/final_A_v1_report.txt`, manifest `cache/v3/manifest_final_A_v1.json`) | COMPOSER-A2 |
| C score (THE LAST PAGES) | `music/out/v3/final_C.wav` (stems `final_C_score.wav`, `final_C_sfx.wav`) | **FINAL 21:05Z**: battery all PASS (format/length/stems, -16.03 LUFS, TP -1.30 dBTP; level map 28/28; sync 51/51; rules 3/3: the slit is C's loudest, the dawn -2.5 LU under it, the prologue -8.7 LU; breaths 44-56 dB deep; notes 0; clicks 0). Review sheet `analysis/v3/final_C/review_C.jpg` | COMPOSER-C |
| B with REAL effects (SOUND) | `music/out/v3/sound_B.wav` (stems `sound_B_score.wav` = final_B's score through the same master, `sound_B_sfx.wav` = `sfx_B.wav`) | **PASS 21:10Z**: battery all PASS (level map 14/14, dawn -2.3 LU, first half -6.1 LU, sync 25/25, notes 0, clicks 0); effects hits on their frames 15/15 (strikes 0.0-0.1 ms). 48 kHz/24-bit, 10,880,000 samples, -16.05 LUFS, TP -1.30 | SOUND |
| C with REAL effects (SOUND-C) | `music/out/v3/sound_C.wav` (stems `sound_C_score.wav` = final_C's pre-master score through the same master, `sound_C_sfx.wav` = `sfx_C.wav`) | **APPROVED by the director 28 Sep ~01:05Z; EDIT-3 adopts it.** PASS 01:00Z: every page/pen/burn on its MEASURED picture frame (`sound/picture_sync_C.json`); battery all PASS: level map 28/28, rules 3/3, sync 51/51, notes 0, clicks 0, -16.03 LUFS, TP -1.30, 14,400,000 samples. Council on v1 frames: re-check after the Cycles rebuild. **For EDIT:** use in place of `final_C.wav` once the director approves | SOUND-C |

All masters: 48 kHz / 24-bit / stereo WAV, exactly the cut's length (A 6,480 f = 12,960,000 samples; B 5,440 f =
10,880,000; C 7,200 f = 14,400,000), -16 LUFS integrated, true peak <= -1.2 dBTP, from silence to silence, with
stem-linked `<name>_score.wav` + `<name>_sfx.wav` (score + sfx = master).

## SOUND (real effects lane, from 19:00Z): THE INTERFACE + STATE (COMPOSER-A, COMPOSER-C, EDIT: please read)
**STATE AT 21:15Z (SOUND):**
- **B: PASSES.** `out/v3/sound_B.wav` = final_B's pre-master score + the REAL effects, mastered by `render_v3.master()`.
  The battery: level map 14/14, dawn -2.3 LU, first half -6.1 LU (limit -6.0; the score alone is -6.5 there), sync
  25/25, notes 0, clicks 0, TP -1.30, -16.05 LUFS. The effects' own hits land on their frames (lid, 3 strikes, 10
  feeds, her flare: 15/15 within 10 ms). The deliverable effects stem is `out/v3/sfx_B.wav`. Fixes on the way: the
  roar settles within 3 s; the reveal fire recedes with the 520 m pull-back under her CALL; an effects-only peak guard
  keeps the effects out of the master's limiter (74 ms ducked); bed caps no longer misread a sparse synthesized
  crackle; knocks are aligned on their loudest contact.
  **For EDIT:** use `sound_B.wav` in place of `final_B.wav` once the director approves (same length, same score).
- **The flint take (shared by A, B, C):** `src/sound_flint_v3.py`. Real strikes (Freesound 499027, CC0), breath,
  catch and roar, used by all three cuts, each at its own cue level.
- **C:** `src/sound_recipes_C.py` is drafted, covering every cue plus COMPOSER-C's own effects (pen, burns, drop, cock,
  cold tick, seethe, dips, roads, fire remains/rises, read from `score_v3_C.extra_effects`), a hooded murmur (council
  -> ring_set) and the catch. NOT rendered yet. Before rendering, add a hook in `sound_v3.build` that appends
  `R.extra_cues(bm)`. Swap the pages to the Sonniss 344 Audio "slow_page_turns" and the sea to "soft_waves_cliffs".
  The drop is f2200 with an after-drip +0.44 s (MIRROR's note below).
- **A:** recipes not started (the fire act, the watch-fires, the beacons: use the flint module + oil_flare + the
  Freesound fires).
- **Sources:** 505 MB in `cache/sound/` (Freesound originals in `fs/`, ElevenLabs in `el/`, Sonniss picks in `sn/`: pine
  branches, douse, oil flare, soft waves cliffs, antique book pages, paper foley, eye of the storm). All five Sonniss
  bundles were harvested and deleted from ~/Downloads. Freesound OAuth is valid until about 19:00Z tomorrow
  (`freesound_v3.py refresh`). ElevenLabs: about 250 credits used of the 30,000 budget.
- **Measured reverb: WIRED as an option** (agreed with COMPOSER-A): `LONGDAWN_HALL=church python render_v3.py A` uses
  the measured church (`ir_v3.irs("church")`). Its energy normalisation is identical to the synthesized hall's. Unset,
  nothing changes. A needs a re-mix plus a battery re-run (about 5 min, no part re-renders), and that is COMPOSER-A's
  render.

- **What:** SOUND replaces the synthesized effects with REAL recordings: Freesound originals (CC0 / CC-BY), a few
  Sonniss GDC 2026 picks, and ElevenLabs only for the gaps. Every cue of the locked cue sheets keeps its id, time,
  fades and envelope. **Its level is matched to your synthesized design of the same cue**, and it is never louder on
  any scale: beds by integrated loudness (with the loudest 3 s capped at +1 dB); events by the loudest 400 ms, capped
  by the loudest 3 s and by the design's sample peak. Credits: `music/SFX_CREDITS.md`. Source: `src/sound_v3.py`
  (engine), `src/sound_recipes_<cut>.py` (what plays), `src/sound_flint_v3.py` (the shared take), `src/ir_v3.py`
  (MEASURED impulse responses).
- **Files per cut (48 kHz, exact length):**
  - `out/v3/sfxpre_<cut>.wav` is the real effects in the PRE-master domain (float32), after the peak guard.
  - `out/v3/sound_<cut>.wav` is `sound_<cut>_score.wav` + `sound_<cut>_sfx.wav`, made by YOUR `render_v3.master()` on the saved
    `cache/v3/premaster_score_final_<cut>.npy` plus the real effects: the same -16 LUFS / <= -1.2 dBTP chain. The battery
    (`analysis/v3/sound_<cut>/`) is run on it, with final_<cut>'s manifest.
  - `out/v3/sfx_<cut>.wav` is a copy of `sound_<cut>_sfx.wav`, the deliverable effects stem.
- **For COMPOSER-A / COMPOSER-C:** nothing changes in your renders unless you opt in. When your final is locked, run
  `python sound_v3.py <cut>` (about 3 minutes): it re-masters your newest pre-master score with the real effects and
  runs your battery. The cue sheets stay locked. Where the picture changed, the recipe says why. B is THE VIGIL, so
  `B.his_fire` is silent (the far answer is a pinprick) and `B.x.flare` is added (her own flare, 40 b2).

## PICTURE SYNC: C PAGES (SOUND-C <-> PAGES-C; SOUND-C from 28 Sep 00:40Z)
**LIVE CONTRACT (director, ~01:05Z):** SOUND-C re-measures and re-syncs when PAGES-C's re-rendered pages/burns land, when
the X1 burns at 4160 and 5600 are wired in, and when the Cycles council lands. **PAGES-C / ACCORD: message SOUND-C
when your frames are in the delivery master (or in renders/ with the EDL pointing at them).**
**STATE 01:00Z (SOUND-C):** `out/v3/sound_C.wav` rendered with the table below; battery all PASS (level map 28/28,
rules 3/3, sync 51/51, notes 0, clicks 0, -16.03 LUFS, TP -1.30). Each moved effect re-measured in the render: its
loudest moment lands on the picture frame (pages 575/6417/6741/6781 exact; burns rise from the edge/hole frame).
SOUND's uncommitted WIP (crest limiter, --levels, arrival-aligned flint/flares) was committed as f27e56c.
The user: C's effects are "off with the visuals ... around the paper/page stuff mostly". Measured from the frames of the
23:36Z master (`src/sync_audit_v3.py`), final_C's page sounds peak 15-23 f BEFORE the leaf moves (they sit on the cue
frame; the picture turns later). SOUND-C now places every page/pen/burn on the MEASURED picture frame. The table the
sound reads is `sound/picture_sync_C.json` (`f` = the frame the sound's hit lands on). **PAGES-C: if you re-time or
redraw any of these, write your new frame in that JSON (or here) and tell SOUND-C: a re-sync is ~3 min.**
Sync definitions: a page = the frame the leaf CROSSES OVER (its fastest frame; the lift's rustle rises before it, the
landing follows). A burn-through = the frame the hole OPENS (or, for an edge sweep, mid-sweep; the crackle rises from
the edge's first frame). A pen = its first and last stroke (the quill's level follows the drawing between them).

| event | cue f | final_C heard | picture f (measured) | now |
|---|---|---|---|---|
| riffle | 320 | 320 (swell from 303, one leaf) | flutter 320-348, lands 349 | the flicking take, bursts 320-348 |
| pen, the mountain | 336.8 | 337-394 | first stroke 355, last 535 (densest 455-491) | quill 355-542 following the drawing |
| page_turn | 560 | 560 (swell from 542) | lift 564, crosses 575, lands 579 | 575 |
| burn, letters | 840 | 841 | hole opens 842 | 842 |
| burn, the deep | 1680 | 1681 | still 1680-85; edge 1686-1700 | rises 1686, peak 1694 |
| pen, the deep | 1689.6 | 1690-1906 | first stroke 1705, to ~1912 | 1705-1911 |
| burn, the Eye | 1920 | 1920 | glow 1921, hole 1924 | 1924 |
| burn, the map | 4160 | swell 4150 -> 4160 | HARD CUT, no burn (X1 not built) | starts ON the cut |
| burn, remains | 5600 | swell 5594 -> 5600 | HARD CUT, no burn (X1 not built) | starts ON the cut |
| roads | 5606 | 5611 | flames leave the ring ~5604 | 5604 |
| page, plenty | 6160 | 6160 | hard cut; NOTHING turns | silent (a turn into C25 gets it back) |
| page, havens | 6400 | 6400 (swell from 6382) | lift ~6392, crosses 6417, lands 6422 | 6417 |
| page, blank | 6720 | 6720 (swell from 6702) | lift 6708, sweep to 6741, lands 6743 | 6741 |
| page, blank_2 | 6760 | 6760 (swell from 6740) | lift 6766, crosses 6781, lands 6783 | 6781 |
| pen T14 | 6790 | 6790 | write-on 6793-6815 | 6793 |
| burn, title | 6980 | 6980 | first spark 6983, letters 6983-7016 | rises 6983, peak 6995 |
On the picture already (<= 2 f): pen T1 400, T7 1710, T9 3640; the drop 2200; strikes 2980/3009/3178; the catch 3316;
beacons 1,2,4,5,6,7; the seethe's flare 5420 (ring_C). The council (4480-5679) is synced to its v1 frames (the dips
now 5537-5569, when the torches are seen going in): **re-check when the Cycles rebuild lands.** Not measurable (slates
in the master): the snow strike ~2900, the roar 3360, the cold tick 3400.

## PICTURE SYNC: C10 THE MIRROR (MIRROR lane, 21:10Z; for COMPOSER-C and SOUND)
- Picture is `renders/mirror_C` (C 2080-2319), built to barmap_C: the drop STRIKES the water on **f2200 exactly (28 b3)**:
  put the drop's splash transient there (`sfx("drop", ..., ev("drop"))` already does). The falling glint is only
  f2198-2199 (no whoosh needed). A secondary drop (the jet's bead falling back) makes a second, smaller ring at
  **~f2210.6**: if SOUND's drop recording has its own small after-drip, let it land about 0.44 s after the first.
- The golden dawn blooms 2200-2224 and holds to **2248** (the score's `back`), then the fire returns 2248-2288:
  the harp's Ab5 at back+0.2 sits right on it. The breath that brings the fire in crosses 2096-2158 (under the
  harp's D6 at mir+1.0). No other sync points; the picture has no Eye, so nothing needs the Eye's sting.

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

## COMPOSER-A2 (cut A) · STATE 28 Sep: FINAL v2, THE RACE RE-VOICED AS DREAD
- **The note (the user, bars 1-49):** "great in parts but ... a bit discordant at other times (e.g. during the race it was a bit
  too upbeat ... at the start of the towers rising)". **Cause, measured with `who_v3.py`:** at the towers' start (60-65 s) the
  thinking cycle, the prize's bright arpeggio in the glass (E5-E6, sixteenths), led the score by 7.8 dB, and it led the edge's
  first bars too (-30.3 LUFS, over the taiko): a sparkling synth arp over a drum build reads as adventure.
- **What changed (all in `src/score_v3_A.py`, v2 comments in place):**
  * A5 -> A6 turn: the calm fire's bright cycle is cut off with the promise (18 b3) instead of running on into the towers.
  * A6: the prize A7(b9) built from the bottom, darkest first (root; E3-Bb3 tritone + b9; the dim7 closes, G3 C#4; the b9 on
    top); its major third only ever inside. The cycle stays sixteenths but smothered (new seat `cycle_dk`: the same FM, ratio
    3.5, low index, low-passed, damped; the prize circled low and close, A Bb A G A Bb C#), now under the strings (-44.6 vs
    the basses' -40.6). The drums: no pickup; soft timpani on A under bars 21-22; a tightening pulse.
  * A7: the spiccato insistent, not skipping (cellos D D Ab D; violas hammer the tritone Ab Ab G Ab; from 28 second violins
    D D Eb D, first violins the violas' tritone an octave up; nothing above D5; it went to D6). Subdivisions on the low tenor
    only, even (no tenor_hi backbeat or gallop). The horns join the corrupted call in unison with the trombones (D3), not an
    octave up. The gold's ring is a pure fifth (D6 + A6) on each bar's first surge (and its third from 26), not a D-major
    figure on every beat far up with a glockenspiel. The cycle smothered (D Eb D C D Eb Ab). Doubling scheme, taiko, brass
    calls, bellows, sub, roll and brink untouched.
  * **Other mismatches found in the scan:** (1) the A5 -> A6 carry-over above; (2) A16 bar 56 recalled the race in the bright
    glass: now the race's own smothered cycle, clearing into the glass as the giants open (57 b1). A1-A5, A8-A20 checked
    against the locked sheet: no other place where the music's feeling contradicts the picture.
- **Levels:** a darker race lowers the film, the master gain rises, and the brink is pinned by the limiter, so every untouched
  section creeps up (+0.6 LU over the two renders). RIDE A2 -2.3, A3 -2.3, A4 -3.5 (were -1.5/-1.5/-2.8). Margins now thin:
  A2, A3, A4 and A10 are 0.2 LU from their band tops, A17 0.3, the edge rule 0.1 (it was 0.0 in v1). If a re-master (SOUND's
  `sound_v3.py`) tips one: trim that section's RIDE by 0.3 (A10: check the silence-piano syncs, +/-10 ms, after any change).
- **B is dropped. Proposal only (not done):** B's ending passed the ANSWER note by note from horn to farther horn, each then
  silent, and brought back the dead of night's high harmonic at the hand-off. For A20: as the title crumbles, pass the ANSWER
  (D C# B F#) outward, horn -> far -> farther, one note each, each falling silent, over the D add9; and let A11's pure harmonic
  A5 (the false dawn's beating A made pure) return ppp as the rose sky pales. It closes A's false-dawn / true-dawn circle and says
  "carried by many" without getting louder. About one render; the blue hour's whole theme and cadence stay as they are.

## COMPOSER-A (cut A) · STATE 20:40Z: BOTH TASKS DONE
- **1. C fallback: DONE** (table). The ride alone fixed C2 but not C12: C12's loudest 3 s was C11's race drone
  swelling into the black and its hall tail. `fallback_v3.py` now lets a race section cut to silence or black (not
  the brink) fall away over its last two beats, as C's sheet asks at the slip. Only C11 matches (A's and B's part
  keys checked unchanged). A's fallback re-rendered on the current engine too: still passes.
- **2. A's full score: FINAL** (`src/score_v3_A.py`, written fresh on the kit; the plan is its docstring, bar by bar).
  Nothing of the v1/v2 kindling or race is reused. What carries the story, and where to look in the code:
  * the thinking cycle's pace is the plot: far-off 16ths (false dawn); a cycle shrinking 7 -> 5 -> 3 -> 1 notes
    into the point (`opening`); below notice in the calm fire; the prize's arpeggio raced in 16ths (towers, edge);
    16ths -> 8ths -> quarters locked to the bar on 59 b1 (`light`); walking in step, its glass warming toward gold
    (`crossing`, `warm_morph`); warm inside the dawn;
  * the corrupted call at four speeds in the edge (quarters, 8ths, 16ths in the middle strings only, a sustained
    roll: never 32nds); a sweet D-major ring on every surge; the bellows drawing into every beat;
  * the CALL for every fire (violas on the roar; a horn at the karst; the low strings at the desert; near horns on
    the Run's odd fires; the violas inside the chord at every watch-fire); the ANSWER first on the Run's second
    fire (51 b1), first in major on fire 6 (its C# the #11 over G); the Run's harmony rising by thirds from Bb to
    Bm over the D pedal (`RUN_CHORDS`, `contour_lead`); the watchers' corrupted call answered by a far CALL; the
    tritone healing to the fifth as the giants open (57 b1);
  * the whole theme only in the blue hour (CALL 74 b3, ANSWER 76 b1, HOME 77 b1), the first real cadence A7 -> D on
    78 b1; the effects own the impact (the score adds only a short timpani + bass drum under it) and every fire.
  * Levels: a drum-bus limiter (`GROUPS`), the conductor's ride (`RIDE`), measured sync trims (`SYNC_TRIM`) and
    centroid targets (`CENTROID`). Six renders; their findings are in the history below.
- **Lesson for the next render of any cut:** the level map is relative to the loudest 3 s, and that anchor is
  pinned by the master's limiter (the brink takes ~10 dB of gain reduction). Any pass that makes the film quieter
  raises the master gain and lifts every untouched section (render 5: +0.7 LU). Keep ~1 LU of margin, and check
  each story line's prominence with `src/who_v3.py` (per-part loudness in a window), not only the bands.
- **Housekeeping:** `cache/v3/parts` is 20 GB of content-addressed part renders, many stale. Prune (keep only the
  keys in the current `cache/v3/manifest_*.json`) only when NO render is running: a running render's parts are
  not in a manifest until it finishes.
- History of A's renders:
  * Render 1 (19:16Z): format/breaths/notes/clicks PASS; level map 6 FAIL, sync 17/54 FAIL, rule "brink loudest"
    FAIL. Causes: the master crushed the brink by ~13 dB (transient peaks: taiko, the crumble piano, the drum roll),
    so the edge tied it; the glass too hot in bars 12-13; the score's own impact rang into bar 36's true silence.
    Sync: bowed D3 fundamentals arrive 0.15-0.4 s after the kit's broadband attack estimate; legato octave drops
    ~0.55 s; entries that crescendo from nothing through a velocity-layer boundary ~0.25 s.
  * Render 2 (19:40Z): level map 18/20 (the edge -0.4 LU under the brink; A10's piano -7.1 vs -10), sync 50/54
    (the watch tone +60 ms; the low pizz measured in its band caught the previous note's leakage). The master
    still took 6-7 dB off every taiko hit in the edge (pumping) and 8-11 dB off the brink.
  * Render 3 (19:57Z): level map 20/20, sync 55/55, centroid 5/5; one rule FAIL: bars 28-29 (four spiccato
    sections in sixteenths + five brass) came within 0.8 LU of the brink. Checked: bar 36's two seconds are true
    silence (score -111 dBFS; only the effects' ash bed's 0.3 s fade tail).
  * Render 4 (20:08Z): **battery ALL PASS** (level map 20/20, rules 3/3, centroid 5/5, sync 55/55, notes 0, clicks 0,
    -16.04 LUFS, TP -1.29 dBTP). Fix: sparse beats dense in the sixteenths (only violas + second violins double;
    cellos and first violins keep eighths), softer brass call. (Its backup copy was deleted once render 6
    passed.)
  * Render 5 (20:23Z): the balance worked (per-part loudness: every story line now leads its window: the
    promise's line level with the bloom's top, the violas' CALL 1.8 dB over its pad, the first ANSWER leading the
    Run, the watchers' corrupted call over the hold, the dawn's ANSWER and HOME on top). But the master gain rose
    0.7 dB with the brink pinned at the limiter, so A2, A3 and A10 crept 0.2-0.5 LU over their bands, and run fire
    1's CALL arrived +56 ms. Lesson: the level map is relative to a limited anchor, so every quieter pass lifts all
    the untouched sections; keep ~1 LU of margin.
  * Render 6 (20:35Z): `RIDE` A2 -1.5, A3 -1.5, A4 -2.8, A7 -2.4, A10 -5, A16 -1, A17 -0.8; trim run fire 1:
    **battery ALL PASS -> FINAL.**
  * Tool: `src/who_v3.py <cut> <name> t0 t1 ...` ranks the parts by loudness in a window (from the part cache).

## COMPOSER-C (cut C) · STATE
- **21:05Z: C's full score FINAL** (`out/v3/final_C.wav` + stems; render 7). Battery all PASS: 48 kHz/24-bit, exactly
  14,400,000 samples, stems sum to the master, -16.03 LUFS integrated, TP -1.30 dBTP; level map **28/28** against the
  locked cue sheet's bands; **sync 51/51**; rules 3/3 (C's loudest 3 s at the slit, 83.5 s; everything after the Eye
  >= 1.8 LU under it; the dawn -2.5 LU under it: warm, not loud; the prologue -8.7 LU); the three breaths 44-56 dB
  deep; notes 0 (stuck/overlap/rates: percussion never faster than 16ths); clicks 0; master comp GR 5.0 / lim GR 5.6
  dB max (the slit). Effects: -23.7 LUFS vs the score's -16.1 (B-like), and each fire owns its moment (the roar +2 LU
  over its horn; the burns, the seethe, the dips, the blaze within 3-6 LU; the beacons ~6-7 LU under the Run's
  16ths, the ostinato resting on each beacon's 16th). Page rules (`python score_v3_C.py`): no stroke on any
  ignition, no ANSWER before beacon 2, no HOME before the dawn. Director's sheet: `analysis/v3/final_C/review_C.jpg`.
- **What C's score is** (plan bar by bar in `score_v3_C.py`'s docstring): the book as chamber music (the cor anglais
  CALL, harp, the pen and the page), the fire as orchestra; THE RING (D-Ab-D' C' Ab) loops unresolved from the
  mountain page (stopped horns) through the Eye (trombones, the slit = C's loudest), the find (violas, ppp, as a
  sweet D7#11), the fire test (the loop holds its D where the Ring should fall), the council (circling beneath the
  chorale) and the bearer (rising by minor thirds, tremolo), until it HEALS in the white heart (C# and A as the
  letters flare), 275 ms of silence, and cadences into D once (69 b1). **THE FIRE REMAINS** (H5): D major settles;
  the ANSWER passed outward horn to farther horn in canon over G/B - D/A - A; the breath; the sunrise's D add9 bloom
  and the whole theme (CALL on one horn, ANSWER + HOME in octave violins with the solo violin above, home on 76 b1).
  **THE MIRROR** scored (one held A5, harp harmonics on the Ring's D/Ab turning to D major for the drop's breath).
  Nothing Shire-like: no flute, no whistle, no pipes (the map's oboe is doubled by a clarinet).
- **Shared-file change (additive only):** `analyze_v3.py` gains a sync kind `arrivew` (the `arrive` criterion on the
  part's broadband envelope): low violas, the bassoon and a cello's D2 have almost no fundamental (checked by FFT:
  the harmonic series are right) and a solo violin's vibrato leaves the +-60-cent band, so `arrive:<midi>` jumps
  there. No existing kind changed; COMPOSER-A's reports are unaffected.
- **Housekeeping:** pruned 2.5 GB of my superseded part stems (only names unique to C, none in any manifest).
- **20:50Z: render 4 battery**: level map **28/28**, rules 3/3, notes 0, clicks 0, breaths 44-56 dB deep, master
  lim GR 6.2 dB; sync 47/50. The 3 misses were the PROBE, not the music (FFT per part: every harmonic series is
  right, no wrong notes): low violas, the bassoon and a cello's D2 have almost no fundamental, and a solo violin's
  vibrato leaves the +-60-cent band, so `arrive:<midi>` jumps. Added to `analyze_v3.py` (additive: a new kind,
  no existing kind changed) **`arrivew`**: the same arrival on the part's broadband envelope, for entries out of
  their own silence; crescendo blooms are probed by their onset (`bloom`). On all 35 soft entries the two probes
  agree within 40 ms except those. **Render 5 (20:55Z)**: the trims from those probes, the dawn -0.5 dB (its rule
  was at -2.1 of -2.0), the slit's tam-tam/violins -1.5 dB (fewer stacked peaks), the Run's line and timpani -2 dB
  under the horn calls. The blaze now rises from 8 to ~5.6 LU under C21's orchestra; the Havens' sea 7 LU under.
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

