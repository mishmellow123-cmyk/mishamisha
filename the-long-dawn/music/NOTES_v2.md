# THE LONG DAWN v2 — score & sound (music department notes)

Read `../BIBLE_V2.md` first (three cuts, the tone rules, the v2 timeline §3). v1 is untouched:
`music/NOTES.md`, `music/CUES.md`, `music/src/*.py` (v1 engine and score), `music/out/*.wav` and
`music/analysis/*` are exactly as delivered.

## Deliverables (`music/out/v2/`)

| file | what |
|---|---|
| `final_AB.wav`, `final_C.wav` | mastered score + SFX for cuts A/B and cut C: 48 kHz / 24-bit / stereo, exactly 5 936 000 samples (123.667 s), −16 LUFS integrated, true peak ≤ −1.3 dBTP, from silence to silence |
| `score_AB.wav`, `score_C.wav` | music only (master-processed stem) |
| `sfx_AB.wav`, `sfx_C.wav` | sound effects only (master-processed stem). **Stem-linked**: `score + sfx == final` (max error ≈ 5e−7) |
| `sfx_events_AB.json`, `sfx_events_C.json` + `sfx_events/*.wav` | every effect as its own clip (peak −6 dBFS): `frame` = clip start (v2 frames), `hit_frame` = its sync point (a whole number of frames into the clip), `gain_db` = the gain applied to the clip in `sfx_<cut>.wav` (before the master), `gain_db_master` = the same after the master's static gain, `breath_exempt` |
| `analysis_AB/`, `analysis_C/` | `report.txt` (format checks, LUFS / true peak whole + per section, the breaths measured, the sync table measured), `overview.png`, `sync.png`, `spec_<section>.png`, `notes.txt` (note/voicing audit), `roll.png` (piano roll of the second half) |
| `compare_AB_C_v1.png` | short-term loudness of the v1 film, AB and C on one timeline |
| `../../src/*_v2.py` | the code (see the last section); the pre-review versions of the scores are not kept separately: the revision is described below |

## Re-render (one command)

```bash
source ~/.venvs/longdawn/env.sh
cd music/src && python render_v2.py        # both scores, AB then C: parts (cached) -> mix -> SFX -> master -> analysis
python render_v2.py --cut AB               # one score
python render_v2.py --cut C --force        # ignore the part cache (a clean render is ~4 min per cut on this Mac)
python render_v2.py --cut AB --only ab_hn1,ab_vln1     # re-render some parts
python sfx_v2.py --cut C --assemble        # rebuild an SFX stem from an edited sfx_events_C.json (then re-master with render_v2)
python notes_v2.py AB                      # note / voicing / range audit -> analysis_AB/notes.txt
python balance_v2.py AB 2400 2624          # which parts dominate a window (and in which band)
python roll_v2.py C                        # piano roll -> analysis_C/roll.png
python check_samples.py                    # (v1 tool) FFT pitch audit of every sample
```
Parts render in at most 2 worker processes (sampler caches capped at ~0.36 GB per worker; a clean render of
both cuts takes 4.5 min on this Mac with peak RSS ≈ 1.5 GB) and are cached content-addressed in `music/cache/v2/parts/<name>__<key>.npy`, so the v1
first-half parts are rendered once and shared by both cuts. Everything is seeded and reproducible.

## What changed from v1

1. **Timeline**: `timeline_v2.py` — 2968 frames, the BEACON RUN at 1520–1679, everything after it +160.
2. **The first half (0–1199) is the v1 composition, called unchanged** from `score.py`
   (`intro`, `kindling`, `race`, `silence`, v1 seating and faders). Re-rendered from the exported code it
   matches the delivered v1 `score.wav` sample for sample in the intro (residual −115 dB), kindling, grasp and
   silence. Two things are now in the score pipeline instead of the edit:
   * **the breaths** (below), and
   * **the race-entrance reinforcement** that the finished film carried (edit/mix.py `REINFORCE['race']` +
     `PUSH`): the same layers with the score's own instruments (`v2r_*` parts), summed on their own bus with a
     true-peak bus limiter (as the editor did) so they sit under the score's peaks instead of making the master
     pump. The race entrance now lands within ~1 dB of the approved film (short-term loudness), with a real breath.
3. **Two new second halves** (bars 16–37): `score_v2.py` (score AB, cuts A and B) and `score_v2_c.py`
   (score C, the Tolkien cut). See below.
4. **SFX rebuilt** for the v2 timeline (`sfx_v2.py`): v1 designs at their v1 frames before 1520, +160 after;
   new THE BEACON RUN (flight wind, a whoomp/roar on each beacon's beat — farther ones softer, darker, narrower
   and more reverberant — and small far ones filling between the beats), cut C's parchment + small-fire crackle
   under the map (1920–2087), a field of distant torches as the emissaries converge, the carved ring's travelling
   fire-hiss (2320–2380), a hearth flare that peaks exactly at the breath, and (revision 2) a faint echo of the three
   flint strikes as the torch changes hands (2756/2768/2780). Every clip has its own seed, so editing
   the list never changes other clips. v1 had set the hits so low the music masked them (the editor lifted two
   by hand); v2 levels were set by measuring each hit against the music in its band.
5. **Mix / master** (`render_v2.py`): per-part seating, distance EQ, hall as v1; new: mix-time EQ per part (a
   gentle "air" shelf on the second-half strings/organ/harp — measured, they read far darker than the first
   half's glass, cymbals and ticks), bus groups with their own limiter, the breaths, the push. Master: stem-linked
   glue compression (1.5:1) + 4× oversampled true-peak look-ahead limiter (chunked, low memory), −16 LUFS,
   10 ms fade from silence, 350 ms fade to silence, 24-bit TPDF.
6. **Samples**: VSCO-2-CE (CC0) only. v2 uses the sampled pipe organ (Ivy Audio, in VSCO-2-CE: loud manual
   "Man3Open", soft stopped flute "Man3Quiet", loud 16′ pedal — measured to sound an octave below its key — and
   quiet pedal), registered in `sampler_v2.py` so `sampler.py` stays v1. `check_samples.py` re-run: all 64
   patches load every mapped region (1252 samples, the same 121 flags as v1). **Additional sources considered**
   (the director allowed CC0/PD): VCSL (CC0 — no choir, no solo strings we needed), the Discord SFZ GM bank
   (choir entries are placeholders), freesound CC0 (no usable multisampled choir), University of Iowa MIS
   (usable, but VSCO already covers the solo lines this score needs). No choir is used in either score: nothing
   CC0 was good enough, and the synthesized choir is exactly what the producers heard as fake. C's "choral
   colour" is a pianissimo brass chorale instead.

## The breaths: why `score.BREATHS` didn't apply in v1

**The delivered v1 audio predates the breath code.** The department added `BREATHS` (score.py) and
`breath_env` (render.py), plus a bigger race-entrance hit, and exported the code without re-rendering:
* `render.breath_env` produces exactly the designed windows (−45 dB, 30 ms fade down, 5 ms up ending on the
  downbeat) when run on the exported code — the code is right.
* Re-rendering the v1 first half from the exported code reproduces the delivered `score.wav` sample for sample
  outside the race (intro residual −115 dB; kindling −39 dB, grasp −23 dB, silence −23 dB — master-gain
  differences only), but not inside the race (correlation 0.66–0.97 from 26.7 to 41 s) **even with breaths
  disabled**; the residual is almost entirely the race-entrance impact/timpani/cymbal (correlation −0.66 with the
  exported `impact` part): the entrance hit was also edited after the last render.
* The department's own `analysis/report.txt`, exported with that audio, measured no breath: the RACE entrance
  "rise −8 dB", the IMPACT "rise 2 dB" (a real breath gives +16…+48 dB, see v2 reports).
* `handoff/HANDOFF.md`: at export the department "was working on a louder climax arrival at 93.333 s and a
  clearer race-drum entrance at 26.667 s".

Two latent pipeline bugs would have spoiled the breaths even after a re-render, and are fixed in v2:
* the SFX stem (and its outdoor/hall reverbs) was never ducked — the storm roar's tails filled the IMPACT suck;
* v1 ducked the hall **output** only inside the window, so the pre-breath hall tail came straight back after
  the downbeat. v2 convolves the reverb **per segment between breaths** and discards each segment's tail at the
  next breath: a real suck.

v2 breaths (`score_v2.BREATHS`, applied to every score part, the score hall, every SFX clip and the SFX
spaces): 200 ms before the RACE (640, the reversed cymbal exempt as designed), the suck 1037→1040, and 275 ms of
silence before the DAWN (2400). `analyze_v2` measures each one (all non-exempt parts + hall, pre-master) and in
the final files: see the reports (depth 47–56 dB).

## Revision 2 — after the critics' review (26 Sep)
Two independent critics (`_local_logs/review/critic_tone.md` M8, M9, M12, m4, m6; `critic_framing.md` m6)
found that score AB still followed v1's rejected arc (organ-led, major-key uplift). Changes:
* **AB 1360**: strings + low brass on B♭, no organ manuals (the 16′ pedal as a floor); the call moved from
  horns a2 into the violas, inside the chord; no strokes at 1360 or 1480 (the SFX own the fires).
* **AB Beacon Run**: no scale climb, no parallel tenths, no hit on any beacon. A D pedal (the shepherd's C7 lands
  on it deceptively); the harmony moves by thirds above it (Dm – F/D – Am/D – C/D → F at the desert); the string
  16ths follow the chords, so the texture rises by thirds; horns call near and answer far. The SFX whoomps own the
  beacons (raised 4–5 dB; the shepherd and the montage ignitions too).
* **AB montage and world entry**: no organ manuals, no strokes; the harmony change marks each ignition.
* **AB dawn**: awe from harmony, not melody. A bloom out of the silence (no stroke): two string sections on
  D add9 with the thinking fire's cycle glowing in it, 80 frames with no tune, then E/D (Lydian) – Gmaj9/D – D add9
  over the D pedal. The call enters late (2480) on one voice, the cellos in unison, mf. No brass, no organ manual.
  Peak −13.5 LUFS short-term, below the race (−12.7).
* **Ring sweep (both scores)**: a swell on the held chord, not a rising figure.
* **AB coda**: the answering fires are string swells only (one family).
* **C run**: horn signals on beacons 1, 3, 5, 7 only; the others are fire (SFX) only.
* **C dawn**: a bloom (a timpani roll starting in the light, no stroke); the eagles' phrase continues the line and
  settles home (B–A–F♯–D), with no rising run and no accent.
* **C coda**: the intro's plain flute (no cut/tap/turn, no drone), the distant horns answering the fires and the
  solo violin rising to the ninth kept.
* **Both SFX lists**: a faint echo of the three flint strikes as the torch changes hands (2756, 2768, 2780):
  the same strikes, regenerated from their own seeds, farther and darker. It is the wordless cue that she lit the
  first beacon (cut B has no text to say it).
* Kept, as the critics asked: no choir; C's pianissimo brass chorale; AB's 2160 lock into one cycle; the glass
  arpeggio turning warm in the world answers; the breaths.
* Consequence: the second half is quieter, so at −16 LUFS integrated the master gain rose ~1.5 dB. The shared
  first half now plays ~0.9 dB above the approved film, and the limiter takes up to ~3.6 dB on the race entrance
  (2.7 on the IMPACT).

## Score AB — the second half (cuts A and B)
*The first half's sound world carried on (strings, low brass, piano, the thinking fire's FM colour). The organ is
only a 16′ floor and the soft flute stop inside the world-answers web. Nothing in the music hits a beacon, an
ignition or an oath.*

| bars / frames | picture | music |
|---|---|---|
| 16–17 / 1200–1359 | dark, wind, flint, the catch | only a low D under the piano's dying fifths (the SILENCE's); 1318 B♭/D blooms in low strings; 1340 C/D |
| 18 / 1360 | the beacon ROARS | strings + low brass open on B♭ over the 16′ pedal; the CALL in half notes inside the chord (violas): D – A – D; B♭ · F/A · Gm |
| 19 / 1440–1519 | the shepherd | a pulse starts (cellos 8ths, violas 16ths); 1480 the theme's ANSWER in the violas (C – B♭), C9sus4 → C9; a timpani roll into the run |
| 20–21 / 1520–1679 | THE BEACON RUN | a D pedal (the C9 lands on D minor, deceptively); above it the harmony moves by thirds, Dm – F/D – Am/D – C/D; the string 16ths rise with the chords; a horn calls near (D, A) and another answers far (F, C); a timpani roll grows under the second half; the SFX whoomps own the seven beacons |
| 22–24 / 1680–1919 | desert · ice · karst · city · sea | the harmony moves by thirds, one chord per ignition (F B♭ Gm E♭ C F/A) under a held high D; the string pulse goes on; a slow horn line rises through it |
| 25–26 / 1920–2079 | the world answers | **B♭maj9(♯11), the IGNITION chord, now in root position.** The thinking fire's arpeggio is heard exactly as it was in the KINDLING (FM glass) and turns warm note by note, while a web of cycles of different lengths drifts against itself; inside it the theme's answer and third phrase (horn + solo violin) |
| 27–30 / 2080–2399 | the Accord | D minor hush: the KINDLING's own cycle alone on the piano. **2160: the flames merge, and every voice locks into one 8-note cycle that fits the bar.** The build over a D pedal (Dm · B♭/D · C/D, changes on off-beats, never on an oath); 2320 B♭, 2360 C, the ring as a tremolo swell; **275 ms of silence** |
| 31–33 / 2400–2623 | DAWN | **a bloom out of the silence**: two string sections on D add9 with the thinking cycle glowing in it (piano, warm FM), 80 frames with no tune; over the D pedal the harmony turns E/D (the Lydian light) → Gmaj9/D → D add9; **the CALL enters at 2480 on one voice, the cellos in unison, mf**, and comes home on D as the chord does; no stroke, no brass, no organ manual |
| 34–37 / 2624–2967 | the hill, the question, the torch, the fires | the piano (the SILENCE's) asks the call again; D5 hangs through the question; the answer during the hand-over (C♯ – B → F♯ as the torch changes hands, with the faint echo of her flint strikes); 2800 a warm string swell (Gmaj9/D); the answering fires = seven string swells spreading outward, building **D add9**; 2880 the final chord, silent by 2968 |

## Score C — the second half (cut C, the epic-fantasy tradition)
*Original melodies only: the film's Beacon theme and new material. Noble, grave, heartfelt.*

| bars / frames | picture | music |
|---|---|---|
| 16–17 | dark, flint, the catch | low strings, a distant timpani roll like far thunder; 1318 B♭/D; 1340 C/D + timpani roll |
| 18 / 1360 | the beacon ROARS | timpani, strings on D minor, **a lone horn cries the CALL** (D4 A4 D5) |
| 19 / 1480 | the shepherd | **a second horn answers from the far peak** (right, distant) on the dominant; the ride begins (cellos 8ths, violas 16ths in 3+3+2) |
| 20–21 / 1520–1679 | THE BEACON RUN | driving strings (16ths grouped 3+3+2) and timpani; **the call passed from peak to peak on beacons 1, 3, 5, 7** (near left, near right, far left, far right), the beacons between are fire only |
| 22–24 / 1680–1919 | desert … sea | **the whole horn section sings the call** at the desert; every later ignition is answered by a horn from another place (C, B♭, G, C): F · C · B♭ · Gm · C7 → |
| 25–26 / 1920–2087 | the map | **wonder**: F major, the theme in the violins with the solo violin an octave above (the call on F, then the answer over B♭ with its raised fourth, the IGNITION colour now bright); harp arpeggios (damped, not a sweep), horns, an oboe answering |
| 27–30 / 2080–2399 | the Council | hushed D minor, a clarinet intones the call; 2160 the Ring in the fire: a pianissimo low-brass chorale (Dm · B♭/D · Gm/D, moving on off-beats, never on a vow); 2320 B♭, 2360 A7sus4, 2380 A; the ring as a string swell; timpani roll; a suspended-cymbal swell sucked into **275 ms of silence** |
| 31–33 / 2400–2655 | the sun over the eastern ranges | D major, **a bloom out of the silence** (a timpani roll in the light, no stroke): the theme in violins in octaves + solo violin (the call, the answer, then a continuation that settles home, B–A–F♯–D, with no accent where the eagles cross); horns in counterpoint, low brass, harp; no crash, no victory |
| 34–37 / 2624–2967 | the hill … the fires | **the intro's plain flute** (no ornaments, no drone): the CALL; the D5 hangs through the question; the answer during the hand-over (with the echo of the flint strikes); 2800 warm strings (Gmaj7/D); **the answering fires are distant horn calls from the hills**; the solo violin rises to the ninth over D add9 |

## Measured (final renders)
| | final LUFS | final TP | score LUFS / TP | sfx LUFS / TP | length | stem-sum error |
|---|---|---|---|---|---|---|
| AB | -16.04 | -1.30 dBTP | -16.20 / -1.27 dBTP | -25.54 / -4.75 dBTP | 5936000 samples | 4.77e-07 |
| C | -16.05 | -1.30 dBTP | -16.17 / -1.27 dBTP | -26.73 / -5.71 dBTP | 5936000 samples | 4.77e-07 |

Loudness per section (integrated LUFS / short-term max) and true peak, final mix:

| section | AB | C |
|---|---|---|
| intro | -23.62 / -22.1 (TP -12.03) | -24.46 / -22.9 (TP -12.87) |
| kindling | -17.03 / -12.7 (TP -3.07) | -17.78 / -13.2 (TP -3.70) |
| race | -13.42 / -12.7 (TP -1.30) | -13.95 / -13.1 (TP -1.30) |
| grasp | -13.57 / -11.5 (TP -1.30) | -14.17 / -12.0 (TP -1.57) |
| silence | -15.74 / -11.5 (TP -1.30) | -16.11 / -11.9 (TP -1.30) |
| first_beacon | -18.05 / -16.7 (TP -5.60) | -18.29 / -17.5 (TP -6.84) |
| far_peak | -15.95 / -15.1 (TP -3.36) | -16.57 / -15.2 (TP -4.86) |
| beacon_run | -14.37 / -13.8 (TP -1.33) | -13.91 / -13.3 (TP -1.30) |
| montage | -16.27 / -14.8 (TP -4.06) | -15.86 / -14.3 (TP -3.55) |
| world_answers | -15.53 / -14.7 (TP -3.96) | -18.44 / -16.3 (TP -6.92) |
| accord | -17.05 / -12.8 (TP -1.92) | -18.27 / -12.7 (TP -1.58) |
| dawn | -15.79 / -13.5 (TP -3.24) | -12.60 / -11.8 (TP -2.29) |
| coda | -25.17 / -21.6 (TP -10.70) | -25.45 / -18.3 (TP -9.70) |

Cross-checked with ffmpeg's EBU R128 meter: both finals I = −16.0 LUFS, true peak −1.3 dBTP. In AB the dawn now
peaks below the race (−13.5 vs −12.7 LUFS short-term); in C the dawn stays the loudest moment (−11.8), the
epic tradition's eucatastrophe.

Breaths (all non-exempt parts + hall, pre-master): AB: 26.667 s 47.3 dB, 43.333 s 55.4 dB, 100.000 s 54.7 dB; C: 26.667 s 47.3 dB, 43.333 s 55.4 dB, 100.000 s 55.9 dB.

Key sync points (measured onset offsets in ms; percussion at its steepest slope). In score AB the music no longer
strikes the fires, so its beacons, ignitions and the shepherd are measured on the SFX; the timpani offsets are C's.
The C3 timpani reaches its attack ~12 ms after the stroke begins, so score C places that drum 4 ms early.
Slow-attack entries (horn calls, the cellos' call, the flute) are "soft" and anticipated by the sampler so they
speak on the beat. Everything in both cuts is within ±10 ms except v1's storm crash at 800 (+11 ms, first half,
as composed; v1 measured +10.5 ms).

| frame | AB | C |
|---|---|---|
| 1236 | +1.0 ms (sfx) | +1.0 ms (sfx) |
| 1262 | +0.5 ms (sfx) | +0.5 ms (sfx) |
| 1290 | +0.5 ms (sfx) | +0.5 ms (sfx) |
| 1360 | +3.5 ms (sfx) | +0.0 ms (c_timp); -20.5 ms (c_hn_solo) |
| 1480 | +0.0 ms (sfx) | +1.0 ms (c_timp) |
| 1520 | -17.5 ms (ab_hn1) | -2.0 ms (c_timp) |
| 1540 | +0.0 ms (sfx) | +5.5 ms (c_timp); +0.0 ms (sfx) |
| 1560 | +0.0 ms (sfx); -20.5 ms (ab_hnfar) | +1.0 ms (c_timp); +0.0 ms (sfx) |
| 1580 | +0.0 ms (sfx) | +4.0 ms (c_timp); +0.0 ms (sfx) |
| 1600 | +0.0 ms (sfx) | +2.0 ms (c_timp); +0.0 ms (sfx) |
| 1620 | +0.0 ms (sfx) | -1.0 ms (c_timp); +0.0 ms (sfx) |
| 1640 | +0.0 ms (sfx) | +1.5 ms (c_timp); +0.0 ms (sfx) |
| 1660 | -5.5 ms (sfx) | +1.5 ms (c_timp); -5.5 ms (sfx) |
| 1700 | +3.5 ms (sfx) | +4.5 ms (c_timp) |
| 1760 | +0.0 ms (sfx) | -2.0 ms (c_timp) |
| 1810 | +3.0 ms (sfx) | +2.0 ms (c_timp) |
| 1850 | +0.0 ms (sfx) | +0.0 ms (c_timp) |
| 1890 | +0.0 ms (sfx) | -1.0 ms (c_timp) |
| 1920 | +0.0 ms (ab_morph); +0.0 ms (ab_piano) | +4.5 ms (c_timp) |
| 2160 | +0.0 ms (ab_pnoarp) | - |
| 2400 | -5.0 ms (final); +0.0 ms (ab_piano) | -4.5 ms (final); +80.5 ms (c_vln1) |
| 2800 | +0.0 ms (sfx) | +0.0 ms (sfx) |
| 2880 | +4.5 ms (ab_piano) | - |

## Notes for the editor
* The SFX stems are built **from the JSON**: change `frame` / `gain_db` in `sfx_events_<cut>.json`, run
  `python sfx_v2.py --cut <cut> --assemble` then `python render_v2.py --cut <cut>` (parts are cached; ~1 min).
* `final_<cut>.wav` is the mastered mix; `score_<cut>.wav` + `sfx_<cut>.wav` sum to it exactly, so the
  editor can re-balance the stems and stay loudness-consistent.
* Most important hits: flint strikes 1236/1262/1290 (and their faint echo 2756/2768/2780), the kindling catch 1318,
  the beacon roar 1360, the shepherd 1480, the seven run beacons 1540–1660 (and seven small far ones between), the
  five montage ignitions, the flames merging 2160, the ring sweep 2320, the hearth flare into the breath, the
  child's beacon 2800 and the answering-fire whumps 2805–2876. In score AB the music hits none of these: the SFX
  own every fire.
* Nothing in the music hits the oaths/vows 2200/2240/2280 (harmony moves on off-beats 2230, 2290).
* Quiet stretches for text: the first beacon before 1318 (≈ −40 LUFS short-term), the Accord hush 2080–2160,
  the coda question 2660–2730 (the piano alone).

## Engine additions (v1 files untouched)
`timeline_v2.py` (grid), `score_v2.py` + `score_v2_c.py` (the scores, breaths, bus groups, mix EQ, push,
sync tables), `sampler_v2.py` (extra VSCO patches: organ pedals, quiet strings…), `synth_v2.py` (fmwarm — the
thinking colour warm, with per-note timbre so it can morph from the KINDLING's glass; deephit; subpulse; all v1
voices unchanged), `sfx_v2.py`, `render_v2.py`, `analyze_v2.py`, `notes_v2.py`, `balance_v2.py`, `roll_v2.py`.
