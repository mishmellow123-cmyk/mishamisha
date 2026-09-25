# THE LONG DAWN — score & sound (music department notes)

## Deliverables (`music/out/`)
| file | what |
|---|---|
| `score.wav` | music only — 48 kHz / 24-bit / stereo, exactly 5 616 000 samples (117.000 s) |
| `sfx.wav` | sound effects & ambience only (same format) |
| `mix.wav` | score + sfx at the intended balance, mastered (≈ −16 LUFS integrated, true peak ≤ −1.25 dBTP) |
| `sfx_events/*.wav` + `sfx_events.json` | every SFX as its own clip: `frame` = where the clip starts, `hit_frame` = its sync point (always a whole number of frames into the clip), `gain_db` = the gain that reproduces `sfx.wav` (clips are peak-normalised to −6 dBFS; master gain included, master limiting not) |

The stems are **stem-linked**: the master's glue-compression and limiter gain curves were computed on the full mix and applied identically to both stems, so `score.wav + sfx.wav == mix.wav` (max error ≈ 5e−7). The editor can rebalance or replace SFX freely and still sum back to the mix.

## Re-render (one command)
```
cd music/src && python3 render.py          # parts (cached) -> mix -> sfx -> master -> analysis
python3 render.py --force                  # ignore the per-part cache
python3 render.py --only choir,vln1        # re-render some parts
python3 analyze.py notes                   # note/range/voicing audit -> analysis/notes.txt
python3 check_samples.py                   # FFT pitch audit of every sample -> analysis/sample_pitch.txt
```
Samples: `music/samples/VSCO-2-CE` (sparse clone of github.com/sgossner/VSCO-2-CE, CC0) and `music/samples/sfz` (the `.sfz` maps extracted from its `SFZ` branch). Re-create with
`git clone --filter=blob:none --sparse https://github.com/sgossner/VSCO-2-CE` + `git sparse-checkout set` on the folders listed in `render`/`sampler.py`
(Strings sections + solo contrabass/violin/harp, F horn, trumpet, tenor trombone, tuba, flute/oboe/clarinet/bassoon, upright piano, organ, `Percussion`, `VSCO 1 Percussion/drums`, `VSCO 1 Percussion/varMetal`).
Parts render with at most 2 worker processes; a full clean render is ~2 min of parts + ~3 min mix/master/analysis on an idle machine (much longer when the picture departments load all 4 cores). Everything is seeded (CRC32 of the part name) and reproducible.

## How it is built (`music/src/`)
* `dsl.py` — score DSL: global beats (`gb(bar, beat)`, `fb(frame)`; 1 beat = 20 frames = 40 000 samples), `Part`/`Note` (pitch, start, dur, velocity, articulation, legato, `sync`), per-part dynamics curves (CC1-style), melody strings (`"D4:1 A4:1 D5:2"`).
* `score.py` — **the composition** (all notes, dynamics, orchestration, seating, faders, and the `BREATHS` list).
* `sampler.py` — VSCO-2-CE sampler: SFZ key/velocity/round-robin maps, equal-power velocity-layer crossfades (static or following the dynamics curve), round-robin rotation plus micro detune/gain/offset variation on repeats, soxr pitch-shift+resample in one pass (shifts stay a few semitones), detected sample onsets aligned to the note time (hits land on the frame), anticipation for slow-speaking sustains, crossfaded sustain extension for long notes, release envelopes, attack-skipping legato. Measured tuning fixes: glockenspiel samples sound an octave above their SFZ keys (corrected), timpani principal tones were 42–78 cents off their mapped keys (corrected per drum), sustain patches get their measured fine-tune (±15–45 c) from `check_samples.py`.
* `synth.py` — what the library lacks: formant choir (8 detuned glottal-wavetable voices per note, vibrato/drift/jitter/breath, "ah/oh/oo" formant banks per register), additive pipe organ (principal chorus + pedal ranks, chiff, wind), taiko ensemble (pitched membranes + skins, 2–3 players per hit), FM glass bells / celesta ("thinking" arpeggios), noise risers, Shepard tone, reversed cymbal, sample-based tremolo glissandi, hybrid impact, sub drop, detuned-saw BRAAM, sub sine, clock tick, church bell.
* `sfx.py` — all SFX synthesised: gusty/howling wind with spindrift hiss, torch crackle (Poisson crackles + flame breath), ember whoosh, ignition whoomps and roars, storm roar, steel-on-flint strikes (scrape + ring + spark ticks), kindling catch, desert/ice/jungle/city/sea ambiences (stick-slip ice and hull creaks, insects, hum, swell), converging torches, hearth flare, cloth rustle, distant whumps.
* `mix.py` / `render.py` / `master.py` — orchestral seating (violins L, violas L-C, cellos C-R, basses R, horns L-C rear, trumpets C-R rear, trombones/tuba R rear, timpani C rear, choir wide), distance EQ, one synthesized true-stereo hall IR (early reflections + band-wise exponential tail, RT60 ≈ 3.1 s, decorrelated), per-part sends; SFX get a short outdoor space plus a little hall. **Breaths**: 200 ms before the RACE entrance (26.667 s), the suck 1037→1040, and 275 ms before the CLIMAX (93.333 s) every part *and the reverb tail* are ducked to −45 dB — only the reversed cymbal rushes in — then the downbeat lands on the frame. Master: stem-linked glue compression (1.5:1), 4× oversampled true-peak look-ahead limiter, loudness to −16 LUFS, 24-bit TPDF dither, exact length, fades from/to silence.
* `analyze.py` / `calibrate.py` / `check_samples.py` — my ears: onset measurement at every sync point (2 ms dB-slope detector), LUFS per section (integrated + short-term), true peak, L/R correlation and mono fold-down, DC, click detector, spectrogram per section (`analysis/spec_*.png`), `overview.png`, `sync.png`; per-part level calibration for the faders; per-sample FFT pitch audit.

## The music
72 BPM, 4/4, D minor → D major. The **Beacon theme** (call D4–A4–D5, falling answer) as briefed.
* **INTRO** — wind + low D drone from silence (cellos/basses + sub); string pad Dm – B♭/D – F/D – C/D over the pedal; solo flute plays A1–A4 from bar 2; harp and a distant bell now and then.
* **KINDLING** — 7-note "thinking" cycles on a 16th grid (FM glass + glock), so the accents drift against the bar; accelerating to sextuplets/32nds and converging to one point → **IGNITION (480)**: B♭maj9(♯11)/D (Lydian: wonder with unease), hit + gong + glass bloom; the chord breathes; bar 8 = A7(♭9), the prize.
* **RACE** — breath, then a big first hit; taiko/giant drum/bass drum lock every beat (tower surges), 16th tenor drums, spiccato ostinati (3+3+2 violas), clock ticks; D pedal Dm | E♭/D; the **corrupted call D–A♭–D** in horns/trombones/tuba, trumpets join at the storm (800) with a chromatic string rise and brass clusters.
* **GRASP** — tremolo glissandi, Shepard tone, cluster crescendo, reversed cymbal; everything cuts at 1037 (suck) → **IMPACT (1040)**: impact + sub drop + BRAAM on D/A♭ + gong/bass drum/timpani, then silence.
* **SILENCE** — piano: D4 … A4 … D5, then the low fifth D2+A2 joins under "Together"; damped as the flame catches.
* **FIRST BEACON** — breathing low D; at 1318 a B♭/D chord blooms; B♭/C with a timpani roll; **1360** F major, harp sweep, solo horn call F–C–F + answer.
* **BEACONS** — harmony changes *on each ignition* (accelerating harmonic rhythm), bass rising F–A–B♭–C–D–E–F; each ignition = the call in a new voice (horns, trumpet, flute+glock, cellos at the call's original pitch, trombones, high violins + solo violin), overlapping as a stretto.
* **GLOBE** — choir enters on F with organ pedal; F – B♭/F – C/F; tiny calls in celesta/glock scattered across the stereo field (the web spreading).
* **ACCORD** — hush on D minor (clarinet intones the call); the four oaths each light a new chord (B♭/D, C, F/A, G/A) with timpani + low bell + choir swell while the top line climbs F5–G5–A5–B5; A pedal; the "together" ring = harp glissando panned L→R; C♯6 held over A.
* **DAWN** — breath, then D MAJOR on 93.333 s: cymbals, gong, timpani, bass drum, impact, low brass, organ pedal, choir (two layers, the widest image of the film); the theme in trumpets/violins/horns: call – answer (Bm, D/F♯) – third phrase (G, then G minor/B♭: the old sorrow remembered) resolving to D.
* **CODA** — solo flute call; the D5 hangs alone through "Who lit the first one?"; the answer (C♯–B–F♯) during the silent hand-over; 2640 warm swell; 12 celesta/glock bells for the answering fires (2645–2716) spreading outward L/R and receding; final D add9 at 2720, faded to silence by 117.0 s.

## Notes for the editor
* Quietest stretches (room for titles/breath): intro (~−24 LUFS), the silence (piano ≈ −33 LUFS short-term), first beacon before 1318 (≈ −38 LUFS short-term), coda (~−23 LUFS). The DAWN is the loudest section.
* Most important SFX: flint strikes 1236/1262/1290, kindling catch 1318, beacon roar 1360, the six montage ignitions, torches merging (clip 2020–2080), hearth flare into 2240, beacon catch 2640 and the answering-fire whumps (clips are frame-exact; nudge freely if picture moves).
* The answering-fire bells (2645–2716) are in the score stem; if HILLS' fire timing differs, the whumps in `sfx_events` can follow picture, and the bell times live in `score.py` (`coda()`, `times`).
* The storm roar SFX cuts dead at 1037 to respect the suck before the IMPACT.
