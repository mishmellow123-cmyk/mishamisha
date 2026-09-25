# THE LONG DAWN — score & sound cue sheet

Read `../BIBLE.md` first (story, time grid, sync table). This is the detailed
brief for the music and sound department.

## Format
* 48 kHz, 24-bit, stereo. Exactly **117.000 s** long (5 616 000 samples).
* 72 BPM, 4/4, no tempo changes. Beat = 0.8333 s = 20 video frames. Bar *b*
  starts at (b−1)·3.3333 s. Frame *f* is at f/24 s.
* Deliver to `music/out/`:
  - `score.wav` — music only
  - `sfx.wav` — sound effects/ambience only
  - `mix.wav` — score + sfx at the intended balance, mastered
  - `sfx_events/*.wav` + `sfx_events.json` (`[{"name","file","frame","gain_db"}]`)
    so the editor can nudge any effect if picture timing moves
  - `NOTES.md` — how it was made, how to re-render, the actual hit times
* Loudness: full mix ≈ **−16 LUFS integrated**, true peak ≤ **−1.0 dBTP**. Keep
  the dynamics cinematic: the quiet parts must be genuinely quiet (the SILENCE
  section around −35 LUFS short-term), the DAWN the loudest moment of the film.

## The theme ("the Beacon theme")
A simple, singable, hymn-like melody with a folk/Celtic-Nordic plainness. Its
identity is the **call**: a rising perfect fifth then fourth to the octave
(D4–A4–D5 in D), like a horn call across mountains, followed by a gently
falling **answer**. Suggested first statement in D minor/Dorian (quarter = 1):

```
bar A1 (Dm):  D4 1   A4 1   D5 2
bar A2 (Bb):  D5 1   C5 .5  Bb4 .5  F4 2
bar A3 (F):   F4 .5  G4 .5  A4 1    C5 1   A4 1
bar A4 (C):   G4 1.5 F4 .5  E4 2
bar A5 (Dm):  D4 1   A4 1   D5 2
bar A6 (C/E): E5 1   D5 .5  C5 .5   A4 2
bar A7 (Bb):  Bb4 1  A4 .5  G4 .5   F4 1   G4 1
bar A8 (A→Dm): E4 2         D4 2
```
Improve it if you can make it more beautiful and memorable — but keep the
call (rising 5th + 4th) because it is used symbolically:
* in the RACE the call is corrupted: the perfect fifth becomes a **tritone**
  (D–G♯/A♭–D) in low brass — the call turned into an alarm;
* in the BEACONS each beacon ignition is answered by **the call in a new voice**;
* at the DAWN it returns in **D major**, whole and triumphant.

## Section by section

| bars | time (s) | frames | picture | music |
|------|----------|--------|---------|-------|
| 1–4 | 0.00–13.33 | 0–319 | the hill at dusk, far future | From silence: wind (sfx), a low warm D drone (cellos/basses + sub), soft string pad. Bar 2 (3.33 s): **solo flute** plays the theme A1–A4, intimate, a little breathy, with reverb like an open valley. Harmony Dm – Bb – F – C. A distant bell or harp note now and then. |
| 5–8 | 13.33–26.67 | 320–639 | glyphs converge → IGNITION at 480 (20.00 s) → towers → crown | A crystalline, glassy **arpeggio shimmer** (celesta/glock/FM-bell, 16ths) enters — the sound of *thinking*. Strings swell. Dm – Bb, a riser into **20.00 s IGNITION** (reverse cymbal into a deep, bright, awed hit; the chord opens: e.g. Gm(add9) or Bb(add9)/D — wonder with unease). Bars 7–8 build toward the dominant (A), uneasy. |
| 9–12 | 26.67–40.00 | 640–959 | RACE: towers surge on every beat; walls; red; storm from 33.33 s | **Drums enter at 26.67 s** (taiko/low toms/bass drum ensemble: driving 16ths with accents on every beat — each accent is a tower-surge on screen). Spiccato string ostinato on D. Harmony D pedal: Dm – E♭/D – Dm – E♭/D (Phrygian menace). High ticking (clock-like) in 16ths = time pressure. The **corrupted call** (D–A♭–D) in horns/trombones. From 33.33 s everything intensifies: brass clusters, rising chromatic line. |
| 13 | 40.00–43.33 | 960–1039 | GRASP: a hand of embers closes on the crown | Riser: string tremolo glissandi + Shepard-tone + brass crescendo + reversed cymbal, all funnelling into… |
| — | **43.33** | **1040** | cut to black | **IMPACT**: huge low hit + sub drop + brass BRAAM, then **silence** as the tail decays over ~3 s. |
| 14–15 | 43.33–50.00 | 1040–1199 | black; text "Then even its makers said: Slow us down. All of us. Together." | **Solo piano**, very sparse, very quiet: fragments of the call (D4 … A4 … D5), long gaps, soft sustain pedal, the last note left hanging. Almost nothing else. |
| 16 | 50.00–53.33 | 1200–1279 | darkness; flint strikes at 1236, 1262, 1290 (51.50, 52.58, 53.75 s) | Near silence; cold wind (sfx). The flint strikes are sfx. Maybe the lowest string pad breathing in underneath. |
| 17 | 53.33–56.67 | 1280–1359 | kindling catches at 1318 (54.92 s) | A soft string chord blooms with the small flame (Bb major colour), expectant. Timpani soft roll building into… |
| 18 | 56.67–60.00 | 1360–1439 | **the beacon ROARS at 1360 (56.67 s)**; camera reveals the peaks | Swell into **F major**; **solo horn** sings the call + answer (F–C–F … or keep D–A–D over a B♭ chord — your call), noble, lonely, hopeful. |
| 19–22 | 60.00–73.33 | 1440–1759 | BEACONS montage; ignitions at 1480, 1540, 1600, 1650, 1690, 1730 (61.67, 64.17, 66.67, 68.75, 70.42, 72.08 s) | Strings ostinato in 8ths (major), timpani & snare building, momentum. **At each ignition the call sounds in a new voice**: horn → trumpet → flute+glock → cellos → trombones → high violins, stacking and accelerating. F – C/E – Dm – B♭ …, lifting harmonically toward… |
| 23–24 | 73.33–80.00 | 1760–1919 | globe at night; the golden web spreads | **Choir enters** ("aah", wide), organ pedal, strings sustain; broad, growing. |
| 25–28 | 80.00–93.33 | 1920–2239 | the ACCORD; oaths glow at 2000, 2040, 2080, 2120 (83.33, 85.00, 86.67, 88.33 s); the "together" ring sweeps 2160–2220 (90.0–92.5 s); flare to white 2225–2240 | The long build. Each oath gets an accent (timpani + low bell + choir swell). The ring sweep = harp glissando / shimmering bells. Bars 27–28 sit on an **A (dominant) pedal** — maximum anticipation — cymbal swell + snare roll into… |
| 29–31 | **93.33**–103.33 | 2240–2479 | **CLIMAX: the sun breaks over Earth's limb at 2240**; text "It did not end our differences. It ended the race." | **D MAJOR.** The same tonic, transformed (the world is the same world, in new light). Full orchestra + organ + choir sing the theme whole, warm, noble, not bombastic — the moment people cry. D – A/C♯ – Bm – G – D/F♯ – G – A – D (or better). Huge cymbal + timpani + choir "AAH" exactly on 93.333 s. |
| 32–33 | 103.33–110.00 | 2480–2639 | CODA: the hill; "Who lit the first one?" (104.2–107.1 s); the Elder silently hands over the torch (107.5–109.2 s) | Diminuendo to intimacy: **solo flute** plays the call + answer in D major over a soft string pad; at the question the music thins to a single held note — the silence of the unanswered question. |
| 34 | 110.00–113.33 | 2640–2719 | the child lights the beacon at 2640 (110.00 s); answering fires across the hills 2645–2720 | A warm swell (strings + choir pp) and a rising cascade of soft bells/celesta/glock — one note per answering fire, spreading across the stereo field like the fires spreading across the hills. |
| 35 | 113.33–117.00 | 2720–2807 | title THE LONG DAWN; fade to black | Final chord **D major add9** (D–A–E–F♯), ringing, fading to silence by 117.0 s. |

## Instruments & production
* Samples: **VSCO-2 Community Edition** (CC0) — `git clone --filter=blob:none
  --sparse https://github.com/sgossner/VSCO-2-CE` then `git sparse-checkout set
  <folders>` for only what you need (Strings: Violin/Viola/Cello sections, Solo
  Contrabass, Harp; Brass: F Horn, Trumpet, Tenor Trombone, Tuba; Woodwinds:
  Flute; Keys: Upright Piano, Organ; Percussion: Timpani, Glock, bass drums…).
  The repo's `SFZ` branch has the SFZ mappings (key ranges, velocity layers,
  round-robins) — use them to write a small sampler in Python (resampling for
  in-between notes, velocity crossfades, release tails, round-robin rotation).
* Synthesize what the library lacks, carefully: choir pads (many detuned formant
  voices, vowel "ah"/"oh", vibrato, breath), organ if the samples are thin
  (additive 16'/8'/4'/2' ranks), taiko/tom ensemble (layered pitched bodies +
  skins + room), risers, impacts, sub drops, the glassy "thinking" arpeggio (FM
  bells), BRAAMs (layered detuned brass + saturation).
* Make it breathe like a real orchestra: humanised timing (±8–15 ms), velocity
  and dynamics curves (crescendi, swells, phrase shaping), legato overlaps,
  release tails, orchestral seating (violins L, violas L-C, cellos C-R, basses R,
  horns L-C rear, trumpets C-R rear, trombones/tuba R rear, timpani C rear,
  choir wide), one convincing hall reverb (convolution with a synthesized IR:
  early reflections + ~2.8–3.5 s decay, stereo-decorrelated) with more
  send on distant sections. Gentle bus compression, a clean limiter.
* **You cannot listen**, so verify by analysis: render and look at
  spectrograms per section, onset/RMS envelopes against the sync table (hits
  must land within ±10 ms), check chord/note lists for wrong notes, check
  every sample's pitch (FFT) against its mapped note, check phase/mono
  compatibility, clipping, LUFS per section. Treat these checks as your ears.

## Sound effects (separate stem)
All synthesized (noise shaping, filtered impulses, resonators). Keep them
cinematic and restrained — the music leads.
* wind: gentle hilltop wind 0–360 and 2460–2807; howling cold summit wind
  with spindrift hiss 1200–1440; ambiences under each montage shot (desert
  wind 1520–1579; icy wind + distant ice creak 1580–1639; night jungle hush
  1640–1679; faint city hum 1680–1719; sea swell, hull creak 1720–1767).
* torch crackle close on the Elder's torch 0–340 and 2480–2640.
* ember whoosh rising into the fire 290–345.
* ignition whoomp at 480 (layer with music).
* storm roar building 800–1040; the impact at 1040 is shared with the score.
* flint strikes 1236, 1262, 1290 (steel on flint: sharp scrape + spark tick).
* kindling catch 1318 (crackle + soft breath of flame).
* beacon roar 1360 (whoosh + roar, then sustained crackle under the reveal).
* montage ignitions 1480, 1540, 1600, 1650, 1690, 1730 (varied whoosh + roar).
* torches merging 2020–2060 (several whooshes converging into one roar).
* hearth flare 2225–2240 (rising whoosh into the climax).
* torch handover rustle 2580–2620; beacon catch 2636–2640; very soft distant
  whumps for answering fires 2645–2720, panned across the field.
