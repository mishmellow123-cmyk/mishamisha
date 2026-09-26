# THE LONG DAWN v2: tone & writing review ("the cringe detector")

**Date:** 26 Sep 2026, 09:10.

**What I reviewed:**
- The masters `out/the_long_dawn_A.mp4`, `_B.mp4` and `_C.mp4` (C was assembled at 08:48). I made 1-fps contact sheets of all three, denser sheets around every line of text, and full-res crops.
- `edit/titles.py`.
- `BIBLE_V2.md` §1–3.
- The music docs: `music/NOTES_v2.md`, `analysis_AB/` and `analysis_C/report.txt`, both `roll.png`, and `compare_AB_C_v1.png`.

**Limits:**
- **I can't hear.** The music notes judge intent and design as the notes and measurements document them. A listening pass should confirm them.
- **Known placeholders are not scored.** These are the Beacon Run, the heroine close-ups, C's DAWN and the desert/karst/city shots. Where I mention one, the note is preventive and aimed at the lane doing the rebuild.

**Evidence:** `review/critic_tone_evidence.jpg` has labelled frames for every picture note rated BLOCKER or MAJOR.

---

## Verdict per cut

**A (The Allegory) commits to its identity.**
- The chain of lines is precise and mostly excellent: the race logic, then "none of them could stop alone", then "someone else", then the oaths. It fixes v1's problem of casting the builders as heroes.
- It fails at three points:
  1. The frame card can't be read.
  2. The resolution line "It ended the race." is set bold, like a slogan, and its pronoun is broken.
  3. The climax image is the stock "connected planet" of a telecom ad: a gold mesh of nodes and arcs over the Earth at sunrise, under a star-filter sun.

**B (The Legend) commits, and the story reads without words.**
- It reads through RACE → SILENCE → FIRST BEACON → the answering fires.
- It is the cut most exposed by the picture problems, because nothing else carries it:
  - The network globe and dawn. Without words, the web can even read as the thinking fire spreading across the world.
  - The coda, where the elder and child look like dolls.
  - The glitter ending.
- It loses the film's one reveal, that the elder lit the first beacon, because the child's question is gone. It needs a wordless cue (m4).

**C (The Tolkien Cut): the pictures are a grown-up homage.**
- What works: the ink Himalaya catching fire, the Eye, and the Ring lying small in the council fire. It also has the best line in the project ("…This one could not be unmade.").
- The words tip it toward fan fiction in two places:
  - the meme quote, captioned before the Ring even exists;
  - "the free peoples", which reads as a political side-taking.
- The music's Celtic-flute coda is Shire pastiche.
- DAWN_C's eagles are the next trap; rules for them are in M12.

---

## BLOCKERS

### B1: The elder and the child are jointed wooden dolls
**Cuts and frames:** all cuts. Coda 2656–2800 (worst at 2688–2784, the torch handover); intro 160–300.

**What's wrong.** In medium shot the figures read as articulated mannequins (evidence: A 2770):
- The rim light outlines every limb segment: upper arm, forearm, mitten, the child's cylinder arms and block legs.
- The elder's lit profile is a smooth, featureless head.
- The scarf is a rigid plank.
- The cairn stones are rounded pillows.

**Why it matters.**
- "Characters that look like clay dolls in close-up" is on the blacklist in so many words.
- v2 fixed only the heroine (1236–1375).
- These two figures carry the film's first image and its last human gesture, in all three cuts, directly under "Who lit the first one?".

**Fix:** HILLS department, `shots/hills/characters.py` and `coda.py`, output to `renders/hills_v2/` (src 2496–2640 and 120–300).
- Shade each figure as **one merged silhouette**. Union the primitives *before* the bevel/sheen pass, so rim light exists only on the outer contour and never on internal joints. In the coda medium shots, lower `sheen` for coat/wool/skin from 1.0–1.3 to ≤ 0.4.
- Keep the faces unlit: at most a 1–2 px edge on the torch side. No lit mannequin profile at 2760–2784.
- Make the scarf cloth: a tapered, fluttering tail with secondary motion instead of a plank. Let the coat hems move.
- Merge the child's arm into the silhouette of sleeve and coat, so no cylinder segments show.
- Rebuild the cairn as irregular, flat-bedded dry stone with hard edges. The dry-stone cairn in MOUNTAIN's `bl/beacon.py` is the reference.
- If none of that can land in time: pull the coda medium shot back so the figures are ≤ 40% of frame height, and backlight harder. That gives true shadow-play (the *Three Brothers* reference), where puppetry reads as a deliberate choice.

### B2: "Connected planet" imagery where there should be fire
**Cuts and frames:** A and B at 1990–2087 (the globe's web) and 2400–2655 (the dawn); C at 2020–2087 (the map).

**What's wrong.**
- The answering world ends as a uniform lattice of nodes and great-circle arcs.
- The same mesh stays printed over the sunlit Earth at the climax, under an 8-ray diffraction starburst.
- C's map ends as a Delaunay triangulation over every landmass, Sahara and Amazon included.
- The torch rivers at 2088–2112 continue the look of radial lines.
- Evidence: A 2050, A 2500, C 2070.

**Why it matters.**
- Glowing arcs between nodes on a night Earth, plus a star-filter sun, is the stock opening shot of telecom, consulting and "global AI" brand films. That is the corporate register the cringe list exists to prevent, and here it is the climax.
- It is also the visual grammar of a network graph. With no words (cut B), the web can read as the thinking fire itself spreading over the world, which is the opposite of the story.
- The MAP lane already flagged its own ending as "a bit network diagram".

**Fix:**
- **GLOBE, world answers.** In `shots/globe/web.py`, output to `renders/globe_v2/` and `globe_B/` (src 1760–1927):
  - Make each link a travelling ember: a bright head with a short fading tail, gone about 1 s after it arrives. No arcs that persist.
  - Make the nodes small flickering fires of uneven size, each with a brief flare when it ignites, clustered along mountain ranges, coasts and rivers.
  - By 2072 the night side should be studded with fire, not meshed.
- **GLOBE, dawn.** Src 2240–2495:
  - No arcs at all.
  - The night-side fires pale as the terminator reaches them: the sun takes over from the beacons.
  - Replace the 8-ray starburst with a clean disc glare and a restrained anamorphic streak.
- **MAP.** In `shots/map/webmap.py`, output to `renders/map_C/` (2010–2087):
  - Draw no new edges after about 2010.
  - Burnt edges fade to brown scorch lines in the parchment.
  - The end state is flame glyphs of varying size on the hatched ranges and coasts, with scorch trails between them. No lattice. Keep the compass rose.
- **MOUNTAIN.** Apply the same rule in `shots/mountain/post_run.py` (dawn): drop the planned 8-ray starburst from DAWN_C.
- **Option for B only:** use DAWN_C without the eagles as B's dawn, which keeps the legend on the ground.

---

## MAJOR

### M1: "It ended the race." (A, 2512–2600)
**What's wrong.**
- The narration hasn't had a subject for about 45 s; its last line was at 1435. Meanwhile the last "IT" on screen is the fire, in the oaths ("NO SINGLE HAND SHALL HOLD IT", "WHAT IT GIVES, IT GIVES TO ALL"). So the literal reading is "the fire ended the race", and "ended the race" rings as "ended the *human* race".
- It is set at weight 600, bolder than every other line, over the sunrise and the loudest music in the film. That makes it the victory slogan the blacklist bans.

**Fix** (`edit/titles.py`): **"And the race was over."** at 2512–2600, weight 560. This is the storyteller's voice, with no agent to misread and no victory in it. Fallback: "The race was over."

Optional: move it to 2300–2372, as the camera pulls up from the carved oaths (the race ends *at the table*), so the sunrise plays wordless. Do this only if the Accord's musical build leaves room for it.

### M2: The thinking fire is a hanging light bulb (all cuts, 490–600)
**What's wrong.**
- The shape is a round glowing core at the bottom, a parallel-sided neck of particles rising out of the top of frame, and squiggly white filaments inside. That is a pendant incandescent bulb.
- In A and C it is captioned "And it began to think." / "A fire that could think.", which turns it into the cartoon "bright idea" icon, as a pun. Evidence: A 520.

**Why it matters:** this is the protagonist's first appearance. A laugh here costs the whole first act its awe.

**Fix:** EMBERS department, `renders/embers_v2/`, `embers_B/` and `embers_C/` (src 480–600).
- Frame the whole flame with its **tip inside the frame**.
- Taper the plume into 2–3 licking tongues, asymmetric and flickering.
- Move the hot core up into the body of the flame instead of a sphere at its base.
- Let the filaments grow out past the core, so there is no edge that reads as a glass envelope.
- Cheapest version: pull the camera 20–25% further back, or tilt it, so the tip is visible.

### M3: "One ring to rule them all." (C, 565–635)
**What's wrong.**
- It lands at 565 on the white thinking fire, but the gold band only forms at 600–630 (evidence: C 580).
- It is the most-memed line in the canon, and "one model to rule them all" is a stock tech/AI headline. The audience that reads every beat will hear a pun.
- Lowercase "ring" contradicts "the Ring" at 1060. It is also set at weight 600.

**Fix** (`edit/titles.py`): **cut it.** The burning band with its inscription *is* the quotation. The sequence "A fire that could think." → [the Ring forms] → "And every kingdom wanted it." reads perfectly.

If the producers keep the quote: "One Ring to rule them all." at 628–695, weight 560, and move "And every kingdom wanted it." to 705–770.

### M4: "And the free peoples answered." (C, 1935–2040)
**What's wrong.**
- In our world, "free peoples" reads as "the free world": the democracies answered and the autocracies didn't.
- It comes after "…our enemies will." and lands over fire running across China and Central Asia. That turns the allegory into the us-versus-them race logic the film argues against.
- It contradicts the map, where every continent lights up.
- It also lands early: at 1945 only one spark is on screen.

**Fix:** **"And all the peoples answered."** at 1960–2040. It keeps Tolkien's word "peoples" and sets up a real contrast: *every kingdom wanted it; all the peoples answered.*

### M5: "the others" / "our enemies" lands on a globe of East Asia (A and C in text; B by picture; 884–912)
**What's wrong.**
- "Each said: if we stop, the others won't." (A) and "…if we do not take it, our enemies will." (C) are still on screen when the ember globe arrives at about 882.
- The globe faces China, Korea, Japan, Southeast Asia and Australia, so the caption points at one region for about 1 s. Evidence: C 890.

**Why it matters:** the film would appear to name the enemy.

**Fix:**
- In the edit: run both lines at **800–866**, so they are gone by about 878 and sit only on the storm, the vortex and the Eye.
- Better fix, from EMBERS: turn the cracking globe to a polar or Atlantic-centred view, or spin it ≥ 120° during 882–960, so no single region reads as "the one breaking".

### M6: The ending, where glitter stands in for fires and the title cuts through the flame (all cuts, 2805–2880)
**What's wrong.**
- The answering fires read as a band of golden bokeh, or fairy dust, across the valley (2816–2846).
- The title's out-of-focus gathering embers add more of the same.
- The letters form while the elder, the child and the blazing beacon are still in frame, and the flame cuts through "G D" at 2846–2866. Evidence: A 2856.

**Why it matters:** the payoff (the child's fire answered across the land, rhyming with the Beacon Run) gets less than 1.5 s clean before cute sparkle and a title cover it.

**Fix:**
- **HILLS** (`coda.py`, src 2645–2720): show 12–20 distinct small fires on successive ridgelines, each a flame with its own glow and a wisp of smoke, igniting outward in a wave. No field of bokeh.
- **TITLE** (`edit/ember_title.py`): keep the sync to the chord at 2880, but:
  - cut the out-of-focus gathering embers by about 70% and make them rise visibly *from the answering fires*;
  - keep the letters clear of the flame, either by raising the title about 90 px (y ≈ 312) or by having HILLS start the crane-up 16 frames earlier, so the beacon is below the band of letters by 2845.

### M7: A's frame card can't be read (A, 4–82)
**What's wrong.**
- The card is 88 characters. Both lines are fully legible only at **46–66 (0.8 s)**, and legible at all only at 18–82.
- From frame 56 the hill fades up under the card, and line 2 runs across the cairn and the figures. Evidence: A 72.

**Why it matters.** The card is what makes A an allegory. If "about the years we are living through" isn't read, the explicit mapping loses its key.

The words themselves are right, so keep them. "Might" does real work: the future is conditional.

**Fix** (`edit/titles.py`, cut A):
- Place the card lines at y = 600 and 658, on the dark hillside below the figures. Size 48, `t_in` 4 and 12, fade 18, both `t_out` 128 (gone by 144).
- Move "Why do we light the fires?" to 150–212 and "So we remember how close we came." to 226–312.
- Leave the picture fade in `assemble.py` (which starts at 56) unchanged.

### M8: Score AB still turns into organ-led, major-key uplift in the second half (A and B, 1360–2623)
Judged from `NOTES_v2.md` and the analysis files, not by ear.

**What's wrong.**
- **1360.** The organ enters at the first beacon, with horns a2 on the call. That is the same pivot v1 made.
- **The Beacon Run (1520–1679).**
  - It is a rising F-major scale (F G A B♭ C D E F), with a deep hit on every step and violins and horns in parallel tenths above it. That is literally uplift by numbers.
  - Each hit also doubles the SFX whoomp on the same frame.
- **The dawn (2400–2623).**
  - Organ with 16′ pedal, horns a4, violins and cellos sing the call broadly in D major.
  - It is the loudest moment of the film: short-term max −10.4 LUFS, against −13.5 for the race.
  - That is the same level as v1's rejected dawn (see `compare_AB_C_v1.png`).

**Why it matters:** the producers' fourth note was exactly this: "from the first beacon on … a traditional triumphal hymn … organ … victory at the dawn". Without the choir, glockenspiel and cymbals, it is still the same arc.

**Fix** (`music/src/score_v2.py`):
- **1360:** strings and low brass on B♭, with no organ manuals. A 16′ pedal as a floor at most.
- **1520–1679:**
  - No diatonic scale climb and no parallel tenths. Hold a pedal and move the harmony by thirds, which is the montage's own logic.
  - Write the horns as a near/far call and response.
  - Drop the timpani and bass-drum hit on each beacon, and let the SFX whoomps own the beacons.
- **2400:**
  - The awe should come from harmony, not melody. Two string sections plus the thinking-fire cycle glowing in D add9, sustained for the first ~70 frames with no tune.
  - The call enters late (≥ 2470) on one voice: solo horn, or cellos in unison, at mf.
  - No full organ manual.
  - Peak the section at or below the race (about −13 LUFS short-term).

### M9: The Celtic-flute coda (C, 2624–2967)
**What's wrong.** `score_v2_c.py` §CODA specifies "a Celtic-leaning flute (cut, tap, turn) over a soft D–A drone" and calls the drone "the pipes' drone".

**Why it matters.**
- An ornamented whistle over a drone is the most recognisable, and most parodied, signature of Shore's Shire music.
- Even with an original tune it reads as pastiche, which is fan fiction in sound, at the film's most intimate moment.
- The hill is our future, not the Shire.

**Fix:** use the intro's plain flute, which is the film's own voice, with no grace notes and no drone. Alternatively, give the call to one distant solo horn, which is C's beacon voice. Keep the distant horns answering the fires and the violin rising to the ninth.

### M10: Emissaries that read as pawns, and a frieze that reads as a seal (all cuts, 2160–2399)
**What's wrong.**
- **The emissaries.** Seen from directly above, they are rounded hood-caps in rose, olive and tan, holding cream peg-torches around a round board. They read as game pieces. Evidence: A 2255 and 2385.
- **The frieze.** Once lit (2345–2399), its 48 gold three-tongued fires read as a printed laurel wreath around a medallion: an institutional emblem.

**Why it matters.**
- v1's "pastel toy-like emissaries" note is only half fixed.
- The emblem sits close to unity-poster imagery, one step away from the "together" ring it replaced.

**Fix:** ACCORD department, `geom.py` (cloth albedo) and `ring.py` (`frieze_lit`), output to `renders/accord_A`, `_B` and `_C` (src 1960–2247).
- **Emissaries:**
  - Hoods and cloaks near-black, desaturated into one charcoal-to-umber range. Vary weave and sheen, not colour.
  - Spent torches charred black, not cream.
  - More cloth overhang, so the shape from above isn't a round cup.
  - Heavier defocus at the frame edges during 2160–2300.
- **Frieze:**
  - Firelight in carved grooves: a dim orange ember glow, plus short real flame tongues at the head of the sweep that die down behind it.
  - Never a flat cream-gold fill, and never brighter than the hearth.

### M11: Poster-hero poses (all cuts; for the lanes doing rebuilds; 1360–1439, 1760–1799, 1840–1879)
**What's wrong.**
- The heroine placeholder raises a **thumbs-up fist** beside the roaring beacon (1360–1380). Then she stands with a rigid arm stretched out over the summit (1380–1439).
- The ice answerer (1776) and the city answerer (1870) hold their torches overhead in the Statue-of-Liberty or Olympic pose.

**Why it matters:** these are the "uplift" poses, and the thumbs-up would get a laugh.

**Fix** (HEROINE, MONTAGE-3D and MONTAGE lanes): working gestures only. Kneel, shield the face from the heat, thrust the torch into the kindling, step back. At the reveal the heroine simply stands and watches the far range, scarf whipping. No fist, no thumb, no torch held aloft.

### M12: DAWN_C's eagles (C, 2400–2655; rendering now, so this note is preventive)
**Why it matters.**
- The eagles are where C tips into fan fiction: "the Eagles are coming", and the eagles meme.
- In the allegory they mean rescue from above, which is the opposite of this film's thesis that we got ourselves through.
- The current plan (from the MOUNTAIN notes) is six great eagles crossing in front of the light at 2430–2560, plus an 8-ray starburst.

**Fix** (MOUNTAIN, `shot_dawn.py` and `post_run.py`):
- Keep the eagles far and small, with wingspan ≤ about 4% of frame width.
- Have them glide rather than flap.
- One pass only: never toward the camera, never across the sun disc.
- A loose, irregular group, not a formation.
- No starburst.

**Music** (`score_v2_c.py`): the "rising consequent for the eagles" should continue the phrase, not put an accent on their entrance.

---

## MINOR

- **m1: Lines bleed over cuts (A and C).**
  - A's "So someone else lit a beacon." and C's "And on a cold mountain, someone lit a beacon." end at `t_out` 1435 and fade until 1447, so they sit on the shepherd for 7 frames after the cut at 1440 (A 1442). "Someone" lands on the wrong person. Set `t_out` 1426.
  - The lines on black end at `t_out` 1190 and fade until 1202, so they ghost for 2 frames over the first beacon. Set `t_out` 1186.
- **m2: A's rhythm and landings.**
  - "So" opens three lines (215, 660, 1370), which makes the legend sound like a syllogism. Keep the first and last, which are structural. Change the middle one to **"The companies raced for it, then the countries."** at 668–738. This drops the tic and adds a true, specific beat: the labs raced first, then the states.
  - There is only a 3-frame gap between "And it began to think." (550, faded out by 562) and "Whoever held it alone…" (565), and the second line lands before the crown exists. Move it to **580–645**, so it completes as the crown forms at about 600.
- **m3: Winks in the glyph field (all cuts, 340–470).**
  - `shots/embers/glyphs.py` contains legible English words ('fire', 'light', 'dream', 'mind', 'We', 'Word') and 'GATTACA'.
  - At 396, "fire" floats directly above the caption's "fire".
  - In B these are the only readable English before the title.
  - Keep the scripts, letters and fragments. Remove the whole English words and GATTACA, and thin out the repeated "ACGT" clusters.
- **m4: B needs a wordless reveal.**
  - Without "Who lit the first one?", the handover is just the idiom "passing the torch" made literal, and the reveal (the elder is the young woman) rests on the scarf alone.
  - Fix: as the torch changes hands (2760–2780), add a faint echo of the three flint strikes from 1236/1262/1290 to both `sfx_events_AB.json` and `sfx_events_C.json`. Doing it in all cuts helps A and C too, and avoids needing a B-only SFX stem.
- **m5: Title type (A and B).**
  - Gold Cinzel caps with wide tracking is the epic-trailer poster face. It's right for C and less right for A and B.
  - Optional: set A and B's ember title in Cormorant roman caps, the storyteller's face. The re-render with `ember_title.py` takes about 65 s.
- **m6: Small music notes.**
  - **AB coda.** The seven swells for the answering fires each come in a new colour: horn, clarinet, viola, cello, flute, violins. That is the "theme handed round the orchestra" v1 was faulted for. Keep it in one family, strings or horns.
  - **The ring sweep (both scores, 2320–2380).** The rising violin scale Mickey-mouses the ornament. Make it a swell.
  - **2400.** The third suck-to-silence, followed by a timpani stroke on the sun, is trailer grammar. A bloom out of silence with no stroke would give awe rather than impact.
  - **C's run.** A horn signal on every beacon: keep beacons 1, 3, 5 and 7, and let the far ones be SFX only.
- **m7: The crown (A and B, 600–800).** Its uniform triangle points read as a cartoon crown or tiara. Make them irregular licking tongues (EMBERS).
- **m8: The first-beacon wide shot (all cuts, 1376–1439).** The terrain is smeared and stretched, and the heroine is large and rigid. If the first-beacon rebuild doesn't replace this shot, it should; the bible says "she is tiny".
- **m9: The shepherd's flame (1488–1519).** Its magenta tip reads as synthetic. Keep the fire ramp orange to yellow.
- **m10: C's "For the night the beacons were lit." (215–310).** For half a second it reads as a duration ("for the night") before it resolves. Keep it anyway: it is the right echo without being a quote.

---

## What works (protect it)

- **The embers act (300–1199).** Glyphs, ignition, towers, race, vortex, cracking world, grasp, black: it is coherent and beautiful, and it reads without words.
- **A's spine.**
  - "Each said: if we stop, the others won't." / "Many in the race wanted it to stop. / None of them could stop alone." / "So someone else lit a beacon." These lines are precise and humane, and they fix v1's heroic-builders problem.
  - "So we remember how close we came." is the best answer to the opening question the film has had.
- **The four oaths.** They are specific, carved, crisp and legible, and nothing in the music lands on them; the harmony moves off the beat. Keep them in A and C.
- **C's best material.**
  - "In the old story, the Ring was unmade in the fire that forged it. / This one could not be unmade." This is the sharpest line in the project; don't touch it.
  - The small Ring lying in the council fire.
  - The Eye, which follows Tolkien's own description.
  - The first half of the map (1920–2000): the ink Himalaya catching fire, with fire running out of the high mountains.
- **The sea shot (1880–1919):** restraint.
- **Music.**
  - No choir anywhere, and C's pianissimo brass chorale in place of one.
  - At 2160 in AB, every voice locks into one cycle.
  - In the world-answers section, the thinking fire's glass arpeggio turns warm note by note.
  - The breath before the race and the suck into the IMPACT.
- **The endings.** "Who lit the first one?" and the handover she leaves unanswered (A and C). The title crumbling back into embers.

---

## Line by line

**A (The Allegory)**

| v2 frames | line | verdict |
|---|---|---|
| 4–66 | *A story they might tell, a hundred years from now, / about the years we are living through.* | Keep the words; fix timing and placement (M7). |
| 120–200 | *Why do we light the fires?* | Keep. Move to 150–212 if the card is extended. |
| 215–310 | *So we remember how close we came.* | Keep. Move to 226–312. |
| 340–440 | *We taught a new kind of fire to read everything we had ever written.* | Keep. It is the most precise mapping line; long (69 characters) but readable in 100 frames. |
| 490–550 | *And it began to think.* | Keep; fix the bulb under it (M2). |
| 565–635 | *Whoever held it alone would hold the world.* | Keep; move to 580–645 so it lands on the crown (m2). |
| 660–730 | *So the companies and the countries raced.* | Change to *The companies raced for it, then the countries.* at 668–738 (m2). |
| 820–900 | *Each said: if we stop, the others won't.* | Keep; move to 800–866 (M5). |
| 1060–1190 | *Many in the race wanted it to stop. / None of them could stop alone.* | Keep; the best line in A. `t_out` 1186. |
| 1370–1435 | *So someone else lit a beacon.* | Keep; `t_out` 1426 (m1). |
| 2160–2280 | the four oaths (in the stone) | Keep. |
| 2512–2600 | *It ended the race.* | Change to *And the race was over.* at weight 560 (M1). |
| 2660–2730 | *Who lit the first one?* | Keep. |

**C (The Tolkien Cut)**

| v2 frames | line | verdict |
|---|---|---|
| 120–200 | *Why do we light the fires?* | Keep. |
| 215–310 | *For the night the beacons were lit.* | Keep (m10). |
| 340–440 | *From every word we had ever written, we forged a new power.* | Keep. "We forged" makes us the Ring-maker, which is the point. |
| 490–550 | *A fire that could think.* | Keep; fix the bulb (M2). |
| 565–635 | *One ring to rule them all.* | Cut it (M3). If kept: *One Ring to rule them all.* at 628–695, weight 560. |
| 660–730 | *And every kingdom wanted it.* | Keep. Move to 705–770 only if the quote stays. |
| 820–900 | *Each said: if we do not take it, our enemies will.* | Keep; move to 800–866 (M5). |
| 1060–1190 | *In the old story, the Ring was unmade in the fire that forged it. / This one could not be unmade.* | Keep; the sharpest line. `t_out` 1186. |
| 1370–1435 | *And on a cold mountain, someone lit a beacon.* | Keep; `t_out` 1426. |
| 1935–2040 | *And the free peoples answered.* | Change to *And all the peoples answered.* at 1960–2040 (M4). |
| 2160–2280 | the four vows | Keep. |
| 2660–2730 | *Who lit the first one?* | Keep. |

**B (The Legend):** the title only. Keep it that way; the fixes B needs are all in the picture and SFX (B1, B2, M2, M6, m3, m4).

---

## Top 5 changes, in order

1. **Fix the coda** (HILLS `characters.py` / `coda.py` and TITLE `ember_title.py`). The same character fix also covers the intro. (B1, M6)
   - Rebuild the elder and child as single merged silhouettes: no rim light on joints, a cloth scarf, unlit faces.
   - Show the answering fires as distinct fires on the ridgelines, not bokeh.
   - Keep the title out of the flame and cut its gathering glitter by about 70%.
2. **Fires, not graphs** (GLOBE `web.py`, MAP `webmap.py`, MOUNTAIN `post_run.py`). (B2)
   - No persistent arcs on the globe, and none at all on the dawn.
   - No Delaunay lattice on C's map.
   - No 8-ray starburst anywhere.
3. **One text pass in `edit/titles.py`.** (M1, M3, M4, M5, M7, m1, m2)
   - A: "And the race was over." at weight 560.
   - A: frame card at y = 600/658, held to 128, with the question and answer nudged later.
   - C: cut the Ring caption (or run "One Ring…" at 628–695, weight 560).
   - C: "And all the peoples answered." at 1960–2040.
   - A and C: "Each said…" at 800–866.
   - `t_out` 1186 and 1426.
   - A: "Whoever held it alone…" at 580–645, and "The companies raced for it, then the countries." at 668–738.
4. **EMBERS: make the thinking fire a flame, not a pendant bulb** (480–600, all three variants). In the same pass, strip the English words and GATTACA from `glyphs.py`. (M2, m3)
5. **MUSIC: take the hymn out of score AB and fix C's coda.** (M8, M9)
   - AB: no organ manuals at 1360 or 2400; no scale-in-tenths with a hit on every step in the run (the SFX owns the beacons); a sustained dawn no louder than the race.
   - C: replace the Celtic flute and drone with the plain intro flute.
