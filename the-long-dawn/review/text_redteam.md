# THE LONG DAWN v2: red-team text review, cuts A and C

26 Sep 2026. Fresh-eyes critic.

**Scope.** Every on-screen word in cuts A and C: `edit/titles.py` `story_lines()` at `f2a75f2`, plus the carved oaths.
**Config gate at start:** GATE PASS (`cloud/verify_session.py`).

**What I read:**
- `BIBLE_V2.md` §1–3 and `edit/titles.py`.
- `review_sheets/critic_tone.md` and `critic_framing.md`. I don't repeat their resolved points unless the fix is still weak.
- The picture under each line, from the committed sheets (`embers_v2`, `embers_C`, `map_C`, `coda_v2`, `cloud_globe_dawnA`, `critic_tone_evidence`) and from the source timings in `shots/embers/tolkien.py`, `scene_a.py`, `scene_b.py`, `timeline.py` and `shots/accord/`.

**Limits.** The v2 masters and `renders/` aren't in the repo. I judged the picture under the text from the sheets and code timings, not from motion. I can't hear the mix.

**How reading time was measured.** I ran every line, current and proposed, through `titles.Line` itself, so the fade-in, the 10-frame letter stagger and the fade-out are all accounted for.
- **Legible** runs from the first letters reaching 50% opacity to the last letters dropping below 50%. For these fades, that equals `t_out − t_in` to within a frame.
- **wps** = words ÷ legible seconds. Up to 3.5 is comfortable.
- **cps** (characters per second) is given where it matters; subtitle guides usually cap adult reading between 15 and 20.
- **px** is the line's width at its size, in the 1920-px frame.

Every alternative below was measured the same way and fits its slot. Frames are v2 frames at 24 fps.

## Summary

**A is nearly there.** Its chain of cause and effect is precise, fair to the builders, and never says "AI". Its ending refuses triumph. It has two problems:
- its longest line is too wide for the busiest background in the film;
- its admission that nobody can see inside the fire has no beat of its own.

**C has the best architecture in the project, and the weakest 12 seconds in either cut (565–846).**
- The architecture: old story → our difference → the turn → a new ending.
- The weak stretch packs a meme quote, a generic-fantasy "guild" and a verbless line together with no gap between them. As a result, "…our enemies." is still on screen while the Eye forms.
- Two more of C's lines can't be read in their slots.

Every TOP 5 fix is text and timing only: re-assemble, no department re-render.

---

## Cut A: THE ALLEGORY

| frames (v2) · measured | current line | verdict | alternatives (same slot) | why |
|---|---|---|---|---|
| 4–104<br>4.2 s · 11 w · 2.7 wps | *A story they might tell of us, a lifetime from now.* | **keep** | — | States the frame without a trailer voice. "Might" keeps it conditional, and "a lifetime" makes the elder's age add up. It runs on pure black for its first 3 s. Its last words sit over the dark cairn as the hill fades up, which does no harm. |
| 140–212<br>3.0 s · 6 w · 2.0 | *Why do we light the fires?* | **keep** | — | The child asks and the elder answers. That convention (questions belong to the child) is what makes the last line work. |
| 226–312<br>3.6 s · 6 w · 1.7 | *To remember how close we came.* | **keep** | — | Dread and hope in six plain words. With "So we remember" gone, the film's only "So" is the turn at 1370, and that is why the turn hits. |
| 340–440<br>4.2 s · 14 w · **3.4 wps · 1463 px** | *We taught a new kind of fire to read everything we had ever written.* | **tweak** | 1. *We fed a new kind of fire everything we had ever written.* (12 w · 2.9 · 1225 px)<br>2. *We taught a new kind of fire to read all we had written.* (13 w · 3.1 · 1193 px) | It is the widest line in the film (76% of frame, 68 characters), it sits at the reading limit, and its background is the busiest in the film: the drifting glyph field. "Fed" is the fire's own verb and describes the picture: the letters are drawn into the point that ignites at 480. It trades the schoolroom warmth of "taught… to read" for hunger, which makes the next beat colder: *fed… And it began to think.* |
| 490–565<br>3.1 s · 10 w · 3.2 | *And it began to think. No one could see how.* | **tweak (staging only)** | Same words, as two beats on one row:<br>`L += row(['And it began to think.', 'No one could see how.'], [490, 514], 565, y, size=S)` | Both sentences fade in together, so the dry second one gets no beat of its own and reads as a run-on. Delayed by 24 frames, it lands as the dread note that NO FASTER THAN WE CAN SEE answers. The second phrase then runs at 2.4 wps. |
| 580–648<br>2.8 s · 8 w · 2.8 | *Whoever held it alone would hold the world.* | **keep** | — | The race's real premise. In 2017 a head of state said publicly that "whoever becomes the leader in this sphere will become the ruler of the world", so this reads as the world's belief, not the film's invention. "Alone" is the word Oath I pays off. |
| 668–738<br>2.9 s · 8 w · 2.7 | *The companies raced for it, then the countries.* | **keep** | — | The one place the allegory touches ground, and it has to. It names both actors, casts no villain, and gets the order right. "For it" ties the race to the prize in the line before. |
| 800–866<br>2.8 s · 7 w · 2.5 | *Each said: if not us, someone worse.* | **keep** | — | Reported speech, and the argument racers actually believe. "Each" is what makes it fair: the most safety-minded lab and the most hawkish ministry say it alike. That includes the makers' own developers, whose public case for building is a version of this sentence. That is the "as yet undetermined" stance, neither flattery nor blame. The line is gone by 878, before the globe. |
| 1060–1186 + 1104–1186<br>(black) · 12 w · 2.3 | *Many in the race said it should slow.* / *None would slow alone.* | **keep words · tweak timing** | Line 2 `t_in` 1104 → **1114** | It reports what they said, and "would" makes it a choice inside a trap, neither helpless nor villainous. Line 1 needs about 2.3 s, but line 2 starts pulling the eye after 1.8 s. Starting line 2 at 1114 gives line 1 2.25 s and lets line 2 land as a verdict. |
| 1370–1426 (snow)<br>2.3 s · 6 w · 2.6 | *So someone else lit a beacon.* | **keep** | — | The hinge of the film. "Else" puts the turn outside the race, and "someone" keeps it anonymous so the last line can ask. Gone by 1438, before the cut to the shepherd. |
| 2160–2280 (carved) | NO SINGLE HAND SHALL HOLD IT · NO FASTER THAN WE CAN SEE · NO FORGE IN THE DARK · WHAT IT GIVES, IT GIVES TO ALL | **keep** | — | Each oath pays off a line: *held it alone* → I; *no one could see how* → II and III; *if not us* → IV (share the gains, and the reason to race goes). IV is the one close to mission-statement language. It works only because it is carved, unnarrated and unscored, so never caption it or voice it. |
| 2512–2600 (dawn)<br>3.7 s · 6 w · 1.6 | *In time, the race was over.* | **keep** | — | The right note: no agent and no victory. "In time" is the film's one admission that this was slow, and the title then confirms it. The line enters 4.7 s after the sun breaks, so the climax itself stays wordless. A residual risk, not worth a change: after nearly a minute without the word, a few viewers may briefly hear "the (human) race". |
| 2660–2730<br>2.9 s · 5 w · 1.7 | *Who lit the first one?* | **keep** | — | The right last line. By the film's own convention it is the child's question, and it is the one question left unanswered. The elder's silence and the handover keep it from becoming a lone-hero myth, and "first" asks the viewer too. No slogan, no end card. |

**A's arc and last line.** The shape is right:
1. The argument comes first, where the ideas are: five lines in 22 s (340–878).
2. The builders' couplet sits alone on black.
3. The hinge comes at 1370.
4. Then 45 s of picture, whose only words are the carved oaths.
5. Two short lines close it, refusing triumph.

It never says "AI". Three lines make the mapping unmistakable to anyone living through these years: *everything we had ever written*, *the companies… then the countries*, *if not us, someone worse*. Nothing flatters or accuses a lab or a nation. The builders are reported, not judged, and the turn comes from outside them.

The last line is the film's best-judged decision: keep it exactly, with no card after it.

One risk that is about layout, not wording: an italic line centred under the orbital sunrise is the keynote end-card layout. It disappears with the dawn changes already requested (tone B2, framing M5).

---

## Cut C: THE TOLKIEN CUT

| frames (v2) · measured | current line | verdict | alternatives (same slot) | why |
|---|---|---|---|---|
| 120–200<br>3.3 s · 6 w · 1.8 | *Why do we light the fires?* | **keep** | — | The same question as in A. |
| 215–310<br>4.0 s · 7 w · 1.8 | *For the night the beacons were lit.* | **tweak (minor)** | 1. *For the night when the beacons were lit.* (8 w · 2.0)<br>2. *To remember the night the beacons were lit.* (8 w · 2.0) | It echoes the famous beacon cry in the past tense without quoting it, which is right. But straight after "light the fires", "For the night" first reads as a duration ("to last the night"), and the reader has to back up. The tone review kept it (m10). One word, "when", removes the stumble and makes the line three clean anapests. |
| 340–440<br>4.2 s · 12 w · 2.9 · 1331 px | *From every word we had ever written, we forged a new power.* | **keep** | — | "Forged" and "power" are the Rings' own vocabulary. "Every word we had ever written" is the same specific tell as in A, and "we" makes us the Ring-makers. |
| 490–565<br>3.1 s · 11 w · **3.52 wps** | *A fire that could think, with a will of its own.* | **tweak** | 1. *A fire that could think, and had a will.* (9 w · 2.6 at 490–572)<br>2. *A fire that thought, and wanted.* (6 w · 1.8) | Keep the idea: it lets the Eye read as the fire's own will looking back, not as a nation. Drop the phrase, for three reasons. It is over the reading limit. It is a stock idiom (cars, hair and weather all have "a will of their own"). And it is how the 2001 film's prologue describes the Ring ("…has a will of its own"), which the bible's no-film-dialogue rule exists to prevent. |
| 628–695<br>2.8 s · 6 w · 2.2 | *One Ring to rule them all.* | **cut** | 1. **No line.** The Ring is forged and inscribed in silence (596–642).<br>2. Only if the producers insist: keep it at 628–686 with the fallback chain in TOP 5 #1. It fits, but with no gaps, and "enemies" clears the Eye by only 4 frames. | The earlier fixes (a capital R, retiming it onto the finished band) don't touch the problem. The inscription burning up out of the metal at 616–642 already quotes the line to everyone who knows it. Putting its most-quoted line under the Ring is the meme format itself. And "One ___ to rule them all" is a stock tech headline, so in an AI allegory it lands as a pun: a wink. It also uses 67 frames that C doesn't have (see the next two rows). |
| 705–770<br>2.7 s · 8 w · 3.0 | *And every kingdom and every guild wanted it.* | **tweak** | 1. *And every kingdom and every forge wanted it.* (656–722 · 2.9)<br>2. *And every kingdom and every smith wanted it.* (656–722 · 2.9) | "Guild" is where the translation shows: it reads as generic fantasy (thieves' guilds, game guilds), not as Tolkien. "Forge" keeps the companies as institutions, echoes "we forged", and gives NO FORGE IN THE DARK someone to address. "Smith" is Tolkien's own word for the Ring-makers, but it pins the wanting on people rather than institutions, so it is the second choice. The current line also starts while the quote is still fading (gap −2 frames). |
| 780–834<br>2.3 s · 7 w · 3.1 | *Each said: if not us, our enemies.* | **tweak** | 1. *Each said: better us than our enemies.* (744–804 · 2.8)<br>2. *Each said: if not us, then our enemies.* (744–804 · 3.2) | A's "if not us, someone worse" is an idiom. This transplant drops the verb and reads like a telegram. "Better us than our enemies" is idiomatic, keeps the symmetry of "Each", and is Boromir's argument in five words. The move to 744–804 is the real fix. Now the line is still at half opacity at 840, when the Eye's iris is already on screen, although framing B1 asked for it to be gone by about 836. At 744–804 it is gone by 816, 20 frames before the Eye begins to form. |
| 1060–1186 + 1112–1186<br>(black) · 20 w · **3.8 wps** | *In the old story, the Ring was unmade in the fire that forged it.* / *This fire could not be unmade.* | **keep words · tweak timing** | Line 1 `t_in` **1052**, line 2 `t_in` **1130**, both `t_out` **1190** with `fade_out=8`, so both are gone by 1198. | The best idea in either cut, and the only line that names its source, is under-timed. Line 1 (14 w, 1400 px) is alone for only 2.2 s before line 2 pulls the eye away, and the pair runs at 3.8 wps. Retimed, line 1 gets 3.25 s alone, the pair runs at 3.5 wps, and line 1 still starts half a second after the IMPACT. Keep "This fire", not "This one": it is what lets the 2512 line unmake the Ring without contradiction. |
| 1370–1426 (snow)<br>2.3 s · 9 w · **3.9 wps · 19 cps** | *And on a cold mountain, someone lit a beacon.* | **tweak** | 1. *And someone else lit a beacon.* (6 w · 2.6)<br>2. *On a cold mountain, someone lit a beacon.* (8 w · 3.4) | The fastest line in the film sits over its brightest background: moonlit snow, which needs the strongest halo in the film. The slot can't grow, because the cut comes at 1440. "On a cold mountain" says what the picture already shows. What C lacks is A's one indispensable word, "else", which puts the turn outside the kingdoms and the forges. Use option 2 only if the mountain must stay. |
| 1960–2040 (map)<br>3.3 s · 5 w · 1.5 | *And all the peoples answered.* | **tweak** | 1. *And hill by hill, the peoples answered.* (7 w · 2.1)<br>2. *And hill by hill, the world answered.* (7 w · 2.1) | "All" is the one absolute left in C, and the picture contradicts it: at 1960 only a few sparks are lit around the Himalaya, and the Americas fill in only as the line leaves. The line is also nearly the King James formula for a people entering a covenant ("And all the people answered…", Exodus 19:8), which is grandeur the film doesn't need right before a council. "Hill by hill" says it spread and took time, and it matches the map. |
| 2160–2280 (carved) | the four vows (as in A) | **keep** | — | As in A. In C, NO FORGE IN THE DARK is also exact Tolkien: the Ring was forged in secret. With "every forge" earlier, it now has someone to address. |
| 2512–2600<br>3.7 s · 10 w · 2.7 | *The Ring was unmade in a fire everyone had lit.* | **keep** | — | The payoff of the silence line and the governance claim in one sentence: the fire can't be uninvented, but one hand's prize can be melted down in a fire every emissary fed. That is literally what the council showed. Preventive note: the eagles planned for 2430–2560 must not be on screen under these words. The Ring destroyed and then the eagles is the book's own sequence, and staged under this line it turns homage into re-enactment. Clear them by about 2500, or drop them. |
| 2660–2730<br>2.9 s · 5 w · 1.7 | *Who lit the first one?* | **keep** | — | It lands even better than in A: after "a fire everyone had lit", it asks who started it. The shared "lit" is a rhyme, not a pun. |

**C's arc and last line.** The spine is the strongest writing in the project: the old story (the Ring unmade in its forge) → our difference (this fire can't be unmade) → the turn (someone lights a beacon) → a new ending (the Ring unmade in a fire everyone lit) → *Who lit the first one?*

Without the quote, it is unmistakably Tolkien through its own sentences, not borrowed ones.

Every problem sits in one stretch, 565–846:
- a quote, a transplanted "guild" and a verbless line are packed together with no gap;
- this is exactly where both earlier reviews found the fan-fiction and side-taking risks.

Fix that stretch (TOP 5 #1–2) and C reads as one continuous telling. The last line is right.

---

## TOP 5 changes, in priority order

1. **C: cut "One Ring to rule them all." and re-space the race (490–816).**
   - **New timings:**
     - *A fire that could think, …* at 490–572. It is gone by 584, before the forging starts at 596.
     - The Ring is forged and inscribed with no caption (596–642).
     - *And every kingdom and every … wanted it.* at 656–722.
     - *Each said: …* at 744–804. It is gone by 816.
   - **What it fixes:**
     - The last big cringe risk in either cut: a meme caption on an image that already quotes it.
     - The two lines that start while the previous one is still fading.
     - The half of framing B1 that is still open. "…our enemies." is at half opacity over the Eye's iris at 840; after this change it is gone 20 frames before the Eye begins to form.
   - **If the producers keep the quote**, the only timing that fits is 628–686 / 698–756 / 768–820. Every line stays at about 3.3 wps or less, but there are no gaps at all, and "enemies" clears the Eye by only 4 frames. That is the cost of the quote.

2. **C: three word fixes in the same stretch** (these stand on their own, with or without #1):
   - *A fire that could think, and had a will.* replaces "with a will of its own", which is a stock phrase and also the film prologue's phrasing.
   - *And every kingdom and every forge wanted it.* "Guild" is generic fantasy. "Forge" belongs to Tolkien's world and gives NO FORGE IN THE DARK someone to address.
   - *Each said: better us than our enemies.* The verbless "if not us, our enemies" reads like a telegram.

3. **C: give the silence line its time** (line 1 at 1052, line 2 at 1130, both `t_out` 1190, `fade_out=8`). The sentence the whole cut rests on is the one most likely to be half-read: 20 words at 3.8 wps, with 14 of them alone for only 2.2 s. Retimed, the pair runs at 3.5 wps and line 1 gets 3.25 s alone. No words change.

4. **C: "And someone else lit a beacon."** (1370–1426). The current line is the fastest in the film: 9 words at 3.9 wps (19 cps), over bright snow, in a slot the cut at 1440 won't let grow. The replacement runs at 2.6 wps and gains "else", the word that puts the turn outside the race, as in A.

5. **A: "We fed a new kind of fire everything we had ever written."** (340–440), and **"No one could see how." as its own beat** (a two-phrase row at 490/514). These are A's only two text problems:
   - the widest line in the film (1463 px at 3.4 wps) over its busiest background;
   - the film's admission that nobody could see inside the fire, which now arrives glued to the sentence before it.

**Next in line** (small, but on the way to perfect):
- C: *And hill by hill, the peoples answered.* (the absolute "all", the scriptural echo, and the mismatch with the map).
- C: *For the night when the beacons were lit.*
- A: *None would slow alone.* starting at 1114.
- C: no eagles under the 2512–2600 line.

**Protect (do not touch):**
- *So someone else lit a beacon.*
- *Each said: if not us, someone worse.*
- *Many in the race said it should slow. / None would slow alone.*
- C's silence couplet (its words, not its timing).
- *The Ring was unmade in a fire everyone had lit.*
- The four oaths, carved and unnarrated.
- *Who lit the first one?*, as the last words of both cuts, with nothing after it but the title.

---

## Appendix: the proposed `story_lines()` bodies

This includes everything above, TOP 5 and next-in-line, and was validated with the real `titles.Line` and `titles.row` code:
- **A:** the worst single-line rate drops from 3.4 to 2.9 wps.
- **C:** the worst single line drops from 3.9 to 2.9 wps, and the pair on black from 3.8 to 3.5.
- In both cuts, apart from the deliberate pairs, no line starts before the previous one has faded.
- In C, the race-logic line is gone by 816.

```python
    if cut == 'A':
        L.append(Line('A story they might tell of us, a lifetime from now.', 4, 104, y=402, size=50,
                      opacity=0.92, glow=0.2, fade=20, fade_out=16))
        L.append(Line('Why do we light the fires?', 140, 212, y=y, size=S))
        L.append(Line('To remember how close we came.', 226, 312, y=y, size=S))
        L.append(Line('We fed a new kind of fire everything we had ever written.', 340, 440, y=y, size=S))
        L += row(['And it began to think.', 'No one could see how.'], [490, 514], 565, y, size=S)
        L.append(Line('Whoever held it alone would hold the world.', 580, 648, y=y, size=S))
        L.append(Line('The companies raced for it, then the countries.', 668, 738, y=y, size=S))
        L.append(Line('Each said: if not us, someone worse.', 800, 866, y=y, size=S))
        L.append(Line('Many in the race said it should slow.', 1060, 1186, y=372, **black))
        L.append(Line('None would slow alone.', 1114, 1186, y=440, **black))
        L.append(Line('So someone else lit a beacon.', 1370, 1426, y=y, size=S, **snow))
        L.append(Line('In time, the race was over.', 2512, 2600, y=y, size=S))
        L.append(Line('Who lit the first one?', 2660, 2730, y=y, size=S))
    elif cut == 'C':
        L.append(Line('Why do we light the fires?', 120, 200, y=y, size=S))
        L.append(Line('For the night when the beacons were lit.', 215, 310, y=y, size=S))
        L.append(Line('From every word we had ever written, we forged a new power.', 340, 440, y=y, size=S))
        L.append(Line('A fire that could think, and had a will.', 490, 572, y=y, size=S))
        # 596-642: the Ring is forged and its inscription burns in -- no caption
        L.append(Line('And every kingdom and every forge wanted it.', 656, 722, y=y, size=S))
        L.append(Line('Each said: better us than our enemies.', 744, 804, y=y, size=S))   # gone by 816; Eye 836+
        L.append(Line('In the old story, the Ring was unmade in the fire that forged it.', 1052, 1190,
                      y=372, fade_out=8, **black))
        L.append(Line('This fire could not be unmade.', 1130, 1190, y=440, fade_out=8, **black))
        L.append(Line('And someone else lit a beacon.', 1370, 1426, y=y, size=S, **snow))
        L.append(Line('And hill by hill, the peoples answered.', 1960, 2040, y=y, size=S))
        L.append(Line('The Ring was unmade in a fire everyone had lit.', 2512, 2600, y=y, size=S))
        L.append(Line('Who lit the first one?', 2660, 2730, y=y, size=S))
```

If the producers keep the quote in C, replace the two race lines with:
- `Line('One Ring to rule them all.', 628, 686, …)`
- `Line('And every kingdom and every forge wanted it.', 698, 756, …)`
- `Line('Each said: better us than our enemies.', 768, 820, …)`
