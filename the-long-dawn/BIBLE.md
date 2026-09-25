# THE LONG DAWN — production bible

A ~1:57 animated short. Mythic, serious, emotionally sincere. The archetype is
*The Lord of the Rings* — the beacons of Gondor, the Council, the Ring that must
not be wielded by any one hand, eucatastrophe at dawn — retold as the (hoped-for)
story of how humanity got through the arrival of powerful AI: rivals uniting,
pacing the frontier by treaty, refusing concentration of power, avoiding war,
and coming out the other side into a flourishing future.

We never say "AI", never show a logo, never name a country. The metaphor is a
**new fire** — a fire that can think. Everyone in AI will read every beat;
everyone else will feel the myth. Grounding comes from precise, true-to-life
beats (the race logic, the labs themselves asking to be slowed, verification,
benefit-sharing), not from literal imagery.

Take it seriously. No irony, no cuteness, no clip-art. Every frame should look
like it cost money: controlled darkness, fire as the only saturated light, real
atmospheric depth, deliberate composition, motion with weight.

---------------------------------------------------------------------------

## 1. Story (the whole film in one breath)

Far in a good future, on the night of a yearly festival, an old woman and a
small child stand on a hill at dusk while beacon fires are lit on every hill
around them. The child asks why. The old woman tells the legend of the
**Kindling**: how we made a fire that could think, how every kingdom raced to
hold it alone, how the race nearly consumed the world — until its own makers
asked to be slowed, and *someone* on a cold mountain lit a beacon, and then
another was lit, and another, across deserts, ice, jungles, cities and seas,
until rivals and enemies gathered around one stone table and swore four oaths.
It did not end their differences. It ended the race. The sun rose on a world
that shared the fire. Back on the hill, the child asks who lit the first one.
The old woman doesn't answer — she just hands the child her torch. (Attentive
viewers have noticed: the young woman on the cold mountain wore the same long
red scarf the old woman wears now. It was her.)

## 2. On-screen text (added in the edit, not by shot renderers)

| id | frames | text |
|----|--------|------|
| T1 | 120–200 | *Why do we light the fires?* |
| T2 | 215–310 | *To remember the night the whole world answered.* |
| T3 | 340–440 | *In the age of the Kindling, we made a new kind of fire.* |
| T4 | 490–550 | *A fire that could think.* |
| T5 | 565–635 | *Whoever held it alone would hold the world.* |
| T6 | 660–730 | *So the kingdoms raced.* |
| T7 | 820–900 | *Each said: if we stop, they win.* |
| T8 | 1055–1195 | *Then even its makers said:* / *Slow us down.* / *All of us.* / *Together.* |
| T9 | 1370–1435 | *And on a cold mountain, someone lit a beacon.* |
| T10 | 1530–1600 | *Rivals. Strangers. Enemies.* |
| T11 | 1620–1700 | *They answered anyway.* |
| T12 | 1922–1998 | *They agreed on little — but they agreed on this:* |
| (diegetic) | 2000–2230 | the four oaths glow into the stone table (see §6 ACCORD) |
| T13 | 2270–2345 | *It did not end our differences.* |
| T14 | 2352–2440 | *It ended the race.* |
| T15 | 2500–2570 | *Who lit the first one?* |
| TITLE | 2668–2780 | THE LONG DAWN |

Text sits centred in the **lower third** of the 1920×804 picture. During those
windows keep the band y≈560–700 calm (no bright, busy detail right behind the
words). It does not need to be black, just not competing.

## 3. Time grid (everything is locked to this)

* 24 fps. 72 BPM, 4/4. **1 beat = 20 frames. 1 bar = 80 frames.**
* Bar *b* starts at global frame (b−1)·80. `look.bar_frame(bar, beat)` does this.
* Total picture: frames 0–2807 (117.0 s). Frame numbers are **global** everywhere.

| bars | frames | section | owner |
|------|--------|---------|-------|
| 1–4 | 0–319 | INTRO — the hill at dusk (future) | HILLS |
| 5–8 | 320–639 | KINDLING — glyphs → the thinking fire → towers → crown | EMBERS |
| 9–12 | 640–959 | RACE — towers surge, walls, red, the storm | EMBERS |
| 13 | 960–1039 | GRASP — a hand of embers closes on the crown | EMBERS |
| 14–15 | 1040–1199 | SILENCE — black, one dying ember, text T8 | EMBERS |
| 16–18 | 1200–1439 | FIRST BEACON — flint in darkness, the cold mountain | HILLS |
| 19–22 | 1440–1759 | BEACONS — montage around the world | MONTAGE |
| 23–24 | 1760–1919 | THE WORLD ANSWERS — globe at night, web of fire | GLOBE |
| 25–28 | 1920–2239 | THE ACCORD — torches converge; the stone table; oaths | ACCORD |
| 29–31 | 2240–2479 | DAWN — sunrise over Earth's limb (the climax) | GLOBE |
| 32–35 | 2480–2807 | CODA — the hill; the question; the child lights it; title | HILLS |

Key sync points (music hits these exactly):

| frame | event |
|------|-------|
| 0 | wind + low drone from silence |
| 80 | solo flute: the Beacon theme (1st statement) |
| 320 | KINDLING: crystalline arpeggio shimmer begins |
| 480 | IGNITION of the thinking fire (swell → hit) |
| 640 | RACE: drums enter; every beat thereafter is a tower-surge |
| 800 | storm swells (music intensifies) |
| 960 | GRASP: riser begins |
| 1036–1040 | fingers close; white-red flash; **1040 IMPACT → cut to black, silence** |
| 1060–1190 | sparse solo piano under T8 |
| 1236, 1262, 1290 | three flint strikes in the dark (sparks) |
| 1318 | kindling catches (small flame) |
| 1360 | the beacon ROARS; camera begins pull-back; the theme on solo horn |
| 1480, 1540, 1600, 1650, 1690, 1730 | beacon ignitions in the montage (one per shot) |
| 1760 | globe: choir enters |
| 2000, 2040, 2080, 2120 | Oath I, II, III, IV glow into the stone |
| 2160–2220 | outer ring of "together" in many scripts ignites in a sweep |
| 2225–2240 | hearth flares to white |
| 2240 | **CLIMAX** — the sun breaks over Earth's limb; full orchestra + choir |
| 2480 | coda: solo flute theme returns, quiet |
| 2640 | the child's torch lights the beacon |
| 2645–2720 | answering fires ignite across the hills (soft bells, one per fire) |
| 2720 | final chord; ring out; picture fades to black by 2807 |

## 4. Look

* Frame: **1920×804** (2.39:1), 24 fps. Render exactly this size.
* Pipeline: render linear-light float HDR → `lib/look.py: finish()` → `save_png()`.
  `finish()` applies bloom, optional anamorphic streak, vignette, ACES tonemap,
  sRGB. Tune exposure/bloom per shot; do not invent another tonemapper.
* **No film grain, no letterbox, no text** in delivered frames (the edit adds
  one unified grain/grade/titles pass). Diegetic carved text in ACCORD is the
  only exception.
* **Fire is the protagonist.** It is always the brightest, warmest thing in
  frame, with real HDR intensity (cores 5–30× the scene's ambient) so the bloom
  halates. Everything else lives in blue darkness. Blacks are deep but not dead.
* Colour arc of the fire across the film:
  - normal fire (beacons, torches): `fire_core → fire_hot → fire_mid → fire_deep`
  - the *thinking fire* (KINDLING): white core with **ice-blue** inner light and
    gold edges (`mind_core`, `mind_ice`, `mind_gold`) — unmistakably *not* ordinary fire
  - RACE: everything bleeds to `race_red / race_crimson`
  - ACCORD → DAWN: pure warm gold (`accord_*`, `dawn_*`)
* Palette hex values live in `look.PALETTE`. Night skies: zenith `#070B1C` →
  horizon `#27335E`, moonlight `#9DB4D9`. Silhouettes `#05060B`.
* Depth: atmospheric perspective (distant layers fade into haze colour), depth
  of field where it helps, parallax on every camera move. Nothing static — even
  "still" shots breathe (slow push, drifting embers, wind).
* Motion: ease in/out, weight, no linear robot moves. Fast particles get motion
  blur (streaks), not strobing dots.
* Reference feelings: the LOTR beacon sequence; Deakins' flare-lit night in
  *1917*; the particle-sand of the *Rings of Power* titles; the shadow-play of
  *Tale of the Three Brothers*; *Interstellar*'s awe; *Arrival*'s restraint.

## 5. Characters (silhouettes; readable at a glance)

* **The Elder** — small, slightly stooped old woman in a long coat. A **long deep
  red scarf** (`scarf_red`) wound at the neck, one long tail streaming in the
  wind — her signature shape. Carries a lit hand torch (INTRO) which she hands to
  the child (CODA). Warm firelight catches the scarf's red; the rest reads black.
* **The Child** — ~6 years old, small, knitted hat with a pom-pom (instant "child"
  silhouette), holds the Elder's hand, looks up at her.
* **The Young Woman** (the Elder, ~60 years earlier) — slim, heavy coat, hair
  whipping, the **same long red scarf** streaming in hard wind. Alone on a snowy
  peak at night. Kneels, strikes flint, stands as the beacon roars.
* **The Answerers** (MONTAGE) — ordinary people of every kind: a figure in desert
  robes, a researcher in a fur-hooded parka, a young person in a hoodie on a city
  rooftop, a sailor in oilskins, a shepherd on a peak. Dignified, never costume-y.
* **The Emissaries** (ACCORD) — hooded, cloaked shapes seen from above around
  a round stone table.
* **The Beacon** — a waist-high stone cairn topped with an iron fire-basket of
  stacked wood. When lit: a tall roaring flame, a column of sparks, smoke lit from
  beneath.

## 6. Shots (detailed briefs for each owner)

### INTRO (HILLS) frames 0–359 (deliver to 359; crossfade into EMBERS 300–340)
Blue hour, far future, peaceful. Layered hills recede into violet-blue haze; a
last amber band on the horizon; first stars. In the sky, subtle world-building
clues that this is a flourishing future: a **thin luminous orbital ring** arcing
across the sky, and a crescent **Moon whose dark side is sprinkled with tiny
city lights**. Never explained. On distant hills, festival beacons ignite one by
one. Foreground: the Elder (torch lit, scarf streaming) and the Child on a
hilltop beside an unlit beacon cairn. Text T1/T2 happen here. Ends by pushing
slowly into the Elder's torch flame until rising embers fill the frame — those
embers become the next section.

### KINDLING / RACE / GRASP / SILENCE (EMBERS) frames 300–1199
The legend, seen *in the fire*: everything is made of glowing embers/particles
in 3D space on darkness, with a moving perspective camera, depth of field and
motion blur.
* 300–340: rising embers (continuous with the torch) swirl, slow, begin to glow
  as tiny letters.
* 320–470: thousands of glyphs from every script and discipline (Latin, Greek,
  Cyrillic, Arabic, Hebrew, Devanagari, CJK, Hangul, Ge'ez, Tamil, math, musical
  notes, DNA letters, code) drift in like fireflies from all directions, then
  spiral inward, accelerating, compressing to one blinding point. (*everything
  we ever wrote*)
* **480 IGNITION**: the point blooms into the thinking fire — white/ice-blue core,
  gold edges, fine branching filaments inside it like thought. It breathes.
* 520–640: towers of embers (6–8, each architecturally distinct = different
  kingdoms) rise in a ring around it, inner faces lit by it. The fire lifts and
  shapes itself into a floating **crown/ring** of fire — the prize.
* 640–960 RACE: each beat (every 20 frames) the towers surge upward in jolts
  spraying sparks; walls of red embers divide them; palette bleeds to crimson;
  camera flies up between the towers. From 800 the crown-fire balloons into a
  vast unstable vortex dwarfing everything; ~880–950 a glimpse of the whole
  world as an ember globe cracking with fire.
* 960–1040 GRASP: a colossal hand of embers rises and reaches for the crown;
  fingers close at 1036; white-red flash 1036–1039.
* **1040: black.** 1040–1199: pure black except one small ember drifting down,
  dimming, dying by ~1150. Keep the lower third clear (text T8).

### FIRST BEACON (HILLS) frames 1200–1439
Black. Wind. Flint strikes at 1236, 1262, 1290 — each a burst of sparks that
for an instant lights a young woman's hands, face in profile, the red scarf. The
kindling catches at 1318 (small flame, her face lit, breath visible). At 1360
the beacon **roars** — flame and sparks leap up — and the camera pulls back and
up to reveal: a snowy summit at night, spindrift blowing, a sea of moonlit peaks
below under a vast starfield, the scarf whipping. She is tiny and the fire is
the only warm thing in a cold world. Text T9 over the reveal.

### BEACONS (MONTAGE) frames 1440–1767 (cuts on the beat)
Six places, each with a person lighting a beacon on the beat. Each shot is
ruthlessly composed, one clear idea, subtle camera move.
* 1440–1519 **far peak**: a figure in the foreground (seen from behind) watches a
  tiny new light flare on the far horizon (the first beacon), turns, thrusts a
  torch into their own beacon — ignition **1480**, fire fills the foreground.
* 1520–1579 **desert**: moonlit dunes, robed figure on a crest; ignition **1540**.
* 1580–1639 **ice**: ice cliffs, green aurora overhead, figure in fur-hooded
  parka; ignition **1600**.
* 1640–1679 **karst jungle**: mist-filled valleys, jagged limestone pinnacles
  (Guilin/Zhangjiajie feel); a beacon on a pinnacle; ignition **1650**.
* 1680–1719 **city**: modern skyline at night, lit windows; a young person on a
  rooftop lights a beacon; other rooftops answer in the distance; ignition **1690**.
* 1720–1767 **sea**: small boat on dark swells, moon-glitter; a sailor raises a
  burning flare at the bow; along the far coast a chain of beacons ignites;
  ignition **1730**. (1760–1767 is a handle.)

### THE WORLD ANSWERS (GLOBE) frames 1752–1935
Earth at night from orbit, city lights, thin atmosphere rim. From a point in
the high Himalaya a web of golden fire spreads: beacon to beacon, branching,
arcs leaping oceans, until the visible night side is laced with gold. Slow
majestic camera. (Handles 1752–1759 and 1920–1935 for dissolves.)

### THE ACCORD (ACCORD) frames 1912–2247
Top-down, descending. 1912–2010: high above a dark plain, rivers of torchlight
converge from every direction toward a ring of standing stones. 2010–2060: we
are above a great round stone table; cloaked emissaries stand around it, their
long shadows radiating outward from the central hearth like a sundial; each
extends a torch into the hearth; the flames merge into one **golden** fire. The
four oaths then glow into a ring carved in the stone, each one rotating to the
top (upright, readable) as it lights:
* 2000 **NO SINGLE HAND SHALL HOLD IT**
* 2040 **NO FASTER THAN WE CAN SEE**
* 2080 **NO FORGE IN THE DARK**
* 2120 **WHAT IT GIVES, IT GIVES TO ALL**

(Draw them in Cinzel, `assets/fonts/Cinzel.ttf`, glowing like moonlit ithildin
but gold.) 2160–2220: an outer ring ignites in a sweep: the word *together* in
many languages and scripts (list in §8). 2225–2247: the hearth flares to white
(match-cut to the sun).

### DAWN (GLOBE) frames 2232–2495
The eucatastrophe. We're in orbit above Earth's curved limb, night below, the
golden web still glowing on the dark side. At **2240** the sun breaks over the
limb — starburst, anamorphic flare, the atmosphere igniting in a band of
orange-rose-blue — and the terminator sweeps across the planet. On the remaining
night side, new lights bloom into places that were dark (the fire reaching every
hearth). The camera rises slowly. Text T13/T14 over it. (Handles for dissolves
2232–2239 and 2480–2495.)

### CODA (HILLS) frames 2460–2807
Back on the hill (same place as INTRO, deeper blue now, more stars, the orbital
ring and the lit Moon clearer). Medium shot: the Child looks up — T15 *Who lit
the first one?* (2500–2570). The Elder says nothing; the wind lifts the red
scarf; she places her torch in the Child's hands (2580–2620). Wide: the Child
touches the torch to the beacon — it catches at **2640** — and across every hill
to the horizon answering fires ignite in a rolling wave (2645–2720). The camera
cranes up into the sky: stars, the ringed sky, the Moon's lights; a few slow
moving lights (ships). Hold for the TITLE (2668–2780); picture should be quiet
and dark enough for it. Picture can end at 2807 (the edit fades out).

## 7. Deliverables & conventions (all visual owners)

* Code lives in `shots/<owner>/` (lowercase: `hills`, `embers`, `montage`,
  `globe`, `accord`). Frames go to `renders/<owner>/f_%05d.png` using **global**
  frame numbers (see `look.frame_path`). Frames are 8-bit PNG via `look.save_png`.
* Also write `renders/<owner>/preview.mp4` (`look.preview_mp4`) and
  `renders/<owner>/contact.png` (`look.contact_sheet`) and a short
  `shots/<owner>/NOTES.md` (what you built, how to re-render, any deviations
  from this bible and why).
* **Quality bar:** look at your own frames (the Read tool shows PNGs) at every
  step. Render low-res tests first; iterate until you'd put it in a trailer.
  Check motion by looking at sequential frames/contact sheets, not just stills.
* **CPU etiquette:** the machine has 4 cores shared by 5–6 people. Use at most
  **2** worker processes / numba threads (`NUMBA_NUM_THREADS=2`, `cv2` threads 2)
  and keep RAM under ~3 GB. Prefer numpy vectorisation + numba over pure Python.
* **Don't run git commands that change the repo** (no add/commit/checkout);
  the director commits. Don't touch other owners' directories.
* Everything must be made by you, from code (procedural), or from the
  open-licensed data already in `assets/` (Natural Earth vectors, three.js Earth
  textures, OFL fonts, system Noto fonts). No downloaded images of scenes,
  characters, or stock footage.

## 8. "Together" ring (ACCORD outer ring)

together · juntos · ensemble · zusammen · insieme · вместе · разом · razem ·
μαζί · 一起 · 共に · 함께 · معًا · ביחד · با هم · एक साथ · ایک ساتھ · একসাথে ·
ਇਕੱਠੇ · ஒன்றாக · ด้วยกัน · cùng nhau · bersama · pamoja · birlikte ·
ერთად · միասին

(Use the installed Noto fonts: `fc-list | grep -i noto` — Noto Serif / Noto
Sans + Noto Serif CJK / Arabic / Hebrew / Devanagari / Bengali / Gurmukhi /
Tamil / Thai / Georgian / Armenian. Render with a shaping-capable path: PIL with
libraqm, or skia with shaping, and verify Arabic/Urdu/Persian join and read
right-to-left.)

## 9. Music (SCORE owner) — summary; full cue sheet in `music/CUES.md`
Original score, 72 BPM, D minor → D major. A simple, singable hymn-like
**Beacon theme** (rising "call" D–A–D then a falling answer) first on solo flute,
twisted dark in the RACE, fragmented on piano in the SILENCE, passed from
instrument to instrument in the BEACONS (each beacon = the next phrase in a new
voice), triumphant with full orchestra, organ and choir at the DAWN, and home on
solo flute in the CODA.
