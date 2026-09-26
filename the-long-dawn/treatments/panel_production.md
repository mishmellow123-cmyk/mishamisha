# PANEL: PRODUCTION AND VISUALS

*The production-and-visuals critic on the v3 treatments. Writers' room, 26 Sep 2026. Session gate: GATE PASS.*

*Read in full: the poet, the dramatist, the essayist and the contrarian. There is no `claude/treatment-mythmaker`
branch, so there are four treatments, not five. Also used: the producers' verdict and updates 2–4, both bibles, the
department NOTES and both red-team reports (surveyed for costs and open issues), the review sheets, frames pulled from
the DAWN_C and `run_fix` render branches, and the cloud render logs, which are where the costs below come from.*

---------------------------------------------------------------------------------------------------------------

## 0. Verdict

| rank | treatment | feasibility (as written) | visual ambition | in one line |
|---|---|:-:|:-:|---|
| **1** | **DRAMATIST**: *The Crossing · The Longest Night · The Red Book* | **6** | **8** | The most producible ambition. Every showpiece is built from parts we already own, and B is a different film because it is *made* differently. |
| **2** | **CONTRARIAN**: *Every Step Closer · The Keeper · The Last Pages* | **3** | **9** | The best ideas in the room (THE EDGE, THE LIFETIME SHOT, THE LIVING INK ILLUSTRATION), inside a 17-minute slate that needs twice the schedule. Harvest it. |
| **3** | **POET**: *The Watch · The Stolen Sun · The Refusal* | **5** | **8** | The best image-writing. But each cut's centre sits on its riskiest shot: a 16-second face, a filter over reused renders, a two-renderer composite. |
| **4** | **ESSAYIST**: *No One Runs · The Open Hand · The Last Pages* | **7** | **6** | The safest slate (every risky shot falls back to renders on disk) and the least divergent: its B is draft 1's B again. Harvest THE DEAD HILL, the stills and THE MIRROR. |

**How I scored.** *Feasibility:* can the slate, as written, be made at the quality bar in about twenty hours of
wall-clock with our lanes and machines? 10 means comfortably; 5 means it needs about 1.5× the time or heavy cuts; 1
means impossible. *Visual ambition:* how far it stretches the picture (3-D and motion design), how differently the
three cuts look and move, and whether the ambition stays in taste. Ambition that invites a doll-read, a poster or a
filter look scores lower.

**A and B each have a strong, distinct concept. Keep three films; do not consolidate.**
* **A:** the dramatist's THE CROSSING is the strongest and the most producible. The contrarian's EVERY STEP CLOSER is
  a close second and has the best race imagery (THE EDGE), which A should take.
* **B:** there are two strong, distinct Bs. The dramatist's THE LONGEST NIGHT is one range, one night and three waves of
  light, with no embers act, no council and no words. The contrarian's THE KEEPER is sixty years in one locked frame.
  Both differ from A in plot *and* in picture. Production prefers THE LONGEST NIGHT (shorter, cheaper, purer); THE
  KEEPER's lifetime shot can be built in the same set as its named upgrade. The poet's THE STOLEN SUN is a strong myth
  but shows A's embers pictures for half its length. The essayist's THE OPEN HAND is not distinct enough.

**The achievable best film, in one paragraph.** Make the dramatist's three films, trimmed to about fourteen minutes,
and graft in five pieces from the others:
* THE EDGE (contrarian) as A's race;
* THE DEAD HILL (essayist) as A's "forever";
* fires catching on far ridges in the same breath as hers (contrarian, poet), so A has no lone hero;
* THE RING FALLS (poet) and the Ring found in the snow (contrarian) as C's turn;
* THE LIVING INK ILLUSTRATION (contrarian) as C's look.

Build every new mountain shot in the one mountain renderer that has already delivered on the cloud (the Run world),
and make the cuts diverge by *rendering that world three ways*: A at night with the race glowing red under the cloud,
B in natural light from dusk to dawn, C as a moving ink drawing. Seven new showpieces are worth the effort (§5.3).
Nine of the proposed ones are not, in this schedule (§5.5).

---------------------------------------------------------------------------------------------------------------

## 1. Ground truth

### 1.1 What is already on disk, and who reuses it

P = poet, D = dramatist, E = essayist, Ct = contrarian. Letters are the cuts that reuse the render.

| render (v2 frames) | what it is | P | D | E | Ct |
|---|---|---|---|---|---|
| `hills_v2` intro, 0–359 | the elder and the child at dusk; far fires; the push into the torch | A, C | A | A (as stills) | retired |
| `embers_v2` / `_B` / `_C`, 300–1199 | glyphs → thinking fire → towers → crown or Ring → race → vortex or Eye → ember globe → grasp → silence | A, B (recoloured), C | A, C | A, B, C | A, B (a dream), C |
| `hills_v2` first beacon, 1200–1439 | flint, the catch, the roar, the reveal | all | all | all | all (slower take) |
| `montage_v2` s1, 1440–1519 | the shepherd answers (s1's world, which the Run world grew from; re-rendered on the cloud with the Run's flames) | A, C | all | all | A, B |
| `run_v2`, 1520–1679 | THE BEACON RUN: a glider flight past seven beat beacons over the cloud sea | A, C (+8 s) | all (+ red under-glow) | B, C (longer) | A, B |
| `montage` + `montage_v2` city | desert, ice, karst, city, sea | A | A (three), B (five) | A, B, C | A, B |
| `globe_v2`, 1760–1927 | the world answers (fires, not fibre) | A | B | B | B |
| `map_C`, 1920–2087 | our Earth as a Tolkien map | C | C | C | C (+ an unlit prologue) |
| `accord_A` / `_B` / `_C`, 1912–2247 | the stone table from above | A, B, C | C | A, B, C | A (the rim), C |
| `globe_v2` / `globe_B` dawn, 2232–2495 | the orbital sunrise | — | — | — | — |
| `dawn_C`, 2400–2655 | the sun over the eastern ranges, the cloud sea, six eagles | B (no eagles), C | C | B, C (26 s) | B, C |
| `hills_v2` coda, 2460–2807 | the question, the handover, the answering fires, the crane | A, B (as its opening) | A | A, B, C | B |
| `edit/ember_title.py` | THE LONG DAWN kindles and crumbles | all | all | all | all |

Three things stand out.
* **The orbital sunrise is retired by all four** (the red team's "keynote slide"). It is a sunk cost, and the right
  call.
* **Every treatment routes its new mountain shots through `shots/run`**: the Run world and `dawn.py`, the numba world
  that also makes the shepherd, the Run, the first beacon's reveal and DAWN_C. That convergence is the production
  backbone of v3. It is also its hazard: used naively, it makes the three films look alike.
* **Much of what the room judged is out of date.** The `run.jpg` review sheet shows the Blender Run, which is in no
  current cut; the Run in the edit is the numba `run_v2`, and I looked at tonight's `run_fix` frames instead (the chain
  reads; the beacons are real flames with smoke; the peaks are a forest of similar spikes). Heroine v2b (the ignition
  fixed; the "weird" read unchanged), globe v3, map rev 3, `run_fix` and title M6 all landed tonight, and none has been
  assembled into a cut or re-reviewed.

### 1.2 What a frame costs (measured)

| renderer | makes | where | cost | evidence |
|---|---|---|---|---|
| **Run world** (`shots/run`: a numba height-field marcher with terrain, snow, cloud deck and `fire2` beacons) | the Run, the shepherd, the reveal; every new mountain shot | cloud, 4 processes a box | 58–77 s a frame per process after a ≈ 4-minute compile: **≈ 200 frames an hour per box** | `run_v2_b` log; `run_fix`: 160 frames on one box in ≈ 47 min |
| **`dawn.py`** (the same world at sunrise) | DAWN_C; every new dusk and dawn | cloud | 28–35 s a frame per process: **≈ 450 frames an hour per box** | DAWN_C: four boxes × 64 frames, 7–9 min each |
| **EMBERS** (3-D particle splatting, depth of field, motion blur) | the embers act | Mac so far | median 3.4–5 s a frame (the grasp ≈ 8 s) | 609 frames in 69 min on two contended workers |
| **HILLS** (2.5-D cards, silhouettes) | the hill, the coda | Mac so far | ≈ 4.5 s a frame per worker | 708 frames in 26 min |
| **HEROINE** (a 3-D SDF figure, sphere-traced) | the first beacon | Mac so far | 14–26 s a frame on average; strike frames ≈ 90 s | 240 frames in 27–52 min |
| **GLOBE** | the world answers | cloud | ≈ 4 s a frame | 712 frames in ≈ 17 min |
| **MAP** (ink, parchment, fire front) | `map_C`; any ink page | cloud | ≈ 1 s a frame, after a 6-min bake | 168 frames in under 5 min |
| **ACCORD** (SDF table, carved text) | the table | Mac so far | 7–21 s a frame | 1,008 frames in 124 min on two workers |
| **Blender EEVEE Next** | the CITY shot (the only Blender footage in any current cut) | the Mac only: an M2 with 8 GB shared by every department; one Blender at a time | CITY 52 s a frame; the Blender Run ≈ 20 s a frame | a memory crash on record; "lumpy mounds" on this Mac (the director); the Blender mountain world is in no current cut |
| **cloud concurrency** | — | — | **twelve boxes at once** | 19:25–19:48 tonight: DAWN_C ×4, the Run ×4, s1 ×2, the globe ×2 |

The treatments' "≈ 6 min a frame" (the Run) and "2–4 min a frame" (`dawn.py`) are Mac figures. The same code runs four
to six times faster per thread on a cloud core.

### 1.3 Twenty hours is not "week one"

Every treatment plans in weeks: "days 1–2", "keep or kill by day 5", "week-1 greyboxes". The schedule is about twenty
hours of wall-clock with parallel agents. So:
* **"Week one" is hours 0–8.** Every test protocol in the treatments has to shrink to stills at hour 3 and 48-frame
  motion tests at hour 8.
* **CPU is not the constraint.** The heaviest slate needs about 25 box-hours of new mountain frames: under three hours
  on ten boxes. The embers, hills, map, globe and accord renderers are an order of magnitude cheaper.
* **The constraint is the busiest lane.** A new showpiece costs an agent six to eight hours of build, look-development,
  a motion test and one fix before its render starts. With about thirteen lanes and about sixteen useful hours each
  (≈ 200 lane-hours), the question for each treatment is whether its busiest lanes (RUN, EMBERS, HEROINE) fit in
  sixteen hours.

Effort scale used below: **S** ≤ 2 h · **M** 3–4 h · **L** 6–8 h · **XL** ≥ 10 h (build, look-dev, motion test and one
fix; render time excluded). Lane-hours are estimates; box-hours use the measured rates above.

| | runtime (A + B + C) | NEW builds (S / M / L / XL) | new lane-hours | busiest lanes | new mountain frames | fits twenty hours? |
|---|---|---|---|---|---|---|
| poet | 5:26 + 4:06 + 5:50 = **15:22** | 4 / 9 / 6 / 2 | ≈ 110 | HEROINE ≈ 30 h, RUN ≈ 30 h, EMBERS ≈ 28 h | ≈ 2,900 (≈ 11 box-h) | no: ≈ 1.6× |
| dramatist | 5:08 + 3:42 + 5:40 = **14:30** | 4 / 10 / 6 / 1 | ≈ 95 | RUN ≈ 45 h (three lanes), EMBERS ≈ 30 h, MAP ≈ 20 h | ≈ 6,100 (≈ 23 box-h) | nearly: ≈ 1.25×, and it trims cleanly |
| essayist | 4:42 + 3:54 + 5:20 = **13:56** | 3 / 8 / 5 / 2 | ≈ 85 | EMBERS ≈ 35 h (the torch race), RUN ≈ 30 h (the ascent) | ≈ 2,700 (≈ 9 box-h) | yes, once its two XL shots fall back |
| contrarian | 5:10 + 5:20 + 6:40 = **17:10** | 5 / 11 / 6 / 4 | ≈ 140 | RUN ≈ 55 h, EMBERS ≈ 40 h, HILLS/HEROINE ≈ 30 h | ≈ 5,200 (≈ 26 box-h) | no: ≈ 2× |

On top of each: **about 35 lane-hours of fixes that all four assume**. The red team's notes on reused shots are still
open: the thinking fire reads as a candle, the crown as a gas-hob burner, the Ring as a glitter bangle; the vortex
is stock and the Eye a sticker; the race frames carry a brown haze; the grasp is a raised protest fist; the desert reads
as snow; the ice's aurora overpowers the fire; the sea's coast beacons are a string of pearls; the table is a coin seen
from above, with pawns and white-pill torches; the hill is vector wallpaper with a flat scarf and a brick-chimney cairn.
That is a quarter to a third of the work before a single new shot, and it is the same in every treatment.

**One pipeline unknown.** Cloud frames travel as PNG commits to `claude/render-*` branches, at about 1.5 MB each, and how
they reach the Mac for the edit is not documented. That was fine for tonight's 2,250 frames (about 3.4 GB). A v3 slate
means 12,000–15,000 new frames, roughly 20 GB through git. Decide the transport at hour 0; don't discover it at hour 14.

### 1.4 The score is 124 seconds, and nobody can hear it

Today's score runs 124 s: a shared first half (v1's score, called unchanged) and second halves AB and C. It is Python:
a score DSL, a VSCO-2 CE sampler, synthesized effects. Rendering both scores takes 4.5 minutes; **composing is the
cost**, because every line, voicing and dynamics breakpoint is placed by hand and anchored to absolute frames. Any
change to a cut's timeline means re-composing the sections it touches. And the composer cannot listen: the music notes
say "verify by analysis". The producers are the only ears in the pipeline.

Every treatment needs roughly seven times today's score: 13:56 (essayist) to 17:10 (contrarian). What carries over is
the motif kit (the call and the answer, the glass FM cycle, the IGNITION chord, the race's taiko, the riser and the
suck, the silence piano), the first half's cues stretched by whole bars, and score C's second-half material for C.
Nearly everything after the first beacon is new in every treatment. So:
* **Runtime is a music decision.** Every extra minute is composed blind, by hand.
* **The best music plans are rules a composer can implement and check by analysis.** The dramatist's "the harmony
  moves only when the line passes a watch-fire", the poet's deceleration by note value and the contrarian's chaconne
  (one variation a year) all qualify.
* **A sparse B is the cheapest score to write well.** A cello, far horns and wind carry the dramatist's B.
* **Timings are baked into more than the music.** The embers text bands, the effects list, the ember title and the
  coda's answering fires all depend on the timeline, so the bar maps must lock before any of them render.

### 1.5 Six production laws for this schedule

1. **New 3-D goes in the Run world, on the cloud.** It is the mountain renderer that has shipped in parallel on the
   cloud (the Run, the shepherd, DAWN_C). Blender runs only on the Mac (one M2, 8 GB shared by every department, one
   Blender at a time), has crashed on memory, and its mountain world is in no current cut. Use it only where nothing
   else will do. None of the shots I recommend needs it.
2. **Render on the cloud, and each shot on one kind of machine.** The Mac is 8 GB shared by everyone, and Mac and cloud
   renders of the same frame don't match exactly (the globe's differ at about 30 dB PSNR), which flickers if a shot
   mixes them. The Mac keeps the edit and the music.
3. **No new face close-ups.** The elder and the child were the tone critic's blocker ("read as jointed dolls"), the red
   team called the heroine a mannequin, and heroine v2b leaves the producers' "a bit weird" unchanged (the suspects
   are the dark hair mass, the scarf tail read as a dark band, and locks that read as twigs). New character beats are
   hands, silhouettes and backlit small figures, until a still of the new pose passes.
4. **Lock each cut's sections in bars by hour 3.** Every renderer draws a frame as a pure function of time, so a shot
   can be rendered to any length. Music and picture then work in parallel against one bar map.
5. **One world, three renderings.** The cuts diverge through light, weather, a non-photoreal pass, camera grammar and
   type, not through new worlds. That is the only way Update 2's divergence fits the schedule.
6. **Every new shot names its fallback on disk before it starts**, and is killed at a gate, never after its render.

---------------------------------------------------------------------------------------------------------------

## 2. The four treatments

### 2.1 POET: A · THE WATCH (5:26) · B · THE STOLEN SUN (4:06) · C · THE REFUSAL (5:50)

**Reuses** nearly the whole v2 picture, each cut through its own lens: the hill and the coda (A, and B's opening), the
embers act (A as it is; B repainted as the sleeping sun, the old kingdoms, Night and a coal-black hand; C with the Ring
and the Eye), the first beacon, the shepherd, the Run, the five places (A), the globe's answer (A), `map_C`, all three
tables, DAWN_C (C, and B without its eagles) and the ember title.

**Extends:** the race with gold rain; the ember globe cooled to ash; the grasp restaged three ways; the reveal
re-plated in the Run world three ways; the Run 8 s longer (C); A's table with the thinking fire in the hearth and
shadows cast as towers; torches carried outward from the table (A).

**Work lands on** EMBERS (heaviest), HEROINE (a face, a hand, a seated watcher), RUN (a composite, a morph, a dawn),
HILLS (a paper one-take), EDIT (a shadow-play pass over all of B) and ACCORD (a melt). No Blender.

**New builds**

| cut | shot | lane / renderer | frames | effort | risk | verdict |
|---|---|---|---|---|---|---|
| A | GOLD RAIN: on every surge, gold streams *up* into the forges' windows | EMBERS layer | ≈ 290 | S–M | low | **build**: the cheapest correct picture of "every step pays" |
| A | THE SPRINT: a low flight between the towers as their shutters slam | EMBERS camera + window parameter | ≈ 190 | M | medium | build |
| A | THE VERTIGO: a dolly-zoom into the fire; warm lattices one way, grey ash the other | EMBERS camera + recursive filaments | ≈ 144 | M | medium (reads as a diagram) | test |
| A | the ember globe goes to ash, and stays | EMBERS | ≈ 48 | S | low | build |
| A | FORGES IN THE LIGHT: the Run world's plain and the embers' towers, composited by depth | RUN + EMBERS | ≈ 380 | XL | high: two renderers, one light | **cheaper twin**: the towers' backs lit by a ring of watchfires, inside one renderer (the contrarian's) |
| A | THE WATCH AT FIRST LIGHT and THE THAW: a seated watcher, the snow line lifting, the first green | `dawn.py` + HEROINE + a terrain time term | ≈ 672 | L | high ("CG green"; a seated heroine) | build without the green; the watcher a backlit silhouette |
| B | the shadow-play pass over every reused render (silhouettes, paper grain, halation) | EDIT post | all shared B frames | L | high: a filter reads as a filter | test on three shots before B's look depends on it |
| B | embers settle into stars | EMBERS | ≈ 336 | M | low | build (if B is this B) |
| B | THE ROAD EAST: a 44-second lateral one-take through cut-paper worlds | HILLS 2.5-D cards + translucency | ≈ 1,056 | L | medium-high (the red team's "vector wallpaper") | B's signature; cheap to render; test |
| B | THE HILL AT DAWN: the child asleep in the scarf | HILLS new poses + a dawn sky | ≈ 624 | M | medium | build (silhouettes at dawn are forgiving) |
| C | THE RING FALLS: (a) tumbling out of the black through cloud; (b) a gold star over her summit, a hiss and steam in the snow | EMBERS + RUN | ≈ 430 | M | medium | **build**: the best spectacle per hour in C |
| C | THE RING IN THE SNOW: sixteen seconds on her face | HEROINE close-up | ≈ 384 | XL | very high | **cut the face**; shoot the poet's own fallback (over the shoulder, the Ring the only light) |
| C | the Ring slid into the kindling | HEROINE hand insert | ≈ 96 | M | medium | build, hands only |
| C | THE LAND BECOMES THE MAP: height to zero, rock to parchment | RUN marcher + MAP sheet | ≈ 240 | L | high | test; or the ink-pass dissolve (§5.2) |
| C | THE UNMAKING: the Ring melts in the hearth | ACCORD `ring.py` | ≈ 240 | M | medium | build |
| C | THE SCAR: an old palm opened to the fire, a mitten laid over it | HEROINE hand close-up + HILLS seated figures | ≈ 816 | L | high (a clay hand) | the fallback from the start: the hand at the edge of focus |

**Divergence.** The strongest on paper: night photography in two camera speeds (A), shadow-play (B), an illuminated
chronicle (C). A's device costs nothing, because it is camera keys and cutting, and it is the best formal idea in the
room for pace: before the turn the camera sprints; after it, it never moves faster than a person walks. B's look is the
weak point. Apart from THE ROAD EAST, B's shadow-play is a post pass over reused 3-D renders, and a filter over particle
renders reads as a filter. B's first 86 seconds would still *move* like A's embers act, because they are A's embers act,
recoloured.

**Ambition and taste.** GOLD RAIN and THE RING FALLS are the two best ratios of picture to cost in any treatment.
THE ROAD EAST is real motion design, and it plays to the HILLS engine's strength. But the risks sit exactly at each
cut's centre:
* C's temptation is held for sixteen seconds on the heroine's face, the weakest asset in the pipeline.
* FORGES IN THE LIGHT asks two renderers to share one frame and one light.
* THE THAW's first green is the "CG" risk the poet names; gold light on wet stone carries the flourishing without it.
* One continuity gap will cost a shot: in C she drops the Ring into her beacon's kindling, and it next lies cold in the
  council's hearth. Someone has to carry it there.

**Music.** Three idioms past the shared opening: A's deceleration by note value over a continuous low D (elegant, and
checkable by analysis); B's frame-drum song (a percussion colour the score hasn't used); C's epic, with a Ring motif.
15:22 in all, the second-heaviest load.

**Feasibility 5 · visual ambition 8.**

### 2.2 DRAMATIST: A · THE CROSSING (5:08) · B · THE LONGEST NIGHT (3:42) · C · THE RED BOOK (5:40)

**Reuses.** A: the hill, the embers act (the glyphs slowed 1.4× by sampling fractional time, so the motion blur stays
true), the silence, the first fire, the reveal, the shepherd, the Run, three watchers (city, ice, sea), the coda and the
title. B: only the first fire, the reveal, the shepherd, the Run, the five places and the globe's answer, about 90 s of
its 222. C: the embers act with the Ring and the Eye, the first fire, the Run, `map_C`, `accord_C` and DAWN_C.

**Extends:** the thinking-fire, crown, vortex, Eye and grasp fixes; the reveal held 48 frames longer; the reveal, the
shepherd and the Run re-rendered with a red under-glow in the cloud (A, C); the globe's answer (B); `map_C` with a
dissolve from the Run's last wide (C); DAWN_C with a single eagle.

**Work lands on** RUN (heaviest: seven new mountain shots and three re-renders, so three lanes), EMBERS, MAP (the book
and its pages), HEROINE (four new setups in B) and ACCORD (a melt). Blender only if THE KEEPERS RISE uses MONTAGE-3D's
figures; I'd keep it to lights.

**New builds**

| cut | shot | lane / renderer | frames | effort | risk | verdict |
|---|---|---|---|---|---|---|
| A, C | THE GIFT: a seed, a shell becoming a spiral of stars, wheat, held in the fire's beam | EMBERS point forms | 336 | M | medium (a tech-promise montage) | build small: natural forms only |
| A | gold running down the towers; a spark falling on dry grass | EMBERS layer | ≈ 280 | S–M | low | build |
| A | THE KEEPER AT THE BELLOWS | EMBERS ember-crust figure | 96 | M | medium-high (a doll) | drop for time |
| A | THE FALL: the camera dives with a spark onto the ember globe | EMBERS camera | ≈ 170 | M | low-medium | optional |
| A, C | THE RED UNDER-GLOW: the race glowing beneath the cloud, all around | RUN cloud-deck emission + three re-renders | ≈ 400 | S (+ CPU) | low | **build**: the cheapest big idea in the room |
| A | THE KEEPERS RISE: white lanterns climb out of the red cloud and meet at a saddle | RUN + lantern sprites (+ Blender figures) | 470 | M–L | medium-high (the figures) | build as lights; figures only if a still passes |
| A (+ C) | **THE CROSSING**: a roped line of lanterns on a knife-edge while the sky wheels | RUN arête + 20-px figures + sprites + star rotation + trail buffer | 840 (+ 480) | L | medium | **build: the signature** |
| A | THE LONG DAWN on the ridge: each watch-fire pales as the sun reaches it | `dawn.py` + a per-fire sun-elevation rule | 720 | M | low-medium | build |
| B | DUSK: the light leaves the peaks, the lowest first | `dawn.py` run backward | 624 | M | low-medium | build |
| B | THE CLIMB | RUN + the heroine at ≤ 60 px, carrying a glowing pot | ≈ 290 | M | medium (her gait) | build: a trudge, the pot the subject |
| B | THE DEAD EMBER | HEROINE close-up (hands) + a clay pot | ≈ 216 | M–L | medium | still first; her hands are the accepted asset |
| B | THE VIGIL: she sits by the beacon while the sky wheels | RUN + the heroine seated | 770 | L | medium-high | build; the silhouette fallback |
| B | **THE HAND-BACK**: the sun walks across the range and each beacon pales; hers is last | `dawn.py` + the vigil set + torch threads to the valleys | ≈ 1,300 | L | medium | **build: B's signature** |
| C | THE RED BOOK by the hearth, the two shadows on the wall | MAP 2.5-D + leather + wall shadows | ≈ 336 | M | low-medium | build |
| C | THE INK PAGES ×7 | MAP ink engine | ≈ 1,050 | L–XL | medium-high (clip-art) | **build four** |
| C | BURN-THROUGHS | EDIT 2-D | small | S | low | build |
| C | THE MAP RISES: inked hills lift into 3-D | RUN marcher with MAP's height field | ≈ 190 | L | high (a morph) | replace with the ink-pass dissolve (§5.2); test if time |
| C | THE UNMAKING | ACCORD `ring.py` | ≈ 240 | M | medium | build |
| C | ink write-ons and fire-letters | EDIT titles | — | S | low | build |

**Divergence.** B is the best-diverged cut in the room: no embers act, no thinking fire, no council, no future frame;
one range, one night, natural light. It will look like a different film because it is made differently. A and C
share more than the treatment admits. Both use the red-under-glow reveal, shepherd and Run, and C's Fellowship is A's
Crossing set with nine lanterns for twelve. C's difference lives in the book and its pages, so C's middle would look
like A's. The fix is to render C's mountains through an ink pass (§5.2).

**Ambition and taste.**
* THE CROSSING (a roped line of lanterns on a knife-edge; the sky wheels; the watch-fires burn low and are fed; the red
  in the cloud goes out patch by patch) is the best single 3-D image in the four treatments for the core message. It
  is built from cheap parts (a crag, 20-px figures, light sprites, a star rotation, a decaying trail buffer) in a
  renderer that has shipped. Its one design demand: the Run world's peaks are a forest of similar spikes, so the
  knife-edge has to be a *designed* arête, shot on a long lens.
* THE RED UNDER-GLOW, one emission term on the cloud deck, is the smartest production idea in the room. It carries the
  hidden race through four shots for a few lines of code and a re-render.
* B's THREE WAVES (the light leaving each peak at dusk, fires catching on the same peaks, the sun returning and each
  fire paling as it arrives) is a structure you can *see*, and it is `dawn.py` run backward and then forward, the
  cheapest mountain renderer we have.
* C's Red Book is the most knowledgeable Tolkien in the room, and ink pages cost about a second a frame.

The risks: B puts the heroine in four new setups (a climb at 60 px, a close-up of hands and a pot as the ember dies, a
32-second seated vigil, a standing silhouette at dawn), and only the hands use her accepted asset. THE MAP RISES is C's
boldest shot and a real morph risk. Seven ink pages are too many for one lane; four carry the argument. THE KEEPERS
RISE should stay as lights (the treatment's own fallback), not Blender figures composited into the numba world.

**Music.** Three clear idioms, including the cheapest B score in the room (cello, strings and far horns; no drums, no
glass, no piano). A accelerates by doubling its rhythm at a fixed 72 BPM, so the frame grid holds. C's chamber-versus-
orchestra split is its book-versus-fire split: each burn-through is a change of orchestration, never a hit.

**Feasibility 6 · visual ambition 8.**

### 2.3 ESSAYIST: A · NO ONE RUNS (4:42) · B · THE OPEN HAND (3:54) · C · THE LAST PAGES (5:20)

**Reuses** the most: the embers act in all three cuts (A: the glyphs, the ignition, the globe and the grasp; B and C:
the whole race), the table in all three, the five places in all three, the globe's answer, DAWN_C and the coda.

**Work lands on** EMBERS (heaviest, because of a new running-figure system), RUN (an ascent to orbit, a vigil, a
daylight medium shot), HILLS (stills, a dead hill, a daylight valley), MAP (five pages), ACCORD (three tables and a
melt) and MONTAGE (the Mirror's water). No Blender.

**New builds**

| cut | shot | lane / renderer | frames | effort | risk | verdict |
|---|---|---|---|---|---|---|
| A | THE DEAD HILL: the opening hill, same lens, no people, no fires, ash falling, a dark Moon | HILLS variant | ≈ 150 | S | low | **build**: the best cheap stakes image anyone wrote |
| A | the customs, as stills in which only the fire moves (a procession; hands passing a torch) | HILLS 2.5-D | ≈ 430 | M + M | medium | build if A is this A |
| A, B, C | THE PROMISE: for one breath, a horizon with a sun on it, in embers | EMBERS | ≈ 100 a cut | M | medium | build one version |
| A | THE TORCH RACE: ember runners on an FK run cycle; the grass catching behind them | EMBERS, a new character system | ≈ 580 | XL | very high | **cut**; the cheaper twin is THE EDGE |
| A | THE ASCENT: summit to orbit in one move, the join hidden in the cloud | RUN + GLOBE | ≈ 640 | XL | high | test only; its fallback is a cut |
| A | THE LONG DAWN on the hill: the valley in daylight, fields, a river, a town | HILLS cards in daylight | ≈ 530 | L | high (kitsch) | restrained: the future only in the sky |
| B | THE WAITING: a locked wide; the stars wheel; she feeds the fire | RUN + the heroine at ≈ 120 px | ≈ 550 | M–L | medium | build |
| B | HER DAWN: a medium shot, warming her hands in the sun | `dawn.py` + HEROINE | ≈ 384 | L | high (daylight shows the doll) | silhouette only |
| C | THE INK PAGES ×5 | MAP ink | ≈ 1,000 | L | medium-high | build three or four |
| C | THE MIRROR: a basin of dark water; the lands burning, a golden dawn, the Eye | the SEA shot's water + composited plates | ≈ 190 | M | medium | test: lovely, and the vertigo of futures |
| C | THE RING MELTS | ACCORD `ring.py` | ≈ 120 | M–L | medium-high | build |
| C | THE EYE FALLS, played as an exhale | EMBERS `tolkien.py` | ≈ 240 | M | medium (an explosion reads as victory) | build |
| C | the letters lift off the Ring into the glyph field | EMBERS | ≈ 240 | S–M | low | build |
| C | THE LONG RUN, 330 frames | RUN | 330 | M (CPU) | low | build |

**Divergence.** A's first movement is the boldest cheap idea anyone proposed: stills in which only the fire moves,
like *La Jetée* lit by fire. It needs no new renderer (one plate held, a fire-only pass per frame), and it makes A's
tempo its argument. B is not distinct: its first 65 seconds are the embers act (the thinking fire, the towers, the gold,
the storm, the cracking world, the hand), which are A's and C's pictures without the words. That was draft 1's B, and the
producers named it. The "one travelling spark" match-cut chain is lovely and cheap, but it is a way of cutting, not a
look.

**Ambition and taste.** The two biggest 3-D ideas in the room are here, and both are the riskiest kind of shot.
* THE TORCH RACE needs a new character-animation system. Running is the hardest motion to make read, and the
  red team's "LED dots" note on the Run applies twice over to a crowd.
* THE ASCENT (one move, two renderers, a join hidden inside the cloud) is the most technically ambitious shot anyone
  proposed. Its payoff is the orbital globe, the image the red team called a keynote slide.

THE DEAD HILL is worth building in any version of A. HER DAWN is where the doll shows; stay backlit. THE MIRROR is a
lovely, moderately priced Tolkien beat made from the sea shot's water. *Party trick* and the grey pointed hat at the
council are the room's two biggest cringe risks in C.

**Music.** The lightest load (13:56). A's race is the only tempo change anyone proposes (72 → 96 → a feel of 144). The
score engine has never changed tempo, and every beat-locked surge would have to be re-keyed to 15- and 10-frame beats.

**Feasibility 7 · visual ambition 6.** The least likely to fail, and the least ambitious where the producers asked for
ambition: divergence.

### 2.4 CONTRARIAN: A · EVERY STEP CLOSER (5:10) · B · THE KEEPER (5:20) · C · THE LAST PAGES (6:40)

**Reuses** less, on purpose: it retires the intro hill and the orbital dawn. It keeps the embers act (A as it is, B as a
dream, C in gold), the first fire (a slower take), the shepherd, the Run, the five places, the globe's answer (B),
`map_C`, the table (A's rim, C's council), DAWN_C and the coda (B).

**Work lands on** every department at once: RUN (a false dawn, a lifetime of plates, an ink pass, a crowd, a valley, a
dawn), EMBERS (a new crater, a brink, ash), HILLS and HEROINE (five ages, a road west, a year of plenty, the slower
take), MAP (a book), ACCORD (a rim and a council) and MONTAGE (a ship). No Blender, though the cage borrows MOUNTAIN's
Blender basket as a design.

**New builds**

| cut | shot | lane / renderer | frames | effort | risk | verdict |
|---|---|---|---|---|---|---|
| A | FALSE DAWN: a cold glow at midnight beyond the ranges; the stars near it go out | RUN + a sub-horizon emitter | 480 | M | medium (reads as city glow) | optional; lovely |
| A, B, C | THE PROMISE: terraces, orchards, a river and roofs drawn in embers (C: in ink) | EMBERS + MAP | ≈ 360 | M | medium (a screensaver) | build one version |
| A | towers lit only on the side that faces the fire; later, their backs lit by the watchers | EMBERS shading | extend | S | low | **build** |
| A, B, C | **THE EDGE**: a crater of fire; the towers lean over its rim; each surge gilds the nearest while the rim crumbles | EMBERS: new rim geometry + a gold term + falling debris | ≈ 550 (A) | L | medium-high | **build for A: the signature** |
| A, B, C | THE BRINK: a tower's crown breaks off; the camera tips over the rim | EMBERS camera + a rigid point cloud | ≈ 240 | M | medium | build |
| B | ASH: the dream's world burns to a cinder | EMBERS | 240 | S–M | low | only with this B |
| C | THE RING FALLS | EMBERS `tolkien.py` | 160 | S | low | build |
| A | DARK ADAPTATION: after the black, the stars come back | a sky plate | 240 | S | low | **build**: nearly free |
| all | the first fire as one slower take | HEROINE keys | ≈ 320 | M | medium (waits on HEROINE) | build if HEROINE lands |
| C | RIDDLES IN THE DARK: the Ring in the snow by her knee; her hand closes on it | HEROINE hand + `ring.py` | ≈ 100 | M | medium | **build**: hands only |
| all | the reveal re-plated; in A, fires catching on every ridge at once | RUN | ≈ 240 a cut | M | low-medium | build |
| B | **THE LIFETIME SHOT**: sixty years in one locked frame | RUN, about eight plates + composited layers; her silhouette at five ages; a growing cairn | ≈ 2,560 | XL (design) | medium-high (a slideshow) | **B's signature, if B is THE KEEPER** |
| B | the far peak: a parent and a small child who points | MONTAGE s1 + new figures | 120 | M | medium | only with this B |
| C | **THE LIVING INK ILLUSTRATION**: the Run as a moving Tolkien drawing | RUN depth/normal/slope passes + hatching anchored to the terrain + MAP's paper | ≈ 480 (+ the reveal) | XL | high (strokes must not boil) | **build: C's signature** |
| A | THE RING OF WATCHERS: fires on every summit around the crater's glow | RUN | 240 | M | medium | build if A uses it |
| C | THE RED BOOK | MAP 2.5-D | 320 | M | low-medium | build |
| A / C | THE RIM (shadows cast as towers) / THE COUNCIL (the peoples as silhouettes; the fist that won't open) | ACCORD | ≈ 480 each | M / L | medium / high (costume) | A optional; C simplified |
| A | THE CARRYING and THE KNIFE-EDGE: 200–400 torchbearers, a cage on long poles | RUN + an instanced crowd | 960 | XL | very high | **cheaper twin**: the dramatist's roped lanterns |
| A | THE LONG DAWN from the summit | `dawn.py` | 640 | M | low-medium | build (or the dramatist's ridge) |
| A, B | VALLEYS AT MORNING: terraces, orchards, a river, roofs | RUN: a new valley function, trees, roofs | ≈ 500 | L | high (a utopia poster) | haze and smoke only |
| C | THE YEAR OF PLENTY: daylight hills, a silver-barked tree, round doors | HILLS + ink wash | 480 | L | medium-high (Shire pastiche) | make it an ink page |
| C | THE ROAD WEST | HILLS | 400 | M | medium | cut |
| C | THE HAVENS and THE EMPTY SEA: a grey ship, farewell fires along the coast | MONTAGE sea + a ship + ink wash | ≈ 1,100 | L | medium | cut, or make it an ink page |

**Divergence.** The clearest system in the room: EMBERS AND BLACK (A), STONE AND SKY (B), INK AND GOLD (C), each with its
own camera grammar and texture, on three time-scales (a night, a lifetime, an age). Its best tool is THE LIVING INK
ILLUSTRATION: render the Run world's depth, normals and slope, hatch along the fall line with strokes anchored to the
terrain (so they cannot boil), and keep the fires in full colour with a gold-leaf bloom. A marcher knows the world
position of every pixel, so terrain-anchored strokes are natural here, not a hack. One investment makes the shared
world C's own.

**Ambition and taste.** The best source of ideas in the room.
* THE EDGE is the single best picture of Update 4 in any treatment: a bomb that prints gold, shown and never explained.
  It lives in the embers engine, at a few seconds a frame.
* THE LIFETIME SHOT (a stone a year on the cairn, a far pinprick answering, torchbearers taking the flame down, valleys
  lighting under the cloud, the ring assembling in the sky) is the most moving showpiece anyone proposed. It costs
  design, not CPU: about eight terrain plates and composited layers.
* DARK ADAPTATION costs almost nothing.

But the slate runs 17:10, its C runs 6:40, and three items do not fit this schedule: THE CARRYING's crowd, THE HAVENS,
and C's chain of endings (plenty, the road west, the Havens, the empty sea: the "too many endings" trap the treatment
itself names). B's opening dream is the embers act again, which costs B its purity. The council's Tolkien silhouettes
(long hair, a braided beard, a wide-brimmed hat, a small figure) are one step from costume.

**Music.** The heaviest: 17:10, composed blind, including a sixty-year chaconne (to be fair, the most natural form
there is for a composer that works in code: variations over a ground bass, one a year) and an elegy.

**Feasibility 3 · visual ambition 9.**

---------------------------------------------------------------------------------------------------------------

## 3. Across the four

### 3.1 Where they converge (low-risk decisions: take them)
* Every new mountain shot goes through the Run world and `dawn.py`. (All four.)
* The orbital sunrise is retired. (All four.)
* The grasp is restaged as a claw that descends palm-down, never a raised fist. (All four.)
* The incentive is gold: gold rain (poet), gold running down the towers (dramatist), gold into the runners (essayist),
  gilding at the rim (contrarian).
* The dawn is the fires paling as the sunlight reaches them (dramatist, contrarian, and the poet's smoke columns). It is
  one rule, beacon intensity against the local sun elevation, which the globe's dawn already uses.
* C is a book: the Red Book and its blank last pages (dramatist, essayist, contrarian), ink pages (the same three),
  parchment and gold leaf (all four).
* C's turn is the Ring leaving the hand and coming to the mountain (poet, contrarian).
* A slow line of lights crosses a knife-edge (dramatist, contrarian).
* "Two first" at the turn (all four). See §3.3.

### 3.2 Every expensive idea has a cheaper twin

| expensive idea (treatment) | the beat it serves | the cheaper twin that keeps the beat |
|---|---|---|
| THE TORCH RACE (E) | the race and its pay | THE EDGE (Ct), or GOLD RAIN on the existing towers (P) |
| THE CARRYING, 200–400 bearers (Ct) | pace, together, in the open | THE CROSSING, twelve roped lanterns (D) |
| FORGES IN THE LIGHT, two renderers (P) | in the light, each saw the others | the towers' backs lit by a ring of watchfires, one renderer (Ct) |
| THE MAP RISES / THE LAND BECOMES THE MAP (D, P) | the old story becomes ours | dissolve an ink page of the range into the ink-rendered range, same camera (§5.2) |
| THE RING IN THE SNOW on her face (P) | the temptation | hands and the Ring, the Ring the key light (P's fallback; Ct's RIDDLES) |
| HER DAWN in daylight (E) | the sun is the fire, returned | the wide: her silhouette against the sun, the scarf lit (E's fallback) |
| THE HAVENS, 1,100 frames (Ct) | the cost; the bearer departs | an ink page in the book: a grey ship |
| THE YEAR OF PLENTY in 3-D (Ct) | abundance, earned | the gold-leaf tree on an ink page (D) |
| VALLEYS AT MORNING (Ct), THE THAW's green (P), the daylight valley (E) | flourishing, not survival | the cloud sea thinning over valleys: hearth smoke and lit terraces half-seen in haze, and the coda's future sky |
| THE ASCENT (E) | the world answers | the Run's last wide, cut to the globe (both on disk) |
| B's shadow-play regrade (P) | B looks like its own film | a B made without the embers act (D), which needs no filter |
| seven ink pages (D) | Tolkien's own beats | four pages from one engine |

### 3.3 What the pictures could give away

The producers' caution is about images as much as words, and the one image all four share is the most decodable in
the project.

| image | where | exposure | handling |
|---|---|---|---|
| **"two first"**: two lanterns, two torches or the two tallest forges move before the rest | all four | high: a matched pair of great powers moving first is exactly the recognisable real-world mechanism the producers ruled out | keep the beat as the natural order of a ceremony: asymmetric staging, never mirrored, never two colours, never an east and a west; the others join within a beat. If a test viewer names two countries, randomise it or cut it |
| the carved oaths in A | P, E, Ct | medium: the tone critic's "seal band"; the closest thing to a poster in the film | the dramatist's choice: no oaths in A (the crossing *is* the accord, in motion); carved vows only in C, where Tolkien's world makes vows belong |
| the valleys of plenty | Ct, P, E | medium: a utopia poster, a keynote | haze, smoke and half-seen terraces; no spires, nothing futuristic on the ground |
| a ring of watchfires around the forge | Ct, P | low: it reads as myth | fine |
| glass lanterns, a fire you can see into | D | low | keep: the most tasteful trace of the north star in the room |

### 3.4 Tolkien on screen: what the references cost

The producers asked C to lean far more into *The Lord of the Rings*. The treatments show that the book frame is what
makes that affordable: a reference drawn as an ink page costs about a second a frame and can't read as costume, while
the same reference built in 3-D costs a lane and invites cosplay.

| reference | treatments | as a picture | cost | verdict |
|---|---|---|---|---|
| the Ring, its fire-letters, the Eye, the hand that takes it | all | `embers_C`, on disk | the red-team fixes | keep |
| the seven beacon-hills | D, E, Ct | the Run *already* has seven beat beacons | free | keep, and say nothing |
| the map | all | `map_C`, on disk | the fixes | keep |
| the Council, the Ring in the hearth, the vows | all | `accord_C`, on disk | the fixes | keep; the peoples as varied cloaks, not costumes |
| the Ring unmade in fire | all | a melt in `ring.py` | M | keep |
| the Ring leaves its bearer and is found by chance in the dark | P, Ct | a gold star over the range; hands in the snow | M | keep |
| the Red Book and its last pages | D, E, Ct | a 2.5-D book by a hearth | M | keep |
| the smiths' rings, the forge in the Mountain, the wise's furnaces, the Dwarves' deep, the year of plenty, the Havens | D, E, Ct | ink pages | S–M a page | pages, never 3-D |
| the Mirror | E | the sea shot's water in a basin | M | test |
| eagles after the deed | all | six eagles, on disk in DAWN_C | trim | fewer, later, farther |
| the cock crow; horns answering at dawn | E, Ct | sound only | S | keep: free and deep |
| a wizard's hat, elvish hair, a dwarf's beard, a small figure at the table | E, Ct | silhouettes from above | L | no: costume |

---------------------------------------------------------------------------------------------------------------

## 4. A and B, explicitly

**Does A have a strong, distinct concept? Yes.** Three of the four As are strong, and they agree on the spine that makes
A an allegory: a gift; a race that pays and cannot see itself; a turn from outside the race that makes things visible;
a slow, open carrying to a dawn. The strongest is the dramatist's THE CROSSING. The best race imagery is the contrarian's
(EVERY STEP CLOSER). The best formal device is the poet's (THE WATCH: two camera speeds). The essayist's NO ONE RUNS is
the most elegant as an argument and the least cinematic.

**Does B have a strong, distinct concept? Yes: two.** The dramatist's THE LONGEST NIGHT and the contrarian's THE KEEPER.
Both are wordless, both are driven by physical acts, and neither needs the allegory's pictures. (THE KEEPER's opening
dream does use them; it should be cut, or replaced by THE LONGEST NIGHT's dusk and dead ember.) The poet's THE STOLEN SUN
is a strong solstice myth, but its first half is A's embers act recoloured, so it meets Update 2's visual half only if
its shadow-play look is built natively rather than filtered. The essayist's THE OPEN HAND is draft 1's B.

**So there is no case for consolidation.** A and B differ in plot (a race and its undoing; a woman keeping a fire
through a night, or a lifetime), in picture (embers and night; stone and sky) and in sound (glass, taiko and a walking
pulse; a cello and far horns).

---------------------------------------------------------------------------------------------------------------

## 5. The most achievable path to the best film

### 5.1 The three films

**A · THE CROSSING (about 5:00).** The dramatist's A, with four grafts and one cut.
1. The frame line; THE HILL; the child's question. *(on disk, with fixes)*
2. The glyphs, slowed; IGNITION; THE GIFT. *(extend; one new embers build)*
3. THE FORGES: the towers, gold running down their faces, a spark on dry grass. *(extend)*
4. **THE EDGE** replaces the middle of the race: the crater opens, the towers lean in, the nearest gild, the rim
   crumbles. *(contrarian; new)*
5. The storm, the cracking ember globe, THE GRASP restaged. *(extend)*
6. **THE DEAD HILL**, four seconds, no words. *(essayist; new, small)*
7. Black; the last ember; the builders' couplet. *(on disk)*
8. THE FIRST FIRE; THE REVEAL with the red under-glow, **and fires catching on the far ridges in the same breath as
   hers.** *(on disk; extend; contrarian and poet)*
9. THE SHEPHERD and THE RUN with the red under-glow; the watchers (city, ice, sea). *(extend)*
10. THE KEEPERS RISE, as lights; **THE CROSSING**; THE LONG DAWN on the ridge. *(new)*
11. The coda and the title. *(on disk, with fixes)*

Cut: the keeper at the bellows (a doll risk for four seconds). Keep the carved oaths out of A, as the dramatist does.
If the producers want the flourishing more explicit than the coda's future sky, test the contrarian's valleys, in haze
only, as the cloud sea thins at the end of THE LONG DAWN.

**B · THE LONGEST NIGHT (about 3:40).** The dramatist's B as written: DUSK → THE CLIMB → THE DEAD EMBER → THE FIRST
FIRE → the reveal in moonlit silver → the shepherd → the Run → the five places → the planet → THE VIGIL → first grey →
**THE HAND-BACK** → the fire goes home → the title. No text, no embers, no red anywhere. If DESERT and KARST are not
fixed in time (the desert reads as snow; KARST's Blender rebuild is unfinished), B shows three places, not five.

*The named upgrade, if the producers want B to carry a lifetime:* THE KEEPER's lifetime shot in place of THE VIGIL, in
the same set and the same locked frame (a stone a year on the cairn, far pinpricks answering, torchbearers descending,
valleys lighting under the cloud, the ring and the Moon's lights assembling in the sky, her silhouette ageing), ending
on THE HAND-BACK and the coda's handover, which is on disk. It costs about eight more hours of design in one lane and
about 45 seconds of runtime, and it *lowers* the heroine risk, because she is a small silhouette throughout. It keeps
B free of the embers act, which THE KEEPER as written is not.

**C · THE RED BOOK (about 5:20).** The dramatist's book, the poet's and the contrarian's turn, the contrarian's look.
1. THE RED BOOK by the hearth; the elder's and the child's shadows on the wall. *(new)*
2. **Four ink pages**, drawn stroke by stroke: the smiths' rings; the Mountain's forge and the One, its letters
   visible only as fire; the wise's tower of furnaces and its felled forest (or the Dwarves' deep halls and bright
   vein); the seven beacon-hills. A fifth, the young tree in gold leaf, is kept for the end. *(new; one engine)*
3. BURN-THROUGHS between page and fire; the book heals at the end. *(new, 2-D)*
4. The embers act in C's grade: the letters lift off the Ring into the glyph field (essayist); IGNITION and THE GIFT; the
   Ring forged; the race; the Eye over every tower; the grasp on the Ring. *(extend)*
5. Black: *In the old story… / This fire could not be unmade.* *(on disk)*
6. **THE RING FALLS**: a gold band tumbling out of the black through cloud, then a gold star streaking down over her
   summit, a hiss and steam in the snow. *(poet; new)*
7. THE FIRST FIRE, with the Ring in the snow in her spark's light: hands and the Ring only. *(contrarian; new inserts)*
   Whether she refuses it (poet) or cannot let it go (contrarian) is the story panel's call; the shots are the same
   hands either way.
8. THE REVEAL and THE RUN as **THE LIVING INK ILLUSTRATION**. *(contrarian; new)*
9. THE MAP ANSWERS; THE COUNCIL; **THE UNMAKING**. *(extend; one new melt)*
10. *Optional:* THE FELLOWSHIP, nine lanterns on A's crossing set, rendered through the ink pass. Cheap once both exist.
11. DAWN (DAWN_C in the ink-and-gold grade, the eagles fewer, later and farther); THE YEAR AFTER (the gold-leaf page);
    THE LAST PAGES; the title in fire-letters cooling to ink. *(on disk; extend)*

### 5.2 One world, three renderings

The three cuts share the Run world's geometry, its shot code and its cloud jobs. They do not share its light.

| | A · night | B · natural light | C · ink and gold |
|---|---|---|---|
| **the Run world** | moonlit night; THE RED UNDER-GLOW in the cloud (the race, hidden); cold-white lanterns; the red goes out patch by patch as the crossing goes on | the day's own arc: alpenglow leaving the peaks, moonlit silver cloud, the sun returning; no red anywhere | THE LIVING INK ILLUSTRATION: hatching down the fall line, contours at depth breaks, a parchment ground; the fires in full colour with a gold-leaf bloom |
| **the embers act** | as v2: the crimson race, the gold at the edge | none | graded gold and brown, the Eye the one saturated red, entered and left through burn-throughs |
| **camera** | kinetic and cut on the beat until the grasp; after the turn, never faster than a walk (the poet's two speeds) | locked frames and slow cranes; time passes inside the frame | the epic's: lateral page moves, long flights along the beacon hills |
| **cutting** | hard cuts on the beat, then long takes | long dissolves | chapters: page, burn-through, fire |
| **type** | Cormorant italic, lower third | the title only | ink write-ons on the page; fire-letters on black |
| **score** | glass, taiko, dark strings, piano; a walking ostinato | a cello, strings, far horns; wind as the floor | chamber music for the book, the orchestra for the fire |

The ink pass also retires C's riskiest morph. The bridge from book to world becomes a dissolve from an ink page of the
beacon-hills into the ink-rendered range, seen from the same camera, with the ink giving way to fire as her beacon
catches. Both halves are ink, so the dissolve reads as the drawing coming alive, and there is no uncanny middle. THE MAP
RISES can still be tested as a standalone motion piece (Update 2 allows one) if the producers want it.

### 5.3 The new shots worth the effort

| # | shot | cut | lane | effort | new frames | why it earns the hours | fallback on disk |
|---|---|---|---|---|---|---|---|
| 1 | **THE CROSSING** | A (and C's Fellowship) | RUN-A | L | ≈ 840 (+ 480) | the core message in one image: together, in the open, at a pace we can see by; cheap parts in a proven renderer | the Run's last wide, held, with the lantern line composited |
| 2 | **THE LIVING INK ILLUSTRATION** | C | RUN-C | XL | ≈ 600 | gives C its own look wherever it touches the shared world, and replaces the morph | ink on the far ranges only, the near terrain graded warm |
| 3 | **THE EDGE** | A | EMBERS | L | ≈ 550 | Update 4's dread and its insane incentive in one picture | the v2 race with GOLD RAIN |
| 4 | **THE HAND-BACK** and **THE LONG DAWN**: one rule, two framings | B, A | RUN-B, RUN-A | L + M | ≈ 1,300 + 720 | the dawn as the fire's promise returned, long because deliberate; the cheapest mountain renderer we have | DAWN_C without eagles |
| 5 | **THE RED BOOK, the four ink pages, the burn-throughs** | C | MAP (+ ACCORD helping) | M + L + S | ≈ 1,500 | the most divergence per hour in the project; C's Tolkien lives here | the map's parchment carrying the lines alone |
| 6 | **DUSK** and **THE VIGIL** | B | RUN-B + HEROINE | M + L | ≈ 624 + 770 | B's first and second waves; B has no other pictures | the reveal wide held, far fires composited |
| 7 | **THE RING FALLS** | C | EMBERS + RUN-C | M | ≈ 430 | the most spectacular Tolkien beat per hour of work | the tumbling Ring in the embers world only |

Small and worth it: THE RED UNDER-GLOW (S, A); THE DEAD HILL (S, A); fires catching on far ridges at the reveal (S, A);
DARK ADAPTATION (S, A, if it fits the cut); THE UNMAKING (M, C); THE GIFT (M, A and C); THE KEEPERS RISE as lights (M,
A); THE CLIMB and THE DEAD EMBER (M each, B, hands and small figures only); the Ring in the snow (M, C, hands only).

New mountain frames in this plan: about 6,500 (Run world ≈ 3,700, `dawn.py` ≈ 2,650), or about 25 box-hours: under
three hours on ten boxes. The plan's cost is in the lanes, not the machines.

### 5.4 Test, and keep only if they win
* THE LIFETIME SHOT: B's named upgrade. A greybox animatic at hour 3 decides it.
* THE MAP RISES: a six-second bridge in C, or a standalone motion piece.
* THE MIRROR: C's vertigo of futures, from the sea shot's water.
* THE ASCENT: A, and only if RUN-A finishes THE CROSSING early.
* THE KEEPERS RISE with figures; THE FALL's camera dive; THE PROMISE's forms of plenty.

### 5.5 Not in this schedule
* THE TORCH RACE and THE CARRYING's crowd: new character systems for a few seconds each, where a cheaper twin keeps the
  beat.
* FORGES IN THE LIGHT as a two-renderer composite.
* THE ROAD EAST: it belongs to a B we are not making. Keep it on file; it is good.
* B's shadow-play regrade of reused renders.
* Every new face close-up: the Ring in the snow on her face, HER DAWN in daylight, THE SCAR in focus.
* THE HAVENS, THE ROAD WEST and THE YEAR OF PLENTY in 3-D. The book makes each of them a page.
* VALLEYS AT MORNING in clear light and THE THAW's green.
* The council in Tolkien costume, and the grey pointed hat.
* Pages five to seven of the seven-page book; the keeper at the bellows.

### 5.6 The twenty hours

| lane | H0–3 | H3–8 | H8–14 | H14–18 | H18–20 |
|---|---|---|---|---|---|
| EDIT | three animatics from renders on disk plus slugs; bar maps locked; a temp mix from the v2 stems; the frame-transport decision | cut tests into the animatics; first screening | assemble finals as they land; each cut's titles | grade, type, conform | masters |
| MUSIC ×3 (one a cut) + FX/MIX | the motif kit; section maps in bars | drafts to the animatics; a temp-score screening for the producers' ears at H8 | full drafts to locked bars | the mix to picture; a second listening at H16 | masters |
| EMBERS-1 | thinking-fire fix; the grasp restaged (crown, Ring) | THE EDGE: stills, then a motion test | THE EDGE renders; the Eye fix | fixes | — |
| EMBERS-2 | THE GIFT stills; the gold layer | THE RING FALLS (a); the crown and Ring fixes | renders | fixes | — |
| RUN-A | THE RED UNDER-GLOW; THE CROSSING greybox | THE CROSSING motion test; THE LONG DAWN; far-ridge ignitions | cloud renders | fixes | — |
| RUN-B | a DUSK still; THE VIGIL greybox (or THE LIFETIME SHOT's) | DUSK, THE CLIMB, THE VIGIL tests | THE HAND-BACK; cloud renders | fixes | — |
| RUN-C (ink) | depth/normal/slope passes; hatching stills | a 24-frame ink Run at half resolution, judged in motion | the ink Run and reveal; THE RING FALLS (b) | fixes | — |
| HEROINE | review v2b; a still of THE DEAD EMBER | THE DEAD EMBER; the vigil poses; the Ring in the snow (hands) | cloud renders | fixes | — |
| MAP | THE RED BOOK set; two pages in pencil | four pages; burn-throughs | `map_C` review and fixes; renders | fixes | — |
| ACCORD | `accord_C` fixes | THE UNMAKING | renders | helps MAP | — |
| HILLS | intro and coda fixes | THE DEAD HILL | renders | helps HEROINE | — |
| GLOBE / MONTAGE | ice, sea and desert fixes; the planet for B | renders | — | — | — |

Three rules make the lanes safe.
* **Fork the shot code per lane at hour 0** (a crossing module, a dusk-and-dawn module, an ink module), and let
  `world.py` take only additive flags after hour 1 (the under-glow, the extra render passes), so no lane breaks
  another's frames.
* **Launch every fixed re-render of a reused shot by hour 2**, on the cloud, so the machines are busy while the new
  shots are still in look-dev.
* **Keep the Mac for the edit and the music.** Everything else renders on the cloud, a whole shot per machine type.

### 5.7 The music

* **Runtime:** 124 s today; about 14 minutes across the three cuts (A about 5:00, B about 3:40, C about 5:20). That is
  still seven times today's score, composed by hand and blind, which is why no cut should run longer than its idea.
* **Reuse:** the first half's cues, stretched by whole bars (the kindling, the race, the riser, the silence) in A and C;
  score C's second-half material in C; the motif kit everywhere.
* **New, and written as rules the composer can check by analysis:** A's gilding shimmer at THE EDGE, its accelerando by
  doubling, the crossing's walking ostinato (the harmony moves only when the line passes a watch-fire), the dawn's
  bloom. All of B, sparse by design: a cello, strings, far horns and wind, and at the hand-back the horns falling silent
  one by one as the beacons pale. C's chamber book, the Ring's loop that cadences once (at the melt), the map's wonder
  and the dawn.
* **The rules the critics earned:** no choir; no organ manuals; no borrowed melodies; the effects own every fire;
  nothing lands on a vow; no stroke on a sunrise; every dawn a bloom out of silence, below the race's loudness.
* **Lanes:** three composers sharing one motif kit, plus one effects-and-mix lane. Bar maps by hour 3, full drafts by
  hour 12, the mix from hour 14. The producers listen at hours 8 and 16, because nobody else can.

### 5.8 Gates and kill rules
* **Hour 3, stills.** Any new shot whose still reads as a doll, clip-art, a poster or a tech demo goes to its fallback
  now.
* **Hour 8, motion.** A 48-frame half-resolution test, cut into its animatic, must beat its fallback, or it goes. The
  bar maps lock.
* **Hour 12.** No full render starts unless its fallback is worse than nothing.
* **Hour 14:** every shot in at least one full render. **Hour 18:** the fix pass closes. **Hour 20:** masters.
* Screen each cut at hours 8 and 16 with fresh eyes and the treatments' three questions: what do the fires do? what
  was the danger? what changed? If anyone names a country pair, a treaty or an institution, go subtler, never louder.
