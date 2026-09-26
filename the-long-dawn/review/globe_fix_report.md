# GLOBE fix (globe_fix): report

Department: GLOBE (`shots/globe/`). Answers the picture red-team (`review/picture_redteam.md` on
`claude/critique-picture`): §11 and top-8 #1 (THE WORLD ANSWERS), §14 and top-8 #3 (DAWN).
Code: `claude/fix-globe_fix`. Frames: `claude/render-globe_fix` (rendered in full on the cloud
with `cloud/jobs/globe_fix.json`). Sheet: `review/globe_fix_before_after.jpg` (left of each
pair is the v2 cloud render, right is v3). Frame numbers are src (v2 edit = src + 160).

## 1. THE WORLD ANSWERS (A/B, 1752–1935)

**The problem.** The spread could be read three wrong ways, and cut B has no words to steer it:
war (an orange comet flying off the subcontinent over the Arabian Sea at 1860 is a launch), the AI
itself spreading (two short comets leaving each new fire make forks with dots at the tips: a neural
net), or a corporate slide (evenly beaded lines, then an even sprinkle of dots).

**What changed** (`fires.py`, `shots.py`, `globe.py`):
* **Nothing flies.** The comets, embers and ocean leaps (`F_LEAP`) are gone from the picture. The
  unit of the spread is a fire catching: a watcher sees the fire on the next height and lights his
  own. A sea crossing is a pause, and then the far shore simply catches.
* **One child per hop.** A new deterministic schedule (`FireNet._spread_hops`, derived from the
  shipped `fires_net_v8_3.npz` and shipped itself as `data/fires_hops_v8_3_h1.npz`). Each fire
  passes the flame to ONE next site, the cheapest that goes on in the same direction (crest links
  first), so the fire runs as lines along ranges, rivers and coasts. A fire's other neighbours
  answer it only after the line has moved on. In the shot, two children of the same fire never
  catch less than 3.5 frames apart (median 7), so there is never a fork at a tip.
* **Each fire is a flame, never a dot.** Each fire is a small standing flame: a tapered stroke
  along the local vertical as the camera sees it. It is ~2.5–5 px tall at full res, taller when
  the camera is close, and upright near the limb. Its glow follows the flame up, so it is never a
  round orb, and its light still falls on the moonlit snow and cloud. The tip of a line is simply
  its newest flame: no head, no spark burst, no dot.
* **Irregular, and carrying time.**
  * Brightness spreads about 3:1 (each fire's size ranks it; a few great fires go past that; the
    lonely fires between the lines are dimmer).
  * 30% of the crest sites beyond the first beacon's own hills stay dark, in clustered gaps, so
    the lines break into runs. The lit crest fires stand on summits and passes up to 22 km either
    side of the crest line.
  * A catching fire flares as a fat, bright burst of flame (~3 frames). It burns high and yellow
    for about half a second, then settles over ~2 s to a lower, redder fire, so the newest fires
    lead every line.
  * Tended fires breathe slowly (4–10 s). A beacon's village fires join 10–30 frames after it.
* **Warm means fire.** The city lights are 1.5 stops lower and a pale blue-white at the same
  luminance, in both shots.

**Beats kept.**

| frame | beat |
|---|---|
| 1752 | the first beacon burns alone |
| 1760 | it flares (the range wash and spark stream kept) |
| 1766 / 1771 / 1777 | the old "throws" beat, now the first answers: west along the crest, east, then the Ganges |
| 1766–1800 | two lines run along the Himalaya, a hop every ~7 frames at first |
| 1800–1840 | the Himalaya catches (all 29 kept crest sites lit by 1840) |
| 1840–1900 | the acceleration (37–58 visible ignitions per 10 frames) |
| ~1910 | the view is studded; nothing new catches after 1911 |

Also kept:
* The camera move.
* Geographic placement: no continent is dark.
* Determinism: every draw comes from the shipped npz files. The pace is now an explicit clock,
  because v2's s^0.35 law started too slowly without the flights.

## 2. DAWN (A and B, 2232–2495; `--both` into `renders/globe_v2` and `renders/globe_B`)

**The problem.** The hit frame (2240) was a grey-beige veil over half the frame, with stars inside
the sun's glare. After it, a static white star with a hairline streak sat over a keynote limb.

**What changed** (`shots.Dawn`, `lens.py`):
* **No stars in the light.** The stars die under the glare, masked by the glare's own
  brightness, and are gone from the sky by 2240 (they fade out over 2234–2240).
* **A gold flash.** Everything in the lens layer except the disc's own white-hot core is now gold
  (`HOT / GOLD / AMBER`). The flash is a compact, intense gold burst with a small amber veil, so
  the space round the sun goes gold, never milky grey. It lingers ~9 frames (v2: 5), so the edit's
  white hold (v2 2399–2402) clears to gold and then to the sunrise. The morning light is warmer
  too (`DAWN_SUN`), and the late anamorphic line fades to 30%.
* **The terminator keeps coming.** The lighting lead runs 0° → 34° (v2: 26°) and keeps moving
  through the whole shot. The day crosses Arabia, the Horn and the Sahel toward us, and by 2495
  only the nearest land is still in night.
* **The fires pale one by one.** Every answering fire from THE WORLD ANSWERS burns on the night
  side, drawn with the same flame. Each pales as the day reaches it, at its own moment: a per-fire
  threshold of ±3° of sun elevation and a ~1 s fade, with its light on the land going first. This
  handover from the beacons to the sun is what makes the shot particular.

**Kept.**
* The orbit and the 'aden' framing.
* The sun breaking the limb at 2240 at screen (981, 352), for the match-cut from the hearth.
* The camera rise, the atmosphere and the weather.
* Cut A's text calm, 2343–2449.

## 3. Remaining weaknesses
* **Scale.** At orbital scale a flame is 2.5–5 px. On a big screen the flame shape, the flare and
  the settling carry it; on a phone the fires read as warm points. I judged motion from
  consecutive-frame strips, not from playback. Someone should watch the shot for the flicker
  (it is a 1–4 Hz breath plus a small fast lick on the light only).
* **Lines are still lines.** The Himalaya and Kunlun are ranges, so their fires still form
  lines. The gaps and the brightness spread break them into runs, but on a small thumbnail a run
  can still look beaded.
* **Missed escarpments.** The crest detector still misses escarpments (Western Ghats, Great
  Dividing Range, Urals); coasts and rivers carry those regions.
* **Still an orbital sunrise.** The critic's first choice was a mountain dawn from the Run's world
  (DAWN_C without the eagles). That belongs to the MOUNTAIN department. This fix keeps the orbit
  as the brief asked and makes it gold, starless in the glare and moving, with our fires handing
  over to the day.
* **Not addressed here.** The critic's §12 (cut C's map: dendrites and dot-density world) belongs
  to the MAP department.
