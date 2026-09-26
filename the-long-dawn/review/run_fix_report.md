# RUN — `run_fix` report (THE BEACON RUN, v2 1520–1679, all cuts)

Answer to the picture red-team's **#5, "The Beacon Run is a terrain demo, not the beacons"**
(`review/picture_redteam.md`, branch `claude/critique-picture`). Code: `shots/run/` on `claude/fix-run_fix`.
Frames: `renders/run_v2/f_01520–01679.png` on `claude/render-run_fix` (cloud job `cloud/jobs/run_fix.json`).
Before/after sheet: `review/run_fix_before_after.jpg` (left: the v2 finals from `claude/render-run-v2-*`;
right: run_fix).

The critic reviewed an older local sheet; the v2 finals on the render branches already differed (a banked
canyon run, dark islands), but its diagnosis held for them: dot beacons, speckled snow, a flat cloud floor, no
air between the ranges, fast and low. Every point is addressed below, plus defects that only showed up at full
resolution.

## What changed and why

**The beacons (`beacons.py`, `fire2.py`, `render_run.py`)**
* **Every fire is a bonfire, at every size.** The beat beacons use the pyre's tongue recipe (separated tongues,
  dark gaps, two tall ones off centre). A beat's flame is sized from its distance and the lens at its beat.
  Measured on the final frames (fire-coloured pixels, flame plus its tight glow), every beat is 26–45 px tall
  from 2 frames after its catch onward:

  | beat | +2 | +6 | +16 |
  |---|---|---|---|
  | 1540 | 42 | 43 | 40 |
  | 1560 | 38 | 45 | 45 |
  | 1600 | 32 | 35 | 28 |
  | 1620 | 37 | 37 | 32 |
  | 1640 | 29 | 30 | 28 |
  | 1660 | 36 | 40 | 26 |

  On the catch frame itself it is 11–17 px, born under the flare. Big fires breathe slower (tongue periods
  ∝ Hf^0.35), so a far fire doesn't flicker like a candle. Far fires are rendered supersampled and
  area-filtered, so even a 4 px fire keeps a hot base and a ragged tip. The old "warm point plus soft aura"
  path, the LED look, is gone.
* **The catch.** A 2–3 frame gold flare runs up the flame (tall, not round; weights 1, 0.51, 0.26), and the
  pool of light on the summit flares with it.
* **Smoke columns lit from below.** New `Plume`: the catch throws the first puffs up as a rising cap, then a
  slower buoyant column leans downwind. Each puff is lit orange from below by its own fire (inverse-cube),
  grey-blue on its moon side, and fogged by its distance.
* **A pool of light on the snow** out to about two fire heights, at roughly moonlight level (about 3× that at
  the fire's foot). It also falls on the cloud tops, where a low beacon warms the sea of cloud.
* **Fillers vary and flicker.** Each is seeded for size and for brightness (each ~2.8:1) and flickers at its
  own rate. The chains race to the horizon as small flames that shrink and fade into the haze.
* **Fires sit on their peaks.** The catalogue's 60 m grid had left far fires floating beside their summits or
  sunk into them. The heightfield marcher also steps 0.35 % of distance, so it skips thin needle tips and draws
  far peaks blunter than their geometry, and the fires ignored the Earth curvature the terrain is drawn with.
  Together these left far fires 10–25 px above the drawn peaks. Each fire is now snapped to its true summit,
  then seated on the terrain eroded by half a march step, then dropped by d²/2R. Verified against terrain-only
  renders.
* **Crisp through the pans.** Flames, glows and flares are drawn after the vector blur. Each is laid down along
  its own screen path through the shutter, over half the true path, because the eye tracks the fires. The
  reconstruction filter had been smearing small bright fires into striped "lanterns".
* **The heroine's beacon** (lit at 1360) is drawn with s1's own far-light recipe, so across the cut from s1's
  last frame it is the same light.
* **The missing 1550 filler** (SFX `run_far_1`, pan +0.8) is added in code (`beacons.EXTRA`). The catalogue
  search had never placed it. `beacons.npy` and `summits.npy` are **not** regenerated.

**The camera (`run.py`)**
* **Slower and higher before the pass.** 55–68 m/s instead of 92. The camera no longer dives off the lip: it
  rides the cliff's updraft 80–100 m over the valley, then settles to eye level with the pyre from ~1576. Bank
  is ×0.13 (was ×0.22). The pre-pass swing to the right is world-anchored at half the old peak rate, so the
  first catches aren't smeared.
* **The pyre pass is kept**: it catches at 1580, 42 m ahead, and is passed at 14 m at ~1594 with its sparks
  streaking through the lens.
* **Long lens after 1600.** The camera climbs to ~330 m while the lens eases from 54° to 42°
  (1590–1614). The great peak, the 1610 and 1620 fires and the start of 1620's chain then share one frame, the
  signal spreading across the range. Once the chain has raced away it zooms on to 29° (1632–1666), stacking the
  ranges for the last beats and chain 2. Each stage has a brisk start and a long, gentle landing in log focal
  length. Pitch holds the cloud-sea horizon at 23.5 % of frame height through the zoom. The pan is
  world-anchored at ≤26 px per frame and follows the signal right to left. Bank fades to a stabilised head.
* **1520 is unchanged** (position, yaw and pitch identical; roll differs by 0.3°), so the match cut holds.

**The world (`world.py`, opt-in through `run_light()` / `CLOUD_RELIEF`; s1 and DAWN_C are untouched)**
* **Snow by form.** Slope comes from a smooth central difference at the snow scale; before, the fine relief
  speckled it through a one-sided difference. A convexity term makes ridges, shoulders and ledges hold snow and
  gullies go bare. The noise terms are cut to a third (0.18 → 0.06, 0.06 → 0.02) and their domains are sheared
  by height, so nothing streaks down a steep face. Walls steeper than ~65° stay bare, judged on a 12 m slope
  (a rib-scale cap drew dark flutes). Near terrain uses the same rule on smoother normals, so the foothills'
  dalmatian speckle is gone.
* **The 1560–1580 "curtain".** On the pyre tower's moonlit facet edge, the convexity term (and before it the
  3 m-normal rule) laid a bright vertical sheet of snow, stretched by the heightfield and the blur. The tower's
  walls are now bare rock, with the steepness cap everywhere else. The horizontal smear of the near snowy spine
  is also much reduced by the slower speed and pan.
* **The cloud sea has relief.** Billowed tops (|noise| fBm from 640 m, 60 m amplitude) are shaded by the normal
  of their larger forms (cloud is translucent: soft heaps, not a rough rug) with a tighter wrap (0.35 vs 0.8), so
  the low moon models lit domes and shadowed hollows. Moonlit **wisps** wrap the mountains' feet where they meet
  the cloud, layered with height.
* **Haze between the ranges.** A haze layer over the cloud sea (7e-5/m at the cloud top, 450 m scale height)
  makes each range paler and bluer than the last.
* **An artefact fixed.** Rays that went below the horizon but found nothing within 90 km were shaded as "sky
  below the horizon". That drew an inverted mountain band along the top of every high, long-lens frame
  (1616–1679 in the v2 finals). They now end in haze.

## What was kept
* The match cut from s1's last frame: the same camera at 1520, with the left mountain, the pointed peak and the
  heroine's beacon in place (her glow is now exactly s1's).
* The beat timings and sound sync. Catches are at 1540, 1560, 1580 (the pyre), 1600 (the great peak), 1620,
  1640 and 1660; fillers at 1550–1670; chains from 1620 and 1660. Each beat lands on its SFX side at its
  ignition frame:

  | frame | fire | screen x | SFX pan |
  |---|---|---|---|
  | 1540 | beat, 1.0 km | 0.42 L | −0.55 |
  | 1550 | filler, 8.8 km | 0.83 R | +0.80 |
  | 1560 | beat, 0.43 km | 0.78 R | +0.50 |
  | 1570 | filler, 6.8 km | 0.30 L | −0.75 |
  | 1580 | the pyre, 42 m | 0.31 L | −0.35 |
  | 1590 | filler, 6.8 km | 0.79 R | +0.65 |
  | 1600 | the great peak, 7.5 km | 0.54 R | +0.40 |
  | 1610 | filler, 3.9 km | 0.01 L (edge) | −0.80 |
  | 1620 | beat, 3.9 km | 0.27 L | −0.25 |
  | 1630 | filler, 10.6 km | 0.82 R | +0.70 |
  | 1640 | beat, 8.7 km | 0.57 R | +0.30 |
  | 1650 | filler, 11.3 km | off-screen L | −0.60 |
  | 1660 | beat, 8.3 km | 0.27 L | −0.10 |
  | 1670 | filler, 8.2 km | 0.71 R | +0.55 |
* The pyre event, the spark fountain through the lens, the world's seeds, the moon, sky and palette, and the
  flame ramp.
* `beacons.npy` and `summits.npy`, bit for bit.
* **A fire is always in frame** (checked on all 160 final frames). From 1520 to 1539 it is the heroine's far
  beacon, as in s1. From 1540 there are always several.

## Remaining weaknesses
* **1520–1539 has only the heroine's far light.** It is small by nature (32 km, s1's look). No other fire exists
  in the story before the 1540 catch.
* **The far great beacons are a legibility cheat.** To stay ≥25 px, the far beats are nominally 70–140 m
  flames. On their peaks they read in proportion (fire about 8–10 % of the peak's visible height, like the near
  ones on their knolls), but a physics-minded eye could call them enormous.
* **Chain 1's first two links catch just off the left edge.** Following them would have needed a 35–40 px/frame
  whip pan through the long lens. The camera arrives as links 3–6 catch, and the whole chain is in frame from
  ~1640 to the end.
* **The long-lens islands are backlit** (the moon is 40° left of the view), so 1620–1650 carries a large dark
  island in the left half. The haze separates the ranges behind it, but the frame is low-key.
* **Flame silhouettes repeat.** They are one tongue model with seeded variation.
* **The pyre tower's summit is a flat shelf.** It reads as a pedestal from eye level, as it did before.
* **Motion was judged on stills only here.** The two-stage zoom, the pans (≤26 px/frame) and the half-shutter
  fires should be checked in motion in the edit.
* **Cost.** The final render took 47.7 min on 4 cores (4 workers × 1 thread). Per frame on one core that is a
  median of 79 s before 1600 (max 223 s, the first frames incl. numba compile) and 51 s after.
* **DAWN_C** (already rendered) used the old pyre position. The tower now stands ~55 m back along the track,
  behind DAWN_C's camera and invisible there; a DAWN_C re-render would follow automatically.
