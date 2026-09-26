# MAP (cut C) fix, "And hill by hill, the peoples answered." (v2 1920–2087)

* **Frames:** 168, rendered in the cloud by `cloud/jobs/map_fix.json` into `renders/map_C/` and pushed to
  `claude/render-map_fix`.
* **Code and notes:** `claude/fix-map_fix` (`shots/map/relay.py` is new; `render.py`, `fire.py`,
  `features.py` and `NOTES.md` changed; `webmap.py`, `answer.py` and `places_globe.py` deleted).
* **Sheet:** `review/map_fix_before_after.jpg`. The left of each pair is the old code, re-rendered here
  at the same frames.

## What changed and why (red-team §12 and TOP 8 #1)

1. **The unit of spread is a flame.**
   * `relay.py` replaces the thread web. Each fire lights one fire further along its line. The next fire
     stands on the summit of a drawn peak or the crown of a drawn hill.
   * Hops are 1.6–5.4 map degrees, so the spacing is irregular and some peaks are skipped.
   * Nothing is drawn between fires: no threads, no travelling fronts, no forked tips and no dots.
     The old point flashes at ignition are gone as well.
2. **No forks.** A second line may leave a fire only once that fire's own line has moved on two hops.
   No fire lights more than two (checked: 14 second lines, 0 forks at a tip).
   * The first beacon is the exception. It lights three: its fixed first answers at 1946, 1951 and 1954.
3. **Lines along real ranges.** Each line keeps to a ±1.8° corridor around a range and leaves the frame:
   * west: Himalaya, Karakoram, Hindu Kush, Elburz, Caucasus and Anatolia;
   * east: Hengduan, Qinling, Taihang and Korea;
   * south: the Ghats and Sri Lanka;
   * Myanmar, Indochina and south China;
   * north: Tian Shan, Altai and Mongolia;
   * Makran, Oman, Yemen and the Horn;
   * the Zagros and the Levant.
4. **No comets over water.** `LEAPS` is gone with `webmap.py`. A strait (Hormuz, Bab-el-Mandeb, Palk,
   Tsushima) is crossed by a pause, then a fire on the far shore.
5. **The newest fires flare and the older ones settle.**
   * A fire catches in ~2.5 frames and flares for ~6 frames: taller, brighter, and throwing its light
     wide.
   * Over ~24 frames it then sinks to its own resting level, lower and redder.
   * Across the settled fires, brightness spans ~2.5:1 (p90/p10), ~4:1 at the extremes. The flaring
     fires sit above that.
6. **A regional end with legible ink.**
   * The crane goes from 42 to ~79 map degrees wide; it used to go to 372. It ends over Iran, Arabia's
     south coast, India, the Himalaya, Myanmar and western China.
   * At ~24 px per degree the peak glyphs and their hatching stay legible. Flames are 9–14 px when
     settled and ~20–30 px when flaring; the first beacon is ~33 px.
   * Line pace is balanced so that both edges are still catching at the end. New fires in the left and
     right thirds: 7/8 over 2020–40, 9/7 over 2040–60, 6/3 after 2060. No region is visibly last.
7. **Grade.** The hearth burns down further and the moonlit fill rises. At the end, unlit paper is a
   neutral moonlit grey (≈57,54,53), so the map is warm only where fire stands (cross-shot grade note).
8. **Nobody left dark.** The relief proxy drew almost no heights in India, so I added 52 hand-traced hill
   marks: the Western and Eastern Ghats, Vindhya, Satpura, Aravalli, Chota Nagpur, Sri Lanka and
   Sulaiman/Kirthar.
   * They are placed last, with their own seed, so every other mark on the sheet is where it was.
   * The sheet needs a re-bake, which the job does if the cache is missing.
9. **Bug fix.** The light-pool irradiance was divided by an extra cell²/(2πσ²). That made every fire's
   pool ~1000× too weak and its brightness depended on the camera.

## What I kept

* **1920–1945 is unchanged:** the close view of the Himalaya, one flame, its flare, the spark fountain
  and the table-top DoF. The mean difference at 1920 is 0.6/255.
* **The timing:** first answers at the old first landings; the line window 1960–2040 with its hushed
  lower third; the cut points 1920 and 2087; the dissolve, with light gathering to the centre and the
  vignette rising from 2050.
* **The sheet itself:** parchment, ink, folds, coasts, rivers and glyphs. The only additions are the
  hills in India.
* **Motion:** frame-to-frame differences over all 167 cuts show no pops or spikes. The largest is the
  designed flare at 1922.

## Deliberately not done

* **No scorch line behind each hop** (the critic's suggestion). It would draw the relay's branching and
  bring back the dendrite reading. The order of the catches along each range carries the line.
* **No dotted ship-lines.** No open-sea crossing is in frame, only straits.

## Remaining weaknesses

* At thumbnail size (≤1/3) the end still reads as warm points strung along the ranges. At full size each
  is a flame with a flicker, and there are ~90 in frame, not thousands.
* The fold crease at 71°E runs through the left-centre of the last framing.
* Motion was judged from stills and difference statistics, not playback. Watch the cadence of catches
  during the line.
* India's hills are hand-traced (~1°). The Horn is only reached in the last second, at the frame's
  corner.
