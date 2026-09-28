# BURN-C: real filmed burn elements on C's pages (user: "the paper burning should look far more realistic")

## STATE (BURN-C, 28 Sep ~03:05 local) -- READ FIRST
* main 02:55: "go on (a)": real flame tongues along the front (hold (b) curl and (c) ash until main has seen (a)).
  Deliver a 24-frame strip + a side-by-side (footage matte on top, footage + (a) below), then STOP and send
  "BURN (a) READY". Usage is nearly spent: no side experiments.
* (a) BUILT (ftburn.flame_tongues, `--tongues 1 --tgain`): real flames on black (the first seconds of PureRaw
  8828898 [24:96] and 8828892 [36:84], before the green shows; 640x360 uint8 bank, crossfaded loops) stood on the
  hole's contour and rising screen-up: one strip column per edge pixel, the column picked by a page-fixed coordinate
  (fx + 0.7 fy) x SU 3.5, overlapping tiles crossfaded (M 120), height 8-140 px and brightness by the front's speed
  (the union swept over 4 clip frames), short on near-vertical edge runs (else they stack into streaks).
  DONE 03:14 (full 841-1039, scratchpad fullA): BURN (a) READY sent. Review: _local_logs/review/burn_C/A_* and
  C5_A_matte_vs_tongues.mp4 (top = footage matte, bottom = + (a)). Fixes on the way: tongue bases smoothed along
  the contour (+-12 px; 1-px columns on slanted edges sheared into slivers), no tongues on steep runs, the clip's
  black level clipped (its noise stacked into faint streaks), the over-hole cores dropped (h264 blocks).
* CLEAN PLATES: job `cloud/jobs/pages_book_plate.json` (book_c `--no-burn`, pushed 339b1e3; 324 frames: 841-1039,
  1680-1717, 1905-1991 -> renders/book_C_plate + _matte). main launched the TEST (req 0928-025304-pagesbookplate-31317,
  log ~/.cache/ldfarm/req/0928-025304-pagesbookplate-31317.log; frames 900, 1700, 1930, 1985 -> renders/_farmtest/
  pages_book_plate/). I OWN THE REST: check at 1:1 (burn off, light kept, nothing else changed); if clean launch
  finals `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/pages_book_plate.json --nodes 3 --detach`
  and add a row "(lane-launched)" to _local_logs/handoff/LAUNCHED_BY_MAIN.md; if not, fix, push, re-test.
* The C5 test verdict (sent 02:50): PARTLY: the footage FRONT is far more real; the footage's own flames don't
  transfer; hybrid recommended -> (a) now.
* Review files (not in git): `~/mishamisha/_local_logs/review/burn_C/` (SHEET_compare.jpg: now vs footage at 12
  frames; STRIP24_866-889.jpg; STILL_898_975.jpg; C5_burn_now_vs_footage.mp4: top now, bottom footage, 841-1039).
  Frames: session scratchpad `burn/full3/` (199 JPGs; delete when superseded).
* Footage: `assets/burn_footage/` (7 clips + the 4K of 8828893, ~175 MB; mp4s git-ignored; LICENSES.md committed).
* Code: `shots/map/ftburn.py` (committed 7f6ffdd + 72f9625):
  `geo`, `probe`, `c5 --frames --out [--speed --gain --pool --grade --e15 --layers]`. Page plate = book_C f840
  reprojected per frame through cam_letters (a leaf ray-cast in numpy); footage -> leaf by a similarity (K_CM 0.0106
  cm per 4K px, birth (2545,1425) pinned to Fw, footage-up = the page's far side); hole = running union of a green
  key that also takes the flame's warm-white wash next to the hole; fire light = the flames + ember rim over the
  BLACK SHEET only (real emission; despilled; warm only) + a 3 px glow over the edge; the hole is black; char = the
  measured ramp (Pixabay 189188) by distance from the edge (to ~0.6 cm) x page-space noise; the flames light a pool.
  Retime: clip frame 18 (0.72 s) = C 841, then real time (25 -> 24 fps): the hole opens 841-842 (SOUND-C's 842).
* 8 min per 199 frames through `RENDERQ_SLOTS=2 python ~/mishamisha/_local_logs/renderq.py -- ...` (main's rule).
* SCOPE (main 02:05): the second half (the Mirror, 2080, onward) is being rewritten: HOLD #17/#19 and any second-half
  page burn until main sends the new shot list. First half goes on: #4 841-1039, #6 1680-1717, #7 1905-1991.
* Rollout (if main says go): farm-render CLEAN plates (book_c with the burn off: a guarded flag) for 841-1039,
  1680-1717, 1905-1991; ftburn comps -> `renders/book_C_ft` + `_matte` (same conventions as book_C, so EDIT's
  book()/matte comp read them unchanged); other clips for #6/#7 (8828892/8828897 sweep, 8829001/8828896 hole; 2.5-3x
  with frame blending); SOUND-C re-measures (hole-open frames stay 842 / 1686 / 1921-1924; C5's page now lingers to
  ~1020, not ~900).

## Log
* 01:30 read COMMON (SAFE MODE), NOTES_PAGES (24-row spec), EDIT NOTES (kind `burn`: out = O*keep + I*(1-cover) + glow,
  display space). Burns in C: #4 C5 840-1039 (book_C, baked), #6 1680-1717 X1 reversed (book_C), #7 1905-1991 the
  Deep -> the Eye (book_C + EDIT matte comp), #17 4150-4185 and #19 5594-5640 (x1burn.py layers, EDIT `burn`).
* Current C5: hole born ~843 at the heart, a smooth radial front fills the frame by ~900 (2.4 s); a thin orange rim,
  no flames on the page (e15's single candle flame stands at the heart 830-1039, EDIT add).
* 02:10-02:48 the C5 test, five passes (t1-t7, full3): (1) the green key missed the flame-washed hole -> cream sheets;
  (2) the over-hole flame from subtraction became a flat orange fill with the torn edge printed on it; (3) warm
  excess, then un-greenness: still a flat fill; (4) dropped the over-hole flame: the hole is black, fire light from
  the sheet side only; (5) a grey band = the flame's warm-white wash keyed as sheet -> keyed as hole when next to it.
