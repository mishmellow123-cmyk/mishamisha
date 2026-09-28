# BURN-C: real filmed burn elements on C's pages (user: "the paper burning should look far more realistic")

## STATE (BURN-C, 28 Sep ~02:00 local) -- READ FIRST
* Route: REAL FOOTAGE FIRST (user's call); procedural only if it doesn't look good enough. ONE solid test, then report.
* Test target: C5 burn-through 841-1039 (book_C has PAGES' burn v2 baked in; e15's flame is added by EDIT).
* Footage: `assets/burn_footage/` (7 clips + the 4K of 8828893, ~175 MB; mp4s git-ignored; LICENSES.md committed).
  The test uses Pexels 8828893 (PureRaw: black sheet on green, hole born at (0.66, 0.66) at 0.72 s). Pixabay 189188
  (a lit kraft sheet burning) gave the char band's colour ramp (ftburn.CHAR).
* Code: `shots/map/ftburn.py` (mine): `geo` (Fw on screen: 962,458 at 841; the render's hole is born at 955,465),
  `probe`, `c5 --frames --out [--speed --gain --pool --grade --e15 --layers]`. The page plate = book_C f840 reprojected
  per frame through cam_letters (the leaf ray-cast, numpy); footage -> leaf by a similarity (K_CM 0.0106 cm/px,
  footage-up = the page's far side); hole = running union of the green key; flames = light over the backdrop (colour
  rebuilt from R over the green); char = measured ramp by distance from the hole edge (cm) x page-space noise; the
  flames light a pool on the paper.
* 02:05 main: don't bypass the memory guard; run tests as `RENDERQ_SLOTS=2 python ~/mishamisha/_local_logs/renderq.py
  -- <cmd>` (the free slot, still waits for 1.5 GB). The duplicate waiter is cancelled; the test is queued on slot 2.
* SCOPE (main 02:05): C's SECOND HALF is being rewritten (the Mirror, 2080, onward): HOLD the map X1s (#17 4150-4185,
  #19 5594-5640) and any second-half page burn until main sends the new shot list. First-half burns go on: #4 C4/C5
  letters 841-1039, #6 1680-1717 (the sweep into the Deep), #7 1905-1991 (the Deep burns through into the Eye).
* 02:40 TEST STATE: ftburn c5 works (8 min for 199 frames incl. the renderq wait). Findings so far:
  - the footage's HOLE (edge shape, ragged fingers, irregular uneven growth) transfers beautifully;
  - its FLAMES do not: they are big defocused flames, and over the green the camera's roll-off makes them non-additive
    (the green washes pale, reads 3x brighter than over the sheet) -> any over-hole flame estimate became a flat
    orange fill with the torn edge printed on it (CG-looking). Now: the hole is BLACK; fire light = the flames and
    ember rim OVER THE BLACK SHEET only (real emission, despilled, warm only) + a 3 px glow over the edge;
  - the char band (measured ramp, narrowed to ~0.6 cm) reads as scorch + soot;
  - no crisp flames licking the edge, no visible curl (the footage's sheet is unlit black).
* RUNNING: full 841-1039 -> scratchpad full2 + cmp2/compare.mp4 (top = now, bottom = footage). Then: sheet + strip +
  verdict to main.

## Log
* 01:30 read COMMON (SAFE MODE), NOTES_PAGES (24-row spec), EDIT NOTES (kind `burn`: out = O*keep + I*(1-cover) + glow,
  display space). Burns in C: #4 C5 840-1039 (book_C, baked), #6 1680-1717 X1 reversed (book_C), #7 1905-1991 the
  Deep -> the Eye (book_C + EDIT matte comp), #17 4150-4185 and #19 5594-5640 (x1burn.py layers, EDIT `burn`).
* Current C5: hole born ~843 at the heart, a smooth radial front fills the frame by ~900 (2.4 s); a thin orange rim,
  no flames on the page (e15's single candle flame stands at the heart 830-1039, EDIT add).
