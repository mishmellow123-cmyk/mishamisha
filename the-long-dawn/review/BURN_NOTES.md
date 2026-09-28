# BURN-C: real filmed burn elements on C's pages (user: "the paper burning should look far more realistic")

## STATE (BURN-C, 28 Sep ~09:45 local) -- LANE DONE (stood down)
* **BURN ROLLOUT DONE.** `renders/book_C_ft` + `renders/book_C_ft_matte` (324 frames each: 841-1039, 1680-1717,
  1905-1991), every frame on the farm's clean plates (renders/book_C_plate, landed 09:21, request 0928-091213).
  Same conventions as book_C, so EDIT reads them unchanged (C is on hold in EDIT until the new EDL):
  - 841-1039 (C4-C5): rgb = the page (premultiplied) + the fire; matte = cover; EDIT's under is black, e15 added.
  - 1680-1717 (C8 sweep): the held race (embers_C3 f1679) burning away over the page is BAKED; matte = 1.
  - 1905-1991 (C8-C9 the Eye): rgb = the page + the fire; matte = cover; EDIT adds the live storm through it.
  Joins with book_C: 840 -> 841 mean |diff| 2.9/255; 1717 -> 1718 3.7 (= book_C's own frame-to-frame 3.7);
  1904 -> 1905 18.6 (= book_C's own 18.0: the camera is moving fast there); book_C vs plate at 1905: 2.9. No pops.
  Checked 24-frame strips (_local_logs/review/burn_C/FINAL_{c5,sweep,eye}_strip24_*.jpg): continuous; the sweep
  is a fast (12x) busy wipe, the Eye tears up the vein from the glow and opens onto the storm.
* (a) ADOPTED by main (the user: "pretty decent"); (b) curl and (c) ash CANCELLED. Second-half page burns (the map
  X1s etc.) come later, once Codex's new pages exist: ftburn.SHOTS takes a new preset per burn (clip, mapping,
  retime by the render's cover), `ftburn.py shot --shot <name> --out renders/book_C_ft`.
* 05:40 main: option (a): WAIT for the plates (the farm test 0928-025304 sat 2h44m at "0 node(s) rendering, 9 up or
  booting"); watcher re-armed for 5 h (1-min polls, fires on EXIT/landed/failed). If still not started on wake: DO NOT
  resubmit; the morning session moves it to GPU nodes (noted in HANDOVER.md). When the plates land: check 1:1, launch
  the finals (--nodes 3 --detach, lane-launched row), then `ftburn shot` x3 into renders/book_C_ft, strips, commit,
  push, "BURN ROLLOUT DONE", stand down.
* 09:12 the plate TEST landed (09:11, GPU ldf-g03, 3.06 s/frame): checked 1:1 at 900/1700/1930/1985: the burn and its
  smoke are off, the fire's and the glow's light kept, the rest within render noise (mean |diff| 1.6-2.4/255 away from
  the burn); matte 1 everywhere. FINALS LAUNCHED by the lane: request 0928-091213-pagesbookplate-44915 (--nodes 3,
  log ~/.cache/ldfarm/req/0928-091213-pagesbookplate-44915.log), row added to LAUNCHED_BY_MAIN.md. Watcher armed.
  NEXT when they land: `ftburn shot --shot c5|sweep|eye --out ../../renders/book_C_ft --preview <scratch>` (each via
  RENDERQ_SLOTS=2 renderq), a 24-frame strip each, commit, push, "BURN ROLLOUT DONE", stand down.
* ROLLOUT CODE READY (ftburn.py `shot --shot c5|sweep|eye [--frames] --out renders/book_C_ft [--preview DIR]`):
  - c5: as adopted (leaf geometry, real time from the clip's birth at 841, tongues);
  - sweep: Pexels 8828892 in screen space, flipped (its burn climbs from the lower left -> falls from the upper left),
    what burns is the HELD RACE (embers_C3 f1679), BAKED into book_C_ft with matte 1 (so EDIT's under adds nothing),
    the page shows through the hole;
  - eye: Pexels 8829001 (2560x1440, scaled to 1920x1080) in screen space pinned on the glow (1037,602 at 1922,
    drifting -0.7,-0.95 px/f with the camera), foreshortened 0.72, flipped; matte = cover, EDIT adds the live storm;
  - sweep + eye are RETIMED BY COVER: each frame takes the clip frame whose hole covers the screen as much as the
    render's burn does (book_C_matte), so every hole-open/sweep frame SOUND-C synced to is kept.
  - The page = renders/book_C_plate (the farm's clean plates) when present, else a stand-in (tests only: c5 = f840
    reprojected, sweep = f1708 held, eye = f1905 held). FINALS WAIT FOR THE PLATES.
* SOUND (SOUND-C's picture_sync_C.json; C is on hold in EDIT, so this is for the re-sync when C comes back):
  | burn | hole opens | before (burn v2) | now (filmed) |
  |---|---|---|---|
  | letters C5 | 841-842 (unchanged) | the hole fills the frame by ~900 | a slit tears open 842-870, the flames flare 845-900 (the fast tear), the page LINGERS: the rim leaves frame ~1000-1020 (last paper ~1020) |
  | the Deep's sweep | edge 1689-1705, peak ~1699 (unchanged: retimed by cover) | same | same frames; the flames flare with the front (1690-1704) |
  | the Eye | 1921-1922, open by ~1965 (unchanged: retimed by cover) | same | same frames; flames lick the tear 1925-1960 |
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
