# PAGES-C (lane PAGES): C's BOOK, PAPER and PAGE material, and C's TRANSITIONS (succeeds MAP-L, the book's author)

# >>> STATE (PAGES-C, 28 Sep ~01:25Z local 21:25) <<<
* Diagnosis: `~/mishamisha/_local_logs/review/C_PAGES.md` (sheets `_local_logs/review/c_pages/`). Director approved the
  C24->C25 PLATE version and asked for the transitions spec (below; relayed to EDIT-3 via main).
* DONE + pushed (2c8cb6e, abe373c, a05da80, pass 2): burn edge v2 (`burn.field2`: torn at birth, crinkled char band
  with a lit lip, beaded ember line, flecks; creep so C5's rim leaves the frame); smoke off the burning page
  (`smoke.py`); incandescent fire letters (`fire_k`); paper tooth/laid+chain lines/pulp cloud/show-through; gold
  crinkle so it glints; a living hearth (`B.hearth`); leaf z-fight fix; 16-sample turns, 24-sample riffle; the 6400
  blank frame; the 320 pop; C2 opening (the hearth flares up on the red book; the hand at 140 px/cm); the C24->C25
  plate (pixel match: mean diff 0.27/255 at 6160); the Havens framed closer; the end fade; C's ink lines T1/T14 written
  into the page (`inkline.py`); C2's ship = the Havens' swan-ship; the Mountain's ring bolder, inner flame tongues;
  the roundel 2.5 cm; `x1burn.py` (the two map X1s).
* LANDED: renders/x1_map_C(+_matte,+_cover) 4150-4185 and x1_map_C71(+...) 5594-5640 (v2, one t_open each; EDIT kind
  `burn` requested). Tests: renders/_farmtest/pages_book (47 stills, pass 1) and pages_book2 (22 stills, pass 2).
* JOB READY SENT (~21:35 local) `cloud/jobs/pages_book.json` (2312 frames: 80-1039, 1680-1991, 6160-7199 ->
  renders/book_C + _matte; C22's 5360-5519 untouched), after pass 3 (c67fec6): `farm.py ... --nodes 8`. On landing:
  count the frames (2312 in each), spot-check 80-120 (the fade-up), 330-345 (riffle), 562-594 (turn), 840-1039 (burn),
  1680-1717 (sweep), 1920-1991 (burn-through), 6160 (== runC_illum f_02877), 6400-6434, 6720-6792 (turns), 7160-7199
  (fade); then delete renders/_farmtest/pages_book{,2,3} and pages_x1map. EDIT (told via main): drop T1 and T14 from
  titles.py for C; `finish_ramp` #21 goes live on these frames; `burn` #17 #19 are live on the v2 layers (25e39dc).
* Owned: book.py, book_c.py, burn.py (v2 additive; MAP-L2's road.py still calls the v1 `field`), pages.py, pen.py,
  redbook.py, ringpage.py, titleburn.py, smoke.py, inkline.py, x1burn.py; jobs pages_book, pages_x1map.
  Not mine: C19-C23 (council), C18's map (MAP-L2, paused), E15's sparks and flame (EMBERS-C).
* Open notes for others: EMBERS-C's e15 streams swoop in two big arcs (764-788) and a soft orange blob at 800;
  E5-C's first flame should match e15's at 1039/1040.

## C's TRANSITIONS: the spec (PAGES-C, 28 Sep; director: "design each in the book's own language")
C has three materials: the BOOK in a dark room (~0.3-0.55), the INK world on bright parchment (~0.75), and FIRE on
black (~0.05). Moving between them is always one of the book's own acts: the page burns through, a leaf turns, ink
bleeds in, or the picture turns out to be a page. EDIT kinds: `burn` (NEW, below), `x1` (MAP-L2's old formula),
`dissolve` (linear light, smoothstep), `finish_ramp` (NEW, below). "render" = baked in the owner's frames.

| # | cut (C frames) | from -> to | method | owner / state |
|---|---|---|---|---|
| 1 | 80-116 | C1 black -> C2 the red book | **the hearth flares up out of the black** on a view that shows the red-bound book lying open on the table by the fire, then drifts in over the last leaves | render (PAGES, book_c.shot_red_book); EDIT: plain cut from C1's black |
| 2 | 320 | C2 sheaf -> C3 the riffle back | continuous: the riffle, C2's recto and bounce carried across (the pop fixed) | render (PAGES) |
| 3 | 560-594 | C3 Mountain -> C4 the dense leaf | **page turn** (one solid leaf, 16 samples; the z-fight fixed) | render (PAGES) |
| 4 | 840-1039 | C4 -> C5 | **burn-through onto black** (burn v2: torn at birth, char, lip, embers, smoke); the rim creeps out of frame by ~1025 ("for a moment it is only light") | render (PAGES) + EMBERS-C's fire (add) |
| 5 | 1040 | C5 the fire alone -> C6 the forging | **match on the flame**: e15's flame at 1039 and E5-C's at 1040 in the same place, size and colour; if they differ, a 6 f dissolve 1037-1043 | EMBERS-C (match) / EDIT (dissolve only if needed) |
| 6 | 1680-1717 | C7 the race -> C8 the Deep | **X1 reversed**: a ragged diagonal ember edge sweeps the (held) race away and leaves parchment, smoke trailing it (1.55 s) | render (PAGES) over EDIT's hold of embers_C3 f1679 (unchanged) |
| 7 | 1905-1991 | C8 -> C9 the Eye | **the paper smokes, then burns through** into the live storm (burn v2); fully open by 1991 | render (PAGES) + EDIT matte comp (unchanged) |
| 8 | 2080 | C9 the Eye -> C10 the Mirror | hard cut: the loudest moment to silence (keep) | - |
| 9 | 2320 | C10 the Mirror -> C11 the grasp | **12 f dissolve 2314-2326**: the fire in the ripples becomes the claw's embers | EDIT |
| 10 | 2470-2480 | C11 the band falls -> C12 black | the band must fall OUT of frame into the black (render); if it cannot, a 10 f fade to black 2470-2480 | EMBERS-C / EDIT |
| 11 | 2720 | black -> C13 the glint | hard cut from black (keep) | - |
| 12 | 2840 | E13 the glint -> R13 the ink star | **8 f dissolve 2836-2844** on the falling light (RUN-C matched the glint's path): the fire becomes a drawn star | EDIT |
| 13 | 2960 / 3080 / 3150 / 3360 | the star's strike -> flint; find -> vision -> catch; -> fire test | hard cuts (story cuts; keep) | - |
| 14 | 3000 | H1 flint (warm) -> find_a (cold blue) | a 4 f dissolve 2998-3002, or grade find_a's first second warmer | EDIT / MONTAGE-3D |
| 15 | 3600 | C15 the fire test -> C16 the ink reveal | hard cut (the scale reveal; keep) | - |
| 16 | 3840 | C16 reveal -> C17 the ink run | hard cut on bar 49 b1; if it reads as a jump cut, a 6 f dissolve 3837-3843 | EDIT |
| 17 | **4150-4185** | C17 the seventh beacon -> C18 the map | **burn-through onto the map** (x1burn.py v2, one t_open 4152): EDIT kind `burn` | PAGES (layers: renders/x1_map_C, _matte, _cover) + EDIT |
| 18 | 4480 | C18 the ring -> C19 the council | council side (not mine) | MAP-L2 / COUNCIL |
| 19 | **5594-5640** | bar 70 the stone -> bar 71 the roads of fire | **burn-through** (x1burn.py v2, one t_open 5597): EDIT kind `burn` | PAGES (renders/x1_map_C71, _matte, _cover) + EDIT |
| 20 | 5680 | bar 71 the map -> C24 the illumination | **16 f dissolve 5672-5688** through the map's breath (5668-5676): the drawing fills with the dawn | EDIT |
| 21 | **6160-6236** | C24 the dawn -> C25 the red book | **the picture turns out to be a page**: C25 opens on C24's last frame (6159) pixel for pixel as a plate on the verso and pulls back to the red book by the hearth; the plate takes the fire's light by 6222 | render (PAGES) + EDIT `finish_ramp` 6160-6224 (ink look -> film look), or the match breaks at the grain/grade |
| 22 | 6400 / 6720 / 6760 | the Havens; blank; the next | **page turns** (16 samples; the 6400 blank frame fixed) | render (PAGES) |
| 23 | 6960 | C27 -> C28 | continuous camera (keep) | render |
| 24 | 7120-7199 | the end | **the hearth sinks** after the plagal close; black by 7199 | render (PAGES) |

EDIT kinds wanted:
* `burn` (for #17, #19): three folders, `glow` (display sRGB), `keep` (RGB matte: the outgoing picture scorched and
  charred, 0 in the hole) and `cover` (1 - hole): `out = O * keep + I * (1 - cover) + glow`. Read `keep` as colour
  (it carries the scorch's tint and the dark char band), not gray. The old `x1` formula still works as a fallback.
* `finish_ramp` (for #21): on book_C frames 6160-6224 the FINISH blends from the ink look to the film look,
  `lerp(ink(img), film(img), smoothstep((f - 6160) / 64))`, because C24 (runC_illum) is finished with the ink look
  and book_C with the film look; without it the pixel match at 6160 breaks.
