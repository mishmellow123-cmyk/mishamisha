# PAGES-C (lane PAGES): C's BOOK, PAPER and PAGE material, and C's TRANSITIONS (succeeds MAP-L, the book's author)

# >>> HANDOFF / STATE (PAGES-C, 28 Sep ~22:50 local) -- READ THIS FIRST <<<
**Delivered:** `pages_book` LANDED 22:39 (4624/4624: renders/book_C + _matte, C 80-1039, 1680-1991, 6160-7199; every
frame fresh; C22 5360-5519 untouched). Verified: 6160 == runC_illum f_02877 (mean diff 0.27/255); C9's matte open by
1991; the fade reaches black at 7199. Map X1s: renders/x1_map_C{,_matte,_cover} 4150-4185 and x1_map_C71{...}
5594-5640 (x1burn.py, v2; EDIT kind `burn` live, 25e39dc).
**LAUNCHED by main (--nodes 8, log _local_logs/jobs/pages_book_polish_farm.log); the director reviews the landed frames. Lane DONE. Lesson: keep passing check stills until the director has seen them.** Was: **JOB READY SENT (22:50 local): `pages_book_polish`** (cloud/jobs/pages_book_polish.json, 1425 frames: 80-331,
360-594, 760-850, 1680-1991, 6160-6434, 6741-7000; code at the tip after 'PAGES-C polish 2'). The director's notes:
T14 1.5x off the gutter; the initial re-illuminated (a burnished gold ground with an ink diaper and vermilion bezants,
the vermilion letter filling it, penwork curls and frame lines, a gilt border stem with ivy down the margin;
RB.Page ink_over_gilt); gold as hammered leaf (few large facets: gnoise 4.5 + 11 cycles/cm, rough 0.24); the Deep's
seam one solid band; the ~800 orange blob (the page's fire lights held 11 mm up and softened, EMBERS-C4's find);
the Deep's smoke thins away before 1992. Checked on 17 farm stills (deleted). ON LANDING: count the six spans in
renders/book_C and _matte (1425 each, fresh), look at 118, 215, 520, 800-816, 1850, 1880, 6300, 6860; nothing
else is pending in PAGES-C's area.
**How it is built (for a successor):**
* book.py = the engine (a numba ray-marched 2.5-D book; `shade_kernel` materials: paper(), ink, gilt, rubric, fire,
  burn v2 via `BURN.field2`). book_c.py = C's shots on C's timeline (`Book3.shot_*`, cameras `cam_*`), the post burn
  (`burn_post`), smoke (`smoke.py`), the C25 plate (`plate_hdr` in HDR per sample + `plate_post` in display space for
  the match), ink lines (`inkline.py` via `RB.Page.texture(extra=)`). redbook.py = C2's last leaves (the initial, the
  swan-ship) + the Epilogue. pages.py = the Mountain, the Deep, Plenty, the Havens (+ roundel). burn.py: v1 `field`
  is MAP-L2's (road.py); v2 = 20-float blocks (`BURN.v2(block, rag, creep)`).
* Tests: `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/pages_book.json --test K --frames ...
  --local-out the-long-dawn/renders/_farmtest/<name>` (SAFE MODE: never render locally). Push before every farm run,
  and NEVER push while a book job is rendering (units fetch the tip when they start).
* EDIT owns: T7 (screen overlay), the dissolves, `burn`, `finish_ramp` 6160-6224; EDIT must drop T1/T14 (in book_C).
**Open (not done):** the Deep's arcades are copy-paste identical (left: the H5 critic's "best graphic"); the Mountain's
smoke is billow circles (engraving convention; could break the outlines); the C2 initial's ground is still a flat
vermilion square at heart (now framed by penwork and gold). Notes for others: E15's two-arc swoosh (EMBERS-C4);
E5-C's first flame should match e15 at 1039/1040.
* Owned: book.py, book_c.py, burn.py (v2), pages.py, pen.py, redbook.py, ringpage.py, titleburn.py, smoke.py,
  inkline.py, x1burn.py; jobs pages_book, pages_book_polish, pages_x1map. Not mine: C19-C23, C18's map, E15.
* Diagnosis + status: `~/mishamisha/_local_logs/review/C_PAGES.md`.

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
