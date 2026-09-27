# MAP-L2 (lane MAP): C18 THE MAP ANSWERS + THE ROAD, and C23 bar 71 THE ROADS OUT, on an INVENTED world

# >>> STATE (MAP-L2, 27 Sep ~20:40Z; usage pause 21:25Z -> ~23:30Z). RESUME HERE <<<
* THE MAP IS ON AN INVENTED WORLD (terra.py, WORLD_VER 4): no real-world data anywhere in the map lane. Code pushed
  (a624990 + earlier). Design: the ring at (0, 18) on the High Moor at the sheet's centre (dark: no relay fire on it);
  her beacon on the knee of her range (13, 8); the Run's seven pre-lit along its east arm (relay.CHAIN, the seventh at
  C17's burn-through point 1130,485 via road.opening_key); the west river from the knee's lake to the western bay; the
  northern range, a NE desert with a salt lake, the hook (NW), the long cape (SW), an archipelago on the great western
  sea; four labels in the book hand (terra.LABELS). The crane: CAM_KEYS in road.py (opening solved from the relay).
* FARM RUNS IN FLIGHT at 20:40Z (logs in the scratchpad `terra/`): `map_sketch` --test 7 (geography stills 1-7 ->
  renders/_farmtest/map_sketch; 7 = a data card with the relay/road/camera numbers) and `map_l2_c18` --test 6 (the REAL
  shot: ROI bake + relay + 6 frames -> renders/_farmtest/map_l2_c18/). Neither was reviewed by me yet.
* NEXT (on resume): (1) look at renders/_farmtest/map_sketch (f1 whole sheet design colours, f3 the camera's world inked
  flat, f4 the widest view, f5 opening, f6 end, f7 numbers) and renders/_farmtest/map_l2_c18 (the shot); (2) if the
  director approved the geography, fix any nitpicks and run `python3 the-long-dawn/cloud/farm.py
  the-long-dawn/cloud/jobs/map_l2_c18.json --nodes 1` (only after the director's approval of JOB READY); (3) review the
  landed frames (4160 open + burn-through match, 4300, 4390 widest, 4440 arrival, 4479 end on the ring, bar 71 5600-5679
  roads out and hearths), then EDIT's comp notes (X1 centres: C18 1130,485 = C17's seventh; bar 71 960,402).
* Known open items: the Road's line and the relay were never seen rendered in the shot yet; labels never seen; the moor
  must read as open downs (not a crater); the desert must be a light ragged stipple (v3 was a dense bullseye, fixed in
  v4 code, unseen); bar 71's hearths/roads on the new land unseen. Old real-Earth caches in renders/map_C/cache (807 MB)
  and v2 frames renders/map_C/f_01920-02087 (real Earth, superseded, unused by EDIT's v3 list) can be deleted.

# >>> STATE AT HANDOFF (MAP-L2 splits the MAP half off MAP/BOOK, 27 Sep ~19:00Z local) <<<
* MAP-L keeps the BOOK (book.py, pages.py, pen.py, redbook.py, book_c.py, ringpage.py and `NOTES.md`). MAP-L2 owns the map
  files: geo.py, features.py, rivers.py, sheet.py, bake.py, relay.py, render.py, road.py (+ the rev-2 legacy webmap.py,
  answer.py, places_globe.py). fire.py, ink.py, noise.py and burn.py are shared: only small, noted commits. These notes are
  the map half's; `NOTES.md` stays MAP-L's (two lanes in one folder: never commit the other lane's file).
* THE BLOCKER (MAP-L, on import): C18's and bar 71's map was our REAL EARTH (her beacon on Everest, road.py 27.99/86.93;
  the council ring at the Pamir knot; India, Arabia, the Gulf and the Caspian legible at 4300-4440). It breaks H5 #2
  (LANDSCAPES, NOT LANDMARKS) and the north star, and a council drawn in Central Asia is a geopolitical statement.
  Real-world inputs at handoff: geo.py (Natural Earth 50m land, the Earth normal map as elevation, Blue Marble biomes and
  lakes), rivers.py (80 real rivers), features.HILL_RANGES (Indian hill ranges), relay.ROUTES + EVEREST (real ranges),
  road.RING_LL/ROUTE_LL, sheet.ROSE_C (South Pacific), webmap/places_globe (real cities, ocean leaps).
* Nothing of the map was ever rendered in v3: `renders/map_C` holds only the v2 real-Earth frames 1920-2087 (MAP v2's
  delivery, superseded) and the real-Earth caches in `renders/map_C/cache` (807 MB). Both v3 jobs are HELD;
  `cloud/jobs/map_v3_road.json` must NOT be launched (real Earth).
* THE JOB: keep the engine (Cam/Sheet/LightGrid, the bake, the hand, the parchment, the fire relay, the pen-line Road,
  FireRemains), and replace the real-world inputs with an INVENTED CONTINENT in the same map-degree frame. Brief (director,
  27 Sep): a whole world in the spirit of a hand-drawn map in an old book (ranges and passes, rivers from flow
  accumulation, forests, plains, a desert, coasts, islands, a great western sea); no real coastline, range, river or city;
  not Middle-earth's shapes; her beacon's range stands in for the ink Run's range; the ring of stones near the heart of the
  map on a high place that belongs to no one; any labels sparse, in the BOOK HAND (pen.Hand), never readable.
  Beats: C18 (4160-4479, arrive 56b3 = 4440), C23 bar 71 (5600-5679). Delivery renders/map_C in C numbering.
  FIRST: one still of the whole invented sheet to main EARLY (geography approval before the bake).

## WORK LOG (MAP-L2)
* 27 Sep ~15:10-16:00 local: `terra.py` (the invented world) + geo/features/relay/road rewired to it; no real-world data is
  read anywhere in the map lane any more (Natural Earth, the Earth normal map, Blue Marble, rivers.py, HILL_RANGES,
  EVEREST, RING_LL are all out of the code path). Caches: `renders/map_C/cache_w/v<WORLD_VER>/` (versioned, so a farm
  node's parked disk never serves a stale world). Sketch stills via the FARM: `cloud/jobs/map_sketch.json`
  (`python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/map_sketch.json --test 6` -> renders/_farmtest/map_sketch).
* Iterations: v1 a round blob with hundreds of lakes (rejected by me); v2 the continent fills the sheet's east and runs off
  its edges, but a firth under the hook echoed the Gulf of Lhun, stipple everywhere, the moor read as a crater ring;
  v3 fixes those (no firth; designed desert zones; scattered downs; clumped foothills; the northern range and the desert
  brought into the widest frame; the Road keeps off riverbanks).
* THE DESIGN (terra.py DESIGN block): the ring at (0, 18) = the sheet's centre, on the High Moor (lobed, open, dark:
  no relay fire stands on it: it is no one's). Her beacon on the knee of her range (13, 8): the west arm runs SW, the
  east arm ESE then NE across the continent; the Run's seven pre-lit along the east arm at ~2.2 deg (relay.CHAIN), the
  seventh at C17's burn-through point on screen (road.SEVENTH_SCREEN 1130,485 from RUN-C's render_ink.scroll_beacons at
  local 310-319; the opening camera is solved for it, road.opening_key). The west river runs from the knee's lake to the
  western bay; the northern range and a desert (NE) close the widest frame; the hook (NW), the long cape (SW) and an
  archipelago on the great western sea.
