# MAP-L2 (lane MAP): C18 THE MAP ANSWERS + THE ROAD, and C23 bar 71 THE ROADS OUT, on an INVENTED world

# >>> STATE (MAP-L2, 27 Sep ~20:50Z; usage pause 21:25Z -> ~23:30Z). RESUME HERE <<<
* THE MAP IS ON AN INVENTED WORLD (terra.py; geo.WORLD_VER 5 since 7bf1b43): no real-world data anywhere in the map lane.
  Design: the ring at (0, 18) on the High Moor at the sheet's centre (dark: no relay fire stands on it); her beacon on
  the knee of her range (12.9, 9.0); the Run's seven pre-lit along its east arm (relay.CHAIN), the seventh at C17's
  burn-through point 1130,485 (road.opening_key solves the opening: tx 23.6, ty 8.4, w 50.3, head -5.3 in v4); the
  west river from the knee's lake to the western bay; the northern range; a ragged NE desert with a salt lake; the hook
  (NW), the long cape (SW), an archipelago on the great western sea; four labels in the book hand (terra.LABELS).
  The Road: 16.5 deg, a least-cost walk (road.road_path) that keeps off riverbanks. Crane keys: road.CAM_KEYS.
* SENT to main (20:37Z, 20:47Z): the world description + the v4 sketch stills (renders/_farmtest/map_sketch f1-f7).
  Awaiting approval of the geography. v4 sketch verdict (mine): desert light and ragged (fixed), moor open (fixed),
  labels present and small; relay too sparse (88 fires: coast/desert branch lines never started) -> fixed in 7bf1b43
  (branch radius 5, side lines 0.3); the parchment kernel's centre crease (sheet.parchment) moved to the thirds.
* FARM: `map_l2_c18 --test 6` started 20:43Z on ldf-g01 with the PRE-FIX code (v4) -> renders/_farmtest/map_l2_c18/
  (C18 + bar 71 frames + X1). Unreviewed at 20:50Z. The v5 fixes are pushed, so any new run takes them.
* NEXT (on resume): (1) review renders/_farmtest/map_l2_c18 (fires, the Road, light, labels; 4160 opening + X1 match;
  4440 arrival; 4479 end on the ring; bar 71 roads/hearths); (2) if the director approved JOB READY, the final is
  `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/map_l2_c18.json --nodes 1` (maybe already run by the
  director in the gap: check renders/map_C 4160-4479, 5600-5679 and renders/x1_map_C*); (3) a relay-density pass if
  still sparse (rev 3 had ~400 fires; target 200-300 in the widest frame), then a cheap re-render.

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
