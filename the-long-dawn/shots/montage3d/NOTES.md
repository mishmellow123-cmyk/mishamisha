# MONTAGE-3D — notes

## >>> STATE AT HANDOFF (MONTAGE-3D-5, 27 Sep ~23:55Z; succeeds MONTAGE-3D-4) <<<
**Scope (director):** (1) C14 hands-only: strike 1 (C 2960-2999) and strike 3 -> blow -> catch (C 3150-3359), new
RING_SHOTs `flint_a` / `flint_b` in ringc.py; (2) C15 fire test fixes: restage the left-hand catch (the doll hand), no
hand poking in at the roar (it was the waiting LEFT glove, parked near the lens), break the flat flame wall, make her
C-shaped steel read (a longer back, both bends clear the fist), grained + creased leather; (3) find polish (find_a
3009 flash: cold-white snow, warmth only in speculars/sparks; 3033-3060 warm knuckle rims + a stitch line; find_b
cam_dists 0.20/0.165 -> 0.28/0.24). meltC (5360-5519) belongs to MONTAGE-MELT: never touch meltc.py or its jobs;
kit/*.py, fireparts.py and render.py are shared with it, so they stay unchanged.
**SAFE MODE (28 Sep 00:30Z):** no local Blender/renders at all; every test on the farm (`ringC_look2.json`, one lane
per RING_SHOT; `--frames` must fall inside a lane's range, a single-shot job clips the list to its own range).
**PRIORITY (coordinator):** C15 first (a SLATE in C's master), then C14, then the find polish. Hands must move
naturally: check 24+ consecutive frames (farm --test of a RANGE, a filmstrip of every 2nd frame) before JOB READY.
**C15 now = two shots:** `fire` 3360-3565 (roar, entrance, hang, tip, the draw begins) + `fire_catch` 3566-3599 (an
insert looking down on her open left palm, find_b's rig: the steel's tip tips the band in, the fingers close).
Round 2 verdict: roar/entrance/hang PASS (flames with dark depth, the C's upper bend reads, dark waxy leather); the
in-shot catch FAILED twice (the band lost in the dark, a claw-like curled hand, cheesy coal bed) -> the insert.
Jobs re-cut: `ringC_fire_1..7` = 3360-3565, `ringC_fire_8` = fire_catch 3566-3599.
**Glove v2 (glove.py):** fingers slimmed ~16 % + tapered; leather v2 (joint wrinkles, flexion creases, burnished
knuckles, stitched points, side-seam dashes, a visible grain). **C-steel:** back 8.6 cm (both bends clear the fist).
**C14 redesign (round 1 FAILED: cage-like basket grid, unreadable fists, a pale-cube flint, straight rain sparks,
wicker nest):** ash bed + cold cinders (no bars), a knapped flint nodule, an irregular fine-fibre nest, parabolic
orange sparks, breath paths through frame, look-down cameras (22 deg strikes; 40 deg over the nest for the blow).
**Running (28 Sep ~01:25Z):** C15 round 3 = motion ranges 3380-3405, 3476-3500, 3556-3599 (95 fr); C14 round 2 =
2966-2999 + 9 flint_b stills. Both -> `renders/_farmtest/ringC_look2/`.
**EDIT:** when finals land, C14 rows 2960-3000 and 3150-3360 and C15 3360-3600 read `renders/ring_C` (stem `ring`).
**QUEUED after C15 + C14 (coordinator, A-FIX; spec `review/A_FIX_NOTES.md` item 6):** re-render DESERT (A ~3880) and
KARST (A3815), 60 frames each on the farm GPU: (1) desert.py's step-back `back = 0.42 * ease((f - 1540.5) / 5)` slides
her 0.42 m in 5 frames with planted feet (A3881-3885): keep her planted, lower the arm slowly instead; (2) the fire's
ignition is a one-frame switch (fireparts.ignite_env): give it a soft catch over ~6-10 frames, LOCALLY in karst.py /
desert.py (fireparts.py is shared with meltc.py: do not edit it). Motion check 24+ consecutive frames, then JOB READY.

## >>> LAST-HOUR UPDATE (MONTAGE-3D-4, 27 Sep ~21:00Z): find_a APPROVED + LAUNCHED; find_b JOB READY sent <<<
* **find_a (3000-3079): APPROVED by the director, launched by main** (`ringC_find_a_{1,2,3}`, --nodes 3, log
  `_local_logs/jobs/ringC_find_a_farm.log`) -> `renders/ring_C/f_03000-03079`. When it lands: decode + contact-sheet
  check. Director's praise: 3017 is the hero frame. Director's POLISH notes (a re-render of only the touched frames):
  (a) **3033-3060 the glove** reads as a smooth dark mass ("a seal's back"): a thin warm rim from the embers on the finger
  ridges and knuckles so the digits separate, and a seam or stitch line catching light; it must read as a gloved hand at
  a glance. (b) **3009-3013 under strike 2's warm flash** the snow goes mauve-brown (suede/sand): keep the snow
  cold-white, warmth only in the specular and the sparks (e.g. the flash lamp's diffuse down / specular up, or a cooler
  flash colour with the warm sparks kept).
* **find_b (3080-3149): JOB READY SENT ~20:58Z** (`ringC_find_b_{1,2,3}`), awaiting approval. The fist fix is pushed
  (d7ad5ed): the night fill 1.3 -> 16 on the closed fist only (key k 1 -> 0.025 over the vision, so the vision frames are
  unchanged). Tested on the farm: `renders/_farmtest/ringC_look/ring_C/f_{03080,03146}.jpg` (mean 50-55/255, was 9: grained
  leather, curled fingers, readable). Polish (not blocking): the bookends are tight (three fingers fill the frame);
  cam_dists (0.20, 0.112, 0.165) -> ~(0.28, 0.112, 0.24) would show the whole fist.
* Farm note: a NEW GPU node spends ~6-7 min on the Cycles kernel JIT (416 s for one frame on ldf-g09); warm nodes
  render at 12-27 s/frame.
* **Left for the 23:29Z successor (director):** the fire-test catch (FIX 2 below) and the other fire/melt FIXes; the
  C14 hands-only sequence (strike 1 2960-2999; strike 3 -> blow -> catch 3150-3359); RUN-C's R13b data is below.

## >>> STATE AT HANDOFF (MONTAGE-3D-4 -> successor, 27 Sep ~20:40Z) <<<
**Brief:** C's Ring close-ups, gloved hands included (brief3_MONTAGE3D2.md + COMMON.md): C14 THE FIND, C15 THE FIRE TEST,
C22 THE MELT. Cycles, canonical script of fire (`assets/ring/`), delivery `renders/ring_C/f_%05d` (C numbering; EDIT's
`ring()` reads it) + `renders/ring_C_mask/` (the melt's band mask for ACCORD's AC3). JOB READY: find_a APPROVED and
rendering; find_b sent (see the LAST-HOUR UPDATE above); fire + melt still have the FIXes below. No local process of
this lane is running.

**SCOPE CHANGE (director, ~20:30Z): ALL of C14 is now ours.** HEROINE's two H1-C approaches failed their gate. Build C14
(2960-3359) as ONE hands-only sequence in Blender, with the same gloved hands and C-steel as C15, on H1's master timing
(C numbering): strike 1 = 2980 (bar 38 b2), strike 2 = 3009 (+29; its spark shows the band = find_a), the hand closes
3060, opens 3090, the vision 3090-3138, the fist 3140 (find_b), strike 3 = 3178 (bar 40 b3.9), the long blow ~3204-3300,
the catch = 3316 (bar 42 b2.8), the roar = 3360 (= C15's first frame). So NEW shots: strike 1 (2960-2999) and strike 3 ->
blow -> catch (3150-3359): her gloved fists, the C-steel striking a flint, short orange falling sparks into the tinder
in her basket, her breath, the catch; hands, sparks, tinder only (no face, no bare hand). Tell EDIT (EDIT-2) when they
exist: its C14 rows 2960-3000 and 3150-3360 still point at HEROINE's h1_v3h5.

**Files:** `ringc.py` (RING_SHOT=find_a|find_b|fire|melt; all four implemented), `glove.py` (the leather glove rig),
`render.py` (driver; `post_aux` hook; PREP_ONLY_FRAMES preps only the frames rendered, +-1; farm-safe: no Mac GPU lock
and a per-range job tag under MT3D_BLENDER). Jobs: `cloud/jobs/ringC_{find_a_1-3,find_b_1-3,fire_1-8,melt_1-6}.json`
(<= 30 frames each, Cycles 128 spp, MT3D_NOISE 0.03; the melt jobs also ship ring_C_mask) and `ringC_look.json` (4 lanes,
one per shot: all look-dev frames on ONE GPU node, one kernel JIT). THE FARM (COMMON.md), from ~/mishamisha, PUSH FIRST:
`python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/ringC_look.json --frames 3017,3112,... --test <K>` ->
`renders/_farmtest/ringC_look/{ring_C,ring_C_mask}/`. farm.py picks the h100 GPU pool itself (the command names
MT3D_BLENDER). Measured: 12-27 s/frame full res at 128 spp (23 frames in 5.9 min, $0.74; 29 in 13 min when queued).
Finals ONLY after the director's OK: `farm.py the-long-dawn/cloud/jobs/ringC_<shot>_<k>.json --nodes 1-3`.
Round-2 look frames (the latest, commit fd59b71): `renders/_farmtest/ringC_look/ring_C/f_{3017,3080,3112,3141,3147,3372,
3450,3488,3530,3572,3580,3586,3597,5362,5363,5366,5400,5424,5438,5470,5500}.jpg`.

**Verdicts, round 2 (farm, full res):**
* **find_a 3000-3079: PASS** (3009 flash reveal, 3017 sparks on the snow and the ice, the band in a clean pit with a
  level meltwater disc (the 0.6 mm step that drew a double jagged glint is fixed), the glove low from the left 3033-3060).
* **find_b 3080-3149: vision PASS, fist FIX.** 3112: the band in the cupped leather palm, five forge-towers of C6's kind
  glowing on its far inner face and bowing toward her: reads. FIX: the fist frames (3080-3084, 3141-3149) are near black:
  raise `fill` (opts, now 1.3; try ~6) and/or lighten the leather here; the fist must read as a gloved fist.
* **fire 3360-3599: the hang and the tip PASS, the catch FAIL.** 3450/3488/3530: the band hangs on the C-steel's long arm
  in the flames, letters awake, the lip; it slides to the lip and holds; the arm's end glows dull red; her right fist a
  dark leather shape (matte now). FIX 1: at 3372 (the roar) a lump of her fist pokes in lower left: start the grip off
  frame (`grip()` before 3384: offset -0.11 -> ~-0.25). FIX 2 (the big one): the catch 3560-3599 = her LEFT hand huge,
  soft and rubbery in the foreground (a DOLL read). Restage: keep the lens on the steel as she draws it out toward frame
  left; her left glove comes in from the left edge at the band's scale, in focus, dark and fire-rimmed, and plucks the band
  off the lip, closing on it (or tip it into a palm framed small and sharp). FIX 3: the backdrop is a flat bright flame
  wall; break it with darker gaps and more depth (fewer, dimmer bgfire cards; smoke).
* **melt 5360-5519: the H5 points PASS** (dark, never overexposed; a crisp 3-frame drop from above + a bounce; the letters
  stay letters (wake 5368, flare 5420, out 5440) while the far side runs into the bead; never a flat ring, no donut; white
  from 5476 to near-white by 5519). FIX: (a) the bead has a pinched crease on its right side (the arc-to-dome mapping in
  `_melt_positions` seams at us = +-1: blend the end caps into the dome); (b) the molten metal reads as orange candy
  (lower `hot`, keep it a mirror of the fire: gold, not orange plastic); (c) the stone reads as flat asphalt: char, a
  crack, ash, one or two coals in the foreground.

**Next steps, in order:** (1) the FIXes above (find_b fill; fire grip start, the catch restage, the backdrop; melt bead
crease, hot, stone), one farm `--test` round on `ringC_look.json`, look at every frame at full res + crops; (2) one
full-res check still per shot -> SendMessage main `JOB READY ringC_<shot>_1..N: ...` with the stills (est. 20 jobs, 550
frames, ~2.5-4 GPU-h on the farm); (3) the new C14 hands shots (strike 1, strike 3 -> blow -> catch) in ringc.py as new
RING_SHOTs with their own jobs; (4) after finals land: check every frame (decode, flicker, a contact sheet), tell ACCORD
(AC3 uses `renders/ring_C/f_05360-05519` + `renders/ring_C_mask/`) and EDIT; delete `renders/_farmtest/ringC_look/`.

**For RUN-C (R13b: the Ring's fall over the ink range strikes the snow beside the old cairn on her summit, C 2900):**
* The "old cairn" is the BEACON's dry-stone base (she kneels there to strike), NOT keeper.CAIRN (B's counting cairn is
  2.7 m from her knee). The hollow (the strike point): `keeper.shelf_pt(0.45, 0.45)`, on the snow at the foot of the
  beacon's base on her side: ~0.41 m from KNEEL = shelf_pt(0.05, 0.35) (by her knee), ~0.47 m from BEACON =
  shelf_pt(0.9, 0.6). The pit it melts: 3.6 cm radius at the top, 1.2 cm deep, a 1.9 cm disc of refrozen meltwater.
* The moon: use keeper's own (world.MOON_DIR = s1's (-0.80, 0.36, 0.48): screen-left of keeper's lens, 21 deg up). The
  find conforms to it: find_a is shot INTO the moon (it sits ~16 deg right of the lens axis), cheated down to 15 deg in
  the close-up so the pit's rim shades the band until the flash; colour (0.55, 0.66, 1.0). find_a's local frame (x right,
  y into frame, z up) is free: turn it so +y faces the moon's bearing. TODO for us: move find_a's `flash_pos` (now
  (-0.20, 0.05, 0.25), up-left beyond) to the beacon's actual side in that frame (about left and toward the lens).


## >>> MONTAGE-MELT (C22 THE MELT, C 5360-5519; lane split off MONTAGE-3D-4 by the director, 27 Sep ~19:15Z) <<<
* **STATE (21:05Z, before the usage gap; resume ~23:30Z):** `meltc.py` built and pushed (46b8014). The six
  full-res farm keys PASSED my check -> `renders/_farmtest/meltC/ring_C/f_{05363,05400,05421,05436,05444,05458}.jpg`
  (+ `_mask`); 1:1 sheet `_local_logs/review/melt_keys.jpg`. **JOB READY meltC sent to the director at 21:02Z**
  (160 fr, 5360-5519, 192 spp, -> `renders/ring_C` + `renders/ring_C_mask`). If approved and not yet launched:
  `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/meltC.json --nodes 3` (~35-45 min; `--missing` to
  resume). The job renders through `melt_local.py` (exits non-zero when a frame is missing: render.py exits 0 on a
  Blender failure, which hid the first farm run's crash, a scalar-vs-vector bug in the char material, now fixed).
  **On resume:** (1) check `renders/ring_C` has all 160 + the masks (`farm.py status`); (2) contact sheet every 8
  frames + the run 5436-5460 frame by frame; look at the fall (5360-5366) and the rattle for pops; (3) the nitpicks
  below as a cheap re-render if the director wants; (4) tell ACCORD-3 the plates are in (their NOTES).
* **MONTAGE-3D-4 keeps** THE FIND, THE FIRE TEST, `ringc.py`, `glove.py`: MONTAGE-MELT never edits them. `ringc.py`'s
  `_melt` (RING_SHOT=melt) and `cloud/jobs/ringC_melt_1..6.json` are the pre-split draft: SUPERSEDED, never launch.
* **Delivery:** `renders/ring_C/f_05360..05519` (C numbering; EDIT's C22 reads `ring_C` first) + `renders/ring_C_mask/`
  (8-bit grey coverage of the metal, the 'ringmask' AOV, for ACCORD-3's AC3; format in `shots/accord/NOTES.md`).
* **The design:** a low macro across the flat hearth stone (lens 100 mm, f/20, 8 deg up, ~62 mm frame easing in to
  ~52); the canonical band + the script of fire; ONE open-tube mesh (480 x 48, both ends at the back where it breaks)
  whose per-frame shape the venv computes (`melt_verts`, numpy) and Blender keys as shape keys (deformation blur;
  the letters ride the metal). 5360-5363 it falls into frame (0.33 g, two frames seen, letters streaking), strikes,
  one hop, an Euler's-disk rattle, still by 5381; 5388-5433 the slump (the back sags and leans, lobes, the front
  stands, letters legible); 5420 the flare (hotter orange, never white); 5427-5440 a neck forms at the back; 5440 the
  letters go out and the loop BREAKS at the back on the cadence; both ends run forward as squat round bulbs (volume
  kept by construction) and bridge at once into one bead 22 deg right of the lens axis (never a mirror-symmetric
  face/moustache), 5451; the bead (11 mm, a sessile drop) trembles and settles; 5476-5519 the flare to white
  (lamps x12, post lift). Look: the gold mirrors the fire (a canopy of fire overhead + a dim warm bounce behind the
  lens, reflections only; the lamps never show in the metal), front face dark gold so the letters read; the molten
  glow is facing-weighted (emissivity = 1 - reflectance); the hearth's charred kindling burns 7-17 cm behind with the
  film's bonfire sprites; sooty ash-dusted stone (dull, no worktop sheen). Tools: `tests/t_meltprev.py` (numpy
  preview of the geometry through the real camera, seconds), `melt_local.py` (render.py with its own lock),
  `MELT_PREVIEW=1` (Workbench motion check).
* **Known nitpicks (cheap re-render later):** the blurred burning kindling behind reads a little lava-orange (tone the
  char glow down, `_char_mat` x4.0 -> ~2, and the kB/kC sprite gains); two frames of the run (5447-5448) show a soft
  crumple where the bulbs meet (smooth kv along the arc in `_profile`); ACCORD's council fire partly replaces the
  surround in AC3 anyway.
* **H5 calls (all mine):** darker, never overexposed; a crisp drop; letters that STAY letters until they go out; NO
  doubled rim (glazed donut / dentures); no plughole (no ring seen flat from above with a dark hole).
* **Sync (music/v3/barmap_C.json):** 5360 the white heart, her fingers forced open (ACCORD's P2 handle), the drop;
  5420 (68 b4) the letters flare once; 5433-5440 the breath; 5440 (69 b1) the letters go out, the loop breaks =
  the Ring loop cadences into D; 5519 (near) full white for ACCORD's P3.

## >>> STATE (MONTAGE-3D-4, 27 Sep ~19:40Z): all four Ring shots BUILT; look-dev on the FARM (`ringC_look`) <<<
* **Farm (COMMON.md):** heavy tests go to `cloud/farm.py`: `ringC_look.json` = 4 lanes (find_a, find_b, fire, melt) so
  one GPU node renders every check frame: `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/ringC_look.json
  --frames <list> --test <K>` (from ~/mishamisha) -> `renders/_farmtest/ringC_look/{ring_C,ring_C_mask}/`. PUSH FIRST.
  render.py is safe for several renders on one node (no Mac lock under MT3D_BLENDER; per-range job tag).
* **Local half-res verdicts:** find_b t_fb2 = the macro works (band on grained leather, towers glowing on the far inner
  face); fixed since: the fist frames were black (sky 0.05 -> 0.12, a cool 'nightfill' from the lens side), the vision
  stays alight until the fist shuts it in (fade 3133-3141), push-in 20 -> 11 cm to the vision and back to 16.5 cm on the
  close, a deeper, more golden bow. fire t_fi1 = the hanging band, the lip, the tip slide and the red end read, BUT the
  glove read as a cream clay mitt (DOLL) and the flames as a blown-out wall: in Cycles the emissive flame cards LIGHT the
  scene -> `_card_no_light` (camera + glossy only: the gold still reflects the fire), lights down (bed 1.1, tongue 0.45,
  rim 3.5 from behind: the fist is a dark shape rimmed by the fire), backdrop gains halved, two more licks round the
  band, the heat a dull cherry gradient (was neon paint), f/9; the catch restaged (left hand rises from below, palm up,
  fingers away; the band drops off the lip 3577-3580; the fist closes 3582-3592 with the band drawn into its hollow;
  the steel withdraws). melt smoke t_me0 = the slumping band keeps its letters (the brief's point) but was cream and
  overexposed with giraffe-print coals -> the same card fix, lights / 4, coals = dull 'fire' type pushed back 8.5 cm.

## >>> (earlier) STATE (MONTAGE-3D-4, 27 Sep ~19:25Z): all four Ring shots BUILT in `ringc.py` <<<
* **find_a (3000-3079):** the disc step fixed (level meltwater, C1 wall: no double polygon glint); the moon now LOW (15 deg,
  0.34) so the pit's far rim shades the floor and strike 2's flash is the reveal; the flash is the strike itself just out of
  frame above left (`flash_pos` (-0.20,0.05,0.25), 0.55, tau 1.5 f): it falls off across the frame (was a far lamp that
  turned the whole frame beige). Finals: MT3D_NOISE 0.03 (identical to 0.015 by eye, 30 % faster).
* **find_b (3080-3149):** REDESIGNED after test t_fb1 failed (a black fist filling the frame, the palm invisible, the vision
  a red neon crescent). Now a macro (lens 100, 14 cm, f/25, ~50 mm frame) down her palm at 35 deg; the fist opens 3081-3090;
  the band lies in the cup (placed by ray-casting the posed glove); the vision on the FAR INNER face (strip u 0.30-0.70,
  flipped to read true): `_vision_strip` = five of C6's kiln-like forge-towers (ember rings, a glow at each crown, smoky
  forge-light), bowing toward the middle 3094-3126 (two earlier versions failed: a cartoon skyline with grid windows =
  clip-art; thin towers with line flames = a crowd of raised arms). A diffuse-only warm light spills the vision on the
  leather; a moon at the lens's mirror angle lays a sheen on it. The fist closes 3136-3141.
* **fire (3360-3599):** the roar (flames surge 3360-3380, sparks), her right fist brings the C-steel in from the left
  (3388-3404): `_steel_path` = forged flat stock 2.6 mm deep, the back in her fist, the long upper arm with an up-turned lip,
  the lower arm curled into a scroll; the band HANGS on the upper arm (its inner face on the arm's top edge), letters awake
  3401-3419; the arm's end glows dull red (blackbody, 'heat') while the gold stays cool; 3480 it tips, the band slides to
  the lip and her wrist catches it (3489); 3560 drawn out; her LEFT palm rises under the tip (3564-3578), the band drops in
  (3580-3583) and the fist closes (3585-3594). Fire = the film's bonfire sprites (bgfire + lick + lick2) on additive cards,
  embers + burning kindling below, three iron basket bars + rim behind.
* **melt (5360-5519):** `band_mesh_open` (a seam opposite the pool + end caps) deformed per frame by `_melt_positions` into
  SHAPE KEYS (one per rendered frame, keyed 0-1-0 linear -> deformation blur); the far side melts first and runs into one
  bead while the near side, letters toward us, stands until 5432, then gives way (5432-5446): never a flat ring (no donut);
  letters wake 5368, flare 5420, out 5440 (deep orange-red); the molten metal glows (gold shader `molten` = the keyed front);
  crisp drop 5360-5364 (accelerating), one bounce + a coin's settle to 5378; the fire whitens from 5468, post() pushes to
  near-white by 5519; `post_aux` writes the 'ringmask' AOV to `renders/ring_C_mask/` for ACCORD's AC3.
* **Jobs written (not READY yet):** `cloud/jobs/ringC_{find_a_1-3,find_b_1-3,fire_1-8,melt_1-6}.json` (<= 30 frames, 128 spp,
  noise 0.03, Cycles; farm.py picks GPU because of MT3D_BLENDER). render.py: `post_aux` hook; PREP_ONLY_FRAMES preps the
  exact frames +-1 (sparse tests cheap).

## >>> STATE AT HANDOFF (MONTAGE-3D-4 fresh lane, 27 Sep 18:45Z) <<<
* **Taken over by MONTAGE-3D-4.** Brief: `_local_logs/handoff/brief3_MONTAGE3D2.md` + the director's task: C's three Ring
  close-ups (gloved hands included), Cycles in the cloud, jobs of <= 30 frames, JOB READY with one full-res check still
  per shot, delivery `renders/ring_C/f_%05d` in C numbering (EDIT's `ring()` reads it).
* **On disk (checked 18:40Z; no MONTAGE-3D process running; only the stray Blender.app pid 77512):** `ringc.py` find_a
  implemented (tests `t_fa7` sheet, `t_fa9`/`t_fa9b` full-res 3017); find_b / fire / melt = NotImplementedError.
  `renders/ring_C/` empty. KARST/DESERT v3 finals in `renders/montage3d_v3/` (done). `renders/montage_v2` kept.
* **Findings at takeover:** t_fa9 vs t_fa9b (MT3D_NOISE 0.015 vs 0.03) are identical to the eye (mean |diff| 0.5/255)
  -> finals use 0.03. BUG: the ice disc sits 0.6 mm PROUD of the pit floor (`_snow_pit`: bottom = -dp + 0.0006 inside
  R < r_disc, the wall = -dp at r_disc) -> a step whose two edges catch the moon as a double jagged polygon line.
* **ACCORD (AC3) wants:** her fist opening in white-hot fire at 5360, the Ring falling out of her palm; (near) full
  white by 5519; a band alpha/mask pass welcome (-> `renders/ring_C_mask/`).


## >>> STATE (MONTAGE-3D-3, paused ~16:36Z by the director; RESUME 20:00Z) <<<
* **Built (committed):** `ringc.py` (the v3 Ring shots, RING_SHOT=find_a|find_b|fire|melt, Cycles, C numbering,
  `PREP_ONLY_FRAMES` so a cloud job preps only its own frames) + `glove.py` (her thin leather glove: a ~330-vert
  subdivision cage + armature with DQS, poses keyed as bone rotations -> true deformation blur, no per-frame meshes;
  seams, pebble grain, palm creases in rest coords; `tests/t_glove.py` = clay/leather turnarounds) + `render.py`
  (prep limited to the rendered frames when the shot sets PREP_ONLY_FRAMES).
* **find_a (3000-3079) WORKS in tests** (`tests/t_fa7/sheet.jpg` half-res key frames; `tests/t_fa8`, `t_fa9*` full-res):
  moonlit night snow (moon 0.26 from behind, sky 0.04), a clean shallow pit (r 3.6 cm, 1.2 cm deep) with a wet
  translucent ice disc (no manhole), the Ring lying on it; strike 2 = flash from above left on 3009 + five short
  orange sparks streaking down (motion blur), one dies glowing on the ice beside the band (lights it to ~3055);
  her LEFT glove enters from the frame's left edge LOW over the snow (3033-3050, never from above), the fingers
  reach into the pit past the band, curl and draw it into the fist (3055-3060), the fist lifts away left.
  Just applied, NOT yet re-checked: dead sparks move off (a black bead hung in the air at 3017), softer disc edge
  (the ice channel ramp), Burley SSS. **Timing (local Metal ~= cloud CPU, calibrated 70 vs 71 s on the probe):**
  full-res 64 spp: random-walk SSS 204 s -> Burley 110 s -> Burley + MT3D_NOISE=0.03 80 s (3017, the heaviest).
* **Next (in order):** (1) look at `tests/t_fa9/f_03017.png` vs `t_fa9b` (noise 0.03): is 0.03 clean enough?
  (2) find_b (`_find_b`: palm-up POV, fist opens 3081-3090, the vision strip `_vision_strip` on the INNER face
  over u 0.30-0.70 -> rotate the band so that arc faces the camera, fist 3140); (3) fire (`_fire`: C-shaped steel =
  a flat 7x3.5 mm bar bent into a C, her right fist below-left out of the flames, the Ring hanging on the upper arm's
  tip in the flames; the tip glows red over 3400-3560 while the gold stays cool; tips 3480, slides to the very tip,
  holds; 3560 drawn out, tipped off into her LEFT palm (which enters bottom-left) and the fist closes: a rhyme with
  the melt's drop); (4) melt (`_melt`: deform the band mesh per frame via `band_mesh` params so the letters ride
  the metal; drop + bounce 5360-5372, hot 5380+, slump 5390-5440 into one bead, flare 5420, out 5440, white by
  5519; AOV 'ringmask' -> post writes renders/ring_C_mask for ACCORD's AC3 composite); (5) one full-res still per
  shot; (6) cloud jobs (16: find 5 x ~30, fire 6 x 40 or 8 x 30, melt 5 x 32; setup = mt3d_probe's; render =
  `cd shots/montage3d && MT3D_ENGINE=CYCLES MT3D_DEVICE=CPU MT3D_BLENDER=$HOME/bpyenv/bin/python RING_SHOT=<shot>
  python3 render.py ringc --final --force --outdir renders/ring_C --range a-b`); (7) JOB READY to main with stills.

## >>> STATE AT HANDOFF (MONTAGE-3D-3 fresh lane, 27 Sep 15:25Z) <<<
* **Taken over by MONTAGE-3D-3.** Brief: `_local_logs/handoff/brief2_MONTAGE3D2.md` + the 15:20Z task: C's three Ring
  close-ups rebuilt to the H5 calls and rendered in CYCLES IN THE CLOUD (4-6 jobs of ~30 frames per shot, the
  `cloud/jobs/mt3d_probe.json` pattern: pip bpy==4.5.14 in its own venv; every job pushes `cloud_logs/<job>_status.txt`).
* **On disk (checked 15:22Z, no MONTAGE-3D process running):** KARST v3 + DESERT v3 finals (60 fr each) in
  `renders/montage3d_v3/{karst_slow,desert}`; the probe stills `renders/montage3d_v3/probe_{eevee,cycles}/f_00100.jpg`
  (cloud Cycles CPU 71 s/frame, EEVEE 151 s); the canonical inscription `assets/ring/` (APPROVED); the protected
  bake-off stills `tests/ring_final_{find,fire,melt}`. `renders/montage_v2` kept as it is (fallbacks).
* **Delivery:** `renders/ring_C/f_%05d` in C numbering: find_a 3000-3079, find_b 3080-3149, fire 3360-3599, melt
  5360-5519 (EDIT's EDL `ring()` reads `ring_C` first). Sync (music/v3/barmap_C.json): strike2 3009, hand_closes 3060,
  vision 3090, fist 3140, roar 3360, steel 3400, steel_tips 3480, fist_again 3560, white_heart 5360,
  letters_flare 5420, unmade 5440; P3 needs (near) full white by 5519 (ACCORD).
* **H5 calls to apply (all Ring):** find = a clean snow pit, meltwater/ice sheen on the disc, never a rust-yellow stain;
  her thin leather glove enters from the SIDE; lit only by the spark-flash (no jewellery-ad fill). Fire test = the Ring
  on the tip of her C-shaped fire-steel in the beacon's flames, unmarked, letters awake, no glow fringe. Melt = dark,
  never overexposed; a crisp drop; letters stay letters until they go out; no doubled rim.

## >>> CANONICAL RING INSCRIPTION (published ~10:25Z, letterforms revised ~10:35Z per the director; APPROVED; RING ONLY, not the book): HEROINE, EMBERS, ACCORD-v3 match THIS <<<
* **Texture:** `the-long-dawn/assets/ring/inscription_outer.png` (9040x640) and `inscription_inner.png` (7264x640),
  16-bit grey, white = letter coverage. Mapping + proportions: `assets/ring/inscription.json`. Specimen:
  `assets/ring/inscription_specimen.png`. Generator at any resolution: `shots/montage3d/ring_script.py`
  (`strip(W, H, 'outer'|'inner')`, W/H = circumference/band width; `python ring_script.py publish`).
* **The script of fire** (invented; no real alphabet, no real text): each letter is the outline of a small flame in
  one fine line, heavy at the base, a hairline at the tip (the book: "fine lines ... lines of fire that seemed to
  form the letters of a flowing script", outside AND inside). No broad nib, no marks above the line, no dots, no
  line inside a flame. Letters are joined base to base within a word.
* **Mapping:** u once round (outer: counter-clockwise seen from +axis; inner: clockwise, so both read left to right
  and never mirrored); row 0 = the +axis edge; x-height 0.30 of the band width, baseline 0.66 from the top edge.
  One fixed sequence: never re-randomise. The metal is plain when cold; the letters exist only in fire (emission
  through this mask, deep orange-red, never white).

## >>> STATE (13:12Z): KARST v3 + DESERT v3 FINALS DONE (local); the Ring rebuild waits for the 15:00Z RESUME <<<
* **KARST v3 FINAL:** `renders/montage3d_v3/karst_slow/f_00000-00059` (60 fr, 2/3 speed, flare on 15, 96 spp, 62 min,
  no post failures, only diff spike = the flare). Review sheet `_local_logs/review/montage3d_karst_v3.jpg`.
* **DESERT v3 FINAL:** `renders/montage3d_v3/desert/f_00000-00059` (= src 1520-1579 at speed, the catch on 20, 71 min,
  all 64 cloak meshes rebuilt; only diff spike = the catch). Review sheet `_local_logs/review/montage3d_desert_v3.jpg`.
* **Remaining weaknesses:** KARST: the spires left of centre are a little alike; the hero summit is a flat table
  (needed so she and the beacon stand on it). DESERT: at 1:1 the cloaked figure is smooth CG and the prints can
  read hoof-like, so WIDE FRAMING ONLY (as the H5 call says); her "flinch" reads as a hand held out to the fire.
* **Superseded and deleted:** the v2 karst_slow frames (pines). `renders/montage_v2` (v2 KARST 1640-1679, DESERT
  1520-1579, CITY) is left in place for the director (v2 is superseded by H5; say if it should go).
* **Verified stills kept for reference:** `tests/t_k3_full/f_00030.png`, `tests/t_des8f/f_01545.png`,
  `tests/t_cyc1/f_00100.png` (Cycles/Metal ring), `tests/ring_final_*` (the protected bake-off stills).
* **Probe:** `mt3d_probe` launched by the director (session_01JYJ7jHKrczNMo7LzYBKm8N) -> `renders/montage3d_v3/
  probe_{eevee,cycles}/` via the importer. **Next (15:00Z RESUME):** read the probe; build the three Ring shots (plan
  below) in Cycles; cloud jobs.

## >>> STATE (paused ~10:45Z; history, plans still valid) <<<
**Delivery folders (EDIT-v3 EDL convention):** KARST -> `renders/montage3d_v3/karst_slow/f_00000-00059`;
DESERT -> `renders/montage3d_v3/desert/f_00000-00059` (= src 1520-1579 at speed); THE RING -> `renders/ring_C/`
in C frame numbering (find 3000-3149, fire test 3360-3599, melt 5360-5519).
**Done since the handoff:** canonical inscription (APPROVED, above). Pipeline: `MT3D_ENGINE=CYCLES` (kit/core
`use_cycles`: Metal GPU here, CPU in the cloud; adaptive + OIDN) and `MT3D_BLENDER=<python with pip bpy>` (render.py
runs bl_main.py as a bpy-module script). Cycles/Metal verified on the bake-off fire still (`tests/t_cyc1`, new
letters awake; the first run spent ~8 min compiling Metal kernels, now cached). Probe job `cloud/jobs/mt3d_probe.json`
(pip bpy==4.5.14 in its own venv, since bpy pins numpy<2; the same still in EEVEE via Mesa/EGL and in Cycles CPU,
seconds stamped on the frame) = JOB READY, not launched.
**KARST v3 = code written, NOT rendered:** `towers.py` (karstgen's r(theta,z) with fluting ~off, reshaped in numpy:
one-sided waists/bulges, strata of hard/soft beds with bevels + bedding notches, lean, a bitten broken crown;
styles stack/spire/split/slab) + `karst.py` `V3` (towers for hero/nearL/MIDS3/FAR3, no trees/scrub, horizontal
strata in the rock shader; the hero crown kept whole on the camera side for the beacon; `KARST_V2=1` = old).
NEXT: `MT3D_REKARST=1 KARST_SLOW=1 python render.py karst --frames 0,15,59 --scale 0.5 --samples 24 --out t_k3a`
(EEVEE), look (mushroom/bottle/chess-piece silhouettes? rim light? glow?), then the 60-frame slow final.
**DESERT v3 = code written, NOT rendered:** ripples (two families 8/13 cm, 18 deg apart, mixed in patches, 2-scale
domain warp, breathing amplitude, fading over the last ~3 m to every crest), footprints (0.30 m steps +-15%, 25x10 cm,
toe-out, toe dig, downhill kick, 30-85% wind-filled), `cloak_body` (bare dark head + hair knot lifting downwind,
rolled wool collar, tunic/trousers/boots, gloved hands; the cloak a continuous ~3 cm sheet of flattened ellipsoids
with in-out folds, a downwind billow and a flap wave; SDF h=0.007). The first (tube) cloak read as a grass skirt;
the sheet version is UNSEEN: preview with the numpy turnaround (`$SCR/figprev.py`, needs `tprev.py` beside it)
before Blender. NEXT: `MT3D_REFIG=1 python render.py desert --frames 1520,1540,1545,1579 --scale 0.5 --samples 24
--out t_des8`, then full-res crops of the prints/ripples and the figure (wide framing only in the cut).
**THE RING = not rebuilt yet** (only the inscription + engine). Plan: Cycles for all three (local Metal tests,
cloud CPU finals). C14 find: dark, lit ONLY by strike 2's spark flash (decaying light + short orange falling
sparks), the band in a clean hollow, unmarked; a thin leather glove enters from the SIDE (not descending) and
closes on it; find_b: palm open, the towers-bowing vision in the polish, fist. C15: the band on the tip of a
C-shaped fire-steel (an oval loop, one end a curled tip) in the beacon's flames, unmarked, letters awake (outer +
inner strips), the tip dips on 3480 and it does not fall, drawn out 3560 into the gloved fist; no glow fringe.
C22 melt: darker exposure, a crisp drop (no haze), one slump surface (no second skin = no doubled rim/donut),
letters stay letters (emission follows the UVs on the slumping surface) until they go out at 5440.
**Engine question for the director (pending):** KARST/DESERT are accepted EEVEE looks; cloud EEVEE needs the probe to
pass (Mesa llvmpipe, likely slow). Recommendation: render those two 60-frame finals locally (one Blender, ~1.5 h and
~45 min) unless the probe shows cloud EEVEE is practical; the Ring finals go to the cloud in Cycles.

## >>> STATE AT HANDOFF (MONTAGE-3D-2 fresh lane, 27 Sep 10:15Z; history) <<<
**What exists (checked 10:12Z; no MONTAGE-3D process running):**
* KARST v2 (pines, Zhangjiajie profile): `renders/montage_v2/f_01640-01679` (40 fr) and the 2/3-speed re-time
  `renders/montage3d_v3/karst_slow/f_00000-00059` (60 fr, complete). BOTH SUPERSEDED by H5 (de-specify: no pines,
  no Zhangjiajie/Huangshan pillar profile, irregular weathered towers in mist). Kept only until the new karst lands.
  Re-time code is in `karst.py` (`KARST_SLOW=1`: source time 1640 + i*2/3, flare at i=15; `--outdir`).
* DESERT v2: `renders/montage_v2/f_01520-01579` (60 fr). Needs the H5 fixes (wide only; plain cloak with wind in
  its folds, no hooded robe, no figurine; ripples randomised in frequency/direction and fading at the crest;
  footprints smaller, irregular, half-filled, in a real gait).
* Ring bake-off stills (the PROTECTED find + fire test): `tests/ring_final_{find,fire,melt}/f_00100.png`, sheet
  `_local_logs/review/ring_bakeoff.jpg`. `ring3d.py` also holds UNTESTED animated inserts (find_a 3000-3079,
  find_b 3080-3149, fire 3360-3599, melt 5360-5519). Their prep died writing the vision EXR (OpenCV EXR codec off
  outside render.py); the 158 knitted-glove meshes it made (`cache/ring3d/data/glove_find_*`, ~700 MB) are the
  descending "bear paw" the H5 critic rejected -> deleted.
**H5 calls to apply before any new render:** KARST de-specified + 2/3 speed (60 fr, flare 15 in); DESERT wide,
cloak, ripples, prints; Ring: canonical inscription; find = her glove enters from the SIDE, thin leather, lit only
by the spark flash (no jewellery-ad fill); fire test = the Ring on the tip of her C-shaped fire-steel in the
beacon's flames, unmarked, letters awake, no glow fringe; melt = darker, crisp drop, letters stay letters until
they go out, no doubled rim.
**Heavy renders -> cloud jobs** (`cloud/jobs/<name>.json`, `"ship": "jpg"`): Blender in the cloud = `pip install
bpy==4.5.14` (the same 4.5.14 LTS as this Mac; cp311 manylinux wheel, 373 MB). No earlier job installed Blender.


## >>> HANDOFF (MONTAGE-3D-2, 2026-09-27) <<<
**Right now:** KARST ACCEPTED (renders/montage_v2/f_01640-01679). **DESERT FINAL DONE** (f_01520-01579, 60 frames,
2374 s = ~40 s/frame, 96 spp full res, peak 368 MB, no post failures, contact-checked: no pops), review sheet
`~/mishamisha/_local_logs/review/montage3d_desert.jpg`, awaiting the director. Calls made: moon az 100; blown sand
dropped (did not read at full res; `opts.streamers`); ripple fade gone before < ~5 px/ripple (no shimmer);
megaripples (1.7 m) give the far dune flanks wind texture; footprint rims softened. Remaining weaknesses: the
prints still read a little like stamped ovals at 1:1; the robe silhouette is simple (no hands at this size).
RING BAKE-OFF (v3 cut C) DONE, stills only: `ring3d.py` (RING_SHOT=find|fire|melt; `python render.py ring3d --frames 100
--scale 1.0 --samples 128 --out ring_final_<shot>`), sheet `~/mishamisha/_local_logs/review/ring_bakeoff.jpg`. Analytic band
(superellipse profile, 512x64, smooth normals), ring.py's invented inscription as emission, full-res screen tracing + a sphere
probe for reflections. MELT uses an SDF slump (`_melt_mesh`) with UVs by angle. Verdicts: FIND Blender, FIRE Blender (clearly),
MELT Blender but weakest (hazy; crisper drop + darker exposure needed). ~20-110 s per still, peak 258 MB.

### 1. State
* **CITY (src 1680-1719): FINAL, accepted by the director, in the cut.** 40 frames in
  `renders/montage_v2/f_01680..f_01719.png` (51.6 s/frame, 96 samples, DOF f/1.8, motion blur). Don't touch.
* **KARST (src 1640-1679, IGN 1650): FINAL RENDERED, awaiting the director's review.** 40 frames in
  `renders/montage_v2/f_01640..f_01679.png`; review sheet `~/mishamisha/_local_logs/review/montage3d_karst.jpg`.
  Kept from MONTAGE-3D-1: camera (z=30, drift x +1.6 -> -2.2, +0.8 m fwd, kick), pillar layout, ATMOS mist
  sea, 4 mist banks, figure action. Fixed:
  (a) pines: new `_pine` = crooked trunk leaning out over the drop + 3-6 tiers of flat RAGGED needle
      clouds (10-30 small noisy blobs each, not one smooth pad), dark needles with AO in the gaps + a
      fine tuft bump (sheen made them read as popcorn); ledge pines lean out of cracks (yaw = outward
      from the pillar axis); 12 variants (crown 0-4, ledge 5-7, low-poly far 8-11) + 2 scrub variants.
      Hero crown composed by hand (`HERO_PINES`, all behind/beside the beacon) + dart-thrown scrub
      clear of the beacon, the figure and the sight line; crown flattened (cap 0.1 R0) and roughened.
  (b) rim: moon az -58 el 16, strength 5.5 (volume factor 0.4 keeps the mist as it was); rock albedo
      up a touch. Moon-facing clefts now catch light (the long vertical joint on mid0 is real geometry).
  (c) glow: fire lights volume_factor 2.2 (halo in the air without over-lighting rock), `beacon_halo`
      glow_haze r 17 m d 0.013, and a wispy `veil` mist bank wrapping the upper shaft 4.5 m under the
      crown, lit from above by the fire. Sparks toned down (burst 120, rate 30) -- no firework.
  (d) memory: pillars are meshed only in the camera's visible z window with pixel-matched spacing
      (`_pillar_mesh` in karst.py; same shapes/seeds as karstgen) -> 1.30 M unique verts (was 5.2 M);
      new foreach_set mesh loader (`_mesh_fast`). Blender peak 616 MB half-res / 640 MB full-res tile 4
      (was 2.2 GB). Final: 96 samples, full res, vol tile 4, motion blur; ~90 s/frame on this Mac.
* **DESERT (src 1520-1579, IGN 1540): BUILT (`desert.py`), look nearly settled in half-res stills;
  NOT rendered final (the director reviews KARST first).** Latest stills: `tests/t_des6/` (moon az 100),
  `tests/t_des5/` (5 key frames, az 118), full-density summit crop `tests/t_des5c/f_01545.png`.
  Design (full Blender, no numba plate): every dune is its own crest-conforming mesh (rows along the
  crest, one column exactly on it, split windward/slip objects) -> true knife edges, never lumpy; radial
  ground grid with gentle mid-scale rolling; 0.61 M verts; Blender peak ~340 MB half-res, ~5 s/frame at
  half-res 24 spp. The camera stands on a high point of the hero crest (eye 1.62 m); the crest dips
  through a saddle (we look down on it: its moon-shadowed slip face runs as a dark band) and climbs to a
  small summit plateau (`_plateau`, cairn seated flat) where the robed, hooded SDF figure (robe + hood
  + neck wrap; hem billows downwind) thrusts the torch into the cairn beacon on 1539.5, lets go, steps
  back 0.42 m with the forearm up (elbow forward), watches. Flame/smoke/sparks stream downwind (-x).
  Shader ripples (wind-aligned, asymmetric, faded by pixel footprint), grainflow on slip faces, a line
  of footprints up the crest to her (UV procedural, `_footprints`), trampled sand round the cairn, faint
  fire-lit sand streamers off the summit crest, Milky Way core low left, horizon haze u=1/7000.
  Fixed on the way: rings in the stacked-ellipsoid robe (-> two tapered cones + hem billow); scarf end
  read as an outstretched arm from behind (tube, then band) -> removed; PCHIP extrapolation wall; hero
  mesh open end + summit objects threw odd moon shadows into the lower-left -> dune tapered at both ends,
  moon az 118 -> 100 (the summit's long shadow now falls out of frame).
  **Open before DESERT final:** (1) regenerate with the committed moon (az 100, el 17, I 2.3):
  `MT3D_REDUNE=1 MT3D_REFIG=1 python render.py desert --frames 1520,1539,1541,1545,1560,1579 --scale 0.5 --samples 24 --out t_des7`
  and check all key frames (the az-100 test only covered 1520/1545); (2) full-density crops of the summit
  (`--scale 0.33334 --opts '{"crop": [1150, 300, 640]}'`) and of the foreground footprints/ripples
  (`{"crop": [700, 700, 640]}`) -- ripples may alias near 30-90 m; (3) the left mid-ground dune face is
  plain -- acceptable, maybe a touch darker/softer; (4) streamers are subtle (dens 1.2): keep or drop;
  (5) 2 full-res frames at 64 spp (tile 4) for memory/time before the final; (6) review sheet + report.

### 2. Next steps (commands from `the-long-dawn/shots/montage3d/`, after `source ~/.venvs/longdawn/env.sh`)
1. `pgrep -fl Blender` -- the idle stray Blender app (pid 77512, ~9 MB, not ours) is always listed; any
   OTHER Blender means someone is rendering: wait. Because of that stray, `render.py --final` would wait
   forever -- pass `--force` after checking by hand.
2. DESERT stills: `python render.py desert --frames 1520,1539,1541,1545,1560,1579 --scale 0.5 --samples 24 --out t_desN`
   (MT3D_REDUNE=1 rebuilds the dune meshes, MT3D_REFIG=1 the robe meshes -- ~10 min for 64 frames).
   Full-density crops: `--scale 0.33334 --opts '{"crop": [cx, cy, 640]}'` (cx, cy in full-res pixels).
   Composition preview without Blender: numpy raymarch of `desert.ground_at` (see gotchas).
3. DESERT final (after the director's OK): `nohup python render.py desert --final --force > cache/desert/final_run.log 2>&1 &`
   (60 frames -> renders/montage_v2/f_01520..f_01579.png). Review sheet:
   `python tests/review_sheet.py desert ~/mishamisha/_local_logs/review/montage3d_desert.jpg 1520,1540,1545,1579 1545 <x0,y0,w,h>`

### 3. Memory-safe Blender on this Mac (8 GB, shared)
* ONE Blender process at a time (render.py writes `cache/blender.lock`; also check `pgrep -fl Blender.app`
  — the driver only auto-waits for `--final` jobs > 3 frames). Never run a test while a final renders.
* Volumetrics: `volumetric_tile_size` 8 for tests, 4 for finals (tile 2 at full res = 2.2 GB peak —
  never), custom range as tight as possible, 48-96 slices.
* `shadow_pool_size` '256' (core default is '512'; pass shadow_pool='256' to setup_render for karst).
* Geometry budget < ~2 M verts. Mesh only what the camera can see: karst pillars are meshed on the
  visible z window (the mist sea hides everything under ~-26 m, the near-left pillar only shows z 12-52)
  at ~2 px spacing for their distance (`karst._res_for`). Load big meshes with foreach_set
  (`karst._mesh_fast`, `desert._mesh_ch`), never Python lists of tuples (doubles the peak RAM).
* Instancing (GN `C.instancer`) for trees/tanks/far pillars; source collections view-layer-excluded.
* EXRs are deleted after post (keep with `--keep-exr` only for tests); 32-bit multilayer ~20 MB/frame.
* No textures at all (everything procedural), so texture limits don't apply.

### 4. Gotchas learned
* Blender's bundled numpy can't load on macOS 12 — keep Blender-side code numpy-free; do heavy work in
  the venv and pass binary meshes (`figures.write_mesh` format; 4th index -1 = triangle) / JSON.
* numba: every kernel needs `cache=True` (NUMBA_CACHE_DIR=cache/numba) — `_pillar_r` takes ~5-10 MIN
  to compile the first time (heavy inlined mt.noise fbm); after that ~1 s per pillar.
* EEVEE blended (additive) flame cards don't appear in screen-space reflections.
* Separate rigid capsule bodies always read as mannequins — use the SDF body (`figures.py`:
  smooth-min round cones/ellipsoids + surface nets, per-frame meshes swapped by bl_main's per_frame).
  FK abduction sign was wrong in the old kit version (fixed in figures.fk).
* Voronoi crease bumps on cloth look like cracked stone — don't.
* ATMOS shader fog needs geometry under every downward ray (the karst 'valley' plane), otherwise rays
  fall through to the world and pillars show through the mist. `enters` test: Incoming.z > 0 = looking down.
* Tree placement must be spacing-based (dart throwing), not per-vertex probability (mesh-resolution
  dependent: 12,891 trees once).
* numba invalidates a file's whole cache when the SOURCE FILE changes (mtime/size): editing karstgen.py
  costs a 5-10 min `_pillar_r` recompile -- put new venv helpers in the shot module (karst.py) instead.
* PCHIP crest tables extrapolate wildly past their last control point: clip/taper outside the range
  (`desert.hero_height`) -- a 'wall' at the horizon in a preview came from that.
* Quick composition check with Blender busy: raymarch the venv height function in numpy at ~288x121
  (desert: `ground_at`, moon-only shading, figure/beacon points projected) -- ~2 min, no shadows.
* Crop test at full-res pixel density: `--scale w/1920 --opts '{"crop": [cx, cy, w]}'` (karst & desert):
  narrower lens + shift, same pixels as the final; post (sparks) still projects right.
* Fire-lit foliage: white-ish sheen + smooth convex blobs = popcorn/cotton. Dark albedo, AO node in the
  material, a fine bump, many small blobs; keep scrub > 5 m from the fire.
* `~/mishamisha/_local_logs/sheet.py` resolves paths under renders/ — for tests use
  `render.make_sheet(dir, frames, out, cols, tw)`.
* zsh doesn't word-split unquoted vars; no `timeout` command on macOS; Bash tool timeout max 10 min —
  use nohup/background for long jobs.

### 5. Taste bar
* One clear idea per shot, ruthlessly composed; the fire is the brightest, warmest thing — everything
  else lives in blue darkness with real atmospheric depth (layers that fade), never a flat backdrop.
* Figures are silhouettes carved by firelight with working gestures — never a doll/mannequin read,
  never a torch held aloft; judge figures in neutral grey turnarounds (`tests/t_figview.py`) first.
* Only replace a v1 shot if it's clearly better side by side at full res; if in doubt, keep v1.
* Real-world specificity sells it (water tanks, office-floor bands, clinging pines, ripples) — but
  nothing that reads as a tech demo (no firework sparks, no snow-cap crowns, no mushroom trees).

## Pipeline (built)
* `render.py` (venv driver): flame sprites + timing tables (montage toolkit `ignite_env`/`flicker`, so
  envelopes match every other beacon) → one Blender process (`bl_main.py`, lock file, waits for other
  departments' Blender before long full-res jobs) → per-frame post in a thread (EXR read, sparks, heat
  shimmer, distant fire glows) → `look.finish()` → `look.save_png()`. Tests → `tests/<name>/`,
  finals → `renders/montage_v2/f_%05d.png` (src numbering).
* `exr.py`: numpy multilayer-EXR reader (Blender's ZIP/FLOAT). `fireparts.py`: sprite generator (montage
  `_bonfire`), exact Blender-camera projection (`BCam` from per-frame `cam_%05d.json`), Z-up sparks adapter.
* `figures.py` (venv): SDF people (fk/ik, hoodie_body, mesh_sdf surface nets, write_mesh).
  `karstgen.py` (venv): pillar meshes (lobes + striations + fractured blocks + crown), tree_points.
  `city.py`, `karst.py`: shot modules (venv spec + prep + post; Blender build + per_frame).
  `tests/t_figview.py` (figure turnarounds), `tests/t_pillar.py` (view any .bin mesh).
* `kit/` (inside Blender, no numpy — Blender's bundled numpy can't load on macOS 12):
  `nodes.py` node DSL; `core.py` render setup (EEVEE Next, Standard view, 32-bit multilayer EXR +
  Depth), camera keys (yaw/pitch/roll), GN instancer, ATMOS shader-fog group (exact exponential height
  fog + street-glow term + moon forward scatter), procedural night sky (gradient, moon, Voronoi stars,
  Milky Way); `fire.py` cairn/basket/wood/drum, flame card (sprite sequence, camera-facing, additive
  BLENDED), fire lights (keyed per frame), smoke plume + glow haze volumes (fire-lit only: moon
  volume_factor 0); `figure.py` rigid capsule body + keyed transforms, 2-bone IK, parametric cloth
  (robe, hood, sleeves, scarf ribbon) as per-frame shape keys, torch.

## Shot designs (planned)
* CITY 1680–1719 (IGN 1690, answers 1694–1716): eye-level rooftop, wet tar deck, parapet at ~12 m,
  hooded young person right of centre + drum (right third); grid city rotated 28°, near neighbours
  around our height with water tanks, downtown towers right of centre; camera = original move.
* KARST 1640–1679 (IGN 1650): 3-D fluted/cracked sandstone-limestone pillars (Python + mathutils
  noise meshes, instanced variants) with GN-instanced clinging pines on ledges/crowns; mist sea via
  ATMOS (noise-modulated top) + mist sheets; fire-lit local mist volume around the hero pinnacle so the
  glow blooms inside it; lateral drift camera as the original.
* DESERT 1520–1579 (IGN 1540): heightfield dunes (venv-generated, loaded via raw floats), knife-edge
  S crest, shader wind ripples, scalloped slip faces, blown-sand streamers off the crest, robed
  hooded figure (robe/hood/sleeve/scarf cloth) backlit, Milky Way; slow truck right as the original.
