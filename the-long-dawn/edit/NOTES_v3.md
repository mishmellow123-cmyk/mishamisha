# EDIT v3 (lane EDIT): the three EDLs, text, animatics

## >>> STATE (EDIT-3, from 27 Sep 23:45Z; succeeds EDIT-2) <<<

- **FILM B IS DROPPED (user, 28 Sep ~01:00Z: its story was too unclear).** The deliverables are A (+ A's coded-towers
  ALT master) and C. B is out of the watcher (`FILMS`, default AC), the masters (`deliver.sh` CUTS default "A C"),
  the animatics, the H9 kit, the titles (`title_v3.sh`: A only) and the previews (`previews.FILMS`). B's renders and
  assets stay on disk (its best shots may be harvested into A or C); its last master, QC and preview are parked in
  `_local_logs/delivery/dropped_B/`. The B-only items below (sound_B, the B4 crop, the B14 plate, the B kit) are moot.
- **Next:** A's and C's masters and transitions; CLARITY-A (A) and PAGES-C (C's book and pages) send concrete
  transition fixes (see TRANSITIONS below). PEOPLE MOVEMENT gate (COMMON.md): flag any jittery or sliding figure
  seen in A's or C's masters.
- **Kept EDIT-2's unpublished work** (snapshot `refs/wip/resume-20260927-2345`): the 21:26Z coverage block below and
  the EDL exports (A20/B14 `kind: title`, A20's desc). Re-exported with `assemble.py --edl` after PR #1, so every
  take now carries `final_eligible` (A 3, B 3, C 11 provisional takes); nothing else moved.
- **B's sound = SOUND's `music/out/v3/sound_B.wav`** (director APPROVED ~23:40Z: final_B's pre-master score + real
  recorded effects through COMPOSER's master chain; level map 14/14, sync 25/25, dawn rule, -16.05 LUFS, TP -1.30).
  Wired EDIT-side: `assemble.ADOPTED_AUDIO` is consulted first by `resolve_audio`/`resolve_audio_label` and the
  watcher's signature; the MASTERS table in music/NOTES_v3.md is untouched (the music lanes' table). Outside
  `deliver._code_hash()`, so B re-muxes without re-encoding picture.
- **Watcher** restarted on the merged (PR #1) `refresh_watch.sh` (the old one, pid 74916, ran the pre-PR script).
  The Mac crashed (memory) at 00:15Z 28 Sep; SAFE MODE (COMMON.md top): renderq = 1 slot, niced, waits for 1.5 GB.
  Every watcher step self-queues through renderq; relaunched with `WORKERS=2` (log `_local_logs/animatic/watch_<HHMM>.log`).
- **B4 crop (H9 CRITIC B M4, 28 Sep):** `B_H1_CROP = (0.12, 0.08, 0.40, 0.40)` (was 0.17,0.22,0.26,0.26): 2.5x; on the
  strikes the glove, steel and sparks read, and the basket reads as a basket; on the blow her profile enters at the right
  edge as a dim silhouette (a warm rim on the lips/nose around src 1356). Not done: Lanczos resampling (it is `Ctx.read`,
  inside `_code_hash()`, so changing it re-encodes every segment of every film).
- **B14's plate (director ~23:55Z, option a):** `handback_B` stops at 5199, so RUN-B-3 renders `dawntitle_B`
  5200-5439: continuous with handback at 5199, then a slow tilt up into the dawn sky so the title sits over clean sky
  at y~360. No held frame. Until it lands B14 stays the stand-in sky + ember title (the kit calls it out as the
  proxy); the watcher picks the plate up (B14's second take) and `title_v3.sh` re-throws the sparks from its fires.
- **Previews (EDIT-2's handover):** the killed 21:11Z run left A_bars01-49, B_bars01-68 and C_bars01-12 but no
  `.previews.json`/README, so the next previews step re-cuts every stretch as NEW (the watcher exits 3: send main
  one line per file). NEVER run two preview jobs at once (the .part files collide). README fixes: it now names the
  provisional source itself (`AS.provisional_sources`: C9's burn-through = its under-layer embers_C3_half), names
  C's council takes (`accord`) as v1 timing renders (`previews.TIMING`), and says the film finish is in.
- **Disk:** ~14 GB free; delivery/ 7.1 GB (grained masters 0.5-1 GB each + MP4 screeners); dropping the MP4
  screeners frees ~2.7 GB if space gets tight.
- **B's H9 kit (for a fresh-context critic):** `~/mishamisha/_local_logs/review/h9_B/` (`h9_kit.py --cuts B --out`),
  plus `CRITIC_README.md` there: the B master to watch with sound, and what is known-pending (DUSK v2, the cairn
  re-renders, the B14 plate).

## TRANSITIONS in A's and C's comps (for CLARITY-A and PAGES-C; user, 28 Sep: "some of the transitions weren't great")

**NEW (28 Sep 01:05Z): EDIT transitions `EDL.TRANS`** (edl_v3.py; comp in `assemble._transitions`, installed by
`_init` after the finish, like it: no Ctx method changed, so only the segments a window touches re-key). A window
`dict(f0, f1, cut, kind, ...)` joins the outgoing shot (its frames, then its last frame held) and the incoming (its
first frame held, then its frames), each finished with its own look, by `kind`: `x1` (a MAP burn layer: `O * keep
+ I * (1 - keep) + glow`, from `glow`/`keep` folders) or `dissolve` (linear light, smoothstep). `ready=False` holds
a window back (plain cut). New kinds go in `_transitions`. **C's spec is PAGES-C's (shots/map/NOTES_PAGES.md, 24 rows);
EDIT's rows are live (28 Sep ~01:40Z, checked on finished stills across each window):** dissolves 2314-2326 (#9),
2836-2844 (#12; plays as the cut while E13 is a slate), 2998-3002 (#14), 3837-3843 (#16, optional, taken), 5672-5688
(#20); kind `burn` (O * keep_rgb + I * (1 - cover) + glow) at 4150-4185 (#17) and 5594-5640 (#19) on PAGES' v2 layers
(one t_open each); `finish_ramp` 6160-6224 (#21: ink look -> film look) live with PAGES-C's pages_book re-render
(28 Sep ~01:30Z launch; C25 opens pixel for pixel on C24 6159). T1 and T14 are now in the page (set `in_picture`). Not taken: #5 1037-1043 (the flames match); #10's fallback fade 2470-2480 waits on EMBERS-C4.
C's ink lines (T1, T7, T9, T14) are now always IRON (titles.py): T14 on C27's page was white.

Until then, how EDIT joins shots: every EDL boundary is a **hard cut** (segments are byte-joined; there is no EDIT
dissolve anywhere). EDIT makes only four transition devices: (1) **matte comp** `rgb + (1 - matte) * under` (MAP's
`book_C_matte` over an under-layer: C8, C9), (2) an **additive layer** (`add`: C4-C5's `embers_C3_e15`), (3) the
**fade to black** in A (6456-6479, `Ctx`), (4) the X3 **ember-title** layers (A20). Every other transition
(dissolves, white-outs, page turns, burn-throughs, holds) is **baked in the department's render**: "continuous" below
means the same folder runs on across the section line with no cut, so a change there is that lane's re-render.

| film | frame | from -> to | kind | how it is made / who owns it |
|---|---|---|---|---|
| A | 80 | A1 black -> A2 R1 FALSE DAWN | hard cut from black | falsedawn_A's first frame (RUN-A-L); any fade-in is in the render |
| A | 560 | A2 R1 -> A3 E1 INTO THE LIGHT | hard cut | falsedawn_A -> embers_A3 (the push into the glow is EMBERS') |
| A | 960-3119 | A3 -> A4 -> A5 -> ... -> A10 | continuous | one embers_A3 render (EMBERS-A2/A3): the point, the ignition, towers, edge, brink, "out of the white" at 2640 (A8 -> A9) and the drop to black at 2800 (A9 -> A10) are baked |
| A | 3120 | A10 E9 ember on black -> A11 X2 stars | hard cut | embers_A3 -> stars_A (RUN-A-L); the ember sits at (960, 548) on both sides |
| A | 3360 | A11 X2 -> A12 H1-A FIRST FIRE | hard cut | stars_A -> h1_v3h5 src 1236 (the shared H1 take) |
| A | 3600 | A12 -> A13 H1-A ROAR | continuous | the same H1 take |
| A | 3680 | A13 ROAR -> R2-A EVERY RIDGE | hard cut | h1_v3h5 -> reveal_A (RUN-A) |
| A | 3800 / 3860 | R2-A -> M5 KARST -> M5 DESERT | hard cuts | reveal_A -> montage3d_v3/karst_slow -> /desert (MONTAGE-3D, Blender) |
| A | 3920 | M5 DESERT -> A14 R3 BEACON RUN | hard cut | -> beaconrun_A (RUN-A-L) |
| A | 4240 | A14 -> A15 R16 WATCHERS | hard cut | -> watchers_A (RUN-A-L) |
| A | 4400 | A15 -> A16 E10 TOWERS IN THE LIGHT | hard cut | watchers_A -> embers_A3 |
| A | 4720 | A16 -> A17 E10 THE FIRE, SEEN | continuous | embers_A3 |
| A | 4880 | A17 -> A18 R6 THE CROSSING | hard cut | embers_A3 -> crossing_A (RUN-A2) |
| A | 5840 | A18 -> A19 R7 THE BLUE HOUR | hard cut | crossing_A -> bluehour_A (RUN-A2) |
| A | 6240 | A19 -> A20 X3 TITLE | continuous (planned) | bluehour_A runs on under the ember title (EDIT's title layer); now the stand-in sky, an EDIT proxy |
| A | 6456-6479 | A20 -> end | fade to black | EDIT (`Ctx`, bar 81 b3.8) |
| A ALT | 1440-2639, 4400-4879 | A6-A8, A16-A17 | same cuts | embers_A3_alt_codedtowers replaces embers_A3 frame for frame |
| C | 80 | C1 black -> C2 P1 THE RED BOOK | hard cut from black | book_C's first frame (MAP-L, book engine) |
| C | 80-1039 | C2 -> C3 -> C4 -> C5 | continuous | one book_C render + its matte: the riffle back (320), the page darkening to letters-to-fire (560) are MAP's |
| C | 700-1039 | C4 THE FIRE CATCHES, C5 THE FIRE, ALONE | EDIT additive layer | + embers_C3_e15 (EMBERS-C3) added over the page; frames count only where both exist |
| C | 1040 | C5 -> C6 E5-C THE FORGING | hard cut | book_C + e15 -> embers_C3 |
| C | 1440 | C6 -> C7 E11 THE RACE | continuous | embers_C3 |
| C | 1680-1919 | C7 -> C8 P2 THE DEEP | EDIT matte comp (wipe) | book_C over a HOLD of embers_C3 f1679: MAP's matte is the ember edge that sweeps the frozen race away to parchment |
| C | 1920-1991 | C8 -> C9 THE EYE (burn-through) | EDIT matte comp | book_C over embers_C3 at the same frame (live storm under the page): the burn-through is MAP's matte opening |
| C | 1992 | C9 burn-through -> C9 THE EYE ONTO NOTHING | hard cut | book_C + under -> embers_C3 alone (seamless only if the matte is fully open by 1991) |
| C | 2080 | C9 -> C10 M4 THE MIRROR | hard cut | embers_C3 -> mirror_C (MIRROR) |
| C | 2320 | C10 -> C11 E8-C THE GRASP | hard cut | mirror_C -> embers_C3 |
| C | 2480 / 2720 | C11 -> C12 black -> C13 THE RING FALLS | hard cuts to and from black | embers_C3 -> black (EDIT) -> embers_C3 |
| C | 2840 | C13 E13 -> C13 R13 THE STAR | hard cut | embers_C3 -> ringfall_C (RUN-C-3) |
| C | 2960 | C13 R13 -> C14 H1-C FLINT | hard cut | ringfall_C -> h1_C |
| C | 3000 / 3080 / 3150 | FLINT -> THE FIND -> THE VISION -> THE CATCH | hard cut, continuous, hard cut | h1_C -> ring_C (find_a, find_b: MONTAGE-3D-4) -> h1_C |
| C | 3360 | C14 -> C15 H2 THE FIRE TEST | hard cut | h1_C -> ring_C |
| C | 3600 / 3840 | C15 -> C16 R2-C REVEAL -> C17 THE LIVING INK RUN | hard cuts | ring_C -> runC_reveal -> runC_scroll (RUN-C ink) |
| C | 4160 | C17 -> C18 P4 THE MAP ANSWERS | hard cut (the "burn-through onto the map" is in map_C) | runC_scroll -> map_C (MAP-L2) |
| C | 4480 | C18 -> C19 AC1 THE COUNCIL | hard cut | map_C -> accord_C (v1 timing renders; COUNCIL-C rebuilds) |
| C | 4800 / 5120 | C19 -> C20 -> C21 | continuous | accord_C |
| C | 5360 | C21 -> C22 AC3 · H3 THE UNMAKING | hard cut | accord_C -> ring_C / montage3d melt (MONTAGE-3D) |
| C | 5520 | C22 -> C23 THE FIRE REMAINS · THE STONE | hard cut ("out of the white" is in the renders) | melt -> accord_C |
| C | 5600 | THE STONE -> ROADS OF FIRE | hard cut (the burn-through is in map_C) | accord_C -> map_C |
| C | 5680 | C23 -> C24 R15 THE ILLUMINATION | hard cut | map_C -> runC_illum |
| C | 6160 | C24 -> C25 THE YEAR OF PLENTY | hard cut | runC_illum -> book_C |
| C | 6160-7199 | C25 -> C26 -> C27 -> C28 TITLE | continuous | one book_C render + matte: the page turns (6400, 6720) and the title burn-on (6980-7160) are MAP's |

## >>> STATE (EDIT-2, resumed 18:38Z on the new account; every lane runs continuously) <<<

- **Director's orders (18:38Z):** (1) keep the animatics and the H9 kit fresh with the watcher; (2) wire in
  final_B, fallback_A, fallback_C, later the A and C scores; (3) a DELIVERY CHAIN: full-res masters per film + A's
  ALT master, with QC (black frames, flashes, audio peaks, exact lengths), tested on the partial coverage so the
  last render to a finished master takes minutes. Local renders go through `_local_logs/renderq.py` (3 slots);
  commits name their paths (`git commit -m ... -- <paths>`; the index is shared).
- **Sound:** the resolver already takes the MASTERS table (`music/NOTES_v3.md`): A = `fallback_A.wav`, B =
  `final_B.wav`, C = `fallback_C.wav` (COMPOSER-C re-rendering it at 18:40Z with a level ride; the snapshot waits
  for it to settle). The A and C scores (`final_A.wav`, `final_C.wav`, COMPOSER-A / COMPOSER-C) take over by
  name as soon as they exist. New picture since 15:47Z: RUN-B's `dusk_B` (B1, 640 f, in B frames).
- **Self-queueing scripts:** `animatic.sh`, `h9_kit.sh`, `deliver.sh` re-exec themselves through renderq unless
  `EDIT_Q=1` (set for their children), so a caller cannot bypass the queue and nothing nests.
- **Delivery chain (NEW, `edit/deliver.py` + `edit/deliver.sh`):** see "## Delivery chain" below. First test: B's
  master, queued 18:45Z behind three other lanes' jobs.
- **FILM FINISH (FINISH lane, f855ea5; director's pick 250D_2383_fire):** the masters' last picture stage (see the
  FINISH block below). EDIT depends on a contract FINISH keeps stable (finish/wire_edit.py, eae08fb):
  `AS._finishing(ctx, look=None)`, `deliver._finish_id()`, `deliver.segment_key(..., fin)`. previews.py accepts
  finished or unfinished segment keys (the FINISH backlog) and the H9 kit's stills are finished too (H9_FINISH=0
  = unfinished). If you rewrite `_init`, `segment_key` or `build_video`, re-run `python3 finish/wire_edit.py`.
- **X3 EMBER TITLES (director ~20:25Z; A20 6240-6479, B14 5200-5439):** `edit/ember_title_v3.py` plays the v2
  engine (`edit/ember_title.py`, unchanged) in a v3 scene (`edit/title_scene.py`): static camera, no occluders,
  sparks thrown by the fires found in the plate's first frame (stand-in fires until it lands), titles.py's v3
  type (Cinzel 92/500/0.28 at y 360), the engine's clock mapped from v3 frames by a PCHIP curve (A: letters land
  ~6300-6330, hold, crumble ~6405-6437, dark before the fade at 6456; B: land ~5260-5290, hold, fade into the
  light 5340-5400, never crumble). Layers: `renders/title_{A,B}/f_*.exr` (linear, additive, half float),
  rendered beside the old set and swapped whole, skipped when inputs are unchanged (`meta.json`). The edit
  composites linear_to_srgb(soft_clip(srgb_to_linear(picture) + layer)) and drops titles.py's title while the
  layer plays. Until the plates land (RUN-A2 `bluehour_A`; RUN-B-3 `handback_B`/`dawntitle_B`) the picture is a
  STAND-IN SKY, an EDIT proxy: A = blue hour over serrated knife-edge ranges with watch-fires, paling to rose;
  B = dusk_B's first frame (B's massifs in alpenglow) growing into dawn light. `title_v3.sh` runs on the
  watcher's path before the animatics whenever A or B changes.
- **PREVIEWS (standing request, ~20:25Z):** `edit/previews.py` (`previews.sh`, on the watcher's path after the
  masters): every continuous, fully rendered stretch >= 25 s, cut frame-exactly from the film's master with its
  sound into `~/Downloads/The Long Dawn v3 - PREVIEWS/<film>_bars<a>-<b>_<what>.mp4` + README.txt; only from a
  master built from the current renders; re-exported when it grows or improves. The watcher exits 3 on a new
  file so EDIT can send the director one line. Qualifying now: B bars 1-8 dusk; A bars 14-23 promise-to-giants;
  A bars 34-46 valley-to-roar (once A's master has A9/A10).
- **20:21Z: the watcher runs CONTINUOUSLY** (`bash edit/refresh_watch.sh`, log `_local_logs/animatic/watch_<HHMM>.log`,
  each step's full output in `edit/cache/refresh_watch.run.all`): every settled change refreshes the animatics,
  the kit and the masters for the cuts that changed, then it keeps watching. It exits (waking EDIT) only on a
  failed step, a QC FAIL, a master newly complete (no slates), or 6 h with nothing new; `ONCE=1` = one refresh.
  The 20:00Z refresh proved the incremental path: A, A ALT and C re-muxed onto the newer scores with 0 segments
  encoded; B re-encoded once (433 s) under the per-kind keys with dusk_B.
- **19:54Z: all four masters exist** (`_local_logs/delivery/`): A 155 MB (606 s cold), A ALT (1 of 23 segments
  encoded: 88 s), C 237 MB (718 s cold), B 99.5 MB (19:09Z). Every technical check PASSES: exact lengths, full
  BT.709, black runs only where planned, flashes <= 1/s, true peak -1.30 dBTP, -16.0 LUFS, AAC screeners <=
  -1.15 dBTP; RESULT WARN only for slates. Sound: A = COMPOSER-A's `final_A.wav` (landed 19:14Z, re-rendered
  19:37Z), B = `final_B.wav`, C = COMPOSER-C's `final_C.wav` (19:30Z). The A ember is at (960, 548) in the master.
  Colour verified end to end (a decoded master frame vs its source: mean |d| ~2 at CRF 14 with no hue bias; the
  -2/255 offset first seen was the QC decode's default swscale flags, now accurate_rnd). Watcher re-armed 19:59Z
  (new signature format, so its first refresh is all films: B's animatic with dusk_B, A re-muxed on the newer
  final_A, all animatic segments once under the per-kind keys).
- **~19:25Z:** A11's X2 proxy ember sits still at (960, 548) = (0.5 W, 0.68 H), where EMBERS-A2's A10 ember rests
  at the cut (their A9/A10 final is on the farm into `embers_A3` 2640-3119), until RUN-A-L's X2 plate lands.
  Segment keys are now per kind (frame code for all; slate code only where slates show; X2 code only for A11;
  text code only under text). The watcher's signature now includes each shot's chosen take and its folders'
  mtimes (dusk_B had replaced B1's test at the same 640-frame count and went unseen).
- **Director's EDL calls ~19:20Z:** C14 H1-C rows take `h1_C` first (`H1_C`); C's title is MAP-L's burn-on in
  book space (book_C 6980-7160), so titles.py draws no C title (set `in_picture`; the text table keeps the row).
  Segment keys now split frame code from text code, so a titles.py change re-encodes only shots with text.
- **E15 comp (director, ~19:00Z):** C4 is split at 700; C 700-1039 = `book_C` (page-only) + `embers_C3_e15`
  added (`T(add=...)`: a frame counts only when both layers have it; the X1 test stands in until then).
  EMBERS-C should dim its e15 under C-T2 (text 580-716, so 700-716) and C-T5a (920-1030).
- **B master test (19:02-19:09Z, after 916 s in the queue): the chain works.** 5,440 f and 10,880,000 samples
  exact; no black, no flashes; true peak -1.30 dBTP, -16.1 LUFS; AAC screener -1.22 dBTP; 405 s cold for 15
  segments (13 fps at full res on a loaded Mac); 99.5 MB mov + 43 MB mp4. One defect found and fixed: the
  picture read `bt709/unknown/unknown` (x264 wrote only the matrix), so the join now stamps full BT.709 VUI with
  `h264_metadata` (no re-encode).
- **Animatics now use the same engine** (`animatic.sh` -> `deliver.py --profile animatic`, `animatic_clean` with
  CLEAN=1; `OLD=1` = the old assemble.py path): refreshes re-encode only the shots that changed.
- **Watcher armed 19:13Z** (`_local_logs/animatic/watch_1913.log`): its first refresh is all three films
  (fallback_A/fallback_C/dusk_B since its 15:47Z baseline): the animatics (cold on the engine), the kit, then the
  masters (B re-joined with the VUI fix; A, A ALT and C built cold, ~25 min of slots). The render queue held ~45
  jobs at 19:10Z, so the refresh waits its turn. Re-arm it after each refresh.

## FINISH (lane FINISH): the film finish is LIVE in the masters (27 Sep ~20:55Z, commit f855ea5)

- **The look (director, ~20:35Z): `250D_2383_fire`**: spektrafilm's Kodak Vision3 250D negative printed to 2383
  (baked LUTs), halation, grain 0.5 that renews every frame (seeded per cut frame; no bank, nothing loops), blend
  0.75, the fire rule (never yellower than the source). RUN-C's ink (`runC_*`, plain sRGB): grain only. Code and
  notes: `finish/` (`finish/NOTES.md`); sheets and QC evidence in `_local_logs/review/finish/`. The spektrafilm
  credit is in `edit/CREDITS.md` (for the end crawl).
- **Where (applied by `finish/wire_edit.py`, idempotent):** `assemble._init(..., finish)` wraps the worker Ctx's
  `picture` with `_finishing()`: after the take (crop, per-cut grade, book matte, add layers), before titles and
  burn-ins; slates, black and EDIT proxies are untouched. No Ctx method changed, so `_code_hash()` and the
  animatics' segment keys did not move. `deliver.PROFILES['master']['finish'] = True` (animatics: no finish; the H9
  kit's stills are unfinished too, since `h9_kit.py` builds its own Ctx: `AS._init(cut, v, 1.0, True, True)` or
  `AS._finishing(ctx)` would give it the finish if you want the critic to see it).
- **Incremental:** a master segment with rendered frames is keyed with the finish's identity; if it must be encoded
  anyway (new renders, code changes) it is finished at once; if only the finish is missing (its unfinished twin is
  cached) it joins a BACKLOG finished `FINISH_BUDGET` frames per film per run (default 1200; `FINISH_ALL=1` clears
  it). The log line, the build stats and the QC report it (`[WARN] finish: ... still unfinished` while any
  remains, else `[INFO] finish: ... every rendered frame finished`). Right now (after your 16834cf) every master
  segment is stale anyway, so the next master build finishes all rendered frames: ~5,700 frames, ~76 CPU-min
  (about 25 min extra on the 3 pool workers), then each new render is finished as it lands.
- **Sanity check (director's condition) PASSED:** a finished B master built through this exact chain (wired copies,
  scratch folder) got EDIT's QC: lengths exact (5,440 f, 10,880,000 samples), format 1920x804 bt709, no black
  frames, 0 flashes, -1.30 dBTP, -16.1 LUFS; WARN only for the slates. The darkest night skies (A DESERT, A KARST,
  A embers on black, B's H1 crop) through the CRF 14 master encode show no banding (the grain dithers: the largest
  flat patch of one code in the DESERT sky 3.8% -> 0.5%, KARST 8.1% -> 1.4%). Bitrate: the finished embers shot was
  smaller than before (16.2 vs 17.7 Mb/s), so no `-tune grain` needed.
- **If it ever fails in a worker:** `finish/luts/` must hold the baked cubes (`bash finish/bake_luts.sh`, ~35 s a
  stock; git-ignored); `_finish_id()` raises in the main process first with that instruction.

## Delivery chain (edit/deliver.py, edit/deliver.sh)

- **Masters** in `~/mishamisha/_local_logs/delivery/`: `<cut>_master.mov` (H.264 High, x264 CRF 14 slow, no
  B-frames, 1920x804, 24 fps, BT.709 limited range with a real BT.709 RGB->YUV matrix, + the cut's sound master
  as 24-bit PCM 48 kHz), `<cut>_master.mp4` (same picture, AAC 320k: the screener), `A_master_codedtowers.*`.
- **Incremental:** one segment per EDL shot in `delivery/cache/master/<key>.h264`; the key hashes the shot's EDL
  entry and take, every source frame's path/size/mtime (and the book matte and under-layers), the text over it,
  and the frame-pipeline code (Ctx, grade, slates, burn-ins, lookup, titles.py). Only stale segments are encoded;
  the film is byte-joined (Annex B, no B-frames: frame-exact), muxed with a settled snapshot of the sound master,
  then QC'd. MAIN and ALT share every segment except A6. Unreferenced segments older than 2 h are pruned.
- **QC** (`<name>_QC.txt` / `.json`, RESULT PASS / WARN / FAIL): exact lengths (frames counted from the stream,
  samples decoded), format (codec, 1920x804, 24 fps, bt709, 48 kHz stereo), black frames (sRGB luma p99.5 < 0.03;
  runs inside black shots, the ember on black, X2 and A's fade are planned, others WARN), flashes (BT.1702-style:
  opposing runs of >= 0.1 relative luminance with the darker < 0.8 over >= 25% of the screen; > 3 a second FAILs),
  sound (sample peak, 4x-oversampled true peak <= -1.0 dBTP, clipped samples, integrated LUFS vs -16 +/- 1; the
  AAC screener's true peak), coverage (slate frames left, by shot).

## STATE AT HANDOFF (EDIT-2, 27 Sep ~15:50Z)

- **Animatics** (`~/mishamisha/_local_logs/animatic/`, all exact lengths, checked): `A_animatic.mp4` 4:30.000,
  `A_animatic_codedtowers.mp4` (the ALT master) 4:30.000, `B_animatic.mp4` 3:46.667, `C_animatic.mp4` 5:00.000.
  Picture: A 1,560 f (24%: A5 + A6 from `embers_A3`, H1 re-key, KARST v3 + DESERT v3 finals, X2 proxy), B 892 f
  (16%), C 1,850 f (26%; C4-C11 are still stand-ins: MAP's X1 motion test and EMBERS' pre-H5 half-res preview).
- **ALT master** differs from MAIN only in A6 (bars 19-23, 400 f of `embers_A3_alt_codedtowers`). `animatic.sh`
  rebuilds it whenever ALT frames exist; the H9 kit shows MAIN beside ALT on those bars.
- **Audio:** A = COMPOSER's FALLBACK `music/out/v3/fallback_A.wav` (picked up by name; the MASTERS table row is
  still "pending"); B = the FINAL score `final_B.wav` (15:24Z render, battery all PASS); C = click track until
  `fallback_C.wav`/`final_C.wav` land. Masters are now SNAPSHOTTED before each encode (`assemble.snapshot_audio`:
  waits until the file is the cut's exact length and unchanged 5 s, copies it to `edit/cache/audio_<cut>.wav`,
  deletes it after): COMPOSER rewrote final_B and fallback_A under two encodes today. The MASTERS-table parser now
  takes the first `.wav` in the row's file cell, notes and all. A re-run of A + ALT is queued behind COMPOSER's
  15:40Z `render_v3.py A --fallback` (the 15:34Z A animatic has the previous fallback).
- **EDL/look edits this session:** KARST/DESERT take notes (v3 finals). **B's H1 crop re-framed** for the re-key:
  `B_H1_CROP` (0.17, 0.22, 0.26, 0.26) centres gloves, steel and tinder, clear of the scarf; the 12-frame roar
  (B5) has its own `B_H1_ROAR_CROP` (0.20, 0.08, 0.30, 0.30) so the flame stays in frame as the take pulls back;
  **the B grade** keeps the fire (warm hues AND white-hot cores) and takes everything else to deep night blue
  (the re-key's lifted sky had become flat blue triangles between the basket bars, and the old cool multiply
  turned the roar's white core blue). Slate lines: A8 = one roaring updraft (no vortex, per H5), B8/B10 without
  "(cutaway CUT)", B13 names the fire-steel hand-off, C10 without "if a lane frees up".
- **H9 critic kit READY:** `bash edit/h9_kit.sh` (about 40 s) rebuilds `~/mishamisha/_local_logs/review/h9/` from
  the EDL and the renders on disk (not from the mp4s): `INDEX.md` (reading order, legend, the gate's four
  questions), `COVERAGE.md` (per section: RENDERED / STAND-IN / SLATE / BLACK / EDIT PROXY, and the ALT master),
  `TEXT.md` (every line as drawn, frames, bar and beat, time, set, and the shot and status under it),
  `{A,B,C}_overview.jpg` (every bar on one page with a per-frame status stripe), `{A,B,C}_bars_<a>-<b>.jpg`
  (one frame per bar, 12 a page, bar badge, caption with frame, time, shot, status and the words on screen, or
  the slate's intended shot), `A_ALT_codedtowers_bars_19-23.jpg`, `stills/<cut>/` (full-res clean frames). The
  frame for a bar is beat 3, or the nearest frame where the most lines are fully drawn; every A and C line
  lands fully drawn on at least one sheet frame. Re-run it just before the gate.
- **Seen in the kit, for the director (not EDIT's to fix):** A-T5 and T6a/b cross bright tower columns in A6
  (bars 19-23) but stay legible (EMBERS' `text_band` dims them); B4's between-strikes frames (e.g. src 1276)
  read dark and abstract even in the new crop.
- **NEXT:** keep the animatics fresh as renders land (`bash edit/animatic.sh` in the background; next expected:
  EMBERS' EDGE/BRINK 1840-2639, RUN-C's `runC_{reveal,scroll,illum}`, MONTAGE-3D's Ring shots in `ring_C`);
  wire C's fallback and the A/C scores (automatic on a re-run); `bash edit/h9_kit.sh` before the gate.

## Earlier state (EDIT-v3, 27 Sep ~11:00Z; paused by the director for usage pacing, RESUME 15:00Z)

- **Done and pushed** (`claude/long-dawn-v2`): `edit/edl_v3.py` (the three EDLs as data, checked against the bar maps
  on every run), `edit/assemble.py` (v3; the v2 one is `edit/assemble_v2.py`), `edit/titles.py` (`lines_v3`,
  `composite_v3`; the v2 API is untouched), `edit/animatic.sh`, `edit/edl/edl_{A,B,C}.json`.
- **Animatics** (verified: frame counts, durations, streams, contact sheets of every cut):
  `~/mishamisha/_local_logs/animatic/{A,B,C}_animatic.mp4`, 960x402 H.264 CRF 23 + AAC; A 4:30.000 (click),
  B 3:46.667 (COMPOSER `final_B.wav`, the 10:17Z VIGIL render), C 5:00.000 (click). About 5 min for all three.
  The director re-runs `edit/animatic.sh` as renders land (he will when `renders/h1_v3h5` completes).
- **The director's calls since (applied):** B3 takes HEROINE's job numbering (`heroine_B` 960-1199 = B3 880-1119,
  off +80) after `deadember_B` in locked numbering. The folder table below is now every department's delivery
  convention: finals in these folder names, numbered in cut frames.
- **H5 CALLS in the edit:** the six lines (A-T2, A-T7, A-T9, A-T10a; C-T6, C-T7; frames unchanged), C23 = THE FIRE
  REMAINS (bar 70 ACCORD, bar 71 MAP), E14 gone, B = THE VIGIL (M2/M3 cutaways cut; H1 in B cropped to hands, tinder
  and sparks with a cool grade; its roar crop is 12 f because the take dissolves to A/C's wide from src ~1488).
- **Verified:** EMBERS' text-band dimming (`shots/embers/render.py` TEXT['A3'], TEXT['C3']) matches the bar maps to
  the frame and is enabled for A3 and C3 (`variant.text_band()`); the band (y 560-700) covers the lower third. The
  `--variant codedtowers` path passed a synthetic test (ALT frames replace MAIN only where they exist; coverage
  reports VARIANT). A-T6a/T6b share one row; A-T10a/b and C-T8a/b stack on black; C-T2 is legible at 1:1 on the
  brighter page (f596).
- **NEXT (at RESUME, the director's order):** (1) the ALT master `A_animatic_codedtowers.mp4` (it builds only once
  `renders/embers_A3_alt_codedtowers` has A-timeline frames; the variant is otherwise identical to MAIN); (2) the
  fallback-master audio for A and C (the resolver already takes `music/out/v3/fallback_<cut>.wav` or the MASTERS
  table in `music/NOTES_v3.md`; re-run the animatics when they land); (3) H9 gate prep (clean versions with
  `CLEAN=1`, contact sheets, the lines).
- **Open, lower priority:** the X1 glow/matte comps other than the book matte (`x1_map_C` onto the map at C17 to C18);
  the MOTION-LAB E2 challenger is not in the cut; C ink-line positions are the lower third until MAP gives page
  coordinates; `edit/mix.py` is the v1 re-mix and is not used in v3 (the masters come from `music/src/render_v3.py`).

## How the EDLs work

- Every shot is `[f0, f1)` on its cut's own timeline, with TAKES in priority order (`edl_v3.py`). A take is a render
  folder STEM plus `off` (`src = cut frame + off`) and a mode: `v3` = `renders/<stem>_<CUT>/` then `<stem>_v3/`;
  `layered` = those, then `<stem>_v2/`, then `renders/<stem>/` (REUSE of v2 frames in src numbering); `exact` =
  `renders/<stem>/` only (a department's own v3 folder, e.g. `embers_A3`, `runC_scroll`).
- `--variant codedtowers` tries `<folder>_alt_codedtowers` before every folder (the ALT folders hold only the
  frames that differ). A's ALT master is built by `animatic.sh` only once `renders/embers_A3_alt_codedtowers` has
  frames; until then it would equal MAIN.
- A shot uses the first take that covers every frame; otherwise the take covering the most frames, with the gaps as
  slates. A shared take can require a complete src range (`need=`): the H5 re-key of H1 (`renders/h1_v3h5`, landing
  now) replaces `h1_v3` in all three cuts at once when all of 1200-1555 is in, so no cut mixes the two looks.
- Book shots (MAP-v3) composite `rgb + (1 - matte) * under` from `book_C` + `book_C_matte`, per MAP's spec.
- Slates: the shot code, owner, name, one line, bars and frames in the top third (the lower third and centre stay
  free for the text), a progress line with beat ticks, and a beat light (brighter on downbeats) for sync checks.
- Burn-ins (off with `--clean` or `CLEAN=1`): shot and source frame top left; bar, beat, frame and seconds top right.

## Where each department's frames are expected (one line in `edl_v3.py` to change)

| lane | shots | folder (numbering) |
|---|---|---|
| EMBERS | A3-A10, A16-A17; C6-C7, C9, C11, C13a; C4b-C5 fire | `embers_A3` (+`_alt_codedtowers`), `embers_C3` (cut frames); `embers_C3_half` as a stand-in; **`embers_C3_e15` (C 700-1039, added over MAP's page)** |
| RUN-A | A2, A13 R2-A, A14, A15, A18, A19-A20 | `falsedawn_A`, `reveal_A`, `beaconrun_A`, `watchers_A`, `crossing_A`, `bluehour_A`, or `run_A` (A frames) |
| RUN-B | B1, B5 R2-B, B6-B12, B13-B14 | `dusk_B`, `reveal_B`, `vigil_B`, `handback_B` (B frames; the cloud jobs already use these) |
| RUN-C | C13b, C16, C17, C24 | `ringfall_C`/`run_C` (C frames); `runC_reveal` (-3600), `runC_scroll` (-3840), `runC_illum` (+2398-5680) |
| HEROINE | H1 all cuts; B2, B3; C14 H1-C | `h1_v3h5` then `h1_v3` (src 1200-1555; strike 1 = 1236, roar = 1476); `climb_B`/`deadember_B`/`heroine_B` (B frames); **`h1_C` (C frames 2960-2999 and 3150-3359: HEROINE-L's hands and flint, first in both C14 H1-C rows)** |
| MONTAGE-3D-2 | A13 KARST, DESERT; C14 find, C15 fire test, C22 melt | `montage3d_v3/karst_slow` (0-59); `montage3d_v3/desert` or `montage_v2` 1520-1579; `ring_C` or `montage3d_v3/{find_a,find_b,fire,melt}` (C frames) |
| MAP | C2-C5, C8, C9 head, C18, C23 bar 71, C25-C28 | `book_C` + `book_C_matte` (**page-only at C 700-1039**: EMBERS-C's `embers_C3_e15` is added over it: book_rgb + (1 - matte) * black + e15); `map_C` 4160-4479 and 5600-5679 (C frames) |
| ACCORD | C19-C23 bar 70 | `accord_C` (C frames 4480-5599) |

Heroine note: the RENDER_SPEC on `claude/v3-heroine` puts THE DEAD EMBER at "B 960-1199" (pre-lock numbering); the
locked sheet has B3 at 880-1119. Frames must arrive in the locked numbering or the EDL line needs an `off`.

## Text (titles.py v3)

- Words from `music/v3/barmap_<cut>.json`, with `H5_WORDS` applied on top; the table prints with
  `python3 -c "import titles; print(titles.text_table('A'))"` from `edit/`.
- A: Cormorant Garamond italic 56, lower third (y 648), letters staggered inside a 12-frame fade in and a 12-frame
  fade out, all inside [f_in, f_out); T6a and T6b on one row (T6b joins at 1720); T10a/T10b centred on black.
- C: ink lines (T1, T7, T9, T14) in EB Garamond italic, a slanted pen-nib wipe over 24 frames with a little bleed at
  the nib, a grainy dissolve out over 12; iron-gall ink on a light page, parchment white on a dark ground. Fire lines
  (T2, T5a, T5b, T6, T10, T11, T12; T8a/T8b stacked on black) kindle over 12 frames (staggered heat, glow) and crumble
  over 12 into rising sparks.
- Titles (X3, Cinzel): A kindles in the sky, holds, crumbles into rising sparks (last 46 f) before the fade to black
  from bar 81 b3.8 (f 6456); B kindles and fades into the light (last 60 f); C burns on in fire-letters and cools to ink.
- **EMBERS text dimming (checked):** `shots/embers/render.py` TEXT['C3'] = (1056, 1190) T5b and (1446, 1550) T6, which
  are exactly the bar map's windows for the lines over EMBERS' C shots (T8a/b sit on black; T7 is on MAP's page).
  TEXT['A3'] = T4, T5, T6a/b, T7, T8, T9 also match the bar map to the frame. The H5 calls change words only, so no
  dimming window moves. If EMBERS ever renders C5 (E15's tail) it needs (920, 1030) for T5a.

## Coverage
<!-- COVERAGE:BEGIN -->
_Generated by `edit/assemble.py --coverage --notes` at 28 Sep 00:59Z._

**A** (6480 f): 5600 f picture (86%), audio: COMPOSER master `music/out/v3/final_A.wav`

| sec | shot | frames | bars | status | source / expected folder |
|---|---|---|---|---|---|
| A1 | A1 BLACK | 0-80 | 1-1 | EDIT (black) | edit |
| A2 | R1 FALSE DAWN | 80-560 | 2-7 | RENDERED | falsedawn_A 80-559 |
| A3 | E1 INTO THE LIGHT · GLYPHS | 560-960 | 8-12 | RENDERED | embers_A3 560-959 (EMBERS v3, A timeline) |
| A4 | E1 THE POINT | 960-1040 | 13-13 | RENDERED | embers_A3 960-1039 (EMBERS v3, A timeline) |
| A5 | E2 · E3 IGNITION · THE PROMISE | 1040-1440 | 14-18 | RENDERED | embers_A3 1040-1439 (EMBERS v3, A timeline) |
| A6 | E5-A THE TOWERS · TWO GIANTS | 1440-1840 | 19-23 | RENDERED | embers_A3 1440-1839 (EMBERS v3, A timeline) |
| A7 | E6 THE EDGE | 1840-2400 | 24-30 | RENDERED | embers_A3 1840-2399 (EMBERS v3, A timeline) |
| A8 | E7-A THE BRINK · OVER THE RIM | 2400-2640 | 31-33 | RENDERED | embers_A3 2400-2639 (EMBERS v3, A timeline) |
| A9 | E4 THE DEAD VALLEY | 2640-2800 | 34-35 | RENDERED | embers_A3 2640-2799 (EMBERS v3, A timeline) |
| A10 | E9 BLACK · THE EMBER | 2800-3120 | 36-39 | RENDERED | embers_A3 2800-3119 (EMBERS v3, A timeline) |
| A11 | X2 DARK ADAPTATION | 3120-3360 | 40-42 | RENDERED | stars_A 3120-3359 |
| A12 | H1-A THE FIRST FIRE | 3360-3600 | 43-45 | RENDERED | h1_v3h5 1236-1475 (H1 v3 master timing with the H5 calls (hood, shawl, thin gloves, orange sparks)) |
| A13 | H1-A THE ROAR | 3600-3680 | 46-46 | RENDERED | h1_v3h5 1476-1555 (H1 v3 master timing with the H5 calls (hood, shawl, thin gloves, orange sparks)) |
| A13 | R2-A EVERY RIDGE | 3680-3800 | 47-48 | RENDERED | reveal_A 3680-3799 |
| A13 | M5 KARST | 3800-3860 | 48-49 | RENDERED | montage3d_v3/karst_slow 0-59 (Blender KARST v3 final, 2/3 speed (H5: rock towers in mist)) |
| A13 | M5 DESERT | 3860-3920 | 49-49 | RENDERED | montage3d_v3/desert 0-59 (Blender DESERT v3 final (H5: wide, plain cloak, irregular prints)) |
| A14 | R3 THE BEACON RUN | 3920-4240 | 50-53 | SLATE | expects renders/beaconrun_A, renders/run_A |
| A15 | R16 THE WATCHERS | 4240-4400 | 54-55 | SLATE | expects renders/watchers_A, renders/run_A |
| A16 | E10 TOWERS IN THE LIGHT | 4400-4720 | 56-59 | RENDERED | embers_A3 4400-4719 (EMBERS v3, A timeline) |
| A17 | E10 THE FIRE, SEEN | 4720-4880 | 60-61 | RENDERED | embers_A3 4720-4879 (EMBERS v3, A timeline) |
| A18 | R6 THE CROSSING | 4880-5840 | 62-73 | RENDERED | crossing_A 4880-5839 |
| A19 | R7 THE BLUE HOUR | 5840-6240 | 74-78 | SLATE | expects renders/bluehour_A, renders/run_A |
| A20 | X3 TITLE | 6240-6480 | 79-81 | EDIT proxy | stand-in sky (plate pending: renders/bluehour_A, renders/run_A) + ember title renders/title_A |

A with `--variant codedtowers`: A6 VARIANT (400 alt frames), A7 VARIANT (560 alt frames), A8 VARIANT (240 alt frames), A16 VARIANT (320 alt frames), A17 VARIANT (160 alt frames)

**B** (5440 f): 5440 f picture (100%), audio: SOUND master `music/out/v3/sound_B.wav`

| sec | shot | frames | bars | status | source / expected folder |
|---|---|---|---|---|---|
| B1 | R8 DUSK | 0-640 | 1-8 | RENDERED | dusk_B 0-639 |
| B2 | H4 THE CLIMB | 640-880 | 9-11 | RENDERED | climb_B 640-879 |
| B3 | H5 THE DEAD EMBER | 880-1120 | 12-14 | RENDERED | deadember_B 880-1119 |
| B4 | H1-B THE FIRST FIRE | 1120-1360 | 15-17 | RENDERED | h1_v3h5 1236-1475 (H1 v3 master timing with the H5 calls (hood, shawl, thin gloves, orange sparks)) |
| B5 | H1-B THE ROAR | 1360-1372 | 18-18 | RENDERED | h1_v3h5 1476-1487 (H1 v3 master timing with the H5 calls (hood, shawl, thin gloves, orange sparks)) |
| B5 | R2-B THE REVEAL | 1372-1520 | 18-19 | RENDERED | reveal_B 1372-1519 |
| B6 | R9 THE VIGIL · FIRST HOURS | 1520-2000 | 20-25 | RENDERED | vigil_B 1520-1999 (RUN-B THE VIGIL) |
| B7 | R9 THE VIGIL · NOTHING ANSWERS | 2000-2480 | 26-31 | RENDERED | vigil_B 2000-2479 (RUN-B THE VIGIL) |
| B8 | R9 THE VIGIL · SOMEONE HAS SEEN | 2480-2640 | 32-33 | RENDERED | vigil_B 2480-2639 (RUN-B THE VIGIL) |
| B9 | R9 THE VIGIL · THE FIRST ANSWERS | 2640-3120 | 34-39 | RENDERED | vigil_B 2640-3119 (RUN-B THE VIGIL) |
| B10 | R9 THE VIGIL · FIRE ANSWERS FIRE | 3120-3280 | 40-41 | RENDERED | vigil_B 3120-3279 (RUN-B THE VIGIL) |
| B11 | R9 THE VIGIL · THE WORLD COMES | 3280-3600 | 42-45 | RENDERED | vigil_B 3280-3599 (RUN-B THE VIGIL) |
| B12 | R9 THE VIGIL · THE CHILD | 3600-3840 | 46-48 | RENDERED | vigil_B 3600-3839 (RUN-B THE VIGIL) |
| B13 | R11 THE HAND-BACK | 3840-5200 | 49-65 | RENDERED | handback_B 3840-5199 |
| B14 | X3 TITLE | 5200-5440 | 66-68 | EDIT proxy | stand-in sky (plate pending: renders/handback_B, renders/dawntitle_B) + ember title renders/title_B |

**C** (7200 f): 6840 f picture (95%), audio: SOUND master `music/out/v3/sound_C.wav`

| sec | shot | frames | bars | status | source / expected folder |
|---|---|---|---|---|---|
| C1 | C1 BLACK · THE HEARTH | 0-80 | 1-1 | EDIT (black) | edit |
| C2 | P1 THE RED BOOK | 80-320 | 2-4 | RENDERED | book_C 80-319 (MAP-v3 book engine) |
| C3 | P2 INK PAGE · THE MOUNTAIN | 320-560 | 5-7 | RENDERED | book_C 320-559 (MAP-v3 book engine) |
| C4 | E15 · X1 LETTERS TO FIRE | 560-700 | 8-9 | RENDERED | book_C 560-699 (MAP-v3 book engine) |
| C4 | E15 · X1 LETTERS TO FIRE · THE FIRE CATCHES | 700-880 | 9-11 | RENDERED | book_C 700-879 (MAP-L page (page-only) + EMBERS-C E15 fire, additive) |
| C5 | E15 THE FIRE, ALONE | 880-1040 | 12-13 | RENDERED | book_C 880-1039 (MAP-L page (page-only) + EMBERS-C E15 fire, additive) |
| C6 | E5-C THE FORGING | 1040-1440 | 14-18 | RENDERED | embers_C3 1040-1439 (EMBERS v3, C timeline) |
| C7 | E11 THE RACE UNDER THE RING | 1440-1680 | 19-21 | RENDERED | embers_C3 1440-1679 (EMBERS v3, C timeline) |
| C8 | P2 INK PAGE · THE DEEP | 1680-1920 | 22-24 | RENDERED | book_C 1680-1919 (MAP-v3 book engine) |
| C9 | E12 THE EYE (burn-through) | 1920-1992 | 25-25 | RENDERED | book_C 1920-1991 (MAP-v3 book engine) |
| C9 | E12 THE EYE ONTO NOTHING | 1992-2080 | 25-26 | RENDERED | embers_C3_half 1992-2079 (EMBERS v3 half-res preview, pre-H5) |
| C10 | M4 THE MIRROR (optional) | 2080-2320 | 27-29 | RENDERED | mirror_C 2080-2319 |
| C11 | E8-C THE GRASP THAT CANNOT HOLD | 2320-2480 | 30-31 | RENDERED | embers_C3_half 2320-2479 (EMBERS v3 half-res preview, pre-H5) |
| C12 | C12 BLACK · THE OLD LAW | 2480-2720 | 32-34 | EDIT (black) | edit |
| C13 | E13 THE RING FALLS | 2720-2840 | 35-36 | SLATE | expects renders/embers_C3, renders/embers_C3_half |
| C13 | R13 THE RING FALLS · THE STAR | 2840-2960 | 36-37 | RENDERED | runC_ringfall 2840-2959 (RUN-C ink final) |
| C14 | H1-C FLINT | 2960-3000 | 38-38 | RENDERED | h1_v3h5 1216-1255 (H1 v3 master timing with the H5 calls (hood, shawl, thin gloves, orange sparks)) |
| C14 | H2 THE FIND | 3000-3080 | 38-39 | RENDERED | ring_C 3000-3079 (MONTAGE-3D-2 Blender Ring) |
| C14 | H2 THE FIND · THE VISION | 3080-3150 | 39-40 | RENDERED | ring_C 3080-3149 (MONTAGE-3D-2 Blender Ring) |
| C14 | H1-C FLINT · THE CATCH | 3150-3360 | 40-42 | RENDERED | h1_v3h5 1266-1475 (H1 v3 master timing with the H5 calls (hood, shawl, thin gloves, orange sparks)) |
| C15 | H2 THE FIRE TEST | 3360-3600 | 43-45 | SLATE | expects renders/ring_C, renders/montage3d_v3/fire |
| C16 | R2-C THE REVEAL | 3600-3840 | 46-48 | RENDERED | runC_reveal 0-239 (RUN-C ink final) |
| C17 | R12 THE LIVING INK RUN | 3840-4160 | 49-52 | RENDERED | runC_scroll 0-319 (RUN-C ink final) |
| C18 | P4 THE MAP ANSWERS · THE ROAD | 4160-4480 | 53-56 | RENDERED | map_C 4160-4479 (MAP-v3 map (C timeline)) |
| C19 | AC1 THE COUNCIL | 4480-4800 | 57-60 | RENDERED | accord_C 4480-4799 |
| C20 | AC4 BRING OUT THE RING | 4800-5120 | 61-64 | RENDERED | accord_C 4800-5119 |
| C21 | AC2 THE BEARER | 5120-5360 | 65-67 | RENDERED | accord_C 5120-5359 |
| C22 | AC3 · H3 THE UNMAKING | 5360-5520 | 68-69 | RENDERED | ring_C 5360-5519 (MONTAGE-3D-2 Blender Ring) |
| C23 | AC THE FIRE REMAINS · THE STONE | 5520-5600 | 70-70 | RENDERED | accord_C 5520-5599 |
| C23 | P4 · X1 THE FIRE REMAINS · ROADS OF FIRE | 5600-5680 | 71-71 | RENDERED | map_C 5600-5679 (MAP-v3 map (C timeline)) |
| C24 | R15 THE ILLUMINATION | 5680-6160 | 72-77 | RENDERED | runC_illum 2398-2877 (RUN-C ink final) |
| C25 | P2 THE YEAR OF PLENTY | 6160-6400 | 78-80 | RENDERED | book_C 6160-6399 (MAP-v3 book engine) |
| C26 | P2 THE HAVENS | 6400-6720 | 81-84 | RENDERED | book_C 6400-6719 (MAP-v3 book engine) |
| C27 | P1 THE LAST PAGES | 6720-6960 | 85-87 | RENDERED | book_C 6720-6959 (MAP-v3 book engine) |
| C28 | X3 TITLE | 6960-7200 | 88-90 | RENDERED | book_C 6960-7199 (MAP-v3 book engine) |

<!-- COVERAGE:END -->
