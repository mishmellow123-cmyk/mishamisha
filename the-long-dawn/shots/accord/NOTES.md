# ACCORD v3 (ACCORD-v3, 27 Sep): cut C only, C frames 4480-5679 -> `renders/accord_C3/f_%05d.png`

## >>> ACCORD-CROWD STATE (27 Sep 20:40Z; usage pause ~21:25Z -> resume ~23:30Z) <<<
* **Code (all pushed, 1bbea59 + 54b9c5b):** `crowd/crowd3.py` (roads, 1,123 walkers, the ring round the stones, her walk-in,
  the bar-70 relight wave + walk-out, torchlight grids, the land, figures, flames, smoke), `crowd/accord3c.py` (driver),
  `crowd/crowdmap.py` (top-down map, no render). accord3.py carries the guarded hook (crowd installs on import;
  `CROWD=0` off; `CROWD_REQUIRED=1` finals).
* **Looks:** look 1 `renders/_farmtest/crowd3_look_v1/`, look 2 `.../crowd3_look_v2/` (check sheet
  `renders/accord_C/tests/crowd/check_look2.jpg`). Look 3 (the fixes since look 2: the R_FAR road bug, irregular crowd,
  no scallop capes, land folds, small heather) = farm request `0927-163255-crowd3look-69903` -> lands in
  `renders/_farmtest/crowd3_look/` (frames 4480..4690, 5556, 5566, 5580..5640).
* **JOB READY sent 20:37Z:** `cloud/jobs/crowd3_ac1.json` (4464-4719, --nodes 2) and `cloud/jobs/crowd3_p3.json`
  (5520-5679, --nodes 2; only with ACCORD-3's P3 signed off). Launch on approval (from ~/mishamisha):
  `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/crowd3_ac1.json --nodes 2 --detach` (same for p3).
  Frames land in `renders/accord_C/` (C numbering). Check them: 4480 (ring centred ~960,402, ~105 px), 4600, 4690
  (her red shawl beside the council), 5566 (front rank's dip), 5620/5640 (the tide outward).
* **Next (after 23:30Z):** review look 3 + the landed finals; nitpicks: figures from straight above still read a little
  as beans at 60 m (options: a longer shadow from a lower torch, a visible shaft); P3 near silhouettes (the shared
  sd_fig) read as smooth pawns in the 5560 low oblique; ACCORD-3's floor inside the stones reads as camouflage from
  20-60 m up.

## ACCORD-3 WORKING STATE (27 Sep ~19:25Z; renders go to THE FARM now, see COMMON.md)
* **Crowd handed over.** ACCORD-CROWD (director's split, 18:50Z) owns the rivers, the crowd and the walk-out. My
  plain3 draft is deleted and accord3 is back to `igz` grids / P_WI 0 / default P_IGC_*/P_IGF_* until their hooks
  land; shade3's P_CROWD/P_WI terms untouched. **Accepted:** her walk-in: `scene3.her_state` takes
  `crowd3.her_walkin(t)` for t < 4760 (guarded import of `crowd/crowd3.py`; without it she stands in her place).
* **Built this session (pushed, fbb822b):**
  - Bar 70's fire that remains: `flame3.calm_density` (CF flames from `scene3.calm_flames(t)`: a tall heart + 5 on
    the coals on the stone's top, 16 thin tongues along the burning logs; one shared tongue/crinkle field sized to
    the owning flame; translucent), tight 2-interval march; `accord3.calm_lights` in the flame body (CALM_LIGHT 9);
    coal bed on the stone (`shade3` P_COAL); ash sooted darker, deep-red sparse embers, log glow along the checks;
    haze x0.35 in P3; P3 camera oblique behind her right shoulder (P3_PHI0 187, tilt 39 -> 5 by 5604, h 3.25 -> 62);
    the bearers stand 0.40 m closer (P3_STEP) and dip into the fire's flank.
  - Hands: `geom3.sd_hand` arm mode (HD[16:24]: elbow/shoulder; wool sleeve with creases wrist -> elbow -> her
    shoulder; `scene3.set_arm`), her P2 pose relaxed per finger, she leans in 34 deg over the hearth (kneels at
    1.00 m); stitched points + sheen on the leather; the gilded crust = `geom3.gilt_crust` (lobed sheet + tongues
    along 3 fingers, raised 1.6 mm with a bead) shaded as metal (no diffuse, sharp + broad spec, cracks, live embers).
  - The Ring's glint (`accord3.ring_glint`, P1 only): sum of torch irradiance at the band, gold, after the lens.
* **Look-dev:** `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/accord3_look.json --test 12` (from
  ~/mishamisha) -> `renders/_farmtest/accord3_look/`; frames 4660,4840,4960,5040,5100,5160,5250,5300,5548,5566,
  5590,5625. Fire alone locally (light): `python shots/accord/firelook3.py out.jpg 5548,5566`.

## ACCORD-CROWD (crowd lane, split off 27 Sep ~18:50Z): the interface with ACCORD-3
* **ACCORD-CROWD STATE (27 Sep ~20:15Z).** crowd3 is pushed and HOOKED: accord3.py's guarded block (54b9c5b) calls
  `crowd3.install()` on import, so `accord3.py range/still` renders the crowd. **`CROWD=0`** = council only (fast
  look-dev); **`CROWD_REQUIRED=1`** for finals. The crowd's flames follow `accord3.TORCH_I`, your flame alpha and
  fire_clip (they render into your `tb`), your per-frame haze (recorded from `FL3.airlight`) and `P_OWNK`.
  Director APPROVED the bar-70 wave (27 Sep): the crowd gives its fire to the hearth at 5120 (torches down, out); after
  the white it comes back out torch to torch (front rank ~5566, the stones ~5610, r 12 m ~5640) and leaves along the
  roads; near ranks read as a gesture (tilt, touch, draw back), far ranks as a spreading tide.
  Look 1 (farm, `renders/_farmtest/crowd3_look_v1/`, 12 frames, ~31 s/frame/process): the structure works; fixed in
  look 2 (farm queue): pale cloth read as beige shells/cupcakes -> darker wools; torch held out at arm's length; the
  far-field light now blocked by the crowd's bodies (density transmittance: shadows read, no tan floor); the moon on
  the open moor (x4, the far dark reads as land), roads darker than the grass (no beams); torch smoke plumes.
  Files: `crowd/crowd3.py`, `crowd/accord3c.py` (driver), `crowd/crowdmap.py` (top-down debug map, no render),
  `cloud/jobs/crowd3_look.json`.
* **Finals plan (proposed; renders go through accord3.py itself, the hook installs the crowd, CROWD_REQUIRED=1):**
  `cloud/jobs/crowd3_ac1.json` = AC1's descent 4464-4719 (4464-4479 a head handle for EDIT's dissolve from MAP's ring;
  the camera holds at ~294 m), `--nodes 2`, ~17 min per node. `cloud/jobs/crowd3_p3.json` = bar 70 5520-5679, to run
  once ACCORD-3's P3 fire and council are signed off too (either lane may launch it). ACCORD-3 renders 4720-5519 with
  the hook in place (the crowd there is light only: its torchlight on the floor and P_CROWD; it is off screen).
* **For ACCORD-3 (seen in look 1, your side):** from 20-60 m up the floor inside the stones reads as camouflage
  blotches (ground_albedo's turf/earth fbm at ~0.6 m); the council from straight above reads as pale sacks; in the
  5560 low oblique the near silhouettes read as smooth chess pawns (the torch arm out from the body helps; I did that
  for the crowd).
* **Split.** ACCORD-3 keeps the hearth, the Ring, the gloves, the beats and the inner council (scene3's 13 + her).
  ACCORD-CROWD has everyone else: the rivers of torches (AC1's descent), the crowd's ring round the stones, its torches
  through AC4/AC2, the bar-70 walk-out, and the land beyond the stones (r > 9 m: worn roads, gentle relief).
* **Module:** `shots/accord/crowd/crowd3.py`, in its own subfolder so its edits never flush accord3's numba cache (it
  keeps its own stamp over `accord/*.py` + `crowd/*.py`).
* **How it plugs in (no edits to your pipeline):** `crowd3.install(accord3)` points accord3's `SH` and `FL3` names at
  thin proxies. `SH.render_surfaces` runs yours with the crowd's ground grids (igf/igc, PR[P_WI]/P_IGC_*/P_IGF_*),
  `OC` plus the crowd's bodies near the fire (P3) and `P_CROWD` scaled by the crowd's lit fraction; after your AA pass
  it adds the land beyond r 9 m (as a ratio to shade3's ground, so no seam) and the crowd's figures (z-tested,
  anti-aliased). `FL3.torch_flames` runs yours, then the crowd's flames (flame3.torch_density, supersampled when
  small) and their airlight. Every crowd step is guarded: a crowd failure prints a traceback and the frame renders
  without it, unless `CROWD_REQUIRED=1` (finals). `CROWD=0` renders without the crowd (fast council look-dev).
  Driver: `python shots/accord/crowd/accord3c.py range --frames ... --outdir ...` (= accord3 + crowd).
  **The one hook in accord3.py** (added once crowd3 has passed its farm test; a guarded block before `__main__`) calls
  `install()` on import, so your own `accord3.py range` renders the crowd too.
* **Layout and beats (C numbering).** Roads (trunks with merging tributaries) end in aisles between the stones.
  4480-4700 the rivers converge and the arrivals settle in ranks round the stones (r 8.4-14 m; nobody inside the
  stones but your 14). Through AC4/AC2 every crowd hand holds a lit torch upright and nobody moves toward the centre.
  5120-5200 the crowd's torches come down with yours (P_CROWD fades). **Bar 70 (proposed; asked of the director):**
  after the white the crowd has closed in round the fire (front rank r ~2.6 m, behind your council at 1.32 m), its
  torches spent; your 13 dip at ~5534 and draw back lit; then the flame passes torch to torch outward (front rank
  ~5566, the stones ~5610, r 12 m ~5640): each bearer dips toward the lit torch in front, draws back lit, turns and
  walks out along the roads, so the lights stream OUTWARD under MAP's burn-through (5600-5640).
* **Asks of ACCORD-3:** (a) keep shade3's `P_CROWD`/`P_WI`/igf/igc terms as they are (the crowd now drives them);
  (b) her walk-in: she arrives along the road from screen-right at 4480 (MAP's Road comes from the east) and crosses
  the open floor to her place by ~4700. `CR.her_walkin(t)` gives pos/ang/walk/phase for t < 4730. You own her, so wire
  it into `her_state` if you agree.

## STATE AT HANDOFF (27 Sep 18:40Z, ACCORD-3 taking over from ACCORD-2; brief = `_local_logs/handoff/brief3_ACCORD.md`)
* **On disk:** v3 engine pushed (118eafd + e0ea266). No ACCORD render process running. Test stills only, in
  `renders/accord_C/tests/` (sheets a-e = ACCORD-2's last look: 4840/4960/5040/5100/5160/5180/5540-5595).
  No v3 frames delivered. `renders/accord_C3/tests/` = the first look (superseded; to delete).
* **MONTAGE-3D-3's melt (`renders/ring_C/f_05360..05519`) does not exist yet** (their order: find_a, find_b, fire, melt).
* **ACCORD-3's order (director):** (1) the hearth fire so bar 70 is unmistakably warm; (2) the gloves (hers in P2 =
  a puppet's hand; the gilded one = a gold brick); (3) the Ring's glint at 4840; (4) the crowd + rivers of torches
  (AC1 descent 4480-4700); (5) check stills per beat, cloud jobs <= 45 min each, JOB READY; (6) AC3 once the melt exists.
* **My read of sheets a-e:** bar 70's fire is a pale donut round a pale octagon (the stone), glitter for embers, the
  figures pale beige "cups" (airlight veil + fill); torches from above = white cotton (over-exposed: ACES whites them);
  P2 5160 = a grey dish of pills with a puppet arm. All frames sepia-muddy (the haze veil).


## >>> PAUSED 27 Sep ~18:00Z (ACCORD-2; director: usage window end; RESUME 20:00Z). No render running. <<<
**Done this session (source pushed: it landed inside RUN-B's commit 118eafd, which swept up the shared git index while
ACCORD-2's files were staged; commit with `git commit <paths>` to avoid the race; test stills only, in
`renders/accord_C/tests/`):**
1. Torch flames rebuilt (`flame3.torch_density/torch_flames`): big tongues (fBm) + crinkle (turbulence), shell
   emission, hot middle third, flicker `torch_flicker()` shared with the torch's light; tight world-box march;
   Hf 0.44, Rf 0.088, I 22; a breeze `_WIND` (0.40,-0.16); the lit head glows as fuel under the flame (no corn cob).
   Look-dev: `python shots/accord/flamelook3.py out.jpg`. Airlight haze `flame3.airlight` (HAZE 0.006, lam 0.35 m).
2. Hearth: faceted 7-face slab (`FLAT_A/FLAT_D`, chips, frost crack, dark granite, STONE_TOP now 0.30), irregular
   sunk kerb field stones, charcoal scattered by 4.5 cm cells (`sd_charcoal`), 8 star-fire logs (bark furrows + char
   crackle keyed on each log's axis), kindling, soft grey ash, ground contact AO, ground relief/turf/pebbles.
   **Bug fixed:** `sd_hearth` returned 1e9 where no part was evaluated (rays leapt out: the "dotted"/cow-print
   artefacts) and false-hit its bound from far cameras (black disc at 5595): now a safe bound with a 1.2 cm margin.
3. Figures: `_hood3` (close hood, seam, peak/liripipe/capuchin) + a sloping capelet (the collar torus = the
   "mushroom brim" is gone); no velvet edge-glow; no olive/green cloth; her shawl a clinging madder wool drawn up
   round her neck (SCARF 0.070,0.008,0.010); rounder darker fists; the hearth rim light scales with albedo (figures
   no longer go white in the fire).
4. **Gilded hand = figure 0** (broad black mantle, beside her); `scene3.gilt_find_cam`: 4990-5042 the orbit descends
   to 1.4 m from the knuckles, 34 deg off vertical, from the back-of-hand side; exposure stops down; the hero glove's
   cloak sleeve ends short; the gold is specular (diffuse 0.10). P1 orbit before 4990 is unchanged (4480 match kept).
5. P2 camera low over the slab (2.6 -> 1.72 m; her head out of frame, only her arm enters); exposure stops down as
   the torches come down; torch light centre = flame base + 0.22 m.
6. Bar 70: calm fire converges over the stone, soft ember bed (no glitter), per-lump charcoal glow, fire-blackened slab.
7. Hearth fire density rebuilt (tongues + crinkle + shell, hot roots): IN PROGRESS. Look-dev:
   `python shots/accord/firelook3.py out.jpg 5210,5300,5560,5580` (fire volume alone, ~1 min incl. compile).
**Verdict of the last stills (half res):** 4840 reads (she kneels and reaches) but the Ring is invisible (needs a
glint/closer); 4960 decent but brownish-muddy; 5040 the hand is framed large but reads as a "gold brick" (fingers,
partial crust over leather, ember cracks needed); P2 5160 slab OK now, her glove a puppet hand, ash blown out;
bar 70 staging works (every torch dips in, draws back lit, turns, walks out) but the fire is a pale flat disc.
**NEXT (in order):** (1) finish the hearth fire in firelook3 (P2: less white, orange tongues, the hollow; P3:
denser convergence so the fire covers the stone, warm orange-gold); (2) the gilded glove's read; (3) her glove in
P2; (4) the Ring's glint at 4840; (5) the crowd + rivers of torches for AC1's descent 4480-4700 (NOT STARTED:
P1 above ~10 m is still an empty plain); (6) cloud jobs `cloud/jobs/accord3_*.json` (ship jpg) + JOB READY + check
stills; (7) AC3 composite once `renders/ring_C/f_05360..` exist (MONTAGE-3D-2's melt: not yet rendered).
Numba: any edit to any *.py in this folder recompiles everything (~2 min on the first still).
Delete when superseded: `renders/accord_C3/` (the previous agent's first look), `renders/accord_C/tests/*`.


## STATE AT HANDOFF (27 Sep 15:15Z, ACCORD-2 taking over from ACCORD-v3; brief = `_local_logs/handoff/brief2_ACCORD.md`)
* **On disk:** v3 engine committed + pushed (7442e33, a951565): `accord3.py`, `scene3.py`, `geom3.py`, `shade3.py`,
  `flame3.py`, `flamelook3.py`. v2 files untouched (`ring.py`'s `inscription()` is imported by MONTAGE-3D-2: never change it).
  Only renders: the first-look stills `renders/accord_C3/tests/still_4960_*` (NOT festival grade). No v3 frames in
  `renders/accord_C/` yet (that folder holds v2's src 1912-2247 = the bible's AC1 fallback; do not collide: v3 is 4480+).
  No ACCORD render processes running. No log_ACCORD.md exists. The H5 critic has no ACCORD items (the council was
  not reviewed); the H5 CALLS that bite here: silhouettes + gloved hands only, the shawl is woven wool, the canonical
  Ring (`assets/ring/inscription_outer.png`), C23 = THE FIRE REMAINS (bar 70 = ours, 5520-5599 + handle to 5679).
* **Director calls (brief2):** naturalistic night render, NO ink filter; EDIT note: a 12-16 f dissolve at 4480 with MAP's
  inked ring holding over the lit stones; bar 70 unmistakably warm (the fire lives on, shared, leaves with the torches);
  the gilded hand readable in one beat; real volumetric torch flames.
* **MONTAGE-3D-2's melt (5360-5519) does not exist yet** (their Ring rebuild resumes 15:00Z; delivery `renders/ring_C/`).
  AC3 composite waits for it.
* **ACCORD-2's order of work:** the first-look fix list below (flames, hoods, torch head + fists, ash + stone + kerb,
  shawl), then test stills 4840 -> 5040 -> AC2 (P2) -> bar 70 (P3); then the crowd + rivers of torches; cloud jobs
  (JOB READY + check stills); the AC3 composite once `renders/ring_C/f_05360..` exist.


## >>> PAUSED 27 Sep ~11:05Z (director: usage pacing; RESUME in the 15:00Z window) <<<
**Where it stands.** The v3 engine is written, committed (7442e33) and renders (numba, CPU): `accord3.py` (driver:
still / range / sheet), `scene3.py` (timeline, cameras P1/P2/P3, the roster of 13 + her, poses, the hand rig, the Ring,
the fire timeline), `geom3.py` (flat stone, cold hearth with kerb/logs/charcoal, figures with kneel/lean/walk, gloved
fists, the detailed hand `sd_hand`, the real-size Ring, analytic capsule soft shadows), `shade3.py` (surfaces),
`flame3.py` (volumetric torch flames, the hearth fire that catches from every torch, sparks), `flamelook3.py` (flame
look-dev: `python shots/accord/flamelook3.py out.jpg`, side + top views). v2 files untouched.
**Output:** EDIT-v3's convention is `renders/accord_C/` in C numbering (4480-5599; bar 70 = 5520-5599);
`accord3.OUT` points there (v2's src frames 1912-2247 in the same folder do not collide). Test stills so far in
`renders/accord_C3/tests/` (delete that folder once superseded). A still: `python shots/accord/accord3.py still 4960
--scale 0.5` (~45 s half res); a full-res crop: `--window x0,y0,w,h --scale 1.0`.
**First-look verdict (4960, the AC4 orbit, half res): NOT yet festival grade.** Fix list, in order:
1. Torch flames read as faint grainy streaks / cotton buds: too thin (envelope and noise threshold in
   `flame3.torch_density`), leaning too hard in the wind (`accord3._WIND`), too few steps. Tune in `flamelook3.py`
   until a vigorous 30-40 cm flame with tongues, a yellow core and orange edges; then bloom and a soft smoke wisp.
2. Hoods read as smooth mushroom caps (v2's "game pieces" risk): add hood folds and peaks, cut the velvet edge-glow
   (`vel` in `shade3`), keep the cloth near-black with warm rims, drop the olive cast (lower `P_CROWD`).
3. The torch head's wrap ridges read as a corn cob; the gloved fist as a beige block (too bright, too boxy).
4. The ash bed reads as a cow-print (big black blots): soft grey ash, small 3-D charcoal, visible half-burnt logs.
   The flat stone reads as a loaf: darker, crisper edge, pitted granite. Kerb: irregular sizes and spacing.
5. Her shawl reads as a red plastic ring: a thinner layer with drape and folds, a darker wool red with texture.
**Then:** stills 4840 (the Ring set), 5040 (the gilded hand), 5130/5165/5210/5340 (P2), 5530/5560/5590 (bar 70);
the walkers and crowd (a `plain3`) for AC1's descent and bar 70's crane; the 4480 match; cloud jobs
(`cloud/jobs/accord3_*.json`, ship jpg); the AC3 composite once MONTAGE-3D-2's melt 5360-5519 exists.
Canonical inscription: `assets/ring/inscription_outer.png` (MONTAGE-3D-2), loaded by `accord3.inscription_mips()`.

## STATE AT HANDOFF (27 Sep ~10:15Z, ACCORD-v3 taking over; no brief/log files for this lane)
* **On disk (v2, do not touch):** `renders/accord_C/` src 1912-2247 (vows + Ring in the hearth; the bible's AC1
  fallback), `accord_A/`, `accord_B/` (retired by BIBLE_V3 section 8: A and B have no table), `accord/` (v1), cache
  `renders/accord_cache/` (carved-text maps 723 MB: not needed by v3). v2 engine files are unchanged and still render v2.
* **`ring.py` is imported by MONTAGE-3D-2** (`shots/montage3d/ring3d.py` calls `ring.inscription()` for the canonical
  inscription). Never change `inscription()`'s signature or output.
* **v3 work:** none yet at handoff. No v3 render processes exist.
* **H5 CALLS in this lane:** silhouettes + gloved hands only (the v2 emissaries' bare `M_SKIN` hand spheres, the
  head-wrap and the veil types are v2-only and are dropped in v3); no regional dress; the red scarf is a woven wool
  shawl; the Ring = MONTAGE-3D-2's canonical ring (ring.py's inscription); C23 is THE FIRE REMAINS and bar 70 is ours;
  the melt must be darker, crisp, no donut rim (MONTAGE owns the melt itself; AC3 is our composite).

## v3 PLAN (the shots, on C's locked bar map: bar n starts at (n-1)*80, a beat is 20 frames)
Three rendered plates plus one composite, all C numbering, one engine (`accord3.py` + `scene3.py`, `geom3.py`,
`shade3.py`, `flame3.py`; reuses nbcore, fire.py kernels, plain.py paths/walkers, particles, ring.py):

| plate | C frames | beats |
|---|---|---|
| **P1 THE COUNCIL + BRING OUT THE RING** (AC1 C19 + AC4 C20, one continuous orbit) | 4480-5119 | 4480 matches MAP's drawn ring (11 stones, centred ~960,402, ~110 px); a slow descent 10-15 deg off vertical with a slow orbit; rivers of torches converge on the ring of stones; ~4700 the council: emissaries in hoods and cloaks of many cuts and heights round a cold hearth with one flat stone; one small figure in a red wool shawl. 4800 she steps to the hearth, kneels; **4840 (61 b3) sets the Ring on the stone**; the orbit goes on; every hand holds a torch upright, none reaches (T11 4900-5030 over the orbit); **5040 (64 b1) the orbit finds the gilded, ember-scarred hand** on a torch. |
| **P2 THE BEARER** (AC2 C21) | 5120-5379 (5360-5379 = handle) | a cut to top-down and close on the stone: **5120 (65 b1) the torches come down together** from every side toward the cold hearth; **5160 (65 b3) her gloved hand, in from the side, goes back to the Ring and closes on it**; **5200 (66 b1) the fire everyone lit catches from every torch at once** and rises round her fist; it grows white-hot to 5359. Handle 5360-5379: her fingers forced open, the Ring drops out of her palm (for EDIT's cut into the melt). |
| **AC3 THE UNMAKING** (C22, composite) | 5360-5519 | MONTAGE-3D-2's Blender melt (`ring3d.py` RING_SHOT=melt, 5360-5519) set into the council fire: our fire's tongues, embers and heat haze round the frame, one grade, and the hearth's flare to full white by 5519 so it hands straight to P3. |
| **P3 THE FIRE REMAINS** (C23 bar 70) | 5520-5679 (5600-5679 = handle under MAP's bar-71 burn-through) | out of the white, top-down: the white settles to a warm, steady fire on the stone where the Ring was; **every spent torch dips into it, catches, and turns away**; the bearers walk outward carrying flame; she kneels by the fire, her hand to her chest; the camera cranes up so that by 5600-5679 the ring of stones sits where MAP's drawn ring will burn through, the flames leaving through the stones like the roads out. |

**DIRECTOR (10:25Z): CONFIRMED** naturalistic night render, no ink filter over 3-D (C's grammar: the world and the
book are drawn; fire, hands, the Ring and the gathering are lit and real; burn-throughs carry us between). **EDIT: at
4480 a 12-16 f dissolve in which MAP's inked ring holds a beat over the lit stones.** Bar 70 must be unmistakably warm:
the fire lives on, shared, and leaves with the torches. The gilded hand must read in one beat.

**Look.** Blue darkness, gold firelight (the red-team fixes: 10-15 deg off vertical, a slow orbit, real torch flames,
the whiteout made of light). No illumination/ink NPR pass over the council (the locked sheet does not ask for it; a
sketch filter over a 3-D render was the H5 critic's C4 complaint); instead a night-piece grade (warm gold where firelight
falls, umber-black shadows, blue only in the far dark). Ground inside the stones is trampled turf and earth, not a paved
dais (landscapes, not landmarks). 11 unhewn standing stones of varied height, no lintels. **Real torch flames**:
volumetric (ray-marched per torch), never splatted discs; walkers' torches at altitude as small flame streaks.
**People**: 13 emissaries + her; hoods, cowls, a capuchin point, a hood thrown back, a broad mantle, a tall staff-bearer,
a stooped elder; one near-black-to-umber-to-undyed wool range; she is small, hooded, the red woven wool shawl the only
red in the frame. Hands are gloved (dark leather) or in sleeves; the gilded hand is a leather-gloved hand whose back is
crusted and seamed with gold, with a few ember-glowing cracks (the grasp's crust, cooled).
**The Ring**: the canonical band (R_in 9.4 mm, 2.3 mm thick, 5.2 mm wide, ring.py's script), true size on the stone:
a small gold ring (~8-16 px in P1, ~20-40 px in P2); letters dark until the fire takes it.

**Order of work:** (1) the new centre (flat stone, cold hearth, kindling) + her + torches with real flames, stills of
4840 / 4960 / 5040 (AC4) and 5540 / 5580 (bar 70); (2) P3 bar 70; (3) P1's AC4 half, then AC1's descent/match;
(4) P2; (5) the AC3 composite once MONTAGE's melt frames exist. Tests local at half res; finals as cloud jobs
(`cloud/jobs/accord3_*.json`, `"ship": "jpg"`).

## COORDINATION (read by MAP-v3, MONTAGE-3D-2, EDIT-v3)
* **MAP (C18 end 4479 -> C19 4480, and bar 70 -> bar 71 5600):** our ring = 11 standing stones laid out like
  `road.py`'s drawn ring (rng 17: a = k*2pi/11 + N(0,0.06)), radius 7.5 m, with one flat stone at the centre. 4480 frames
  the ring centred at ~(960, 402), ~110 px across, to match 4479. For bar 71 we deliver the plate to 5679 with the camera
  craning up over the council; exact screen position/size of the ring at 5600 will be published here once rendered.
* **MONTAGE-3D-2 (AC3):** we composite your melt frames 5360-5519 (`renders/montage3d_v3/melt/` or wherever you publish;
  please note the folder in your NOTES). P2 hands off at 5360 with her fist opening in white-hot fire and the Ring
  falling out of her palm; P3 starts from full white at 5520, so your white flare should reach (near) full white by 5519.
  A flame-free or alpha pass of the band would let our fire sit in front of and behind it; plain RGB also works.
* **MONTAGE-MELT -> ACCORD-3 (AC3 plate format, PROPOSED 27 Sep 19:35Z; the melt is now its own lane, `montage3d/meltc.py`;
  reply here, one line, and I build to it):**
  - **Two plates, C numbering, 1920x804, 24 fps; the farm ships JPEG q95 4:4:4, so the alpha is a SEPARATE plate:**
    `renders/ring_C/f_05360..05519` = the complete melt (my own stone, fire, sparks and grade, and the white flare, so
    EDIT can cut it in as it is; C22 reads `ring_C` first) and `renders/ring_C_mask/f_05360..05519` = the metal's
    coverage (8-bit grey, 255 = the band or bead; anti-aliased, with the same motion blur and DOF as the RGB). Comp:
    your fire BEHIND the metal = `yours * (1 - mask)` added over my plate; in front = plain add. The glove is not in the
    mask (say if you want it).
  - **Camera (as built, 20:40Z):** a low macro across the hearth stone, 8 deg above it, 100 mm lens (hfov 20.4 deg,
    sensor 36 mm), f/20; the frame ~62 mm wide at the Ring easing in to ~52 mm. No glove in my plate: your P2 handle
    shows her fingers forced open; my Ring falls in at the top of frame (5362-5363), strikes 5363, hops, rattles,
    still by 5381 (EDIT can cut on the drop anywhere 5360-5362). The bead forms 22 deg right of the lens axis. The white heart of the fire is BEHIND the Ring (the upper
    half of the frame); the tongues rise vertically in frame; embers drift up. No horizon, no figures, no stones' edges.
  - **Beats:** the letters are awake from 5360 (the fire already had it in her fist); 5420 they flare once; 5433-5440
    the breath (stillness); 5440 they go out and the band breaks at the back and runs forward into one bead (5440-5454);
    the bead glows and trembles; the hearth flares from ~5476 to (near) full white at 5519 for your P3.
* **EDIT:** C numbering. P1 4480-5119, P2 5120-5359 (+ handle to 5379), AC3 5360-5519, P3 5520-5599 (+ handle to 5679),
  all in `renders/accord_C3/`. T11 4900-5030 and T12 5130-5260 sit over P1/P2 with nothing busy under them.

---

# ACCORD v2 (below: the v2 notes, unchanged)

# ACCORD v2 — src frames 1912–2247 (v2 timeline = src + 160) → `renders/accord_A|B|C/f_%05d.png`

Three variants from one renderer (`--variant A|B|C`); v1 (`renders/accord/`, cloud) is untouched.
* **A — Allegory.** The four oaths carved and written in gold at 2000/2040/2080/2120, each rolled upright to the top of frame (as v1), crisp.
* **B — Legend (wordless).** No words. The oath band becomes a carved braid with a fire rising at each emissary's station, lit in the same ember language as the frieze: each station fire catches as its emissary reaches over it with a torch (≈1990–1998), then after the merge the fire runs along the braid from every station until the ring is one (2004–2060) and settles to embers. The stepped roll is replaced by one slow, continuous, eased turn; the council is framed centred (h ≈ 10.5 m) so the whole mandala turns in frame.
* **C — Tolkien.** As A, plus the Ring (see below).

## What changed from v1 (all variants)
* **Emissaries** (`geom.sd_fig`, `scene.py`). Rebuilt as dark, dignified hooded figures seen from above: cloth with crease-profile folds fanning out from the neck, a folded mantle that overhangs every figure (longer on the torch side), hood shells with a real cavity, a brim that overhangs the face and a peak behind; six silhouettes (cowl with liripipe, deep hood, head-wrap with a tail, bare head with the hood thrown back, a slighter veiled figure, a pointed capuchin) at varied heights and widths. All cloth sits in one near-black charcoal-to-umber range; they differ by weave and sheen (`F_WEAVE`, `F_SHEENK`) and cut, not colour. Wool/velvet response (faces turned to us sink dark, edges hold the firelight), a faint warm back-rim from the crowd's torches, a soft fill from the upper flames so the hoods read. After the merge they hold their charred-black spent torches upright before them, hands withdrawn into the sleeves. Near-field depth of field 2000–2140 (`accord.near_dof`: table in focus, heavier toward the frame edges) keeps the edge figures soft. Through the flare their hearth light is compressed (`P_FIGK`), so they stay dark silhouettes against the blaze instead of going pastel.
* **The "together" ring is gone.** In its place the floor band carries an original carved frieze: 48 small three-tongued fires on a running line, the grander one in each emissary's shadow. In the same clockwise sweep (2160–2220) fire runs round it: short tapered flame tongues ride the head of the sweep, blown forward, and die down behind it (`particles.frieze_tongues`), leaving dim, mottled orange embers in the grooves (`shade.frieze_fire`) — firelight in carved stone, never a flat gold fill, far below the hearth. (Rejected on the way: a braid with medallions, which read as a ring of eyes; and a gold fill, which read as a laurel emblem.)
* **No T12 dimming** (no screen band is calmed any more).
* **Temporal fixes** (critics' pass): the ignition glare tapers to zero by 2016 (no pop at 2011); the camera dissolves through the mist layers over 5–14 frames (no strobes at 1946/1955); the light the fire throws on the room breathes at ≤ 1 rad/frame and ≤ ~3.5 % (no 10 Hz dips; the flames themselves still dance). Checked by frame-mean luma: no single-frame dip > 3 % in 1926–1975, 2003–2019, 2176–2231.
* **A/C:** the pull-back to the mandala starts 14 frames later than v1 (keys 2150/2163/2176) so Oath IV holds full size.
* **Crisp text.** Cause of the soft local renders: `accord.py range` inherited argparse's `--scale 0.5` (meant for `still`), so every local "delivery" frame was rendered at half resolution and bilinearly upscaled. A local full-res render of 2010 matches the cloud frame (edge-gradient p99 301 vs 297). `range` now defaults to full resolution, and any reduced scale is written to `renders/accord_<V>/draft/`, never the delivery folder.

## The Ring (C)
`ring.py`. A plain heavy gold band (OD 0.37 m — mythic scale, ~90 px at the oath framing) lying askew (27°) on a bed of grey ash in the hearth. It carries an inscription on both faces in an **invented** broad-nib script generated here from original strokes (arches with curling legs, flame-like stems with coiled heads, U-bowls with rising tails, coils, marks above the line) — no real alphabet, no real text. It glints in the torchlight in the cold hearth as the flames stream in; the letters kindle as the merged fire takes it (2001–2016) and then breathe with a slow pulse (58 frames). The fire is hollowed around it (`FP_HOLLOW`) so it lies visible among the flames; the flare closes over it and the white swallows it.

## Re-render
```
source ~/.venvs/longdawn/env.sh; export NUMBA_NUM_THREADS=1
python shots/accord/accord.py range 1912 2247 --variant A --worker 0/2 &   # + --worker 1/2
python shots/accord/accord.py finish --variant A                          # preview.mp4 + contact.png
python shots/accord/accord.py still 2100 --variant B --scale 0.5          # test still -> renders/accord_B/tests/
python shots/accord/accord.py still 2060 --variant C --window 760,300,400,400   # full-res crop
```
Caches live in `renders/accord_cache/` (carved maps per kind `text`/`orn` as memory-mapped .npy, the plain, the Ring's inscription); delete to rebuild (`python shots/accord/textmaps.py text orn`). Look-dev helpers: `figlook.py` (emissaries from above), `keysheet.py` (key-frame sheet), `textmaps.py preview <kind> out.png`, `ring.py preview out.png`.
Speed: 7–21 s/frame single-threaded uncontended (oath framing is the slowest); the full pass (3 × 336 frames, 2 workers, heavy contention from other departments) took 124 min, i.e. ~15 s/frame/worker on average. Performance work: shadow rays use a cheap silhouette proxy of the figures, the hood math only runs inside a head bounding sphere, and extra AA samples go only to id edges and high-contrast pixels; the fire is ray-marched into its own buffer and softened by a sub-pixel blur (removes v1's cross-hatch jitter).

## Deviations / cheats
* Figure lighting cheats: the crowd back-rim, the upper-flame fill, torches lighting their own bearers at half strength, and the flare compression.
* The oath letters (A, C) stay gold for legibility; every other carving that lights (frieze, B's braid) is ember-light.
* The v1 deviations still hold (merge on the 2000 hit, two-line oaths, lighting cheats for the radial shadows, exposure ride).

Review sheet (v1 | A | B | C at matched frames): `~/mishamisha/_local_logs/review/accord_v2.jpg` (`_local_logs/accord_v2_sheet.py`).
