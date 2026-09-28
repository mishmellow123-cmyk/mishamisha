# COUNCIL-C: C's council rebuilt in Cycles (C19-C23, C 4464-5679)

## >>> STATE: PAUSED by the director (28 Sep ~03:40Z; C's story is being reworked). NO new rounds/builds without a go. <<<
* **Look-dev rounds (farm GPU, 1 lane per frame pair; stamps = Cycles render s/frame, 2 processes sharing one H100):**
  - round 4 (e4c1f4e, CURRENT BEST): `renders/_farmtest/councilC_look/f_{05160,05161,05566,05567}.jpg`
    (5160/5161 = P2 top-down, her glove closing on the Ring; 5566/5567 = bar 70, spent torches dipping into the fire).
  - round 3: `renders/_farmtest/councilC_look_v3/`; round 2: `..._v2/` (round 1 deleted: pale robes, cotton-swab
    torches, cake slab, giraffe logs, 17 min/frame).
* **Timing (round 4):** P2 without the hearth fire 87-88 s/frame; P3 with the fire 381 s (5566, shared GPU) and
  272 s (5567, alone on the GPU for most of it): the fire frames' real cost is ~4.5 min until the domain fix. Round 3 was 79-81 / 365 / 270 (5567 alone). Kernel JIT ~6-7 min on a NEW node.
* **Projection (880 Cycles frames: 4640-4799 AC1 low part, 4800-5119 AC4, 5120-5359 AC2, 5520-5679 bar 70; AC1's
  far wides 4464-4639 = ACCORD-CROWD v1 kept, dissolve at 4640-4663; C22 = the melt's ring_C):** as is, 560 no-fire
  frames x ~75 s + 320 fire frames x ~4.5-6 min = ~36-44 GPU-h (+ ~2 h JIT). With the known P3 fix below, ~20-24 GPU-h.
* **P3 cost, probable cause:** flame DOMAINS, not shading: every torch flame's box is sized by its max lean over the
  job (the walk-out's velocity lean -> up to ~2 m boxes; 13 council + 15 crowd + 22 hearth domains overlap the centre),
  and Cycles marches every domain a ray crosses (empty space included). Fix: domains offset along the lean and capped
  (lean clamp ~0.6 for sizing), fewer hearth domains (merge the heart + crown), step rate 0.4 -> 0.8. Glossy-invisible
  flames (round 4) did NOT help.
* **Frank look assessment (round 4):** a real step up from v1 (dark wool hooded silhouettes, real flame tongues at
  oblique angles, stone/char/litter at 1 mm/px), but NOT festival grade yet: (1) the slab still reads as a tray from
  above (dark top, lit bevel) and as a pale platter with a ring of orange crumbs in P3 (the coal pile must heap and
  cover the middle); (2) top-down torch flames read as orange capsules; (3) flames are smooth orange "feathers"
  (need inner structure/brighter cores); (4) staves near the fire read pale; (5) the Ring at 5160 is a ~16 px glint at
  her fingertips (light-linked glint light works but is faint); (6) hoods still a little helmet-like from behind.
* **C20 staging bug to fix (ACCORD data):** at the Ring-set (4822-4866) her left arm's IK (0.28+0.27 m) clamps ~0.3 m
  short of the Ring (she kneels at r 1.0, the Ring lies at r 0.22): the hand never reaches it. Fix: kneel closer during
  the reach (HER_KNEEL_R 1.0 -> ~0.80) or more lean; her left glove rig is already built (`her2`).
* **Uncommitted -> committed with this note:** her left glove for the P1 reach (`her2`), COUNCIL_SCALE (half-res
  motion checks; set the job's "shape" to [402, 960]).
* **Resume:** read this block, then the director's new council beats; re-run `councilC_look.json --test 4 --gpu`
  only after a go.

## (earlier) STATE (28 Sep ~01:40Z, COUNCIL-C)
* **Scope (director):** rebuild the council's LOOK in Blender Cycles on the farm GPUs (MONTAGE-3D's pipeline:
  `shots/montage3d/render.py` + `bl_main.py`), keeping ACCORD's staging, beats and timing exactly. SAFE MODE: no local
  Blender/renders at all; every test is a farm `--test` (GPU picked automatically: the commands name MT3D_BLENDER).
* **Code (all in `shots/montage3d/`):**
  - `councilc.py`: the shot. `prep` (venv, on the node) runs ACCORD-3's `scene3` (camera, the 13 + her, torches,
    her hand, the Ring, the fire) and ACCORD-CROWD's `crowd3` (only people in or near the view, LOD by distance) for
    the frames rendered, writes `cache/councilc/state_<tag>.json`, and BAKES every mesh (`geo_<tag>.bin/.json`) and
    the figures' per-frame vertices (`figs_<tag>_<f>.bin`): Blender's Python needs no numpy. `build`/`per_frame`
    (Blender) = the look. `post` = ACCORD's per-frame exposure (`scene3.exposure`) and white-out (`white_level`).
  - `council_geo.py` (numpy): cloaked figures from geom3 F rows (hanging wool folds, turned hem, deep hood with a
    rolled rim and a VOID inside, mantle or her red shawl, torch-arm sleeve, crowd fist), the slab (irregular outline,
    no rim), kerb rocks, split logs (wedge section, burnt-down inner end), kindling, charcoal chunks (angular),
    standing stones, heather tufts (trodden variants), the torch (stave + wound pitch head).
  - `council_local.py`: render.py with its own lock + a missing-frame check (exits 1).
  - Look: wool (sheen rim, twill, damp hem), void hoods, leather gloves = `glove.py` rigs (emissaries' hammer grip
    round the stave, the gilded one gold-crusted over the knuckles with live ember scars; her right hand in P2 =
    find_b's leather), torches with volumetric emission flames (tapered tongues, transparent tips, blackbody core ->
    tips) + a flickering point light each; spent torches cold black with ash; slab = gritstone, soot toward the middle,
    lichen on the weathered edge, never emissive; logs = warped-voronoi alligator char (no straight cracks) that glows
    only when burning (keyed heat); ground = peaty earth, grit, trodden litter, a powdery ash bed; heather instanced.
  - P2 fire rows are re-derived with FIXED slots (`_p2_rows`): ACCORD's p2_flames list grows as torches catch, which
    would make flames swap identity and pop.
* **Look-dev:** `cloud/jobs/councilC_look.json` (5160 + 5566, one lane each) ->
  `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/councilC_look.json --test 2` (from ~/mishamisha) ->
  `renders/_farmtest/councilC_look/accord_C/`.
* **Output plan:** finals to `renders/accord_C` (EDIT's C19-C23 take), replacing ACCORD's v1-for-timing frames, only
  after the director approves.
* **Knobs (env):** COUNCIL_EXPO (post exposure x), COUNCIL_TORCH_W (torch light W), COUNCIL_FIRE_W, COUNCIL_FLAME_E
  (flame emission), COUNCIL_NOHEATHER, COUNCIL_NOCROWD, COUNCIL_REPREP.
* **Open / next:** the look-dev verdict; then per-shot jobs (P1 AC1 descent + AC4 incl. the gilded-glove find 5040,
  P2 AC2, P3 bar 70); far crowds (AC1 4464-4719, the walk-out wides) may composite ACCORD-CROWD's crowd; the
  walk-out must not read as sliding/robotic (check 24+ consecutive frames before any JOB READY).
