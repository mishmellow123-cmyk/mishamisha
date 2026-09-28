# COUNCIL-C: C's council rebuilt in Cycles (C19-C23, C 4464-5679)

## >>> STATE (28 Sep ~01:40Z, COUNCIL-C) <<<
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
