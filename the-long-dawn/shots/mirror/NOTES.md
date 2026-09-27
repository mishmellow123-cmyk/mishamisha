# MIRROR: C10 THE MIRROR (C bars 27-29, frames 2080-2319 -> renders/mirror_C)

## STATE AT HANDOFF (MIRROR, 27 Sep ~20:55Z; read this first)
* CODE: pushed at 5ba520d (claude/long-dawn-v2): `mirror.py` (renderer, look pass 2), `plates.py`, `plates/fire.mp4`,
  `plates/dawn.jpg`, `cloud/jobs/mirror_C.json` (1 lane `--frames 2080-2319 --threads 8 --ss 1.5`, jpg ->
  renders/mirror_C).
* JOB READY mirror_C SENT to main at ~20:53Z (240 f, ~12 min on 3 cpu nodes). If approved, launch:
  `cd ~/mishamisha && python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/mirror_C.json --nodes 3`
  (and `--missing` to top up). EDIT-2 confirmed that the EDL already reads renders/mirror_C (C numbering, jpg or
  png), so nothing is needed there.
* Farm look-dev IN FLIGHT at 20:52Z: `--test 3 --frames 2160,2222,2300` -> renders/_farmtest/mirror_C/ (log: scratchpad
  mirror/farm_t2.log). Review them at full res (a 1:1 crop of the lip and the ripples) before the final lands.
* Local 0.3x check (look pass 2): review/mirror_C_check/local_0.3x_2160_2206_2222_2300.jpg. VERDICT: good with
  nitpicks.
  - It reads as rippled water in a stone basin: the lip on the right, the dark ground beyond. No porthole, no Eye.
  - 2160: the burning lands in dark water with stars.
  - 2206: the drop's rings bloom the dawn out from the sun.
  - 2222: the golden dawn and the drawn ranges.
  - 2300: the fire back (red walls, towers, sparks).
* FARM STILLS REVIEWED (21:10Z; renders/_farmtest/mirror_C/ 2160, 2222, 2300, at full res): PASS.
  - A basin, not a porthole. The fire reads, the golden dawn reads, the fire is back. No Eye.
  - The 1:1 crop showed star reflections smeared into scratch-like streaks over the bright visions. FIXED in the
    kernel: the sky reflection is dimmed where the vision is bright, and the star gain went 45 -> 34.
  - The fix was verified locally (2088 keeps its stars, 2222 is clean) and is pushed. The final picks it up from the
    branch tip.
  - Farm cost: ~24 s/frame for isolated test frames. That includes re-simulating from the start, so a final
    slice of 80 frames on one node should run ~10-15 min.
* NITPICKS LEFT (a cheap re-render):
  - The dawn's sun is a soft glow, not the drawn disc (lower the added glow in Mirror.__init__).
  - The drawn ranges are faint under the gold (the `land` factor).
  - The lip could take more warm glow (rim_glow 3.0*E).
* Music: the sync note is in music/NOTES_v3.md ("PICTURE SYNC: C10"). The drop strikes f2200, the after-drip ~2210.6,
  the dawn runs 2200-2248, the fire is back 2248-2288. COMPOSER-C's score_v3_C.mirror() already matches.
* NEXT STEPS, in order:
  1. The farm stills: check at full res, fix anything glaring, and push.
  2. On approval, launch the final. When it lands, check the landed frames (contact sheet 2080-2319 every 16) and
     play them through (ffmpeg preview).
  3. Nitpick pass (above), then re-render.
  4. When EMBERS-C's H5 race lands in renders/embers_C3 (1560-1679), or RUN-C's final f2480 changes:
     - run `python3 shots/mirror/plates.py`;
     - check that the flanks (FLANK_L/FLANK_R) still exclude the Ring;
     - commit the plates and re-render.

## DESIGN (as built)
* Beats (sync = music/v3/barmap_C.json; COMPOSER-C's score_v3_C.mirror() already uses these):
  - 2080 (27 b1): dark water, stars in it, the hewn stone lip lit only by a faint cool key and the sky.
  - 2096-2158: a breath crosses the water from the left (the book: Galadriel breathes on the water); its
    capillary ripples carry in the lands burning (the fire develops just behind the ripple front).
  - 2158-2199: the lands burning, rippling; the lip lit red by the water's own glow.
  - 2198-2199: the drop's glint falls (a short streak toward the strike point) and its reflection rises to meet it.
  - **2200 (28 b3): the drop strikes** (the sound's transient belongs exactly here). Crown sparkles, the jet's
    bead (2200-2207), a secondary drop at ~2210.6 (a second, smaller ring train).
  - 2200-2224: a golden dawn blooms out from the strike (RUN-C's drawn world at sunrise, the sun under the drop).
  - 2224-2248: the dawn holds, one breath.
  - 2248-2288: the returning ripples carry the fire back over it, inward from the walls to the drop point.
  - 2288-2319: the lands burning; cut to C11 THE GRASP (the claw descends: top-down to top-down).
* No Eye anywhere. The basin is framed off-centre (the water fills the frame, the lip on the right with the dark
  ground beyond, the left lip half out of frame) and the drop falls off-centre, so no ring/iris/porthole read.
* Optics (numba, no Blender): the water is a height field from a linear gravity-capillary wave solver
  (w^2 = (gk + s/r k^3) tanh(kh), exact per-mode propagation, 4 substeps per frame, viscous damping, the hewn
  wall as a reflecting mask; 768^2 over 0.86 m at full res). Per pixel: exact Fresnel (n 1.333); the reflection
  samples the stone lip (short march) or the star sky (gnomonic texture: 2600 stars with a real magnitude spread
  and colours, a faint galaxy band); the refraction samples the vision on a plane 0.55 m below the surface
  (so the vision has parallax under the camera push and the ripples lens it). The stone is a ray-marched height
  field (hewn radius, rounded lip, chipped bullnose, cracks, pits, lichen, moss, a water-stained wet band),
  lit by the water's own glow per angle (fire red, then dawn gold), a faint cool key, and the sky.
* Plates (`plates.py`): FIRE = EMBERS-C C7 1560-1679, the two flanks of each frame joined with a soft seam so the
  Ring is never in the water; played at half speed; pre-H5 source (embers_C3_half) until the H5 race lands.
  DAWN = RUN-C C24 illumination f2480 (the drawn world, the sun just up): the Mirror's glimpse is the very image
  C24 pays off. Graded gold (land darkened, sun glow added).
* Fallback (bible): comp-only Mirror (the fire and a dawn plate warped through procedural ripple normals, no
  basin). Not needed unless the gate fails.
