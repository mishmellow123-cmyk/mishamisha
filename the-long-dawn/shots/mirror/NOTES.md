# MIRROR: C10 THE MIRROR (C bars 27-29, frames 2080-2319 -> renders/mirror_C)

## STATE AT HANDOFF (MIRROR, 27 Sep ~21:15Z; read this first)
* COMMITTED + PUSHED (e572d41 on claude/long-dawn-v2): `mirror.py` (renderer), `plates.py` (plate baker),
  `plates/fire.mp4` (4.4 MB, 120 f), `plates/dawn.jpg`, `cloud/jobs/mirror_C.json` (1 lane,
  `--frames 2080-2319 --threads 8 --ss 1.5`, ship jpg -> renders/mirror_C).
* Farm look-dev: `farm.py cloud/jobs/mirror_C.json --test 4 --frames 2088,2160,2222,2300` -> renders/_farmtest/mirror_C/
  (log: scratchpad mirror/farm_t1.log). The local 0.3-scale check of the same frames is in the scratchpad (mirror/t2).
* The RESULT and the JOB READY status are in the block right below (updated last thing before the window closed).
* NEXT STEPS, in order:
  1. Look at the farm stills (reduced for composition, 1:1 crop for detail). Gate: stars read in dark water at
     2088; the burning lands read (not "hammered glass") at 2160; the golden dawn with the sun under the drop at
     2222; the fire back at 2300; the stone reads as stone; nothing reads as an eye, a porthole or a screen.
  2. If it passes (or passes with nitpicks), send JOB READY mirror_C to main (240 f; farm est. ~10 min on 3 nodes)
     and launch the final when approved: `python3 the-long-dawn/cloud/farm.py the-long-dawn/cloud/jobs/mirror_C.json --nodes 3`.
  3. Tell EDIT-2 (aed1b4c6f62d96a63) the folder: renders/mirror_C, C numbering 2080-2319 (C10, bars 27-29).
  4. When EMBERS-C's H5 race (renders/embers_C3 1560-1679) or RUN-C's final f2480 changes: `python3
     shots/mirror/plates.py`, check the flank crop still excludes the Ring (FLANK_L/FLANK_R), commit the plates,
     re-render with `--missing` after deleting the old frames.

## JOB STATUS
* (see the bottom of this file; updated at each step)

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
