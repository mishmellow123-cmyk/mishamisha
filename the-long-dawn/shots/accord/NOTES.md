# ACCORD v3 (ACCORD-v3, 27 Sep): cut C only, C frames 4480-5679 -> `renders/accord_C3/f_%05d.png`

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
