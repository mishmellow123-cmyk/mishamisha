# ACCORD — plan (global frames 1912–2247, cut 1920–2239)

## Story beat
Every people answers. From far above a dark plain, rivers of torchlight flow
in along worn paths toward a ring of standing stones. We fall toward the
stones and the great round table inside them. Twelve hooded emissaries stand
around it. They reach their torches toward the cold hearth, the flames lean in
and run together, and at **2000** they become one GOLDEN fire. Its light
throws their shadows outward like the hour lines of a sundial. The oaths burn
into the stone one by one (2000/2040/2080/2120), each turning to the top of
frame as it lights. Then "together" in 27 tongues ignites around the rim
(2160–2220), and the hearth flares white (2225→), which the edit match-cuts to
the sun.

## Key decisions
* **Merge on the hit.** The torches' flames lean in over 1986–1999 and fuse at
  **2000**, the same frame as Oath I (NO SINGLE HAND SHALL HOLD IT: one fire
  held by every hand). The fire keeps blooming through 2000–2060 with embers
  streaming up at camera, so the bible's 2010–2060 golden fire is still there;
  it just starts on the downbeat. (Deviation noted in NOTES.md.)
* **Legible oaths (director's note).** One ring holding all four oaths gives
  ~12 px caps in a hearth-centred frame, which is unreadable. Instead each oath
  is carved as **two concentric lines** in one inscription band:
  `NO SINGLE HAND / SHALL HOLD IT`, `NO FASTER THAN / WE CAN SEE`,
  `NO FORGE / IN THE DARK`, `WHAT IT GIVES, / IT GIVES TO ALL`. The four
  blocks sit around the band ~90° apart. Each line spans ~75–95° of its ring.
  During the oaths the camera is low, top-down, and rolls about the hearth so
  each block lands upright across the top of frame. Target cap height is
  **~38–45 px** (Cinzel Bold), with the hearth just below centre and the table
  overfilling the frame.
* **"Together" ring** = two rows of words carved on the table rim just outside
  the oath band (27 words, bible §8 order, Noto fonts shaped with RAQM). It is
  visible at ~22–26 px caps when the camera eases up a little for the sweep.
  Every script is checked in a zoomed test image.
* **Mandala.** The table is low (top z=0.8) and the hearth light sits high
  (z≈2.3), so each figure throws a long wedge of shadow outward from about 3.6 m
  to 12 m. Fire light falls off softer than inverse-square (a lighting cheat)
  so the spokes read against a very dim moonlit ground. The full circle shows
  at the ignition during the descent. In the low frames the spokes fan out at
  the wide left and right of the 2.39 frame. At the flare the lit wedges burn
  white-gold between dark spokes, a sunburst that sets up the match-cut.

## World (metres, z up, origin = table centre on ground)
| thing | size |
|---|---|
| hearth bowl | r 0.50, lip ring, coals; fire light centre z≈2.3 |
| table | R 2.45, top z 0.80, rounded rim, weathered granite |
| oath band | cap 0.15 m, inner baseline r≈1.25, outer baseline r≈1.475, carved borders |
| together band | 2 rows, cap≈0.14, baselines ≈1.87 / 2.10, rim 2.45 |
| emissaries | 12, r≈2.8, 1.62–1.85 m tall, hooded cloaks, arm + torch |
| dais | flagstone floor r≤5.2 |
| standing stones | ~18 at r≈10, 2.6–4.2 m tall |
| crowd | arrived walkers r 12–20 m, torches |
| paths | ~10 main paths plus tributaries from ±450 m, worn earth |

## Timeline (global frames)
| frames | what |
|---|---|
| 1912–1990 | Top-down descent from ~650 m to ~25 m on an eased log-height curve with a slow spiral roll. Torch rivers flow in on time-lapse (fast at altitude, slowing as we descend). Mist layers at ~70/150/280 m slide past for parallax. 1922–1998: lower band y 560–700 kept dim (soft grad on the torch splats). |
| 1978–1992 | Emissaries raise their torches, staggered around the circle |
| 1986–1999 | Flames lean in and stream as ribbons to the hearth; the last holdout joins at ~1997 |
| **2000** | IGNITION: one gold fire, burst of embers, the shadow mandala snaps on. **Oath I** starts writing (line 1 2000–2011, line 2 2010–2022) |
| 2000–2010 | Descent finishes into the oath framing (low, hearth just below centre) |
| 2004–2030 | Arms withdraw, spent torches lowered |
| 2022–2040 | Eased roll so Oath II lands at the top; **2040** Oath II writes |
| 2062–2080 | Roll; **2080** Oath III |
| 2102–2120 | Roll; **2120** Oath IV |
| 2140 | Oath band border rings close with light |
| 2136–2164 | Camera eases up to the together framing (hearth centred), slow roll |
| **2160–2220** | "together" ring ignites in a clockwise sweep from the top, with a hot leading edge |
| 2222–2247 | Hearth flares to white (×60) and the camera pushes in; by 2240 the white fills the centre |

## Rendering approach (numba, CPU, 2 threads)
* **Camera**: perspective, straight down (roll about the view axis, plus a small
  shift so the hearth can sit off-centre), FOV_h 50°.
* **Primary visibility** per pixel: analytic ground plane, table
  (cylinder/disk with rim bevel and hearth bowl), and SDF sphere-tracing
  inside bounding volumes for figures (cloak cone with folds, shoulders, hooded
  head, sleeve arm, torch) and standing stones (rounded tapered boxes).
  Adaptive AA: 1 spp everywhere, then 4 more jittered spp on
  geometry/ID edges.
* **Carved text**: Cinzel (wght 700) lines rendered to strips with PIL and
  warped onto their arcs in a Cartesian table-top texture (≈1 mm/texel), then a
  distance transform gives a **V-cut chisel depth** (depth ∝ distance inside
  the stroke, capped). The normal comes from the depth gradient, so the
  hearth's low raking light lights the outer walls and shades the inner walls.
  Mip pyramid for filtering.
* **Gold "ithildin" fill**: per-texel fill time = arc position +
  (1 − normalised depth) × k. Light runs along the letters and floods the
  groove centres first, with a white-hot leading edge that cools to gold, plus
  a soft spill onto the surrounding stone (blurred mip).
* **Lighting**: hearth = spherical light with flickering position/intensity
  and soft shadows by SDF cone-marching against figures and stones plus an
  analytic test for the table. Torches and ribbons are unshadowed point
  lights. Walker torchlight is splatted into world-space irradiance grids.
  Moon is a dim cool directional light plus sky ambient and AO-ish contact
  darkening.
* **Fire**: hearth = ray-marched emissive volume (advected fbm with swirl,
  gold temperature ramp from `accord_amber` through `accord_gold` and
  `accord_pale` to white, cores ~15–30). Torch flames and ribbons are
  flickering additive splats (orange `fire_*` → gold as they merge).
* **Embers / sparks**: closed-form particle trajectories (deterministic per
  frame, order-independent), rising in a vortex toward camera, drawn as
  motion-blurred streaks with DOF-sized discs.
* **Walkers**: ~1500 analytic path followers (torch dot + trail + hooded
  silhouette sprite + ground light pool).
* **Post**: camera motion blur by reprojecting the depth buffer (rolls and
  descent), then embers, then `look.finish()`.

## Process
1. Text maps + multi-script zoom test (check RTL joining and no tofu).
2. Table/text/light still at the oath framing, checked for legibility at full res.
3. Figures, stones, shadows; look at the mandala.
4. Fire, embers, merge.
5. Plain, paths, walkers, mist; the descent.
6. Full low-res pass → contact sheets → fixes → full-res render → preview/contact/NOTES.
