# The Long Dawn: rendered studies, 27 September 2026

Open `index.html` in a browser. The slider reveals the alternative on the left and baseline on the right; the buttons show either full frame. These are actual CPU-rendered frames. No production branch, original laptop, farm job or cloud node was changed.

## A15 mountain cutoff: verified technical fix

The baseline has a straight vertical cutoff in the distant blue ridge. Its marcher receives a 700 m height ceiling, while this terrain's highest ridge endpoint is 2,332.665 m. Actual rays beyond the edge therefore report sky instead of the far wall. A controlled 3,000 m diagnostic render removed the cutoff.

The final fix derives a conservative 2,784.5 m ceiling from the complete A terrain, including upward displacement and smooth-union envelopes. Independent source review checked the derivation. The viewer's mountain comparison shows final renders at 4280, 4360 and 4399 with this helper; camera, terrain geometry, lighting and the accepted cairn are unchanged.

Four regressions exercise the actual A15 marcher on a 25-column strip retaining all source rows, check the four A-world entrypoints, preserve the shared pipeline's default for other worlds, and verify that raised crest parameters raise the ceiling. Restoring 700 m makes the far-wall regression fail. In that strip, the derived bound exactly matches 3,000 m and preserves all previously finite hits. This does not prove that every other terrain recipe has a sufficient bound.

A2/A11/A14/A15 share this terrain and receive the correction. Other fixed-height callers remain unverified; none was changed merely because it uses 700 m. The fix is on the technical branch, independently adoptable from the council proposal.

## Council material candidate: retain for comparison

Only `shots/accord/shade3.py` changes. The candidate fragments the char checks, replaces bright connected ember cracks with sparse patches, and removes the uniform emissive floor from torch fuel. It preserves the camera, geometry, crowd, light array, flame volumes, timing and finish.

Frames 4960, 5160, 5566 and 5596 compare source at `07d5dca` with the material patch. 5160 and 5566 are native 1920×804; the others are 960×402. Full crowd was required. The baseline runtime was frozen separately before rendering.

Observed result: the solid glowing torch-head appearance is reduced, particularly at 5566; log emission is less continuous. At 5160 some lowered torches now read as smouldering. The rounded figures, slab-like centre, pale fire and overall miniature appearance remain. **This is a narrow candidate, not an accepted council redesign.**

`council-motion-5160-5175.mp4` contains sixteen consecutive frames at 24 fps, baseline left and candidate right (0.667 seconds, no sound). It is a short material comparison; it does not settle the sequence's timing or full motion quality.

A proposed flame-anchor shift was rejected before implementation. A finite-grid probe of actual flame density, fuel geometry and camera projection at 5160 found that moving lowered flames toward the upper fuel surface would crop more of two right-edge flames. This measurement omits full ray integration, smoke, grading and other occluders. It argues against that particular shift; the weak burning appearance remains unresolved. The diagnostic source/results are supplied under `anchor-probe/`.

## A15 cairn comparisons: all three alternatives rejected

The harness `shots/run/studies/cairn_shapes.py` replaces only `watchers_a.hearth_scene` within its own process and restores it on exit. Accepted renderer files remain untouched. Camera, light, finish, fire recipe and material 5 are held constant. The latter has a fixed shader; changing the object's nominal RGB alone would not recolour these stones.

At frames 4280, 4360 and 4399 (960×402):

- **Packing-only** keeps the existing 21 capsule stones but changes placement and removes smooth blending. It exposes floating-looking gaps and retains the smooth lobes.
- **Fractured-fieldstone** uses unequal flat box fragments. It reads as stepped masonry courses; this repeats a previously rejected visual direction.
- **Uneven-fieldstone** removes the courses and varies heights, proportions and tilt. One refinement shortened the plank-like axes and rounded their edges. It still reads as rectangular chunks, not convincing weathered stone.

These failures do not establish that the renderer cannot make a better cairn. They establish that these geometry recipes do not solve it. Surface response and contact shading remain plausible contributors; no alteration of either was tested here. Do not replace the accepted cairn with these studies by default.

All candidates match the baseline's conservative primitive top bound for the seed and scale (seed 7070, scale 1: 0.6252963897877976 m). An earlier 0.64 m normalization was corrected, and the supplied final comparisons were rendered again. Bounds checks do not certify physical contact or visual quality. The shader, camera, lighting, terrain and fire are unchanged within each cairn pair. These cairn snapshots retain the old terrain bound in both images; reproducing the harness after the technical fix will correct the distant background in both variants.

To reproduce on an isolated checkout, from `the-long-dawn`, in the renderer's Python environment:

```sh
NUMBA_NUM_THREADS=2 python -B shots/run/studies/cairn_shapes.py --variant baseline --frames 4280,4360,4399 --out /absolute/study/baseline
NUMBA_NUM_THREADS=2 python -B shots/run/studies/cairn_shapes.py --variant uneven-fieldstone --frames 4280,4360,4399 --out /absolute/study/uneven
CROWD=1 CROWD_REQUIRED=1 NUMBA_NUM_THREADS=2 python -B shots/accord/accord3.py range --frames 4960,5160,5566,5596 --scale 0.5 --outdir /absolute/study/council
```

Use distinct output directories and preserve the source version for each comparison. Never target the farm's delivery directories with these studies. The outputs folder also contains the separate technical handback with integration details and regression tests.
