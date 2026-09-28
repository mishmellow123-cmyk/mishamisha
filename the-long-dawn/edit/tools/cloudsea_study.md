# A cloud-sea relief study

This is a geometry proposal for director comparison. No production shot enables it. The shared A world reads an optional fourth world parameter, `P[3]`, as a strength from 0 to 1; an absent fourth parameter or zero preserves the prior cloud geometry. The base `world.h_cloud` is unchanged, including for B's separate `bworld` renderer.

The candidate adds rounded lobes at 360, 125 and 45 m cell scales, with maximum contributions of 100, 55 and 25 m. Cell centres are jittered; the smaller fields are rotated relative to the larger field. The added field is fixed in world coordinates. Each scale fades as its pixel footprint grows. Existing cloud advection, holes, cameras, lighting, fog and finishing remain as implemented. New heights consequently affect normals, occlusion, shadows and the existing under-glow's surface classification; no shading controls change.

The bound is analytic. Each compact cap is `q²`, where `q=max(0,1-distance²/radius²)`, so a cap lies in [0,1]. Its contribution is zero with zero slope at the edge. A field is the product union `1-product(1-cap)`, also in [0,1], multiplied by a footprint fade in [0,1]. The three amplitudes sum to 180 m; the strength is clamped to [0,1], and the final result is clamped to `CLOUD_RELIEF_MAX`. Centres stay inside [cell+0.2, cell+0.8] and radii are at most 1.05 cells. An omitted cell outside the surrounding 3×3 neighbourhood has its nearest possible centre at least 1.2 cells away, so truncation drops no nonzero cap. The cloud upper bound becomes −650+382+180 = −88 m before holes or Earth curvature. A2's conservative scene ceiling is still dominated by terrain.

`h_cloud_cr` applies the relief before existing cloud holes. `hfun` supplies it to the marcher and soft-shadow rays. `shade` uses the same strength for cloud classification and both normal samples. `cloud_glow` uses it for classification, trough bias and its existing fog calculation. `hills/beacon._world_keep_module` copies the whole shared world source into its generated module, so A13 inherits the same geometry without modifying its WIP source.

## Reproduce

Run from the repository root with an absolute Python executable and output directory. One thread is the default; coordinate with the owner of the shared M4 before increasing it (maximum 4). Run each condition in a fresh process. `--source-root` can point to an original `the-long-dawn` tree for the third, original-source control; that tree permits only `--relief 0` when it lacks the new API.

```sh
python the-long-dawn/edit/tools/cloudsea_study.py --shot A2 --frames 380,500 --relief 0 --out /absolute/study/A2/off
python the-long-dawn/edit/tools/cloudsea_study.py --shot A2 --frames 380,500 --relief 1 --out /absolute/study/A2/on
python the-long-dawn/edit/tools/cloudsea_study.py --shot A13 --frames 3720,3799 --relief 0 --out /absolute/study/A13/off
python the-long-dawn/edit/tools/cloudsea_study.py --shot A13 --frames 3720,3799 --relief 1 --out /absolute/study/A13/on
python the-long-dawn/edit/tools/cloudsea_study.py --shot A13 --frames 3720,3799 --relief 0 --source-root /absolute/original/the-long-dawn --out /absolute/study/A13/original
```

At the default scale, outputs are complete 960×402 frames. A2 cut frames 380/500 map to shot frames 300/420, use its ordinary 1.5× supersampling, and receive its `FINISH`. A13 calls its full `render` (foreground, simulations and its own finishing included); it is not a world-layer preview. The driver wraps the actual world module's Python entry points, copies their parameter arrays, sets the same strength for march/shade/glow, and restores the original callables even on failure. It records the actual world parameters observed by each entry point. The flag is data passed to compiled functions, not a mutable Numba global.

Before importing A13, the driver requires its existing `shots/run/reveal_a_fires.npy` to be a nonempty, finite, real N×6 table (x, y, z, ignition frame, size, seed). It installs that validated table in A13's ordinary fire cache and records the file SHA256 in the source manifest. It never builds or replaces the table. A2 has no such input requirement.

PNG export uses deterministic nearest-integer quantization without random dither. Each directory has a manifest with source hashes, Git revision, frame hashes, render times and completion status. Each frame's `march_depths` list records the shape, dtype and SHA256 of every completed source-depth buffer, independently of the finished RGB hashes; no large depth arrays are saved. Compare these only across matching cameras and shapes. Existing frame/manifest paths are refused to prevent one condition overwriting another. Source-hash keys are relative to the selected source tree; an external study driver is recorded as `driver/cloudsea_study.py`. The skyline cache is cleared before every frame in both conditions.

## Verification and limits

The numerical tests compile small kernels but perform no full shot renders. Schedule them when compute is available. The driver tests use NumPy/OpenCV and synthetic frames, without renderer imports or JIT.

```sh
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s the-long-dawn/shots/run/tests -p test_cloudsea_relief.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s the-long-dawn/edit/tools/tests -p test_cloudsea_study.py -v
```

Tests check the scalar envelope and positive control, cell-boundary continuity, the original default-off height/hole contract, off/on/off parameter behavior through a single compiled signature, actual marcher depth changes, shadow occlusion, three-element compatibility, and the actual hills module generator. Driver tests check frame conversion, finishing, all three parameter routes, post-march depth hashing, restoration after an injected failure, and deterministic PNG decode.

Before calling default-off equivalence verified, compare complete original-source frames against revised-off frames with this same driver. Then inspect enabled frames beside off frames at equal exposure and framing. Numerical tests establish contracts; they do not establish visual quality.

The relief is a heightfield, without volumetric depth or overhangs. `falsedawn.skyline` and `nighta.skyline` already approximate the cloud skyline with a fixed `CLOUD_Y`; `nighta.fire_trans` also uses the fixed cloud-top mist altitude. Those atmospheric and cold-glow approximations are unchanged so the study isolates geometry under the accepted lighting model. It does not make every optical consumer an exact measurement of the new surface. The separate `dawn`, `relight` and ink renderers sample base `h_cloud` directly and are outside this opt-in study. In the A13 render, `HEROINE_V2` uses `peaks.render(..., summit_only=True)`: its baked far/cloud grids are empty, so a second, unflagged cloud surface does not cover the opted-in world layer. No continuity join or shot enablement is adopted by these stills. A13's 3679→3680 join still needs director review before any production enablement.
