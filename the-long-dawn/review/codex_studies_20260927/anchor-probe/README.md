# Council flame-anchor probe

The proposed raised anchor was **not applied**. These files preserve a diagnostic of frame 5160; they do not establish final visual quality or resolve the weak burning appearance at the right edge.

`diagnose.py` calls the actual `scene3` state, `flame3.torch_density`, `geom3.sd_fig`/`trace_fig` and camera projection. For each of torches 0, 5, 6 and 7 it samples 55,440 world-axis points relative to the existing flame base: x = [-0.13, 0.26), y = [-0.21, 0.14), z = [-0.10, 0.57), with 0.012 m spacing. It weights each positive-density point by `density * (0.05 + temperature**2.6)`. The hot subset means temperature > 0.6. The denominator is the sum of these weights within this finite grid, separately for each torch and subset.

The hypothetical alternative moves a downward torch's base toward its proximal fuel cap and upper surface: `smooth(max(-axis_z, 0)) * (-0.08 * axis + [0, 0, 0.034])`, where `smooth(x) = x*x*(3-2*x)`. At 5160 this moves the base about 96 mm upward and 39 mm outward. The density field is translated with no energy, dimensions, timing or intensity change. Upright torches would have zero displacement.

| Torch | Current hot weight in frame and unoccluded | Hypothetical raised anchor |
|---|---:|---:|
| 0, lower left | 72.17% | 58.00% |
| 5, lower right | 68.86% | 56.06% |
| 6, middle right | 74.44% | 37.50% |
| 7, upper right | 87.94% | 97.96% |

All thirteen inner torches have `lit=1` at this frame. For the three right-hand torches, the nominal core point at `base + [0,0,0.22*flame_height]` is outside the actual fuel geometry. Own-figure occlusion accounts for 11.92%, 10.47% and 15.00% of all sampled emission weight respectively. Raising the anchor reduces overlap but shifts the projected base about 30–35 pixels outward at 960×402, so cropping outweighs that gain for torches 5 and 6. This is evidence against this particular anchor proposal as a reliable correction; it does not rule out every possible anchor adjustment.

Limits: the grid is finite and not an exhaustive flame integral; there is no convergence study. Only the torch's own figure occludes the samples. Other figures, terrain, smoke, ray-march attenuation, emitted RGB, perspective integration, bloom, exposure, grading and pixel sampling are omitted. The percentages are measured fractions of this diagnostic's sample weights, **not fractions of final image brightness**. “Nominal core” is a geometric sample at the temperature envelope's transition, not the measured brightest voxel. Source hashes, exact values, runtime versions and sample counts are in `results.json`. The import uses `CROWD=0` because crowd occlusion is not part of this diagnostic; the comparison renders below require the crowd.

Reproduce from the repository root in the renderer's Python environment (replace the output path):

```sh
NUMBA_NUM_THREADS=2 python -B the-long-dawn/review/codex_studies_20260927/anchor-probe/diagnose.py --repo . --out /absolute/study/anchor-results.json
```

The saved result omits the original absolute checkout path. Source hashes and measured data are unchanged. The diagnostic is distinct from the rendered material candidate; no anchor change was applied.
