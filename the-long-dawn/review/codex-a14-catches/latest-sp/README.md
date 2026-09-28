# Shared figure renderer: latest production change

Compared production `6ae71801bae7b188a8f2d9533761c7e13824fc06` with `70d102b148ff225947f252f3183aca9503a6d4fd` using source and AST only. `static-receipt.json` records the full hashes, function hashes and caller expressions; `static_probe.py` reproduces the check without importing project code.

Three of the eleven compiled functions changed: `_prim`, `_horn`, and `render`. The additional changed Python functions are `Scene.box`, `lantern_v3`, and `stone_ring`. The earlier compiled-function identity finding stops at `6ae7180`.

Current A14 calls `WA.draw_figures`; A15 uses the same function. `watchers_a.py` calls only `SP.Scene`, `SP.traveller`, and `SP.render`. Its wrap is a material-0 bell. The traveller uses material 0 for clothes and material 2 for optional wooden staffs. Its boxes omit roughness and glass arguments.

| New behavior | Required input | Current watcher path |
|---|---|---|
| Box displacement (`sdfppl.py:293`) | Type 1 and positive `P[14]` | Box defaults leave `P[14:16]` zero. Bell fold depth also uses `P[14]`, but bells have type 4 and bypass this branch. |
| Revised horn emission (`:415`) | A glass hit and `O[14] > 0.5` | `Scene.begin` initializes this flag to zero; no watcher call creates horn panes or glass. |
| Revised rough-stone shading (`:506`) | Material at least 5.5 | The watcher scene uses only materials 0 and 2. The hearth has its own `hearth_a` tracer. |
| Changed lantern bars and stone-ring shapes | Calls to `lantern_v3` or `stone_ring` | Neither constructor is called by this watcher scene. |

All five production shot inputs (`watchers_a`, `beaconrun_a`, `nighta`, `hearth_a`, `fire_near_a`) are byte-identical between those two production commits. This statement excludes the A14 candidate's own changes.

The separate geometry probe records actual array comparisons. This static check does not run that probe, compile either shader or compare pixels. Unselected source branches do not prove machine-code identity after recompilation. The earlier image check retains its original source receipt and cannot be relabelled as verification of this newer shader.

[Actual geometry receipt](geometry-receipt.json); all five old/new P/O array pairs are exact. Reproduce with `geometry_probe.py --repo <repo> --output <new-output> --old-ref 6ae7180 --new-ref 70d102b` (explicit refs override its historical defaults).

A [compiled primitive smoke check](primitive-smoke-receipt.json) exercises one actual watcher box at four points and a copied positive-roughness control. All outputs are finite, default results exactly equal `_sd_box`, and the rough control changes them. `_prim` compiled in nopython mode; `SP.render` was not compiled or invoked by this probe. [Script](primitive_smoke.py).
