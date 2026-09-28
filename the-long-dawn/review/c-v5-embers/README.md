# C v5 ember shots 11, 16 and 17

The standalone [c_v5.py](../../shots/embers/c_v5.py) implements the approved overnight structure and the subsequent **two front runners** amendment. Existing ember entrypoints and shared source files are unchanged; frames 0–2079 remain outside its accepted ranges. No PR10 noise proposal is included.

| Shot | Absolute C frames | Output stem | Behavior |
|---|---|---|---|
| 11 THE TRAP | 2320–2639 | `embers_C5_trap` | The near-left forge withdraws; the others surge; it returns; two matching towers rise neck and neck toward the Ring. |
| 16 THE FORGES GO COLD | 3840–3999 | `embers_C5_cold` | Every forge extinguishes on C3848. Smoke continues above cold masonry; the Ring stays gold. |
| 17 THE RING, UNFINISHED | 4000–4239 | `embers_C5_unfinished` | Gold beads cease, the Ring turns grey and the storm clears before the last 24 frames. |

Source base: `339b1e3`. Reused ember/`look.py` paths had no intervening changes at production `ebe5546`. [receipt.json](receipt.json) records exact renderer hashes, current images and measured results.

## Look and timing

C3 supplies the tower layout; the Eye shot supplies the medieval forge forms and ashlar treatment. The Ring is the canonical `ringsolid` object, fire comes from `cflame`, and the storm reuses `c2.storm_layer` on the Eye's layer basis. The camera makes a slow push on the established C azimuth.

Forge 8 drops to 6% fire over C2420–2438 and stays low through C2539. The others brighten from C2480 and rise through C2538. The low fire returns over C2540–2552; its tower follows through C2578. Forges 4 and 5 then climb together over C2580–2636. Both use existing forge form 5, equal starting/growth heights and the same fire intensity/color recipe. Their separate crowns approach either side of the Ring. The pair introduces no flags, symbols or differentiated national architecture. These heights hold through both later shots.

Each flame, smoke source, Ring point light and falling-gold endpoint follows the actual transformed crown. This corrects an alignment issue found during the pair amendment: geometry bent inward while the inherited `top()` stayed on the unbent axis. The override is confined to this new renderer.

C3848 is a simultaneous emission/flame switch; that frame's shutter starts on the downbeat. Neutral illumination retains faint cold stone and smoke. Ring heat and inscription drain over C4000–4184; its own layer loses saturation. Drops cease by C4048. Storm and smoke reach zero by C4216; camera and Ring orientation hold through C4239. Captions, page dissolve and sound belong to EDIT.

## Current visual evidence

Start with [full-resolution C2639](stills/f_02639.jpg), then [C2460](stills/f_02460.png), [C2510](stills/f_02510.png), [C2560](stills/f_02560.png), the shutdown pair [C3847](stills/f_03847.png)/[C3848](stills/f_03848.png), and [C4216](stills/f_04216.png).

All 12 current stills and the six motion windows below were regenerated from the pair source. Each strip was inspected as reduced composition and native-pixel subject crops; the full-resolution JPEG was also inspected at native size.

| Action | Frames | Composition strip | Native subject strip |
|---|---|---|---|
| Withdrawal | 2418–2441 | [24 frames](motion/withdrawal-24.png) | [Native crop](motion/withdrawal-native-24.png) |
| Other forges respond | 2478–2501 | [24 frames](motion/others-surge-24.png) | [Native crop](motion/others-surge-native-24.png) |
| Low forge returns | 2538–2561 | [24 frames](motion/return-24.png) | [Native crop](motion/return-native-24.png) |
| Two rivals advance | 2592–2615 | [24 frames](motion/rivals-24.png) | [Native crop](motion/rivals-native-24.png) |
| Simultaneous shutdown | 3840–3863 | [24 frames](motion/shutdown-24.png) | [Native crop](motion/shutdown-native-24.png) |
| Ring cooling | 4100–4123 | [24 frames](motion/cooling-24.png) | [Native crop](motion/cooling-native-24.png) |

The current 144 motion frames took 116.309 seconds inside frame calls (median 0.790 seconds; peak RSS 706,707,456 bytes). The native C2639 took 1.675 seconds for the frame, plus 0.230 seconds import and 1.339 seconds initialization; peak RSS was 1,033,125,888 bytes. The JPEG is 1920×804, quality setting 95, measured 4:4:4 sampling. All renders used Numba/BLAS 1 and OpenCV `setNumThreads(0)` (reported 1). These samples ran alongside other work and are not controlled throughput comparisons.

The pair remains visibly separate beside the complete Ring; the earlier withdrawal/response sequence and simultaneous shutdown remain readable in these inspected windows. No new human figures were introduced. This was internal review, with no unbriefed-viewer comprehension test or real-time playback. Motion coverage samples six windows rather than every frame. EDIT captions, transitions and sound still require assembly review. PNG output uses shared `look.save_png` TPDF dither; prequantization equality of the clean-hold endpoints was not measured.

## Superseded evidence

Commit `0a5c545` held a single-leader version. Its imagery is superseded by the current pair images; no final sequence was rendered from it. [The historical receipt](superseded-single-leader-receipt.json) retains its measurements and hashes, with artifact paths explicitly relative to that old commit. `render-first-render.log`, `render-pass2.log`, `render-pass3-stills.log` and `render-motion.log` are historical diagnostics, including the first rejected Ring-clipping camera and the subsequently corrected hidden withdrawal tower. Current render logs are [render-pair-review.log](render-pair-review.log), [render-pair-motion.log](render-pair-motion.log) and [fullres-render.log](fullres-render.log).

## Tests and delivery

[Sixteen lightweight tests](tests.log) cover ownership, timing, paired forms/heights/intensity, crown alignment, simultaneous shutdown, actual frame dispatch, cold Ring/stone/smoke, shared-global restoration, framing, overwrite refusal and the farm's actual `LaneCmd` subset rewriting. The [unbent-crown negative control](crown-negative-control.log) fails the alignment test; the earlier [forced-hot negative control](negative-control.log) also failed as expected. [CLI output protection](cli-safety.json) was checked on the unchanged CLI before the pair amendment.

From the repository root:

```sh
NUMBA_DISABLE_JIT=1 python -m unittest discover -s the-long-dawn/shots/embers/tests -v
```

From `the-long-dawn/`, with the renderer's NumPy/Numba/SciPy/OpenCV/Pillow/fonttools environment:

```sh
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python shots/embers/c_v5.py --frames 2360,2460,2510,2560,2639 --scale .5 --out /absolute/new/review-directory
```

The CLI requires an absolute output path and refuses existing frame files. Farm jobs use rewritable `--frames`; they render PNGs for `cloud/run_job.py` to ship as JPEG q95/4:4:4: [Trap](../../cloud/jobs/embers_C5_trap.json), [Cold](../../cloud/jobs/embers_C5_cold.json), [Unfinished](../../cloud/jobs/embers_C5_unfinished.json). No farm job has been launched.

Local 720-frame delivery is authorized after the pair review, with AC power confirmed. It will use `--scale 1 --format jpg` in the three distinct `~/ldfarm/out/<stem>` directories, absolute C filenames, per-frame dimension/decode/hash checks and silent 24fps previews. Completion is reported separately; this review receipt does not assert that the full delivery has finished.

## Handback

Inspect full-resolution C2639, the rivals strip, the C2460/C2510/C2560 sequence, the shutdown strip and the final grey Ring. The only new renderer is `shots/embers/c_v5.py`; the former C11 grasp remains on its old entrypoint. Use the new stems when adopting the new timeline. Current runtime source will stay frozen throughout final delivery. The remaining assembly work is caption/transition/sound review; no such changes are included here.
