# C v5 ember shots 11, 16 and 17

The new standalone [c_v5.py](../../shots/embers/c_v5.py) implements the three ember-world beats in the approved overnight v5.2 structure. Existing ember entrypoints and source files are unchanged; frames 0–2079 remain outside this entrypoint's accepted ranges. No PR10 noise proposal is included.

| Shot | Absolute C frames | Output stem | Behavior |
|---|---|---|---|
| 11 THE TRAP | 2320–2639 | `embers_C5_trap` | All forges burn; the prominent near-left forge withdraws, the others rise, it returns, then one tower approaches the Ring. |
| 16 THE FORGES GO COLD | 3840–3999 | `embers_C5_cold` | All forge flames and hot edges extinguish on C3848; grey smoke continues and the Ring stays gold. |
| 17 THE RING, UNFINISHED | 4000–4239 | `embers_C5_unfinished` | Gold beads cease, the Ring loses its heat/inscription and turns grey, and the storm clears before the last 24 frames. |

The source base is `339b1e3`. Reused ember and `look.py` paths were checked against production `ebe5546` before packaging; those paths had no intervening changes. Source hashes and measured receipts are in [receipt.json](receipt.json).

## Look and timing

The geometry comes from C3's tower layout and the Eye shot's medieval forge designs, irregular heights and ashlar treatment. The Ring uses the existing canonical `ringsolid` geometry/inscription; the flames use `cflame`; the storm reuses `c2.storm_layer` with the Eye's layer basis. The camera stays on the established C azimuth with a slow push. A new schedule controls each forge without changing shared A/C recipes.

C2420–2438 lowers forge 8 to 6% fire; it stays low through C2539. The other forges brighten from C2480 and rise through C2538. The low fire returns over C2540–2552 and its tower follows through C2578. Forge 5 advances over C2580–2636 until its crown is close to the Ring. Tower heights then hold through the two later shots.

The C3848 shutdown is a single-frame switch, including the tower emission override and the separate flame dispatch; its shutter starts at the downbeat. Cold masonry and smoke retain faint neutral illumination. Ring heat and inscription fade over C4000–4184; its own layer progressively loses saturation. Drops fade out by C4048. Storm and smoke reach zero by C4216, and the camera/Ring orientation hold through C4239 for EDIT's dissolve. There is no page dissolve or caption baked here.

Two visual corrections were made after rendering: the first camera clipped the Ring, so its target/field were adjusted; the first withdrawal selected a tower hidden behind a nearer forge, so the prominent near-left forge now carries the beat. The existing forge geometry and layout were preserved.

## Review evidence

Start with [C2360](stills/f_02360.png), [C2460](stills/f_02460.png), [C2510](stills/f_02510.png), [C2560](stills/f_02560.png), and [full-resolution C2639](stills/f_02639.jpg). Then compare [C3847](stills/f_03847.png) with [C3848](stills/f_03848.png), and [C4000](stills/f_04000.png), [C4110](stills/f_04110.png), [C4216](stills/f_04216.png).

Six consecutive 24-frame windows were rendered at 960×402 and inspected as reduced compositions and native-pixel subject crops:

| Action | Frames | Composition strip | Native subject strip |
|---|---|---|---|
| Withdrawal | 2418–2441 | [24 frames](motion/withdrawal-24.png) | [Native crop](motion/withdrawal-native-24.png) |
| Other forges respond | 2478–2501 | [24 frames](motion/others-surge-24.png) | [Native crop](motion/others-surge-native-24.png) |
| Low forge returns | 2538–2561 | [24 frames](motion/return-24.png) | [Native crop](motion/return-native-24.png) |
| Leader advances | 2592–2615 | [24 frames](motion/leader-24.png) | [Native crop](motion/leader-native-24.png) |
| Simultaneous shutdown | 3840–3863 | [24 frames](motion/shutdown-24.png) | [Native crop](motion/shutdown-native-24.png) |
| Ring cooling | 4100–4123 | [24 frames](motion/cooling-24.png) | [Native crop](motion/cooling-native-24.png) |

The 144-frame batch took 69.587 seconds inside the frame calls (median 0.482 seconds/frame); peak RSS was 692,912,128 bytes. The cold first test included kernel compilation and took 33.262 seconds wall time, with peak RSS 1,240,350,720 bytes. The corrected warmed stills and motion used the existing compiled cache.

The full-resolution C2639 test took 2.439 seconds for the frame and 4.963 seconds total including import/initialization, with peak RSS 845,447,168 bytes. Its saved JPEG is 1920×804, quality setting 95, with measured 4:4:4 sampling. These are measured samples, not a whole-shot throughput guarantee. All renders used Numba/BLAS 1 and OpenCV `setNumThreads(0)` (reported 1).

Inspection found a readable withdrawal/response sequence, a distinct leader, a simultaneous shutdown, and a clear grey Ring at the end. No real-place/company references or new human figures were introduced. This was internal visual review; no unbriefed-viewer comprehension test or real-time playback was conducted. The strips cover six one-second windows, not every frame of the 720-frame group. EDIT transitions, caption placement and sound synchronization still need the director's assembly review.

The clean-hold endpoint PNG files differ by at most two 8-bit code values; the shared `look.save_png` adds unseeded TPDF dither. The floating finished RGB was not saved for that pair, so byte equality before quantization is not claimed.

## Tests and rendering

[Fourteen tests](tests.log) cover frame ownership, event order, tower growth, simultaneous shutdown, actual flame/tower dispatch, grey Ring shading, smoke, shared-global restoration, Ring framing, output protection and the farm's actual `LaneCmd` subset rewriting. A [forced-hot negative control](negative-control.log) fails the tower-emission regression as expected. An actual CLI call also [refused an existing output](cli-safety.json) before building the scene; the original PNG hash was unchanged.

From `the-long-dawn/`, using an environment with NumPy, Numba, SciPy, OpenCV, Pillow and fonttools:

```sh
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python shots/embers/c_v5.py --frames 2360,2460,2510,2560,2639 --scale .5 --out /absolute/new/review-directory
```

The CLI requires an absolute output directory and refuses existing frame files. Positional frame syntax remains an alias; farm jobs use `--frames` so the farm can select subsets. The three jobs render PNGs, then `cloud/run_job.py` ships JPEG q95/4:4:4:

- [Trap job](../../cloud/jobs/embers_C5_trap.json)
- [Cold job](../../cloud/jobs/embers_C5_cold.json)
- [Unfinished job](../../cloud/jobs/embers_C5_unfinished.json)

For local full-resolution finals, use `--scale 1 --format jpg` and the appropriate absolute `~/ldfarm/out/<stem>` path (expanded by the caller). **The 720 local finals have not been rendered and no farm job has been launched.** Extended rendering was held because the M4 remained on battery; the single full-resolution test was explicitly allocated. These jobs are ready for the director's render decision.

Run the tests from the repository root with `NUMBA_DISABLE_JIT=1 python -m unittest discover -s the-long-dawn/shots/embers/tests -v`. The image kernels were exercised by the actual renders, while the tests use synthetic arrays/stubs to isolate scheduling and composition dispatch. `make_sheets.py` rebuilds the strips from saved PNGs without invoking a renderer.

## Handback

Inspect the full-resolution C2639, the C2460/C2510/C2560 sequence, the shutdown strip and the last grey-Ring frame first. The only new executable renderer is `shots/embers/c_v5.py`; no shared ember source was edited. The former C11 grasp remains on the old entrypoint and is not overwritten. Use the new stems when EDIT adopts the new timeline.

No operations remain running from this lane. The remaining work is full-shot delivery and EDIT/sound/transition review. If local delivery is chosen after power is available, render the three ranges above to their distinct output stems, validate frame counts/dimensions and inspect the resulting full sequences before reporting finals complete.
