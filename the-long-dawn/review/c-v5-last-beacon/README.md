# C v5.2 — LAST BEACON

Base: `339b1e370f6cbd01acc18c6d8a2e50f5e7f302e2`. Shot 15 owns **C3440–3839**. Structure approved in the overnight package; caption wording remains provisional and is left to EDIT.

Tracked receipts normalize personal filesystem paths to portable placeholders. Exact raw receipts are retained locally outside the repository; image and renderer hashes are unchanged.

`shots/map/last_beacon.py` is an opt-in renderer. It reuses the existing terra v5 geography, `bake.bake_region` parchment and terrain ink, `render.Cam`/`Sheet` projection and mip sampling, and `road.RoadShot` pen flames and living fire. It introduces eight fictional territories, a separate ignition schedule, and a camera that settles for the last dark kingdom. It never constructs the old RoadShot ring or road. Existing map, book, BURN, edit, sound and farm code are unchanged.

The eighth kingdom remains dark during C3724–3783. Its catch occupies C3784–3791; C3792–3839 provides the 48-frame all-lit caption hold. Beacons cross near/far and left/right during the spread. All frames can be rendered independently.

The crop atlas uses the original baker at 48 pixels per map unit, in serial 768-pixel tiles. It has separate source-pinned geography and texture caches under `renders/map_last_beacon_C/`. The loader fails on missing, truncated, wrong-shaped or source-mismatched texture data. It does not read an unrelated accepted map cache and claim fresh provenance for it.

`cloud/jobs/map_last_beacon_c5.json` builds then renders serially, and declares all 400 frames. Its PNGs satisfy `cloud/run_job.py`'s input contract; the runner ships JPEG quality 95, 4:4:4. Local renderer output defaults to the same JPEG settings and `f_%05d.jpg` C numbering. Images are published by atomic rename after encoding.

## Validation (2026-09-28)

- Seven editorial/cache tests pass: 400-frame span, prescribed hold intervals, last catch, eight distinct territories, irregular spatial order, beacon visibility at every frame, camera settling, atlas coverage, rejected malformed cache, source hashes and valid C ranges.
- Five existing cloud-runner tests pass (the suite emits pre-existing unclosed-file ResourceWarnings).
- Syntax and the serial job's PNG/output declaration checked.
- Original world/glyph bootstrap and all 28 atlas tiles completed in 674.60 seconds; maximum RSS 1,264,549,888 bytes. The unchanged world generator's fractional-power warning is retained in `bootstrap.log`. All eight positions passed the terrain land/lake checks.
- Reviewed 33 reduced frames and native C3724/C3792; the parent task independently reviewed the native dark hold and catch strip and approved local delivery. Its 480px check prompted one correction: heavier ink for the existing unlit stack, smoothly reduced as its flame catches. No new symbol or light was added.
- `catch-strip-full.jpg`, `catch-strip-detail.jpg` and `catch-24frames.mp4` contain consecutive C3772–3795. The video is verified H.264, 960×402, 24 frames at 24 fps, silent. All 24 frame hashes differ.
- Rendering C3724, then C3839, then C3724 gave exact equality of the two C3724 float arrays. Every review JPEG decoded as RGB at its expected size with 4:4:4 sampling. `validation.json` records hashes of all 35 review images.
- Native C3724/C3792 measured 1.40 and 0.98 seconds respectively; maximum RSS for that process was 755,302,400 bytes. `runtime.json` distinguishes the earlier preview source from the frozen delivery renderer (`cee6cc2a7b6fd97c497503ace1578b1877333833f69c4026b95f64e096b41601`).
- All 400 final frames passed full JPEG decode, exact C3440–3839 coverage, 1920×804 RGB, 4:4:4 sampling and quality-95 quantization. Both independently rendered native reference images match the final sequence byte-for-byte. `delivery.json` records every image hash; its frame-manifest SHA256 is `c0b55daa3f8e523818a5f01b54ddd14da1b923bc84fe3f6967931cdf712acc3e`.
- Final rendering took 256.47 seconds, maximum RSS 1,085,554,688 bytes. Delivery: `~/ldfarm/out/map_last_beacon_C/f_03440.jpg` through `f_03839.jpg`.
- Silent full preview: `~/ldfarm/comms/files/c5_map_last_beacon_3440_3839_silent_24fps.mp4`. `preview-probe.json` verifies H.264, 1920×804, 400 frames at 24 fps, 16.666667 seconds, one video stream and no audio stream.

Commands (from `the-long-dawn/`; all render work uses one Numba/BLAS thread and OpenCV threading is disabled by the entry point):

```sh
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 shots/map/last_beacon.py bake
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 shots/map/last_beacon.py frames --frames 3440,3582,3724,3783,3792,3839 --scale .5 --out review/c-v5-last-beacon/reduced
```

No farm job has been launched. The recipe pins the package versions used by local validation; it has not been executed on a farm node.
