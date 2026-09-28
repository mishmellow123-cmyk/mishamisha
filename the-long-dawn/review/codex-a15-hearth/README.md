# A15 — hearth, watcher and near-fire review

Review candidate for `CODEX_HANDOVER_3.md`, from code commit `8f56370`. The director still owns creative acceptance and final rendering. The job is prepared; no farm operation was launched.

The old three-course rounded cairn is replaced by a low, broken course of chipped stones and dark rubble. Actual terrain varies by about 1.65 m across the first complete ring: that candidate produced an artificial staircase. The adopted placement omits stones far above/below the fuel bed and lets the existing rock and the watcher's body conceal most of the hearth. This follows the handover's staging alternative. The fine pits and material details are largely unresolved or hidden in the finished framing; the images do not establish a close-up stone-material result.

The principal watcher stands closer to the fire, the camera pitches up one degree, and the light pool/cleared patch stay close to the hearth. A separate red shoulder wrap preserves the closed cowl. Hand retraction uses continuous targets; ankle positions remain fixed during the upper-body turn. Breathing and cloth variation are reduced. Tests establish continuity and stationary targets; they do not establish full physical support across the rough ledge. A finite sole-sampling audit found terrain contact on both boots, with uneven support.

David flagged the old near flame's solid white cone. The seventh fire now uses an opt-in world-space emitting volume, with irregular fuel patches, unequal roots and continuous advected turbulence. Foreground depth clips the actual volume. It has soft, stylized tongues; whether that treatment fits A is a director judgment. Distant beacon flame rendering keeps its existing recipe. The new close-depth/smoke policy applies to A14/A15; default nighta callers retain their prior behavior.

A14 shares this hearth and the landing camera. Its renderer therefore receives the same seventh-fire fuel seat, local lighting and volume in this PR, and both shots suppress the single sky shadow-ray fan. The subsequent A14 task owns larger glide fires, catch-core work and motion-blur depth. These are required companion changes to the shared A15 staging, not a completed claim about the rest of A14.

## Review images

Four check frames cover A4240, A4280, A4360 and A4399. The A4360 check is also supplied at 1920×804. The 24 consecutive motion frames are A4320–4343. Review both the composition strip and the separate native-pixel figure/fire crops. Receipts record the ordered frame list, source/input hashes, float-RGB hashes, PNG hashes and runtime thread settings.

The native motion crops use `(x0, y0, x1, y1) = (550, 135, 810, 402)` on each 960×402 image; the full-resolution A4360 detail uses `(1090, 345, 1490, 775)` on the 1920×804 image. Right/bottom bounds are exclusive. The composition strip alone is resized (320×134 per frame); labels occupy separate bars and never overwrite rendered pixels. [Packaging receipt](packaging.json) records every panel, its source frames, crop/resizing operations and output hash. Source receipts: [A15 checks/motion](receipt-final.json), [full-resolution A15](receipt-full.json), [baseline](receipt-baseline.json), [A14 join](receipt-join.json).

[Before / after](before-after.png) · [24-frame composition strip](filmstrip.png) · [Motion detail 1](motion-detail-1.png) · [Motion detail 2](motion-detail-2.png) · [Motion detail 3](motion-detail-3.png) · [Motion detail 4](motion-detail-4.png)

[4240](check-4240.png) · [4280](check-4280.png) · [4360](check-4360.png) · [4399](check-4399.png) · [4360 full resolution](full-4360.png) · [4360 native detail](detail-4360.png) · [A14/A15 join](join.png)

## Verification

Visual audit: the implementing session inspected all four final check stills, the full-resolution A4360 crop, all 24 motion frames in the composition strip and native-pixel panels, and the A4239/A4240 join. The watcher stays steady through the sampled second, the shawl reads dark red, and the cowl stays closed and unlit. The new flame breaks up the former solid white cone; the prominent rounded cairn and sky ray are absent. The terrain still has sculpted, faceted shading, and a warm section of ledge remains visible to the left. Those material/stylistic judgments remain open for director review. This sample does not certify every frame of the final 160-frame shot.

The compiled RUN suite passes 51 tests with Numba JIT enabled, two Numba threads and sequential OpenCV. Coverage includes actual foreground occlusion, oblique-ray depth, continuous time/temperature, single atmospheric transmission, planted ankle targets, continuous glove geometry, shared-lighting call sites, fuel-seat dispatch and the A14/A15 camera boundary. Existing terrain-ceiling and camera regressions remain unchanged and pass. In-memory negative controls fail for the old multi-metre fire bias, abrupt hand release, rotating ankle targets, ignored/incorrect volume depth, duplicate atmospheric attenuation and omitted shared-lighting/seat dispatch.

```sh
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B -m unittest discover -s shots/run/tests -p 'test_*.py'
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 shots/run/watchers_a.py --frames 4240,4280,4360,4399 --scale 0.5 --out renders/_codex_a15_checks
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 shots/run/watchers_a.py --range 4320-4343 --scale 0.5 --out renders/_codex_a15_motion
```

The checked-in review PNGs use deterministic rounding after the existing finish pass; the production CLI writer adds its usual output dither. Exact review hashes refer to the supplied images, not to that separate export path.

Final job: `cloud/jobs/watchers_a_hearth3.json`, A4240–4399, output `renders/watchers_A_hearth3`, two single-thread workers. The separate output directory preserves existing finals and prevents their `--skip` reuse. After approval and rendering, the director/EDIT must select the new output; this PR changes no EDL.
