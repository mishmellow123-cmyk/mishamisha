# A14 — beacon catches and opaque motion depth

This candidate follows A15 PR #7. Review and merge that dependency first. The current farm fetches `claude/long-dawn-v2`; the job's `branch` field does not select PR source. No farm job was launched.

The three approaching ridge fires (chain 1, 3 and 5) use a copied size table enlarged by one third. Their positions and ignition frames stay fixed; the three far answers and seventh hearth keep their sizes. The shared night recipe couples size to width, light pool, clearance, smoke and flame base. A15 inherits those same changes through `beaconrun_a.FIRES` and `fires_table()`, so the A14/A15 boundary and an A15 check are included here.

Each catch gains a compact warm core inside its existing halo. The two-frame rise and nine-frame decay remain unchanged. The seventh core sits just above the measured fuel seat introduced by PR #7; all catch components use the smaller near-fire depth tolerance so they cannot paint through a nearby cowl or stone.

The opaque hearth/figure pass wrote forward depth to `fr.zb` but left terrain-only ray distance in `fr.dist`. Camera motion blur therefore reconstructed the wrong world position at those pixels. A14 now converts only new, nearer opaque hits to Euclidean ray distance before warping and blur. Unchanged terrain/sky values remain intact. This repairs camera-induced blur; it does not add velocity for intrinsic figure motion or translucent flame/smoke.

## Review and verification

Visual audit: the implementing session inspected the four catch comparisons, every A4200–4239 frame in the composition strip and native-pixel panels, the A4239/A4240 boundary and the updated A4360 check. The rise and hand withdrawal remain continuous through the old A4208 pop point and settle into A15's pose. The close flare no longer washes through the watcher's silhouette. The approaching fires are more legible, especially the fifth, while far answers remain small points. The shared near flame is still stylized and some warm faceted ledge remains visible; those creative limitations carry over from PR #7. The sampled frames do not certify every frame of the 320-frame shot.

The local review uses 960×402 frames with two Numba threads and sequential OpenCV. Render receipts pin code/input hashes and ordered frames. The PNGs use deterministic rounding after the existing finish pass; the production CLI adds its usual output dither.

The compiled RUN suite passes 59 tests (JIT enabled; two Numba threads; sequential OpenCV). The new tests exercise actual catch pixels behind/in front of depth, the unchanged pulse timing, fuel-seat placement, resolution scaling, single atmospheric transmission, immutable source tables and the loaded shared chain. The depth integration executes the live render body with small pass substitutes, then the real warp and velocity functions. Negative controls fail when the synchronization call is removed, forward depth is mistaken for ray distance, or the old three-metre flare bias returns.

Prepared job: `cloud/jobs/beaconrun_a_catches3.json`, A3920–4239, two single-thread workers, output `renders/beaconrun_A_catches3`. Its `--out ../beaconrun_A_catches3` is intentional: this renderer resolves relative output paths from `renders/beaconrun_A`, unlike A15. Existing finals are preserved. Director/EDIT output selection is required after approval and rendering; no EDL is changed.

<!-- media-links:start -->

Check stills: [A3962](check-3962.png) · [A4042](check-4042.png) · [A4122](check-4122.png) · [A4202](check-4202.png).

Baseline/current pairs: [A3962](comparison-3962.png) · [A4042](comparison-4042.png) · [A4122](comparison-4122.png) · [A4202](comparison-4202.png).

[All 40 frames, A4200–4239](filmstrip-40.png) · [24 consecutive frames, A4207–4230](filmstrip-24.png).

Native figure/fire panels: [A4200–4205](motion-detail-1.png) · [A4206–4211](motion-detail-2.png) · [A4212–4217](motion-detail-3.png) · [A4218–4223](motion-detail-4.png) · [A4224–4229](motion-detail-5.png) · [A4230–4235](motion-detail-6.png) · [A4236–4239](motion-detail-7.png).

[A14/A15 join](join.png) · [A15 A4360 after the shared changes](a15-check-4360.png).

[Packaging receipt and crop bounds](packaging.json) · [Check sources](receipt-catches.json) · [Motion sources](receipt-motion.json) · [A15 sources](receipt-a15-after.json) · [Baseline sources](receipt-baseline.json).

<!-- media-links:end -->
