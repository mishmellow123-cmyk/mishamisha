# A14 — beacon catches and opaque motion depth

This candidate follows A15 PR #7, which the director merged at `287b1a1`; frame-QC #5 and legacy reach #9 are also merged. Handover 4 reports that A15's final job is launched and EDIT selects `renders/watchers_A_hearth3`. PR #8 targets `claude/long-dawn-v2` and includes production `70d102b` through merge `ee085f1`. Codex launched no farm job.

The three approaching ridge fires (chain 1, 3 and 5) use a copied size table enlarged by one third. Their positions and ignition frames stay fixed; the three far answers and seventh hearth keep their sizes. The shared night recipe couples size to width, light pool, clearance, smoke and flame base. A15 inherits those same changes through `beaconrun_a.FIRES` and `fires_table()`, so the A14/A15 boundary and an A15 check are included here.

Each catch gains a compact warm core inside its existing halo. The two-frame rise and nine-frame decay remain unchanged. The seventh core sits just above the measured fuel seat introduced by PR #7; all catch components use the same smaller near-fire depth tolerance (2.5–6 cm inside 60 m).

The opaque hearth/figure pass wrote forward depth to `fr.zb` but left terrain-only ray distance in `fr.dist`. Camera motion blur therefore reconstructed the wrong world position at those pixels. A14 now converts only new, nearer opaque hits to Euclidean ray distance before warping and blur. Unchanged terrain/sky values remain intact. This repairs camera-induced blur; it does not add velocity for intrinsic figure motion or translucent flame/smoke.

## Review and verification

Visual audit: the implementing session inspected the four catch comparisons, every A4200–4239 frame in the composition strip and native-pixel panels, the A4239/A4240 boundary and the updated A4360 check. The rise and hand withdrawal remain continuous through the old A4208 pop point and settle into A15's pose. The close flare no longer washes through the watcher's silhouette. The approaching fires are more legible, especially the fifth, while far answers remain small points. The shared near flame is still stylized and some warm faceted ledge remains visible; those creative limitations carry over from PR #7. The sampled frames do not certify every frame of the 320-frame shot.

The local review uses 960×402 frames with two Numba threads and sequential OpenCV. Render receipts pin code/input hashes and ordered frames. The PNGs use deterministic rounding after the existing finish pass; the production CLI adds its usual output dither.

The combined branch passes **64 compiled RUN tests** after integrating production `70d102b` (JIT enabled; two Numba threads; sequential OpenCV). The [test log](tests-latest.log) records runtime and source provenance. The new tests exercise actual catch pixels behind/in front of depth, the unchanged pulse timing, fuel-seat placement, resolution scaling, single atmospheric transmission, immutable source tables and the loaded shared chain. The depth integration executes the live render body with small pass substitutes, then the real warp and velocity functions. Negative controls fail when the synchronization call is removed, forward depth is mistaken for ray distance, or the old three-metre flare bias returns.

The two completed integration frames, [A4240](integration-4240.png) and [A4360](integration-4360.png), match the earlier candidate exactly: unquantized RGB hashes and every PNG pixel. The implementing session inspected both at composition and native size. [Comparison](comparison-integration.json), [source receipt](receipt-integration.json). This verifies RUN-A4's first shader expansion through `6ae7180`; the intervening legacy correction changes no compiled function ([static receipt](receipt-source-integration.json), [probe](check_source_integration.py)).

Production subsequently added rough-box/horn/stone changes at `8bcc818`. Those change compiled functions, so the preceding pixel result is **not** relabelled as verification of that newer shader. The merged tip has five byte-identical watcher geometry samples and source/flag checks showing the new behavior inactive in this scene; see [latest source checks](latest-sp/README.md). Its compiled unit suite passes. No complete frame was re-rendered with the default newer shader.

A separate disposable experiment reduced forced noise inlining and produced an identical A4240 in 45.605 seconds total. It is an [unadopted performance proposal](COMPILER_EXPERIMENT.md), with unmatched timing conditions and only one-frame numerical coverage. No experimental shader change is in this PR.

## A15 dependency and prepared jobs

The original `join.png` uses #8 for both shots. The [mixed-source comparison](join-existing-a15.png) also shows local #7 A4240: its right ridge fire is visibly smaller than in #8 A4239/A4240. [Native fire detail](join-existing-a15-fire-detail.png), [watcher detail](join-existing-a15-figure-detail.png) and the [source receipt](receipt-mixed-join.json) distinguish those versions. These are local review frames. The actual farm A4240 and its source receipt are unavailable here and were requested in [the first dependency comment](https://github.com/mishmellow123-cmyk/mishamisha/pull/8#issuecomment-5862450228) and [the follow-up](https://github.com/mishmellow123-cmyk/mishamisha/pull/8#issuecomment-5862477890).

Prepared jobs, neither launched by Codex:

- [`beaconrun_a_catches3.json`](../../cloud/jobs/beaconrun_a_catches3.json): A3920–4239, output `renders/beaconrun_A_catches3`. Its `--out ../beaconrun_A_catches3` is intentional: A14 resolves relative output paths from `renders/beaconrun_A`.
- [`watchers_a_catches3.json`](../../cloud/jobs/watchers_a_catches3.json): all 160 A15 frames, A4240–4399, output `renders/watchers_A_catches3`. This companion preserves the accepted `watchers_A_hearth3` output for comparison. The continuing flames, light pools and smoke inherit the changed sizes; replacing only A15's opening frames would leave a version change inside the shot.

Both jobs use two single-thread workers. The director decides whether to launch the companion if #8 is accepted, then reviews the rendered pair before EDIT selection. No EDL is changed.

**Source changes can affect an already-launched job.** `cloud/farm_node.py:48,95–103,212–219` fetches the live `claude/long-dawn-v2` tip for each unit; the JSON `branch` field does not select renderer source. Before merging #8, confirm the existing A15 run's completion and unit commit provenance with the director. Pending units could otherwise acquire #8 under the existing output name. The separate companion output does not by itself prevent that risk.

<!-- media-links:start -->

Check stills: [A3962](check-3962.png) · [A4042](check-4042.png) · [A4122](check-4122.png) · [A4202](check-4202.png).

Baseline/current pairs: [A3962](comparison-3962.png) · [A4042](comparison-4042.png) · [A4122](comparison-4122.png) · [A4202](comparison-4202.png).

[All 40 frames, A4200–4239](filmstrip-40.png) · [24 consecutive frames, A4207–4230](filmstrip-24.png).

Native figure/fire panels: [A4200–4205](motion-detail-1.png) · [A4206–4211](motion-detail-2.png) · [A4212–4217](motion-detail-3.png) · [A4218–4223](motion-detail-4.png) · [A4224–4229](motion-detail-5.png) · [A4230–4235](motion-detail-6.png) · [A4236–4239](motion-detail-7.png).

[A14/A15 join](join.png) · [A15 A4360 after the shared changes](a15-check-4360.png).

[Packaging receipt and crop bounds](packaging.json) · [Check sources](receipt-catches.json) · [Motion sources](receipt-motion.json) · [A15 sources](receipt-a15-after.json) · [Baseline sources](receipt-baseline.json).

<!-- media-links:end -->
