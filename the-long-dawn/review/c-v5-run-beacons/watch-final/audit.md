# Watch final-delivery audit

2026-09-28 UTC. All 240 C4480–4719 images are delivered. No new visual blocker was found in the sampled final frames. This is a final-image sample audit; full-speed playback, EDIT joins and audience comprehension were not evaluated.

## Delivery and provenance

The delivery validator fully decoded every JPEG at 1920×804, checked q95 quantization and 4:4:4 sampling, and matched every file hash to its renderer receipt. The completed raw delivery receipt has SHA256 `adfa5569738086df615aae437f35bd12d7c40f30e1c6647916c1717ad56a2c39`. The silent preview contains 240 frames at 24 fps.

The initial serial run's failed receipt is retained deliberately: it records the authorized interruption after C4480–4481, whose original bytes were kept and verified. The two replacement ranges, C4482–4600 and C4601–4719, both completed. There is no missing or duplicated frame number in the combined delivery.

All 118 source/data hashes pinned by the PR's `runtime.json` still match. The frozen Watch driver is SHA256 `1c2a1cc7a797eaa29c338f6b43f9efc85301a352e986a67184db19141973e4e0`. The final C4600 JPEG is byte-identical to the native review image rendered before production. That comparison covers one frame; it does not establish all-frame equivalence between different thread settings.

## Visual evidence

Root inspected native final C4480, C4600, C4610 and C4719, a 24-frame consecutive sheet spanning C4592–4615, and 25 keyframes distributed across the shot. The union is 45 unique final images, each independently hash-matched to the delivery receipt in `evidence.json`. Sheet tiles are generated at 480×201; the viewer may scale the full sheet to fit.

The consecutive sheet crosses the split-worker boundary at C4600/C4601. Terrain silhouettes, ink lines and the camera's progression remain coherent across that boundary; no obvious one-frame positional jump or change of exposure appeared. The distributed samples show the continuous lateral move through the established terrain, with gold beacon marks present throughout the inspected states. No new ignition beat or apparent blackout appeared in these samples.

The larger foreground fires resolve clearly at native size. The most distant marks are tiny, and at reduced size they approach gold dots; the images do not establish that a viewer can count all eight beacons in every pose. The eight already-lit catalogue entries are verified by the implementation tests, independently of that perceptual limitation. Existing ink-fire shapes and terrain remain unchanged.

## Whole-shot numerical checks and limits

The existing `frame_qc.py` examined 240/240 frames and 239/239 adjacent pairs: no missing/undecodable/wrong-size files, black or near-constant frames, duplicate pairs or pop candidates. The checker was verified against the current production branch before use. Its thresholds are uncalibrated review aids; zero candidates does not prove the absence of flicker or other visual defects. No neighboring-shot joins were measured.

The 195 unsampled final images were mechanically checked but were not visually reviewed in this audit. Ordered stills do not establish real-time motion quality, timing with captions or music, or narrative comprehension. Those remain assembly-review questions. No source, finished frame, EDIT or sound file was changed.

Artifacts: `worker-join.jpg`, `shot-samples.jpg`, `evidence.json`, `frame-qc.json` and `frame-qc.log`. Raw audit and delivery files remain local; published copies normalize home/workspace paths and carry a separate manifest of raw and published hashes.
