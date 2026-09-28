# Paired Reveal final-delivery audit

2026-09-28 UTC. **All 240 C2880–3119 final JPEGs are delivered.** The completed sample audit found no new visual blocker. PR review readiness does not establish audience comprehension, full-speed playback quality or approval of the assembled film.

## Delivery and provenance

Both one-thread render workers completed their disjoint 120-frame ranges with exit code 0. The delivery validator finished at 10:10:32 UTC; its queue finished at 10:10:40 UTC. Every final JPEG passed exact frame coverage, full decode at 1920×804 RGB, q95 quantization, 4:4:4 sampling and hash agreement with its renderer receipt. The immutable raw completion receipt is SHA256 `d9627d6a8ade28d99c2488789555dfe9373c493704bfd351d4fb3189a6431101`.

The silent H.264 preview has 240 decoded frames at 24 fps and ten seconds duration, with no audio stream. Its SHA256 is `14929dfe20e2a195a3baef65ed98f5d3a99502e0af92a62efbeb9ff3f64eeaa5`. The original delivery receipt's `visual_final_audit: pending` records the validator's earlier state; this later report completes the bounded final-image review without modifying that receipt.

All 118 source/data files pinned in `runtime.json` still match. Final C2960 and C3119 are byte-identical to their earlier native review JPEGs. Those two comparisons establish identity for those images; they are not an all-frame comparison between thread configurations.

## Final visual coverage

The opening, late and completion audits together inspected **91 distinct delivered frames out of 240**. Every sample hash now matches the completed delivery receipt, including all 26 opening and 25 late samples that had previously been matched to snapshots of running receipts. `combined-sample-coverage.json` contains the exact union and hashes.

Coverage includes native start/middle/end states; three consecutive 24-frame windows, C2880–2903, C2988–3011 and C3072–3095; and distributed reduced compositions across the whole range. The opening and late audits also include native-pixel crops of both fires. Root's final pass inspected native C2999/C3000 across the worker boundary and C3119, plus the completed boundary and distributed sheets. Composition tiles are generated at 480×201 and may be scaled by the viewer to fit the sheet.

Both fires are visible from C2880 and grow together during the opening window. They remain at separate peaks in the inspected later states. The C2999/C3000 boundary preserves the positions and outlines of terrain and both fire marks, with no obvious positional jump or exposure change. The endpoint retains the two-site composition, with both fires within frame. No new dropout or broken terrain outline appeared in the reviewed windows or key states.

The farther fire remains a tiny, low-contrast gold mark at 480px. Its position is distinct to an informed viewer, but this audit does not prove that an unprompted viewer notices both fires or infers the reciprocal promise. The existing stylized ink-fire language is preserved; no realism claim is made.

## Whole-shot checks and remaining limits

The existing frame checker read all 240 frames and compared all 239 adjacent pairs. There were no missing, undecodable, wrong-size or unsupported images; no black/constant/near-constant/transparent candidates; no duplicate pairs or pop candidates. Its version was checked against production through `9ed020d`, which introduced no relevant renderer or checker change. These default thresholds are uncalibrated review aids and do not establish the absence of perceptual flicker. No neighboring-shot joins were measured.

The other 149 images received mechanical checks but were not visually inspected. Real-time playback, complete-story comprehension, provisional captions, sound synchronization and EDIT transitions remain assembly-review work. The source and all rendered output files stayed frozen. This package publishes review evidence only.

Final frames: `~/ldfarm/out/runC_reveal_pair_v5/f_%05d.jpg`. Preview and raw receipt: `~/ldfarm/comms/files/c5_reveal_pair_v5_2880_3119_silent_24fps.mp4` and `c5_reveal_pair_v5_delivery.json`. Full audit mirrors: `C5_reveal_final_audit/`, `C5_reveal_opening_final_audit/` and `C5_reveal_late_final_audit/` in the same mailbox files directory. Published text copies normalize private paths; `evidence-manifest.json` records raw and published hashes.
