# C v5: paired Reveal and the Watch

Two opt-in drivers implement C2880–3119 and C4480–4719. Both use the existing RUN terrain, ink pass and beacon drawings. The accepted render dispatch, EDIT, sound and farm controller are unchanged.

## Staging

**Paired Reveal:** the original near beacon and a second beacon on catalogue summit 886 catch together at C2880. Both use the existing ignition envelope. The far position is refined onto its terrain summit; seven sampled camera poses verify that both sites remain inside the frame, separated and visible through the actual terrain. The existing reveal's wide camera poses 160–239 span the full 240 frames, while the fire clock advances at 24 fps. This widens the opening enough to keep both fires small. The original reveal began already lit; the simultaneous ignition is deliberate new behavior.

**Watch:** the original scroll's last 240 poses, source frames 80–319, retain their original world/camera/fire clock. A copied beacon catalogue sets all eight ignition times to −400. The original catalogue is not mutated.

## Verification

- All 73 RUN tests pass, including actual shared-world visibility and ignition checks, farm command splitting, catalogue preservation, and recorded setup failure. The five cloud runner tests also pass; their existing file-handle ResourceWarnings remain visible in the log.
- Root inspected both 24-frame strips at composition and native crop scale, the Watch's 1920×804 C4600 and ending C4719 sample, and paired Reveal's 1920×804 C2960/C3119. The current ink-fire style is preserved. The far fire is intentionally small; these samples establish separation, not a blind audience test of the entire story.
- The Watch's C4600 one-thread and four-thread JPEGs are byte-identical. That single frame does not establish all-frame cross-thread equivalence. Existing-frame refusal returned exit 2 without changing the image. Setup-failure tests prove that failure receipts cannot be reused as successful retries.
- Native Watch sample receipt: 30.368416 seconds. Paired samples: 92.500392 and 86.590769 seconds under concurrent rendering. These are observed wall times from different loads, not a controlled speed comparison or completion estimate.

`runtime.json` pins the source trees and installed libraries. `evidence-manifest.json` records raw and published file hashes; home/workspace paths are normalized only in published review copies. Raw local evidence is retained.

## Delivery and remaining work

**Watch is delivered:** all 240 C4480–4719 final JPEGs passed exact coverage, full decode, 1920×804 dimensions, q95 quantization, 4:4:4 sampling and renderer-receipt hash checks. Both split ranges completed; C4480–4481 retain their verified original bytes. The initial serial receipt still says `failed` because that owned process was deliberately interrupted for rescheduling. Preserve that receipt as provenance for those two frames.

The [Watch final audit](watch-final/audit.md) covers 45 unique final frames, including native start/middle/end images and a 24-frame strip across C4600/C4601, the split-worker boundary. No new visual blocker was observed. Whole-shot numerical QC covers 240 frames and 239 adjacent pairs without review candidates. All 118 pinned source/data hashes still match; final C4600 is byte-identical to its earlier native review image. Distant fires remain tiny at reduced size. These are sampled still/strip observations, not full-speed playback or audience testing. The immutable automated delivery receipt records visual audit as pending at its creation; the later audit report supplies the sampled review result.

**Paired Reveal is rendering:** the local queue launched two disjoint one-thread workers at 09:06 UTC after Ember and all three Pages deliveries passed their gates. Final delivery and its final-image audit remain outstanding, so this PR stays draft. Its supervisor preserves finished images, verifies hashes and encoding, and publishes a silent preview on completion. The queue's 5 GiB launch/2 GiB stop gates and four-process RUN cap remain in force.

Output directories are `~/ldfarm/out/runC_watch_v5/` and `~/ldfarm/out/runC_reveal_pair_v5/`, each contracted for 240 images named `f_%05d.jpg` with absolute C frames. The completed Watch's silent 24 fps preview is `~/ldfarm/comms/files/c5_watch_v5_4480_4719_silent_24fps.mp4`; receipt: `c5_watch_v5_delivery.json` in the same directory. Final audit artifacts are mirrored to `~/ldfarm/comms/files/C5_watch_final_audit/`. Consult the latest mailbox/handback before starting any duplicate render. EDIT joins, captions and sound still require assembly review.

Fallback farm recipes are `cloud/jobs/runC_watch_v5.json` and `cloud/jobs/runC_reveal_pair_v5.json`: named frame selectors support farm subdivision, the renderer writes PNG and the existing shipper produces JPEG. Neither job has been launched on the farm. Keep both drivers together: paired Reveal imports the Watch's image writer.

The production branch was checked through `0d5e8f6`; no newer changes existed in the relevant shared renderer trees. Do not adopt the separate PR10 compiler experiment for these renders. No merge or final creative approval is implied by this package.
