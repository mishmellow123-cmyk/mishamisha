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

## Completed delivery and review limits

Both shots are delivered: 480 final JPEGs in total, each 1920×804 RGB, q95, 4:4:4, with exact absolute-frame coverage, full decode and per-frame renderer-receipt hash verification. Both silent previews contain 240 frames at 24 fps.

| Shot | Finals | Preview and receipt under `~/ldfarm/comms/files/` |
|---|---|---|
| Paired Reveal, C2880–3119 | `~/ldfarm/out/runC_reveal_pair_v5/` | `c5_reveal_pair_v5_2880_3119_silent_24fps.mp4`; `c5_reveal_pair_v5_delivery.json` |
| Watch, C4480–4719 | `~/ldfarm/out/runC_watch_v5/` | `c5_watch_v5_4480_4719_silent_24fps.mp4`; `c5_watch_v5_delivery.json` |

The [completed Reveal audit](reveal-final/audit.md) reconciles 91 unique final images across opening, worker-boundary and late 24-frame windows plus native/distributed key states. The [earlier opening](reveal-opening-final/audit.md) and [late](reveal-late-final/audit.md) records retain their running-receipt snapshots; all 51 of those sample hashes now match the completed receipt. Root inspected native C2999/C3000 and C3119 at closure. No new blocker appeared; both sites remain visible and separated, and the worker boundary is coherent. Final C2960/C3119 are byte-identical to their earlier native review images. The far fire remains subtle at 480px; audience recognition of both fires is unproven.

The [Watch audit](watch-final/audit.md) covers 45 unique final images, including native start/middle/end and the 24-frame C4592–4615 worker-boundary strip. No new blocker was observed. Final C4600 matches its earlier native review JPEG. Preserve the initial serial receipt: its intentional interruption left two retained, verified frames, C4480–4481; both replacement ranges subsequently completed.

Each full-shot numerical QC read 240 frames and compared 239 adjacent pairs without candidates. All 118 pinned source/data hashes still match. The automated completion receipts' pending visual labels predate these later sampled reviews and remain immutable. Audit mirrors are `C5_reveal_final_audit/` and `C5_watch_final_audit/` in the mailbox files directory.

Reveal finished at 10:10 UTC after the Ember/Pages dependencies and resource gates passed; both workers exited 0. No owned RUN renderer remains active. These final audits use native stills and ordered strips. Full-speed playback, audience comprehension, provisional captions, sound synchronization and neighboring-shot joins remain assembly-review work. PR readiness is not final creative approval.

Fallback farm recipes are `cloud/jobs/runC_watch_v5.json` and `cloud/jobs/runC_reveal_pair_v5.json`: named frame selectors support farm subdivision, the renderer writes PNG and the existing shipper produces JPEG. Neither job has been launched on the farm. Keep both drivers together: paired Reveal imports the Watch's image writer.

The production branch was checked through `9ed020d`; no newer changes existed in the relevant shared renderer trees. Do not adopt the separate PR10 compiler experiment for these renders. No merge or final creative approval is implied by this package.
