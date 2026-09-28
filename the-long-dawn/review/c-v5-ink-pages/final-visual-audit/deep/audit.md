# Deep Abandoned final-delivery sample audit

2026-09-28 UTC. No new delivery/provenance defect found in the inspected samples. **Small-format readability remains limited:** at 480 pixels wide, the lantern is an ambiguous roofed outline and ladder rungs are difficult to distinguish. This audit does not establish that abandonment reads at a glance at that size.

Measured coverage: **30 distinct RGB frames / 240 and their 30 mattes / 240** fully decoded and SHA256-matched to `work/c5-page-review/book_C5_deep_abandoned-final.json`. The inspected `book_c_v5.py`, `v5_inkpages.py` and underlying `pages.py` also match the receipt's source hashes. Exact sample/frame/hash records: [sample-verification.json](sample-verification.json). The existing 240-frame/239-pair numerical QC was not repeated.

Visual coverage: native 1920×804 C4240, C4360 and C4479; native crops of the lantern/top gallery, bottom ladder and deep gold vein; seven keyframes at 480×201; and **all 24 consecutive C4352–4375 frames** at 480×201. C4360 overlaps the strip, producing 30 distinct samples.

Observations:

- I saw no miners in the five gallery levels of the native start/middle/end frames. The ladders are empty. Native detail resolves the side rails and rungs; the surface symbol has a small handle, roof and cage-like body beside the entrance. Its base stays aligned with the surface line, without an obvious suspended-object defect.
- The gold vein remains the dominant bright feature from the entrance down through the galleries to the root at the bottom. Its branches and the surrounding arches, stairs and rock hatching remain in place across the sampled endpoints and consecutive strip. I saw no obvious mark dropout, redraw or positional jump; subtle lighting differences are consistent with the time-dependent hearth source.
- At 480px width, the mine's stacked galleries and bright vein remain clear. In my reduced view, the surface symbol could be read as a tiny building; I cannot reliably identify a set-down lantern without the source context. The ladders reduce largely to dark vertical bars. The native detail is present, but the reduced composition does not independently prove the intended abandonment cue is legible.
- The composition is static from its first sampled frame. There is no new drawing growth to expect here. The final page stays visually complete through C4479 in the inspected stills.

Source expectations: `book_c_v5.py:50–62` builds the complete texture once with `texture(1e9)`, uses a fixed camera and varies the light with shot time. `v5_inkpages.py:127–130` inherits the existing Deep geometry and discards miners' strokes while consuming their original random draws; `:132–156` adds the lantern and empty ladders. `pages.py:649–669` defines the existing vein/hall layout. The code path preserves that geometry by construction; this audit did not regenerate or pixel-compare the original pre-v5 mine plate, so the visual finding is that the vein is present and stable in the delivered samples.

Limits: 210 RGB frames were not visually inspected here. Sequential stills are not playback evidence and do not establish whole-shot flicker absence, emotional timing, audience comprehension, caption readability or EDIT joins. Hash checks on sampled mattes establish identity, not a downstream compositing test. No renderer, PR, mailbox or process changes were made.

Artifacts: [keyframes](deep-keyframes-480.png), [24 consecutive frames](deep-consecutive-4352-4375-480.png), [480px final composition](deep-end-480.png), [native lantern/top-gallery crop](deep-end-lantern-top-native.png), [native bottom ladder](deep-end-bottom-ladder-native.png), [native gold vein](deep-end-gold-vein-native.png).

Publication note: this report is a public copy. Private paths are normalized; package-link changes and original file SHA256 are recorded in [PUBLICATION.json](PUBLICATION.json). Original delivery-receipt and sample hashes are preserved. Helper scripts are not included.
