# Refusal final-delivery sample audit

2026-09-28 UTC. No blocker found in the inspected samples. This is a bounded visual/provenance audit, not approval of full-speed playback or first-time-viewer comprehension.

Measured coverage: **33 distinct RGB frames / 240 and their 33 matching mattes / 240**, all fully decoded and SHA256-matched to `work/c5-page-review/book_C5_refusal-final.json`. The two inspected source files also match that receipt's recorded hashes. Exact frames, hashes and receipt identity are in [sample-verification.json](sample-verification.json). This did not repeat the existing full-sequence numeric QC.

Visual coverage: full native 1920×804 C2080, C2200 and C2319; the final artwork in a native crop; twelve keyframes at 480×201; and **all 24 consecutive C2192–2215 frames** at 480×201. The latter overlaps three keyframes, which is why the distinct sample total is 33. The final matte was also viewed reduced; this audit did not assess a downstream matte composite.

Observed:

- C2080 is an empty right page. The offering forearm/hand appears first, then the ring, then the other figure and its shading. In C2192–2215, the raised glove contour visibly grows through C2196–2201; palm marks and connected wrist follow, then longer cloak folds and hatching. I saw no positional jump or detached final glove in that strip.
- At native size, the offering hand reads as open, with a cuff and four extended fingers; the ring's gold outline is distinct immediately above the hand. At480px width, both ring and offering gesture remain identifiable. I did not see an obvious floating-object defect in the final drawing.
- The raised hand connects to its sleeve. The cowl opening is dark hatchwork with no visible facial features; its opening points away from the offer. **The figure is drawn in a turned-away pose; these samples do not show a separate body-turn animation.** This matches a drawing of the gesture, but should not be described as a performed turn.
- The completed robe has directional folds and broken hatching that follow the paper's ink treatment. Its broad outline is simple during construction. I did not see an obvious pasted graphic, doll-like face or disconnected final cloth contour; those are visual judgments, not measured quality guarantees. Ground marks below the hem help seat the completed figure. Caption space below the drawing is clear in the inspected frames.

Source timing checked without importing or rendering: `book_c_v5.py:37` maps `(Cframe−2080)/24` to shot time; `v5_inkpages.py:53,59,95,123` schedules the offering hand at 0.35–2.4s, ring at 2.4–3.15s, figure/raised glove at 3.15–5.4s, and final hatching at 5.4–7.05s. The sampled construction/hold agrees with those stages. Current PR 12 handback notes were read, including the earlier hood/sleeve revisions; the images assessed here came from the final delivery directory.

Limits: 207 RGB frames were not visually inspected here. Sequential stills show drawing progression but cannot establish smoothness at 24 fps, flicker absence across the entire shot, emotional timing, audience interpretation, caption legibility, or the adjacent EDIT joins. No source, PR, mailbox, render, or process changes were made.

Review artifacts: [keyframes](refusal-keyframes-480.png), [24 consecutive frames](refusal-consecutive-2192-2215-480.png), [native final artwork crop](refusal-end-native-art-crop.png), [480px final composition](refusal-end-480.png).

Publication note: this report is a public copy. Private paths are normalized; package-link changes and original file SHA256 are recorded in [PUBLICATION.json](PUBLICATION.json). Original delivery-receipt and sample hashes are preserved. Helper scripts are not included.
