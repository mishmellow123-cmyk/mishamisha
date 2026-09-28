# Paired Reveal: opening final-image audit

2026-09-28 09:30 UTC. **Partial audit while the final render is running.** No new blocker was found in the 26 inspected final images. This does not mark the 240-frame shot delivered or visually approved.

## Evidence and observations

Root inspected all 24 consecutive C2880–2903 final JPEGs as reduced composition tiles and unchanged native-pixel crops of each fire. Native C2880 was inspected in the earlier checkpoint; native C2903 and C3023 were inspected during this audit. C3000 and C3023 were also viewed at 480×201. The union is 26 distinct final images. Each fully decoded at 1920×804 and matched its renderer receipt's SHA256 entry. The two growing receipts were captured as immutable snapshots for this check; their status was `running`, with 44 and 47 frame records respectively. These snapshots are evidence of partial progress, not completed deliveries.

Both near and far fire shapes are present in C2880, then enlarge together over the opening frames before settling into their individual flicker. The native crops show the two changes occurring within the same early window, without one fire waiting several frames for the other. There is no preceding unlit frame inside this shot, so the images alone do not establish an off-to-on transition before C2880; the common C2880 ignition is also pinned by source and the existing tests.

The two marked peaks remain widely separated and unobscured in the inspected states. At 480 pixels wide, the near flame remains recognizable and the farther flame is a tiny gold mark above its gold summit. Its position is distinct, but this is an informed inspection, not evidence that an unprompted viewer will read the intended reciprocal promise. C3000/C3023 retain the same two-site composition at the camera's later position.

The consecutive window shows coherent terrain and camera progression. No obvious single-frame shape dropout, unexpected exposure change or positional jump appeared in the inspected crops or composition strip. The pre-existing stylized ink-fire drawings remain in use. The stills do not establish real-time motion quality or a final verdict on fire realism.

All 118 source/data files pinned in the PR's `runtime.json` still match their recorded hashes. No rendering source, output frame or active process was changed for this audit.

## Outstanding checks

The completed delivery receipt, all-frame encoding/hash validation, whole-shot numerical QC, native endpoint C3119 and the C2988–3011 split-worker boundary strip remain pending. On completion, reconcile these 26 sample hashes against the final delivery receipt. Full-speed playback, captions, sound, adjacent-shot joins and audience comprehension are outside this audit.

Artifacts: `composition-2880-2903.png`, `near-native-2880-2903.png`, `far-native-2880-2903.png`, `C3000-480.png`, `C3023-480.png`, `evidence.json` and the two `.snapshot` files. The composition sheet contains 480×201 tiles and may be scaled by the viewer; native detail sheets preserve the source crop pixels. The crop bounds are recorded in `evidence.json`.
