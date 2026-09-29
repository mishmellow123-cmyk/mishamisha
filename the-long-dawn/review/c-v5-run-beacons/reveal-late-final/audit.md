# Paired Reveal: delivered late-window audit

No new visual blocker in this bounded sample. I inspected delivered C2960 and C3095 at native 1920×804, and every image in C3072–3095 as a 480×201 composition and unchanged native-pixel crops of both fires. That is **25 unique final JPEGs**, including a 24-frame consecutive window. The worker boundary and final C3119 are outside this review.

## Visual observations

**The two beacons remain separate and present.** In both native key frames the nearer fire occupies the lower central ridge and the far fire sits on the upper-right summit, with open landscape between them. Neither is cropped or hidden by a foreground ridge. In every image of C3072–3095, both flame marks remain visible at their respective peaks. Their outlines change across the strip; I observed no one-frame disappearance, location jump or joining of the two shapes.

**Native detail carries more than the reduction.** At native size the nearer beacon has several distinct flame tongues, dark ink outlines and a warm area around its base; the far beacon retains a smaller outlined flame and gold summit patch. At 480 pixels wide the nearer beacon remains the easier focal point. The far one becomes a tiny gold mark with very limited flame detail against similarly warm paper. I can locate both throughout the sheet with their positions in mind; this does not establish that an unprompted viewer will notice both. The small-scale limitation remains relevant to the intended reveal even though no sampled dropout occurs.

**The late image stays coherent.** The terrain lines, horizon and peaks shift gradually with the camera. The flame shapes vary while remaining registered to their peaks. I observed no abrupt background displacement, exposure jump or broken mountain outline within the 24-frame window. This is a sheet inspection, not real-time playback or a test of the entire shot's pacing.

## Measured provenance

- All **25/25 JPEG SHA256 hashes** match immutable copies of the two renderer receipts captured while their status was `running`. Every selected image decoded at 1920×804. The snapshots contain 102 and 107 recorded frames respectively; only the selected 25 were image-hashed and visually examined here.
- Final C2960 is **byte-identical** to the previously approved native sample at `work/c-v5-reveal-pair-tests/fullres/f_02960.jpg`. Both SHA256 values are `d6433ce1f4a0e90e27a41a5c8c39e37fc88efe9e6eed27fd453a27a58a7521f7`.
- The two current entrypoint files, `watch_c_v5.py` and `reveal_pair_c_v5.py`, match their frozen hashes in the delivery configuration. This is an entrypoint check, not an audit of every imported module.

See [evidence.json](evidence.json) for exact sample hashes, receipt-snapshot hashes, dimensions and crop coordinates. The local-only `work/c-v5-reveal-final-audit/late/build_evidence.py` helper reads existing files, preserves the receipt snapshots and creates the sheets without renderer imports or calls. The crops retain original decoded pixels; the reduced compositions use area averaging. No enhancement was applied.

- [C2960/C3095 at 480 pixels wide](key-composition-480.png)
- [24 consecutive reduced compositions](composition-3072-3095.png)
- [24 native crops: near fire](near-native-3072-3095.png)
- [24 native crops: far fire](far-native-3072-3095.png)
- [Native key-frame near crops](key-near-native.png) and [far crops](key-far-native.png)

No source, final image, process, PR or mailbox was changed. This audit does not repeat full-shot numeric QC, approve adjacent shots/captions/sound, claim audience comprehension, or cover uninspected frames. The snapshots preserve running receipts; later completion receipts may supersede them.
