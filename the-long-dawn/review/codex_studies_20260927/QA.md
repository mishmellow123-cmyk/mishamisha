# Comparison review log

27 September 2026. Source base: `07d5dca`.

- Actual council renders inspected at 4960, 5160, 5566 and 5596. Full-resolution 5160/5566 inspected; the comparison retains the observed weak lowered-flame limitation.
- A15 baseline and all three final cairn recipes rendered at 4280/4360/4399. All proposals remain rejected. Final variant bounds are normalized to the measured baseline bounds, not the earlier nominal 0.64 m.
- Final derived-bound A15 renders completed at 4280, 4360 and 4399; frame 4360 inspected against the baseline and the 3,000 m diagnostic. The tested depth strip matches 3,000 m exactly. Full PNGs differ by up to 3 code values and are not claimed byte-identical.
- All 26 comparison PNGs decode successfully. The final mountain correction was also inspected in the viewer. The video contains 16 frames, 1920×402, 24/1 fps, duration 0.666667 s; baseline and candidate each occupy one half.
- Browser checks cover five comparison states, image loading, baseline/alternative buttons and keyboard slider (50→51 updates the image clip to 49%).
- Desktop and 390 px layout inspected; narrow layout has 390 px document width and no horizontal overflow. The temporary viewport override was reset.
- Video playback checked with `document.hidden=false`: two successive readings gave 0.557370 s and 0.348949 s with looping enabled, and the displayed hand/flame frame changed. Playback was then paused. This verifies player/frame progression, not the full scene's artistic quality.
- Browser error/warning log was empty at inspection. The viewer has no CSS transitions, external resources or build dependencies.

The shader patch is intentionally separate from verified renderer fixes. No final film acceptance is implied by this log. The original laptop's current masters were unavailable.
