# Handoff: state of THE LONG DAWN

Read `../BIBLE.md` (story, frame-exact time grid, shot briefs) and `../music/CUES.md` first.

## What exists
- **Rendered picture** (1920×804, global frame numbers) is exported as clips in `handoff/clips/<dept>_<start>-<end>.mp4`
  (H.264 CRF 15). Rebuild the PNG frames with `python3 handoff/import.py`, which writes `renders/<dept>/f_%05d.png`.
  - hills: complete (0–359 intro, 1200–1439 first beacon, 2460–2807 coda)
  - embers: complete (300–1199)
  - globe: complete v1 (1752–1935, 2232–2495); a cloud-fix re-render of the dawn was in progress at export time
  - accord: ~90% (1912–2247); check the clips for missing frames
  - montage: ~70% (1440–1767); check the clips for missing frames
  - Missing frames can be re-rendered from code: see each `shots/<dept>/NOTES.md` or `PLAN.md`, and each `render.py`.
- **Audio**: `handoff/audio/{mix,score,sfx}.flac` (v1, 48 kHz, exactly 117.000 s, −16 LUFS), plus `sfx_events/` and
  `music/out/sfx_events.json`. Convert back with `ffmpeg -i handoff/audio/mix.flac music/out/mix.wav` (likewise
  score/sfx). The score department's sync report is `music/analysis/report.txt`. It was working on a louder climax
  arrival at 93.333 s and a clearer race-drum entrance at 26.667 s; v1 is usable as is.

## Finishing
```bash
pip install numpy scipy opencv-python-headless pillow soundfile pedalboard pyloudnorm numba   # + apt: ffmpeg
python3 handoff/import.py
ffmpeg -i handoff/audio/mix.flac music/out/mix.wav      # (and score.wav/sfx.wav if using edit/mix.py)
python3 edit/assemble.py --draft                        # fast half-res check -> out/draft.mp4
python3 edit/assemble.py                                # master -> out/the_long_dawn.mp4 (titles, grain, letterbox, audio)
```
`edit/assemble.py` holds the cut (dissolves in linear light, the flash-cut into the sunrise, fades).
`edit/titles.py` holds all on-screen text and its timing. Missing frames show as grey "MISSING" placeholders.
Keep the master under 100 MB for GitHub: the defaults are CRF 17 with maxrate 6 Mb/s, about 90 MB for 117 s.

## Known weak spots, if there's time
- embers 800–830: the vortex is seen edge-on before it fills the sky; 960–1036: the hand is a bit tubular.
- hills FIRST BEACON: the mountains are 2.5-D cards; the stand-up pose blend is stiff.
- Text size: consider 54 px instead of 50 px for phone legibility (`edit/titles.py`, `story_lines()`).
