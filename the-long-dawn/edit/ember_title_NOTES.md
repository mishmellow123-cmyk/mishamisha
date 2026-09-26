# TITLE dept (ember title) — NOTES

## >>> PAUSED mid-refinement (critic_tone.md §M6), 2026-09-26 — resume here <<<
Nothing is rendering. `renders/title/` still holds the complete, valid v3 render (title at
y=402); `edit/ember_title.py` is unchanged since that render (no edits made for M6 yet).

Asked (director): (1) cut the out-of-focus gathering embers ~70% and make them rise visibly FROM
the answering fires (HILLS will redo those as 12-20 distinct ridgeline fires; match their real
positions if they change); (2) keep the letters clear of the beacon flame (raise to y~312, or form
later); keep the chord sync at 2880 and the crumble. Re-render, update the sheet, report briefly.

Measured (plate, x 980-1260): beacon flame (luma > 0.8) top edge by v2 frame:
2846:254  2848:299  2850:336  2852:356  2854:361  2856:403  2858:428  2860:445  2862:452  2866:468.
Band at y=402 is 363-431 (flame clears ~2859); at y=312 it is 273-341 (clears ~2850).

Plan (decided):
1. `TITLE_Y = 312` and `TA0, TA1 = 2853, 2870` (G/D land from ~2856 when the flame is clear;
   the last N is done by ~2880).
2. Letter embers ~225 (from 750), mostly sharp (only ~1/3 mildly defocused, CoC <= 4). Each is
   LAUNCHED FROM A VISIBLE LIT FIRE near its target's x (rate-limited per fire), integrated
   FORWARD with `_v_free`: fast launch decaying to a float, curl, wind, crane coupling that
   decays after launch (add a coupling-decay param). Aim the rise with a per-ember scale so it
   arrives 30-150 px below the target at TG (2 iterations), then the existing curved glide.
3. Reveal driven by the embers: farthest-point-sample the targets per letter; the glyph's
   reveal field A(x,y) = min_i(TA_i + |p - T_i| / 2.6 px/frame), so each landing ember ignites the
   stroke and the fire runs along it (the flash term already exists).
4. Free embers 440 -> ~150, all born at visible fires; remove the 14 foreground bokeh.
5. Fires sync: in `ember_title_fires.py` add `--write` (rewrites FIRES/CREST in place, preferring
   `coda.Coda().wave/.old`), store a hash of shots/hills/coda.py + hillworld.py, and have the CLI
   re-extract automatically (subprocess, PYTHONDONTWRITEBYTECODE=1, NUMBA_CACHE_DIR=tmp) when
   that hash changes, so a re-render always follows HILLS' current fires.
6. Re-render (~65 s), review sheet at the usual frames plus a crop of 2846-2866 around the
   flame, then update the report (formula unchanged; the title is now at y=312).

---

**Status of the v3 delivery: DONE (2026-09-26).** 167 frames rendered: `renders/title/t_02800.exr` … `t_02966.exr`
(half-float linear-light RGB, 1920x804, additive, v2 numbering, 230 MB). Review sheet:
`~/mishamisha/_local_logs/review/title.jpg` (2810 2830 2850 2865 2880 2905 2935 2950, composited
over the real plate with the edit's fade to black).

## Files
- `edit/ember_title.py` — the title layer: `layer(f)` (loads the EXR; renders on the fly if one
  is missing; None outside 2800-2966), `composite(img_srgb, f)`, `soft_clip()`, `render(f)`,
  `review()`, CLI.
- `edit/ember_title_fires.py` — CODA fire world positions + our hilltop's crest profile,
  extracted once (read-only) from shots/hills; embedded data, no runtime dependency.

## Integration (director)
After `picture(f)`, before `titles.composite` / `finish`:

    import ember_title
    img = ember_title.composite(img, f)
    # == look.linear_to_srgb(soft_clip(look.srgb_to_linear(img) + ember_title.layer(f)))
    # soft_clip(x) = x for x <= 0.8, else 0.8 + 0.2 * (1 - exp(-(x - 0.8) / 0.2))

and drop `_title(L)` from `titles.story_lines` (all cuts). The layer handles its own fade (zero
through 2806 and from 2961). `ember_title` sets `OPENCV_IO_ENABLE_OPENEXR=1` on import; if cv2
already read an EXR earlier in the process, set it in the environment instead.

## Re-render
    source ~/.venvs/longdawn/env.sh
    python edit/ember_title.py                   # 2800-2966, 2 workers, ~65 s, < 400 MB each
    python edit/ember_title.py --range 2840 2890 --workers 1
    python edit/ember_title.py --review          # the sheet
Deterministic (seeded); every frame is a pure function of f.

## Timeline (v2)
2807 first sparks rise from the answering fires · 2830-2880 embers gather out of focus and glide
into the letters left to right, racking into focus (THE ~2843-2858 … N ~2862-2878) · 2880-2925
hold (gold, fine ember stipple, drifting heat, halo) · 2923-2945 letters crumble left to right,
tops first, into ~2950 embers that lift, cool to ember red and die · 0 from 2961.

## Known, accepted
- A few defocused embers overlap the elder's thin scarf for a frame or two around 2840-2852
  (the figures/cairn are occluders, the scarf is not); invisible at speed.
